
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel   # for BERT
from Data_with_NLP import *

def bert_embedding(dataset, column, baseline_list, prefix, model_name='sentence-transformers/multi-qa-MiniLM-L6-cos-v1'):

    # Get the sentences from the dataset (to be compared to the basiline list)
    referrals = dataset[column].tolist()

    # Initialize the BERT tokenizer per sentence on the baseline list
    for index, protocol in enumerate(baseline_list):
        # initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # create list of sentences to be encoded, being the first the baseline protocol
        sentences = [protocol] + referrals

        # initialize dictionary to store tokenized sentences
        tokens = {'input_ids': [], 'attention_mask': []}

        # loop through sentences
        for sentence in sentences:

            # encode each sentence and append to dictionary
            new_tokens = tokenizer.encode_plus(sentence, max_length=128,
                                            truncation=True, padding='max_length',
                                            return_tensors='pt')
            tokens['input_ids'].append(new_tokens['input_ids'][0])
            tokens['attention_mask'].append(new_tokens['attention_mask'][0])

        # reformat list of tensors into single tensor
        tokens['input_ids'] = torch.stack(tokens['input_ids'])
        tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

        # run through model and get embeddings    
        outputs = model(**tokens)
        outputs.keys()

        print(index)
        # get embeddings from last hidden state
        embeddings = outputs.last_hidden_state

        # get attention mask  
        attention_mask = tokens['attention_mask']

        # create mask for embeddings
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()

        # mask embeddings
        masked_embeddings = embeddings * mask

        # sum the embeddings along the axis of the tokens
        summed = torch.sum(masked_embeddings, 1)

        # sum the number of tokens that should be considered in each sentence
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        summed_mask.shape

        # divide the summed embeddings by the number of tokens that should be considered in each sentence
        mean_pooled = summed / summed_mask

        # convert from PyTorch tensor to numpy array
        mean_pooled = mean_pooled.detach().numpy()

        # calculate cosine similarity between baseline protocol and referrals
        results = cosine_similarity(
            [mean_pooled[0]],
            mean_pooled[1:]
        )
        # add results to dataset
        column_name = prefix + str(index)
        dataset[column_name] = results[0]


def bert_easy(dataset, column_name, baseline_list, suffix = 'max'):
    
    from sklearn.metrics.pairwise import cosine_similarity

    # Creating a list to store the text
    referrals = dataset[column_name].tolist()

    # Creating a list to store the column names
    columns_bert = []

    # Looping through the baseline list
    for index, protocol in enumerate(baseline_list):

        # Creating a list with the protocol and the referrals
        sentences = [protocol] + referrals
        
        # Encoding the sentences
        sentences_vectors = model.encode(sentences)

        # Calculating the cosine similarity
        results = cosine_similarity([sentences_vectors[0]], sentences_vectors[1:])

        # Appending the information on the dataset
        column_name = 'prot_' + str(index)
        columns_bert.append(column_name)
        dataset[column_name] = results[0]

    dataset[suffix + '_score'] = dataset[columns_bert].max(axis=1)
    dataset.drop(columns_bert, axis=1, inplace=True)
    
    return dataset

def bert_protocols(data, reference):

    split_text_df = data[['COD_REFERENCIA', 'Texto', 'text_length']]
    split_text_df = split_text_df[split_text_df['text_length'] > 0]
    split_text_df['text_split'] = split_text_df['Texto'].apply(lambda x: x.split("."))
    split_text_df =  split_text_df.explode('text_split')
    lower_text(split_text_df,'text_split', 'text_split')
    remove_stop_words(split_text_df,'text_split', 'text_split')
    spacy_lemmatizer(split_text_df,'text_split', 'text_split')
    
    for i, protocol in enumerate(reference):

        bert_easy(split_text_df, 'text_split', protocol, suffix = str(i))

    return split_text_df