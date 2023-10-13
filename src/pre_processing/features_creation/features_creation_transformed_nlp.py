from src.pre_processing.features_creation.features_creation_nlp import *
#from src.pre_processing.features_creation.features_creation_bert import *
#from src.pre_processing.features_creation.features_creation_tfidf import *

### NLP ###
def lowering_text(alertP1):
    lower_text(alertP1,"Texto","clean_text_caveman")
    return(alertP1)
def clean_text(alertP1):
   lower_text(alertP1,"Texto","clean_text")
   remove_stop_words(alertP1, "clean_text", "clean_text")
   spacy_lemmatizer(alertP1, "clean_text", "clean_text")
   return(alertP1)
#Medications
def medication_column(alertP1):
   categorize_medication(alertP1,"clean_text_caveman", "../../src/pre_processing/features_creation/data/list_medications.xlsx", 80)
   add_textcount_columns(alertP1,"clean_text_caveman","medication_level_1")
   add_textcount_columns(alertP1,"clean_text_caveman","medication_level_2")
   add_textcount_columns(alertP1,"clean_text_caveman","medication_level_3")   
   return(alertP1)
#Symptoms
def symptoms_column(alertP1):
   categorize_symptoms(alertP1,"clean_text_caveman", "../../src/pre_processing/features_creation/data/list_symptoms.xlsx", 80)
   add_textcount_columns(alertP1,"clean_text_caveman","symptom_1")
   add_textcount_columns(alertP1,"clean_text_caveman","symptom_0")  
   return(alertP1)

def exams(alertP1):
   categorize_exams(alertP1,"clean_text_caveman", "../../src/pre_processing/features_creation/data/list_exams.xlsx", 95)
   add_textcount_columns(alertP1,"clean_text_caveman","exam_identified") 
   return(alertP1)
def comorbidities(alertP1):
   categorize_comorbidities(alertP1,"clean_text_caveman", "../../src/pre_processing/features_creation/data/list_comorbidities.xlsx", 85)
   add_textcount_columns(alertP1,"clean_text_caveman","comorbidity_identified") 
   return(alertP1)
def synonyms(alertP1):
   check_synonyms("Data/synonyms_dict.xlsx", alertP1, "../../src/pre_processing/features_creation/data/list_concepts.xlsx", 80, process_all=True)
   return(alertP1)
def LDA(alertP1):
   train_and_predict_lda(alertP1, 'clean_text',n_components=2, learning_decay=0.5, random_state=16)
   return(alertP1) 

#def bert(alertP1):
   protocols = pd.read_csv('BERT/protocols_to_bert.csv')
   baseline = protocols.columns.to_list()
   reference = []

   for col in baseline:
    
     protocols2 = protocols.copy()
     protocols2 = protocols2[[col]].dropna()

     lower_text(protocols2,col, col)
     remove_stop_words(protocols2,col, col)
     spacy_lemmatizer(protocols2,col, col)

     reference.append(list(protocols2[col].dropna()))
   alertP1=bert_split_referrals(alertP1, reference, model_name = 'sentence-transformers/msmarco-MiniLM-L-6-v3')
   return (alertP1)
import math
#from gensim.models import Word2Vec
#import numpy as np
# def w2v(alertP1):
#     # Split data into train and test
#     AlertP1_sorted = alertP1.sort_values(by='DATA_RECEPCAO')

#     # calculate the index for the split
#     split_index = math.ceil(0.8 * len(AlertP1_sorted))

#     # split the data frame into test and train sets
#     train_set = AlertP1_sorted.iloc[:split_index]


#     #Converting text into list of sentences
#     sentences = train_set['clean_text'].tolist()
#     #W2V model building
#     model = Word2Vec(sentences, window=3, min_count=5, workers=1,sg=0,alpha=0.01)  # Adjust parameters as needed
    
#     #Featurization
#     def get_sentence_vector(sentence):
#         vectors = []
#         for word in sentence:
#             if word in model.wv:
#                 vectors.append(model.wv[word])#If the word in the text exists in the W2V vocabulary, it assigns the vector 
#         if vectors:
#             return np.mean(vectors, axis=0)#Takes the mean of the vectors for that referral
#         else:
#             return np.zeros(model.vector_size) #if it can't find 
        
#     alertP1['word2vec_feature'] = alertP1['clean_text'].apply(lambda x: get_sentence_vector(x)) #assigning w2v to the correct columns.
#     # Define the number of dimensions in the word2vec vectors
#     num_dimensions = 100
#     # Extract the word2vec vectors as a NumPy array
#     vectors = np.array(alertP1['word2vec_feature'].tolist())

#     # Split the "word2vec_feature" column into separate columns
#     alertP1[[f"dim_{i+1}" for i in range(num_dimensions)]] = pd.DataFrame(vectors.tolist(), index=alertP1.index)# Remove the original "word2vec_feature" column
#     alertP1.drop("word2vec_feature", axis=1, inplace=True)
#     return(alertP1)
# def tfidf(alertP1):
#    alertP1=calculate_presence(alertP1,'clean_text')
#    return (alertP1)
