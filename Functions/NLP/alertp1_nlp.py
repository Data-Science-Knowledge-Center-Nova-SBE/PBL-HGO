#Text strings changes

def remove_names(text):
    import re
    # Find all words that start with a capital letter
    names = re.findall(r'\b[A-Z][a-z]+\b', text)
    
    # Replace the names with an empty string
    for name in names:
        text = text.replace(name, '')
        
    return text

def text_list(text):

    # Create an empty list to store the text
    text_list = []

    # Loop through the 'text' column
    for text in text.str.lower(): # Transform every word to lower case
        text_list.append(text)

    return text_list

def spacy_lemmatizer(text_list):
    # Spacy is required
    # $pip install -U spacy
    # $python -m spacy download pt_core_news_md
    # Additional information: https://spacy.io/usage
    
    import spacy
    import pt_core_news_md
    nlp = pt_core_news_md.load()

    doclist = list(nlp.pipe(text_list))

    text_list=[]
    for i, doc in enumerate(doclist):
        text_list.append(' '.join([listitem.lemma_ for listitem in doc]))
        
    return text_list

def remove_stop_words(text_list):

    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    # Download the Portuguese stop words
    nltk.download('stopwords')
    nltk.download('punkt')

    # Get the Portuguese stop words
    stop_words = set(stopwords.words('portuguese'))
    stop_words.update(['.', ',','(',')',':','-','?','+','/',';','drª','``','','desde','doente','consulta','alterações','se',"''",'cerca','refere','hgo','utente','vossa','s','...','ainda','c','filha','costa','dr.','pereira','ja','--','p','dr','h','n','>','q','//','..','b','++','%'])
    ###### I THINK WE SHOULD REMOVE ALETRAÇOES FROM THIS LIST#########

    # Create a new list to store the filtered text
    filtered_text = []

    # Loop through the text_list and remove the stop words
    for text in text_list:
        words = word_tokenize(text)
        words = [word for word in words if word.lower() not in stop_words]
        filtered_text.append(" ".join(words))
    
    return filtered_text

# Dataframe changes 

def age_text(df, column):
    import pandas as pd
    
    #This should be applied to the filtered text column of the dataset
    
    # Create a boolean mask to identify rows containing the words "anos" and "idade" in the "Texto" column
    mask = df[column].str.contains('anos') & df[column].str.contains('idade')

    # Use the boolean mask to extract all the rows that contain the desired words
    extracted_rows = df[mask]

    # Extract the numbers before "anos" and save it as a new column "Age"
    extracted_rows['Age'] = extracted_rows['Texto'].str.extract(r'(\d+) anos')

    # Convert the "Age" column from string to integer, converting non-numeric values to NaN
    extracted_rows['Age'] = pd.to_numeric(extracted_rows['Age'], errors='coerce')

    # Drop the rows with NaN values in the "Age" column
    extracted_rows = extracted_rows.dropna(subset=['Age'])

    # Add age column
    df["Age"] = extracted_rows['Age']

    # Create a new column with age range
    df['Age_range'] = df['Age'].apply(lambda x: '1-20' if (20 > x >= 1) else '20-40' if (40 > x > 20) else '40-60' if (60 > x > 40) else '60-80' if (80 > x > 60) else '80-100' if (100 > x > 80) else '100>' if (100 > x) else '0')

    return df

def medications_text(df, excel_file):
    import pandas as pd

    # Import excel file with medications
    medications = pd.read_excel('allPackages.xls')
    medications['Nome do medicamento'].count()
    medications = medications.groupby(['Nome do medicamento']).size().reset_index(name='counts').sort_values(by = 'counts')
    medications = pd.DataFrame(medications)

    #word_list = medications.groupby(['Nome do medicamento']).size()
    word_list = medications['Nome do medicamento']

    # Create a column containing all words in the dataframe
    df['all_words'] = df.apply(lambda x: ' '.join(x.astype(str)), axis=1)
    
    # Create a boolean mask indicating whether each word is in the list
    mask = df['all_words'].apply(lambda x: any(word in x for word in word_list))
    
    mask = df['Texto'].apply(lambda x: any(word in x for word in word_list))
    
    result = df.loc[mask, :]
    
