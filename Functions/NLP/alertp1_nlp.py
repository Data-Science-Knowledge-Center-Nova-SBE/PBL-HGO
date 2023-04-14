import pandas as pd

#Text strings changes

def remove_names(string):
    import re
    # Find all words that start with a capital letter
    names = re.findall(r'\b[A-Z][a-z]+\b', text)
    
    # Replace the names with an empty string
    for name in names:
        text = text.replace(name, '')
        
    return text

def lower_text(df, column, new_column):

    df[new_column] = df[column].str.lower()

    return df

def remove_stop_words(df, original_column, new_column):

    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    # Download the Portuguese stop words
    nltk.download('stopwords')
    nltk.download('punkt')

    # Lower Case strings
    df[new_column] = df[original_column].str.lower()

    # Get the Portuguese stop words
    stop_words = set(stopwords.words('portuguese'))
    stop_words.update(['.', ',','(',')',':','-','?','+','/',';','drª','``','','se',"''",'cerca','refere','hgo','utente','vossa','s','...','ainda','c','filha','costa','dr.','pereira','ja','--','p','dr','h','n','>','q','//','..','b','++','%'])
    ###### I THINK WE SHOULD REMOVE ALETRAÇOES FROM THIS LIST#########

    # Create a new list to store the filtered text
    filtered_text = []

    # Loop through the text_list and remove the stop words
    for text in df[new_column]:
        words = word_tokenize(text)
        words = [word for word in words if word.lower() not in stop_words]
        filtered_text.append(" ".join(words))
    
    df[new_column] = filtered_text

    return df

def spacy_lemmatizer(df, original_column, new_column):
    # Spacy is required
    # $pip install -U spacy
    # $python -m spacy download pt_core_news_md
    # Additional information: https://spacy.io/usage
    
    import spacy
    import pt_core_news_md
    nlp = pt_core_news_md.load()

    doclist = list(nlp.pipe(df[original_column]))

    text_list=[]
    for i, doc in enumerate(doclist):
        text_list.append(' '.join([listitem.lemma_ for listitem in doc]))
    
    df[new_column] = text_list

    return df

# Dataframe changes 

def disease_class(df, column):
    list = ["cefaleia","demência","convulsão", "epilepsia", "sincope", "vertigem", "tremor", "acidente vascular cerebral"]
    for d in list:
        df[d] = df[column].str.contains(d)
    return df


def age_text(df, column):
    
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
    medications = pd.read_excel('excel_file')
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
    
def categorize_medication(df, column, medications_excel, threshold=80):
    #The file should have a header and follow the structure [0] = Drug Name and [1] = Level
    
    #Install of fuzzywuzzy is required
    #pip install fuzzywuzzy

    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process
    import re

    medications_df = pd.read_excel(medications_excel,header= 0)

    for i, text in df[column].iteritems():
        words = re.findall(r'\b\w+\b', text)
        for index, row in medications_df.iterrows():
            medication_name = row["name"].lower()
            medication_level = 'medication_level_' + str(row['level'])
            for word in words:
                ratio = fuzz.ratio(medication_name, word)
                if ratio >= threshold:
                    text = re.sub(r'\b' + word + r'\b(?![\w])', medication_level, text)
        df.at[i, column] = text

    return df

def categorize_symptoms(df, column, symptoms_excel, threshold=75):
    #The file should have a header and follow the structure [0] = symptom and [1] = consultation
    
    #Install of fuzzywuzzy is required
    #pip install fuzzywuzzy

    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process
    import re

    symptoms_df = pd.read_excel(symptoms_excel,header= 0)

    for i, text in df[column].iteritems():
        words = re.findall(r'\b\w+\b', text)
        for index, row in symptoms_df.iterrows():
            symptom_name = row["symptom"].lower()
            symptom_consultation = 'symptom_' + str(row['consultation'])
            for word in words:
                ratio = fuzz.ratio(symptom_name, word)
                if ratio >= threshold:
                    text = re.sub(r'\b' + word + r'\b(?![\w])', symptom_consultation, text)
        df.at[i, column] = text

    return df

def categorize_symptoms_simple(df, column, symptoms_excel, threshold=80):
    #The file should have a header and follow the structure [0] = symptom 
    
    #Install of fuzzywuzzy is required
    #pip install fuzzywuzzy

    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process
    import re

    symptoms_df = pd.read_excel(symptoms_excel,header= 0)

    for i, text in df[column].iteritems():
        words = re.findall(r'\b\w+\b', text)
        for index, row in symptoms_df.iterrows():
            symptom_name = row["symptom"].lower()
            symptom_consultation = 'symptom_identified'
            for word in words:
                ratio = fuzz.ratio(symptom_name, word)
                if ratio >= threshold:
                    text = re.sub(r'\b' + word + r'\b(?![\w])', symptom_consultation, text)
        df.at[i, column] = text

    return df

def categorize_comorbidities(df, column, commorbidities_excel, threshold=80):
    #The file should have a header and follow the structure [0] = comorbidity
    
    #Install of fuzzywuzzy is required
    #pip install fuzzywuzzy

    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process
    import re

    comorbidities_df = pd.read_excel(commorbidities_excel,header= 0)

    for i, text in df[column].iteritems():
        words = re.findall(r'\b\w+\b', text)
        for index, row in comorbidities_df.iterrows():
            comorbidity_name = row["comorbidity"].lower()
            comorbidity_consultation = 'comorbidity_identified'
            for word in words:
                ratio = fuzz.ratio(comorbidity_name, word)
                if ratio >= threshold:
                    text = re.sub(r'\b' + word + r'\b(?![\w])', comorbidity_consultation, text)
        df.at[i, column] = text

    return df

def categorize_exams(df, column, exams_excel, threshold=90):
    #The file should have a header and follow the structure [0] = exam
    
    #Install of fuzzywuzzy is required
    #pip install fuzzywuzzy

    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process
    import re

    exams_df = pd.read_excel(exams_excel,header= 0)

    for i, text in df[column].iteritems():
        words = re.findall(r'\b\w+\b', text)
        for index, row in exams_df.iterrows():
            exam_name = row["exam"].lower()
            exam_consultation = 'exam_identified'
            for word in words:
                ratio = fuzz.ratio(exam_name, word)
                if ratio >= threshold:
                    text = re.sub(r'\b' + word + r'\b(?![\w])', exam_consultation, text)
        df.at[i, column] = text

    return df

def get_word_list (df, column):
    
    from collections import Counter
    words = df[column].str.split(expand=True).stack()
    word_counts = Counter(words)
    word_counts_df = pd.DataFrame.from_dict(word_counts, orient='index', columns=['count']).reset_index().rename(columns={'index': 'word'})
    word_counts_df = word_counts_df.sort_values(by='count', ascending=False)

    return word_counts_df


def add_textcount_columns(df, text_column, text):
    # Define a function to count the occurrences of a substring in a string
    def count_substring(string, substring):
        return string.count(substring)

    # Create new columns by applying the count_substring function to the Text column
    df[text] = df[text_column].apply(lambda x: count_substring(x, text))

    return df
def check_synonyms(excel_file, df, column, threshold, process_all=False, word=None):

    from fuzzywuzzy import fuzz
    import re

    # Load the list of synonyms from the Excel file
    synonyms_df = pd.read_excel(excel_file)

    # Define a function to check if any of the synonyms are present in the text
    def check_text(text, synonyms):
        count = 0
        words = re.findall(r'\b\w+\b', text.lower())
        for word in words:
            # Use fuzzywuzzy to allow for some mistakes in the way words are written
            scores = [fuzz.ratio(word, synonym) for synonym in synonyms]
            if max(scores) >= threshold:
                count += 1
        return count

    if process_all:
        # Process all column names from the Excel file
        for word in synonyms_df.columns:
            synonyms = synonyms_df[word].dropna().str.lower().tolist()
            df[f"count_{word}"] = df[column].apply(lambda x: check_text(x, synonyms))
    else:
        # Process only the given word
        if word is None:
            raise ValueError("Please provide a word to process.")
        if word not in synonyms_df.columns:
            raise ValueError(f"Invalid word: {word}. Available words: {list(synonyms_df.columns)}")
        synonyms = synonyms_df[word].dropna().str.lower().tolist()
        df[f"count_{word}"] = df[column].apply(lambda x: check_text(x, synonyms))

    return df