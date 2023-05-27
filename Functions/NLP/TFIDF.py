
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
def append_tfidf_to_dataframe1(data, text_column):
    # Remove numbers from the clean text using regular expressions
    data['clean_text'] = data[text_column].apply(lambda x: re.sub(r'\d+', '', x))

    # Create a TfidfVectorizer object with desired parameters
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, max_features=20)

    # Fit and transform the text data for combined set
    tfidf_matrix_combined = tfidf_vectorizer.fit_transform(data['clean_text'])

    # Create a dataframe for TfidfVectorizer output with top 20 words as columns for combined set
    tfidf_df_combined = pd.DataFrame(tfidf_matrix_combined.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # Append the TF-IDF dataframe to the original dataframe
    data = pd.concat([data, tfidf_df_combined], axis=1)

    return data


# ## If the word is present, return 1, if the word is not present, return 0


def append_tfidf_to_dataframe2(data, text_column):
    # Remove numbers from the clean text using regular expressions
    data['clean_text'] = data[text_column].apply(lambda x: re.sub(r'\d+', '', x))

    # Create a TfidfVectorizer object with desired parameters
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, max_features=20, binary=True)

    # Fit and transform the text data
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['clean_text'])

    # Create a dataframe for TfidfVectorizer output with top 20 words as columns
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # Convert the TF-IDF values to binary (1 if non-zero, 0 otherwise)
    tfidf_df = tfidf_df.apply(np.sign)

    # Append the TF-IDF dataframe to the original dataframe
    data = pd.concat([data, tfidf_df], axis=1)

    return data
def tfidf(alertP1):
    append_tfidf_to_dataframe1(alertP1,'cle')





