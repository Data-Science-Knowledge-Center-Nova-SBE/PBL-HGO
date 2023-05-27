#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import math

def calculate_presence(data, text_column):
    # Remove numbers from the clean text using regular expressions
    data['clean_text'] = data[text_column].apply(lambda x: re.sub(r'\d+', '', x))
    
    # Split the data into train and test sets
    # Calculate the index for the split
    split_index = math.ceil(0.8 * len(data))
    train_set = data.iloc[:split_index].reset_index(drop=True)
    test_set = data.iloc[split_index:].reset_index(drop=True)

    # Create a TfidfVectorizer object with desired parameters
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, max_features=20)

    # Fit and transform the text data
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set['clean_text'])
    
    # Fit and transform the text data
    tfidf_matrix_test = tfidf_vectorizer.transform(test_set['clean_text'])

    # Create a dataframe for TfidfVectorizer output with top 20 words as columns
    tfidf_df_train = pd.DataFrame(tfidf_matrix_train.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    tfidf_df_test = pd.DataFrame(tfidf_matrix_test.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    
    # Merge both dataframes
    tfidf_df = pd.concat([tfidf_df_train,tfidf_df_test])

    # Replace TF-IDF scores with presence (1 if word is present, 0 otherwise)
    tfidf_df = tfidf_df.applymap(lambda x: 1 if x > 0 else 0)

    # Append the features to the original data frame
    data_with_features = pd.concat([data, tfidf_df], axis=1)

    return data_with_features


# In[ ]:


def calculate_presence(data, text_column):
    # Remove numbers from the clean text using regular expressions
    data['clean_text'] = data[text_column].apply(lambda x: re.sub(r'\d+', '', x))
    
    # Split the data into train and test sets
    # Calculate the index for the split
    split_index = math.ceil(0.8 * len(data))
    train_set = data.iloc[:split_index].reset_index(drop=True)
    test_set = data.iloc[split_index:].reset_index(drop=True)

    # Create a TfidfVectorizer object with desired parameters
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, max_features=20)

    # Fit and transform the text data
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set['clean_text'])
    
    # Fit and transform the text data
    tfidf_matrix_test = tfidf_vectorizer.transform(test_set['clean_text'])

    # Create a dataframe for TfidfVectorizer output with top 20 words as columns
    tfidf_df_train = pd.DataFrame(tfidf_matrix_train.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    tfidf_df_test = pd.DataFrame(tfidf_matrix_test.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    
    # Merge both dataframes
    tfidf_df = pd.concat([tfidf_df_train,tfidf_df_test])

    # Append the features to the original data frame
    data_with_features = pd.concat([data, tfidf_df], axis=1)

    return data_with_features

