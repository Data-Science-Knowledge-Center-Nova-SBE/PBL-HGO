#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gensim
from gensim.models import Word2Vec
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def get_word2vec_features(text_data):
    # split the text data into sentences
    sentences = [sentence.split() for sentence in text_data]

    # train a Word2Vec model on the sentences
    model = Word2Vec(sentences, min_count=1, vector_size=100, workers=4)

    # get the Word2Vec vectors for the words in the vocabulary
    vectors = model.wv.vectors

    # get the vocabulary of the Word2Vec model
    vocab = model.wv.key_to_index.keys()

    # create a TF-IDF vectorizer
    tfidf = TfidfVectorizer()
    tfidf.fit(text_data)

    # transform the text data into a TF-IDF matrix
    tfidf_matrix = tfidf.transform(text_data)

    # compute the TF-IDF weighted Word2Vec vectors
    weighted_vectors = []
    for i, word in enumerate(vocab):
        tfidf_score = tfidf.idf_[tfidf.vocabulary_[word]] if word in tfidf.vocabulary_ else 1
        weighted_vector = vectors[i] * tfidf_score
        weighted_vectors.append(weighted_vector)

    # create a dictionary mapping words to vectors
    word_to_vec = dict(zip(vocab, weighted_vectors))

    # define a function to compute the average word vector for a sentence
    def sentence_to_avg_vector(sentence):
        words = sentence.split()
        vectors = [word_to_vec[word] for word in words if word in word_to_vec]
        if len(vectors) > 0:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(100)

    # compute the average word vector for each sentence
    features = np.array([sentence_to_avg_vector(sentence) for sentence in text_data])
    
    return features


# In[ ]:




