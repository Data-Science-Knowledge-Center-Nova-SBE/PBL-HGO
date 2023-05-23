#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def extract_top_words(train_set_filtered, num_words=20):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        train_set_filtered['filtered_text'],
        train_set_filtered['result'],
        test_size=0.2,
        random_state=42
    )

    # Create the TF-IDF vectorizer object
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=5)

    # Fit and transform the vectorizer on the train set
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    # Extract the top words with the highest scores for the train set
    train_scores = np.asarray(X_train_tfidf.mean(axis=0)).ravel().tolist()
    train_features = tfidf_vectorizer.get_feature_names_out()
    top_train_indices = np.argsort(train_scores)[::-1][:num_words]
    top_train_words = [train_features[i] for i in top_train_indices]
    top_train_scores = [train_scores[i] for i in top_train_indices]

    # Transform the vectorizer on the test set
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Extract the top words with the highest scores for the test set
    test_scores = np.asarray(X_test_tfidf.mean(axis=0)).ravel().tolist()
    test_features = tfidf_vectorizer.get_feature_names_out()
    top_test_indices = np.argsort(test_scores)[::-1][:num_words]
    top_test_words = [test_features[i] for i in top_test_indices]
    top_test_scores = [test_scores[i] for i in top_test_indices]

    # Return the top words and scores for both train and test sets
    return {
        'train': {
            'words': top_train_words,
            'scores': top_train_scores
        },
        'test': {
            'words': top_test_words,
            'scores': top_test_scores
        }
    }

# Assuming you have the `train_set_filtered` DataFrame
num_words = 20
top_words = extract_top_words(train_set_filtered, num_words=num_words)

# Create a new DataFrame to store the extracted features
extracted_features = pd.DataFrame()

# Iterate over each row in the original dataset
for index, row in train_set_filtered.iterrows():
    word_scores = []
    # Compute the word scores for the top words
    for word in top_words['train']['words']:
        # Compute the word frequency or TF-IDF score for the current row
        score = row['filtered_text'].count(word)  # Example: Using word frequency
        word_scores.append(score)
    # Append the word scores to the new DataFrame
    extracted_features = extracted_features.append(pd.Series(word_scores), ignore_index=True)

