# import the classes
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, precision_recall_curve, auc, confusion_matrix, cohen_kappa_score

#random forest
def random_forest(features,target):
    # split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2,random_state=16,shuffle=False)
    # instantiate the model (using the default parameters)
    rndmforest =RandomForestClassifier(n_estimators=500, random_state=42)
    # fit the model with data
    clf=rndmforest.fit(X_train, y_train)
    # Make preictions on the train set
    y_train_scores = rndmforest.predict_proba(X_train)[:, 1]
    # Make predictions on the test set
    y_test_scores = rndmforest.predict_proba(X_test)[:, 1]
    return y_train_scores, y_test_scores, clf.feature_importances_,X_train, X_test, y_train, y_test

