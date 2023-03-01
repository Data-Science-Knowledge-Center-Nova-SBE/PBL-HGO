# import the classes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#logistic regression 
def log_regression(features,target):
    # split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2,random_state=16)
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression(random_state=16)
    # fit the model with data
    clf=logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    return y_pred,clf.coef_, clf.intercept_,X_train, X_test, y_train, y_test


# Evaluate the accuracy of the train set
#accuracy = accuracy_score(y_train, logreg.predict(X_train))
#print("Accuracy of the train set:", accuracy)

# Evaluate the recall of the train set
#recall = recall_score(y_train,logreg.predict(X_train), average='macro')
#print("Recall of the train set:", recall)
# Evaluate the precision of the train set
#precision = precision_score(y_train, logreg.predict(X_train), average='macro')
#print("Precision of the train set:", precision)