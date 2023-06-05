# import the classes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


#logistic regression with probability
def log_regression_proba(features,target):
    # split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2,random_state=16,shuffle=False)
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression(random_state=16)
    # fit the model with data
    clf=logreg.fit(X_train, y_train)
     # Make preictions on the train set
    y_train_scores = logreg.predict_proba(X_train)[:, 1]
    # Make predictions on the test set
    y_test_scores = logreg.predict_proba(X_test)[:, 1]
    
    return y_train_scores, y_test_scores, clf.coef_, X_train, X_test, y_train, y_test 
# normal log regression
def log_regression(features,target):
    # split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2,random_state=16,shuffle=False)
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression(random_state=16)
    # fit the model with data
    clf=logreg.fit(X_train, y_train)
    y_pred_train = logreg.predict(X_train)
    y_pred_test = logreg.predict(X_test)

    return y_pred_train,y_pred_test,clf.coef_, clf.intercept_,X_train, X_test, y_train, y_test


# Evaluate the accuracy of the train set
#accuracy = accuracy_score(y_train, logreg.predict(X_train))
#print("Accuracy of the train set:", accuracy)

# Evaluate the recall of the train set
#recall = recall_score(y_train,logreg.predict(X_train), average='macro')
#print("Recall of the train set:", recall)
# Evaluate the precision of the train set
#precision = precision_score(y_train, logreg.predict(X_train), average='macro')
#print("Precision of the train set:", precision)