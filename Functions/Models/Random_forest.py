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
    logreg =RandomForestClassifier(n_estimators=500, random_state=42)
    # fit the model with data
    clf=logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    return y_pred, clf.feature_importances_,X_train, X_test, y_train, y_test


def evaluate_thresholds(y_pred, y_test, k = 211):
    #Precion@k
    def precision_at_k(y_true, y_pred, k):
        y_true = y_true.to_numpy()
        # Get the indices of the top k predictions
        top_k_indices = np.argsort(y_pred)[-k:]
        # Get the top k predictions
        top_k_predictions = y_true[top_k_indices]
        # Calculate precision@k
        precision = np.sum(top_k_predictions) / k
        return precision

    # Default threshold (0.5)
    default_threshold = 0.5
    default_y_pred_binary = (y_pred >= default_threshold).astype(int)
    default_accuracy = accuracy_score(y_test, default_y_pred_binary)
    default_recall = recall_score(y_test, default_y_pred_binary)
    default_precision = precision_score(y_test, default_y_pred_binary)
    default_precision_at_k = precision_at_k(y_test, default_y_pred_binary, k)
    
    # Define thresholds
   

    # Precision-Recall trade-off
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    f1_scores = 2 * precision * recall / (precision + recall)
    best_index = pd.Series(f1_scores).idxmax()
    pr_threshold = thresholds[best_index]
    pr_y_pred_binary = (y_pred >= pr_threshold).astype(int)
    pr_accuracy = accuracy_score(y_test, pr_y_pred_binary)
    pr_recall = recall_score(y_test, pr_y_pred_binary)
    pr_precision = precision_score(y_test, pr_y_pred_binary)
    pr_precision_at_k = precision_at_k(y_test, pr_y_pred_binary, k)
    
    thresholds = np.linspace(0, 1, 100)

    # Balanced Accuracy
    cms = [confusion_matrix(y_test, (y_pred >= t).astype(int)) for t in thresholds]
    sensitivity = [cm[1, 1] / (cm[1, 1] + cm[1, 0]) for cm in cms]
    specificity = [cm[0, 0] / (cm[0, 0] + cm[0, 1]) for cm in cms]
    balanced_accuracy = [(s + sp) / 2 for s, sp in zip(sensitivity, specificity)]
    ba_threshold = thresholds[np.argmax(balanced_accuracy)]
    ba_y_pred_binary = (y_pred >= ba_threshold).astype(int)
    ba_accuracy = accuracy_score(y_test, ba_y_pred_binary)
    ba_recall = recall_score(y_test, ba_y_pred_binary)
    ba_precision = precision_score(y_test, ba_y_pred_binary)
    ba_precision_at_k = precision_at_k(y_test, ba_y_pred_binary,k)
    
    # Cohen's Kappa
    kappas = [cohen_kappa_score(y_test, (y_pred >= t).astype(int)) for t in thresholds]
    kappa_threshold = thresholds[np.argmax(kappas)]
    kappa_y_pred_binary = (y_pred >= kappa_threshold).astype(int)
    kappa_kappa = cohen_kappa_score(y_test, kappa_y_pred_binary)
    kappa_accuracy = accuracy_score(y_test, kappa_y_pred_binary)
    kappa_recall = recall_score(y_test, kappa_y_pred_binary)
    kappa_precision = precision_score(y_test, kappa_y_pred_binary)
    kappa_precision_at_k = precision_at_k(y_test, kappa_y_pred_binary, k)

    # Limit to K
    limit_to_k_threshold = np.sort(y_pred)[-k]
    limit_to_k_y_pred_binary = (y_pred >= limit_to_k_threshold).astype(int)
    limit_to_k_accuracy = accuracy_score(y_test, limit_to_k_y_pred_binary)
    limit_to_k_recall = recall_score(y_test, limit_to_k_y_pred_binary)
    limit_to_k_precision = precision_score(y_test, limit_to_k_y_pred_binary)
    limit_to_k_precision_at_k = precision_at_k(y_test, limit_to_k_y_pred_binary, k)
    
    # Create output dataframe
    data = {
        'Analysis': ['Default', 'Precision-Recall trade-off', 'Balanced Accuracy', "Cohen's Kappa", 'Limit to K'],
        'Threshold': [round(default_threshold, 2), round(pr_threshold, 2), round(ba_threshold, 2), round(kappa_threshold, 2), round(limit_to_k_threshold, 2)],
        'Accuracy': [round(default_accuracy, 2), round(pr_accuracy, 2), round(ba_accuracy, 2), round(kappa_accuracy, 2), round(limit_to_k_accuracy, 2)],
        'Recall': [round(default_recall, 2), round(pr_recall, 2), round(ba_recall, 2), round(kappa_recall, 2), round(limit_to_k_recall, 2)],
        'Precision': [round(default_precision, 2), round(pr_precision, 2), round(ba_precision, 2), round(kappa_precision, 2), round(limit_to_k_precision, 2)],
        'Precision@211': [round(default_precision_at_k, 2), round(pr_precision_at_k, 2), round(ba_precision_at_k, 2), round(kappa_precision_at_k, 2), round(limit_to_k_precision_at_k, 2)]
    }
    
    df = pd.DataFrame(data)
    return df