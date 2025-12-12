
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

class CustomerClassifier:
    def __init__(self):
        self.models = {
            'NaiveBayes': GaussianNB(),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42), # Probability for ROC/LogLoss
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        self.results = {}

    def prepare_data(self, rfm_df):
        """
        Prepares target: Return Customer (1) or One-time (0).
        Based on Frequency > 1.
        Features: Recency, Monetary.
        """
        rfm_df['IsReturn'] = (rfm_df['Frequency'] > 1).astype(int)
        X = rfm_df[['Recency', 'Monetary']] # Don't use Frequency to predict derived Frequency!
        y = rfm_df['IsReturn']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_evaluate_all(self, X_train, X_test, y_train, y_test):
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            self.results[name] = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1": f1_score(y_test, y_pred, zero_division=0),
                "ConfusionMatrix": confusion_matrix(y_test, y_pred).tolist(),
            }
            
            if y_prob is not None:
                try:
                    self.results[name]["AUC_ROC"] = roc_auc_score(y_test, y_prob)
                    self.results[name]["LogLoss"] = log_loss(y_test, y_prob)
                except:
                    pass
                    
        return self.results
