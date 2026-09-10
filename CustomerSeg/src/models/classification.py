import time
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

# Max rows used for slow O(n^2) models to prevent UI freeze
_SVM_MAX_SAMPLES = 3000
_KNN_MAX_SAMPLES = 5000

class CustomerClassifier:
    def __init__(self):
        # LinearSVC is O(n) vs SVC O(n^2-n^3); wrap in CalibratedClassifierCV for predict_proba
        self.models = {
            'NaiveBayes':    GaussianNB(),
            'DecisionTree':  DecisionTreeClassifier(random_state=42),
            'SVM (Linear)':  CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42)),
            'KNN':           KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', n_jobs=-1)
        }
        self.results = {}
        self._sample_warnings = {}

    def prepare_data(self, rfm_df):
        rfm_df = rfm_df.copy()
        rfm_df['IsReturn'] = (rfm_df['Frequency'] > 1).astype(int)
        X = rfm_df[['Recency', 'Monetary']]
        y = rfm_df['IsReturn']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def _subsample(self, X_train, y_train, max_n, model_name):
        if len(X_train) > max_n:
            idx = np.random.RandomState(42).choice(len(X_train), max_n, replace=False)
            self._sample_warnings[model_name] = (
                f"Trained on {max_n:,} of {len(X_train):,} rows (capped for speed)"
            )
            return X_train.iloc[idx], y_train.iloc[idx]
        return X_train, y_train

    def train_evaluate_all(self, X_train, X_test, y_train, y_test):
        caps = {
            'SVM (Linear)': _SVM_MAX_SAMPLES,
            'KNN':          _KNN_MAX_SAMPLES,
        }
        for name, model in self.models.items():
            print(f"Training {name}...")
            t0 = time.time()
            _X_tr, _y_tr = self._subsample(
                X_train, y_train, caps.get(name, len(X_train)), name
            )
            model.fit(_X_tr, _y_tr)
            elapsed = time.time() - t0
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            self.results[name] = {
                "Accuracy":        accuracy_score(y_test, y_pred),
                "Precision":       precision_score(y_test, y_pred, zero_division=0),
                "Recall":          recall_score(y_test, y_pred, zero_division=0),
                "F1":              f1_score(y_test, y_pred, zero_division=0),
                "ConfusionMatrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
                "TrainTime_s":     round(elapsed, 3),
                "TrainSamples":    len(_X_tr),
                "SampleWarning":   self._sample_warnings.get(name, None),
            }
            if y_prob is not None:
                try:
                    self.results[name]["AUC_ROC"] = roc_auc_score(y_test, y_prob)
                    self.results[name]["LogLoss"]  = log_loss(y_test, y_prob)
                except Exception:
                    pass
            print(f"  done in {elapsed:.2f}s")
        return self.results
