
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

class AdvancedModels:
    def __init__(self):
        self.pca = None
        self.mlp_reg = None
        self.mlp_clf = None

    def apply_pca(self, X, n_components=2):
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)
        explained_variance = self.pca.explained_variance_ratio_
        return X_pca, explained_variance

    def train_mlp_regression(self, X_train, y_train):
        # Neural Network for Sales Prediction
        self.mlp_reg = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        self.mlp_reg.fit(X_train, y_train)
        return self.mlp_reg

    def train_mlp_classification(self, X_train, y_train):
        # Neural Network for Churn/Return
        self.mlp_clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        self.mlp_clf.fit(X_train, y_train)
        return self.mlp_clf
