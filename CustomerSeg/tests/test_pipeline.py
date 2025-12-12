
import pandas as pd
import numpy as np
from src.models.regression import SalesRegressor
from src.models.classification import CustomerClassifier
from src.models.clustering import ClusterAnalysis
from src.models.rules import MarketBasketAnalysis
from src.models.advanced import AdvancedModels

def test_pipeline():
    print("Testing Pipeline...")
    
    # Dummy RFM data
    rfm = pd.DataFrame({
        'Recency': np.random.randint(1, 365, 100),
        'Frequency': np.random.randint(1, 50, 100),
        'Monetary': np.random.rand(100) * 1000
    })
    
    # Regression
    print("Testing Regression...")
    reg = SalesRegressor()
    X_train, X_test, y_train, y_test = reg.prepare_data(rfm)
    reg.train_linear(X_train, y_train)
    reg.train_polynomial(X_train, y_train)
    
    # Classification
    print("Testing Classification...")
    clf = CustomerClassifier()
    X_train, X_test, y_train, y_test = clf.prepare_data(rfm)
    clf.train_evaluate_all(X_train, X_test, y_train, y_test)
    
    # Clustering
    print("Testing Clustering...")
    clust = ClusterAnalysis()
    clust.kmeans_clustering(rfm[['Recency', 'Frequency', 'Monetary']].values)
    
    # Advanced
    print("Testing Advanced...")
    adv = AdvancedModels()
    adv.apply_pca(X_train, n_components=2)
    adv.train_mlp_regression(X_train, y_train)
    
    print("All tests passed successfully!")

if __name__ == "__main__":
    test_pipeline()
