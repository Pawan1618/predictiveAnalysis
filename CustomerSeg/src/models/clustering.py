
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

class ClusterAnalysis:
    def __init__(self):
        self.kmeans = None
        self.hierarchical = None

    def kmeans_clustering(self, X, n_clusters=3):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = self.kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        return labels, score, self.kmeans.cluster_centers_

    def hierarchical_clustering(self, X, n_clusters=3):
        self.hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        labels = self.hierarchical.fit_predict(X)
        score = silhouette_score(X, labels) # computationally expensive on large data
        return labels, score
