
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

class SalesRegressor:
    def __init__(self):
        self.model = None
        self.poly_model = None
        self.poly_features = None

    def prepare_data(self, rfm_df):
        """Prepares X and y for regressions. 
        Predicting Monetary (CLV) based on Recency and Frequency."""
        X = rfm_df[['Recency', 'Frequency']]
        y = rfm_df['Monetary']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_linear(self, X_train, y_train):
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        return self.model

    def train_polynomial(self, X_train, y_train, degree=2):
        self.poly_features = PolynomialFeatures(degree=degree)
        X_poly = self.poly_features.fit_transform(X_train)
        self.poly_model = LinearRegression()
        self.poly_model.fit(X_poly, y_train)
        return self.poly_model

    def predict(self, X, model_type='linear'):
        if model_type == 'linear':
            return self.model.predict(X)
        elif model_type == 'polynomial':
            X_poly = self.poly_features.transform(X)
            return self.poly_model.predict(X_poly)

    def evaluate(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return {"MAE": mae, "RMSE": rmse, "R2": r2}
