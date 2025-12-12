import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.train_df = None
        self.test_df = None
        self.scaler = StandardScaler()
        self.encoders = {}

    def load_data(self):
        """Loads dataset from CSV."""
        print(f"Loading data from {self.file_path}...")
        try:
            self.df = pd.read_csv(self.file_path, encoding='ISO-8859-1') # specific encoding for Online Retail often needed
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.file_path)
        print(f"Data loaded. Shape: {self.df.shape}")
        return self.df

    def clean_data(self):
        """Cleans the dataset: missing values, cancelled orders."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        initial_shape = self.df.shape
        
        # Drop rows with missing Customer ID (essential for customer analysis)
        # Note: Actual column name might vary slightly ('Customer ID' or 'CustomerID')
        col_map = {c: c.strip() for c in self.df.columns}
        self.df.rename(columns=col_map, inplace=True)
        
        if 'Customer ID' in self.df.columns:
            self.df.dropna(subset=['Customer ID'], inplace=True)
        elif 'CustomerID' in self.df.columns:
            self.df.rename(columns={'CustomerID': 'Customer ID'}, inplace=True)
            self.df.dropna(subset=['Customer ID'], inplace=True)
            
        # Remove cancelled orders (Invoice starting with 'C')
        self.df['Invoice'] = self.df['Invoice'].astype(str)
        self.df = self.df[~self.df['Invoice'].str.startswith('C')]
        
        # Remove invalid quantity/price
        self.df = self.df[(self.df['Quantity'] > 0) & (self.df['Price'] > 0)]
        
        print(f"Data cleaned. Rows removed: {initial_shape[0] - self.df.shape[0]}. New Shape: {self.df.shape}")
        return self.df

    def feature_engineering(self):
        """Adds TotalAmount, parses Dates."""
        if self.df is None:
            return
            
        # TotalAmount
        self.df['TotalAmount'] = self.df['Quantity'] * self.df['Price']
        
        # Date Parsing
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        self.df['Year'] = self.df['InvoiceDate'].dt.year
        self.df['Month'] = self.df['InvoiceDate'].dt.month
        self.df['Hour'] = self.df['InvoiceDate'].dt.hour
        self.df['DayOfWeek'] = self.df['InvoiceDate'].dt.dayofweek
        
        print("Feature engineering completed.")
        return self.df

    def encode_normalize(self):
        """Encodes categorical variables and normalizes numerical ones."""
        # Encode Country
        le_country = LabelEncoder()
        self.df['Country_Encoded'] = le_country.fit_transform(self.df['Country'])
        self.encoders['Country'] = le_country
        
        # Normalize numeric features for ML (optional per model, but asked in Unit I)
        # We'll normalize Price and TotalAmount for demonstration, usually stored in separate scaled types
        # Note: Normalizing everything is not always good for interpretation, but we create scaled cols.
        self.df['Price_Scaled'] = self.scaler.fit_transform(self.df[['Price']])
        self.df['TotalAmount_Scaled'] = self.scaler.fit_transform(self.df[['TotalAmount']])
        
        print("Encoding and Normalization completed.")
        return self.df

    def split_data(self, test_size=0.2):
        """Splits data into train and test sets."""
        # For simple random split (might not be ideal for time-series, but standard for Unit I syllabus)
        self.train_df, self.test_df = train_test_split(self.df, test_size=test_size, random_state=42)
        print(f"Data split. Train shape: {self.train_df.shape}, Test shape: {self.test_df.shape}")
        return self.train_df, self.test_df

    def get_customer_data(self):
        """Aggregates data by Customer for Customer-level prediction (RFM)."""
        # Recency, Frequency, Monetary
        snapshot_date = self.df['InvoiceDate'].max() + pd.Timedelta(days=1)
        
        rfm = self.df.groupby('Customer ID').agg({
            'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
            'Invoice': 'nunique',
            'TotalAmount': 'sum'
        }).reset_index()
        
        rfm.rename(columns={
            'InvoiceDate': 'Recency',
            'Invoice': 'Frequency',
            'TotalAmount': 'Monetary'
        }, inplace=True)
        
        return rfm

if __name__ == "__main__":
    # Test the pipeline
    processor = DataPreprocessor('data/raw/online_retail_II.csv')
    processor.load_data()
    processor.clean_data()
    processor.feature_engineering()
    processor.encode_normalize()
    processor.split_data()
    rfm = processor.get_customer_data()
    print("RFM Head:")
    print(rfm.head())
