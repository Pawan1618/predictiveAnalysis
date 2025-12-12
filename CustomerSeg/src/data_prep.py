import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
import os

# Configure Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.train_df = None
        self.test_df = None
        self.scaler = StandardScaler()
        self.encoders = {}

    def load_data(self):
        """Loads dataset from CSV with robust encoding handling."""
        logger.info(f"Loading data from {self.file_path}...")
        try:
            # 'ISO-8859-1' is common for retail datasets containing European characters
            self.df = pd.read_csv(self.file_path, encoding='ISO-8859-1') 
        except UnicodeDecodeError:
            logger.warning("ISO-8859-1 encoding failed. Trying default UTF-8.")
            self.df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            raise

        logger.info(f"Data loaded successfully. Initial Shape: {self.df.shape}")
        return self.df

    def clean_data(self):
        """
        Cleans the dataset:
        1. Standardization of column names.
        2. Dropping missing Customer IDs (crucial for segmentation).
        3. Removing cancelled orders ('C' prefix).
        4. Handling duplicates.
        5. Removing invalid numerical values.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        initial_rows = self.df.shape[0]
        
        # 1. Standardize Column Names
        self.df.columns = [c.strip() for c in self.df.columns]
        
        # Mapping common variations to standard names
        rename_map = {'CustomerID': 'Customer ID', 'Total': 'TotalAmount'}
        self.df.rename(columns=rename_map, inplace=True)
        
        # 2. Drop Missing Customer IDs
        if 'Customer ID' in self.df.columns:
            missing_cust = self.df['Customer ID'].isnull().sum()
            self.df.dropna(subset=['Customer ID'], inplace=True)
            logger.info(f"Dropped {missing_cust} rows with missing Customer ID.")
        
        # 3. Remove Duplicates
        duplicates = self.df.duplicated().sum()
        self.df.drop_duplicates(inplace=True)
        logger.info(f"Dropped {duplicates} duplicate rows.")

        # 4. Remove Cancelled Orders
        # Ensure Invoice is string
        self.df['Invoice'] = self.df['Invoice'].astype(str)
        cancelled_mask = self.df['Invoice'].str.startswith('C')
        n_cancelled = cancelled_mask.sum()
        self.df = self.df[~cancelled_mask]
        logger.info(f"Removed {n_cancelled} cancelled orders.")
        
        # 5. Remove Invalid Values (Negative/Zero Quantity or Price)
        # Note: Some returns might have negative quantity, but we removed 'C' invoices.
        # Check for any remaining anomalies.
        invalid_mask = (self.df['Quantity'] <= 0) | (self.df['Price'] <= 0)
        n_invalid = invalid_mask.sum()
        self.df = self.df[~invalid_mask]
        logger.info(f"Removed {n_invalid} rows with invalid Quantity/Price.")
        
        final_rows = self.df.shape[0]
        logger.info(f"Data cleaning complete. Rows retained: {final_rows} ({(final_rows/initial_rows)*100:.2f}%)")
        return self.df

    def feature_engineering(self):
        """
        Adds derived features:
        - TotalAmount (Revenue)
        - Date info (Year, Month, Hour, DayOfWeek)
        """
        if self.df is None:
            return
            
        logger.info("Starting Feature Engineering...")
        
        # Revenue Calculation
        self.df['TotalAmount'] = self.df['Quantity'] * self.df['Price']
        
        # Date Parsing
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        self.df['Year'] = self.df['InvoiceDate'].dt.year
        self.df['Month'] = self.df['InvoiceDate'].dt.month
        self.df['Hour'] = self.df['InvoiceDate'].dt.hour
        self.df['DayOfWeek'] = self.df['InvoiceDate'].dt.dayofweek
        self.df['DayName'] = self.df['InvoiceDate'].dt.day_name()
        
        # Time of Day Segment
        self.df['TimeOfDay'] = pd.cut(self.df['Hour'], 
                                      bins=[0, 6, 12, 18, 24], 
                                      labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                      include_lowest=True)
        
        logger.info("Feature engineering completed.")
        return self.df

    def encode_normalize(self):
        """Encodes categorical variables and normalizes numerical ones."""
        logger.info("Starting Encoding & Normalization...")
        
        # Encode Country
        le_country = LabelEncoder()
        self.df['Country_Encoded'] = le_country.fit_transform(self.df['Country'])
        self.encoders['Country'] = le_country
        
        # Normalize numeric features
        # We create scaled versions but keep originals for interpretation
        self.df['Price_Scaled'] = self.scaler.fit_transform(self.df[['Price']])
        self.df['TotalAmount_Scaled'] = self.scaler.fit_transform(self.df[['TotalAmount']])
        
        logger.info("Encoding and Normalization completed.")
        return self.df

    def get_customer_data(self):
        """Aggregates data by Customer for RFM Analysis."""
        logger.info("Aggregating data for RFM Analysis...")
        
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
        
        logger.info(f"RFM table created. Shape: {rfm.shape}")
        return rfm

    def save_processed_data(self, output_dir='data/processed'):
        """Saves processed dataframes to CSV."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        logger.info(f"Saving processed data to {output_dir}...")
        
        # Save Transaction Level Data
        # We drop Scaled columns for the CSV usually to save space unless needed specifically
        save_cols = [c for c in self.df.columns if not c.endswith('_Scaled')]
        self.df[save_cols].to_csv(f"{output_dir}/cleaned_transactions.csv", index=False)
        
        # Save Customer Level Data (RFM)
        rfm = self.get_customer_data()
        rfm.to_csv(f"{output_dir}/rfm_customer_data.csv", index=False)
        
        logger.info("Data saved successfully.")

if __name__ == "__main__":
    # Production Pipeline Execution
    RAW_PATH = 'data/raw/online_retail_II.csv'
    
    processor = DataPreprocessor(RAW_PATH)
    processor.load_data()
    processor.clean_data()
    processor.feature_engineering()
    processor.encode_normalize()
    processor.save_processed_data()
