
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_prep import DataPreprocessor
from src.models.regression import SalesRegressor
from src.models.classification import CustomerClassifier
from src.models.clustering import ClusterAnalysis
from src.models.rules import MarketBasketAnalysis
from src.models.advanced import AdvancedModels
from src.models.ensemble import EnsembleModels

# Page Config
st.set_page_config(page_title="Customer Purchase Prediction AI", layout="wide")

st.title("🛍️ Customer Purchase Prediction & Segmentation AI")

# Sidebar
st.sidebar.title("Navigation")
options = [
    "Project Intro",
    "Unit I: Data Prep",
    "Unit II: Regression (Sales)",
    "Unit III: Classification (Churn)",
    "Unit IV: Clustering & Rules",
    "Unit V: PCA & Neural Networks",
    "Unit VI: Ensemble & Eval"
]
choice = st.sidebar.radio("Go to:", options)

# Caching Data Load
@st.cache_resource
def load_and_prep_data():
    processor = DataPreprocessor('data/raw/online_retail_II.csv')
    df = processor.load_data()
    # Use smaller sample for demo speed if needed, but we'll try full load first
    df = processor.clean_data()
    df = processor.feature_engineering()
    df = processor.encode_normalize()
    rfm = processor.get_customer_data()
    return processor, df, rfm

try:
    with st.spinner("Loading and Processing Data... This may take a moment."):
        processor, df, rfm = load_and_prep_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

if choice == "Project Intro":
    st.image("https://miro.medium.com/max/1400/1*9G3_f7W4e4f5v9_z8x9g.jpeg", caption="Retail Analytics", use_column_width=True)
    st.markdown("""
    ### 🎯 Project Objectives
    1. **Predict Sales/CLV** using Regression.
    2. **Predict Customer Churn** using Classification.
    3. **Segment Customers** using K-Means.
    4. **Market Basket Analysis** using Apriori.
    5. **Advanced Modeling** with PCA & Neural Networks.
    """)

elif choice == "Unit I: Data Prep":
    st.header("Unit I: Data Preparation")
    st.subheader("Raw Data Preview")
    st.write(df.head())
    st.write(f"**Shape:** {df.shape}")
    
    st.subheader("RFM Table (Customer Level)")
    st.write(rfm.head())
    
    st.markdown("### 🛠️ Techniques Used")
    st.markdown("- Missing Value Handling (Customer ID removed)")
    st.markdown("- Feature Engineering (TotalAmount, RFM)")
    st.markdown("- Normalization (StandardScaler)")

elif choice == "Unit II: Regression (Sales)":
    st.header("Unit II: Regression - Predict CLV")
    
    reg_model = SalesRegressor()
    X_train, X_test, y_train, y_test = reg_model.prepare_data(rfm)
    
    model_choice = st.selectbox("Select Model", ["Linear Regression", "Polynomial Regression"])
    
    if st.button("Train Regression Model"):
        if model_choice == "Linear Regression":
            model = reg_model.train_linear(X_train, y_train)
            type_m = 'linear'
        else:
            model = reg_model.train_polynomial(X_train, y_train, degree=2)
            type_m = 'polynomial'
            
        y_pred = reg_model.predict(X_test, model_type=type_m)
        metrics = reg_model.evaluate(y_test, y_pred)
        
        st.success("Training Complete!")
        st.json(metrics)
        
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        st.pyplot(fig)

elif choice == "Unit III: Classification (Churn)":
    st.header("Unit III: Classification - Predict Return Customer")
    
    clf_model = CustomerClassifier()
    X_train, X_test, y_train, y_test = clf_model.prepare_data(rfm)
    
    if st.button("Train Classifiers"):
        results = clf_model.train_evaluate_all(X_train, X_test, y_train, y_test)
        st.write("### Model Performance")
        st.table(pd.DataFrame(results).T)

elif choice == "Unit IV: Clustering & Rules":
    st.header("Unit IV: Clustering & Association Rules")
    
    st.subheader("K-Means Clustering (Customer Segmentation)")
    
    cluster_model = ClusterAnalysis()
    # Use scaled RFM for clustering
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    k = st.slider("Select K Clusters", 2, 6, 3)
    if st.button("Run K-Means"):
        labels, score, centers = cluster_model.kmeans_clustering(X_scaled, n_clusters=k)
        rfm['Cluster'] = labels
        st.write(f"Silhouette Score: {score:.4f}")
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(rfm['Recency'], rfm['Frequency'], rfm['Monetary'], c=labels, cmap='viridis')
        ax.set_xlabel('Recency')
        ax.set_ylabel('Frequency')
        ax.set_zlabel('Monetary')
        st.pyplot(fig)
        
    st.divider()
    st.subheader("Market Basket Analysis (Apriori)")
    st.info("Using a subset of recent transactions for speed")
    
    if st.button("Run Association Rules"):
        # Subset for speed
        subset_df = df[df['Country'] == 'United Kingdom'].tail(2000) 
        mba = MarketBasketAnalysis()
        basket = mba.prepare_basket(subset_df)
        rules = mba.run_apriori(basket, min_support=0.03, min_confidence=0.2)
        
        st.write(f"Found {len(rules)} rules")
        st.dataframe(rules.sort_values(by='lift', ascending=False).head(10))

elif choice == "Unit V: PCA & Neural Networks":
    st.header("Unit V: PCA & Neural Networks")
    
    adv_model = AdvancedModels()
    X_train, X_test, y_train, y_test = SalesRegressor().prepare_data(rfm)
    
    st.subheader("PCA Visualization")
    if st.button("Run PCA"):
        X_pca, var = adv_model.apply_pca(X_train, n_components=2)
        st.write(f"Explained Variance Ratio: {var}")
        fig, ax = plt.subplots()
        ax.scatter(X_pca[:,0], X_pca[:,1], alpha=0.5)
        st.pyplot(fig)
        
    st.subheader("Neural Network (MLP) Regressor")
    if st.button("Train MLP"):
        mlp = adv_model.train_mlp_regression(X_train, y_train)
        score = mlp.score(X_test, y_test)
        st.write(f"MLP R² Score: {score:.4f}")

elif choice == "Unit VI: Ensemble & Eval":
    st.header("Unit VI: Ensemble Learning")
    
    ens_model = EnsembleModels()
    clf_model = CustomerClassifier()
    X_train, X_test, y_train, y_test = clf_model.prepare_data(rfm)
    
    if st.button("Train Ensembles"):
        results_ens = ens_model.train_ensemble_clf(X_train, y_train)
        
        res_data = {}
        for name, model in results_ens.items():
            acc = model.score(X_test, y_test)
            res_data[name] = acc
            
        st.bar_chart(res_data)
        st.write(res_data)

