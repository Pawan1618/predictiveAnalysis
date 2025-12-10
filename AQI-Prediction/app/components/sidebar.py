import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.markdown("## 📊 Navigation")
        
        # You could add a logo here if available
        # st.image("app/assets/logo.png", use_column_width=True)
        
        st.markdown("---")
        
        st.markdown("### 🛠 Project Settings")
        st.selectbox("Select Model Version", ["Random Forest v1.0", "XGBoost v0.9 (Beta)"])
        st.checkbox("Show Raw Data", value=False)
        st.checkbox("Enable Dark Mode Analysis", value=True)
        
        st.markdown("---")
        
        st.markdown("### ℹ️ About")
        st.info(
            """
            **AQI Prediction App**
            
            This application estimates the Air Quality Index (AQI) based on environmental parameters.
            
            **Version:** 1.0.0
            **Developer:** Pawan
            """
        )
        
        st.markdown("---")
        st.markdown("© 2025 AQI Project")
