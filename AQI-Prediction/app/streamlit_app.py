import streamlit as st
from components.sidebar import render_sidebar
from components.prediction_form import render_prediction_form
from components.charts import render_charts

st.set_page_config(page_title="AQI Prediction", layout="wide")

def main():
    st.title("Air Quality Index (AQI) Prediction")
    
    # Sidebar
    render_sidebar()
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Input Parameters")
        input_data = render_prediction_form()
        
    with col2:
        st.header("Analysis & Prediction")
        if input_data:
            st.write("Prediction result will appear here.")
            # TODO: Call prediction model
            render_charts()

if __name__ == "__main__":
    main()
