import streamlit as st
from components.sidebar import render_sidebar
from components.prediction_form import render_prediction_form
from components.charts import render_charts
from components.style import apply_custom_css

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="AQI Prediction Pro",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Apply custom CSS
    apply_custom_css()
    
    st.title("🌬️ Air Quality Index (AQI) Prediction")
    st.markdown("Predict air quality levels in real-time using advanced machine learning models.")
    st.markdown("---")
    
    # Sidebar
    render_sidebar()
    
    # Layout using Tabs or Columns
    # Here we use columns but tabs could also be nice "Prediction" | "Historical Data"
    
    # Main content
    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        # Input Section
        input_data = render_prediction_form()
        
    with col2:
        # Results Section
        # Only show results if data is submitted, or show a placeholder/demo
        if input_data:
            render_charts(input_data)
        else:
            # Placeholder content when no prediction has been made yet
            st.info("👈 Please adjust the parameters in the form and click 'Predict AQI' to see the analysis.")
            st.image("https://images.unsplash.com/photo-1623227413713-356a62372c3d?q=80&w=2670&auto=format&fit=crop", 
                     caption="Clean Air Matters", use_column_width=True)

if __name__ == "__main__":
    main()
