import streamlit as st
import pandas as pd
import numpy as np
from components.style import metric_card

def render_charts(input_data=None):
    st.subheader("📊 Analysis Results")
    
    # Mock Prediction for display purposes
    # In a real app, 'input_data' would be passed to the model
    mock_aqi = np.random.randint(50, 300)
    
    # Determined AQI Category
    if mock_aqi <= 50:
        category, color = "Good", "green"
    elif mock_aqi <= 100:
        category, color = "Moderate", "yellow"
    elif mock_aqi <= 150:
        category, color = "Unhealthy for Sensitive Groups", "orange"
    elif mock_aqi <= 200:
        category, color = "Unhealthy", "red"
    elif mock_aqi <= 300:
        category, color = "Very Unhealthy", "purple"
    else:
        category, color = "Hazardous", "maroon"

    # Top level metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card("Predicted AQI", f"{mock_aqi}", delta=None)
    with m2:
        metric_card("Category", category, delta=None)
    with m3:
        metric_card("Confidence", "92%", delta=1.5)
    
    st.markdown("---")
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("### 📈 Pollutant Contribution")
        # Radar Chart or Bar Chart Data
        pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'Ozone']
        values = np.random.randint(10, 100, size=6)
        
        chart_data = pd.DataFrame(
            {'Pollutant': pollutants, 'Concentration': values}
        ).set_index('Pollutant')
        
        st.bar_chart(chart_data)

    with c2:
        st.markdown("### 📉 Historical Trend")
        # Line Chart
        history = pd.DataFrame(
            np.random.randn(24, 1) + 50,
            columns=['AQI']
        )
        st.line_chart(history)
