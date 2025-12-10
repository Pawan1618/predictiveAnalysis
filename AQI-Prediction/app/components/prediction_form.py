import streamlit as st

def render_prediction_form():
    """
    Renders the input form for AQI prediction.
    Returns a dictionary of input values if submitted, else None.
    """
    with st.form("prediction_form"):
        st.subheader("🌍 Environmental Parameters")
        
        # Organize inputs into columns for a cleaner layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, max_value=1000.0, value=35.0, help="Particulate Matter < 2.5 micrometers")
            so2 = st.number_input("SO2 (ppb)", min_value=0.0, max_value=500.0, value=10.0, help="Sulfur Dioxide")
            temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, value=25.0)

        with col2:
            pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, max_value=1000.0, value=65.0, help="Particulate Matter < 10 micrometers")
            co = st.number_input("CO (ppm)", min_value=0.0, max_value=50.0, value=1.0, help="Carbon Monoxide")
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)

        with col3:
            no2 = st.number_input("NO2 (ppb)", min_value=0.0, max_value=500.0, value=20.0, help="Nitrogen Dioxide")
            ozone = st.number_input("Ozone (ppb)", min_value=0.0, max_value=500.0, value=45.0, help="Ozone")
            wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=100.0, value=5.0)
            
        st.markdown("---")
        
        # Submit Button with full width utilizing custom CSS class if needed
        submitted = st.form_submit_button("🚀 Predict AQI", type="primary")
        
        if submitted:
            return {
                "PM2.5": pm25,
                "PM10": pm10,
                "NO2": no2,
                "SO2": so2,
                "CO": co,
                "Ozone": ozone,
                "Temperature": temperature,
                "Humidity": humidity,
                "Wind Speed": wind_speed
            }
    return None
