import streamlit as st

def render_sidebar():
    with st.sidebar:
        # st.image("app/assets/logo.png", use_column_width=True) # Placeholder path
        st.title("Navigation")
        st.info("This app predicts AQI based on environmental factors.")
