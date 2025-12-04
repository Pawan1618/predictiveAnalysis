import streamlit as st
import pandas as pd
import numpy as np

def render_charts():
    st.subheader("AQI Trends")
    # Dummy data
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['a', 'b', 'c'])
    st.line_chart(chart_data)
