import streamlit as st

def apply_custom_css():
    st.markdown("""
        <style>
        /* Main Container Styling */
        .reportview-container {
            background: #f0f2f6;
        }
        
        /* Headers */
        h1, h2, h3 {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            color: #2c3e50;
        }
        
        h1 {
            font-weight: 700;
            padding-bottom: 20px;
            border-bottom: 2px solid #e1e4e8;
            margin-bottom: 30px;
        }

        /* Card Styling for Metrics */
        .metric-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-5px);
        }
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-label {
            font-size: 1rem;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Sidebar Styling */
        .css-1d391kg {
            padding-top: 2rem;
        }
        
        /* Button Styling */
        .stButton>button {
            background-color: #2c3e50;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #34495e;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        /* Input Fields */
        .stNumberInput > div > div > input {
            border-radius: 5px;
        }

        /* Expander Styling */
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
            border-radius: 5px;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

def metric_card(label, value, delta=None):
    delta_html = ""
    if delta:
        color = "#27ae60" if delta > 0 else "#c0392b"
        arrow = "↑" if delta > 0 else "↓"
        delta_html = f'<div style="color: {color}; font-size: 0.9rem; margin-top: 5px;">{arrow} {abs(delta)}%</div>'
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)
