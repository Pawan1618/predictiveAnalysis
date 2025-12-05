import os
import sys

def main():
    print("Starting AQI Prediction App...")
    os.system(f"{sys.executable} -m streamlit run app/streamlit_app.py")

if __name__ == "__main__":
    main()
