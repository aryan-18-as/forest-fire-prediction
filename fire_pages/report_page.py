import streamlit as st

def run():

    st.title("📘 Project Report")

    st.markdown("""
    ## AI-Based Forest Fire Prediction System  
    *A Machine Learning Project for Predicting Forest Fire Risk*

    ### 👨‍💻 Team Members  
    - **Aryan Saxena (BETN1CS22163)**
    - **Amaan Haque (BETN1CS22100)**
    - **Kuldeep Rana (BETN1CS22040)**

    **Guide:** *Ms. Nidhi Dandotiya*

    ---

    ## 📂 1. Project Overview

    Forest fires cause massive ecological and economic damage.  
    This project predicts the probability of a fire occurring in a forest using ML.

    Instead of manually entering parameters, the system automatically fetches:

    - Temperature  
    - Humidity  
    - Wind Speed  
    - NDVI  
    - Drought Index  
    - Forest Cover  
    - Geographic Elevation  
    - Landcover Class  

    Based on these environmental features, the model predicts **fire or no fire**.

    ---

    ## 🔧 2. Methodology

    ### **2.1 Data Processing**
    - Missing values handled  
    - Categorical label encoding  
    - Feature scaling  
    - Correlation analysis  

    ### **2.2 Model Training**
    The following models were tested:

    - Random Forest  
    - XGBoost  
    - Logistic Regression  
    - Gradient Boosting  

    **Random Forest** achieved the best accuracy and was selected.

    ---

    ## 🌐 3. External API Used

    ### **OpenCage Geocoding API**  
    - Converts forest names into coordinates  
    - Used to generate environmental parameters  

    No other external API required (OpenWeather was removed to reduce errors).

    ---

    ## 🖥 4. Application Workflow

    1. User enters forest name  
    2. App fetches latitude & longitude  
    3. Auto-generates environmental features  
    4. Applies encoding + feature scaling  
    5. ML model predicts fire probability  
    6. Shows location on map and risk level  

    ---

    ## 📌 5. Key Features

    - Fully automated prediction  
    - EDA dashboard  
    - Custom danger calculator  
    - Dataset explorer  
    - Responsive UI  
    - Professional multipage architecture  

    ---

    ## 📦 6. Files Included

    - `app.py`  
    - `fire_model.pkl`  
    - `scaler (2).pkl`  
    - `encoder.pkl`  
    - `feature_columns_1.pkl`  
    - `fire_dataset.csv`  
    - `pages/` folder 

    ---

    ## 🏁 Conclusion

    The system demonstrates how ML can support wildfire monitoring by providing
    early warning signals based on environmental indicators.

    """)
