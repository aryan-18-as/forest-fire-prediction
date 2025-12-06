import streamlit as st

st.set_page_config(page_title="Project Report", page_icon="ℹ️", layout="wide")

st.title("📘 Project Report – AI-Based Forest Fire Risk Prediction")

st.markdown("""
## 📌 Project Overview
This project predicts forest fire risk using environmental and geographical factors derived from forest coordinates.
A machine learning model evaluates whether a forested region is likely to experience fire based on climate, vegetation, and terrain indicators.

---

## 🎯 Objectives
- Predict forest fire occurrence using ML.  
- Allow users to input any forest name globally.  
- Automatically fetch latitude & longitude using a geocoding API.  
- Generate environmental features for prediction.  
- Provide an interactive dashboard with prediction, analytics, and dataset exploration.

---

## 🛠 Technology Stack

### Languages & Libraries
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  
- Joblib  
- Requests  

### Frontend/UI Framework
- Streamlit (Multi-Page Dashboard)

---

## 📡 APIs Used

### 1️⃣ OpenCage Geocoding API
Purpose: Convert **forest name → latitude & longitude**  
API Link: https://opencagedata.com/api  

Used for fetching exact coordinates of forests worldwide.

---

## 📂 Dataset Description
The dataset used for model training contains:

- temperature_c  
- humidity_pct  
- precip_mm  
- wind_speed_m_s  
- ndvi  
- drought_code  
- fwi_score  
- forest_cover_pct  
- landcover_class  
- elevation_m  
- slope_deg  
- population_density  
- fire_occurrence (Target: 0 = No Fire, 1 = Fire)

---

## 🤖 Machine Learning Model

### Model Used: **Random Forest Classifier**

### Why Random Forest?
- High accuracy  
- Suitable for environmental/climate features  
- Handles non-linear relationships  
- Robust & low overfitting  

### Model Pipeline Contains:
- Label Encoding  
- Standard Scaling  
- Column Alignment  
- Random Forest Prediction  

Saved files include:
fire_model.pkl
scaler.pkl
encoder.pkl
feature_columns.pkl


---

## 🔁 System Workflow

User inputs forest name
↓
OpenCage API converts name → coordinates
↓
Environmental features are generated
↓
Features encoded & scaled
↓
Random Forest model predicts risk
↓
Result shown on UI (Fire / No Fire)



---

## 📊 Application Features

### 1️⃣ Fire Risk Predictor  
- Enter forest name  
- Auto-fetch coordinates  
- Map visualization  
- Fire/No Fire prediction  

### 2️⃣ EDA Analytics  
- Dataset preview  
- Correlation matrix  
- Feature statistics  

### 3️⃣ Danger Calculator  
- Manual input mode for testing values  
- Real-time prediction  

### 4️⃣ Dataset Explorer  
- Full dataset view  
- Filter, scroll & inspect  

---

## 👥 Team Members

| Name | Enrollment No. |
|------|----------------|
| **Aryan Saxena** | BETN1CS22163 |
| **Amaan Haque** | BETN1CS22100 |
| **Krishna Jain** | BETN1CS22179 |
| **Kuldeep Rana** | BETN1CS22040 |

---

## 🧑‍🏫 Faculty Guide  
**Nidhi Dandotiya**  
Department of Computer Science & Engineering

---

## 📦 Conclusion
This AI-powered system combines machine learning, geospatial intelligence, and synthetic climate modeling to create a reliable Forest Fire Risk Prediction tool.  
It is scalable, fast, and suitable for real-world environmental risk assessment applications.

""")
