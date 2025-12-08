# 🔥 AI-Based Forest Fire Prediction System

This project is an end-to-end Machine Learning and AI-based system that predicts forest fire risk using geolocation and environmental conditions. The application is deployed using **Streamlit** and provides interactive dashboards, analytics, and automated fire-risk forecasting.

---

## 🌲 Project Overview
The system predicts whether a forest area is at **risk of fire** based on:
- Location (latitude & longitude)
- Temperature
- Humidity
- Wind Speed
- Vegetation Index (NDVI)
- Drought Conditions
- FWI Score
- Landcover Category
- Population Exposure

Environmental values are generated using geolocation-based functions and processed through an ML pipeline.

---

## 🚀 Key Components
### ✅ **Fire Prediction Dashboard**
- Enter any forest name (e.g., “Amazon”, “Sundarbans”)
- Coordinates fetched using **OpenCage Geocoding API**
- Model predicts fire risk (High / Low)
- Displays:
  - Location Map  
  - Key environmental metrics  
  - Fire risk outcome  

### 📊 **EDA Analytics**
- View dataset insights  
- Correlation heatmaps  
- Feature distributions  
- Summary statistics  

### 🌡 **Danger Calculator**
- Users manually input environment parameters  
- Model outputs instant prediction  

### 🗂 **Dataset Explorer**
- Explore dataset used for model training  
- Sort, filter, preview data  

### 📘 **Project Report Page**
Contains:
- Abstract  
- Problem definition  
- Data description  
- ML pipeline  
- Results  

---

## 🧠 Machine Learning
### Algorithm:
- **Random Forest Classifier**

### ML Pipeline:
- Label Encoding  
- Scaling with StandardScaler  
- Feature selection via `feature_columns_1.pkl`

Model & preprocessing files used:
- `fire_model.pkl`  
- `scaler.pkl`  
- `encoder.pkl`  
- `feature_columns.pkl`

---

## 👥 Team
- **Aryan Saxena (BETN1CS22163)**  
- **Amaan Haque (BETN1CS22100)**  
- **Kuldeep Rana (BETN1CS22040)**  

### Guide:
- **Nidhi Dandotiya**

---

## 🌐 Technologies Used
- Python  
- Streamlit  
- Scikit-Learn  
- Pandas / NumPy  
- OpenCage API  

---

