import streamlit as st

st.set_page_config(page_title="Project Report", page_icon="ℹ️", layout="wide")

st.title("📘 Project Report – AI-Based Forest Fire Risk Prediction")

st.markdown("""
## 📌 Project Overview
Forest fires are increasing worldwide due to climate change, low humidity, and extreme weather conditions.  
This system predicts **whether a forest is at risk of catching fire**, using environmental variables derived from geolocation.

The project integrates:
- Machine Learning  
- Geospatial APIs  
- Automated environmental data generation  
- User-friendly Streamlit interface  

---

## 🎯 Objectives
- Predict fire occurrence using environmental and geographical factors.  
- Allow users to input **any forest name globally**.  
- Automatically fetch coordinates of the forest.  
- Generate realistic environmental features for prediction.  
- Provide an interactive dashboard with analytics and insights.

---

## 🛠 Tech Stack
### **Languages & Tools**
- Python  
- Streamlit  
- Pandas, NumPy  
- Scikit-learn  
- Joblib  
- Requests (API Calls)

---

## 📡 APIs Used
### **1️⃣ Opencage Geocoding API**
Used to convert **forest name → latitude & longitude**  
- API Link: https://opencagedata.com/api  
- Purpose: Convert user input to geospatial coordinates

### **2️⃣ Weather-based Synthetic Feature Generator**
Instead of using paid weather APIs, the system generates:
- temperature  
- humidity  
- drought code  
- FWI (Fire Weather Index)  
- NDVI  
using mathematical transformations.

This ensures:
- Zero API cost  
- Consistent results  
- Fast predictions

---

## 📂 Dataset Description
The model is trained on a custom dataset containing:
- Temperature  
- Humidity  
- Wind Speed  
- Precipitation  
- NDVI  
- FWI Score  
- Drought Code  
- Forest Cover  
- Land Cover Class  
- Elevation  
- Slope  
- Population Density  
- Fire Occurrence (Target: 0 = No Fire, 1 = Fire)

File used:
