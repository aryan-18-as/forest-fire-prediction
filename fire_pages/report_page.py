import streamlit as st

def run():
    st.title("📘 Project Report")

    st.markdown("""
# **AI-Based Forest Fire Prediction System**

## **1. Introduction**
This project predicts forest fire risks using geolocation, environmental parameters, ML models, and AI explanations.

---

## **2. Dataset**
The dataset contains:
- Temperature  
- Humidity  
- Wind Speed  
- NDVI  
- FWI Score  
- Drought Code  
- Vegetation & Terrain features  

---

## **3. Methodology**
1. Data preprocessing  
2. Feature scaling  
3. Label encoding  
4. ML Model: **Random Forest Classifier**  
5. Synthetic environment generator  
6. API Integration:
   - OpenCage Geocoding
   - Groq LLaMA 3.3 AI for explanations

---

## **4. AI Features**
- Forest overview generation  
- Explanation of ML predictions  
- Fire safety recommendations  

---

## **5. Results**
- High accuracy in predicting fire risk  
- Useful explanations using LLaMA 3.3  

---

## **6. Team Members**
- **Aryan Saxena (BETN1CS22163)**  
- **Amaan Haque (BETN1CS22100)**  
- **Kuldeep Rana (BETN1CS22040)**  

**Guide:** Dr.Manali Shukla

---

## **7. Conclusion**
This system provides a complete AI + ML solution for early forest fire detection.
    """)
