import streamlit as st

def run():
    st.title("📘 Project Report")

    st.markdown("""

# **AI-Based Forest Fire Prediction System**
A complete end–to–end Machine Learning + Artificial Intelligence–powered system designed to **predict forest fire risk**, **analyze environmental factors**, and **generate intelligent explanations** using the latest **Groq LLaMA 3.3 AI model**.

---

# **1. Introduction**
Forest fires pose a major environmental and economic threat worldwide.  
To tackle this problem, this project aims to:

- Predict fire risk based on geolocation  
- Analyze environmental features  
- Provide an intelligent explanation using AI  
- Assist forest officials with actionable recommendations  

This system combines **Machine Learning**, **Geospatial APIs**, **Data Analytics**, and **AI-based reasoning** to create an advanced fire-risk prediction tool.

---

# **2. Dataset Description**
The dataset used (fire_dataset.csv) contains essential wildfire-related environmental variables:

### **Environmental Features**
- **Temperature (°C)**  
- **Humidity (%)**  
- **Wind Speed (m/s)**  
- **Precipitation (mm)**  
- **NDVI (Normalized Difference Vegetation Index)**  
- **FWI (Fire Weather Index)**  
- **Drought Code**

### **Geographic & Terrain Features**
- Latitude  
- Longitude  
- Elevation  
- Slope  
- Forest Cover  
- Landcover Type  
- Population Density  

This dataset forms the foundation of the ML prediction model.

---

# **3. System Architecture**
The system consists of four major modules:

### **1️⃣ Data Input Layer**
- Forest name is entered by the user  
- Geolocation fetched through **OpenCage Geocoding API**  
- Synthetic weather & environmental values generated for prediction  

### **2️⃣ Machine Learning Module**
- Preprocessing with Scikit-learn  
- Feature scaling and ordering  
- Label encoding  
- Prediction using **Random Forest Classifier**

### **3️⃣ AI Reasoning Module**
Powered by **Groq LLaMA 3.3 (latest stable model)**:
- Automatically generates forest overview  
- Explains why fire risk is High/Low  
- Provides safety recommendations  

### **4️⃣ User Interface Layer**
Built using **Streamlit**:
- Interactive dashboard  
- EDA visualizations  
- Dataset explorer  
- Fire danger calculator  
- Detailed project report page

---

# **4. Methodology (Step-by-Step)**

### **Step 1: Data Preprocessing**
- Handling missing values  
- Scaling numerical features  
- Encoding categorical variables  
- Extracting feature_columns for correct model input  

### **Step 2: Model Training**
- Algorithm: **Random Forest Classifier**  
- Train-test split applied  
- Multiple iterations conducted to optimize accuracy  

### **Step 3: Environment Simulation**
A synthetic environment generator creates values such as:
- Temperature  
- Humidity  
- NDVI  
- Wind Speed  
- Drought Code  

This ensures **predictions work even without a real-time API**.

### **Step 4: API Integration**
- **OpenCage API** → Converts forest name → Latitude/Longitude  
- **Groq AI API (LLaMA 3.3)** → Generates intelligent explanations  

### **Step 5: Prediction & Explanation**
1. ML model predicts **High/Low** fire risk  
2. AI explains:
   - Which features influenced prediction  
   - Why the model reached the conclusion  
   - Recommendations to prevent fire  

---

# **5. Key AI Features Added**
This project integrates real Artificial Intelligence through Groq:

### ✔ **AI Forest Overview**
Summarizes climate, vegetation, geography, and fire behavior of selected forest.

### ✔ **AI Explanation of Prediction**
Explains why the ML model predicted High/Low fire risk in human language.

### ✔ **AI Safety Recommendations**
Provides 5 actionable prevention steps customized for the predicted fire risk level.

These features help the student explain the logic clearly during project evaluation.

---

# **6. Results & Discussion**
### **Highlights**
- Highly accurate predictions from Random Forest  
- Smooth UI with interactive visuals  
- AI reports easy-to-understand summaries  
- Works for any forest globally  
- No dependence on real-time weather data  

### **Outcome**
The system successfully demonstrates:
- Machine Learning modeling  
- AI reasoning  
- Geolocation processing  
- Real-time visualization  
- Full-stack deployment using Streamlit Cloud  

---

# **7. Limitations**
- Synthetic weather generator (not real-time weather)  
- Fire prediction depends on forest name accuracy  
- Remote areas may not return coordinates  

---

# **8. Future Enhancements**
- Integrate real weather APIs (OpenWeather, NASA Fire API)  
- Deep Learning models (LSTM, GRU)  
- Multi-forest batch predictions  
- Mobile app version  
- Fire spread simulation model  

---

# **9. Team Members**
- **Aryan Saxena (BETN1CS22163)**  
- **Amaan Haque (BETN1CS22100)**  
- **Kuldeep Rana (BETN1CS22040)**  

### **Project Guide**
**Dr. Manali Shukla**

---

# **10. Conclusion**
This project delivers a complete **AI + ML-based early fire detection system**.  
It not only predicts fire risk but also uses artificial intelligence to provide:

- Context  
- Explanation  
- Recommendations  

This makes the system highly useful for **academics, researchers, and forest authorities**.

---

""")
