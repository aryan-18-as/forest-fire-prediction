import streamlit as st

st.set_page_config(page_title="Project Report", page_icon="ℹ️", layout="wide")

st.title("ℹ️ Project Overview & Report")

st.markdown("""
### 🔥 Project: AI-Based Forest Fire Risk Prediction System

**Goal:**  
Early prediction of forest fire risk based on environmental variables such as temperature, humidity, wind speed, FWI, drought code, NDVI, etc.

**Key Components:**
- Machine Learning model (trained on historical climate + fire-occurence data)
- Streamlit front-end for:
  - Automatic forest-based prediction (using OpenCage geocoding)
  - Manual danger calculator
  - Dataset & EDA pages
- Deployment on Streamlit Cloud / web for real-time access.

---

### 🧠 Model Details
- Algorithm: (RandomForest / XGBoost / whichever you used)
- Input Features: 14 (latitude, longitude, temperature_c, precip_mm, humidity_pct, wind_speed_m_s, fwi_score, drought_code, ndvi, forest_cover_pct, landcover_class_encoded, elevation_m, slope_deg, population_density)
- Target: `fire_occurrence` (0 = No Fire, 1 = Fire)

*(Yahan pe tu apne notebook / report se points copy paste kar sakta hai: accuracy, precision, recall, etc.)*

---

### 🛠 Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (ML model)
- Joblib (model persistence)
- Streamlit (web app)
- OpenCage API (geocoding)

---

### 👨‍💻 Team / Credits
- Major Project: Forest Fire Prediction  
- Developed by: **[Aryan Saxena]**  
- Guided by: **[Nidhi Dandotiya]**

""")
