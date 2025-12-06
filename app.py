import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Forest Fire Predictor",
    layout="wide",
    page_icon="🔥"
)

# ---------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------
st.sidebar.title("📂 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🔥 Fire Risk Predictor",
        "📊 EDA Analytics",
        "🌡 Danger Calculator",
        "🗂 Dataset Explorer",
        "ℹ️ Project Report",
    ]
)

# ---------------------------------------------------
# CUSTOM CSS (your premium UI styles preserved)
# ---------------------------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 48px;
    font-weight: 900;
    text-align: center;
    padding: 18px;
    background: linear-gradient(90deg, #ff512f, #dd2476);
    -webkit-background-clip: text;
    color: transparent;
}
.card {
    padding: 20px;
    background: white;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    margin-bottom: 20px;
}
.pred-box {
    font-size: 30px;
    font-weight: bold;
    text-align: center;
    padding: 25px;
    border-radius: 12px;
    margin-top: 20px;
}
[data-testid="stSidebar"] {
    background: #1e1e1e;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# API KEY
# ---------------------------------------------------
OPENCAGE_API_KEY = "95df23a7370340468757cad17a479691"

# ---------------------------------------------------
# LOAD MODEL FILES
# ---------------------------------------------------
@st.cache_resource
def load_all():
    model = joblib.load("fire_model.pkl")
    scaler = joblib.load("scaler (2).pkl")
    encoder = joblib.load("encoder.pkl")
    feature_cols = joblib.load("feature_columns_1.pkl")
    return model, scaler, encoder, feature_cols

model, scaler, encoder, feature_cols = load_all()

# ---------------------------------------------------
# GEOCODING
# ---------------------------------------------------
def geocode_forest(name):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={name}&key={OPENCAGE_API_KEY}"
    r = requests.get(url).json()
    if r["total_results"] == 0:
        return None, None
    return r["results"][0]["geometry"]["lat"], r["results"][0]["geometry"]["lng"]

# ---------------------------------------------------
# ENVIRONMENT DATA GENERATOR
# ---------------------------------------------------
def generate_environment(lat, lon):
    temperature = 20 + abs(lat % 10)
    humidity = 40 + abs(lon % 20)
    wind_speed = 2 + (abs(lat + lon) % 5)
    precip = abs(lat - lon) % 3

    ndvi = np.clip(humidity / 100 - 0.3, 0, 1)
    fwi = wind_speed * (1 - humidity / 100) * 25
    drought_code = max(20, (temperature * 2) - precip)

    return pd.DataFrame([{
        "latitude": lat,
        "longitude": lon,
        "temperature_c": temperature,
        "precip_mm": precip,
        "humidity_pct": humidity,
        "wind_speed_m_s": wind_speed,
        "fwi_score": fwi,
        "drought_code": drought_code,
        "ndvi": ndvi,
        "forest_cover_pct": 70,
        "landcover_class": "Deciduous Forest",
        "elevation_m": 300,
        "slope_deg": 12,
        "population_density": 18
    }])


# ===================================================
# PAGE 1: MAIN FIRE RISK PREDICTOR
# ===================================================
if page == "🔥 Fire Risk Predictor":

    st.markdown("<div class='main-title'>🔥 AI-Based Forest Fire Risk Predictor</div>", unsafe_allow_html=True)

    forest_name = st.text_input("🌲 Enter Forest Name", "Amazon", placeholder="Enter any Forest Name…")

    if st.button("🔍 Predict Fire Risk", use_container_width=True):

        lat, lon = geocode_forest(forest_name)

        if lat is None:
            st.error("Forest not found.")
            st.stop()

        df = generate_environment(lat, lon)

        # Encode
        try:
            df["landcover_class_encoded"] = encoder.transform(df["landcover_class"])
        except:
            df["landcover_class_encoded"] = encoder.transform(["Deciduous Forest"])

        df = df.drop(columns=["landcover_class"])
        df = df.reindex(columns=feature_cols)

        df_scaled = scaler.transform(df)
        pred = model.predict(df_scaled)[0]

        # MAP
        st.subheader("📍 Forest Location")
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))

        # CARDS
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Temperature", f"{df.temperature_c.values[0]} °C")
        col2.metric("Humidity", f"{df.humidity_pct.values[0]} %")
        col3.metric("Wind", f"{df.wind_speed_m_s.values[0]} m/s")
        col4.metric("NDVI", round(df.ndvi.values[0], 2))
        col5.metric("FWI Score", round(df.fwi_score.values[0], 2))

        # RESULT BOX
        if pred == 1:
            st.markdown(
                "<div class='pred-box' style='background:#ffcccc; color:#b30000;'>🔥 HIGH FIRE RISK</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='pred-box' style='background:#ccffcc; color:#006600;'>🌿 LOW / NO FIRE RISK</div>",
                unsafe_allow_html=True
            )

        # INPUT DATA
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Input Environmental Data")
        st.json(df.to_dict(orient="records")[0])
        st.markdown("</div>", unsafe_allow_html=True)


# ===================================================
# PAGE 2: EDA ANALYTICS
# ===================================================
elif page == "📊 EDA Analytics":
    st.title("📊 EDA & Analytics")
    try:
        df = pd.read_csv("fire_dataset.csv")
        st.dataframe(df, use_container_width=True)

        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 1:
            st.subheader("Correlation Matrix")
            st.dataframe(df[num_cols].corr(), use_container_width=True)
    except:
        st.error("Dataset 'fire_dataset.csv' is missing.")


# ===================================================
# PAGE 3: MANUAL DANGER CALCULATOR
# ===================================================
elif page == "🌡 Danger Calculator":
    st.title("🌡 Manual Danger Calculator")

    temp = st.slider("Temperature (°C)", 0, 50, 25)
    hum = st.slider("Humidity (%)", 0, 100, 50)
    wind = st.slider("Wind Speed (m/s)", 0, 20, 5)
    precip = st.slider("Precipitation (mm)", 0, 10, 2)
    ndvi = st.slider("NDVI", 0.0, 1.0, 0.4)
    fwi = st.slider("FWI Score", 0.0, 100.0, 10.0)
    drought = st.slider("Drought Code", 0, 800, 50)

    if st.button("Calculate Risk"):
        df = pd.DataFrame([{
            "latitude": 10,
            "longitude": 20,
            "temperature_c": temp,
            "precip_mm": precip,
            "humidity_pct": hum,
            "wind_speed_m_s": wind,
            "fwi_score": fwi,
            "drought_code": drought,
            "ndvi": ndvi,
            "forest_cover_pct": 70,
            "landcover_class": "Deciduous Forest",
            "elevation_m": 300,
            "slope_deg": 12,
            "population_density": 18
        }])

        df["landcover_class_encoded"] = encoder.transform(["Deciduous Forest"])
        df = df.drop(columns=["landcover_class"])
        df = df.reindex(columns=feature_cols)

        pred = model.predict(scaler.transform(df))[0]

        if pred == 1:
            st.error("High Fire Risk")
        else:
            st.success("Low / No Fire Risk")


# ===================================================
# PAGE 4: DATASET EXPLORER
# ===================================================
elif page == "🗂 Dataset Explorer":
    st.title("🗂 Dataset Explorer")
    try:
        df = pd.read_csv("fire_dataset.csv")
        st.dataframe(df, use_container_width=True)
    except:
        st.error("Dataset missing.")


# ===================================================
# PAGE 5: PROJECT REPORT
# ===================================================
elif page == "ℹ️ Project Report":
    st.title("ℹ️ Project Report")
    st.markdown("""
### Overview  
This project predicts forest fire risk using environmental factors derived from forest coordinates.

### Workflow  
- Location → Environment Data  
- Encoding & Scaling  
- Model Prediction  

### Output  
Binary classification: **Fire / No Fire**.
""")
