import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib

# ==========================================
# 🌟 Page Config
# ==========================================
st.set_page_config(
    page_title="AI Forest Fire Predictor",
    page_icon="🔥",
    layout="centered"
)

# ==========================================
# 🌟 Custom CSS for Modern UI
# ==========================================
st.markdown("""
    <style>
        .main-title {
            font-size: 42px;
            font-weight: bold;
            text-align: center;
            color: white;
            background: linear-gradient(90deg, #ff512f, #dd2476);
            padding: 18px;
            border-radius: 10px;
        }
        .sub-card {
            background-color: #ffffff;
            padding: 18px;
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
            margin-top: 20px;
        }
        .prediction-box {
            padding: 25px;
            border-radius: 12px;
            font-size: 26px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 🌟 Title
# ==========================================
st.markdown("<div class='main-title'>🔥 AI Forest Fire Risk Predictor</div>", unsafe_allow_html=True)

# ==========================================
# 🔐 API KEY
# ==========================================
OPENCAGE_API_KEY = "95df23a7370340468757cad17a479691"

# ==========================================
# 📦 Load ML Artifacts
# ==========================================
@st.cache_resource
def load_all():
    model = joblib.load("fire_model.pkl")
    scaler = joblib.load("scaler (2).pkl")
    feature_cols = joblib.load("feature_columns_1.pkl")
    return model, scaler, feature_cols

model, scaler, feature_cols = load_all()

# ==========================================
# 📍 Geocode Forest Name → Coordinates
# ==========================================
def geocode_forest(name):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={name}&key={OPENCAGE_API_KEY}"
    r = requests.get(url).json()
    if r["total_results"] == 0:
        return None, None
    return r["results"][0]["geometry"]["lat"], r["results"][0]["geometry"]["lng"]

# ==========================================
# 🌦 Generate Environment Data (No API)
# ==========================================
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

# ==========================================
# 🌳 Input Card
# ==========================================
st.markdown("<div class='sub-card'>", unsafe_allow_html=True)
st.subheader("🌲 Enter Forest Name for Prediction")
forest_name = st.text_input("Forest Name", "Amazon")
st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 🔮 Predict Button
# ==========================================
if st.button("🔍 Predict Fire Risk", use_container_width=True):

    # Get coordinates
    lat, lon = geocode_forest(forest_name)
    if lat is None:
        st.error("❌ Forest not found. Try another name.")
        st.stop()

    # Generate environmental data
    df = generate_environment(lat, lon)

    # Using raw landcover_class (NO encoding needed)
    df = df.reindex(columns=feature_cols)

    # Scale
    df_scaled = scaler.transform(df)

    # Predict
    pred = model.predict(df_scaled)[0]

    # =======================
    # 🎨 Result UI
    # =======================
    if pred == 1:
        st.markdown(
            "<div class='prediction-box' style='background:#ffcccc; color:#b30000;'>🔥 YES — High Fire Risk Detected!</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='prediction-box' style='background:#ccffcc; color:#006600;'>🌿 NO — Fire Risk Not Detected</div>",
            unsafe_allow_html=True
        )

    # Show data card
    st.markdown("<div class='sub-card'>", unsafe_allow_html=True)
    st.subheader("📊 Environmental Data Used")
    st.json(df.to_dict(orient="records")[0])
    st.markdown("</div>", unsafe_allow_html=True)
