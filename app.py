import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib

st.set_page_config(page_title="Forest Fire Predictor", layout="centered")

# -------------------------------
# API KEY (Only OpenCage Needed)
# -------------------------------
OPENCAGE_API_KEY = "95df23a7370340468757cad17a479691"   # Your key

# -------------------------------
# Load ML Files
# -------------------------------
@st.cache_resource
def load_all():
    model = joblib.load("fire_model.pkl")
    scaler = joblib.load("scaler (2).pkl")
    encoder = joblib.load("encoder.pkl")
    feature_cols = joblib.load("feature_columns_1.pkl")
    return model, scaler, encoder, feature_cols

model, scaler, encoder, feature_cols = load_all()

# -------------------------------
# 1️⃣ Get coordinates
# -------------------------------
def geocode_forest(forest_name):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={forest_name}&key={OPENCAGE_API_KEY}"
    r = requests.get(url).json()

    if r["total_results"] == 0:
        return None, None

    lat = r["results"][0]["geometry"]["lat"]
    lon = r["results"][0]["geometry"]["lng"]
    return lat, lon

# -------------------------------
# 2️⃣ Generate environment data (NO API)
# -------------------------------
def generate_environment(lat, lon):
    # safe static & math-based values
    temperature = 20 + abs(lat % 10)         # 20–30
    humidity = 40 + abs(lon % 20)            # 40–60
    wind_speed = 2 + (abs(lat + lon) % 5)    # 2–7
    precip = (abs(lat - lon) % 3)            # 0–3

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
        "landcover_class": "Deciduous Forest",   # Encoder-safe
        "elevation_m": 300,
        "slope_deg": 12,
        "population_density": 18
    }])

# -------------------------------
# UI
# -------------------------------
st.title("🔥 AI Forest Fire Prediction (Final Version)")
forest_name = st.text_input("🌲 Enter Forest Name", "Amazon")

if st.button("Predict Fire Risk"):

    lat, lon = geocode_forest(forest_name)
    if lat is None:
        st.error("Forest not found. Try another name.")
        st.stop()

    df = generate_environment(lat, lon)

    # Encode landcover
    try:
        df["landcover_class_encoded"] = encoder.transform(df["landcover_class"])
    except:
        df["landcover_class_encoded"] = encoder.transform(["Deciduous Forest"])

    df = df.drop(columns=["landcover_class"])

    # Correct Column Order
    df = df.reindex(columns=feature_cols)

    # Scale
    df_scaled = scaler.transform(df)

    # Predict
    pred = model.predict(df_scaled)[0]

    if pred == 1:
        st.error("🔥 YES — High Forest Fire Risk Detected")
    else:
        st.success("🌿 NO — Fire Risk Not Detected")

    st.subheader("📊 Input Data Used")
    st.json(df.to_dict(orient="records"))


give you all code please improve it's ui-ux make it look beautiful 
