import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

# -------------------------------
# 🔑 API KEY (direct embed)
# -------------------------------
OPENCAGE_API_KEY = "95df23a7370340468757cad17a479691"

# -------------------------------
# 📌 Load ML components
# -------------------------------
@st.cache_resource
def load_components():
    model = joblib.load("fire_model.pkl")
    scaler = joblib.load("scaler (2).pkl")
    encoder = joblib.load("encoder.pkl")
    return model, scaler, encoder

model, scaler, encoder = load_components()

# -------------------------------
# 📌 FINAL Correct Column Order
# -------------------------------
FEATURE_COLS = [
    "latitude",
    "longitude",
    "temperature_c",
    "precip_mm",
    "humidity_pct",
    "wind_speed_m_s",
    "fwi_score",
    "drought_code",
    "ndvi",
    "forest_cover_pct",
    "landcover_class_encoded",
    "elevation_m",
    "slope_deg",
    "population_density"
]

# --------------------------------------------------
# 🌍 Function to get location (lat/lon) from forest name
# --------------------------------------------------
def geocode_forest(forest_name):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={forest_name}&key={OPENCAGE_API_KEY}"
    response = requests.get(url).json()

    try:
        lat = response["results"][0]["geometry"]["lat"]
        lon = response["results"][0]["geometry"]["lng"]
        return lat, lon
    except:
        return None, None

# --------------------------------------------------
# 🌦️ Fetch environmental values from Open-Meteo API
# --------------------------------------------------
def fetch_environment_data(lat, lon):

    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,precipitation,relativehumidity_2m,windspeed_10m"
    )
    weather = requests.get(weather_url).json()

    return {
        "temperature_c": weather["hourly"]["temperature_2m"][0],
        "precip_mm": weather["hourly"]["precipitation"][0],
        "humidity_pct": weather["hourly"]["relativehumidity_2m"][0],
        "wind_speed_m_s": weather["hourly"]["windspeed_10m"][0]
    }


# --------------------------------------------------
# 🌲 Static environmental variables (fallback)
# --------------------------------------------------
def get_static_defaults():
    return {
        "fwi_score": 15,
        "drought_code": 80,
        "ndvi": 0.55,
        "forest_cover_pct": 70,
        "landcover_class": "Forest",
        "elevation_m": 300,
        "slope_deg": 10,
        "population_density": 20
    }


# -------------------------------
# 🎨 Streamlit UI
# -------------------------------
st.title("🔥 AI Based Forest Fire Risk Predictor (Final Stable Version)")
forest_name = st.text_input("🌲 Enter Forest Name", placeholder="Amazon, Sundarbans, Gir, etc.")

if st.button("Predict Fire Risk"):

    # --------------------------------------
    # 1️⃣ Get coordinates of the forest
    # --------------------------------------
    lat, lon = geocode_forest(forest_name)

    if lat is None:
        st.error("❌ Forest name not found. Try another name.")
        st.stop()

    # --------------------------------------
    # 2️⃣ Download weather conditions
    # --------------------------------------
    weather = fetch_environment_data(lat, lon)

    # --------------------------------------
    # 3️⃣ Load static defaults
    # --------------------------------------
    defaults = get_static_defaults()

    # Combine all features
    df = pd.DataFrame([{
        "latitude": lat,
        "longitude": lon,
        **weather,
        **defaults
    }])

    # --------------------------------------
    # 4️⃣ Encode categorical values
    # --------------------------------------
    df["landcover_class_encoded"] = encoder.transform(df["landcover_class"])

    df = df.drop(columns=["landcover_class"])

    # --------------------------------------
    # 5️⃣ Reorder correctly (THE FIX)
    # --------------------------------------
    df = df.reindex(columns=FEATURE_COLS)

    # --------------------------------------
    # 6️⃣ Scale numeric data
    # --------------------------------------
    df_scaled = scaler.transform(df)

    # --------------------------------------
    # 7️⃣ Predict
    # --------------------------------------
    result = model.predict(df_scaled)[0]

    # --------------------------------------
    # 8️⃣ Final output YES / NO
    # --------------------------------------
    if result == 1:
        st.error("🔥 YES – High Forest Fire Risk Detected!")
    else:
        st.success("🌿 NO – Forest Fire Risk Not Detected")


# -------------------------------
# Debug (optional—hide later)
# -------------------------------
# st.write("Input DF:", df)
# st.write("Columns:", df.columns.tolist())
