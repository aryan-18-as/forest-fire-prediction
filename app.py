import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests

# ------------------------------------------------------------
# LOAD ML MODEL + SCALER + FEATURE COLUMNS
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("fire_risk_model.pkl")
    scaler = joblib.load("scaler (2).pkl")
    columns = joblib.load("feature_columns.pkl")
    return model, scaler, columns

model, scaler, feature_cols = load_model()

# ------------------------------------------------------------
# OPENCAGE GEOCODING API → FOREST NAME → LAT/LON
# ------------------------------------------------------------
def get_coordinates_from_opencage(forest_name):
    try:
        API_KEY = st.secrets["OPENCAGE_API_KEY"]
    except:
        st.error("❌ OpenCage API key missing. Add OPENCAGE_API_KEY in Streamlit Secrets.")
        st.stop()
        
    url = f"https://api.opencagedata.com/geocode/v1/json?q={forest_name}&key={API_KEY}"
    
    resp = requests.get(url)
    data = resp.json()

    if "results" not in data or len(data["results"]) == 0:
        return None, None

    lat = data["results"][0]["geometry"]["lat"]
    lon = data["results"][0]["geometry"]["lng"]
    return lat, lon

# ------------------------------------------------------------
# OPEN-METEO WEATHER API (NO KEY REQUIRED)
# ------------------------------------------------------------
def get_weather_from_openmeteo(lat, lon):
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
    )

    resp = requests.get(url).json()

    weather = {
        "temperature_c": resp["hourly"]["temperature_2m"][0],
        "humidity_pct": resp["hourly"]["relative_humidity_2m"][0],
        "wind_speed_m_s": resp["hourly"]["wind_speed_10m"][0],
        "precip_mm": resp["hourly"]["precipitation"][0],
    }
    return weather

# ------------------------------------------------------------
# STREAMLIT PAGE UI
# ------------------------------------------------------------
st.set_page_config(page_title="🔥 Forest Fire Predictor", page_icon="🔥", layout="wide")

st.title("🌍 Global Forest Fire Risk Prediction System")
st.markdown("Enter any **Forest Name** → App fetches coordinates + weather automatically → ML model predicts fire risk.")

forest_name = st.text_input("Enter Forest Name (Example: Amazon Rainforest)")

if st.button("Predict Fire Risk", use_container_width=True):
    if forest_name.strip() == "":
        st.error("Please enter a forest name.")
        st.stop()

    # STEP 1: Get coordinates
    st.info("🔍 Finding forest location...")
    lat, lon = get_coordinates_from_opencage(forest_name)

    if lat is None:
        st.error("❌ Forest not found. Try a different name.")
        st.stop()

    st.success(f"📍 Coordinates Found → Lat: {lat}, Lon: {lon}")

    # STEP 2: Get live weather
    st.info("🌦 Fetching live weather conditions...")
    weather = get_weather_from_openmeteo(lat, lon)

    # Show metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌡 Temperature", f"{weather['temperature_c']} °C")
    col2.metric("💧 Humidity", f"{weather['humidity_pct']} %")
    col3.metric("💨 Wind Speed", f"{weather['wind_speed_m_s']} m/s")
    col4.metric("🌧 Rainfall", f"{weather['precip_mm']} mm")

    # STEP 3: Prepare ML input row
    row = {
        "latitude": lat,
        "longitude": lon,
        "temperature_c": weather["temperature_c"],
        "precip_mm": weather["precip_mm"],
        "humidity_pct": weather["humidity_pct"],
        "wind_speed_m_s": weather["wind_speed_m_s"],

        # Missing values – fill 0 (your dataset allows)
        "fwi_score": 0,
        "drought_code": 0,
        "ndvi": 0,
        "forest_cover_pct": 0,
        "elevation_m": 0,
        "slope_deg": 0,
        "population_density": 0
    }

    df = pd.DataFrame([row])
    df = df[feature_cols]   # make sure correct column order

    # STEP 4: Scale + Predict
    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    # STEP 5: Display result
    st.markdown("---")

    if pred == 1:
        st.error(f"🔥 **HIGH FIRE RISK** — Probability: **{prob:.2f}**")
    else:
        st.success(f"🌧 **LOW FIRE RISK** — Probability: **{prob:.2f}**")

    st.markdown("---")
    st.caption("Powered by Machine Learning + OpenCage Geocoding API + Open-Meteo Weather API")
