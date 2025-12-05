import streamlit as st
import requests
import joblib
import pandas as pd
import numpy as np

# Load model & scaler
model = joblib.load("fire_risk_model.pkl")
scaler = joblib.load("scaler (2).pkl")
feature_cols = joblib.load("feature_columns.pkl")

# Predefined forest coordinates
FORESTS = {
    "Amazon Rainforest": (-3.4653, -62.2159),
    "Sundarbans": (21.9497, 89.1833),
    "Jim Corbett": (29.5300, 78.7740),
    "Kaziranga": (26.5775, 93.1711),
    "Black Forest": (48.0793, 8.2070),
    "Borneo Rainforest": (1.6120, 113.5329),
    "California Yosemite": (37.8651, -119.5383)
}

# Fetch weather from Open-Meteo
def get_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
    r = requests.get(url).json()
    
    return {
        "temperature_c": r["hourly"]["temperature_2m"][0],
        "humidity_pct": r["hourly"]["relative_humidity_2m"][0],
        "wind_speed_m_s": r["hourly"]["wind_speed_10m"][0],
        "precip_mm": r["hourly"]["precipitation"][0]
    }

# Streamlit UI
st.set_page_config(page_title="Global Forest Fire Predictor", page_icon="🔥", layout="wide")

st.title("🔥 Global Forest Fire Risk Prediction System")
st.markdown("Enter a forest name — data will auto-fetch from live weather APIs!")

forest = st.selectbox("Select Forest Region", list(FORESTS.keys()))

if st.button("Predict Fire Risk"):
    lat, lon = FORESTS[forest]

    st.info(f"Fetching live environmental data for **{forest}**...")

    w = get_weather(lat, lon)

    # Create input row
    row = {
        "latitude": lat,
        "longitude": lon,
        "temperature_c": w["temperature_c"],
        "precip_mm": w["precip_mm"],
        "humidity_pct": w["humidity_pct"],
        "wind_speed_m_s": w["wind_speed_m_s"],
        "fwi_score": 0,  
        "drought_code": 0,
        "ndvi": 0,
        "forest_cover_pct": 0,
        "elevation_m": 0,
        "slope_deg": 0,
        "population_density": 0
    }

    df = pd.DataFrame([row])

    # Ensure same order of columns
    df = df[feature_cols]

    # Scale
    df_scaled = scaler.transform(df)

    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    if pred == 1:
        st.error(f"🔥 HIGH FIRE RISK — Probability: **{prob:.2f}**")
    else:
        st.success(f"🌧 LOW FIRE RISK — Probability: **{prob:.2f}**")
