import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Fire Danger Calculator", page_icon="🌡", layout="centered")

st.title("🌡 Fire Danger Calculator (Manual Mode)")

@st.cache_resource
def load_all():
    model = joblib.load("fire_model.pkl")
    scaler = joblib.load("scaler (2).pkl")
    encoder = joblib.load("encoder.pkl")
    feature_cols = joblib.load("feature_columns_1.pkl")
    return model, scaler, encoder, feature_cols

model, scaler, encoder, feature_cols = load_all()

col1, col2 = st.columns(2)

with col1:
    temp = st.slider("Temperature (°C)", 5.0, 50.0, 28.0)
    humidity = st.slider("Humidity (%)", 5, 100, 55)
    wind = st.slider("Wind Speed (m/s)", 0.0, 25.0, 5.0)
    precip = st.slider("Precipitation (mm)", 0.0, 20.0, 2.0)

with col2:
    ndvi = st.slider("NDVI", 0.0, 1.0, 0.5)
    fwi = st.slider("FWI Score", 0.0, 100.0, 30.0)
    drought = st.slider("Drought Code", 0.0, 300.0, 80.0)
    pop = st.slider("Population Density", 0, 500, 20)

if st.button("🔍 Calculate Fire Risk"):
    df = pd.DataFrame([{
        "latitude": 0.0,
        "longitude": 0.0,
        "temperature_c": temp,
        "precip_mm": precip,
        "humidity_pct": humidity,
        "wind_speed_m_s": wind,
        "fwi_score": fwi,
        "drought_code": drought,
        "ndvi": ndvi,
        "forest_cover_pct": 70,
        "landcover_class": "Deciduous Forest",
        "elevation_m": 300,
        "slope_deg": 10,
        "population_density": pop
    }])

    try:
        df["landcover_class_encoded"] = encoder.transform(df["landcover_class"])
    except Exception:
        df["landcover_class_encoded"] = encoder.transform(["Deciduous Forest"])

    df = df.drop(columns=["landcover_class"])
    df = df.reindex(columns=feature_cols)
    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)[0]

    if pred == 1:
        st.error("🔥 HIGH FIRE RISK (manual calculation)")
    else:
        st.success("🌿 LOW / NO FIRE RISK (manual calculation)")

    st.json(df.to_dict(orient="records")[0])
