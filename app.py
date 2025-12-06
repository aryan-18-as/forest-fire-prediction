import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Forest Fire Predictor",
    page_icon="🔥",
    layout="wide",
)

# -------------------------------------------------
# OPEN CAGE KEY
# -------------------------------------------------
OPENCAGE_API_KEY = "95df23a7370340468757cad17a479691"

# -------------------------------------------------
# LOAD ML ARTIFACTS
# -------------------------------------------------
@st.cache_resource
def load_all():
    model = joblib.load("fire_model.pkl")
    scaler = joblib.load("scaler (2).pkl")
    encoder = joblib.load("encoder.pkl")
    feature_cols = joblib.load("feature_columns_1.pkl")
    return model, scaler, encoder, feature_cols

model, scaler, encoder, feature_cols = load_all()

# -------------------------------------------------
# GEOCODING
# -------------------------------------------------
def geocode_forest(forest):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={forest}&key={OPENCAGE_API_KEY}"
    r = requests.get(url).json()
    if r["total_results"] == 0:
        return None, None
    return r["results"][0]["geometry"]["lat"], r["results"][0]["geometry"]["lng"]

# -------------------------------------------------
# GENERATE ENVIRONMENT DATA
# -------------------------------------------------
def generate_environment(lat, lon):
    temperature = 20 + abs(lat % 10)
    humidity = 40 + abs(lon % 20)
    wind_speed = 2 + (abs(lat + lon) % 5)
    precip = (abs(lat - lon) % 3)

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

# -------------------------------------------------
# MAIN UI
# -------------------------------------------------
st.title("🔥 AI Forest Fire Risk Predictor")

forest_name = st.text_input("Enter Forest Name", "Amazon")

if st.button("Predict Fire Risk"):

    lat, lon = geocode_forest(forest_name)
    if lat is None:
        st.error("Forest not found.")
        st.stop()

    df = generate_environment(lat, lon)

    # Encode landcover
    try:
        df["landcover_class_encoded"] = encoder.transform(df["landcover_class"])
    except:
        df["landcover_class_encoded"] = encoder.transform(["Deciduous Forest"])

    df = df.drop(columns=["landcover_class"])

    # Reorder columns
    df = df.reindex(columns=feature_cols)

    # Scale
    df_scaled = scaler.transform(df)

    # Predict
    pred = model.predict(df_scaled)[0]

    if pred == 1:
        st.error("High Fire Risk Detected")
    else:
        st.success("Fire Risk Not Detected")

    st.subheader("Input Data")
    st.json(df.to_dict(orient="records"))
