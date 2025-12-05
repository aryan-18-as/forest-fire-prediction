import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib

st.set_page_config(
    page_title="Forest Fire Risk Predictor",
    layout="centered"
)

# -----------------------------------------------
# Load ML Model + Scaler + Encoder + Feature List
# -----------------------------------------------
@st.cache_resource
def load_all():
    model = joblib.load("fire_model.pkl")
    scaler = joblib.load("scaler (2).pkl")
    encoder = joblib.load("encoder.pkl")
    feature_cols = joblib.load("feature_columns_final.pkl")
    return model, scaler, encoder, feature_cols


model, scaler, encoder, feature_cols = load_all()

# -----------------------------------------------
# API KEYS (Embedded)
# -----------------------------------------------

GEOCODE_API_KEY = "95df23a7370340468757cad17a479691"


# -----------------------------------------------
# GET FOREST COORDINATES (OpenCage)
# -----------------------------------------------
def geocode_forest(forest_name):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={forest_name}&key={GEOCODE_API_KEY}"
    r = requests.get(url).json()

    if r["total_results"] == 0:
        return None, None

    lat = r["results"][0]["geometry"]["lat"]
    lon = r["results"][0]["geometry"]["lng"]
    return lat, lon


# -----------------------------------------------
# GET WEATHER + VEGETATION DATA
# -----------------------------------------------
def fetch_environment_data(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
    r = requests.get(url).json()

    temp = r["main"]["temp"]
    humidity = r["main"]["humidity"]
    wind = r["wind"]["speed"]
    rain = r.get("rain", {}).get("1h", 0)

    # Fake but stable NDVI & FWI logic
    ndvi = np.clip((humidity / 100) - 0.3, 0, 1)
    fwi = wind * (1 - humidity / 100) * 20
    drought_code = max(10, (temp * 3) - rain)

    data = {
        "latitude": lat,
        "longitude": lon,
        "temperature_c": temp,
        "precip_mm": rain,
        "humidity_pct": humidity,
        "wind_speed_m_s": wind,
        "fwi_score": fwi,
        "drought_code": drought_code,
        "ndvi": ndvi,
        "forest_cover_pct": 70,               # Fixed constant
        "landcover_class": "Deciduous Forest",  # FIXED to avoid encoder errors
        "elevation_m": 300,
        "slope_deg": 10,
        "population_density": 20
    }

    return pd.DataFrame([data])


# -----------------------------------------------
# STREAMLIT UI
# -----------------------------------------------
st.markdown("<h1>🔥 AI Based Forest Fire Risk Predictor</h1>", unsafe_allow_html=True)

forest_name = st.text_input("🌲 Enter Forest Name", "Amazon")

if st.button("Predict Fire Risk"):

    st.info("Fetching data...")

    # 1. Get coordinates
    lat, lon = geocode_forest(forest_name)

    if lat is None:
        st.error("Forest not found. Try a different name.")
        st.stop()

    # 2. Get environmental data
    df = fetch_environment_data(lat, lon)

    # 3. Encode landcover column
    try:
        df["landcover_class_encoded"] = encoder.transform(df["landcover_class"])
    except:
        # FALLBACK: Use first known class
        df["landcover_class_encoded"] = encoder.transform(["Deciduous Forest"])

    # 4. Keep only required features
    df = df[feature_cols]

    # 5. Scale numerical features
    df_scaled = scaler.transform(df)

    # 6. Predict
    pred = model.predict(df_scaled)[0]

    result = "🔥 YES — HIGH FIRE RISK" if pred == 1 else "🌧 NO — LOW FIRE RISK"
    color = "red" if pred == 1 else "green"

    st.markdown(
        f"<h2 style='color:{color};text-align:center;'>{result}</h2>",
        unsafe_allow_html=True
    )

    st.subheader("📊 Model Input Data")
    st.json(df.to_dict(orient="records"))
