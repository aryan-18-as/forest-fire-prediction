import streamlit as st
import requests
import pandas as pd
import joblib
import numpy as np

# ------------------------------
# DIRECT API KEY
# ------------------------------
OPENCAGE_KEY = "95df23a7370340468757cad17a479691"  # PUT HERE

# ------------------------------
# LOAD MODEL + SCALER
# ------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("fire_model.pkl")
    scaler = joblib.load("scaler (2).pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return model, scaler, feature_cols

model, scaler, feature_cols = load_model()

# ------------------------------
# GEOCODE FOREST → LAT/LON
# ------------------------------
def geocode_forest(name):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={name}&key={OPENCAGE_KEY}"
    resp = requests.get(url).json()

    if len(resp["results"]) == 0:
        return None, None

    lat = resp["results"][0]["geometry"]["lat"]
    lon = resp["results"][0]["geometry"]["lng"]
    return lat, lon

# ------------------------------
# WEATHER API
# ------------------------------
def fetch_weather(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
    )
    r = requests.get(url).json()
    cur = r["current"]

    return {
        "temperature": cur.get("temperature_2m", 25),
        "humidity": cur.get("relative_humidity_2m", 50),
        "precip": cur.get("precipitation", 0),
        "wind_speed": cur.get("wind_speed_10m", 2),
    }

# ------------------------------
# NDVI API (fallback)
# ------------------------------
def fetch_ndvi(lat, lon):
    try:
        r = requests.get(f"https://api.spectator.earth/ndvi?lat={lat}&lon={lon}").json()
        return r.get("ndvi", 0.5)
    except:
        return 0.5

# ------------------------------
# SIMPLE LANDCOVER (Your model expects raw string)
# ------------------------------
def fetch_landcover(lat, lon):
    ndvi = fetch_ndvi(lat, lon)
    if ndvi > 0.6:
        return "Evergreen Forest"
    elif ndvi > 0.4:
        return "Deciduous Forest"
    elif ndvi > 0.2:
        return "Grassland"
    return "Cropland"

# ------------------------------
# ELEVATION
# ------------------------------
def fetch_elevation(lat, lon):
    try:
        r = requests.get(f"https://api.opentopodata.org/v1/test-dataset?locations={lat},{lon}").json()
        return r["results"][0]["elevation"]
    except:
        return 200

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.title("🔥 AI Based Forest Fire Risk Predictor (Final Stable Version)")

forest_name = st.text_input("🌳 Enter Forest Name")

if st.button("Predict Fire Risk"):

    # 1️⃣ GEOCODE
    lat, lon = geocode_forest(forest_name)
    if lat is None:
        st.error("Forest not found.")
        st.stop()

    # 2️⃣ WEATHER
    w = fetch_weather(lat, lon)

    # 3️⃣ OTHER FEATURES
    ndvi = fetch_ndvi(lat, lon)
    landcover = fetch_landcover(lat, lon)
    elevation = fetch_elevation(lat, lon)

    # 4️⃣ CONSTRUCT EXACT FEATURE SET EXPECTED BY MODEL
    df = pd.DataFrame([{
        'latitude': lat,
        'longitude': lon,
        'temperature_c': w["temperature"],
        'precip_mm': w["precip"],
        'humidity_pct': w["humidity"],
        'wind_speed_m_s': w["wind_speed"],
        'fwi_score': 0,              # default value
        'drought_code': 100,         # default
        'ndvi': ndvi,
        'forest_cover_pct': 50,      # default
        'landcover_class': landcover,  # RAW STRING (NO ENCODING)
        'elevation_m': elevation,
        'slope_deg': 10,             # default
        'population_density': 50     # default
    }])

    # ⚠ ORDER EXACTLY AS feature_columns.pkl
    df = df[feature_cols]

    # 5️⃣ SCALE + PREDICT
    df_scaled = scaler.transform(df)

    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    # RESULT DISPLAY
    st.subheader("📊 Prediction Result")
    if pred == 1:
        st.error(f"🔥 HIGH FIRE RISK — Probability: {prob:.2f}")
    else:
        st.success(f"🌿 LOW FIRE RISK — Probability: {prob:.2f}")
