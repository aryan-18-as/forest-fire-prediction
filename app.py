import streamlit as st
import requests
import pandas as pd
import joblib
import numpy as np

# -----------------------------------------------------------
# LOAD MODEL + SCALER + ENCODER + FEATURE COLUMNS
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("fire_model.pkl")
    scaler = joblib.load("scaler (2).pkl")
    encoder = joblib.load("encoder.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return model, scaler, encoder, feature_cols


model, scaler, encoder, feature_cols = load_model()


# -----------------------------------------------------------
# GEOCODING API (OpenCage)
# -----------------------------------------------------------
def geocode_forest(name):
    API_KEY = st.secrets["OPENCAGE_KEY"]  # MUST be added in Streamlit Secrets

    url = f"https://api.opencagedata.com/geocode/v1/json?q={name}&key={API_KEY}"
    resp = requests.get(url).json()

    if len(resp["results"]) == 0:
        return None, None

    lat = resp["results"][0]["geometry"]["lat"]
    lon = resp["results"][0]["geometry"]["lng"]
    return lat, lon


# -----------------------------------------------------------
# WEATHER API (Open-Meteo)
# -----------------------------------------------------------
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


# -----------------------------------------------------------
# SIMPLE NDVI (fallback method)
# -----------------------------------------------------------
def fetch_ndvi(lat, lon):
    try:
        url = f"https://api.spectator.earth/ndvi?lat={lat}&lon={lon}"
        r = requests.get(url).json()
        return r.get("ndvi", 0.5)
    except:
        return 0.5


# -----------------------------------------------------------
# SIMPLE LANDCOVER CLASSIFIER USING NDVI
# -----------------------------------------------------------
def fetch_landcover(lat, lon):
    ndvi = fetch_ndvi(lat, lon)

    if ndvi > 0.6:
        return "Evergreen Forest"
    elif ndvi > 0.4:
        return "Deciduous Forest"
    elif ndvi > 0.2:
        return "Grassland"
    else:
        return "Cropland"


# -----------------------------------------------------------
# ELEVATION API
# -----------------------------------------------------------
def fetch_elevation(lat, lon):
    try:
        url = f"https://api.opentopodata.org/v1/test-dataset?locations={lat},{lon}"
        r = requests.get(url).json()
        return r["results"][0]["elevation"]
    except:
        return 200


# -----------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------
st.title("🔥 AI-Based Forest Fire Risk Prediction (Live API Version)")
st.write("Predict fire risk for ANY forest worldwide using live climate + environment data.")

forest_name = st.text_input("🌳 Enter Forest Name (Example: Sundarbans, Amazon, Jim Corbett)")

if st.button("Predict Fire Risk", use_container_width=True):

    if forest_name.strip() == "":
        st.error("❌ Please enter a forest name.")
        st.stop()

    # -----------------------------------------------------------
    # 1️⃣ GET LAT/LON
    # -----------------------------------------------------------
    lat, lon = geocode_forest(forest_name)

    if lat is None:
        st.error("❌ Unable to locate forest. Try a different name.")
        st.stop()

    st.success(f"📍 Coordinates Found: Latitude {lat}, Longitude {lon}")

    # -----------------------------------------------------------
    # 2️⃣ GET WEATHER
    # -----------------------------------------------------------
    w = fetch_weather(lat, lon)
    st.info("🌤 Live Weather Fetched Successfully")

    # -----------------------------------------------------------
    # 3️⃣ NDVI + LANDCOVER + ELEVATION
    # -----------------------------------------------------------
    ndvi = fetch_ndvi(lat, lon)
    landcover = fetch_landcover(lat, lon)
    elevation = fetch_elevation(lat, lon)

    # -----------------------------------------------------------
    # 4️⃣ DEFAULT VALUES (NO POPULATION API)
    # -----------------------------------------------------------
    population = 50        # SAFE default
    slope = 10            # SAFE default
    forest_cover = 50     # SAFE default
    drought_code = 100    # SAFE default
    fwi_score = 0         # SAFE default

    # -----------------------------------------------------------
    # 5️⃣ BUILD API DATAFRAME
    # -----------------------------------------------------------
    df_api = pd.DataFrame([{
        "latitude": lat,
        "longitude": lon,
        "temperature": w["temperature"],
        "humidity": w["humidity"],
        "precip": w["precip"],
        "wind_speed": w["wind_speed"],
        "ndvi": ndvi,
        "elevation": elevation,
        "slope": slope,
        "population_density": population,
        "forest_cover_pct": forest_cover,
        "drought_code": drought_code,
        "fwi_score": fwi_score,
        "landcover_class": landcover
    }])

    # -----------------------------------------------------------
    # 6️⃣ RENAME COLUMNS TO MATCH TRAINING DATA
    # -----------------------------------------------------------
    df = df_api.rename(columns={
        "temperature": "temperature_c",
        "humidity": "humidity_pct",
        "wind_speed": "wind_speed_m_s",
        "elevation": "elevation_m",
        "slope": "slope_deg",
        "precip": "precip_mm",
    })

    # -----------------------------------------------------------
    # 7️⃣ ENCODE LANDCOVER CLASS
    # -----------------------------------------------------------
    df["landcover_class_encoded"] = encoder.transform(df[["landcover_class"]])
    df = df.drop(columns=["landcover_class"])

    # -----------------------------------------------------------
    # 8️⃣ ALIGN FEATURE ORDER WITH TRAINING DATA
    # -----------------------------------------------------------
    df = df[feature_cols]

    # -----------------------------------------------------------
    # 9️⃣ SCALE + PREDICT
    # -----------------------------------------------------------
    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    # -----------------------------------------------------------
    # 🔟 FINAL OUTPUT
    # -----------------------------------------------------------
    st.subheader("📊 Prediction Result")
    if pred == 1:
        st.error(f"🔥 HIGH FIRE RISK (Probability: {prob:.2f})")
    else:
        st.success(f"🌿 LOW FIRE RISK (Probability: {prob:.2f})")

    st.info("✔ Prediction complete — using live weather + environmental data.")
