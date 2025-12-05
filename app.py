import streamlit as st
import requests
import pandas as pd
import joblib
import numpy as np

# ----------------------------
# LOAD MODELS
# ----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("fire_model.pkl")
    scaler = joblib.load("scaler (2).pkl")
    encoder = joblib.load("encoder.pkl")
    feature_cols = joblib.load("feature_columns.pkl")  # full column order used during training
    return model, scaler, encoder, feature_cols


model, scaler, encoder, feature_cols = load_model()


# ----------------------------
# OPEN CAGE GEOCODER
# ----------------------------
def geocode_forest(name):
    API_KEY = st.secrets["OPENCAGE_KEY"]  # stored in secrets.toml
    url = f"https://api.opencagedata.com/geocode/v1/json?q={name}&key={API_KEY}"

    resp = requests.get(url).json()

    if len(resp["results"]) == 0:
        return None, None

    lat = resp["results"][0]["geometry"]["lat"]
    lon = resp["results"][0]["geometry"]["lng"]
    return lat, lon


# ----------------------------
# OPEN-METEO WEATHER API
# ----------------------------
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


# ----------------------------
# NDVI API (SentinelHub Free Endpoint)
# ----------------------------
def fetch_ndvi(lat, lon):
    try:
        url = f"https://api.spectator.earth/ndvi?lat={lat}&lon={lon}"
        r = requests.get(url).json()
        return r.get("ndvi", 0.5)
    except:
        return 0.5


# ----------------------------
# ELEVATION API
# ----------------------------
def fetch_elevation(lat, lon):
    try:
        url = f"https://api.opentopodata.org/v1/test-dataset?locations={lat},{lon}"
        r = requests.get(url).json()
        return r["results"][0]["elevation"]
    except:
        return 200


# ----------------------------
# LANDCOVER (simple fallback)
# ----------------------------
def fetch_landcover(lat, lon):
    # Temporary classifier based on NDVI
    ndvi = fetch_ndvi(lat, lon)
    if ndvi > 0.6:
        return "Evergreen Forest"
    elif ndvi > 0.4:
        return "Deciduous Forest"
    elif ndvi > 0.2:
        return "Grassland"
    return "Cropland"


# ----------------------------
# POPULATION (API)
# ----------------------------
def fetch_population(lat, lon):
    try:
        url = f"https://api.api-ninjas.com/v1/reversegeocoding?lat={lat}&lon={lon}"
        headers = {"X-Api-Key": st.secrets["NINJA_KEY"]}
        r = requests.get(url, headers=headers).json()
        return r[0].get("population", 10)
    except:
        return 10


# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("🔥 AI-Based Forest Fire Risk Prediction")

forest_name = st.text_input("🌳 Enter Forest / National Park Name (e.g., Sundarbans, Amazon, Jim Corbett)")

if st.button("Predict Fire Risk"):

    if forest_name == "":
        st.error("❌ Please enter a forest name.")
        st.stop()

    # ---------------------------------------
    # 1️⃣ GEOCODING
    # ---------------------------------------
    lat, lon = geocode_forest(forest_name)
    if lat is None:
        st.error("❌ Could not locate this forest name. Try another.")
        st.stop()

    st.success(f"📍 Location Found: ({lat}, {lon})")

    # ---------------------------------------
    # 2️⃣ FETCH ALL FEATURES
    # ---------------------------------------
    w = fetch_weather(lat, lon)
    ndvi = fetch_ndvi(lat, lon)
    elevation = fetch_elevation(lat, lon)
    landcover = fetch_landcover(lat, lon)
    population = fetch_population(lat, lon)

    # ---------------------------------------
    # 3️⃣ BUILD DATAFRAME
    # ---------------------------------------
    df_api = pd.DataFrame([{
        "latitude": lat,
        "longitude": lon,
        "temperature": w["temperature"],
        "humidity": w["humidity"],
        "precip": w["precip"],
        "wind_speed": w["wind_speed"],
        "ndvi": ndvi,
        "elevation": elevation,
        "slope": 10,                    # fallback
        "population_density": population,
        "landcover_class": landcover
    }])

    # ---------------------------------------
    # 4️⃣ RENAME COLUMNS TO MATCH TRAINING
    # ---------------------------------------
    df = df_api.rename(columns={
        "temperature": "temperature_c",
        "humidity": "humidity_pct",
        "wind_speed": "wind_speed_m_s",
        "elevation": "elevation_m",
        "slope": "slope_deg",
        "precip": "precip_mm",
    })

    # ---------------------------------------
    # 5️⃣ ADD MISSING REQUIRED FEATURES
    # ---------------------------------------
    df["fwi_score"] = 0
    df["drought_code"] = 100
    df["forest_cover_pct"] = 50

    # ---------------------------------------
    # 6️⃣ ENCODE LANDCOVER
    # ---------------------------------------
    df["landcover_class_encoded"] = encoder.transform(df[["landcover_class"]])
    df = df.drop(columns=["landcover_class"])

    # ---------------------------------------
    # 7️⃣ ORDER COLUMNS EXACTLY LIKE TRAINING
    # ---------------------------------------
    df = df[feature_cols]

    # ---------------------------------------
    # 8️⃣ SCALE + PREDICT
    # ---------------------------------------
    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    # ---------------------------------------
    # 9️⃣ OUTPUT
    # ---------------------------------------
    st.subheader("📊 Prediction Result")

    if pred == 1:
        st.error(f"🔥 High Fire Risk! (Probability: {prob:.2f})")
    else:
        st.success(f"🌿 Low Fire Risk (Probability: {prob:.2f})")

    st.info("✔ Prediction completed successfully!")
