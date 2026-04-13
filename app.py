import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib

# ============================================================
# API KEY (KEEP YOURS HERE)
# ============================================================
OPENCAGE_API_KEY = "95df23a7370340468757cad17a479691"

# ============================================================
# LOAD MODEL FILES
# ============================================================
@st.cache_resource
def load_all():
    model = joblib.load("fire_model.pkl")
    scaler = joblib.load("scaler (2).pkl")
    encoder_dict = joblib.load("encoders.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return model, scaler, encoder_dict, feature_cols

model, scaler, encoder_dict, feature_cols = load_all()
encoder_cols = list(encoder_dict.keys())

# ============================================================
# IMPROVED FOREST LIST (IMPORTANT FIX)
# ============================================================
forest_list = [
    "Amazon Rainforest Brazil",
    "Sundarbans India",
    "Jim Corbett National Park India",
    "Gir National Park India",
    "Black Forest Germany",
    "Congo Rainforest Africa",
    "Daintree Rainforest Australia",
    "Sherwood Forest England",
    "Sequoia National Park USA",
    "Nilgiri Forest India",
    "Kaziranga National Park India",
    "Bandipur National Park India",
    "Borneo Rainforest Indonesia",
    "Satpura National Park India",
    "Periyar National Park India",
    "Great Bear Rainforest Canada"
]

# ============================================================
# FALLBACK COORDINATES (NO ERROR GUARANTEE)
# ============================================================
fallback_coords = {
    "Amazon Rainforest Brazil": (-3.4653, -62.2159),
    "Sundarbans India": (21.9497, 89.1833),
    "Jim Corbett National Park India": (29.5300, 78.7747),
    "Gir National Park India": (21.1240, 70.8245),
    "Black Forest Germany": (48.0000, 8.0000),
    "Congo Rainforest Africa": (-2.8797, 23.6560),
    "Daintree Rainforest Australia": (-16.1700, 145.4180),
    "Sherwood Forest England": (53.2000, -1.0667),
    "Sequoia National Park USA": (36.4864, -118.5658),
    "Nilgiri Forest India": (11.4064, 76.6932),
    "Kaziranga National Park India": (26.5775, 93.1711),
    "Bandipur National Park India": (11.7401, 76.6450),
    "Borneo Rainforest Indonesia": (0.7893, 113.9213),
    "Satpura National Park India": (22.4667, 78.4333),
    "Periyar National Park India": (9.4627, 77.2367),
    "Great Bear Rainforest Canada": (52.0000, -127.0000)
}

# ============================================================
# GEOCODING FUNCTION (FIXED)
# ============================================================
def geocode_forest(name):
    try:
        url = f"https://api.opencagedata.com/geocode/v1/json?q={name}&key={OPENCAGE_API_KEY}"
        r = requests.get(url).json()

        if r.get("total_results", 0) > 0:
            lat = r["results"][0]["geometry"]["lat"]
            lon = r["results"][0]["geometry"]["lng"]
            return lat, lon

        # fallback if API fails
        return fallback_coords.get(name, (None, None))

    except:
        return fallback_coords.get(name, (None, None))

# ============================================================
# GENERATE ENVIRONMENT DATA
# ============================================================
def generate_environment(lat, lon):
    temp = 20 + abs(lat % 10)
    hum = 40 + abs(lon % 20)
    wind = 2 + (abs(lat + lon) % 5)
    precip = abs(lat - lon) % 3
    ndvi = np.clip(hum/100 - 0.3, 0, 1)
    fwi = wind * (1 - hum/100) * 25

    return pd.DataFrame([{
        "latitude": lat,
        "longitude": lon,
        "temperature_c": temp,
        "humidity_pct": hum,
        "wind_speed_m_s": wind,
        "precip_mm": precip,
        "ndvi": ndvi,
        "fwi_score": fwi,
        "drought_code": max(20, (temp*2)-precip),
        "forest_cover_pct": 70,
        "landcover_class": "Deciduous Forest",
        "elevation_m": 300,
        "slope_deg": 12,
        "population_density": 18
    }])

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="AI Forest Fire Predictor", layout="wide")

st.title("🔥 AI-Based Forest Fire Predictor")

forest = st.selectbox("Select Forest", forest_list)

if st.button("Predict Fire Risk"):

    lat, lon = geocode_forest(forest)

    if lat is None:
        st.error("Forest not found even after fallback!")
        st.stop()

    st.success(f"Location Found: {lat}, {lon}")

    df = generate_environment(lat, lon)

    # Encoding
    df_oh = pd.get_dummies(df["landcover_class"], prefix="landcover_class")

    for col in encoder_cols:
        df_oh[col] = df_oh.get(col, 0)

    df = pd.concat([df.drop(columns=["landcover_class"]), df_oh[encoder_cols]], axis=1)
    df = df.reindex(columns=feature_cols)

    # Prediction
    pred = int(model.predict(df)[0])

    # MAP
    st.subheader("📍 Location")
    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))

    # METRICS
    col1, col2, col3 = st.columns(3)
    col1.metric("Temperature", f"{df.temperature_c.iloc[0]:.2f} °C")
    col2.metric("Humidity", f"{df.humidity_pct.iloc[0]:.2f} %")
    col3.metric("Wind Speed", f"{df.wind_speed_m_s.iloc[0]:.2f} m/s")

    # RESULT
    if pred == 1:
        st.error("🔥 HIGH FIRE RISK")
    else:
        st.success("🌿 LOW FIRE RISK")
