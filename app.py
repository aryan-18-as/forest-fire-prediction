import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Forest Fire Predictor",
    page_icon="🔥",
    layout="wide"
)

# ---------------------------------------------------------
# PREMIUM CSS
# ---------------------------------------------------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0f0f0f, #1a1a1a);
}

/* Animated Gradient Title */
.title {
    font-size: 52px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #ff8a00, #e52e71, #9b00ff);
    -webkit-background-clip: text;
    color: transparent;
    padding: 20px 0px;
}

/* Glass Card */
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 8px 25px rgba(0,0,0,0.5);
    margin-bottom: 25px;
}

/* Prediction Box */
.prediction-box {
    font-size: 32px;
    padding: 25px;
    font-weight: 700;
    text-align: center;
    border-radius: 16px;
    margin-top: 15px;
    animation: glow 1.7s ease-in-out infinite alternate;
}

@keyframes glow {
    from { box-shadow: 0 0 10px #ff4646; }
    to { box-shadow: 0 0 25px #ff0000; }
}

/* Map Border */
.stMap {
    border-radius: 18px;
    border: 2px solid #333;
    overflow: hidden;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111;
    color: white;
}

.sidebar-title {
    font-weight: 900;
    color: #ff4d4d;
    font-size: 28px;
    padding-bottom: 10px;
}

input {
    border-radius: 12px !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# API Key
# ---------------------------------------------------------
OPENCAGE_API_KEY = "95df23a7370340468757cad17a479691"

# ---------------------------------------------------------
# LOAD ALL MODELS
# ---------------------------------------------------------
@st.cache_resource
def load_all():
    model = joblib.load("fire_model.pkl")
    scaler = joblib.load("scaler (2).pkl")
    encoder = joblib.load("encoder.pkl")
    feature_cols = joblib.load("feature_columns_1.pkl")
    return model, scaler, encoder, feature_cols

model, scaler, encoder, feature_cols = load_all()


# ---------------------------------------------------------
# GEOCODING
# ---------------------------------------------------------
def geocode_forest(name):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={name}&key={OPENCAGE_API_KEY}"
    r = requests.get(url).json()
    if r["total_results"] == 0:
        return None, None
    return r["results"][0]["geometry"]["lat"], r["results"][0]["geometry"]["lng"]

# ---------------------------------------------------------
# Generate environment data
# ---------------------------------------------------------
def generate_environment(lat, lon):
    temperature = 20 + abs(lat % 10)
    humidity = 40 + abs(lon % 20)
    wind_speed = 2 + (abs(lat + lon) % 5)
    precip = abs(lat - lon) % 3

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


# ---------------------------------------------------------
# HEADER TITLE
# ---------------------------------------------------------
st.markdown("<div class='title'>🔥 AI Forest Fire Risk Predictor</div>", unsafe_allow_html=True)

st.sidebar.markdown("<div class='sidebar-title'>⚡ Controls</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------
forest_name = st.sidebar.text_input("🌲 Enter Forest Name", "Amazon")

predict_btn = st.sidebar.button("🚀 Predict Fire Risk", use_container_width=True)

# ---------------------------------------------------------
# MAIN PREDICTION FLOW
# ---------------------------------------------------------
if predict_btn:

    lat, lon = geocode_forest(forest_name)

    if lat is None:
        st.error("❌ Forest not found. Try another name.")
        st.stop()

    df = generate_environment(lat, lon)

    # Encode
    try:
        df["landcover_class_encoded"] = encoder.transform(df["landcover_class"])
    except:
        df["landcover_class_encoded"] = encoder.transform(["Deciduous Forest"])

    df = df.drop(columns=["landcover_class"])

    df = df.reindex(columns=feature_cols)

    df_scaled = scaler.transform(df)

    pred = model.predict(df_scaled)[0]

    # ---------------------------------------------------------
    # ROW 1 — MAP
    # ---------------------------------------------------------
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📍 Forest Location on Map")
    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # ROW 2 — METRICS
    # ---------------------------------------------------------
    with st.container():
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("🌡 Temperature", f"{df.temperature_c.values[0]} °C")
        col2.metric("💧 Humidity", f"{df.humidity_pct.values[0]} %")
        col3.metric("🌬 Wind Speed", f"{df.wind_speed_m_s.values[0]} m/s")
        col4.metric("🌿 NDVI", round(df.ndvi.values[0], 2))
        col5.metric("🔥 FWI Score", round(df.fwi_score.values[0], 2))

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # ROW 3 — PREDICTION OUTPUT
    # ---------------------------------------------------------
    if pred == 1:
        st.markdown(
            "<div class='prediction-box' style='background:#ff1a1a; color:white;'>🔥 HIGH FIRE RISK DETECTED</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='prediction-box' style='background:#22cc88; color:white;'>🌿 LOW / NO FIRE RISK</div>",
            unsafe_allow_html=True
        )

    # ---------------------------------------------------------
    # ROW 4 — INPUT JSON PRETTY CARD
    # ---------------------------------------------------------
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📊 Environment Data Used")
    st.json(df.to_dict(orient="records")[0])
    st.markdown("</div>", unsafe_allow_html=True)
