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
    background: radial-gradient(circle at top, #2b5876 0, #1b1b1b 45%, #000000 100%);
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
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(14px);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 10px 30px rgba(0,0,0,0.6);
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
    from { box-shadow: 0 0 12px #ff4646; }
    to   { box-shadow: 0 0 28px #ff0000; }
}

/* Map Border */
.stMap {
    border-radius: 18px;
    border: 2px solid #333;
    overflow: hidden;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #050505;
    color: #f0f0f0;
}

.sidebar-title {
    font-weight: 900;
    color: #ff4d4d;
    font-size: 26px;
    padding-bottom: 10px;
}

input {
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
OPENCAGE_API_KEY = "95df23a7370340468757cad17a479691"

# ---------------------------------------------------------
# LOAD ALL ARTEFACTS
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
# HELPERS
# ---------------------------------------------------------
def geocode_forest(name: str):
    """Convert forest name to lat/lon using OpenCage."""
    url = f"https://api.opencagedata.com/geocode/v1/json?q={name}&key={OPENCAGE_API_KEY}"
    r = requests.get(url).json()
    if r.get("total_results", 0) == 0:
        return None, None
    return r["results"][0]["geometry"]["lat"], r["results"][0]["geometry"]["lng"]


def generate_environment(lat: float, lon: float) -> pd.DataFrame:
    """Generate synthetic but realistic environmental features."""
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


def run_model(df_env: pd.DataFrame):
    """Encode + reindex + scale + predict."""
    # Encode landcover
    try:
        df_env["landcover_class_encoded"] = encoder.transform(df_env["landcover_class"])
    except Exception:
        df_env["landcover_class_encoded"] = encoder.transform(["Deciduous Forest"])

    df_env = df_env.drop(columns=["landcover_class"])
    df_env = df_env.reindex(columns=feature_cols)
    df_scaled = scaler.transform(df_env)

    pred = model.predict(df_scaled)[0]
    # If model supports probability:
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(df_scaled)[0][1]
    return pred, prob, df_env

# ---------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------
st.markdown("<div class='title'>🔥 AI Forest Fire Risk Predictor</div>", unsafe_allow_html=True)

st.sidebar.markdown("<div class='sidebar-title'>⚡ Controls</div>", unsafe_allow_html=True)
forest_name = st.sidebar.text_input("🌲 Forest Name", "Amazon")
predict_btn = st.sidebar.button("🚀 Predict Fire Risk", use_container_width=True)

st.sidebar.markdown("----")
st.sidebar.write("Made with ❤️ using Streamlit & ML")

if predict_btn:
    # 1. Geocode
    lat, lon = geocode_forest(forest_name)
    if lat is None:
        st.error("❌ Forest not found. Try another name.")
        st.stop()

    # 2. Generate env data
    df_env = generate_environment(lat, lon)

    # 3. Run model
    pred, prob, df_final = run_model(df_env)

    # --------- Map Card ----------
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📍 Forest Location")
    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))
    st.markdown("</div>", unsafe_allow_html=True)

    # --------- Metrics Card ----------
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🌡 Temp", f"{df_final.temperature_c.values[0]:.1f} °C")
    col2.metric("💧 Humidity", f"{df_final.humidity_pct.values[0]:.0f} %")
    col3.metric("🌬 Wind", f"{df_final.wind_speed_m_s.values[0]:.1f} m/s")
    col4.metric("🌿 NDVI", f"{df_final.ndvi.values[0]:.2f}")
    col5.metric("🔥 FWI", f"{df_final.fwi_score.values[0]:.1f}")
    st.markdown("</div>", unsafe_allow_html=True)

    # --------- Prediction Box ----------
    if pred == 1:
        box_html = "<div class='prediction-box' style='background:#ff1a1a; color:white;'>🔥 HIGH FIRE RISK DETECTED</div>"
    else:
        box_html = "<div class='prediction-box' style='background:#22cc88; color:white;'>🌿 LOW / NO FIRE RISK</div>"

    st.markdown(box_html, unsafe_allow_html=True)

    # --------- Input Data Card ----------
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📊 Environment Data Used for Prediction")
    st.json(df_final.to_dict(orient="records")[0])
    if prob is not None:
        st.caption(f"Estimated probability of fire (model output): **{prob*100:.2f}%**")
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("👈 Sidebar se forest name daal ke **Predict Fire Risk** pe click kar.")

