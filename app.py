import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Forest Fire Predictor",
    layout="wide",
    page_icon="🔥"
)

# ---------------------------------------------------
# CUSTOM CSS FOR PREMIUM UI
# ---------------------------------------------------
st.markdown("""
<style>
/* Gradient Title */
.main-title {
    font-size: 48px;
    font-weight: 900;
    text-align: center;
    padding: 18px;
    background: linear-gradient(90deg, #ff512f, #dd2476);
    -webkit-background-clip: text;
    color: transparent;
}

/* Cards */
.card {
    padding: 20px;
    background: white;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    margin-bottom: 20px;
}

/* Prediction Box */
.pred-box {
    font-size: 30px;
    font-weight: bold;
    text-align: center;
    padding: 25px;
    border-radius: 12px;
    margin-top: 20px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1e1e1e;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# API KEY
# ---------------------------------------------------
OPENCAGE_API_KEY = "95df23a7370340468757cad17a479691"

# ---------------------------------------------------
# LOAD MODEL FILES
# ---------------------------------------------------
@st.cache_resource
def load_all():
    model = joblib.load("fire_model.pkl")
    scaler = joblib.load("scaler (2).pkl")
    encoder = joblib.load("encoder.pkl")
    feature_cols = joblib.load("feature_columns_1.pkl")
    return model, scaler, encoder, feature_cols

model, scaler, encoder, feature_cols = load_all()

# ---------------------------------------------------
# GEOCODING
# ---------------------------------------------------
def geocode_forest(name):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={name}&key={OPENCAGE_API_KEY}"
    r = requests.get(url).json()
    if r["total_results"] == 0:
        return None, None
    return r["results"][0]["geometry"]["lat"], r["results"][0]["geometry"]["lng"]

# ---------------------------------------------------
# ENVIRONMENT DATA GENERATOR
# ---------------------------------------------------
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


# ---------------------------------------------------
# MAIN UI
# ---------------------------------------------------
st.markdown("<div class='main-title'>🔥 AI-Based Forest Fire Risk Predictor</div>", unsafe_allow_html=True)

forest_name = st.text_input("🌲 Enter Forest Name", "Amazon", placeholder="Enter any Forest Name…")

# ---------------------------------------------------
# RUN BUTTON
# ---------------------------------------------------
if st.button("🔍 Predict Fire Risk", use_container_width=True):

    lat, lon = geocode_forest(forest_name)

    if lat is None:
        st.error("❌ Forest not found. Try another name.")
        st.stop()

    # Generate data
    df = generate_environment(lat, lon)

    # Encode landcover
    try:
        df["landcover_class_encoded"] = encoder.transform(df["landcover_class"])
    except:
        df["landcover_class_encoded"] = encoder.transform(["Deciduous Forest"])

    df = df.drop(columns=["landcover_class"])

    # Correct Columns
    df = df.reindex(columns=feature_cols)

    # Scale
    df_scaled = scaler.transform(df)

    # Predict
    pred = model.predict(df_scaled)[0]

    # ---------------------------------------------------
    # SHOW MAP
    # ---------------------------------------------------
    st.subheader("📍 Forest Location")
    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))


    # ---------------------------------------------------
    # METRIC CARDS ROW
    # ---------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("🌡️ Temperature", f"{df.temperature_c.values[0]} °C")
    col2.metric("💧 Humidity", f"{df.humidity_pct.values[0]} %")
    col3.metric("🌬️ Wind", f"{df.wind_speed_m_s.values[0]} m/s")
    col4.metric("🌿 NDVI", round(df.ndvi.values[0], 2))
    col5.metric("🔥 FWI Score", round(df.fwi_score.values[0], 2))


    # ---------------------------------------------------
    # PREDICTION BOX
    # ---------------------------------------------------
    if pred == 1:
        st.markdown(
            "<div class='pred-box' style='background:#ffcccc; color:#b30000;'>🔥 HIGH FIRE RISK DETECTED</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='pred-box' style='background:#ccffcc; color:#006600;'>🌿 LOW / NO FIRE RISK</div>",
            unsafe_allow_html=True
        )


    # ---------------------------------------------------
    # INPUT DATA CARD
    # ---------------------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Input Environmental Data Used")
    st.json(df.to_dict(orient="records")[0])
    st.markdown("</div>", unsafe_allow_html=True)
