import streamlit as st
import pandas as pd
import joblib
import requests

st.set_page_config(page_title="AI Forest Fire Risk Predictor", layout="centered")

st.title("🔥 AI Based Forest Fire Risk Predictor (Final Stable Version)")

# -----------------------------
# 1) Load Model, Scaler, Encoder
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("fire_model.pkl")
    scaler = joblib.load("scaler (2).pkl")
    encoder = joblib.load("encoder.pkl")
    return model, scaler, encoder

model, scaler, encoder = load_artifacts()

# -----------------------------
# 2) API KEY (HARDCODED)
# -----------------------------
OPENCAGE_KEY = "95df23a7370340468757cad17a479691"  # your key

# -----------------------------
# 3) FOREST → LAT/LON
# -----------------------------
def geocode_forest(name):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={name}&key={OPENCAGE_KEY}"
    res = requests.get(url)
    data = res.json()

    if "results" not in data or len(data["results"]) == 0:
        return None, None
    
    lat = data["results"][0]["geometry"]["lat"]
    lon = data["results"][0]["geometry"]["lng"]
    return lat, lon

# -----------------------------
# 4) WEATHER + VEGETATION API SIMULATION
# -----------------------------
def get_environment_data(lat, lon):
    return {
        "latitude": lat,
        "longitude": lon,
        "temperature_c": 28.4,
        "precip_mm": 2.1,
        "humidity_pct": 61,
        "wind_speed_m_s": 4.3,
        "fwi_score": 32.5,
        "drought_code": 145,
        "ndvi": 0.49,
        "forest_cover_pct": 72,
        "landcover_class": "Evergreen Forest",
        "elevation_m": 228,
        "slope_deg": 12.5,
        "population_density": 28
    }

# -----------------------------
# 5) Streamlit UI
# -----------------------------
forest_name = st.text_input("🌲 Enter Forest Name")

if st.button("Predict Fire Risk"):
    
    if forest_name.strip() == "":
        st.error("Please enter a valid forest name!")
        st.stop()

    # Step A → Get coordinates
    lat, lon = geocode_forest(forest_name)
    if lat is None:
        st.error("Forest not found. Try another name.")
        st.stop()

    # Step B → Get environmental variables
    env = get_environment_data(lat, lon)
    df = pd.DataFrame([env])

    st.write("📌 API DF Columns:", df.columns.tolist())

    # Step C → Encode landcover_class
    df["landcover_class_encoded"] = encoder.transform(df["landcover_class"])

    # Step D → Drop raw category column
    df = df.drop(columns=["landcover_class"])

    # FINAL feature columns (must match model training)
    feature_cols = [
        "latitude",
        "longitude",
        "temperature_c",
        "precip_mm",
        "humidity_pct",
        "wind_speed_m_s",
        "fwi_score",
        "drought_code",
        "ndvi",
        "forest_cover_pct",
        "landcover_class_encoded",
        "elevation_m",
        "slope_deg",
        "population_density"
    ]

    # DEBUG: Show what's inside df
    st.write("After Encoding:", df.columns.tolist())

    # Step E → Reorder
    df = df[feature_cols]

    # Step F → Scale
    df_scaled = scaler.transform(df)

    # Step G → Predict
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    st.subheader("🔥 Prediction Result")
    if pred == 1:
        st.error(f"⚠ HIGH RISK of Forest Fire ({prob*100:.2f}% probability)")
    else:
        st.success(f"✅ LOW RISK of Forest Fire ({prob*100:.2f}% probability)")
