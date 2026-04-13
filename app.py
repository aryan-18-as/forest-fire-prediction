import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="AI Forest Fire Predictor", layout="wide", page_icon="🔥")

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
# FOREST LIST (FIXED)
# ============================================================
forest_list = [
    "Amazon Rainforest Brazil",
    "Sundarbans India",
    "Jim Corbett National Park India",
    "Gir National Park India",
    "Black Forest Germany",
    "Congo Rainforest Africa",
    "Daintree Rainforest Australia"
]

# ============================================================
# FALLBACK COORDINATES
# ============================================================
fallback_coords = {
    "Amazon Rainforest Brazil": (-3.4653, -62.2159),
    "Sundarbans India": (21.9497, 89.1833),
    "Jim Corbett National Park India": (29.5300, 78.7747),
    "Gir National Park India": (21.1240, 70.8245),
    "Black Forest Germany": (48.0000, 8.0000),
    "Congo Rainforest Africa": (-2.8797, 23.6560),
    "Daintree Rainforest Australia": (-16.1700, 145.4180),
}

# ============================================================
# FUNCTIONS
# ============================================================
def geocode_forest(name):
    try:
        url = f"https://api.opencagedata.com/geocode/v1/json?q={name}"
        r = requests.get(url).json()

        if r.get("total_results", 0) > 0:
            lat = r["results"][0]["geometry"]["lat"]
            lon = r["results"][0]["geometry"]["lng"]
            return lat, lon

        return fallback_coords.get(name, (None, None))
    except:
        return fallback_coords.get(name, (None, None))


def generate_environment(lat, lon):
    return pd.DataFrame([{
        "latitude": lat,
        "longitude": lon,
        "temperature_c": 25,
        "humidity_pct": 50,
        "wind_speed_m_s": 5,
        "precip_mm": 1,
        "ndvi": 0.5,
        "fwi_score": 10,
        "drought_code": 30,
        "forest_cover_pct": 70,
        "landcover_class": "Deciduous Forest",
        "elevation_m": 300,
        "slope_deg": 12,
        "population_density": 18
    }])

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🔥 Fire Prediction Suite")
    menu = st.radio("Navigation", [
        "Prediction Dashboard",
        "EDA Analytics",
        "Danger Calculator",
        "Dataset Explorer",
        "Project Report"
    ])

# ============================================================
# PAGE 1: PREDICTION DASHBOARD
# ============================================================
if menu == "Prediction Dashboard":

    st.title("🔥 AI-Based Forest Fire Predictor")

    forest = st.selectbox("Select Forest", forest_list)

    if st.button("Predict Fire Risk"):

        lat, lon = geocode_forest(forest)

        if lat is None:
            st.error("Forest not found!")
            st.stop()

        df = generate_environment(lat, lon)

        # Encoding
        df_oh = pd.get_dummies(df["landcover_class"], prefix="landcover_class")

        for col in encoder_cols:
            df_oh[col] = df_oh.get(col, 0)

        df = pd.concat([df.drop(columns=["landcover_class"]), df_oh[encoder_cols]], axis=1)
        df = df.reindex(columns=feature_cols)

        pred = int(model.predict(df)[0])

        st.subheader("📍 Location")
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))

        col1, col2, col3 = st.columns(3)
        col1.metric("Temperature", f"{df.temperature_c.iloc[0]:.2f} °C")
        col2.metric("Humidity", f"{df.humidity_pct.iloc[0]:.2f} %")
        col3.metric("Wind Speed", f"{df.wind_speed_m_s.iloc[0]:.2f} m/s")

        if pred == 1:
            st.error("🔥 HIGH FIRE RISK")
        else:
            st.success("🌿 LOW FIRE RISK")

# ============================================================
# PAGE 2: EDA ANALYTICS
# ============================================================
elif menu == "EDA Analytics":
    st.title("📊 EDA Analytics")

    data = pd.DataFrame({
        "Temperature": np.random.randint(20, 40, 50),
        "Humidity": np.random.randint(30, 80, 50),
        "Wind Speed": np.random.randint(1, 10, 50)
    })

    st.dataframe(data)
    st.line_chart(data)

# ============================================================
# PAGE 3: DANGER CALCULATOR
# ============================================================
elif menu == "Danger Calculator":
    st.title("🔥 Danger Score Calculator")

    temp = st.slider("Temperature (°C)", 0, 50, 25)
    hum = st.slider("Humidity (%)", 0, 100, 50)
    wind = st.slider("Wind Speed", 0, 20, 5)

    score = temp * 0.5 + wind * 2 - hum * 0.2

    st.subheader(f"🔥 Danger Score: {score:.2f}")

    if score > 30:
        st.error("HIGH RISK")
    else:
        st.success("LOW RISK")

# ============================================================
# PAGE 4: DATASET EXPLORER
# ============================================================
elif menu == "Dataset Explorer":
    st.title("📂 Dataset Explorer")

    data = pd.DataFrame({
        "Latitude": np.random.randn(20),
        "Longitude": np.random.randn(20),
        "Temp": np.random.randint(20, 40, 20)
    })

    st.dataframe(data)

# ============================================================
# PAGE 5: PROJECT REPORT
# ============================================================
elif menu == "Project Report":
    st.title("📘 Project Report")

    st.markdown("""
    ### AI-Based Forest Fire Prediction System

    This system uses Machine Learning to predict forest fire risk based on environmental factors.

    **Features:**
    - Fire Risk Prediction  
    - Data Analysis  
    - Danger Score Calculator  
    - Dataset Exploration  
    """)
