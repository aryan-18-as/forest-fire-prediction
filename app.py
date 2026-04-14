import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="AI Forest Fire Predictor", layout="wide", page_icon="🔥")

# ============================================================
# LOAD MODEL
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
# FOREST DATA (NO API)
# ============================================================
forest_data = {
    "Amazon Rainforest": (-3.4653, -62.2159),
    "Sundarbans": (21.9497, 89.1833),
    "Jim Corbett Forest": (29.5300, 78.7747),
    "Gir Forest": (21.1240, 70.8245),
    "Black Forest": (48.0000, 8.0000),
    "Congo Rainforest": (-2.8797, 23.6560),
    "Daintree Rainforest": (-16.1700, 145.4180),
    "Borneo Rainforest": (0.9619, 114.5548),
}

forest_list = list(forest_data.keys())

# ============================================================
# GENERATE DATA
# ============================================================
def generate_environment(lat, lon):
    return pd.DataFrame([{
        "latitude": lat,
        "longitude": lon,
        "temperature_c": 25 + abs(lat % 5),
        "humidity_pct": 50 + abs(lon % 10),
        "wind_speed_m_s": 5,
        "precip_mm": 1,
        "ndvi": 0.5,
        "fwi_score": 20,
        "drought_code": 100,
        "forest_cover_pct": 70,
        "landcover_class": "Deciduous Forest",
        "elevation_m": 300,
        "slope_deg": 10,
        "population_density": 20
    }])

# ============================================================
# SIMPLE AI (NO API)
# ============================================================
def ai_forest_profile(forest):
    return f"{forest} is a major forest ecosystem with rich biodiversity and ecological importance."

def ai_fire_explanation(pred):
    return "High fire risk due to dry conditions." if pred else "Low fire risk due to favorable conditions."

def ai_recommend(pred):
    return "Avoid fire sources and monitor closely." if pred else "Maintain safety and awareness."

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
# MAIN PAGE
# ============================================================
if menu == "Prediction Dashboard":

    st.title("🔥 AI-Based Forest Fire Predictor")

    forest = st.selectbox("Select Forest", forest_list)

    if st.button("Predict Fire Risk"):

        lat, lon = forest_data[forest]

        df = generate_environment(lat, lon)

        df_oh = pd.get_dummies(df["landcover_class"], prefix="landcover_class")

        for col in encoder_cols:
            df_oh[col] = df_oh.get(col, 0)

        df = pd.concat([df.drop(columns=["landcover_class"]), df_oh[encoder_cols]], axis=1)
        df = df.reindex(columns=feature_cols)

        pred = int(model.predict(df)[0])

        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))

        if pred:
            st.error("🔥 HIGH FIRE RISK")
        else:
            st.success("🌿 LOW FIRE RISK")

        st.subheader("🌲 Forest Overview")
        st.write(ai_forest_profile(forest))

        st.subheader("🧠 Explanation")
        st.write(ai_fire_explanation(pred))

        st.subheader("⚠️ Recommendations")
        st.write(ai_recommend(pred))

# ============================================================
# MULTI PAGE
# ============================================================
elif menu == "EDA Analytics":
    from fire_pages import eda_page
    eda_page.run()

elif menu == "Danger Calculator":
    from fire_pages import danger_page
    danger_page.run()

elif menu == "Dataset Explorer":
    from fire_pages import dataset_page
    dataset_page.run()

elif menu == "Project Report":
    from fire_pages import report_page
    report_page.run()
