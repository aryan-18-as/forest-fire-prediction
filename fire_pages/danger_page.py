import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from groq import Groq

# ================================================================
# API KEYS
# ================================================================
OPENCAGE_API_KEY = "95df23a7370340468757cad17a479691"
GROQ_API_KEY = "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"

groq_client = Groq(api_key=GROQ_API_KEY)

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(page_title="AI Forest Fire Predictor", layout="wide", page_icon="🔥")

# ================================================================
# LOAD ML FILES
# ================================================================
@st.cache_resource
def load_all():
    model = joblib.load("fire_model.pkl")
    scaler = joblib.load("scaler (2).pkl")
    encoder_dict = joblib.load("encoders.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return model, scaler, encoder_dict, feature_cols

model, scaler, encoder_dict, feature_cols = load_all()
encoder_cols = list(encoder_dict.keys())

# ================================================================
# HELPER FUNCTIONS
# ================================================================
def geocode_forest(name):
    """Return latitude, longitude using OpenCage API."""
    url = f"https://api.opencagedata.com/geocode/v1/json?q={name}&key={OPENCAGE_API_KEY}"
    res = requests.get(url).json()

    if res.get("total_results", 0) == 0:
        return None, None
    
    loc = res["results"][0]["geometry"]
    return loc["lat"], loc["lng"]


def show_map(lat, lon):
    """Show free map (NO MAPBOX API needed)."""
    df = pd.DataFrame({"latitude": [lat], "longitude": [lon]})
    st.map(df, zoom=6)


def generate_environment(lat, lon):
    """Generate synthetic environment values."""
    temp = 20 + abs(lat % 10)
    hum = 40 + abs(lon % 20)
    wind = 2 + (abs(lat + lon) % 5)
    precip = abs(lat - lon) % 3
    ndvi = np.clip(hum / 100 - 0.3, 0, 1)
    fwi = wind * (1 - hum / 100) * 25

    return pd.DataFrame([{
        "latitude": lat,
        "longitude": lon,
        "temperature_c": temp,
        "humidity_pct": hum,
        "wind_speed_m_s": wind,
        "precip_mm": precip,
        "ndvi": ndvi,
        "fwi_score": fwi,
        "drought_code": max(20, (temp * 2) - precip),
        "forest_cover_pct": 70,
        "landcover_class": "Deciduous Forest",
        "elevation_m": 300,
        "slope_deg": 12,
        "population_density": 18
    }])

# ================================================================
# AI FUNCTIONS
# ================================================================
def groq_ai(prompt):
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"


def ai_forest_profile(forest):
    return groq_ai(
        f"Give a 5-line expert overview of the forest '{forest}' including vegetation, climate, and fire vulnerability."
    )

def ai_fire_explanation(df, pred, forest):
    return groq_ai(
        f"Explain in 5 lines why the model predicted {'HIGH' if pred else 'LOW'} fire risk for '{forest}'. "
        f"Use this data: {df.to_dict()}."
    )

def ai_recommend(pred):
    return groq_ai(
        f"Give 5 expert-level safety recommendations for {'HIGH' if pred else 'LOW'} fire risk."
    )


# ================================================================
# FOREST LIST
# ================================================================
forest_list = [
    "Amazon", "Sundarbans", "Jim Corbett Forest", "Gir Forest",
    "Black Forest", "Congo Rainforest", "Daintree Rainforest",
    "Sherwood Forest", "Sequoia National Forest", "Nilgiri Forest",
    "Kaziranga Forest", "Bandipur Forest", "Borneo Rainforest",
    "Satpura Forest", "Periyar Forest", "Great Bear Rainforest"
]

# ================================================================
# SIDEBAR NAVIGATION
# ================================================================
with st.sidebar:
    st.title("🔥 Fire Prediction Suite")
    menu = st.radio("Menu", ["Prediction Dashboard", "EDA Analytics",
                             "Danger Calculator", "Dataset Explorer",
                             "Project Report"])

# ================================================================
# PREDICTION DASHBOARD
# ================================================================
if menu == "Prediction Dashboard":
    st.markdown(
        "<h1 style='text-align:center; color:#ff3366;'>AI-Based Forest Fire Predictor</h1>",
        unsafe_allow_html=True
    )

    forest = st.selectbox("🌲 Select Forest", forest_list)

    if st.button("Predict Fire Risk", use_container_width=True):

        # --- Geocode ---
        lat, lon = geocode_forest(forest)
        if lat is None:
            st.error("Forest not found in geocoding API.")
            st.stop()

        # --- Show Map ---
        st.subheader("📍 Forest Location")
        show_map(lat, lon)

        # --- Generate Environment ---
        df = generate_environment(lat, lon)

        # --- One-Hot Encoding ---
        df_oh = pd.get_dummies(df["landcover_class"], prefix="landcover_class")
        for col in encoder_cols:
            if col not in df_oh.columns:
                df_oh[col] = 0

        df = pd.concat([df.drop(columns=["landcover_class"]), df_oh[encoder_cols]], axis=1)
        df = df.reindex(columns=feature_cols)

        # --- Prediction ---
        pred = int(model.predict(df)[0])

        # --- Metrics ---
        c1, c2, c3 = st.columns(3)
        c1.metric("Temperature", f"{df.temperature_c.iloc[0]:.1f} °C")
        c2.metric("Humidity", f"{df.humidity_pct.iloc[0]:.1f} %")
        c3.metric("Wind Speed", f"{df.wind_speed_m_s.iloc[0]:.1f} m/s")

        # --- Output ---
        if pred == 1:
            st.markdown("<div class='pred-high'>🔥 HIGH FIRE RISK</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='pred-low'>🌿 LOW FIRE RISK</div>", unsafe_allow_html=True)

        st.subheader("🌲 Forest Overview (AI)")
        st.write(ai_forest_profile(forest))

        st.subheader("🧠 AI Explanation")
        st.write(ai_fire_explanation(df, pred, forest))

        with st.expander("💡 Safety Recommendations"):
            st.write(ai_recommend(pred))

# ================================================================
# OTHER PAGES
# ================================================================
elif menu == "EDA Analytics":
    import fire_pages.eda_page as p
    p.run()

elif menu == "Danger Calculator":
    import fire_pages.danger_page as p
    p.run()

elif menu == "Dataset Explorer":
    import fire_pages.dataset_page as p
    p.run()

elif menu == "Project Report":
    import fire_pages.report_page as p
    p.run()
