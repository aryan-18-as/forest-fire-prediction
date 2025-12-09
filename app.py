import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from groq import Groq

# ================================================================
# API KEYS
# ================================================================
LOCATIONIQ_API_KEY = "a7271ad912be4dd1b3db39fe46004c09"
GROQ_API_KEY = "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"

groq_client = Groq(api_key=GROQ_API_KEY)

# ================================================================
# STREAMLIT PAGE CONFIG
# ================================================================
st.set_page_config(page_title="AI Forest Fire Predictor",
                   layout="wide",
                   page_icon="🔥")

# ================================================================
# LOAD ML FILES
# ================================================================
@st.cache_resource
def load_all():
    model = joblib.load("fire_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder_dict = joblib.load("encoders.pkl")       # one-hot encoding dict
    feature_cols = joblib.load("feature_columns.pkl")
    return model, scaler, encoder_dict, feature_cols

model, scaler, encoder_dict, feature_cols = load_all()
encoder_cols = list(encoder_dict.keys())             # one-hot column names

# ================================================================
# FREE GEOCODING (LocationIQ)
# ================================================================
forest_coordinates = {
    "Amazon": (-3.4653, -62.2159),
    "Congo Rainforest": (-1.4419, 15.5560),
    "Borneo Rainforest": (0.9619, 114.5548),
    "Great Bear Rainforest": (52.0, -127.5),
    "Sundarbans": (21.9497, 89.1833),
    "Jim Corbett Forest": (29.5300, 78.7740),
    "Gir Forest": (21.1240, 70.8240),
    "Black Forest": (48.1000, 8.2000),
    "Daintree Rainforest": (-16.1700, 145.4185),
    "Sherwood Forest": (53.2000, -1.0667),
    "Sequoia National Forest": (36.2950, -118.5640),
    "Nilgiri Forest": (11.4916, 76.7337),
    "Kaziranga Forest": (26.5775, 93.1711),
    "Bandipur Forest": (11.6577, 76.6295),
    "Satpura Forest": (22.5021, 78.3495),
    "Periyar Forest": (9.4669, 77.1560)
}

def geocode_forest(name):
    # Fallback fixed coordinates
    if name in forest_coordinates:
        return forest_coordinates[name]
    
    url = f"https://us1.locationiq.com/v1/search?key={LOCATIONIQ_API_KEY}&q={name}&format=json"
    try:
        r = requests.get(url).json()
        lat = float(r[0]["lat"])
        lon = float(r[0]["lon"])
        return lat, lon
    except:
        return None, None

# ================================================================
# STATIC MAP (FREE)
# ================================================================
def show_static_map(lat, lon):
    map_url = (
        f"https://maps.locationiq.com/v3/staticmap?"
        f"key={LOCATIONIQ_API_KEY}"
        f"&center={lat},{lon}"
        f"&zoom=7"
        f"&size=900x500"
        f"&markers=icon:large-red-cutout|{lat},{lon}"
    )
    st.image(map_url, caption="Forest Location (Map)", use_column_width=True)

# ================================================================
# ENVIRONMENT FEATURE GENERATION
# ================================================================
def generate_environment(lat, lon):
    temp = 20 + abs(lat % 11)
    hum = 40 + abs(lon % 21)
    wind = 2 + (abs(lat + lon) % 6)
    precip = abs(lat - lon) % 4
    ndvi = np.clip(hum/100 - 0.3, 0, 1)
    fwi = wind * (1 - hum/100) * 22

    return pd.DataFrame([{
        "latitude": lat,
        "longitude": lon,
        "temperature_c": temp,
        "humidity_pct": hum,
        "wind_speed_m_s": wind,
        "precip_mm": precip,
        "ndvi": ndvi,
        "fwi_score": fwi,
        "drought_code": max(20, temp*2 - precip),
        "forest_cover_pct": 70,
        "landcover_class": "Deciduous Forest",
        "elevation_m": 300,
        "slope_deg": 12,
        "population_density": 18
    }])

# ================================================================
# AI EXPLANATION (GROQ)
# ================================================================
def groq_ai(prompt):
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"AI Error: {e}"

def ai_forest_profile(forest): 
    return groq_ai(f"Give a 5-line expert overview of '{forest}' forest ecosystem & climate.")

def ai_fire_explanation(df, pred, forest):
    return groq_ai(
        f"Explain briefly why forest '{forest}' fire risk predicted as {'HIGH' if pred else 'LOW'} "
        f"based on this environmental data: {df.to_dict()}."
    )

def ai_recommend(pred):
    return groq_ai(
        f"Give 5 useful fire safety recommendations for {'high' if pred else 'low'} fire risk."
    )

# ================================================================
# SIDEBAR NAVIGATION
# ================================================================
with st.sidebar:
    st.title("🔥 Fire Prediction Suite")
    menu = st.radio("Navigate", [
        "Prediction Dashboard",
        "EDA Analytics",
        "Danger Calculator",
        "Dataset Explorer",
        "Project Report"
    ])

# ================================================================
# MAIN DASHBOARD
# ================================================================
if menu == "Prediction Dashboard":

    st.markdown("<h1 style='text-align:center;color:#ff3366;'>AI-Based Forest Fire Predictor</h1>", unsafe_allow_html=True)

    forest = st.selectbox("Select a Forest", list(forest_coordinates.keys()))

    if st.button("Predict Fire Risk", use_container_width=True):

        lat, lon = geocode_forest(forest)
        if lat is None:
            st.error("Forest not found.")
            st.stop()

        df = generate_environment(lat, lon)

        # One-hot encoding
        df_oh = pd.get_dummies(df["landcover_class"], prefix="landcover_class")
        for col in encoder_cols:
            df_oh[col] = df_oh.get(col, 0)

        df = pd.concat([df.drop(columns=["landcover_class"]), df_oh[encoder_cols]], axis=1)
        df = df.reindex(columns=feature_cols)

        pred = int(model.predict(df)[0])

        st.subheader("📍 Forest Location on Map")
        show_static_map(lat, lon)

        # SHOW METRICS
        col1, col2, col3 = st.columns(3)
        col1.metric("Temperature", f"{df.temperature_c.iloc[0]:.2f}°C")
        col2.metric("Humidity", f"{df.humidity_pct.iloc[0]:.2f}%")
        col3.metric("Wind Speed", f"{df.wind_speed_m_s.iloc[0]:.2f} m/s")

        # RESULT BOX
        if pred == 1:
            st.markdown("<div class='pred-high'>🔥 HIGH FIRE RISK</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='pred-low'>🌿 LOW FIRE RISK</div>", unsafe_allow_html=True)

        st.subheader("🌲 Forest Overview (AI)")
        st.write(ai_forest_profile(forest))

        st.subheader("🧠 Why This Prediction? (AI)")
        st.write(ai_fire_explanation(df, pred, forest))

        with st.expander("♡ Fire Safety Recommendations (AI)"):
            st.write(ai_recommend(pred))

# ================================================================
# EXTRA PAGES
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
