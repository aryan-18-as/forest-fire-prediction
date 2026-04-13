import streamlit as st
import pandas as pd
import numpy as np
import joblib
from groq import Groq

# ============================================================
# 🔑 PUT YOUR GROQ API KEY HERE (OPTIONAL)
# ============================================================
GROQ_API_KEY = "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"  # If not available, app still works

# Safe Groq init
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except:
    groq_client = None

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Forest Fire Predictor",
    page_icon="🔥",
    layout="wide"
)

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
# 🌍 FOREST LIST + COORDINATES (NO API NEEDED)
# ============================================================
forest_data = {
    "Amazon Rainforest Brazil": (-3.4653, -62.2159),
    "Sundarbans India": (21.9497, 89.1833),
    "Jim Corbett National Park India": (29.5300, 78.7747),
    "Gir National Park India": (21.1240, 70.8245),
    "Black Forest Germany": (48.0000, 8.0000),
    "Congo Rainforest Africa": (-2.8797, 23.6560),
    "Daintree Rainforest Australia": (-16.1700, 145.4180),
    "Sherwood Forest England": (53.2000, -1.0667),
    "Sequoia National Park USA": (36.4864, -118.5658),
    "Nilgiri Forest India": (11.4102, 76.6950),
    "Kaziranga National Park India": (26.5775, 93.1711),
    "Bandipur National Park India": (11.6586, 76.6293),
    "Borneo Rainforest Indonesia": (0.9619, 114.5548),
    "Satpura National Park India": (22.5420, 78.3450),
    "Periyar National Park India": (9.4627, 77.2367),
    "Great Bear Rainforest Canada": (52.5000, -128.0000)
}

forest_list = list(forest_data.keys())

# ============================================================
# 🔥 GENERATE ENVIRONMENT DATA
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
# 🤖 AI FUNCTIONS (SAFE)
# ============================================================
def groq_ai(prompt):
    if groq_client is None:
        return "AI not configured. Please add API key."

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except:
        return "AI temporarily unavailable."

def ai_forest_profile(forest):
    return groq_ai(f"Give a short overview of {forest}.")

def ai_fire_explanation(pred, forest):
    return groq_ai(f"Explain why fire risk is {'HIGH' if pred else 'LOW'} for {forest}.")

def ai_recommend(pred):
    return groq_ai(f"Give safety tips for {'high' if pred else 'low'} fire risk.")

# ============================================================
# 📌 SIDEBAR
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
# 🏠 MAIN PAGE
# ============================================================
if menu == "Prediction Dashboard":

    st.markdown("<h1 style='text-align:center;color:#ff3366;'>AI-Based Forest Fire Predictor</h1>",
                unsafe_allow_html=True)

    forest = st.selectbox("Select Forest", forest_list)

    if st.button("Predict Fire Risk", use_container_width=True):

        lat, lon = forest_data[forest]

        df = generate_environment(lat, lon)

        # Encoding
        df_oh = pd.get_dummies(df["landcover_class"], prefix="landcover_class")
        for col in encoder_cols:
            df_oh[col] = df_oh.get(col, 0)

        df = pd.concat([df.drop(columns=["landcover_class"]), df_oh[encoder_cols]], axis=1)
        df = df.reindex(columns=feature_cols)

        pred = int(model.predict(df)[0])

        # Map
        st.subheader("📍 Location")
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Temperature", f"{df.temperature_c.iloc[0]:.2f} °C")
        c2.metric("Humidity", f"{df.humidity_pct.iloc[0]:.2f} %")
        c3.metric("Wind Speed", f"{df.wind_speed_m_s.iloc[0]:.2f} m/s")

        # Result UI
        if pred == 1:
            st.error("🔥 HIGH FIRE RISK")
        else:
            st.success("🌿 LOW FIRE RISK")

        # AI Sections
        st.markdown("## 🌲 Forest Overview (AI)")
        st.write(ai_forest_profile(forest))

        st.markdown("## 🧠 AI Explanation")
        st.write(ai_fire_explanation(pred, forest))

        st.markdown("## ⚠️ Safety Recommendations")
        st.write(ai_recommend(pred))

# ============================================================
# 🔀 MULTI PAGE ROUTING
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
