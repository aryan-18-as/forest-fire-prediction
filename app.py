import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Optional Groq (will not break if missing)
try:
    from groq import Groq
    GROQ_API_KEY = "YOUR_GROQ_KEY"
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
# GENERATE DATA
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
# AI FUNCTIONS (HYBRID - GROQ + FALLBACK)
# ============================================================
def groq_ai(prompt):
    if groq_client is None:
        return None
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except:
        return None

def ai_forest_profile(forest):
    res = groq_ai(f"Give a short overview of {forest}.")
    if res:
        return res
    return f"{forest} is an important forest ecosystem with rich biodiversity and ecological value."

def ai_fire_explanation(pred, forest):
    res = groq_ai(f"Explain fire risk for {forest}.")
    if res:
        return res
    return "High fire risk due to dry conditions." if pred else "Low fire risk due to favorable conditions."

def ai_recommend(pred):
    res = groq_ai("Give fire safety tips.")
    if res:
        return res
    return (
        "• Avoid open flames\n• Monitor forest\n• Inform authorities\n• Maintain firebreaks"
        if pred else
        "• Maintain cleanliness\n• Monitor weather\n• Educate visitors\n• Keep safety ready"
    )

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
        st.write(ai_fire_explanation(pred, forest))

        st.subheader("⚠️ Recommendations")
        st.write(ai_recommend(pred))

# ============================================================
# MULTI PAGE ROUTING
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
