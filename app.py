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
# FUNCTIONS
# ================================================================
def geocode_forest(name):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={name}&key={OPENCAGE_API_KEY}"
    r = requests.get(url).json()
    if r.get("total_results", 0) == 0:
        return None, None
    return r["results"][0]["geometry"]["lat"], r["results"][0]["geometry"]["lng"]

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

# ================================================================
# AI FUNCTIONS (Groq)
# ================================================================
def groq_ai(prompt):
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"AI unavailable. (Error: {e})"

def ai_forest_profile(forest):
    return groq_ai(f"Give a 5-line overview about the forest '{forest}'.")

def ai_fire_explanation(df, pred, forest):
    return groq_ai(
        f"Explain in 5 lines why the prediction for forest '{forest}' is "
        f"{'HIGH' if pred else 'LOW'} based on the data: {df.to_dict()}."
    )

def ai_recommend(pred):
    return groq_ai(
        f"Give 5 safety recommendations for {'high' if pred else 'low'} fire risk."
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
# SIDEBAR
# ================================================================
with st.sidebar:
    st.title("🔥 Fire Prediction Suite")
    menu = st.radio("Navigation", [
        "Prediction Dashboard", "EDA Analytics",
        "Danger Calculator", "Dataset Explorer", "Project Report"
    ])

# ================================================================
# MAIN PAGE: PREDICTION DASHBOARD
# ================================================================
if menu == "Prediction Dashboard":

    st.markdown("<h1 style='text-align:center;color:#ff3366;'>AI-Based Forest Fire Predictor</h1>",
                unsafe_allow_html=True)

    forest = st.selectbox("Select Forest", forest_list)

    if st.button("Predict Fire Risk", use_container_width=True):

        lat, lon = geocode_forest(forest)
        if lat is None:
            st.error("Forest not found.")
            st.stop()

        df = generate_environment(lat, lon)

        df_oh = pd.get_dummies(df["landcover_class"], prefix="landcover_class")
        for col in encoder_cols:
            df_oh[col] = df_oh.get(col, 0)

        df = pd.concat([df.drop(columns=["landcover_class"]), df_oh[encoder_cols]], axis=1)
        df = df.reindex(columns=feature_cols)

        pred = int(model.predict(df)[0])

        # 📍 MAP
        st.subheader("📍 Location")
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))

        # TEMP / HUM / WIND
        c1, c2, c3 = st.columns(3)
        c1.metric("Temperature", f"{df.temperature_c.iloc[0]:.2f} °C")
        c2.metric("Humidity", f"{df.humidity_pct.iloc[0]:.2f} %")
        c3.metric("Wind", f"{df.wind_speed_m_s.iloc[0]:.2f} m/s")

        # ============================================================
        # FIRE RISK UI
        # ============================================================
        if pred == 1:
            st.markdown("""
                <div style="
                    padding:28px;
                    border-radius:18px;
                    text-align:center;
                    font-size:36px;
                    font-weight:900;
                    color:white;
                    background: linear-gradient(135deg, #ff0000, #ff5722);
                    box-shadow:0 0 25px rgba(255, 0, 0, 0.7);
                    animation: pulse 1.5s infinite;
                ">
                    🔥🔥 HIGH FIRE RISK 🔥🔥
                </div>

                <style>
                @keyframes pulse {
                    0% { transform: scale(1); }
                    50% { transform: scale(1.05); }
                    100% { transform: scale(1); }
                }
                </style>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="
                    padding:28px;
                    border-radius:18px;
                    text-align:center;
                    font-size:36px;
                    font-weight:900;
                    color:#003d1f;
                    background: linear-gradient(135deg, #b2f7e9, #7effb2);
                    box-shadow:0 0 18px rgba(0, 200, 100, 0.5);
                ">
                    🌿 SAFE — LOW FIRE RISK 🌿
                </div>
            """, unsafe_allow_html=True)

        # ============================================================
        # AI OUTPUT CARDS
        # ============================================================

        # FOREST OVERVIEW
        st.markdown("""<div style="
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 20px 25px;
            margin-top: 25px;
            border: 1px solid rgba(255,255,255,0.15);
            box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        "><h2 style="color:#00e676;font-weight:800;">🌲 Forest Overview (AI)</h2></div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="
            background: rgba(0,0,0,0.30);
            padding:18px;
            border-radius:12px;
            border-left:4px solid #00e676;
            font-size:18px;
            color:white;
        ">{ai_forest_profile(forest)}</div>
        """, unsafe_allow_html=True)

        # EXPLANATION
        st.markdown("""<div style="
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding:20px 25px;
            border:1px solid rgba(255,255,255,0.15);
            margin-top:20px;
        "><h2 style="color:#40c4ff;font-weight:800;">🧠 AI Explanation</h2></div>""",
                     unsafe_allow_html=True)

        st.markdown(f"""
        <div style="
            background: rgba(0,0,0,0.30);
            padding:18px;
            border-radius:12px;
            border-left:4px solid #40c4ff;
            font-size:18px;
            color:white;
        ">{ai_fire_explanation(df, pred, forest)}</div>
        """, unsafe_allow_html=True)

        # SAFETY
        st.markdown("""<div style="
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(10px);
            border-radius:16px;
            padding:20px 25px;
            border:1px solid rgba(255,255,255,0.15);
            margin-top:20px;
        "><h2 style="color:#ff8a80;font-weight:800;">♡ Safety Recommendations (AI)</h2></div>""",
                     unsafe_allow_html=True)

        with st.expander("Click to view recommendations"):
            st.markdown(f"""
            <div style="
                background: rgba(0,0,0,0.30);
                padding:18px;
                border-radius:12px;
                border-left:4px solid #ff8a80;
                font-size:18px;
                color:white;
            ">{ai_recommend(pred)}</div>
            """, unsafe_allow_html=True)


# ================================================================
# OTHER PAGES
# ================================================================
elif menu == "EDA Analytics":
    import fire_pages.eda_page as p; p.run()

elif menu == "Danger Calculator":
    import fire_pages.danger_page as p; p.run()

elif menu == "Dataset Explorer":
    import fire_pages.dataset_page as p; p.run()

elif menu == "Project Report":
    import fire_pages.report_page as p; p.run()
