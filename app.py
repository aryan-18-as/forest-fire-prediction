import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from groq import Groq

# ============================================
# 🔐 INSERT YOUR API KEYS HERE
# ============================================
OPENCAGE_API_KEY = "95df23a7370340468757cad17a479691"
GROQ_API_KEY = "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"

# Initialize Groq Client
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY != "YOUR_GROQ_KEY" else None


# ============================================
# STREAMLIT PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="AI Forest Fire Predictor",
    page_icon="🔥",
    layout="wide"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
[data-testid="stSidebar"] { background:#1e1e1e; color:white; }

.sidebar-title {
    font-size:26px; font-weight:800; padding:15px 5px; color:white;
}

.main-title {
    font-size:40px; font-weight:900; text-align:center;
    background:linear-gradient(90deg,#ff512f,#dd2476);
    -webkit-background-clip:text; color:transparent;
    padding:15px;
}

.pred-high {
    padding:28px; border-radius:16px;
    text-align:center; font-size:32px; font-weight:800;
    color:white;
    background:linear-gradient(135deg,#ff512f,#dd2476);
    box-shadow:0 4px 18px rgba(255,60,60,0.45);
    margin-top:25px;
}

.pred-low {
    padding:28px; border-radius:16px;
    text-align:center; font-size:32px; font-weight:800;
    color:#004d25;
    background:linear-gradient(135deg,#b9f6ca,#69f0ae);
    box-shadow:0 4px 18px rgba(0,200,120,0.35);
    margin-top:25px;
}
</style>
""", unsafe_allow_html=True)


# ============================================
# LOAD ML FILES
# ============================================
@st.cache_resource
def load_all():
    model = joblib.load("fire_model.pkl")
    scaler = joblib.load("scaler (2).pkl")
    encoder = joblib.load("encoder.pkl")
    feature_cols = joblib.load("feature_columns_1.pkl")
    return model, scaler, encoder, feature_cols

model, scaler, encoder, feature_cols = load_all()


# ============================================
# HELPER FUNCTIONS
# ============================================

def geocode_forest(name):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={name}&key={OPENCAGE_API_KEY}"
    r = requests.get(url).json()
    if r.get("total_results", 0) == 0:
        return None, None
    return r["results"][0]["geometry"]["lat"], r["results"][0]["geometry"]["lng"]


def generate_environment(lat, lon):
    temperature = 20 + abs(lat % 10)
    humidity = 40 + abs(lon % 20)
    wind = 2 + (abs(lat + lon) % 5)
    precip = abs(lat - lon) % 3

    ndvi = np.clip(humidity / 100 - 0.3, 0, 1)
    fwi = wind * (1 - humidity / 100) * 25

    return pd.DataFrame([{
        "latitude": lat,
        "longitude": lon,
        "temperature_c": temperature,
        "humidity_pct": humidity,
        "wind_speed_m_s": wind,
        "precip_mm": precip,
        "ndvi": ndvi,
        "fwi_score": fwi,
        "drought_code": max(20, (temperature * 2) - precip),
        "forest_cover_pct": 70,
        "landcover_class": "Deciduous Forest",
        "elevation_m": 300,
        "slope_deg": 12,
        "population_density": 18
    }])


# ============================================
# AI FUNCTIONS USING GROQ (llama-3.3-70b-versatile)
# ============================================

def groq_ai(prompt):
    if groq_client is None:
        return "AI unavailable – please add Groq API key."

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"AI unavailable. (Error: {e})"


def ai_forest_profile(forest):
    prompt = f"""
Give a clear 5-line overview for the forest '{forest}' covering:
- Region / country
- Climate
- Type of vegetation
- Fire vulnerability
- Notable environmental behavior
"""
    return groq_ai(prompt)


def ai_fire_explanation(df, pred, forest):
    prompt = f"""
Explain in exactly 5 lines why the forest '{forest}' has a predicted
fire risk level of {'HIGH' if pred else 'LOW'} using this data:
{df.to_dict()}

Use simple language but expert-style reasoning.
"""
    return groq_ai(prompt)


def ai_recommend(pred):
    prompt = f"""
Give exactly 5 bullet points with safety recommendations for {'HIGH' if pred else 'LOW'} fire risk areas.
Short and practical.
"""
    return groq_ai(prompt)


# ============================================
# SIDEBAR NAVIGATION
# ============================================
with st.sidebar:
    st.markdown("<div class='sidebar-title'>🔥 Fire Prediction Suite</div>", unsafe_allow_html=True)

    menu = st.radio(
        "Navigation",
        ["Prediction Dashboard", "EDA Analytics", "Danger Calculator", "Dataset Explorer", "Project Report"],
        key="navmenu"
    )


# ============================================
# PAGE 1: PREDICTION DASHBOARD
# ============================================
if menu == "Prediction Dashboard":

    st.markdown("<div class='main-title'>AI-Based Forest Fire Risk Predictor</div>", unsafe_allow_html=True)

    forest = st.text_input("Enter Forest Name", "Amazon")

    if st.button("Predict Fire Risk", use_container_width=True):

        lat, lon = geocode_forest(forest)

        if lat is None:
            st.error("Forest not found. Try another name.")
            st.stop()

        df = generate_environment(lat, lon)

        df["landcover_class_encoded"] = encoder.transform(["Deciduous Forest"])
        df = df.drop(columns=["landcover_class"])
        df = df.reindex(columns=feature_cols)

        scaled = scaler.transform(df)
        pred = int(model.predict(scaled)[0])

        st.subheader("📍 Location")
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))

        col1, col2, col3 = st.columns(3)
        col1.metric("Temperature", f"{df.temperature_c.iloc[0]:.2f} °C")
        col2.metric("Humidity", f"{df.humidity_pct.iloc[0]:.2f} %")
        col3.metric("Wind", f"{df.wind_speed_m_s.iloc[0]:.2f} m/s")

        if pred == 1:
            st.markdown("<div class='pred-high'>🔥 HIGH FIRE RISK</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='pred-low'>🌿 NO FIRE RISK</div>", unsafe_allow_html=True)

        st.subheader("🌲 Forest Overview (AI)")
        st.write(ai_forest_profile(forest))

        st.subheader("🧠 AI Explanation")
        st.write(ai_fire_explanation(df, pred, forest))

        with st.expander("♡ Safety Recommendations (AI)"):
            st.write(ai_recommend(pred))


# ============================================
# OTHER PAGES
# ============================================

elif menu == "EDA Analytics":
    try:
        import fire_pages.eda_page as p
        p.run()
    except:
        st.error("EDA page missing.")

elif menu == "Danger Calculator":
    try:
        import fire_pages.danger_page as p
        p.run()
    except:
        st.error("Danger Calculator page missing.")

elif menu == "Dataset Explorer":
    try:
        import fire_pages.dataset_page as p
        p.run()
    except:
        st.error("Dataset Explorer page missing.")

elif menu == "Project Report":
    try:
        import fire_pages.report_page as p
        p.run()
    except:
        st.error("Project Report page missing.")
