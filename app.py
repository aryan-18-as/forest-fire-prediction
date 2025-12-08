import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import google.generativeai as genai

# -----------------------------
# GEMINI AI CONFIG
# -----------------------------
GEMINI_API_KEY = "AIzaSyAgIYugXh-IpNsg9IpsjlItYz0fEmd5gec"   # your key
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel("gemini-pro")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Forest Fire Predictor", page_icon="🔥", layout="wide")

# -----------------------------
# GLOBAL CSS
# -----------------------------
st.markdown("""
<style>

[data-testid="stSidebar"] {
    background: #1e1e1e;
    color: white;
}

.sidebar-title {
    font-size: 28px;
    font-weight: 800;
    color: white;
    text-align: center;
    padding: 15px 0;
}

.main-title {
    font-size: 48px;
    font-weight: 900;
    text-align: center;
    padding: 18px;
    background: linear-gradient(90deg, #ff512f, #dd2476);
    -webkit-background-clip: text;
    color: transparent;
}

.pred-high {
    padding: 28px;
    border-radius: 12px;
    text-align: center;
    font-size: 32px;
    font-weight: 800;
    color: white;
    background: linear-gradient(135deg, #ff512f, #dd2476);
    box-shadow: 0 4px 15px rgba(255,0,0,0.25);
    margin-top: 25px;
}

.pred-low {
    padding: 28px;
    border-radius: 12px;
    text-align: center;
    font-size: 32px;
    font-weight: 800;
    color: #16561F;
    background: linear-gradient(135deg, #b9f6ca, #69f0ae);
    box-shadow: 0 4px 15px rgba(0,200,0,0.25);
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL FILES
# -----------------------------
@st.cache_resource
def load_all():
    model = joblib.load("fire_model.pkl")
    scaler = joblib.load("scaler (2).pkl")
    encoder = joblib.load("encoder.pkl")
    feature_cols = joblib.load("feature_columns_1.pkl")
    return model, scaler, encoder, feature_cols

model, scaler, encoder, feature_cols = load_all()

# -----------------------------
# OTHER KEYS
# -----------------------------
OPENCAGE_API_KEY = "95df23a7370340468757cad17a479691"

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def geocode_forest(name):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={name}&key={OPENCAGE_API_KEY}"
    res = requests.get(url).json()
    if res["total_results"] == 0:
        return None, None
    return res["results"][0]["geometry"]["lat"], res["results"][0]["geometry"]["lng"]

def generate_environment(lat, lon):
    temperature = 20 + abs(lat % 10)
    humidity = 40 + abs(lon % 20)
    wind_speed = 2 + (abs(lat + lon) % 5)
    precip = abs(lat - lon) % 3
    
    ndvi = np.clip(humidity/100 - 0.3, 0, 1)
    fwi = wind_speed * (1 - humidity/100) * 25
    drought_code = max(20, (temperature*2) - precip)

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


# -----------------------------
# GEMINI AI FUNCTIONS
# -----------------------------
def ai_fire_explanation(df, prediction, forest):
    info = df.to_dict(orient="records")[0]
    prompt = f"""
    Forest: {forest}
    Prediction: {"High" if prediction == 1 else "Low"} Fire Risk
    Data: {info}

    Write a 5–6 line expert explanation describing:
    - Why this fire risk level was predicted
    - Key contributing environmental factors
    - Safety interpretation of values
    - Meaningful insights about fire probability
    """

    try:
        return model_gemini.generate_content(prompt).text
    except:
        return "AI explanation unavailable."


def ai_forest_profile(forest):
    prompt = f"""
    Write a 4–6 line professional overview of the forest '{forest}' including:
    - Climate 
    - Vegetation
    - Fire susceptibility
    - Human interaction impacts
    """
    try:
        return model_gemini.generate_content(prompt).text
    except:
        return "Forest profile unavailable."


def ai_recommendations(prediction):
    prompt = f"""
    Fire risk: {"High" if prediction == 1 else "Low"}.
    Give 5 practical, expert-level safety recommendations.
    """
    try:
        return model_gemini.generate_content(prompt).text
    except:
        return "Recommendations unavailable."


# -----------------------------
# SIDEBAR MENU
# -----------------------------
st.sidebar.markdown("<div class='sidebar-title'>🔥 Fire Prediction Suite</div>", unsafe_allow_html=True)

menu = st.sidebar.radio("", ["Prediction Dashboard", "EDA Analytics", "Danger Calculator", "Dataset Explorer", "Project Report"])

# -----------------------------
# PAGE 1 — PREDICTION
# -----------------------------
if menu == "Prediction Dashboard":

    st.markdown("<div class='main-title'>Forest Fire Risk Predictor</div>", unsafe_allow_html=True)

    forest_name = st.text_input("Enter Forest Name", "Amazon")

    if st.button("Predict Fire Risk", use_container_width=True):

        lat, lon = geocode_forest(forest_name)
        if lat is None:
            st.error("Forest not found.")
            st.stop()

        df = generate_environment(lat, lon)

        try:
            df["landcover_class_encoded"] = encoder.transform(df["landcover_class"])
        except:
            df["landcover_class_encoded"] = encoder.transform(["Deciduous Forest"])

        df = df.drop(columns=["landcover_class"])
        df = df.reindex(columns=feature_cols)
        df_scaled = scaler.transform(df)

        pred = model.predict(df_scaled)[0]

        # Show Map
        st.subheader("Location")
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Temperature", f"{df.temperature_c.values[0]}°C")
        col2.metric("Humidity", f"{df.humidity_pct.values[0]}%")
        col3.metric("Wind", f"{df.wind_speed_m_s.values[0]} m/s")

        # Prediction Box
        if pred == 1:
            st.markdown("<div class='pred-high'>🔥 FIRE RISK DETECTED</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='pred-low'>🌿 NO FIRE RISK</div>", unsafe_allow_html=True)

        # AI Sections
        with st.expander("🌲 Forest Overview (AI)"):
            st.write(ai_forest_profile(forest_name))

        st.subheader("🧠 AI-Generated Explanation")
        st.write(ai_fire_explanation(df, pred, forest_name))

        with st.expander("🛡 Safety Recommendations (AI)"):
            st.write(ai_recommendations(pred))


# -----------------------------
# OTHER PAGES
# -----------------------------
elif menu == "EDA Analytics":
    import fire_pages.eda_page as page
    page.run()

elif menu == "Danger Calculator":
    import fire_pages.danger_page as page
    page.run()

elif menu == "Dataset Explorer":
    import fire_pages.dataset_page as page
    page.run()

elif menu == "Project Report":
    import fire_pages.report_page as page
    page.run()
