# app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from groq import Groq

# ---------------------------------------------------
# CONFIG: ADD YOUR KEYS HERE
# ---------------------------------------------------
GROQ_API_KEY = "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"

import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from groq import Groq

# ---------------------------------------------------
# CONFIG: ADD YOUR KEYS HERE
# ---------------------------------------------------
GROQ_API_KEY = "# app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from groq import Groq

# ---------------------------------------------------
# CONFIG: ADD YOUR KEYS HERE
# ---------------------------------------------------
GROQ_API_KEY = "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"
OPENCAGE_API_KEY = "95df23a7370340468757cad17a479691"

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Forest Fire Predictor",
    layout="wide",
    page_icon="🔥",
)

# ---------------------------------------------------
# CUSTOM CSS (Premium UI)
# ---------------------------------------------------
st.markdown(
    """
<style>
/* Sidebar background */
[data-testid="stSidebar"] {
    background: #1e1e1e;
    color: white;
}

/* Sidebar Title */
.sidebar-title {
    font-size: 26px;
    font-weight: 800;
    color: white;
    text-align: left;
    padding: 10px 5px 20px 5px;
}

/* Gradient Main Title */
.main-title {
    font-size: 42px;
    font-weight: 900;
    text-align: center;
    padding: 18px;
    background: linear-gradient(90deg, #ff512f, #dd2476);
    -webkit-background-clip: text;
    color: transparent;
}

/* Prediction Result Box */
.pred-high {
    padding: 28px;
    border-radius: 16px;
    text-align: center;
    font-size: 32px;
    font-weight: 800;
    color: #ffffff;
    background: linear-gradient(135deg, #ff512f, #dd2476);
    box-shadow: 0 4px 18px rgba(255, 60, 60, 0.45);
    margin-top: 25px;
}

.pred-low {
    padding: 28px;
    border-radius: 16px;
    text-align: center;
    font-size: 32px;
    font-weight: 800;
    color: #004d25;
    background: linear-gradient(135deg, #b9f6ca, #69f0ae);
    box-shadow: 0 4px 18px rgba(0, 200, 120, 0.35);
    margin-top: 25px;
}

/* Section headings */
section-title {
    font-size: 22px;
    font-weight: 700;
}

/* Metric labels a little bolder */
.css-1xarl3l, .css-10trblm {
    font-weight: 600 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# INITIALISE GROQ CLIENT (SAFE)
# ---------------------------------------------------
groq_client = None
if GROQ_API_KEY and GROQ_API_KEY != "gsk_d5he5aZmgnXwnFPo8IdZWGdyb3FYwzBWgXHLkMxJjc0UdKesIn1pE":
    groq_client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------
# LOAD ML FILES
# ---------------------------------------------------
@st.cache_resource
def load_all():
    model = joblib.load("fire_model.pkl")
    scaler = joblib.load("scaler (2).pkl")          # file name as you saved it
    encoder = joblib.load("encoder.pkl")
    feature_cols = joblib.load("feature_columns_1.pkl")
    return model, scaler, encoder, feature_cols


model, scaler, encoder, feature_cols = load_all()

# ---------------------------------------------------
# HELPER FUNCTIONS: GEO & ENVIRONMENT
# ---------------------------------------------------
def geocode_forest(name: str):
    """
    Get latitude and longitude for a forest name using OpenCage API.
    """
    if not OPENCAGE_API_KEY or OPENCAGE_API_KEY == "95df23a7370340468757cad17a479691":
        st.error("OpenCage API key is missing. Please add it in app.py.")
        return None, None

    url = f"https://api.opencagedata.com/geocode/v1/json?q={name}&key={OPENCAGE_API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return None, None

    if data.get("total_results", 0) == 0:
        return None, None

    lat = data["results"][0]["geometry"]["lat"]
    lon = data["results"][0]["geometry"]["lng"]
    return lat, lon


def generate_environment(lat: float, lon: float) -> pd.DataFrame:
    """
    Generate synthetic-but-consistent environmental data from coordinates.
    """
    temperature = 20 + abs(lat % 10)
    humidity = 40 + abs(lon % 20)
    wind_speed = 2 + (abs(lat + lon) % 5)
    precip = abs(lat - lon) % 3

    ndvi = np.clip(humidity / 100 - 0.3, 0, 1)
    fwi = wind_speed * (1 - humidity / 100) * 25
    drought_code = max(20, (temperature * 2) - precip)

    return pd.DataFrame(
        [
            {
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
                "population_density": 18,
            }
        ]
    )

# ---------------------------------------------------
# AI FUNCTIONS USING GROQ (LLAMA-3)
# ---------------------------------------------------
def groq_ai(prompt: str) -> str:
    """
    Call Groq LLaMA-3 to generate a short, focused answer.
    """
    if groq_client is None:
        return "AI unavailable (Groq API key not configured)."

    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI unavailable. ({e})"


def ai_forest_profile(forest: str) -> str:
    prompt = f"""
Provide a 5–6 line summary of the forest '{forest}' including:
- Location/region
- Typical climate
- Dominant vegetation
- Natural fire vulnerability
Use clear, non-technical language.
"""
    return groq_ai(prompt)


def ai_fire_explanation(df: pd.DataFrame, prediction: int, forest: str) -> str:
    info = df.to_dict(orient="records")[0]
    risk_level = "HIGH" if prediction == 1 else "LOW"

    prompt = f"""
Forest: {forest}
Predicted fire risk: {risk_level}

Environmental inputs:
{info}

Explain in 4–6 lines:
- Which variables influence this prediction the most
- Why these values indicate {risk_level} fire risk
- A concise expert-style explanation that a student can present.
"""
    return groq_ai(prompt)


def ai_recommendations(prediction: int) -> str:
    risk_level = "HIGH" if prediction == 1 else "LOW"
    prompt = f"""
Current fire risk level: {risk_level}.

List 5 practical wildfire safety recommendations as bullet points.
Keep them short and specific.
"""
    return groq_ai(prompt)

# ---------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------
st.sidebar.markdown(
    "<div class='sidebar-title'>🔥 Fire Prediction Suite</div>",
    unsafe_allow_html=True,
)

menu = st.sidebar.radio(
    "Navigation",
    [
        "Prediction Dashboard",
        "EDA Analytics",
        "Danger Calculator",
        "Dataset Explorer",
        "Project Report",
    ],
)

# ---------------------------------------------------
# PAGE 1: PREDICTION DASHBOARD
# ---------------------------------------------------
if menu == "Prediction Dashboard":
    st.markdown(
        "<div class='main-title'>AI-Based Forest Fire Risk Predictor</div>",
        unsafe_allow_html=True,
    )

    forest_name = st.text_input("Enter Forest Name", "Amazon")

    predict_btn = st.button("Predict Fire Risk", use_container_width=True)

    if predict_btn:
        # 1. Geocode
        lat, lon = geocode_forest(forest_name)
        if lat is None:
            st.error("Forest not found. Please try another name.")
        else:
            # 2. Generate environment data
            df = generate_environment(lat, lon)

            # 3. Encode landcover class
            try:
                df["landcover_class_encoded"] = encoder.transform(
                    df["landcover_class"]
                )
            except Exception:
                df["landcover_class_encoded"] = encoder.transform(
                    ["Deciduous Forest"]
                )

            df = df.drop(columns=["landcover_class"])

            # 4. Align columns & scale
            df = df.reindex(columns=feature_cols)
            df_scaled = scaler.transform(df)

            # 5. Predict
            pred = int(model.predict(df_scaled)[0])

            # 6. Show map
            st.subheader("Location on Map")
            st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))

            # 7. Key metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Temperature", f"{df['temperature_c'].iloc[0]:.2f} °C")
            c2.metric("Humidity", f"{df['humidity_pct'].iloc[0]:.2f} %")
            c3.metric("Wind Speed", f"{df['wind_speed_m_s'].iloc[0]:.2f} m/s")

            # 8. Prediction box
            if pred == 1:
                st.markdown(
                    "<div class='pred-high'>🔥 HIGH FIRE RISK DETECTED</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='pred-low'>🌿 NO FIRE RISK</div>",
                    unsafe_allow_html=True,
                )

            # 9. AI sections
            st.markdown("## 🌲 Forest Overview (AI)")
            st.markdown(ai_forest_profile(forest_name))

            st.markdown("## 🧠 AI-Generated Explanation")
            st.markdown(ai_fire_explanation(df, pred, forest_name))

            with st.expander("♡ Safety Recommendations (AI)"):
                st.markdown(ai_recommendations(pred))

            # 10. Show full input data (optional)
            st.markdown("## 📊 Environmental Data Used")
            st.json(df.to_dict(orient="records")[0])

# ---------------------------------------------------
# PAGE 2: EDA ANALYTICS
# ---------------------------------------------------
elif menu == "EDA Analytics":
    try:
        import fire_pages.eda_page as page

        page.run()
    except ImportError:
        st.error("EDA page module not found (fire_pages/eda_page.py).")

# ---------------------------------------------------
# PAGE 3: DANGER CALCULATOR
# ---------------------------------------------------
elif menu == "Danger Calculator":
    try:
        import fire_pages.danger_page as page

        page.run()
    except ImportError:
        st.error("Danger Calculator page module not found (fire_pages/danger_page.py).")

# ---------------------------------------------------
# PAGE 4: DATASET EXPLORER
# ---------------------------------------------------
elif menu == "Dataset Explorer":
    try:
        import fire_pages.dataset_page as page

        page.run()
    except ImportError:
        st.error("Dataset Explorer page module not found (fire_pages/dataset_page.py).")

# ---------------------------------------------------
# PAGE 5: PROJECT REPORT
# ---------------------------------------------------
elif menu == "Project Report":
    try:
        import fire_pages.report_page as page

        page.run()
    except ImportError:
        st.error("Project Report page module not found (fire_pages/report_page.py).")
"
OPENCAGE_API_KEY = "YOUR_OPENCAGE_API_KEY_HERE"

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Forest Fire Predictor",
    layout="wide",
    page_icon="🔥",
)

# ---------------------------------------------------
# CUSTOM CSS (Premium UI)
# ---------------------------------------------------
st.markdown(
    """
<style>
/* Sidebar background */
[data-testid="stSidebar"] {
    background: #1e1e1e;
    color: white;
}

/* Sidebar Title */
.sidebar-title {
    font-size: 26px;
    font-weight: 800;
    color: white;
    text-align: left;
    padding: 10px 5px 20px 5px;
}

/* Gradient Main Title */
.main-title {
    font-size: 42px;
    font-weight: 900;
    text-align: center;
    padding: 18px;
    background: linear-gradient(90deg, #ff512f, #dd2476);
    -webkit-background-clip: text;
    color: transparent;
}

/* Prediction Result Box */
.pred-high {
    padding: 28px;
    border-radius: 16px;
    text-align: center;
    font-size: 32px;
    font-weight: 800;
    color: #ffffff;
    background: linear-gradient(135deg, #ff512f, #dd2476);
    box-shadow: 0 4px 18px rgba(255, 60, 60, 0.45);
    margin-top: 25px;
}

.pred-low {
    padding: 28px;
    border-radius: 16px;
    text-align: center;
    font-size: 32px;
    font-weight: 800;
    color: #004d25;
    background: linear-gradient(135deg, #b9f6ca, #69f0ae);
    box-shadow: 0 4px 18px rgba(0, 200, 120, 0.35);
    margin-top: 25px;
}

/* Section headings */
section-title {
    font-size: 22px;
    font-weight: 700;
}

/* Metric labels a little bolder */
.css-1xarl3l, .css-10trblm {
    font-weight: 600 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# INITIALISE GROQ CLIENT (SAFE)
# ---------------------------------------------------
groq_client = None
if GROQ_API_KEY and GROQ_API_KEY != "YOUR_GROQ_API_KEY_HERE":
    groq_client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------
# LOAD ML FILES
# ---------------------------------------------------
@st.cache_resource
def load_all():
    model = joblib.load("fire_model.pkl")
    scaler = joblib.load("scaler (2).pkl")          # file name as you saved it
    encoder = joblib.load("encoder.pkl")
    feature_cols = joblib.load("feature_columns_1.pkl")
    return model, scaler, encoder, feature_cols


model, scaler, encoder, feature_cols = load_all()

# ---------------------------------------------------
# HELPER FUNCTIONS: GEO & ENVIRONMENT
# ---------------------------------------------------
def geocode_forest(name: str):
    """
    Get latitude and longitude for a forest name using OpenCage API.
    """
    if not OPENCAGE_API_KEY or OPENCAGE_API_KEY == "YOUR_OPENCAGE_API_KEY_HERE":
        st.error("OpenCage API key is missing. Please add it in app.py.")
        return None, None

    url = f"https://api.opencagedata.com/geocode/v1/json?q={name}&key={OPENCAGE_API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return None, None

    if data.get("total_results", 0) == 0:
        return None, None

    lat = data["results"][0]["geometry"]["lat"]
    lon = data["results"][0]["geometry"]["lng"]
    return lat, lon


def generate_environment(lat: float, lon: float) -> pd.DataFrame:
    """
    Generate synthetic-but-consistent environmental data from coordinates.
    """
    temperature = 20 + abs(lat % 10)
    humidity = 40 + abs(lon % 20)
    wind_speed = 2 + (abs(lat + lon) % 5)
    precip = abs(lat - lon) % 3

    ndvi = np.clip(humidity / 100 - 0.3, 0, 1)
    fwi = wind_speed * (1 - humidity / 100) * 25
    drought_code = max(20, (temperature * 2) - precip)

    return pd.DataFrame(
        [
            {
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
                "population_density": 18,
            }
        ]
    )

# ---------------------------------------------------
# AI FUNCTIONS USING GROQ (LLAMA-3)
# ---------------------------------------------------
def groq_ai(prompt: str) -> str:
    """
    Call Groq LLaMA-3 to generate a short, focused answer.
    """
    if groq_client is None:
        return "AI unavailable (Groq API key not configured)."

    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI unavailable. ({e})"


def ai_forest_profile(forest: str) -> str:
    prompt = f"""
Provide a 5–6 line summary of the forest '{forest}' including:
- Location/region
- Typical climate
- Dominant vegetation
- Natural fire vulnerability
Use clear, non-technical language.
"""
    return groq_ai(prompt)


def ai_fire_explanation(df: pd.DataFrame, prediction: int, forest: str) -> str:
    info = df.to_dict(orient="records")[0]
    risk_level = "HIGH" if prediction == 1 else "LOW"

    prompt = f"""
Forest: {forest}
Predicted fire risk: {risk_level}

Environmental inputs:
{info}

Explain in 4–6 lines:
- Which variables influence this prediction the most
- Why these values indicate {risk_level} fire risk
- A concise expert-style explanation that a student can present.
"""
    return groq_ai(prompt)


def ai_recommendations(prediction: int) -> str:
    risk_level = "HIGH" if prediction == 1 else "LOW"
    prompt = f"""
Current fire risk level: {risk_level}.

List 5 practical wildfire safety recommendations as bullet points.
Keep them short and specific.
"""
    return groq_ai(prompt)

# ---------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------
st.sidebar.markdown(
    "<div class='sidebar-title'>🔥 Fire Prediction Suite</div>",
    unsafe_allow_html=True,
)

menu = st.sidebar.radio(
    "Navigation",
    [
        "Prediction Dashboard",
        "EDA Analytics",
        "Danger Calculator",
        "Dataset Explorer",
        "Project Report",
    ],
)

# ---------------------------------------------------
# PAGE 1: PREDICTION DASHBOARD
# ---------------------------------------------------
if menu == "Prediction Dashboard":
    st.markdown(
        "<div class='main-title'>AI-Based Forest Fire Risk Predictor</div>",
        unsafe_allow_html=True,
    )

    forest_name = st.text_input("Enter Forest Name", "Amazon")

    predict_btn = st.button("Predict Fire Risk", use_container_width=True)

    if predict_btn:
        # 1. Geocode
        lat, lon = geocode_forest(forest_name)
        if lat is None:
            st.error("Forest not found. Please try another name.")
        else:
            # 2. Generate environment data
            df = generate_environment(lat, lon)

            # 3. Encode landcover class
            try:
                df["landcover_class_encoded"] = encoder.transform(
                    df["landcover_class"]
                )
            except Exception:
                df["landcover_class_encoded"] = encoder.transform(
                    ["Deciduous Forest"]
                )

            df = df.drop(columns=["landcover_class"])

            # 4. Align columns & scale
            df = df.reindex(columns=feature_cols)
            df_scaled = scaler.transform(df)

            # 5. Predict
            pred = int(model.predict(df_scaled)[0])

            # 6. Show map
            st.subheader("Location on Map")
            st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))

            # 7. Key metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Temperature", f"{df['temperature_c'].iloc[0]:.2f} °C")
            c2.metric("Humidity", f"{df['humidity_pct'].iloc[0]:.2f} %")
            c3.metric("Wind Speed", f"{df['wind_speed_m_s'].iloc[0]:.2f} m/s")

            # 8. Prediction box
            if pred == 1:
                st.markdown(
                    "<div class='pred-high'>🔥 HIGH FIRE RISK DETECTED</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='pred-low'>🌿 NO FIRE RISK</div>",
                    unsafe_allow_html=True,
                )

            # 9. AI sections
            st.markdown("## 🌲 Forest Overview (AI)")
            st.markdown(ai_forest_profile(forest_name))

            st.markdown("## 🧠 AI-Generated Explanation")
            st.markdown(ai_fire_explanation(df, pred, forest_name))

            with st.expander("♡ Safety Recommendations (AI)"):
                st.markdown(ai_recommendations(pred))

            # 10. Show full input data (optional)
            st.markdown("## 📊 Environmental Data Used")
            st.json(df.to_dict(orient="records")[0])

# ---------------------------------------------------
# PAGE 2: EDA ANALYTICS
# ---------------------------------------------------
elif menu == "EDA Analytics":
    try:
        import fire_pages.eda_page as page

        page.run()
    except ImportError:
        st.error("EDA page module not found (fire_pages/eda_page.py).")

# ---------------------------------------------------
# PAGE 3: DANGER CALCULATOR
# ---------------------------------------------------
elif menu == "Danger Calculator":
    try:
        import fire_pages.danger_page as page

        page.run()
    except ImportError:
        st.error("Danger Calculator page module not found (fire_pages/danger_page.py).")

# ---------------------------------------------------
# PAGE 4: DATASET EXPLORER
# ---------------------------------------------------
elif menu == "Dataset Explorer":
    try:
        import fire_pages.dataset_page as page

        page.run()
    except ImportError:
        st.error("Dataset Explorer page module not found (fire_pages/dataset_page.py).")

# ---------------------------------------------------
# PAGE 5: PROJECT REPORT
# ---------------------------------------------------
elif menu == "Project Report":
    try:
        import fire_pages.report_page as page

        page.run()
    except ImportError:
        st.error("Project Report page module not found (fire_pages/report_page.py).")
"
OPENCAGE_API_KEY = "YOUR_OPENCAGE_API_KEY_HERE"

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Forest Fire Predictor",
    layout="wide",
    page_icon="🔥",
)

# ---------------------------------------------------
# CUSTOM CSS (Premium UI)
# ---------------------------------------------------
st.markdown(
    """
<style>
/* Sidebar background */
[data-testid="stSidebar"] {
    background: #1e1e1e;
    color: white;
}

/* Sidebar Title */
.sidebar-title {
    font-size: 26px;
    font-weight: 800;
    color: white;
    text-align: left;
    padding: 10px 5px 20px 5px;
}

/* Gradient Main Title */
.main-title {
    font-size: 42px;
    font-weight: 900;
    text-align: center;
    padding: 18px;
    background: linear-gradient(90deg, #ff512f, #dd2476);
    -webkit-background-clip: text;
    color: transparent;
}

/* Prediction Result Box */
.pred-high {
    padding: 28px;
    border-radius: 16px;
    text-align: center;
    font-size: 32px;
    font-weight: 800;
    color: #ffffff;
    background: linear-gradient(135deg, #ff512f, #dd2476);
    box-shadow: 0 4px 18px rgba(255, 60, 60, 0.45);
    margin-top: 25px;
}

.pred-low {
    padding: 28px;
    border-radius: 16px;
    text-align: center;
    font-size: 32px;
    font-weight: 800;
    color: #004d25;
    background: linear-gradient(135deg, #b9f6ca, #69f0ae);
    box-shadow: 0 4px 18px rgba(0, 200, 120, 0.35);
    margin-top: 25px;
}

/* Section headings */
section-title {
    font-size: 22px;
    font-weight: 700;
}

/* Metric labels a little bolder */
.css-1xarl3l, .css-10trblm {
    font-weight: 600 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# INITIALISE GROQ CLIENT (SAFE)
# ---------------------------------------------------
groq_client = None
if GROQ_API_KEY and GROQ_API_KEY != "YOUR_GROQ_API_KEY_HERE":
    groq_client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------
# LOAD ML FILES
# ---------------------------------------------------
@st.cache_resource
def load_all():
    model = joblib.load("fire_model.pkl")
    scaler = joblib.load("scaler (2).pkl")          # file name as you saved it
    encoder = joblib.load("encoder.pkl")
    feature_cols = joblib.load("feature_columns_1.pkl")
    return model, scaler, encoder, feature_cols


model, scaler, encoder, feature_cols = load_all()

# ---------------------------------------------------
# HELPER FUNCTIONS: GEO & ENVIRONMENT
# ---------------------------------------------------
def geocode_forest(name: str):
    """
    Get latitude and longitude for a forest name using OpenCage API.
    """
    if not OPENCAGE_API_KEY or OPENCAGE_API_KEY == "YOUR_OPENCAGE_API_KEY_HERE":
        st.error("OpenCage API key is missing. Please add it in app.py.")
        return None, None

    url = f"https://api.opencagedata.com/geocode/v1/json?q={name}&key={OPENCAGE_API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return None, None

    if data.get("total_results", 0) == 0:
        return None, None

    lat = data["results"][0]["geometry"]["lat"]
    lon = data["results"][0]["geometry"]["lng"]
    return lat, lon


def generate_environment(lat: float, lon: float) -> pd.DataFrame:
    """
    Generate synthetic-but-consistent environmental data from coordinates.
    """
    temperature = 20 + abs(lat % 10)
    humidity = 40 + abs(lon % 20)
    wind_speed = 2 + (abs(lat + lon) % 5)
    precip = abs(lat - lon) % 3

    ndvi = np.clip(humidity / 100 - 0.3, 0, 1)
    fwi = wind_speed * (1 - humidity / 100) * 25
    drought_code = max(20, (temperature * 2) - precip)

    return pd.DataFrame(
        [
            {
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
                "population_density": 18,
            }
        ]
    )

# ---------------------------------------------------
# AI FUNCTIONS USING GROQ (LLAMA-3)
# ---------------------------------------------------
def groq_ai(prompt: str) -> str:
    """
    Call Groq LLaMA-3 to generate a short, focused answer.
    """
    if groq_client is None:
        return "AI unavailable (Groq API key not configured)."

    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI unavailable. ({e})"


def ai_forest_profile(forest: str) -> str:
    prompt = f"""
Provide a 5–6 line summary of the forest '{forest}' including:
- Location/region
- Typical climate
- Dominant vegetation
- Natural fire vulnerability
Use clear, non-technical language.
"""
    return groq_ai(prompt)


def ai_fire_explanation(df: pd.DataFrame, prediction: int, forest: str) -> str:
    info = df.to_dict(orient="records")[0]
    risk_level = "HIGH" if prediction == 1 else "LOW"

    prompt = f"""
Forest: {forest}
Predicted fire risk: {risk_level}

Environmental inputs:
{info}

Explain in 4–6 lines:
- Which variables influence this prediction the most
- Why these values indicate {risk_level} fire risk
- A concise expert-style explanation that a student can present.
"""
    return groq_ai(prompt)


def ai_recommendations(prediction: int) -> str:
    risk_level = "HIGH" if prediction == 1 else "LOW"
    prompt = f"""
Current fire risk level: {risk_level}.

List 5 practical wildfire safety recommendations as bullet points.
Keep them short and specific.
"""
    return groq_ai(prompt)

# ---------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------
st.sidebar.markdown(
    "<div class='sidebar-title'>🔥 Fire Prediction Suite</div>",
    unsafe_allow_html=True,
)

menu = st.sidebar.radio(
    "Navigation",
    [
        "Prediction Dashboard",
        "EDA Analytics",
        "Danger Calculator",
        "Dataset Explorer",
        "Project Report",
    ],
)

# ---------------------------------------------------
# PAGE 1: PREDICTION DASHBOARD
# ---------------------------------------------------
if menu == "Prediction Dashboard":
    st.markdown(
        "<div class='main-title'>AI-Based Forest Fire Risk Predictor</div>",
        unsafe_allow_html=True,
    )

    forest_name = st.text_input("Enter Forest Name", "Amazon")

    predict_btn = st.button("Predict Fire Risk", use_container_width=True)

    if predict_btn:
        # 1. Geocode
        lat, lon = geocode_forest(forest_name)
        if lat is None:
            st.error("Forest not found. Please try another name.")
        else:
            # 2. Generate environment data
            df = generate_environment(lat, lon)

            # 3. Encode landcover class
            try:
                df["landcover_class_encoded"] = encoder.transform(
                    df["landcover_class"]
                )
            except Exception:
                df["landcover_class_encoded"] = encoder.transform(
                    ["Deciduous Forest"]
                )

            df = df.drop(columns=["landcover_class"])

            # 4. Align columns & scale
            df = df.reindex(columns=feature_cols)
            df_scaled = scaler.transform(df)

            # 5. Predict
            pred = int(model.predict(df_scaled)[0])

            # 6. Show map
            st.subheader("Location on Map")
            st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))

            # 7. Key metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Temperature", f"{df['temperature_c'].iloc[0]:.2f} °C")
            c2.metric("Humidity", f"{df['humidity_pct'].iloc[0]:.2f} %")
            c3.metric("Wind Speed", f"{df['wind_speed_m_s'].iloc[0]:.2f} m/s")

            # 8. Prediction box
            if pred == 1:
                st.markdown(
                    "<div class='pred-high'>🔥 HIGH FIRE RISK DETECTED</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='pred-low'>🌿 NO FIRE RISK</div>",
                    unsafe_allow_html=True,
                )

            # 9. AI sections
            st.markdown("## 🌲 Forest Overview (AI)")
            st.markdown(ai_forest_profile(forest_name))

            st.markdown("## 🧠 AI-Generated Explanation")
            st.markdown(ai_fire_explanation(df, pred, forest_name))

            with st.expander("♡ Safety Recommendations (AI)"):
                st.markdown(ai_recommendations(pred))

            # 10. Show full input data (optional)
            st.markdown("## 📊 Environmental Data Used")
            st.json(df.to_dict(orient="records")[0])

# ---------------------------------------------------
# PAGE 2: EDA ANALYTICS
# ---------------------------------------------------
elif menu == "EDA Analytics":
    try:
        import fire_pages.eda_page as page

        page.run()
    except ImportError:
        st.error("EDA page module not found (fire_pages/eda_page.py).")

# ---------------------------------------------------
# PAGE 3: DANGER CALCULATOR
# ---------------------------------------------------
elif menu == "Danger Calculator":
    try:
        import fire_pages.danger_page as page

        page.run()
    except ImportError:
        st.error("Danger Calculator page module not found (fire_pages/danger_page.py).")

# ---------------------------------------------------
# PAGE 4: DATASET EXPLORER
# ---------------------------------------------------
elif menu == "Dataset Explorer":
    try:
        import fire_pages.dataset_page as page

        page.run()
    except ImportError:
        st.error("Dataset Explorer page module not found (fire_pages/dataset_page.py).")

# ---------------------------------------------------
# PAGE 5: PROJECT REPORT
# ---------------------------------------------------
elif menu == "Project Report":
    try:
        import fire_pages.report_page as page

        page.run()
    except ImportError:
        st.error("Project Report page module not found (fire_pages/report_page.py).")
