import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go

# ----------------------------------------------------
# Streamlit page config
# ----------------------------------------------------
st.set_page_config(
    page_title="F-PAS | Fire Prediction & Alert System",
    page_icon="🔥",
    layout="wide",
)

st.title("🔥 F-PAS – Fire Prediction & Alert System")
st.caption("Global fire-risk estimation using live weather data (OpenWeather API).")

st.markdown("""
Ye system kisi bhi location ke **live weather** ke basis par 
ek **Fire Risk Score (0–100)** calculate karta hai aur 
Low / Moderate / High fire danger batata hai.
""")

# ----------------------------------------------------
# API KEY HANDLING
# ----------------------------------------------------
# Pehle Streamlit secrets me se key lene ki koshish:
API_KEY = st.secrets.get("OPENWEATHER_API_KEY", None)

# Agar local run kar rahe ho aur secrets use nahi kar rahe:
if not API_KEY:
    st.warning("OPENWEATHER_API_KEY secret nahi mila. Neeche manually daal sakte ho (sirf local testing ke liye).")
    API_KEY = st.text_input("OpenWeather API Key (for local use)", type="password")

if not API_KEY:
    st.stop()

# ----------------------------------------------------
# Helper: OpenWeather API calls
# ----------------------------------------------------
def geocode_city(city: str, api_key: str):
    """City name se lat/lon nikalna (OpenWeather geocoding)."""
    url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {"q": city, "limit": 1, "appid": api_key}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    return data[0]["lat"], data[0]["lon"]


def fetch_weather(lat: float, lon: float, api_key: str):
    """Given lat/lon, current weather fetch karo (metric units)."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

# ----------------------------------------------------
# Fire-risk score (FWI-style heuristic, 0–100)
# ----------------------------------------------------
def compute_fire_risk(temp_c, rh, wind_ms, rain_mm):
    """
    Simple FWI-style index (0–100):

    - Zyada temperature  → zyada risk
    - Kam humidity       → zyada risk
    - Zyada wind         → zyada risk
    - Zyada recent rain  → risk kam
    """
    # Normalize features to 0–1
    temp_score = np.clip((temp_c - 0) / 35, 0, 1)          # 0–35°C
    rh_score   = np.clip((100 - rh) / 100, 0, 1)           # low RH = high score
    wind_score = np.clip(wind_ms / 20, 0, 1)               # 0–20 m/s
    rain_score = np.clip(rain_mm / 20, 0, 1)               # 0–20 mm

    index = (
        0.4 * temp_score +
        0.3 * rh_score +
        0.2 * wind_score -
        0.2 * rain_score
    )

    index = float(np.clip(index, 0, 1))
    return index * 100  # 0–100


def risk_level(score):
    if score < 30:
        return "Low", "🌿", "green"
    elif score < 60:
        return "Moderate", "🟡", "orange"
    else:
        return "High", "🔥", "red"


# ----------------------------------------------------
# Sidebar – Location input
# ----------------------------------------------------
st.sidebar.header("📍 Location Selection")

tab1, tab2 = st.sidebar.tabs(["City name", "Lat/Lon"])

with tab1:
    city_input = st.text_input("City, Country (e.g. Delhi, IN)", "Delhi, IN")

with tab2:
    lat = st.number_input("Latitude", value=28.6139, format="%.4f")
    lon = st.number_input("Longitude", value=77.2090, format="%.4f")

mode = st.sidebar.radio("Location mode", ["City search", "Manual lat/lon"])

if mode == "City search":
    location_label = city_input
else:
    location_label = f"{lat:.4f}, {lon:.4f}"

st.markdown(f"### 📍 Selected Location: `{location_label}`")

# ----------------------------------------------------
# Fetch weather on button click
# ----------------------------------------------------
if st.button("🔄 Fetch Live Weather & Predict Fire Risk", use_container_width=True):
    try:
        # 1) Get coordinates
        if mode == "City search":
            geo = geocode_city(city_input, API_KEY)
            if geo is None:
                st.error("City not found. Spelling check karo ya different location try karo.")
                st.stop()
            lat, lon = geo

        # 2) Fetch weather
        weather = fetch_weather(lat, lon, API_KEY)

        temp_c = weather["main"]["temp"]
        rh     = weather["main"]["humidity"]
        wind   = weather.get("wind", {}).get("speed", 0.0)

        rain_data = weather.get("rain", {})
        rain_1h   = rain_data.get("1h", 0.0)
        rain_3h   = rain_data.get("3h", 0.0)
        rain_mm   = max(rain_1h, rain_3h)  # rough approx

        # 3) Compute fire risk
        score = compute_fire_risk(temp_c, rh, wind, rain_mm)
        level, emoji, color = risk_level(score)

        # ------------------------------------------------
        # Layout: metrics + gauge
        # ------------------------------------------------
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🌡 Temp (°C)", f"{temp_c:.1f}")
        col2.metric("💧 Humidity (%)", f"{rh:.0f}")
        col3.metric("💨 Wind (m/s)", f"{wind:.1f}")
        col4.metric("🌧 Rain (mm)", f"{rain_mm:.1f}")

        st.markdown("---")

        gcol1, gcol2 = st.columns([2, 1])

        with gcol1:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=score,
                    title={'text': "Fire Risk Score (0–100)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "orange"},
                        'steps': [
                            {'range': [0, 30], 'color': "#3CB371"},
                            {'range': [30, 60], 'color': "#FFD700"},
                            {'range': [60, 100], 'color': "#FF4500"},
                        ],
                    },
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        with gcol2:
            st.subheader("Overall Assessment")
            st.markdown(
                f"### {emoji} **{level} Fire Danger**\n"
                f"Score: **{score:.1f} / 100**"
            )

            if level == "High":
                st.error(
                    "⚠ Conditions are critical. Dry fuel + low humidity + wind "
                    "milke **high probability of fire ignition and spread** banate hain."
                )
            elif level == "Moderate":
                st.warning(
                    "⚠ Fire possible hai dry areas me. Open flames, littered glass, "
                    "aur careless activities avoid karo."
                )
            else:
                st.success(
                    "✅ Fire risk currently low hai, lekin local forest guidelines "
                    "aur regulations follow karna zaroori hai."
                )

        st.markdown("---")
        st.caption("Note: Ye ek simplified FWI-style heuristic hai, jo sirf weather ke basis pe risk estimate karta hai.")

    except requests.RequestException as e:
        st.error(f"🌐 Network/API error: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
