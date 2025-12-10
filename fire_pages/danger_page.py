import streamlit as st
import pandas as pd
import numpy as np
import joblib

def run():

    st.title("🌡 Danger Score Calculator")

    st.markdown("""
    This tool allows you to calculate a custom danger score based on temperature, humidity,
    wind speed, and drought conditions.
    """)

    col1, col2 = st.columns(2)

    with col1:
        temp = st.slider("Temperature (°C)", 0, 50, 25)
        humidity = st.slider("Humidity (%)", 0, 100, 50)

    with col2:
        wind = st.slider("Wind Speed (m/s)", 0, 30, 5)
        drought = st.slider("Drought Code", 0, 500, 100)

    # Formula
    score = (temp * 0.4) + (wind * 0.3) + (drought * 0.2) - (humidity * 0.1)
    score = round(score, 2)

    st.subheader("🔥 Danger Score")
    st.metric(label="", value=score)

    if score > 80:
        st.error("Severe Risk")
    elif score > 50:
        st.warning("Moderate Risk")
    else:
        st.success("Low Risk")
    st.title("🌡 Fire Danger Calculator")

    model = joblib.load("fire_model.pkl")
    scaler = joblib.load("scaler (2).pkl")
    encoder = joblib.load("encoder.pkl")
    feature_cols = joblib.load("feature_columns_1.pkl")

    st.subheader("Enter Environmental Conditions")

    temp = st.number_input("Temperature (°C)", 0, 60, 25)
    humidity = st.number_input("Humidity (%)", 0, 100, 40)
    wind = st.number_input("Wind Speed (m/s)", 0, 40, 5)
    precip = st.number_input("Precipitation (mm)", 0, 500, 2)

    if st.button("Calculate Fire Risk"):
        ndvi = np.clip(humidity/100 - 0.3, 0, 1)
        fwi = wind * (1 - humidity/100) * 25
        drought = max(20, (temp*2)-precip)

        df = pd.DataFrame([{
            "latitude": 10,
            "longitude": 10,
            "temperature_c": temp,
            "precip_mm": precip,
            "humidity_pct": humidity,
            "wind_speed_m_s": wind,
            "fwi_score": fwi,
            "drought_code": drought,
            "ndvi": ndvi,
            "forest_cover_pct": 70,
            "landcover_class": "Deciduous Forest",
            "elevation_m": 300,
            "slope_deg": 12,
            "population_density": 18
        }])

        df["landcover_class_encoded"] = encoder.transform(df["landcover_class"])
        df = df.drop(columns=["landcover_class"])
        df = df.reindex(columns=feature_cols)

        pred = int(model.predict(scaler.transform(df))[0])

        if pred == 1:
            st.error("🔥 HIGH FIRE RISK")
        else:
            st.success("🌿 LOW FIRE RISK")
