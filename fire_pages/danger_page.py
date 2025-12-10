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
   
