import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load global objects from main app
def run():
    st.title("🔥 Custom Danger Calculator")

    st.markdown("Enter custom environmental values and get fire-risk prediction.")

    # Load required files
    model = joblib.load("fire_model.pkl")
    scaler = joblib.load("scaler (2).pkl")
    encoder_dict = joblib.load("encoders.pkl")
    feature_cols = joblib.load("feature_columns.pkl")

    encoder_cols = list(encoder_dict.keys())

    # ===========================
    # USER INPUT
    # ===========================
    lat = st.number_input("Latitude", -90.0, 90.0, 20.0)
    lon = st.number_input("Longitude", -180.0, 180.0, 70.0)
    temp = st.number_input("Temperature (°C)", 0.0, 60.0, 30.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)
    wind = st.number_input("Wind Speed (m/s)", 0.0, 40.0, 5.0)
    precip = st.number_input("Precipitation (mm)", 0.0, 500.0, 2.0)
    ndvi = st.number_input("NDVI (0-1)", 0.0, 1.0, 0.4)
    fwi = st.number_input("FWI Score", 0.0, 100.0, 20.0)
    drought = st.number_input("Drought Code", 0.0, 800.0, 200.0)
    forest_cover = st.number_input("Forest Cover %", 0, 100, 70)
    elevation = st.number_input("Elevation (m)", 0, 5000, 300)
    slope = st.number_input("Slope (°)", 0, 90, 10)
    pop = st.number_input("Population Density", 0, 5000, 20)

    landcover = st.selectbox(
        "Landcover Class",
        ["Deciduous Forest", "Grassland", "Savanna", "Cropland", "Rainforest"]
    )

    # ===========================
    # PREDICT BUTTON
    # ===========================
    if st.button("Calculate Risk"):

        df = pd.DataFrame([{
            "latitude": lat,
            "longitude": lon,
            "temperature_c": temp,
            "humidity_pct": humidity,
            "wind_speed_m_s": wind,
            "precip_mm": precip,
            "ndvi": ndvi,
            "fwi_score": fwi,
            "drought_code": drought,
            "forest_cover_pct": forest_cover,
            "landcover_class": landcover,
            "elevation_m": elevation,
            "slope_deg": slope,
            "population_density": pop
        }])

        # ============ FIXED ONE-HOT (same method as app.py) ============
        df_oh = pd.get_dummies(df["landcover_class"], prefix="landcover_class")

        for col in encoder_cols:
            if col not in df_oh.columns:
                df_oh[col] = 0

        df = pd.concat([df.drop(columns=["landcover_class"]), df_oh[encoder_cols]], axis=1)

        # Reorder columns
        df = df.reindex(columns=feature_cols)

        # Predict
        pred = int(model.predict(df)[0])

        if pred == 1:
            st.error("🔥 HIGH FIRE RISK")
        else:
            st.success("🌿 LOW FIRE RISK")
