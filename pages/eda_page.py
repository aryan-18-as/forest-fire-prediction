import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run():

    st.title("📊 Exploratory Data Analysis Dashboard")

    st.markdown("""
    This module provides insights into the forest fire dataset using statistical summaries,
    correlation heatmaps, and distribution plots.
    """)

    # Load dataset
    try:
        df = pd.read_csv("fire_dataset.csv")
    except:
        st.error("Dataset 'fire_dataset.csv' not found in project root.")
        return

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # ---------------------
    # STATS SUMMARY
    # ---------------------
    st.subheader("📊 Statistical Summary")
    st.dataframe(df.describe())

    # ---------------------
    # CORRELATION HEATMAP
    # ---------------------
    st.subheader("🔥 Correlation Heatmap")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for heatmap.")

    # ---------------------
    # DISTRIBUTIONS
    # ---------------------
    st.subheader("📈 Feature Distributions")

    for col in numeric_cols[:6]:  # Limit to 6 graphs
        st.write(f"🔹 {col}")
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

