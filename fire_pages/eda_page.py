import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run():
    st.title("📊 Exploratory Data Analysis")

    try:
        df = pd.read_csv("fire_dataset.csv")
    except:
        st.error("Dataset file 'fire_dataset.csv' not found.")
        return

    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe())

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Distribution Plots")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df["temperature_c"], kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.histplot(df["humidity_pct"], kde=True, ax=ax)
        st.pyplot(fig)
