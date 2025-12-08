import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run():
    st.title("📊 EDA Analytics")

    try:
        df = pd.read_csv("fire_dataset.csv")
    except:
        st.error("❌ Could not load fire_dataset.csv. Upload it to root directory.")
        return

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe())

    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    if numeric_df.empty:
        st.warning("⚠ No numeric columns available for Correlation Heatmap.")
    else:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False, ax=ax)
        st.pyplot(fig)

    st.subheader("Feature Distributions")
    for col in numeric_df.columns[:5]:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)
