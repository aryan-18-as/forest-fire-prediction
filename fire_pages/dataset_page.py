import streamlit as st
import pandas as pd

def run():

    st.title("🗂 Dataset Explorer")

    st.markdown("Use this tool to explore, filter, and analyze the input dataset.")

    try:
        df = pd.read_csv("fire_dataset.csv")
    except:
        st.error("Dataset 'fire_dataset.csv' not found. Upload it in project root.")
        return

    st.subheader("📄 Full Dataset")
    st.dataframe(df)

    st.subheader("🔍 Search / Filter")
    col = st.selectbox("Select Column", df.columns)
    val = st.text_input("Enter filter value")

    if val:
        filtered = df[df[col].astype(str).str.contains(val, case=False)]
        st.write(f"Results Found: {filtered.shape[0]}")
        st.dataframe(filtered)
