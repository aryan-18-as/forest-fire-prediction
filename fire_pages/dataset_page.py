import streamlit as st
import pandas as pd

def run():
    st.title("🗂 Dataset Explorer")

    try:
        df = pd.read_csv("fire_dataset.csv")
    except:
        st.error("Dataset 'fire_dataset.csv' not found.")
        return

    st.subheader("Preview Dataset")
    st.dataframe(df)

    st.subheader("Filter Columns")
    cols = st.multiselect("Select columns to view", df.columns.tolist(), default=df.columns.tolist())
    st.dataframe(df[cols])

    st.subheader("Download Filtered Data")
    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        "filtered_dataset.csv",
        "text/csv"
    )
