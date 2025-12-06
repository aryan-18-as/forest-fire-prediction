import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dataset Explorer", page_icon="🗂", layout="wide")

st.title("🗂 Dataset Explorer")

DATA_PATH = "fire_dataset.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    df = load_data()
except Exception as e:
    st.error(f"⚠ Dataset `{DATA_PATH}` nahi mila. Error: {e}")
    st.stop()

st.subheader("🔍 Browse Data")
st.dataframe(df, use_container_width=True)

st.subheader("🔎 Column Filter")
col = st.selectbox("Select column", df.columns)
st.write(df[col].value_counts())
