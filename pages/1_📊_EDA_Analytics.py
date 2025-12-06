import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="EDA Analytics", page_icon="📊", layout="wide")

st.title("📊 EDA & Analytics – Forest Fire Dataset")

# Path to your dataset
DATA_PATH = "fire_dataset.csv"   # yahan apne CSV ka naam rakhna

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    df = load_data()
except Exception as e:
    st.error(f"⚠ Dataset load nahi hua. File `{DATA_PATH}` repo me daalo.\n\nError: {e}")
    st.stop()

st.subheader("🔎 Quick Glance")
st.write(df.head())

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Rows", df.shape[0])
with col2:
    st.metric("Total Columns", df.shape[1])

st.subheader("📦 Column Summary")
st.write(df.describe(include="all").T)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if numeric_cols:
    st.subheader("📈 Correlation Heatmap (numeric columns)")
    corr = df[numeric_cols].corr()
    st.dataframe(corr.style.background_gradient(cmap="RdYlGn"), use_container_width=True)

    st.subheader("📊 Distribution Explorer")
    col = st.selectbox("Select numeric feature", numeric_cols)
    st.bar_chart(df[col])
else:
    st.info("No numeric columns found in dataset.")
