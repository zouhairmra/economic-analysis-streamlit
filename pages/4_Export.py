import streamlit as st
import pandas as pd

st.header("Export Results")
uploaded = st.file_uploader("Upload CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.download_button("Download data as CSV", df.to_csv(index=False), "data.csv")
