import streamlit as st
import pandas as pd
import plotly.express as px

st.header("Exploratory Analysis")
uploaded = st.file_uploader("Upload CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.write(df.describe())
    st.plotly_chart(px.imshow(df.corr(), text_auto=True))
