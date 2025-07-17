import streamlit as st
import pandas as pd
import plotly.express as px

st.header("Exploratory Analysis")
uploaded = st.file_uploader("Upload CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.write(df.describe())
    numeric_df = df.select_dtypes(include='number')
st.plotly_chart(px.imshow(numeric_df.corr(), text_auto=True))
