import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.header("Multivariate Regression")
uploaded = st.file_uploader("Upload CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    targets = st.multiselect("Select explanatory variables (X)", df.columns)
    target = st.selectbox("Select target (Y)", df.columns)
    if st.button("Run regression"):
        X = df[targets]
        y = df[target]
        model = LinearRegression().fit(X, y)
        st.write("Coefficients:", dict(zip(targets, model.coef_)))
        st.write("Intercept:", model.intercept_)
        st.write("R²:", model.score(X, y))
