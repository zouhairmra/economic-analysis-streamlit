# 3_multivariate.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ardl import ardl_select_order

st.title("Multivariate Econometric Analysis")

uploaded_file = st.file_uploader("Upload a multivariate CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.write(df.head())

    # Handle datetime index if available
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("You need at least 2 numeric columns.")
    else:
        ycol = st.selectbox("Select dependent variable (Y)", numeric_cols)
        xcols = st.multiselect("Select independent variable(s) (X)", [col for col in numeric_cols if col != ycol])

        if ycol and xcols:
            # Subset and clean data
            data = df[[ycol] + xcols].dropna()

            st.write(f"Number of usable observations: {len(data)}")

            # --------------------------
            # OLS ESTIMATION
            # --------------------------
            st.subheader("OLS Regression")
            X = sm.add_constant(data[xcols])
            ols_model = sm.OLS(data[ycol], X).fit()
            st.text(ols_model.summary())

            # --------------------------
            # ARDL ESTIMATION
            # --------------------------
            st.subheader("ARDL Estimation")

            try:
                # Automatically select best lags using AIC
                selected_ardl = ardl_select_order(
                    endog=data[ycol],
                    exog=data[xcols],
                    maxlag=4,         # Max lag for dependent variable
                    maxorder=4,       #
