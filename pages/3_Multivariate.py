# 3_multivariate.py

import streamlit as st
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ardl import ARDL
import matplotlib.pyplot as plt

st.title("Multivariate Econometric Analysis")

uploaded_file = st.file_uploader("Upload a multivariate CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.write(df.head())

    # Handle date column if exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric variables for multivariate analysis.")
    else:
        ycol = st.selectbox("Select dependent variable (Y)", numeric_cols)
        xcols = st.multiselect("Select independent variables (X)", [col for col in numeric_cols if col != ycol])

        if ycol and xcols:
            # Drop NA
            data = df[[ycol] + xcols].dropna()
            st.write(f"Using {len(data)} observations.")

            # Add constant
            data_with_const = sm.add_constant(data[xcols])
            model = sm.OLS(data[ycol], data_with_const).fit()

            st.subheader("OLS Regression Results")
            st.text(model.summary())

            # --- ARDL Model ---
            st.subheader("ARDL Estimation")
            try:
                y = data[ycol]
                X = data[xcols]

                # Automatically choose optimal lags (max p=4, q=2 for simplicity)
                ardl_model = ARDL(y, X, lags=4)
                ardl_res = ardl_model.fit()

                st.text(ardl_res.summary())

                # Plot ARDL residuals
                fig, ax = plt.subplots()
                ax.plot(ardl_res.resid)
                ax.set_title("ARDL Residuals")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"ARDL estimation failed: {e}")
