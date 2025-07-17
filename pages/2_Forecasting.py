import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# Title
st.title("📈 Forecasting Analysis")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    # Read CSV
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File successfully loaded!")

        # Display the first few rows
        st.subheader("📄 Data Preview")
        st.write(df.head())

        # Convert any date columns
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'year' in col.lower()]
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass

        # Drop rows with NA after conversion
        df.dropna(inplace=True)

        # Choose target column
        st.subheader("🎯 Select Forecasting Target")
        ycol = st.selectbox("Choose the column to forecast", df.columns)

        # Choose feature (independent variable)
        xcol = st.selectbox("Choose the time column", [col for col in df.columns if np.issubdtype(df[col].dtype, np.datetime64) or np.issubdtype(df[col].dtype, np.number)])

        # Ensure numeric time input
        X = df[[xcol]]
        if np.issubdtype(X[xcol].dtype, np.datetime64):
            X[xcol] = X[xcol].map(pd.Timestamp.toordinal)

        y = df[ycol]

        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Plot
        st.subheader("📊 Forecast Plot")
        fig, ax = plt.subplots()
        ax.plot(X[xcol], y, label="Actual", marker='o')
        ax.plot(X[xcol], y_pred, label="Predicted", linestyle='--')
        ax.legend()
        st.pyplot(fig)

        # Metrics
        st.subheader("📉 Model Performance")
        st.write(f"R² Score: {r2_score(y, y_pred):.4f}")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")

        # Econometric analysis with statsmodels
        st.subheader("📘 Econometric Summary")
        X_const = sm.add_constant(X)
        ols_model = sm.OLS(y, X_const).fit()
        st.text(ols_model.summary())

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("Please upload a CSV file to begin.")
