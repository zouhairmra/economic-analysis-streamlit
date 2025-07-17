import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import statsmodels.api as sm

# Title
st.title("📈 Forecasting & Econometric Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Convert date column if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')

    df = df.dropna()  # drop rows with NaNs

    # Let user choose target variable
    ycol = st.selectbox("Select target variable (Y)", df.columns)

    # Run forecasting if 'year' and ycol are selected
    if 'year' in df.columns and ycol in df.columns:
        X = df[['year']]
        y = df[ycol]

        # sklearn Linear Regression
        model = LinearRegression()
        model.fit(X, y)
        df['prediction'] = model.predict(X)

        # Evaluation Metrics
        rmse = np.sqrt(mean_squared_error(y, df['prediction']))
        mae = mean_absolute_error(y, df['prediction'])
        r2 = r2_score(y, df['prediction'])

        st.subheader("📊 Descriptive Statistics")
        st.write(df[[ycol, 'prediction']].describe())

        st.subheader("📉 Evaluation Metrics")
        st.markdown(f"- RMSE: `{rmse:.3f}`")
        st.markdown(f"- MAE: `{mae:.3f}`")
        st.markdown(f"- R²: `{r2:.3f}`")

        st.subheader("📈 Forecast vs Actual")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['year'], y=df[ycol], mode='lines+markers', name='Actual'))
        fig.add_trace(go.Scatter(x=df['year'], y=df['prediction'], mode='lines+markers', name='Prediction'))
        st.plotly_chart(fig)

        # Optional: OLS Regression using statsmodels
        st.subheader("📘 OLS Regression Summary")
        X_const = sm.add_constant(X)
        ols_model = sm.OLS(y, X_const).fit()
        st.text(ols_model.summary())
    else:
        st.warning("Ensure 'year' column and selected target are valid numeric columns.")
else:
    st.info("Please upload a CSV file to begin.")
