import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px

st.title("📈 Forecasting")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Let user select the column to forecast
    ycol = st.selectbox("Select the variable to forecast", df.columns)

    # Convert year to numeric (if not already)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df[ycol] = pd.to_numeric(df[ycol], errors='coerce')

    # Drop rows with missing values
    df = df.dropna(subset=['year', ycol])

    # Fit model
    model = LinearRegression()
    model.fit(df[['year']], df[ycol])

    # Predict
    future_years = pd.DataFrame({'year': np.arange(df['year'].max() + 1, df['year'].max() + 6)})
    predictions = model.predict(future_years)

    forecast_df = future_years.copy()
    forecast_df[ycol] = predictions

    # Display plot
    fig = px.line(df, x='year', y=ycol, title="Actual vs Forecast")
    fig.add_scatter(x=forecast_df['year'], y=forecast_df[ycol], mode='lines+markers', name='Forecast')
    st.plotly_chart(fig)

    # Show forecast table
    st.subheader("Forecasted Values")
    st.dataframe(forecast_df)
