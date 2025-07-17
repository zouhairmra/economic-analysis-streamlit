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

    # Try to find a column containing year or date
    if 'year' not in df.columns:
        st.error("Column 'year' is required in the dataset.")
    else:
        # Ensure both 'year' and target column are numeric
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df[ycol] = pd.to_numeric(df[ycol], errors='coerce')

        # Drop rows with missing or invalid values
        df_clean = df.dropna(subset=['year', ycol])
        if df_clean.empty:
            st.error("No valid data available after cleaning. Please check your file.")
        else:
            # Fit model
            model = LinearRegression()
            model.fit(df_clean[['year']], df_clean[ycol])

            # Forecast next 5 years
            future_years = pd.DataFrame({'year': np.arange(df_clean['year'].max() + 1, df_clean['year'].max() + 6)})
            predictions = model.predict(future_years)

            forecast_df = future_years.copy()
            forecast_df[ycol] = predictions

            # Plot
            fig = px.line(df_clean, x='year', y=ycol, title="Actual vs Forecast")
            fig.add_scatter(x=forecast_df['year'], y=forecast_df[ycol], mode='lines+markers', name='Forecast')
            st.plotly_chart(fig)

            # Show forecast table
            st.subheader("Forecasted Values")
            st.dataframe(forecast_df)
