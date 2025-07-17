import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("📈 Forecasting Economic Indicators")

# Upload the dataset
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    try:
        # Convert and extract year
        df['year'] = pd.to_datetime(df['year'], errors='coerce')
        df = df.dropna(subset=['year'])
        df['year'] = df['year'].dt.year

        # Choose target variable
        ycol = st.selectbox("Choose a variable to forecast", df.select_dtypes(include='number').columns.drop("year"))
        
        if ycol:
            model = LinearRegression()
            model.fit(df[['year']], df[ycol])
            df['prediction'] = model.predict(df[['year']])

            # Plot actual vs predicted
            fig, ax = plt.subplots()
            ax.plot(df['year'], df[ycol], label="Actual")
            ax.plot(df['year'], df['prediction'], label="Prediction")
            ax.legend()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error processing the file: {e}")
