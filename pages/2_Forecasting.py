import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
df['year'] = pd.to_numeric(df['year'], errors='coerce')
st.header("Forecasting")
uploaded = st.file_uploader("Upload CSV with year + target", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    if 'year' not in df.columns or df.shape[1] < 2:
        st.error("Need a 'year' column + one numeric column.")
    else:
        ycol = st.selectbox("Choose target:", df.columns.drop('year'))
        model = LinearRegression()
        model.fit(df[['year']], df[ycol])
        df['predicted'] = model.predict(df[['year']])
        fig = px.line(df, x='year', y=[ycol, 'predicted'])
        st.plotly_chart(fig)
        st.write(df[['year', ycol, 'predicted']])
