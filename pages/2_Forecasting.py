import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score

st.title("📈 Forecasting and Econometric Analysis")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv_
