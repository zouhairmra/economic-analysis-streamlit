import streamlit as st
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.ardl import ARDL

st.set_page_config(page_title="Multivariate Analysis", layout="wide")

st.title("📊 Multivariate Econometric Analysis")

uploaded_file = st.file_uploader("Upload your multivariate dataset (CSV format):", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.write("### Data Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    ycol = st.selectbox("Select dependent variable (Y):", options=numeric_cols)
    xcols = st.multiselect("Select independent variable(s) (X):", options=[col for col in numeric_cols if col != ycol])

    if ycol and xcols:
        X = df[xcols]
        y = df[ycol]

        st.subheader("1. OLS Regression Results")
        X_const = sm.add_constant(X)
        ols_model = sm.OLS(y, X_const).fit()
        st.text(ols_model.summary())

        st.subheader("2. Correlation Matrix")
        corr = df[[ycol] + xcols].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader("3. ARDL Model (AutoRegressive Distributed Lag)")
        try:
            ardl_formula = f"{ycol} ~ " + " + ".join([f"L1.{col}" for col in xcols])
            ardl_model = ARDL.from_formula(ardl_formula, data=df, lags=2)
            ardl_res = ardl_model.fit()
            st.text(ardl_res.summary())
        except Exception as e:
            st.warning(f"ARDL estimation failed: {e}")
else:
    st.info("Please upload a dataset to begin.")
