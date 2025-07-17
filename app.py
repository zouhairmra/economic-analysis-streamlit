
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("📊 Economic Analysis with AI")

uploaded_file = st.file_uploader("📂 Upload your economic data file (CSV)", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("🧾 Data preview:")
    st.dataframe(data.head())

    col_x = st.selectbox("📈 Select explanatory variable (X)", data.columns)
    col_y = st.selectbox("📉 Select dependent variable (Y)", data.columns)

    if st.button("🔍 Run analysis"):
        X = data[[col_x]].dropna()
        Y = data[col_y].loc[X.index]

        model = LinearRegression()
        model.fit(X, Y)

        st.write(f"**Regression Coefficient**: {model.coef_[0]:.4f}")
        st.write(f"**Intercept**: {model.intercept_:.4f}")
        st.write(f"**R² Score**: {model.score(X, Y):.4f}")

        fig, ax = plt.subplots()
        ax.scatter(X, Y)
        ax.plot(X, model.predict(X), color="red")
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        st.pyplot(fig)
