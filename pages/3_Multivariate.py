import streamlit as st
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.ardl import ARDL
import itertools

st.title("Multivariate Econometric Analysis")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.dropna(inplace=True)

    st.subheader("Data Preview")
    st.write(df.head())

    columns = df.columns.tolist()

    ycol = st.selectbox("Select dependent variable", columns)
    xcols = st.multiselect("Select independent variables", [col for col in columns if col != ycol])

    if ycol and xcols:
        st.markdown("### Stationarity Tests (ADF)")
        adf_results = {}
        for col in [ycol] + xcols:
            adf_stat, pval, _, _, _, _ = adfuller(df[col])
            adf_results[col] = {"ADF Statistic": adf_stat, "p-value": pval}
        st.write(pd.DataFrame(adf_results).T)

        st.markdown("### OLS Regression")
        X = df[xcols]
        X = sm.add_constant(X)
        y = df[ycol]
        ols_model = sm.OLS(y, X).fit()
        st.text(ols_model.summary())

        st.markdown("### Variance Inflation Factor (VIF)")
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        st.write(vif_data)

        st.markdown("### Durbin-Watson Test")
        dw_stat = durbin_watson(ols_model.resid)
        st.write(f"Durbin-Watson statistic: {dw_stat:.4f}")

        st.markdown("### Breusch-Pagan Test")
        bp_test = het_breuschpagan(ols_model.resid, X)
        labels = ["LM Stat", "LM p-value", "F Stat", "F p-value"]
        st.write(dict(zip(labels, bp_test)))

        st.markdown("### Cointegration Test (Engle-Granger)")
        if len(xcols) == 1:
            coint_stat, pval, _ = coint(df[ycol], df[xcols[0]])
            st.write({"Test Stat": coint_stat, "p-value": pval})
        else:
            st.info("Engle-Granger test supports only one regressor.")

        st.markdown("### ARDL Estimation")
        max_lag_y = st.number_input("Max lags for dependent variable", min_value=1, max_value=10, value=1)
        max_lags_x = {x: st.number_input(f"Max lags for {x}", min_value=0, max_value=10, value=1) for x in xcols}

        try:
            ardl_model = ARDL(endog=df[ycol], lags=max_lag_y, exog=df[xcols], exog_lags=max_lags_x).fit()
            st.text(ardl_model.summary())

            if st.button("Export Results"):
                with open("ARDL_summary.txt", "w") as f:
                    f.write(str(ardl_model.summary()))
                st.success("ARDL summary saved to ARDL_summary.txt")
        except Exception as e:
            st.error(f"ARDL estimation failed: {e}")
