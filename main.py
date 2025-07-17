import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm

st.set_page_config(page_title="DataStatPro Clone", layout="wide")
st.markdown("""
<style>
/* Make the app look cleaner and more modern */
body {
    background-color: #f7f9fc;
}
.sidebar .sidebar-content {
    background-color: #004466;
    color: white;
}
.sidebar .sidebar-content .block-container {
    padding: 1rem;
}
.css-1d391kg {
    background-color: #004466;
    color: white;
}
.stButton>button {
    background-color: #006699;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
}
.stButton>button:hover {
    background-color: #005580;
}
</style>
""", unsafe_allow_html=True)

# Simulated top navigation bar
st.markdown("""
<style>
.navbar {
    background-color: #004466;
    padding: 10px;
    color: white;
    font-size: 24px;
    font-weight: bold;
}
.navbar a {
    color: white;
    margin-right: 15px;
    text-decoration: none;
}
</style>
<div class="navbar">
    DataStatPro Clone
    <a href='#home'>Home</a>
    <a href='#upload'>Upload</a>
    <a href='#analysis'>Analysis</a>
    <a href='#export'>Export</a>
</div>
""", unsafe_allow_html=True)

st.write("")

# Sidebar menu for navigation
menu = st.sidebar.selectbox("Menu", ["Home", "Upload Data", "Analysis", "Export Results"])

if "df" not in st.session_state:
    st.session_state.df = None

if menu == "Home":
    st.title("Welcome to DataStatPro Clone")
    st.write("""
        This app lets you upload your data and perform simple economic/financial analyses.
        Use the sidebar to navigate between pages.
    """)

elif menu == "Upload Data":
    st.title("Upload your data (CSV or Excel)")

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.success("File uploaded successfully!")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading file: {e}")

elif menu == "Analysis":
    st.title("Data Analysis")

    if st.session_state.df is None:
        st.warning("Please upload data first!")
    else:
        df = st.session_state.df
        st.write("Data preview:")
        st.dataframe(df.head())

        analysis = st.selectbox("Choose analysis", [
    "Descriptive Statistics",
    "Correlation Matrix",
    "Linear Regression",
    "Multiple Linear Regression",
    "Histogram",
    "Scatter Plot",
    "Time Series Line Chart"
])

        if analysis == "Descriptive Statistics":
    st.write("Summary statistics of numeric variables:")
    st.dataframe(df.describe())

         elif analysis == "Histogram":
    col = st.selectbox("Choose column for histogram", df.select_dtypes(include="number").columns)
    fig = px.histogram(df, x=col)
    st.plotly_chart(fig, use_container_width=True)
elif analysis == "Multiple Linear Regression":
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    y_var = st.selectbox("Dependent variable (Y)", numeric_cols, key="mlr_y")
    x_vars = st.multiselect("Independent variables (X)", [c for c in numeric_cols if c != y_var], key="mlr_x")
    if st.button("Run MLR"):
        if x_vars:
            X = df[x_vars]
            X = sm.add_constant(X)
            y = df[y_var]
            model = sm.OLS(y, X).fit()
            st.write(model.summary())
        else:
            st.warning("Please select at least one independent variable.")

elif analysis == "Time Series Line Chart":
    time_col = st.selectbox("Time column", df.columns)
    value_col = st.selectbox("Value column", df.select_dtypes(include="number").columns)
    try:
        df[time_col] = pd.to_datetime(df[time_col])
        df_sorted = df.sort_values(by=time_col)
        fig = px.line(df_sorted, x=time_col, y=value_col)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Time series plot failed: {e}")

  if analysis == "Correlation Matrix":
            corr = df.corr()
            st.write("Correlation matrix:")
            st.dataframe(corr)
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', origin='lower')
            st.plotly_chart(fig, use_container_width=True)

        elif analysis == "Linear Regression":
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            y_var = st.selectbox("Dependent variable (Y)", numeric_cols)
            x_var = st.selectbox("Independent variable (X)", [c for c in numeric_cols if c != y_var])
            if st.button("Run Regression"):
                X = df[[x_var]]
                X = sm.add_constant(X)
                y = df[y_var]
                model = sm.OLS(y, X).fit()
                st.write(model.summary())
                fig = px.scatter(df, x=x_var, y=y_var, trendline="ols")
                st.plotly_chart(fig, use_container_width=True)

        elif analysis == "Scatter Plot":
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            x_var = st.selectbox("X axis", numeric_cols, key="scatter_x")
            y_var = st.selectbox("Y axis", numeric_cols, key="scatter_y")
            fig = px.scatter(df, x=x_var, y=y_var)
            st.plotly_chart(fig, use_container_width=True)

elif menu == "Export Results":
    st.title("Export your uploaded data")
    if st.session_state.df is None:
        st.warning("Upload data first!")
    else:
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="dataset.csv", mime="text/csv")
