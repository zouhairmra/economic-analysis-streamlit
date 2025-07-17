import streamlit as st
import pandas as pd

st.title("📁 Upload Your Data")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ File successfully uploaded!")
    st.write("### Data Preview")
    st.dataframe(df)
    st.session_state["data"] = df
else:
    st.info("Upload a dataset to get started.")
