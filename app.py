
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="NARZY | Financial Risk Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("NARZY")
st.subheader("AI-Powered Financial Risk and Fraud Analytics")

st.write(
    "An early warning system designed to help analysts "
    "identify companies that may be at risk of financial distress."
)

st.divider()

st.header("Company Financial Data")

st.write(
    "Upload a CSV file containing company financial information "
    "to begin exploring the data."
)

uploaded_file = st.file_uploader(
    "Upload financial data (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.success("File uploaded successfully!")

        st.subheader("Data Preview")
        st.dataframe(df, use_container_width=True)

        st.subheader("Dataset Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Rows", len(df))

        with col2:
            st.metric("Total Columns", len(df.columns))

    except Exception as e:
        st.error(f"Could not read the CSV file: {e}")

st.divider()

st.caption(
    "NARZY is a prototype. Uploaded data is displayed for exploration; "
    "this version does not yet predict bankruptcy or prove fraud."
)




