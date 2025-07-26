import streamlit as st

st.set_page_config(page_title="Stock Fundamental Dashboard", layout="wide", page_icon="📈")

st.title("Stock Fundamental Dashboard")

st.markdown(
    """
    Use the left sidebar to navigate:

    - **Dashboard**: summary of all stocks (annual).
    - **Add or Edit**: manage annual & quarterly data.
    - **View Stock**: deep‑dive per stock with ratios and charts.
    """
)

