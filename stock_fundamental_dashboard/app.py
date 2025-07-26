import streamlit as st

st.set_page_config(
    page_title="Stock Fundamental Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)
try:
    st.image("assets/logo.png", width=120, caption="Stock Fundamental Dashboard", use_column_width=False)
except Exception:
    pass

st.title("Stock Fundamental Dashboard")

st.write("""
Navigate pages using the menu on the left.

- **Dashboard:** View & compare all stocks.
- **Add/Edit:** Enter or update stock details.
- **View Stock:** See full financial details, TTM, CAGR, and graphics.
""")

st.sidebar.success("Select a page above.")

