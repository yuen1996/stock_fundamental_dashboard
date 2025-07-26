import streamlit as st
import pandas as pd
from utils import io_helpers, calculations

st.header("📊 Dashboard: All Stocks & Ratios")
df = io_helpers.load_data()
if df is None or df.empty:
    st.warning("No stock data found. Please add data in 'Add/Edit'.")
    st.stop()

# Optionally filter by industry
industries = ["All"] + sorted(df["Industry"].dropna().unique())
industry_sel = st.selectbox("Filter by Industry", industries)
df_view = df if industry_sel == "All" else df[df["Industry"] == industry_sel]

stocks = df_view["Name"].unique()
if not stocks.size:
    st.warning("No stocks for this filter.")
    st.stop()

st.dataframe(df_view.sort_values(["Name", "Year"]))

# Multi-stock, multi-year ratio table
for name in stocks:
    st.markdown(f"### {name}")
    stock = df_view[df_view["Name"] == name].sort_values("Year")
    if stock.empty:
        continue

    # Calculate ratios for each year
    ratios = []
    for _, row in stock.iterrows():
        r = calculations.calc_ratios(row)
        r["Year"] = row["Year"]
        ratios.append(r)
    r_df = pd.DataFrame(ratios).set_index("Year").round(4)
    st.dataframe(r_df)

