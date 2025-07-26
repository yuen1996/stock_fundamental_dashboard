import streamlit as st
import pandas as pd
from utils import io_helpers, calculations

st.header("📊 Dashboard: All Stocks & Ratios")

df = io_helpers.load_data()
if df is None or df.empty or "Name" not in df.columns:
    st.warning("No stock data found. Please add data in 'Add/Edit'.")
    st.stop()

# Only show annual rows for the summary dashboard
if "IsQuarter" not in df.columns:
    df["IsQuarter"] = False
df_view = df[df["IsQuarter"] != True].copy()

# Optional filter by industry
industries = ["All"] + sorted([x for x in df_view["Industry"].dropna().unique()])
industry_sel = st.selectbox("Filter by Industry", industries, index=0)
if industry_sel != "All":
    df_view = df_view[df_view["Industry"] == industry_sel]

if df_view.empty:
    st.info("No annual rows to display for this filter.")
    st.stop()

st.dataframe(df_view.sort_values(["Name", "Year"]), use_container_width=True)

# Multi-stock, multi-year ratio table
for name in df_view["Name"].dropna().unique():
    st.markdown(f"### {name}")
    stock = df_view[df_view["Name"] == name].sort_values("Year")
    if stock.empty:
        continue

    ratios = []
    for _, row in stock.iterrows():
        r = calculations.calc_ratios(row)
        r["Year"] = row["Year"]
        ratios.append(r)

    r_df = pd.DataFrame(ratios).set_index("Year").round(4)
    st.dataframe(r_df, use_container_width=True)

