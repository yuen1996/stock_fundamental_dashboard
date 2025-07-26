import streamlit as st
import pandas as pd
from utils import io_helpers

# --- FIELD DEFINITIONS ---

INCOME_STATEMENT_FIELDS = [
    ("Net Profit", "NetProfit"),
    ("Gross Profit", "GrossProfit"),
    ("Revenue", "Revenue"),
    ("Cost Of Sales", "CostOfSales"),
    ("Finance Costs", "FinanceCosts"),
    ("Administrative Expenses", "AdminExpenses"),
    ("Selling & Distribution Expenses", "SellDistExpenses"),
]
BALANCE_SHEET_FIELDS = [
    ("Number of Shares", "NumShares"),
    ("Current Asset", "CurrentAsset"),
    ("Other Receivables", "OtherReceivables"),
    ("Trade Receivables", "TradeReceivables"),
    ("Biological Assets", "BiologicalAssets"),
    ("Inventories", "Inventories"),
    ("Prepaid Expenses", "PrepaidExpenses"),
    ("Intangible Asset", "IntangibleAsset"),
    ("Current Liability", "CurrentLiability"),
    ("Total Asset", "TotalAsset"),
    ("Total Liability", "TotalLiability"),
    ("Shareholder Equity", "ShareholderEquity"),
    ("Reserves", "Reserves"),
]
OTHER_DATA_FIELDS = [
    ("Dividend pay cent", "Dividend"),
    ("End of year share price", "SharePrice"),
]

QUARTERLY_FIELDS = [
    ("Quarterly Net Profit", "Q_NetProfit"),
    ("Quarterly Gross Profit", "Q_GrossProfit"),
    ("Quarterly Revenue", "Q_Revenue"),
    ("Quarterly Cost Of Sales", "Q_CostOfSales"),
    ("Quarterly Finance Costs", "Q_FinanceCosts"),
    ("Quarterly Administrative Expenses", "Q_AdminExpenses"),
    ("Quarterly Selling & Distribution Expenses", "Q_SellDistExpenses"),
    # Balance sheet fields (quarterly)
    ("Number of Shares", "Q_NumShares"),
    ("Current Asset", "Q_CurrentAsset"),
    ("Other Receivables", "Q_OtherReceivables"),
    ("Trade Receivables", "Q_TradeReceivables"),
    ("Biological Assets", "Q_BiologicalAssets"),
    ("Inventories", "Q_Inventories"),
    ("Prepaid Expenses", "Q_PrepaidExpenses"),
    ("Intangible Asset", "Q_IntangibleAsset"),
    ("Current Liability", "Q_CurrentLiability"),
    ("Total Asset", "Q_TotalAsset"),
    ("Total Liability", "Q_TotalLiability"),
    ("Shareholder Equity", "Q_ShareholderEquity"),
    ("Reserves", "Q_Reserves"),
    # Other Data
    ("Current Share Price", "Q_SharePrice"),
    ("Each end per every quarter price", "Q_EndQuarterPrice"),
]

# --- LOAD DATA ---
st.header("➕ Add / Edit Stock (Annual & Quarterly)")
df = io_helpers.load_data()

# Ensure guard columns exist for backward compatibility
if "IsQuarter" not in df.columns:
    df["IsQuarter"] = False
if "Quarter" not in df.columns:
    df["Quarter"] = pd.NA

stock_name = st.text_input("Stock Name")
industry = st.text_input("Industry")

if not stock_name:
    st.info("Enter a stock name to begin.")
    st.stop()

# --- ANNUAL DATA ---
st.subheader("Annual Financial Data")
years_for_stock = sorted(df[(df["Name"] == stock_name) & (df["IsQuarter"] != True)]["Year"].dropna().unique()) if stock_name else []
years = st.multiselect("Years to edit/add (Annual)", options=[y for y in range(2000, 2036)], default=years_for_stock or [2023])
tab_annual = st.tabs([f"Year {y}" for y in years]) if years else []

annual_data = {}
for i, year in enumerate(years):
    with tab_annual[i]:
        year_data = {}
        st.markdown(f"#### Year: {year}")

        row = df[(df["Name"] == stock_name) & (df["Year"] == year) & (df["IsQuarter"] != True)]
        prefill = row.iloc[0].to_dict() if not row.empty else {}

        # Income Statement
        st.markdown("##### Income Statement")
        for label, key in INCOME_STATEMENT_FIELDS:
            year_data[key] = st.number_input(label, value=float(prefill.get(key, 0.0) or 0.0), key=f"{key}_{year}_annual", step=1.0, format="%.4f")
        # Balance Sheet
        st.markdown("##### Balance Sheet")
        for label, key in BALANCE_SHEET_FIELDS:
            year_data[key] = st.number_input(label, value=float(prefill.get(key, 0.0) or 0.0), key=f"{key}_{year}_annual", step=1.0, format="%.4f")
        # Other Data
        st.markdown("##### Other Data")
        for label, key in OTHER_DATA_FIELDS:
            year_data[key] = st.number_input(label, value=float(prefill.get(key, 0.0) or 0.0), key=f"{key}_{year}_annual", step=0.0001, format="%.4f")
        annual_data[year] = year_data

        # --- DELETE BUTTON FOR YEAR ---
        if row.shape[0] and st.button(f"Delete Year {year}", key=f"del_year_{year}"):
            df = df[~((df["Name"] == stock_name) & (df["Year"] == year) & (df["IsQuarter"] != True))]
            io_helpers.save_data(df)
            st.warning(f"Deleted year {year}. Please refresh.")
            st.stop()

# --- QUARTERLY DATA ---
st.subheader("Quarterly Financial Data")
all_quarters = ["Q1", "Q2", "Q3", "Q4"]

# Track (year, quarter) combos for this stock
quarters_in_df = df[(df["Name"] == stock_name) & (df["IsQuarter"] == True)][["Year", "Quarter"]].dropna().drop_duplicates()
quarter_choices = [(y, q) for y in range(2000, 2036) for q in all_quarters]

default_quarters = [(int(row["Year"]), str(row["Quarter"])) for _, row in quarters_in_df.iterrows()] or [(2023, "Q4")]
quarters_to_edit = st.multiselect(
    "Quarters to edit/add",
    options=quarter_choices,
    format_func=lambda x: f"{x[1]} {x[0]}",
    default=default_quarters
)
tab_quarters = st.tabs([f"{q[1]} {q[0]}" for q in quarters_to_edit]) if quarters_to_edit else []

quarterly_data = {}
for i, (year, quarter) in enumerate(quarters_to_edit):
    with tab_quarters[i]:
        q_data = {}
        st.markdown(f"#### Quarter: {quarter} {year}")

        row = df[(df["Name"] == stock_name) & (df["Year"] == year) & (df["Quarter"] == quarter) & (df["IsQuarter"] == True)]
        prefill = row.iloc[0].to_dict() if not row.empty else {}

        # Income Statement
        st.markdown("##### Quarterly Income Statement")
        for label, key in QUARTERLY_FIELDS[:7]:
            q_data[key] = st.number_input(label, value=float(prefill.get(key, 0.0) or 0.0), key=f"{key}_{year}_{quarter}_q", step=1.0, format="%.4f")
        # Balance Sheet
        st.markdown("##### Quarterly Balance Sheet")
        for label, key in QUARTERLY_FIELDS[7:21]:
            q_data[key] = st.number_input(label, value=float(prefill.get(key, 0.0) or 0.0), key=f"{key}_{year}_{quarter}_q", step=1.0, format="%.4f")
        # Other Data
        st.markdown("##### Quarterly Other Data")
        for label, key in QUARTERLY_FIELDS[21:]:
            q_data[key] = st.number_input(label, value=float(prefill.get(key, 0.0) or 0.0), key=f"{key}_{year}_{quarter}_q", step=0.0001, format="%.4f")

        quarterly_data[(year, quarter)] = q_data

        # --- DELETE BUTTON FOR QUARTER ---
        if row.shape[0] and st.button(f"Delete {quarter} {year}", key=f"del_{year}_{quarter}"):
            df = df[~((df["Name"] == stock_name) & (df["Year"] == year) & (df["Quarter"] == quarter) & (df["IsQuarter"] == True))]
            io_helpers.save_data(df)
            st.warning(f"Deleted {quarter} {year}. Please refresh.")
            st.stop()

# --- SAVE BUTTON ---
if st.button("💾 Save All Changes"):
    if not stock_name or not industry:
        st.error("Please enter stock name and industry.")
        st.stop()

    # Save annual
    for year in annual_data:
        row = {"Name": stock_name, "Industry": industry, "Year": year, "IsQuarter": False, "Quarter": ""}
        row.update(annual_data[year])
        cond = (df["Name"] == stock_name) & (df["Year"] == year) & (df["IsQuarter"] != True)
        if cond.any():
            df.loc[cond, row.keys()] = list(row.values())
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Save quarterly
    for (year, quarter) in quarterly_data:
        row = {"Name": stock_name, "Industry": industry, "Year": year, "IsQuarter": True, "Quarter": quarter}
        row.update(quarterly_data[(year, quarter)])
        cond = (df["Name"] == stock_name) & (df["Year"] == year) & (df["Quarter"] == quarter) & (df["IsQuarter"] == True)
        if cond.any():
            df.loc[cond, row.keys()] = list(row.values())
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    io_helpers.save_data(df)
    st.success("Saved all changes.")

