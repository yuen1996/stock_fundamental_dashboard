import streamlit as st
import pandas as pd
from utils import io_helpers

# ---------- Force Wide Layout on Page ----------
st.set_page_config(layout="wide")

# ---------- Page CSS ----------
BASE_CSS = """
<style>
html, body, [class*="css"] { font-size: 16px !important; }
h1, h2, h3, h4 { color:#0f172a !important; font-weight:800 !important; letter-spacing:.2px; }
.stApp { background: radial-gradient(1100px 600px at 18% -10%, #f7fbff 0%, #eef4ff 45%, #ffffff 100%); }
.stDataFrame, .stDataEditor { font-size: 15px !important; }
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

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

# --- ANNUAL / QUARTERLY FORMS (show only when a name is provided) ---
if stock_name:
    st.subheader("Annual Financial Data")
    years_for_stock = sorted(df[(df["Name"] == stock_name) & (df["IsQuarter"] != True)]["Year"].dropna().unique())
    years = st.multiselect("Years to edit/add (Annual)", options=[y for y in range(2000, 2036)],
                           default=years_for_stock or [2023])
    tab_annual = st.tabs([f"Year {y}" for y in years]) if years else []

    annual_data = {}
    for i, year in enumerate(years):
        with tab_annual[i]:
            year_data = {}
            st.markdown(f"#### Year: {year}")

            row = df[(df["Name"] == stock_name) & (df["Year"] == year) & (df["IsQuarter"] != True)]
            prefill = row.iloc[0].to_dict() if not row.empty else {}

            # --- Reset (Re-key) button: zero all inputs for this year ---
            if st.button("Reset all fields to 0 for this year", key=f"reset_{year}_annual"):
                for _, key in INCOME_STATEMENT_FIELDS + BALANCE_SHEET_FIELDS + OTHER_DATA_FIELDS:
                    st.session_state[f"{key}_{year}_annual"] = 0.0
                st.experimental_rerun()

            st.markdown("##### Income Statement")
            for label, key in INCOME_STATEMENT_FIELDS:
                year_data[key] = st.number_input(
                    label,
                    value=float(prefill.get(key, 0.0) or 0.0),
                    key=f"{key}_{year}_annual",
                    step=1.0, format="%.4f"
                )
            st.markdown("##### Balance Sheet")
            for label, key in BALANCE_SHEET_FIELDS:
                year_data[key] = st.number_input(
                    label,
                    value=float(prefill.get(key, 0.0) or 0.0),
                    key=f"{key}_{year}_annual",
                    step=1.0, format="%.4f"
                )
            st.markdown("##### Other Data")
            for label, key in OTHER_DATA_FIELDS:
                year_data[key] = st.number_input(
                    label,
                    value=float(prefill.get(key, 0.0) or 0.0),
                    key=f"{key}_{year}_annual",
                    step=0.0001, format="%.4f"
                )
            annual_data[year] = year_data

            if row.shape[0] and st.button(f"Delete Year {year}", key=f"del_year_{year}"):
                df = df[~((df["Name"] == stock_name) & (df["Year"] == year) & (df["IsQuarter"] != True))]
                io_helpers.save_data(df)
                st.warning(f"Deleted year {year}. Please refresh.")
                st.stop()

    # -------------------- QUARTERLY SECTION --------------------
    st.subheader("Quarterly Financial Data")

    all_quarters = ["Q1", "Q2", "Q3", "Q4"]

    # === NEW: Quick action to create missing quarterly rows for selected years ===
    with st.container():
        st.markdown("**Quick action:** create empty quarterly rows (Q1–Q4) for selected years if missing.")
        # union of annual years you chose and any existing annual years for this stock
        candidate_years = sorted(set(years) | set(years_for_stock))
        years_to_generate = st.multiselect(
            "Years to generate quarterly rows",
            options=candidate_years or [2023],
            default=candidate_years or [2023],
            key="gen_quarter_years"
        )
        if st.button("➕ Create missing quarters (Q1–Q4)", key="btn_gen_quarters"):
            if not stock_name or not industry:
                st.error("Please enter Stock Name and Industry first.")
            else:
                new_rows = []
                for y in years_to_generate:
                    for q in all_quarters:
                        mask = (df["Name"] == stock_name) & (df["Year"] == y) & \
                               (df["Quarter"] == q) & (df["IsQuarter"] == True)
                        if not mask.any():
                            row = {"Name": stock_name, "Industry": industry,
                                   "Year": y, "IsQuarter": True, "Quarter": q}
                            # initialize all quarterly fields as 0.0 for easy re-key later
                            for _, key in QUARTERLY_FIELDS:
                                row[key] = 0.0
                            new_rows.append(row)
                if new_rows:
                    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                    io_helpers.save_data(df)
                    st.success(f"Added {len(new_rows)} quarterly row(s).")
                    st.experimental_rerun()
                else:
                    st.info("No missing quarters for the selected year(s).")

    # Existing quarterly editor UI (unchanged)
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

            row = df[(df["Name"] == stock_name) & (df["Year"] == year) &
                     (df["Quarter"] == quarter) & (df["IsQuarter"] == True)]
            prefill = row.iloc[0].to_dict() if not row.empty else {}

            # Reset quarter inputs quickly
            if st.button("Reset all fields to 0 for this quarter", key=f"reset_{year}_{quarter}_q"):
                for _, key in QUARTERLY_FIELDS:
                    st.session_state[f"{key}_{year}_{quarter}_q"] = 0.0
                st.experimental_rerun()

            st.markdown("##### Quarterly Income Statement")
            for label, key in QUARTERLY_FIELDS[:7]:
                q_data[key] = st.number_input(
                    label,
                    value=float(prefill.get(key, 0.0) or 0.0),
                    key=f"{key}_{year}_{quarter}_q",
                    step=1.0, format="%.4f"
                )
            st.markdown("##### Quarterly Balance Sheet")
            for label, key in QUARTERLY_FIELDS[7:21]:
                q_data[key] = st.number_input(
                    label,
                    value=float(prefill.get(key, 0.0) or 0.0),
                    key=f"{key}_{year}_{quarter}_q",
                    step=1.0, format="%.4f"
                )
            st.markdown("##### Quarterly Other Data")
            for label, key in QUARTERLY_FIELDS[21:]:
                q_data[key] = st.number_input(
                    label,
                    value=float(prefill.get(key, 0.0) or 0.0),
                    key=f"{key}_{year}_{quarter}_q",
                    step=0.0001, format="%.4f"
                )

            quarterly_data[(year, quarter)] = q_data

            if row.shape[0] and st.button(f"Delete {quarter} {year}", key=f"del_{year}_{quarter}"):
                df = df[~((df["Name"] == stock_name) & (df["Year"] == year) &
                          (df["Quarter"] == quarter) & (df["IsQuarter"] == True))]
                io_helpers.save_data(df)
                st.warning(f"Deleted {quarter} {year}. Please refresh.")
                st.stop()

    if st.button("💾 Save All Changes"):
        if not stock_name or not industry:
            st.error("Please enter stock name and industry.")
            st.stop()

        # Upsert annual rows
        for year in annual_data:
            row = {"Name": stock_name, "Industry": industry, "Year": year, "IsQuarter": False, "Quarter": ""}
            row.update(annual_data[year])
            cond = (df["Name"] == stock_name) & (df["Year"] == year) & (df["IsQuarter"] != True)
            if cond.any():
                df.loc[cond, row.keys()] = list(row.values())
            else:
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        # Upsert quarterly rows
        for (year, quarter) in quarterly_data:
            row = {"Name": stock_name, "Industry": industry, "Year": year, "IsQuarter": True, "Quarter": quarter}
            row.update(quarterly_data[(year, quarter)])
            cond = (df["Name"] == stock_name) & (df["Year"] == year) & \
                   (df["Quarter"] == quarter) & (df["IsQuarter"] == True)
            if cond.any():
                df.loc[cond, row.keys()] = list(row.values())
            else:
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        io_helpers.save_data(df)
        st.success("Saved all changes.")
else:
    st.info("Tip: enter a Stock Name above to add/edit annual & quarterly values. You can still use the quick editors below.")

# --- QUICK EDIT (ANNUAL) ---
st.divider()
st.subheader(✏️ Quick Edit Existing Annual Rows (All Stocks)")

annual_all = df[df["IsQuarter"] != True].copy()

# Filters
fc1, fc2 = st.columns([1,2])
with fc1:
    industries = ["All"] + sorted([x for x in annual_all["Industry"].dropna().unique()])
    industry_sel = st.selectbox("Filter by Industry", industries, index=0, key="qe_industry")
with fc2:
    search_text = st.text_input("🔎 Search stock name or industry", key="qe_search")

v = annual_all if industry_sel == "All" else annual_all[annual_all["Industry"] == industry_sel]
if search_text.strip():
    q = search_text.strip().lower()
    v = v[
        v["Name"].str.lower().str.contains(q, na=False) |
        v["Industry"].str.lower().str.contains(q, na=False)
    ]

if v.empty:
    st.info("No annual rows to edit for the current filter.")
else:
    for name in sorted(v["Name"].dropna().unique()):
        sv = v[v["Name"] == name].sort_values("Year").reset_index(drop=True).copy()
        if sv.empty:
            continue
        yr_min = int(sv["Year"].min()) if pd.notna(sv["Year"]).any() else "-"
        yr_max = int(sv["Year"].max()) if pd.notna(sv["Year"]).any() else "-"
        with st.expander(f"{name}  •  Years {yr_min}–{yr_max}", expanded=False):
            sv.insert(0, "RowKey", sv.apply(lambda r: f"{r['Name']}|{int(r['Year'])}|A", axis=1))
            sv.insert(1, "Delete", False)
            edited = st.data_editor(
                sv,
                use_container_width=True,
                height=360,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "RowKey": st.column_config.TextColumn("RowKey", help="Internal key", disabled=True, width="small"),
                    "Delete": st.column_config.CheckboxColumn("Delete", help="Tick to delete this year"),
                    "Name": st.column_config.TextColumn("Name", disabled=True),
                    "Year": st.column_config.NumberColumn("Year", disabled=True, format="%d"),
                },
                key=f"qe_editor_{name}",
            )
            if st.button(f"💾 Save changes for {name}", key=f"qe_save_{name}"):
                # deletions
                del_keys = set(edited.loc[edited["Delete"] == True, "RowKey"].tolist())
                for key in del_keys:
                    s_name, s_year, _ = key.split("|")
                    mask = (df["Name"] == s_name) & (df["Year"] == int(s_year)) & (df["IsQuarter"] != True)
                    df.drop(df[mask].index, inplace=True)
                # updates
                keep = edited[edited["Delete"] != True].copy()
                for _, er in keep.iterrows():
                    s_name, s_year, _ = er["RowKey"].split("|")
                    mask = (df["Name"] == s_name) & (df["Year"] == int(s_year)) & (df["IsQuarter"] != True)
                    update_cols = [c for c in keep.columns if c not in ("RowKey", "Delete")]
                    for c in update_cols:
                        if c in df.columns:
                            df.loc[mask, c] = er[c]
                io_helpers.save_data(df)
                st.success(f"Saved changes for {name}.")
                st.experimental_rerun()

# --- QUICK EDIT (QUARTERLY) ---
st.divider()
st.subheader("✏️ Quick Edit Existing Quarterly Rows (All Stocks)")

quarter_all = df[df["IsQuarter"] == True].copy()

# Filters (same as annual)
qfc1, qfc2 = st.columns([1,2])
with qfc1:
    q_industries = ["All"] + sorted([x for x in quarter_all["Industry"].dropna().unique()])
    q_industry_sel = st.selectbox("Filter by Industry", q_industries, index=0, key="qe_q_industry")
with qfc2:
    q_search_text = st.text_input("🔎 Search stock name or industry", key="qe_q_search")

qv = quarter_all if q_industry_sel == "All" else quarter_all[quarter_all["Industry"] == q_industry_sel]
if q_search_text.strip():
    qtxt = q_search_text.strip().lower()
    qv = qv[
        qv["Name"].str.lower().str.contains(qtxt, na=False) |
        qv["Industry"].str.lower().str.contains(qtxt, na=False)
    ]

if qv.empty:
    st.info("No quarterly rows to edit for the current filter.")
else:
    for name in sorted(qv["Name"].dropna().unique()):
        svq = qv[qv["Name"] == name].sort_values(["Year", "Quarter"]).reset_index(drop=True).copy()
        if svq.empty:
            continue
        yr_min = int(svq["Year"].min()) if pd.notna(svq["Year"]).any() else "-"
        yr_max = int(svq["Year"].max()) if pd.notna(svq["Year"]).any() else "-"
        with st.expander(f"{name}  •  Periods {yr_min}–{yr_max}", expanded=False):
            # internal key
            svq.insert(0, "RowKey", svq.apply(lambda r: f"{r['Name']}|{int(r['Year'])}|{str(r['Quarter'])}|Q", axis=1))
            svq.insert(1, "Delete", False)

            edited_q = st.data_editor(
                svq,
                use_container_width=True,
                height=360,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "RowKey": st.column_config.TextColumn("RowKey", help="Internal key", disabled=True, width="small"),
                    "Delete": st.column_config.CheckboxColumn("Delete", help="Tick to delete this quarter"),
                    "Name": st.column_config.TextColumn("Name", disabled=True),
                    "Year": st.column_config.NumberColumn("Year", disabled=True, format="%d"),
                    "Quarter": st.column_config.TextColumn("Quarter", disabled=True),
                    # To allow changing quarter, replace the above with:
                    # "Quarter": st.column_config.SelectboxColumn("Quarter", options=["Q1","Q2","Q3","Q4"])
                },
                key=f"qe_q_editor_{name}",
            )

            if st.button(f"💾 Save quarterly changes for {name}", key=f"qe_q_save_{name}"):
                # deletions
                del_keys = set(edited_q.loc[edited_q["Delete"] == True, "RowKey"].tolist())
                for key in del_keys:
                    s_name, s_year, s_quarter, _ = key.split("|")
                    mask = (df["Name"] == s_name) & (df["Year"] == int(s_year)) & \
                           (df["Quarter"] == s_quarter) & (df["IsQuarter"] == True)
                    df.drop(df[mask].index, inplace=True)
                # updates
                keep = edited_q[edited_q["Delete"] != True].copy()
                for _, er in keep.iterrows():
                    s_name, s_year, s_quarter, _ = er["RowKey"].split("|")
                    mask = (df["Name"] == s_name) & (df["Year"] == int(s_year)) & \
                           (df["Quarter"] == s_quarter) & (df["IsQuarter"] == True)
                    update_cols = [c for c in keep.columns if c not in ("RowKey", "Delete")]
                    for c in update_cols:
                        if c in df.columns:
                            df.loc[mask, c] = er[c]
                io_helpers.save_data(df)
                st.success(f"Saved quarterly changes for {name}.")
                st.experimental_rerun()

