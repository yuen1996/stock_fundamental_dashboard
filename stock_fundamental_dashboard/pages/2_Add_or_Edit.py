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
    # Quarterly P&L
    ("Quarterly Net Profit", "Q_NetProfit"),
    ("Quarterly Gross Profit", "Q_GrossProfit"),
    ("Quarterly Revenue", "Q_Revenue"),
    ("Quarterly Cost Of Sales", "Q_CostOfSales"),
    ("Quarterly Finance Costs", "Q_FinanceCosts"),
    ("Quarterly Administrative Expenses", "Q_AdminExpenses"),
    ("Quarterly Selling & Distribution Expenses", "Q_SellDistExpenses"),
    # Quarterly Balance Sheet
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
    # Quarterly Other
    ("Current Share Price", "Q_SharePrice"),
    ("Each end per every quarter price", "Q_EndQuarterPrice"),
]

# ---------- Load data ----------
st.header("➕ Add / Edit Stock (Annual & Quarterly)")
df = io_helpers.load_data()
if df is None:
    df = pd.DataFrame()

# Ensure guard columns exist for backward compatibility
if "IsQuarter" not in df.columns:
    df["IsQuarter"] = False
if "Quarter" not in df.columns:
    df["Quarter"] = pd.NA

stock_name = st.text_input("Stock Name")
industry = st.text_input("Industry")

# ---------- Annual & Quarterly Forms ----------
if stock_name:
    st.subheader("Annual Financial Data")

    years_for_stock = sorted(
        df[(df["Name"] == stock_name) & (df["IsQuarter"] != True)]["Year"].dropna().unique().tolist()
    )
    years = st.multiselect(
        "Years to edit/add (Annual)",
        options=[y for y in range(2000, 2036)],
        default=years_for_stock or [2023],
    )
    tab_annual = st.tabs([f"Year {y}" for y in years]) if years else []

    annual_data = {}
    for i, year in enumerate(years):
        with tab_annual[i]:
            year_data = {}
            st.markdown(f"#### Year: {year}")

            row = df[(df["Name"] == stock_name) & (df["Year"] == year) & (df["IsQuarter"] != True)]
            prefill = row.iloc[0].to_dict() if not row.empty else {}

            # --- Reset (Re-key) for this Year ---
            if st.button("Reset all fields to 0 for this year", key=f"reset_{year}_annual"):
                for _, key in INCOME_STATEMENT_FIELDS + BALANCE_SHEET_FIELDS + OTHER_DATA_FIELDS:
                    st.session_state[f"{key}_{year}_annual"] = 0.0
                st.experimental_rerun()

            # --- Create empty quarters for THIS year (Q1–Q4) if missing ---
            if st.button("➕ Create empty quarters for this year (Q1–Q4)", key=f"gen_quarters_single_{year}"):
                all_quarters_local = ["Q1", "Q2", "Q3", "Q4"]
                new_rows = []
                for q in all_quarters_local:
                    mask = (df["Name"] == stock_name) & (df["Year"] == year) & \
                           (df["Quarter"] == q) & (df["IsQuarter"] == True)
                    if not mask.any():
                        row_new = {
                            "Name": stock_name, "Industry": industry,
                            "Year": year, "IsQuarter": True, "Quarter": q
                        }
                        for _, key in QUARTERLY_FIELDS:
                            row_new[key] = 0.0
                        new_rows.append(row_new)
                if new_rows:
                    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                    io_helpers.save_data(df)
                    st.success(f"Added {len(new_rows)} quarterly row(s) for {year}.")
                    st.experimental_rerun()
                else:
                    st.info(f"All quarters already exist for {year}.")

            st.markdown("##### Income Statement")
            for label, key in INCOME_STATEMENT_FIELDS:
                year_data[key] = st.number_input(
                    label,
                    value=float(prefill.get(key, 0.0) or 0.0),
                    key=f"{key}_{year}_annual",
                    step=1.0,
                    format="%.4f",
                )

            st.markdown("##### Balance Sheet")
            for label, key in BALANCE_SHEET_FIELDS:
                year_data[key] = st.number_input(
                    label,
                    value=float(prefill.get(key, 0.0) or 0.0),
                    key=f"{key}_{year}_annual",
                    step=1.0,
                    format="%.4f",
                )

            st.markdown("##### Other Data")
            for label, key in OTHER_DATA_FIELDS:
                year_data[key] = st.number_input(
                    label,
                    value=float(prefill.get(key, 0.0) or 0.0),
                    key=f"{key}_{year}_annual",
                    step=0.0001,
                    format="%.4f",
                )

            annual_data[year] = year_data

            # Delete this year
            if row.shape[0] and st.button(f"Delete Year {year}", key=f"del_year_{year}"):
                df = df[~((df["Name"] == stock_name) & (df["Year"] == year) & (df["IsQuarter"] != True))]
                io_helpers.save_data(df)
                st.warning(f"Deleted year {year}. Please refresh.")
                st.stop()

    # -------------------- Quarterly Section --------------------
    st.subheader("Quarterly Financial Data")

    all_quarters = ["Q1", "Q2", "Q3", "Q4"]

    # Bulk create missing quarters for selected years
    with st.container():
        st.markdown("**Quick action:** Create empty quarterly rows (Q1–Q4) for selected years if missing.")
        candidate_years = sorted(set(years) | set(years_for_stock))
        years_to_generate = st.multiselect(
            "Years to generate quarterly rows",
            options=candidate_years or [2023],
            default=candidate_years or [2023],
            key="gen_quarter_years",
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
                            row_ins = {
                                "Name": stock_name, "Industry": industry,
                                "Year": y, "IsQuarter": True, "Quarter": q
                            }
                            for _, key in QUARTERLY_FIELDS:
                                row_ins[key] = 0.0
                            new_rows.append(row_ins)
                if new_rows:
                    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                    io_helpers.save_data(df)
                    st.success(f"Added {len(new_rows)} quarterly row(s).")
                    st.experimental_rerun()
                else:
                    st.info("No missing quarters for the selected year(s).")

    # Select quarters to edit/add
    quarters_in_df = df[(df["Name"] == stock_name) & (df["IsQuarter"] == True)][["Year", "Quarter"]].dropna().drop_duplicates()
    quarter_choices = [(y, q) for y in range(2000, 2036) for q in all_quarters]

    default_quarters = (
        [(int(row["Year"]), str(row["Quarter"])) for _, row in quarters_in_df.iterrows()] or [(2023, "Q4")]
    )
    quarters_to_edit = st.multiselect(
        "Quarters to edit/add",
        options=quarter_choices,
        format_func=lambda x: f"{x[1]} {x[0]}",
        default=default_quarters,
    )
    tab_quarters = st.tabs([f"{q[1]} {q[0]}" for q in quarters_to_edit]) if quarters_to_edit else []

    quarterly_data = {}
    for i, (yr, qtr) in enumerate(quarters_to_edit):
        with tab_quarters[i]:
            q_data = {}
            st.markdown(f"#### Quarter: {qtr} {yr}")

            row_q = df[(df["Name"] == stock_name) & (df["Year"] == yr) &
                       (df["Quarter"] == qtr) & (df["IsQuarter"] == True)]
            prefill_q = row_q.iloc[0].to_dict() if not row_q.empty else {}

            # Reset (Re-key) for this Quarter
            if st.button("Reset all fields to 0 for this quarter", key=f"reset_{yr}_{qtr}_q"):
                for _, key in QUARTERLY_FIELDS:
                    st.session_state[f"{key}_{yr}_{qtr}_q"] = 0.0
                st.experimental_rerun()

            st.markdown("##### Quarterly Income Statement")
            for label, key in QUARTERLY_FIELDS[:7]:
                q_data[key] = st.number_input(
                    label,
                    value=float(prefill_q.get(key, 0.0) or 0.0),
                    key=f"{key}_{yr}_{qtr}_q",
                    step=1.0,
                    format="%.4f",
                )

            st.markdown("##### Quarterly Balance Sheet")
            for label, key in QUARTERLY_FIELDS[7:21]:
                q_data[key] = st.number_input(
                    label,
                    value=float(prefill_q.get(key, 0.0) or 0.0),
                    key=f"{key}_{yr}_{qtr}_q",
                    step=1.0,
                    format="%.4f",
                )

            st.markdown("##### Quarterly Other Data")
            for label, key in QUARTERLY_FIELDS[21:]:
                q_data[key] = st.number_input(
                    label,
                    value=float(prefill_q.get(key, 0.0) or 0.0),
                    key=f"{key}_{yr}_{qtr}_q",
                    step=0.0001,
                    format="%.4f",
                )

            quarterly_data[(yr, qtr)] = q_data

            # Delete this quarter
            if row_q.shape[0] and st.button(f"Delete {qtr} {yr}", key=f"del_{yr}_{qtr}"):
                df = df[~((df["Name"] == stock_name) & (df["Year"] == yr) &
                          (df["Quarter"] == qtr) & (df["IsQuarter"] == True))]
                io_helpers.save_data(df)
                st.warning(f"Deleted {qtr} {yr}. Please refresh.")
                st.stop()

    # Save all changes (upsert)
    if st.button("💾 Save All Changes"):
        if not stock_name or not industry:
            st.error("Please enter stock name and industry.")
            st.stop()

        # Upsert annual rows
        for year in annual_data:
            row_up = {
                "Name": stock_name, "Industry": industry, "Year": year, "IsQuarter": False, "Quarter": ""
            }
            row_up.update(annual_data[year])
            cond = (df["Name"] == stock_name) & (df["Year"] == year) & (df["IsQuarter"] != True)
            if cond.any():
                df.loc[cond, row_up.keys()] = list(row_up.values())
            else:
                df = pd.concat([df, pd.DataFrame([row_up])], ignore_index=True)

        # Upsert quarterly rows
        for (year, quarter) in quarterly_data:
            row_uq = {
                "Name": stock_name, "Industry": industry, "Year": year, "IsQuarter": True, "Quarter": quarter
            }
            row_uq.update(quarterly_data[(year, quarter)])
            cond = (df["Name"] == stock_name) & (df["Year"] == year) & \
                   (df["Quarter"] == quarter) & (df["IsQuarter"] == True)
            if cond.any():
                df.loc[cond, row_uq.keys()] = list(row_uq.values())
            else:
                df = pd.concat([df, pd.DataFrame([row_uq])], ignore_index=True)

        io_helpers.save_data(df)
        st.success("Saved all changes.")
else:
    st.info("Tip: enter a Stock Name above to add/edit annual & quarterly values. You can still use the quick editors below.")

# =================================================================
# QUICK EDIT BY STOCK (Annual & Quarterly in tabs; add/edit/delete)
# =================================================================
st.divider()
st.subheader("🛠 Quick Edit by Stock (Annual & Quarterly)")

# Base filters
all_rows = df.copy()
c1, c2 = st.columns([1, 2])
with c1:
    industries = ["All"] + sorted([x for x in all_rows["Industry"].dropna().unique()])
    f_industry = st.selectbox("Filter by Industry", industries, index=0, key="qeb_industry")
with c2:
    f_query = st.text_input("🔎 Search stock name or industry", key="qeb_search")

if f_industry != "All":
    all_rows = all_rows[all_rows["Industry"] == f_industry]
if f_query.strip():
    q = f_query.strip().lower()
    all_rows = all_rows[
        all_rows["Name"].str.lower().str.contains(q, na=False) |
        all_rows["Industry"].str.lower().str.contains(q, na=False)
    ]

if all_rows.empty:
    st.info("No rows for the current filter.")
else:
    for name in sorted(all_rows["Name"].dropna().unique()):
        st.markdown("---")
        with st.expander(name, expanded=False):
            tabs = st.tabs(["Annual", "Quarterly"])

            # ----------------- Annual Tab -----------------
            with tabs[0]:
                av = df[(df["Name"] == name) & (df["IsQuarter"] != True)].sort_values("Year").reset_index(drop=True).copy()
                # Prepare editor frame
                if not av.empty and "Industry" in av.columns:
                    industry_default = av["Industry"].iloc[0]
                else:
                    industry_default = ""

                # RowKey + Delete + visible columns
                av.insert(0, "RowKey", av.apply(lambda r: f"{r['Name']}|{int(r['Year'])}|A", axis=1))
                av.insert(1, "Delete", False)

                # Allow adding new annual rows
                edited_a = st.data_editor(
                    av,
                    use_container_width=True,
                    height=360,
                    hide_index=True,
                    num_rows="dynamic",
                    column_config={
                        "RowKey": st.column_config.TextColumn("RowKey", help="Internal key", disabled=True, width="small"),
                        "Delete": st.column_config.CheckboxColumn("Delete", help="Tick to delete this year"),
                        "Name": st.column_config.TextColumn("Stock", disabled=True),
                        "Industry": st.column_config.TextColumn("Industry", help="Optional"),
                        "Year": st.column_config.NumberColumn("Year", format="%d"),
                    },
                    key=f"qeb_annual_{name}",
                )

                if st.button(f"💾 Save Annual for {name}", key=f"qeb_save_a_{name}"):
                    # Identify deletions
                    del_keys = set(edited_a.loc[edited_a["Delete"] == True, "RowKey"].tolist())

                    # Keep non-deleted
                    keep = edited_a[edited_a["Delete"] != True].copy()

                    # Upsert each row (existing or new)
                    for _, er in keep.iterrows():
                        # If RowKey is blank -> new row
                        if pd.isna(er.get("RowKey")) or er.get("RowKey") == "":
                            y = er.get("Year")
                            if pd.isna(y):
                                continue
                            y = int(y)
                            row_up = {"Name": name,
                                      "Industry": er.get("Industry", industry_default),
                                      "Year": y,
                                      "IsQuarter": False,
                                      "Quarter": ""}
                            # copy other known columns present in df
                            for col in df.columns:
                                if col in ("Name", "Industry", "Year", "IsQuarter", "Quarter"):
                                    continue
                                if col in er:
                                    row_up[col] = er[col]
                            df = pd.concat([df, pd.DataFrame([row_up])], ignore_index=True)
                        else:
                            s_name, s_year, _ = er["RowKey"].split("|")
                            mask = (df["Name"] == s_name) & (df["Year"] == int(s_year)) & (df["IsQuarter"] != True)
                            for c in keep.columns:
                                if c in ("RowKey", "Delete"):
                                    continue
                                if c in df.columns:
                                    df.loc[mask, c] = er[c]

                    # Apply deletions
                    for key in del_keys:
                        s_name, s_year, _ = key.split("|")
                        mask = (df["Name"] == s_name) & (df["Year"] == int(s_year)) & (df["IsQuarter"] != True)
                        df.drop(df[mask].index, inplace=True)

                    io_helpers.save_data(df)
                    st.success(f"Saved annual changes for {name}.")
                    st.experimental_rerun()

            # ----------------- Quarterly Tab -----------------
            with tabs[1]:
                qv = df[(df["Name"] == name) & (df["IsQuarter"] == True)].sort_values(["Year", "Quarter"]).reset_index(drop=True).copy()
                quarters = ["Q1", "Q2", "Q3", "Q4"]

                if not qv.empty and "Industry" in qv.columns:
                    industry_default_q = qv["Industry"].iloc[0]
                else:
                    industry_default_q = ""

                qv.insert(0, "RowKey", qv.apply(lambda r: f"{r['Name']}|{int(r['Year'])}|{str(r['Quarter'])}|Q", axis=1))
                qv.insert(1, "Delete", False)

                edited_q = st.data_editor(
                    qv,
                    use_container_width=True,
                    height=360,
                    hide_index=True,
                    num_rows="dynamic",
                    column_config={
                        "RowKey": st.column_config.TextColumn("RowKey", help="Internal key", disabled=True, width="small"),
                        "Delete": st.column_config.CheckboxColumn("Delete", help="Tick to delete this period"),
                        "Name": st.column_config.TextColumn("Stock", disabled=True),
                        "Industry": st.column_config.TextColumn("Industry", help="Optional"),
                        "Year": st.column_config.NumberColumn("Year", format="%d"),
                        "Quarter": st.column_config.SelectboxColumn("Quarter", options=quarters),
                    },
                    key=f"qeb_quarter_{name}",
                )

                if st.button(f"💾 Save Quarterly for {name}", key=f"qeb_save_q_{name}"):
                    # Deletions
                    del_keys = set(edited_q.loc[edited_q["Delete"] == True, "RowKey"].tolist())

                    # Keep non-deleted
                    keep = edited_q[edited_q["Delete"] != True].copy()

                    # Upsert
                    for _, er in keep.iterrows():
                        if pd.isna(er.get("RowKey")) or er.get("RowKey") == "":
                            y = er.get("Year")
                            qt = er.get("Quarter")
                            if pd.isna(y) or pd.isna(qt) or qt not in quarters:
                                continue
                            y = int(y)
                            row_uq = {"Name": name,
                                     "Industry": er.get("Industry", industry_default_q),
                                     "Year": y,
                                     "IsQuarter": True,
                                     "Quarter": qt}
                            for col in df.columns:
                                if col in ("Name", "Industry", "Year", "IsQuarter", "Quarter"):
                                    continue
                                if col in er:
                                    row_uq[col] = er[col]
                            df = pd.concat([df, pd.DataFrame([row_uq])], ignore_index=True)
                        else:
                            s_name, s_year, s_quarter, _ = er["RowKey"].split("|")
                            mask = (df["Name"] == s_name) & (df["Year"] == int(s_year)) & \
                                   (df["Quarter"] == s_quarter) & (df["IsQuarter"] == True)
                            for c in keep.columns:
                                if c in ("RowKey", "Delete"):
                                    continue
                                if c in df.columns:
                                    df.loc[mask, c] = er[c]

                    # Apply deletions
                    for key in del_keys:
                        s_name, s_year, s_quarter, _ = key.split("|")
                        mask = (df["Name"] == s_name) & (df["Year"] == int(s_year)) & \
                               (df["Quarter"] == s_quarter) & (df["IsQuarter"] == True)
                        df.drop(df[mask].index, inplace=True)

                    io_helpers.save_data(df)
                    st.success(f"Saved quarterly changes for {name}.")
                    st.experimental_rerun()

