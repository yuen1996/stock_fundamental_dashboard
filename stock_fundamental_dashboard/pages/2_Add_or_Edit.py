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

# --- helper: infer a default Industry for a stock from existing rows ---
def _infer_industry_for_stock(df: pd.DataFrame, stock: str, fallback: str = "") -> str:
    """
    Pick the most common non-empty Industry for a given stock.
    Falls back to the provided fallback or "".
    """
    if df is None or df.empty:
        return fallback
    s = (
        df.loc[df["Name"] == stock, "Industry"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    if s.empty:
        return fallback
    try:
        m = s.mode()
        return m.iloc[0] if not m.empty else s.iloc[0]
    except Exception:
        return s.iloc[0]

# --- FIELD DEFINITIONS (data-entry fields only; ratios are NOT in quick-edit) ---
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

# Convenience sets for filtering quick-edit columns (to HIDE ratios)
ANNUAL_ALLOWED_BASE = {"Name", "Industry", "Year", "IsQuarter", "Quarter"} \
    | {k for _, k in INCOME_STATEMENT_FIELDS} \
    | {k for _, k in BALANCE_SHEET_FIELDS} \
    | {k for _, k in OTHER_DATA_FIELDS}

QUARTERLY_ALLOWED_BASE = {"Name", "Industry", "Year", "Quarter", "IsQuarter"} \
    | {k for _, k in QUARTERLY_FIELDS}

# ---------- Load data ----------
st.header("➕ Add / Edit Stock (Annual & Quarterly)")
df = io_helpers.load_data()
if df is None:
    df = pd.DataFrame()

# Guard columns (backward compatibility)
if "IsQuarter" not in df.columns:
    df["IsQuarter"] = False
if "Quarter" not in df.columns:
    df["Quarter"] = pd.NA

stock_name = st.text_input("Stock Name")
industry = st.text_input("Industry")

# ==================== TOP FORMS: ANNUAL & QUARTERLY ====================
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

            # Reset year
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

    # -------------------- Quarterly Section (form) --------------------
    st.subheader("Quarterly Financial Data")
    all_quarters = ["Q1", "Q2", "Q3", "Q4"]

    # Quick action: create missing quarterly rows for selected years
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
                            row_ins = {"Name": stock_name, "Industry": industry, "Year": y, "IsQuarter": True, "Quarter": q}
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

    # Form tabs to edit specific quarters
    quarters_in_df = df[(df["Name"] == stock_name) & (df["IsQuarter"] == True)][["Year", "Quarter"]].dropna().drop_duplicates()
    quarter_choices = [(y, q) for y in range(2000, 2036) for q in all_quarters]
    default_quarters = ([(int(r["Year"]), str(r["Quarter"])) for _, r in quarters_in_df.iterrows()] or [(2023, "Q4")])
    tab_quarters = st.tabs([f"{q} {y}" for (y, q) in default_quarters]) if default_quarters else []

    quarterly_data = {}
    for i, (yr, qtr) in enumerate(default_quarters):
        with tab_quarters[i]:
            q_data = {}
            st.markdown(f"#### Quarter: {qtr} {yr}")

            row_q = df[(df["Name"] == stock_name) & (df["Year"] == yr) & (df["Quarter"] == qtr) & (df["IsQuarter"] == True)]
            prefill_q = row_q.iloc[0].to_dict() if not row_q.empty else {}

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

            if row_q.shape[0] and st.button(f"Delete {qtr} {yr}", key=f"del_{yr}_{qtr}"):
                df = df[~((df["Name"] == stock_name) & (df["Year"] == yr) & (df["Quarter"] == qtr) & (df["IsQuarter"] == True))]
                io_helpers.save_data(df)
                st.warning(f"Deleted {qtr} {yr}. Please refresh.")
                st.stop()

    # Save all changes (upsert)
    if st.button("💾 Save All Changes"):
        if not stock_name or not industry:
            st.error("Please enter stock name and industry.")
            st.stop()

        # Annual upserts
        for year in annual_data:
            row_up = {"Name": stock_name, "Industry": industry, "Year": year, "IsQuarter": False, "Quarter": ""}
            row_up.update(annual_data[year])
            cond = (df["Name"] == stock_name) & (df["Year"] == year) & (df["IsQuarter"] != True)
            if cond.any():
                df.loc[cond, row_up.keys()] = list(row_up.values())
            else:
                df = pd.concat([df, pd.DataFrame([row_up])], ignore_index=True)

        # Quarterly upserts
        for (year, quarter) in quarterly_data:
            row_uq = {"Name": stock_name, "Industry": industry, "Year": year, "IsQuarter": True, "Quarter": quarter}
            row_uq.update(quarterly_data[(year, quarter)])
            cond = (df["Name"] == stock_name) & (df["Year"] == year) & (df["Quarter"] == quarter) & (df["IsQuarter"] == True)
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

def _empty_editor_frame(all_columns, required_cols):
    """Create an empty DataFrame with at least the required columns."""
    cols = list(dict.fromkeys(required_cols + [c for c in all_columns if c not in required_cols]))
    return pd.DataFrame(columns=cols)

if all_rows.empty:
    st.info("No rows for the current filter.")
else:
    for name in sorted(all_rows["Name"].dropna().unique()):
        st.markdown("---")
        with st.expander(name, expanded=False):
            tabs = st.tabs(["Annual", "Quarterly"])

            # default Industry for this stock (used when adding rows / fixing blanks)
            industry_default_this = _infer_industry_for_stock(df, name, fallback="")

            # ----------------- Annual Tab -----------------
            with tabs[0]:
                av = df[(df["Name"] == name) & (df["IsQuarter"] != True)].sort_values("Year").reset_index(drop=True).copy()

                if av.empty:
                    av = _empty_editor_frame(
                        df.columns.tolist(),
                        required_cols=["Name", "Industry", "Year", "IsQuarter", "Quarter"]
                    )
                    av.loc[:, "Name"] = name
                    av.loc[:, "Industry"] = industry_default_this   # auto-fill (editable)
                    av.loc[:, "Year"] = pd.Series(dtype="Int64")
                    av.loc[:, "IsQuarter"] = False
                    av.loc[:, "Quarter"] = ""

                    av.insert(0, "RowKey", "")
                    av.insert(1, "Delete", False)
                else:
                    av.insert(0, "RowKey", av.apply(lambda r: f"{r['Name']}|{int(r['Year'])}|A", axis=1))
                    av.insert(1, "Delete", False)

                # Keep ONLY base-entry columns (hide EPS, ROE, PE, PB, etc.)
                base_a = ["RowKey", "Delete", "Name", "Industry", "Year"]
                extra_a = [
                    c for c in av.columns
                    if c in ANNUAL_ALLOWED_BASE and c not in {"IsQuarter", "Quarter", "Name", "Industry", "Year"}
                ]
                allowed_a = base_a + extra_a
                av_display = av[[c for c in allowed_a if c in av.columns]].copy()

                edited_a = st.data_editor(
                    av_display,
                    use_container_width=True,
                    height=360,
                    hide_index=True,
                    num_rows="dynamic",
                    column_order=allowed_a,
                    column_config={
                        "RowKey": st.column_config.TextColumn("RowKey", help="Internal key", disabled=True, width="small"),
                        "Delete": st.column_config.CheckboxColumn("Delete", help="Tick to delete this year"),
                        "Name": st.column_config.TextColumn("Name", disabled=True),
                        "Industry": st.column_config.TextColumn("Industry", help="Optional"),
                        "Year": st.column_config.NumberColumn("Year", format="%d"),
                    },
                    key=f"qeb_annual_{name}",
                )

                if st.button(f"💾 Save Annual for {name}", key=f"qeb_save_a_{name}"):
                    del_keys = set(edited_a.loc[edited_a.get("Delete", False) == True, "RowKey"].tolist()) if not edited_a.empty else set()
                    keep = edited_a[edited_a.get("Delete", False) != True].copy() if not edited_a.empty else edited_a

                    # upserts
                    for _, er in keep.iterrows():
                        if pd.isna(er.get("Year")):
                            continue
                        y = int(er["Year"])
                        if not er.get("RowKey"):  # new row
                            ind_val = str(er.get("Industry") or "").strip() or industry_default_this
                            row_up = {"Name": name, "Industry": ind_val, "Year": y, "IsQuarter": False, "Quarter": ""}
                            for col in ANNUAL_ALLOWED_BASE:
                                if col in ("Name", "Industry", "Year", "IsQuarter", "Quarter"): continue
                                if col in keep.columns:
                                    row_up[col] = er.get(col, None)
                            df = pd.concat([df, pd.DataFrame([row_up])], ignore_index=True)
                        else:  # update
                            s_name, s_year, _ = er["RowKey"].split("|")
                            mask = (df["Name"] == s_name) & (df["Year"] == int(s_year)) & (df["IsQuarter"] != True)
                            if "Industry" in df.columns:
                                ind_val = str(er.get("Industry") or "").strip() or industry_default_this
                                df.loc[mask, "Industry"] = ind_val
                            for c in keep.columns:
                                if c in ("RowKey", "Delete"): continue
                                if c in ANNUAL_ALLOWED_BASE and c in df.columns:
                                    df.loc[mask, c] = er.get(c, None)

                    # deletions
                    for key_del in del_keys:
                        s_name, s_year, _ = key_del.split("|")
                        mask = (df["Name"] == s_name) & (df["Year"] == int(s_year)) & (df["IsQuarter"] != True)
                        df.drop(df[mask].index, inplace=True)

                    io_helpers.save_data(df)
                    st.success(f"Saved annual changes for {name}.")
                    st.experimental_rerun()

            # ----------------- Quarterly Tab (autofill + live RowKey + robust save) -----------------
            with tabs[1]:
                quarters = ["—", "Q1", "Q2", "Q3", "Q4"]  # include placeholder to avoid cell errors

                # Load current rows for this stock
                qv = (
                    df[(df["Name"] == name) & (df["IsQuarter"] == True)]
                    .sort_values(["Year", "Quarter"])
                    .reset_index(drop=True)
                    .copy()
                )
                if qv.empty:
                    # start with an empty template that already contains Name/Industry defaults
                    qv = _empty_editor_frame(
                        df.columns.tolist(),
                        required_cols=["Name", "Industry", "Year", "Quarter", "IsQuarter"]
                    )
                    qv.loc[:, "Name"] = name
                    qv.loc[:, "Industry"] = industry_default_this
                    qv.loc[:, "Year"] = pd.Series(dtype="Int64")
                    qv.loc[:, "Quarter"] = "—"
                    qv.loc[:, "IsQuarter"] = True
                else:
                    qv["Quarter"] = qv["Quarter"].astype(str).where(qv["Quarter"].isin(quarters[1:]), "—")

                # Build display frame (hide ratios)
                base_q = ["RowKey", "Delete", "Name", "Industry", "Year", "Quarter"]
                extra_q = [
                    c for c in qv.columns
                    if c in QUARTERLY_ALLOWED_BASE and c not in {"IsQuarter", "Name", "Industry", "Year", "Quarter"}
                ]
                allowed_q = base_q + extra_q

                # Create display df with required cols first
                qv_display = pd.DataFrame(columns=[c for c in allowed_q if c not in ("RowKey", "Delete")])
                for c in qv_display.columns:
                    if c in qv.columns:
                        qv_display[c] = qv[c]
                qv_display.insert(0, "RowKey", "")
                qv_display.insert(1, "Delete", False)

                # Session-state buffer so +Add row is prefilled & normalized
                state_key = f"qeb_quarter_{name}_df"

                def _normalize_quarter_editor(df_work: pd.DataFrame) -> pd.DataFrame:
                    """Fill Name/Industry, normalize Quarter & Year, compute RowKey live."""
                    if df_work is None or df_work.empty:
                        return df_work
                    df_work = df_work.copy()

                    # force Name = current stock
                    if "Name" in df_work.columns:
                        df_work["Name"] = name

                    # fill blank/None Industry with default (treat None/'None'/NaN/'nan'/'')
                    if "Industry" in df_work.columns:
                        s = df_work["Industry"].astype(str).str.strip()
                        s = s.replace({"None": "", "none": "", "NaN": "", "nan": ""})
                        s = s.where(s != "", industry_default_this)
                        df_work["Industry"] = s

                    # normalize Quarter values
                    if "Quarter" in df_work.columns:
                        q = df_work["Quarter"].astype(str).str.strip().str.upper()
                        q = q.where(q.isin(quarters[1:]), "—")
                        df_work["Quarter"] = q

                    # coerce Year to numeric (Int64) for validation
                    if "Year" in df_work.columns:
                        df_work["Year"] = pd.to_numeric(df_work["Year"], errors="coerce").astype("Int64")

                    # live RowKey when valid Year & Quarter
                    if "RowKey" in df_work.columns:
                        def _make_key(r):
                            try:
                                y = int(r.get("Year")) if pd.notna(r.get("Year")) else None
                                q = str(r.get("Quarter"))
                                if y is not None and q in {"Q1", "Q2", "Q3", "Q4"}:
                                    return f"{name}|{y}|{q}|Q"
                                return ""
                            except Exception:
                                return ""
                        df_work["RowKey"] = df_work.apply(_make_key, axis=1)

                    # ensure Delete exists
                    if "Delete" not in df_work.columns:
                        df_work.insert(1, "Delete", False)

                    return df_work

                # initialize/refresh buffer BEFORE editor so new rows are prefilled
                if state_key not in st.session_state:
                    st.session_state[state_key] = _normalize_quarter_editor(qv_display)
                else:
                    st.session_state[state_key] = _normalize_quarter_editor(st.session_state[state_key])

                # Editor (DataFrame returned on each run)
                edited_q = st.data_editor(
                    st.session_state[state_key],
                    use_container_width=True,
                    height=380,
                    hide_index=True,
                    num_rows="dynamic",
                    column_order=allowed_q,
                    column_config={
                        "RowKey": st.column_config.TextColumn("RowKey", help="Auto = Name|Year|Quarter|Q", disabled=True, width="large"),
                        "Delete": st.column_config.CheckboxColumn("Delete", help="Tick to delete this period"),
                        "Name": st.column_config.TextColumn("Name", disabled=True),
                        "Industry": st.column_config.TextColumn("Industry", help="Auto-filled; you can change"),
                        "Year": st.column_config.NumberColumn("Year", format="%d"),
                        "Quarter": st.column_config.SelectboxColumn("Quarter", options=quarters),
                    },
                    key=f"qeb_quarter_{name}",
                )

                # keep normalized value for next rerun (so Name/Industry/RowKey appear)
                st.session_state[state_key] = _normalize_quarter_editor(edited_q)

                # -------- Save Quarterly (robust: coerce Year, normalize Quarter) --------
                if st.button(f"💾 Save Quarterly for {name}", key=f"qeb_save_q_{name}"):
                    buf = _normalize_quarter_editor(st.session_state[state_key])  # normalize once more
                    if buf is None or buf.empty:
                        st.info("Nothing to save.")
                    else:
                        # consider only rows with valid Year & Quarter
                        valid = buf[(buf["Year"].notna()) & (buf["Quarter"].isin(["Q1", "Q2", "Q3", "Q4"]))]

                        to_delete = valid[valid.get("Delete", False) == True]
                        to_upsert = valid[valid.get("Delete", False) != True]

                        if to_delete.empty and to_upsert.empty:
                            st.info("Nothing to save.")
                        else:
                            # deletions
                            for _, er in to_delete.iterrows():
                                mask = (df["Name"] == name) & (df["IsQuarter"] == True) & \
                                       (df["Year"] == int(er["Year"])) & (df["Quarter"] == str(er["Quarter"]))
                                df.drop(df[mask].index, inplace=True)

                            # upserts (insert or update)
                            for _, er in to_upsert.iterrows():
                                y = int(er["Year"]); qt = str(er["Quarter"])
                                row_uq = {
                                    "Name": name,
                                    "Industry": (str(er.get("Industry") or "").strip() or industry_default_this),
                                    "Year": y,
                                    "IsQuarter": True,
                                    "Quarter": qt,
                                }
                                for col in QUARTERLY_ALLOWED_BASE:
                                    if col in ("Name", "Industry", "Year", "IsQuarter", "Quarter"):
                                        continue
                                    if col in to_upsert.columns:
                                        row_uq[col] = er.get(col, None)

                                mask = (df["Name"] == name) & (df["IsQuarter"] == True) & (df["Year"] == y) & (df["Quarter"] == qt)
                                if mask.any():
                                    df.loc[mask, row_uq.keys()] = list(row_uq.values())
                                else:
                                    df = pd.concat([df, pd.DataFrame([row_uq])], ignore_index=True)

                            io_helpers.save_data(df)
                            st.success(f"Saved quarterly changes for {name}.")
                            # clear buffer so it reloads fresh from disk next time
                            st.session_state.pop(state_key, None)
                            st.experimental_rerun()

