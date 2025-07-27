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

# --- FIELD DEFINITIONS (data-entry only; ratios are NOT in quick-edit) ---
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
# Annual "Other" (current price is per‑stock, so not here)
OTHER_DATA_FIELDS = [
    ("Dividend pay cent", "Dividend"),
    ("End of year share price", "SharePrice"),
]

# Quarterly fields (no Q_SharePrice; current price is per stock)
QUARTERLY_FIELDS = [
    ("Quarterly Net Profit", "Q_NetProfit"),
    ("Quarterly Gross Profit", "Q_GrossProfit"),
    ("Quarterly Revenue", "Q_Revenue"),
    ("Quarterly Cost Of Sales", "Q_CostOfSales"),
    ("Quarterly Finance Costs", "Q_FinanceCosts"),
    ("Quarterly Administrative Expenses", "Q_AdminExpenses"),
    ("Quarterly Selling & Distribution Expenses", "Q_SellDistExpenses"),
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
    ("Each end per every quarter price", "Q_EndQuarterPrice"),
]

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

# -------------- Stock settings (per‑stock meta) --------------
stock_name = st.text_input("Stock Name")
industry = st.text_input("Industry")

if stock_name:
    mask_stock = df["Name"] == stock_name
    # Prefer CurrentPrice if exists, otherwise fall back to legacy Price/SharePrice
    current_price_default = 0.0
    if "CurrentPrice" in df.columns and df.loc[mask_stock, "CurrentPrice"].notna().any():
        current_price_default = float(df.loc[mask_stock, "CurrentPrice"].dropna().iloc[0])
    elif "Price" in df.columns and df.loc[mask_stock, "Price"].notna().any():
        current_price_default = float(df.loc[mask_stock, "Price"].dropna().iloc[0])
    elif "SharePrice" in df.columns and df.loc[mask_stock, "SharePrice"].notna().any():
        current_price_default = float(df.loc[mask_stock, "SharePrice"].dropna().iloc[-1])

    st.subheader("Stock settings")
    cp = st.number_input("Current Price (per stock, used by TTM / ratios)", value=float(current_price_default), step=0.0001, format="%.4f", key="cur_price_stock")

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("💾 Save stock settings (sync price)", key="save_stock_meta"):
            # Ensure columns exist
            for col in ("CurrentPrice", "Price"):
                if col not in df.columns:
                    df[col] = pd.NA

            # Update all rows for this stock
            df.loc[mask_stock, "CurrentPrice"] = float(cp)
            df.loc[mask_stock, "Price"] = float(cp)  # keep compatibility

            # Optionally align Industry for blanks
            if industry:
                if "Industry" not in df.columns:
                    df["Industry"] = ""
                blank_ind = df["Industry"].astype(str).str.strip().isin(["", "None", "nan"])
                df.loc[mask_stock & blank_ind, "Industry"] = industry

            io_helpers.save_data(df)
            st.success("Stock settings saved and synced. All rows updated with current price.")

# ==================== TOP FORMS IN TABS ====================
if stock_name:
    tabs_top = st.tabs(["Annual Form", "Quarterly Form"])

    # ------------------ Annual Form Tab ------------------
    with tabs_top[0]:
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

                if row.shape[0] and st.button(f"Delete Year {year}", key=f"del_year_{year}"):
                    df = df[~((df["Name"] == stock_name) & (df["Year"] == year) & (df["IsQuarter"] != True))]
                    io_helpers.save_data(df)
                    st.warning(f"Deleted year {year}. Please refresh.")
                    st.stop()

        if st.button("💾 Save All Annual Changes"):
            if not stock_name or not industry:
                st.error("Please enter stock name and industry.")
                st.stop()
            for year in annual_data:
                row_up = {"Name": stock_name, "Industry": industry, "Year": year, "IsQuarter": False, "Quarter": ""}
                row_up.update(annual_data[year])
                cond = (df["Name"] == stock_name) & (df["Year"] == year) & (df["IsQuarter"] != True)
                if cond.any():
                    df.loc[cond, row_up.keys()] = list(row_up.values())
                else:
                    df = pd.concat([df, pd.DataFrame([row_up])], ignore_index=True)
            io_helpers.save_data(df)
            st.success("Saved annual changes.")

    # ------------------ Quarterly Form Tab ------------------
    with tabs_top[1]:
        st.subheader("Quarterly Financial Data")
        all_quarters = ["Q1", "Q2", "Q3", "Q4"]

        st.markdown("**Edit / Add a quarter**")

        existing_years = sorted(set(df.loc[df["Name"] == stock_name, "Year"].dropna().astype(int).tolist()))
        wide_years = list(range(2000, 2036))
        year_options = sorted(set(existing_years + wide_years))
        default_year = max(existing_years) if existing_years else 2023

        ca, cb = st.columns([1, 1])
        with ca:
            sel_year = st.selectbox("Year", options=year_options, index=year_options.index(default_year), key="q_form_year")
        with cb:
            sel_quarter = st.selectbox("Quarter", options=all_quarters, index=3, key="q_form_quarter")

        row_q = df[
            (df["Name"] == stock_name) &
            (df["IsQuarter"] == True) &
            (df["Year"] == int(sel_year)) &
            (df["Quarter"] == sel_quarter)
        ]
        prefill_q = row_q.iloc[0].to_dict() if not row_q.empty else {}

        if st.button("Reset all fields to 0 for this quarter", key=f"reset_{sel_year}_{sel_quarter}_q"):
            for _, key in QUARTERLY_FIELDS:
                st.session_state[f"{key}_{sel_year}_{sel_quarter}_q"] = 0.0
            st.experimental_rerun()

        st.markdown("##### Quarterly Income Statement")
        for label, key in QUARTERLY_FIELDS[:7]:
            wkey = f"{key}_{sel_year}_{sel_quarter}_q"
            val = st.session_state.get(wkey, float(prefill_q.get(key, 0.0) or 0.0))
            st.number_input(label, value=val, key=wkey, step=1.0, format="%.4f")

        st.markdown("##### Quarterly Balance Sheet")
        for label, key in QUARTERLY_FIELDS[7:20]:
            wkey = f"{key}_{sel_year}_{sel_quarter}_q"
            val = st.session_state.get(wkey, float(prefill_q.get(key, 0.0) or 0.0))
            st.number_input(label, value=val, key=wkey, step=1.0, format="%.4f")

        st.markdown("##### Quarterly Other Data")
        for label, key in QUARTERLY_FIELDS[20:]:
            wkey = f"{key}_{sel_year}_{sel_quarter}_q"
            val = st.session_state.get(wkey, float(prefill_q.get(key, 0.0) or 0.0))
            st.number_input(label, value=val, key=wkey, step=0.0001, format="%.4f")

        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            if st.button("💾 Save this quarter", key=f"save_{sel_year}_{sel_quarter}_q"):
                new_row = {"Name": stock_name, "Industry": industry, "Year": int(sel_year), "IsQuarter": True, "Quarter": sel_quarter}
                for _, k in QUARTERLY_FIELDS:
                    new_row[k] = float(st.session_state.get(f"{k}_{sel_year}_{sel_quarter}_q", prefill_q.get(k, 0.0)) or 0.0)
                cond = (
                    (df["Name"] == stock_name) &
                    (df["IsQuarter"] == True) &
                    (df["Year"] == int(sel_year)) &
                    (df["Quarter"] == sel_quarter)
                )
                if cond.any():
                    df.loc[cond, new_row.keys()] = list(new_row.values())
                else:
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                io_helpers.save_data(df)
                st.success(f"Saved {sel_quarter} {sel_year} for {stock_name}.")

        with c2:
            if st.button("🗑️ Delete this quarter", key=f"delete_{sel_year}_{sel_quarter}_q"):
                cond = (
                    (df["Name"] == stock_name) &
                    (df["IsQuarter"] == True) &
                    (df["Year"] == int(sel_year)) &
                    (df["Quarter"] == sel_quarter)
                )
                if cond.any():
                    df.drop(df[cond].index, inplace=True)
                    io_helpers.save_data(df)
                    st.warning(f"Deleted {sel_quarter} {sel_year} for {stock_name}.")
                else:
                    st.info("No row to delete for this selection.")
else:
    st.info("Tip: enter a Stock Name above to add/edit annual & quarterly values. You can still use the quick editors below.")

# =================================================================
# QUICK EDIT BY STOCK (Annual & Quarterly)  — with Current Price
# =================================================================
st.divider()
st.subheader("🛠 Quick Edit by Stock (Annual & Quarterly)")

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
    cols = list(dict.fromkeys(required_cols + [c for c in all_columns if c not in required_cols]))
    return pd.DataFrame(columns=cols)

if all_rows.empty:
    st.info("No rows for the current filter.")
else:
    for name in sorted(all_rows["Name"].dropna().unique()):
        st.markdown("---")
        with st.expander(name, expanded=False):

            # ---- per‑stock Current Price right under the title
            mask_name = df["Name"] == name
            cur_default = 0.0
            if "CurrentPrice" in df.columns and df.loc[mask_name, "CurrentPrice"].notna().any():
                cur_default = float(df.loc[mask_name, "CurrentPrice"].dropna().iloc[0])
            elif df.loc[mask_name, "SharePrice"].notna().any():
                cur_default = float(df.loc[mask_name, "SharePrice"].dropna().iloc[-1])

            colcp1, colcp2 = st.columns([1, 1])
            with colcp1:
                cur_price_edit = st.number_input(
                    "Current Price (this stock)",
                    value=float(cur_default),
                    step=0.0001, format="%.4f",
                    key=f"cur_price_quick_{name}"
                )
            with colcp2:
                if st.button("💾 Save current price", key=f"save_cur_price_{name}"):
                    for col in ("CurrentPrice", "Price"):
                        if col not in df.columns:
                            df[col] = pd.NA
                    df.loc[mask_name, "CurrentPrice"] = float(cur_price_edit)
                    df.loc[mask_name, "Price"] = float(cur_price_edit)
                    io_helpers.save_data(df)
                    st.success("Current price saved.")

            tabs = st.tabs(["Annual", "Quarterly"])
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
                    av.loc[:, "Industry"] = industry_default_this
                    av.loc[:, "Year"] = pd.Series(dtype="Int64")
                    av.loc[:, "IsQuarter"] = False
                    av.loc[:, "Quarter"] = ""

                    av.insert(0, "RowKey", "")
                    av.insert(1, "Delete", False)
                else:
                    av.insert(0, "RowKey", av.apply(lambda r: f"{r['Name']}|{int(r['Year'])}|A", axis=1))
                    av.insert(1, "Delete", False)

                # Move RowKey to the END to avoid focus jump
                base_a = ["Delete", "Name", "Industry", "Year"]
                extra_a = [
                    c for c in av.columns
                    if c in ANNUAL_ALLOWED_BASE and c not in {"IsQuarter", "Quarter", "Name", "Industry", "Year"}
                ]
                allowed_a = base_a + extra_a + ["RowKey"]
                av_display = av[[c for c in allowed_a if c in av.columns]].copy()

                edited_a = st.data_editor(
                    av_display,
                    use_container_width=True,
                    height=360,
                    hide_index=True,
                    num_rows="dynamic",
                    column_order=allowed_a,
                    column_config={
                        "RowKey":  st.column_config.TextColumn("RowKey", help="Internal key", disabled=True),
                        "Delete":  st.column_config.CheckboxColumn("Delete", help="Tick to delete this year"),
                        "Name":    st.column_config.TextColumn("Name", disabled=True),
                        "Industry":st.column_config.TextColumn("Industry", help="Optional"),
                        "Year":    st.column_config.NumberColumn("Year", format="%d"),
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

            # ----------------- Quarterly Tab (buffered; RowKey last) -----------------
            with tabs[1]:
                quarters = ["—", "Q1", "Q2", "Q3", "Q4"]

                qv = (
                    df[(df["Name"] == name) & (df["IsQuarter"] == True)]
                    .sort_values(["Year", "Quarter"])
                    .reset_index(drop=True)
                    .copy()
                )
                if qv.empty:
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
                    qv["Quarter"] = qv["Quarter"].astype(str).str.strip().str.upper()
                    qv["Quarter"] = qv["Quarter"].where(qv["Quarter"].isin(quarters[1:]), "—")

                # Move RowKey to the END to avoid focus jump
                base_q = ["Delete", "Name", "Industry", "Year", "Quarter"]
                extra_q = [
                    c for c in qv.columns
                    if c in QUARTERLY_ALLOWED_BASE and c not in {"IsQuarter", "Name", "Industry", "Year", "Quarter"}
                ]
                allowed_q = base_q + extra_q + ["RowKey"]

                # ---------- SESSION BUFFER ----------
                state_key = f"qeb_quarter_{name}_buf"
                if state_key not in st.session_state:
                    buf = qv[[c for c in allowed_q if c not in ("RowKey", "Delete")]].copy()
                    st.session_state[state_key] = buf
                else:
                    buf = st.session_state[state_key]

                # ---------- DISPLAY COPY (compute RowKey only for UI) ----------
                disp = buf.copy()
                if "Name" in disp.columns:
                    disp["Name"] = name
                if "Industry" in disp.columns:
                    ind = disp["Industry"].astype("string").str.strip()
                    ind = ind.replace({"None": "", "none": "", "NaN": "", "nan": ""})
                    disp["Industry"] = ind.where(ind != "", industry_default_this)
                if "Quarter" in disp.columns:
                    disp["Quarter"] = disp["Quarter"].astype("string").str.strip().str.upper()
                    disp["Quarter"] = disp["Quarter"].where(disp["Quarter"].isin(quarters[1:]), "—")
                if "Year" in disp.columns:
                    disp["Year"] = pd.to_numeric(disp["Year"], errors="coerce").astype("Int64")

                disp["RowKey"] = disp.apply(
                    lambda r: f"{name}|{int(r['Year'])}|{r['Quarter']}|Q"
                    if pd.notna(r.get("Year")) and r.get("Quarter") in {"Q1","Q2","Q3","Q4"} else "",
                    axis=1
                ).astype("string")
                disp.insert(0, "Delete", False)

                edited_q = st.data_editor(
                    disp,
                    use_container_width=True,
                    height=380,
                    hide_index=True,
                    num_rows="dynamic",
                    column_order=allowed_q,
                    column_config={
                        "RowKey":  st.column_config.TextColumn("RowKey", help="Auto = Name|Year|Quarter|Q", disabled=True, width="large"),
                        "Delete":  st.column_config.CheckboxColumn("Delete", help="Tick to delete this period"),
                        "Name":    st.column_config.TextColumn("Name", disabled=True),
                        "Industry":st.column_config.TextColumn("Industry", help="Auto-filled; you can change"),
                        "Year":    st.column_config.NumberColumn("Year", format="%d"),
                        "Quarter": st.column_config.SelectboxColumn("Quarter", options=quarters),
                    },
                    key=f"qeb_quarter_{name}",
                )

                # Update buffer with user edits (drop UI-only columns)
                st.session_state[state_key] = edited_q.drop(columns=["RowKey", "Delete"], errors="ignore")

                # ---------- Save ----------
                def _normalise_for_save(df_work: pd.DataFrame) -> pd.DataFrame:
                    if df_work is None or df_work.empty:
                        return df_work
                    out = df_work.copy()
                    out["Name"] = name
                    out["Industry"] = (
                        out["Industry"].astype("string").str.strip()
                        .replace({"None": "", "none": "", "NaN": "", "nan": ""})
                        .where(lambda s: s != "", industry_default_this)
                    )
                    if "Quarter" in out.columns:
                        out["Quarter"] = out["Quarter"].astype("string").str.strip().str.upper()
                    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
                    return out

                if st.button(f"💾 Save Quarterly for {name}", key=f"qeb_save_q_{name}"):
                    buf = _normalise_for_save(st.session_state[state_key])
                    if buf is None or buf.empty:
                        st.info("Nothing to save.")
                    else:
                        valid = buf[(buf["Year"].notna()) & (buf["Quarter"].isin(["Q1","Q2","Q3","Q4"]))]

                        to_del = valid[valid.get("Delete", False) == True] if "Delete" in valid.columns else valid.iloc[0:0]
                        to_up  = valid[valid.get("Delete", False) != True] if "Delete" in valid.columns else valid

                        for _, r in to_del.iterrows():
                            mask = (df["Name"] == name) & (df["IsQuarter"] == True) & \
                                   (df["Year"] == int(r["Year"])) & (df["Quarter"] == str(r["Quarter"]))
                            df.drop(df[mask].index, inplace=True)

                        for _, r in to_up.iterrows():
                            y = int(r["Year"]); q = str(r["Quarter"])
                            row = {"Name": name, "Industry": (str(r.get("Industry") or "").strip() or industry_default_this),
                                   "Year": y, "IsQuarter": True, "Quarter": q}
                            for col in QUARTERLY_ALLOWED_BASE:
                                if col in ("Name","Industry","Year","IsQuarter","Quarter"):
                                    continue
                                if col in to_up.columns:
                                    row[col] = r.get(col, None)
                            mask = (df["Name"] == name) & (df["IsQuarter"] == True) & (df["Year"] == y) & (df["Quarter"] == q)
                            if mask.any():
                                df.loc[mask, row.keys()] = list(row.values())
                            else:
                                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

                        io_helpers.save_data(df)
                        st.success(f"Saved quarterly changes for {name}.")
                        st.experimental_rerun()
