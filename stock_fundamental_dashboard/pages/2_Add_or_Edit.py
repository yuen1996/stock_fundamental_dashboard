# --- make project root importable so we can import io_helpers/calculations/rules ---
import os, sys
_THIS = os.path.dirname(__file__)
_PARENT = os.path.abspath(os.path.join(_THIS, ".."))         # .../stock_fundamental_dashboard
_GRANDP = os.path.abspath(os.path.join(_THIS, "..", ".."))   # repo root (one level above)

for p in (_PARENT, _GRANDP):
    if p not in sys.path:
        sys.path.insert(0, p)

# now these imports will work whether files are in the package root or repo root
try:
    import io_helpers, calculations, rules
except ModuleNotFoundError:
    from utils import io_helpers, calculations, rules  # fallback if you move them under utils/

# add config import (works if config is at root or under utils/)
try:
    import config
except ModuleNotFoundError:
    from utils import config  # type: ignore


import streamlit as st
import pandas as pd
from utils import io_helpers
import config

def comma_number_input(label, value, key, decimals=0):
    """
    A text-input that:
      • shows commas,
      • keeps a numeric value in st.session_state[key],
      • returns the numeric value.
    """
    txt_key = key + "_txt"

    # init both the display and numeric state once
    if txt_key not in st.session_state:
        st.session_state[txt_key] = f"{float(value):,.{decimals}f}"
    if key not in st.session_state:
        try:
            st.session_state[key] = float(value)
        except:
            st.session_state[key] = 0.0

    def _fmt():
        raw = str(st.session_state.get(txt_key, "")).replace(",", "")
        try:
            num = float(raw)
            st.session_state[txt_key] = f"{num:,.{decimals}f}"
            st.session_state[key] = num
        except:
            # leave as-is on parse error
            pass

    txt = st.text_input(label, value=st.session_state[txt_key], key=txt_key, on_change=_fmt)

    # also set numeric on every render
    try:
        num = float(str(txt).replace(",", ""))
    except:
        num = 0.0
    st.session_state[key] = num
    return num



# — Insert these two lines to prefill from Dashboard “Edit” link —
params = st.query_params
default_stock = params.get("stock_name", [""])[0]

# ---------- Force Wide Layout on Page ----------
st.set_page_config(layout="wide")

# === Unified CSS (same as Dashboard; fonts 16px) ===
BASE_CSS = """
<style>
:root{
  --bg:#f6f7fb;               /* app background */
  --surface:#ffffff;          /* cards/tables background */
  --text:#0f172a;             /* main text */
  --muted:#475569;            /* secondary text */
  --border:#e5e7eb;           /* card & table borders */
  --shadow:0 8px 24px rgba(15, 23, 42, .06);

  /* accent colors for section stripes */
  --primary:#4f46e5;          /* indigo */
  --info:#0ea5e9;             /* sky   */
  --success:#10b981;          /* green */
  --warning:#f59e0b;          /* amber */
  --danger:#ef4444;           /* red   */
}
html, body, [class*="css"]{
  font-size:16px !important; color:var(--text);
}
.stApp{
  background: radial-gradient(1000px 500px at 10% -10%, #f0f3fb 0%, var(--bg) 60%), var(--bg);
}
h1, h2, h3, h4{
  color:var(--text) !important; font-weight:800 !important; letter-spacing:.2px;
}

/* Section header card */
.sec{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:14px;
  box-shadow:var(--shadow);
  padding:.65rem .9rem;
  margin:1rem 0 .6rem 0;
  display:flex; align-items:center; gap:.6rem;
}
.sec .t{ font-size:1.05rem; font-weight:800; margin:0; padding:0; }
.sec .d{ color:var(--muted); font-size:.95rem; margin-left:.25rem; }
.sec::before{
  content:""; display:inline-block;
  width:8px; height:26px; border-radius:6px; background:var(--primary);
}
.sec.info::before    { background:var(--info); }
.sec.success::before { background:var(--success); }
.sec.warning::before { background:var(--warning); }
.sec.danger::before  { background:var(--danger); }

/* Tables / editors */
.stDataFrame, .stDataEditor{ font-size:15px !important; }
div[data-testid="stDataFrame"] table, div[data-testid="stDataEditor"] table{
  border-collapse:separate !important; border-spacing:0;
}
div[data-testid="stDataFrame"] table tbody tr:hover td,
div[data-testid="stDataEditor"] table tbody tr:hover td{
  background:#f8fafc !important;
}
div[data-testid="stDataFrame"] td, div[data-testid="stDataEditor"] td{
  border-bottom:1px solid var(--border) !important;
}

/* Inputs */
div[data-baseweb="input"] input, textarea, .stNumberInput input{
  font-size:15px !important;
}
.stSlider > div [data-baseweb="slider"]{ margin-top:.25rem; }

/* Buttons */
.stButton>button{
  border-radius:12px !important; padding:.55rem 1.1rem !important; font-weight:700;
}

/* Tabs */
.stTabs [role="tab"]{ font-size:15px !important; font-weight:600 !important; }

/* Sidebar theme (keeps your dark gradient) */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg, #0b1220 0%, #1f2937 100%) !important;
}
[data-testid="stSidebar"] *{ color:#e5e7eb !important; }

/* Optional: hide Streamlit's default page list
section[data-testid="stSidebarNav"]{ display:none !important; }
*/
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

# --- Streamlit rerun compat (works on old/new versions) ---
def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun() 
        except Exception:
            pass



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

def _infer_bucket_for_stock(df: pd.DataFrame, stock: str, fallback: str = "General") -> str:
    if df is None or df.empty or "IndustryBucket" not in df.columns:
        return fallback
    s = (
        df.loc[df["Name"] == stock, "IndustryBucket"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    if s.empty:
        return fallback
    try:
        m = s.mode()
        chosen = m.iloc[0] if not m.empty else s.iloc[0]
        return chosen if chosen in config.INDUSTRY_BUCKETS else fallback
    except Exception:
        return fallback



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
    ("Cash & Cash Equivalents", "Cash"),
    ("Total Debt / Borrowings", "TotalDebt"),

]
# Annual "Other" (current price is per-stock, so not here)
OTHER_DATA_FIELDS = [
    ("Dividend pay cent", "Dividend"),
    ("End of year share price", "SharePrice"),
]

# --- Cash Flow Statement (Annual) — NEW fields used by radar
ANNUAL_CF_FIELDS = [
    ("Operating Cash Flow", "CFO"),            # Net cash from operating activities
    ("Capital Expenditure", "CapEx"),          # Purchase of PPE — enter as POSITIVE outflow
    ("Income Tax Expense", "IncomeTax"),
    ("Depreciation & Amortization", "DepAmort"),
    ("EBITDA (optional)", "EBITDA"),
]


# Quarterly fields (no Q_SharePrice; current price is per stock)
# Quarterly Income Statement
QUARTERLY_IS_FIELDS = [
    ("Quarterly Net Profit", "Q_NetProfit"),
    ("Quarterly Gross Profit", "Q_GrossProfit"),
    ("Quarterly Revenue", "Q_Revenue"),
    ("Quarterly Cost Of Sales", "Q_CostOfSales"),
    ("Quarterly Finance Costs", "Q_FinanceCosts"),
    ("Quarterly Administrative Expenses", "Q_AdminExpenses"),
    ("Quarterly Selling & Distribution Expenses", "Q_SellDistExpenses"),
]

# Quarterly Balance Sheet
QUARTERLY_BS_FIELDS = [
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
]

# NEW: Quarterly Cash Flow Statement
QUARTERLY_CF_FIELDS = [
    ("Quarterly Operating Cash Flow", "Q_CFO"),                 # net cash FROM operating activities
    ("Quarterly Capital Expenditure", "Q_CapEx"),               # purchase/payment for PPE (enter as positive outflow)
    ("Quarterly Income Tax", "Q_Tax"),                          # tax expense
    ("Quarterly Depreciation & Amortization", "Q_DepAmort"),    # for EBITDA fallback
    ("Quarterly EBITDA (optional)", "Q_EBITDA"),                # if you have it directly
]

# Quarterly Other Data
QUARTERLY_OTHER_FIELDS = [
    ("Each end per every quarter price", "Q_EndQuarterPrice"),
]

# Keep a single combined list so existing save/reset code keeps working
QUARTERLY_FIELDS = QUARTERLY_IS_FIELDS + QUARTERLY_BS_FIELDS + QUARTERLY_CF_FIELDS + QUARTERLY_OTHER_FIELDS

ANNUAL_ALLOWED_BASE = {"Name", "Industry", "IndustryBucket", "Year", "IsQuarter", "Quarter"} \
    | {k for _, k in INCOME_STATEMENT_FIELDS} \
    | {k for _, k in BALANCE_SHEET_FIELDS} \
    | {k for _, k in OTHER_DATA_FIELDS} \
    | {k for _, k in ANNUAL_CF_FIELDS}

QUARTERLY_ALLOWED_BASE = {"Name", "Industry", "IndustryBucket", "Year", "Quarter", "IsQuarter"} \
    | {k for _, k in QUARTERLY_FIELDS}


# ---------- Load data ----------
st.header("➕ Add / Edit Stock (Annual & Quarterly)")
df = io_helpers.load_data()
if df is None:
    df = pd.DataFrame()

# Ensure the new bucket column exists (backward compatible)
if "Industry" not in df.columns:
    df["Industry"] = ""
if "IndustryBucket" not in df.columns:
    df["IndustryBucket"] = ""


# Guard columns (backward compatibility)
if "IsQuarter" not in df.columns:
    df["IsQuarter"] = False
if "Quarter" not in df.columns:
    df["Quarter"] = pd.NA

# -------------- Stock settings (per-stock meta) --------------
# Section card (visual only)
st.markdown(
    '<div class="sec"><div class="t">⚙️ Stock Settings</div>'
    '<div class="d">Per-stock metadata & current price</div></div>',
    unsafe_allow_html=True
)

stock_name = st.text_input("Stock Name", value=default_stock)

# Free-text industry + dropdown bucket (new)
col_i1, col_i2 = st.columns([1, 1])

with col_i1:
    # Free text (kept)
    industry_text_prefill = ""
    if stock_name:
        industry_text_prefill = _infer_industry_for_stock(df, stock_name, fallback="")
    industry = st.text_input("Industry (free text)", value=industry_text_prefill)

with col_i2:
    # Dropdown bucket (new)
    bucket_prefill = "General"
    if stock_name:
        bucket_prefill = _infer_bucket_for_stock(df, stock_name, fallback="General")
    industry_bucket = st.selectbox(
        "Industry Bucket (dropdown)",
        options=list(config.INDUSTRY_BUCKETS),
        index=list(config.INDUSTRY_BUCKETS).index(bucket_prefill)
    )


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
    cp = comma_number_input(
        "Current Price (per stock…)",
        value=float(current_price_default),
        key="cur_price_stock",
        decimals=4
    )

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("💾 Save stock settings (sync price)", key="save_stock_meta"):
            # Ensure columns exist
            for col in ("CurrentPrice", "Price", "Industry", "IndustryBucket"):
                if col not in df.columns:
                    df[col] = pd.NA

            # Update all rows for this stock
            df.loc[mask_stock, "CurrentPrice"] = float(cp)
            df.loc[mask_stock, "Price"] = float(cp)  # keep compatibility

            # Fill Industry free-text for blanks only (preserve existing if user left it empty)
            # Sync Industry & IndustryBucket to ALL rows for this stock (overwrite)
            if industry:
                df.loc[mask_stock, "Industry"] = industry
            if industry_bucket:
                df.loc[mask_stock, "IndustryBucket"] = industry_bucket


            io_helpers.save_data(df)
            st.success("Stock settings saved and synced. All rows updated with current price and industry fields.")


# ==================== TOP FORMS IN TABS ====================
if stock_name:
    tabs_top = st.tabs(["Annual Form", "Quarterly Form"])

    # ------------------ Annual Form Tab ------------------
    with tabs_top[0]:
        # Section card
        st.markdown(
            '<div class="sec info"><div class="t">📅 Annual Financial Data</div>'
            '<div class="d">Income, balance sheet & other (Year-based)</div></div>',
            unsafe_allow_html=True
        )

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
                    _safe_rerun()

                st.markdown("##### Income Statement")
                for label, key in INCOME_STATEMENT_FIELDS:
                    year_data[key] = comma_number_input(
                        label,
                        value=float(prefill.get(key, 0.0) or 0.0),
                        key=f"{key}_{year}_annual",
                        decimals=0
                    )

                st.markdown("##### Balance Sheet")
                for label, key in BALANCE_SHEET_FIELDS:
                    year_data[key] = comma_number_input(
                        label,
                        value=float(prefill.get(key, 0.0) or 0.0),
                        key=f"{key}_{year}_annual",
                        decimals=0
                    )

                st.markdown("##### Other Data")
                for label, key in OTHER_DATA_FIELDS:
                    year_data[key] = comma_number_input(
                        label,
                        value=float(prefill.get(key, 0.0) or 0.0),
                        key=f"{key}_{year}_annual",
                        decimals=4
                    )

                # --- Cash Flow Statement (Annual) — NEW ---
                st.markdown("##### Cash Flow Statement (Annual)")
                for label, key in ANNUAL_CF_FIELDS:
                    year_data[key] = comma_number_input(
                        label,
                        value=float(prefill.get(key, 0.0) or 0.0),
                        key=f"{key}_{year}_annual",
                        decimals=0
                    )

                annual_data[year] = year_data


                if row.shape[0] and st.button(f"Delete Year {year}", key=f"del_year_{year}"):
                    df = df[~((df["Name"] == stock_name) & (df["Year"] == year) & (df["IsQuarter"] != True))]
                    io_helpers.save_data(df)
                    st.warning(f"Deleted year {year}. Please refresh.")
                    st.stop()

if st.button("💾 Save All Annual Changes"):
    if not stock_name or not industry:
        st.error("Please enter stock name and industry (free text).")
        st.stop()
    for year in annual_data:
        row_up = {
            "Name": stock_name,
            "Industry": industry,
            "IndustryBucket": industry_bucket,  # NEW
            "Year": year,
            "IsQuarter": False,
            "Quarter": ""
        }
        row_up.update(annual_data[year])
        cond = (df["Name"] == stock_name) & (df["Year"] == year) & (df["IsQuarter"] != True)
        # upsert
        if cond.any():
            # ensure columns exist
            for c in row_up.keys():
                if c not in df.columns:
                    df[c] = pd.NA
            df.loc[cond, row_up.keys()] = list(row_up.values())
        else:
            df = pd.concat([df, pd.DataFrame([row_up])], ignore_index=True)
    # Also sync the chosen bucket / free-text industry across all rows for this stock
    df.loc[df["Name"] == stock_name, "IndustryBucket"] = industry_bucket
    if industry:
        df.loc[df["Name"] == stock_name, "Industry"] = industry

    io_helpers.save_data(df)
    st.success("Saved annual changes.")



    
        # ------------------ Quarterly Form Tab ------------------
    with tabs_top[1]:
        # Section card
        st.markdown(
            '<div class="sec success"><div class="t">🗓 Quarterly Financial Data</div>'
            '<div class="d">Quarterly inputs (Q1–Q4)</div></div>',
            unsafe_allow_html=True
        )

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
            _safe_rerun()

        # ---------------- Sections (ALL inside the tab) ----------------
        q_vals = {}  # <- collect all quarter field values here

        st.markdown("##### Quarterly Income Statement")
        for label, key in QUARTERLY_IS_FIELDS:
            wkey = f"{key}_{sel_year}_{sel_quarter}_q"
            default = float(prefill_q.get(key, 0.0) or 0.0)
            val = comma_number_input(label, value=st.session_state.get(wkey, default), key=wkey, decimals=0)
            q_vals[key] = val

        st.markdown("##### Quarterly Balance Sheet")
        for label, key in QUARTERLY_BS_FIELDS:
            wkey = f"{key}_{sel_year}_{sel_quarter}_q"
            default = float(prefill_q.get(key, 0.0) or 0.0)
            val = comma_number_input(label, value=st.session_state.get(wkey, default), key=wkey, decimals=0)
            q_vals[key] = val

        st.markdown("##### Quarterly Cash Flow Statement")
        for label, key in QUARTERLY_CF_FIELDS:
            wkey = f"{key}_{sel_year}_{sel_quarter}_q"
            default = float(prefill_q.get(key, 0.0) or 0.0)
            val = comma_number_input(label, value=st.session_state.get(wkey, default), key=wkey, decimals=0)
            q_vals[key] = val

        st.markdown("##### Quarterly Other Data")
        for label, key in QUARTERLY_OTHER_FIELDS:
            wkey = f"{key}_{sel_year}_{sel_quarter}_q"
            default = float(prefill_q.get(key, 0.0) or 0.0)
            decs = 4 if key == "Q_EndQuarterPrice" else 0
            val = comma_number_input(label, value=st.session_state.get(wkey, default), key=wkey, decimals=decs)
            q_vals[key] = val


        # ---------------- Save / Delete (STILL inside the tab) ----------------
        c1, c2, _ = st.columns([1, 1, 2])

        with c1:
            if st.button("💾 Save this quarter", key=f"save_{sel_year}_{sel_quarter}_q"):
                new_row = {
                    "Name": stock_name,
                    "Industry": industry,
                    "IndustryBucket": industry_bucket,
                    "Year": int(sel_year),
                    "IsQuarter": True,
                    "Quarter": sel_quarter,
                    **q_vals,  # <- all captured inputs
                }

                # ensure all columns exist before assignment
                for c in new_row.keys():
                    if c not in df.columns:
                        df[c] = pd.NA

                cond = (
                    (df["Name"] == stock_name) &
                    (df["IsQuarter"] == True) &
                    (df["Year"] == int(sel_year)) &
                    (df["Quarter"] == sel_quarter)
                )

                if cond.any():
                    df.loc[cond, list(new_row.keys())] = list(new_row.values())
                else:
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                # sync industry fields across all rows for this stock
                df.loc[df["Name"] == stock_name, "IndustryBucket"] = industry_bucket
                if industry:
                    df.loc[df["Name"] == stock_name, "Industry"] = industry

                io_helpers.save_data(df)
                st.success(f"Saved {sel_quarter} {sel_year} for {stock_name} and synced industry fields.")



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

# Section card
st.markdown(
    '<div class="sec warning"><div class="t">🛠 Quick Edit by Stock</div>'
    '<div class="d">Fast editing for existing rows</div></div>',
    unsafe_allow_html=True
)

st.subheader("🛠 Quick Edit by Stock (Annual & Quarterly)")

all_rows = df.copy()
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    industries = ["All"] + sorted([x for x in all_rows["Industry"].dropna().unique()])
    f_industry = st.selectbox("Filter by Industry (free text)", industries, index=0, key="qeb_industry")
with c2:
    buckets = ["All"] + list(config.INDUSTRY_BUCKETS)
    f_bucket = st.selectbox("Filter by Industry Bucket", buckets, index=0, key="qeb_bucket")
with c3:
    f_query = st.text_input("🔎 Search name / industry / bucket", key="qeb_search")

if f_industry != "All":
    all_rows = all_rows[all_rows["Industry"] == f_industry]
if f_bucket != "All":
    if "IndustryBucket" not in all_rows.columns:
        all_rows["IndustryBucket"] = ""
    all_rows = all_rows[all_rows["IndustryBucket"] == f_bucket]
if f_query.strip():
    q = f_query.strip().lower()
    # include bucket in search
    if "IndustryBucket" not in all_rows.columns:
        all_rows["IndustryBucket"] = ""
    all_rows = all_rows[
        all_rows["Name"].str.lower().str.contains(q, na=False) |
        all_rows["Industry"].str.lower().str.contains(q, na=False) |
        all_rows["IndustryBucket"].str.lower().str.contains(q, na=False)
    ]


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

            # ---- per-stock Current Price right under the title
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
            bucket_default_this = _infer_bucket_for_stock(df, name, fallback="General")
            industry_default_this = _infer_industry_for_stock(df, name, fallback="")


            # ----------------- Annual Tab -----------------
            with tabs[0]:
                av = df[(df["Name"] == name) & (~df["IsQuarter"].astype(bool))] \
                    .sort_values("Year") \
                    .reset_index(drop=True) \
                    .copy()

                if av.empty:
                    av = _empty_editor_frame(
                        df.columns.tolist(),
                        required_cols=["Name", "Industry", "Year", "IsQuarter", "Quarter"]
                    )
                    av.loc[:, "Name"] = name
                    av.loc[:, "Industry"] = industry_default_this
                    av.loc[:, "IndustryBucket"] = bucket_default_this   # NEW
                    av.loc[:, "Year"] = pd.Series(dtype="Int64")
                    av.loc[:, "IsQuarter"] = False
                    av.loc[:, "Quarter"] = ""


                    av.insert(0, "RowKey", "")
                    av.insert(1, "Delete", False)
                else:
                    av.insert(0, "RowKey", av.apply(lambda r: f"{r['Name']}|{int(r['Year'])}|A", axis=1))
                    av.insert(1, "Delete", False)

                # Ensure all ANNUAL columns exist so they show up in the editor
                for col in (ANNUAL_ALLOWED_BASE - {"Name", "Industry", "Year", "IsQuarter", "Quarter"}):
                    if col not in av.columns:
                        av[col] = pd.NA

                # ✅ ensure IndustryBucket column exists with a sensible default
                if "IndustryBucket" not in av.columns:
                    av["IndustryBucket"] = bucket_default_this


                # Move RowKey to the END to avoid focus jump
                base_a = ["Delete", "Name", "Industry", "IndustryBucket", "Year"]
                extra_a = [
                    c for c in av.columns
                    if c in ANNUAL_ALLOWED_BASE and c not in {"IsQuarter", "Quarter", "Name", "Industry", "IndustryBucket", "Year"}
                ]
                allowed_a = base_a + extra_a + ["RowKey"]
                # de-dup the column order list (keeps first occurrence)
                allowed_a = list(dict.fromkeys(allowed_a))

                av_display = av[[c for c in allowed_a if c in av.columns]].copy()
                # extra safety: remove any duplicate-named columns in the DataFrame
                av_display = av_display.loc[:, ~av_display.columns.duplicated()]


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
                        "Industry": st.column_config.TextColumn("Industry (free text)", help="Optional"),
                        "IndustryBucket": st.column_config.SelectboxColumn(
                            "Industry Bucket",
                            options=list(config.INDUSTRY_BUCKETS),
                            help="Pick a bucket for scoring"
                        ),

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
                            buck_val = str(er.get("IndustryBucket") or "").strip() or bucket_default_this
                            row_up = {
                                "Name": name,
                                "Industry": ind_val,
                                "IndustryBucket": buck_val,      # NEW
                                "Year": y, "IsQuarter": False, "Quarter": ""
                            }

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

                            # ✅ NEW: update the IndustryBucket alongside Industry
                            if "IndustryBucket" in df.columns:
                                buck_val = str(er.get("IndustryBucket") or "").strip() or bucket_default_this
                                df.loc[mask, "IndustryBucket"] = buck_val


                            for c in keep.columns:
                                if c in ("RowKey", "Delete"): continue
                                if c in ANNUAL_ALLOWED_BASE and c in df.columns:
                                    df.loc[mask, c] = er.get(c, None)

                    # deletions
                    for key_del in del_keys:
                        s_name, s_year, _ = key_del.split("|")
                        mask = (df["Name"] == s_name) & (df["Year"] == int(s_year)) & (df["IsQuarter"] != True)
                        df.drop(df[mask].index, inplace=True)

                    # ⬇️ NEW: unify bucket across ALL rows for this stock using the edited value(s)
                    try:
                        nb_series = keep.get("IndustryBucket", pd.Series(dtype="string")).astype("string").str.strip()
                        nb_series = nb_series[nb_series != ""]
                        if not nb_series.empty:
                            new_bucket = nb_series.mode().iloc[0]
                            if new_bucket in config.INDUSTRY_BUCKETS:
                                df.loc[df["Name"] == name, "IndustryBucket"] = new_bucket
                    except Exception:
                        pass    

                    io_helpers.save_data(df)
                    st.success(f"Saved annual changes for {name}.")
                    _safe_rerun()

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
                    qv.loc[:, "IndustryBucket"] = bucket_default_this    # NEW
                    qv.loc[:, "Year"] = pd.Series(dtype="Int64")
                    qv.loc[:, "Quarter"] = "—"
                    qv.loc[:, "IsQuarter"] = True

                else:
                    qv["Quarter"] = qv["Quarter"].astype(str).str.strip().str.upper()
                    qv["Quarter"] = qv["Quarter"].where(qv["Quarter"].isin(quarters[1:]), "—")

                # Ensure all QUARTERLY columns exist so they show up in the editor
                for col in (QUARTERLY_ALLOWED_BASE - {"Name", "Industry", "Year", "Quarter", "IsQuarter"}):
                    if col not in qv.columns:
                        qv[col] = pd.NA

                # ✅ ensure IndustryBucket column exists with a sensible default
                if "IndustryBucket" not in qv.columns:
                    qv["IndustryBucket"] = bucket_default_this


                # Put stable, non-checkbox columns first; move Delete/RowKey to the END
                base_q = ["Name", "Industry", "IndustryBucket", "Year", "Quarter"]
                extra_q = [
                    c for c in qv.columns
                    if c in QUARTERLY_ALLOWED_BASE and c not in {"IsQuarter", "Name", "Industry", "IndustryBucket", "Year", "Quarter"}
                ]
                allowed_q = base_q + extra_q + ["Delete", "RowKey"]
                allowed_q = list(dict.fromkeys(allowed_q))


                # ---------- SESSION BUFFER ----------
                state_key = f"qeb_quarter_{name}_buf"
                if state_key not in st.session_state:
                    buf = qv[[c for c in allowed_q if c not in ("RowKey", "Delete")]].copy()
                    st.session_state[state_key] = buf
                else:
                    buf = st.session_state[state_key]

                # ---------- DISPLAY COPY (compute RowKey only for UI) ----------
                disp = buf.copy()
                disp["Name"] = name
                ind = disp["Industry"].astype("string").str.strip().replace({"None": "", "none": "", "NaN": "", "nan": ""})
                disp["Industry"] = ind.where(ind != "", industry_default_this)

                # ✅ default/clean the IndustryBucket for display & editing
                disp["IndustryBucket"] = (
                    disp.get("IndustryBucket", pd.Series(dtype="string"))
                    .astype("string").str.strip()
                    .where(lambda s: s.notna() & (s != ""), bucket_default_this)
                )

                disp["Quarter"] = disp["Quarter"].astype("string").str.strip().str.upper().where(
                    disp["Quarter"].isin(quarters[1:]), "—"
                )
                disp["Year"] = pd.to_numeric(disp["Year"], errors="coerce").astype("Int64")
                disp["RowKey"] = disp.apply(
                    lambda r: f"{name}|{int(r['Year'])}|{r['Quarter']}|Q"
                    if pd.notna(r["Year"]) and r["Quarter"] in quarters[1:] else "",
                    axis=1
                ).astype("string")

                # ---- Buffer edits in a form: no rerun while typing ----
                with st.form(f"form_quarter_editor_{name}", clear_on_submit=False):
                    disp["Delete"] = False  # UI-only
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
                            "IndustryBucket": st.column_config.SelectboxColumn(
                                    "Industry Bucket",
                                    options=list(config.INDUSTRY_BUCKETS),
                                    help="Bucket for industry scoring",
                                ),
                            "Year":    st.column_config.NumberColumn("Year", format="%d"),
                            "Quarter": st.column_config.SelectboxColumn("Quarter", options=quarters),
                        },
                        key=f"qeb_quarter_{name}",
                    )
                    submit_q = st.form_submit_button(f"💾 Save Quarterly for {name}")

                # After submit, update buffer and write to CSV
                # After submit, update buffer and write to CSV (handle deletions + upserts)
                if submit_q:
                    edited = edited_q.copy()

                    # 1) Deletions from Delete + RowKey (format: Name|Year|Quarter|Q)
                    del_keys = set(
                        edited.loc[edited.get("Delete", False) == True, "RowKey"].dropna().astype(str).tolist()
                    )
                    deleted = 0
                    for key_del in del_keys:
                        try:
                            s_name, s_year, s_quarter, _ = key_del.split("|")
                            mask = (
                                (df["Name"] == s_name) &
                                (df["IsQuarter"] == True) &
                                (df["Year"] == int(s_year)) &
                                (df["Quarter"] == s_quarter)
                            )
                            deleted += int(mask.sum())
                            df.drop(df[mask].index, inplace=True)
                        except Exception:
                            pass

                    # 2) Keep (non-deleted) rows in the session buffer
                    st.session_state[state_key] = edited[edited.get("Delete", False) != True] \
                        .drop(columns=["RowKey", "Delete"], errors="ignore")

                    # 3) Normalise + validate for upserts
                    def _normalise_for_save(df_work: pd.DataFrame) -> pd.DataFrame:
                        out = df_work.copy()
                        out["Name"] = name
                        out["Industry"] = (
                            out["Industry"].astype("string").str.strip()
                            .replace({"None": "", "none": "", "NaN": "", "nan": ""})
                            .where(lambda s: s != "", industry_default_this)
                        )
                        # ✅ normalise IndustryBucket with fallback
                        out["IndustryBucket"] = (
                            out.get("IndustryBucket", pd.Series(dtype="string"))
                            .astype("string").str.strip()
                            .where(lambda s: s.notna() & (s != ""), bucket_default_this)
                        )
				
                        out["Quarter"] = out["Quarter"].astype("string").str.strip().str.upper()
                        out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
                        return out

                    buf_save = _normalise_for_save(st.session_state[state_key])
                    valid = buf_save[(buf_save["Year"].notna()) & (buf_save["Quarter"].isin(quarters[1:]))]

                    # 4) Upserts (insert or update remaining rows)
                    for _, r in valid.iterrows():
                        y, q = int(r["Year"]), r["Quarter"]
                        row = {"Name": name, "Industry": r["Industry"], "IndustryBucket": r["IndustryBucket"],
                            "Year": y, "IsQuarter": True, "Quarter": q}
                        for col in QUARTERLY_ALLOWED_BASE - {"Name", "Industry", "IndustryBucket", "Year", "IsQuarter", "Quarter"}:
                            row[col] = r.get(col, None)

                        mask = (df["Name"] == name) & (df["IsQuarter"] == True) & (df["Year"] == y) & (df["Quarter"] == q)
                        if mask.any():
                            df.loc[mask, row.keys()] = list(row.values())
                        else:
                            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

                    # ⬇️ NEW: unify bucket across ALL rows for this stock using the edited value(s)
                    try:
                        nb_series = edited.get("IndustryBucket", pd.Series(dtype="string")).astype("string").str.strip()
                        nb_series = nb_series[nb_series != ""]
                        if not nb_series.empty:
                            new_bucket = nb_series.mode().iloc[0]
                            if new_bucket in config.INDUSTRY_BUCKETS:
                                df.loc[df["Name"] == name, "IndustryBucket"] = new_bucket
                    except Exception:
                        pass

                    # 5) Save + messages + rerun
                    io_helpers.save_data(df)
                    if deleted:
                        st.warning(f"Deleted {deleted} quarter(s) for {name}.")
                    st.success(f"Saved quarterly changes for {name}.")
                    _safe_rerun()


