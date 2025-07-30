import streamlit as st
import pandas as pd
import numpy as np
import os
import streamlit.components.v1 as components
from datetime import datetime, timedelta
from utils import io_helpers, calculations

# ---------- Force Wide Layout on Page ----------
st.set_page_config(layout="wide")

# ---------- Page CSS ----------
BASE_CSS = """
<style>
html, body, [class*="css"] { font-size: 17px !important; }
h1, h2, h3, h4 { color:#0f172a !important; font-weight:800 !important; letter-spacing:.2px; }
.stApp { background: radial-gradient(1100px 600px at 18% -10%, #f7fbff 0%, #eef4ff 45%, #ffffff 100%); }
.stDataEditor, .stDataFrame { font-size: 16px !important; }
.stButton>button { border-radius: 12px; padding: .6rem 1.1rem; font-weight: 700; }
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

st.header("📊 Dashboard")

# --- Load Data ---
df = io_helpers.load_data()
if df is None or df.empty or "Name" not in df.columns:
    st.warning("No stock data found. Please add data in 'Add or Edit'.")
    st.stop()

# Ensure proper types
if "IsQuarter" not in df.columns:
    df["IsQuarter"] = False
df["IsQuarter"] = df["IsQuarter"].astype(bool)

# Get data file last modified timestamp
DATA_PATH = io_helpers.DATA_PATH            # ← from utils/io_helpers.py
if os.path.exists(DATA_PATH):
    file_ts  = os.path.getmtime(DATA_PATH)
    last_mod = datetime.fromtimestamp(file_ts)
else:
    last_mod = None

# Work with annual rows only for summary
annual = df[~df["IsQuarter"]].copy()

# --- FILTERS: Industry, Search, Date ---
colf1, colf2, colf3 = st.columns([1, 2, 1])
with colf1:
    industries = ["All"] + sorted(annual["Industry"].dropna().unique())
    industry_sel = st.selectbox("Filter by Industry", industries, index=0)
with colf2:
    search_text = st.text_input("🔍 Search Stock Name or Industry", placeholder="Type to filter...")
with colf3:
    date_opts = ["Any", "> 7 days", "> 14 days", "> 1 month", "> 3 months"]
    date_sel = st.selectbox("Filter by last update", date_opts, index=0)

# Apply filters
df_view = annual.copy()
if industry_sel != "All":
    df_view = df_view[df_view["Industry"] == industry_sel]
if search_text.strip():
    q = search_text.lower()
    df_view = df_view[
        df_view["Name"].str.lower().str.contains(q, na=False) |
        df_view["Industry"].str.lower().str.contains(q, na=False)
    ]
if last_mod and date_sel != "Any":
    age = datetime.now() - last_mod
    if date_sel == "> 7 days":
        thresh = timedelta(days=7)
    elif date_sel == "> 14 days":
        thresh = timedelta(days=14)
    elif date_sel == "> 1 month":
        thresh = timedelta(days=30)
    else:
        thresh = timedelta(days=90)
    if age <= thresh:
        df_view = df_view.iloc[0:0]  # no rows pass

# ===============================================================
#  A) SUMMARY (one row per stock — latest year only)
# ===============================================================
st.subheader("📌 Latest-year Summary")
if df_view.empty:
    st.info("No rows to show for the current filter.")
else:
    latest = (
        df_view
        .sort_values(["Name", "Year"])
        .groupby("Name", as_index=False)
        .tail(1)
    )
    rows = []
    for _, r in latest.iterrows():
        ratio = calculations.calc_ratios(r)
        cur   = r.get("CurrentPrice", np.nan)
        if pd.isna(cur):
            cur = r.get("SharePrice", np.nan)
        rows.append({
            "Stock": r["Name"],
            "Industry": r["Industry"],
            "Year": r["Year"],
            "Current Price": cur,
            "Revenue": ratio.get("Revenue"),
            "NetProfit": ratio.get("NetProfit"),
            "EPS": ratio.get("EPS"),
            "ROE (%)": ratio.get("ROE (%)"),
            "P/E": ratio.get("P/E"),
            "P/B": ratio.get("P/B"),
            "Net Profit Margin (%)": ratio.get("Net Profit Margin (%)"),
            "Dividend Yield (%)": ratio.get("Dividend Yield (%)"),
            "LastModified": r.get("LastModified", "N/A")
        })
    summary = pd.DataFrame(rows).sort_values("Stock").reset_index(drop=True)
    # render centered header and plain table (no Edit column)
        # centered heading
    st.markdown(
        "<h4 style='text-align:center'>📌 Latest-year Summary</h4>",
        unsafe_allow_html=True
    )
    # Streamlit’s native DataFrame style
    st.dataframe(summary, use_container_width=True, height=320)



# ===============================================================
#  📊 Latest-quarter Summary (one row per stock)
# ===============================================================
# centered heading
st.markdown(
    "<h4 style='text-align:center'>📌 Latest-quarter Summary</h4>",
    unsafe_allow_html=True
)

quarterly = df[df["IsQuarter"]].copy()
if industry_sel != "All":
    quarterly = quarterly[quarterly["Industry"] == industry_sel]
if search_text.strip():
    lowq = search_text.lower()
    quarterly = quarterly[
        quarterly["Name"].str.lower().str.contains(lowq, na=False) |
        quarterly["Industry"].str.lower().str.contains(lowq, na=False)
    ]

latest_q = (
    quarterly
    .sort_values(["Name", "Year", "Quarter"])
    .groupby("Name", as_index=False)
    .tail(1)
)

qrows = []
for _, r in latest_q.iterrows():
    ratio = calculations.calc_ratios(r)
    cur = r.get("CurrentPrice", np.nan)
    if pd.isna(cur):
        cur = r.get("Q_EndQuarterPrice", np.nan)
    qrows.append({
        "Stock": r["Name"],
        "Industry": r["Industry"],
        "Year": r["Year"],
        "Quarter": r["Quarter"],
        "Current Price": cur,
        "Revenue": ratio.get("Revenue"),
        "NetProfit": ratio.get("NetProfit"),
        "EPS": ratio.get("EPS"),
        "ROE (%)": ratio.get("ROE (%)"),
        "P/E": ratio.get("P/E"),
        "P/B": ratio.get("P/B"),
        "Net Profit Margin (%)": ratio.get("Net Profit Margin (%)"),
        "Dividend Yield (%)": ratio.get("Dividend Yield (%)"),
        "LastModified": r["LastModified"]  # now per‑row
    })

qsummary = pd.DataFrame(qrows).sort_values("Stock").reset_index(drop=True)
st.dataframe(qsummary, use_container_width=True, height=320)




# ===============================================================
#  B) 🎯 Watchlist & Target Price
# ===============================================================
st.subheader("🎯 Watchlist & Target Price")

# Latest current price per stock: prefer CurrentPrice anywhere in df,
# otherwise fall back to latest annual SharePrice.
pref_current = (
    df.dropna(subset=["CurrentPrice"])
      .sort_values(["Name", "Year"])
      .groupby("Name", as_index=False)
      .tail(1)[["Name", "CurrentPrice"]]
      .set_index("Name")
)

fallback_from_annual = (
    annual.sort_values(["Name", "Year"])
          .groupby("Name", as_index=False)
          .tail(1)[["Name", "SharePrice"]]
          .rename(columns={"SharePrice": "CurrentPrice"})
          .set_index("Name")
)

latest_prices = fallback_from_annual.copy()
for n, row in pref_current.iterrows():
    latest_prices.loc[n, "CurrentPrice"] = row["CurrentPrice"]

# ---- Add / Update form
with st.form("watchlist_add_form", clear_on_submit=True):
    c1, c2 = st.columns([2,1])
    with c1:
        name_input = st.text_input(
            "Stock to watch",
            placeholder="Type a name (matches existing stock names or add custom)",
            help="Type an existing stock name to link the current price, or a custom one."
        )
    with c2:
        target_input = st.number_input("Target price", min_value=0.0, step=0.0001, format="%.4f")
    notes_input = st.text_input("Notes (optional)", placeholder="e.g., buy zone, catalyst, etc.")
    submitted = st.form_submit_button("Add / Update")

if submitted:
    wl = io_helpers.load_watchlist()
    if not name_input:
        st.error("Please enter a stock name for the watchlist.")
    else:
        # Upsert
        exists = wl["Name"].str.lower().eq(name_input.lower()).any()
        if exists:
            wl.loc[wl["Name"].str.lower() == name_input.lower(), ["TargetPrice", "Notes", "Active"]] = [target_input, notes_input, True]
        else:
            wl = pd.concat([
                wl,
                pd.DataFrame([{"Name": name_input, "TargetPrice": target_input, "Notes": notes_input, "Active": True}])
            ], ignore_index=True)
        io_helpers.save_watchlist(wl)
        st.success(f"Watchlist updated for '{name_input}'.")
        st.experimental_rerun()  # show the updated list immediately

# ---- Editor
wl = io_helpers.load_watchlist()
if wl.empty:
    st.info("No items in your watchlist yet. Use the form above to add one.")
else:
    # Enrich with computed cols (ensure proper dtypes for editor)
    disp = wl.copy()
    disp.insert(0, "Delete", False)

    # Ensure text dtype for Notes to avoid Streamlit type mismatch
    disp["Notes"] = disp["Notes"].astype("string").fillna("")

    # Map current price from latest_prices; keep NaN if not available
    disp["CurrentPrice"] = disp["Name"].map(lambda n: latest_prices.loc[n, "CurrentPrice"] if n in latest_prices.index else np.nan)

    # numeric conversions
    disp["TargetPrice"]   = pd.to_numeric(disp["TargetPrice"], errors="coerce")
    disp["CurrentPrice"]  = pd.to_numeric(disp["CurrentPrice"], errors="coerce")

    # upside (avoid division by zero)
    disp["Upside %"] = np.where(
        disp["CurrentPrice"] > 0,
        (disp["TargetPrice"] - disp["CurrentPrice"]) / disp["CurrentPrice"] * 100.0,
        np.nan
    ).round(2)

    edited = st.data_editor(
        disp,
        use_container_width=True,
        height=420,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "Delete":       st.column_config.CheckboxColumn("Delete"),
            "Name":         st.column_config.TextColumn("Name"),
            "TargetPrice":  st.column_config.NumberColumn("Target Price", format="%.4f"),
            "CurrentPrice": st.column_config.NumberColumn("Current Price", disabled=True, format="%.4f"),
            "Upside %":     st.column_config.NumberColumn("Upside %", disabled=True, help="(Target - Current) / Current * 100"),
            "Notes":        st.column_config.TextColumn("Notes"),
            "Active":       st.column_config.CheckboxColumn("Active"),
        },
        key="watchlist_editor",
    )

    if st.button("💾 Save Watchlist"):
        keep = edited[edited["Delete"] != True].copy()
        # Normalize types before saving
        keep["Notes"]       = keep["Notes"].astype("string").fillna("")
        keep["TargetPrice"] = pd.to_numeric(keep["TargetPrice"], errors="coerce")
        keep["Active"]      = keep["Active"].fillna(True).astype(bool)
        keep = keep[["Name","TargetPrice","Notes","Active"]]
        io_helpers.save_watchlist(keep)
        st.success("Watchlist saved.")
        st.experimental_rerun()
