import streamlit as st
import pandas as pd
import numpy as np
from utils import io_helpers, calculations

# ---------- Force Wide Layout on Page ----------
st.set_page_config(layout="wide")

# ---------- Page CSS (larger font / clean background) ----------
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

if "IsQuarter" not in df.columns:
    df["IsQuarter"] = False

# Work with annual rows only for summary + latest prices
annual = df[df["IsQuarter"] != True].copy()

# --- FILTERS: Industry, Search ---
colf1, colf2 = st.columns([1, 2])
with colf1:
    industries = ["All"] + sorted([x for x in annual["Industry"].dropna().unique()])
    industry_sel = st.selectbox("Filter by Industry", industries, index=0)
with colf2:
    search_text = st.text_input("🔍 Search Stock Name or Industry", placeholder="Type to filter...")

view = annual if industry_sel == "All" else annual[annual["Industry"] == industry_sel]
if search_text.strip():
    q = search_text.strip().lower()
    view = view[
        view["Name"].str.lower().str.contains(q, na=False) |
        view["Industry"].str.lower().str.contains(q, na=False)
    ]

# ===============================================================
#  A) SUMMARY (one row per stock — latest year only)
# ===============================================================
st.subheader("📌 Latest-year Summary (one row per stock)")

if view.empty:
    st.info("No rows to show for the current filter.")
else:
    latest = view.sort_values(["Name", "Year"]).groupby("Name", as_index=False).tail(1).copy()

    rows = []
    for _, r in latest.iterrows():
        ratio = calculations.calc_ratios(r)
        rows.append({
            "Stock": r["Name"],
            "Industry": r.get("Industry", None),
            "Year": r.get("Year", None),
            "Price": r.get("SharePrice", None),
            "Revenue": ratio.get("Revenue"),
            "NetProfit": ratio.get("NetProfit"),
            "EPS": ratio.get("EPS"),
            "ROE (%)": ratio.get("ROE (%)"),
            "P/E": ratio.get("P/E"),
            "P/B": ratio.get("P/B"),
            "Net Profit Margin (%)": ratio.get("Net Profit Margin (%)"),
            "Dividend Yield (%)": ratio.get("Dividend Yield (%)"),
        })
    summary = pd.DataFrame(rows).sort_values(["Stock"]).reset_index(drop=True)
    st.dataframe(summary, use_container_width=True, height=320)

# ===============================================================
#  B) 🎯 Watchlist & Target Price
# ===============================================================
st.subheader("🎯 Watchlist & Target Price")

# Helper: latest price per stock
latest_prices = (
    annual.sort_values(["Name", "Year"])
    .groupby("Name", as_index=False)
    .tail(1)[["Name", "SharePrice"]]
    .rename(columns={"SharePrice": "CurrentPrice"})
    .set_index("Name")
)

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
