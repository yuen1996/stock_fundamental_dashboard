import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils import io_helpers, calculations, rules

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

# Work with annual rows only for summary
annual = df[~df["IsQuarter"]].copy()

# --- FILTERS: Industry, Search, Updated in last ... ---
colf1, colf2, colf3 = st.columns([1, 2, 1])
with colf1:
    industries = ["All"] + sorted(annual["Industry"].dropna().unique())
    industry_sel = st.selectbox("Filter by Industry", industries, index=0)
with colf2:
    search_text = st.text_input("🔍 Search Stock Name or Industry", placeholder="Type to filter...")
with colf3:
    date_opts = ["Any", "Last 7 days", "Last 14 days", "Last 1 month", "Last 3 months"]
    date_sel = st.selectbox("Updated in", date_opts, index=0)

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

# Row-level "updated in last ..." filter using LastModified
if date_sel != "Any" and "LastModified" in df_view.columns:
    now = datetime.now()
    if date_sel == "Last 7 days":
        cutoff = now - timedelta(days=7)
    elif date_sel == "Last 14 days":
        cutoff = now - timedelta(days=14)
    elif date_sel == "Last 1 month":
        cutoff = now - timedelta(days=30)
    else:
        cutoff = now - timedelta(days=90)
    lm = pd.to_datetime(df_view["LastModified"], errors="coerce")
    df_view = df_view[lm >= cutoff]

# ===============================================================
#  A) 📌 Latest-year Summary (one row per stock)
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
    st.dataframe(summary, use_container_width=True, height=320)

# ===============================================================
#  B) 📌 Latest-quarter Summary (one row per stock)
# ===============================================================
st.subheader("📌 Latest-quarter Summary")

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
        "LastModified": r.get("LastModified", "N/A")
    })

qsummary = pd.DataFrame(qrows).sort_values("Stock").reset_index(drop=True)
st.dataframe(qsummary, use_container_width=True, height=320)

# ===============================================================
#  C) 🎯 Watchlist & Target Price
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
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()  # for older Streamlit

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
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

# ===============================================================
#  D) 🧭 Trade Readiness (Rules PASS × Risk/Reward)
# ===============================================================
st.subheader("🧭 Trade Readiness (Rules PASS × Risk/Reward)")

# Controls
cA, cB = st.columns([1, 3])
with cA:
    strategy_dash = st.selectbox(
        "Ruleset",
        list(rules.RULESETS.keys()),
        index=0,
        key="dash_ruleset"
    )
with cB:
    min_rr = st.slider("Minimum acceptable R:R", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

# 1) Evaluate latest annual rows (same logic as Systematic Decision page)
annual_only = df[df["IsQuarter"] != True].copy()
latest_annual = (
    annual_only
    .sort_values(["Name", "Year"])
    .groupby("Name", as_index=False)
    .tail(1)
)

eval_rows = []
for _, r in latest_annual.iterrows():
    m = calculations.calc_ratios(r)
    price = r.get("CurrentPrice", np.nan)
    if pd.isna(price):
        price = r.get("SharePrice", np.nan)
    ev = rules.evaluate(m, strategy_dash)
    eval_rows.append({
        "Name": r["Name"],
        "Industry": r.get("Industry", ""),
        "Year": int(r["Year"]),
        "CurrentPrice": price,
        "Score": ev["score"],
        "Decision": "PASS" if ev["pass"] else "REJECT",
        "Unmet": "; ".join(ev["reasons"]),
    })
ev_df = pd.DataFrame(eval_rows)

# 2) Load Trade Queue (entry/stop/take/rr already supported by helper)
tq = io_helpers.load_trade_queue().copy()

# Robust numeric casting for queue fields
for col in ["Entry", "Stop", "Take", "RR", "Shares"]:
    if col in tq.columns:
        tq[col] = pd.to_numeric(tq[col], errors="coerce")

# 3) Join rules + plan on Name (keep all plans)
joined = (tq.merge(ev_df, on="Name", how="left", suffixes=("", "_Eval"))
            .rename(columns={"Strategy": "PlanStrategy"}))

# 4) Derived risk & reward
joined["Risk (MYR)"]   = (joined["Shares"] * (joined["Entry"] - joined["Stop"])).where(
    joined["Shares"].notna() & joined["Entry"].notna() & joined["Stop"].notna()
)
joined["Reward (MYR)"] = (joined["Shares"] * (joined["Take"] - joined["Entry"])).where(
    joined["Shares"].notna() & joined["Take"].notna() & joined["Entry"].notna()
)
joined["Cost (MYR)"]   = (joined["Shares"] * joined["Entry"]).where(
    joined["Shares"].notna() & joined["Entry"].notna()
)
joined["Risk % of Position"] = (100.0 * joined["Risk (MYR)"] / joined["Cost (MYR)"]).where(
    joined["Risk (MYR)"].notna() & joined["Cost (MYR)"].notna() & (joined["Cost (MYR)"] > 0)
)

# 5) RR Band, Viability, Action
def rr_band_label(x):
    x = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if pd.isna(x):
        return "N/A"
    if x < 1.5:
        return "Low"
    if x < 2.0:
        return "OK"
    return "Good"

joined["RR Band"] = joined["RR"].apply(rr_band_label)

def viability(row):
    if row.get("Decision") != "PASS":
        return "⛔ Fails rules"
    rr = pd.to_numeric(pd.Series([row.get("RR")]), errors="coerce").iloc[0]
    sh = pd.to_numeric(pd.Series([row.get("Shares")]), errors="coerce").iloc[0]
    if pd.notna(rr) and rr >= min_rr and pd.notna(sh) and sh > 0:
        return "✅ Ready"
    if pd.notna(rr) and rr < min_rr:
        return "⚠️ Low R"
    return "⏳ Incomplete"
joined["Viability"] = joined.apply(viability, axis=1)

def action_hint(row):
    v = row.get("Viability")
    if v == "✅ Ready":
        return "Place order per plan."
    if v == "⚠️ Low R":
        return f"Improve R (target ≥ {min_rr:.1f}) or adjust stop."
    if v == "⛔ Fails rules":
        return "Fails rules — review fundamentals."
    return "Complete plan (shares/targets)."
joined["Action"] = joined.apply(action_hint, axis=1)

# 6) Display
cols_order = [
    "Name", "PlanStrategy", "Decision", "Score",
    "Entry", "Stop", "Take", "RR", "RR Band", "Shares",
    "Risk (MYR)", "Reward (MYR)", "Risk % of Position",
    "Viability", "Action", "Reasons"
]
disp_cols = [c for c in cols_order if c in joined.columns]
st.dataframe(joined[disp_cols], use_container_width=True, height=420)

# 7) KPIs
ready_count   = (joined["Viability"] == "✅ Ready").sum()
pass_count    = (ev_df["Decision"] == "PASS").sum() if not ev_df.empty else 0
planned_count = len(tq)
low_r_count   = (pd.to_numeric(joined["RR"], errors="coerce") < min_rr).sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("✅ Ready trades", int(ready_count))
k2.metric("PASS (rules)",    int(pass_count))
k3.metric("Plans in queue",  int(planned_count))
k4.metric(f"Low R (< {min_rr:.1f})", int(low_r_count))

