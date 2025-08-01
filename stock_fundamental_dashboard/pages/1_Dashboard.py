import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils import io_helpers, calculations, rules

# ──────────────────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────────────────
st.set_page_config(layout="wide")

# === Unified CSS (paste the same block in every page after set_page_config) ===
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

/* Sidebar theme */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg, #0b1220 0%, #1f2937 100%) !important;
}
[data-testid="stSidebar"] *{ color:#e5e7eb !important; }

/* Optional: hide Streamlit default page list
section[data-testid="stSidebarNav"]{ display:none !important; }
*/
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# Title
# ──────────────────────────────────────────────────────────
st.header("📊 Dashboard")

# ──────────────────────────────────────────────────────────
# Load Data
# ──────────────────────────────────────────────────────────
df = io_helpers.load_data()
if df is None or df.empty or "Name" not in df.columns:
    st.warning("No stock data found. Please add data in **Add / Edit**.")
    st.stop()

if "IsQuarter" not in df.columns:
    df["IsQuarter"] = False
df["IsQuarter"] = df["IsQuarter"].astype(bool)

annual = df[~df["IsQuarter"]].copy()

# ──────────────────────────────────────────────────────────
# Filters row
# ──────────────────────────────────────────────────────────
colf1, colf2, colf3 = st.columns([1, 2, 1])
with colf1:
    industries = ["All"] + sorted(annual["Industry"].dropna().unique())
    industry_sel = st.selectbox("Filter by Industry", industries, index=0)
with colf2:
    search_text = st.text_input("🔍 Search Stock Name or Industry", placeholder="Type to filter...")
with colf3:
    date_opts = ["Any", "Last 7 days", "Last 14 days", "Last 1 month", "Last 3 months"]
    date_sel = st.selectbox("Updated in", date_opts, index=0)

df_view = annual.copy()
if industry_sel != "All":
    df_view = df_view[df_view["Industry"] == industry_sel]
if search_text.strip():
    q = search_text.lower()
    df_view = df_view[
        df_view["Name"].str.lower().str.contains(q, na=False) |
        df_view["Industry"].str.lower().str.contains(q, na=False)
    ]
if date_sel != "Any" and "LastModified" in df_view.columns:
    now = datetime.now()
    if date_sel == "Last 7 days":      cutoff = now - timedelta(days=7)
    elif date_sel == "Last 14 days":   cutoff = now - timedelta(days=14)
    elif date_sel == "Last 1 month":   cutoff = now - timedelta(days=30)
    else:                               cutoff = now - timedelta(days=90)
    lm = pd.to_datetime(df_view["LastModified"], errors="coerce")
    df_view = df_view[lm >= cutoff]

# ──────────────────────────────────────────────────────────
# SECTION 1: Fundamentals (Annual + Quarterly)
# ──────────────────────────────────────────────────────────
st.markdown('<div class="sec primary"><div class="t">📌 Fundamentals</div><div class="d">Latest annual & quarterly snapshots</div></div>', unsafe_allow_html=True)

# --- Latest-year Summary ---
st.markdown('<div class="sec"><div class="t">📌 Latest-year Summary</div><div class="d">One row per stock (latest Year)</div></div>', unsafe_allow_html=True)
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
        if pd.isna(cur): cur = r.get("SharePrice", np.nan)
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

# --- Latest-quarter Summary ---
st.markdown('<div class="sec info"><div class="t">📌 Latest-quarter Summary</div><div class="d">One row per stock (latest Year & Quarter)</div></div>', unsafe_allow_html=True)

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
    if pd.isna(cur): cur = r.get("Q_EndQuarterPrice", np.nan)
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

# ──────────────────────────────────────────────────────────
# SECTION 2: Trade Readiness (rules × risk/reward)
# ──────────────────────────────────────────────────────────
st.markdown('<div class="sec success"><div class="t">🧭 Trade Readiness (Rules PASS × Risk/Reward)</div><div class="d">Combine rules evaluation with your planned R:R from the queue</div></div>', unsafe_allow_html=True)

# Controls
cA, cB = st.columns([1, 3])
with cA:
    strategy_dash = st.selectbox("Ruleset", list(rules.RULESETS.keys()), index=0, key="dash_ruleset")
with cB:
    min_rr = st.slider("Minimum acceptable R:R", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

# Evaluate latest annual rows
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
    if pd.isna(price): price = r.get("SharePrice", np.nan)
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

# Join with Trade Queue plans
tq = io_helpers.load_trade_queue().copy()
for col in ["Entry", "Stop", "Take", "RR", "Shares"]:
    if col in tq.columns: tq[col] = pd.to_numeric(tq[col], errors="coerce")

joined = (tq.merge(ev_df, on="Name", how="left", suffixes=("", "_Eval"))
            .rename(columns={"Strategy": "PlanStrategy"}))

# Derived
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

def rr_band_label(x):
    x = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if pd.isna(x): return "N/A"
    if x < 1.5: return "Low"
    if x < 2.0: return "OK"
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
    if v == "✅ Ready": return "Place order per plan."
    if v == "⚠️ Low R": return f"Improve R (target ≥ {min_rr:.1f}) or adjust stop."
    if v == "⛔ Fails rules": return "Fails rules — review fundamentals."
    return "Complete plan (shares/targets)."
joined["Action"] = joined.apply(action_hint, axis=1)

cols_order = [
    "Name", "PlanStrategy", "Decision", "Score",
    "Entry", "Stop", "Take", "RR", "RR Band", "Shares",
    "Risk (MYR)", "Reward (MYR)", "Risk % of Position",
    "Viability", "Action", "Reasons"
]
disp_cols = [c for c in cols_order if c in joined.columns]
st.dataframe(joined[disp_cols], use_container_width=True, height=420)

# KPIs
ready_count   = (joined["Viability"] == "✅ Ready").sum()
pass_count    = (ev_df["Decision"] == "PASS").sum() if not ev_df.empty else 0
planned_count = len(tq)
low_r_count   = (pd.to_numeric(joined["RR"], errors="coerce") < min_rr).sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("✅ Ready trades", int(ready_count))
k2.metric("PASS (rules)",    int(pass_count))
k3.metric("Plans in queue",  int(planned_count))
k4.metric(f"Low R (< {min_rr:.1f})", int(low_r_count))

with st.expander("📥 Trade Queue (raw view)"):
    st.dataframe(tq, use_container_width=True, height=240)


