# 7_Ongoing_Trades.py  – row-exact close logic + filters

# --- path patch so this page can import from project root ---
import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))  # project root (parent of /pages)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# robust imports: prefer package (utils), fall back to top-level
try:
    from utils import io_helpers
except Exception:        # fallback if running as flat repo
    import io_helpers     # type: ignore

# ---------- Page setup ----------
st.set_page_config(page_title="Ongoing Trades", layout="wide")

# =========== Unified CSS (same as other pages) ===========
BASE_CSS = """
<style>
:root{
  --bg:#f6f7fb; --surface:#ffffff; --text:#0f172a; --muted:#475569;
  --border:#e5e7eb; --shadow:0 8px 24px rgba(15,23,42,.06);
  --primary:#4f46e5; --info:#0ea5e9; --success:#10b981; --warning:#f59e0b; --danger:#ef4444;
}
html, body, [class*="css"]{ font-size:16px !important; color:var(--text); }
.stApp{ background: radial-gradient(1000px 500px at 10% -10%, #f0f3fb 0%, var(--bg) 60%), var(--bg); }
h1,h2,h3,h4{ color:var(--text) !important; font-weight:800 !important; letter-spacing:.2px; }
.sec{ background:var(--surface); border:1px solid var(--border); border-radius:14px;
      box-shadow:var(--shadow); padding:.65rem .9rem; margin:1rem 0 .6rem 0; display:flex; align-items:center; gap:.6rem; }
.sec .t{ font-size:1.05rem; font-weight:800; margin:0; padding:0; }
.sec .d{ color:var(--muted); font-size:.95rem; margin-left:.25rem; }
.sec::before{ content:""; display:inline-block; width:8px; height:26px; border-radius:6px; background:var(--primary); }
.sec.info::before{ background:var(--info); } .sec.success::before{ background:var(--success); }
.sec.warning::before{ background:var(--warning); } .sec.danger::before{ background:var(--danger); }
.stDataFrame, .stDataEditor{ font-size:15px !important; }
div[data-testid="stDataFrame"] td, div[data-testid="stDataEditor"] td{ border-bottom:1px solid var(--border) !important; }
div[data-baseweb="input"] input, textarea, .stNumberInput input{ font-size:15px !important; }
.stButton>button{ border-radius:12px !important; padding:.55rem 1.1rem !important; font-weight:700; }
.stTabs [role="tab"]{ font-size:15px !important; font-weight:600 !important; }
[data-testid="stSidebar"]{ background:linear-gradient(180deg,#0b1220 0%,#1f2937 100%) !important; }
[data-testid="stSidebar"] *{ color:#e5e7eb !important; }
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

# ---------- safe rerun helper ----------
def _safe_rerun():
    try:            st.rerun()
    except Exception:
        try:        st.experimental_rerun()
        except Exception:
            pass

st.header("📈 Ongoing Trades")

# ─────────────────────────────────────────
# Load live positions & give each a RowId
# ─────────────────────────────────────────
open_df = io_helpers.load_open_trades()
if open_df.empty:
    st.info("No ongoing trades. Use **Systematic Decision → Manage Queue → Mark Live** to open a position.")
    st.stop()

# Preserve original row ids to act on exact rows even after filtering
open_df = open_df.reset_index().rename(columns={"index": "RowId"})

# ─────────────────────────────────────────
# Filters (search by name, strategy, recent period)
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec"><div class="t">🔎 Filters</div>'
    '<div class="d">Narrow the list before taking action</div></div>',
    unsafe_allow_html=True
)

f1, f2, f3 = st.columns([2, 1, 1])
with f1:
    q = st.text_input("Search name", placeholder="Type part of a stock name…")
with f2:
    strategies = ["All"] + sorted([s for s in open_df["Strategy"].dropna().unique()])
    strat_sel = st.selectbox("Strategy", strategies, index=0)
with f3:
    period = st.selectbox("Opened in", ["Any", "Last 7 days", "Last 14 days", "Last 1 month", "Last 3 months"], index=0)

filtered = open_df.copy()

# apply filters
if q.strip():
    qq = q.lower()
    filtered = filtered[filtered["Name"].str.lower().str.contains(qq, na=False)]

if strat_sel != "All":
    filtered = filtered[filtered["Strategy"] == strat_sel]

if period != "Any" and "OpenDate" in filtered.columns:
    now = datetime.now()
    if period == "Last 7 days":      cutoff = now - timedelta(days=7)
    elif period == "Last 14 days":   cutoff = now - timedelta(days=14)
    elif period == "Last 1 month":   cutoff = now - timedelta(days=30)
    else:                             cutoff = now - timedelta(days=90)
    dt = pd.to_datetime(filtered["OpenDate"], errors="coerce")
    filtered = filtered[dt >= cutoff]

if filtered.empty:
    st.info("No rows match the current filters.")
    st.stop()

# ─────────────────────────────────────────
# Overview (KPIs)
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec"><div class="t">📊 Overview</div>'
    '<div class="d">Open positions &amp; exposure (filtered)</div></div>',
    unsafe_allow_html=True
)

shares = pd.to_numeric(filtered.get("Shares"), errors="coerce")
entry  = pd.to_numeric(filtered.get("Entry"),  errors="coerce")
total_cost = (shares * entry).fillna(0).sum()

k1, k2 = st.columns(2)
k1.metric("Open Positions (shown)", len(filtered))
k2.metric("Total Cost (MYR)", f"{total_cost:,.2f}")

# ─────────────────────────────────────────
# Editor table (filtered set)
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec success"><div class="t">🧾 Open Positions</div>'
    '<div class="d">Specify Close Price &amp; Reason, then tick rows to close</div></div>',
    unsafe_allow_html=True
)

CLOSE_REASONS = [
    "Target hit", "Stop hit", "Trailing stop", "Time stop",
    "Thesis changed", "Portfolio rebalance", "Other (specify)",
]

table = filtered.copy()
table.insert(0, "Select", False)
table["ClosePrice"]  = 0.0
table["CloseReason"] = CLOSE_REASONS[0]
table["Detail"]      = ""

edited = st.data_editor(
    table,
    use_container_width=True,
    height=460,
    hide_index=True,
    column_config={
        "Select":      st.column_config.CheckboxColumn("Sel"),
        "RowId":       st.column_config.NumberColumn("RowId", disabled=True),
        "Name":        st.column_config.TextColumn("Name", disabled=True),
        "Strategy":    st.column_config.TextColumn("Strategy", disabled=True),
        "Entry":       st.column_config.NumberColumn("Entry", format="%.4f", disabled=True),
        "Stop":        st.column_config.NumberColumn("Stop",  format="%.4f", disabled=True),
        "Take":        st.column_config.NumberColumn("Take",  format="%.4f", disabled=True),
        "Shares":      st.column_config.NumberColumn("Shares", format="%d", disabled=True),
        "RR":          st.column_config.NumberColumn("RR Init", format="%.2f", disabled=True),
        "TP1":         st.column_config.NumberColumn("TP1", format="%.4f", disabled=True),
        "TP2":         st.column_config.NumberColumn("TP2", format="%.4f", disabled=True),
        "TP3":         st.column_config.NumberColumn("TP3", format="%.4f", disabled=True),
        "OpenDate":    st.column_config.TextColumn("Open Date", disabled=True),
        "Reasons":     st.column_config.TextColumn("Notes", disabled=True),
        "ClosePrice":  st.column_config.NumberColumn("Close Price", format="%.4f"),
        "CloseReason": st.column_config.SelectboxColumn("Close Reason", options=CLOSE_REASONS),
        "Detail":      st.column_config.TextColumn("Detail (if Other)"),
    },
    key="open_trades_editor",
)

# ─────────────────────────────────────────
# Actions
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec warning"><div class="t">🔒 Actions</div>'
    '<div class="d">Close selected positions (row-exact)</div></div>',
    unsafe_allow_html=True
)

if st.button("🔒 Close selected"):
    closed, invalid = 0, 0
    for _, r in edited.iterrows():
        if not r.Select:
            continue
        px = float(r.ClosePrice or 0)
        reason = r.CloseReason or ""
        det = r.Detail or ""
        if px <= 0 or (reason == "Other (specify)" and not det.strip()):
            invalid += 1
            continue
        reason_txt = reason if reason != "Other (specify)" else f"{reason}: {det.strip()}"
        ok = io_helpers.close_open_trade_row(int(r.RowId), px, reason_txt)
        if ok:
            closed += 1
    msg = f"Closed {closed} trade(s)."
    if invalid:
        msg += f" {invalid} skipped (price/reason missing)."
    st.success(msg)
    _safe_rerun()

