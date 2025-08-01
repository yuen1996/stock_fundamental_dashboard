# 7_Ongoing_Trades.py

# --- path patch so this page can import from project root ---
import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))  # project root (parent of /pages)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# robust imports: prefer package (utils), fall back to top-level
try:
    from utils import io_helpers
except Exception:
    import io_helpers

# ---------- Page setup ----------
st.set_page_config(page_title="Ongoing Trades", layout="wide")

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

/* Section header card (visual separators) */
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

/* Inputs & buttons */
div[data-baseweb="input"] input, textarea, .stNumberInput input{ font-size:15px !important; }
.stSlider > div [data-baseweb="slider"]{ margin-top:.25rem; }
.stButton>button{ border-radius:12px !important; padding:.55rem 1.1rem !important; font-weight:700; }

/* Tabs */
.stTabs [role="tab"]{ font-size:15px !important; font-weight:600 !important; }

/* Sidebar theme (dark, same as Dashboard) */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg, #0b1220 0%, #1f2937 100%) !important;
}
[data-testid="stSidebar"] *{ color:#e5e7eb !important; }
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

# ---- safe rerun helper for all Streamlit versions ----
def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

st.header("📈 Ongoing Trades")

open_df = io_helpers.load_open_trades()
if open_df.empty:
    st.info("No ongoing trades. Use **Systematic Decision → Manage Queue → Mark Live** to open a position.")
    st.stop()

# ─────────────────────────────────────────
# Overview (KPIs)
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec"><div class="t">📊 Overview</div>'
    '<div class="d">Open positions & exposure</div></div>',
    unsafe_allow_html=True
)

try:
    shares = pd.to_numeric(open_df.get("Shares"), errors="coerce")
    entry  = pd.to_numeric(open_df.get("Entry"),  errors="coerce")
    cost   = (shares * entry).fillna(0).sum()
    k1, k2 = st.columns(2)
    k1.metric("Open Positions", len(open_df))
    k2.metric("Total Cost (MYR)", f"{cost:,.2f}")
except Exception:
    pass

# ─────────────────────────────────────────
# Open positions table (with close controls)
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec success"><div class="t">🧾 Open Positions</div>'
    '<div class="d">Close price & reason required to exit</div></div>',
    unsafe_allow_html=True
)

CLOSE_REASONS = [
    "Target hit",
    "Stop hit",
    "Trailing stop",
    "Time stop",
    "Thesis changed",
    "Portfolio rebalance",
    "Other (specify)",
]

table = open_df.copy()
table.insert(0, "Select", False)
table["ClosePrice"] = 0.0
table["CloseReason"] = CLOSE_REASONS[0]
table["Detail"] = ""

edited_open = st.data_editor(
    table,
    use_container_width=True,
    height=440,
    hide_index=True,
    column_config={
        "Select":      st.column_config.CheckboxColumn("Select"),
        "Name":        st.column_config.TextColumn("Name", disabled=True),
        "Strategy":    st.column_config.TextColumn("Strategy", disabled=True),
        "Entry":       st.column_config.NumberColumn("Entry", format="%.4f", disabled=True),
        "Stop":        st.column_config.NumberColumn("Stop", format="%.4f", disabled=True),
        "Take":        st.column_config.NumberColumn("Take", format="%.4f", disabled=True),
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
    '<div class="d">Close selected positions</div></div>',
    unsafe_allow_html=True
)

c1, _, _ = st.columns([1.6, 1, 1])
with c1:
    if st.button("🔒 Close selected"):
        closed, invalid = 0, 0
        for _, r in edited_open.iterrows():
            if not r.get("Select"):
                continue
            px = float(r.get("ClosePrice") or 0)
            reason = r.get("CloseReason") or ""
            det = r.get("Detail") or ""
            if px <= 0:
                invalid += 1
                continue
            if reason == "Other (specify)" and not det.strip():
                invalid += 1
                continue
            reason_text = reason if reason != "Other (specify)" else f"{reason}: {det.strip()}"
            ok = io_helpers.close_open_trade(
                name=r["Name"], strategy=r["Strategy"],
                close_price=px, close_reason=reason_text
            )
            if ok:
                closed += 1
        if closed:
            st.success(f"Closed {closed} trade(s).")
            _safe_rerun()
        if invalid and not closed:
            st.warning(f"{invalid} selected row(s) invalid (need Close Price and/or Detail for 'Other').")


