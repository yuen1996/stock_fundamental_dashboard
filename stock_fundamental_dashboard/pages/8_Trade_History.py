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



# 8_Trade_History.py

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
except Exception:
    import io_helpers

# ---- safe rerun for all Streamlit versions ----
def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

# ---------- Page setup ----------
st.set_page_config(page_title="Trade History", layout="wide")

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

st.header("📘 Trade History")

# ---------- Load data ----------
full = io_helpers.load_closed_trades()
if full.empty:
    st.info("No closed trades yet.")
    st.stop()

# Preserve a RowId for deletion mapping
full = full.reset_index().rename(columns={"index": "RowId"})

# ─────────────────────────────────────────────────────────
# Filters
# ─────────────────────────────────────────────────────────
st.markdown(
    '<div class="sec"><div class="t">🔎 Filters</div>'
    '<div class="d">Strategy, period, R threshold & search</div></div>',
    unsafe_allow_html=True
)

colA, colB, colC, colD = st.columns([1, 1, 1, 2])
with colA:
    strat = st.selectbox("Strategy", ["All"] + sorted(full["Strategy"].dropna().unique()), index=0)
with colB:
    period = st.selectbox("Period", ["All", "YTD", "Last 30 days", "Last 90 days", "Last 1 year"], index=1)
with colC:
    min_rr = st.slider("Min RR_Init", 0.0, 5.0, 0.0, 0.1)
with colD:
    search = st.text_input("Search Name / CloseReason / Notes", "")

flt = full.copy()
# Parse CloseDate from epoch-ms or ISO string → datetime
def _parse_ms_or_iso(s: pd.Series) -> pd.Series:
    ms = pd.to_numeric(s, errors="coerce")
    dt_ms  = pd.to_datetime(ms, unit="ms", errors="coerce")         # epoch ms
    dt_iso = pd.to_datetime(s, errors="coerce")                      # ISO strings
    return dt_ms.fillna(dt_iso)

flt["CloseDate_dt"] = _parse_ms_or_iso(flt["CloseDate"])


if strat != "All":
    flt = flt[flt["Strategy"] == strat]

if period != "All":
    now = datetime.now()
    if period == "YTD":
        cutoff = datetime(now.year, 1, 1)
    elif period == "Last 30 days":
        cutoff = now - timedelta(days=30)
    elif period == "Last 90 days":
        cutoff = now - timedelta(days=90)
    else:
        cutoff = now - timedelta(days=365)
    flt = flt[flt["CloseDate_dt"] >= cutoff]

flt["RR_Init"] = pd.to_numeric(flt["RR_Init"], errors="coerce")
flt = flt[flt["RR_Init"].fillna(0) >= min_rr]

if search.strip():
    q = search.lower()
    flt = flt[
        flt["Name"].astype(str).str.lower().str.contains(q, na=False)
        | flt["CloseReason"].astype(str).str.lower().str.contains(q, na=False)
        | flt["Reasons"].astype(str).str.lower().str.contains(q, na=False)
    ]

# Show friendly string, keep dt for filtering
flt["CloseDate"] = flt["CloseDate_dt"].dt.strftime("%Y-%m-%d %H:%M")


# ─────────────────────────────────────────────────────────
# Overview (KPIs)
# ─────────────────────────────────────────────────────────
wins = (pd.to_numeric(flt["PnL"], errors="coerce") > 0).sum()
loss = (pd.to_numeric(flt["PnL"], errors="coerce") <= 0).sum()
total_pnl = pd.to_numeric(flt["PnL"], errors="coerce").sum()
avg_r = pd.to_numeric(flt["RMultiple"], errors="coerce").mean()

st.markdown(
    '<div class="sec success"><div class="t">📊 Overview</div>'
    '<div class="d">Win rate & performance</div></div>',
    unsafe_allow_html=True
)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Trades", len(flt))
k2.metric("Win Rate", f"{(wins / max(len(flt),1))*100:.1f}%")
k3.metric("Total PnL (MYR)", f"{(total_pnl or 0):,.2f}")
k4.metric("Avg R multiple", f"{(avg_r or 0):.2f}")

# ─────────────────────────────────────────────────────────
# History Records (table)
# ─────────────────────────────────────────────────────────
st.markdown(
    '<div class="sec info"><div class="t">📚 History Records</div>'
    '<div class="d">Closed trades with outcomes</div></div>',
    unsafe_allow_html=True
)

cols = [
    "RowId", "CloseDate", "Name", "Strategy",
    "Entry", "Stop", "Take", "Shares", "RR_Init",
    "ClosePrice", "HoldingDays", "PnL", "ReturnPct", "RMultiple",
    "CloseReason", "Reasons",
]
cols = [c for c in cols if c in flt.columns]
table = flt[cols].sort_values("CloseDate", ascending=False).reset_index(drop=True)
table.insert(0, "Select", False)

edited = st.data_editor(
    table,
    use_container_width=True,
    height=520,
    hide_index=True,
    column_config={
        "Select":      st.column_config.CheckboxColumn("Select"),
        "RowId":       st.column_config.TextColumn("RowId", disabled=True),
        "CloseDate":   st.column_config.TextColumn("Close Date", disabled=True),
        "Name":        st.column_config.TextColumn("Name", disabled=True),
        "Strategy":    st.column_config.TextColumn("Strategy", disabled=True),
        "Entry":       st.column_config.NumberColumn("Entry", format="%.4f", disabled=True),
        "Stop":        st.column_config.NumberColumn("Stop", format="%.4f", disabled=True),
        "Take":        st.column_config.NumberColumn("Take", format="%.4f", disabled=True),
        "Shares":      st.column_config.NumberColumn("Shares", format="%d", disabled=True),
        "RR_Init":     st.column_config.NumberColumn("RR Init", format="%.2f", disabled=True),
        "ClosePrice":  st.column_config.NumberColumn("Close Price", format="%.4f", disabled=True),
        "HoldingDays": st.column_config.NumberColumn("Days", format="%d", disabled=True),
        "PnL":         st.column_config.NumberColumn("PnL (MYR)", format="%.2f", disabled=True),
        "ReturnPct":   st.column_config.NumberColumn("Return %", format="%.2f", disabled=True),
        "RMultiple":   st.column_config.NumberColumn("R multiple", format="%.2f", disabled=True),
        "CloseReason": st.column_config.TextColumn("Close Reason", disabled=True),
        "Reasons":     st.column_config.TextColumn("Notes", disabled=True),
    },
    key="trade_history_editor",
)

# ─────────────────────────────────────────────────────────
# Bulk Actions
# ─────────────────────────────────────────────────────────
st.markdown(
    '<div class="sec warning"><div class="t">🧹 Bulk Actions</div>'
    '<div class="d">Delete selected or all shown (after filters)</div></div>',
    unsafe_allow_html=True
)

c1, c2, _ = st.columns([1.6, 2, 3])
with c1:
    if st.button("🗑️ Delete selected"):
        sel_ids = set(edited.loc[edited["Select"] == True, "RowId"].tolist())
        if not sel_ids:
            st.warning("No rows selected.")
        else:
            base = full.copy()
            base = base[~base["RowId"].isin(sel_ids)]
            base = base.drop(columns=["RowId"], errors="ignore")
            io_helpers.save_closed_trades(base)
            st.success(f"Deleted {len(sel_ids)} row(s) from Trade History.")
            _safe_rerun()

with c2:
    if st.button("🧹 Delete ALL shown (after filters)"):
        shown_ids = set(edited["RowId"].tolist())
        if not shown_ids:
            st.warning("No rows to delete for the current filter.")
        else:
            base = full.copy()
            base = base[~base["RowId"].isin(shown_ids)]
            base = base.drop(columns=["RowId"], errors="ignore")
            io_helpers.save_closed_trades(base)
            st.success(f"Deleted {len(shown_ids)} row(s) from Trade History.")
            _safe_rerun()

st.caption("Use filters to narrow results, then **Delete ALL shown** to clear older records quickly.")


