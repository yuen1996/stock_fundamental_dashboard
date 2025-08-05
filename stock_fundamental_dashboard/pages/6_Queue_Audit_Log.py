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



# 6_Queue_Audit_Log.py

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

# --- robust imports: prefer package (utils), fall back to top-level ---
try:
    from utils import io_helpers
except Exception:
    import io_helpers

# ---------- Page setup ----------
st.set_page_config(page_title="Trade Queue Audit Log", layout="wide")

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

# ---- safe rerun for all Streamlit versions ----
def _safe_rerun():
    try:
        st.rerun()  # newer versions
    except Exception:
        try:
            st.experimental_rerun()  # older versions
        except Exception:
            pass

st.header("🧾 Trade Queue Audit Log")

# ---------- Load full log ----------
log_full = io_helpers.load_queue_audit()
if log_full.empty:
    st.info("No audit records yet.")
    st.stop()

log_full = log_full.reset_index().rename(columns={"index": "RowId"})  # stable id per load
log_full["Timestamp"] = pd.to_datetime(log_full["Timestamp"], errors="coerce")

# ─────────────────────────────────────────────────────────
# Filters
# ─────────────────────────────────────────────────────────
st.markdown(
    '<div class="sec"><div class="t">🔎 Filters</div>'
    '<div class="d">Event, period, and text search</div></div>',
    unsafe_allow_html=True
)

colA, colB, colC = st.columns([1, 2, 2])
with colA:
    event = st.selectbox("Event", ["All", "UPSERT", "MARK_LIVE", "CLOSE", "DELETE"], index=0)
with colB:
    search = st.text_input("Search Name / Strategy / Reasons", "")
with colC:
    period = st.selectbox("Period", ["All", "Last 7 days", "Last 30 days", "Last 90 days"], index=2)

df = log_full.copy()

# event filter
if event != "All":
    df = df[df["Event"] == event]

# period filter
if period != "All":
    now = datetime.now()
    days = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}[period]
    cutoff = now - timedelta(days=days)
    df = df[df["Timestamp"] >= cutoff]

# search filter
if search.strip():
    q = search.lower()
    df = df[
        df["Name"].astype(str).str.lower().str.contains(q, na=False)
        | df["Strategy"].astype(str).str.lower().str.contains(q, na=False)
        | df["Reasons"].astype(str).str.lower().str.contains(q, na=False)
        | df["AuditReason"].astype(str).str.lower().str.contains(q, na=False)
        | df["Event"].astype(str).str.lower().str.contains(q, na=False)
    ]

df = df.sort_values("Timestamp", ascending=False).reset_index(drop=True)

# ─────────────────────────────────────────────────────────
# Audit Records (table)
# ─────────────────────────────────────────────────────────
st.markdown(
    '<div class="sec success"><div class="t">📚 Audit Records</div>'
    '<div class="d">All queue changes with reasons</div></div>',
    unsafe_allow_html=True
)

table = df.copy()
table.insert(0, "Select", False)

cols = [
    "Select", "RowId", "Timestamp", "Event", "Name", "Strategy",
    "Score", "CurrentPrice", "Entry", "Stop", "Take", "Shares", "RR",
    "TP1", "TP2", "TP3", "Reasons", "AuditReason",
]
cols = [c for c in cols if c in table.columns]

edited = st.data_editor(
    table[cols],
    use_container_width=True,
    height=520,
    hide_index=True,
    column_config={
        "Select":       st.column_config.CheckboxColumn("Select"),
        "RowId":        st.column_config.TextColumn("RowId", disabled=True),
        "Timestamp":    st.column_config.TextColumn("Timestamp", disabled=True),
        "Event":        st.column_config.TextColumn("Event", disabled=True),
        "Name":         st.column_config.TextColumn("Name", disabled=True),
        "Strategy":     st.column_config.TextColumn("Strategy", disabled=True),
        "Score":        st.column_config.NumberColumn("Score", disabled=True),
        "CurrentPrice": st.column_config.NumberColumn("Current", disabled=True),
        "Entry":        st.column_config.NumberColumn("Entry", disabled=True),
        "Stop":         st.column_config.NumberColumn("Stop", disabled=True),
        "Take":         st.column_config.NumberColumn("Take", disabled=True),
        "Shares":       st.column_config.NumberColumn("Shares", disabled=True),
        "RR":           st.column_config.NumberColumn("RR", disabled=True),
        "TP1":          st.column_config.NumberColumn("TP1", disabled=True),
        "TP2":          st.column_config.NumberColumn("TP2", disabled=True),
        "TP3":          st.column_config.NumberColumn("TP3", disabled=True),
        "Reasons":      st.column_config.TextColumn("Row Reasons", disabled=True),
        "AuditReason":  st.column_config.TextColumn("Audit Reason", disabled=True),
    },
    key="queue_audit_editor",
)

# ─────────────────────────────────────────────────────────
# Bulk Actions
# ─────────────────────────────────────────────────────────
st.markdown(
    '<div class="sec warning"><div class="t">🧹 Bulk Actions</div>'
    '<div class="d">Delete selected or all shown (after filters)</div></div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns([1.4, 1.8, 3])
with c1:
    if st.button("🗑️ Delete selected"):
        selected_ids = set(edited.loc[edited["Select"] == True, "RowId"].tolist())
        if not selected_ids:
            st.warning("No rows selected.")
        else:
            base = log_full.copy()
            base = base[~base["RowId"].isin(selected_ids)]
            # drop helper column before save
            base = base.drop(columns=["RowId"], errors="ignore")
            io_helpers.save_queue_audit(base)
            st.success(f"Deleted {len(selected_ids)} row(s) from audit log.")
            _safe_rerun()

with c2:
    if st.button("🧹 Delete ALL shown (after filters)"):
        shown_ids = set(edited["RowId"].tolist())
        if not shown_ids:
            st.warning("No rows to delete for the current filter.")
        else:
            base = log_full.copy()
            base = base[~base["RowId"].isin(shown_ids)]
            base = base.drop(columns=["RowId"], errors="ignore")
            io_helpers.save_queue_audit(base)
            st.success(f"Deleted {len(shown_ids)} row(s) from audit log.")
            _safe_rerun()

st.caption("Tip: Use filters (Event/Period/Search) to narrow, then **Delete ALL shown** to clear older records quickly.")


