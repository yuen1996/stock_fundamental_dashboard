# pages/6_Queue_Audit_Log.py
from __future__ import annotations

# --- Auth ---
from auth_gate import require_auth
require_auth()

# --- UI & CSS ---
try:
    from utils.ui import setup_page, section, register_page_css, render_page_title
except Exception:
    from ui import setup_page, section, register_page_css  # fallback
    try:
        from ui import render_page_title  # type: ignore
    except Exception:
        def render_page_title(page_name: str) -> None:
            import streamlit as _st
            _st.title(f"📊 Fundamentals Dashboard — {page_name}")

# Register page-specific CSS (same KPI/area look)
register_page_css("6_Queue_Audit_Log", """
.sec {margin:.75rem 0;}
.sec .t{font-weight:900;font-size:1.05rem;margin:0;}
.sec .d{color:#6b7280;font-size:.95rem;margin-top:.2rem;}
.sec + [data-testid="stVerticalBlock"]{margin:.35rem 0 1rem 0;}
.sec + [data-testid="stVerticalBlock"]:not(:has(> [data-testid="stVerticalBlock"])){
  background:#fff;border:1px solid var(--border);border-radius:14px;box-shadow:var(--shadow);
  padding:.8rem 1rem;
}
.sec + [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:not(:has(> [data-testid="stVerticalBlock"])){
  background:#fff;border:1px solid var(--border);border-radius:14px;box-shadow:var(--shadow);
  padding:.8rem 1rem;
}
.tbl-note{color:#6b7280;font-size:.9rem;margin:.25rem 0;}
.badge-chip{display:inline-block;padding:.15rem .5rem;border-radius:999px;background:#eef2ff;color:#4338ca;font-weight:800;font-size:.75rem;margin-right:.25rem;}
""")

setup_page("Queue Audit Log", "6_Queue_Audit_Log")
render_page_title("Queue Audit Log")

# --- Std libs ---
import os, sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st

# --- Make sure EVERY page uses the SAME data dir BEFORE importing io_helpers ---
_THIS   = os.path.dirname(__file__)
_PARENT = os.path.abspath(os.path.join(_THIS, ".."))
_GRANDP = os.path.abspath(os.path.join(_PARENT, ".."))

def _dir_writable(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        _t = os.path.join(path, ".sfd_write_test")
        with open(_t, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(_t)
        return True
    except Exception:
        return False

_DEFAULT_DATA_DIR  = os.path.join(_PARENT, "data")
_FALLBACK_DATA_DIR = os.path.join(os.path.expanduser("~"), ".sfd_data")
if _dir_writable(_DEFAULT_DATA_DIR):
    os.environ["SFD_DATA_DIR"] = _DEFAULT_DATA_DIR
else:
    os.makedirs(_FALLBACK_DATA_DIR, exist_ok=True)
    os.environ["SFD_DATA_DIR"] = _FALLBACK_DATA_DIR

# Robust import pathing
for p in (_PARENT, _GRANDP):
    if p not in sys.path:
        sys.path.insert(0, p)

# --- IO helpers (reload after setting SFD_DATA_DIR) ---
try:
    from utils import io_helpers as ioh
except Exception:
    import io_helpers as ioh  # type: ignore

import importlib as _importlib
ioh = _importlib.reload(ioh)

# ---------- Header ----------
st.markdown(section("📒 Queue Audit Log", "Every queue change + Mark Live action is recorded."), unsafe_allow_html=True)

# ---------- Load audit ----------
try:
    df = ioh.load_queue_audit()
except Exception:
    df = None

if df is None or df.empty:
    st.info("No audit records yet. When you update queue rows or click **Mark Live**, entries will appear here.")
    st.stop()

# Normalize + parse timestamps
def _parse_ts(x):
    try:
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        return pd.NaT

if "Timestamp" in df.columns:
    df["Timestamp_dt"] = df["Timestamp"].map(_parse_ts)
    df["Timestamp"]    = df["Timestamp_dt"].dt.strftime("%Y-%m-%d %H:%M")

# ---------- Filters ----------
st.markdown(section("Filters", "Narrow down the history you want to inspect"), unsafe_allow_html=True)
st.markdown('<div class="area">', unsafe_allow_html=True)

colA, colB, colC = st.columns([1, 2, 2])
with colA:
    event = st.selectbox("Event", ["All", "UPSERT", "UPDATE", "MARK_LIVE", "DELETE"], index=0)
with colB:
    search = st.text_input("Search Name / Strategy / Reasons", "")
with colC:
    period = st.selectbox("Period", ["All", "Last 7 days", "Last 30 days", "Last 90 days"], index=0)

view = df.copy()

# event filter
if event != "All" and "Event" in view.columns:
    view = view[view["Event"] == event]

# period filter
if period != "All" and "Timestamp_dt" in view.columns:
    now = datetime.now()
    days = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}[period]
    cutoff = now - timedelta(days=days)
    view = view[view["Timestamp_dt"] >= cutoff]

# search filter
if search.strip():
    q = search.lower()
    def _has(col): return col in view.columns
    mask = False
    for c in ["Name","Strategy","Reasons","AuditReason","Event"]:
        if _has(c):
            mask = mask | view[c].astype(str).str.lower().str.contains(q, na=False)
    view = view[mask]

# Sorting + prep
sort_col = "Timestamp_dt" if "Timestamp_dt" in view.columns else "Timestamp"
if sort_col in view.columns:
    view = view.sort_values(sort_col, ascending=False).reset_index(drop=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Table ----------
st.markdown(section("📚 Audit Records", "All queue changes with reasons", "success"), unsafe_allow_html=True)
st.markdown('<div class="area">', unsafe_allow_html=True)

# add a stable RowId for selection
show = view.copy().reset_index().rename(columns={"index": "RowId"})
if "RowId" not in show.columns:
    show.insert(0, "RowId", range(len(show)))

# Ensure columns exist
for c in ["TP1","TP2","TP3","AuditReason"]:
    if c not in show.columns:
        show[c] = ""

# Pretty config
cfg = {
    "RowId":        st.column_config.NumberColumn("RowId", disabled=True, help="Session index"),
    "Timestamp":    st.column_config.TextColumn("When", disabled=True),
    "Event":        st.column_config.TextColumn("Event", disabled=True),
    "Name":         st.column_config.TextColumn("Name", disabled=True),
    "Strategy":     st.column_config.TextColumn("Strategy", disabled=True),
    "Score":        st.column_config.NumberColumn("Score", format="%.0f", disabled=True),
    "CurrentPrice": st.column_config.NumberColumn("Price", format="%.4f", disabled=True),
    "Entry":        st.column_config.NumberColumn("Entry", format="%.4f", disabled=True),
    "Stop":         st.column_config.NumberColumn("Stop",  format="%.4f", disabled=True),
    "Take":         st.column_config.NumberColumn("Take",  format="%.4f", disabled=True),
    "Shares":       st.column_config.NumberColumn("Shares", format="%d",   disabled=True),
    "RR":           st.column_config.NumberColumn("RR",    format="%.2f", disabled=True),
    "TP1":          st.column_config.NumberColumn("TP1",   format="%.4f", disabled=True),
    "TP2":          st.column_config.NumberColumn("TP2",   format="%.4f", disabled=True),
    "TP3":          st.column_config.NumberColumn("TP3",   format="%.4f", disabled=True),
    "Reasons":      st.column_config.TextColumn("Row Reasons", disabled=True),
    "AuditReason":  st.column_config.TextColumn("Audit Reason", disabled=True),
}

show = show[
    [c for c in ["RowId","Timestamp","Event","Name","Strategy","Score","CurrentPrice",
                 "Entry","Stop","Take","Shares","RR","TP1","TP2","TP3","Reasons","AuditReason"]
     if c in show.columns]
]

# selection column
show.insert(0, "Select", False)

edited = st.data_editor(
    show,
    hide_index=True,
    use_container_width=True,
    height=min(520, 72 + 28*min(len(show), 14)),
    column_config={"Select": st.column_config.CheckboxColumn("Sel"), **cfg},
    key="queue_audit_editor",
)

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Bulk actions ----------
st.markdown(section("🧹 Bulk Actions", "Delete selected or all shown (after filters)", "warning"), unsafe_allow_html=True)
st.markdown('<div class="area">', unsafe_allow_html=True)
c1, c2, _ = st.columns([1.4, 1.8, 3])

def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

with c1:
    if st.button("🗑️ Delete selected"):
        ids = set(edited.loc[edited["Select"] == True, "RowId"].tolist())
        if not ids:
            st.warning("No rows selected.")
        else:
            base = view.copy().reset_index().rename(columns={"index": "RowId"})
            base = base[~base["RowId"].isin(ids)]
            # drop helper column before save
            base = base.drop(columns=["RowId"], errors="ignore")
            # Merge back with untouched rows outside current filter
            keep = pd.concat([base, df[~df.index.isin(view.index)]], ignore_index=True)
            ioh.save_queue_audit(keep)
            st.success(f"Deleted {len(ids)} row(s) from audit log.")
            _safe_rerun()

with c2:
    if st.button("🧹 Delete ALL shown (after filters)"):
        ids = set(edited["RowId"].tolist())
        if not ids:
            st.warning("No rows to delete for the current filter.")
        else:
            base = view.copy().reset_index().rename(columns={"index": "RowId"})
            base = base[~base["RowId"].isin(ids)]
            base = base.drop(columns=["RowId"], errors="ignore")
            keep = pd.concat([base, df[~df.index.isin(view.index)]], ignore_index=True)
            ioh.save_queue_audit(keep)
            st.success(f"Deleted {len(ids)} row(s) from audit log.")
            _safe_rerun()

st.markdown('</div>', unsafe_allow_html=True)
