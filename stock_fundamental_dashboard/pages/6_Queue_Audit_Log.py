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

st.set_page_config(layout="wide")
st.title("🧾 Trade Queue Audit Log")

# Load full log once; preserve original row ids for deletion mapping
log_full = io_helpers.load_queue_audit()
if log_full.empty:
    st.info("No audit records yet.")
    st.stop()

log_full = log_full.reset_index().rename(columns={"index": "RowId"})  # stable id per load
log_full["Timestamp"] = pd.to_datetime(log_full["Timestamp"], errors="coerce")

# ---------- Filters ----------
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

# ---------- Table with selection ----------
df = df.sort_values("Timestamp", ascending=False).reset_index(drop=True)

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

# ---------- Bulk actions ----------
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
            st.rerun()

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
            st.rerun()

st.caption("Tip: Use filters (Event/Period/Search) to narrow, then **Delete ALL shown** to clear older records quickly.")

