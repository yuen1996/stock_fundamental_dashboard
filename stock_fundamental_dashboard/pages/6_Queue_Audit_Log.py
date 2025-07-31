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

log = io_helpers.load_queue_audit()
if log.empty:
    st.info("No audit records yet.")
    st.stop()

# Filters
colA, colB, colC = st.columns([1, 2, 2])
with colA:
    event = st.selectbox("Event", ["All", "UPSERT", "DELETE"], index=0)
with colB:
    search = st.text_input("Search Name / Strategy / Reasons", "")
with colC:
    period = st.selectbox("Period", ["All", "Last 7 days", "Last 30 days", "Last 90 days"], index=2)

df = log.copy()
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

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
    ]

# ordering and display
df = df.sort_values("Timestamp", ascending=False).reset_index(drop=True)

cols = [
    "Timestamp", "Event", "Name", "Strategy",
    "Score", "CurrentPrice", "Entry", "Stop", "Take", "Shares", "RR",
    "TP1", "TP2", "TP3",
    "Reasons", "AuditReason",
]
cols = [c for c in cols if c in df.columns]
st.dataframe(df[cols], use_container_width=True, height=520)

st.caption("This log records every queue **UPSERT** (add/update) and **DELETE** with the chosen reason and key plan ratios.")
