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

try:
    from utils import io_helpers
except Exception:
    import io_helpers

st.set_page_config(layout="wide")
st.title("📘 Trade History")

df = io_helpers.load_closed_trades()
if df.empty:
    st.info("No closed trades yet.")
    st.stop()

# Filters
colA, colB, colC, colD = st.columns([1, 1, 1, 2])
with colA:
    strat = st.selectbox("Strategy", ["All"] + sorted(df["Strategy"].dropna().unique()), index=0)
with colB:
    period = st.selectbox("Period", ["All", "YTD", "Last 30 days", "Last 90 days", "Last 1 year"], index=1)
with colC:
    min_rr = st.slider("Min RR_Init", 0.0, 5.0, 0.0, 0.1)

flt = df.copy()
flt["CloseDate"] = pd.to_datetime(flt["CloseDate"], errors="coerce")

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
    flt = flt[flt["CloseDate"] >= cutoff]

flt["RR_Init"] = pd.to_numeric(flt["RR_Init"], errors="coerce")
flt = flt[flt["RR_Init"].fillna(0) >= min_rr]

# KPIs
wins = (pd.to_numeric(flt["PnL"], errors="coerce") > 0).sum()
loss = (pd.to_numeric(flt["PnL"], errors="coerce") <= 0).sum()
total_pnl = pd.to_numeric(flt["PnL"], errors="coerce").sum()
avg_r = pd.to_numeric(flt["RMultiple"], errors="coerce").mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Trades", len(flt))
k2.metric("Win Rate", f"{(wins / max(len(flt),1))*100:.1f}%")
k3.metric("Total PnL (MYR)", f"{(total_pnl or 0):,.2f}")
k4.metric("Avg R multiple", f"{(avg_r or 0):.2f}")

cols = [
    "CloseDate", "Name", "Strategy",
    "Entry", "Stop", "Take", "Shares", "RR_Init",
    "ClosePrice", "HoldingDays", "PnL", "ReturnPct", "RMultiple",
    "CloseReason", "Reasons",
]
cols = [c for c in cols if c in flt.columns]
st.dataframe(flt.sort_values("CloseDate", ascending=False)[cols], use_container_width=True, height=520)
