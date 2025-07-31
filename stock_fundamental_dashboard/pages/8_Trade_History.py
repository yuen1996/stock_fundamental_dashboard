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

# ---- safe rerun for all Streamlit versions ----
def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

st.set_page_config(layout="wide")
st.title("📘 Trade History")

full = io_helpers.load_closed_trades()
if full.empty:
    st.info("No closed trades yet.")
    st.stop()

# Preserve a RowId for deletion mapping
full = full.reset_index().rename(columns={"index": "RowId"})

# ---------- Filters ----------
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

if search.strip():
    q = search.lower()
    flt = flt[
        flt["Name"].astype(str).str.lower().str.contains(q, na=False)
        | flt["CloseReason"].astype(str).str.lower().str.contains(q, na=False)
        | flt["Reasons"].astype(str).str.lower().str.contains(q, na=False)
    ]

# ---------- KPIs ----------
wins = (pd.to_numeric(flt["PnL"], errors="coerce") > 0).sum()
loss = (pd.to_numeric(flt["PnL"], errors="coerce") <= 0).sum()
total_pnl = pd.to_numeric(flt["PnL"], errors="coerce").sum()
avg_r = pd.to_numeric(flt["RMultiple"], errors="coerce").mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Trades", len(flt))
k2.metric("Win Rate", f"{(wins / max(len(flt),1))*100:.1f}%")
k3.metric("Total PnL (MYR)", f"{(total_pnl or 0):,.2f}")
k4.metric("Avg R multiple", f"{(avg_r or 0):.2f}")

# ---------- Table with selection + bulk delete ----------
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

c1, c2, c3 = st.columns([1.6, 2, 3])
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

