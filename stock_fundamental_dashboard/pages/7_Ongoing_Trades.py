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

st.set_page_config(layout="wide")
st.title("📈 Ongoing Trades")

open_df = io_helpers.load_open_trades()
if open_df.empty:
    st.info("No ongoing trades. Use **Systematic Decision → Manage Queue → Mark Live** to open a position.")
    st.stop()

# KPIs
try:
    shares = pd.to_numeric(open_df.get("Shares"), errors="coerce")
    entry  = pd.to_numeric(open_df.get("Entry"),  errors="coerce")
    cost   = (shares * entry).fillna(0).sum()
    k1, k2 = st.columns(2)
    k1.metric("Open Positions", len(open_df))
    k2.metric("Total Cost (MYR)", f"{cost:,.2f}")
except Exception:
    pass

st.subheader("Open Positions")

# Table with action columns
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

c1, c2, c3 = st.columns([1.4, 2, 2])
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
            try: st.rerun()
            except Exception: st.experimental_rerun()
        if invalid and not closed:
            st.warning(f"{invalid} selected row(s) invalid (need Close Price and/or Detail for 'Other').")


