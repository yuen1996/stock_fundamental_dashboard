# 7_Ongoing_Trades.py  – row-exact close logic
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
except Exception:        # fallback if running as flat repo
    import io_helpers     # type: ignore

# ---------- Page setup ----------
st.set_page_config(page_title="Ongoing Trades", layout="wide")

# =========== Unified CSS (same as Dashboard) ===========
BASE_CSS = """<style>
/* (identical CSS you already use – kept for brevity) */
</style>"""
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

open_df = open_df.reset_index().rename(columns={"index": "RowId"})  # unique row key

# ─────────────────────────────────────────
# Overview (KPIs)
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec"><div class="t">📊 Overview</div>'
    '<div class="d">Open positions &amp; exposure</div></div>',
    unsafe_allow_html=True
)

shares = pd.to_numeric(open_df.get("Shares"), errors="coerce")
entry  = pd.to_numeric(open_df.get("Entry"),  errors="coerce")
total_cost = (shares * entry).fillna(0).sum()

k1, k2 = st.columns(2)
k1.metric("Open Positions", len(open_df))
k2.metric("Total Cost (MYR)", f"{total_cost:,.2f}")

# ─────────────────────────────────────────
# Editor table
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

# build editable table
table = open_df.copy()
table.insert(0, "Select", False)
table["ClosePrice"]  = 0.0
table["CloseReason"] = CLOSE_REASONS[0]
table["Detail"]      = ""

edited = st.data_editor(
    table,
    use_container_width=True,
    height=440,
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


