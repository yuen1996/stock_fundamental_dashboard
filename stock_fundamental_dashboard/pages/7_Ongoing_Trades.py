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

# Optional KPIs
try:
    shares = pd.to_numeric(open_df.get("Shares"), errors="coerce")
    entry  = pd.to_numeric(open_df.get("Entry"),  errors="coerce")
    cost   = (shares * entry).fillna(0).sum()
    st.metric("Open Positions", len(open_df))
    st.metric("Total Cost (MYR)", f"{cost:,.2f}")
except Exception:
    pass

st.subheader("Open Positions")
st.dataframe(open_df, use_container_width=True, height=360)

st.markdown("### Close a Position (requires Close Price + Reason)")
CLOSE_REASONS = [
    "Target hit",
    "Stop hit",
    "Trailing stop",
    "Time stop",
    "Thesis changed",
    "Portfolio rebalance",
    "Other (specify)",
]

for i, row in open_df.reset_index(drop=True).iterrows():
    with st.container():
        c1, c2, c3, c4, c5, c6 = st.columns([2, 1.2, 1.2, 1.2, 1.6, 1.4])
        name = row.get("Name")
        strat= row.get("Strategy")
        entry= row.get("Entry")
        shares = row.get("Shares")
        c1.markdown(f"**{name}**  \n_Strategy:_ {strat}")
        c2.write(f"Entry: {entry:,.4f}" if pd.notna(entry) else "Entry: N/A")
        c3.write(f"Shares: {int(shares):,}" if pd.notna(shares) else "Shares: N/A")

        # Close controls
        close_px = c4.number_input("Close Price", min_value=0.0, step=0.001, format="%.4f", key=f"close_px_{i}_{name}_{strat}")
        reason_sel = c5.selectbox("Reason", CLOSE_REASONS, index=0, key=f"close_reason_{i}_{name}_{strat}")

        if reason_sel == "Other (specify)":
            detail = c6.text_input("Detail", key=f"close_detail_{i}_{name}_{strat}")
            can_close = (close_px > 0) and bool(detail.strip())
            if c6.button("Close Trade", key=f"btn_close_other_{i}_{name}_{strat}", disabled=not can_close):
                reason_text = f"{reason_sel}: {detail.strip()}"
                ok = io_helpers.close_open_trade(
                    name=name, strategy=strat,
                    close_price=close_px,
                    close_reason=reason_text
                )
                if ok:
                    st.success(f"Closed: {name} ({strat}) @ {close_px:.4f}")
                    try: st.rerun()
                    except Exception: st.experimental_rerun()
                else:
                    st.error("Failed to close trade.")
        else:
            # Non-"Other" reasons don't require extra detail
            if c6.button("Close Trade", key=f"btn_close_{i}_{name}_{strat}", disabled=(close_px <= 0)):
                ok = io_helpers.close_open_trade(
                    name=name, strategy=strat,
                    close_price=close_px,
                    close_reason=reason_sel
                )
                if ok:
                    st.success(f"Closed: {name} ({strat}) @ {close_px:.4f}")
                    try: st.rerun()
                    except Exception: st.experimental_rerun()
                else:
                    st.error("Failed to close trade.")

