# --- path patch so this page can import from project root ---
import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))  # project root (parent of /pages)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from math import floor

# --- robust imports: prefer package (utils), fall back to top-level ---
try:
    from utils import io_helpers, calculations, rules
except Exception:
    import io_helpers, calculations, rules


st.set_page_config(layout="wide")
st.title("📐 Risk / Reward Planner")

# =============== Helpers ===============
def fmt(x, d=4, pct=False):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "N/A"
    try:
        return (f"{x:,.{d}f}%" if pct else f"{x:,.{d}f}")
    except Exception:
        try:
            x = float(x)
            return (f"{x:,.{d}f}%" if pct else f"{x:,.{d}f}")
        except Exception:
            return "N/A"

def latest_current_price(stock_rows: pd.DataFrame) -> float | None:
    """Prefer CurrentPrice across any row; else latest annual SharePrice."""
    cur_val = None
    if "CurrentPrice" in stock_rows.columns:
        s = stock_rows["CurrentPrice"].dropna()
        if not s.empty:
            cur_val = s.iloc[-1]
    if cur_val is None:
        annual = stock_rows[stock_rows.get("IsQuarter", False) != True]
        if "SharePrice" in annual.columns and not annual.empty:
            s2 = annual.sort_values("Year")["SharePrice"].dropna()
            if not s2.empty:
                cur_val = s2.iloc[-1]
    try:
        return float(cur_val) if cur_val is not None else None
    except Exception:
        return None

def latest_annual_metrics(latest_annual_row: pd.Series | None) -> dict:
    if latest_annual_row is None:
        return {}
    return calculations.calc_ratios(latest_annual_row)

def get_latest_annual_row(stock_rows: pd.DataFrame) -> pd.Series | None:
    annual = stock_rows[stock_rows.get("IsQuarter", False) != True].copy()
    if annual.empty or "Year" not in annual.columns:
        return None
    annual = annual.dropna(subset=["Year"])
    if annual.empty:
        return None
    annual = annual.sort_values("Year")
    return annual.iloc[-1]

def valid_prices(entry, stop, take):
    """Check long setup: stop < entry < take."""
    if entry is None or stop is None or take is None:
        return False, "Please provide Entry, Stop, and Take-Profit."
    if stop >= entry:
        return False, "Stop-loss must be **below** Entry for a long setup."
    if take <= entry:
        return False, "Take-profit must be **above** Entry for a long setup."
    return True, ""

# =============== Data load ===============
df = io_helpers.load_data()
if df is None or df.empty or "Name" not in df.columns:
    st.warning("No data found. Please add stocks in **Add/Edit** first.")
    st.stop()

stocks = sorted([s for s in df["Name"].dropna().unique()])
left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("1) Pick a Stock & Strategy")
    colA, colB = st.columns([2, 1])
    with colA:
        stock_name = st.selectbox("Stock", options=stocks, index=0 if stocks else None, key="rr_stock")
    with colB:
        strategy = st.selectbox("Strategy", options=list(rules.RULESETS.keys()), index=0, key="rr_strategy")

    stock_rows = df[df["Name"] == stock_name].sort_values(["Year"])
    price_now = latest_current_price(stock_rows)
    latest_row = get_latest_annual_row(stock_rows)
    metrics = latest_annual_metrics(latest_row)
    ev = rules.evaluate(metrics, strategy) if metrics else {"score": 0, "pass": False, "reasons": [], "mandatory": [], "scored": []}

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Current Price", fmt(price_now, 4) if price_now is not None else "N/A")
    with c2:
        st.metric("Decision Score", f"{ev['score']}%")
    with c3:
        st.metric("Decision Status", "✅ PASS" if ev["pass"] else "❌ REJECT")
    with c4:
        if not ev["pass"] and ev["reasons"]:
            st.write("Unmet: " + "; ".join(ev["reasons"]))
        else:
            st.write("Looks OK by chosen strategy.")

with right:
    st.subheader("Help")
    st.markdown(
        """
- **Entry**: your intended buy price (default = current).
- **Stop‑loss**: below entry (fixed % or manual).
- **Take‑profit**: above entry (R multiple or manual).
- **R : R** = (Take − Entry) / (Entry − Stop).
- **Position size** caps the **$ risk** at your chosen risk %.
"""
    )

st.divider()

# =============== Risk & Reward Inputs ===============
st.subheader("2) Define Your Entry / Stop / Take & Risk")
cA, cB, cC, cD = st.columns(4)

default_entry = price_now if price_now is not None else 0.0
with cA:
    entry = st.number_input("Entry Price", min_value=0.0, value=float(default_entry), step=0.001, format="%.4f", key="rr_entry")

with cB:
    stop_mode = st.radio("Stop-loss Mode", ["% below entry", "Manual price"], horizontal=True, key="rr_stopmode")
    if stop_mode == "% below entry":
        stop_pct = st.number_input("Stop % below Entry", min_value=0.1, max_value=50.0, value=8.0, step=0.1, format="%.1f", key="rr_stoppct")
        stop = entry * (1 - stop_pct / 100.0)
    else:
        stop = st.number_input("Stop-loss Price", min_value=0.0, value=max(0.0, entry * 0.92), step=0.001, format="%.4f", key="rr_stop")

with cC:
    take_mode = st.radio("Take‑profit Mode", ["R multiple", "Manual price"], horizontal=True, key="rr_tpmode")
    if take_mode == "R multiple":
        r_multiple = st.number_input("Target (R)", min_value=0.5, max_value=10.0, value=2.0, step=0.5, format="%.1f", key="rr_rmult")
        risk_per_share = max(entry - stop, 0.0)
        take = entry + r_multiple * risk_per_share if risk_per_share > 0 else entry
    else:
        take = st.number_input("Take‑profit Price", min_value=0.0, value=float(entry * 1.15 if entry else 0.0), step=0.001, format="%.4f", key="rr_take")

with cD:
    acct = st.number_input("Account Size (MYR)", min_value=0.0, value=10000.0, step=100.0, format="%.2f", key="rr_acct")
    risk_pct = st.number_input("Risk % per Trade", min_value=0.1, max_value=5.0, value=1.0, step=0.1, format="%.1f", key="rr_riskpct")
    lot_size = st.number_input("Lot Size (shares)", min_value=1, value=100, step=1, key="rr_lot")

# =============== Calculations ===============
ok, msg = valid_prices(entry, stop, take)
if not ok:
    st.warning(msg)
    st.stop()

risk_per_share = entry - stop
reward_per_share = take - entry
rr = (reward_per_share / risk_per_share) if risk_per_share > 0 else None

cash_risk = acct * (risk_pct / 100.0)
raw_shares = floor(cash_risk / risk_per_share) if risk_per_share > 0 else 0
# round down to nearest lot
shares = max((raw_shares // lot_size) * lot_size, 0)
position_cost = shares * entry
potential_loss = shares * (entry - stop)
potential_gain = shares * (take - entry)

# =============== Output Cards ===============
st.subheader("3) Result")
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.metric("Stop‑loss", fmt(stop, 4))
with k2:
    st.metric("Take‑profit", fmt(take, 4))
with k3:
    st.metric("R : R", f"{rr:.2f}" if rr is not None and np.isfinite(rr) else "N/A")
with k4:
    st.metric("Shares", f"{shares:,d}")
with k5:
    st.metric("Position Cost", fmt(position_cost, 2))
with k6:
    st.metric("Risk (MYR)", fmt(potential_loss, 2))

# detail table
detail = pd.DataFrame(
    {
        "Entry": [entry],
        "Stop": [stop],
        "Take": [take],
        "Risk/Share": [risk_per_share],
        "Reward/Share": [reward_per_share],
        "R:R": [rr],
        "Shares": [shares],
        "Cost": [position_cost],
        "Potential Loss": [potential_loss],
        "Potential Gain": [potential_gain],
    }
).T
detail.columns = [stock_name]
st.dataframe(detail.round(4), use_container_width=True, height=240)

# =============== Integrate with Trade Queue ===============
st.subheader("4) Save Plan")
disabled = not (shares > 0 and np.isfinite(risk_per_share) and risk_per_share > 0)

reason_lines = [
    f"Plan for {stock_name} ({strategy})",
    f"Entry={entry:.4f}, Stop={stop:.4f}, Take={take:.4f}",
    f"R:R={rr:.2f}" if rr is not None and np.isfinite(rr) else "R:R=N/A",
    f"Shares={shares:,d}, Cost={position_cost:,.2f}, Risk={potential_loss:,.2f}, Gain={potential_gain:,.2f}",
]
reason_text = " | ".join(reason_lines)

colL, colR = st.columns([1, 1])
with colL:
    st.text_area("Plan summary (stored as `Reasons` in Trade Queue)", value=reason_text, height=80, key="rr_reason")
with colR:
    if st.button("➕ Add to Trade Queue", type="primary", use_container_width=True, disabled=disabled):
        try:
            io_helpers.push_trade_candidate(
                name=stock_name,
                strategy=strategy,
                score=ev.get("score", 0),
                current_price=price_now if price_now is not None else entry,
                reasons=st.session_state["rr_reason"],
            )
            st.success("Added to Trade Queue.")
        except Exception as e:
            st.error(f"Failed to add to Trade Queue: {e}")

st.caption("Tip: Adjust lot size to your market (e.g., 100 shares on Bursa). This page does not change schemas; plans are saved into the existing Trade Queue via the 'Reasons' field.")
