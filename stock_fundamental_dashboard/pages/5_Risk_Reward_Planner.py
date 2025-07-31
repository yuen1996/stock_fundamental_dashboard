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

# ==== Query params (prefill support) ====
try:
    qp = dict(st.query_params)
except Exception:
    qp = {k: v[0] for k, v in st.experimental_get_query_params().items()}  # Streamlit < 1.30 fallback

def qget(name, cast=None, default=None):
    v = qp.get(name, default)
    if v is None:
        return default
    if cast is None:
        return v
    try:
        return cast(v)
    except Exception:
        return default

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

def rr_band(rr):
    """Return (label, color) by R:R."""
    if rr is None or not np.isfinite(rr):
        return "N/A", "#64748b"  # slate
    if rr < 1.5:
        return "Low (<1.5)", "#ef4444"  # red
    if rr < 2.0:
        return "OK (1.5–2.0)", "#f59e0b"  # amber
    return "Good (≥2.0)", "#16a34a"      # green

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

# Prefill selections from query params
stock_q   = qget("stock",   str, None)
strategy_q= qget("strategy",str, None)
entry_q   = qget("entry",   float, None)
stop_q    = qget("stop",    float, None)
take_q    = qget("take",    float, None)
r_q       = qget("r",       float, None)
acct_q    = qget("acct",    float, None)
risk_q    = qget("riskpct", float, None)
lot_q     = qget("lot",     int,   None)

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("1) Pick a Stock & Strategy")
    colA, colB = st.columns([2, 1])

    # Stock
    if stock_q and stock_q in stocks:
        stock_index = stocks.index(stock_q)
    else:
        stock_index = 0 if stocks else None
    with colA:
        stock_name = st.selectbox("Stock", options=stocks, index=stock_index, key="rr_stock")

    # Strategy
    strategies = list(rules.RULESETS.keys())
    if strategy_q and strategy_q in strategies:
        strat_index = strategies.index(strategy_q)
    else:
        strat_index = 0
    with colB:
        strategy = st.selectbox("Strategy", options=strategies, index=strat_index, key="rr_strategy")

    # Lookups
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
- **Entry**: intended buy price (default = current).
- **Stop-loss**: % below entry, **ATR-based**, or manual price.
- **Take-profit**: by R multiple or manual price.
- **R : R** = (Take − Entry) / (Entry − Stop).
- **Position size** caps the **MYR risk** at your chosen risk %.
"""
    )

st.divider()

# Try to fetch latest ATR if OHLC exists
atr_period_default = 14
atr_val, atr_date = io_helpers.latest_atr(stock_q or "", period=atr_period_default) if (stock_q or stock_index is not None) else (None, None)
atr_available = atr_val is not None

# =============== Risk & Reward Inputs ===============
st.subheader("2) Define Your Entry / Stop / Take & Risk")
cA, cB, cC, cD = st.columns(4)

default_entry = entry_q if entry_q is not None else (price_now if price_now is not None else 0.0)
with cA:
    entry = st.number_input("Entry Price", min_value=0.0, value=float(default_entry), step=0.001, format="%.4f", key="rr_entry")

with cB:
    stop_modes = ["% below entry", "Manual price"] + (["ATR-based"] if atr_available else [])
    stop_mode = st.radio("Stop-loss Mode", stop_modes, horizontal=True, key="rr_stopmode")

    if stop_mode == "% below entry" and stop_q is None:
        stop_pct = st.number_input("Stop % below Entry", min_value=0.1, max_value=50.0, value=8.0, step=0.1, format="%.1f", key="rr_stoppct")
        stop = entry * (1 - stop_pct / 100.0)
    elif stop_mode == "Manual price" or (stop_q is not None and stop_mode != "ATR-based"):
        stop_default = stop_q if stop_q is not None else max(0.0, entry * 0.92)
        stop = st.number_input("Stop-loss Price", min_value=0.0, value=float(stop_default), step=0.001, format="%.4f", key="rr_stop")
    else:
        # ATR-based
        st.markdown(f"**ATR({atr_period_default})** latest ≈ **{fmt(atr_val, 4)}**"
                    + (f" (as of {pd.to_datetime(atr_date).date()})" if atr_date is not None else ""))
        atr_mult = st.number_input("ATR Multiplier", min_value=0.5, max_value=5.0, value=2.0, step=0.5, format="%.1f", key="rr_atr_mult")
        stop = max(0.0, entry - (atr_val or 0) * atr_mult) if atr_available else max(0.0, entry * 0.92)

with cC:
    take_mode = st.radio("Take-profit Mode", ["R multiple", "Manual price"], horizontal=True, key="rr_tpmode")
    if take_mode == "R multiple" and (take_q is None):
        r_default = r_q if r_q is not None else 2.0
        r_multiple = st.number_input("Target (R)", min_value=0.5, max_value=10.0, value=float(r_default), step=0.5, format="%.1f", key="rr_rmult")
        risk_per_share = max(entry - stop, 0.0)
        take = entry + r_multiple * risk_per_share if risk_per_share > 0 else entry
    else:
        take_default = take_q if take_q is not None else (entry * 1.15 if entry else 0.0)
        take = st.number_input("Take-profit Price", min_value=0.0, value=float(take_default), step=0.001, format="%.4f", key="rr_take")

with cD:
    acct = st.number_input("Account Size (MYR)", min_value=0.0, value=float(acct_q if acct_q is not None else 10000.0), step=100.0, format="%.2f", key="rr_acct")
    risk_pct = st.number_input("Risk % per Trade", min_value=0.1, max_value=5.0, value=float(risk_q if risk_q is not None else 1.0), step=0.1, format="%.1f", key="rr_riskpct")
    lot_size = st.number_input("Lot Size (shares)", min_value=1, value=int(lot_q if lot_q is not None else 100), step=1, key="rr_lot")

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

# Risk band badge + trade viability
band_label, band_color = rr_band(rr)
viability = ("✅ Ready" if (rr is not None and np.isfinite(rr) and rr >= 1.5 and shares > 0)
             else "⚠️ Low R or zero size" if (rr is not None and rr < 1.5)
             else "⏳ Incomplete")
st.markdown(
    f"""<div style="display:inline-block;padding:.25rem .6rem;border-radius:999px;
        background:{band_color};color:white;font-weight:700;margin:.25rem 0;">
        R:R Band — {band_label}</div>""",
    unsafe_allow_html=True
)
st.write(f"**Trade Viability:** {viability}")

# Multi-target TPs (R multiples)
st.subheader("2a) Optional Multi-Target Take-Profits")
m1, m2, m3 = st.columns(3)
with m1:
    tp1_r = st.number_input("TP1 (R)", min_value=0.25, max_value=10.0, value=1.0, step=0.25, format="%.2f", key="tp1r")
with m2:
    tp2_r = st.number_input("TP2 (R)", min_value=0.25, max_value=10.0, value=2.0, step=0.25, format="%.2f", key="tp2r")
with m3:
    tp3_r = st.number_input("TP3 (R)", min_value=0.25, max_value=10.0, value=3.0, step=0.25, format="%.2f", key="tp3r")

TP1 = entry + tp1_r * risk_per_share
TP2 = entry + tp2_r * risk_per_share
TP3 = entry + tp3_r * risk_per_share

tp_table = pd.DataFrame({
    "Target": ["TP1", "TP2", "TP3"],
    "Multiple (R)": [tp1_r, tp2_r, tp3_r],
    "Price": [TP1, TP2, TP3],
    "Gain/Share": [TP1 - entry, TP2 - entry, TP3 - entry],
})
st.dataframe(tp_table.round(4), use_container_width=True, height=180)

# =============== Output Cards ===============
st.subheader("3) Result")
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.metric("Stop-loss", fmt(stop, 4))
with k2:
    st.metric("Take-profit", fmt(take, 4))
with k3:
    st.metric("R : R", f"{rr:.2f}" if rr is not None and np.isfinite(rr) else "N/A")
with k4:
    st.metric("Shares", f"{shares:,d}")
with k5:
    st.metric("Position Cost", fmt(position_cost, 2))
with k6:
    st.metric("Risk (MYR)", fmt(potential_loss, 2))

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
        "TP1": [TP1],
        "TP2": [TP2],
        "TP3": [TP3],
        "Viability": [viability],
    }
).T
detail.columns = [stock_name]
st.dataframe(detail.round(4), use_container_width=True, height=260)

# =============== Integrate with Trade Queue ===============
st.subheader("4) Save Plan")
disabled = not (shares > 0 and np.isfinite(risk_per_share) and risk_per_share > 0)

reason_lines = [
    f"Plan for {stock_name} ({strategy})",
    f"Entry={entry:.4f}, Stop={stop:.4f}, Take={take:.4f}",
    f"R:R={rr:.2f}" if rr is not None and np.isfinite(rr) else "R:R=N/A",
    f"Shares={shares:,d}, Cost={position_cost:,.2f}, Risk={potential_loss:,.2f}, Gain={potential_gain:,.2f}",
    f"TP1={TP1:.4f}, TP2={TP2:.4f}, TP3={TP3:.4f}",
    f"Band={band_label}, Viability={viability}",
]
reason_text = " | ".join(reason_lines)

colL, colR = st.columns([1, 1])
with colL:
    st.text_area("Plan summary (stored as `Reasons`)", value=reason_text, height=90, key="rr_reason")
with colR:
    if st.button("➕ Add to Trade Queue", type="primary", use_container_width=True, disabled=disabled):
        try:
            io_helpers.push_trade_candidate(
                name=stock_name,
                strategy=strategy,
                score=ev.get("score", 0),
                current_price=price_now if price_now is not None else entry,
                reasons=st.session_state["rr_reason"],
                entry=entry, stop=stop, take=take, shares=shares, rr=rr,
                tp1=TP1, tp2=TP2, tp3=TP3,
            )
            st.success("Added to Trade Queue.")
        except Exception as e:
            st.error(f"Failed to add to Trade Queue: {e}")

st.caption("ATR-based stop is available when OHLC data is present in data/ohlc/<Name>.csv or data/ohlc.csv. Risk band colors: <1.5R = red, 1.5–2.0R = amber, ≥2.0R = green.")

