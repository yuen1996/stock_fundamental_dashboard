# --- make project root importable so we can import io_helpers/calculations/rules ---
import os, sys
_THIS = os.path.dirname(__file__)
_PARENT = os.path.abspath(os.path.join(_THIS, ".."))         # .../stock_fundamental_dashboard
_GRANDP = os.path.abspath(os.path.join(_THIS, "..", ".."))   # repo root (one level above)

for p in (_PARENT, _GRANDP):
    if p not in sys.path:
        sys.path.insert(0, p)

# now these imports will work whether files are in the package root or repo root
try:
    import io_helpers, calculations, rules
except ModuleNotFoundError:
    from utils import io_helpers, calculations, rules  # fallback if you move them under utils/


# 5_Risk_Reward_Planner.py — queue-aware version with unified CSS + ATR + multi-TP
# --- path patch so this page can import from project root ---
import os, sys, math
ROOT = os.path.dirname(os.path.dirname(__file__))  # project root (parent of /pages)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np

# --- robust imports: prefer package (utils), fall back to top-level ---
try:
    from utils import io_helpers, calculations, rules
except Exception:
    import io_helpers, calculations, rules

# ---------- Page setup ----------
st.set_page_config(page_title="Risk / Reward Planner", layout="wide")

# === Unified CSS (same as Dashboard; fonts 16px) ===
BASE_CSS = """
<style>
:root{
  --bg:#f6f7fb;               /* app background */
  --surface:#ffffff;          /* cards/tables background */
  --text:#0f172a;             /* main text */
  --muted:#475569;            /* secondary text */
  --border:#e5e7eb;           /* card & table borders */
  --shadow:0 8px 24px rgba(15, 23, 42, .06);

  /* accent colors for section stripes */
  --primary:#4f46e5;          /* indigo */
  --info:#0ea5e9;             /* sky   */
  --success:#10b981;          /* green */
  --warning:#f59e0b;          /* amber */
  --danger:#ef4444;           /* red   */
}
html, body, [class*="css"]{
  font-size:16px !important; color:var(--text);
}
.stApp{
  background: radial-gradient(1000px 500px at 10% -10%, #f0f3fb 0%, var(--bg) 60%), var(--bg);
}
h1, h2, h3, h4{
  color:var(--text) !important; font-weight:800 !important; letter-spacing:.2px;
}

/* Section header card (visual separators) */
.sec{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:14px;
  box-shadow:var(--shadow);
  padding:.65rem .9rem;
  margin:1rem 0 .6rem 0;
  display:flex; align-items:center; gap:.6rem;
}
.sec .t{ font-size:1.05rem; font-weight:800; margin:0; padding:0; }
.sec .d{ color:var(--muted); font-size:.95rem; margin-left:.25rem; }
.sec::before{
  content:""; display:inline-block;
  width:8px; height:26px; border-radius:6px; background:var(--primary);
}
.sec.info::before    { background:var(--info); }
.sec.success::before { background:var(--success); }
.sec.warning::before { background:var(--warning); }
.sec.danger::before  { background:var(--danger); }

/* Tables / editors */
.stDataFrame, .stDataEditor{ font-size:15px !important; }
div[data-testid="stDataFrame"] table, div[data-testid="stDataEditor"] table{
  border-collapse:separate !important; border-spacing:0;
}
div[data-testid="stDataFrame"] table tbody tr:hover td,
div[data-testid="stDataEditor"] table tbody tr:hover td{
  background:#f8fafc !important;
}
div[data-testid="stDataFrame"] td, div[data-testid="stDataEditor"] td{
  border-bottom:1px solid var(--border) !important;
}

/* Inputs & buttons */
div[data-baseweb="input"] input, textarea, .stNumberInput input{ font-size:15px !important; }
.stSlider > div [data-baseweb="slider"]{ margin-top:.25rem; }
.stButton>button{ border-radius:12px !important; padding:.55rem 1.1rem !important; font-weight:700; }

/* Tabs */
.stTabs [role="tab"]{ font-size:15px !important; font-weight:600 !important; }

/* Sidebar theme (dark, same as Dashboard) */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg, #0b1220 0%, #1f2937 100%) !important;
}
[data-testid="stSidebar"] *{ color:#e5e7eb !important; }
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

st.header("📐 Risk / Reward Planner")

# ========== 0) Pick an existing queued row (or create new) ==========
queue_df = io_helpers.load_trade_queue().reset_index().rename(columns={"index": "RowId"})
queue_options = ["— New Plan —"] + [
    f"{int(r.RowId)} — {r.Name} ({r.Strategy})"
    for _, r in queue_df.iterrows()
]
sel = st.selectbox("Queued idea to edit (or choose “New Plan”)", queue_options, index=0)
editing_rowid = None
prefill = {}
if sel != "— New Plan —" and len(queue_df):
    try:
        editing_rowid = int(sel.split(" — ")[0])
        r = queue_df.loc[queue_df.RowId == editing_rowid].iloc[0]
        prefill = {
            "stock":    r.Name,
            "strategy": r.Strategy,
            "entry":    r.Entry,
            "stop":     r.Stop,
            "take":     r.Take,
            "r":        r.RR,
        }
    except Exception:
        editing_rowid = None
        prefill = {}

# ======= Query params fallback (old links still work) =======
try:
    qp = dict(st.query_params)
except Exception:
    qp = {k: v[0] for k, v in st.experimental_get_query_params().items()}

def qget(k, cast=None, default=None):
    """queued-row values override URL params"""
    if k in prefill and prefill[k] is not None and not (isinstance(prefill[k], float) and np.isnan(prefill[k])):
        return prefill[k]
    v = qp.get(k, default)
    if v is None: return default
    if cast is None: return v
    try: return cast(v)
    except Exception: return default

# =============== Helpers ===============
def fmt(x, d=4, pct=False):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "N/A"
    try:
        return (f"{x:,.{d}f}%" if pct else f"{x:,.{d}f}")
    except Exception:
        return "N/A"

def rr_band(rr):
    if rr is None or not np.isfinite(rr): return "N/A", "#64748b"
    if rr < 1.5:  return "Low (<1.5)", "#ef4444"
    if rr < 2.0:  return "OK (1.5–2.0)", "#f59e0b"
    return "Good (≥2.0)", "#16a34a"

def latest_current_price(stock_rows: pd.DataFrame) -> float | None:
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

def get_latest_annual_row(stock_rows: pd.DataFrame) -> pd.Series | None:
    annual = stock_rows[stock_rows.get("IsQuarter", False) != True].copy()
    if annual.empty or "Year" not in annual.columns:
        return None
    annual = annual.dropna(subset=["Year"]).sort_values("Year")
    return annual.iloc[-1] if not annual.empty else None

# =============== Data load ===============
df = io_helpers.load_data()
if df is None or df.empty or "Name" not in df.columns:
    st.warning("No data found. Please add stocks in **Add/Edit** first.")
    st.stop()

stocks = sorted([s for s in df["Name"].dropna().unique()])

# Prefill selections
stock_q    = qget("stock",    str, None)
strategy_q = qget("strategy", str, None)
entry_q    = qget("entry",    float, None)
stop_q     = qget("stop",     float, None)
take_q     = qget("take",     float, None)
r_q        = qget("r",        float, None)

# ─────────────────────────────────────────
# 1) Pick a Stock & Strategy
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec"><div class="t">1) Pick a Stock & Strategy</div>'
    '<div class="d">Use an existing queued idea or start a new plan</div></div>',
    unsafe_allow_html=True
)

colA, colB = st.columns([2, 1])

with colA:
    stock_index = stocks.index(stock_q) if (stock_q in stocks) else 0
    stock_name = st.selectbox("Stock", options=stocks, index=stock_index, key="rr_stock")

with colB:
    strategies = list(rules.RULESETS.keys())
    strat_index = strategies.index(strategy_q) if (strategy_q in strategies) else 0
    strategy = st.selectbox("Strategy", options=strategies, index=strat_index, key="rr_strategy")

# Lookups for current price & decision context
stock_rows = df[df["Name"] == stock_name].sort_values(["Year"])
price_now  = latest_current_price(stock_rows)
latest_row = get_latest_annual_row(stock_rows)
metrics    = calculations.calc_ratios(latest_row) if latest_row is not None else {}
ev         = rules.evaluate(metrics, strategy) if metrics else {"score": 0, "pass": False, "reasons": []}

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Current Price", (f"{price_now:,.4f}" if price_now is not None else "N/A"))
with c2: st.metric("Decision Score", f"{ev.get('score',0)}%")
with c3: st.metric("Decision Status", "✅ PASS" if ev.get("pass") else "❌ REJECT")
with c4:
    if (not ev.get("pass")) and ev.get("reasons"):
        st.write("Unmet: " + "; ".join(ev["reasons"]))
    else:
        st.write("Looks OK by chosen strategy.")

st.divider()

# Try to fetch latest ATR if OHLC helper exists
atr_fn = getattr(io_helpers, "latest_atr", None)
atr_period_default = 14
atr_val, atr_date = (atr_fn(stock_name, period=atr_period_default) if callable(atr_fn) else (None, None))
atr_available = atr_val is not None

# ─────────────────────────────────────────
# 2) Define Entry / Stop / Take & Risk
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec success"><div class="t">2) Define Your Entry / Stop / Take & Risk</div>'
    '<div class="d">ATR stop available when OHLC is present</div></div>',
    unsafe_allow_html=True
)

cA, cB, cC, cD = st.columns(4)

default_entry = (
    float(entry_q) if entry_q is not None else
    float(prefill.get("entry")) if prefill.get("entry") is not None and not pd.isna(prefill.get("entry")) else
    float(price_now) if price_now is not None else 0.0
)

with cA:
    entry = st.number_input("Entry Price", min_value=0.0, value=default_entry, step=0.001, format="%.4f", key="rr_entry")

with cB:
    stop_modes = ["% below entry", "Manual price"] + (["ATR-based"] if atr_available else [])
    stop_mode = st.radio("Stop-loss Mode", stop_modes, horizontal=True, key="rr_stopmode")

    if stop_mode == "% below entry" and (stop_q is None and pd.isna(prefill.get("stop"))):
        stop_pct = st.number_input("Stop % below Entry", min_value=0.1, max_value=50.0, value=8.0, step=0.1, format="%.1f", key="rr_stoppct")
        stop = entry * (1 - stop_pct / 100.0)
    elif stop_mode == "Manual price" or (stop_q is not None) or (prefill.get("stop") is not None and not pd.isna(prefill.get("stop"))):
        stop_default = (
            float(stop_q) if stop_q is not None else
            float(prefill.get("stop")) if prefill.get("stop") is not None and not pd.isna(prefill.get("stop")) else
            max(0.0, entry * 0.92)
        )
        stop = st.number_input("Stop-loss Price", min_value=0.0, value=stop_default, step=0.001, format="%.4f", key="rr_stop")
    else:
        # ATR-based
        st.markdown(f"**ATR({atr_period_default})** latest ≈ **{(f'{atr_val:,.4f}' if atr_val is not None else 'N/A')}**"
                    + (f" (as of {pd.to_datetime(atr_date).date()})" if atr_date is not None else ""))
        atr_mult = st.number_input("ATR Multiplier", min_value=0.5, max_value=5.0, value=2.0, step=0.5, format="%.1f", key="rr_atr_mult")
        stop = max(0.0, entry - (atr_val or 0) * atr_mult) if atr_available else max(0.0, entry * 0.92)

with cC:
    take_mode = st.radio("Take-profit Mode", ["R multiple", "Manual price"], horizontal=True, key="rr_tpmode")
    if take_mode == "R multiple" and (take_q is None and pd.isna(prefill.get("take"))):
        r_default = float(r_q) if r_q is not None else float(prefill.get("r")) if prefill.get("r") is not None and not pd.isna(prefill.get("r")) else 2.0
        r_multiple = st.number_input("Target (R)", min_value=0.5, max_value=10.0, value=r_default, step=0.5, format="%.1f", key="rr_rmult")
        risk_per_share = max(entry - stop, 0.0)
        take = entry + r_multiple * risk_per_share if risk_per_share > 0 else entry
    else:
        take_default = (
            float(take_q) if take_q is not None else
            float(prefill.get("take")) if prefill.get("take") is not None and not pd.isna(prefill.get("take")) else
            (entry * 1.15 if entry else 0.0)
        )
        take = st.number_input("Take-profit Price", min_value=0.0, value=take_default, step=0.001, format="%.4f", key="rr_take")

with cD:
    acct = st.number_input(
        "Account Size (MYR)",
        min_value=0.0, value=10000.0, step=100.0, format="%.2f",
        key="rr_acct"
    )
    # Cap risk at 30% (you can choose lower, but not higher)
    risk_pct = st.number_input(
        "Risk % per Trade",
        min_value=0.1, max_value=30.0, value=1.0, step=0.1, format="%.1f",
        key="rr_riskpct"
    )
    lot_size = st.number_input(
    "Lot Size (shares)",
    min_value=1, value=100, step=1,
    key="rr_lot"
)

# ==== Limits & Caps (risk/cost) ====
if entry and entry > 0 and stop is not None:
    rps = max(entry - stop, 0.0)                       # risk/share
    cash_risk = acct * (risk_pct / 100.0)              # MYR

    # Max by risk (rounded to lot size)
    max_sh_risk_raw = math.floor(cash_risk / rps) if rps > 0 else 0
    max_sh_risk = max((max_sh_risk_raw // lot_size) * lot_size, 0)
    lots_risk = (max_sh_risk // lot_size) if lot_size else 0

    # Max by cost/buying power (rounded to lot size)
    max_sh_cost = max((math.floor(acct / entry) // lot_size) * lot_size, 0)
    lots_cost = (max_sh_cost // lot_size) if lot_size else 0

    # Final allowed (must satisfy both caps)
    final_allowed = min(max_sh_risk, max_sh_cost) if max_sh_cost > 0 else max_sh_risk
    final_lots = (final_allowed // lot_size) if lot_size else 0

    # Display as a compact 4-metric row
    m1, m2, m3, m4 = st.columns([1.3, 1.7, 1.7, 1.7])
    with m1: st.metric("Cash risk budget (MYR)", f"{cash_risk:,.2f}")
    with m2: st.metric("Max by risk", f"{max_sh_risk:,} sh (≈ {lots_risk} lots)")
    with m3: st.metric("Max by cost", f"{max_sh_cost:,} sh (≈ {lots_cost} lots)")
    with m4: st.metric("Final allowed", f"{final_allowed:,} sh (≈ {final_lots} lots)")
else:
    st.info("Enter Entry & Stop to see risk, caps and allowed size.")




# =============== Calculations ===============
def valid_prices(entry, stop, take):
    if entry is None or stop is None or take is None:
        return False, "Please provide Entry, Stop, and Take-Profit."
    if stop >= entry:
        return False, "Stop-loss must be **below** Entry for a long setup."
    if take <= entry:
        return False, "Take-profit must be **above** Entry for a long setup."
    return True, ""

ok, msg = valid_prices(entry, stop, take)
if not ok:
    st.warning(msg)
    st.stop()

risk_per_share   = entry - stop
reward_per_share = take - entry
rr = (reward_per_share / risk_per_share) if risk_per_share > 0 else None

cash_risk = acct * (risk_pct / 100.0)

# Risk-based sizing (STOP is already used: risk_per_share = entry - stop)
raw_shares = math.floor(cash_risk / risk_per_share) if (risk_per_share and risk_per_share > 0) else 0
shares_by_risk = max((raw_shares // lot_size) * lot_size, 0)

# Cost-based cap: cannot spend more than Account Size
max_sh_by_cost = 0
if entry and entry > 0:
    max_sh_by_cost = max((math.floor(acct / entry) // lot_size) * lot_size, 0)

# Final shares must satisfy BOTH caps
shares = min(shares_by_risk, max_sh_by_cost) if max_sh_by_cost > 0 else shares_by_risk

# Optional: tell user if the cost cap binds
if max_sh_by_cost > 0 and shares_by_risk > max_sh_by_cost:
    st.info("Shares limited by account buying power (cost cap).")
elif shares_by_risk > 0 and shares == shares_by_risk and (max_sh_by_cost == 0 or shares_by_risk <= max_sh_by_cost):
    st.info("Shares limited by your risk cap.")


position_cost  = shares * entry
potential_loss = shares * (entry - stop)
potential_gain = shares * (take - entry)


# Band & viability
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

# ─────────────────────────────────────────
# 2a) Optional Multi-Target Take-Profits
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec warning"><div class="t">2a) Optional Multi-Target Take-Profits</div>'
    '<div class="d">Define TP1/TP2/TP3 by R multiples</div></div>',
    unsafe_allow_html=True
)
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

# ─────────────────────────────────────────
# 3) Results (metrics table)
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec info"><div class="t">3) Result</div>'
    '<div class="d">Key metrics & projected P/L</div></div>',
    unsafe_allow_html=True
)
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1: st.metric("Stop-loss", fmt(stop, 4))
with k2: st.metric("Take-profit", fmt(take, 4))
with k3: st.metric("R : R", f"{rr:.2f}" if rr is not None and np.isfinite(rr) else "N/A")
with k4: st.metric("Shares", f"{shares:,d}")
with k5: st.metric("Position Cost", f"{position_cost:,.2f}")
with k6: st.metric("Risk (MYR)", f"{potential_loss:,.2f}")

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

# ─────────────────────────────────────────
# 4) Save / Update
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec success"><div class="t">4) Save Plan</div>'
    '<div class="d">Add NEW queue row or UPDATE the selected queued idea</div></div>',
    unsafe_allow_html=True
)

summary_lines = [
    f"Plan for {stock_name} ({strategy})",
    f"Entry={entry:.4f}, Stop={stop:.4f}, Take={take:.4f}",
    f"R:R={rr:.2f}" if rr is not None and np.isfinite(rr) else "R:R=N/A",
    f"Shares={shares:,d}, Cost={position_cost:,.2f}, Risk={potential_loss:,.2f}, Gain={potential_gain:,.2f}",
    f"TP1={TP1:.4f}, TP2={TP2:.4f}, TP3={TP3:.4f}",
    f"Band={band_label}, Viability={viability}",
]
reason_text = " | ".join(summary_lines)

colL, colR = st.columns([1, 1])
with colL:
    st.text_area("Plan summary (stored as `Reasons`)", value=reason_text, height=90, key="rr_reason")

disabled = not (shares > 0 and np.isfinite(risk_per_share) and risk_per_share > 0)

with colR:
    if editing_rowid is None:
        if st.button("➕ Add NEW plan to Queue", type="primary", use_container_width=True, disabled=disabled):
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
    else:
        if st.button("💾 Update EXISTING queue row", type="secondary", use_container_width=True, disabled=disabled):
            ok = io_helpers.update_trade_candidate(
                editing_rowid,
                Entry=entry, Stop=stop, Take=take, Shares=shares, RR=rr,
                TP1=TP1, TP2=TP2, TP3=TP3,
                Reasons=st.session_state["rr_reason"],
            )
            if ok:
                st.success("Updated the selected Trade Queue row.")
            else:
                st.error("Could not find the selected queue row to update.")


