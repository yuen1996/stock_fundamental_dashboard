# pages/5_Risk_Reward_Planner.py
from __future__ import annotations

# --- Auth ---
from auth_gate import require_auth
require_auth()

# --- Std libs ---
import os, sys, math, io, json
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st
from utils import bus, name_link  # to locate OHLC like View Stock
import re  # used by _safe_key

# --- UI wiring ---
try:
    from utils.ui import setup_page, section, render_stat_cards, render_page_title
except Exception:
    try:
        from ui import setup_page, section, render_stat_cards, render_page_title  # fallback if utils/ not present
    except Exception:
        from ui import setup_page  # type: ignore

        def section(title: str, subtitle: str = "", tone: str | None = None) -> str:
            badge = "" if not tone else f" <small>[{tone}]</small>"
            sub = f'<div style="color:#6b7280;font-size:.95rem;margin-top:.15rem">{subtitle}</div>' if subtitle else ""
            return f'<div style="margin:.75rem 0"><h3 style="margin:0">{title}{badge}</h3>{sub}</div>'
        
        def render_stat_cards(items, columns: int = 3, caption: str | None = None):
            # very small fallback: show as plain text
            if caption: st.caption(caption)
            cols = st.columns(columns)
            for i, it in enumerate(items):
                with cols[i % columns]:
                    st.write(f"**{it.get('label','')}**")
                    st.write(it.get('value',''))
        def render_page_title(page_name: str) -> None:
            st.title(f"📊 Fundamentals Dashboard — {page_name}")

# --- Robust import pathing (works from /pages or project root) ---
_THIS   = os.path.dirname(__file__)
_PARENT = os.path.abspath(os.path.join(_THIS, ".."))
_GRANDP = os.path.abspath(os.path.join(_PARENT, ".."))
for p in (_PARENT, _GRANDP):
    if p not in sys.path:
        sys.path.insert(0, p)
        
# --- Data version etag (shared across pages) & unified OHLC folder ---
_VERSION_FILE = os.path.join(_GRANDP, ".data.version")
def _data_etag() -> int:
    try:
        return int(os.stat(_VERSION_FILE).st_mtime_ns)
    except Exception:
        return 0

# Keep all OHLC CSVs under project /data/ohlc (shared with Momentum & Tech Charts)
OHLC_DIR = os.path.join(_PARENT, "data", "ohlc")

# --- io_helpers only (no rules linkage) ---
try:
    from utils import io_helpers
except Exception:
    import io_helpers  # type: ignore

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
setup_page("Risk / Reward Planner")
render_page_title("Risk / Reward Planner")

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _is_num(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def _fmt(x, d=4, pct=False):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "N/A"
    try:
        return (f"{x:,.{d}f}%" if pct else f"{x:,.{d}f}")
    except Exception:
        return "N/A"

def _fmt_cell_display(v):
    """Safe string for Streamlit table (avoids Arrow numeric-cast issues)."""
    try:
        if v is None:
            return "—"
        # ints stay as integers with commas
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,d}"
        # floats: nice formatting (more precision for small price-like values)
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return "—"
            # 4dp for sub-100 values, otherwise 2dp
            return f"{v:,.4f}" if abs(v) < 100 else f"{v:,.2f}"
        # strings/emojis/etc.
        s = str(v).strip()
        return s if s else "—"
    except Exception:
        return "—"
    
def _safe_key(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name or "")).lower()
  
def _rr_band(rr):
    if rr is None or not np.isfinite(rr): return "N/A", "neutral"
    if rr < 1.5:  return "Low (<1.5)", "bad"
    if rr < 2.0:  return "OK (1.5–2.0)", "warn"
    return "Good (≥2.0)", "good"

def _latest_current_price(stock_rows: pd.DataFrame) -> float | None:
    if stock_rows is None or stock_rows.empty:
        return None
    stock_name = (
        str(stock_rows["Name"].dropna().iloc[-1])
        if "Name" in stock_rows.columns and stock_rows["Name"].dropna().size
        else None
    )
    if not stock_name:
        return None

    pref = st.session_state.get(f"price_src_{_safe_key(stock_name)}", "Momentum")
    pref = str(pref or "Auto").strip().lower()

    manual = _manual_price_from_rows(stock_rows)
    mom, _asof = _momentum_close_before_today(stock_name, stock_rows)

    if pref == "add/edit":
        return manual if manual is not None else mom
    # 'momentum' and 'auto' prefer Momentum, then fall back to manual
    return mom if mom is not None else manual

@st.cache_data(show_spinner=False)
def _load_ohlc_local(name: str, _etag: int) -> pd.DataFrame | None:

    try:
        safe = "".join(ch if ch.isalnum() or ch in ("-","_") else "_" for ch in str(name)).strip("_")
        path = os.path.join(OHLC_DIR, f"{safe}.csv")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            try:
                df["Date"] = df["Date"].dt.tz_localize(None)
            except Exception:
                pass
        for c in ("Open","High","Low","Close","Adj Close","Volume"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return (df.dropna(subset=["Date","Close"])
                  .drop_duplicates(subset=["Date"])
                  .sort_values("Date")
                  .reset_index(drop=True))
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def _compute_atr_from_ohlc(name: str, period: int = 14) -> tuple[Optional[float], Optional[pd.Timestamp]]:
    df = _load_ohlc_local(name, _data_etag())
    if df is None or df.empty:
        return None, None
    hi = pd.to_numeric(df.get("High"), errors="coerce")
    lo = pd.to_numeric(df.get("Low"), errors="coerce")
    cl = pd.to_numeric(df.get("Close"), errors="coerce")
    prev_close = cl.shift(1)
    tr = pd.DataFrame({
        "hl": hi - lo,
        "hc": (hi - prev_close).abs(),
        "lc": (lo - prev_close).abs(),
    }).max(axis=1)
    atr = tr.rolling(int(period)).mean()
    v = float(atr.iloc[-1]) if not atr.empty and np.isfinite(atr.iloc[-1]) else None
    d = df["Date"].iloc[-1] if "Date" in df.columns and not df.empty else None
    return v, d

def _latest_atr(name: str, period: int = 14) -> tuple[Optional[float], Optional[pd.Timestamp]]:
    # Prefer project's helper if you add it later
    fn = getattr(io_helpers, "latest_atr", None)
    if callable(fn):
        try:
            v, d = fn(name, period=period)
            return (float(v) if v is not None else None, d)
        except Exception:
            pass
    # Fallback to local OHLC
    return _compute_atr_from_ohlc(name, period=period)

# --- OHLC loader & Momentum close (same behavior as View Stock) ---

def _resolve_ohlc_dir() -> str:
    # reuse your OHLC_DIR if defined; else fallback to common paths
    try:
        return OHLC_DIR
    except NameError:
        candidates = [
            os.path.abspath(os.path.join(_PARENT, "data", "ohlc")),
            os.path.abspath(os.path.join(os.getcwd(), "data", "ohlc")),
            os.path.abspath(os.path.join(_THIS, "..", "data", "ohlc")),
            os.path.abspath(os.path.join(_GRANDP, "data", "ohlc")),
        ]
        for d in candidates:
            if os.path.isdir(d):
                return d
        return candidates[0]

@st.cache_data(show_spinner=False)
def _load_ohlc_for_name(stock_name: str, _etag: int, *, ticker: str | None = None) -> pd.DataFrame | None:
    ohlc_dir = _resolve_ohlc_dir()
    path = name_link.find_ohlc_path(stock_name, ohlc_dir=ohlc_dir, ticker=ticker)
    if not path or not os.path.exists(path):
        # last resort: old "<name>.csv"
        path = os.path.join(ohlc_dir, f"{stock_name}.csv")
        if not os.path.exists(path):
            return None
    dfp = pd.read_csv(path)
    dfp["Date"] = pd.to_datetime(dfp["Date"], errors="coerce")
    if "Close" not in dfp.columns and "Adj Close" in dfp.columns:
        dfp["Close"] = pd.to_numeric(dfp["Adj Close"], errors="coerce")
    for c in ["Open","High","Low","Close","Volume"]:
        if c in dfp.columns:
            dfp[c] = pd.to_numeric(dfp[c], errors="coerce")
    return (dfp.dropna(subset=["Date","Close"])
              .drop_duplicates(subset=["Date"])
              .sort_values("Date")
              .reset_index(drop=True))

def _momentum_close_before_today(stock_name: str, stock_rows: pd.DataFrame) -> tuple[float | None, pd.Timestamp | None]:
    ticker_hint = None
    for col in ("Ticker", "Code", "Symbol"):
        if col in stock_rows.columns and stock_rows[col].dropna().size:
            ticker_hint = str(stock_rows[col].dropna().iloc[-1]); break
    etag = int(bus.etag("ohlc")) if hasattr(bus, "etag") else 0
    dfp = _load_ohlc_for_name(stock_name, etag, ticker=ticker_hint)
    if dfp is None or dfp.empty:
        return None, None
    today = pd.Timestamp.today().normalize()
    prior = dfp[dfp["Date"] < today]
    if prior.empty:
        return None, None
    last_row = prior.iloc[-1]
    px = float(last_row["Close"]) if pd.notna(last_row["Close"]) else None
    asof = pd.to_datetime(last_row["Date"])
    return (px if (px is not None and np.isfinite(px)) else None), asof

def _manual_price_from_rows(rows: pd.DataFrame) -> float | None:
    for k in ("CurrentPrice", "EndQuarterPrice", "Price", "SharePrice"):
        s = rows.get(k)
        if s is not None and s.dropna().size:
            try:
                v = float(pd.to_numeric(s, errors="coerce").dropna().iloc[-1])
                if np.isfinite(v):
                    return v
            except Exception:
                pass
    return None

# History-based strategy stats (optional; independent of rules)
def _strategy_stats(strategy: str) -> dict:
    """Try to read your closed-trades stats via io_helpers.load_closed_trades(); fall back to generic."""
    try:
        fn = getattr(io_helpers, "load_closed_trades", None)
        if callable(fn):
            h = fn()
            if h is not None and not h.empty:
                df = h.copy()
                if "Strategy" in df.columns:
                    df = df[df["Strategy"].astype(str) == str(strategy)]
                if "RMultiple" in df.columns and not df.empty:
                    r = pd.to_numeric(df["RMultiple"], errors="coerce").dropna()
                    if not r.empty:
                        wins = r[r > 0]; losses = r[r <= 0]
                        wr = float(len(wins)) / float(len(r)) if len(r) else 0.0
                        rwin = float(wins.mean()) if len(wins) else 2.0
                        rloss = float(abs(losses.mean())) if len(losses) else 1.0
                        return {"wr": wr, "rwin": rwin, "rloss": rloss, "n": int(len(r))}
    except Exception:
        pass
    return {"wr": 0.50, "rwin": 2.0, "rloss": 1.0, "n": 0}

def _ev(win_rate: float, avg_win_r: float, avg_loss_r: float) -> Optional[float]:
    try:
        return win_rate * avg_win_r - (1.0 - win_rate) * avg_loss_r
    except Exception:
        return None

def _atomic_write_csv_local(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    df2 = df.copy().where(pd.notna(df.copy()), "")
    try:
        df2.to_csv(tmp, index=False)
    except Exception:
        for c in df2.columns:
            df2[c] = df2[c].astype("object")
        df2.to_csv(tmp, index=False)
    os.replace(tmp, path)

def _queue_path() -> str:
    p = ""
    try:
        dbg = getattr(io_helpers, "debug_queue_paths", lambda: {})()
        p = dbg.get("QUEUE_FILE", "")
    except Exception:
        p = ""
    if p:
        return p
    return os.path.join(_PARENT, "data", "trade_queue.csv")

def _update_queue_row_by_index(row_id: int, **fields) -> bool:
    """Fallback updater if io_helpers.update_trade_candidate() is not available."""
    try:
        q = io_helpers.load_trade_queue()
        if q is None or q.empty or row_id < 0 or row_id >= len(q):
            return False
        for k, v in fields.items():
            if k in q.columns:
                q.at[row_id, k] = v
            else:
                q[k] = q.get(k, "")
                q.at[row_id, k] = v
        _atomic_write_csv_local(q, _queue_path())
        return True
    except Exception:
        return False

# ------------------------------------------------------------------
# 0) Pick an existing queued row (QUEUED ONLY)
# ------------------------------------------------------------------
qdf_raw = io_helpers.load_trade_queue()
if qdf_raw is None or qdf_raw.empty:
    st.warning("Trade Queue is empty. Go to **Systematic Decision** and push ideas to queue first.")
    st.stop()

queue_df = qdf_raw.reset_index().rename(columns={"index": "RowId"})
options = ["— Choose queued idea —"] + [f"{int(r.RowId)} — {r.Name} ({r.Strategy})" for _, r in queue_df.iterrows()]
sel = st.selectbox("Queued idea to edit", options, index=0)

# Reset per-row widget state when the selected queue row changes
_form_keys = (
    "rr_stock", "rr_stock_label", "rr_strategy",
    "rr_entry","rr_stop_mode","rr_stop","rr_stop_pct","rr_atr_mult",
    "rr_take","rr_acct","rr_riskpct","rr_lot","rr_planned_lots",
    "tp1r","tp2r","tp3r","rr_reason"
)

if st.session_state.get("_rr_last_sel") != sel:
    for k in _form_keys:
        st.session_state.pop(k, None)
    st.session_state["_rr_last_sel"] = sel

if sel == "— Choose queued idea —":
    st.info("Pick a **queued idea** to continue. New plans are disabled (systematic mode).")
    st.stop()

editing_rowid = int(sel.split(" — ")[0])
row = queue_df.loc[queue_df.RowId == editing_rowid].iloc[0]

# Prefill payload from queue
prefill = {
    "stock":    row.Name,
    "strategy": row.Strategy,
    "entry":    row.Entry,
    "stop":     row.Stop,
    "take":     row.Take,
    "r":        row.RR,
}

# Query params (legacy links still work)
try:
    qp = dict(st.query_params)
except Exception:
    qp = {k: v[0] for k, v in st.experimental_get_query_params().items()}

def qget(k, cast=None, default=None):
    if k in prefill and prefill[k] is not None and not (isinstance(prefill[k], float) and np.isnan(prefill[k])):
        return prefill[k]
    v = qp.get(k, default)
    if v is None: return default
    if cast is None: return v
    try: return cast(v)
    except Exception: return default

# ------------------------------------------------------------------
# Load master data (for price lookups only; no rules)
# ------------------------------------------------------------------
df = io_helpers.load_data()
if df is None or df.empty or "Name" not in df.columns:
    st.warning("No data found. Please add stocks in **Add/Edit** first.")
    st.stop()

stocks = sorted([s for s in df["Name"].dropna().unique()])

# 1) Pick stock & strategy (strategy is just a label here)
st.markdown(
    section("1) Pick a Stock & Strategy",
            "Select an idea from your Trade Queue (new plans disabled)"),
    unsafe_allow_html=True
)

# Wrap inputs in an area card
st.markdown('<div class="area">', unsafe_allow_html=True)
cA, cB = st.columns([2, 1])
with cA:
    # Read-only stock name from the selected queue row
    stock_name = str(qget("stock", str, None) or "")
    st.text_input("Stock", value=stock_name, disabled=True, key=f"rr_stock_label_{editing_rowid}")

with cB:
    strat_options = sorted([str(s) for s in queue_df["Strategy"].dropna().unique()]) or ["General"]
    default_strat = qget("strategy", str, None)
    strat_index = strat_options.index(default_strat) if (default_strat in strat_options) else 0
    strategy = st.selectbox("Strategy (label only)", options=strat_options, index=strat_index, key="rr_strategy")
st.markdown('</div>', unsafe_allow_html=True)

# Lookups for current price
stock_rows = df[df["Name"] == stock_name].sort_values(["Year"])
price_now  = _latest_current_price(stock_rows)

# Latest ATR for this stock (prefer helper/fallback)
atr_period_default = 14
atr_val, atr_date = _latest_atr(stock_name, period=atr_period_default)
atr_available = atr_val is not None

# Header KPIs (include ATR)
render_stat_cards([
    {"label": "Queued Row",     "value": str(editing_rowid), "badge": "ID"},
    {"label": "Current Price",  "value": (f"{price_now:,.4f}" if price_now is not None else "N/A"), "badge": "Live"},
    {"label": "Strategy",       "value": strategy or "—"},
    {"label": f"ATR({atr_period_default})", "value": (f"{atr_val:,.4f}" if atr_available else "—"),
     "badge": (atr_date.strftime("%Y-%m-%d") if isinstance(atr_date, (pd.Timestamp, datetime)) else "")},
], columns=4, caption=None)

# ------------------------------------------------------------------
# 2) Entry / Stop / Take & Risk
# ------------------------------------------------------------------
st.markdown(
    section("2) Entry / Stop / Take", "Define your plan & risk controls", "info"),
    unsafe_allow_html=True
)

st.markdown('<div class="area">', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 2.2])
with c1:
    entry_default = float(qget("entry", float, None) or (row.Entry if _is_num(row.Entry) else (price_now or 0.0)))
    entry = st.number_input("Entry Price", min_value=0.0, value=entry_default, step=0.001, format="%.4f", key="rr_entry")
with c2:
    # --- ATR Stop Mode support here ---
    stop_mode = st.radio("Stop-loss Mode",
                         ["Manual price", "% below entry", "ATR-based"],
                         index=0, horizontal=True, key="rr_stop_mode")

    if stop_mode == "Manual price":
        stop_default = float(qget("stop", float, None) or (row.Stop if _is_num(row.Stop) else (entry * 0.95 if entry else 0.0)))
        stop  = st.number_input("Stop-loss Price", min_value=0.0, value=stop_default, step=0.001, format="%.4f", key="rr_stop")
    elif stop_mode == "% below entry":
        pct = st.number_input("Stop % below entry", min_value=0.1, max_value=90.0,
                              value=5.0, step=0.1, format="%.1f", key="rr_stop_pct")
        stop = max(0.0, entry * (1.0 - pct/100.0))
        st.metric("Derived Stop", f"{stop:,.4f}")
    else:  # ATR-based
        atr_mult = st.number_input("ATR multiplier (×)", min_value=0.25, max_value=10.0,
                                   value=2.0, step=0.25, format="%.2f", key="rr_atr_mult")
        if atr_available and entry > 0:
            stop = max(0.0, entry - atr_val * atr_mult)
            st.metric("Derived Stop", f"{stop:,.4f}")
        else:
            st.warning("ATR not available — falling back to 5% stop.")
            stop = max(0.0, entry * 0.95)
with c3:
    take_default = float(qget("take", float, None) or (row.Take if _is_num(row.Take) else (entry * 1.15 if entry else 0.0)))
    take  = st.number_input("Take-profit Price", min_value=0.0, value=take_default, step=0.001, format="%.4f", key="rr_take")
with c4:
    acct = st.number_input("Account Size (MYR)", min_value=0.0, value=10000.0, step=100.0, format="%.2f", key="rr_acct")
    risk_pct = st.number_input("Risk % per Trade", min_value=0.1, max_value=30.0, value=1.0, step=0.1, format="%.1f", key="rr_riskpct")
    lot_size = st.number_input("Lot Size (shares)", min_value=1, value=100, step=1, help="Shares per 1 lot (e.g. 100).", key="rr_lot")
    planned_lots = st.number_input("Planned lots to buy", min_value=0, value=int(row.Shares // lot_size if _is_num(row.Shares) else 0), step=1, help="Your intended lots.", key="rr_planned_lots")
    st.caption("Tip: If you want 13 lots, keep Lot Size=100 and set **Planned lots to buy = 13**.")
st.markdown('</div>', unsafe_allow_html=True)

# Math & caps
risk_per_share   = max(entry - stop, 0.0)
reward_per_share = max(take - entry, 0.0)
rr = (reward_per_share / risk_per_share) if risk_per_share > 0 else None

lots_planned_now   = int(st.session_state.get("rr_planned_lots", planned_lots) or 0)
planned_shares_now = int(lots_planned_now * lot_size)

cash_risk = acct * (risk_pct / 100.0) if acct and risk_pct else 0.0
max_sh_cost = max((math.floor(acct / entry) // lot_size) * lot_size, 0) if entry > 0 and lot_size else 0
lots_cost   = (max_sh_cost // lot_size) if lot_size else 0
final_allowed_lots   = max(lots_cost - lots_planned_now, 0)
final_allowed_shares = final_allowed_lots * lot_size

# Band & viability
band_label, band_tone = _rr_band(rr)
viability = ("✅ Ready" if (rr is not None and np.isfinite(rr) and rr >= 1.5 and planned_shares_now > 0)
             else "⚠️ Low R or zero size" if (rr is not None and rr < 1.5)
             else "⏳ Incomplete")

# Strategy EV (historic, optional)
stats = _strategy_stats(strategy)
ev_r = _ev(stats["wr"], stats["rwin"], stats["rloss"])
oneR_money = planned_shares_now * max(entry - stop, 0.0)

# KPI rows
render_stat_cards([
    {"label":"Risk/Share (RM)", "value": _fmt(risk_per_share, 4), "badge": "RISK"},
    {"label":"Reward/Share (RM)","value": _fmt(reward_per_share, 4), "badge": "REWARD"},
    {"label":"R : R", "value": (f"{rr:.2f}×" if rr and np.isfinite(rr) else "N/A"),
     "badge": band_label.split(" ")[0] if band_label!="N/A" else "",
     "tone": {"good":"good","warn":"warn","bad":"bad"}.get(band_tone,"")},
], columns=3)

render_stat_cards([
    {"label":"Max Lots (by COST)","value": f"{lots_cost:,d}"},
    {"label":"Planned Lots","value": f"{lots_planned_now:,d}"},
    {"label":"Final Allowed (Lots)","value": f"{final_allowed_lots:,d}"},
    {"label":"1R (MYR)","value": (f"{oneR_money:,.2f}" if oneR_money else "0.00")},
], columns=4)

render_stat_cards([
    {"label":"Strategy EV / 1R", "value": (f"{ev_r:.2f}R" if ev_r is not None else "N/A"),
     "note": f"WR≈{stats['wr']*100:.1f}% | AvgWin≈{stats['rwin']:.2f}R | AvgLoss≈{stats['rloss']:.2f}R | N={stats['n']}"},
    {"label":"Viability", "value": viability,
     "tone": ("good" if "Ready" in viability else "bad" if "Low" in viability else "warn")},
], columns=2)

# ------------------------- Locks (with session unlock support) -------------------------
min_rr = st.slider("Minimum allowed R:R", 1.2, 3.0, 1.8, 0.1,
                   help="Block Save/Update if plan R:R is below this")
rr_ok_planned = bool(rr is not None and np.isfinite(rr) and rr >= min_rr and planned_shares_now > 0)

ev_thresh, ev_block_days = -0.25, 7
ev_bad = (ev_r is not None and np.isfinite(ev_r) and ev_r < ev_thresh)

# Keys
block_key = f"ev_block_until_{strategy or 'ALL'}"
clear_key = f"ev_unlocked_{strategy or 'ALL'}"      # per-strategy “unlocked” flag
global_clear_key = "ev_unlocked_ALL"                # global “unlocked” flag for this session

now = datetime.now()

# If globally cleared, make sure no per-strategy block survives
if st.session_state.get(global_clear_key):
    st.session_state.pop(block_key, None)

block_until = st.session_state.get(block_key, None)
unlocked = bool(st.session_state.get(clear_key) or st.session_state.get(global_clear_key))

# Only create/extend the lock if NOT unlocked
if ev_bad and not unlocked:
    if (not block_until) or (now >= block_until):
        st.session_state[block_key] = now + timedelta(days=ev_block_days)
        block_until = st.session_state[block_key]
else:
    # If unlocked, ensure the lock is removed
    if block_key in st.session_state:
        st.session_state.pop(block_key, None)
    block_until = None

blocked_active = bool(block_until and now < block_until)

# User messaging
if planned_shares_now <= 0:
    st.warning("Planned lots is 0. Set **Planned lots to buy** to enable Save/Update.")
elif not rr_ok_planned:
    st.warning("R:R below minimum. Increase take-profit or tighten stop.")
if ev_bad:
    st.warning(f"Strategy EV {ev_r:.2f}R is below threshold ({ev_thresh:.2f}R).")
if blocked_active:
    st.error(f"EV block active for '{strategy}'. Unlocks on {block_until.strftime('%Y-%m-%d %H:%M')}.")

# ------------------------------------------------------------------
# 2a) Optional multi-target take-profits
# ------------------------------------------------------------------
st.markdown(
    section("2a) Optional Multi-Target Take-Profits", "Define TP1/TP2/TP3 by R multiples", "warning"),
    unsafe_allow_html=True
)
st.markdown('<div class="area">', unsafe_allow_html=True)
c_tp1, c_tp2, c_tp3 = st.columns(3)
with c_tp1: tp1_r = st.number_input("TP1 (R)", min_value=0.25, max_value=10.0, value=1.0, step=0.25, format="%.2f", key="tp1r")
with c_tp2: tp2_r = st.number_input("TP2 (R)", min_value=0.25, max_value=10.0, value=2.0, step=0.25, format="%.2f", key="tp2r")
with c_tp3: tp3_r = st.number_input("TP3 (R)", min_value=0.25, max_value=10.0, value=3.0, step=0.25, format="%.2f", key="tp3r")

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
st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# 3) Result metrics (table)
# ------------------------------------------------------------------
st.markdown(section("3) Result", "Key metrics & projected P/L", "info"), unsafe_allow_html=True)
st.markdown('<div class="area">', unsafe_allow_html=True)

# Planned numbers (based on planned lots set above)
lots_planned   = int(st.session_state.get("rr_planned_lots", lots_planned_now) or 0)
planned_shares = int(lots_planned * lot_size)
planned_cost   = planned_shares * entry
planned_loss   = planned_shares * max(entry - stop, 0.0)
planned_gain   = planned_shares * max(take - entry, 0.0)

# ATR-based loss readout (reuse the ATR we computed earlier)
atr_stop_price = None
atr_loss = None
if atr_available and (atr_val is not None):
    atr_mult_used = float(st.session_state.get("rr_atr_mult", 2.0) or 2.0)
    atr_stop_price = max(0.0, entry - atr_val * atr_mult_used)
    atr_loss = planned_shares * max(entry - atr_stop_price, 0.0)

detail = pd.DataFrame(
    {
        "Entry": [entry],
        "Stop": [stop],
        "Take": [take],
        "Risk/Share": [risk_per_share],
        "Reward/Share": [reward_per_share],
        "R:R": [rr],
        "Planned Lots": [lots_planned],
        "Planned Shares": [planned_shares],
        "Planned Cost (MYR)": [planned_cost],
        "Planned Loss @ Stop (MYR)": [planned_loss],
        "Planned Gain @ Take (MYR)": [planned_gain],
        "Loss @ ATR Stop (MYR)": [atr_loss],
        "Max Risk Budget (MYR)": [cash_risk],
        "TP1": [TP1],
        "TP2": [TP2],
        "TP3": [TP3],
        "Viability": [viability],
    }
).T
detail.columns = [stock_name]

# Arrow-safe display: make every cell a string for the rendered table
detail_disp = detail.copy()
detail_disp[stock_name] = detail_disp[stock_name].map(_fmt_cell_display)

st.table(detail_disp)
st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# 4) Save / Update
# ------------------------------------------------------------------
st.markdown(section("4) Save Plan", "Update the selected queued idea", "success"), unsafe_allow_html=True)
st.markdown('<div class="area">', unsafe_allow_html=True)

summary_lines = [
    f"Plan for {stock_name} ({strategy})",
    f"Entry={entry:.4f}, Stop={stop:.4f}, Take={take:.4f}",
    f"R:R={rr:.2f}" if rr is not None and np.isfinite(rr) else "R:R=N/A",
    f"Shares={planned_shares:,d}, Cost={planned_cost:,.2f}, Risk={planned_loss:,.2f}, Gain={planned_gain:,.2f}",
    f"TP1={TP1:.4f}, TP2={TP2:.4f}, TP3={TP3:.4f}",
    f"Band={band_label} ({viability})",
]
reason_text = " | ".join(summary_lines)

colL, colR = st.columns([1, 1])
with colL:
    st.text_area("Plan summary (stored as `Reasons`)", value=reason_text, height=110, key="rr_reason")

disabled = (not (planned_shares > 0 and np.isfinite(risk_per_share) and risk_per_share > 0 and rr_ok_planned)) or blocked_active

with colR:
    if st.button("💾 Update queued plan", type="primary", use_container_width=True, disabled=disabled):
        ok = False
        update_fn = getattr(io_helpers, "update_trade_candidate", None)
        fields = dict(
            Entry=entry, Stop=stop, Take=take, Shares=int(planned_shares), RR=(float(rr) if rr is not None and np.isfinite(rr) else ""),
            TP1=TP1, TP2=TP2, TP3=TP3,
            Reasons=st.session_state.get("rr_reason", reason_text),
        )
        if callable(update_fn):
            try:
                ok = bool(update_fn(editing_rowid, **fields))
            except Exception:
                ok = False
        if not ok:
            ok = _update_queue_row_by_index(editing_rowid, **fields)

        if ok:
            st.success("Updated the selected Trade Queue row.")
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
        else:
            st.error("Could not find/update the selected queue row.")
st.markdown('</div>', unsafe_allow_html=True)
