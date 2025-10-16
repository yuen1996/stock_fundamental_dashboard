# pages/10_QUANT_TECH_CHARTS.py
from __future__ import annotations

# --- Auth ---
from auth_gate import require_auth
require_auth()

# --- UI wiring ---
import streamlit as st
from utils.ui import setup_page, section, render_page_title

# --- Std libs ---
import os, re, math
from datetime import date, timedelta
import numpy as np
import pandas as pd

# --- Charting ---
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Project helpers (stock master list) ---
try:
    from io_helpers import load_data  # master list of stocks
except Exception:
    from utils.io_helpers import load_data  # fallback for /pages

# Try to import the bus used by Momentum Data for cache-busting
try:
    from utils import bus  # preferred
except Exception:
    import bus  # fallback if bus.py is at project root

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
setup_page("Quant Tech Charts")
render_page_title("Quant Tech Charts")

_THIS   = os.path.dirname(__file__)
_PARENT = os.path.abspath(os.path.join(_THIS, ".."))
_GRANDP = os.path.abspath(os.path.join(_PARENT, ".."))

_VERSION_FILE = os.path.join(_GRANDP, ".data.version")
def _data_etag() -> int:
    try:
        return int(os.stat(_VERSION_FILE).st_mtime_ns)
    except Exception:
        return 0

# ---- Where OHLC CSVs live (match Momentum Data's save location) ----
def _resolve_ohlc_dir() -> str:
    """
    Prefer the same folder Momentum Data writes to:
      <project>/data/ohlc  (parent of /pages)
    Fall back to a few common spots; create the canonical one if missing.
    """
    candidates = [
        os.path.join(_PARENT, "data", "ohlc"),
        os.path.join(os.getcwd(), "data", "ohlc"),
        os.path.join(_THIS, "..", "data", "ohlc"),
        os.path.join(_GRANDP, "data", "ohlc"),
    ]
    for d in candidates:
        if os.path.isdir(d):
            return os.path.abspath(d)
    d = os.path.join(_PARENT, "data", "ohlc")
    os.makedirs(d, exist_ok=True)
    return os.path.abspath(d)

OHLC_DIR = _resolve_ohlc_dir()

def _safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_")

def _ohlc_path(name: str) -> str:
    return os.path.join(OHLC_DIR, f"{_safe_name(name)}.csv")

def _alias_candidates(x: str) -> list[str]:
    """Generate tolerant filename candidates (handles Berhad/Bhd/PLC suffixes)."""
    base = _safe_name(x)
    cands = {base}
    for suf in ("_Berhad", "_Bhd", "_PLC"):
        if base.endswith(suf):
            cands.add(base[: -len(suf)])
    cands.add(re.sub(r'_(Berhad|Bhd|PLC)$', '', base, flags=re.I))
    return [c for c in dict.fromkeys([c for c in cands if c])]

def _ohlc_path_for(stock_name: str) -> str | None:
    """Exact match first; else tolerant alias + relaxed prefix."""
    exact = os.path.join(OHLC_DIR, f"{_safe_name(stock_name)}.csv")
    if os.path.exists(exact):
        return exact
    try:
        files = {
            fn.lower(): os.path.join(OHLC_DIR, fn)
            for fn in os.listdir(OHLC_DIR) if fn.lower().endswith(".csv")
        }
        for cand in _alias_candidates(stock_name):
            target = f"{cand}.csv".lower()
            if target in files:
                return files[target]
            for fname, full in files.items():
                if fname.startswith(cand.lower()):
                    return full
    except Exception:
        pass
    return exact

@st.cache_data(show_spinner=False)
def _load_ohlc(name: str, _etag: int) -> pd.DataFrame | None:
    """Load a stock OHLC CSV (cache busts via bus etag if present)."""
    try:
        p = _ohlc_path_for(name)
        if not p or not os.path.exists(p):
            return None
        df = pd.read_csv(p)
        if "Date" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if hasattr(df["Date"], "dt"):
            try:
                df["Date"] = df["Date"].dt.tz_localize(None)
            except Exception:
                pass
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = (
            df.dropna(subset=["Date", "Close"])
              .drop_duplicates(subset=["Date"])
              .sort_values("Date")
        )
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c not in df.columns:
                df[c] = np.nan
        return df.reset_index(drop=True)
    except Exception:
        return None

# ----------------------- Indicators -----------------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(1, n//2)).mean()

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=max(1, n//2)).mean()

def wma(s: pd.Series, n: int) -> pd.Series:
    if n <= 1:
        return s
    weights = np.arange(1, n+1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)

def hma(s: pd.Series, n: int) -> pd.Series:
    n = int(max(1, n))
    wma_half = wma(s, max(1, n//2))
    wma_full = wma(s, n)
    raw = 2*wma_half - wma_full
    return wma(raw, int(round(math.sqrt(n))))

def kama(s: pd.Series, n_er: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    change = s.diff(n_er).abs()
    vol = s.diff().abs().rolling(n_er).sum()
    er = (change / vol).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    sc_fast = 2.0 / (fast + 1.0)
    sc_slow = 2.0 / (slow + 1.0)
    sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
    out = pd.Series(index=s.index, dtype=float)
    if len(s) == 0:
        return out
    out.iloc[0] = float(s.iloc[0])
    for i in range(1, len(s)):
        prev = out.iloc[i-1]; cur = s.iloc[i]
        k = sc.iloc[i] if np.isfinite(sc.iloc[i]) else sc_slow**2
        out.iloc[i] = prev + k * (cur - prev)
    return out

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0.0); down = -delta.clip(upper=0.0)
    rs = ema(up, n) / (ema(down, n) + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(s, fast) - ema(s, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line

def bbands(s: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(s, n)
    std = s.rolling(n, min_periods=max(1, n//2)).std()
    return mid, mid + k*std, mid - k*std

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    a = df["High"] - df["Low"]
    b = (df["High"] - prev_close).abs()
    c = (df["Low"] - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1.0/n, adjust=False).mean()

def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = df[["Date","Open","High","Low","Close"]].copy()
    ha["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0
    ha["HA_Open"] = np.nan
    ha.loc[ha.index[0], "HA_Open"] = (df.loc[df.index[0], "Open"] + df.loc[df.index[0], "Close"]) / 2.0
    for i in range(1, len(ha)):
        ha.loc[ha.index[i], "HA_Open"] = (ha.loc[ha.index[i-1], "HA_Open"] + ha.loc[ha.index[i-1], "HA_Close"]) / 2.0
    ha["HA_High"] = ha[["HA_Open","HA_Close"]].join(df["High"]).max(axis=1)
    ha["HA_Low"]  = ha[["HA_Open","HA_Close"]].join(df["Low"]).min(axis=1)
    return ha

def anchored_vwap(df: pd.DataFrame, anchor_date: date) -> pd.Series:
    dfa = df[df["Date"].dt.date >= anchor_date].copy()
    if dfa.empty: return pd.Series(index=df.index, dtype=float)
    tp = (dfa["High"] + dfa["Low"] + dfa["Close"]) / 3.0
    vol = dfa["Volume"].fillna(0.0)
    if vol.sum() <= 0.0 or vol.replace(0, np.nan).isna().all():
        vwap_local = tp.cumsum() / pd.Series(np.arange(1, len(tp)+1), index=tp.index)
    else:
        vwap_local = (tp * vol).cumsum() / vol.cumsum().replace(0, np.nan)
    out = pd.Series(index=df.index, dtype=float); out.loc[dfa.index] = vwap_local
    return out

# ----------------------- Utilities -----------------------
def _resample_ohlc(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "D": return df.copy()
    rule = {"W": "W-FRI", "M": "M"}[freq]
    agg = {"Open":"first", "High":"max", "Low":"min", "Close":"last", "Volume":"sum"}
    out = df.set_index("Date").resample(rule).agg(agg).dropna(subset=["Open","High","Low","Close"])
    return out.reset_index()

def _norm_index_line(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0: return s
    base = s.dropna().iloc[0]
    return (s / (base if base else 1.0)) * 100.0

def _lb_to_days(lb: str) -> int | None:
    return {"3M":90, "6M":180, "1Y":365, "2Y":730, "5Y":1825, "Max":None}.get(lb, 365)

def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))

def _cross_dn(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))

# ----------------------- UI helpers (new) -----------------------
def _price_decimals(y: pd.Series) -> int:
    """Pick sensible decimals for price formatting."""
    try:
        m = float(pd.to_numeric(y, errors="coerce").dropna().median())
    except Exception:
        m = 0.0
    if m == 0 or not np.isfinite(m): return 2
    if m < 1:   return 4
    if m < 10:  return 3
    if m < 100: return 2
    return 2

def _fmt_num(x: float, d: int) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)): return "—"
    return f"{x:.{d}f}"

def _inject_kpi_css() -> None:
    if st.session_state.get("_kpi_css_injected"):  # avoid stacking styles on reruns
        return
    st.markdown("""
<style>
.kpi-card{border:1px solid rgba(0,0,0,.06); border-radius:18px; padding:14px 16px;
    background:linear-gradient(180deg,rgba(2,132,199,.06),rgba(2,132,199,0));
    box-shadow:0 2px 6px rgba(0,0,0,.04);}
.kpi-head{display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;}
.kpi-title{font-weight:700; font-size:1.05rem; display:flex; gap:.5rem; align-items:center;}
.kpi-pill{font-size:.85rem; padding:.25rem .6rem; border-radius:999px; background:rgba(0,0,0,.06);}
.kpi-grid{display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; margin-top:4px;}
.kpi{padding:8px 10px; border-radius:12px; background:rgba(255,255,255,.6); border:1px solid rgba(0,0,0,.05);}
.kpi .label{font-size:.8rem; color:#6b7280; margin-bottom:2px;}
.kpi .value{font-size:1.35rem; font-weight:700; line-height:1.2;}
.delta{margin-top:4px; display:inline-block; font-size:.85rem; padding:.15rem .45rem; border-radius:999px;}
.delta.up{color:#15803d; background:rgba(21,128,61,.12);}
.delta.dn{color:#b91c1c; background:rgba(185,28,28,.12);}
.delta.neu{color:#64748b; background:rgba(100,116,139,.12);}
.note{margin-top:8px; color:#6b7280; font-size:.85rem;}
@media (max-width: 900px){ .kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr));}}
</style>
    """, unsafe_allow_html=True)
    st.session_state["_kpi_css_injected"] = True

def _render_last_candle_card(df: pd.DataFrame, po: pd.Series, ph: pd.Series, pl: pd.Series, pc: pd.Series, *,
                             title_tf: str, use_heikin: bool) -> None:
    """Render a KPI-style card with last candle OHLC + date and delta on Close."""
    last_idx = pc.last_valid_index()
    if last_idx is None:
        return

    d = df.loc[last_idx, "Date"]
    op, hi, lo, cl = (po.iloc[last_idx], ph.iloc[last_idx], pl.iloc[last_idx], pc.iloc[last_idx])

    # previous valid close for delta
    prev_series = pc.iloc[: last_idx].dropna()
    prev_cl = prev_series.iloc[-1] if not prev_series.empty else np.nan

    delta_abs = cl - prev_cl if (np.isfinite(cl) and np.isfinite(prev_cl)) else np.nan
    delta_pct = (delta_abs / prev_cl * 100.0) if (np.isfinite(delta_abs) and prev_cl not in (0, np.nan)) else np.nan

    dec = _price_decimals(pc)
    op_s, hi_s, lo_s, cl_s = (_fmt_num(float(op), dec), _fmt_num(float(hi), dec),
                              _fmt_num(float(lo), dec), _fmt_num(float(cl), dec))

    if np.isfinite(delta_abs) and abs(delta_abs) > 0:
        trend_cls = "up" if delta_abs > 0 else "dn"
        sign = "▲" if delta_abs > 0 else "▼"
        delta_s = f"{sign} {_fmt_num(abs(delta_abs), dec)} ({delta_pct:.2f}%)"
    elif np.isfinite(cl):
        trend_cls = "neu"
        delta_s = "— 0.00%".
    else:
        trend_cls = "neu"
        delta_s = "—"

    date_str = pd.to_datetime(d).strftime("%Y-%m-%d")
    src_note = "Heikin-Ashi" if use_heikin else "Price"

    _inject_kpi_css()
    st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-head">
    <div class="kpi-title">🔎 Last Candle</div>
    <div class="kpi-pill">{date_str} · {title_tf}</div>
  </div>
  <div class="kpi-grid">
    <div class="kpi"><div class="label">Open</div><div class="value">{op_s}</div></div>
    <div class="kpi"><div class="label">High</div><div class="value">{hi_s}</div></div>
    <div class="kpi"><div class="label">Low</div><div class="value">{lo_s}</div></div>
    <div class="kpi">
      <div class="label">Close</div>
      <div class="value">{cl_s}</div>
      <div class="delta {trend_cls}">{delta_s}</div>
    </div>
  </div>
  <div class="note">Values shown are <b>{src_note}</b>.</div>
</div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Page content
# ------------------------------------------------------------------
st.markdown(
    section(
        "What this does",
        "Charts from your saved daily OHLC in `data/ohlc/&lt;Stock_Name&gt;.csv`. "
        "**Simple mode** shows a concise, proven set (SMA20/50/200, BB(20,2), RSI, MACD). "
        "Switch to Advanced for extra tools. Turn on *Trade signals* to see buy/sell markers."
    ),
    unsafe_allow_html=True
)

# --- Stock list from master data ---
master = load_data()
stocks = sorted(master["Name"].dropna().astype(str).unique().tolist()) \
         if (master is not None and not master.empty and "Name" in master.columns) else []
if not stocks:
    st.warning("No stocks found in your master data. Add names on the Add/Edit page first.")
    st.stop()

# --- Sidebar controls ---
with st.sidebar:
    st.subheader("Controls")
    sel = st.multiselect("Stocks", options=stocks, default=[stocks[0]] if stocks else [])
    lookback = st.select_slider("Lookback", options=["3M","6M","1Y","2Y","5Y","Max"], value="1Y")
    freq = st.radio("Timeframe", ["D","W","M"], index=0, horizontal=True)
    use_heikin = st.checkbox("Heikin-Ashi candles", value=False)
    simple_mode = st.checkbox("Simple mode (core set)", value=True)

    st.markdown("---")
    if simple_mode:
        st.caption("Core set options")
        show_bb = True; bb_len = 20; bb_k = 2.0
        show_rsi = True; rsi_len = 14; rsi_ob = 70; rsi_os = 30
        show_macd = True; macd_fast = 12; macd_slow = 26; macd_sig = 9
        show_compare = st.checkbox("Show comparison panel (normalized to 100)", value=False)
        use_vwap = st.checkbox("Add an Anchored VWAP", value=False)
        default_anchor = date.today() - timedelta(days=365)
        anchor_date_1 = st.date_input("Anchor date", value=default_anchor, key="avwap1_simple", disabled=not use_vwap)
        show_atr = False; atr_len = 14
        show_sma_cross = True; sma_fast = 20; sma_slow = 50
        ma_type = "SMA"; ma_len = 50
        show_kc = False; kc_len = 20; kc_mult = 1.5
        show_squeeze = False
        show_super = False; st_n = 10; st_mult = 3.0
        show_don = False; don_n = 20
        show_rsc = False; bench = "<none>"
        use_adx = False; adx_n = 14; adx_thr = 25
        use_anchor_ytd = False; use_anchor_lb = False; extra_anchor_2 = None
    else:
        st.caption("Moving Averages")
        ma_type = st.selectbox("Primary MA", ["SMA","EMA","KAMA","HMA"], index=0)
        ma_len  = st.number_input("Primary MA length", 5, 400, 50, step=1)
        show_sma_cross = st.checkbox("Show SMA fast/slow cross", value=True)
        sma_fast = st.number_input("SMA Fast", 5, 400, 20, step=1)
        sma_slow = st.number_input("SMA Slow", 10, 600, 50, step=5)

        st.markdown("---")
        st.caption("Bands")
        show_bb = st.checkbox("Bollinger Bands", value=True)
        bb_len  = st.number_input("BB Period", 5, 200, 20, step=1)
        bb_k    = st.number_input("BB Width (σ)", 1.0, 4.0, 2.0, step=0.5)
        show_kc = st.checkbox("Keltner Channels", value=True)
        kc_len  = st.number_input("KC Period", 5, 200, 20, step=1)
        kc_mult = st.number_input("KC Width (ATR×)", 0.5, 5.0, 1.5, step=0.1)
        show_squeeze = st.checkbox("TTM Squeeze", value=True)

        st.markdown("---")
        st.caption("Other overlays")
        show_super = st.checkbox("Supertrend", value=False)
        st_n = st.number_input("Supertrend ATR period", 5, 100, 10, step=1)
        st_mult = st.number_input("Supertrend ATR mult", 1.0, 6.0, 3.0, step=0.5)
        show_don = st.checkbox("Donchian Channels", value=False)
        don_n = st.number_input("Donchian lookback", 5, 200, 20, step=1)

        st.markdown("---")
        st.caption("Oscillators")
        show_rsi = st.checkbox("RSI", value=True)
        rsi_len  = st.number_input("RSI (len)", 5, 100, 14, step=1)
        rsi_ob   = st.number_input("Overbought", 50, 95, 70, step=1)
        rsi_os   = st.number_input("Oversold", 5, 50, 30, step=1)
        show_macd = st.checkbox("MACD", value=True)
        macd_fast = st.number_input("MACD fast", 2, 60, 12, step=1)
        macd_slow = st.number_input("MACD slow", 5, 120, 26, step=1)
        macd_sig  = st.number_input("Signal", 2, 60, 9, step=1)
        show_atr = st.checkbox("ATR pane", value=False)
        atr_len  = st.number_input("ATR (len)", 5, 100, 14, step=1)

        st.markdown("---")
        show_compare = st.checkbox("Show comparison panel (normalized to 100)", value=False)
        st.markdown("---")
        st.caption("Relative Strength vs Benchmark")
        show_rsc = st.checkbox("Show RSC (Mansfield) panel", value=False)
        bench = st.selectbox("Benchmark", options=["<none>"] + stocks, index=0)

        st.markdown("---")
        st.caption("Regime Filter (ADX)")
        use_adx = st.checkbox("Enable ADX regime flag", value=True)
        adx_n = st.number_input("ADX period", 5, 100, 14, step=1)
        adx_thr = st.number_input("Trend threshold (ADX≥)", 5, 50, 25, step=1)

        st.markdown("---")
        st.caption("Anchored VWAPs")
        use_vwap = st.checkbox("Enable AVWAPs", value=False)
        default_anchor = date.today() - timedelta(days=365)
        anchor_date_1 = st.date_input("Anchor #1", value=default_anchor, key="avwap1")
        use_anchor_ytd = st.checkbox("Add YTD anchor", value=False)
        use_anchor_lb  = st.checkbox("Add Lookback-start anchor", value=False)
        extra_anchor_2 = st.date_input("Anchor #2 (optional)", value=default_anchor, key="avwap2")

    # --- Trade signals controls ---
    st.markdown("---")
    st.caption("Trade signals")
    show_signals = st.checkbox("Show buy/sell signals", value=True)
    use_sig_sma  = st.checkbox("SMA fast/slow cross", value=True)
    use_sig_macd = st.checkbox("MACD line/signal cross", value=False)
    use_sig_rsi  = st.checkbox("RSI threshold cross", value=False)

# --- Lookback ---
lb_days = _lb_to_days(lookback)
if not sel:
    st.info("Pick at least one stock from the sidebar.")
    st.stop()

# --- Load data for main chart ---
try:
    _bus_etag = bus.etag("ohlc")
except Exception:
    _bus_etag = None
etag = (_bus_etag or _data_etag())

main_name = sel[0]
df0 = _load_ohlc(main_name, etag)
if df0 is None or df0.empty:
    tried = _ohlc_path_for(main_name)
    st.warning(
        f"No OHLC file found for **{main_name}**.\n\n"
        f"I looked for: `{tried}`\n\n"
        "Use the Momentum Data page to fetch/upload it."
    )
    st.stop()

if lb_days is not None:
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=lb_days)
    df0 = df0[df0["Date"] >= cutoff]

df = _resample_ohlc(df0, freq).reset_index(drop=True)

# Heikin-Ashi (optional)
if use_heikin:
    ha = heikin_ashi(df)
    price_open, price_high, price_low, price_close = ha["HA_Open"], ha["HA_High"], ha["HA_Low"], ha["HA_Close"]
else:
    price_open, price_high, price_low, price_close = df["Open"], df["High"], df["Low"], df["Close"]

# --- Indicators for main chart ---
if simple_mode:
    sma20 = sma(price_close, 20)
    sma50 = sma(price_close, 50)
    sma200 = sma(price_close, 200)
    primary_ma = sma50
else:
    if   ma_type == "SMA":  primary_ma = sma(price_close, int(ma_len))
    elif ma_type == "EMA":  primary_ma = ema(price_close, int(ma_len))
    elif ma_type == "KAMA": primary_ma = kama(price_close, n_er=min(10, int(ma_len)), fast=2, slow=max(10, int(ma_len)))
    else:                   primary_ma = hma(price_close, int(ma_len))

# Always compute fast/slow SMAs for **signals**
_sma_fast_sig = sma(price_close, int(sma_fast))
_sma_slow_sig = sma(price_close, int(sma_slow))

# Plot-only (advanced) cross MAs
sma_fast_ser = sma(price_close, int(sma_fast)) if (show_sma_cross and not simple_mode) else None
sma_slow_ser = sma(price_close, int(sma_slow)) if (show_sma_cross and not simple_mode) else None

bb_mid = bb_up = bb_lo = None
if show_bb:
    bb_mid, bb_up, bb_lo = bbands(price_close, int(bb_len), float(bb_k))

# Optional AVWAPs
vwap_series = {}
if use_vwap:
    bdf = pd.DataFrame({"Date": df["Date"], "High": price_high, "Low": price_low, "Close": price_close, "Volume": df["Volume"]})
    vwap_series["Anchor"] = anchored_vwap(bdf, anchor_date_1)
    if not simple_mode:
        if use_anchor_ytd:
            vwap_series["YTD"] = anchored_vwap(bdf, date(date.today().year, 1, 1))
        if use_anchor_lb and lb_days is not None:
            vwap_series["Lookback"] = anchored_vwap(bdf, (pd.Timestamp.now().normalize() - pd.Timedelta(days=lb_days)).date())
        if extra_anchor_2:
            vwap_series["Anchor #2"] = anchored_vwap(bdf, extra_anchor_2)

# Oscillators
need_rsi  = show_rsi or (show_signals and use_sig_rsi)
need_macd = show_macd or (show_signals and use_sig_macd)
rsi_series = rsi(price_close, int(rsi_len)) if need_rsi else None
macd_line = macd_signal = macd_hist = None
if need_macd:
    macd_line, macd_signal, macd_hist = macd(price_close, int(macd_fast), int(macd_slow), int(macd_sig))

# ----------------------- Build signals -----------------------
signals = []
if show_signals:
    if use_sig_sma:
        up  = _cross_up(_sma_fast_sig, _sma_slow_sig)
        dn  = _cross_dn(_sma_fast_sig, _sma_slow_sig)
        for idx in df.index[up.fillna(False)]:
            signals.append({"Date": df.loc[idx, "Date"], "Action": "Buy",  "Reason": f"SMA{int(sma_fast)}/{int(sma_slow)} bull cross", "Price": float(price_close.iloc[idx])})
        for idx in df.index[dn.fillna(False)]:
            signals.append({"Date": df.loc[idx, "Date"], "Action": "Sell", "Reason": f"SMA{int(sma_fast)}/{int(sma_slow)} bear cross", "Price": float(price_close.iloc[idx])})

    if use_sig_macd and macd_line is not None:
        up  = _cross_up(macd_line, macd_signal)
        dn  = _cross_dn(macd_line, macd_signal)
        for idx in df.index[up.fillna(False)]:
            signals.append({"Date": df.loc[idx, "Date"], "Action": "Buy",  "Reason": "MACD bull cross", "Price": float(price_close.iloc[idx])})
        for idx in df.index[dn.fillna(False)]:
            signals.append({"Date": df.loc[idx, "Date"], "Action": "Sell", "Reason": "MACD bear cross", "Price": float(price_close.iloc[idx])})

    if use_sig_rsi and rsi_series is not None:
        up  = _cross_up(rsi_series, pd.Series([float(rsi_os)]*len(rsi_series), index=rsi_series.index))
        dn  = _cross_dn(rsi_series, pd.Series([float(rsi_ob)]*len(rsi_series), index=rsi_series.index))
        for idx in df.index[up.fillna(False)]:
            signals.append({"Date": df.loc[idx, "Date"], "Action": "Buy",  "Reason": f"RSI cross↑ {int(rsi_os)}", "Price": float(price_close.iloc[idx])})
        for idx in df.index[dn.fillna(False)]:
            signals.append({"Date": df.loc[idx, "Date"], "Action": "Sell", "Reason": f"RSI cross↓ {int(rsi_ob)}", "Price": float(price_close.iloc[idx])})

    if signals:
        df_sig = pd.DataFrame(signals).sort_values("Date").reset_index(drop=True)
        merged = []
        for d, grp in df_sig.groupby(["Date", "Action"], sort=True):
            reasons = ", ".join(grp["Reason"].tolist())
            price = grp["Price"].iloc[-1]
            merged.append({"Date": d[0], "Action": d[1], "Reason": reasons, "Price": price})
        signals = sorted(merged, key=lambda x: x["Date"])

# ----------------------- Formatting for hovers -----------------------
_price_dec = _price_decimals(price_close)

# Build custom hovertext for candles (older Plotly compatible)
open_ser  = price_open if use_heikin else df["Open"]
high_ser  = price_high if use_heikin else df["High"]
low_ser   = price_low  if use_heikin else df["Low"]
close_ser = price_close if use_heikin else df["Close"]

_candle_hovertext = []
for i in range(len(df)):
    d = pd.to_datetime(df.loc[i, "Date"]).strftime("%Y-%m-%d")
    o = _fmt_num(float(open_ser.iloc[i]),  _price_dec)
    h = _fmt_num(float(high_ser.iloc[i]),  _price_dec)
    l = _fmt_num(float(low_ser.iloc[i]),   _price_dec)
    c = _fmt_num(float(close_ser.iloc[i]), _price_dec)
    _candle_hovertext.append(f"<b>{d}</b><br>Open: {o}<br>High: {h}<br>Low: {l}<br>Close: {c}")

# ----------------------- Figure layout -----------------------
rows = [("price","Price / Volume")]
if show_rsi and rsi_series is not None:  rows.append(("rsi","RSI"))
if show_macd and macd_line is not None: rows.append(("macd","MACD"))

fig = make_subplots(
    rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.035,
    row_heights=[0.60] + [0.20]*(len(rows)-1),
    specs=[[{"secondary_y": True}]] + [[{"secondary_y": False}] for _ in rows[1:]]
)

# Candles (use hovertext for compatibility)
fig.add_trace(go.Candlestick(
    x=df["Date"],
    open=open_ser, high=high_ser, low=low_ser, close=close_ser,
    name="Heikin-Ashi" if use_heikin else "Price",
    hovertext=_candle_hovertext, hoverinfo="text"
), row=1, col=1, secondary_y=False)

# Primary MA + (in simple mode) SMA20/50/200 bundle
fig.add_trace(go.Scatter(x=df["Date"], y=primary_ma, name=(f"{'SMA' if simple_mode else ma_type}({50 if simple_mode else int(ma_len)})"), mode="lines"), row=1, col=1)
if simple_mode:
    fig.add_trace(go.Scatter(x=df["Date"], y=sma20,  name="SMA(20)",  mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=sma200, name="SMA(200)", mode="lines"), row=1, col=1)

# Bollinger Bands
if show_bb and bb_mid is not None:
    fig.add_trace(go.Scatter(x=df["Date"], y=bb_mid, name=f"BB mid({int(bb_len)})", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=bb_up,  name="BB upper", mode="lines", line=dict(dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=bb_lo,  name="BB lower", mode="lines", line=dict(dash="dot")), row=1, col=1)

# Anchored VWAP(s)
for label, ser in (vwap_series.items() if vwap_series else []):
    fig.add_trace(go.Scatter(x=df["Date"], y=ser, name=f"AVWAP {label}", mode="lines"), row=1, col=1)

# Volume (secondary y)
if df["Volume"].notna().sum() > 0:
    _vol_hover = []
    for i in range(len(df)):
        d = pd.to_datetime(df.loc[i, "Date"]).strftime("%Y-%m-%d")
        v = df["Volume"].iloc[i]
        v_str = f"{int(v):,}" if pd.notna(v) else "—"
        _vol_hover.append(f"<b>{d}</b><br>Volume: {v_str}")
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Volume"], name="Volume", opacity=0.3,
        hovertext=_vol_hover, hoverinfo="text"
    ), row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Price", tickformat=f".{_price_dec}f", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Vol",   row=1, col=1, secondary_y=True)

# RSI pane
row_idx = 2
if show_rsi and rsi_series is not None:
    fig.add_trace(go.Scatter(x=df["Date"], y=rsi_series, name=f"RSI({int(rsi_len)})", mode="lines"), row=row_idx, col=1)
    fig.add_hrect(y0=30, y1=70, line_width=0, fillcolor="LightGray", opacity=0.25, row=row_idx, col=1)
    fig.add_hline(y=70, line=dict(dash="dot"), row=row_idx, col=1)
    fig.add_hline(y=30, line=dict(dash="dot"), row=row_idx, col=1)
    fig.update_yaxes(title_text="RSI", row=row_idx, col=1, range=[0, 100])
    row_idx += 1

# MACD pane
if show_macd and macd_line is not None:
    fig.add_trace(go.Bar(x=df["Date"], y=macd_hist, name="MACD hist", opacity=0.45), row=row_idx, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=macd_line,  name="MACD",   mode="lines"), row=row_idx, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=macd_signal, name="Signal", mode="lines"), row=row_idx, col=1)
    fig.update_yaxes(title_text="MACD", row=row_idx, col=1)

# --- Signal markers on price chart ---
if show_signals and signals:
    sig_df = pd.DataFrame(signals).sort_values("Date")
    buys  = sig_df[sig_df["Action"] == "Buy"]
    sells = sig_df[sig_df["Action"] == "Sell"]

    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys["Date"], y=buys["Price"], mode="markers",
            marker=dict(symbol="triangle-up", size=12, line=dict(width=1), color="#2ca02c"),
            name="BUY signal", hovertext=buys["Reason"], hoverinfo="text+x+y"
        ), row=1, col=1)

    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells["Date"], y=sells["Price"], mode="markers",
            marker=dict(symbol="triangle-down", size=12, line=dict(width=1), color="#d62728"),
            name="SELL signal", hovertext=sells["Reason"], hoverinfo="text+x+y"
        ), row=1, col=1)

    for d in sig_df["Date"].tail(6):
        fig.add_vline(x=d, line_width=1, line_dash="dot", line_color="gray", opacity=0.35, row=1, col=1)

# Layout
fig.update_layout(
    height=760 if simple_mode else 880,
    margin=dict(l=6, r=6, t=10, b=70),
    xaxis=dict(showspikes=True),
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0)
)

# --- Title ---
title_tf = {"D":"Daily","W":"Weekly","M":"Monthly"}[freq]
display_name = str(main_name).strip().upper()
st.subheader(f"{display_name} — {title_tf} Technicals")

# Last candle KPI card (always true OHLC; change to price_* variables if you want HA)
_render_last_candle_card(df, df["Open"], df["High"], df["Low"], df["Close"],
                         title_tf=title_tf, use_heikin=use_heikin)

# ----------------------- Tabs -----------------------
tab_chart, tab_signals = st.tabs(["📈 Chart", "🔔 Signals"])

with tab_chart:
    st.plotly_chart(fig, use_container_width=True, key="main_chart")

with tab_signals:
    if show_signals and signals:
        df_sig = pd.DataFrame(signals).sort_values("Date", ascending=False).reset_index(drop=True)
        df_sig["Date"] = df_sig["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(df_sig[["Date","Action","Price","Reason"]], use_container_width=True, height=360)
        st.caption("Signals shown are generated by the selected engines (SMA cross / MACD cross / RSI threshold). "
                   "Triangle-up = Buy, Triangle-down = Sell. Use these as cues, not guarantees.")
    elif show_signals:
        st.info("No signals found for the current settings/lookback.")
    else:
        st.info("Enable **Show buy/sell signals** in the sidebar to populate this tab.")

# ----------------------- Comparison panel (optional) -----------------------
if show_compare and len(sel) > 1:
    st.markdown("#### 📊 Comparison (normalized to 100)")
    comp_dfs = []
    for name in sel:
        df_ = _load_ohlc(name, etag)
        if df_ is None or df_.empty:
            continue
        if lb_days is not None:
            cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=lb_days)
            df_ = df_[df_["Date"] >= cutoff]
        df_ = _resample_ohlc(df_, freq).reset_index(drop=True)
        comp_dfs.append((name, df_[["Date","Close"]].reset_index(drop=True)))

    if comp_dfs:
        base = comp_dfs[0][1].rename(columns={"Close": comp_dfs[0][0]})
        for nm, dfi in comp_dfs[1:]:
            base = base.merge(dfi.rename(columns={"Close": nm}), on="Date", how="outer")
        base = base.sort_values("Date").reset_index(drop=True)

        fig2 = go.Figure()
        for nm in [n for n,_ in comp_dfs]:
            fig2.add_trace(go.Scatter(x=base["Date"], y=_norm_index_line(base[nm]), name=nm, mode="lines"))
        fig2.update_layout(height=360, margin=dict(l=6, r=6, t=32, b=6),
                           title=f"Normalized Close = 100 @ first point ({title_tf}, {lookback})",
                           legend=dict(orientation="h"))
        st.plotly_chart(fig2, use_container_width=True, key="cmp_chart")
    else:
        st.info("Not enough data to render comparison.")

# ----------------------- Footer notes -----------------------
st.caption("Tip: Use the Momentum Data page to fetch/upload OHLC files. This page reads those CSVs directly and renders charts with Plotly.")
