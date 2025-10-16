# pages/10_QUANT_TECH_CHARTS.py
from __future__ import annotations

# --- Auth ---
from auth_gate import require_auth
require_auth()

# --- UI wiring ---
import streamlit as st
from utils.ui import setup_page, section, render_page_title, render_stat_cards

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
        os.path.join(_PARENT, "data", "ohlc"),      # <project>/data/ohlc  ← Momentum saves here
        os.path.join(os.getcwd(), "data", "ohlc"),  # CWD fallback
        os.path.join(_THIS, "..", "data", "ohlc"),  # pages/.. fallback
        os.path.join(_GRANDP, "data", "ohlc"),      # legacy/grandparent
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

def _ohlc_path_for(stock_name: str) -> str | None:
    """
    Exact match first; else tolerant alias + relaxed prefix (e.g., Public_Bank ↔ Public_Bank_Berhad.csv).
    Returns a *candidate* path even if it doesn't exist (for clearer warnings).
    """
    base = _safe_name(stock_name)
    exact = os.path.join(OHLC_DIR, f"{base}.csv")
    if os.path.exists(exact):
        return exact
    try:
        files = {
            fn.lower(): os.path.join(OHLC_DIR, fn)
            for fn in os.listdir(OHLC_DIR) if fn.lower().endswith(".csv")
        }
        # also try trimming Berhad/Bhd/PLC suffixes
        variants = {base}
        for suf in ("_Berhad", "_Bhd", "_PLC"):
            if base.endswith(suf):
                variants.add(base[: -len(suf)])
        variants.add(re.sub(r'_(Berhad|Bhd|PLC)$', '', base, flags=re.I))

        for cand in variants:
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
    """Load OHLC CSV with tolerant filename matching; return tidy DataFrame."""
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

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0.0); down = -delta.clip(upper=0.0)
    rs = ema(up, n) / (ema(down, n) + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    a = df["High"] - df["Low"]
    b = (df["High"] - prev_close).abs()
    c = (df["Low"] - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1.0/n, adjust=False).mean()

def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume (cumulative)."""
    sign = np.sign(df["Close"].diff()).fillna(0.0)
    return (sign * df["Volume"].fillna(0.0)).cumsum()

def _resample_ohlc(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "D": return df.copy()
    rule = {"W": "W-FRI", "M": "M"}[freq]
    agg = {"Open":"first", "High":"max", "Low":"min", "Close":"last", "Volume":"sum"}
    out = df.set_index("Date").resample(rule).agg(agg).dropna(subset=["Open","High","Low","Close"])
    return out.reset_index()

def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))

def _cross_dn(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))

def _fmt_px(x: float | int | None) -> str:
    try:
        v = float(x)
    except Exception:
        return "—"
    if abs(v) >= 2:   return f"{v:,.2f}"
    if abs(v) >= 0.2: return f"{v:,.3f}"
    return f"{v:,.4f}"


# ------------------------------------------------------------------
# Page content
# ------------------------------------------------------------------
st.markdown(
    section(
        "What this does",
        "Charts from your saved daily OHLC in `data/ohlc/<Stock_Name>.csv`. "
        "This view shows EMA(10) & EMA(30), Volume, RSI, ATR, OBV and EMA-cross Buy/Sell markers. "
        "KPI cards show the last candle’s Date / O / H / L / C."
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

    st.markdown("---")
    st.caption("Panels")
    show_rsi = st.checkbox("RSI (14)", value=True)
    show_atr = st.checkbox("ATR (14)", value=True)
    show_obv = st.checkbox("OBV", value=True)

    st.markdown("---")
    st.caption("Signals")
    show_signals = st.checkbox("EMA(10/30) cross markers", value=True)

# --- Lookback ---
def _lb_to_days(lb: str) -> int | None:
    return {"3M":90, "6M":180, "1Y":365, "2Y":730, "5Y":1825, "Max":None}.get(lb, 365)

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
    # simple HA, only used for candles (not EMA/indicators)
    ha = df[["Date","Open","High","Low","Close"]].copy()
    ha["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0
    ha["HA_Open"] = np.nan
    ha.loc[ha.index[0], "HA_Open"] = (df.loc[df.index[0], "Open"] + df.loc[df.index[0], "Close"]) / 2.0
    for i in range(1, len(ha)):
        ha.loc[ha.index[i], "HA_Open"] = (ha.loc[ha.index[i-1], "HA_Open"] + ha.loc[ha.index[i-1], "HA_Close"]) / 2.0
    ha["HA_High"] = ha[["HA_Open","HA_Close"]].join(df["High"]).max(axis=1)
    ha["HA_Low"]  = ha[["HA_Open","HA_Close"]].join(df["Low"]).min(axis=1)
    price_open, price_high, price_low, price_close = ha["HA_Open"], ha["HA_High"], ha["HA_Low"], ha["HA_Close"]
else:
    price_open, price_high, price_low, price_close = df["Open"], df["High"], df["Low"], df["Close"]

# --- Core overlays ---
ema10 = ema(price_close, 10)
ema30 = ema(price_close, 30)

# --- Oscillators / momentums ---
rsi_series = rsi(price_close, 14) if show_rsi else None
atr_series = atr(df, 14) if show_atr else None
obv_series = obv(df) if show_obv else None

# ----------------------- Signals (EMA 10/30 cross) -----------------------
signals = []
up = _cross_up(ema10, ema30)
dn = _cross_dn(ema10, ema30)
for idx in df.index[up.fillna(False)]:
    signals.append({"Date": df.loc[idx, "Date"], "Action": "Buy",  "Reason": "EMA10/30 bull cross", "Price": float(price_close.iloc[idx])})
for idx in df.index[dn.fillna(False)]:
    signals.append({"Date": df.loc[idx, "Date"], "Action": "Sell", "Reason": "EMA10/30 bear cross", "Price": float(price_close.iloc[idx])})

# ----------------------- Figure layout -----------------------
# Build explicit row map so traces never land in the wrong pane
row = 1
row_map = {"price": row}; row += 1
if rsi_series is not None: row_map["rsi"] = row; row += 1
if atr_series is not None: row_map["atr"] = row; row += 1
if obv_series is not None: row_map["obv"] = row; row += 1

# First row gets secondary_y for Volume; others don't
specs = []
for r in range(1, row):
    specs.append([{"secondary_y": True} if r == row_map["price"] else {"secondary_y": False}])

row_heights = [0.60] + [0.20] * (len(specs) - 1)
fig = make_subplots(
    rows=len(specs), cols=1, shared_xaxes=True, vertical_spacing=0.035,
    row_heights=row_heights, specs=specs
)

# --- PRICE (candles + EMA10/30) ---
r_price = row_map["price"]
fig.add_trace(go.Candlestick(
    x=df["Date"],
    open=(price_open if use_heikin else df["Open"]),
    high=(price_high if use_heikin else df["High"]),
    low=(price_low if use_heikin else df["Low"]),
    close=(price_close if use_heikin else df["Close"]),
    name="Heikin-Ashi" if use_heikin else "Price",
    increasing_line_color="#16a34a", decreasing_line_color="#ef4444"
), row=r_price, col=1, secondary_y=False)

# EMA(10) blue, EMA(30) red — strictly on price row
fig.add_trace(go.Scatter(x=df["Date"], y=ema10, name="EMA(10)", mode="lines",
                         line=dict(color="#1f77b4", width=2)), row=r_price, col=1)
fig.add_trace(go.Scatter(x=df["Date"], y=ema30, name="EMA(30)", mode="lines",
                         line=dict(color="#d62728", width=2)), row=r_price, col=1)

# Volume — secondary y of price row only
if df["Volume"].notna().sum() > 0:
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume", opacity=0.30),
                  row=r_price, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Price", row=r_price, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Vol",   row=r_price, col=1, secondary_y=True)

# --- RSI (STRICT line-only; no price/volume here) ---
if "rsi" in row_map:
    r_rsi = row_map["rsi"]
    fig.add_hrect(
        y0=30, y1=70, line_width=0,
        fillcolor="rgba(140,117,255,0.10)", opacity=1.0,
        row=r_rsi, col=1
    )
    fig.add_trace(go.Scatter(
        x=df["Date"], y=rsi_series,
        name="RSI(14)", mode="lines",
        line=dict(color="#6C5CE7", width=2),
        showlegend=False
    ), row=r_rsi, col=1)
    fig.add_hline(y=70, line=dict(dash="dot", color="#999"), row=r_rsi, col=1)
    fig.add_hline(y=30, line=dict(dash="dot", color="#999"), row=r_rsi, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], zeroline=False, row=r_rsi, col=1)

# --- ATR ---
if "atr" in row_map:
    r_atr = row_map["atr"]
    fig.add_trace(go.Scatter(x=df["Date"], y=atr_series, name="ATR(14)", mode="lines"),
                  row=r_atr, col=1)
    fig.update_yaxes(title_text="ATR", row=r_atr, col=1)

# --- OBV ---
if "obv" in row_map:
    r_obv = row_map["obv"]
    fig.add_trace(go.Scatter(x=df["Date"], y=obv_series, name="OBV", mode="lines"),
                  row=r_obv, col=1)
    fig.update_yaxes(title_text="OBV", row=r_obv, col=1)

# --- Signal markers on PRICE row only ---
if show_signals and signals:
    sig_df = pd.DataFrame(signals).sort_values("Date")
    buys  = sig_df[sig_df["Action"] == "Buy"]
    sells = sig_df[sig_df["Action"] == "Sell"]
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys["Date"], y=buys["Price"], mode="markers",
            marker=dict(symbol="triangle-up", size=12, line=dict(width=1), color="#16a34a"),
            name="BUY signal", hovertext=buys["Reason"], hoverinfo="text+x+y"
        ), row=r_price, col=1)
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells["Date"], y=sells["Price"], mode="markers",
            marker=dict(symbol="triangle-down", size=12, line=dict(width=1), color="#ef4444"),
            name="SELL signal", hovertext=sells["Reason"], hoverinfo="text+x+y"
        ), row=r_price, col=1)

# --- Make the chart 'stick' (no weekend/holiday gaps)
if freq == "D" and not df.empty:
    # Hide weekends plus any missing weekdays (public holidays) between min/max
    all_days = pd.date_range(df["Date"].min().normalize(), df["Date"].max().normalize(), freq="D")
    have = pd.to_datetime(df["Date"].dt.normalize().unique())
    missing_weekdays = [d for d in all_days if (d not in have and d.weekday() < 5)]
    fig.update_xaxes(rangebreaks=[
        dict(bounds=["sat", "mon"]),        # weekends
        dict(values=missing_weekdays)       # holidays (no data weekdays)
    ])

# Layout
fig.update_layout(
    height=820,
    margin=dict(l=6, r=6, t=10, b=70),
    xaxis=dict(showspikes=True),
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0)
)

# --- Big title ABOVE the chart (always visible, ALL CAPS) ---
title_tf = {"D":"Daily","W":"Weekly","M":"Monthly"}[freq]
display_name = str(main_name).strip().upper()
st.subheader(f"{display_name} — {title_tf} Technicals")

# ----------------------- Tabs -----------------------
tab_chart, tab_signals = st.tabs(["📈 Chart", "🔔 Signals"])

# --- KPI cards (Date / O / H / L / C) ---
def _render_kpis(_df: pd.DataFrame):
    if _df.empty: return
    last = _df.iloc[-1]
    last_d = last["Date"]
    items = [
        {"label": "Last Candle", "value": last_d.strftime("%Y-%m-%d"), "badge": title_tf},
        {"label": "Open",  "value": _fmt_px(last.get("Open")) , "badge": "O"},
        {"label": "High",  "value": _fmt_px(last.get("High")) , "badge": "H"},
        {"label": "Low",   "value": _fmt_px(last.get("Low"))  , "badge": "L"},
        {"label": "Close", "value": _fmt_px(last.get("Close")), "badge": "C"},
    ]
    render_stat_cards(items, columns=5)

with tab_chart:
    _render_kpis(df)
    st.plotly_chart(fig, use_container_width=True, key="main_chart")

with tab_signals:
    if show_signals and signals:
        df_sig = pd.DataFrame(signals).sort_values("Date", ascending=False).reset_index(drop=True)
        df_sig["Date"] = df_sig["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(df_sig[["Date","Action","Price","Reason"]], use_container_width=True, height=360)
        st.caption("Markers are based on EMA(10/30) crosses. Triangle-up = Buy, Triangle-down = Sell. Use as cues, not guarantees.")
    elif show_signals:
        st.info("No EMA cross signals in the current window.")
    else:
        st.info("Enable **EMA(10/30) cross markers** in the sidebar to populate this tab.")

# ----------------------- Footer notes -----------------------
st.caption("Tip: Use the Momentum Data page to fetch/upload OHLC files. This page reads those CSVs directly and renders charts with Plotly.")
