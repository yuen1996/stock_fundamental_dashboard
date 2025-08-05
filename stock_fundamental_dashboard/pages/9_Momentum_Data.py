# pages/9_Momentum_Data.py
import os, re
import pandas as pd
import numpy as np
import streamlit as st

# Try your helpers (works whether utils/ is used or not)
try:
    from io_helpers import load_data, load_ohlc
except Exception:
    from utils.io_helpers import load_data, load_ohlc  # fallback if placed under utils/

OHLC_DIR = "data/ohlc"

def _sanitize_name(n: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(n)).strip("_")

def _compute_momentum_preview(ohlc: pd.DataFrame) -> dict:
    """Compute last price, MA200, 12m return if possible."""
    if ohlc is None or ohlc.empty:
        return {"rows": 0, "price": None, "ma200": None, "ret_12m": None}
    df = ohlc.copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
    rows = len(df)
    price = float(df["Close"].iloc[-1]) if rows else None
    ma200 = float(df["Close"].rolling(200, min_periods=200).mean().iloc[-1]) if rows >= 200 else None
    ret_12m = None
    if rows >= 252:
        c0 = float(df["Close"].iloc[-252])
        if c0 != 0:
            ret_12m = price / c0 - 1.0
    return {"rows": rows, "price": price, "ma200": ma200, "ret_12m": ret_12m}

st.set_page_config(page_title="Momentum Data (OHLC)", page_icon="📈", layout="wide")
st.title("📈 Momentum Data — Daily OHLC")
st.write(
    "Provide daily OHLC so the **Snowflake / Momentum** pillar can compute "
    "**12-month return** and the **200-DMA flag**."
)

# Load names from your master data file
df = load_data()
names = sorted(df["Name"].dropna().unique().tolist())
left, right = st.columns([1, 1])

with left:
    stock_name = st.selectbox("Pick a stock (Name in your system)", names, index=0 if names else None)
with right:
    # Allow custom/override name
    custom_name = st.text_input("Or type a custom stock name (overrides the picker)", value="")
if custom_name.strip():
    stock_name = custom_name.strip()

st.divider()
tab_fetch, tab_upload, tab_check = st.tabs(["Download from Yahoo", "Upload CSV", "Verify / Preview"])

# --------------------------- Fetch from Yahoo ---------------------------
with tab_fetch:
    st.caption("Uses the **yfinance** library to fetch daily data and saves to your app as CSV.")
    ticker = st.text_input("Yahoo ticker (try 5024.KL or HUPSENG.KL)", value="5024.KL", key="yf_ticker")
    years_back = st.slider("Years of daily history to fetch", 1, 20, 5)
    auto_adjust = st.checkbox("Use Adjusted Close as Close (recommended)", True)
    do_fetch = st.button("Fetch & Save to app", use_container_width=True)

    if do_fetch:
        try:
            import yfinance as yf
        except Exception:
            st.error("`yfinance` not installed. Add `yfinance>=0.2.40` to requirements.txt and restart.")
            st.stop()
        with st.spinner(f"Fetching {years_back} years from Yahoo for {ticker}…"):
            df_yf = yf.download(ticker, period=f"{years_back}y", interval="1d", auto_adjust=auto_adjust)
        if df_yf is None or df_yf.empty:
            st.error("No data returned. Try the other ticker variant (5024.KL ↔ HUPSENG.KL) or increase years.")
        else:
            # Normalize columns
            df_yf = df_yf.rename(columns={"Adj Close": "Close"})
            need = ["Open", "High", "Low", "Close"]
            missing = [c for c in need if c not in df_yf.columns]
            if missing:
                st.error(f"Downloaded data missing columns: {missing}")
            else:
                os.makedirs(OHLC_DIR, exist_ok=True)
                out_path = os.path.join(OHLC_DIR, f"{_sanitize_name(stock_name)}.csv")
                out = df_yf[["Open", "High", "Low", "Close"]].copy()
                if "Volume" in df_yf.columns:
                    out["Volume"] = df_yf["Volume"]
                out.index.name = "Date"
                out.reset_index().to_csv(out_path, index=False)
                st.success(f"Saved {len(out)} rows → {out_path}")
                st.dataframe(out.tail(10), use_container_width=True)

# --------------------------- Upload CSV ---------------------------
with tab_upload:
    st.caption("Upload a CSV with headers: Date,Open,High,Low,Close[,Volume]. It will be saved for this stock.")
    up = st.file_uploader("Upload OHLC CSV", type=["csv"])
    if up is not None:
        try:
            raw = pd.read_csv(up)
            # Flexible header matching
            cols = {c.lower().strip(): c for c in raw.columns}
            def pick(name):  # return the original-cased column if a lowercase match exists
                return cols.get(name.lower())
            need = [pick("Date"), pick("Open"), pick("High"), pick("Low"), pick("Close")]
            if any(c is None for c in need):
                st.error("CSV must contain at least Date, Open, High, Low, Close columns.")
            else:
                out = raw[[need[0], need[1], need[2], need[3], need[4]]].copy()
                out.columns = ["Date", "Open", "High", "Low", "Close"]
                vol = pick("Volume")
                if vol:
                    out["Volume"] = raw[vol]
                out = out.dropna(subset=["Date", "Close"]).sort_values("Date")
                os.makedirs(OHLC_DIR, exist_ok=True)
                out_path = os.path.join(OHLC_DIR, f"{_sanitize_name(stock_name)}.csv")
                out.to_csv(out_path, index=False)
                st.success(f"Saved {len(out)} rows → {out_path}")
                st.dataframe(out.tail(10), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")

# --------------------------- Verify / Preview ---------------------------
with tab_check:
    st.caption("Checks whether Momentum can be computed now (need ≥200 days for MA200 and ≥252 for 12-month return).")
    if stock_name:
        ohlc = load_ohlc(stock_name)
        if ohlc is None or ohlc.empty:
            st.info("No OHLC file found yet. Use Download/Upload tabs above.")
        else:
            m = _compute_momentum_preview(ohlc)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows (days)", f"{m['rows']:,}")
            c2.metric("Last Close", f"{m['price']:.4f}" if m["price"] is not None else "–")
            c3.metric("MA200", f"{m['ma200']:.4f}" if m["ma200"] is not None else "–")
            c4.metric("12-month return", f"{m['ret_12m']*100:.2f}%" if m["ret_12m"] is not None else "–")
            if m["rows"] < 200:
                st.warning("Need ≥200 daily rows for the 200-DMA flag.")
            if m["rows"] < 252:
                st.warning("Need ≥252 daily rows for the 12-month return.")
            st.dataframe(ohlc.tail(15), use_container_width=True)
