# pages/9_Momentum_Data.py
import os, re
import pandas as pd
import numpy as np
import streamlit as st
import time
import datetime as dt
import requests

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

# --------------------------- Fetch (Yahoo or Alpha Vantage) ---------------------------
with tab_fetch:
    src = st.radio("Data source", ["Yahoo (yfinance)", "Alpha Vantage"], horizontal=True)

    if src == "Yahoo (yfinance)":
        st.caption("Pulls via the yfinance API (not the website export). Use Daily for Momentum.")
        ticker = st.text_input("Yahoo ticker", value="5024.KL", key="yf_ticker")

        freq_label = st.selectbox("Frequency", ["Daily", "Weekly", "Monthly"], index=0)
        interval = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}[freq_label]

        mode = st.radio("Range", ["Period (incl. max)", "Custom date range"], horizontal=True)
        if mode == "Period (incl. max)":
            period = st.selectbox("Period", ["1y", "2y", "5y", "10y", "max"], index=4)
            start = end = None
        else:
            c1, c2 = st.columns(2)
            default_start = dt.date.today() - dt.timedelta(days=365*5)
            start = c1.date_input("Start date", value=default_start)
            end   = c2.date_input("End date",   value=dt.date.today())
            period = None

        auto_adjust = st.checkbox("Use Adjusted Close as Close (recommended)", True)
        btn = st.button("Fetch & Save", use_container_width=True)

        if btn:
            try:
                import yfinance as yf
            except Exception:
                st.error("`yfinance` is not installed. Add `yfinance>=0.2.40` to requirements.txt and restart.")
                st.stop()

            with st.spinner("Downloading from Yahoo…"):
                if period:
                    # This usually returns full history when period='max'
                    df_yf = yf.download(ticker, period=period, interval=interval, auto_adjust=auto_adjust)
                else:
                    # Some networks get throttled by long ranges; download in chunks and merge
                    def dl_chunked(tkr, start_d, end_d):
                        step = dt.timedelta(days=720)  # ~2 years per request
                        cur_s = start_d
                        frames = []
                        pbar = st.progress(0, text="Chunking…")
                        i = 0
                        total = max(1, int((end_d - start_d).days / step.days) + 1)
                        while cur_s <= end_d:
                            i += 1
                            cur_e = min(end_d, cur_s + step)
                            df_part = yf.download(
                                tkr,
                                start=cur_s,
                                end=cur_e + dt.timedelta(days=1),  # include end
                                interval=interval,
                                auto_adjust=auto_adjust,
                                progress=False,
                            )
                            if df_part is not None and not df_part.empty:
                                frames.append(df_part)
                            pbar.progress(min(i/total, 1.0))
                            time.sleep(0.6)  # be polite to API
                            cur_s = cur_e + dt.timedelta(days=1)
                        pbar.empty()
                        if frames:
                            out = pd.concat(frames).sort_index()
                            out = out[~out.index.duplicated(keep="last")]
                            return out
                        return pd.DataFrame()

                    df_yf = dl_chunked(ticker, start, end)

            if df_yf is None or df_yf.empty:
                st.error("No data returned. Try the other ticker variant (5024.KL ↔ HUPSENG.KL) or switch source.")
            else:
                df_yf = df_yf.rename(columns={"Adj Close":"Close"})
                need = ["Open","High","Low","Close"]
                miss = [c for c in need if c not in df_yf.columns]
                if miss:
                    st.error(f"Downloaded data missing columns: {miss}")
                else:
                    os.makedirs(OHLC_DIR, exist_ok=True)
                    out_path = os.path.join(OHLC_DIR, f"{_sanitize_name(stock_name)}.csv")
                    out = df_yf[["Open","High","Low","Close"]].copy()
                    if "Volume" in df_yf.columns:
                        out["Volume"] = df_yf["Volume"]
                    out.index.name = "Date"
                    out = out.reset_index().sort_values("Date")
                    out.to_csv(out_path, index=False)
                    st.success(f"Saved {len(out):,} rows → {out_path}")
                    if interval != "1d":
                        st.info("Momentum requires DAILY data (≥252 rows) for 12-month return and ≥200 for MA200.")
                    st.dataframe(out.tail(10), use_container_width=True)

    else:  # Alpha Vantage
        st.caption("Alpha Vantage daily adjusted (free). Create a free API key at alphavantage.co.")
        symbol  = st.text_input("Symbol", value="5024.KL", key="av_symbol")
        api_key = st.text_input("API Key", type="password", help="Get one at alphavantage.co")
        btn2 = st.button("Fetch & Save", use_container_width=True)

        if btn2:
            if not api_key:
                st.error("Please enter your API key.")
            else:
                with st.spinner("Downloading from Alpha Vantage…"):
                    url = (
                        "https://www.alphavantage.co/query?"
                        "function=TIME_SERIES_DAILY_ADJUSTED"
                        f"&symbol={symbol}&outputsize=full&apikey={api_key}&datatype=csv"
                    )
                    r = requests.get(url, timeout=30)
                    if r.status_code != 200:
                        st.error(f"HTTP {r.status_code}: failed to fetch.")
                    else:
                        df = pd.read_csv(pd.compat.StringIO(r.text)) if hasattr(pd.compat, "StringIO") \
                             else pd.read_csv(pd.io.common.StringIO(r.text))
                        if df is None or df.empty or "timestamp" not in df.columns:
                            st.error("Alpha Vantage returned no data.")
                        else:
                            df = df.rename(columns={
                                "timestamp":"Date", "open":"Open", "high":"High", "low":"Low",
                                "adjusted_close":"Close", "volume":"Volume"
                            })[["Date","Open","High","Low","Close","Volume"]].sort_values("Date")
                            os.makedirs(OHLC_DIR, exist_ok=True)
                            out_path = os.path.join(OHLC_DIR, f"{_sanitize_name(stock_name)}.csv")
                            df.to_csv(out_path, index=False)
                            st.success(f"Saved {len(df):,} rows → {out_path}")
                            st.dataframe(df.tail(10), use_container_width=True)



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
