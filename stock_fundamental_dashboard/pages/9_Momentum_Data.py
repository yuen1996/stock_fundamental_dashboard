# pages/9_Momentum_Data.py
import os, re
import pandas as pd
import numpy as np
import streamlit as st

# --- Helpers from your project (works whether utils/ folder exists or not)
try:
    from io_helpers import load_data, load_ohlc
except Exception:
    from utils.io_helpers import load_data, load_ohlc  # fallback

# Where we store per-stock OHLC files
OHLC_DIR = "data/ohlc"

# ---------- Styling to match other pages ----------
BASE_CSS = """
<style>
:root{
  --bg:#f6f7fb; --surface:#ffffff; --text:#0f172a; --muted:#475569;
  --border:#e5e7eb; --shadow:0 8px 24px rgba(15, 23, 42, .06);
  --primary:#4f46e5; --info:#0ea5e9; --success:#10b981; --warning:#f59e0b; --danger:#ef4444;
}
.stApp{ background: radial-gradient(1000px 500px at 10% -10%, #f0f3fb 0%, var(--bg) 60%), var(--bg); }
h1,h2,h3,h4{ color:var(--text)!important; font-weight:800!important; letter-spacing:.2px; }
.sec{
  background:var(--surface); border:1px solid var(--border); border-radius:14px; box-shadow:var(--shadow);
  padding:.65rem .9rem; margin:1rem 0 .6rem 0; display:flex; align-items:center; gap:.6rem;
}
.sec .t{ font-size:1.05rem; font-weight:800; margin:0; }
.sec .d{ color:var(--muted); font-size:.95rem; margin-left:.25rem; }
.sec::before{ content:""; width:8px; height:26px; border-radius:6px; background:var(--primary); }
.sec.info::before{ background:var(--info); } .sec.success::before{ background:var(--success); }
.sec.warning::before{ background:var(--warning); } .sec.danger::before{ background:var(--danger); }
.stDataFrame td, .stDataFrame th{ border-bottom:1px solid var(--border)!important; }
</style>
"""
st.set_page_config(page_title="Momentum Data (CSV upload only)", page_icon="📈", layout="wide")
st.markdown(BASE_CSS, unsafe_allow_html=True)

st.title("📈 Momentum Data — Daily OHLC (CSV upload)")
st.write(
    "Upload **daily** price data so the **Momentum** pillar can compute **12-month return** "
    "and the **200-DMA flag**. We support **TradingView exports** (columns `time, close, Volume`) "
    "or generic OHLC CSVs (`Date, Open, High, Low, Close[, Volume]`)."
)

# -------------------------- Utilities --------------------------
def _safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_")

def _ensure_dir():
    os.makedirs(OHLC_DIR, exist_ok=True)

def _normalize_csv_to_ohlc(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts:
      - TradingView: 'time','close'[, 'Volume']  (dates like DD/MM/YYYY are OK)
      - Generic OHLC: 'Date','Open','High','Low','Close'[, 'Volume']
    Returns cols: Date, Open, High, Low, Close[, Volume]
    """
    cols = {c.lower().strip(): c for c in raw.columns}
    def has(*names): return any(n.lower() in cols for n in names)
    def pick(name):  return cols.get(name.lower())

    # TradingView shape: time, close, (optional) volume
    if has("time") and has("close"):
        out = pd.DataFrame({
            "Date":  pd.to_datetime(raw[pick("time")], errors="coerce", dayfirst=True),
            "Close": pd.to_numeric(raw[pick("close")], errors="coerce"),
        })
        # Synthesize OHLC from Close (enough for Momentum)
        out["Open"] = out["Close"]
        out["High"] = out["Close"]
        out["Low"]  = out["Close"]
        if has("volume"):
            out["Volume"] = pd.to_numeric(raw[pick("volume")], errors="coerce")

    # Generic OHLC shape
    elif has("date") and has("open") and has("high") and has("low") and has("close"):
        out = pd.DataFrame({
            "Date":  pd.to_datetime(raw[pick("date")],  errors="coerce", dayfirst=True),
            "Open":  pd.to_numeric(raw[pick("open")],   errors="coerce"),
            "High":  pd.to_numeric(raw[pick("high")],   errors="coerce"),
            "Low":   pd.to_numeric(raw[pick("low")],    errors="coerce"),
            "Close": pd.to_numeric(raw[pick("close")],  errors="coerce"),
        })
        if has("volume"):
            out["Volume"] = pd.to_numeric(raw[pick("volume")], errors="coerce")
    else:
        raise ValueError(
            "CSV must contain either (time, close[, Volume]) OR (Date, Open, High, Low, Close[, Volume])."
        )

    out = (
        out.dropna(subset=["Date", "Close"])
           .drop_duplicates(subset=["Date"])
           .sort_values("Date")
    )
    return out

def _preview_stats(df: pd.DataFrame):
    """Return (rows, last_close, (ma200, ret12)). Always a 3-tuple."""
    if df is None or getattr(df, "empty", True):
        return 0, None, (None, None)

    df = df.copy().sort_values("Date")
    px = pd.to_numeric(df["Close"], errors="coerce")
    rows = int(px.dropna().shape[0])

    last = float(px.iloc[-1]) if rows else None
    ma200 = float(px.rolling(200, min_periods=200).mean().iloc[-1]) if rows >= 200 else None

    ret12 = None
    if rows >= 252:
        base = float(px.iloc[-252])
        if base != 0:
            ret12 = float(last / base - 1.0)

    return rows, last, (ma200, ret12)

def _save_csv_for(name: str, df: pd.DataFrame):
    _ensure_dir()
    path = os.path.join(OHLC_DIR, f"{_safe_name(name)}.csv")
    df.to_csv(path, index=False)
    return path

def _existing_path(name: str) -> str:
    return os.path.join(OHLC_DIR, f"{_safe_name(name)}.csv")

# -------------------------- Load master list --------------------------
master = load_data()
stocks = sorted(master["Name"].dropna().unique().tolist()) if master is not None and not master.empty else []

# -------------- Global guidance + Danger zone --------------
st.markdown(
    '<div class="sec info">'
    '<div class="t">How much data do I need?</div>'
    '<div class="d">'
    '• Upload **daily** prices (TradingView export is OK: <code>time, close, Volume</code>). '
    '• **≥200** daily rows for **MA200**. '
    '• **≥252** daily rows for **12-month return**. '
    '• File is saved as <code>data/ohlc/&lt;Stock_Name&gt;.csv</code> and used automatically by the Snowflake Momentum spoke.'
    '</div></div>',
    unsafe_allow_html=True
)

with st.expander("🧨 Danger zone — bulk delete", expanded=False):
    ok = st.checkbox("I understand this will remove ALL momentum CSV files.", key="mom_bulk_confirm")
    if st.button("Delete ALL momentum files (data/ohlc/*.csv)",
                 type="primary", use_container_width=True, disabled=not ok):
        _ensure_dir()
        cnt = 0
        for fn in os.listdir(OHLC_DIR):
            if fn.lower().endswith(".csv"):
                try:
                    os.remove(os.path.join(OHLC_DIR, fn)); cnt += 1
                except Exception:
                    pass
        st.success(f"Removed {cnt} file(s).")
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

st.divider()

# -------------- Per-stock grouped UI --------------
if not stocks:
    st.info("No stocks found in your main dataset. Add a stock first, then return here to upload momentum CSV.")
else:
    for name in stocks:
        with st.expander(name, expanded=False):
            # Current status (if any)
            existing = load_ohlc(name)
            rows, last_close, (ma200, ret12) = _preview_stats(existing)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows (days)", f"{rows:,}")
            c2.metric("Last Close", f"{last_close:,.4f}" if last_close is not None else "–")
            c3.metric("MA200", f"{ma200:,.4f}" if ma200 is not None else "–")
            c4.metric("12-month return", f"{ret12*100:,.2f}%" if ret12 is not None else "–")

            if rows >= 252 and ma200 is not None:
                st.markdown("**Data Health:** ✅ Ready (Momentum can be scored)")
            else:
                missing = []
                if rows < 200: missing.append("need ≥200 rows for MA200")
                if rows < 252: missing.append("need ≥252 rows for 12-month return")
                st.markdown(f"**Data Health:** ⚠️ Incomplete — {', '.join(missing) if missing else 'add more daily rows'}")

            # Upload box (TradingView or generic OHLC)
            st.markdown(
                '<div class="sec"><div class="t">Upload / Replace CSV</div>'
                '<div class="d">TradingView: <code>time, close[, Volume]</code> • '
                'Generic: <code>Date,Open,High,Low,Close[,Volume]</code></div></div>',
                unsafe_allow_html=True
            )
            up = st.file_uploader(f"Upload CSV for {name}", type=["csv"], key=f"up_{name}")
            if up is not None:
                try:
                    # Auto-detect delimiter; fallback encodings
                    try:
                        raw = pd.read_csv(up, sep=None, engine="python", encoding="utf-8")
                    except UnicodeDecodeError:
                        raw = pd.read_csv(up, sep=None, engine="python", encoding="latin-1")

                    out = _normalize_csv_to_ohlc(raw)
                    path = _save_csv_for(name, out)
                    nr, lc, _ = _preview_stats(out)
                    st.success(f"Saved {nr:,} rows → {path}")
                    st.dataframe(out.tail(15), use_container_width=True, height=260)

                    # Refresh so metrics above reflect new file
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to process CSV: {e}")

            # Quick management buttons
            colA, colB, colC = st.columns([1,1,1])
            with colA:
                if st.button("Download existing CSV", use_container_width=True, key=f"dlex_{name}"):
                    if existing is None or existing.empty:
                        st.info("No file yet.")
                    else:
                        tmp = existing.copy()
                        st.download_button(
                            "Click to download",
                            data=tmp.to_csv(index=False).encode("utf-8"),
                            file_name=f"{_safe_name(name)}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key=f"dbtn_{name}"
                        )
            with colB:
                if st.button("Delete this stock file", use_container_width=True, key=f"del_{name}"):
                    p = _existing_path(name)
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                            st.success("Deleted.")
                            try:
                                st.rerun()
                            except Exception:
                                st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Failed to delete: {e}")
                    else:
                        st.info("Nothing to delete.")
            with colC:
                st.caption("Upload again anytime — the new file will **replace** the old one.")

            # Show tail for quick check (existing)
            if existing is not None and not existing.empty:
                st.dataframe(existing.tail(10), use_container_width=True, height=240)


