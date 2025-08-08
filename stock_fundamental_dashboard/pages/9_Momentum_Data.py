# pages/9_Momentum_Data.py
import os, re, io
import pandas as pd
import numpy as np
import streamlit as st

# --- Helpers from your project (works whether utils/ folder exists or not)
try:
    from io_helpers import load_data, load_ohlc
except Exception:
    from utils.io_helpers import load_data, load_ohlc  # fallback

OHLC_DIR = "data/ohlc"

# ---------- Styling (match Dashboard.py) ----------
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

/* Section header card */
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

/* Inputs */
div[data-baseweb="input"] input, textarea, .stNumberInput input{
  font-size:15px !important;
}
.stSlider > div [data-baseweb="slider"]{ margin-top:.25rem; }

/* Buttons */
.stButton>button{
  border-radius:12px !important; padding:.55rem 1.1rem !important; font-weight:700;
}

/* Tabs */
.stTabs [role="tab"]{ font-size:15px !important; font-weight:600 !important; }

/* Sidebar theme */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg, #0b1220 0%, #1f2937 100%) !important;
}
[data-testid="stSidebar"] *{ color:#e5e7eb !important; }

/* Optional: hide Streamlit default page list
section[data-testid="stSidebarNav"]{ display:none !important; }
*/
</style>
"""

st.set_page_config(page_title="Momentum Data (CSV upload only)", page_icon="📈", layout="wide")
st.markdown(BASE_CSS, unsafe_allow_html=True)

st.title("📈 Momentum Data — Daily OHLC (CSV upload)")
st.write(
    "Upload **daily** price data so the **Momentum** pillar can compute **12-month return** "
    "and the **200-DMA flag**. We support **TradingView exports** (columns `time, close[, Volume]`) "
    "or generic OHLC CSVs (`Date, Open, High, Low, Close[, Volume]`)."
)

# -------------------------- Utilities --------------------------
def _safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_")

def _ensure_dir():
    os.makedirs(OHLC_DIR, exist_ok=True)

def _pick(cols_map, *names):
    for n in names:
        if n.lower() in cols_map:
            return cols_map[n.lower()]
    return None

def _parse_time_series(s: pd.Series) -> pd.Series:
    """Robust date parsing: epoch ms/s, DD/MM/YYYY, YYYY-MM-DD."""
    try:
        if np.issubdtype(s.dropna().infer_objects().dtype, np.number) or s.astype(str).str.fullmatch(r"\d+").all():
            v = pd.to_numeric(s, errors="coerce")
            if v.max() and v.max() > 1e12:
                dt = pd.to_datetime(v, unit="ms", utc=True)
            else:
                dt = pd.to_datetime(v, unit="s",  utc=True)
            return dt.tz_localize(None)
    except Exception:
        pass
    d1 = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if d1.isna().mean() > 0.4:
        d2 = pd.to_datetime(s, errors="coerce", dayfirst=False)
        return d2
    return d1

def _normalize_csv_to_ohlc(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    raw_rows = len(raw)
    cols = {c.lower().strip(): c for c in raw.columns}

    d_t = _pick(cols, "time", "date", "timestamp")
    c_t = _pick(cols, "close", "adj close", "adjusted_close", "adj_close")
    o_t = _pick(cols, "open")
    h_t = _pick(cols, "high")
    l_t = _pick(cols, "low")
    v_t = _pick(cols, "volume", "vol")

    if d_t is None or c_t is None:
        raise ValueError("CSV must contain either (time, close[, Volume]) OR (Date, Open, High, Low, Close[, Volume]).")

    date_parsed = _parse_time_series(raw[d_t])
    close_parsed = pd.to_numeric(raw[c_t], errors="coerce")

    if all(_pick(cols, n) is not None for n in ["open","high","low"]):
        df = pd.DataFrame({
            "Date":  date_parsed,
            "Open":  pd.to_numeric(raw[o_t], errors="coerce"),
            "High":  pd.to_numeric(raw[h_t], errors="coerce"),
            "Low":   pd.to_numeric(raw[l_t], errors="coerce"),
            "Close": close_parsed,
        })
    else:
        df = pd.DataFrame({
            "Date":  date_parsed,
            "Close": close_parsed,
        })
        df["Open"] = df["Close"]
        df["High"] = df["Close"]
        df["Low"]  = df["Close"]

    if v_t is not None:
        df["Volume"] = pd.to_numeric(raw[v_t], errors="coerce")

    na_date  = df["Date"].isna()
    na_close = df["Close"].isna()
    before_drop = len(df)
    df = df[~(na_date | na_close)].copy()
    dropped = before_drop - len(df)

    dup = int(df["Date"].duplicated(keep="last").sum())
    df = df.drop_duplicates(subset=["Date"], keep="last")

    df = df.sort_values("Date").reset_index(drop=True)

    diag = {
        "raw_rows": int(raw_rows),
        "parsed_rows": int(len(df)),
        "dropped_rows": int(dropped),
        "duplicate_dates_removed": int(dup),
        "date_min": (df["Date"].min().date().isoformat() if len(df) else None),
        "date_max": (df["Date"].max().date().isoformat() if len(df) else None),
        "used_columns": {
            "date": d_t, "close": c_t, "open": o_t, "high": h_t, "low": l_t, "volume": v_t
        }
    }
    return df, diag

def _preview_stats(df: pd.DataFrame):
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
    '• Upload <b>daily</b> prices (TradingView export is OK: <code>time, close[, Volume]</code>). '
    '• <b>≥200</b> daily rows for <b>MA200</b>. '
    '• <b>≥252</b> daily rows for <b>12-month return</b>. '
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

            st.markdown(
                '<div class="sec"><div class="t">Upload / Replace CSV</div>'
                '<div class="d">TradingView: <code>time, close[, Volume]</code> • '
                'Generic: <code>Date,Open,High,Low,Close[,Volume]</code></div></div>',
                unsafe_allow_html=True
            )
            up = st.file_uploader(f"Upload CSV for {name}", type=["csv"], key=f"up_{name}")
            if up is not None:
                try:
                    try:
                        raw = pd.read_csv(up, sep=None, engine="python", encoding="utf-8")
                    except UnicodeDecodeError:
                        raw = pd.read_csv(up, sep=None, engine="python", encoding="latin-1")

                    norm, diag = _normalize_csv_to_ohlc(raw)
                    path = _save_csv_for(name, norm)

                    st.success(f"Saved {len(norm):,} rows → {path}")
                    d1, d2 = st.columns(2)
                    with d1:
                        st.markdown("**Diagnostics**")
                        st.write(
                            f"- Raw rows: **{diag['raw_rows']:,}**\n"
                            f"- Parsed rows kept: **{diag['parsed_rows']:,}**\n"
                            f"- Dropped (invalid Date/Close): **{diag['dropped_rows']:,}**\n"
                            f"- Duplicate dates removed: **{diag['duplicate_dates_removed']:,}**\n"
                            f"- Date range: **{diag['date_min']} → {diag['date_max']}**\n"
                            f"- Used columns: Date=`{diag['used_columns']['date']}`, Close=`{diag['used_columns']['close']}`"
                        )
                    with d2:
                        st.markdown("**Preview**")
                        st.caption("First 10 rows")
                        st.dataframe(norm.head(10), use_container_width=True, height=220)
                        st.caption("Last 10 rows")
                        st.dataframe(norm.tail(10), use_container_width=True, height=220)

                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to process CSV: {e}")

            colA, colB, colC = st.columns([1,1,1])
            with colA:
                if st.button("Download existing CSV", use_container_width=True, key=f"dlex_{name}"):
                    if existing is None or existing.empty:
                        st.info("No file yet.")
                    else:
                        st.download_button(
                            "Click to download",
                            data=existing.to_csv(index=False).encode("utf-8"),
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

            if existing is not None and not existing.empty:
                st.dataframe(existing.tail(10), use_container_width=True, height=240)

