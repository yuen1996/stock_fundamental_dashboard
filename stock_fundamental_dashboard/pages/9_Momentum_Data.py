# pages/9_Momentum_Data.py
from __future__ import annotations
import os, re, io
from datetime import date, timedelta
import pandas as pd
import numpy as np
import streamlit as st

# --- Helpers from your project (works whether utils/ folder exists or not)
try:
    from io_helpers import load_data, load_ohlc
except Exception:
    from utils.io_helpers import load_data, load_ohlc  # fallback

# --- Optional Yahoo Finance
try:
    import yfinance as yf
    _YF_OK = True
except Exception:
    _YF_OK = False

OHLC_DIR = "data/ohlc"

# ---------- Styling ----------
st.set_page_config(page_title="Momentum Data (CSV or Yahoo)", page_icon="📈", layout="wide")
BASE_CSS = """
<style>
.sec{margin:16px 0 4px 0;}
.sec .t{font-weight:700}
.sec .d{opacity:.85}
.badge{display:inline-block;padding:.1rem .5rem;border-radius:.5rem;background:#eef2ff;color:#1d4ed8;font-weight:600;font-size:.8rem}
.code{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace}
.tbl caption{font-weight:600;margin-bottom:4px}
.info{background:#eef6ff;border:1px solid #cfe4ff;padding:.6rem .8rem;border-radius:.5rem}
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

st.title("📈 Momentum Data — Daily OHLC")
st.write(
    "Two options:\n"
    "1) **Import directly from Yahoo Finance** (pick ticker + date range), or\n"
    "2) **Upload a CSV** — TradingView (`time,close[,Volume]`) or generic OHLC (`Date,Open,High,Low,Close[,Volume]`).\n"
    "Saved as **`data/ohlc/<Stock_Name>.csv`** and used by your Momentum pillar."
)

# -------------------------- Utilities --------------------------
def _safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_")

def _ensure_dir():
    os.makedirs(OHLC_DIR, exist_ok=True)

def _pick(cols_map, *names):
    """Pick a column by any of the provided candidate names (case-insensitive)."""
    for n in names:
        if n.lower() in cols_map:
            return cols_map[n.lower()]
    return None

def _parse_time_series(s: pd.Series) -> pd.Series:
    """Robust date parsing: epoch ms/s, DD/MM/YYYY, YYYY-MM-DD."""
    try:
        # numeric-like?
        if s.dropna().astype(str).str.fullmatch(r"\d+").all():
            v = pd.to_numeric(s, errors="coerce")
            if v.max() and v.max() > 1e12:    # epoch ms
                dt = pd.to_datetime(v, unit="ms", utc=True)
            else:                              # epoch s
                dt = pd.to_datetime(v, unit="s",  utc=True)
            return dt.tz_localize(None)
    except Exception:
        pass
    d1 = pd.to_datetime(s, errors="coerce", dayfirst=True)
    # if dayfirst confuses, try default
    if d1.isna().mean() > 0.4:
        d2 = pd.to_datetime(s, errors="coerce")
        d2 = d2.dt.tz_localize(None) if hasattr(d2, "dt") else d2
        return d2
    return d1.dt.tz_localize(None)

def _normalize_csv_to_ohlc(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Accepts a raw DataFrame and returns a normalized OHLC DataFrame with:
    Date, Open, High, Low, Close[, Volume]
    Also returns diagnostics.
    """
    diag = {"raw_rows": len(raw), "parsed_rows": 0, "dropped_rows": 0,
            "duplicate_dates_removed": 0, "date_min": None, "date_max": None,
            "used_columns": {}}

    colmap = {c.lower(): c for c in raw.columns}
    # candidate names
    date_col  = _pick(colmap, "date", "time", "timestamp")
    close_col = _pick(colmap, "close", "closing", "adj close", "adj_close", "price")
    open_col  = _pick(colmap, "open")
    high_col  = _pick(colmap, "high")
    low_col   = _pick(colmap, "low")
    vol_col   = _pick(colmap, "volume", "vol")

    if date_col is None or close_col is None:
        raise ValueError("CSV must have at least Date/time and Close columns.")

    df = raw.copy()
    df["Date"] = _parse_time_series(df[date_col])
    for c in [open_col, high_col, low_col, close_col, vol_col]:
        if c is not None:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # TradingView export: keep Close only if no OHLC
    if open_col and high_col and low_col:
        out = df[["Date", open_col, high_col, low_col, close_col] + ([vol_col] if vol_col else [])].copy()
        out.columns = ["Date", "Open", "High", "Low", "Close"] + (["Volume"] if vol_col else [])
    else:
        out = df[["Date", close_col] + ([vol_col] if vol_col else [])].copy()
        out.rename(columns={close_col: "Close", vol_col: "Volume" if vol_col else "Volume"}, inplace=True)
        # Create dummy O/H/L from Close so downstream calc still works
        out["Open"] = out["Close"]
        out["High"] = out["Close"]
        out["Low"]  = out["Close"]
        # reorder
        keep = ["Date", "Open", "High", "Low", "Close"]
        if "Volume" in out.columns: keep.append("Volume")
        out = out[keep]

    # clean
    out = out[~(out["Date"].isna() | out["Close"].isna())].copy()
    before = len(out)
    out = out.drop_duplicates(subset=["Date"], keep="last")
    dup_removed = before - len(out)

    out = out.sort_values("Date").reset_index(drop=True)
    diag.update({
        "parsed_rows": len(out),
        "dropped_rows": diag["raw_rows"] - len(out),
        "duplicate_dates_removed": dup_removed,
        "date_min": out["Date"].min(),
        "date_max": out["Date"].max(),
        "used_columns": {"date": date_col, "close": close_col}
    })
    return out, diag

def _save_csv_for(name: str, df: pd.DataFrame):
    _ensure_dir()
    path = os.path.join(OHLC_DIR, f"{_safe_name(name)}.csv")
    df.to_csv(path, index=False)
    return path

def _existing_path(name: str) -> str:
    return os.path.join(OHLC_DIR, f"{_safe_name(name)}.csv")

def _first_mid_last(df: pd.DataFrame, n: int = 10):
    if df is None or df.empty:
        return df, df, df
    d = df.sort_values("Date").reset_index(drop=True)
    head = d.head(n)
    tail = d.tail(n)
    if len(d) <= 2*n:
        mid = d.iloc[0:0].copy()
    else:
        start = (len(d)//2) - (n//2)
        mid = d.iloc[start:start+n].copy()
    return head, mid, tail

# ---------- NEW: Coverage stats helpers (non-breaking) ----------
def _window_stats(df: pd.DataFrame, start_d: date, end_d: date):
    """Compute row counts and coverage between start_d..end_d (inclusive)."""
    if df is None or df.empty:
        return None
    s = pd.Timestamp(start_d)
    e = pd.Timestamp(end_d)
    d = df[(df["Date"] >= s) & (df["Date"] <= e)].copy().sort_values("Date")
    rows = len(d)
    first = d["Date"].min()
    last = d["Date"].max()
    cal_days = (e - s).days + 1
    biz_days = len(pd.bdate_range(s, e))
    coverage = (rows / biz_days) if biz_days else None
    have_ma200 = rows >= 200
    have_ret12m = rows >= 252
    gaps = d["Date"].diff().dt.days.dropna()
    max_gap = int(gaps.max()) if len(gaps) else 0
    mean_gap = float(gaps.mean()) if len(gaps) else 0.0
    return {
        "rows": rows, "first": first, "last": last,
        "biz_days": biz_days, "cal_days": cal_days, "coverage": coverage,
        "have_ma200": have_ma200, "have_ret12m": have_ret12m,
        "max_gap_days": max_gap, "mean_gap_days": mean_gap,
    }

def _render_stats(stats, label="Selected range"):
    if not stats:
        st.info("No rows in selected range.")
        return
    ok_ma = "✅" if stats["have_ma200"] else "❌"
    ok_12 = "✅" if stats["have_ret12m"] else "❌"
    cov = f"{stats['coverage']*100:.1f}%" if stats["coverage"] is not None else "—"
    st.markdown(
        f"**{label}:** {stats['first'].date()} → {stats['last'].date()}  "
        f"• rows: **{stats['rows']:,}**  • business days: **{stats['biz_days']:,}**  "
        f"• coverage: **{cov}**  • MA200: {ok_ma}  • 12-mo return: {ok_12}  "
        f"• gaps (max/avg days): {stats['max_gap_days']}/{stats['mean_gap_days']:.1f}"
    )

@st.cache_data(show_spinner=False)
def _fetch_yahoo_ohlc(ticker: str, start_d: date, end_d: date) -> pd.DataFrame:
    """
    Fetch daily OHLC from Yahoo Finance and normalize to:
    Date, Open, High, Low, Close[, Volume]
    Uses Ticker.history() (robust) with a download() fallback.
    """
    if not _YF_OK:
        raise RuntimeError("yfinance is not installed in this environment.")

    sym = str(ticker).strip()
    if not sym:
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume"])

    # 1) Primary path: Ticker.history (more reliable than download for some symbols)
    try:
        tk = yf.Ticker(sym)
        df = tk.history(
            start=start_d,
            end=end_d + timedelta(days=1),  # inclusive end
            interval="1d",
            auto_adjust=False,
            actions=False,
        )
    except Exception:
        df = None

    # 2) Fallback path: download()
    if df is None or df.empty:
        try:
            df = yf.download(
                sym,
                start=start_d.isoformat(),
                end=(end_d + timedelta(days=1)).isoformat(),
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=False,
            )
        except Exception:
            df = None

    if df is None or df.empty:
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume"])

    # Some yfinance versions return MultiIndex columns on fallback; flatten
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            pass

    # Normalize columns
    df = df.reset_index()
    rename = {
        "Date": "Date",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Adj Close": "AdjClose",
        "Volume": "Volume",
    }
    df = df.rename(columns=rename)

    # Coerce types & drop bad rows
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if hasattr(df["Date"], "dt"):
        # drop tz if present
        try:
            df["Date"] = df["Date"].dt.tz_localize(None)
        except Exception:
            pass

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[~(df["Date"].isna() | df["Close"].isna())].copy()
    df = df.drop_duplicates(subset=["Date"], keep="last").sort_values("Date").reset_index(drop=True)

    keep = ["Date", "Open", "High", "Low", "Close"]
    if "Volume" in df.columns:
        keep.append("Volume")
    return df[keep]

# -------------------------- Load master list --------------------------
master = load_data()
stocks = sorted(master["Name"].dropna().unique().tolist()) if master is not None and not master.empty else []

# -------------- Guidance + Danger zone --------------
st.markdown(
    '<div class="sec info">'
    '<div class="t">How much data do I need?</div>'
    '<div class="d">'
    '• Upload <b>daily</b> prices (TradingView export is OK: <code>time,close[,Volume]</code>). '
    '• <b>≥200</b> daily rows for <b>MA200</b>. '
    '• <b>≥252</b> daily rows for <b>12-month return</b>. '
    '• Files are saved as <code>data/ohlc/&lt;Stock_Name&gt;.csv</code> and used by the Momentum spoke.'
    '</div></div>', unsafe_allow_html=True
)

with st.expander("🧨 Danger zone — bulk delete", expanded=False):
    ok = st.checkbox("I understand this will remove ALL momentum CSV files.", key="mom_bulk_confirm")
    if st.button("Delete ALL momentum files (data/ohlc/*.csv)", type="primary", disabled=not ok):
        try:
            _ensure_dir()
            removed = 0
            for fn in os.listdir(OHLC_DIR):
                if fn.lower().endswith(".csv"):
                    os.remove(os.path.join(OHLC_DIR, fn)); removed += 1
            st.success(f"Deleted {removed} file(s).")
        except Exception as e:
            st.error(f"Failed to delete: {e}")

# -------------------------- Main UI --------------------------
if not stocks:
    st.warning("No stock list found in master data. Add names to your master file first.")
else:
    for name in stocks:
        with st.expander(name, expanded=False):
            # show existing file quick stats
            existing = load_ohlc(name)
            if existing is not None and not existing.empty:
                pth = _existing_path(name)
                st.caption(f"Existing file: **{pth}**  •  rows: **{len(existing):,}**  •  range: "
                           f"**{existing['Date'].min().date()} → {existing['Date'].max().date()}**")
                h, m, t = _first_mid_last(existing, 10)
                cA, cB, cC = st.columns(3)
                with cA:
                    st.caption("First 10")
                    st.dataframe(h, use_container_width=True, height=220)
                with cB:
                    st.caption("Middle 10")
                    st.dataframe(m, use_container_width=True, height=220)
                with cC:
                    st.caption("Last 10")
                    st.dataframe(t, use_container_width=True, height=220)

                # ---- NEW: Coverage for full file + window checker ----
                try:
                    full_start = existing["Date"].min().date()
                    full_end = existing["Date"].max().date()
                    stats_full = _window_stats(existing, full_start, full_end)
                    _render_stats(stats_full, "Full file range")

                    min_d = full_start
                    max_d = full_end
                    default_start = max_d - timedelta(days=400)
                    st.caption("Check a time window")
                    w1, w2 = st.columns(2)
                    with w1:
                        win_s = st.date_input("Window start", value=max(min_d, default_start),
                                              min_value=min_d, max_value=max_d, key=f"ws_{name}")
                    with w2:
                        win_e = st.date_input("Window end", value=max_d,
                                              min_value=min_d, max_value=max_d, key=f"we_{name}")
                    if win_s > win_e:
                        st.error("Window start must be before end date.")
                    else:
                        stats_sel = _window_stats(existing, win_s, win_e)
                        _render_stats(stats_sel, "Selected window")
                except Exception as _e:
                    st.warning(f"Could not compute window stats: {_e}")
            else:
                st.caption("No existing file found for this stock.")

            st.divider()

            # --- Import from Yahoo Finance
            st.markdown('<div class="sec"><div class="t">Import from Yahoo Finance</div>'
                        '<div class="d">Provide the Yahoo ticker (with exchange suffix, e.g. <code>7113.KL</code>) '
                        'and choose your date range.</div></div>', unsafe_allow_html=True)

            default_ticker = name  # you can change in UI
            today = date.today()
            default_start = today - timedelta(days=365*5)
            default_end   = today

            cols_y = st.columns([2, 2, 2, 1])
            with cols_y[0]:
                yf_ticker = st.text_input("Yahoo ticker", value=default_ticker, key=f"yf_t_{name}")
            with cols_y[1]:
                start_d = st.date_input("Start", value=default_start, key=f"yf_s_{name}")
            with cols_y[2]:
                end_d = st.date_input("End", value=default_end, key=f"yf_e_{name}")
            with cols_y[3]:
                st.write("")  # spacer
                fetch = st.button("Fetch", key=f"yf_btn_{name}", use_container_width=True)

            if fetch:
                if not _YF_OK:
                    st.error("`yfinance` is not installed in this environment.")
                elif not yf_ticker:
                    st.error("Please enter a Yahoo ticker (e.g., 7113.KL).")
                elif start_d > end_d:
                    st.error("Start date must be before End date.")
                else:
                    with st.spinner("Contacting Yahoo Finance..."):
                        try:
                            df_y = _fetch_yahoo_ohlc(yf_ticker.strip(), start_d, end_d)
                        except Exception as e:
                            df_y = None
                            st.error(f"Yahoo fetch failed: {e}")

                    if df_y is not None and not df_y.empty:
                        h, m, t = _first_mid_last(df_y, 10)
                        st.markdown("**Preview (first / middle / last 10)**")
                        p1, p2, p3 = st.columns(3)
                        with p1:
                            st.caption("First 10")
                            st.dataframe(h, use_container_width=True, height=240)
                        with p2:
                            st.caption("Middle 10")
                            st.dataframe(m, use_container_width=True, height=240)
                        with p3:
                            st.caption("Last 10")
                            st.dataframe(t, use_container_width=True, height=240)

                        # ---- NEW: Coverage for the fetched date range ----
                        try:
                            stats_fetch = _window_stats(df_y, start_d, end_d)
                            _render_stats(stats_fetch, "Fetched range")
                        except Exception as _e:
                            st.warning(f"Could not compute coverage: {_e}")

                        path = _save_csv_for(name, df_y)
                        st.success(f"Saved {len(df_y):,} rows to **{path}**")

                        try:
                            st.rerun()
                        except Exception:
                            st.experimental_rerun()
                    else:
                        st.warning("No rows returned for that date range / ticker.")

            st.divider()

            # --- Upload / Replace CSV (kept same behavior as before)
            st.markdown('<div class="sec"><div class="t">Upload / Replace CSV</div>'
                        '<div class="d">TradingView (<code>time,close[,Volume]</code>) '
                        'or generic OHLC (<code>Date,Open,High,Low,Close[,Volume]</code>).</div></div>',
                        unsafe_allow_html=True)

            uploaded = st.file_uploader("Upload CSV", type=["csv"], key=f"up_{name}")
            if uploaded is not None:
                try:
                    raw_bytes = uploaded.read()
                    raw = pd.read_csv(io.BytesIO(raw_bytes))
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

                    # ---- NEW: Coverage for the uploaded full file range ----
                    try:
                        full_start = norm["Date"].min().date()
                        full_end = norm["Date"].max().date()
                        stats_file = _window_stats(norm, full_start, full_end)
                        _render_stats(stats_file, "Full file range")
                    except Exception as _e:
                        st.warning(f"Could not compute coverage: {_e}")

                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to process CSV: {e}")

            # --- Delete this stock's file
            st.markdown('<div class="sec"><div class="t">Replace / Delete existing file</div></div>',
                        unsafe_allow_html=True)
            colA, colB, colC = st.columns([1, 1, 3])
            with colA:
                st.caption("Remove the existing file for this stock.")
                if st.button("Delete existing CSV", key=f"del_{name}"):
                    p = _existing_path(name)
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                            st.success(f"Deleted {p}")
                            try:
                                st.rerun()
                            except Exception:
                                st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Failed to delete: {e}")
                    else:
                        st.info("Nothing to delete.")
            with colB:
                st.caption("Upload again anytime — the new file will **replace** the old one.")

