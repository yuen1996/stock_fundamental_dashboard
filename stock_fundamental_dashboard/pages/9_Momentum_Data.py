# pages/9_Momentum_Data.py
from __future__ import annotations

# --- Auth ---
from auth_gate import require_auth
require_auth()

# --- UI wiring (no CSS here; uses your utils/ui) ---
import streamlit as st
from utils.ui import setup_page, section, render_page_title

# --- Std libs ---
import os, io, re
from datetime import date, timedelta
import pandas as pd
import numpy as np

# --- Project helpers ---
try:
    from io_helpers import load_data   # master list
except Exception:
    from utils.io_helpers import load_data  # fallback if running from /pages

# --- Optional Yahoo Finance ---
try:
    import yfinance as yf
    _YF_OK = True
except Exception:
    _YF_OK = False

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
setup_page("Momentum Data")  # uses your global UI styling
render_page_title("Momentum Data")

OHLC_DIR = "data/ohlc"

def _safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_")

def _ensure_dir():
    os.makedirs(OHLC_DIR, exist_ok=True)

def _ohlc_path(name: str) -> str:
    return os.path.join(OHLC_DIR, f"{_safe_name(name)}.csv")

def _load_ohlc_local(name: str) -> pd.DataFrame | None:
    """Local loader for previews; your Snowflake/summary can read the same files."""
    try:
        p = _ohlc_path(name)
        if not os.path.exists(p):
            return None
        df = pd.read_csv(p)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            if hasattr(df["Date"], "dt"):
                try: df["Date"] = df["Date"].dt.tz_localize(None)
                except Exception: pass
        for c in ["Open","High","Low","Close","Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Date","Close"]).drop_duplicates(subset=["Date"]).sort_values("Date")
        return df.reset_index(drop=True)
    except Exception:
        return None

def _save_ohlc(name: str, df: pd.DataFrame) -> str:
    _ensure_dir()
    path = _ohlc_path(name)
    df.to_csv(path, index=False)
    return path

def _parse_time_series(s: pd.Series) -> pd.Series:
    # handle unix seconds/millis or normal dates
    try:
        if s.dropna().astype(str).str.fullmatch(r"\d+").all():
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
        d2 = pd.to_datetime(s, errors="coerce")
        return d2.dt.tz_localize(None) if hasattr(d2, "dt") else d2
    return d1.dt.tz_localize(None)

# -- UPDATED: punctuation-tolerant column matcher --------------------------------
def _colkey(s: str) -> str:
    # normalize: lowercase and strip punctuation/spaces (e.g., "Vol." -> "vol")
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def _pick(colmap: dict[str,str], *names) -> str | None:
    # colmap is typically {c.lower(): c}; match using normalized keys
    norm = {_colkey(k): v for k, v in colmap.items()}
    for n in names:
        k = _colkey(n)
        if k in norm:
            return norm[k]
    return None

# -- NEW: parse volumes like '592.30K', '1.17M', '5.43B' -------------------------
def _parse_human_volume(s: pd.Series) -> pd.Series:
    """
    Convert '274.50K', '1.17M', '5.43B', '—', '-' -> float shares.
    Leaves normal numeric strings as-is.
    """
    def to_num(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        t = str(x).strip().replace(",", "")
        if t in ("", "-", "—"):
            return np.nan
        m = re.match(r'^([+-]?\d*\.?\d+)\s*([KMB]?)$', t, flags=re.I)
        if not m:
            try:
                return float(t)
            except Exception:
                return np.nan
        num = float(m.group(1))
        suf = (m.group(2) or "").upper()
        mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get(suf, 1.0)
        return num * mult
    return s.apply(to_num)

# -- UPDATED: normalizer supports Investing.com headers --------------------------
def _normalize_csv_to_ohlc(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts:
      • TradingView: time, close[, volume]
      • Generic OHLC: Date, Open, High, Low, Close[, Volume]
      • Investing.com: Date, Price, Open, High, Low, Vol., Change %
    Returns standard: Date, Open, High, Low, Close[, Volume]
    """
    colmap = {c.lower(): c for c in raw.columns}

    # tolerate "Price" for Close; tolerate "Vol." for Volume
    date_col  = _pick(colmap, "date", "time", "timestamp")
    close_col = _pick(colmap, "close", "closing", "adj close", "adj_close", "price")
    open_col  = _pick(colmap, "open")
    high_col  = _pick(colmap, "high")
    low_col   = _pick(colmap, "low")
    vol_col   = _pick(colmap, "volume", "vol", "vol.", "volumen")  # add variants

    if date_col is None or close_col is None:
        raise ValueError("CSV must have at least Date/time and Close/Price columns.")

    df = raw.copy()

    # Parse Date/time robustly (handles dd/mm, mm/dd, and unix seconds/millis)
    df["Date"] = _parse_time_series(df[date_col])

    # Coerce nums for OHLC
    for c in [open_col, high_col, low_col, close_col]:
        if c is not None and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Parse Vol. like '592.30K' -> 592300
    if vol_col is not None and vol_col in df.columns:
        df[vol_col] = _parse_human_volume(df[vol_col])

    # Build output
    if open_col and high_col and low_col:
        out = df[["Date", open_col, high_col, low_col, close_col] + ([vol_col] if vol_col else [])].copy()
        out.columns = ["Date", "Open", "High", "Low", "Close"] + (["Volume"] if vol_col else [])
    else:
        # synth OHLC from Close if no O/H/L present
        out = df[["Date", close_col] + ([vol_col] if vol_col else [])].copy()
        out.rename(columns={close_col: "Close"}, inplace=True)
        if vol_col:
            out.rename(columns={vol_col: "Volume"}, inplace=True)
        out["Open"] = out["Close"]; out["High"] = out["Close"]; out["Low"] = out["Close"]
        keep = ["Date", "Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in out.columns else [])
        out = out[keep]

    # Clean & sort
    out = (
        out[~(out["Date"].isna() | out["Close"].isna())]
        .drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    return out

@st.cache_data(show_spinner=False)
def _fetch_yahoo_ohlc(ticker: str, start_d: date, end_d: date) -> pd.DataFrame:
    """
    Fetch daily OHLC via yfinance and normalize to: Date, Open, High, Low, Close[, Volume]
    """
    if not _YF_OK:
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume"])

    sym = str(ticker).strip()
    if not sym:
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume"])

    try:
        tk = yf.Ticker(sym)
        df = tk.history(
            start=start_d,
            end=end_d + timedelta(days=1),
            interval="1d",
            auto_adjust=False,
            actions=False,
        )
    except Exception:
        df = None

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

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            pass

    df = df.reset_index()
    df = df.rename(columns={
        "Date":"Date", "Open":"Open", "High":"High", "Low":"Low",
        "Close":"Close", "Adj Close":"AdjClose", "Volume":"Volume"
    })

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if hasattr(df["Date"], "dt"):
        try: df["Date"] = df["Date"].dt.tz_localize(None)
        except Exception: pass

    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Date","Close"]).drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    keep = ["Date","Open","High","Low","Close"] + (["Volume"] if "Volume" in df.columns else [])
    return df[keep]

# ------------------------------------------------------------------
# Page content
# ------------------------------------------------------------------

st.markdown(
    section(
        "What this does",
        "Store daily prices per stock (CSV or Yahoo). Files live at data/ohlc/<Stock_Name>.csv and are used by your Snowflake Momentum spoke."
    ),
    unsafe_allow_html=True
)

master = load_data()
stocks = sorted(master["Name"].dropna().astype(str).unique().tolist()) if (master is not None and not master.empty and "Name" in master.columns) else []

if not stocks:
    st.warning("No stock list found in your master data. Add names on the Add/Edit page first.")
    st.stop()

# Bulk tools
with st.expander("🧨 Danger zone — bulk delete", expanded=False):
    _ensure_dir()
    ok = st.checkbox("I understand this will remove ALL momentum CSV files under data/ohlc/", key="mom_bulk_confirm")
    if st.button("Delete ALL momentum files", type="primary", disabled=not ok):
        removed = 0
        for fn in os.listdir(OHLC_DIR):
            if fn.lower().endswith(".csv"):
                try:
                    os.remove(os.path.join(OHLC_DIR, fn)); removed += 1
                except Exception:
                    pass
        st.success(f"Deleted {removed} file(s).")

for name in stocks:
    safe = _safe_name(name)
    with st.expander(name, expanded=False):
        # Existing file quick info
        existing = _load_ohlc_local(name)
        if existing is not None and not existing.empty:
            pth = _ohlc_path(name)
            st.caption(
                f"Existing file: **{pth}** · rows: **{len(existing):,}** · "
                f"range: **{existing['Date'].min().date()} → {existing['Date'].max().date()}**"
            )
            # small previews
            prev = existing.copy()
            if "Volume" not in prev.columns:
                prev["Volume"] = np.nan
            c1, c2, c3 = st.columns(3)
            with c1:
                st.caption("First 10")
                st.dataframe(prev.head(10)[["Date","Open","High","Low","Close","Volume"]], use_container_width=True, height=220, key=f"first_{safe}")
            with c2:
                st.caption("Middle 10")
                mid_start = max(0, len(prev)//2 - 5)
                st.dataframe(prev.iloc[mid_start:mid_start+10][["Date","Open","High","Low","Close","Volume"]], use_container_width=True, height=220, key=f"mid_{safe}")
            with c3:
                st.caption("Last 10")
                st.dataframe(prev.tail(10)[["Date","Open","High","Low","Close","Volume"]], use_container_width=True, height=220, key=f"last_{safe}")
        else:
            st.caption("No existing momentum file found for this stock.")

        st.divider()

        # --- Import via Yahoo Finance ---
        st.markdown(section("Import from Yahoo Finance", "Provide the Yahoo ticker (e.g., 7113.KL) and a date range."), unsafe_allow_html=True)
        today = date.today()
        default_start = today - timedelta(days=365*5)
        col_y1, col_y2, col_y3, col_y4 = st.columns([2, 2, 2, 1])
        with col_y1:
            yf_ticker = st.text_input("Yahoo ticker", value=name, key=f"yf_t_{safe}")
        with col_y2:
            start_d = st.date_input("Start", value=default_start, key=f"yf_s_{safe}")
        with col_y3:
            end_d = st.date_input("End", value=today, key=f"yf_e_{safe}")
        with col_y4:
            st.write("")
            fetch = st.button("Fetch", key=f"yf_btn_{safe}", use_container_width=True)

        if fetch:
            if not _YF_OK:
                st.error("`yfinance` is not installed in this environment.")
            elif not yf_ticker:
                st.error("Please enter a Yahoo ticker, e.g. 7113.KL")
            elif start_d > end_d:
                st.error("Start date must be before End date.")
            else:
                with st.spinner("Fetching daily prices…"):
                    df_y = _fetch_yahoo_ohlc(yf_ticker.strip(), start_d, end_d)

                if df_y is None or df_y.empty:
                    st.warning("No rows returned for that range / ticker.")
                else:
                    if "Volume" not in df_y.columns:
                        df_y["Volume"] = np.nan
                    st.caption(f"Preview — fetched **{len(df_y):,}** rows for `{yf_ticker.strip()}`")
                    p1, p2, p3 = st.columns(3)
                    with p1: st.dataframe(df_y.head(10), use_container_width=True, height=240, key=f"yf_prev1_{safe}")
                    with p2:
                        mid = df_y.iloc[max(0, len(df_y)//2 - 5): max(0, len(df_y)//2 - 5) + 10]
                        st.dataframe(mid, use_container_width=True, height=240, key=f"yf_prev2_{safe}")
                    with p3: st.dataframe(df_y.tail(10), use_container_width=True, height=240, key=f"yf_prev3_{safe}")

                    path = _save_ohlc(name, df_y)
                    st.success(f"Saved to **{path}**")
                    st.download_button("⬇️ Download fetched CSV", data=df_y.to_csv(index=False).encode("utf-8"),
                                       file_name=f"{_safe_name(name)}.csv", mime="text/csv",
                                       use_container_width=False, key=f"yf_dl_{safe}")
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()

        st.divider()

        # --- Upload / Replace CSV ---
        st.markdown(section(
            "Upload / Replace CSV",
            "TradingView, generic OHLC, **or Investing.com (Price History)** — we'll auto-detect headers (`Date, Price, Open, High, Low, Vol., Change %`)."
        ), unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key=f"up_{safe}")
        if uploaded is not None:
            try:
                # autodetect comma vs tab using python engine's sniffer
                raw = pd.read_csv(io.BytesIO(uploaded.read()), sep=None, engine="python")
                norm = _normalize_csv_to_ohlc(raw)
                path = _save_ohlc(name, norm)
                st.success(f"Saved {len(norm):,} rows → {path}")

                norm_show = norm.copy()
                if "Volume" not in norm_show.columns:
                    norm_show["Volume"] = np.nan

                cA, cB = st.columns(2)
                with cA:
                    st.caption("First 10 rows")
                    st.dataframe(norm_show.head(10), use_container_width=True, height=220, key=f"up_prev1_{safe}")
                with cB:
                    st.caption("Last 10 rows")
                    st.dataframe(norm_show.tail(10), use_container_width=True, height=220, key=f"up_prev2_{safe}")

                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to process CSV: {e}")

        st.divider()

        # --- Delete this stock’s file ---
        colD1, colD2 = st.columns([1,3])
        with colD1:
            if st.button("Delete existing CSV", key=f"del_{safe}"):
                p = _ohlc_path(name)
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
        with colD2:
            st.caption("Upload again anytime — a new file will replace the old one.")
