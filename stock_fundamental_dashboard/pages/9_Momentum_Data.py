# pages/9_Momentum_Data.py
from __future__ import annotations

# --- Auth ---
from auth_gate import require_auth
require_auth()

# --- UI wiring ---
import streamlit as st
from utils.ui import setup_page, section, render_page_title

# --- Std libs ---
import os, io, re, json, traceback
from datetime import date, timedelta, datetime
import pandas as pd
import numpy as np
import requests 

# --- Project helpers (master list) ---
try:
    from io_helpers import load_data   # master list
except Exception:
    from utils.io_helpers import load_data  # fallback if running from /pages

# --- Event bus to invalidate caches on write ---
from utils import bus

# --- Unified OHLC resolver + duplicate-proof keys ---
from utils.ohlc_resolver import load_ohlc, unique_key

# --- Optional Yahoo Finance ---
_YF_LAST_ERR = None
try:
    import yfinance as yf
    _YF_OK = True
except Exception as e:
    _YF_OK = False
    _YF_LAST_ERR = f"yfinance import failed: {e}"

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
setup_page("Momentum Data")
render_page_title("Momentum Data")

# --- Canonical & fallback OHLC write locations ---
_THIS   = os.path.dirname(__file__)
_PARENT = os.path.abspath(os.path.join(_THIS, ".."))
_GRANDP = os.path.abspath(os.path.join(_PARENT, ".."))

def _home_state_dir() -> str:
    d = os.path.join(os.path.expanduser("~"), ".app_state")
    os.makedirs(d, exist_ok=True)
    return d

def _alt_ohlc_dir() -> str:
    d = os.path.join(_home_state_dir(), "ohlc")
    os.makedirs(d, exist_ok=True)
    return d

def _resolve_ohlc_dir() -> str:
    """
    Prefer project path; allow env override; always have a home fallback.
    Env override: ST_OHLC_DIR=/path  (or SFD_OHLC_DIR=/path or OHLC_DIR=/path)
    """
    env = (os.environ.get("ST_OHLC_DIR")
           or os.environ.get("SFD_OHLC_DIR")
           or os.environ.get("OHLC_DIR"))
    if env:
        try:
            os.makedirs(env, exist_ok=True)
            return os.path.abspath(env)
        except Exception:
            pass

    # Try several sensible locations; create the first we can write to.
    candidates = [
        os.path.join(_PARENT,  "data", "ohlc"),                      # <project>/data/ohlc  ← canonical
        os.path.join(os.getcwd(), "data", "ohlc"),                   # CWD fallback
        os.path.abspath(os.path.join(_THIS, "..", "data", "ohlc")),  # pages/.. fallback
        os.path.join(_GRANDP,  "data", "ohlc"),                      # legacy/grandparent
        _alt_ohlc_dir(),                                             # home fallback (always writable)
    ]
    for d in candidates:
        try:
            os.makedirs(d, exist_ok=True)
            return os.path.abspath(d)
        except Exception:
            continue

    # Last resort: home fallback
    return _alt_ohlc_dir()

# Force a guaranteed fallback env var, then resolve active dir
os.environ.setdefault("SFD_OHLC_DIR", _alt_ohlc_dir())
OHLC_DIR = _resolve_ohlc_dir()


# ---------------------- App-state (persisted settings) ----------------------
_ACTIVE_PREFS_PATH: str | None = None

def _prefs_paths() -> list[str]:
    """
    Keep both paths IN SYNC, so next rerun reads same mapping regardless of order:
      1) ~/.app_state/momentum_yf_prefs.json
      2) <parent_of_OHLC_DIR>/momentum_yf_prefs.json
    """
    paths: list[str] = []
    try:
        paths.append(os.path.join(_home_state_dir(), "momentum_yf_prefs.json"))
    except Exception:
        pass
    try:
        proj_dir = os.path.abspath(os.path.join(OHLC_DIR, ".."))
        os.makedirs(proj_dir, exist_ok=True)
        paths.append(os.path.join(proj_dir, "momentum_yf_prefs.json"))
    except Exception:
        pass
    # unique while preserving order
    out, seen = [], set()
    for p in paths:
        if p not in seen:
            out.append(p); seen.add(p)
    return out

def _load_prefs() -> dict:
    default = {"map": {}, "last_auto_fetch_date": None, "max_auto_per_load": 10}
    for p in _prefs_paths():
        try:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data.setdefault("map", {})
                    data.setdefault("last_auto_fetch_date", None)
                    data.setdefault("max_auto_per_load", 10)
                    globals()["_ACTIVE_PREFS_PATH"] = p
                    return data
        except Exception:
            continue
    paths = _prefs_paths()
    globals()["_ACTIVE_PREFS_PATH"] = paths[0] if paths else None
    return default

def _save_prefs(prefs: dict) -> str | None:
    """
    Write to ALL valid paths; return first successful path.
    """
    first_ok = None
    for p in _prefs_paths():
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(prefs, f, indent=2, ensure_ascii=False)
            if first_ok is None:
                first_ok = p
        except Exception:
            pass
    if first_ok:
        globals()["_ACTIVE_PREFS_PATH"] = first_ok
    return first_ok

def _safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_")

def _ticker_key_variants(name: str) -> list[str]:
    raw = str(name).strip()
    safe = _safe_name(raw).lower()
    compact = re.sub(r"[^0-9A-Za-z]+", "", raw).lower()
    return [raw, safe, compact]

def _get_saved_ticker(name: str, prefs: dict | None = None) -> str | None:
    p = prefs or _load_prefs()
    m = p.get("map", {})
    for k in _ticker_key_variants(name):
        v = m.get(k)
        if v:
            return str(v).strip()
    return None

def _set_saved_ticker(name: str, ticker: str) -> str | None:
    p = _load_prefs()
    m = p.setdefault("map", {})
    for k in _ticker_key_variants(name):
        m[k] = str(ticker).strip()
    return _save_prefs(p)

def _active_prefs_path() -> str | None:
    return globals().get("_ACTIVE_PREFS_PATH")

def _mark_auto_fetch_today() -> None:
    p = _load_prefs()
    p["last_auto_fetch_date"] = date.today().isoformat()
    _save_prefs(p)

def _auto_already_done_today() -> bool:
    p = _load_prefs()
    return p.get("last_auto_fetch_date") == date.today().isoformat()

# ---------------------- IO helpers ----------------------
def _ensure_dir(d: str | None = None):
    os.makedirs(d or OHLC_DIR, exist_ok=True)

def _ohlc_path_canonical(name: str, base: str | None = None) -> str:
    b = base or OHLC_DIR
    return os.path.join(b, f"{_safe_name(name)}.csv")

def _load_ohlc_with_path(name: str) -> tuple[pd.DataFrame | None, str | None]:
    """
    Prefer the resolver (it finds + cleans + handles legacy locations).
    Fall back to canonical path if resolver can't find anything.
    """
    try:
        dfp, path, _how = load_ohlc(name)
        if dfp is not None and not dfp.empty:
            return dfp.reset_index(drop=True), path
    except Exception:
        pass

    p = _ohlc_path_canonical(name)
    if not os.path.exists(p):
        return None, None
    try:
        df = pd.read_csv(p)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            if hasattr(df["Date"], "dt"):
                try: df["Date"] = df["Date"].dt.tz_localize(None)
                except Exception: pass
        for c in ["Open","High","Low","Close","Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Date","Close"]).drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        return df, p
    except Exception:
        return None, p

def _last_csv_date(name: str) -> date | None:
    df, _p = _load_ohlc_with_path(name)
    if df is None or df.empty:
        return None
    try:
        return df["Date"].max().date()
    except Exception:
        return None

def _save_ohlc(name: str, df: pd.DataFrame) -> str:
    """
    Atomic write to canonical path; if permission denied, fall back to home ohlc dir
    and switch this session to use that dir (so subsequent reads work).
    """
    global OHLC_DIR
    def _atomic_write_to(path: str):
        tmp = f"{path}.tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)

    # 1) try canonical
    _ensure_dir()
    canon = _ohlc_path_canonical(name)
    try:
        _atomic_write_to(canon)
        try: bus.bump("ohlc")
        except Exception: pass
        return canon
    except Exception as e1:
        # 2) fall back to home ohlc dir
        alt_dir = _alt_ohlc_dir()
        _ensure_dir(alt_dir)
        OHLC_DIR = alt_dir  # switch for this session so next reads/writes are consistent
        alt = _ohlc_path_canonical(name, base=alt_dir)
        try:
            _atomic_write_to(alt)
            try: bus.bump("ohlc")
            except Exception: pass
            return alt
        except Exception as e2:
            raise RuntimeError(f"Save failed. Canonical error: {e1}; Fallback error: {e2}")

def _parse_time_series(s: pd.Series) -> pd.Series:
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

def _colkey(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def _pick(colmap: dict[str,str], *names) -> str | None:
    norm = {_colkey(k): v for k, v in colmap.items()}
    for n in names:
        k = _colkey(n)
        if k in norm:
            return norm[k]
    return None

def _parse_human_volume(s: pd.Series) -> pd.Series:
    def to_num(x):
        if pd.isna(x): return np.nan
        if isinstance(x, (int, float)): return float(x)
        t = str(x).strip().replace(",", "")
        if t in ("", "-", "—"): return np.nan
        m = re.match(r'^([+-]?\d*\.?\d+)\s*([KMB]?)$', t, flags=re.I)
        if not m:
            try: return float(t)
            except Exception: return np.nan
        num = float(m.group(1)); suf = (m.group(2) or "").upper()
        mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get(suf, 1.0)
        return num * mult
    return s.apply(to_num)

def _normalize_csv_to_ohlc(raw: pd.DataFrame) -> pd.DataFrame:
    colmap = {c.lower(): c for c in raw.columns}
    date_col  = _pick(colmap, "date", "time", "timestamp")
    close_col = _pick(colmap, "close", "closing", "adj close", "adj_close", "price")
    open_col  = _pick(colmap, "open")
    high_col  = _pick(colmap, "high")
    low_col   = _pick(colmap, "low")
    vol_col   = _pick(colmap, "volume", "vol", "vol.", "volumen")

    if date_col is None or close_col is None:
        raise ValueError("CSV must have at least Date/time and Close/Price columns.")

    df = raw.copy()
    df["Date"] = _parse_time_series(df[date_col])

    for c in [open_col, high_col, low_col, close_col]:
        if c is not None and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if vol_col is not None and vol_col in df.columns:
        df[vol_col] = _parse_human_volume(df[vol_col])

    if open_col and high_col and low_col:
        out = df[["Date", open_col, high_col, low_col, close_col] + ([vol_col] if vol_col else [])].copy()
        out.columns = ["Date", "Open", "High", "Low", "Close"] + (["Volume"] if vol_col else [])
    else:
        out = df[["Date", close_col] + ([vol_col] if vol_col else [])].copy()
        out.rename(columns={close_col: "Close"}, inplace=True)
        if vol_col: out.rename(columns={vol_col: "Volume"}, inplace=True)
        out["Open"] = out["Close"]; out["High"] = out["Close"]; out["Low"] = out["Close"]
        keep = ["Date", "Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in out.columns else [])
        out = out[keep]

    out = (out[~(out["Date"].isna() | out["Close"].isna())]
           .drop_duplicates(subset=["Date"], keep="last")
           .sort_values("Date").reset_index(drop=True))
    return out

@st.cache_data(show_spinner=False, ttl=600)
def _fetch_yahoo_ohlc(ticker: str, start_d: date, end_d: date) -> pd.DataFrame:
    """
    Robust Yahoo fetch with a real browser User-Agent. Tries download() first,
    then Ticker(...).history(). Returns normalized OHLC.
    """
    global _YF_LAST_ERR
    _YF_LAST_ERR = None

    if not _YF_OK:
        _YF_LAST_ERR = "yfinance not installed"
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume"])

    sym = str(ticker).strip()
    if not sym or start_d > end_d:
        _YF_LAST_ERR = "empty ticker or start > end"
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume"])

    # Build a session with UA (Yahoo blocks some headless/default clients)
    try:
        sess = requests.Session()
        sess.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        })
    except Exception as e:
        sess = None
        _YF_LAST_ERR = f"requests session init failed: {type(e).__name__}: {e}"

    df = None

    # Try download() first (lets us pass session)
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
            session=sess,  # <-- important
        )
    except Exception as e:
        _YF_LAST_ERR = f"yf.download({sym}) failed: {type(e).__name__}: {e}"
        df = None

    # Fallback to Ticker(..., session=sess).history()
    if df is None or df.empty:
        try:
            tk = yf.Ticker(sym, session=sess) if sess is not None else yf.Ticker(sym)
            df = tk.history(
                start=start_d,
                end=end_d + timedelta(days=1),
                interval="1d",
                auto_adjust=False,
                actions=False,
            )
            if df is None or df.empty:
                _YF_LAST_ERR = f"Ticker.history({sym}) returned empty"
        except Exception as e:
            _YF_LAST_ERR = f"Ticker.history({sym}) failed: {type(e).__name__}: {e}"
            df = None

    if df is None or df.empty:
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume"])

    # yfinance sometimes returns MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        try: df.columns = df.columns.get_level_values(0)
        except Exception: pass

    df = df.reset_index().rename(columns={
        "Date":"Date", "Open":"Open", "High":"High", "Low":"Low",
        "Close":"Close", "Adj Close":"AdjClose", "Volume":"Volume"
    })

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if hasattr(df["Date"], "dt"):
        try: df["Date"] = df["Date"].dt.tz_localize(None)
        except Exception: pass

    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Date","Close"]).drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    keep = ["Date","Open","High","Low","Close"] + (["Volume"] if "Volume" in df.columns else [])
    return df[keep]


def _yf_candidates(name: str, ticker: str) -> list[str]:
    """
    Build a small, ordered list of candidate Yahoo symbols based on the UI value + heuristics.
    """
    cands: list[str] = []
    t = (ticker or "").strip()
    if t:
        cands.append(t)

    nm = str(name).lower().strip()

    # SKP Resources Berhad
    if re.search(r"\bskp\b", nm) or "skp resources" in nm:
        for s in ("7155.KL", "SKPRESOURCES.KL"):
            if s not in cands:
                cands.append(s)

    # Genting Malaysia Berhad
    if "genting malaysia" in nm or "genm" in nm:
        for s in ("GENM.KL", "4715.KL"):
            if s not in cands:
                cands.append(s)

    # De-dup while preserving order
    out, seen = [], set()
    for s in cands:
        if s and s not in seen:
            out.append(s); seen.add(s)
    return out

# ---------------------- Auto-fetch engine ----------------------
def _auto_fetch_saved(stocks: list[str], *, limit: int = 10, force: bool = False):
    if not _YF_OK or not stocks:
        return 0, 0
    if (not force) and _auto_already_done_today():
        return 0, 0

    updated = 0
    skipped = 0
    today_d = date.today()
    prefs = _load_prefs()

    for name in stocks:
        if limit and updated >= limit:
            break

        yf_ticker = _get_saved_ticker(name, prefs)
        if not yf_ticker:
            skipped += 1
            continue

        last_d = _last_csv_date(name)
        start_d = (last_d + timedelta(days=1)) if last_d else (today_d - timedelta(days=365*5))
        end_d   = today_d

        if last_d is not None and start_d > end_d:
            skipped += 1
            continue

        df_new = _fetch_yahoo_ohlc(yf_ticker, start_d, end_d)
        if df_new is None or df_new.empty:
            skipped += 1
            continue

        existing, _p = _load_ohlc_with_path(name)
        merged = (pd.concat([existing, df_new], ignore_index=True)
                  if (existing is not None and not existing.empty) else df_new)
        merged = merged.drop_duplicates(subset=["Date"], keep="last").sort_values("Date").reset_index(drop=True)

        _save_ohlc(name, merged)
        updated += 1

    if not force:
        _mark_auto_fetch_today()

    return updated, skipped

# ---------------------- flash helpers (persist messages across reruns) --------
def flash(msg: str, level: str = "info"):
    st.session_state["__flash_msg__"] = (level, msg)

def show_flash():
    tup = st.session_state.pop("__flash_msg__", None)
    if not tup: return
    level, msg = tup
    if level == "success": st.success(msg)
    elif level == "warning": st.warning(msg)
    elif level == "error": st.error(msg)
    else: st.info(msg)

# ------------------------------------------------------------------
# Page content
# ------------------------------------------------------------------
ok_dir = os.access(OHLC_DIR, os.W_OK)
with st.expander("📂 Storage locations & health", expanded=True):
    st.markdown(section(
        "Locations",
        f"OHLC dir (active): `{OHLC_DIR}` · Writeable: **{ok_dir}**"
    ), unsafe_allow_html=True)
    st.caption(f"Prefs path in use: `{_active_prefs_path()}` (exists: {os.path.exists(_active_prefs_path() or '')})")
    try:
        files = sorted([f for f in os.listdir(OHLC_DIR) if f.lower().endswith('.csv')])
        preview = ", ".join(files[:30]) + (" …" if len(files) > 30 else "")
        st.caption(f"Files in OHLC dir ({len(files)}): {preview or '(none)'}")
    except Exception as e:
        st.caption(f"(Could not list dir: {e})")
    if _YF_LAST_ERR:
        st.caption(f"Last Yahoo error: `{_YF_LAST_ERR}`")

show_flash()

master = load_data()
stocks = sorted(master["Name"].dropna().astype(str).unique().tolist()) \
         if (master is not None and not master.empty and "Name" in master.columns) else []

if not stocks:
    st.warning("No stock list found in your master data. Add names on the Add/Edit page first.")
    st.stop()

# ---- Auto-fetch controls (global) ----
prefs = _load_prefs()
with st.expander("⚡ Auto-fetch saved tickers (daily)", expanded=True):
    colA, colB, colC = st.columns([2,1,1])
    with colA:
        auto_on = st.checkbox("Enable on first visit each day", value=True, key=unique_key("auto_on", "momentum"))
    with colB:
        max_auto = st.number_input("Max per load", 1, 100, int(prefs.get("max_auto_per_load", 10)), step=1, key=unique_key("auto_limit", "momentum"))
    with colC:
        run_now = st.button("Run now", key=unique_key("run_auto_now", "momentum"))

    if int(prefs.get("max_auto_per_load", 10)) != int(max_auto):
        prefs["max_auto_per_load"] = int(max_auto)
        _save_prefs(prefs)

    if run_now:
        with st.spinner("Running auto-fetch now…"):
            done, skipped = _auto_fetch_saved(stocks, limit=int(max_auto), force=True)
        flash(f"Auto-fetch complete: updated {done}, skipped {skipped}", "success")
        st.rerun()
    elif auto_on:
        if not _auto_already_done_today():
            with st.spinner("Checking saved tickers for new data…"):
                done, skipped = _auto_fetch_saved(stocks, limit=int(max_auto), force=False)
            if done:
                flash(f"Auto-fetch complete: updated {done} ticker(s).", "success")
                st.rerun()

with st.expander("🔧 Yahoo diagnostics", expanded=False):
    colx, coly, colz = st.columns([2,1,1])
    with colx:
        st.caption("Versions")
        try:
            st.write(f"yfinance: {getattr(yf, '__version__', '?')}")
        except Exception:
            st.write("yfinance: ?")
        import sys, pandas as _pd, numpy as _np
        st.write(f"pandas: {_pd.__version__} · numpy: {_np.__version__} · py: {sys.version.split()[0]}")
        st.caption("If fetch keeps failing, upgrade in your venv:")
        st.code("source venv/bin/activate && pip install --upgrade yfinance pandas numpy requests", language="bash")
    with coly:
        if st.button("🧹 Clear Yahoo cache", use_container_width=True, key=unique_key('clear_yf_cache','momentum')):
            try:
                _fetch_yahoo_ohlc.clear()
                flash("Yahoo cache cleared.", "success")
            except Exception as e:
                flash(f"Clear cache failed: {e}", "error")
            st.rerun()
    with colz:
        test_sym = st.text_input("Quick test symbol", value="GENM.KL", key=unique_key('yf_test_sym','momentum'))
        if st.button("Run test (30d)", use_container_width=True, key=unique_key('run_yf_test','momentum')):
            try:
                df_test = _fetch_yahoo_ohlc(test_sym, date.today()-timedelta(days=30), date.today())
                if df_test is None or df_test.empty:
                    if _YF_LAST_ERR:
                        flash(f"Test fetch empty for {test_sym}. Last error: `{_YF_LAST_ERR}`", "warning")
                    else:
                        flash(f"Test fetch empty for {test_sym}.", "warning")
                else:
                    flash(f"Test fetch OK for {test_sym}: {len(df_test)} rows, {df_test['Date'].min().date()} → {df_test['Date'].max().date()}", "success")
            except Exception as e:
                flash(f"Test fetch exception: {e}", "error")
            st.rerun()

# ---- Bulk tools -------------------------------------------------
with st.expander("🧨 Danger zone — bulk delete", expanded=False):
    ok = st.checkbox(
        "I understand this will remove ALL momentum CSV files under the active OHLC dir",
        key=unique_key("mom_bulk_confirm", "momentum")
    )
    if st.button("Delete ALL momentum files", type="primary", disabled=not ok, key=unique_key("mom_bulk_delete", "momentum")):
        removed = 0
        try:
            for fn in os.listdir(OHLC_DIR):
                if fn.lower().endswith(".csv"):
                    try:
                        os.remove(os.path.join(OHLC_DIR, fn)); removed += 1
                    except Exception:
                        pass
            try: bus.bump("ohlc")
            except Exception: pass
            flash(f"Deleted {removed} file(s).", "success")
        except Exception as e:
            flash(f"Delete failed: {e}", "error")
        st.rerun()

# ---- Per-stock panels ---------------------------------------------------------
for name in stocks:
    with st.expander(name, expanded=False):
        # --- Debug snapshot for THIS stock ---
        canon = _ohlc_path_canonical(name)
        exists = os.path.exists(canon)
        size   = (os.path.getsize(canon) if exists else 0)
        st.caption(f"Canonical path: `{canon}` · exists: **{exists}** · size: **{size:,} bytes**")

        # Existing file quick info (resolver)
        existing, pth = _load_ohlc_with_path(name)
        if existing is not None and not existing.empty:
            st.caption(
                f"Existing file: **{pth or canon}** · rows: **{len(existing):,}** · "
                f"range: **{existing['Date'].min().date()} → {existing['Date'].max().date()}**"
            )
            prev = existing.copy()
            if "Volume" not in prev.columns: prev["Volume"] = np.nan
            c1, c2, c3 = st.columns(3)
            with c1:
                st.caption("First 10")
                st.dataframe(prev.head(10)[["Date","Open","High","Low","Close","Volume"]],
                             use_container_width=True, height=220,
                             key=unique_key("first10", name))
            with c2:
                st.caption("Middle 10")
                mid_start = max(0, len(prev)//2 - 5)
                st.dataframe(prev.iloc[mid_start:mid_start+10][["Date","Open","High","Low","Close","Volume"]],
                             use_container_width=True, height=220,
                             key=unique_key("mid10", name))
            with c3:
                st.caption("Last 10")
                st.dataframe(prev.tail(10)[["Date","Open","High","Low","Close","Volume"]],
                             use_container_width=True, height=220,
                             key=unique_key("last10", name))
        else:
            st.caption("No existing momentum file found for this stock.")

        st.divider()

        # --- Quick write test ---
        colT1, colT2 = st.columns([1,3])
        with colT1:
            if st.button("🧪 Force write test", key=unique_key("force_write", name), use_container_width=True):
                test_df = pd.DataFrame([{
                    "Date": pd.Timestamp(date.today()),
                    "Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": np.nan
                }])
                try:
                    test_path = _save_ohlc(name, test_df)
                    flash(f"Write OK → {test_path}", "success")
                except Exception as e:
                    flash(f"Write failed: {e}", "error")
                st.rerun()
        with colT2:
            st.caption("If this fails, fix folder permission for the active OHLC dir shown above.")

        # --- Import via Yahoo Finance ---
        st.markdown(section("Import from Yahoo Finance", "Save a default ticker once; Momentum will auto-fetch daily on first visit."), unsafe_allow_html=True)
        today = date.today()
        default_start = today - timedelta(days=365*5)

        saved = _get_saved_ticker(name, prefs)
        prefill = saved or ("7155.KL" if re.search(r"\bskp\b", name, flags=re.I) else name)

        col_y1, col_y2, col_y3, col_y4, col_y5 = st.columns([2, 2, 2, 1, 1])

        with col_y1:
            yf_ticker = st.text_input("Yahoo ticker", value=prefill, key=unique_key("yf_ticker", name))
        with col_y2:
            start_d = st.date_input("Start", value=default_start, key=unique_key("yf_start", name))
        with col_y3:
            end_d = st.date_input("End", value=today, key=unique_key("yf_end", name))
        with col_y4:
            st.write("")
            fetch = st.button("Fetch", key=unique_key("yf_fetch", name), use_container_width=True)
        with col_y5:
            st.write("")
            if st.button("💾 Save default", key=unique_key("yf_save_default", name), use_container_width=True):
                t = str(yf_ticker or "").strip()
                if not t:
                    flash("Please enter a Yahoo ticker (e.g. 7155.KL).", "warning")
                else:
                    used_path = _set_saved_ticker(name, t)
                    prefs.update(_load_prefs())  # refresh mapping immediately
                    if used_path:
                        flash(f"Saved: {name} → {t}\nPrefs synced at: {used_path}", "success")
                    else:
                        flash("Failed to save preferences to any location.", "error")
                st.rerun()

        if fetch:
            if not _YF_OK:
                flash("`yfinance` is not installed in this environment.", "error")
                st.rerun()
            elif not yf_ticker:
                flash("Please enter a Yahoo ticker, e.g. 7113.KL", "warning")
                st.rerun()
            elif start_d > end_d:
                flash("Start date must be before End date.", "warning")
                st.rerun()
            else:
                candidates = _yf_candidates(name, str(yf_ticker))
                got = None
                for sym in candidates:
                    with st.spinner(f"Fetching daily prices for {sym}…"):
                        df_y = _fetch_yahoo_ohlc(sym, start_d, end_d)
                    if df_y is not None and not df_y.empty:
                        got = (sym, df_y); break

                if not got:
                    if _YF_LAST_ERR:
                        flash(f"No rows returned for: {', '.join(candidates)}\nLast Yahoo error: `{_YF_LAST_ERR}`", "warning")
                    else:
                        flash(f"No rows returned for: {', '.join(candidates)}", "warning")
                    st.rerun()
                else:
                    sym, df_y = got
                    if "Volume" not in df_y.columns: df_y["Volume"] = np.nan
                    st.caption(f"Preview — fetched **{len(df_y):,}** rows for `{sym}`")
                    p1, p2, p3 = st.columns(3)
                    with p1:
                        st.dataframe(df_y.head(10), use_container_width=True, height=240, key=unique_key("yf_prev_head", name))
                    with p2:
                        mid = df_y.iloc[max(0, len(df_y)//2 - 5): max(0, len(df_y)//2 - 5) + 10]
                        st.dataframe(mid, use_container_width=True, height=240, key=unique_key("yf_prev_mid", name))
                    with p3:
                        st.dataframe(df_y.tail(10), use_container_width=True, height=240, key=unique_key("yf_prev_tail", name))

                    existing2, _pth2 = _load_ohlc_with_path(name)
                    merged = (pd.concat([existing2, df_y], ignore_index=True)
                              if (existing2 is not None and not existing2.empty) else df_y)
                    merged = merged.drop_duplicates(subset=["Date"], keep="last").sort_values("Date").reset_index(drop=True)

                    try:
                        path = _save_ohlc(name, merged)
                        size = os.path.getsize(path) if os.path.exists(path) else 0
                        flash(f"Saved to **{path}** ({size:,} bytes)", "success")
                    except Exception as e:
                        flash(f"Save failed: {e}", "error")
                    st.rerun()

        st.divider()

        # --- Upload / Replace CSV ---
        st.markdown(section(
            "Upload / Replace CSV",
            "TradingView, generic OHLC, **or Investing.com (Price History)** — we'll auto-detect headers (`Date, Price, Open, High, Low, Vol., Change %`)."
        ), unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload CSV", type=["csv"], key=unique_key("upload_csv", name))
        if uploaded is not None:
            try:
                raw = pd.read_csv(io.BytesIO(uploaded.read()), sep=None, engine="python")
                norm = _normalize_csv_to_ohlc(raw)
                existing3, _pth3 = _load_ohlc_with_path(name)
                merged = (pd.concat([existing3, norm], ignore_index=True)
                          if (existing3 is not None and not existing3.empty) else norm)
                merged = merged.drop_duplicates(subset=["Date"], keep="last").sort_values("Date").reset_index(drop=True)

                path = _save_ohlc(name, merged)
                size = os.path.getsize(path) if os.path.exists(path) else 0
                flash(f"Saved {len(merged):,} rows → {path} ({size:,} bytes)", "success")
            except Exception as e:
                flash(f"Failed to process CSV: {e}", "error")
            st.rerun()

        st.divider()

        # --- Delete this stock’s file ---
        colD1, colD2 = st.columns([1,3])
        with colD1:
            if st.button("Delete existing CSV", key=unique_key("delete_csv", name)):
                p = pth or _ohlc_path_canonical(name)
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                        try: bus.bump("ohlc")
                        except Exception: pass
                        flash(f"Deleted {p}", "success")
                    except Exception as e:
                        flash(f"Failed to delete: {e}", "error")
                else:
                    flash("Nothing to delete.", "info")
                st.rerun()
        with colD2:
            st.caption("Upload again anytime — a new file will replace the old one.")
