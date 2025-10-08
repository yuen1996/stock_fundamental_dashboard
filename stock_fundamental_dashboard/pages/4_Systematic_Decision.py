# 4_Systematic_Decision.py  — CLEAN, LINKED TO rules.py
from __future__ import annotations

from auth_gate import require_auth
require_auth()

import streamlit as st
import pandas as pd
import numpy as np
from utils import bus  # ← ADD THIS

# ✅ PASTE THIS BLOCK HERE -----------------------------
def _safe_key(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name or "")).lower()

def get_bucket(stock: str):
    return st.session_state.get(f"bucket_{_safe_key(stock)}")

def get_syn_idx(stock: str) -> dict:
    return st.session_state.get(f"syn_idx_{_safe_key(stock)}", {}) or {}

def canon(stock: str, label: str) -> str:
    s = (label or "").strip().lower()
    idx = get_syn_idx(stock)
    return idx.get(s) \
        or idx.get(s.replace(" (%)","")) or idx.get(s + " (%)") \
        or idx.get(s.replace(" (x)",""))  or idx.get(s + " (x)") \
        or idx.get(s.replace(" (×)",""))  or idx.get(s + " (×)") \
        or label

def get_summary_ttm(stock: str) -> dict:
    """TTM ratios exactly as seen on View Stock Summary (canonical labels)."""
    return st.session_state.get(f"ttm_dict_{_safe_key(stock)}", {}) or {}

def get_ttm_kpis(stock: str):
    """Returns (period_label, {label->value}) for the card."""
    rec = (st.session_state.get("TTM_KPI_SYNC", {}) or {}).get(stock) or {}
    return rec.get("period", "TTM"), (rec.get("values") or {})

def get_cashflow_kpis(stock: str):
    """Returns (basis, {key->value})"""
    rec = (st.session_state.get("CF_SYNC", {}) or {}).get(stock) or {}
    return rec.get("basis", "TTM"), (rec.get("values") or {})

def get_cagr(stock: str):
    """Returns (end_basis, N, {base_label -> CAGR %})"""
    rec = (st.session_state.get("CAGR_SYNC", {}) or {}).get(stock) or {}
    return rec.get("end_basis", "TTM"), int(rec.get("N", 5)), (rec.get("values_pct") or {})

def get_metric(stock: str, label: str, source: str = "summary"):
    """
    source: 'summary' (default), 'ttm_kpi', or 'cashflow'
    Uses canonical labels where available.
    """
    if source == "summary":
        return get_summary_ttm(stock).get(canon(stock, label))
    if source == "ttm_kpi":
        _, vals = get_ttm_kpis(stock)
        return vals.get(canon(stock, label)) or vals.get(label)
    if source == "cashflow":
        _, vals = get_cashflow_kpis(stock)
        # CF keys are literal like 'FCF', 'FCF Margin (%)', etc.
        return vals.get(label) or vals.get(canon(stock, label))
    return None
# ------------------------------------------------------

# UI helpers
try:
    from utils.ui import (
        setup_decision_page,
        section,
        render_stat_cards,
        render_page_title,
    )
except Exception:
    from ui import setup_decision_page, section, render_stat_cards  # fallback
    try:
        from ui import render_page_title  # type: ignore
    except Exception:
        def render_page_title(page_name: str) -> None:
            st.title(f"📊 Fundamentals Dashboard — {page_name}")

# IO + rules (robust imports + correct pathing)
import os, sys

# Make project root(s) importable when running from /pages
_THIS   = os.path.dirname(__file__)
_PARENT = os.path.abspath(os.path.join(_THIS, ".."))       # project root
_GRANDP = os.path.abspath(os.path.join(_PARENT, ".."))     # one level higher (just in case)
for p in (_PARENT, _GRANDP):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---- Ensure a writable data dir (MUST be BEFORE importing io_helpers) ----
_DEFAULT_DATA_DIR = os.path.join(_PARENT, "data")
_FALLBACK_DATA_DIR = os.path.join(os.path.expanduser("~"), ".sfd_data")

def _dir_writable(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        _test = os.path.join(path, ".sfd_write_test")
        with open(_test, "w") as f:
            f.write("ok")
        os.remove(_test)
        return True
    except Exception:
        return False

# Prefer project /data if writable, else fall back to ~/.sfd_data
if _dir_writable(_DEFAULT_DATA_DIR):
    os.environ["SFD_DATA_DIR"] = _DEFAULT_DATA_DIR
else:
    os.makedirs(_FALLBACK_DATA_DIR, exist_ok=True)
    os.environ["SFD_DATA_DIR"] = _FALLBACK_DATA_DIR

# --------------------------------------------------------------------------

# io_helpers
try:
    from utils import io_helpers as ioh
except Exception:
    import io_helpers as ioh  # fallback

# (Force-reload in case it was imported earlier with bad env)
import importlib as _importlib
ioh = _importlib.reload(ioh)

# ---- UI-side audit fallback (guarantees a record even if an older ioh was loaded) ----
def _queue_row_payload(row_id: int) -> dict:
    """Snapshot the current queue row to include in audit (best-effort)."""
    try:
        q = ioh.load_trade_queue().reset_index().rename(columns={"index": "RowId"})
        r = q.loc[q["RowId"] == int(row_id)].iloc[0]
        return r.to_dict()
    except Exception:
        return {"RowId": int(row_id)}

def _audit_fallback(event: str, payload: dict, reason: str):
    """Call ioh.append_queue_audit if available (non-blocking)."""
    try:
        fn = getattr(ioh, "append_queue_audit", None)
        if callable(fn):
            fn(event, payload, audit_reason=reason)
    except Exception:
        pass

# rules (prefer package import; else load by file path)
try:
    from utils import rules as rules_mod  # type: ignore
except Exception:
    import importlib.util
    CANDIDATES = [
        os.path.join(_PARENT,  "rules.py"),
        os.path.join(_PARENT,  "utils", "rules.py"),
        os.path.join(_GRANDP,  "rules.py"),
        os.path.join(_GRANDP,  "utils", "rules.py"),
    ]
    rules_mod = None
    for path in CANDIDATES:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("rules", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["rules"] = mod
            assert spec.loader is not None
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            rules_mod = mod
            break
    if rules_mod is None:
        raise ImportError(f"Cannot locate rules.py. Tried: {CANDIDATES}")

STRICT_NO_COMPUTE = bool(getattr(rules_mod, "STRICT_NO_COMPUTE", True))

import json, math
_SETTINGS_FILE = os.path.join(_PARENT, "data", "app_settings.json")

def _load_app_settings() -> dict:
    try:
        with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

# --- Optional config defaults (MUST be above _fd_default/_epf_default) ---
try:
    import config
    DEFAULT_FD  = float(getattr(config, "FD_RATE", 0.035))
    DEFAULT_EPF = float(getattr(config, "EPF_RATE", 0.058))
except Exception:
    DEFAULT_FD, DEFAULT_EPF = 0.035, 0.058


def _fd_default() -> float:
    s = _load_app_settings()
    try:
        v = float(s.get("fd_eps_rate", DEFAULT_FD))
        return v if math.isfinite(v) else DEFAULT_FD
    except Exception:
        return DEFAULT_FD

def _epf_default() -> float:
    s = _load_app_settings()
    try:
        v = float(s.get("epf_rate", DEFAULT_EPF))
        return v if math.isfinite(v) else DEFAULT_EPF
    except Exception:
        return DEFAULT_EPF

setup_decision_page("Systematic Decision")
render_page_title("Systematic Decision")

from utils.dictionary import dictionary_flyout
dictionary_flyout(key_prefix="decision", width_px=440)


with lay.main:
    # ---------- Page header ----------
    st.markdown(
        section(
            "🚦 Systematic Decision — Funnel Scanner (Bucket-aware, no gates)",
            "Uses ONLY View-Stock data + bucket anchors in rules.py. Scores are synced to your View page columns (direct ratios preferred).",
            "info"
        ),
        unsafe_allow_html=True,
    )

    # ---------- Load data ----------
    df = ioh.load_data()
    if df is None or df.empty or "Name" not in df.columns:
        st.info("No data yet. Add stocks in **Add / Edit** first.")
        st.stop()

    # Normalize context columns
    if "IndustryBucket" not in df.columns:
        df["IndustryBucket"] = "General"

    # Latest annual snapshot for metadata
    annual_only = df[df.get("IsQuarter") != True].copy() if "IsQuarter" in df.columns else df.copy()
    if annual_only.empty:
        st.info("No annual rows available.")
        st.stop()

    latest_meta = (
        annual_only.sort_values(["Name", "Year"])
        .groupby("Name", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    # ----- tiny local helper (kept UI-only) -----
    def _resolve_price_for_name(name: str) -> float | None:
        rows = df[df["Name"] == name]

        # helper: always return a Series, even if the source is a scalar/None
        def _series(x):
            # If x is already a pandas Series, return as-is; else wrap in a 1D Series
            return x if isinstance(x, pd.Series) else pd.Series([x])

        # Try CurrentPrice first
        s_cur = pd.to_numeric(_series(rows.get("CurrentPrice")), errors="coerce").dropna()
        if not s_cur.empty:
            return float(s_cur.iloc[-1])

        # Fallbacks
        s_sh  = pd.to_numeric(_series(rows.get("SharePrice")), errors="coerce").dropna()
        if not s_sh.empty:
            return float(s_sh.iloc[-1])

        s_any = pd.to_numeric(_series(rows.get("Price")), errors="coerce").dropna()
        if not s_any.empty:
            return float(s_any.iloc[-1])

        return None

    def _block_contrib_table(blocks: dict, bucket: str) -> pd.DataFrame:
        """Block Score + bucket Weight% + Contribution (Score * weight)."""
        w = (getattr(rules_mod, "weights_for", lambda b: {})(bucket) or {})
        pairs = [
            ("cashflow_first", "Cash-flow (5Y)"),
            ("ttm_vs_lfy",     "TTM vs LFY"),
            ("growth_quality", "Growth & Quality (5Y)"),
            ("valuation_entry","Valuation @ Entry"),
            ("dividend",       "Dividend"),
            ("momentum",       "Momentum"),
        ]
        raw_w = [float(w.get(k, w.get(lbl, 0.0)) or 0.0) for k, lbl in pairs]
        tot = sum(raw_w) or 1.0
        norm = [x / tot for x in raw_w]
        rows = []
        for (key, label), wn in zip(pairs, norm):
            b = (blocks.get(key) or {})
            s = float(b.get("score") or 0.0)
            rows.append({
                "Block": label,
                "Score": round(s, 1),
                "Weight %": round(100.0 * wn, 1),
                "Contribution": round(s * wn, 1),
                "Label": b.get("label", ""),
                "Conf %": round(100.0 * (b.get("conf", 1.0) or 0.0), 0),
            })
        return pd.DataFrame(rows)

    # ---------- Detailed breakdown helpers (UI only) ----------
    import re
    from typing import Optional, Dict, Any, Tuple

    def _annual_only(stock_df: pd.DataFrame) -> pd.DataFrame:
        a = stock_df[stock_df.get("IsQuarter") != True].copy() if "IsQuarter" in stock_df.columns else stock_df.copy()
        if "Year" in a.columns:
            a["Year"] = pd.to_numeric(a["Year"], errors="coerce")
            a = a.sort_values("Year")
        return a

    def _pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        # Try rules.py's helper first (keeps behavior identical)
        try:
            return getattr(rules_mod, "_pick_col")(df, candidates)  # type: ignore[attr-defined]
        except Exception:
            pass
        # Minimal fallback
        norm = {re.sub(r"[^a-z0-9]+","", str(c).lower()): c for c in df.columns}
        for c in candidates:
            n = re.sub(r"[^a-z0-9]+","", str(c).lower())
            if n in norm: return norm[n]
        for c in candidates:
            n = re.sub(r"[^a-z0-9]+","", str(c).lower())
            for k, orig in norm.items():
                if n and n in k: return orig
        return None

    def _last_float(df: pd.DataFrame, aliases: list[str]) -> Optional[float]:
        try:
            return getattr(rules_mod, "_last_float")(df, aliases)  # same as rules.py
        except Exception:
            col = _pick_col(df, aliases)
            if not col: return None
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            return float(s.iloc[-1]) if not s.empty else None

    def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
        try:
            return getattr(rules_mod, "_safe_div")(a, b)
        except Exception:
            if a is None or b in (None, 0): return None
            try: return float(a)/float(b)
            except Exception: return None

    def _coerce_float(v):
        """
        Best-effort: turn v into a float or return None.
        Handles strings with %, Series/arrays, dashes, etc.
        """
        if v is None:
            return None
        try:
            import re
            import numpy as np
            import pandas as pd

            # pandas Series → last non-null
            if isinstance(v, pd.Series):
                v = v.dropna()
                if v.empty:
                    return None
                v = v.iloc[-1]

            # list/tuple/ndarray → last element
            if isinstance(v, (list, tuple, np.ndarray)):
                if len(v) == 0:
                    return None
                v = v[-1]

            # strings → strip symbols/commas
            if isinstance(v, str):
                s = v.strip()
                if s in {"", "-", "—", "NaN", "nan", "None"}:
                    return None
                s = s.replace(",", "")
                # remove trailing percent or ×/x symbol
                s = re.sub(r"[%×x]\s*$", "", s, flags=re.I)
                # keep only number-ish characters (very conservative)
                s = re.sub(r"[^0-9eE\.\-\+]", "", s)
                return float(s)

            return float(v)
        except Exception:
            return None


    def _band_score(v, lo: float, hi: float) -> float:
        """
        Map v into [0..100] across [lo..hi] (or reversed if lower is better).
        Tries rules.py’s _band_score if present; otherwise uses a robust fallback.
        """
        # Try rules.py version first (keeps behavior identical if available)
        try:
            fn = getattr(rules_mod, "_band_score", None)
            if callable(fn):
                return float(fn(v, lo, hi))
        except Exception:
            pass

        # Robust fallback
        import numpy as np
        x = _coerce_float(v)
        if x is None or not np.isfinite(x):
            return 0.0
        if lo == hi:
            return 50.0
        if lo < hi:
            t = (x - float(lo)) / (float(hi) - float(lo))        # higher better
        else:
            t = (float(lo) - x) / (float(lo) - float(hi))        # lower better
        return float(np.clip(t, 0.0, 1.0) * 100.0)

    def _fmt_val(v: Any, unit: str | None = None) -> str:
        x = _coerce_float(v)
        if x is None:
            return "—"
        if unit == "%":
            return f"{x:.1f}%"
        if unit == "x":
            return f"{x:.2f}×"
        return f"{x:.4g}"

    def _ttm_summary(stock_df: pd.DataFrame, bucket: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Build the same TTM Summary column used by the View page (so values match rules)."""
        a   = _annual_only(stock_df)
        qtr = stock_df[stock_df.get("IsQuarter") == True].copy() if "IsQuarter" in stock_df.columns else pd.DataFrame()
        price_fallback = _last_float(stock_df, ["CurrentPrice","SharePrice","Price","Annual Price per Share (RM)"])
        try:
            try:
                from utils import calculations as _calc
            except Exception:
                import calculations as _calc
            sum_df = _calc.build_summary_table(
                annual_df=a, quarterly_df=qtr, bucket=bucket, include_ttm=True, price_fallback=price_fallback
            )
            if isinstance(sum_df, pd.DataFrame) and not sum_df.empty and "Metric" in sum_df.columns:
                ttm_col = next((c for c in reversed(sum_df.columns) if isinstance(c, str) and c.upper().startswith("TTM")), None)
                return sum_df, ttm_col
        except Exception:
            pass
        return None, None

    # ---- Momentum helpers: read OHLC saved on Momentum page and compute 12M change
    def _resolve_ohlc_dir() -> str:
        candidates = [
            os.path.abspath(os.path.join(_PARENT, "data", "ohlc")),
            os.path.abspath(os.path.join(os.getcwd(), "data", "ohlc")),
            os.path.abspath(os.path.join(_THIS, ".", "data", "ohlc")),
            os.path.abspath(os.path.join(_GRANDP, "data", "ohlc")),
        ]
        for d in candidates:
            if os.path.isdir(d):
                return d
        return candidates[0]

    _OHLC_DIR = _resolve_ohlc_dir()

    def _ohlc_dir_etag() -> int:
        # Prefer the cross-page bus etag; fall back to directory mtime
        try:
            return int(bus.etag("ohlc"))
        except Exception:
            try:
                return int(os.stat(_OHLC_DIR).st_mtime_ns)
            except Exception:
                return 0

    def _safe_ohlc_name(x: str) -> str:
        import re
        return re.sub(r"[^0-9A-Za-z]+", "_", str(x)).strip("_")

    def _ohlc_path_for(stock_name: str) -> str:
        return os.path.join(_OHLC_DIR, f"{_safe_ohlc_name(stock_name)}.csv")

    @st.cache_data(show_spinner=False)
    def _load_ohlc_for(stock_name: str, _etag: int) -> pd.DataFrame | None:
        p = _ohlc_path_for(stock_name)
        if not os.path.exists(p):
            return None
        df = pd.read_csv(p)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = (
            df.dropna(subset=["Date", "Close"])
            .drop_duplicates(subset=["Date"])
            .sort_values("Date")
            .reset_index(drop=True)
        )
        return df

    def _mom_change_pct_df(df: pd.DataFrame, months: int = 12) -> float | None:
        if df is None or df.empty or "Date" not in df.columns or "Close" not in df.columns:
            return None
        df = df.dropna(subset=["Date","Close"]).sort_values("Date")
        if df.empty:
            return None
        last_dt = df["Date"].iloc[-1]
        days = 30 * months if months < 12 else 365
        start_dt = last_dt - pd.Timedelta(days=days)

        prior = df[df["Date"] <= start_dt]
        if not prior.empty:
            prior_close = float(prior["Close"].iloc[-1])
        else:
            if len(df) < 200:
                return None
            idx = max(0, len(df) - 252)  # ~1Y trading days
            prior_close = float(df["Close"].iloc[idx])

        last_close = float(df["Close"].iloc[-1])
        if prior_close and np.isfinite(prior_close) and prior_close != 0 and np.isfinite(last_close):
            return (last_close / prior_close - 1.0) * 100.0
        return None

    def _ret_12m_pct_from_ohlc(stock_name: str) -> float | None:
        df_px = _load_ohlc_for(stock_name, _ohlc_dir_etag())
        if df_px is None:
            return None
        return _mom_change_pct_df(df_px, months=12)

    def _via_summary(sum_df: Optional[pd.DataFrame], ttm_col: Optional[str], label: str) -> Optional[float]:
        if sum_df is None or not ttm_col: return None
        row = sum_df.loc[sum_df["Metric"] == label, ttm_col]
        if row.empty: return None
        s = pd.to_numeric(row, errors="coerce").dropna()
        return float(s.iloc[0]) if not s.empty else None

    def _value_for_metric(
        stock_df: pd.DataFrame,
        bucket: str,
        key: str,
        fd_rate_decimal: float,
        sum_df: Optional[pd.DataFrame],
        ttm_col: Optional[str],
        *,
        stock_name: Optional[str] = None,
    ) -> Tuple[Optional[float], str, str]:
        a = _annual_only(stock_df)

        if stock_name is None and "Name" in stock_df.columns:
            s_name = stock_df["Name"].dropna()
            if not s_name.empty:
                stock_name = str(s_name.iloc[0])

        def _via(label: str) -> Optional[float]:
            if sum_df is None or not ttm_col:
                return None
            row = sum_df.loc[sum_df["Metric"] == label, ttm_col]
            if row.empty:
                return None
            s = pd.to_numeric(row, errors="coerce").dropna()
            return float(s.iloc[0]) if not s.empty else None

        def _via_lfy(label: str):
            # prefer FY YYYY or plain YYYY, pick the latest (right-most)
            if sum_df is None or "Metric" not in sum_df.columns:
                return None, None
            import re
            cols = [c for c in sum_df.columns if isinstance(c, str)]
            fy_cols = [c for c in cols if re.match(r"^(FY\s*)?\d{4}$", c.strip(), flags=re.I)]
            if not fy_cols:
                return None, None
            col = fy_cols[-1]
            row = sum_df.loc[sum_df["Metric"] == label, col]
            s = pd.to_numeric(row, errors="coerce").dropna()
            return (float(s.iloc[0]) if not s.empty else None), col

        def _yoy_from_annual(names: list[str]) -> Optional[float]:
            col = _pick_col(a, names)
            if not col:
                return None
            s = pd.to_numeric(a[col], errors="coerce").dropna()
            if len(s) >= 2 and float(s.iloc[-2]) != 0:
                return (float(s.iloc[-1]) / float(s.iloc[-2]) - 1.0) * 100.0
            return None

        # --- STRICT: View/TTM summary only (with a few safe rescues) -------------
        if STRICT_NO_COMPUTE:
            if key == "cfo_margin_pct":
                v = _via("Operating CF Margin (%)") or _via("CFO Margin (%)")
                return v, "TTM Summary: CFO Margin (%)" if v is not None else "—", "%"

            if key == "fcf_margin_pct":
                v = _via("FCF Margin (%)")
                return v, "TTM Summary: FCF Margin (%)" if v is not None else "—", "%"

            if key in ("cash_conv_cfo_ebitda", "cash_conv_cfo_ebitda_pct"):
                # accept (×) or (%) and normalize to ×
                v_x = _via("CFO/EBITDA (×)") or _via("CFO/EBITDA (x)")
                if v_x is not None:
                    return v_x, "TTM Summary: CFO/EBITDA (×)", "x"
                v_pct = _via("CFO/EBITDA (%)") or _via("Cash Conversion (CFO/EBITDA)")
                if v_pct is not None:
                    return float(v_pct) / 100.0, "TTM Summary: CFO/EBITDA (%)", "x"
                return None, "—", "x"

            if key == "ebitda_margin_pct":
                v = _via("EBITDA Margin (%)")
                return v, "TTM Summary: EBITDA Margin (%)" if v is not None else "—", "%"

            if key in ("icr_x", "icr"):
                v = _via("Interest Coverage (x)") or _via("Interest Coverage (×)")
                return v, "TTM Summary: Interest Coverage (x)" if v is not None else "—", "x"

            if key == "avg_cost_debt_pct":
                v = _via("Average Cost of Debt (%)")
                return v, "TTM Summary: Average Cost of Debt (%)" if v is not None else "—", "%"

            if key == "capex_to_rev_pct":
                v = _via("Capex/Revenue (%)") or _via("Capex to Revenue (%)")
                return v, "TTM Summary: Capex/Revenue (%)" if v is not None else "—", "%"

            if key == "eps_yoy_pct":
                v = _via("EPS YoY (%)")
                if v is None:
                    v = _yoy_from_annual(["EPS (RM)", "EPS", "Earnings per Share"])
                    if v is not None:
                        return v, "Annual: YoY from EPS", "%"
                return v, "TTM Summary: EPS YoY (%)" if v is not None else "—", "%"

            if key == "revenue_yoy_pct":
                v = _via("Revenue YoY (%)") or _via("Revenue YoY")
                if v is None:
                    v = _yoy_from_annual(["Revenue"])
                    if v is not None:
                        return v, "Annual: YoY from Revenue", "%"
                return v, "TTM Summary: Revenue YoY (%)" if v is not None else "—", "%"

            if key == "payout_pct":
                # Prefer last FY if present; else TTM; do not compute in strict
                v_lfy, fy_col = _via_lfy("Payout Ratio (%)")
                if v_lfy is not None:
                    return v_lfy, f"Annual Summary: Payout Ratio (%) [{fy_col}]", "%"
                v = _via("Payout Ratio (%)")
                if v is not None:
                    return v, "TTM Summary: Payout Ratio (%)", "%"
                return None, "—", "%"

            if key in ("mom_12m_pct", "price_change_12m_pct"):
                v = _via("12M Price Change (%)") or _via("Price Change 12M (%)") or _via("Total Return 1Y (%)")
                if v is not None:
                    return v, "TTM Summary: 12M Price Change (%)", "%"
                if stock_name:
                    v = _ret_12m_pct_from_ohlc(stock_name)
                    return v, "Momentum OHLC (12M price change)" if v is not None else "—", "%"
                return None, "—", "%"

            # Banking/REITs direct rows via Summary
            direct = {
                "nim_pct":       "NIM (%)",
                "cir_pct":       "Cost-to-Income Ratio (%)",
                "npl_ratio_pct": "NPL Ratio (%)",
                "wale_years":    "WALE (years)",
                "occupancy_pct": "Occupancy (%)",
            }
            if key in direct:
                v = _via(direct[key])
                return v, f"TTM Summary: {direct[key]}" if v is not None else "—", "%" if key.endswith("_pct") else ""

            if key == "dy_vs_fd_x":
                # Show raw Dividend Yield (%) in the details table
                v = _via("Dividend Yield (%)") or _via("Distribution Yield (%)")
                if v is None:
                    v_lfy, fy_col = _via_lfy("Dividend Yield (%)")
                    if v_lfy is not None:
                        return v_lfy, f"Annual Summary: Dividend Yield (%) [{fy_col}]", "%"
                return v, "TTM Summary: Dividend Yield (%)" if v is not None else "—", "%"

            return None, "—", ""

        # --- Non-strict: keep your fallbacks; add CFO/EBITDA (%) tolerance --------
        if key == "cfo_margin_pct":
            v = _via("Operating CF Margin (%)") or _via("CFO Margin (%)")
            if v is not None: return v, "TTM Summary: CFO Margin (%)", "%"
            num = _last_float(a, ["CFO","Operating Cash Flow","Cash Flow from Ops"]); den = _last_float(a, ["Revenue"])
            vv = _safe_div(num, den); return (vv*100.0 if vv is not None else None), "Annual: CFO/Revenue", "%"

        if key == "fcf_margin_pct":
            v = _via("FCF Margin (%)")
            if v is not None: return v, "TTM Summary: FCF Margin (%)", "%"
            cfo = _last_float(a, ["CFO","Operating Cash Flow","Cash Flow from Ops"])
            cap = _last_float(a, ["Capex","CAPEX"])
            rev = _last_float(a, ["Revenue"])
            vv = _safe_div((cfo - cap) if (cfo is not None and cap is not None) else None, rev)
            return (vv*100.0 if vv is not None else None), "Annual: (CFO - Capex)/Revenue", "%"

        if key in ("cash_conv_cfo_ebitda","cash_conv_cfo_ebitda_pct"):
            v = _via("CFO/EBITDA (×)") or _via("CFO/EBITDA (x)")
            if v is None:
                v_pct = _via("CFO/EBITDA (%)") or _via("Cash Conversion (CFO/EBITDA)")
                if v_pct is not None:
                    return float(v_pct)/100.0, "TTM Summary: CFO/EBITDA (%)", "x"
            if v is not None: return v, "TTM Summary: CFO/EBITDA (×)", "x"
            cfo = _last_float(a, ["CFO","Operating Cash Flow","Cash Flow from Ops"])
            ebitda = _last_float(a, ["EBITDA"])
            vv = _safe_div(cfo, ebitda); return vv, "Annual: CFO/EBITDA", "x"

        if key == "ebitda_margin_pct":
            v = _via("EBITDA Margin (%)")
            if v is not None: return v, "TTM Summary: EBITDA Margin (%)", "%"
            e = _last_float(a, ["EBITDA"]); r = _last_float(a, ["Revenue"])
            vv = _safe_div(e, r); return (vv*100.0 if vv is not None else None), "Annual: EBITDA/Revenue", "%"

        if key in ("icr_x","icr"):
            v = _via("Interest Coverage (x)") or _via("Interest Coverage (×)")
            if v is not None: return v, "TTM Summary: Interest Coverage (x)", "x"
            ebit = _last_float(a, ["Operating Profit","EBIT","Operating profit"])
            ie   = _last_float(a, ["Interest Expense","Finance Costs"])
            vv = _safe_div(ebit, abs(ie) if ie is not None else None); return vv, "Annual: EBIT / Interest Expense", "x"

        if key == "avg_cost_debt_pct":
            v = _via("Average Cost of Debt (%)")
            if v is not None: return v, "TTM Summary: Average Cost of Debt (%)", "%"
            ie = _last_float(a, ["Interest Expense","Finance Costs"]); tb = _last_float(a, ["Total Borrowings"])
            vv = _safe_div(ie, tb); return (vv*100.0 if vv is not None else None), "Annual: Interest Expense / Total Borrowings", "%"

        if key == "capex_to_rev_pct":
            v = _via("Capex/Revenue (%)") or _via("Capex to Revenue (%)")
            if v is not None: return v, "TTM Summary: Capex/Revenue (%)", "%"
            cap = _last_float(a, ["Capex","CAPEX"]); r = _last_float(a, ["Revenue"])
            vv = _safe_div(cap, r); return (vv*100.0 if vv is not None else None), "Annual: Capex / Revenue", "%"

        if key == "eps_yoy_pct":
            v = _via("EPS YoY (%)")
            if v is None:
                col = _pick_col(a, ["EPS (RM)", "EPS", "Earnings per Share"])
                if col:
                    s = pd.to_numeric(a[col], errors="coerce").dropna()
                    if len(s) >= 2 and float(s.iloc[-2]) != 0:
                        v = (float(s.iloc[-1]) / float(s.iloc[-2]) - 1.0) * 100.0
                        return v, f"Annual: YoY from {col}", "%"
            return v, ("TTM Summary: EPS YoY (%)" if v is not None else "—"), "%"

        if key == "revenue_yoy_pct":
            v = _via("Revenue YoY (%)") or _via("Revenue YoY")
            if v is None:
                col = _pick_col(a, ["Revenue"])
                if col:
                    s = pd.to_numeric(a[col], errors="coerce").dropna()
                    if len(s) >= 2 and float(s.iloc[-2]) != 0:
                        v = (float(s.iloc[-1]) / float(s.iloc[-2]) - 1.0) * 100.0
                        return v, f"Annual: YoY from {col}", "%"
            return v, ("TTM Summary: Revenue YoY (%)" if v is not None else "—"), "%"

        if key == "dy_vs_fd_x":
            v = _via("Dividend Yield (%)") or _via("Distribution Yield (%)")
            if v is None:
                v, fy_col = _via_lfy("Dividend Yield (%)")
                if v is None:
                    v, fy_col = _via_lfy("Distribution Yield (%)")
                if v is not None:
                    return v, f"Annual Summary: Dividend/Distribution Yield (%) [{fy_col}]", "%"
            return v, ("TTM Summary: Dividend/Distribution Yield (%)" if v is not None else "—"), "%"

        if key == "payout_pct":
            v = _via("Payout Ratio (%)")
            if v is not None: return v, "TTM Summary: Payout Ratio (%)", "%"
            eps = _last_float(a, ["EPS (RM)","EPS","Earnings per Share"])
            dps = _last_float(a, ["DPS","Dividend per Share (TTM, RM)","DPU"])
            vv = _safe_div(dps, eps); return (vv*100.0 if vv is not None else None), "Annual: DPS / EPS", "%"

        if key in ("mom_12m_pct", "price_change_12m_pct"):
            v = _via("12M Price Change (%)") or _via("Price Change 12M (%)") or _via("Total Return 1Y (%)")
            src = "TTM Summary: 12M Price Change (%)" if v is not None else ""
            if v is None and stock_name:
                v = _ret_12m_pct_from_ohlc(stock_name)
                src = "Momentum OHLC (12M price change)" if v is not None else "—"
            return v, (src or "—"), "%"

        direct_map = {
            "nim_pct":       ["NIM (%)","NIM","Net Interest/Financing Margin (%)"],
            "cir_pct":       ["Cost-to-Income Ratio (%)","Cost-to-Income","CIR","C/I (%)"],
            "npl_ratio_pct": ["NPL Ratio (%)","NPL (%)"],
            "wale_years":    ["WALE (years)","WALE"],
            "occupancy_pct": ["Occupancy (%)","Hotel Occupancy (%)"],
        }
        if key in direct_map:
            col = _pick_col(a, direct_map[key])
            if col:
                return _last_float(a, [col]), f"Annual: {col}", "%" if key.endswith("_pct") else ""
            v = _via(direct_map[key][0])
            return v, f"TTM Summary: {direct_map[key][0]}" if v is not None else "—", "%" if key.endswith("_pct") else ""

        return None, "—", ""

    def _metric_breakdown(stock_df: pd.DataFrame, bucket: str, fd_rate_decimal: float) -> Dict[str, pd.DataFrame]:
        """
        Returns {block_name -> DataFrame} with metric, raw value, band, score, source.
        """
        sum_df, ttm_col = _ttm_summary(stock_df, bucket)
        blocks = getattr(rules_mod, "anchors_for", lambda b: {})(bucket) or {}
        out: Dict[str, pd.DataFrame] = {}

        # For the dividend panel we’ll also surface FD/EPF if available.
        fd_pct  = getattr(rules_mod, "_fd_rate_pct",  lambda: None)()  # type: ignore[attr-defined]
        epf_pct = getattr(rules_mod, "_epf_rate_pct", lambda: None)()  # type: ignore[attr-defined]
        score_dy = getattr(rules_mod, "_score_dy_snowflake_style", None)

        for block, spec in blocks.items():
            name_hint = None
            if "Name" in stock_df.columns:
                s_name = stock_df["Name"].dropna()
                if not s_name.empty:
                    name_hint = str(s_name.iloc[0])
            rows = []
            for key, (lo, hi) in spec.items():
                # ---- Robust unpack: accept 3-tuple, 2-tuple, or scalar ----
                ret = _value_for_metric(stock_df, bucket, key, fd_rate_decimal, sum_df, ttm_col, stock_name=name_hint)
                if isinstance(ret, tuple):
                    if len(ret) == 3:
                        v, src, unit = ret
                    elif len(ret) == 2:
                        v, src = ret
                        unit = ""
                    elif len(ret) == 1:
                        v = ret[0]
                        src, unit = "—", ""
                    else:
                        v, src, unit = None, "—", ""
                else:
                    v, src, unit = ret, "—", ""

                # ---- Score + band text ----
                if key == "dy_vs_fd_x" and score_dy is not None:
                    # score_dy expects Dividend Yield in PERCENT
                    try:
                        if v is None:
                            dy_pct = None
                        elif unit == "%":
                            dy_pct = float(v)
                        else:
                            # If older function returned DY/FD multiple, convert to %
                            dy_pct = float(v) * float(fd_rate_decimal) * 100.0
                    except Exception:
                        dy_pct = None

                    sc = float(score_dy(dy_pct))
                    band_txt = f"FD { _fmt_val(fd_pct, '%') } → EPF { _fmt_val(epf_pct, '%') } → Cap 35%"
                    # Show raw as a percent for clarity if unit missing
                    if unit == "" and dy_pct is not None:
                        unit = "%"
                        v = dy_pct
                else:
                    sc = _band_score(v, float(lo), float(hi))
                    arrow = "↑ better" if float(lo) < float(hi) else "↓ better"
                    band_txt = f"{_fmt_val(lo, unit)} → {_fmt_val(hi, unit)}  ({arrow})"

                rows.append({
                    "Metric": key,
                    "Raw": _fmt_val(v, unit),
                    "Band": band_txt,
                    "Score": round(sc, 1),
                    "Source": src or "—",
                })
            out[block] = pd.DataFrame(rows)
        return out

    def _valuation_breakdown(stock_df: pd.DataFrame, bucket: str, current_price: Optional[float]) -> Tuple[pd.DataFrame, str]:
        prof = getattr(rules_mod, "BUCKET_PROFILES", {}).get(bucket) or getattr(rules_mod, "BUCKET_PROFILES", {}).get("General") or {}
        entries = prof.get("entry", []) or []
        bands   = prof.get("value", {}) or {}

        # Use rules.py's internal metric aggregator so values match the page
        vals = {}
        try:
            vals = getattr(rules_mod, "_valuation_metrics")(stock_df, current_price, bucket=bucket)  # type: ignore[attr-defined]
        except Exception:
            vals = {}

        rows = []
        for k in entries:
            b = bands.get(k, {})
            lo, hi = b.get("lo"), b.get("hi")
            if lo is None or hi is None: continue
            lower_better = bool(b.get("lower_better", False))
            lo_eff, hi_eff = (hi, lo) if lower_better else (lo, hi)
            v  = vals.get(k)
            sc = _band_score(v, float(lo_eff), float(hi_eff))
            dir_txt = "lower better" if lower_better else "higher better"
            unit = "%" if k in ("fcf_yield","dy") else "x" if k in ("pe","pb","p_nav","p_rnav","ev_ebitda","ev_sales") else ""
            rows.append({
                "Entry metric": k,
                "Raw": _fmt_val(v, unit),
                "Band": f"{_fmt_val(lo_eff, unit)} → {_fmt_val(hi_eff, unit)}  ({dir_txt})",
                "Score": round(sc, 1),
            })

        if rows:
            avg = float(np.mean([r["Score"] for r in rows]))
            label = "Cheap" if avg>=70 else ("Fair" if avg>=40 else "Rich")
        else:
            avg = 0.0; label = "—"
        return pd.DataFrame(rows), label

    # ---------- Sidebar ----------
    st.sidebar.subheader("Filters")
    all_buckets = list(dict.fromkeys(latest_meta["IndustryBucket"].astype(str).tolist()))
    pick_bucket = st.sidebar.multiselect("Industry Bucket", all_buckets, default=all_buckets)

    # Pull FD from global settings (could be stored as % or decimal); normalize to decimal.
    _fd_from_settings = float(_fd_default())
    fd_rate = (_fd_from_settings / 100.0) if (_fd_from_settings is not None and _fd_from_settings > 1) else _fd_from_settings
    # NEW — normalize EPF like FD
    _epf_from_settings = float(_epf_default())
    epf_rate = (_epf_from_settings / 100.0) if (_epf_from_settings is not None and _epf_from_settings > 1) else _epf_from_settings


    # ---------- Evaluate via rules.py (no gates) ----------
    rows = []
    eval_json_by_name: dict[str, dict] = {}

    # pick the scorer from rules.py (compatible with older copies)
    _scoring = (
        getattr(rules_mod, "funnel_score_from_view", None)
        or getattr(rules_mod, "compute_funnel_from_view_stock", None)
        or getattr(rules_mod, "funnel_rule", None)  # final fallback
    )

    if _scoring is None:
        st.error("No funnel scorer found in rules.py. Please ensure `funnel_score_from_view(...)` exists.")
        st.stop()

    for _, meta in latest_meta.iterrows():
        name   = str(meta["Name"])
        bucket = str(meta.get("IndustryBucket", "General") or "General")
        if pick_bucket and bucket not in pick_bucket:
            continue

        cur_price  = _resolve_price_for_name(name)
        stock_rows = df[df["Name"] == name].copy()
        if cur_price is not None:
            stock_rows["CurrentPrice"] = float(cur_price)

        # Try with the current param name first, then gracefully fall back
        try:
            res = _scoring(
                name=name,
                stock_df=stock_rows,
                bucket=bucket,
                current_price=cur_price,
                fd_rate_decimal=float(fd_rate),      # ← matches rules.py today
            ) or {}
        except TypeError:
            try:
                # Older scorer variants might expect a different kw name
                res = _scoring(
                    name=name,
                    stock_df=stock_rows,
                    bucket=bucket,
                    current_price=cur_price,
                    fd_rate_sidebar_decimal=float(fd_rate),
                ) or {}
            except TypeError:
                # Last resort: positional
                res = _scoring(name, stock_rows, bucket, cur_price, float(fd_rate)) or {}

        eval_json_by_name[name] = res

        blocks = (res.get("blocks") or {})
        fscore = float(res.get("composite") or 0.0)

        # thresholds from rules.py
        _min_score, _ = getattr(rules_mod, "min_thresholds_for", lambda b: (65, 50))(bucket)
        decision = "PASS" if fscore >= int(_min_score) else "REJECT"
        unmet    = "" if decision == "PASS" else f"FunnelScore < {_min_score}"

        rows.append({
            "Name":         name,
            "Industry":     meta.get("Industry", ""),
            "Bucket":       bucket,
            "Year":         int(meta.get("Year", np.nan)) if pd.notna(meta.get("Year")) else None,
            "Price":        cur_price,

            "FunnelScore":  round(fscore, 1),
            "MinScore":     int(_min_score),
            "Decision":     decision,
            "Unmet":        unmet,

            "Cash":         round((blocks.get("cashflow_first", {}) or {}).get("score", np.nan), 1),
            "TTM":          round((blocks.get("ttm_vs_lfy", {}) or {}).get("score", np.nan), 1),
            "Growth":       round((blocks.get("growth_quality", {}) or {}).get("score", np.nan), 1),
            "Valuation":    (blocks.get("valuation_entry", {}) or {}).get("label", ""),
            "Dividend":     round((blocks.get("dividend", {}) or {}).get("score", np.nan), 1),
            "Momentum":     round((blocks.get("momentum", {}) or {}).get("score", np.nan), 1),
        })

    dec = pd.DataFrame(rows)
    if dec.empty:
        st.info("No stocks matched your filters.")
        st.stop()

    # ---------- Summary (KPI cards) ----------
    render_stat_cards(
        [
            {"label": "Universe size", "value": f"{len(latest_meta):,}", "badge": "Universe"},
            {"label": "Scanned",       "value": f"{len(dec):,}",          "badge": "Run"},
            {
                "label": "PASS",
                "value": str(int((dec["Decision"] == "PASS").sum())),
                "badge": "Count",
                "tone":  "good" if (dec["Decision"] == "PASS").any() else "warn",
            },
        ],
        columns=3,
    )

    # ---------- Tables ----------
    def _show_table(label: str, df_: pd.DataFrame):
        # pretty section badge
        tone = "success" if "PASS" in label else "danger"
        st.markdown(section(label, "", tone), unsafe_allow_html=True)

        if df_.empty:
            st.caption("— none —")
            return
        cfg = {
            "Price":       st.column_config.NumberColumn("Price", format="%.4f"),
            "FunnelScore": st.column_config.NumberColumn("FunnelScore", format="%.0f"),
            "MinScore":    st.column_config.NumberColumn("MinScore", format="%.0f"),
        }
        cols = [c for c in ["Name","Industry","Bucket","Year","Price","FunnelScore","MinScore","Decision","Unmet","Valuation","Cash","TTM","Growth","Dividend","Momentum"] if c in df_.columns]
        st.data_editor(
            df_[cols].sort_values(["Decision","FunnelScore","Name"], ascending=[True, False, True]),
            use_container_width=True,
            hide_index=True,
            height=min(540, 72 + 28*min(len(df_), 12)),
            column_config=cfg,
            disabled=True,
            key=f"tbl_{label}"
        )

    _show_table("✅ PASS (FunnelScore ≥ MinScore)", dec[dec["Decision"] == "PASS"])
    _show_table("❌ REJECT (below MinScore)",       dec[dec["Decision"] == "REJECT"])

    # ---------- Drilldown ----------
    with st.expander("📊 Funnel details — scores, bands & data sources", expanded=False):
        pick = st.selectbox("Pick a stock", dec["Name"].tolist())
        row = dec[dec["Name"] == pick].iloc[0]
        bucket = row["Bucket"]
        res = eval_json_by_name.get(pick) or {}
        stock_rows = df[df["Name"] == pick].copy()

        blocks  = (res.get("blocks") or {})
        fscore  = float(res.get("composite") or 0.0)
        ms, mv  = getattr(rules_mod, "min_thresholds_for", lambda b: (65, 50))(bucket)
        weights = getattr(rules_mod, "weights_for", lambda b: {})(bucket)

        # Top summary cards for this single pick
        render_stat_cards(
            [
                {"label": "Decision",    "value": row["Decision"], "badge": "Status", "tone": "good" if row["Decision"]=="PASS" else "bad"},
                {"label": "FunnelScore", "value": f"{fscore:.0f}",  "badge": "Composite"},
                {"label": "MinScore",    "value": f"{ms}",          "badge": bucket},
            ],
            columns=3,
        )

        # Block contributions (same numbers used to form the composite)
        st.markdown("**Block contributions (weight × score × availability)**")
        st.dataframe(_block_contrib_table(blocks, bucket), use_container_width=True, height=220)

        # Valuation entry breakdown (per-metric values + bands)
        st.markdown("**Valuation @ Entry — per-metric bands & scores**")
        cur_price  = _resolve_price_for_name(pick)
        vdf, vlabel = _valuation_breakdown(stock_rows, bucket, cur_price)
        c1, c2 = st.columns([3,1])
        with c1:
            st.dataframe(vdf, use_container_width=True, height=220)
        with c2:
            tone = {"Cheap": "good", "Fair": "warn", "Rich": "bad"}.get(vlabel, "neutral")
            render_stat_cards(
                [{"label": "Valuation", "value": vlabel, "badge": "Entry", "tone": tone}],
                columns=1,
            )

        # Metric-level breakdown for each block in anchors
        st.markdown("**Block metric details — raw values, bands, score & source**")
        details = _metric_breakdown(stock_rows, bucket, fd_rate)
        for ui_name, pretty in [
            ("cashflow_first", "Cash-flow (5Y)"),
            ("ttm_vs_lfy",     "TTM vs LFY"),
            ("growth_quality", "Growth & Quality (5Y)"),
            ("dividend",       "Dividend"),
            ("momentum",       "Momentum"),
        ]:
            if ui_name in details:
                st.markdown(section(pretty, "", "info"), unsafe_allow_html=True)
                st.dataframe(details[ui_name], use_container_width=True, height=220)

        st.caption("Notes: ‘Band’ shows the score band used (arrow indicates direction). ‘Source’ tells you where the raw value came from (Annual columns, derived formula, or the View page’s TTM Summary). Dividend uses your saved FD & EPF to map dividend yield to a 0–100 score (Snowflake-style).")

        # --- JSON export (single pick; everything in this expander) ---
        export_payload = {
            "stock":        pick,
            "industry":     row.get("Industry", ""),
            "bucket":       bucket,
            "year":         (int(row["Year"]) if pd.notna(row.get("Year")) else None),
            "price":        cur_price,
            "decision":     row.get("Decision", ""),
            "composite":    round(float(fscore), 1),
            "min_thresholds": {
                "min_score":     int(ms),
                "min_valuation": int(mv),
            },
            "weights":      (weights or {}),
            # blocks already contains each block’s score/label/conf from rules.py
            "blocks":       (blocks or {}),
            # exact numbers used to form composite (weight × score × availability)
            "block_contributions": _block_contrib_table(blocks, bucket).to_dict(orient="records"),
            # valuation panel (per-metric bands & scores) + Cheap/Fair/Rich label
            "valuation": {
                "label": vlabel,
                "rows":  vdf.to_dict(orient="records"),
            },
            # metric-level tables for every block (raw value, band, score, source)
            "details": {
                k: df.to_dict(orient="records")
                for k, df in (details or {}).items()
                if isinstance(df, pd.DataFrame)
            },
        }

        st.download_button(
            "⬇️ Download funnel details (JSON)",
            data=json.dumps(export_payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"{_safe_key(pick)}_funnel_details.json",
            mime="application/json",
            use_container_width=True,
        )

    # === PASS actions — push selected rows to Trade Queue (two buttons) ===
    st.markdown(
        section("✅ PASS — push to Trade Queue",
                "Tick the rows you want to queue, then choose a push action.",
                "success"),
        unsafe_allow_html=True,
    )

    pass_df = dec[dec["Decision"] == "PASS"].copy()
    if pass_df.empty:
        st.info("No PASS candidates right now.")
    else:
        # Build a compact, user-friendly editor
        price_col = "Price" if "Price" in pass_df.columns else None
        cols_keep = ["Name", "Industry", "Bucket", "Year"]
        if price_col:
            cols_keep += [price_col]
        cols_keep += ["FunnelScore", "MinScore", "Valuation", "Dividend", "Momentum"]
        cols_keep = [c for c in cols_keep if c in pass_df.columns]

        # Compute DY% and LongTrade eligibility (DY > EPF)
        dy_pct_map: dict[str, float | None] = {}
        lt_eligible: dict[str, bool] = {}

        for _, r in pass_df.iterrows():
            name   = str(r["Name"])
            bucket = str(r.get("Bucket", "General") or "General")
            stock_rows = df[df["Name"] == name].copy()

            sum_df, ttm_col = _ttm_summary(stock_rows, bucket)
            v, _src, unit = _value_for_metric(
                stock_rows, bucket, key="dy_vs_fd_x",
                fd_rate_decimal=fd_rate, sum_df=sum_df, ttm_col=ttm_col,
                stock_name=name,
            )

            dy_pct = None
            try:
                if unit == "%":
                    dy_pct = float(v) if v is not None else None
                elif unit == "" and v is not None:
                    # older rules might return DY/FD multiple → convert to %
                    dy_pct = float(v) * float(fd_rate) * 100.0
            except Exception:
                dy_pct = None

            dy_pct_map[name] = dy_pct
            lt_eligible[name] = (dy_pct is not None) and (epf_rate is not None) and (dy_pct > float(epf_rate) * 100.0)

        # Build table once
        tbl = pass_df[cols_keep].copy()
        tbl.insert(0, "SelectPush", False)  # checkbox users can tick
        tbl["DY (%)"] = tbl["Name"].map(lambda n: dy_pct_map.get(n))
        tbl["LongTrade? (DY>EPF)"] = tbl["Name"].map(lambda n: bool(lt_eligible.get(n, False)))

        edited = st.data_editor(
            tbl,
            use_container_width=True,
            hide_index=True,
            height=min(360, 72 + 28*min(len(tbl), 10)),
            column_config={
                "SelectPush":  st.column_config.CheckboxColumn("Push"),
                "FunnelScore": st.column_config.NumberColumn("FunnelScore", format="%.0f", disabled=True),
                "MinScore":    st.column_config.NumberColumn("MinScore",    format="%.0f", disabled=True),
                "Valuation":   st.column_config.TextColumn("Valuation @ Entry", disabled=True),
                "Dividend":    st.column_config.TextColumn("Dividend", disabled=True),
                "Momentum":    st.column_config.NumberColumn("Momentum", format="%.0f", disabled=True),
                **({price_col: st.column_config.NumberColumn(price_col, format="%.4f", disabled=True)} if price_col else {}),
                "DY (%)":      st.column_config.NumberColumn("DY (%)", format="%.1f", disabled=True),
                "LongTrade? (DY>EPF)": st.column_config.CheckboxColumn("LongTrade? (DY>EPF)", disabled=True),
            },
            key="pass_push_editor",
        )

        def _push_selected(rows_df: pd.DataFrame, strategy: str):
            from utils import io_helpers as _ioh

            to_push = []
            for _, r in rows_df.iterrows():
                if r.get("SelectPush", False) is not True:
                    continue

                name = str(r["Name"])
                score_val = r.get("FunnelScore")
                score = float(score_val) if pd.notna(score_val) else 0.0

                price_val = r.get(price_col) if price_col else None
                cpx = float(price_val) if (price_col and pd.notna(price_val)) else None

                to_push.append({
                    "name": name,
                    "strategy": strategy,
                    "score": score,
                    "current_price": cpx,
                    "entry_price": cpx,
                    "reasons": f"Auto-pushed from PASS ({strategy})",
                })

            if not to_push:
                st.info("No rows were ticked. Tick **Push** first, then click a button.")
                return

            if hasattr(_ioh, "push_trade_candidates"):
                pushed = _ioh.push_trade_candidates(to_push)
                failed = len(to_push) - pushed
                failed_names = []
            else:
                pushed, failed = 0, 0
                failed_names = []
                for r in to_push:
                    ok = _ioh.push_trade_candidate(
                        name=r["name"], strategy=r["strategy"], score=r["score"],
                        current_price=r["current_price"], entry_price=r["entry_price"],
                        reasons=r["reasons"],
                    )
                    if ok: pushed += 1
                    else:
                        failed += 1
                        failed_names.append(r["name"])

            if pushed > 0:
                st.success(f"Pushed {pushed} stock(s) to Trade Queue as “{strategy}”.")
                (getattr(st, "rerun", None) or getattr(st, "experimental_rerun", lambda: None))()
            elif failed:
                st.error("Failed to push: " + ", ".join(failed_names))
            else:
                st.info("Nothing to do.")

        eligible_set = {n for n, ok in lt_eligible.items() if ok}
        has_any_eligible = len(eligible_set) > 0

        c1, c2 = st.columns(2)
        with c1:
            if st.button("📥 Push selected (PASS)", type="primary", use_container_width=True):
                _push_selected(edited, strategy="IndustryFunnel")

        with c2:
            if st.button("📥 Push selected (Long trade — DY > EPF)", use_container_width=True, disabled=not has_any_eligible):
                sel = edited[(edited["SelectPush"] == True) & (edited["Name"].isin(eligible_set))]
                if sel.empty:
                    st.info("No selected rows are eligible for Long trade (DY ≤ EPF).")
                else:
                    _push_selected(sel, strategy="LongTrade")

    # === Manage Queue (row-exact) — same behavior as old page ===
    st.markdown(
        section("🔧 Manage Queue", "Mark Live / Delete — acts on exact RowId", "warning"),
        unsafe_allow_html=True,
    )

    tq = ioh.load_trade_queue().copy()
    if tq.empty:
        st.info("Queue is empty.")
    else:
        # Give each current row its visible RowId (index right now)
        tq = tq.reset_index().rename(columns={"index": "RowId"})

        # numeric coercion for plan fields (NOW includes TP1/TP2/TP3)
        for c in ["Entry", "Stop", "Take", "Shares", "RR", "TP1", "TP2", "TP3", "CurrentPrice", "Score"]:
            tq[c] = pd.to_numeric(tq.get(c), errors="coerce")

        # A plan is OK when these are present (long trade: stop below entry)
        def _plan_ok_row(r):
            e, s, t, sh, rr = r.get("Entry"), r.get("Stop"), r.get("Take"), r.get("Shares"), r.get("RR")
            return (
                np.isfinite(e) and e > 0 and
                np.isfinite(s) and s > 0 and e > s and
                np.isfinite(t) and t > 0 and
                np.isfinite(rr) and
                (pd.notna(sh) and (int(sh) if not pd.isna(sh) else 0) > 0)
            )

        tq["PlanOK"] = tq.apply(_plan_ok_row, axis=1)

        # Show TP1/TP2/TP3 in the table
        view_cols = [
            "RowId","Name","Strategy","CurrentPrice","Score",
            "Entry","Stop","Take","Shares","RR","TP1","TP2","TP3","Timestamp","Reasons","PlanOK"
        ]
        table = tq[[c for c in view_cols if c in tq.columns]].copy()
        table.insert(0, "Select", False)

        edited_q = st.data_editor(
            table,
            use_container_width=True,
            height=360,
            hide_index=True,
            column_config={
                "Select":       st.column_config.CheckboxColumn("Sel"),
                "RowId":        st.column_config.NumberColumn("RowId", disabled=True),
                "CurrentPrice": st.column_config.NumberColumn("Price", format="%.4f", disabled=True),
                "Score":        st.column_config.NumberColumn("Score", format="%.0f", disabled=True),
                "Entry":        st.column_config.NumberColumn("Entry", format="%.4f", disabled=True),
                "Stop":         st.column_config.NumberColumn("Stop",  format="%.4f", disabled=True),
                "Take":         st.column_config.NumberColumn("Take",  format="%.4f", disabled=True),
                "Shares":       st.column_config.NumberColumn("Shares", format="%d",   disabled=True),
                "RR":           st.column_config.NumberColumn("RR",    format="%.2f", disabled=True),
                "TP1":          st.column_config.NumberColumn("TP1",   format="%.4f", disabled=True),
                "TP2":          st.column_config.NumberColumn("TP2",   format="%.4f", disabled=True),
                "TP3":          st.column_config.NumberColumn("TP3",   format="%.4f", disabled=True),
                "Timestamp":    st.column_config.TextColumn("Added", disabled=True),
                "Reasons":      st.column_config.TextColumn("Notes/Reasons", disabled=True),
                "PlanOK":       st.column_config.CheckboxColumn("Plan Ready?", disabled=True),
            },
            key="queue_manage_editor",
        )

        c1, c2, _ = st.columns([1.6, 1.6, 3])

        # Mark Live (only if plan is complete)
        with c1:
            if st.button("✅ Mark Live selected"):
                moved, blocked = 0, 0
                blocked_ids = []
                for _, r in edited_q.iterrows():
                    if not r.Select:
                        continue
                    if not bool(r.get("PlanOK", False)):
                        blocked += 1
                        blocked_ids.append(int(r.RowId))
                        continue
                    # just call io_helpers; it already writes MARK_LIVE to the audit file
                    if ioh.mark_live_row(int(r.RowId)):
                        moved += 1

                if moved > 0:
                    st.success(f"Marked live: {moved} row(s).")
                    (getattr(st, "rerun", None) or getattr(st, "experimental_rerun", lambda: None))()
                elif blocked > 0:
                    st.warning(
                        f"{blocked} row(s) blocked — plan incomplete (need Entry, Stop, Take, Shares, RR): "
                        + ", ".join(map(str, blocked_ids))
                        + ". Open **Risk / Reward Planner** to finish the plan."
                    )
                else:
                    st.info("Nothing selected.")

        with c2:
            if st.button("🗑️ Delete selected"):
                sel_ids = [int(r.RowId) for _, r in edited_q.iterrows() if r.Select]
                if not sel_ids:
                    st.info("Nothing selected.")
                else:
                    # just call io_helpers; it already writes DELETE rows to the audit file
                    if hasattr(ioh, "delete_trade_rows"):
                        deleted = ioh.delete_trade_rows(sel_ids)
                    else:
                        deleted = 0
                        for i in sorted(set(sel_ids), reverse=True):
                            if ioh.delete_trade_row(int(i)):
                                deleted += 1

                    if deleted > 0:
                        st.success(f"Deleted {deleted} row(s).")
                        (getattr(st, "rerun", None) or getattr(st, "experimental_rerun", lambda: None))()
                    else:
                        st.info("Nothing deleted.")
