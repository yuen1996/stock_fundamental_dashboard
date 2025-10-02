# stock_fundamental_dashboard/utils/rules.py
"""
Minimal rules module so pages can import it.
Add real validation/derivation rules here when needed.
"""

from __future__ import annotations  

import re
import numpy as np
import pandas as pd

from typing import Dict, Any, Iterable, Set
import pandas as pd

# === STRICT MODE ==============================================================
# When True, funnel reads ONLY what your View page already shows (TTM Summary /
# direct ratio columns) and does not compute any fallbacks.
STRICT_NO_COMPUTE = True
# =============================================================================

def sanitize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replace empty strings with None; keep other values as-is.
    Safe to call before saving.
    """
    out = {}
    for k, v in row.items():
        if isinstance(v, str) and v.strip() == "":
            out[k] = None
        else:
            out[k] = v
    return out

def allowed_keys_for_bucket(bucket: str, all_fields: Iterable[str]) -> Set[str]:
    """
    Placeholder gate: currently allows everything.
    Later, restrict based on bucket-specific policies.
    """
    return set(all_fields)

def pre_save_hook(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hook called before saving the full DataFrame.
    Currently a no-op; return df unchanged.
    """
    return df

# --- Snowflake (pass-through TTM ratios + five-pillar specs) -----------------
from typing import Tuple, Set, Iterable, Optional, Dict, Any
import numpy as np
import pandas as pd

def _qnum(q):
    import re
    if pd.isna(q): return np.nan
    m = re.search(r"(\d+)", str(q).upper())
    return int(m.group(1)) if m else np.nan

# heuristics to detect existing ratio/percent columns (no math)
_RATIO_EXCLUDE = {"name","year","quarter","isquarter","industry","industrybucket"}
def _is_ratio_col_name(col: str) -> bool:
    if not isinstance(col, str): return False
    s = col.strip().lower()
    if s in _RATIO_EXCLUDE: return False
    hints = ("(%)"," margin","yield"," ratio","p/e","peg","ev/ebitda","roe",
             "roa","coverage","turnover"," (x)"," (×)")
    if any(h in s for h in hints): return True
    exacts = {"p/b","pnav","p/nav","icr","current ratio","quick ratio"}
    return s in exacts

def _collect_ratio_cols(df) -> list[str]:
    return [c for c in df.columns if isinstance(c, str) and _is_ratio_col_name(c)]

def _allowed_snowflake_labels(bucket: str) -> Set[str]:
    """
    Use your configured Summary categories to limit which ratios appear.
    Falls back to 'General' if bucket-specific is missing.
    """
    try:
        try:
            from utils import config as _cfg
        except Exception:
            import config as _cfg  # last-resort import path
        cats = (_cfg.INDUSTRY_SUMMARY_RATIOS_CATEGORIES or {})
        m = (cats.get(bucket) or cats.get("General") or {})
        out = set()
        for items in m.values():
            if isinstance(items, dict):
                out.update(str(k) for k in items.keys())
        return out
    except Exception:
        return set()

def snowflake_from_annual(annual_df, *, bucket: str) -> Tuple[pd.Series, str]:
    """
    Return (Series of 'ratio label' -> value, ttm_label) using ONLY the values
    you already stored in ANNUAL rows for the TTM year (last FY + 1).
    No calculations are performed.
    """
    if annual_df is None or annual_df.empty or "Year" not in annual_df.columns:
        return pd.Series(dtype="float64"), "TTM"

    ys = pd.to_numeric(annual_df["Year"], errors="coerce").dropna()
    if ys.empty:
        return pd.Series(dtype="float64"), "TTM"
    ttm_year = int(ys.max()) + 1
    ttm_tag  = f"TTM {ttm_year}"

    ttm_rows = annual_df[pd.to_numeric(annual_df["Year"], errors="coerce") == ttm_year]
    if ttm_rows.empty:
        return pd.Series(dtype="float64"), ttm_tag

    ratio_cols = _collect_ratio_cols(annual_df)
    if not ratio_cols:
        return pd.Series(dtype="float64"), ttm_tag

    allowed = _allowed_snowflake_labels(bucket)
    if allowed:
        ratio_cols = [c for c in ratio_cols if c in allowed]
        if not ratio_cols:
            return pd.Series(dtype="float64"), ttm_tag

    vals = {}
    for rc in ratio_cols:
        s = pd.to_numeric(ttm_rows[rc], errors="coerce").dropna()
        if not s.empty:
            vals[rc] = float(s.iloc[-1])

    return pd.Series(vals, dtype="float64"), ttm_tag


# Optional hint for section ordering in UIs that read rules:
SECTION_AFTER = {
    # put Snowflake block directly after the Cash Flow/Wealth block
    "snowflake": "cashflow_wealth"
}

# =========================
# Snowflake per-industry spec (complete; covers all buckets)
# =========================
# Each pillar is a list of metric-specs. A metric-spec = {
#   "name": "<label exactly as in your Summary table>",
#   "low": float, "high": float,      # map low..high → 0..100 (clamped)
#   "invert": bool,                   # if True, lower is better (e.g. P/E)
#   "src": "auto|cf|cagr|momentum",   # where to pull value from
#   "weight": float                   # optional weight (defaults 1.0)
# }

def _S(name, low, high, invert=False, src="auto", weight=1.0):
    return {"name": name, "low": low, "high": high, "invert": invert, "src": src, "weight": weight}

# ---- shared bases (use (x) labels to match your Summary rows)
_BASE_FV_NONBANK = [
    _S("Dividend Yield (%)", 0.0, 8.0),
    _S("P/E (x)",            10.0, 25.0, invert=True),
    _S("P/B (x)",            0.7,  3.0,  invert=True),
]
_BASE_EQ_STANDARD = [
    _S("ROE (%)",            5.0, 20.0),
    _S("Gross Margin (%)",  10.0, 50.0),
    _S("Net Margin (%)",     0.0, 20.0),
]
_BASE_GC_STANDARD = [
    _S("Revenue CAGR (%)",    -10.0, 15.0, src="cagr"),
    _S("Net Profit CAGR (%)", -10.0, 15.0, src="cagr"),
]
_BASE_CS_FCF = [
    _S("FCF Margin (%)",  -5.0, 10.0, src="cf"),
    _S("FCF Yield (%)",    0.0, 10.0, src="cf"),
]
_BASE_MOM = [_S("12M Price Change (%)", -20.0, 40.0, src="momentum")]

SNOWFLAKE_SPECS = {
    # ---------- Generic (fallback)
    "General": {
        "Future Value":       list(_BASE_FV_NONBANK),
        "Earnings Quality":   list(_BASE_EQ_STANDARD),
        "Growth Consistency": list(_BASE_GC_STANDARD),
        "Cash Strength":      list(_BASE_CS_FCF),
        "Momentum":           list(_BASE_MOM),
    },

    # ---------- Non-financials
    "Manufacturing": {
        "Future Value":       list(_BASE_FV_NONBANK),
        "Earnings Quality":   list(_BASE_EQ_STANDARD),
        "Growth Consistency": list(_BASE_GC_STANDARD),
        "Cash Strength":      list(_BASE_CS_FCF) + [_S("Net Debt / EBITDA (×)", 2.5, 0.0, invert=True)],
        "Momentum":           list(_BASE_MOM),
    },
    "Retail": {
        "Future Value":       list(_BASE_FV_NONBANK),
        "Earnings Quality":   list(_BASE_EQ_STANDARD),
        "Growth Consistency": list(_BASE_GC_STANDARD),
        "Cash Strength":      list(_BASE_CS_FCF) + [_S("Capex/Revenue (%)", 20.0, 10.0, invert=True)],
        "Momentum":           list(_BASE_MOM),
    },
    "Utilities": {
        "Future Value":       [_S("Dividend Yield (%)", 0.0, 6.0),
                               _S("P/E (x)", 8.0, 18.0, invert=True),
                               _S("P/B (x)", 0.7, 2.0, invert=True)],
        "Earnings Quality":   [_S("ROE (%)", 6.0, 14.0),
                               _S("EBITDA Margin (%)", 30.0, 55.0),
                               _S("Operating Profit Margin (%)", 10.0, 25.0)],
        "Growth Consistency": [_S("RAB CAGR (%)", 0.0, 6.0, src="cagr")],
        "Cash Strength":      [_S("FCF Margin (%)", -5.0, 10.0, src="cf"),
                               _S("Capex/Revenue (%)", 30.0, 15.0, invert=True),
                               _S("CFO/EBITDA (×)", 0.6, 1.0)],
        "Momentum":           list(_BASE_MOM),
    },
    "Energy/Materials": {
        "Future Value":       list(_BASE_FV_NONBANK),
        "Earnings Quality":   [_S("ROE (%)", 5.0, 18.0),
                               _S("EBITDA Margin (%)", 15.0, 40.0),
                               _S("Net Margin (%)", 0.0, 20.0)],
        "Growth Consistency": [_S("EBITDA CAGR (%)", -10.0, 15.0, src="cagr"),
                               _S("Annual Production CAGR (%)", -5.0, 10.0, src="cagr")],
        "Cash Strength":      [_S("FCF Margin (%)", -5.0, 12.0, src="cf"),
                               _S("CFO/EBITDA (×)", 0.6, 1.0),
                               _S("Net Debt / EBITDA (×)", 3.0, 0.0, invert=True)],
        "Momentum":           list(_BASE_MOM),
    },
    "Tech": {
        "Future Value":       [_S("P/E (x)", 15.0, 40.0, invert=True, weight=0.6),
                               _S("EV/Sales (×)", 2.0, 10.0, invert=True, weight=0.6),
                               _S("Dividend Yield (%)", 0.0, 2.0, weight=0.2)],
        "Earnings Quality":   [_S("Gross Margin (%)", 40.0, 70.0),
                               _S("Net Margin (%)", 0.0, 15.0),
                               _S("Rule of 40 (%)", 20.0, 40.0)],
        "Growth Consistency": [_S("Revenue CAGR (%)", 10.0, 30.0, src="cagr"),
                               _S("ARR CAGR (%)", 10.0, 30.0, src="cagr")],
        "Cash Strength":      [_S("FCF Margin (%)", -5.0, 15.0, src="cf"),
                               _S("CFO/EBITDA (×)", 0.6, 1.0)],
        "Momentum":           list(_BASE_MOM),
    },
    "Healthcare": {
        "Future Value":       list(_BASE_FV_NONBANK),
        "Earnings Quality":   [_S("ROE (%)", 6.0, 18.0),
                               _S("Gross Margin (%)", 25.0, 55.0),
                               _S("Net Margin (%)", 5.0, 18.0)],
        "Growth Consistency": list(_BASE_GC_STANDARD),
        "Cash Strength":      list(_BASE_CS_FCF),
        "Momentum":           list(_BASE_MOM),
    },
    "Telco": {
        "Future Value":       [_S("Dividend Yield (%)", 0.0, 6.0),
                               _S("EV/EBITDA (x)", 6.0, 10.0, invert=True),
                               _S("P/E (x)", 10.0, 25.0, invert=True)],
        "Earnings Quality":   [_S("EBITDA Margin (%)", 30.0, 50.0),
                               _S("Operating Profit Margin (%)", 10.0, 25.0),
                               _S("Net Margin (%)", 5.0, 18.0)],
        "Growth Consistency": [_S("Subscribers CAGR (%)", 0.0, 6.0, src="cagr"),
                               _S("ARPU CAGR (%)", -2.0, 4.0, src="cagr")],
        "Cash Strength":      [_S("FCF Margin (%)", -5.0, 12.0, src="cf"),
                               _S("Capex/Revenue (%)", 25.0, 12.0, invert=True)],
        "Momentum":           list(_BASE_MOM),
    },
    "Construction": {
        "Future Value":       list(_BASE_FV_NONBANK),
        "Earnings Quality":   list(_BASE_EQ_STANDARD),
        "Growth Consistency": [_S("Orderbook CAGR (%)", 0.0, 15.0, src="cagr"),
                               _S("Net Profit CAGR (%)", -10.0, 15.0, src="cagr")],
        "Cash Strength":      [_S("Net Debt / EBITDA (×)", 3.0, 0.0, invert=True),
                               _S("Interest Coverage (x)", 1.5, 5.0)],
        "Momentum":           list(_BASE_MOM),
    },
    "Plantation": {
        "Future Value":       list(_BASE_FV_NONBANK),
        "Earnings Quality":   [_S("ROE (%)", 5.0, 15.0),
                               _S("EBITDA Margin (%)", 10.0, 35.0),
                               _S("Net Margin (%)", 0.0, 20.0)],
        "Growth Consistency": [_S("Tonnes Sold CAGR (%)", -5.0, 10.0, src="cagr"),
                               _S("EBITDA CAGR (%)", -10.0, 15.0, src="cagr")],
        "Cash Strength":      [_S("FCF Margin (%)", -5.0, 12.0, src="cf"),
                               _S("Net Debt / EBITDA (×)", 2.5, 0.0, invert=True)],
        "Momentum":           list(_BASE_MOM),
    },
    "Property": {
        "Future Value":       [_S("Distribution/Dividend Yield (%)", 0.0, 8.0),
                               _S("P/NAV (x)", 0.6, 1.1, invert=True),
                               _S("P/E (x)", 8.0, 20.0, invert=True)],
        "Earnings Quality":   [_S("Gross Margin (%)", 15.0, 35.0),
                               _S("Net Margin (%)", 5.0, 18.0)],
        "Growth Consistency": [_S("Unbilled Sales CAGR (%)", 0.0, 12.0, src="cagr"),
                               _S("Net Profit CAGR (%)", -10.0, 15.0, src="cagr")],
        "Cash Strength":      [_S("Net Gearing (Net Debt/Equity, %)", 60.0, 20.0, invert=True),
                               _S("Interest Coverage (x)", 1.5, 5.0)],
        "Momentum":           list(_BASE_MOM),
    },
    "Transportation/Logistics": {
        "Future Value":       [_S("Dividend Yield (%)", 0.0, 6.0),
                               _S("P/E (x)", 8.0, 20.0, invert=True),
                               _S("EV/EBITDA (x)", 5.0, 10.0, invert=True)],
        "Earnings Quality":   [_S("EBITDA Margin (%)", 10.0, 30.0),
                               _S("Net Margin (%)", 3.0, 12.0)],
        "Growth Consistency": [_S("EBITDA CAGR (%)", -10.0, 15.0, src="cagr"),
                               _S("TEU Throughput CAGR (%)", 0.0, 10.0, src="cagr")],
        "Cash Strength":      [_S("Net Debt / EBITDA (×)", 3.0, 0.0, invert=True),
                               _S("Interest Coverage (x)", 1.5, 5.0)],
        "Momentum":           list(_BASE_MOM),
    },
    "Leisure/Travel": {
        "Future Value":       [_S("Dividend Yield (%)", 0.0, 6.0),
                               _S("P/E (x)", 10.0, 25.0, invert=True),
                               _S("EV/EBITDA (x)", 6.0, 12.0, invert=True)],
        "Earnings Quality":   [_S("EBITDA Margin (%)", 15.0, 35.0),
                               _S("Net Margin (%)", 5.0, 18.0)],
        "Growth Consistency": [_S("Revenue CAGR (%)", -10.0, 15.0, src="cagr"),
                               _S("Visits CAGR (%)", 0.0, 12.0, src="cagr")],
        "Cash Strength":      [_S("FCF Margin (%)", -5.0, 12.0, src="cf"),
                               _S("Interest Coverage (x)", 1.5, 5.0)],
        "Momentum":           list(_BASE_MOM),
    },

    # ---------- Financials (non-bank financials)
    "Financials": {
        "Future Value":       list(_BASE_FV_NONBANK),
        "Earnings Quality":   [_S("ROE (%)", 6.0, 18.0),
                               _S("Net Margin (%)", 5.0, 20.0),
                               _S("Operating Expense Ratio (%)", 60.0, 40.0, invert=True)],
        "Growth Consistency": [_S("Net Profit CAGR (%)", -10.0, 15.0, src="cagr")],
        "Cash Strength":      [_S("FCF Margin (%)", -5.0, 12.0, src="cf"),
                               _S("Interest Coverage (x)", 1.5, 5.0)],
        "Momentum":           list(_BASE_MOM),
    },

    # ---------- Banking (now has its own spec)
    "Banking": {
        "Future Value": [
            _S("Dividend Yield (%)", 0.0, 8.0),
            _S("P/E (x)",             9.0, 16.0, invert=True),
            _S("P/B (x)",             0.7, 1.6,  invert=True),
        ],
        "Earnings Quality": [
            _S("ROE (%)",                    5.0, 14.0),
            _S("NIM (%)",                    1.5, 3.0),
            _S("Cost-to-Income Ratio (%)",  60.0, 40.0, invert=True),
        ],
        "Growth Consistency": [
            _S("Net Profit CAGR (%)",  -5.0, 10.0, src="cagr"),
            _S("Gross Loans CAGR (%)",  0.0, 8.0,  src="cagr"),
        ],
        "Cash Strength": [
            _S("Loan-Loss Coverage (×)", 0.8, 1.5),
            _S("NPL Ratio (%)",          5.0, 1.5, invert=True),
            _S("CASA Ratio (%)",        20.0, 45.0),
        ],
        "Momentum": list(_BASE_MOM),
    },

    # ---------- REITs (note the plural key)
    "REITs": {
        "Future Value":       [_S("Distribution Yield (%)", 0.0, 8.0),
                               _S("P/NAV (x)", 0.6, 1.1, invert=True)],
        "Earnings Quality":   [_S("Occupancy (%)", 85.0, 98.0),
                               _S("WALE (years)", 2.0, 6.0)],
        "Growth Consistency": [_S("NPI CAGR (%)", 0.0, 6.0, src="cagr")],
        "Cash Strength":      [_S("Gearing (x)", 0.6, 0.3, invert=True),
                               _S("Interest Coverage (x)", 2.0, 5.0),
                               _S("Average Cost of Debt (%)", 6.0, 3.0, invert=True)],
        "Momentum":           list(_BASE_MOM),
    },
}

# --- Snowflake (pass-through TTM ratios; robust) --------------------
from typing import Tuple, Set
import numpy as np

_RATIO_EXCLUDE = {"name","year","quarter","isquarter","industry","industrybucket"}

def _is_ratio_col_name(col: str) -> bool:
    if not isinstance(col, str): return False
    s = col.strip().lower()
    if s in _RATIO_EXCLUDE: return False
    hints = ("(%)"," margin","yield"," ratio","p/e","peg","ev/ebitda",
             "roe","roa","coverage","turnover"," (x)"," (×)")
    if any(h in s for h in hints): return True
    exacts = {"p/b","pnav","p/nav","icr","current ratio","quick ratio"}
    return s in exacts

def _collect_ratio_cols(df) -> list[str]:
    return [c for c in df.columns if isinstance(c, str) and _is_ratio_col_name(c)]

def _allowed_labels_and_aliases(bucket: str) -> Set[str]:
    """
    Allow canonical labels *and* their aliases from INDUSTRY_SUMMARY_RATIOS_CATEGORIES.
    """
    try:
        try:
            from utils import config as _cfg
        except Exception:
            import config as _cfg
        cats = (_cfg.INDUSTRY_SUMMARY_RATIOS_CATEGORIES or {})
        m = (cats.get(bucket) or cats.get("General") or {})
        out: Set[str] = set()
        for items in m.values():
            if isinstance(items, dict):
                for canonical, aliases in items.items():
                    out.add(str(canonical))
                    if isinstance(aliases, (list, tuple, set)):
                        out.update(str(a) for a in aliases)
        return out
    except Exception:
        return set()

def snowflake_from_annual(annual_df, *, bucket: str) -> Tuple[pd.Series, str]:
    """
    Return (Series of 'ratio label' -> value, ttm_label) using ONLY values already
    stored in ANNUAL rows for the TTM year. No calculations are performed.
    Accepts Year as:
      - next FY number (last FY + 1), or
      - 'TTM', or 'TTM <next FY>'
    """
    import pandas as pd, re
    if annual_df is None or annual_df.empty or "Year" not in annual_df.columns:
        return pd.Series(dtype="float64"), "TTM"

    year_str = annual_df["Year"].astype(str).str.strip()
    year_num = pd.to_numeric(year_str, errors="coerce")
    last_fy = int(year_num.dropna().max()) if not year_num.dropna().empty else None
    ttm_year = (last_fy + 1) if last_fy is not None else None
    ttm_tag  = f"TTM {ttm_year}" if ttm_year is not None else "TTM"

    # Accept numeric next FY OR 'TTM' OR 'TTM YYYY'
    mask_ttm = (year_num == ttm_year)
    mask_ttm = mask_ttm | year_str.str.upper().eq("TTM")
    if ttm_year is not None:
        mask_ttm = mask_ttm | year_str.str.upper().eq(f"TTM {ttm_year}")

    ttm_rows = annual_df[mask_ttm]
    if ttm_rows.empty:
        return pd.Series(dtype="float64"), ttm_tag

    ratio_cols = _collect_ratio_cols(annual_df)
    if not ratio_cols:
        return pd.Series(dtype="float64"), ttm_tag

    allowed = _allowed_labels_and_aliases(bucket)
    if allowed:
        ratio_cols = [c for c in ratio_cols if str(c) in allowed]
        if not ratio_cols:
            return pd.Series(dtype="float64"), ttm_tag

    def _coerce(v):
        if pd.isna(v): return None
        if isinstance(v, (int, float, np.floating)): return float(v)
        s = str(v).strip().replace(",", "")
        s = re.sub(r"[%×x]", "", s, flags=re.I)
        try:
            return float(s)
        except Exception:
            return None

    vals = {}
    for rc in ratio_cols:
        col = ttm_rows[rc].dropna() if rc in ttm_rows.columns else pd.Series(dtype="float64")
        if not col.empty:
            # take last non-null, after coercion
            for x in reversed(col.tolist()):
                v = _coerce(x)
                if v is not None:
                    vals[str(rc)] = v
                    break

    return pd.Series(vals, dtype="float64"), ttm_tag

# Keep this so View page can place the block just after Cash Flow/Wealth
SECTION_AFTER = {
    "snowflake": "cashflow_wealth"
}


# Funnel rule (bucket-aware scoring + soft gate penalties)
from typing import Mapping, Optional

# Prefer config values if present; else use safe defaults identical to the page.
try:
    try:
        from utils import config as _cfg  # package-style
    except Exception:
        import config as _cfg             # repo-root fallback
except Exception:
    _cfg = type("Cfg", (), {})()

# --- Global FD/EPF reader (from Add/Edit settings) + Dividend scorer ----------
import os, json

def _settings_file_candidates():
    base = os.path.dirname(__file__)
    parents = [
        base,
        os.path.abspath(os.path.join(base, "..")),
        os.path.abspath(os.path.join(base, "..", "..")),
    ]
    return [os.path.join(p, "data", "app_settings.json") for p in parents]

def _load_global_settings() -> dict:
    for p in _settings_file_candidates():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return {}

def _fd_rate_pct() -> float | None:
    s = _load_global_settings()
    v = s.get("fd_eps_rate")
    if v is None:
        # fallback to config default (decimal -> %)
        try:
            return float(getattr(_cfg, "FD_RATE", 0.035)) * 100.0
        except Exception:
            return 3.5
    try:
        v = float(v)
        return v if np.isfinite(v) else None
    except Exception:
        return None

def _epf_rate_pct() -> float | None:
    s = _load_global_settings()
    v = s.get("epf_rate")
    if v is None:
        # fallback to config default (decimal -> %)
        try:
            return float(getattr(_cfg, "EPF_RATE", 0.058)) * 100.0
        except Exception:
            return 5.8
    try:
        v = float(v)
        return v if np.isfinite(v) else None
    except Exception:
        return None

def _score_dy_snowflake_style(dy_pct: float | None, cap: float = 35.0) -> float:
    """
    0–100 score using FD & EPF (inputs as PERCENT, e.g., 3.5 for 3.5%):
      0→FD:     0 → 40
      FD→EPF:  40 → 70
      EPF→cap: 70 → 98
      >= cap:  98
    """
    fd = _fd_rate_pct()
    epf = _epf_rate_pct()

    try:
        dy  = 0.0 if dy_pct is None else float(dy_pct)
        fd  = float(fd) if fd is not None else 0.0
        epf = float(epf) if epf is not None else fd
    except Exception:
        return 0.0

    if not np.isfinite(dy) or dy <= 0.0:
        return 0.0
    if epf <= fd:
        epf = fd

    def lerp(x, x0, y0, x1, y1):
        if x1 <= x0:
            return float(y1)
        t = (x - x0) / (x1 - x0)
        t = float(np.clip(t, 0.0, 1.0))
        return float(y0 + (y1 - y0) * t)

    if dy < fd:
        return lerp(dy, 0.0, 0.0, fd, 40.0)
    elif dy < epf:
        return lerp(dy, fd, 40.0, epf, 70.0)
    elif dy < cap:
        return lerp(dy, epf, 70.0, cap, 98.0)
    else:
        return 98.0


_DEFAULT_GATE_POLICY = {"mode": "soft", "penalty_per_fail": 0.08, "max_penalty": 0.25}

_DEFAULT_MIN_SCORE_BY_BUCKET = {
    "General": 65, "Manufacturing": 65, "Retail": 62, "Financials": 63,
    "Banking": 62, "REITs": 60, "Utilities": 64, "Energy/Materials": 62,
    "Tech": 66, "Healthcare": 64, "Telco": 62, "Construction": 60,
    "Plantation": 62, "Property": 60, "Transportation/Logistics": 62,
    "Leisure/Travel": 61,
}
_DEFAULT_MIN_VALUATION_BY_BUCKET = {k: 50 for k in _DEFAULT_MIN_SCORE_BY_BUCKET.keys()}

_DEFAULT_BLOCK_WEIGHTS = {
    "cashflow_first": 0.25,
    "ttm_vs_lfy": 0.15,
    "growth_quality": 0.25,
    "valuation_entry": 0.20,
    "dividend": 0.10,
    "momentum": 0.05,
}
_DEFAULT_BLOCK_WEIGHTS_BY_BUCKET = {
    "General": _DEFAULT_BLOCK_WEIGHTS,
    "Manufacturing": {"cashflow_first": 0.28, "ttm_vs_lfy": 0.15, "growth_quality": 0.27, "valuation_entry": 0.18, "dividend": 0.07, "momentum": 0.05},
    "Retail": {"cashflow_first": 0.25, "ttm_vs_lfy": 0.22, "growth_quality": 0.18, "valuation_entry": 0.20, "dividend": 0.10, "momentum": 0.05},
    "Financials": {"cashflow_first": 0.22, "ttm_vs_lfy": 0.18, "growth_quality": 0.25, "valuation_entry": 0.20, "dividend": 0.10, "momentum": 0.05},
    "Banking": {"cashflow_first": 0.26, "ttm_vs_lfy": 0.18, "growth_quality": 0.22, "valuation_entry": 0.22, "dividend": 0.07, "momentum": 0.05},
    "REITs": {"cashflow_first": 0.25, "ttm_vs_lfy": 0.15, "growth_quality": 0.20, "valuation_entry": 0.25, "dividend": 0.10, "momentum": 0.05},
    "Utilities": {"cashflow_first": 0.26, "ttm_vs_lfy": 0.16, "growth_quality": 0.20, "valuation_entry": 0.23, "dividend": 0.10, "momentum": 0.05},
    "Energy/Materials": {"cashflow_first": 0.28, "ttm_vs_lfy": 0.16, "growth_quality": 0.20, "valuation_entry": 0.22, "dividend": 0.09, "momentum": 0.05},
    "Tech": {"cashflow_first": 0.18, "ttm_vs_lfy": 0.22, "growth_quality": 0.32, "valuation_entry": 0.18, "dividend": 0.05, "momentum": 0.05},
    "Healthcare": {"cashflow_first": 0.24, "ttm_vs_lfy": 0.18, "growth_quality": 0.24, "valuation_entry": 0.20, "dividend": 0.09, "momentum": 0.05},
    "Telco": {"cashflow_first": 0.26, "ttm_vs_lfy": 0.18, "growth_quality": 0.16, "valuation_entry": 0.25, "dividend": 0.10, "momentum": 0.05},
    "Construction": {"cashflow_first": 0.22, "ttm_vs_lfy": 0.22, "growth_quality": 0.22, "valuation_entry": 0.19, "dividend": 0.10, "momentum": 0.05},
    "Plantation": {"cashflow_first": 0.27, "ttm_vs_lfy": 0.18, "growth_quality": 0.20, "valuation_entry": 0.20, "dividend": 0.10, "momentum": 0.05},
    "Property": {"cashflow_first": 0.25, "ttm_vs_lfy": 0.20, "growth_quality": 0.15, "valuation_entry": 0.25, "dividend": 0.10, "momentum": 0.05},
    "Transportation/Logistics": {"cashflow_first": 0.26, "ttm_vs_lfy": 0.18, "growth_quality": 0.18, "valuation_entry": 0.23, "dividend": 0.10, "momentum": 0.05},
    "Leisure/Travel": {"cashflow_first": 0.25, "ttm_vs_lfy": 0.20, "growth_quality": 0.18, "valuation_entry": 0.22, "dividend": 0.10, "momentum": 0.05},
}

# === Minimal anchors & valuation profiles (used by the funnel) ===============

# Anchors tell the funnel which metrics to read (from your View/TTM summary)
# and how to map them to 0–100 scores per block.
ANCHORS_BY_BUCKET = {
    "General": {
        "cashflow_first": {
            "cfo_margin_pct": (0.0, 20.0),
            "fcf_margin_pct": (-5.0, 10.0),
            "cash_conv_cfo_ebitda": (0.6, 1.0),
        },
        "ttm_vs_lfy": {
            "eps_yoy_pct": (-10.0, 20.0),
            "revenue_yoy_pct": (-10.0, 15.0),
        },
        "growth_quality": {
            "ebitda_margin_pct": (10.0, 40.0),
            "icr_x": (1.5, 5.0),
        },
        "dividend": {
            "dy_vs_fd_x": (0.5, 1.5),
            "payout_pct": (0.0, 80.0),
        },
        "momentum": {
            "mom_12m_pct": (-20.0, 40.0),       # 12M price change % band, like Snowflake
        },
    },
        "Banking": {
        "cashflow_first": {"cfo_margin_pct": (0.0, 20.0)},
        "ttm_vs_lfy": {"eps_yoy_pct": (-10.0, 20.0)},
        "growth_quality": {
            "nim_pct": (1.5, 3.5),
            "cir_pct": (60.0, 40.0),  # lower is better (lo>hi handled)
        },
        "dividend": {
            "dy_vs_fd_x": (0.6, 1.6),
            "payout_pct": (0.0, 70.0),
        },
        # ↓↓↓ replace this line ↓↓↓
        "momentum": {"mom_12m_pct": (-20.0, 40.0)},
    },

}

# Valuation profiles define what “entry” metrics to combine for the
# Valuation block and their “cheap↔rich” bands.
BUCKET_PROFILES = {
    "General": {
        "entry": ["pe", "pb", "fcf_yield"],
        "value": {
            "pe":        {"lo": 10.0, "hi": 25.0, "lower_better": True},
            "pb":        {"lo": 0.5,  "hi": 3.0,  "lower_better": True},
            "fcf_yield": {"lo": 0.0,  "hi": 8.0,  "lower_better": False},
        },
    },
    "Banking": {
        "entry": ["pe", "pb", "dy"],
        "value": {
            "pe": {"lo": 9.0, "hi": 16.0, "lower_better": True},
            "pb": {"lo": 0.7, "hi": 1.6,  "lower_better": True},
            "dy": {"lo": 2.0, "hi": 8.0,  "lower_better": False},
        },
    },
}
# ============================================================================ 


def _cfg_or(default, attr):
    v = getattr(_cfg, attr, None)
    return v if isinstance(v, dict) and v else default

def weights_for(bucket: str) -> dict:
    wb = _cfg_or(_DEFAULT_BLOCK_WEIGHTS_BY_BUCKET, "BLOCK_WEIGHTS_BY_BUCKET")
    base = _cfg_or(_DEFAULT_BLOCK_WEIGHTS, "BLOCK_WEIGHTS")
    return (wb.get(bucket) or wb.get("General") or base or {}).copy()

def min_thresholds_for(bucket: str) -> tuple[int, int]:
    msb = _cfg_or(_DEFAULT_MIN_SCORE_BY_BUCKET, "MIN_SCORE_BY_BUCKET")
    mvb = _cfg_or(_DEFAULT_MIN_VALUATION_BY_BUCKET, "MIN_VALUATION_BY_BUCKET")
    return int(msb.get(bucket, msb.get("General", 65))), int(mvb.get(bucket, mvb.get("General", 50)))

def _safe_float(x) -> float | None:
    try:
        import math
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def _composite_from_blocks(blocks: Mapping[str, Mapping[str, float]], w: Mapping[str, float]) -> tuple[float, dict]:
    """Weighted average of block scores using confidence as availability weight."""
    num = 0.0; den = 0.0; details = {}
    for key, meta in (blocks or {}).items():
        score = _safe_float(meta.get("score"))
        conf  = _safe_float(meta.get("conf")) or 0.0
        wt    = float(w.get(key, 0.0))
        contrib = (score or 0.0) * conf * wt
        num += contrib
        den += conf * wt
        details[key] = {"score": score or 0.0, "conf": conf, "weight": wt, "contribution": contrib}
    return (float(num / den) if den > 0 else 0.0), details

def _norm(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def _pick_col(df, candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None
    norm = {_norm(c): c for c in df.columns}
    for c in candidates:
        n = _norm(c)
        if n in norm: return norm[n]
    for c in candidates:
        n = _norm(c)
        for k, orig in norm.items():
            if n and n in k: return orig
    return None

def _annual(df):
    if df is None or df.empty: return df
    a = df[df.get("IsQuarter") != True].copy()
    if "Year" in a.columns:
        a["Year"] = pd.to_numeric(a["Year"], errors="coerce")
        a = a.sort_values("Year")
    else:
        a = a.sort_index()
    return a

def _last_float(df, aliases: list[str]) -> float | None:
    col = _pick_col(df, aliases)
    if not col: return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(s.iloc[-1]) if not s.empty else None

def _safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0): return None
    try: return float(a)/float(b)
    except Exception: return None

# direct ratio aliases (match your View/Annual columns if present)
RATIO_ALIASES = {
    "pe":        ["P/E (×)","P/E (x)","P/E","PE (×)","PE (x)","PE"],
    "pb":        ["P/B (×)","P/B (x)","P/B","PB (×)","PB (x)","PB","P/BV","PBV"],
    "ev_ebitda": ["EV/EBITDA (×)","EV/EBITDA","EV / EBITDA"],
    "ev_sales":  ["EV/Sales (×)","EV/Sales","EV / Sales","EV/Revenue"],
    "fcf_yield": ["FCF Yield (%)","FCF yield (%)"],
    "dy":        ["Dividend Yield (%)","Distribution/Dividend Yield (%)","Distribution Yield (%)","DY (%)"],
    "p_nav":     ["P/NAV (×)","P/NAV (x)","P/NAV","P NAV"],
    "p_rnav":    ["P/RNAV (×)","P/RNAV (x)","P/RNAV","P RNAV"],
}

def _latest_direct_ratio(stock_df, aliases: list[str]) -> float | None:
    a = _annual(stock_df)
    col = _pick_col(a, aliases)
    if not col: return None
    s = pd.to_numeric(a[col], errors="coerce").dropna()
    return float(s.iloc[-1]) if not s.empty else None

def _valuation_metrics(stock_df: pd.DataFrame, current_price: float | None, bucket: str | None = None) -> dict:
    ann = stock_df[stock_df.get("IsQuarter") != True].copy() if "IsQuarter" in stock_df.columns else stock_df.copy()
    qtr = stock_df[stock_df.get("IsQuarter") == True].copy() if "IsQuarter" in stock_df.columns else pd.DataFrame()
    price = float(current_price) if (current_price is not None and np.isfinite(current_price)) else None

    # 1) Direct ratios from ANNUAL rows
    pe_d        = _latest_direct_ratio(ann, RATIO_ALIASES["pe"])
    pb_d        = _latest_direct_ratio(ann, RATIO_ALIASES["pb"])
    ev_ebitda_d = _latest_direct_ratio(ann, RATIO_ALIASES["ev_ebitda"])
    ev_sales_d  = _latest_direct_ratio(ann, RATIO_ALIASES["ev_sales"])
    fcfy_d      = _latest_direct_ratio(ann, RATIO_ALIASES["fcf_yield"])
    dy_d        = _latest_direct_ratio(ann, RATIO_ALIASES["dy"])
    pnav_d      = _latest_direct_ratio(ann, RATIO_ALIASES["p_nav"])
    prnav_d     = _latest_direct_ratio(ann, RATIO_ALIASES["p_rnav"])

    # 2) Prefer the exact TTM Summary your View page shows (tolerant to “x” vs “×”)
    pe_s = pb_s = ev_ebitda_s = ev_sales_s = fcfy_s = dy_s = pnav_s = prnav_s = None
    if bucket:
        try:
            from calculations import build_summary_table
            sum_df = build_summary_table(
                annual_df=ann, quarterly_df=qtr, bucket=bucket,
                include_ttm=True, price_fallback=price,
            )
            if isinstance(sum_df, pd.DataFrame) and not sum_df.empty and "Metric" in sum_df.columns:
                ttm_col = next((c for c in reversed(sum_df.columns) if isinstance(c, str) and c.upper().startswith("TTM")), None)

                def _via_summary(label: str) -> float | None:
                    if not ttm_col: return None
                    row = sum_df.loc[sum_df["Metric"] == label, ttm_col]
                    if row.empty: return None
                    v = pd.to_numeric(row, errors="coerce").dropna()
                    return float(v.iloc[0]) if not v.empty and np.isfinite(v.iloc[0]) else None

                def _via_any(labels: list[str]) -> float | None:
                    for lbl in labels:
                        v = _via_summary(lbl)
                        if v is not None:
                            return v
                    return None

                # REPLACE the old single-label lookups with these tolerant ones:
                pe_s        = _via_any(["P/E (×)", "P/E (x)", "P/E"])
                pb_s        = _via_any(["P/B (×)", "P/B (x)", "P/B"])
                ev_ebitda_s = _via_any(["EV/EBITDA (×)", "EV/EBITDA", "EV / EBITDA"])
                ev_sales_s  = _via_any(["EV/Sales (×)", "EV/Sales", "EV / Sales", "EV/Revenue"])
                fcfy_s      = _via_any(["FCF Yield (%)", "FCF yield (%)"])
                dy_s        = _via_any(["Dividend Yield (%)", "Distribution/Dividend Yield (%)", "Distribution Yield (%)", "DY (%)"])
                pnav_s      = _via_any(["P/NAV (×)", "P/NAV (x)", "P/NAV"])
                prnav_s     = _via_any(["P/RNAV (×)", "P/RNAV (x)", "P/RNAV"])

        except Exception:
            pass

    # 3) SUMMARY FIRST, then Annual direct — no math yet
    vals_direct_or_summary = {
        "pe":        pe_s        if pe_s        is not None else pe_d,
        "pb":        pb_s        if pb_s        is not None else pb_d,
        "ev_ebitda": ev_ebitda_s if ev_ebitda_s is not None else ev_ebitda_d,
        "ev_sales":  ev_sales_s  if ev_sales_s  is not None else ev_sales_d,
        "fcf_yield": fcfy_s      if fcfy_s      is not None else fcfy_d,   # % already
        "dy":        dy_s        if dy_s        is not None else dy_d,     # % already
        "p_nav":     pnav_s      if pnav_s      is not None else pnav_d,
        "p_rnav":    prnav_s     if prnav_s     is not None else prnav_d,
    }

    # --- NEW: sanitize obviously bad entries so they don't score 100 by mistake
    import numpy as _np
    def _posfinite(v, *, cap=1e3):
        v = _safe_float(v)
        return v if (v is not None and _np.isfinite(v) and v > 0.0 and abs(v) <= cap) else None

    for k in ("pe","pb","ev_ebitda","ev_sales","p_nav","p_rnav"):
        vals_direct_or_summary[k] = _posfinite(vals_direct_or_summary.get(k))

    for k in ("fcf_yield","dy"):  # allow 0..100%
        v = _safe_float(vals_direct_or_summary.get(k))
        vals_direct_or_summary[k] = (v if (v is not None and _np.isfinite(v) and v >= 0.0 and v <= 100.0) else None)

    # 4) Compute safe fallbacks from fundamentals (used ONLY if still missing)
    price = price if price is not None else _last_float(ann, ["CurrentPrice","SharePrice","Price","Annual Price per Share (RM)"])
    shares     = _last_float(ann, ["Shares","Shares Outstanding","Units Outstanding"])
    market_cap = (price * shares) if (price not in (None, np.nan) and shares not in (None, np.nan)) else _last_float(ann, ["MarketCap","Market Cap"])
    equity     = _last_float(ann, ["Equity","Equity (Book Value)","Shareholders’ Equity","Shareholders' Equity"])
    bvps       = _safe_div(equity, shares)

    eps_col = _pick_col(ann, ["EPS (RM)", "EPS", "Earnings per Share"])
    eps = None
    if eps_col:
        s = pd.to_numeric(ann[eps_col], errors="coerce").dropna()
        eps = float(s.iloc[-1]) if not s.empty else None
    if eps is None:
        npf = _last_float(ann, ["Net Profit", "NetProfit"])
        if npf is not None and shares not in (None, 0):
            eps = npf / shares

    revenue  = _last_float(ann, ["Revenue"])
    ebitda   = _last_float(ann, ["EBITDA"])
    cfo      = _last_float(ann, ["CFO","Operating Cash Flow","Cash Flow from Ops"])
    capex    = _last_float(ann, ["Capex","CAPEX"])
    dps      = _last_float(ann, ["DPS","Dividend per Share (TTM, RM)","DPU"])
    nav_ps   = _last_float(ann, ["NAV per Unit","NAV per Share"])
    rnav_ps  = _last_float(ann, ["RNAV per Share"])
    tot_borr = _last_float(ann, ["Total Borrowings"])
    cash_ce  = _last_float(ann, ["Cash & Cash Equivalents","Cash & Equivalents"])
    net_debt = (float(tot_borr) - float(cash_ce)) if (tot_borr is not None and cash_ce is not None) else None
    ev       = (market_cap + net_debt) if (market_cap is not None and net_debt is not None) else None

    pe_calc        = (price / eps) if (price is not None and eps and eps > 0) else None
    pb_calc        = (price / bvps) if (price is not None and bvps and bvps > 0) else None
    ev_ebitda_calc = _safe_div(ev, ebitda) if (ev is not None and ebitda and ebitda > 0) else None
    ev_sales_calc  = _safe_div(ev, revenue) if (ev is not None and revenue and revenue > 0) else None
    fcf            = (cfo - capex) if (cfo is not None and capex is not None) else None
    fcfy_calc      = _safe_div(fcf, market_cap)
    if fcfy_calc is not None: fcfy_calc *= 100.0
    dy_calc        = _safe_div(dps, price)
    if dy_calc   is not None: dy_calc   *= 100.0
    pnav_calc      = _safe_div(price, nav_ps) if (price is not None and nav_ps and nav_ps > 0) else None
    prnav_calc     = _safe_div(price, rnav_ps) if (price is not None and rnav_ps and rnav_ps > 0) else None

    # 5) Merge: prefer Summary/Direct, else computed (even if STRICT)
    computed = {
        "pe": pe_calc, "pb": pb_calc, "ev_ebitda": ev_ebitda_calc, "ev_sales": ev_sales_calc,
        "fcf_yield": fcfy_calc, "dy": dy_calc, "p_nav": pnav_calc, "p_rnav": prnav_calc,
    }
    out = {}
    for k in ["pe","pb","ev_ebitda","ev_sales","fcf_yield","dy","p_nav","p_rnav"]:
        v_dir = vals_direct_or_summary.get(k)
        out[k] = v_dir if v_dir is not None else computed.get(k)
    return out

# ---- Momentum helpers (same tolerant matcher as the View page) ----
import os, re

def _resolve_ohlc_dir() -> str:
    base = os.path.dirname(__file__)
    for d in [
        os.path.join(base, "data", "ohlc"),
        os.path.join(os.path.dirname(base), "data", "ohlc"),
        os.path.join(os.path.dirname(os.path.dirname(base)), "data", "ohlc"),
        os.path.join(os.getcwd(), "data", "ohlc"),
    ]:
        if os.path.isdir(d):
            return os.path.abspath(d)
    return os.path.abspath(os.path.join(base, "data", "ohlc"))

_OHLC_DIR = _resolve_ohlc_dir()

def _safe_ohlc_name(x: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(x)).strip("_")

def _alias_candidates(x: str) -> list[str]:
    base = _safe_ohlc_name(x)
    cand = {base}
    for suf in ["_Berhad", "_Bhd", "_PLC"]:
        if base.endswith(suf):
            cand.add(base[: -len(suf)])
    cand.add(re.sub(r'_(Berhad|Bhd|PLC)$', '', base, flags=re.I))
    return [c for c in dict.fromkeys([c for c in cand if c])]

def _ohlc_path_for(stock_name: str) -> str | None:
    exact = os.path.join(_OHLC_DIR, f"{_safe_ohlc_name(stock_name)}.csv")
    if os.path.exists(exact):
        return exact
    try:
        files = {f.lower(): os.path.join(_OHLC_DIR, f)
                 for f in os.listdir(_OHLC_DIR) if f.lower().endswith(".csv")}
        for cand in _alias_candidates(stock_name):
            target = f"{cand}.csv".lower()
            if target in files:
                return files[target]
            for fname, full in files.items():
                if fname.startswith(cand.lower()):
                    return full
    except Exception:
        pass
    return None

def _load_ohlc_for(stock_name: str):
    p = _ohlc_path_for(stock_name)
    if not p or not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (
        df.dropna(subset=["Date","Close"])
          .drop_duplicates(subset=["Date"])
          .sort_values("Date")
          .reset_index(drop=True)
    )
    return df

def _mom_change_pct_df(df, months: int = 12) -> float | None:
    if df is None or df.empty:
        return None
    last_dt = df["Date"].iloc[-1]
    import pandas as _pd
    start_dt = last_dt - _pd.Timedelta(days=365 if months >= 12 else 30*months)
    prior = df[df["Date"] <= start_dt]
    if not prior.empty:
        prior_close = float(prior["Close"].iloc[-1])
    else:
        if len(df) < 200:
            return None
        idx = max(0, len(df) - 252)
        prior_close = float(df["Close"].iloc[idx])
    last_close = float(df["Close"].iloc[-1])
    if prior_close and np.isfinite(prior_close) and prior_close != 0 and np.isfinite(last_close):
        return (last_close / prior_close - 1.0) * 100.0
    return None

def _ret_12m_pct_from_ohlc(stock_name: str) -> float | None:
    df = _load_ohlc_for(stock_name)
    return _mom_change_pct_df(df, months=12) if df is not None else None

def _band_score(v: float | None, lo: float, hi: float) -> float:
    if v is None or not np.isfinite(v): return 0.0
    if lo == hi: return 50.0
    if lo < hi:  t = (float(v)-lo) / (hi-lo)         # higher is better
    else:        t = (lo - float(v)) / (lo - hi)     # lower is better
    return float(np.clip(t, 0.0, 1.0) * 100.0)

# --- keep ONLY this block (replace any duplicates below) ---
def anchors_for(bucket: str) -> dict:
    return (ANCHORS_BY_BUCKET.get(bucket) or ANCHORS_BY_BUCKET.get("General") or {})

def weights_for(bucket: str) -> dict:
    wb = _cfg_or(_DEFAULT_BLOCK_WEIGHTS_BY_BUCKET, "BLOCK_WEIGHTS_BY_BUCKET")
    base = _cfg_or(_DEFAULT_BLOCK_WEIGHTS, "BLOCK_WEIGHTS")
    return (wb.get(bucket) or wb.get("General") or base or {}).copy()

def min_thresholds_for(bucket: str) -> tuple[int, int]:
    msb = _cfg_or(_DEFAULT_MIN_SCORE_BY_BUCKET, "MIN_SCORE_BY_BUCKET")
    mvb = _cfg_or(_DEFAULT_MIN_VALUATION_BY_BUCKET, "MIN_VALUATION_BY_BUCKET")
    return int(msb.get(bucket, msb.get("General", 65))), int(mvb.get(bucket, mvb.get("General", 50)))


def valuation_block_score(stock_df: pd.DataFrame, bucket: str, current_price: float | None) -> tuple[float, str, float]:
    prof = BUCKET_PROFILES.get(bucket) or BUCKET_PROFILES.get("General") or {}
    vals = _valuation_metrics(stock_df, current_price, bucket=bucket)
    entry_keys = prof.get("entry", []) or []
    bands = prof.get("value", {}) or {}
    per = []; avail = 0
    for k in entry_keys:
        b = bands.get(k, {}); lo=b.get("lo"); hi=b.get("hi")
        if lo is None or hi is None: continue
        lower_better = bool(b.get("lower_better", False))
        lo_eff, hi_eff = (hi, lo) if lower_better else (lo, hi)
        v = vals.get(k)
        if v is not None and np.isfinite(v): avail += 1
        per.append(_band_score(v, float(lo_eff), float(hi_eff)))
    if not per: return (0.0, "—", 0.0)
    avg = float(np.mean(per)); conf = float(avail / max(1,len(per)))
    label = "Cheap" if avg>=70 else ("Fair" if avg>=40 else "Rich")
    return (avg, label, conf)

def _score_block_from_anchors(
    stock_df: pd.DataFrame,
    anchors: dict,
    fd_rate_decimal: float,
    *,
    bucket: str,
    stock_name_hint: str | None = None,
) -> tuple[float, float]:
    a   = stock_df[stock_df.get("IsQuarter") != True].copy() if "IsQuarter" in stock_df.columns else stock_df.copy()
    qtr = stock_df[stock_df.get("IsQuarter") == True].copy() if "IsQuarter" in stock_df.columns else pd.DataFrame()

    # Build the same TTM Summary used by View
    sum_df, ttm_col = None, None
    price_fallback = _last_float(stock_df, ["CurrentPrice", "SharePrice", "Price", "Annual Price per Share (RM)"])
    try:
        try:
            from utils import calculations as _calc
        except Exception:
            import calculations as _calc
        sum_df = _calc.build_summary_table(
            annual_df=a, quarterly_df=qtr,
            bucket=bucket, include_ttm=True,
            price_fallback=price_fallback,
        )
        if isinstance(sum_df, pd.DataFrame) and not sum_df.empty and "Metric" in sum_df.columns:
            ttm_col = next((c for c in reversed(sum_df.columns) if isinstance(c, str) and c.upper().startswith("TTM")), None)
    except Exception:
        sum_df, ttm_col = None, None

    def via_summary(label: str) -> float | None:
        if sum_df is None or not ttm_col: return None
        row = sum_df.loc[sum_df["Metric"] == label, ttm_col]
        if row.empty: return None
        s = pd.to_numeric(row, errors="coerce").dropna()
        return float(s.iloc[0]) if not s.empty else None

    def via_last_fy(label: str) -> float | None:
        if sum_df is None or "Metric" not in sum_df.columns:
            return None
        import re
        cols = [c for c in sum_df.columns if isinstance(c, str)]
        fy_cols = [c for c in cols if re.match(r"^(FY\s*)?\d{4}$", c.strip(), flags=re.I)]
        if not fy_cols:
            return None
        col = fy_cols[-1]
        row = sum_df.loc[sum_df["Metric"] == label, col]
        s = pd.to_numeric(row, errors="coerce").dropna()
        return float(s.iloc[0]) if not s.empty else None

    def yoy_from_annual(names: list[str]) -> float | None:
        col = _pick_col(a, names)
        if not col: return None
        s = pd.to_numeric(a[col], errors="coerce").dropna()
        if len(s) >= 2 and float(s.iloc[-2]) != 0:
            return (float(s.iloc[-1]) / float(s.iloc[-2]) - 1.0) * 100.0
        return None

    stock_name_from_df = None
    if "Name" in stock_df.columns:
        s_name = stock_df["Name"].dropna()
        if not s_name.empty:
            stock_name_from_df = str(s_name.iloc[0])

    def _pick_stock_name() -> str | None:
        return stock_name_hint or stock_name_from_df

    def value_for(key: str) -> float | None:
        # ---- STRICT: TTM Summary only (except momentum fallback to OHLC) ----
        if STRICT_NO_COMPUTE:
            if key == "cfo_margin_pct":
                return via_summary("Operating CF Margin (%)") or via_summary("CFO Margin (%)")
            if key == "fcf_margin_pct":
                return via_summary("FCF Margin (%)")
            if key in ("cash_conv_cfo_ebitda","cash_conv_cfo_ebitda_pct"):
                v = via_summary("CFO/EBITDA (×)") or via_summary("CFO/EBITDA (x)")
                if v is None:
                    v_pct = via_summary("CFO/EBITDA (%)") or via_summary("Cash Conversion (CFO/EBITDA)")
                    return (float(v_pct)/100.0) if v_pct is not None else None
                return v
            if key == "ebitda_margin_pct":
                return via_summary("EBITDA Margin (%)")
            if key in ("icr_x","icr"):
                return (via_summary("Interest Coverage (x)") or via_summary("Interest Coverage (×)"))
            if key == "avg_cost_debt_pct":
                return via_summary("Average Cost of Debt (%)")
            if key == "capex_to_rev_pct":
                return (via_summary("Capex/Revenue (%)") or via_summary("Capex to Revenue (%)"))
            if key == "eps_yoy_pct":
                v = via_summary("EPS YoY (%)")
                return v if v is not None else yoy_from_annual(["EPS (RM)", "EPS", "Earnings per Share"])
            if key == "revenue_yoy_pct":
                v = (via_summary("Revenue YoY (%)") or via_summary("Revenue YoY"))
                return v if v is not None else yoy_from_annual(["Revenue"])
            if key == "payout_pct":
                return (via_summary("Payout Ratio (%)") or via_last_fy("Payout Ratio (%)"))
            if key in ("mom_12m_pct","price_change_12m_pct"):
                v = (via_summary("12M Price Change (%)")
                     or via_summary("Price Change 12M (%)")
                     or via_summary("Total Return 1Y (%)"))
                if v is not None:
                    return v
                stock_name = _pick_stock_name()
                return _ret_12m_pct_from_ohlc(stock_name) if stock_name else None
            direct_map = {
                "nim_pct":       "NIM (%)",
                "cir_pct":       "Cost-to-Income Ratio (%)",
                "npl_ratio_pct": "NPL Ratio (%)",
                "wale_years":    "WALE (years)",
                "occupancy_pct": "Occupancy (%)",
            }
            if key in direct_map:
                return via_summary(direct_map[key])
            return None

        # ---- Non-strict (compute if Summary missing) ------------------------
        if key == "cfo_margin_pct":
            cfo = _last_float(a, ["CFO","Operating Cash Flow","Cash Flow from Ops"])
            rev = _last_float(a, ["Revenue"])
            v = _safe_div(cfo, rev)
            return v*100.0 if v is not None else (via_summary("Operating CF Margin (%)") or via_summary("CFO Margin (%)"))

        if key == "fcf_margin_pct":
            cfo = _last_float(a, ["CFO","Operating Cash Flow","Cash Flow from Ops"])
            cap = _last_float(a, ["Capex","CAPEX"])
            rev = _last_float(a, ["Revenue"])
            if cfo is not None and cap is not None and rev not in (None, 0):
                return ((cfo - cap) / rev) * 100.0
            return via_summary("FCF Margin (%)")

        if key in ("cash_conv_cfo_ebitda","cash_conv_cfo_ebitda_pct"):
            cfo    = _last_float(a, ["CFO","Operating Cash Flow","Cash Flow from Ops"])
            ebitda = _last_float(a, ["EBITDA"])
            v = _safe_div(cfo, ebitda)
            if v is not None:
                return v
            v = via_summary("CFO/EBITDA (×)") or via_summary("CFO/EBITDA (x)")
            if v is not None:
                return v
            v_pct = via_summary("CFO/EBITDA (%)") or via_summary("Cash Conversion (CFO/EBITDA)")
            return (float(v_pct)/100.0) if v_pct is not None else None

        if key == "ebitda_margin_pct":
            e = _last_float(a, ["EBITDA"]); r = _last_float(a, ["Revenue"])
            v = _safe_div(e, r)
            return v*100.0 if v is not None else via_summary("EBITDA Margin (%)")

        if key in ("icr_x","icr"):
            ebit = _last_float(a, ["Operating Profit","EBIT","Operating profit"])
            ie   = _last_float(a, ["Interest Expense","Finance Costs"])
            v = _safe_div(ebit, abs(ie) if ie is not None else None)
            return v if v is not None else (via_summary("Interest Coverage (x)") or via_summary("Interest Coverage (×)"))

        if key == "avg_cost_debt_pct":
            ie = _last_float(a, ["Interest Expense","Finance Costs"])
            tb = _last_float(a, ["Total Borrowings"])
            v = _safe_div(ie, tb)
            return v*100.0 if v is not None else via_summary("Average Cost of Debt (%)")

        if key == "capex_to_rev_pct":
            cap = _last_float(a, ["Capex","CAPEX"]); r = _last_float(a, ["Revenue"])
            v = _safe_div(cap, r)
            return v*100.0 if v is not None else (via_summary("Capex/Revenue (%)") or via_summary("Capex to Revenue (%)"))

        if key == "eps_yoy_pct":
            v = yoy_from_annual(["EPS (RM)", "EPS", "Earnings per Share"])
            if v is not None: return v
            # fallback: compute EPS then YoY
            np_col = _pick_col(a, ["Net Profit", "NetProfit"])
            sh_col = _pick_col(a, ["Shares", "Shares Outstanding", "Units Outstanding"])
            if np_col and sh_col:
                num = pd.to_numeric(a[np_col], errors="coerce")
                den = pd.to_numeric(a[sh_col], errors="coerce")
                eps = (num / den).replace([np.inf, -np.inf], np.nan).dropna()
                if len(eps) >= 2 and float(eps.iloc[-2]) != 0:
                    return (float(eps.iloc[-1]) / float(eps.iloc[-2]) - 1.0) * 100.0
            return via_summary("EPS YoY (%)")

        if key == "revenue_yoy_pct":
            v = yoy_from_annual(["Revenue"])
            return v if v is not None else (via_summary("Revenue YoY (%)") or via_summary("Revenue YoY"))

        if key == "payout_pct":
            eps = _last_float(a, ["EPS (RM)","EPS","Earnings per Share"])
            dps = _last_float(a, ["DPS","Dividend per Share (TTM, RM)","DPU"])
            v = _safe_div(dps, eps)
            return v*100.0 if v is not None else via_summary("Payout Ratio (%)")

        if key in ("mom_12m_pct","price_change_12m_pct"):
            v = (via_summary("12M Price Change (%)")
                 or via_summary("Price Change 12M (%)")
                 or via_summary("Total Return 1Y (%)"))
            if v is not None:
                return v
            stock_name = _pick_stock_name()
            return _ret_12m_pct_from_ohlc(stock_name) if stock_name else None

        direct_map = {
            "nim_pct":       ["NIM (%)","NIM","Net Interest/Financing Margin (%)"],
            "cir_pct":       ["Cost-to-Income Ratio (%)","Cost-to-Income","CIR","C/I (%)"],
            "npl_ratio_pct": ["NPL Ratio (%)","NPL (%)"],
            "wale_years":    ["WALE (years)","WALE"],
            "occupancy_pct": ["Occupancy (%)","Hotel Occupancy (%)"],
        }
        if key in direct_map:
            v = _last_float(a, direct_map[key])
            return v if v is not None else via_summary(direct_map[key][0])

        return None

    scores, avail = [], 0

    # Dividend special case (Snowflake-style mapping using FD/EPF)
    if "dy_vs_fd_x" in (anchors or {}):
        dy_pct = (via_summary("Dividend Yield (%)")
                  or via_summary("Distribution Yield (%)")
                  or via_last_fy("Dividend Yield (%)")
                  or via_last_fy("Distribution Yield (%)"))
        if not STRICT_NO_COMPUTE and dy_pct is None:
            dps   = _last_float(a, ["DPS","Dividend per Share (TTM, RM)","DPU"])
            price = _last_float(a, ["CurrentPrice","Price","SharePrice","Annual Price per Share (RM)"])
            v = _safe_div(dps, price)
            dy_pct = (v * 100.0) if v is not None else None
        if dy_pct is not None and np.isfinite(float(dy_pct)):
            scores.append(_score_dy_snowflake_style(float(dy_pct))); avail += 1
        else:
            scores.append(0.0)

    for metric_key, (lo, hi) in ((anchors or {}).items()):
        if metric_key == "dy_vs_fd_x":
            continue
        v = value_for(metric_key)
        if v is not None and np.isfinite(v):
            avail += 1
        scores.append(_band_score(v, float(lo), float(hi)))

    if not scores:
        return 0.0, 0.0
    return float(np.mean(scores)), float(avail / max(1, len(scores)))

# ---------- Public API --------------------------------------------------------
def funnel_rule(
    *,
    name: str,
    stock_df: pd.DataFrame,
    bucket: str,
    current_price: float | None = None,
    fd_rate_decimal: float = 0.035,
) -> dict:
    bucket = bucket or "General"
    anchors_all = anchors_for(bucket)

    # If you keep ANCHORS_BY_BUCKET empty, everything will be 0.
    # (Optional) Tiny built-in fallback so you see non-zero scores immediately.
    if not anchors_all:
        anchors_all = {
            "cashflow_first": {"cfo_margin_pct": (0.0, 20.0), "fcf_margin_pct": (-5.0, 10.0), "cash_conv_cfo_ebitda": (0.6, 1.0)},
            "ttm_vs_lfy":     {"eps_yoy_pct": (-10.0, 20.0), "revenue_yoy_pct": (-10.0, 15.0)},
            "growth_quality": {"ebitda_margin_pct": (10.0, 40.0), "icr_x": (1.5, 5.0)},
            "dividend":       {"dy_vs_fd_x": (0.5, 1.5), "payout_pct": (0.0, 80.0)},
            "momentum":       {"revenue_yoy_pct": (-10.0, 15.0)},
        }

    s_cash, c_cash = _score_block_from_anchors(stock_df, anchors_all.get("cashflow_first", {}), fd_rate_decimal, bucket=bucket, stock_name_hint=name)
    s_ttm,  c_ttm  = _score_block_from_anchors(stock_df, anchors_all.get("ttm_vs_lfy", {}),     fd_rate_decimal, bucket=bucket, stock_name_hint=name)
    s_gq,   c_gq   = _score_block_from_anchors(stock_df, anchors_all.get("growth_quality", {}), fd_rate_decimal, bucket=bucket, stock_name_hint=name)
    s_div,  c_div  = _score_block_from_anchors(stock_df, anchors_all.get("dividend", {}),       fd_rate_decimal, bucket=bucket, stock_name_hint=name)
    s_mom,  c_mom  = _score_block_from_anchors(stock_df, anchors_all.get("momentum", {}),       fd_rate_decimal, bucket=bucket, stock_name_hint=name)

    s_val, lbl_val, c_val = valuation_block_score(stock_df, bucket, current_price)

    w = weights_for(bucket) or {}
    base = {"cashflow_first": .25, "ttm_vs_lfy": .15, "growth_quality": .25,
            "valuation_entry": .20, "dividend": .10, "momentum": .05}
    ww = {**base, **w}
    s = sum(ww.values()) or 1.0
    for k in ww: ww[k] = ww[k]/s

    composite = (
        s_cash*ww["cashflow_first"] + s_ttm*ww["ttm_vs_lfy"] + s_gq*ww["growth_quality"] +
        s_val*ww["valuation_entry"] + s_div*ww["dividend"] + s_mom*ww["momentum"]
    )

    return {
        "name": name, "bucket": bucket, "composite": float(composite),
        "blocks": {
            "cashflow_first": {"score": s_cash, "conf": c_cash},
            "ttm_vs_lfy":     {"score": s_ttm,  "conf": c_ttm},
            "growth_quality": {"score": s_gq,   "conf": c_gq},
            "valuation_entry":{"score": s_val,  "label": lbl_val, "conf": c_val},
            "dividend":       {"score": s_div,  "conf": c_div},
            "momentum":       {"score": s_mom,  "conf": c_mom},
        },
    }
# === end rules.py =============================================================
