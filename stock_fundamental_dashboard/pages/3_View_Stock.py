# pages/3_View_Stock.py

from __future__ import annotations

from auth_gate import require_auth
require_auth()

from utils.ui import (
    setup_page,
    render_kpi_text_grid,
    section,
    render_compare_header,
    render_page_title,
)
setup_page("View Stock")
render_page_title("View Stock")

# ---- Compare dialog: full-height & non-blocking backdrop ----
import streamlit as st
st.markdown("""
<style>
/* —— Wide, white modal —— */
div[data-baseweb="modal"]{
  align-items:flex-start !important;
  padding-top:24px !important;
}

/* Modal panel: wider + white */
div[data-baseweb="modal"] > div,
div[data-baseweb="modal"] [role="dialog"]{
  width:min(1200px, 96vw) !important;   /* a bit wider */
  max-width:96vw !important;
  background:#fff !important;           /* white background */
  border-radius:12px !important;
  border:1px solid #e5e7eb !important;
  box-shadow:0 12px 32px rgba(0,0,0,.18) !important;
  margin:0 auto !important;
}

/* Ensure inner container is white and scrolls if tall */
div[data-baseweb="modal"] [role="document"],
div[data-baseweb="modal"] [role="dialog"] > div{
  background:#fff !important;           /* keep canvas white */
  max-height:calc(100vh - 64px) !important;
  overflow:auto !important;
}

/* Standard dark, click-blocking backdrop */
div[data-baseweb="backdrop"],
div[data-baseweb="modal"] ~ div[data-baseweb="backdrop"],
div[data-baseweb="modal"] [data-baseweb="backdrop"],
div[role="presentation"][style*="background-color"]{
  background:rgba(0,0,0,.5) !important;
  pointer-events:auto !important;
}

/* Let content use full width inside the modal */
div[data-baseweb="modal"] [data-testid="block-container"]{
  max-width:unset !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- Imports (UI-only; no calculations) ----------
import os, sys, re, time, math
import json 
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

# ---------- Path setup so imports work from /pages ----------
_THIS   = os.path.dirname(__file__)
_PARENT = os.path.abspath(os.path.join(_THIS, ".."))
_GRANDP = os.path.abspath(os.path.join(_THIS, "..", ".."))

for p in (_PARENT, _GRANDP):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# Robust imports (no calculations)
try:
    import io_helpers
except Exception:
    from utils import io_helpers  # type: ignore

try:
    import config
except Exception:
    try:
        from utils import config  # type: ignore
    except Exception:
        import importlib
        config = importlib.import_module("config")  # last resort

try:
    import calculations
except Exception:
    from utils import calculations  # type: ignore

# --- View→Decision snapshot helpers ---
def _safe_key(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name or "")).lower()

def _build_syn_idx(bucket: str, sum_df: pd.DataFrame) -> dict:
    """
    Tolerant synonym index so the Decision page can resolve labels identically.
    Pulls from config if available; falls back to what's in the Summary table.
    """
    idx = {}
    try:
        cats = getattr(config, "INDUSTRY_SUMMARY_RATIOS_CATEGORIES", {}) or {}
        src = (cats.get(bucket) or cats.get("General") or {})
        for _cat, items in (src or {}).items():
            if not isinstance(items, dict): 
                continue
            for canonical, syns in items.items():
                canon = str(canonical).strip()
                base  = re.sub(r"\s*\([^)]*\)$", "", canon).strip()
                variants = {
                    canon, base,
                    base + " (%)", base + " (x)", base + " (×)",
                    canon.replace(" (×)", " (x)"),
                    canon.replace(" (x)", " (×)")
                }
                for v in variants:
                    idx[v.strip().lower()] = canon
                for s in (syns or []):
                    idx[str(s).strip().lower()] = canon
    except Exception:
        pass
    if isinstance(sum_df, pd.DataFrame) and "Metric" in sum_df.columns:
        for m in sum_df["Metric"].astype(str):
            idx[m.strip().lower()] = m
    return idx

def save_view_snapshot(name: str, bucket: str, sum_df: pd.DataFrame) -> None:
    """
    Save the exact TTM column the user sees so the Decision page can read it:
      - st.session_state['ttm_dict_<safe>'] = {Metric -> TTM float/None}
      - st.session_state['syn_idx_<safe>']  = {lowercased variants -> canonical}
      - st.session_state['bucket_<safe>']   = last selected bucket
    """
    if sum_df is None or sum_df.empty or "Metric" not in sum_df.columns:
        return
    ttm_col = next((c for c in reversed(sum_df.columns)
                    if isinstance(c, str) and c.upper().startswith("TTM")), None)
    if not ttm_col:
        return

    skey = _safe_key(name)
    ttms = pd.to_numeric(sum_df[ttm_col], errors="coerce")
    ttm_pairs = dict(zip(sum_df["Metric"], ttms))
    ttm_dict = {str(k): (float(v) if pd.notna(v) else None) for k, v in ttm_pairs.items()}

    st.session_state[f"ttm_dict_{skey}"] = ttm_dict
    st.session_state[f"syn_idx_{skey}"]  = _build_syn_idx(bucket, sum_df)
    st.session_state[f"bucket_{skey}"]   = bucket

# ---------- Global data-version (matches Add/Edit page) ----------
_VERSION_FILE = os.path.join(_GRANDP, ".data.version")
def _data_etag() -> int:
    try:
        return int(os.stat(_VERSION_FILE).st_mtime_ns)
    except Exception:
        return 0
    
# --- Global app settings (shared across pages) ---
_SETTINGS_FILE = os.path.join(_GRANDP, "data", "app_settings.json")

def _load_settings() -> dict:
    try:
        with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
            import json as _json
            return _json.load(f)
    except Exception:
        return {}

def _save_settings(d: dict) -> None:
    os.makedirs(os.path.dirname(_SETTINGS_FILE), exist_ok=True)
    with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
        import json as _json
        _json.dump(d, f, indent=2, ensure_ascii=False)
    # bump data version so other pages see the change
    try:
        open(_VERSION_FILE, "a").close()
        os.utime(_VERSION_FILE, None)
    except Exception:
        pass

def get_fd_eps_rate() -> float | None:
    v = _load_settings().get("fd_eps_rate")
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except Exception:
        return None
    
def get_epf_rate() -> float | None:
    v = _load_settings().get("epf_rate")
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def get_epf_rate_prev() -> float | None:
    v = _load_settings().get("epf_rate_prev")
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except Exception:
        return None

# ---------- Helpers (no math; just parsing/formatting/sorting) ----------
def _qnum(q):
    if pd.isna(q): return np.nan
    m = re.search(r"(\d+)", str(q).upper())
    return int(m.group(1)) if m else np.nan

def _auto_h(df_like, row_h=28, base=88, max_h=1400):
    if df_like is None:
        return 200
    n = len(df_like) if hasattr(df_like, "__len__") else 12
    return int(min(max_h, base + row_h * (n + 1)))

def _for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """Make a display-only copy with all column labels as strings (kills the mixed-type warning)."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = pd.MultiIndex.from_tuples([tuple(str(x) for x in t) for t in out.columns])
    else:
        out.columns = [str(c) for c in out.columns]
    return out

def _build_annual_quarter_tables(stock_df: pd.DataFrame):
    """Split + sort without deriving any new numbers."""
    annual = stock_df[stock_df["IsQuarter"] != True].copy()
    qtr    = stock_df[stock_df["IsQuarter"] == True].copy()
    if not annual.empty:
        annual = annual.drop(columns=[c for c in ("IsQuarter","Quarter") if c in annual.columns])
        annual = annual.sort_values("Year")
    if not qtr.empty:
        qtr["Qnum"] = qtr["Quarter"].map(_qnum)
        qtr = qtr.dropna(subset=["Year","Qnum"]).sort_values(["Year","Qnum"])
    return annual, qtr

def _bucket_for(rows: pd.DataFrame) -> str:
    b = rows.get("IndustryBucket")
    if b is not None and b.dropna().size:
        v = str(b.dropna().iloc[-1])
        try:
            return v if v in getattr(config, "INDUSTRY_BUCKETS", ()) else config.resolve_bucket(v)
        except Exception:
            return v
    ind = rows.get("Industry")
    if ind is not None and ind.dropna().size:
        try:
            return config.resolve_bucket(str(ind.dropna().iloc[-1]))
        except Exception:
            return str(ind.dropna().iloc[-1])
    return "General"

def _industry_label(rows: pd.DataFrame) -> str:
    if rows.get("Industry") is not None and rows["Industry"].dropna().size:
        return str(rows["Industry"].dropna().iloc[-1])
    return ""

def _current_price(rows: pd.DataFrame) -> float:
    """Just pick the latest stored price-like field — no math."""
    for k in ("CurrentPrice", "EndQuarterPrice", "Price", "SharePrice"):
        s = rows.get(k)
        if s is not None and s.dropna().size:
            try:
                v = float(pd.to_numeric(s, errors="coerce").dropna().iloc[-1])
                if np.isfinite(v):
                    return v
            except Exception:
                pass
    return float("nan")

# ---------- Ratio table (FY-only) helpers: detect → pivot → style ----------
# Columns we never treat as ratios
_RATIO_EXCLUDE = {
    "name", "year", "quarter", "isquarter", "industry", "industrybucket",
}

# Heuristics to detect ratio-like columns you already have in ANNUAL data
def _is_ratio_col_name(col: str) -> bool:
    if not isinstance(col, str):
        return False
    s = col.strip()
    sl = s.lower()
    if sl in _RATIO_EXCLUDE:
        return False
    # obvious hints
    hints = ("(%)", " margin", "yield", " ratio", "p/e", "peg", "ev/ebitda",
             "roe", "roa", "coverage", "turnover", "(×)", " (x)")
    if any(h in sl for h in hints):
        return True
    # common exacts
    exacts = {"p/b", "pnav", "p/nav", "icr", "current ratio", "quick ratio"}
    return sl in exacts

def _collect_ratio_cols(annual_df: pd.DataFrame) -> list[str]:
    cols = []
    for c in annual_df.columns:
        if isinstance(c, str) and _is_ratio_col_name(c):
            cols.append(c)
    return cols

def _years_sorted(annual_df: pd.DataFrame) -> list[int]:
    ys = pd.to_numeric(annual_df.get("Year", pd.Series(dtype="float64")), errors="coerce").dropna()
    ys = ys[ys < 9000]  # ignore any sentinel values
    return sorted(ys.astype(int).unique().tolist())

def _pivot_ratio_table_from_annual(annual_df: pd.DataFrame) -> pd.DataFrame:
    """
    No calculations: uses the values already present in your ANNUAL rows.
    Output: DataFrame with 'Ratio' + one column per FY year.
    """
    if annual_df is None or annual_df.empty or "Year" not in annual_df.columns:
        return pd.DataFrame(columns=["Ratio"])

    ratio_cols = _collect_ratio_cols(annual_df)
    if not ratio_cols:
        return pd.DataFrame(columns=["Ratio"])

    years = _years_sorted(annual_df)
    if not years:
        return pd.DataFrame(columns=["Ratio"])

    # for each ratio-col, pick the last non-null value within each FY
    rows = []
    base = annual_df.copy()
    base["__Year"] = pd.to_numeric(base["Year"], errors="coerce")

    for rc in ratio_cols:
        row = {"Ratio": rc}
        for y in years:
            g = base.loc[base["__Year"] == y, rc]
            v = pd.to_numeric(g, errors="coerce").dropna()
            row[y] = (float(v.iloc[-1]) if not v.empty and math.isfinite(float(v.iloc[-1])) else np.nan)
        rows.append(row)

    out = pd.DataFrame(rows, columns=["Ratio"] + years)
    # drop rows that are entirely NaN across years
    if len(years) > 0:
        mask_any = out[years].apply(lambda r: pd.to_numeric(r, errors="coerce").notna().any(), axis=1)
        out = out[mask_any].reset_index(drop=True)
    return out

def _is_percentish_label(label: str) -> bool:
    if not isinstance(label, str):
        return False
    s = label.lower()
    if "(%)" in s or " margin" in s or "yield" in s or " ratio" in s:
        return True
    return False

def _fmt_commas(v, decimals=2):
    try:
        f = float(v)
        if not np.isfinite(f):
            return "—"
        return f"{f:,.{decimals}f}"
    except Exception:
        return "—"

def _fmt_pct(v, decimals=2):
    try:
        f = float(v)
        if not np.isfinite(f):
            return "—"
        return f"{f:,.{decimals}f}%"
    except Exception:
        return "—"

def _style_ratio_table(df_ratios: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """
    Apply % formatting to 'percentish' rows; numeric formatting to the rest.
    """
    if df_ratios is None or df_ratios.empty:
        return df_ratios

    years = [c for c in df_ratios.columns if c != "Ratio"]
    if not years:
        return df_ratios

    percent_rows = df_ratios["Ratio"].map(_is_percentish_label)
    num_rows     = ~percent_rows

    sty = df_ratios.style
    try:
        sty = sty.hide(axis="index")
    except Exception:
        try:
            sty = sty.hide_index()
        except Exception:
            pass

    # Apply row-wise formatting by subset
    sty = sty.format(_fmt_pct,    subset=pd.IndexSlice[percent_rows, years])
    sty = sty.format(_fmt_commas, subset=pd.IndexSlice[num_rows,     years])
    return sty

# ---------- Industry-structured Summary (labels only; no numbers) ----------
from collections import OrderedDict

def _ttm_label_from_annual(annual_df: pd.DataFrame) -> str:
    try:
        ys = pd.to_numeric(annual_df.get("Year", pd.Series(dtype="float64")), errors="coerce").dropna()
        last_fy = int(ys.max()) if not ys.empty else None
    except Exception:
        last_fy = None
    return f"TTM {last_fy + 1}" if last_fy is not None else "TTM"

def _ratio_table_structure(annual: pd.DataFrame, bucket: str) -> pd.DataFrame:
    """
    Build an industry/category grouped table: rows = labeled metrics, columns = FY years + TTM label.
    Values left blank (NaN) — structure only.
    """
    CFG = getattr(config, "INDUSTRY_SUMMARY_RATIOS_CATEGORIES", {}) or {}
    ORDER = getattr(config, "SUMMARY_RATIO_CATEGORY_ORDER", [
        "Margins","Returns","Efficiency & Working Capital","Liquidity & Leverage",
        "Cash Flow","Operations / Industry KPIs","Income & Efficiency",
        "Three-Proportion (Banking)","Asset Quality","Capital & Liquidity",
        "Debt & Hedging","Portfolio Quality","Distribution & Valuation","Valuation",
    ])

    # build column set = FY years + TTM label
    years = []
    if isinstance(annual, pd.DataFrame) and not annual.empty and "Year" in annual.columns:
        years = (
            pd.to_numeric(annual["Year"], errors="coerce").dropna().astype(int).sort_values().tolist()
        )
    periods = [*years]
    ttm_tag = _ttm_label_from_annual(annual)
    periods.append(ttm_tag)

    # ordered category map for bucket
    cats = CFG.get(bucket, {}) or CFG.get("General", {}) or {}
    def _ordered(d):
        out = OrderedDict()
        for sec in ORDER:
            if sec in d: out[sec] = OrderedDict(d[sec])
        for sec, items in d.items():
            if sec not in out: out[sec] = OrderedDict(items)
        return out
    cats = _ordered(cats)

    rows = []
    cols = ["Metric"] + years + [ttm_tag]
    for cat, lab_map in cats.items():
        rows.append([f"— {cat} —"] + [np.nan]*len(periods))
        if isinstance(lab_map, dict):
            for display_label in lab_map.keys():
                rows.append([display_label] + [np.nan]*len(periods))

    return pd.DataFrame(rows, columns=cols)

def _style_structured_ratio_table(comp: pd.DataFrame):
    if comp.empty:
        return comp
    style_grid = pd.DataFrame("", index=comp.index, columns=comp.columns)
    is_cat = comp["Metric"].astype(str).str.match(r"^\s*— .* —\s*$")
    cat_rows = comp.index[is_cat]
    CAT_BG = "#eef2ff"
    for ridx in cat_rows:
        for col in comp.columns:
            style_grid.loc[ridx, col] += f"background-color:{CAT_BG}; text-align:center; font-weight:800;"
            if col != "Metric":
                style_grid.loc[ridx, col] += "border-left:none !important; border-right:none !important;"
    style_grid.loc[cat_rows, "Metric"] += "letter-spacing:.2px;"

    try:
        return comp.style.hide(axis="index").apply(lambda _: style_grid, axis=None)
    except Exception:
        return comp
    
def _summary_insert_category_rows(sum_df: pd.DataFrame, *, industry: str | None,
                                  prefix_category_on_metric: bool = True,
                                  prefix_industry: bool = True) -> pd.DataFrame:
    """
    Take the calculated summary table with columns: ['Category','Metric', <years...>, 'TTM...']
    and return a display table where:
      • Each Category becomes a '— Category —' separator row
      • Each metric label can be prefixed with 'INDUSTRY — <industry> — <Category> — <Metric>'
    """
    if sum_df is None or sum_df.empty:
        return pd.DataFrame()

    # keep all value columns in original order
    value_cols = [c for c in sum_df.columns if c not in ("Category", "Metric")]

    rows: list[pd.Series] = []
    for cat, g in sum_df.groupby("Category", sort=False):
        # category separator row
        rows.append(pd.Series({"Metric": f"— {cat} —", **{c: np.nan for c in value_cols}}))

        # metric rows (optionally prefixed)
        for _, r in g.iterrows():
            label = str(r["Metric"])
            if prefix_category_on_metric:
                parts = []
                if prefix_industry and industry:
                    parts.append(f"INDUSTRY — {industry}")
                parts.append(str(cat))
                label = " — ".join(parts) + " — " + label
            row_dict = {"Metric": label}
            for c in value_cols:
                row_dict[c] = r.get(c, np.nan)
            rows.append(pd.Series(row_dict))

    out = pd.DataFrame(rows, columns=["Metric"] + value_cols)
    return out

def _style_summary_with_categories(df_disp: pd.DataFrame):
    """
    Styling to match the screenshot:
      • Category separator rows centered, bold, tinted
      • % rows formatted with percent, others with commas
      • Hide index
    """
    if df_disp is None or df_disp.empty:
        return df_disp

    years = [c for c in df_disp.columns if c != "Metric"]
    cat_mask = df_disp["Metric"].astype(str).str.match(r"^\s*— .* —\s*$")

    sty = df_disp.style
    try:
        sty = sty.hide(axis="index")
    except Exception:
        pass

    # Row-level styles for category separators
    style_grid = pd.DataFrame("", index=df_disp.index, columns=df_disp.columns)
    CAT_BG = "#eef2ff"
    for i in df_disp.index[cat_mask]:
        for col in df_disp.columns:
            style_grid.loc[i, col] += f"background-color:{CAT_BG}; text-align:center; font-weight:800;"
    style_grid.loc[df_disp.index[cat_mask], "Metric"] += "letter-spacing:.2px;"

    sty = sty.apply(lambda _: style_grid, axis=None)

    # Format: percent-like vs numeric (skip category rows)
    pct_mask = (~cat_mask) & df_disp["Metric"].astype(str).map(_is_percentish_label)
    num_mask = (~cat_mask) & ~pct_mask

    if pct_mask.any():
        sty = sty.format(lambda v: "—" if pd.isna(v) else f"{float(v):,.2f}%",
                         subset=pd.IndexSlice[pct_mask, years])
    if num_mask.any():
        sty = sty.format(lambda v: "—" if pd.isna(v) else f"{float(v):,.2f}",
                         subset=pd.IndexSlice[num_mask, years])

    return sty

# ---------- Schema (labels/keys) → raw tables (no derived fields) ----------
def _schema_by_category(bucket: str) -> "OrderedDict[str, list[dict]]":
    """
    Preferred: config.INDUSTRY_FORM_CATEGORIES[bucket] = { "Income Statement":[{"key","label"}, ...], ... }
    Fallback:  config.INDUSTRY_FORM_FIELDS[bucket] or COMMON_UNIVERSAL_FIELDS (items may carry 'section')
    """
    cats = (getattr(config, "INDUSTRY_FORM_CATEGORIES", {}) or {}).get(bucket)
    if isinstance(cats, dict) and cats:
        out = OrderedDict()
        for cat, items in cats.items():
            lst = []
            for it in (items or []):
                if isinstance(it, str):
                    lst.append({"key": it, "label": it})
                elif isinstance(it, dict) and it.get("key"):
                    lst.append({"key": it["key"], "label": it.get("label", it["key"])})
            out[cat] = lst
        return out

    fields = (getattr(config, "INDUSTRY_FORM_FIELDS", {}) or {}).get(
        bucket, getattr(config, "COMMON_UNIVERSAL_FIELDS", [])
    )
    out = OrderedDict()
    for f in (fields or []):
        if not f.get("key"):
            continue
        cat = f.get("section") or f.get("category") or "KPIs"
        out.setdefault(cat, []).append({"key": f["key"], "label": f.get("label", f["key"])})
    return out

def _raw_df_from_schema(df_in: pd.DataFrame, bucket: str, *, quarterly: bool) -> pd.DataFrame:
    """Build raw table using schema keys only; no extra/derived columns."""
    schema = _schema_by_category(bucket)
    df = df_in.copy()
    if quarterly:
        if "Qnum" not in df.columns and "Quarter" in df.columns:
            df["Qnum"] = df["Quarter"].map(_qnum)

    cols, data = [], []

    # Period columns first
    cols.append(("Period", "Year"));      data.append(df.get("Year", pd.Series(dtype="Int64")))
    if quarterly:
        cols.append(("Period", "Quarter")); data.append(df.get("Quarter", pd.Series(dtype="string")))

    # Schema fields (exact keys from config; quarterly columns prefixed with Q_)
    for cat, items in schema.items():
        for it in items:
            key   = it["key"]
            label = it.get("label", key)
            col   = f"Q_{key}" if quarterly else key
            if col not in df.columns:
                df[col] = pd.NA
            cols.append((cat, label))
            data.append(df[col])

    out = pd.concat(data, axis=1)
    out.columns = pd.MultiIndex.from_tuples(cols)

    # Sort by Year (and Quarter if present)
    y = pd.to_numeric(out[("Period","Year")], errors="coerce")
    if quarterly:
        q = df.get("Qnum")
        if q is None and "Quarter" in df:
            q = pd.to_numeric(df["Quarter"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
        q = q.reindex(y.index)
        order_idx = pd.DataFrame({"__Y": y, "__Q": q}).sort_values(["__Y","__Q"], na_position="last").index
        out = out.loc[order_idx]
    else:
        out = out.loc[y.sort_values(na_position="last").index]

    return out

# ---------- YoY / QoQ helper (optional; light display math) ----------
def _add_yoy_to_multiheader(df_multi: pd.DataFrame, *, is_quarterly: bool):
    """Return a styled DataFrame with YoY or QoQ % columns added (optional UI sugar)."""
    df = df_multi.copy()

    # locate some typical labels
    def _find_lab(labs):
        for c in df.columns:
            if isinstance(c, tuple) and c[1] in labs:
                return c
        return None

    rev_col = _find_lab(["Revenue"])
    np_col  = _find_lab(["Net Profit","NetProfit"])
    eps_col = _find_lab(["EPS"])

    years = pd.to_numeric(df[("Period","Year")], errors="coerce")
    quarters = df[("Period","Quarter")].astype(str) if is_quarterly and ("Period","Quarter") in df.columns else None

    def _yoy(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").pct_change() * 100.0

    def _qoq(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        idx = s.index
        t = pd.DataFrame({"_i": np.arange(len(s)), "y": years, "q": quarters, "v": s})
        t["qn"] = pd.to_numeric(t["q"].str.extract(r"(\d+)")[0], errors="coerce")
        t = t.sort_values(["y","qn","_i"])
        t["chg"] = t["v"].pct_change() * 100.0
        t = t.sort_values("_i")
        return pd.Series(t["chg"].to_numpy(), index=idx)

    if is_quarterly:
        if np_col is not None: df[("Analysis", f"QoQ {np_col[1]} (%)")] = _qoq(df[np_col])
        if rev_col is not None: df[("Analysis", f"QoQ {rev_col[1]} (%)")] = _qoq(df[rev_col])
        if eps_col is not None: df[("Analysis", "QoQ EPS (%)")] = _qoq(df[eps_col])
    else:
        if np_col is not None: df[("Analysis", f"YoY {np_col[1]} (%)")] = _yoy(df[np_col])
        if rev_col is not None: df[("Analysis", f"YoY {rev_col[1]} (%)")] = _yoy(df[rev_col])
        if eps_col is not None: df[("Analysis", "YoY EPS (%)")] = _yoy(df[eps_col])

    # simple formatting
    sty = df.style
    try:
        sty = sty.hide(axis="index")
    except Exception:
        pass

    pct_cols = [c for c in df.columns if isinstance(c, tuple) and "(%)" in c[1]]
    if pct_cols:
        sty = sty.format(lambda v: "-" if pd.isna(v) else f"{float(v):,.2f}%", subset=pd.IndexSlice[:, pct_cols])

    return sty

# ---------- Chart helpers (raw columns only) ----------
def _available_plot_metrics(df: pd.DataFrame, *, quarterly: bool, bucket: str) -> dict:
    """Build {display_label -> actual_column} for plotting from schema keys that exist."""
    label_to_col = {}
    try:
        cats = _schema_by_category(bucket)
        for _, items in cats.items():
            for it in items:
                key, label = it["key"], it.get("label", it["key"])
                col = f"Q_{key}" if quarterly else key
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce")
                    if s.notna().sum() > 0 and label not in label_to_col:
                        label_to_col[label] = col
    except Exception:
        pass
    return label_to_col

def _plot_line(df: pd.DataFrame, xcol: str, ycol: str, title: str, key: str):
    s = pd.to_numeric(df.get(ycol, pd.Series([], dtype="float64")), errors="coerce")
    x = df.get(xcol, pd.Series([], dtype="object")).astype(str)
    if s.notna().sum() == 0:
        st.info(f"No data for {title}.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=s, mode="lines+markers", name=title))
    fig.update_layout(title=title, height=280, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key=key)


# ---------- TTM + display formatting helpers ----------
def _ttm_agg_for(category: str, key: str, label: str) -> str:
    # delegate to the central rule in config.py
    return config._ttm_agg_for(category, key, label)

def _ttm_dict_from_quarters(q_df: pd.DataFrame, bucket: str) -> dict:
    """
    Build a TTM dict from the latest up to 4 quarters with category-aware aggregation:
    - Income Statement / Cash Flow: sum last 4 (flows)
    - Balance Sheet: last (stocks), except '(Avg)' -> mean
    - Ratios/%/Rates: last
    - 'Average ...' fields: mean
    - DPS / DPU / fields tagged TTM / 'New Orders (TTM)': sum
    - Counts/levels (Shares, Units, Price, etc.): last
    """
    if q_df is None or q_df.empty:
        return {}

    q = q_df.copy()
    if "Qnum" not in q.columns and "Quarter" in q.columns:
        q["Qnum"] = q["Quarter"].map(_qnum)
    q = q.dropna(subset=["Year", "Qnum"]).sort_values(["Year", "Qnum"])
    if q.empty:
        return {}

    # take the latest 4 rows (or fewer if not enough quarters)
    tail = q.tail(min(4, len(q)))

    out: dict[str, float] = {}
    schema = _schema_by_category(bucket) or {}  # {category: [ {key,label,...}, ... ]}

    for category, items in schema.items():
        for it in items:
            key   = it.get("key")
            if not key:
                continue
            label = it.get("label", key)

            # try Q_<key>, then <key>, then <label>
            cand_cols = [f"Q_{key}", key, label]
            series = None
            for col in cand_cols:
                if col in tail.columns:
                    series = pd.to_numeric(tail[col], errors="coerce")
                    break
            if series is None:
                continue

            how = config._ttm_agg_for(category, key, label)
            if how == "sum":
                v = series.sum(skipna=True)
            elif how == "mean":
                v = series.dropna().mean()
            else:  # "last"
                non_na = series[series.notna()]
                v = non_na.iloc[-1] if not non_na.empty else np.nan

            if pd.notna(v):
                try:
                    out[key] = float(v)
                except Exception:
                    pass

    return out

def _annual_with_appended_ttm(annual_df: pd.DataFrame, qtr_df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    """
    Append a synthetic 'TTM (next FY)' annual row using the latest up to 4 quarters.
    If last FY is 2024, we append Year=2025 with TTM values where available (others left blank).
    """
    if annual_df is None or annual_df.empty or "Year" not in annual_df.columns:
        return annual_df

    years = pd.to_numeric(annual_df["Year"], errors="coerce").dropna()
    if years.empty:
        return annual_df
    next_fy = int(years.max()) + 1

    ttm_vals = _ttm_dict_from_quarters(qtr_df, bucket)

    # Always append the row; fill what we can
    row = {"Year": next_fy}
    for col in annual_df.columns:
        if col in ("Year", "IsQuarter", "Quarter", "Industry", "IndustryBucket", "Name"):
            continue
        # annual_df columns are schema KEYS; ttm_vals uses KEYS
        v = ttm_vals.get(col, np.nan)
        try:
            row[col] = (np.nan if v is None or not np.isfinite(float(v)) else float(v))
        except Exception:
            row[col] = np.nan

    return pd.concat([annual_df, pd.DataFrame([row])], ignore_index=True)

def _ttm_raw_from_annual_row(annual_df: pd.DataFrame) -> dict:
    """
    If Annual has a synthetic TTM row (Year = last_fy + 1), return a dict of raw keys from it.
    """
    if annual_df is None or annual_df.empty or "Year" not in annual_df.columns:
        return {}
    ys = pd.to_numeric(annual_df["Year"], errors="coerce").dropna()
    if ys.empty:
        return {}
    ttm_year = int(ys.max())          # after _annual_with_appended_ttm this will be next FY (e.g., 2025)
    row = annual_df[annual_df["Year"] == ttm_year].tail(1)
    if row.empty:
        return {}

    d = {}
    drop = {"Year","IsQuarter","Quarter","Industry","IndustryBucket","Name"}
    for k, v in row.iloc[0].items():
        if k in drop:
            continue
        if pd.isna(v):
            continue
        try:
            d[k] = float(v)
        except Exception:
            d[k] = v
    return d

def _apply_commas_to_styler(sty: "pd.io.formats.style.Styler", df: pd.DataFrame):
    """Add thousand-separator formatting to numeric cells in multi-index tables."""
    if ("Period", "Year") in df.columns:
        sty = sty.format(lambda v: "-" if pd.isna(v) else f"{int(float(v))}",
                         subset=pd.IndexSlice[:, [("Period","Year")]])
    value_cols = [c for c in df.columns
                  if not (isinstance(c, tuple) and c[0] in ("Period", "Analysis"))]
    if value_cols:
        sty = sty.format(lambda v: "-" if pd.isna(v) else f"{float(v):,.2f}",
                         subset=pd.IndexSlice[:, value_cols])
    return sty

# ---------- JSON exports (period-first only) ----------
def _num_or_none(v):
    try:
        f = float(v)
        if np.isfinite(f):
            return int(f) if abs(f - int(f)) < 1e-9 else f
    except Exception:
        pass
    return None

def _summary_results_by_year_json(
    sum_df: pd.DataFrame, *, name: str, industry: str, bucket: str,
    keep_metrics: list[str] | None = None, include_ttm: bool = True
) -> dict:
    def _num_or_none_local(v):
        try:
            f = float(v)
            if math.isfinite(f):
                return int(f) if abs(f - int(f)) < 1e-9 else f
        except Exception:
            pass
        return None

    out = {"stock": name, "industry": industry or "", "bucket": bucket, "periods": {}}
    if sum_df is None or sum_df.empty:
        return out

    year_cols = sorted([c for c in sum_df.columns if isinstance(c, (int, np.integer))], reverse=True)
    ttm_cols  = [c for c in sum_df.columns if isinstance(c, str) and c.upper().startswith("TTM")]
    ttm_col   = (ttm_cols[0] if ttm_cols else None)

    for y in year_cols:
        out["periods"][str(y)] = {}
    if include_ttm and ttm_col:
        out["periods"][str(ttm_col)] = {}

    for _, r in sum_df.iterrows():
        label = str(r.get("Metric", "")).strip()
        if label.startswith("—") and label.endswith("—"):
            continue
        if keep_metrics and not any(k.lower() == label.lower() or k.lower() in label.lower() for k in keep_metrics):
            continue
        for y in year_cols:
            v = _num_or_none_local(r.get(y))
            if v is not None:
                out["periods"][str(y)][label] = v
        if include_ttm and ttm_col:
            v = _num_or_none_local(r.get(ttm_col))
            if v is not None:
                out["periods"][str(ttm_col)][label] = v
    return out

def _period_label_for(basis: str, *, annual_df: pd.DataFrame, ttm_col: str | None) -> str:
    if str(basis).upper() == "TTM":
        return (ttm_col or _ttm_label_from_annual(annual_df) or "TTM")
    try:
        y = _last_fy(annual_df)
        return f"FY {y}" if y is not None else "FY"
    except Exception:
        return "FY"

def _ttm_kpis_to_yearfirst_json(
    *, labels: list[str], sum_df: pd.DataFrame | None,
    ttm_col: str | None, ttm_raw: dict, syn_idx: dict,
    price_now: float | None, period_label: str,
    name: str, industry: str, bucket: str,
    annual_df: pd.DataFrame | None = None
) -> dict:

    def _canonical(label: str) -> str:
        if not label:
            return ""
        s = str(label).strip().lower()
        return (
            syn_idx.get(s)
            or syn_idx.get(s.replace(" (%)",""))
            or syn_idx.get(s + " (%)")
            or syn_idx.get(s.replace(" (x)",""))
            or syn_idx.get(s + " (x)")
            or syn_idx.get(s.replace(" (×)",""))
            or syn_idx.get(s + " (×)")
            or label
        )

    # choose a TTM column name if not provided
    ttm_col_eff = None
    if isinstance(sum_df, pd.DataFrame):
        if ttm_col and ttm_col in sum_df.columns:
            ttm_col_eff = ttm_col
        else:
            for c in sum_df.columns:
                if isinstance(c, str) and c.upper().startswith("TTM"):
                    ttm_col_eff = c
                    break

    metrics = {}
    for label in (labels or []):
        val = None

        # 1) try Summary (preferred)
        if isinstance(sum_df, pd.DataFrame) and not sum_df.empty and ttm_col_eff:
            canon = _canonical(label)
            hit = sum_df[sum_df["Metric"].astype(str).str.lower() == str(canon).lower()]
            if not hit.empty and ttm_col_eff in hit.columns:
                val = _num_or_none(hit.iloc[0][ttm_col_eff])

        # 2) raw TTM fallbacks (expanded to include banking absolutes)
        if val is None and isinstance(ttm_raw, dict):
            L = str(label).strip().lower()

            # existing generics
            if L in ("revenue","gross profit","operating profit","ebitda","net profit","eps","dps"):
                val = _num_or_none(ttm_raw.get(label))

            elif L == "dividend yield" and ttm_raw.get("DPS") not in (None, 0) and ttm_raw.get("Price") not in (None, 0):
                try:
                    val = _num_or_none((float(ttm_raw["DPS"]) / float(ttm_raw["Price"])) * 100.0)
                except Exception:
                    pass

            elif L == "p/e" and ttm_raw.get("EPS") not in (None, 0) and ttm_raw.get("Price") not in (None, 0):
                try:
                    val = _num_or_none(float(ttm_raw["Price"]) / float(ttm_raw["EPS"]))
                except Exception:
                    pass

            # NEW: banking absolutes
            elif L in (
                "operating income",
                "operating expenses",
                "provisions",
                "nii (incl islamic)",
                "interest expense",
                "gross loans",
                "deposits",
                "demand deposits",
                "savings deposits",
                "time/fixed deposits",
            ):
                val = _num_or_none(ttm_raw.get(label))

        # 3) final fallback: read the Annual TTM row (e.g., 2025) if present
        if val is None and annual_df is not None:
            try:
                annual_ttm = _ttm_raw_from_annual_row(annual_df) or {}
            except Exception:
                annual_ttm = {}
            if annual_ttm:
                L = str(label).strip().lower()
                if L in (
                    "operating income","operating expenses","provisions","nii (incl islamic)",
                    "interest expense","gross loans","deposits","demand deposits","savings deposits","time/fixed deposits",
                    "revenue","gross profit","operating profit","ebitda","net profit","eps","dps"
                ):
                    val = _num_or_none(annual_ttm.get(label))

        metrics[label] = val if val is not None else None

    return {
        "stock": name, "industry": industry or "", "bucket": bucket,
        "periods": { str(period_label): metrics }
    }

def _cagr_results_to_yearfirst_json(
    *, items: list[str], N:int, end_basis:str,
    sum_df: pd.DataFrame | None, annual: pd.DataFrame | None,
    quarterly: pd.DataFrame | None, ttm_col: str | None, price_now: float | None,
    name:str, industry:str, bucket:str
) -> dict:
    """
    JSON for CAGR / MOS / PEG that matches the cards exactly.
    Rules (strict, no window shrinking):
      • FY end:  need N+1 FY years; start = last_fy - N; end = last_fy
      • TTM end: need N   FY years; start = last_fy - (N-1); end = TTM column
    CFO/FCF TTM are built from the last 4 quarters (CFO - |Capex|).
    """
    EB = str(end_basis).upper()
    N  = int(N)

    def _num_or_none_local(v):
        try:
            f = float(v)
            if math.isfinite(f):
                return int(f) if abs(f - int(f)) < 1e-9 else f
        except Exception:
            pass
        return None

    # ----- helpers --------------------------------------------------------
    def _years_in_summary(df: pd.DataFrame) -> list[int]:
        if df is None or df.empty: return []
        return sorted([c for c in df.columns if isinstance(c, (int, np.integer))])

    def _syn_index_for_bucket(bucket_name: str) -> dict:
        cats = (getattr(config, "INDUSTRY_SUMMARY_RATIOS_CATEGORIES", {}) or {}).get(bucket_name) \
            or (getattr(config, "INDUSTRY_SUMMARY_RATIOS_CATEGORIES", {}) or {}).get("General") \
            or {}
        idx = {}
        for _cat, items in (cats or {}).items():
            if isinstance(items, dict):
                for canonical, syns in items.items():
                    key = str(canonical).strip().lower()
                    idx[key] = canonical
                    for s in (syns or []):
                        idx[str(s).strip().lower()] = canonical
                    for v in [
                        canonical.replace(" (%)","").replace(" (x)","").replace(" (×)",""),
                        f"{canonical} (%)" if "(%)" not in canonical else canonical.replace(" (%)",""),
                        f"{canonical} (x)"  if "(x)"  not in canonical and "×" not in canonical else canonical.replace(" (x)",""),
                        f"{canonical} (×)"  if "×"    not in canonical and "(x)" not in canonical else canonical.replace(" (×)",""),
                    ]:
                        idx[str(v).strip().lower()] = canonical
        return idx

    def _find_row(label: str, syn_idx: dict) -> pd.Series | None:
        if sum_df is None or sum_df.empty: return None
        s = str(label).strip().lower()
        canon = syn_idx.get(s) or s
        hit = sum_df[sum_df["Metric"].astype(str).str.lower() == str(canon).lower()]
        return hit.iloc[0] if not hit.empty else None

    # annual series for raw values (used when Summary row is absent)
    def _series_from_annual(label: str) -> pd.Series | None:
        if annual is None or annual.empty or "Year" not in annual.columns:
            return None
        mapping = {
            "Revenue":     ["Revenue","Total Revenue","TotalRevenue","Sales"],
            "EBITDA":      ["EBITDA"],
            "Net Profit":  ["Net Profit","NPAT","Profit After Tax","PATAMI","Net Income","Profit attributable to owners"],
            "CFO":         ["CFO","Cash Flow from Ops","Cash from Operations","Operating Cash Flow"],
            "Orderbook":   ["Orderbook","Order Book"],
            "NAV per Unit":["NAV per Unit","NAV/Unit","NAVPU","NAV"],
            "Gross Loans": ["Gross Loans","Gross Loan"],
            "Deposits":    ["Deposits","Customer Deposits","Total Deposits"],
        }
        if label == "FCF":
            if "CFO" in annual.columns and ("Capex" in annual.columns or "Capital Expenditure" in annual.columns):
                cfo = pd.to_numeric(annual.get("CFO"), errors="coerce")
                cap = pd.to_numeric(annual.get("Capex", annual.get("Capital Expenditure")), errors="coerce").abs()
                yrs = pd.to_numeric(annual["Year"], errors="coerce")
                if cfo.notna().sum() >= 2 and cap.notna().sum() >= 2:
                    fcf = cfo - cap
                    return pd.Series({int(y): float(v) for y, v in zip(yrs, fcf)})
            return None
        for k in mapping.get(label, []):
            if k in annual.columns:
                s   = pd.to_numeric(annual[k], errors="coerce")
                yrs = pd.to_numeric(annual["Year"], errors="coerce")
                good = (~s.isna()) & (~yrs.isna())
                if good.sum() >= 2:
                    return pd.Series({int(y): float(v) for y, v in zip(yrs[good], s[good])})
        return None

    # strict, card-identical CAGR from a Summary **row**
    def _cagr_from_summary_row_strict(row: pd.Series, years: list[int], N: int,
                                      EB: str, ttm_col: str | None) -> float | None:
        if row is None or not years: return None
        yrs = sorted([int(y) for y in years])
        last_fy = int(yrs[-1])

        if EB == "TTM":
            if not (ttm_col and (ttm_col in row) and pd.notna(row.get(ttm_col))):
                return None
            if len(yrs) < N:
                return None
            y0 = yrs[-N]
            v0 = pd.to_numeric(pd.Series([row.get(y0)]), errors="coerce").iloc[0]
            vE = pd.to_numeric(pd.Series([row.get(ttm_col)]), errors="coerce").iloc[0]
        else:
            if len(yrs) <= N:
                return None
            y0 = yrs[-(N+1)]
            v0 = pd.to_numeric(pd.Series([row.get(y0)]), errors="coerce").iloc[0]
            vE = pd.to_numeric(pd.Series([row.get(last_fy)]), errors="coerce").iloc[0]

        try:
            v0 = float(v0); vE = float(vE)
            if not (np.isfinite(v0) and np.isfinite(vE)) or v0 <= 0 or vE <= 0:
                return None
            return (vE / v0) ** (1.0 / N) - 1.0
        except Exception:
            return None

    # strict, card-identical CAGR from **annual series** with optional TTM end override
    def _cagr_from_annual_series_strict(series: pd.Series, N:int, EB:str,
                                        last_fy:int, ttm_end: float | None) -> float | None:
        if series is None or series.dropna().size < 2: return None
        fy_years = sorted([int(y) for y in series.index if isinstance(y, (int, np.integer)) and y <= last_fy])
        if EB == "TTM":
            if len(fy_years) < N or ttm_end is None:
                return None
            y0 = last_fy - (N - 1)
            v0 = series.get(y0, np.nan)
            vE = float(ttm_end)
        else:
            if len(fy_years) <= N:
                return None
            y0 = last_fy - N
            v0 = series.get(y0, np.nan)
            vE = series.get(last_fy, np.nan)

        if not (pd.notna(v0) and pd.notna(vE)) or v0 <= 0 or vE <= 0:
            return None
        try:
            return (float(vE) / float(v0)) ** (1.0 / N) - 1.0
        except Exception:
            return None

    # Build TTM overrides from quarters for CFO/FCF (and a few basics)
    ttm_vals = _ttm_dict_from_quarters(quarterly if isinstance(quarterly, pd.DataFrame) else pd.DataFrame(), bucket) or {}
    ttm_overrides = {
        "Revenue":    ttm_vals.get("Revenue"),
        "EBITDA":     ttm_vals.get("EBITDA"),
        "Net Profit": ttm_vals.get("Net Profit"),
        "CFO":        ttm_vals.get("CFO"),
        "FCF":        ((ttm_vals.get("CFO") or 0.0) - abs(ttm_vals.get("Capex") or 0.0)) if ("CFO" in ttm_vals or "Capex" in ttm_vals) else None,
    }

    # Summary context
    years  = _years_in_summary(sum_df) if (isinstance(sum_df, pd.DataFrame) and not sum_df.empty) else []
    syn_idx = _syn_index_for_bucket(bucket)
    last_fy = (max(years) if years else _last_fy(annual if isinstance(annual, pd.DataFrame) else pd.DataFrame()))

    metrics: dict[str, float | None] = {}

    # ---- compute each requested CAGR ------------------------------------
    for it in (items or []):
        if not it or not it.lower().endswith(" cagr"):
            continue
        base = it.replace(" CAGR","").strip()
        base = {"Operating Cash Flow":"CFO","Free Cash Flow":"FCF"}.get(base, base)

        g = None
        # 1) Try Summary row (strict)
        row = _find_row(base, syn_idx)
        if isinstance(row, pd.Series) and years and last_fy is not None:
            g = _cagr_from_summary_row_strict(row, years, N, EB, ttm_col)

        # 2) Fallback: annual series + strict rules with TTM override
        if g is None and last_fy is not None:
            s = _series_from_annual("Capex" if base=="FCF" else base) if base != "FCF" else _series_from_annual("FCF")
            ttm_end = ttm_overrides.get(base) if EB == "TTM" else None
            g = _cagr_from_annual_series_strict(s, N, EB, int(last_fy), ttm_end)

        metrics[it] = (g * 100.0) if g is not None else None  # store as percent to match cards

    # PEG (Graham) + Margin of Safety (same as cards)
    try:
        peg = calculations.graham_peg(sum_df, years_back=int(N), end_basis=EB)
    except Exception:
        peg = None
    try:
        mos = calculations.margin_of_safety_pct(sum_df, years_back=int(N), end_basis=EB, price_fallback=price_now)
    except Exception:
        mos = None
    metrics["PEG (Graham)"]   = _num_or_none_local(peg)
    metrics["Margin of Safety"] = _num_or_none_local(mos)

    label = _period_label_for(end_basis, annual_df=(annual if isinstance(annual, pd.DataFrame) else pd.DataFrame()), ttm_col=ttm_col)
    return {
        "stock": name, "industry": industry or "", "bucket": bucket,
        "window_years": int(N), "end_basis": EB,
        "periods": { str(label): metrics }
    }

def _cashflow_to_yearfirst_json(cf: dict, *, basis:str, annual_df: pd.DataFrame | None,
                                name:str, industry:str, bucket:str) -> dict:
    label = _period_label_for(basis, annual_df=(annual_df if isinstance(annual_df, pd.DataFrame) else pd.DataFrame()),
                              ttm_col=None)
    per = {
        "CFO": _num_or_none(cf.get("CFO")),
        "Capex": _num_or_none(cf.get("Capex")),
        "FCF": _num_or_none(cf.get("FCF")),
        "FCF Margin (%)": _num_or_none(cf.get("FCF Margin (%)")),
        "FCF Yield (%)": _num_or_none(cf.get("FCF Yield (%)")),
        "Capex to Revenue (%)": _num_or_none(cf.get("Capex to Revenue (%)")),
        "CFO/EBITDA (%)": _num_or_none(cf.get("CFO/EBITDA (%)")),
        "Cash Conversion (%)": _num_or_none(cf.get("Cash Conversion (%)")),
    }
    return {
        "stock": name, "industry": industry or "", "bucket": bucket,
        "basis": basis, "periods": { str(label): per }
    }

def _clean_num(v):
    try:
        f = float(v)
        if np.isfinite(f):
            return f
    except Exception:
        pass
    return None

def _annual_to_human_json(annual_df: pd.DataFrame, *, name:str, industry:str, bucket:str) -> dict:
    if annual_df is None or annual_df.empty:
        rows = []
    else:
        drop_cols = {"IsQuarter", "Quarter", "Industry", "IndustryBucket", "Name"}
        keep_cols = [c for c in annual_df.columns if c not in drop_cols]
        rows = []
        for _, r in annual_df[keep_cols].iterrows():
            d = {}
            for k, v in r.items():
                if pd.isna(v): 
                    continue
                if k == "Year":
                    try:
                        d[k] = int(float(v)); continue
                    except Exception:
                        pass
                n = _clean_num(v)
                d[k] = n if n is not None else str(v)
            if d:
                rows.append(d)
    return {"stock": name, "industry": industry or "", "bucket": bucket, "annual_input_rows": rows}

def _quarterly_to_human_json(q_df: pd.DataFrame, *, name: str, industry: str, bucket: str) -> dict:
    if q_df is None or q_df.empty:
        rows = []
    else:
        q = q_df.copy()
        if "Qnum" not in q.columns:
            q["Qnum"] = q["Quarter"].map(_qnum)
        q = q.dropna(subset=["Year", "Qnum"]).sort_values(["Year", "Qnum"])

        drop_cols = {"IsQuarter", "Industry", "IndustryBucket", "Name", "Qnum"}
        keep_cols = [c for c in q.columns if c not in drop_cols]

        rows = []
        for _, r in q[keep_cols].iterrows():
            d = {}
            for k, v in r.items():
                if pd.isna(v):
                    continue
                if k == "Year":
                    try:
                        d[k] = int(float(v)); continue
                    except Exception:
                        pass
                if k == "Quarter":
                    d[k] = str(v).upper().strip(); continue
                n = _clean_num(v)
                d[k] = n if n is not None else str(v)
            if d:
                rows.append(d)
    return {"stock": name, "industry": industry or "", "bucket": bucket, "quarterly_input_rows": rows}

# ---------- Load data ----------

@st.cache_data(show_spinner=False)
def _load_df(_etag: int = 0) -> pd.DataFrame:
    df0 = io_helpers.load_data()
    return df0 if df0 is not None else pd.DataFrame()

df = _load_df(_data_etag())

if st.button("🔄 Refresh data"):
    _load_df.clear()
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

if df is None or df.empty or "Name" not in df.columns:
    st.warning("No data uploaded yet.")
    st.stop()

# Guard columns used by UI
for c in ("IsQuarter", "Quarter", "Industry", "IndustryBucket"):
    if c not in df.columns:
        df[c] = pd.NA

# ---------- Filter bar ----------
st.markdown(section("🔎 Filter", "Find stocks by name, industry, and bucket"), unsafe_allow_html=True)

def _clear_filters():
    st.session_state.update(q_text="", pick_ind="All", pick_bkt="All")
    st.session_state.pop("sel_names", None)

with st.form("filter_form", clear_on_submit=False):
    c1, c2, c3 = st.columns([2, 1.2, 1.2])
    with c1:
        st.text_input("Find stock (name contains…)", key="q_text")
    with c2:
        ind_opts = ["All"] + sorted([x for x in df["Industry"].dropna().astype(str).unique()])
        st.selectbox("Industry", ind_opts, index=0, key="pick_ind")
    with c3:
        bkt_opts = ["All"] + sorted([x for x in df["IndustryBucket"].dropna().astype(str).unique()])
        st.selectbox("Industry Bucket", bkt_opts, index=0, key="pick_bkt")
    bA, bB = st.columns([1, 1])
    bA.form_submit_button("Apply filters")
    bB.form_submit_button("Clear filters", type="secondary", on_click=_clear_filters)

view = df.copy()
_q  = st.session_state.get("q_text", "").strip()
_pi = st.session_state.get("pick_ind", "All")
_pb = st.session_state.get("pick_bkt", "All")

if _q:
    view = view[view["Name"].astype(str).str.contains(_q, case=False, na=False)]
if _pi != "All":
    view = view[view["Industry"].astype(str) == _pi]
if _pb != "All":
    view = view[view["IndustryBucket"].astype(str) == _pb]

names = sorted(view["Name"].dropna().astype(str).unique().tolist())
st.caption(f"Showing **{len(names)}** stock(s).")
if not names:
    st.stop()

# Seed + pick up to 2 stocks (for UI/compare)
seed = [s for s in st.session_state.get("sel_names", []) if s in names][:2] or names[:1]
try:
    sel_names = st.multiselect(
        "Pick up to 2 stocks to analyze / compare",
        options=names,
        default=seed,
        key="sel_names",
        help="Pick one to analyze, two to compare headers side-by-side (UI only).",
        max_selections=2
    )
except TypeError:
    sel_names = st.multiselect(
        "Pick up to 2 stocks to analyze / compare",
        options=names,
        default=seed,
        key="sel_names",
        help="Pick one to analyze, two to compare headers side-by-side (UI only).",
    )

calc_sel_names = sel_names[:2]
name = (calc_sel_names[0] if calc_sel_names else names[0])
safe_name = re.sub(r"[^0-9A-Za-z_]+", "_", str(name))

# ---------- Compare dialog / fallback ----------

def _last_fy(annual_df: pd.DataFrame) -> int | None:
    if annual_df is None or annual_df.empty or "Year" not in annual_df.columns:
        return None
    ys = pd.to_numeric(annual_df["Year"], errors="coerce").dropna()
    return int(ys.max()) if not ys.empty else None

def _render_compare(a: str, b: str):
    """
    Cleaner, wider compare dialog that focuses on TTM ratios only.
    IMPORTANT: No formulas are done here. We only read ratio values already
    present in your ANNUAL rows for the TTM year (defined as last FY + 1).
    """

    # ---------- pull subsets ----------
    sub_a = df[df["Name"] == a].sort_values(["Year", "Quarter"])
    sub_b = df[df["Name"] == b].sort_values(["Year", "Quarter"])
    annual_a, qtr_a = _build_annual_quarter_tables(sub_a)
    annual_b, qtr_b = _build_annual_quarter_tables(sub_b)
    bucket_a = _bucket_for(sub_a); bucket_b = _bucket_for(sub_b)
    price_a  = _current_price(sub_a); price_b = _current_price(sub_b)

    # ---------- quick facts header ----------
    def _facts(sub):
        price = _current_price(sub)
        return [
            ("Current Price", f"{price:,.4f}" if np.isfinite(price) else "—"),
            ("Industry", _industry_label(sub) or "—"),
            ("Bucket", _bucket_for(sub) or "—"),
        ]

    # New, nicer header using the shared helper
    facts_a = _facts(sub_a)
    facts_b = _facts(sub_b)

    # If you later compute TTM label strings (e.g., "TTM 2025") before this,
    # pass them via ttm_left=..., ttm_right=.... For now we just render the header.
    render_compare_header(
        a, b,
        facts_a, facts_b,
        # ttm_left=ttm_tag_a,   # optional, if you compute it above
        # ttm_right=ttm_tag_b,  # optional, if you compute it above
        columns=2
    )
    # Show global FD/EPS rate (applies to all stocks)
    fd_rate  = get_fd_eps_rate()
    epf_rate = get_epf_rate()

    if (fd_rate is not None) or (epf_rate is not None):
        parts = []
        if isinstance(fd_rate, (int, float)) and math.isfinite(fd_rate):
            parts.append(f"FD: {fd_rate:.2f}%")
        if isinstance(epf_rate, (int, float)) and math.isfinite(epf_rate):
            parts.append(f"EPF: {epf_rate:.2f}%")
        subtitle = " · ".join(parts) if parts else "—"
        st.markdown(section("Global Rates", subtitle, "info"), unsafe_allow_html=True)

    # ---------- helpers (scoped here so you can paste this whole block) ----------
    def _last_fy(annual_df: pd.DataFrame):
        if annual_df is None or annual_df.empty or "Year" not in annual_df.columns:
            return None
        ys = pd.to_numeric(annual_df["Year"], errors="coerce").dropna()
        return int(ys.max()) if not ys.empty else None

    def _ttm_ratio_series(annual_df: pd.DataFrame, quarterly_df: pd.DataFrame, *, bucket: str, price_now: float | None):
        """
        Use ONLY the Industry Summary (calculated) TTM column.
        Returns: (Series[label -> value], ttm_label_str like 'TTM 2025')
        """
        # Build Industry Summary (brings in current price for valuation)
        try:
            sum_df = calculations.build_summary_table(
                annual_df=annual_df,
                quarterly_df=quarterly_df,
                bucket=bucket,
                include_ttm=True,
                price_fallback=price_now
            )
        except Exception:
            sum_df = pd.DataFrame()

        if sum_df is None or sum_df.empty:
            return pd.Series(dtype="float64"), "TTM"

        # Find the exact TTM column name (e.g., 'TTM 2025')
        ttm_col = next((c for c in sum_df.columns if isinstance(c, str) and c.upper().startswith("TTM")), None)
        if not ttm_col:
            return pd.Series(dtype="float64"), "TTM"

        # Series of metric -> value for TTM; keep only ratio-like rows
        s = pd.to_numeric(sum_df.set_index("Metric")[ttm_col], errors="coerce")
        s = s[s.index.map(_is_ratio_col_name)]
        return s, ttm_col  # chip will show the exact 'TTM YYYY'

    # ---------- TTM ratio table (no formula) ----------
    st.markdown(
        section(
            "📊 TTM Ratios",
            "Prefers your TTM ratio row if present; otherwise uses calculated TTM from your formulas."
        ),
        unsafe_allow_html=True
    )

    s_a, ttm_tag_a = _ttm_ratio_series(annual_a, qtr_a, bucket=bucket_a, price_now=price_a)
    s_b, ttm_tag_b = _ttm_ratio_series(annual_b, qtr_b, bucket=bucket_b, price_now=price_b)

    # union of ratio labels (case-insensitive sort)
    all_rows = sorted(set(s_a.index.tolist()) | set(s_b.index.tolist()), key=lambda x: x.lower() if isinstance(x, str) else str(x))

    # --- Restrict rows to those defined in Summary config for the buckets in play
    cfg = getattr(config, "INDUSTRY_SUMMARY_RATIOS_CATEGORIES", {}) or {}

    def _allowed_for(bucket_name: str) -> set[str]:
        out = set()
        cats = (cfg.get(bucket_name, {}) or cfg.get("General", {}) or {})
        for items in cats.values():
            if isinstance(items, dict):
                out.update(str(k) for k in items.keys())
        return out

    allowed = _allowed_for(bucket_a) | _allowed_for(bucket_b)
    if allowed:
        all_rows = [r for r in all_rows if str(r) in allowed]
    
    if not all_rows:
        st.info("No TTM ratio values found in your annual data. Add a TTM (next FY) row with ratio columns to see them here.")
        return

    cmp_df = pd.DataFrame({"Ratio": all_rows})
    cmp_df[a] = [s_a.get(r, np.nan) for r in all_rows]
    cmp_df[b] = [s_b.get(r, np.nan) for r in all_rows]

    # format: percent-ish rows with %; others with commas
    percent_mask = [ _is_percentish_label(str(r)) for r in cmp_df["Ratio"].astype(str) ]
    idx_pct = list(np.where(percent_mask)[0])
    idx_num = list(np.where(~np.array(percent_mask))[0])

    sty = cmp_df.style
    try:
        sty = sty.hide(axis="index")
    except Exception:
        pass

    if idx_pct:
        sty = sty.format(lambda v: "—" if pd.isna(v) else f"{float(v):,.2f}%",
                         subset=pd.IndexSlice[idx_pct, [a, b]])
    if idx_num:
        sty = sty.format(lambda v: "—" if pd.isna(v) else f"{float(v):,.2f}",
                         subset=pd.IndexSlice[idx_num, [a, b]])

    # small badges to show which TTM year each side refers to
    st.markdown(
        f'<span class="ttm-chip">{a}: {ttm_tag_a}</span> &nbsp; '
        f'<span class="ttm-chip">{b}: {ttm_tag_b}</span>',
        unsafe_allow_html=True
    )

    st.dataframe(
        sty,
        use_container_width=True,
        height=min(680, _auto_h(cmp_df, row_h=28, base=140)),
    )

# ---------- Compare: button only ----------
if len(calc_sel_names) == 2:
    a, b = calc_sel_names

    if hasattr(st, "dialog"):
        @st.dialog(f"🆚 Compare: {a} vs {b}")
        def _open_compare_dialog():
            _render_compare(a, b)

        st.button("🆚 Compare", type="primary",
                  on_click=_open_compare_dialog, key=f"cmp_open_{a}_{b}")
    else:
        # Fallback for older Streamlit: only render after the button is clicked
        open_key = f"cmp_open_{a}_{b}"
        if st.button("🆚 Compare", type="primary", key=open_key):
            st.session_state[open_key] = True
        if st.session_state.get(open_key):
            with st.expander(f"🆚 Compare: {a} vs {b}", expanded=True):
                _render_compare(a, b)

# ---------- Per-stock UI ----------
with st.expander(name, expanded=True):
    stock = df[df["Name"] == name].sort_values(["Year","Quarter"])
    annual, qtr = _build_annual_quarter_tables(stock)

    bucket    = _bucket_for(stock)
    industry  = _industry_label(stock)
    price_now = _current_price(stock)

    # Header KPIs (display only)
    fd_rate  = get_fd_eps_rate()
    epf_rate = get_epf_rate()

    render_kpi_text_grid([
        ("Current Price", f"{price_now:,.4f}" if np.isfinite(price_now) else "—"),
        ("Industry",      industry or "—"),
        ("Bucket",        bucket or "—"),
        ("FD rate (global)",  f"{fd_rate:,.2f}%"  if isinstance(fd_rate,  (int, float)) and math.isfinite(fd_rate)  else "—"),
        ("EPF rate (global)", f"{epf_rate:,.2f}%" if isinstance(epf_rate, (int, float)) and math.isfinite(epf_rate) else "—"),
    ])

    st.markdown(section("📚 Financial Reports", "Annual & Quarterly data, charts & downloads (raw only)"), unsafe_allow_html=True)
    tabs = st.tabs(["Summary", "Annual", "Quarterly"])

    # ========================= SUMMARY =========================
    with tabs[0]:
        # ---------- helpers specific to the Summary tab ----------
        def _syn_index_for_bucket(bucket_name: str) -> dict:
            """Build a case-insensitive synonym -> canonical label index from your config categories."""
            cats = (getattr(config, "INDUSTRY_SUMMARY_RATIOS_CATEGORIES", {}) or {}).get(bucket_name) \
                or (getattr(config, "INDUSTRY_SUMMARY_RATIOS_CATEGORIES", {}) or {}).get("General") \
                or {}
            idx = {}
            for _cat, items in (cats or {}).items():
                if not isinstance(items, dict): 
                    continue
                for canonical, syns in items.items():
                    key = str(canonical).strip().lower()
                    idx[key] = canonical
                    for s in (syns or []):
                        idx[str(s).strip().lower()] = canonical
                    # also tolerate unitless variants: add/remove (%) and (x)/(×)
                    for v in [
                        canonical.replace(" (%)","").replace(" (x)","").replace(" (×)",""),
                        f"{canonical} (%)" if "(%)" not in canonical else canonical.replace(" (%)",""),
                        f"{canonical} (x)"  if "(x)"  not in canonical and "×" not in canonical else canonical.replace(" (x)",""),
                        f"{canonical} (×)"  if "×"    not in canonical and "(x)" not in canonical else canonical.replace(" (×)",""),
                    ]:
                        idx[str(v).strip().lower()] = canonical
            # add a few common fallbacks used by the KPI menus
            extras = {
                "gross margin": "Gross Margin (%)",
                "net margin": "Net Margin (%)",
                "dividend yield": "Dividend Yield (%)",
                "p/e": "P/E (×)",
                "p/b": "P/B (×)",
                "net debt / ebitda": "Net Debt / EBITDA (×)",
                "interest coverage": "Interest Coverage (×)",
                "receivables days": "Receivable Days",
                "payables days": "Payable Days",
                "eps yoy": "EPS YoY (%)",
                "dividend payout ratio": "Payout Ratio (%)",
                "payout ratio": "Payout Ratio (%)",
            }
            for k, v in extras.items():
                idx[k] = v
            return idx

        def _canonical(label: str, idx: dict) -> str | None:
            if not label: return None
            s = str(label).strip().lower()
            if s in idx: 
                return idx[s]
            # try unit heuristics if not found
            unit_variants = [
                s.replace(" (%)",""), s + " (%)",
                s.replace(" (x)",""),  s + " (x)",
                s.replace(" (×)",""),  s + " (×)",
            ]
            for v in unit_variants:
                if v in idx: 
                    return idx[v]
            return None

        def _ttm_col_name(df: pd.DataFrame) -> str | None:
            """Pick 'TTM' or 'TTM YYYY' gracefully."""
            for c in df.columns:
                if isinstance(c, str) and c.upper().startswith("TTM"):
                    return c
            return "TTM" if "TTM" in df.columns else None

        def _find_row(sum_df: pd.DataFrame, label: str, idx: dict):
            """Find a metric row with synonym + unit tolerance."""
            if sum_df is None or sum_df.empty: 
                return None
            # exact (case-insensitive)
            hit = sum_df[sum_df["Metric"].astype(str).str.lower() == str(label).lower()]
            if not hit.empty: 
                return hit.iloc[0]
            # via canonical from synonyms
            canon = _canonical(label, idx)
            if canon:
                hit = sum_df[sum_df["Metric"].astype(str).str.lower() == canon.lower()]
                if not hit.empty: 
                    return hit.iloc[0]
            return None

        def _fmt_pct(v):
            try:
                f = float(v)
                return "—" if not np.isfinite(f) else f"{f:,.2f}%"
            except Exception:
                return "—"

        def _fmt_num(v):
            try:
                f = float(v)
                return "—" if not np.isfinite(f) else f"{f:,.2f}"
            except Exception:
                return "—"

        def _years_in_summary(df: pd.DataFrame) -> list[int]:
            if df is None or df.empty: return []
            ys = [c for c in df.columns if isinstance(c, (int, np.integer))]
            return sorted(ys)

        # ---------- build calculated summary table (uses current price for valuation ratios) ----------
        st.markdown(
            section("📊 Industry Summary (calculated)",
                    "Per-bucket formulas from calculations.py; shows FY columns + TTM.",
                    "success"),
            unsafe_allow_html=True
        )

        syn_idx = _syn_index_for_bucket(bucket)  # resolves menu labels -> summary labels

        sum_df = pd.DataFrame()
        if hasattr(calculations, "build_summary_table"):
            try:
                sum_df = calculations.build_summary_table(
                    annual_df=annual,
                    quarterly_df=qtr,
                    bucket=bucket,
                    include_ttm=True,
                    price_fallback=price_now,   # <- ensures P/E, DY, FCF Yield, etc. use current price
                )
            except Exception:
                sum_df = pd.DataFrame()
                
        save_view_snapshot(name, bucket, sum_df)        

        if sum_df is None or sum_df.empty:
            st.info("No calculable metrics for this bucket/data yet (showing structure only).")
            comp = _ratio_table_structure(annual, bucket)
            if comp is not None and not comp.empty:
                comp_show = _for_streamlit(comp)  # normalize column labels to strings (UI-only)
                try:
                    st.dataframe(
                        _style_structured_ratio_table(comp_show),
                        use_container_width=True,
                        height=_auto_h(comp_show),
                    )
                except Exception:
                    st.dataframe(comp_show, use_container_width=True, height=_auto_h(comp_show))

        else:
            # ---- Always show full structure, then fill in computed values ----
            base = _ratio_table_structure(annual, bucket)     # all labels (rows) + FYs + TTM col name
            ttm_tag = _ttm_label_from_annual(annual)
            vals = sum_df.copy()

            # Normalize the TTM column name to match the structure ("TTM" or "TTM YYYY")
            old_ttm = _ttm_col_name(vals)
            if old_ttm and old_ttm != ttm_tag and old_ttm in vals.columns:
                vals = vals.rename(columns={old_ttm: ttm_tag})

            # Keep only columns that exist in the structure (Metric + periods)
            keep_cols = ["Metric"] + [c for c in base.columns if c != "Metric"]

            # Coerce "2023" -> 2023 if needed so it matches vals' int columns
            kc_norm = ["Metric"]
            for c in keep_cols[1:]:
                if isinstance(c, str) and c.isdigit() and (int(c) in vals.columns):
                    kc_norm.append(int(c))
                else:
                    kc_norm.append(c)

            vals = vals[[c for c in kc_norm if c in vals.columns]]

            # (removed) no EPF YoY injection here — keep it out of Industry Summary

            # Fill base with available values from vals (row-wise by Metric)
            disp = base.copy()
            vmap = vals.set_index("Metric")
            for i, m in disp["Metric"].items():
                if isinstance(m, str) and m.strip().startswith("—"):
                    continue  # category separator row
                if m in vmap.index:
                    for c in keep_cols[1:]:
                        if c in vmap.columns:
                            disp.at[i, c] = vmap.at[m, c]

            # Optional: toggle to hide rows that are completely blank (keep category headers)
            hide_empty = st.toggle("Hide metrics with no values", value=False, key=f"sum_hide_empty_{safe_name}")
            if hide_empty:
                mask_vals = disp.drop(columns=["Metric"]).apply(
                    lambda r: pd.to_numeric(r, errors="coerce").notna().any(), axis=1
                )
                is_cat = disp["Metric"].astype(str).str.match(r"^\s*— .* —\s*$")
                disp = disp[mask_vals | is_cat]

            # Style + show
            disp_show = _for_streamlit(disp)  # <-- normalize column labels for display only
            sty = _style_summary_with_categories(disp_show)
            st.dataframe(sty, use_container_width=True, height=_auto_h(disp_show, row_h=28, base=140))

        # ===== Soldier Worm: Diagnostics (Missing + Trace) =====
        try:
            warn_df = calculations.build_soldier_worm_report(
                annual_df=annual, quarterly_df=qtr, bucket=bucket, include_ttm=True, price_fallback=price_now
            )
        except Exception:
            warn_df = pd.DataFrame(columns=["Category","Metric","Period","Missing Inputs"])

        try:
            trace_df = calculations.build_soldier_worm_calc_trace(
                annual_df=annual, quarterly_df=qtr, bucket=bucket, include_ttm=True, price_fallback=price_now
            )
        except Exception:
            trace_df = pd.DataFrame()

        miss_cnt  = 0 if warn_df is None else len(warn_df)
        trace_cnt = 0 if trace_df is None else len(trace_df)

        with st.expander(f"Show diagnostics  —  missing: {miss_cnt} · trace rows: {trace_cnt}", expanded=False):
            # ----- Missing Inputs -----
            st.subheader("Missing Inputs", divider=True)
            colA, colB = st.columns([1.2, 1])
            with colA:
                show = st.selectbox("Show", ["All","Only TTM","Only FY"], index=0, key=f"sw_scope_{safe_name}")
            with colB:
                q_search = st.text_input("Filter by metric text…", key=f"sw_q_{safe_name}")

            view_w = warn_df.copy()
            if not view_w.empty:
                if show == "Only TTM":
                    view_w = view_w[view_w["Period"].astype(str).str.upper() == "TTM"]
                elif show == "Only FY":
                    view_w = view_w[view_w["Period"].astype(str).str.upper() != "TTM"]
                if q_search.strip():
                    s = q_search.strip().lower()
                    view_w = view_w[
                        view_w["Metric"].astype(str).str.lower().str.contains(s)
                        | view_w["Missing Inputs"].astype(str).str.lower().str.contains(s)
                    ]

            if view_w.empty:
                st.caption("No missing inputs detected for the current Summary setup.")
            else:
                try:
                    sty_sw = view_w.style.hide(axis="index")
                except Exception:
                    sty_sw = view_w.style
                st.dataframe(sty_sw, use_container_width=True, height=_auto_h(view_w, row_h=26, base=110))

            # ----- Calc Trace -----
            st.subheader("Calc Trace", divider=True)
            colA, colB, colC = st.columns([1, 1, 1])
            with colA:
                scope = st.selectbox("Show", ["All","Only TTM","Only FY"], index=0, key=f"sw_trace_scope_{safe_name}")
            with colB:
                only_sus = st.toggle("Show suspicious only (flags/noted)", value=False, key=f"sw_trace_sus_{safe_name}")
            with colC:
                q = st.text_input("Filter (metric / inputs / path / flags)…", key=f"sw_trace_q_{safe_name}")

            view_t = trace_df.copy()
            if not view_t.empty:
                if scope == "Only TTM":
                    view_t = view_t[view_t["Period"].astype(str).str.upper() == "TTM"]
                elif scope == "Only FY":
                    p = view_t["Period"].astype(str).str.upper()
                    view_t = view_t[(p == "FY") | (p.str.startswith("FY "))]
                if only_sus:
                    sus = (view_t["Flags"].astype(str) != "") | (view_t["Note"].astype(str) != "")
                    view_t = view_t[sus]
                if q.strip():
                    s = q.strip().lower()
                    view_t = view_t[
                        view_t["Metric"].astype(str).str.lower().str.contains(s)
                        | view_t["Inputs"].astype(str).str.lower().str.contains(s)
                        | view_t["Path Used"].astype(str).str.lower().str.contains(s)
                        | view_t.get("Formula","").astype(str).str.lower().str.contains(s)
                        | view_t.get("Flags","").astype(str).str.lower().str.contains(s)
                    ]

            if view_t.empty:
                st.caption("No calc trace yet (not enough data).")
            else:
                pct_mask = view_t["Metric"].astype(str).str.contains(r"\(%\)|margin|yield| ratio", case=False, regex=True)
                try:
                    sty = view_t.style.hide(axis="index")
                except Exception:
                    sty = view_t.style

                sty = sty.format(lambda v: "—" if pd.isna(v) else f"{float(v):,.2f}%", subset=pd.IndexSlice[pct_mask, ["Value"]])
                sty = sty.format(lambda v: "—" if pd.isna(v) else f"{float(v):,.2f}",  subset=pd.IndexSlice[~pct_mask, ["Value"]])

                def _row_style(r):
                    base = [""] * len(r)
                    has_note = str(r.get("Note","")).strip() != ""
                    flags    = str(r.get("Flags","")).lower()
                    def set_all(css): return [css] * len(r)
                    if "fallback" in flags: return set_all("background-color:#fff7ed;")
                    if "prev fy" in flags:  return set_all("background-color:#f1f5f9;")
                    if "outlier" in flags:  return set_all("background-color:#fee2e2;")
                    if has_note:            return set_all("background-color:#fef9c3;")
                    return base
                sty = sty.apply(_row_style, axis=1)

                st.dataframe(sty, use_container_width=True, height=_auto_h(view_t, row_h=26, base=110))

            # --- Download: Industry Summary (JSON, year-first) ---
            try:
                payload_yearfirst = _summary_results_by_year_json(
                    sum_df, name=name, industry=industry, bucket=bucket,
                    # if you want ONLY valuation stuff, uncomment the next line:
                    # keep_metrics=["P/E (×)", "P/B (×)", "EV/EBITDA (×)", "EV/Sales (×)", "Dividend Yield (%)"]
                )
                st.download_button(
                    label="⬇️ Download Industry Summary (JSON, year-first)",
                    data=(__import__("json").dumps(payload_yearfirst, indent=2, ensure_ascii=False)).encode("utf-8"),
                    file_name=f"{name}_summary_by_year.json",
                    mime="application/json",
                    use_container_width=True,
                    key=f"dl_summary_yearfirst_{safe_name}",
                )
            except Exception:
                pass

        # ---------- raw TTM fallback for cash-flow + basic absolutes (uses current price) ----------
        def _ttm_raw_from_quarters():
            # 1) central calculator if available
            if hasattr(calculations, "ttm_raw_row_from_quarters"):
                try:
                    d = calculations.ttm_raw_row_from_quarters(qtr, current_price=price_now) or {}
                    return d
                except Exception:
                    pass
            # 2) fallback: sum last 4 quarters
            if qtr is None or qtr.empty:
                return {}
            qx = qtr.copy()
            if "Qnum" not in qx.columns:
                qx["Qnum"] = qx["Quarter"].map(_qnum)
            qx = qx.dropna(subset=["Year","Qnum"]).sort_values(["Year","Qnum"])
            tail = qx.tail(min(4, len(qx)))

            def _sum(*cands):
                for c in cands:
                    if c in tail.columns:
                        return pd.to_numeric(tail[c], errors="coerce").sum(skipna=True)
                return np.nan
            def _last(*cands):
                for c in cands:
                    if c in tail.columns:
                        s = pd.to_numeric(tail[c], errors="coerce").dropna()
                        if not s.empty: return float(s.iloc[-1])
                return np.nan

            out = {
                "Revenue": _sum("Q_Revenue","Q_Total Revenue","Q_TotalRevenue","Q_Sales"),
                "Gross Profit": _sum("Q_Gross Profit","Q_GrossProfit"),
                "Operating Profit": _sum("Q_Operating Profit","Q_EBIT","Q_OperatingIncome"),
                "EBITDA": _sum("Q_EBITDA"),
                "Net Profit": _sum("Q_Net Profit","Q_NPAT","Q_Profit After Tax","Q_Profit attributable to owners"),
                "DPS": _sum("Q_DPS"),
                "CFO": _sum("Q_CFO","Q_Cash Flow from Ops","Q_Cash from Operations","Q_Operating Cash Flow"),
                "Capex": _sum("Q_Capex","Q_Capital Expenditure"),
                "Shares": _last("Q_Shares","Q_NumShares","Q_Units"),
                "Price": _last("Q_EndQuarterPrice","Q_Price","Q_SharePrice"),
            }
            # EPS TTM if possible
            try:
                if np.isfinite(out.get("Net Profit", np.nan)) and np.isfinite(out.get("Shares", np.nan)):
                    out["EPS"] = float(out["Net Profit"]) / float(out["Shares"])
            except Exception:
                pass
            # use current price as final fallback for price
            if not (isinstance(out.get("Price"), (int,float)) and np.isfinite(out["Price"])):
                out["Price"] = price_now
            return out

        ttm_raw = _ttm_raw_from_quarters()
        ttm_col = _ttm_col_name(sum_df) if (sum_df is not None and not sum_df.empty) else None

        # ---------- TTM KPI cards (synonym-resolved + raw fallback) ----------
        st.markdown(
            section(
                "⏱️ TTM KPIs",
                "Resolved from calculator output (TTM column) — falls back to raw TTM where sensible"
            ),
            unsafe_allow_html=True
        )

        def _menu_for(bucket_name: str, menu: dict[str, list[str]]) -> list[str]:
            try:
                return (menu or {}).get(bucket_name) or (menu or {}).get("General") or []
            except Exception:
                return []

        def _canonize_label(lbl: str) -> str:
            # prefer canonical label from your synonym index; fall back to itself
            return (_canonical(lbl, syn_idx) or lbl).strip()

        def _dedupe_ttm_labels(items: list[str]) -> list[str]:
            seen = set()
            out: list[str] = []
            for lab in (items or []):
                can = _canonize_label(lab)
                key = can.lower()
                if key in seen:
                    continue
                out.append(can)
                seen.add(key)
            # don't auto-inject any extras; use only what's in config
            return out

        ttm_menu  = getattr(config, "TTM_METRICS_BY_BUCKET", {}) or {}
        ttm_items = _dedupe_ttm_labels(_menu_for(bucket, ttm_menu))


        ttm_pairs = []
        for label in ttm_items:
            row = _find_row(sum_df, label, syn_idx) if (sum_df is not None and not sum_df.empty) else None
            val = np.nan
            if row is not None and ttm_col in row:
                val = row[ttm_col]
            else:
                # fall back to raw TTM for basic absolutes / price-derived items
                L = label.strip().lower()
                if L in ("revenue","gross profit","operating profit","ebitda","net profit","eps","dps"):
                    val = ttm_raw.get(label)
                elif L == "dividend yield" and np.isfinite(ttm_raw.get("DPS", np.nan)) and np.isfinite(ttm_raw.get("Price", np.nan)):
                    val = (float(ttm_raw["DPS"]) / float(ttm_raw["Price"])) * 100.0
                elif L == "p/e" and np.isfinite(ttm_raw.get("EPS", np.nan)) and np.isfinite(ttm_raw.get("Price", np.nan)) and float(ttm_raw["EPS"]) != 0:
                    val = float(ttm_raw["Price"]) / float(ttm_raw["EPS"])
                # (others like ROE/ROA/PB/ND/EBITDA need balance sheet items; leave as NaN if summary didn’t provide)

            if any([
                "(%)" in str(_canonical(label, syn_idx) or label).lower(),
                "margin" in label.lower(),
                "yield" in label.lower(),
                " ratio" in label.lower(),
                "yoy" in label.lower(),
            ]):
                ttm_pairs.append((label, _fmt_pct(val)))
            else:
                ttm_pairs.append((label, _fmt_num(val)))

        render_kpi_text_grid(ttm_pairs)
        
        # --- SYNC: TTM KPI card values (numeric) for Decision page ---
        ttm_values = {}
        period_lbl = (ttm_col or _ttm_label_from_annual(annual) or "TTM")

        for label in (ttm_items or []):
            # Recreate the numeric (unformatted) value exactly like the card did
            row = _find_row(sum_df, label, syn_idx) if (isinstance(sum_df, pd.DataFrame) and not sum_df.empty) else None
            val = None
            if row is not None and (ttm_col in row) and pd.notna(row[ttm_col]):
                try:
                    val = float(row[ttm_col])
                except Exception:
                    val = None
            else:
                L = label.strip().lower()
                if L in ("revenue","gross profit","operating profit","ebitda","net profit","eps","dps"):
                    val = ttm_raw.get(label)
                elif L == "dividend yield" and np.isfinite(ttm_raw.get("DPS", np.nan)) and np.isfinite(ttm_raw.get("Price", np.nan)):
                    val = (float(ttm_raw["DPS"]) / float(ttm_raw["Price"])) * 100.0
                elif L == "p/e" and np.isfinite(ttm_raw.get("EPS", np.nan)) and np.isfinite(ttm_raw.get("Price", np.nan)) and float(ttm_raw["EPS"]) != 0:
                    val = float(ttm_raw["Price"]) / float(ttm_raw["EPS"])

                elif L in ("eps yoy", "eps yoy (%)", "eps yoy%"):
                    # Prefer Summary EPS (TTM + latest FY), otherwise raw TTM + last FY annual.
                    ttm_eps, last_eps = None, None
                    eps_row = _find_row(sum_df, "EPS", syn_idx) if (sum_df is not None and not sum_df.empty) else None
                    if isinstance(eps_row, pd.Series) and (ttm_col in eps_row.index):
                        ttm_eps = eps_row.get(ttm_col)
                        fy_cols = [c for c in eps_row.index if isinstance(c, (int, np.integer))]
                        if fy_cols:
                            last_eps = eps_row.get(max(fy_cols))

                    if (ttm_eps is None or not np.isfinite(float(ttm_eps))) and isinstance(ttm_raw, dict):
                        ttm_eps = ttm_raw.get("EPS")

                    if (last_eps is None or not np.isfinite(float(last_eps))) and isinstance(annual, pd.DataFrame):
                        if "EPS" in annual.columns:
                            s = pd.to_numeric(annual["EPS"], errors="coerce").dropna()
                            if not s.empty:
                                last_eps = float(s.iloc[-1])
                        elif "Net Profit" in annual.columns and "Shares" in annual.columns:
                            np_s = pd.to_numeric(annual["Net Profit"], errors="coerce").dropna()
                            sh_s = pd.to_numeric(annual["Shares"], errors="coerce").dropna()
                            if not np_s.empty and not sh_s.empty:
                                last_eps = float(np_s.iloc[-1]) / float(sh_s.iloc[-1])

                    try:
                        t, p = float(ttm_eps), float(last_eps)
                        val = ((t / p) - 1.0) * 100.0 if np.isfinite(t) and np.isfinite(p) and p != 0 else np.nan
                    except Exception:
                        val = np.nan

                elif L in ("payout ratio", "payout ratio (%)", "dividend payout ratio", "dividend payout ratio (%)"):
                    # Prefer Summary "Payout Ratio (%)"; otherwise compute DPS / EPS from TTM.
                    pr_row = _find_row(sum_df, "Payout Ratio (%)", syn_idx) if (sum_df is not None and not sum_df.empty) else None
                    if isinstance(pr_row, pd.Series) and (ttm_col in pr_row.index) and pd.notna(pr_row.get(ttm_col)):
                        val = pr_row.get(ttm_col)
                    else:
                        dps, eps = None, None
                        if isinstance(sum_df, pd.DataFrame) and not sum_df.empty:
                            dps_row = _find_row(sum_df, "DPS", syn_idx)
                            eps_row = _find_row(sum_df, "EPS", syn_idx)
                            if isinstance(dps_row, pd.Series) and (ttm_col in dps_row.index):
                                dps = dps_row.get(ttm_col)
                            if isinstance(eps_row, pd.Series) and (ttm_col in eps_row.index):
                                eps = eps_row.get(ttm_col)
                        if (dps is None or not np.isfinite(float(dps))) and isinstance(ttm_raw, dict):
                            dps = ttm_raw.get("DPS")
                        if (eps is None or not np.isfinite(float(eps))) and isinstance(ttm_raw, dict):
                            eps = ttm_raw.get("EPS")
                        try:
                            d, e = float(dps), float(eps)
                            val = (d / e) * 100.0 if np.isfinite(d) and np.isfinite(e) and e != 0 else np.nan
                        except Exception:
                            val = np.nan

            # Canonicalize label so Decision can resolve synonyms
            canon = _canonical(label, syn_idx) or label
            try:
                v = float(val) if val is not None and np.isfinite(float(val)) else None
            except Exception:
                v = None
            ttm_values[str(canon)] = v

        st.session_state.setdefault("TTM_KPI_SYNC", {})[name] = {
            "period": period_lbl,
            "values": ttm_values
        }


        # --- Soldier Worm for TTM KPI cards ---
        try:
            ttm_warn = calculations.build_soldier_worm_ttm_kpis(
                sum_df=sum_df, annual_df=annual, quarterly_df=qtr,
                bucket=bucket, labels=ttm_items, price_fallback=price_now
            )
        except Exception:
            ttm_warn = pd.DataFrame(columns=["Metric","Period","Missing Inputs"])

        try:
            ttm_trace = calculations.build_soldier_worm_ttm_kpis_trace(
                sum_df=sum_df, annual_df=annual, quarterly_df=qtr,
                bucket=bucket, labels=ttm_items, price_fallback=price_now
            )
        except Exception:
            ttm_trace = pd.DataFrame()

        # TTM KPI — Diagnostics (Missing + Trace)
        ttm_miss_cnt  = 0 if ttm_warn is None else len(ttm_warn)
        ttm_trace_cnt = 0 if ttm_trace is None else len(ttm_trace)

        with st.expander(f"⏱️ TTM KPI — Diagnostics  —  missing: {ttm_miss_cnt} · trace rows: {ttm_trace_cnt}", expanded=False):
            st.subheader("Missing Inputs", divider=True)
            if ttm_warn is None or ttm_warn.empty:
                st.caption("No missing inputs for selected TTM KPIs.")
            else:
                try:
                    st.dataframe(ttm_warn.style.hide(axis="index"),
                                use_container_width=True,
                                height=_auto_h(ttm_warn, row_h=26, base=110))
                except Exception:
                    st.dataframe(ttm_warn, use_container_width=True,
                                height=_auto_h(ttm_warn, row_h=26, base=110))

            st.subheader("Calc Trace", divider=True)
            if ttm_trace is None or ttm_trace.empty:
                st.caption("No calc trace available for selected TTM KPIs.")
            else:
                qq = st.text_input("Filter TTM calc trace…", key=f"ttm_trace_q_{safe_name}")
                vt = ttm_trace.copy()
                if qq.strip():
                    s = qq.strip().lower()
                    vt = vt[
                        vt["Metric"].astype(str).str.lower().str.contains(s)
                        | vt["Inputs"].astype(str).str.lower().str.contains(s)
                        | vt["Path Used"].astype(str).str.lower().str.contains(s)
                        | vt.get("Formula","").astype(str).str.lower().str.contains(s)
                    ]
                try:
                    st.dataframe(vt.style.hide(axis="index"), use_container_width=True,
                                height=_auto_h(vt, row_h=26, base=110))
                except Exception:
                    st.dataframe(vt, use_container_width=True,
                                height=_auto_h(vt, row_h=26, base=110))
            
            # --- Download: TTM KPI (JSON, period-first) ---
            try:
                period_lbl = (ttm_col or _ttm_label_from_annual(annual) or "TTM")
                payload_ttm_yf = _ttm_kpis_to_yearfirst_json(
                    labels=ttm_items, sum_df=sum_df, ttm_col=ttm_col,
                    ttm_raw=ttm_raw, syn_idx=syn_idx, price_now=price_now,
                    period_label=period_lbl,
                    name=name, industry=industry, bucket=bucket,
                    annual_df=annual  # <-- add this
                )

                st.download_button(
                    label="⬇️ Download TTM KPI (JSON, period-first)",
                    data=(__import__("json").dumps(payload_ttm_yf, indent=2, ensure_ascii=False)).encode("utf-8"),
                    file_name=f"{name}_ttm_kpi_periodfirst.json",
                    mime="application/json",
                    use_container_width=True,
                    key=f"dl_ttm_yearfirst_{safe_name}",
                )
            except Exception:
                pass

        # ---------- CAGR KPI cards (with FY/TTM end toggle + 3y/5y window) ----------
        st.markdown(
            section("📈 CAGR KPIs", "FY or TTM end-point + choose 3y/5y window"),
            unsafe_allow_html=True
        )
        cagr_menu  = getattr(config, "CAGR_METRICS_BY_BUCKET", {}) or {}

        def _menu_for(bucket_name: str, menu: dict[str, list[str]]) -> list[str]:
            try:
                return (menu or {}).get(bucket_name) or (menu or {}).get("General") or []
            except Exception:
                return []

        cagr_items = _menu_for(bucket, cagr_menu)

        # UI toggles
        DEFAULT_CAGR_WIN = 5
        use_ttm = st.toggle("Use TTM as end point", value=True, key=f"cagr_use_ttm_{safe_name}")
        if hasattr(st, "segmented_control"):
            win = st.segmented_control("Window", options=[3,5], selection_mode="single",
                                    default=DEFAULT_CAGR_WIN, key=f"cagr_win_{safe_name}")
        else:
            win = st.radio("Window", options=[3,5], index=1, horizontal=True, key=f"cagr_win_{safe_name}")
        end_basis = "TTM" if use_ttm else "FY"
        ttm_col = _ttm_col_name(sum_df) if (sum_df is not None and not sum_df.empty) else None

        # --- PASTE HERE — CAGR window normalizer (ensures value is 3 or 5) ---
        _raw_win = st.session_state.get(f"cagr_win_{safe_name}", DEFAULT_CAGR_WIN)
        try:
            _w = int(float(_raw_win))
            if _w not in (3, 5):
                _w = DEFAULT_CAGR_WIN
        except Exception:
            _w = DEFAULT_CAGR_WIN
        win = _w  # don't write back to session_state after the widget exists
        # ---------------------------------------------------------------------

        # helpers
        def _years_in_summary(df: pd.DataFrame) -> list[int]:
            if df is None or df.empty: return []
            ys = [c for c in df.columns if isinstance(c, (int, np.integer))]
            return sorted(ys)

        def _cagr_from_summary_row(row: pd.Series, years: list[int], N: int,
                                end_basis: str, ttm_col: str | None) -> float | None:
            """
            Correct window rules:
            • FY end: need N+1 FY years, start = years[-(N+1)], end = value at last FY
            • TTM end: need N FY years,   start = years[-N],     end = value at TTM column
            No window shrinking. If inputs are missing, return None.
            """
            if row is None or not years or len(years) < 2:
                return None

            yrs = sorted([int(y) for y in years])
            EB  = str(end_basis).upper()
            N   = int(N)

            last_fy = int(yrs[-1])
            if EB == "TTM":
                # must have TTM value present
                if not (ttm_col and (ttm_col in row) and pd.notna(row.get(ttm_col))):
                    return None
                # must have at least N FY years
                if len(yrs) < N:
                    return None
                y0   = yrs[-N]
                v0   = pd.to_numeric(pd.Series([row.get(y0)]), errors="coerce").iloc[0]
                vEnd = pd.to_numeric(pd.Series([row.get(ttm_col)]), errors="coerce").iloc[0]
            else:
                # FY end needs N+1 FY years
                if len(yrs) <= N:
                    return None
                y0   = yrs[-(N+1)]
                v0   = pd.to_numeric(pd.Series([row.get(y0)]), errors="coerce").iloc[0]
                vEnd = pd.to_numeric(pd.Series([row.get(last_fy)]), errors="coerce").iloc[0]

            try:
                v0 = float(v0); vEnd = float(vEnd)
                if not (np.isfinite(v0) and np.isfinite(vEnd)) or v0 <= 0 or vEnd <= 0:
                    return None
                return (vEnd / v0) ** (1.0 / N) - 1.0
            except Exception:
                return None


        def _base_label_for_cagr(name: str) -> str:
            base = str(name).replace(" CAGR","").strip()
            mapping = {
                "Operating Cash Flow": "CFO",
                "Free Cash Flow": "FCF",
                "NAV per Unit": "NAV per Unit",
                "Orderbook": "Orderbook",
                "Gross Loans": "Gross Loans",
                "Deposits": "Deposits",
            }
            return mapping.get(base, base)

        # CFO/FCF fallback (annual + optional TTM end override)
        def _series_from_annual(label: str) -> pd.Series | None:
            if annual is None or annual.empty or "Year" not in annual.columns:
                return None

            def _build_series(years_raw, values) -> pd.Series | None:
                data: dict[int, float] = {}
                for raw_year, raw_value in zip(years_raw, values):
                    if pd.isna(raw_year):
                        continue
                    try:
                        year_int = int(float(raw_year))
                    except (TypeError, ValueError):
                        continue

                    if pd.isna(raw_value):
                        data[year_int] = np.nan
                        continue

                    try:
                        data[year_int] = float(raw_value)
                    except (TypeError, ValueError):
                        continue

                return pd.Series(data) if data else None

            # build a year->value series from raw columns
            cand_map = {
                "Revenue": ["Revenue","Total Revenue","TotalRevenue","Sales"],
                "EBITDA": ["EBITDA"],
                "Net Profit": ["Net Profit","NPAT","Profit After Tax","PATAMI","Net Income","Profit attributable to owners"],
                "CFO": ["CFO","Cash Flow from Ops","Cash from Operations","Operating Cash Flow"],
                "Orderbook": ["Orderbook","Order Book","Unbilled Sales"],
                "Store Count": ["Store Count"],
                "ARR": ["ARR"],
                "AUM": ["AUM"],
            }

            years_raw = pd.to_numeric(annual["Year"], errors="coerce")

            if label == "FCF":
                if "CFO" in annual.columns and ("Capex" in annual.columns or "Capital Expenditure" in annual.columns):
                    cfo = pd.to_numeric(annual.get("CFO"), errors="coerce")
                    cap = pd.to_numeric(annual.get("Capex", annual.get("Capital Expenditure")), errors="coerce")
                    if cfo.notna().sum() >= 2 and cap.notna().sum() >= 2:
                        fcf = cfo - cap.abs()
                        return _build_series(years_raw, fcf)                        
                return None
            if label in cand_map:
                for k in cand_map[label]:
                    if k in annual.columns:
                        s = pd.to_numeric(annual[k], errors="coerce")
                        if s.notna().sum() >= 2:
                            years_raw = pd.to_numeric(annual["Year"], errors="coerce")
                            series = _build_series(years_raw, s)
                            if series is not None:
                                return series
            return None

        def _cagr_from_annual_series(label: str, N: int, end_basis: str,
                                    ttm_override: float | None) -> float | None:
            # Build year->value series from annual
            s = _series_from_annual(label)
            if s is None or s.dropna().size < 2:
                return None

            # Determine FY universe (ignore synthetic next-FY TTM rows)
            yrs_all = sorted([int(y) for y in s.index if isinstance(y, (int, np.integer))])
            if annual is None or annual.empty or "Year" not in annual.columns:
                return None
            ys_fy = pd.to_numeric(annual["Year"], errors="coerce").dropna().astype(int)
            if ys_fy.empty:
                return None
            last_fy = int(ys_fy.max())
            # if a synthetic TTM year (last_fy+1) exists in the series, it's fine; we anchor on last_fy

            EB = str(end_basis).upper()
            N = int(N)

            # Availability check: require N FY columns for TTM; N+1 for FY
            fy_years_present = [y for y in yrs_all if y <= last_fy]
            if EB == "TTM":
                if len(fy_years_present) < N:
                    return None
                start_year = last_fy - (N - 1)
                v0 = s.get(start_year, np.nan)
                vE = float(ttm_override) if (ttm_override is not None) else np.nan
            else:
                if len(fy_years_present) <= N:
                    return None
                start_year = last_fy - N
                v0 = s.get(start_year, np.nan)
                vE = s.get(last_fy, np.nan)

            if not (pd.notna(v0) and pd.notna(vE)) or v0 <= 0 or vE <= 0:
                return None
            try:
                return (float(vE) / float(v0)) ** (1.0 / N) - 1.0
            except Exception:
                return None

        # --- TTM overrides for end-point values (so TTM CAGR works for non-CF too)
        ttm_overrides = {
            "Revenue":    (ttm_raw.get("Revenue")    if isinstance(ttm_raw, dict) else None),
            "EBITDA":     (ttm_raw.get("EBITDA")     if isinstance(ttm_raw, dict) else None),
            "Net Profit": (ttm_raw.get("Net Profit") if isinstance(ttm_raw, dict) else None),
            "CFO":        (ttm_raw.get("CFO")        if isinstance(ttm_raw, dict) else None),
            "FCF":        (((ttm_raw.get("CFO") or 0.0) - abs(ttm_raw.get("Capex") or 0.0)) if isinstance(ttm_raw, dict) else None),
        }

        def _last_fy_value(label: str) -> float | None:
            s = _series_from_annual(label)
            if s is None or s.dropna().empty:
                return None
            yrs = sorted(int(y) for y in s.index if isinstance(y, (int, np.integer)))
            return float(s.get(yrs[-1])) if yrs else None

        years = _years_in_summary(sum_df) if (sum_df is not None and not sum_df.empty) else []
        try:
            N = int(win if win is not None else DEFAULT_CAGR_WIN)
        except (TypeError, ValueError):
            N = DEFAULT_CAGR_WIN

        # ---- collect & show CAGR cards, and cache raw numbers for Snowflake sync ----
        cagr_pairs = []
        cagr_sync  = {}   # base label -> numeric percent (not formatted)

        for item in cagr_items:
            if not item.lower().endswith(" cagr"):
                continue

            base_label = _base_label_for_cagr(item)  # e.g., "Revenue", "CFO", "FCF", "Net Profit"
            g = None

            # 1) Try via Summary rows
            row = _find_row(sum_df, base_label, syn_idx) if (sum_df is not None and not sum_df.empty) else None
            if isinstance(row, pd.Series) and years:
                g = _cagr_from_summary_row(row, years, N, end_basis, ttm_col)

            # 2) Fall back to annual series with a TTM end override (if needed)
            if g is None:
                override = None
                if end_basis == "TTM":
                    override = ttm_overrides.get(base_label)
                    if override is None and base_label in ("Orderbook","Store Count","ARR","AUM","NAV per Unit"):
                        override = _last_fy_value(base_label)
                g = _cagr_from_annual_series(base_label, N, end_basis, override)

            # card display (format as %)
            cagr_pairs.append((item, _fmt_pct((g * 100.0) if g is not None else np.nan)))

            # cache raw numeric % for Snowflake
            cagr_sync[base_label] = (None if g is None else float(g * 100.0))

        # PEG / MOS (unchanged below) ...
        try:
            peg = calculations.graham_peg(sum_df, years_back=N, end_basis=end_basis)
            cagr_pairs.append(("PEG (Graham)", _fmt_num(peg)))
        except Exception:
            cagr_pairs.append(("PEG (Graham)", "—"))

        try:
            mos = calculations.margin_of_safety_pct(sum_df, years_back=N, end_basis=end_basis, price_fallback=price_now)
            cagr_pairs.append(("Margin of Safety", _fmt_pct(mos)))
        except Exception:
            cagr_pairs.append(("Margin of Safety", "—"))

        render_kpi_text_grid(cagr_pairs)

        # <-- NEW: publish this panel's settings/results for Snowflake
        st.session_state.setdefault("CAGR_SYNC", {})[name] = {
            "N": int(N),
            "end_basis": str(end_basis).upper(),   # "TTM" or "FY"
            "values_pct": cagr_sync                # { "Revenue": 12.3, "CFO": 9.8, ... } in %
        }

        # --- Soldier Worm for CAGR KPIs (uses same window/end toggles) ---
        try:
            cagr_list = [x for x in (cagr_items or []) if x.lower().endswith(" cagr")] + ["PEG (Graham)", "Margin of Safety"]
            cagr_warn = calculations.build_soldier_worm_cagr_kpis(
                sum_df=sum_df, annual_df=annual, quarterly_df=qtr,
                bucket=bucket, labels=cagr_list, years_back=N, end_basis=end_basis, price_fallback=price_now
            )
        except Exception:
            cagr_warn = pd.DataFrame(columns=["Metric","Period","Missing Inputs"])

        try:
            cagr_trace = calculations.build_soldier_worm_cagr_calc_trace(
                sum_df=sum_df, annual_df=annual, quarterly_df=qtr,
                bucket=bucket, labels=[x for x in (cagr_items or []) if x.lower().endswith(" cagr")],
                years_back=N, end_basis=end_basis, price_fallback=price_now
            )
        except Exception:
            cagr_trace = pd.DataFrame()

        # CAGR / MOS / PEG — Diagnostics (Missing + Trace)
        cg_miss_cnt  = 0 if cagr_warn is None else len(cagr_warn)
        cg_trace_cnt = 0 if cagr_trace is None else len(cagr_trace)

        with st.expander(f"📈 CAGR / MOS / PEG — Diagnostics  —  missing: {cg_miss_cnt} · trace rows: {cg_trace_cnt}", expanded=False):
            st.subheader("Missing Inputs", divider=True)
            if cagr_warn is None or cagr_warn.empty:
                st.caption("No missing inputs for the chosen window/end-basis.")
            else:
                try:
                    st.dataframe(cagr_warn.style.hide(axis="index"),
                                use_container_width=True,
                                height=_auto_h(cagr_warn, row_h=26, base=110))
                except Exception:
                    st.dataframe(cagr_warn, use_container_width=True,
                                height=_auto_h(cagr_warn, row_h=26, base=110))

            st.subheader("Calc Trace", divider=True)
            if cagr_trace is None or cagr_trace.empty:
                st.caption("No calc trace available for selected CAGR end-basis.")
            else:
                q = st.text_input("Filter CAGR calc trace…", key=f"cagr_trace_q_{safe_name}")
                vt = cagr_trace.copy()
                if q.strip():
                    s = q.strip().lower()
                    vt = vt[
                        vt["Metric"].astype(str).str.lower().str.contains(s)
                        | vt["Inputs"].astype(str).str.lower().str.contains(s)
                        | vt["Path Used"].astype(str).str.lower().str.contains(s)
                        | vt.get("Flags","").astype(str).str.lower().str.contains(s)
                    ]
                try:
                    sty = vt.style.hide(axis="index")
                except Exception:
                    sty = vt.style
                sty = sty.format(lambda v: "—" if pd.isna(v) else f"{float(v):,.2f}%", subset=["Value"])
                st.dataframe(sty, use_container_width=True, height=_auto_h(vt, row_h=26, base=110))
                
            # --- Download: CAGR / MOS / PEG (JSON, period-first) ---
            try:
                cagr_list = [x for x in (cagr_items or []) if x.lower().endswith(" cagr")]
                payload_cg_yf = _cagr_results_to_yearfirst_json(
                    items=cagr_list, N=int(N), end_basis=end_basis,
                    sum_df=sum_df, annual=annual, quarterly=qtr, ttm_col=ttm_col, price_now=price_now,
                    name=name, industry=industry, bucket=bucket
                )
                st.download_button(
                    label="⬇️ Download CAGR / MOS / PEG (JSON, period-first)",
                    data=(__import__("json").dumps(payload_cg_yf, indent=2, ensure_ascii=False)).encode("utf-8"),
                    file_name=f"{name}_cagr_mos_peg_periodfirst.json",
                    mime="application/json",
                    use_container_width=True,
                    key=f"dl_cagr_yearfirst_{safe_name}",
                )
            except Exception:
                pass        

        # ---------- Cash Flow KPI cards (toggle Latest FY vs TTM; follow config.CF_METRICS_BY_BUCKET) ----------
        st.markdown(
            section("💧 Cash Flow KPIs", "Toggle TTM or Latest FY; computed by calculations.compute_cashflow_kpis"),
            unsafe_allow_html=True
        )

        cf_basis = "TTM" if st.toggle("Show TTM (else Latest FY)", value=True, key=f"cf_basis_{safe_name}") else "FY"

        try:
            cf = calculations.compute_cashflow_kpis(
                annual_df=annual,
                quarterly_df=qtr,
                basis=cf_basis,           # "TTM" or "FY"
                price_fallback=price_now,
                bucket=bucket             # <- make the card bucket-aware
            ) or {}
        except Exception:
            cf = {}

        # 1) Which labels to show for this bucket?
        _default_cf_labels = [
            "CFO", "Capex", "Free Cash Flow",
            "FCF Margin (%)", "FCF Yield (%)",
            "Capex to Revenue (%)", "CFO/EBITDA (%)", "Cash Conversion (%)",
        ]
        CF_MENU = getattr(config, "CF_METRICS_BY_BUCKET", {}) or {}
        cf_labels = CF_MENU.get(bucket, _default_cf_labels)

        # 2) Map visible labels -> keys returned by compute_cashflow_kpis()
        _CF_LABEL_TO_KEY = {
            "CFO": "CFO",
            "Capex": "Capex",
            "Free Cash Flow": "FCF",
            "FCF Margin": "FCF Margin (%)",
            "FCF Margin (%)": "FCF Margin (%)",
            "FCF Yield": "FCF Yield (%)",
            "FCF Yield (%)": "FCF Yield (%)",
            "Capex to Revenue": "Capex to Revenue (%)",
            "Capex to Revenue (%)": "Capex to Revenue (%)",
            "CFO/EBITDA": "CFO/EBITDA (%)",
            "CFO/EBITDA (%)": "CFO/EBITDA (%)",
            "Cash Conversion": "Cash Conversion (%)",
            "Cash Conversion (%)": "Cash Conversion (%)",
        }

        _percentish_keys = {
            "FCF Margin (%)", "FCF Yield (%)", "Capex to Revenue (%)",
            "CFO/EBITDA (%)", "Cash Conversion (%)",
        }

        def _fmt_cf(label: str, value):
            key = _CF_LABEL_TO_KEY.get(label, label)
            return _fmt_pct(value) if key in _percentish_keys else _fmt_num(value)

        # 3) Build cards strictly from config
        cf_pairs = []
        for lbl in cf_labels:
            key = _CF_LABEL_TO_KEY.get(lbl, lbl)
            cf_pairs.append((lbl, _fmt_cf(lbl, cf.get(key))))

        render_kpi_text_grid(cf_pairs)
        _shown_keys = { _CF_LABEL_TO_KEY.get(lbl, lbl) for lbl in cf_labels }
        
        # --- SYNC: Cash Flow KPI card (numeric) for Decision page ---
        _cf_sync_vals = {}
        for k in _shown_keys:
            try:
                v = cf.get(k)
                _cf_sync_vals[k] = (float(v) if v is not None and np.isfinite(float(v)) else None)
            except Exception:
                _cf_sync_vals[k] = None

        st.session_state.setdefault("CF_SYNC", {})[name] = {
            "basis": cf_basis,   # "TTM" or "FY"
            "values": _cf_sync_vals
        }

        # --- Soldier Worm for Cash Flow KPIs (TTM or FY, per toggle) ---
        try:
            cf_warn = calculations.build_soldier_worm_cashflow_kpis(
                annual_df=annual, quarterly_df=qtr, basis=cf_basis, price_fallback=price_now
            )
        except Exception:
            cf_warn = pd.DataFrame(columns=["Metric","Period","Missing Inputs"])

        try:
            cf_trace = calculations.build_soldier_worm_cashflow_kpis_trace(
                annual_df=annual, quarterly_df=qtr, basis=cf_basis, price_fallback=price_now
            )
        except Exception:
            cf_trace = pd.DataFrame()
            
        # --- filter diagnostics to shown items
        if isinstance(cf_warn, pd.DataFrame) and not cf_warn.empty and "Metric" in cf_warn.columns:
            cf_warn = cf_warn[cf_warn["Metric"].astype(str).isin(_shown_keys)]

        if isinstance(cf_trace, pd.DataFrame) and not cf_trace.empty and "Metric" in cf_trace.columns:
            cf_trace = cf_trace[cf_trace["Metric"].astype(str).isin(_shown_keys)]
            
        # Cash Flow — Diagnostics (Missing + Trace)
        cf_miss_cnt  = 0 if cf_warn is None else len(cf_warn)
        cf_trace_cnt = 0 if cf_trace is None else len(cf_trace)

        with st.expander(f"💧 Cash Flow — Diagnostics  —  missing: {cf_miss_cnt} · trace rows: {cf_trace_cnt}", expanded=False):
            st.subheader("Missing Inputs", divider=True)
            if cf_warn is None or cf_warn.empty:
                st.caption("No missing inputs for the selected basis.")
            else:
                try:
                    st.dataframe(cf_warn.style.hide(axis="index"),
                                use_container_width=True,
                                height=_auto_h(cf_warn, row_h=26, base=110))
                except Exception:
                    st.dataframe(cf_warn, use_container_width=True,
                                height=_auto_h(cf_warn, row_h=26, base=110))

            st.subheader("Calc Trace", divider=True)
            if cf_trace is None or cf_trace.empty:
                st.caption("No calc trace available for selected Cash Flow KPIs.")
            else:
                qq = st.text_input("Filter Cash Flow calc trace…", key=f"cf_trace_q_{safe_name}")
                vt = cf_trace.copy()
                if qq.strip():
                    s = qq.strip().lower()
                    vt = vt[
                        vt["Metric"].astype(str).str.lower().str.contains(s)
                        | vt["Inputs"].astype(str).str.lower().str.contains(s)
                        | vt["Path Used"].astype(str).str.lower().str.contains(s)
                        | vt.get("Formula","").astype(str).str.lower().str.contains(s)
                    ]
                try:
                    st.dataframe(vt.style.hide(axis="index"), use_container_width=True,
                                height=_auto_h(vt, row_h=26, base=110))
                except Exception:
                    st.dataframe(vt, use_container_width=True,
                                height=_auto_h(vt, row_h=26, base=110))
                    
            # --- Download: Cash Flow KPIs (JSON, period-first) ---
            try:
                cf = { key: cf.get(key) for key in _shown_keys }
                payload_cf_yf = _cashflow_to_yearfirst_json(
                    cf, basis=cf_basis, annual_df=annual,
                    name=name, industry=industry, bucket=bucket
                )
                st.download_button(
                    label="⬇️ Download Cash Flow (JSON, period-first)",
                    data=(__import__("json").dumps(payload_cf_yf, indent=2, ensure_ascii=False)).encode("utf-8"),
                    file_name=f"{name}_cashflow_periodfirst.json",
                    mime="application/json",
                    use_container_width=True,
                    key=f"dl_cf_yearfirst_{safe_name}",
                )
            except Exception:
                pass

        # --- ❄️ Snowflake — Five-Pillar Score (reads only values already on this page) ---
        from utils import rules as _rules

        st.markdown(
            section("❄️ Snowflake — Five-Pillar Score",
                    "Scores each pillar from 0–100 using your TTM ratios, Cash-Flow KPIs, and the current CAGR window. No new formulas."),
            unsafe_allow_html=True
        )

        # Pull a spec for this bucket
        _specs = (_rules.SNOWFLAKE_SPECS.get(bucket) or _rules.SNOWFLAKE_SPECS.get("General") or {})

        # ---------- Local helpers (self-contained) ----------
        def __sf_ttm_col_name(df: pd.DataFrame) -> str | None:
            if isinstance(df, pd.DataFrame):
                for c in df.columns:
                    if isinstance(c, str) and c.upper().startswith("TTM"):
                        return c
                if "TTM" in df.columns:
                    return "TTM"
            return None

        def __sf_years_in_summary(df: pd.DataFrame) -> list[int]:
            if df is None or df.empty: return []
            return sorted([c for c in df.columns if isinstance(c, (int, np.integer))])

        def __sf_syn_index(bucket_name: str) -> dict:
            cats = (getattr(config, "INDUSTRY_SUMMARY_RATIOS_CATEGORIES", {}) or {}).get(bucket_name) \
                or (getattr(config, "INDUSTRY_SUMMARY_RATIOS_CATEGORIES", {}) or {}).get("General") \
                or {}
            idx = {}
            for _cat, items in (cats or {}).items():
                if not isinstance(items, dict): 
                    continue
                for canonical, syns in items.items():
                    key = str(canonical).strip().lower()
                    idx[key] = canonical
                    for s in (syns or []):
                        idx[str(s).strip().lower()] = canonical
                    for v in [
                        canonical.replace(" (%)","").replace(" (x)","").replace(" (×)",""),
                        f"{canonical} (%)" if "(%)" not in canonical else canonical.replace(" (%)",""),
                        f"{canonical} (x)"  if "(x)"  not in canonical and "×" not in canonical else canonical.replace(" (x)",""),
                        f"{canonical} (×)"  if "×"    not in canonical and "(x)" not in canonical else canonical.replace(" (×)",""),
                    ]:
                        idx[str(v).strip().lower()] = canonical
            extras = {
                "gross margin": "Gross Margin (%)",
                "net margin": "Net Margin (%)",
                "dividend yield": "Dividend Yield (%)",
                "p/e": "P/E (×)",
                "p/b": "P/B (×)",
                "net debt / ebitda": "Net Debt / EBITDA (×)",
                "interest coverage": "Interest Coverage (×)",
                "receivables days": "Receivable Days",
                "payables days": "Payable Days",
                "ev/ebitda": "EV/EBITDA (×)",
                "p/nav":     "P/NAV (×)",
            }
            idx.update(extras)
            return idx

        def __sf_find_row(sum_df: pd.DataFrame, label: str, idx: dict):
            if sum_df is None or sum_df.empty: 
                return None
            hit = sum_df[sum_df["Metric"].astype(str).str.lower() == str(label).lower()]
            if not hit.empty:
                return hit.iloc[0]
            s = str(label).strip().lower()
            canon = idx.get(s) or idx.get(s.replace(" (%)","")) or idx.get(s + " (%)") \
                or idx.get(s.replace(" (x)","")) or idx.get(s + " (x)") \
                or idx.get(s.replace(" (×)","")) or idx.get(s + " (×)")
            if canon:
                hit = sum_df[sum_df["Metric"].astype(str).str.lower() == canon.lower()]
                if not hit.empty:
                    return hit.iloc[0]
            return None

        def __sf_num_or_none(v):
            try:
                f = float(v)
                if np.isfinite(f):
                    return int(f) if abs(f - int(f)) < 1e-9 else f
            except Exception:
                pass
            return None

        def __sf_cagr_from_summary_row(row: pd.Series, years: list[int], N: int,
                                    end_basis: str, ttm_col: str | None) -> float | None:
            if row is None or not years or len(years) < 2:
                return None
            yrs = sorted([int(y) for y in years])
            EB  = str(end_basis).upper()
            N   = int(N)

            last_fy = int(yrs[-1])
            if EB == "TTM":
                if not (ttm_col and (ttm_col in row) and pd.notna(row.get(ttm_col))):
                    return None
                if len(yrs) < N:
                    return None
                y0   = yrs[-N]
                v0   = pd.to_numeric(pd.Series([row.get(y0)]), errors="coerce").iloc[0]
                vEnd = pd.to_numeric(pd.Series([row.get(ttm_col)]), errors="coerce").iloc[0]
            else:
                if len(yrs) <= N:
                    return None
                y0   = yrs[-(N+1)]
                v0   = pd.to_numeric(pd.Series([row.get(y0)]), errors="coerce").iloc[0]
                vEnd = pd.to_numeric(pd.Series([row.get(last_fy)]), errors="coerce").iloc[0]

            try:
                v0 = float(v0); vEnd = float(vEnd)
                if not (np.isfinite(v0) and np.isfinite(vEnd)) or v0 <= 0 or vEnd <= 0:
                    return None
                return (vEnd / v0) ** (1.0 / N) - 1.0
            except Exception:
                return None


        # Use the already-chosen toggles from the CAGR panel if present (safe read)
        use_ttm_state = bool(st.session_state.get(f"cagr_use_ttm_{safe_name}", True))

        _rawN = st.session_state.get(f"cagr_win_{safe_name}", 5)
        try:
            N = int(float(_rawN))
            if N not in (3, 5):
                N = 5
        except Exception:
            N = 5

        end_basis = "TTM" if use_ttm_state else "FY"

        syn_idx = __sf_syn_index(bucket)
        ttm_col = __sf_ttm_col_name(sum_df) if (isinstance(sum_df, pd.DataFrame) and not sum_df.empty) else None
        years = __sf_years_in_summary(sum_df) if (isinstance(sum_df, pd.DataFrame) and not sum_df.empty) else []

        def _get_from_summary(label: str):
            """
            Prefer Summary TTM; fall back to latest FY if TTM is absent.
            """
            if sum_df is None or sum_df.empty:
                return None
            row = __sf_find_row(sum_df, label, syn_idx)
            if row is None:
                return None

            # 1) TTM if present
            if ttm_col and (ttm_col in row) and pd.notna(row.get(ttm_col)):
                return __sf_num_or_none(row.get(ttm_col))

            # 2) latest FY fallback
            fy_cols = [c for c in row.index if isinstance(c, (int, np.integer))]
            if fy_cols:
                last_fy = max(fy_cols)
                return __sf_num_or_none(row.get(last_fy))

            return None


        def _get_from_cf(label: str):
            if not isinstance(cf, dict):
                return None
            m = {
                "FCF Margin (%)":        "FCF Margin (%)",
                "FCF Yield (%)":         "FCF Yield (%)",
                "Capex/Revenue (%)":     "Capex to Revenue (%)",
                "CFO/EBITDA (×)":        "CFO/EBITDA (%)",   # CF provides percent → convert below
                "CFO/EBITDA (%)":        "CFO/EBITDA (%)",
                "Cash Conversion (%)":   "Cash Conversion (%)",
                # ⬇️ add this
                "Average Cost of Debt (%)": "Average Cost of Debt (%)",
            }
            key = m.get(label, label)
            v = __sf_num_or_none(cf.get(key))
            if v is None:
                return None
            # Unit bridge for CFO/EBITDA when spec asks for × but CF gives %
            if "(×)" in str(label) and "(%)" in str(key):
                return v / 100.0
            return v

        # ---- Momentum helpers: tolerant filename matching (Berhad/Bhd/PLC + relaxed prefix)
        def _resolve_ohlc_dir() -> str:
            # First existing path wins
            candidates = [
                os.path.abspath(os.path.join(_PARENT, "data", "ohlc")),   # project root (where Momentum page writes)
                os.path.abspath(os.path.join(os.getcwd(), "data", "ohlc")),  # CWD fallback
                os.path.abspath(os.path.join(_THIS, "..", "data", "ohlc")),  # pages/.. fallback
                os.path.abspath(os.path.join(_GRANDP, "data", "ohlc")),      # legacy/old location
            ]
            for d in candidates:
                if os.path.isdir(d):
                    return d
            return candidates[0]

        OHLC_DIR = _resolve_ohlc_dir()

        def _ohlc_dir_etag() -> int:
            """Bust cache when files change."""
            try:
                return int(os.stat(OHLC_DIR).st_mtime_ns)
            except Exception:
                return 0

        def _safe_ohlc_name(x: str) -> str:
            return re.sub(r"[^0-9A-Za-z]+", "_", str(x)).strip("_")

        def _alias_candidates(x: str) -> list[str]:
            """
            Generate tolerant filename candidates (handles Berhad/Bhd/PLC suffixes).
            Also returns a base without those suffixes so relaxed prefix matches work.
            """
            base = _safe_ohlc_name(x)
            cand = {base}
            for suf in ["_Berhad", "_Bhd", "_PLC"]:
                if base.endswith(suf):
                    cand.add(base[: -len(suf)])
            cand.add(re.sub(r'_(Berhad|Bhd|PLC)$', '', base, flags=re.I))
            # keep order & de-dup
            return [c for c in dict.fromkeys([c for c in cand if c])]

        def _ohlc_path_for(stock_name: str) -> str | None:
            """Exact match first; else tolerant alias + relaxed prefix (e.g., Public_Bank ↔ Public_Bank_Berhad.csv)."""
            exact = os.path.join(OHLC_DIR, f"{_safe_ohlc_name(stock_name)}.csv")
            if os.path.exists(exact):
                return exact
            try:
                files = {f.lower(): os.path.join(OHLC_DIR, f)
                        for f in os.listdir(OHLC_DIR) if f.lower().endswith(".csv")}
                for cand in _alias_candidates(stock_name):
                    target = f"{cand}.csv".lower()
                    if target in files:
                        return files[target]
                    # relaxed prefix
                    for fname, full in files.items():
                        if fname.startswith(cand.lower()):
                            return full
            except Exception:
                pass
            return None

        @st.cache_data(show_spinner=False)
        def _load_ohlc_for(stock_name: str, _etag: int) -> pd.DataFrame | None:
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

        def _mom_change_pct_df(df: pd.DataFrame, months: int = 12) -> float | None:
            """
            Trailing price change (%) using a calendar-month window.
            If there's no exact prior-date, use the last price <= start date.
            Falls back to ~252 trading days (~1Y) if needed.
            """
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
            df = _load_ohlc_for(stock_name, _ohlc_dir_etag())  # cache busts when files change
            if df is None:
                return None
            return _mom_change_pct_df(df, months=12)


        def _get_momentum(label: str):
            """
            Priority:
            1) Values published by the Momentum page into st.session_state
            2) Summary table (if you compute momentum there)
            3) OHLC CSVs in data/ohlc (fallback; what this page already supports)
            """
            L = (label or "").upper()

            def _bucket(L):
                if "1M" in L or "30D" in L: return "1M"
                if "3M" in L or "90D" in L: return "3M"
                if "6M" in L or "180D" in L: return "6M"
                return "12M"

            key_m = _bucket(L)

            # 1) Session-state from Momentum page (support a few common shapes)
            for cache_name in ("MOMENTUM_SYNC","MOMENTUM_RETURNS","MOM_RETURNS","MOM_CACHE"):
                cache = st.session_state.get(cache_name)
                if isinstance(cache, dict):
                    rec = cache.get(name) or cache.get(safe_name)
                    if isinstance(rec, dict):
                        v = rec.get(key_m) or rec.get(key_m.lower()) or rec.get(key_m.replace("M","m"))
                        if v is not None:
                            return __sf_num_or_none(v)

            # Also support per-key style like mom_<safe>_12m
            single = st.session_state.get(f"mom_{_safe_ohlc_name(name)}_{key_m.lower()}")
            if single is not None:
                return __sf_num_or_none(single)

            # 2) Summary table candidates
            for cand in [label, "12M Price Change (%)", "Price Change 12M (%)", "Total Return 1Y (%)"]:
                v = _get_from_summary(cand)
                if v is not None:
                    return v

            # 3) OHLC CSV fallback
            df_px = _load_ohlc_for(name, _ohlc_dir_etag())
            if df_px is None or df_px.empty:
                return None
            months = 1 if key_m=="1M" else 3 if key_m=="3M" else 6 if key_m=="6M" else 12
            return _mom_change_pct_df(df_px, months=months)

        def _series_from_annual(label: str) -> pd.Series | None:
            if annual is None or annual.empty or "Year" not in annual.columns:
                return None

            def _build_series(years_raw, values) -> pd.Series | None:
                # coerce to numeric, then keep only rows with both year and value present
                yrs = pd.to_numeric(years_raw, errors="coerce")
                vals = pd.to_numeric(values, errors="coerce")
                mask = yrs.notna() & vals.notna()
                if mask.sum() < 2:
                    return None
                yrs_ok = yrs[mask].astype("Int64").dropna().astype(int)
                vals_ok = vals[mask].astype(float)
                return pd.Series(data=vals_ok.to_list(), index=yrs_ok.to_list())

            years_raw = annual["Year"]

            # mappings for common labels
            cand_map = {
                "Revenue":      ["Revenue","Total Revenue","TotalRevenue","Sales"],
                "EBITDA":       ["EBITDA"],
                "Net Profit":   ["Net Profit","NPAT","Profit After Tax","PATAMI","Net Income","Profit attributable to owners"],
                "CFO":          ["CFO","Cash Flow from Ops","Cash from Operations","Operating Cash Flow"],
                "Orderbook":    ["Orderbook","Order Book","Unbilled Sales"],
                "Store Count":  ["Store Count"],
                "ARR":          ["ARR"],
                "AUM":          ["AUM"],
            }

            # FCF = CFO - |Capex|
            if label == "FCF":
                cfo = pd.to_numeric(annual.get("CFO"), errors="coerce") if "CFO" in annual.columns else None
                cap = None
                for cap_col in ("Capex", "Capital Expenditure"):
                    if cap_col in annual.columns:
                        cap = pd.to_numeric(annual[cap_col], errors="coerce").abs()
                        break
                if cfo is not None and cap is not None:
                    fcf = cfo - cap
                    return _build_series(years_raw, fcf)
                return None

            # normal path via mapped columns
            if label in cand_map:
                for k in cand_map[label]:
                    if k in annual.columns:
                        s = annual[k]
                        ser = _build_series(years_raw, s)
                        if ser is not None:
                            return ser

            return None

        def _cagr_from_annual_series(label: str, N: int, end_basis: str, ttm_override: float | None) -> float | None:
            s = _series_from_annual(label)
            if s is None or s.dropna().size < 2:
                return None
            yrs = sorted([int(x) for x in s.index if isinstance(x, (int, np.integer))])
            need_more = (len(yrs) < N) if end_basis == "TTM" else (len(yrs) <= N)
            if need_more:
                return None
            v_end = float(ttm_override) if (end_basis == "TTM" and ttm_override is not None) else float(s.get(yrs[-1], np.nan))
            y0 = yrs[-N] if end_basis == "TTM" else yrs[-(N + 1)]
            v0 = float(s.get(y0, np.nan))
            if not (np.isfinite(v_end) and np.isfinite(v0)) or v_end <= 0 or v0 <= 0:
                return None
            return (v_end / v0) ** (1.0 / N) - 1.0

        # raw TTM for CFO/FCF fallback (already computed earlier as ttm_raw)
        ttm_cfo = ttm_raw.get("CFO") if isinstance(ttm_raw, dict) else None
        ttm_fcf = ((ttm_raw.get("CFO") or 0.0) - abs(ttm_raw.get("Capex") or 0.0)) if isinstance(ttm_raw, dict) else None
        
        # --- TTM end overrides (match the CAGR panel behaviour) ---
        ttm_overrides = {
            "Revenue":    __sf_num_or_none((ttm_raw or {}).get("Revenue")),
            "EBITDA":     __sf_num_or_none((ttm_raw or {}).get("EBITDA")),
            "Net Profit": __sf_num_or_none((ttm_raw or {}).get("Net Profit")),
            "CFO":        __sf_num_or_none((ttm_raw or {}).get("CFO")),
            "FCF":        __sf_num_or_none(((ttm_raw or {}).get("CFO") or 0.0) - abs((ttm_raw or {}).get("Capex") or 0.0)),
        }

        def __sf_base_label_for_cagr(name: str) -> str:
            name = str(name).strip()
            mapping = {
                "Operating Cash Flow": "CFO",
                "Free Cash Flow": "FCF",
                "Net Income": "Net Profit",
            }
            return mapping.get(name, name)

        def _get_cagr(label_with_suffix: str):
            # Accept both "CAGR (%)" and plain "CAGR"
            base = str(label_with_suffix).replace(" CAGR (%)", "").replace(" CAGR", "").strip()
            base = __sf_base_label_for_cagr(base)  # e.g., "Operating Cash Flow" -> "CFO"

            # 0) Use the CAGR panel cache if it matches current window + end-basis
            cache = st.session_state.get("CAGR_SYNC", {}).get(name)
            if isinstance(cache, dict) \
            and str(cache.get("end_basis","")).upper() == str(end_basis).upper() \
            and int(cache.get("N", 0)) == int(N):
                v = (cache.get("values_pct") or {}).get(base)
                if v is not None:
                    return float(v)  # already in percent

            # 1) Try Summary (same rules as the panel)
            row = __sf_find_row(sum_df, base, syn_idx) if (sum_df is not None and not sum_df.empty) else None
            g = None
            if isinstance(row, pd.Series) and years:
                g = __sf_cagr_from_summary_row(row, years, int(N), end_basis, ttm_col)
                if g is not None:
                    return g * 100.0

            # 2) Annual series + TTM override fallback (mirrors the panel)
            ttm_end = None
            if str(end_basis).upper() == "TTM":
                ttm_end = ttm_overrides.get(base)
                if ttm_end is None and base in ("Orderbook","Store Count","ARR","AUM","NAV per Unit","Gross Loans","Deposits"):
                    v = _get_from_summary(base)
                    if v is None:
                        s = _series_from_annual(base)
                        if s is not None and not s.dropna().empty:
                            fy_cols = sorted([int(y) for y in s.index if isinstance(y, (int, np.integer))])
                            v = float(s.get(fy_cols[-1])) if fy_cols else None
                    ttm_end = v

            g = _cagr_from_annual_series(base, int(N), end_basis, ttm_end)
            return None if g is None else g * 100.0

        def _metric_value(spec: dict):
            src = str(spec.get("src", "auto")).lower()
            nm  = spec.get("name", "")
            if src == "cf":        return _get_from_cf(nm)
            if src == "cagr":      return _get_cagr(nm)
            if src == "momentum":  return _get_momentum(nm)
            return _get_from_summary(nm)  # auto/default → Summary (TTM)

        def _score_value(v, low, high, invert=False):
            if v is None:
                return None
            try:
                v = float(v); low = float(low); high = float(high)
            except Exception:
                return None
            if not np.isfinite(v) or high == low:
                return None

            # Always normalize on an ascending range, then (optionally) invert
            lo, hi = (low, high) if low < high else (high, low)
            x = (v - lo) / (hi - lo)
            if invert:
                x = 1.0 - x
            return float(np.clip(x, 0.0, 1.0) * 100.0)
        
        def _score_dividend_yield_specialcase(dy, fd_rate, epf_rate, cap=35.0):
            """
            Returns a 0–100 score for Dividend Yield (%) using FD & EPF:
            - No/zero dividend -> 0
            - 0 .. FD          -> linearly 0 .. 40
            - FD .. EPF        -> linearly 40 .. 70
            - EPF .. cap (35%) -> linearly 70 .. 98
            - >= cap           -> 98  (almost full)
            All inputs expected in PERCENT (e.g., 3.5 for 3.5%).
            """
            try:
                dy  = 0.0 if dy is None else float(dy)
                fd  = float(fd_rate) if fd_rate is not None else 0.0
                epf = float(epf_rate) if epf_rate is not None else fd
            except Exception:
                return 0.0

            if not np.isfinite(dy) or dy <= 0.0:
                return 0.0

            if epf <= fd:
                epf = fd  # degrade gracefully if user entered EPF <= FD

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
                return 98.0  # almost full beyond 35%

        # ---------- Build pillar scores ----------
        pillar_rows: list[dict] = []
        pillar_scores: list[tuple[str, float | None]] = []

        for pillar, items in _specs.items():
            metrics, scores, weights = [], [], []
            for sp in (items or []):
                val = _metric_value(sp)
                nm  = str(sp.get("name", "")).strip().lower()

                if nm.startswith("dividend yield"):
                    # Dynamic scoring using your Global Settings (percent values)
                    fd  = get_fd_eps_rate()
                    epf = get_epf_rate()
                    scr = _score_dividend_yield_specialcase(val, fd, epf, cap=35.0)
                else:
                    scr = _score_value(
                        val,
                        sp.get("low", 0),
                        sp.get("high", 1),
                        sp.get("invert", False)
                    )                                   
                
                w   = float(sp.get("weight", 1.0) or 1.0)
                metrics.append({"Pillar": pillar, "Metric": sp.get("name",""), "Src": sp.get("src","auto"),
                                "Value": val, "Score": scr, "Weight": w})
                if scr is not None:
                    scores.append(scr * w); weights.append(w)
            ps = (sum(scores) / sum(weights)) if weights else None
            pillar_scores.append((pillar, ps))
            pillar_rows.extend(metrics)

        # ---------- Display (Radar by default; details hidden in expander) ----------
        preferred = ["Future Value", "Earnings Quality", "Growth Consistency", "Cash Strength", "Momentum"]
        cats = [p for p in preferred if any(p == q for q, _ in pillar_scores)] + \
            [q for q, _ in pillar_scores if q not in preferred]
        vals = []
        for c in cats:
            v = next((s for (p, s) in pillar_scores if p == c), None)
            vals.append(0.0 if v is None else float(v))

        if not any(v is not None for _, v in pillar_scores):
            st.info("No inputs available yet to score the five pillars for this stock.")
        else:
            if hasattr(st, "segmented_control"):
                view_mode = st.segmented_control("View", options=["Radar", "Bars"],
                                                selection_mode="single", default="Radar",
                                                key=f"snowflake_view_{safe_name}")
            else:
                view_mode = st.radio("View", options=["Radar", "Bars"], index=0, horizontal=True,
                                    key=f"snowflake_view_{safe_name}")

            if view_mode == "Radar":
                theta = cats + [cats[0]]
                r     = vals + [vals[0]]
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(theta=theta, r=r, fill="toself", name=name))
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                                polar=dict(radialaxis=dict(visible=True, range=[0,100], tickvals=[0,20,40,60,80,100]),
                                            angularaxis=dict(direction="clockwise")))
                st.plotly_chart(fig, use_container_width=True, key=f"snowflake_radar_{safe_name}")
            else:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=vals, y=cats, orientation="h"))
                fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10),
                                xaxis=dict(range=[0,100], title="Score"), yaxis=dict(title=""))
                st.plotly_chart(fig, use_container_width=True, key=f"snowflake_bar_{safe_name}")

            # Details (collapsed)
            det = pd.DataFrame(pillar_rows)
            if not det.empty:
                det_disp = det[["Pillar","Metric","Src","Value","Score","Weight"]].copy()
                def _fmt_cell(row):
                    lab = str(row["Metric"]).lower()
                    v = row["Value"]
                    if v is None or (isinstance(v, float) and (pd.isna(v) or not np.isfinite(v))): return "—"
                    if "(%)" in lab or "margin" in lab or "yield" in lab or " ratio" in lab: return f"{float(v):,.2f}%"
                    return f"{float(v):,.2f}"
                det_disp["Value"] = det_disp.apply(_fmt_cell, axis=1)
                with st.expander("Show metric details (click to expand)", expanded=False):
                    try:
                        st.dataframe(det_disp.style.hide(axis="index"),
                                    use_container_width=True, height=_auto_h(det_disp, row_h=26, base=110))
                    except Exception:
                        st.dataframe(det_disp, use_container_width=True, height=_auto_h(det_disp, row_h=26, base=110))

    # ========================= ANNUAL (raw schema table + YoY always + 4 charts with toggle) =========================
    with tabs[1]:
        st.markdown(
            section("📄 Annual — Raw Inputs by Category",
                    "Raw inputs grouped by category (no derived fields).",
                    "info"),
            unsafe_allow_html=True
        )

        if annual is None or annual.empty:
            st.info("No annual rows yet.")
        else:
            # Append TTM row (Year = last FY + 1) from latest 4 quarters
            annual_plus = _annual_with_appended_ttm(annual, qtr, bucket)

            # Raw table — always show YoY (%) columns
            a_tbl = _raw_df_from_schema(annual_plus, bucket, quarterly=False)
            sty = _add_yoy_to_multiheader(a_tbl, is_quarterly=False)  # YoY always on
            sty = _apply_commas_to_styler(sty, a_tbl)
            st.dataframe(sty, use_container_width=True, height=_auto_h(a_tbl, row_h=30))

            # Download as JSON (raw) — includes TTM row
            payload = _annual_to_human_json(annual_plus, name=name, industry=industry, bucket=bucket)
            st.download_button(
                label="⬇️ Download Annual (JSON, raw inputs)",
                data=(__import__("json").dumps(payload, indent=2, ensure_ascii=False)).encode("utf-8"),
                file_name=f"{name}_annual_raw.json",
                mime="application/json",
                use_container_width=True,
                key=f"dl_annual_{safe_name}",
            )

            # ---- Quick Charts (四宫格) ----
            st.markdown(
                section("📈 Quick Charts (pick 4 metrics)",
                        "Line charts over FY sequence. Uses existing raw fields only.",
                        "info"),
                unsafe_allow_html=True
            )

            label_to_col = _available_plot_metrics(annual_plus, quarterly=False, bucket=bucket)
            opts = ["— Select —"] + sorted(list(label_to_col.keys()))

            # Prepare x-axis once (Year order)
            ax = annual_plus.copy()
            ax["_Y"] = pd.to_numeric(ax.get("Year"), errors="coerce")
            ax = ax.dropna(subset=["_Y"]).sort_values("_Y")
            ax["_X"] = ax["_Y"].astype(int).astype(str)

            # -------- Row 1: selectors with charts directly below --------
            a_col1, a_col2 = st.columns(2)

            with a_col1:
                a_lab1 = st.selectbox("Chart #1", options=opts, index=0, key=f"a_c1_{safe_name}")
                if a_lab1 in label_to_col and a_lab1 != "— Select —":
                    _plot_line(ax, "_X", label_to_col[a_lab1], a_lab1, key=f"a_qchart_1_{safe_name}")

            with a_col2:
                a_lab2 = st.selectbox("Chart #2", options=opts, index=0, key=f"a_c2_{safe_name}")
                if a_lab2 in label_to_col and a_lab2 != "— Select —":
                    _plot_line(ax, "_X", label_to_col[a_lab2], a_lab2, key=f"a_qchart_2_{safe_name}")

            # Toggle sits AFTER charts #1 & #2
            a_show_more = st.toggle("Show charts 3 & 4", value=False, key=f"a_toggle_more_{safe_name}")

            # -------- Row 2 (hidden until toggled): selectors with charts below --------
            if a_show_more:
                a_col3, a_col4 = st.columns(2)
                with a_col3:
                    a_lab3 = st.selectbox("Chart #3", options=opts, index=0, key=f"a_c3_{safe_name}")
                    if a_lab3 in label_to_col and a_lab3 != "— Select —":
                        _plot_line(ax, "_X", label_to_col[a_lab3], a_lab3, key=f"a_qchart_3_{safe_name}")
                with a_col4:
                    a_lab4 = st.selectbox("Chart #4", options=opts, index=0, key=f"a_c4_{safe_name}")
                    if a_lab4 in label_to_col and a_lab4 != "— Select —":
                        _plot_line(ax, "_X", label_to_col[a_lab4], a_lab4, key=f"a_qchart_4_{safe_name}")

    # ========================= QUARTERLY (raw schema table + QoQ always + 4 charts with toggle) =========================
    with tabs[2]:
        st.markdown(
            section("📄 Quarterly — Raw Inputs by Category",
                    "Raw inputs grouped by category (no derived fields).",
                    "info"),
            unsafe_allow_html=True
        )

        if qtr is None or qtr.empty:
            st.info("No quarterly rows yet.")
        else:
            # ---- Raw table (always show QoQ) ----
            q_tbl = _raw_df_from_schema(qtr, bucket, quarterly=True)
            sty = _add_yoy_to_multiheader(q_tbl, is_quarterly=True)  # adds QoQ cols
            sty = _apply_commas_to_styler(sty, q_tbl)
            st.dataframe(sty, use_container_width=True, height=_auto_h(q_tbl, row_h=30))

            # Download as JSON (raw)
            payload_q = _quarterly_to_human_json(qtr, name=name, industry=industry, bucket=bucket)
            st.download_button(
                label="⬇️ Download Quarterly (JSON, raw inputs)",
                data=(__import__("json").dumps(payload_q, indent=2, ensure_ascii=False)).encode("utf-8"),
                file_name=f"{name}_quarterly_raw.json",
                mime="application/json",
                use_container_width=True,
                key=f"dl_quarterly_{safe_name}",
            )

            # ---- Quick Charts (四宫格) ----
            st.markdown(
                section("📈 Quick Charts (pick 4 metrics)",
                        "Line charts over Quarter sequence. Uses existing raw fields only.",
                        "info"),
                unsafe_allow_html=True
            )

            label_to_col = _available_plot_metrics(qtr, quarterly=True, bucket=bucket)
            opts = ["— Select —"] + sorted(list(label_to_col.keys()))

            # Prepare x-axis once
            qx = qtr.copy()
            if "Qnum" not in qx.columns:
                qx["Qnum"] = qx["Quarter"].map(_qnum)
            qx = qx.dropna(subset=["Year","Qnum"]).sort_values(["Year","Qnum"]).copy()
            qx["_X"] = qx["Year"].astype(int).astype(str) + " Q" + qx["Qnum"].astype(int).astype(str)

            # -------- Row 1: selectors with charts directly below --------
            col1, col2 = st.columns(2)

            with col1:
                lab1 = st.selectbox("Chart #1", options=opts, index=0, key=f"c1_{safe_name}")
                if lab1 in label_to_col and lab1 != "— Select —":
                    _plot_line(qx, "_X", label_to_col[lab1], lab1, key=f"qchart_1_{safe_name}")

            with col2:
                lab2 = st.selectbox("Chart #2", options=opts, index=0, key=f"c2_{safe_name}")
                if lab2 in label_to_col and lab2 != "— Select —":
                    _plot_line(qx, "_X", label_to_col[lab2], lab2, key=f"qchart_2_{safe_name}")

            # Toggle sits AFTER charts #1 & #2
            show_more = st.toggle("Show charts 3 & 4", value=False, key=f"toggle_more_{safe_name}")

            # -------- Row 2 (hidden until toggled): selectors with charts below --------
            if show_more:
                col3, col4 = st.columns(2)
                with col3:
                    lab3 = st.selectbox("Chart #3", options=opts, index=0, key=f"c3_{safe_name}")
                    if lab3 in label_to_col and lab3 != "— Select —":
                        _plot_line(qx, "_X", label_to_col[lab3], lab3, key=f"qchart_3_{safe_name}")
                with col4:
                    lab4 = st.selectbox("Chart #4", options=opts, index=0, key=f"c4_{safe_name}")
                    if lab4 in label_to_col and lab4 != "— Select —":
                        _plot_line(qx, "_X", label_to_col[lab4], lab4, key=f"qchart_4_{safe_name}")
