# pages/11_AI_Analyst.py
from __future__ import annotations

# ── Auth ──────────────────────────────────────────────────────────────────────
from auth_gate import require_auth
require_auth()

# ── Std lib ───────────────────────────────────────────────────────────────────
import os, io, re, json, math
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

# ── Third-party ───────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np

# --- Shared data version etag (invalidate caches when data changes) ---
_THIS   = os.path.dirname(__file__)
_PARENT = os.path.dirname(_THIS)          # project root
_GRANDP = os.path.dirname(_PARENT)        # one level up (kept for parity)

_VERSION_FILE = os.path.join(_PARENT, ".data.version")
def _data_etag() -> int:
    try:
        return int(os.stat(_VERSION_FILE).st_mtime_ns)
    except Exception:
        return 0

# ── Local helpers (robust fallbacks) ──────────────────────────────────────────
try:
    from utils.ui import setup_page, section, render_stat_cards, render_page_title
except Exception:
    from ui import setup_page, section  # type: ignore
    def render_page_title(page_name: str) -> None:
        st.title(f"📊 Fundamentals Dashboard — {page_name}")
    def render_stat_cards(*args, **kwargs):          # no-op if missing
        pass

try:
    from utils import io_helpers as ioh
except Exception:
    import io_helpers as ioh                         # type: ignore

# Optional calculations import (we degrade if missing)
try:
    from utils import calculations as calc           # type: ignore
except Exception:
    calc = None

# ╭────────────────────────────────────────────────────────────────────────────╮
# │ Page setup                                                                 │
# ╰────────────────────────────────────────────────────────────────────────────╯
setup_page("AI Analyst — Chat")
render_page_title("AI Analyst — Chat")
st.caption(
    "Type your question. I’ll read View Stock caches (TTM, KPIs, CAGR, CF) "
    "and Momentum CSVs automatically."
)

# Hard cap on how many stocks we send to the LLM in one call.
# This keeps total tokens under your 30k TPM limit.
_MAX_LLM_STOCKS = 20

# ╭────────────────────────────────────────────────────────────────────────────╮
# │ API key: load/save (same UX you used)                                      │
# ╰────────────────────────────────────────────────────────────────────────────╯
_APP_STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".app_state")
_API_FILE      = os.path.join(_APP_STATE_DIR, "openai_api_key.txt")

# ╭────────────────────────────────────────────────────────────────────────────╮
# │ Data loading                                                                │
# ╰────────────────────────────────────────────────────────────────────────────╯
@st.cache_data(show_spinner=False)
def _load_master(_etag: int) -> pd.DataFrame:
    df = ioh.load_data()
    if df is None:
        return pd.DataFrame()
    df.columns = [str(c) for c in df.columns]
    if "Name" not in df.columns:
        return pd.DataFrame()
    return df

try:
    df_master = _load_master(_data_etag())
except Exception:
    df_master = pd.DataFrame()

names_all: list[str] = []
if isinstance(df_master, pd.DataFrame) and not df_master.empty and "Name" in df_master.columns:
    names_all = sorted(df_master["Name"].dropna().astype(str).unique().tolist())

def _load_saved_api_key() -> Optional[str]:
    try:
        v = st.secrets.get("OPENAI_API_KEY")
        if v:
            return str(v).strip()
    except Exception:
        pass
    v = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_APIKEY")
    if v:
        return v.strip()
    try:
        if os.path.exists(_API_FILE):
            with open(_API_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
    except Exception:
        pass
    return None

def _save_api_key(k: str) -> bool:
    try:
        os.makedirs(_APP_STATE_DIR, exist_ok=True)
        with open(_API_FILE, "w", encoding="utf-8") as f:
            f.write(k.strip())
        return True
    except Exception:
        return False

with st.sidebar:
    st.subheader("🔐 OpenAI")
    key_default = _load_saved_api_key() or ""
    api_key = st.text_input("API key", value=key_default, type="password")
    remember = st.checkbox("Remember key on this server (plain text)", value=bool(key_default))
    if remember and api_key and api_key != key_default:
        st.caption("✅ Saved." if _save_api_key(api_key) else "⚠️ Could not save. Check permissions.")

    st.subheader("⚙️ Model")
    model = st.selectbox(
        "Model",
        ["gpt-4o", "gpt-4o-mini", "gpt-5-mini"],  # added gpt-5-mini
        index=0,
    )
    model = st.text_input("Override (optional)", value=model)
    temp  = st.slider("Creativity (temperature)", 0.0, 1.2, 0.2, 0.05)

    st.subheader("📦 Packaging")
    total_names = len(names_all)
    slider_max = max(0, total_names)
    base_min = 20 if slider_max > 20 else 1
    slider_min = min(base_min, slider_max - 1 if slider_max > 1 else slider_max)
    slider_val = min(120, slider_max) if slider_max else 0
    if slider_val < slider_min:
        slider_val = slider_max

    if slider_min >= slider_max:
        max_catalog = slider_max
        if slider_max <= 0:
            st.caption("No cached stocks found yet. Add data to adjust the packaging limit.")
        else:
            st.caption("Only one cached stock available; packaging limit fixed to 1.")
    else:
        slider_step = 10 if (slider_max - slider_min) >= 10 else 1
        max_catalog = st.slider(
            "Max stocks to include",
            min_value=slider_min,
            max_value=slider_max,
            value=slider_val,
            step=slider_step,
        )
    st.caption("Tip: mention tickers in your message to prioritize them. I’ll also look at your current selection in View Stock.")

# ╭────────────────────────────────────────────────────────────────────────────╮
# │ Momentum (reads the exact CSVs your Momentum page writes)                  │
# │ data/ohlc/<Safe_Name>.csv  → compute 3/6/12M, MA50/200, 52w off-high       │
# ╰────────────────────────────────────────────────────────────────────────────╯
def _safe_name(x: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(x)).strip("_")

def _ohlc_path(name: str) -> str:
    # same directory convention as your Momentum page (data/ohlc)  # (ref) 9_Momentum_Data
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ohlc", f"{_safe_name(name)}.csv")

@st.cache_data(show_spinner=False)
def _load_ohlc(name: str, _etag: int) -> Optional[pd.DataFrame]:
    p = _ohlc_path(name)
    if not os.path.exists(p):
        return None
    try:
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
    except Exception:
        return None

def _momentum_features(px: Optional[pd.DataFrame]) -> Dict[str, Optional[float]]:
    out = {"ret_3m_pct": None, "ret_6m_pct": None, "ret_12m_pct": None,
           "ma50": None, "ma200": None, "off_high_52w_pct": None}
    if px is None or px.empty or "Date" not in px.columns or "Close" not in px.columns:
        return out
    s = px.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True).copy()
    s["Close"] = pd.to_numeric(s["Close"], errors="coerce")
    s["ma50"]  = s["Close"].rolling(50).mean()
    s["ma200"] = s["Close"].rolling(200).mean()

    last = float(s["Close"].iloc[-1])
    out["ma50"]  = float(s["ma50"].iloc[-1])  if pd.notna(s["ma50"].iloc[-1])  else None
    out["ma200"] = float(s["ma200"].iloc[-1]) if pd.notna(s["ma200"].iloc[-1]) else None

    for win, key in [(63,"ret_3m_pct"),(126,"ret_6m_pct"),(252,"ret_12m_pct")]:
        if len(s) > win and float(s["Close"].iloc[-win]) != 0:
            out[key] = (last / float(s["Close"].iloc[-win]) - 1.0) * 100.0

    lb = s.tail(252)["Close"]
    if not lb.empty:
        hi = float(lb.max())
        if hi:
            out["off_high_52w_pct"] = (last / hi - 1.0) * 100.0
    return out

# ╭────────────────────────────────────────────────────────────────────────────╮
# │ Fallback TTM reconstruction if View page cache is missing                   │
# ╰────────────────────────────────────────────────────────────────────────────╯
def _guess_bucket(rows: pd.DataFrame) -> str:
    if "IndustryBucket" in rows and rows["IndustryBucket"].dropna().size:
        return str(rows["IndustryBucket"].dropna().iloc[-1])
    if "Industry" in rows and rows["Industry"].dropna().size:
        return str(rows["Industry"].dropna().iloc[-1])
    return "General"

def _schema_keys_for_bucket(rows: pd.DataFrame, bucket: str) -> list[tuple[str, str]]:
    """
    Return [(category, key)] from config schema; fallback to Q_ columns.
    """
    try:
        try:
            from utils import config as _cfg  # preferred
        except Exception:
            import config as _cfg             # fallback
        cats = getattr(_cfg, "INDUSTRY_FORM_CATEGORIES", {}) or {}
        d = cats.get(bucket) or cats.get("General") or {}
        out: list[tuple[str, str]] = []
        for cat, items in (d or {}).items():
            for it in (items or []):
                if isinstance(it, dict) and it.get("key"):
                    out.append((cat, it["key"]))
                elif isinstance(it, str):
                    out.append((cat, it))
        if out:
            return out
    except Exception:
        pass
    # Fallback: any quarterly-looking column
    q_cols = [c for c in rows.columns if isinstance(c, str) and c.startswith("Q_")]
    return [("KPIs", c[2:]) for c in q_cols]

def _ttm_agg_rule(category: str, key: str) -> str:
    """
    Match View Stock semantics: flows=sum, balance-sheet=last (avg if marked),
    ratios/percent/rates=last, explicit TTM/DPS= sum, counts/levels=last.
    """
    cat = (category or "").strip()
    lab = (key or "").strip().lower()

    if cat in ("Income Statement", "Cash Flow"):
        return "sum"
    if cat == "Balance Sheet":
        return "mean" if ("avg" in lab or lab.startswith("average ")) else "last"

    # Explicit flows / TTM-like
    if ("ttm" in lab) or ("new orders" in lab) or ("dividend per share" in lab) or (lab == "dpu"):
        return "sum"

    # Averages
    if ("(avg" in lab) or (" avg" in lab) or lab.startswith("average "):
        return "mean"

    # Ratios / % / rates → carry latest reading
    if any(w in lab for w in ["ratio", "%", " margin", "yield", " rate", "nim", "lcr", "nsfr", "cir", "casa",
                              "occupancy", "wale", "reversion", "hedged debt"]):
        return "last"

    # Counts / levels
    if any(w in lab for w in ["shares", "units", "price", "store count", "subscribers",
                              "fleet size", "room count", "throughput"]):
        return "last"

    return "last"

def _reconstruct_ttm_from_quarters(rows: pd.DataFrame, bucket_hint: Optional[str]) -> dict:
    """
    Build a TTM dict from the latest up to 4 quarters using schema-aware rules.
    Used when the View Stock page hasn't saved a ttm_dict_* in this session.
    """
    if rows is None or rows.empty:
        return {}

    q = rows[rows.get("IsQuarter") == True].copy()
    if q.empty or "Quarter" not in q.columns or "Year" not in q.columns:
        return {}

    q["__Q"] = pd.to_numeric(q["Quarter"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    q["__Y"] = pd.to_numeric(q["Year"], errors="coerce")
    q = q.dropna(subset=["__Y", "__Q"]).sort_values(["__Y", "__Q"])
    if q.empty:
        return {}

    tail = q.tail(min(4, len(q)))

    bucket = bucket_hint or _guess_bucket(rows)
    schema_pairs = _schema_keys_for_bucket(rows, bucket)  # [(category, key)]

    out: dict[str, float] = {}
    for category, key in schema_pairs:
        cand_cols = [f"Q_{key}", key]
        s = None
        for col in cand_cols:
            if col in tail.columns:
                s = pd.to_numeric(tail[col], errors="coerce")
                break
        if s is None:
            continue

        how = _ttm_agg_rule(category, key)
        if how == "sum":
            v = s.sum(skipna=True)
        elif how == "mean":
            v = s.dropna().mean()
        else:
            nz = s[s.notna()]
            v = nz.iloc[-1] if not nz.empty else np.nan

        if pd.notna(v):
            try:
                out[key] = float(v)
            except Exception:
                pass

    return out

def _split_annual_quarter(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replicate View Stock's annual/quarter splitting for downstream calcs."""
    if rows is None or rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    annual = rows[rows.get("IsQuarter") != True].copy()
    if not annual.empty:
        annual = annual.drop(columns=[c for c in ("IsQuarter", "Quarter") if c in annual.columns])
        if "Year" in annual.columns:
            annual = annual.sort_values("Year")

    quarterly = rows[rows.get("IsQuarter") == True].copy()
    if not quarterly.empty:
        if "Quarter" in quarterly.columns:
            quarterly["Qnum"] = pd.to_numeric(
                quarterly["Quarter"].astype(str).str.extract(r"(\d+)")[0],
                errors="coerce",
            )
            quarterly = quarterly.dropna(subset=["Year", "Qnum"]).sort_values(["Year", "Qnum"])
        else:
            quarterly = quarterly.dropna(subset=["Year"]).sort_values("Year")

    return annual, quarterly

# ╭────────────────────────────────────────────────────────────────────────────╮
# │ View Stock caches → pull everything you already publish to session_state   │
# │ (TTM snapshot, synonyms, KPIs, CAGR, CF)                                   │
# ╰────────────────────────────────────────────────────────────────────────────╯
def _current_price(rows: pd.DataFrame) -> Optional[float]:
    for k in ("CurrentPrice","EndQuarterPrice","Price","SharePrice","Annual Price per Share (RM)"):
        if k in rows.columns:
            s = pd.to_numeric(rows[k], errors="coerce").dropna()
            if not s.empty:
                return float(s.iloc[-1])
    return None

def _bucket_of(rows: pd.DataFrame) -> str:
    if "IndustryBucket" in rows.columns and rows["IndustryBucket"].dropna().size:
        return str(rows["IndustryBucket"].dropna().iloc[-1])
    if "Industry" in rows.columns and rows["Industry"].dropna().size:
        return str(rows["Industry"].dropna().iloc[-1])
    return "General"

def _industry_of(rows: pd.DataFrame) -> str:
    if "Industry" in rows.columns and rows["Industry"].dropna().size:
        return str(rows["Industry"].dropna().iloc[-1])
    return ""

def _ttm_snapshot_from_view(name: str) -> Dict[str, Any]:
    """
    Read exactly what View Stock saved via save_view_snapshot() + KPI caches.
    Expects:
      st.session_state[f"ttm_dict_{safe}"] = {Metric -> value}
      st.session_state[f"syn_idx_{safe}"]  = {lowercased variant -> canonical}
      st.session_state[f"bucket_{safe}"]   = bucket
      st.session_state["TTM_KPI_SYNC"][name] = {'period':..., 'values':{...}}
      st.session_state["CAGR_SYNC"][name]    = {'N':..., 'end_basis':..., 'values_pct':{...}}
      st.session_state["CF_SYNC"][name]      = {'basis':..., 'values':{...}}
    """
    skey = _safe_name(name).lower()
    ss = st.session_state

    ttm_dict = ss.get(f"ttm_dict_{skey}") or {}
    syn_idx  = ss.get(f"syn_idx_{skey}")  or {}
    bucket   = ss.get(f"bucket_{skey}")   or None

    ttm_kpi = (ss.get("TTM_KPI_SYNC") or {}).get(name) or {}
    cagr    = (ss.get("CAGR_SYNC")    or {}).get(name) or {}
    cf      = (ss.get("CF_SYNC")      or {}).get(name) or {}

    return {
        "ttm_dict": ttm_dict,
        "syn_idx": syn_idx,
        "bucket_hint": bucket,
        "ttm_kpi": ttm_kpi,
        "cagr": cagr,
        "cf": cf,
    }

# ╭────────────────────────────────────────────────────────────────────────────╮
# │ Vector-store helpers                                                       │
# ╰────────────────────────────────────────────────────────────────────────────╯
def _format_for_vector(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)):
        f = float(value)
        if not math.isfinite(f):
            return None
        return f"{f:.6g}"

    text = str(value).strip()
    if not text:
        return None

    try:
        guess = float(text.replace(",", ""))
        if math.isfinite(guess):
            return f"{guess:.6g}"
    except Exception:
        pass

    return text

def _compose_vector_doc(
    name: str,
    industry: str,
    bucket: str,
    price: Optional[float],
    metrics: Dict[str, Optional[float]],
    momentum: Dict[str, Optional[float]],
    ttm_dict: Dict[str, Any],
    ttm_kpi: Dict[str, Any],
    cagr_card: Dict[str, Any],
    cf_card: Dict[str, Any],
) -> str:
    lines: List[str] = []

    def _add_line(label: str, value: Any) -> None:
        formatted = _format_for_vector(value)
        if formatted is None:
            return
        lab = re.sub(r"\s+", " ", str(label)).strip()
        if not lab:
            return
        lines.append(f"{lab}: {formatted}")

    lines.append(f"name: {name}")
    if industry:
        lines.append(f"industry: {industry}")
    if bucket:
        lines.append(f"bucket: {bucket}")
    if price is not None:
        _add_line("price_now", price)

    for key, val in sorted(metrics.items()):
        _add_line(f"metric {key}", val)

    for key, val in sorted(momentum.items()):
        _add_line(f"momentum {key}", val)

    ttm_items = sorted(ttm_dict.items(), key=lambda kv: str(kv[0]))
    for key, val in ttm_items[:200]:
        _add_line(f"ttm {key}", val)

    ttm_values = (ttm_kpi or {}).get("values") or {}
    if ttm_kpi.get("period"):
        lines.append(f"kpi period: {ttm_kpi['period']}")
    for key, val in sorted(ttm_values.items()):
        _add_line(f"kpi {key}", val)

    if cagr_card.get("N"):
        lines.append(f"cagr years: {cagr_card['N']}")
    if cagr_card.get("end_basis"):
        lines.append(f"cagr end_basis: {cagr_card['end_basis']}")
    for key, val in sorted((cagr_card.get("values_pct") or {}).items()):
        _add_line(f"cagr {key}", val)

    if cf_card.get("basis"):
        lines.append(f"cf basis: {cf_card['basis']}")
    for key, val in sorted((cf_card.get("values") or {}).items()):
        _add_line(f"cf {key}", val)

    vector_doc = "\n".join(lines)
    if len(vector_doc) > 4000:
        vector_doc = vector_doc[:4000]
    return vector_doc

# ╭────────────────────────────────────────────────────────────────────────────╮
# │ Build a compact, LLM-friendly bundle per stock                             │
# ╰────────────────────────────────────────────────────────────────────────────╯
def _bundle_for_stock(name: str, master: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Build a compact, LLM-friendly bundle for one stock.
    - Pulls View Stock session caches (TTM snapshot, KPI/CAGR/CF)
    - Falls back to reconstructing TTM from last 4 quarters if cache is absent
    - Adds momentum features from data/ohlc/<Safe_Name>.csv
    """
    rows = master[master["Name"] == name].copy()
    if rows.empty:
        return None

    price    = _current_price(rows)
    bucket0  = _bucket_of(rows)
    industry = _industry_of(rows)

    # Read View Stock caches (ttm_dict_/syn_idx_/bucket_) + KPI/CAGR/CF
    view = _ttm_snapshot_from_view(name)

    # TTM fallback if View page hasn't saved one yet this session
    view_ttm_dict_raw = view.get("ttm_dict")
    view_ttm_dict = dict(view_ttm_dict_raw) if isinstance(view_ttm_dict_raw, dict) else {}
    ttm_dict: Dict[str, Any] = dict(view_ttm_dict)
    if not ttm_dict:
        fallback_ttm = _reconstruct_ttm_from_quarters(
            rows, view.get("bucket_hint") or bucket0
        )
        if fallback_ttm:
            ttm_dict.update(fallback_ttm)

    syn_idx: Dict[str, str] = dict(view.get("syn_idx") or {})
    ttm_kpi_card: Dict[str, Any] = dict(view.get("ttm_kpi") or {})
    cagr_card: Dict[str, Any] = dict(view.get("cagr") or {})
    cf_card: Dict[str, Any] = dict(view.get("cf") or {})
    ttm_kpi_vals: Dict[str, Any] = (ttm_kpi_card.get("values") or {})

    def _parse_float(v: Any) -> Optional[float]:
        try:
            f = float(v)
            return f if math.isfinite(f) else None
        except Exception:
            return None

    def _value_from_dict(values: Dict[str, Any], label: str) -> Optional[float]:
        if not values:
            return None
        direct = _parse_float(values.get(label))
        if direct is not None:
            return direct
        s = str(label).strip().lower()
        candidates = [
            s,
            s.replace(" (%)", ""), s + " (%)",
            s.replace(" (x)",  ""), s + " (x)",
            s.replace(" (×)",  ""), s + " (×)",
        ]
        for cand in candidates:
            canon = syn_idx.get(cand, cand)
            if canon in values:
                parsed = _parse_float(values.get(canon))
                if parsed is not None:
                    return parsed
        return None

    calc_ready = calc is not None and hasattr(calc, "build_summary_table")
    if calc_ready:
        core_labels = [
            "P/E (×)",
            "P/B (×)",
            "EV/EBITDA (×)",
            "FCF Yield (%)",
            "Dividend Yield (%)",
            "Payout Ratio (%)",
            "Interest Coverage (×)",
            "Average Cost of Debt (%)",
            "FCF Margin (%)",
        ]
        view_missing = [lab for lab in core_labels if _value_from_dict(view_ttm_dict, lab) is None]
        if (not view_ttm_dict) or view_missing:
            annual_df, quarterly_df = _split_annual_quarter(rows)
            if not annual_df.empty or not quarterly_df.empty:
                bucket_for_calc = view.get("bucket_hint") or bucket0 or "General"
                try:
                    sum_df = calc.build_summary_table(
                        annual_df=annual_df,
                        quarterly_df=quarterly_df,
                        bucket=bucket_for_calc,
                        include_ttm=True,
                        price_fallback=price,
                    )
                except Exception:
                    sum_df = None
                if sum_df is not None and not sum_df.empty and "Metric" in sum_df.columns:
                    ttm_cols = [
                        c for c in sum_df.columns
                        if isinstance(c, str) and c.upper().startswith("TTM")
                    ]
                    if ttm_cols:
                        def _ttm_col_key(col: str) -> tuple[int, str]:
                            m = re.search(r"(\d{4})", str(col))
                            year = int(m.group(1)) if m else -1
                            return (year, str(col))

                        ttm_col = max(ttm_cols, key=_ttm_col_key)
                        try:
                            ttm_series = pd.to_numeric(
                                sum_df.set_index("Metric")[ttm_col],
                                errors="coerce",
                            )
                        except Exception:
                            ttm_series = pd.Series(dtype="float64")
                        if not ttm_series.empty:
                            for metric, val in ttm_series.dropna().items():
                                parsed_val = _parse_float(val)
                                if parsed_val is None:
                                    continue
                                metric_key = str(metric)
                                existing_val = _parse_float(ttm_dict.get(metric_key))
                                if existing_val is not None:
                                    continue
                                ttm_dict[metric_key] = parsed_val

    # Unit/synonym-tolerant accessor into TTM dict, with KPI fallback
    def _get_ttm(label: str) -> Optional[float]:
        s = str(label).strip().lower()
        candidates = [
            s,
            s.replace(" (%)", ""), s + " (%)",
            s.replace(" (x)",  ""), s + " (x)",
            s.replace(" (×)",  ""), s + " (×)",
        ]

        direct = _parse_float(ttm_dict.get(label))
        if direct is not None:
            return direct

        for cand in candidates:
            canon = syn_idx.get(cand, cand)
            if canon in ttm_dict:
                via_dict = _parse_float(ttm_dict.get(canon))
                if via_dict is not None:
                    return via_dict

        for cand in candidates:
            canon = syn_idx.get(cand, cand)
            if canon in ttm_kpi_vals:
                via_kpi = _parse_float(ttm_kpi_vals.get(canon))
                if via_kpi is not None:
                    return via_kpi

        fallback_direct = _parse_float(ttm_kpi_vals.get(label))
        if fallback_direct is not None:
            return fallback_direct

        return None

    # Core valuation / dividend / risk pulled from TTM snapshot (if present)
    pe     = _get_ttm("P/E (×)")
    pb     = _get_ttm("P/B (×)")
    ev_e   = _get_ttm("EV/EBITDA (×)")
    fcfy   = _get_ttm("FCF Yield (%)")
    dy     = _get_ttm("Dividend Yield (%)")
    payout = _get_ttm("Payout Ratio (%)")
    icr    = _get_ttm("Interest Coverage (×)")
    acd    = _get_ttm("Average Cost of Debt (%)")
    fcfm   = _get_ttm("FCF Margin (%)")

    metrics_block = {
        "pe_x": pe,
        "pb_x": pb,
        "ev_ebitda_x": ev_e,
        "fcf_yield_pct": fcfy,
        "dividend_yield_pct": dy,
        "payout_ratio_pct": payout,
        "interest_coverage_x": icr,
        "avg_cost_debt_pct": acd,
        "fcf_margin_pct": fcfm,
    }

    # Momentum features from OHLC CSV (if available)
    px  = _load_ohlc(name, _data_etag())
    mom = _momentum_features(px)

    bucket_final = (view.get("bucket_hint") or bucket0 or "General")
    vector_doc = _compose_vector_doc(
        name=name,
        industry=industry,
        bucket=bucket_final,
        price=price,
        metrics=metrics_block,
        momentum=mom,
        ttm_dict=ttm_dict,
        ttm_kpi=ttm_kpi_card,
        cagr_card=cagr_card,
        cf_card=cf_card,
    )

    return {
        "name": name,
        "industry": industry,
        "bucket": bucket_final,
        "price_now": price,

        # Handy top-level metrics frequently used by prompts
        "metrics": metrics_block,

        # Momentum snapshot
        "momentum": mom,

        # Expose the merged TTM snapshot (from View cache or reconstructed fallback)
        "ttm": {
            "values": ttm_dict,                 # { "P/E (×)": 12.3, "Dividend Yield (%)": 4.1, ... }
            "synonyms": syn_idx,                # lowercased variants -> canonical labels
            "kpi_cards": ttm_kpi_card,          # {'period': 'TTM 2025', 'values': {...}}
            "cagr_cards": cagr_card,            # {'N': 5, 'end_basis': 'TTM', 'values_pct': {...}}
            "cf_cards": cf_card,                # {'basis': 'TTM'|'FY', 'values': {...}}
        },

        # Keep the raw view dump as well (useful for debugging or richer prompts)
        "view": view,

        # Pre-computed natural-language-ish document for vector search / retrieval
        "vector_doc": vector_doc,
    }

@st.cache_data(show_spinner=False)
def _build_catalog(max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    if df_master.empty:
        return []
    names = sorted(df_master["Name"].dropna().astype(str).unique().tolist())
    if max_rows is not None:
        names = names[:max_rows]
    out: List[Dict[str,Any]] = []
    for n in names:
        b = _bundle_for_stock(n, df_master)
        if b:
            out.append(b)
    return out

def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[0-9A-Za-z_\.]+", str(text).lower())

def _build_vector_store(catalog: List[Dict[str, Any]]) -> Dict[str, Any]:
    docs: List[str] = []
    names: List[str] = []
    for stock in catalog:
        doc = str(stock.get("vector_doc") or "").strip()
        name = stock.get("name")
        if not doc or not name:
            continue
        docs.append(doc)
        names.append(str(name))

    if not docs:
        return {}

    tokenized = [_tokenize(doc) for doc in docs]
    vocab = sorted({tok for toks in tokenized for tok in toks})
    if not vocab:
        return {}

    index = {tok: idx for idx, tok in enumerate(vocab)}
    n_docs = len(docs)

    doc_freq: Dict[str, int] = {}
    for toks in tokenized:
        for tok in set(toks):
            doc_freq[tok] = doc_freq.get(tok, 0) + 1

    idf = np.zeros(len(vocab), dtype="float64")
    for tok, idx in index.items():
        df_count = doc_freq.get(tok, 0)
        idf[idx] = math.log((1.0 + n_docs) / (1.0 + df_count)) + 1.0

    matrix = np.zeros((n_docs, len(vocab)), dtype="float64")
    for i, toks in enumerate(tokenized):
        counts = Counter(toks)
        total = float(sum(counts.values())) or 1.0
        row = matrix[i]
        for tok, cnt in counts.items():
            j = index.get(tok)
            if j is None:
                continue
            tf = cnt / total
            row[j] = tf * idf[j]
        norm = np.linalg.norm(row)
        if norm:
            row /= norm

    return {
        "names": names,
        "documents": docs,
        "vocab": vocab,
        "index": index,
        "idf": idf,
        "matrix": matrix,
    }

def _vector_search(query: str, store: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
    if not store or not query or not store.get("vocab"):
        return []

    tokens = _tokenize(query)
    if not tokens:
        return []

    counts = Counter(tokens)
    total = float(sum(counts.values())) or 1.0

    vocab_size = len(store.get("vocab") or [])
    if vocab_size == 0:
        return []

    vec = np.zeros(vocab_size, dtype="float64")
    index = store.get("index") or {}
    idf = store.get("idf")
    if idf is None or len(idf) != vocab_size:
        return []

    for tok, cnt in counts.items():
        j = index.get(tok)
        if j is None:
            continue
        tf = cnt / total
        vec[j] = tf * float(idf[j])

    norm = np.linalg.norm(vec)
    if not norm:
        return []
    vec /= norm

    matrix = store.get("matrix")
    if matrix is None or not hasattr(matrix, "dot"):
        return []

    sims = matrix.dot(vec)
    if sims.size == 0:
        return []

    order = np.argsort(-sims)
    hits: List[Dict[str, Any]] = []
    for idx in order[:top_k]:
        score = float(sims[idx])
        if score <= 0:
            continue
        doc = store["documents"][idx]
        snippet = re.sub(r"\s+", " ", doc).strip()
        hits.append({
            "name": store["names"][idx],
            "score": score,
            "snippet": snippet[:200],  # reduced from 400 to 200
        })
    return hits

# ╭────────────────────────────────────────────────────────────────────────────╮
# │ Focus resolution: from message text, then View-page selection, then top N  │
# ╰────────────────────────────────────────────────────────────────────────────╯
def _resolve_focus(msg: str, universe: List[str]) -> List[str]:
    if not universe:
        return []

    want: List[str] = []

    def _append(candidate: Optional[str]) -> None:
        if not candidate or candidate in want:
            return
        want.append(candidate)

    text = (msg or "")
    for n in universe:
        if re.search(rf"\b{re.escape(str(n))}\b", text, flags=re.I):
            _append(n)

    # also accept comma lists like "Focus: A, B"
    for seg in re.findall(r"(?:focus|tickers?)\s*:\s*([^\n]+)", text, flags=re.I):
        for tok in seg.split(","):
            t = tok.strip()
            if not t:
                continue
            for n in universe:
                if t.lower() in n.lower():
                    _append(n)

    # fallback to current selection in View Stock (multiselect)
    sel = st.session_state.get("sel_names") or []
    for s in sel:
        if s in universe:
            _append(s)

    return want

def _prune_bundle(catalog: List[Dict[str,Any]], focus: List[str], cap: int) -> List[Dict[str,Any]]:
    if not catalog:
        return []
    if focus:
        order = {name: idx for idx, name in enumerate(focus)}
        pri = [c for c in catalog if c.get("name") in focus]
        pri.sort(key=lambda c: order.get(c.get("name"), len(order)))
        rest = [c for c in catalog if c.get("name") not in focus]
        return (pri + rest)[:cap]
    return catalog[:cap]

def _trim_values_dict(values: Dict[str, Any], core_keys: List[str], max_keys: int) -> Dict[str, Any]:
    """
    Take a dict of metrics and:
      • Always keep core_keys if present
      • Then fill up with other keys until max_keys
    """
    if not isinstance(values, dict) or max_keys <= 0:
        return {}

    trimmed: Dict[str, Any] = {}

    # 1) Keep the core keys first (if present)
    for k in core_keys:
        if k in values and len(trimmed) < max_keys:
            trimmed[k] = values[k]

    # 2) Fill the remaining slots with whatever is left
    for k, v in values.items():
        if len(trimmed) >= max_keys:
            break
        if k in trimmed:
            continue
        trimmed[k] = v

    return trimmed

def _thin_stock_bundle(stock: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reduce each stock bundle to the essentials before sending to OpenAI.
    This keeps the JSON small enough to stay within TPM / context limits.
    """
    name      = stock.get("name")
    industry  = stock.get("industry")
    bucket    = stock.get("bucket")
    price_now = stock.get("price_now")

    metrics   = stock.get("metrics") or {}
    momentum  = stock.get("momentum") or {}
    ttm       = stock.get("ttm") or {}

    ttm_values     = ttm.get("values") or {}
    ttm_kpi        = ttm.get("kpi_cards") or {}
    ttm_kpi_vals   = ttm_kpi.get("values") or {}
    cagr_cards     = ttm.get("cagr_cards") or {}
    cf_cards       = ttm.get("cf_cards") or {}

    # Core valuation / quality metrics we try to keep if present
    core_ttm_keys = [
        "P/E (×)", "P/B (×)", "EV/EBITDA (×)",
        "Dividend Yield (%)", "Payout Ratio (%)",
        "FCF Yield (%)", "FCF Margin (%)",
        "Interest Coverage (×)", "Average Cost of Debt (%)",
        "Net Margin (%)", "ROE (%)", "ROA (%)",
    ]

    # Trim big dicts to a reasonable size
    thin_ttm_values = _trim_values_dict(ttm_values, core_ttm_keys, max_keys=60)
    thin_kpi_values = _trim_values_dict(ttm_kpi_vals, [], max_keys=40)

    if isinstance(cagr_cards, dict):
        cagr_values_pct = cagr_cards.get("values_pct") or {}
    else:
        cagr_values_pct = {}
    thin_cagr_values = _trim_values_dict(cagr_values_pct, [], max_keys=30)

    if isinstance(cf_cards, dict):
        cf_values = cf_cards.get("values") or {}
    else:
        cf_values = {}
    thin_cf_values = _trim_values_dict(cf_values, [], max_keys=30)

    thin_ttm = {
        "values": thin_ttm_values,
        "kpi_summary": {
            "period": ttm_kpi.get("period") if isinstance(ttm_kpi, dict) else None,
            "values": thin_kpi_values,
        },
        "cagr_summary": {
            "N": cagr_cards.get("N") if isinstance(cagr_cards, dict) else None,
            "end_basis": cagr_cards.get("end_basis") if isinstance(cagr_cards, dict) else None,
            "values_pct": thin_cagr_values,
        },
        "cf_summary": {
            "basis": cf_cards.get("basis") if isinstance(cf_cards, dict) else None,
            "values": thin_cf_values,
        },
    }

    return {
        "name": name,
        "industry": industry,
        "bucket": bucket,
        "price_now": price_now,
        "metrics": metrics,
        "momentum": momentum,
        "ttm": thin_ttm,
    }

# ╭────────────────────────────────────────────────────────────────────────────╮
# │ OpenAI call (chat.completions, streaming). No 'input_json' content type.   │
# ╰────────────────────────────────────────────────────────────────────────────╯
SYSTEM_PROMPT = (
    "You are a conservative equity analyst for Malaysian stocks.\n"
    "Task:\n"
    " • Rank the best 5–10 actionable ideas from the provided catalog.\n"
    " • Explain rationale with fundamentals (growth/quality, dividends, risk) and momentum (3/6/12M, MA50/200, off-high).\n"
    " • Add valuation context (PE / PB / EV/EBITDA or FCF yield) if present.\n"
    " • Flag red-flags (weak ICR, high cost of debt, negative FCF margin, high payout with low growth, momentum < MA200 or far off-high).\n"
    " • Use retrieval.matches for semantic suggestions when tickers are not explicit.\n"
    " • Give 3 next steps (what to validate before buying).\n"
    "Be concise, bullet-first, and data-referenced using the JSON bundle."
)

def _chat_openai(api_key: str, model: str, messages: List[Dict[str,Any]], temperature: float = 0.2):
    from openai import OpenAI  # openai>=1.x
    client = OpenAI(api_key=api_key)
    stream = client.chat.completions.create(
        model=model, temperature=float(temperature),
        messages=messages, stream=True,
    )
    placeholder = st.empty()
    out = io.StringIO()
    for chunk in stream:
        try:
            delta = chunk.choices[0].delta.content or ""
        except Exception:
            delta = ""
        if delta:
            out.write(delta)
            placeholder.markdown(out.getvalue())
    return out.getvalue().strip()

# ╭────────────────────────────────────────────────────────────────────────────╮
# │ Chat state + UI                                                            │
# ╰────────────────────────────────────────────────────────────────────────────╯
if "ai_chat" not in st.session_state:
    st.session_state["ai_chat"] = []

# KPIs up top (quick sanity)
_view_cache_count = (
    len(st.session_state.get("TTM_KPI_SYNC") or {})
    + len(st.session_state.get("CAGR_SYNC") or {})
    + len(st.session_state.get("CF_SYNC") or {})
)

render_stat_cards(
    [
        {"label":"Universe", "value": f"{len(names_all):,}", "badge":"Stocks"},
        {"label":"Momentum files", "value": f"{sum(os.path.exists(_ohlc_path(n)) for n in names_all):,}", "badge":"CSV"},
        {"label":"View caches (session)", "value": f"{_view_cache_count:,}", "badge":"items"},
    ],
    columns=3,
)

for msg in st.session_state["ai_chat"]:
    st.chat_message(msg["role"]).write(msg["content"])

def _prior_natural_history(limit: int = 8) -> List[Dict[str, str]]:
    """
    Keep the last few natural chat turns but DROP any prior user turns
    that already embedded a DATA_BUNDLE_JSON. This prevents the model
    from seeing giant JSON bundles multiple times (which can cause it to
    repeat itself).
    """
    hist: List[Dict[str, str]] = []
    for m in st.session_state["ai_chat"]:
        if m.get("role") not in ("user","assistant"):
            continue
        if m.get("role") == "user" and "DATA_BUNDLE_JSON:" in str(m.get("content","")):
            continue
        hist.append({"role": m["role"], "content": m["content"]})
    return hist[-limit:]

user_text = st.chat_input("Ask for a ranked shortlist, ask about specific tickers, or request a style…")
if user_text:
    # 1) Show user bubble
    st.chat_message("user").write(user_text)
    st.session_state["ai_chat"].append({"role":"user","content": user_text})

    # 2) Build data bundle (focus first, then fill to cap)
    catalog_universe = _build_catalog(None)

    # Build vector store from the full universe (uses vector_doc internally)
    vector_store = _build_vector_store(catalog_universe)
    store_size = len(vector_store.get("names", [])) if vector_store else 0

    # Effective cap for this LLM call (respect slider but clamp to _MAX_LLM_STOCKS)
    try:
        cap_setting = int(max_catalog)
    except Exception:
        cap_setting = len(catalog_universe)
    effective_cap = min(cap_setting, _MAX_LLM_STOCKS)

    top_k_hits = min(15, max(5, min(effective_cap, store_size))) if store_size else 0
    if top_k_hits > store_size:
        top_k_hits = store_size

    vector_hits = _vector_search(user_text, vector_store, top_k=top_k_hits) if top_k_hits else []

    focus = _resolve_focus(user_text, [c["name"] for c in catalog_universe])
    for hit in vector_hits:
        name = hit.get("name")
        if name and name not in focus:
            focus.append(name)

    # Prune to the focus + cap, then thin each bundle before sending to OpenAI
    catalog = _prune_bundle(catalog_universe, focus, effective_cap)
    stocks_payload = [_thin_stock_bundle(s) for s in catalog]

    payload = {
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "focus": focus,
        "universe_count": len(catalog_universe),
        "packaged_count": len(stocks_payload),
        "stocks": stocks_payload,
    }

    payload["retrieval"] = {
        "query": user_text,
        "vector_store_size": store_size,
        "matches": [
            {
                "name": hit.get("name"),
                "score": round(float(hit.get("score", 0.0)), 4),
                "snippet": hit.get("snippet"),
            }
            for hit in vector_hits
        ],
    }

    # 3) Compose messages (embed JSON as fenced text — no special block types)
    user_with_data = (
        user_text.strip()
        + "\n\n"
        + f"Focus inferred: {', '.join(focus) if focus else '(none)'}\n"
        + f"Universe packaged: {len(catalog)} of {len(catalog_universe)}\n"
        + "DATA_BUNDLE_JSON:\n```json\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n```"
    )

    # 4) Call OpenAI
    api_key_real = api_key or _load_saved_api_key()
    if not api_key_real:
        st.chat_message("assistant").write("Please add your OpenAI API key in the sidebar.")
    elif not catalog:
        st.chat_message("assistant").write("No data available to analyze. Add stocks and/or refresh View Stock first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    model_use = (model or "").strip() or "gpt-4o"
                    answer = _chat_openai(
                        api_key=api_key_real,
                        model=model_use,
                        messages=[
                            {"role":"system","content": SYSTEM_PROMPT},
                            *_prior_natural_history(limit=6),  # slightly shorter history
                            {"role":"user","content": user_with_data},
                        ],
                        temperature=float(temp),
                    )
                except Exception as e:
                    st.error(f"OpenAI error: {e}")
                    answer = ""

                # ⬇️ DO NOT st.write(answer) here — the stream already rendered it
                if answer:
                    st.session_state["ai_chat"].append({"role":"assistant","content": answer})

# Debug expander (optional)
with st.expander("🔎 Debug / last packaged data (for this run)"):
    try:
        st.json(payload)  # will exist only after a prompt is sent
    except Exception:
        st.caption("Send a message to see the packaged data preview here.")
