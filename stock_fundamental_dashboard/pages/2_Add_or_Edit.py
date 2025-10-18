# pages/2_Add_or_Edit.py

import os
import sys
import time
import pandas as pd
import streamlit as st
import json, math

_fragment = None  # disable fragments to avoid form conflicts & keep things snappy

# ---- Auth ----
from auth_gate import require_auth
require_auth()

from utils.ui import (
    setup_page,
    render_kpi_text_grid,
    section,
    render_stat_cards,
    render_page_title,
)
setup_page("Add / Edit Stock")  # on 2_Add_or_Edit.py
render_page_title("Add / Edit Stock")

from datetime import datetime, timezone 

# Show a toast after rerun when settings were saved
if st.session_state.pop("__settings_saved", False):
    st.toast("Stock settings saved & synced to Quick Edit.", icon="✅")

# ---- Pathing so io_helpers/calculations/rules can import nicely ----
_THIS = os.path.dirname(__file__)
_PARENT = os.path.abspath(os.path.join(_THIS, ".."))
_GRANDP = os.path.abspath(os.path.join(_THIS, "..", ".."))
for p in (_PARENT, _GRANDP):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---- Shared modules ----
try:
    import io_helpers, calculations, rules
except ModuleNotFoundError:
    from utils import io_helpers, calculations, rules  # fallback

try:
    import config
except ModuleNotFoundError:
    from utils import config  # type: ignore

# ---- Helper: does this bucket support "Three Fees"? ----
def _bucket_supports_three_fees(bucket: str) -> bool:
    """True if the bucket includes 'Three Fees' in INDUSTRY_FORM_CATEGORIES."""
    cats = (config.INDUSTRY_FORM_CATEGORIES.get(bucket)
            or config.INDUSTRY_FORM_CATEGORIES.get("General", {}))
    for sec_items in (cats or {}).values():
        for f in (sec_items or []):
            if (f or {}).get("key") == "Three Fees":
                return True
    return False

# (Optional) convenience if you ever need the current bucket without passing it around
def _current_bucket() -> str:
    # prefer explicit IndustryBucket; fallback to Industry; else General
    b = st.session_state.get("IndustryBucket") or st.session_state.get("Industry") or "General"
    return _canonical_bucket(b)

# ---- Global "data version" for cross-page cache busting ----
_VERSION_FILE = os.path.join(_GRANDP, ".data.version")

# ---- Audit log helpers (local, lightweight) ----
_AUDIT_FILE = os.path.join(_GRANDP, "data", "audit_log.jsonl")

def _diff_dict(before: dict | None, after: dict | None, *, allowed: set[str] | None = None) -> dict:
    before = before or {}
    after = after or {}
    allowed = allowed or (set(before.keys()) | set(after.keys()))
    changes: dict[str, list] = {}
    for k in allowed:
        b = before.get(k, None)
        a = after.get(k, None)
        # Treat NaNs as None for stable diffs
        try:
            if pd.isna(b):
                b = None
        except Exception:
            pass
        try:
            if pd.isna(a):
                a = None
        except Exception:
            pass
        if b != a:
            changes[k] = [b, a]
    return changes

def _audit_log_event(
    action: str, *,
    name: str,
    scope: str,
    year: int | None = None,
    quarter: str | None = None,
    changes: dict | None = None,
    before: dict | None = None,
    after: dict | None = None,
    source: str = ""
) -> None:
    try:
        os.makedirs(os.path.dirname(_AUDIT_FILE), exist_ok=True)
        import json as _json
        ev = {
            "ts": datetime.now(timezone.utc).isoformat(),  # tz-aware ISO
            "action": action,
            "scope": scope,
            "name": name,
            "year": year,
            "quarter": quarter,
            "changes": changes or {},
            "source": source,
        }
        with open(_AUDIT_FILE, "a", encoding="utf-8") as f:
            f.write(_json.dumps(ev, ensure_ascii=False) + "\n")
    except Exception:
        # non-fatal
        pass

def _bump_data_etag():
    try:
        open(_VERSION_FILE, "a").close()
        os.utime(_VERSION_FILE, None)
    except Exception:
        pass

def _data_etag() -> int:
    try:
        return int(os.stat(_VERSION_FILE).st_mtime_ns)
    except Exception:
        return 0

def _now_iso() -> str:
    """Local time ISO-ish timestamp for audit trail."""
    try:
        return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        # fallback if timezone not available
        return time.strftime("%Y-%m-%d %H:%M:%S")
       
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
    # bump the same etag so other pages see the change
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

def set_fd_eps_rate(v: float | None) -> None:
    s = _load_settings()
    s["fd_eps_rate"] = (None if v in (None, "", "—") else float(v))
    _save_settings(s)
    
def get_epf_rate() -> float | None:
    v = _load_settings().get("epf_rate")
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def set_epf_rate(v: float | None) -> None:
    s = _load_settings()
    s["epf_rate"] = (None if v in (None, "", "—") else float(v))
    _save_settings(s)
 
# ---- Cached IO ----
@st.cache_data(show_spinner=False)
def get_df():
    df0 = io_helpers.load_data()
    return df0 if df0 is not None else pd.DataFrame()

def save_df(df: pd.DataFrame):
    io_helpers.save_data(df)
    get_df.clear()
    _bump_data_etag()
    st.session_state["data_version"] = st.session_state.get("data_version", 0) + 1

# ---- UI helpers ----
def calc_number_input(
    label: str,
    value: float | None,
    key: str,
    decimals: int = 0,
    placeholder: str | None = None,
):
    """
    Text-input wrapper so empty == None (not 0).
    - Shows current numeric value formatted; blank shows empty.
    - Writes a parsed float (or None) back to st.session_state[key].

    IMPORTANT: We do NOT pass `value=` to st.text_input when also using session_state,
    to avoid the warning: "widget ... was created with a default value but also had
    its value set via the Session State API."
    """
    tkey = f"{key}_txt"

    def _fmt(v):
        try:
            f = float(v)
            s = f"{f:.{decimals}f}" if decimals else f"{f:.0f}"
            return s.rstrip("0").rstrip(".") if decimals else s
        except Exception:
            return ""

    # Seed the paired text key only if it doesn't exist yet
    if tkey not in st.session_state:
        st.session_state[tkey] = _fmt(value)

    # ✅ Do NOT pass `value=` here — the widget will read from session_state via its key
    raw = st.text_input(label, key=tkey, placeholder=placeholder)

    s = (raw or "").replace(",", "").strip()
    if s == "":
        st.session_state[key] = None
        return None

    try:
        f = float(s)
        f = round(f, decimals) if decimals else float(f)
        st.session_state[key] = f
        return f
    except Exception:
        return st.session_state.get(key, None)

def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

# ---- Autofill suppression helpers (keep computed fields empty after Clear) ----
def _suppress_flag_key(scope: str, year: int, quarter: str | None = None) -> str:
    return f"__suppress_autofill_{scope}_{year}" + (f"_{quarter}" if quarter else "")

def mark_autofill_suppressed(scope: str, year: int, quarter: str | None = None) -> None:
    st.session_state[_suppress_flag_key(scope, year, quarter)] = True

def is_autofill_suppressed(scope: str, year: int, quarter: str | None = None) -> bool:
    return bool(st.session_state.get(_suppress_flag_key(scope, year, quarter), False))

def unsuppress_if_user_typed(scope: str, year: int, quarter: str | None, flat_fields: list[dict]) -> None:
    """Lift suppression as soon as the user types any value for that period."""
    if not is_autofill_suppressed(scope, year, quarter):
        return
    for f in flat_fields:
        k = f.get("key")
        if not k:
            continue
        ssk = f"{k}_{year}_annual" if scope == "annual" else f"{k}_{year}_{quarter}_q"
        if st.session_state.get(ssk) is not None:
            st.session_state[_suppress_flag_key(scope, year, quarter)] = False
            break

def _label_with_unit(lbl: str | None, unit: str | None) -> str:
    if not lbl:
        return ""
    if not unit:
        return lbl
    return lbl if unit.lower() in lbl.lower() else f"{lbl} ({unit})"

def _decimals_for_unit(unit: str | None, key: str = "") -> int:
    if unit == "%":
        return 2
    if unit in ("RM", "RM/t", "RM/boe"):
        return 2
    if "Price" in key:
        return 4
    return 0

def _allowed_meta_fields(scope: str) -> set[str]:
    """
    Meta fields that are allowed to be diffed/saved alongside financial keys.
    - Annual rows should never include 'Quarter' in diffs.
    - Quarterly rows should include 'Quarter'.
    """
    base = {"Industry", "IndustryBucket", "Year", "IsQuarter"}
    return base if scope == "annual" else (base | {"Quarter"})

def _qify(keys: set[str]) -> set[str]:
    return {("Q_" + k if not k.startswith("Q_") else k) for k in keys}

def _derived_keys_for(bucket: str, scope: str) -> set[str]:
    base = set(_DERIVED_COMMON)
    base |= _DERIVED_BY_BUCKET.get(bucket, set())
    if bucket == "Banking":
        base |= _BANK_DERIVED_ANNUAL
    if scope == "quarterly":
        base = _qify(base)
    return base

# ---- (NEW) Quick Edit helpers ----
def _bucket_allowed_keys(bucket: str, quarterly: bool = False) -> set[str]:
    # Thin wrapper so Quick Edit can reuse your cached _allowed_keys_for_bucket
    return _allowed_keys_for_bucket(bucket, quarterly=quarterly)

# Optional: lightweight placeholder hints (used when config field has no "help")
_PLACEHOLDERS_BASE = {
    "Revenue": "Top-line sales.",
    "CostOfSales": "Direct costs (COGS).",
    "Gross Profit": "Revenue − COGS.",
    "Operating Profit": "EBIT (operating).",
    "EBITDA": "EBIT + depreciation/amort.",
    "DepPPE": "PPE depreciation.",
    "DepROU": "ROU / lease depreciation.",
    "DepInvProp": "Investment property depreciation.",
    "DepAmort": "Total depreciation & amortisation.",
    "Three Fees": "Selling + Admin + Interest expense.",
    "Interest Expense": "Finance costs.",
    "Net Profit": "Bottom-line profit.",
    "Total Borrowings": "Gross debt.",
    "Cash & Cash Equivalents": "Cash and near-cash.",
    "Units Outstanding": "Units/Shares outstanding.",
    "Equity": "Shareholders’ equity.",
    "Capex": "Capital expenditures.",
    # Banking
    "Gross Loans": "Loan book (gross).",
    "Deposits": "Total customer deposits.",
    "Demand Deposits": "Current accounts (CASA).",
    "Savings Deposits": "Savings (CASA).",
    "Time/Fixed Deposits": "Term deposits.",
    "NII (incl Islamic)": "Interest income − expense (+ Islamic).",
    "Earning Assets": "Interest-earning assets.",
    "TP_Bank_NIM_Num": "Manual override (NIM numerator).",
    "TP_Bank_NIM_Den": "Manual override (NIM denominator).",
    "NIM": "Net Interest Margin (%).",
}

# --- Banking TP mapping: canonical override key -> base source key
_BANK_TP_CANON_TO_BASE = {
    "TP_Bank_NIM_Num": "NII (incl Islamic)",
    "TP_Bank_NIM_Den": "Average Earning Assets",
    "TP_Bank_CIR_Num": "Operating Expenses",
    "TP_Bank_CIR_Den": "Operating Income",
    "TP_Bank_Lev_Num": "Total Assets",
    "TP_Bank_Lev_Den": "Equity",
    "TP_Bank_CFO_Den": "Operating Income",
}

_PLACEHOLDERS_BY_BUCKET = {
    "REITs": {
        "Avg Cost of Debt": "Interest expense ÷ borrowings (%).",
        "NAV per Unit": "Equity ÷ units.",
    },
    "Transportation/Logistics": {
        "ASK": "Available Seat Kilometres.",
        "RPK": "Revenue Passenger Kilometres.",
        "Load Factor": "RPK ÷ ASK (%).",
        "Yield per km": "Revenue ÷ RPK.",
    },
    "Tech": {
        "R&D Expense": "Research & development spend.",
        "R&D Intensity": "R&D ÷ Revenue (%).",
    },
    "Utilities": {"Capex to Revenue": "Capex ÷ Revenue (%)."},
    "Telco": {"Capex to Revenue": "Capex ÷ Revenue (%)."},
    "Property": {"Net Debt": "Borrowings − cash."},
    "Banking": {
        "Loan-to-Deposit Ratio": "Gross loans ÷ deposits (%).",
        "CASA Ratio": "Demand + Savings ÷ Deposits (%).",
        "Average Earning Assets": "Avg of current & prior EA.",
    },
}

# === (NEW) Turn off CIR for selected buckets ===
try:
    CIR_DISABLED_BUCKETS = set(getattr(config, "CIR_DISABLED_BUCKETS", {"Banking"}))
except Exception:
    CIR_DISABLED_BUCKETS = {"Banking"}

def _cir_allowed(bucket: str) -> bool:
    return str(bucket) not in CIR_DISABLED_BUCKETS

def _strip_bank_cir(cats: dict, bucket: str, quarterly: bool) -> dict:
    """Remove CIR override fields from the UI categories when disabled."""
    if _cir_allowed(bucket):
        return cats
    kill = {"TP_Bank_CIR_Num", "TP_Bank_CIR_Den"}
    if quarterly:
        kill = {"Q_" + k for k in kill}
    out = {}
    for sec, items in (cats or {}).items():
        out[sec] = [f for f in (items or []) if f.get("key") not in kill]
    return out

def _placeholder_for(key: str, bucket: str, quarterly: bool = False) -> str | None:
    base_key = key[2:] if (quarterly and key.startswith("Q_")) else key
    if bucket in _PLACEHOLDERS_BY_BUCKET and base_key in _PLACEHOLDERS_BY_BUCKET[bucket]:
        return _PLACEHOLDERS_BY_BUCKET[bucket][base_key]
    return _PLACEHOLDERS_BASE.get(base_key)

def _link_bank_tp_overrides_in_state(*, scope: str, year: int, quarter: str | None = None):
    """
    Ensure canonical TP override fields (TP_Bank_*) are auto-filled from their base
    fields when blank. We only touch canonical keys.
    """
    is_q = (scope == "quarterly")

    def k(key: str) -> str:
        return (f"Q_{key}_{year}_{quarter}_q") if is_q else (f"{key}_{year}_annual")

    for canon, src in _BANK_TP_CANON_TO_BASE.items():
        ck = k(canon)
        sk = k(src)

        cv = _try_float(st.session_state.get(ck))
        sv = _try_float(st.session_state.get(sk))

        # If override is blank but base has a value -> copy base into override
        if cv is None and sv is not None:
            st.session_state[ck] = sv
            tkey = ck + "_txt"  # keep the paired text widget in sync if present
            if tkey in st.session_state:
                st.session_state[tkey] = str(sv).rstrip("0").rstrip(".")

# ---- Categories (from your config) ----
_SECTION_ORDER = ["Income Statement", "Balance Sheet", "Cash Flow", "Other", "Industry KPIs"]

def _strip_three_fees(cats: dict, bucket: str, quarterly: bool = False) -> dict:
    """
    Remove 'Three Fees' only for buckets that do NOT include it in config.
    For buckets where config includes it, keep it visible in the form.
    """
    if not cats:
        return cats or {}

    # If config says this bucket supports Three Fees, don't strip it from the UI
    if _bucket_supports_three_fees(bucket):
        return cats

    drop = {"Three Fees", "Q_Three Fees"}  # safe across scopes
    out = {}
    for sec, items in (cats or {}).items():
        cleaned = []
        for f in (items or []):
            k = (f or {}).get("key", "")
            if k not in drop:
                cleaned.append(f)
        out[sec] = cleaned
    return out

def _get_categories(bucket: str, quarterly: bool = False) -> dict:
    cats = (
        config.INDUSTRY_FORM_CATEGORIES.get(bucket)
        or config.INDUSTRY_FORM_CATEGORIES.get("General", {})
    )
    if not quarterly:
        # If you added the Three Fees stripper earlier, call it (safe if missing)
        if "_strip_three_fees" in globals():
            cats = _strip_three_fees(cats, bucket, quarterly=False)
        # NEW: strip CIR overrides
        cats = _strip_bank_cir(cats, bucket, quarterly=False)
        return cats

    # Prefer explicit quarterly categories if available; else auto-prefix Q_
    cats_q = getattr(config, "INDUSTRY_FORM_CATEGORIES_Q", {}).get(bucket)
    if cats_q is None:
        cats_q = {}
        for sec, items in cats.items():
            cats_q[sec] = [{**f, "key": f"Q_{f['key']}"} for f in items if f.get("key")]

    # If you added the Three Fees stripper earlier, call it (safe if missing)
    if "_strip_three_fees" in globals():
        cats_q = _strip_three_fees(cats_q, bucket, quarterly=True)
    # NEW: strip CIR overrides
    cats_q = _strip_bank_cir(cats_q, bucket, quarterly=True)
    return cats_q

def _flatten_categories(cat_dict: dict) -> list[dict]:
    out = []
    for s in _SECTION_ORDER:
        out.extend(cat_dict.get(s, []))
    for s in cat_dict:
        if s not in _SECTION_ORDER:
            out.extend(cat_dict[s])
    return out

@st.cache_data(show_spinner=False)
def _cats(bucket: str, quarterly: bool) -> list[dict]:
    return _flatten_categories(_get_categories(bucket, quarterly))

@st.cache_data(show_spinner=False)
def _allowed_keys_for_bucket(bucket: str, quarterly: bool = False) -> set[str]:
    return {f["key"] for f in _cats(bucket, quarterly) if f.get("key")}

# ---- Small numeric helpers ----
_Q_ORDER = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}

def _abs_or_none(x):
    v = _try_float(x)
    return None if v is None else abs(v)

def _sum_costs(*vals):
    parts = [abs(v) for v in vals if _try_float(v) is not None for v in [float(v)]]
    return sum(parts) if parts else None

def _try_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        return v if v == v else None
    except Exception:
        return None
    
# --- Loan loss reserve lookup (for Net Loans fallback) ---
_LLR_KEYS = (
    "Loan Loss Reserve",
    "Loan Loss Reserves",
    "Loan Loss Provision",
    "Allowance for ECL",
    "Allowance for Expected Credit Losses",
    "ECL Allowance",
)

# --- Banking field aliases (annual + quarterly via prefix) ---
_BANK_ALIASES = {
    "Net Loans": (
        "Net Loans",
        "Net Loans/Financing", "Net Loans Financing",
        "Net Financing", "Net Loans & Financing",
    ),
    "Gross Loans": (
        "Gross Loans",
        "Gross Loans/Financing", "Gross Loans & Financing",
    ),
    "FVOCI": (
        "FVOCI Investments",
        "Financial Investments at FVOCI",
        "Investment Securities at FVOCI",
        "Financial Assets at FVOCI",
    ),
    "AMORT": (
        "Amortised Cost Investments",
        "Financial Investments at Amortised Cost",
        "Investment Securities at Amortised Cost",
        "Financial Assets at Amortised Cost",
    ),
    "REVREPO": (
        "Reverse Repos",
        "Financial Assets Purchased under Resale/Reverse Repo",
        "Reverse Repo",
        "Assets Purchased under Resale/Reverse Repurchase",
    ),
    "FVTPL": (
        "FVTPL Investments",
        "Financial Assets at FVTPL",
    ),
    # NEW ⬇️
    "PLACEMENTS": (
        "Deposits & Placements with Banks",
        "Deposits and Placements with Banks and Other Financial Institutions",
        "Placements with Banks",
        "Deposits and Placements",
        "Deposits & Placements",
        "Deposits with Financial Institutions",
    ),
}

def _pick_alias_target(available: set[str], names: tuple[str, ...], *, prefix: str = "") -> str | None:
    """Return the first alias key (with optional prefix) that exists in available keys."""
    for n in names:
        k = (prefix + n) if prefix else n
        if k in available:
            return k
    return None

def _first_float(d: dict, keys: tuple[str, ...], *, prefix: str = ""):
    """Return the first _try_float(d[prefix+key]) that is not None."""
    for k in keys:
        v = _try_float(d.get(prefix + k))
        if v is not None:
            return v
    return None
  
# === JSON I/O helpers (used by the Per-stock JSON backup/restore UI) ===
import json
import pandas as pd

def _df_to_json_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert a DataFrame to pretty JSON bytes (UTF-8) for Streamlit download.
    - NaN -> null
    - Dates -> ISO8601
    - Pretty-printed for readability
    """
    if df is None or df.empty:
        return b"[]"
    parsed = json.loads(df.to_json(orient="records", date_format="iso"))
    return json.dumps(parsed, ensure_ascii=False, indent=2).encode("utf-8")

def _json_to_df(payload) -> pd.DataFrame | None:
    """
    Accepts a parsed JSON payload (list[dict] or dict) and returns a DataFrame.
    Returns None if the payload has no tabular data.
    """
    if payload is None:
        return None
    if isinstance(payload, list):
        return pd.DataFrame(payload) if payload else None
    if isinstance(payload, dict):
        # single row object
        return pd.DataFrame([payload])
    return None

def _normalize_row_for_upsert(row: dict) -> dict:
    """
    Light normalization for uploaded rows before upsert:
    - Trim strings
    - Convert blanks to None
    - Keep numbers as numbers when possible
    - Leave unknown columns as-is
    """
    out = {}
    for k, v in (row or {}).items():
        if isinstance(v, str):
            vv = v.strip()
            vv = None if vv in ("", "—", "None", "none", "NaN", "nan") else vv
            out[k] = vv
            continue
        try:
            if pd.isna(v):
                out[k] = None
                continue
        except Exception:
            pass
        out[k] = v
    return out

def _avg_two(cur, prev):
    cur_f, prev_f = _try_float(cur), _try_float(prev)
    if cur_f is None and prev_f is None:
        return None
    if prev_f is None:
        return cur_f
    if cur_f is None:
        return prev_f
    return (cur_f + prev_f) / 2.0

# ---- Prior value lookups ----
def _prev_annual_value(df, stock, year, key):
    row = df[
        (df["Name"].astype(str).str.upper() == str(stock).upper())
        & (df["IsQuarter"] != True)
        & (pd.to_numeric(df["Year"], errors="coerce") == int(year) - 1)
    ]
    if row.empty:
        return None
    return _try_float(row.iloc[0].get(key))

def _clean_str(x, default=""):
    """Return a trimmed string, falling back to default if x is NA/empty."""
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    s = str(x).strip()
    return s if s else default

# ---- Canonical casing helpers (one-cap / proper case) ----
def _one_cap(s: str) -> str:
    s = _clean_str(s, "")
    if not s:
        return ""
    # Title-like: first letter of each token upper, rest lower
    # Keep punctuation/spaces as-is
    return " ".join(w[:1].upper() + w[1:].lower() if w else w for w in str(s).split())


def _canon_name(s: str) -> str:
    return _one_cap(s)


def _canon_industry(s: str) -> str:
    return _one_cap(s)


def _canonical_bucket(s: str, default: str = "General") -> str:
    # Map case-insensitively to a canonical config bucket string
    t = _clean_str(s, "")
    if not t:
        return default
    for b in list(config.INDUSTRY_BUCKETS):
        if str(b).strip().lower() == t.lower():
            return str(b)
    return default


def _prev_quarter_value(df, stock, year, quarter, key):
    q = str(quarter).upper().strip()
    if q not in _Q_ORDER:
        return None
    py, pq = (int(year) - 1, "Q4") if _Q_ORDER[q] == 0 else (int(year), f"Q{_Q_ORDER[q]}")
    row = df[
        (df["Name"].astype(str).str.upper() == str(stock).upper())
        & (df["IsQuarter"] == True)
        & (pd.to_numeric(df["Year"], errors="coerce") == py)
        & (df["Quarter"].astype(str).str.upper() == pq)
    ]
    if row.empty:
        return None
    return _try_float(row.iloc[0].get(key))

# ---- Annual / Quarterly computed fields (banking + common buckets) ----
def _set_if_available_annual(year: int, key: str, val, bucket: str):
    # Respect Clear → suppression for this year
    if is_autofill_suppressed("annual", year):
        return False
    if val is None:
        return False
    keys = {f["key"] for f in _cats(bucket, quarterly=False) if f.get("key")}
    if key not in keys:
        return False

    ssk = f"{key}_{year}_annual"  # numeric store used by logic

    # ⛔️ Don't overwrite if user already typed something non-None
    if st.session_state.get(ssk, None) is not None:
        return False

    st.session_state[ssk] = float(val)

    # Mirror into the visible text widget so the form shows it (only if exists)
    tkey = ssk + "_txt"  # display store used by calc_number_input()
    if tkey in st.session_state:
        st.session_state[tkey] = str(st.session_state[ssk]).rstrip("0").rstrip(".")

    return True

def _set_if_available_quarterly(year: int, quarter: str, key: str, val, bucket: str):
    # Respect Clear → suppression for this quarter
    if is_autofill_suppressed("quarterly", year, quarter):
        return False
    if val is None:
        return False
    keys = {f["key"] for f in _cats(bucket, quarterly=True) if f.get("key")}
    qkey = f"Q_{key}" if not key.startswith("Q_") else key
    if qkey not in keys:
        return False

    ssk = f"{qkey}_{year}_{quarter}_q"  # numeric store used by logic

    # ⛔️ Don't overwrite if user already typed something non-None
    if st.session_state.get(ssk, None) is not None:
        return False

    st.session_state[ssk] = float(val)

    # Mirror into the visible text widget so the form shows it (only if exists)
    tkey = ssk + "_txt"  # display store used by calc_number_input()
    if tkey in st.session_state:
        st.session_state[tkey] = str(st.session_state[ssk]).rstrip("0").rstrip(".")

    return True

def compute_banking_derivatives_annual(
    row_up: dict, *, df, stock_name: str, year: int, available_keys: set[str]
):
    # Loan-to-Deposit Ratio (allows Gross Loans via aliases)
    if "Loan-to-Deposit Ratio" in available_keys:
        gl  = _first_float(row_up, _BANK_ALIASES["Gross Loans"])
        dep = _try_float(row_up.get("Deposits"))
        row_up["Loan-to-Deposit Ratio"] = (
            round(gl / dep * 100.0, 2) if (gl is not None and dep not in (None, 0)) else None
        )

    # CASA Ratio (%): primary = (Demand + Savings) / Deposits; fallback denom = D+S+T only if Deposits is blank
    if "CASA Ratio" in available_keys:
        dd  = _try_float(row_up.get("Demand Deposits"))
        sd  = _try_float(row_up.get("Savings Deposits"))
        dep = _try_float(row_up.get("Deposits"))
        td  = (_try_float(row_up.get("Time/Fixed Deposits"))
               or _try_float(row_up.get("Fixed/Time Deposits")))
        casa_amt = (dd or 0.0) + (sd or 0.0)
        denom = dep if dep not in (None, 0) else (casa_amt + (td or 0.0))
        row_up["CASA Ratio"] = round(casa_amt / denom * 100.0, 2) if denom else None

    # CASA (Core, %): (Demand+Savings) / (Demand+Savings+Fixed)
    if "CASA (Core, %)" in available_keys:
        dd = _try_float(row_up.get("Demand Deposits"))
        sd = _try_float(row_up.get("Savings Deposits"))
        td = _try_float(row_up.get("Time/Fixed Deposits")) or _try_float(row_up.get("Fixed/Time Deposits"))
        casa_core = None
        if td is not None:
            ddsd = (dd or 0.0) + (sd or 0.0)
            denom = ddsd + td
            casa_core = round(ddsd / denom * 100.0, 2) if denom else None
        row_up["CASA (Core, %)"] = casa_core

    # Net Loans fallback: ...
    nl_target = _pick_alias_target(available_keys, _BANK_ALIASES["Net Loans"])
    if nl_target and _try_float(row_up.get(nl_target)) is None:
        gl  = _first_float(row_up, _BANK_ALIASES["Gross Loans"])
        llr = _abs_or_none(_first_float(row_up, _LLR_KEYS))
        if gl is not None:
            row_up[nl_target] = gl - (llr or 0.0)

    # ✅ Always try to compute EA if still blank
    if "Earning Assets" in available_keys and _try_float(row_up.get("Earning Assets")) is None:
        nl    = _first_float(row_up, _BANK_ALIASES["Net Loans"])
        fvoci = _first_float(row_up, _BANK_ALIASES["FVOCI"])
        amort = _first_float(row_up, _BANK_ALIASES["AMORT"])
        rrepo = _first_float(row_up, _BANK_ALIASES["REVREPO"])
        plac  = _first_float(row_up, _BANK_ALIASES["PLACEMENTS"])
        parts = [v for v in (nl, fvoci, amort, plac, rrepo) if v is not None]
        row_up["Earning Assets"] = sum(parts) if parts else None

    # Averages (only if current provided) — leave as-is to avoid guessing which alias to persist
    if "Average Gross Loans" in available_keys:
        gl_cur  = _try_float(row_up.get("Gross Loans"))
        gl_prev = _prev_annual_value(df, stock_name, year, "Gross Loans") if gl_cur is not None else None
        row_up["Average Gross Loans"] = _avg_two(gl_cur, gl_prev) if gl_cur is not None else None

    if "Average Deposits" in available_keys:
        dep_cur  = _try_float(row_up.get("Deposits"))
        dep_prev = _prev_annual_value(df, stock_name, year, "Deposits") if dep_cur is not None else None
        row_up["Average Deposits"] = _avg_two(dep_cur, dep_prev) if dep_cur is not None else None

    if "Average Earning Assets" in available_keys:
        ea_cur  = _try_float(row_up.get("Earning Assets"))
        ea_prev = _prev_annual_value(df, stock_name, year, "Earning Assets") if ea_cur is not None else None
        row_up["Average Earning Assets"] = _avg_two(ea_cur, ea_prev) if ea_cur is not None else None

    # NIM (%): override-aware (still fine even if averages are blank)
    if "NIM" in available_keys:
        num = _try_float(row_up.get("TP_Bank_NIM_Num")) or _try_float(row_up.get("NII (incl Islamic)"))
        den = _try_float(row_up.get("TP_Bank_NIM_Den")) or _try_float(row_up.get("Average Earning Assets"))
        row_up["NIM"] = (round(num / den * 100.0, 2)
                         if (num is not None and den not in (None, 0))
                         else row_up.get("NIM", None))

def compute_banking_derivatives_quarterly(
    new_row: dict, *, df, stock_name: str, year: int, quarter: str, available_keys: set[str]
):
    # Loan-to-Deposit Ratio (Gross Loans via aliases with Q_ prefix)
    if "Q_Loan-to-Deposit Ratio" in available_keys:
        gl  = _first_float(new_row, _BANK_ALIASES["Gross Loans"], prefix="Q_")
        dep = _try_float(new_row.get("Q_Deposits"))
        new_row["Q_Loan-to-Deposit Ratio"] = (
            round(gl / dep * 100.0, 2) if (gl is not None and dep not in (None, 0)) else None
        )

    # CASA Ratio (%)
    if "Q_CASA Ratio" in available_keys:
        dd  = _try_float(new_row.get("Q_Demand Deposits"))
        sd  = _try_float(new_row.get("Q_Savings Deposits"))
        dep = _try_float(new_row.get("Q_Deposits"))
        td  = (_try_float(new_row.get("Q_Time/Fixed Deposits"))
               or _try_float(new_row.get("Q_Fixed/Time Deposits")))
        casa_amt = (dd or 0.0) + (sd or 0.0)
        denom = dep if dep not in (None, 0) else (casa_amt + (td or 0.0))
        new_row["Q_CASA Ratio"] = round(casa_amt / denom * 100.0, 2) if denom else None

    # CASA (Core, %)
    if "Q_CASA (Core, %)" in available_keys:
        dd = _try_float(new_row.get("Q_Demand Deposits"))
        sd = _try_float(new_row.get("Q_Savings Deposits"))
        td = _try_float(new_row.get("Q_Time/Fixed Deposits")) or _try_float(new_row.get("Q_Fixed/Time Deposits"))
        casa_core = None
        if td is not None:
            ddsd = (dd or 0.0) + (sd or 0.0)
            denom = ddsd + td
            casa_core = round(ddsd / denom * 100.0, 2) if denom else None
        new_row["Q_CASA (Core, %)"] = casa_core

    # ───────────────────────────────────────────────────────────────────────────
    # Net Loans fallback (Quarterly): write into whichever Q_ alias exists
    qnl_target = _pick_alias_target(available_keys, _BANK_ALIASES["Net Loans"], prefix="Q_")
    if qnl_target and _try_float(new_row.get(qnl_target)) is None:
        gl  = _first_float(new_row, _BANK_ALIASES["Gross Loans"], prefix="Q_")
        llr = _abs_or_none(_first_float(new_row, _LLR_KEYS, prefix="Q_"))
        if gl is not None:
            new_row[qnl_target] = gl - (llr or 0.0)

    # Earning Assets (Quarterly, Strict+): NL + FVOCI + Amort + Placements + Reverse Repo (exclude FVTPL)
    if "Q_Earning Assets" in available_keys and _try_float(new_row.get("Q_Earning Assets")) is None:
        nl    = _first_float(new_row, _BANK_ALIASES["Net Loans"], prefix="Q_")
        fvoci = _first_float(new_row, _BANK_ALIASES["FVOCI"], prefix="Q_")
        amort = _first_float(new_row, _BANK_ALIASES["AMORT"], prefix="Q_")
        rrepo = _first_float(new_row, _BANK_ALIASES["REVREPO"], prefix="Q_")
        plac  = _first_float(new_row, _BANK_ALIASES["PLACEMENTS"], prefix="Q_")
        parts = [v for v in (nl, fvoci, amort, plac, rrepo) if v is not None]
        new_row["Q_Earning Assets"] = sum(parts) if parts else None

    # ───────────────────────────────────────────────────────────────────────────

    # Averages: (keep using the canonical Q_ keys you’ve defined)
    if "Q_Average Gross Loans" in available_keys:
        gl_cur  = _try_float(new_row.get("Q_Gross Loans"))
        gl_prev = _prev_quarter_value(df, stock_name, year, quarter, "Q_Gross Loans") if gl_cur is not None else None
        new_row["Q_Average Gross Loans"] = _avg_two(gl_cur, gl_prev) if gl_cur is not None else None

    if "Q_Average Deposits" in available_keys:
        dep_cur  = _try_float(new_row.get("Q_Deposits"))
        dep_prev = _prev_quarter_value(df, stock_name, year, quarter, "Q_Deposits") if dep_cur is not None else None
        new_row["Q_Average Deposits"] = _avg_two(dep_cur, dep_prev) if dep_cur is not None else None

    if "Q_Average Earning Assets" in available_keys:
        ea_cur  = _try_float(new_row.get("Q_Earning Assets"))
        ea_prev = _prev_quarter_value(df, stock_name, year, quarter, "Q_Earning Assets") if ea_cur is not None else None
        new_row["Q_Average Earning Assets"] = _avg_two(ea_cur, ea_prev) if ea_cur is not None else None

    # Q_NIM (%): override-aware
    if "Q_NIM" in available_keys:
        num = _try_float(new_row.get("Q_TP_Bank_NIM_Num")) or _try_float(new_row.get("Q_NII (incl Islamic)"))
        den = _try_float(new_row.get("Q_TP_Bank_NIM_Den")) or _try_float(new_row.get("Q_Average Earning Assets"))
        new_row["Q_NIM"] = (round(num / den * 100.0, 2)
                            if (num is not None and den not in (None, 0))
                            else new_row.get("Q_NIM", None))

def _sum_existing(*vals):
    parts = [v for v in vals if v is not None]
    return sum(parts) if parts else None


def autofill_common_fields_annual(*, df, stock_name: str, year: int, bucket: str):
    changed = False

    def _live(k):
        return _try_float(st.session_state.get(f"{k}_{year}_annual"))

    # universal
    rev      = _live("Revenue")
    cogs     = _abs_or_none(_live("CostOfSales"))
    dep_ppe  = _abs_or_none(_live("DepPPE"))
    dep_rou  = _abs_or_none(_live("DepROU"))
    dep_inv  = _abs_or_none(_live("DepInvProp"))
    dep_amort = (
        _sum_costs(dep_ppe, dep_rou, dep_inv)
        if any(v is not None for v in (dep_ppe, dep_rou, dep_inv))
        else _abs_or_none(_live("DepAmort"))
    )
    op       = _live("Operating Profit")
    sell     = _abs_or_none(_live("Selling Expenses"))
    admin    = _abs_or_none(_live("Administrative Expenses"))
    intexp   = _abs_or_none(_live("Interest Expense"))

    changed |= _set_if_available_annual(
        year, "Gross Profit",
        (rev - cogs) if (rev is not None and cogs is not None) else None, bucket
    )
    changed |= _set_if_available_annual(year, "DepAmort", dep_amort, bucket)
    changed |= _set_if_available_annual(
        year, "EBITDA",
        (op + (dep_amort or 0)) if op is not None else None, bucket
    )
    changed |= _set_if_available_annual(
        year, "Three Fees", _sum_costs(sell, admin, intexp), bucket
    )

    if bucket == "Banking":
        ii  = _try_float(_live("Interest Income"))
        ie  = _abs_or_none(_live("Interest Expense"))
        isl = _try_float(_live("Net Islamic Income"))
        nii = (ii - ie + (isl or 0)) if (ii is not None and ie is not None) else None
        changed |= _set_if_available_annual(year, "NII (incl Islamic)", nii, bucket)

    if bucket == "REITs":
        ie, debt = _abs_or_none(_live("Interest Expense")), _try_float(_live("Total Borrowings"))
        changed |= _set_if_available_annual(
            year, "Avg Cost of Debt",
            round(ie / debt * 100.0, 2) if (ie is not None and debt not in (None, 0)) else None,
            bucket
        )
        eq, units = _try_float(_live("Equity")), _try_float(_live("Units Outstanding"))
        changed |= _set_if_available_annual(
            year, "NAV per Unit",
            (eq / units) if (eq is not None and units not in (None, 0)) else None, bucket
        )

    if bucket in ("Utilities", "Telco"):
        capex = _abs_or_none(_live("Capex"))
        changed |= _set_if_available_annual(
            year, "Capex to Revenue",
            round(capex / rev * 100.0, 2) if (capex is not None and rev not in (None, 0)) else None,
            bucket
        )

    if bucket == "Tech":
        rnd = _abs_or_none(_live("R&D Expense"))
        changed |= _set_if_available_annual(
            year, "R&D Intensity",
            round(rnd / rev * 100.0, 2) if (rnd is not None and rev not in (None, 0)) else None,
            bucket
        )

    if bucket == "Construction":
        new_orders, tender = _live("New Orders"), _live("Tender Book")
        changed |= _set_if_available_annual(
            year, "Win Rate",
            round(new_orders / tender * 100.0, 2)
            if (new_orders is not None and tender not in (None, 0)) else None,
            bucket
        )

    if bucket == "Healthcare":
        pdays, admits = _live("Patient Days"), _live("Admissions")
        changed |= _set_if_available_annual(
            year, "ALOS",
            (pdays / admits) if (pdays is not None and admits not in (None, 0)) else None,
            bucket
        )

    if bucket == "Transportation/Logistics":
        ask, rpk = _live("ASK"), _live("RPK")
        changed |= _set_if_available_annual(
            year, "Load Factor",
            round(rpk / ask * 100.0, 2) if (rpk is not None and ask not in (None, 0)) else None,
            bucket
        )
        changed |= _set_if_available_annual(
            year, "Yield per km",
            (rev / rpk) if (rev is not None and rpk not in (None, 0)) else None, bucket
        )

    if bucket == "Property":
        borrow, cash = _live("Total Borrowings"), _live("Cash & Cash Equivalents")
        changed |= _set_if_available_annual(
            year, "Net Debt",
            (borrow - cash) if (borrow is not None and cash is not None) else None, bucket
        )

    return changed

def autofill_common_fields_quarterly(*, df, stock_name: str, year: int, quarter: str, bucket: str):
    changed = False

    def _live(k):
        return _try_float(st.session_state.get(f"Q_{k}_{year}_{quarter}_q"))

    rev      = _live("Revenue")
    cogs     = _abs_or_none(_live("CostOfSales"))
    dep_ppe  = _abs_or_none(_live("DepPPE"))
    dep_rou  = _abs_or_none(_live("DepROU"))
    dep_inv  = _abs_or_none(_live("DepInvProp"))
    dep_amort = (
        _sum_costs(dep_ppe, dep_rou, dep_inv)
        if any(v is not None for v in (dep_ppe, dep_rou, dep_inv))
        else _abs_or_none(_live("DepAmort"))
    )
    op       = _live("Operating Profit")
    sell     = _abs_or_none(_live("Selling Expenses"))
    admin    = _abs_or_none(_live("Administrative Expenses"))
    intexp   = _abs_or_none(_live("Interest Expense"))

    changed |= _set_if_available_quarterly(
        year, quarter, "Gross Profit",
        (rev - cogs) if (rev is not None and cogs is not None) else None, bucket
    )
    changed |= _set_if_available_quarterly(year, quarter, "DepAmort", dep_amort, bucket)
    changed |= _set_if_available_quarterly(
        year, quarter, "EBITDA",
        (op + (dep_amort or 0)) if op is not None else None, bucket
    )
    changed |= _set_if_available_quarterly(
        year, quarter, "Three Fees", _sum_costs(sell, admin, intexp), bucket
    )

    if bucket == "Banking":
        ii  = _try_float(_live("Interest Income"))
        ie  = _abs_or_none(_live("Interest Expense"))
        isl = _try_float(_live("Net Islamic Income"))
        nii = (ii - ie + (isl or 0)) if (ii is not None and ie is not None) else None
        changed |= _set_if_available_quarterly(year, quarter, "NII (incl Islamic)", nii, bucket)

    if bucket == "REITs":
        ie, debt = _abs_or_none(_live("Interest Expense")), _try_float(_live("Total Borrowings"))
        changed |= _set_if_available_quarterly(
            year, quarter, "Avg Cost of Debt",
            round(ie / debt * 100.0, 2) if (ie is not None and debt not in (None, 0)) else None,
            bucket
        )

    if bucket in ("Utilities", "Telco"):
        capex = _abs_or_none(_live("Capex"))
        changed |= _set_if_available_quarterly(
            year, quarter, "Capex to Revenue",
            round(capex / rev * 100.0, 2) if (capex is not None and rev not in (None, 0)) else None,
            bucket
        )

    if bucket == "Tech":
        rnd = _abs_or_none(_live("R&D Expense"))
        changed |= _set_if_available_quarterly(
            year, quarter, "R&D Intensity",
            round(rnd / rev * 100.0, 2) if (rnd is not None and rev not in (None, 0)) else None,
            bucket
        )

    if bucket == "Construction":
        new_orders, tender = _live("New Orders"), _live("Tender Book")
        changed |= _set_if_available_quarterly(
            year, quarter, "Win Rate",
            round(new_orders / tender * 100.0, 2)
            if (new_orders is not None and tender not in (None, 0)) else None,
            bucket
        )

    if bucket == "Healthcare":
        pdays, admits = _live("Patient Days"), _live("Admissions")
        changed |= _set_if_available_quarterly(
            year, quarter, "ALOS",
            (pdays / admits) if (pdays is not None and admits not in (None, 0)) else None,
            bucket
        )

    if bucket == "Transportation/Logistics":
        ask, rpk = _live("ASK"), _live("RPK")
        changed |= _set_if_available_quarterly(
            year, quarter, "Load Factor",
            round(rpk / ask * 100.0, 2) if (rpk is not None and ask not in (None, 0)) else None,
            bucket
        )
        changed |= _set_if_available_quarterly(
            year, quarter, "Yield per km",
            (rev / rpk) if (rev is not None and rpk not in (None, 0)) else None, bucket
        )

    if bucket == "Property":
        borrow, cash = _live("Total Borrowings"), _live("Cash & Cash Equivalents")
        changed |= _set_if_available_quarterly(
            year, quarter, "Net Debt",
            (borrow - cash) if (borrow is not None and cash is not None) else None, bucket
        )

    return changed

# ---- Pure derived computations used at SAVE time (no session_state writes) ----
def compute_common_derivatives_inplace(row: dict, *, bucket: str, prefix: str = "") -> None:
    """
    Recompute key derived fields directly on the row dict before saving.
    Does NOT touch session_state. Works for annual (prefix="") and quarterly (prefix="Q_").
    """
    def gf(k):  # get float from row with prefix
        return _try_float(row.get(prefix + k))

    def absn(v):
        return None if v is None else abs(v)

    # Source fields
    rev      = gf("Revenue")
    cogs     = absn(gf("CostOfSales"))
    dep_ppe  = absn(gf("DepPPE"))
    dep_rou  = absn(gf("DepROU"))
    dep_inv  = absn(gf("DepInvProp"))
    dep_amort = (
        _sum_existing(dep_ppe, dep_rou, dep_inv)
        if any(v is not None for v in (dep_ppe, dep_rou, dep_inv))
        else absn(gf("DepAmort"))
    )
    op       = gf("Operating Profit") or gf("EBIT")
    sell     = absn(gf("Selling Expenses"))
    admin    = absn(gf("Administrative Expenses"))
    intexp   = absn(gf("Interest Expense"))

    # Core derived
    gross_profit = (rev - cogs) if (rev is not None and cogs is not None) else None
    ebitda       = (op + (dep_amort or 0)) if op is not None else None
    three_fees   = _sum_existing(*(v for v in (sell, admin, intexp) if v is not None))

    # Write back
    row[prefix + "Gross Profit"] = gross_profit
    row[prefix + "DepAmort"]     = dep_amort
    row[prefix + "EBITDA"]       = ebitda
    row[prefix + "Three Fees"]   = three_fees

    # Bucket-specific pure ratios
    if bucket in ("Utilities", "Telco"):
        capex = absn(gf("Capex"))
        row[prefix + "Capex to Revenue"] = (
            round(capex / rev * 100.0, 2) if (capex is not None and rev not in (None, 0)) else None
        )

    if bucket == "Tech":
        rnd = absn(gf("R&D Expense"))
        row[prefix + "R&D Intensity"] = (
            round(rnd / rev * 100.0, 2) if (rnd is not None and rev not in (None, 0)) else None
        )

    if bucket == "Property":
        borrow, cash = gf("Total Borrowings"), gf("Cash & Cash Equivalents")
        row[prefix + "Net Debt"] = (
            (borrow - cash) if (borrow is not None and cash is not None) else None
        )

    if bucket == "REITs":
        ie, debt = absn(gf("Interest Expense")), gf("Total Borrowings")
        row[prefix + "Avg Cost of Debt"] = (
            round(ie / debt * 100.0, 2) if (ie is not None and debt not in (None, 0)) else None
        )
    # Banking left to compute_banking_derivatives_*()

# ---- Helper Preview Cards ----
def render_calc_helper(
    *, bucket: str, scope: str, df, stock_name: str, year: int, quarter: str | None = None, fallback: dict | None = None
):
    fallback = fallback or {}

    def _read(key: str):
        try:
            ss_key = f"{key}_{year}_annual" if scope == "annual" else f"Q_{key}_{year}_{quarter}_q"
            if ss_key in st.session_state:
                return _try_float(st.session_state.get(ss_key))
            return _try_float(fallback.get(key))
        except Exception:
            return None

    def _fmt_money(x):
        return "-" if x is None else f"{x:,.0f}"

    def _fmt_pct(x):
        return "-" if x is None else f"{x:.2f}"

    rows = []
    rev      = _read("Revenue")
    cogs     = _abs_or_none(_read("CostOfSales"))
    dep_ppe  = _abs_or_none(_read("DepPPE"))
    dep_rou  = _abs_or_none(_read("DepROU"))
    dep_inv  = _abs_or_none(_read("DepInvProp"))
    dep_amort = (
        _sum_costs(dep_ppe, dep_rou, dep_inv)
        if any(v is not None for v in (dep_ppe, dep_rou, dep_inv))
        else _abs_or_none(_read("DepAmort"))
    )
    op = _read("Operating Profit") or _read("EBIT")
    ebitda = (op + (dep_amort or 0)) if op is not None else None
    gross_profit = (rev - cogs) if (rev is not None and cogs is not None) else _read("Gross Profit")

    if gross_profit is not None:
        rows.append(("Gross Profit (RM)", _fmt_money(gross_profit), f"{_fmt_money(rev)} − {_fmt_money(cogs)}"))
    if ebitda is not None:
        rows.append(("EBITDA (RM)", _fmt_money(ebitda), f"{_fmt_money(op)} + {_fmt_money(dep_amort)}"))
    
    # NEW helper cards:
    if dep_amort is not None:
        rows.append(("DepAmort (RM)", _fmt_money(dep_amort), "DepPPE + DepROU + DepInvProp (if present)"))

    # Only show Three Fees card if this bucket supports it per config
    if _bucket_supports_three_fees(bucket):
        three_fees = _sum_costs(
            _abs_or_none(_read("Selling Expenses")),
            _abs_or_none(_read("Administrative Expenses")),
            _abs_or_none(_read("Interest Expense")),
        )
        if three_fees is not None:
            rows.append(("Three Fees (RM)", _fmt_money(three_fees), "Selling + Admin + Interest"))

    if bucket == "Banking":
        dd = _try_float(_read("Demand Deposits")) or 0.0
        sd = _try_float(_read("Savings Deposits")) or 0.0
        dep = _try_float(_read("Deposits")) or 0.0
        td = _try_float(_read("Time/Fixed Deposits")) or _try_float(_read("Fixed/Time Deposits")) or 0.0
        casa_amt = dd + sd
        denom = dep if dep not in (None, 0) else (dd + sd + td)
        casa_pct = (casa_amt / denom * 100.0) if denom else None

        denom_core = (dd + sd + td) if any(v is not None for v in (dd, sd, td)) else None
        casa_core_pct = (casa_amt / denom_core * 100.0) if denom_core else None

        # Net Loans display (entered or derived Gross − LLR)
        gl  = _try_float(_read("Gross Loans"))
        llr = None
        for k in _LLR_KEYS:
            vv = _try_float(_read(k))
            if vv is not None:
                llr = _abs_or_none(vv)
                break
        nl_entered = _try_float(_read("Net Loans"))
        nl_calc = (gl - (llr or 0.0)) if gl is not None else None
        nl_display = nl_entered if nl_entered is not None else nl_calc
        if nl_display is not None:
            rows.append((
                "Net Loans (RM)",
                _fmt_money(nl_display),
                "As entered" if nl_entered is not None else "Gross − Loan Loss Reserve",
            ))

        # Earning Assets display (Strict+; excludes FVTPL, includes Placements)
        def _read_any(*names):
            for n in names:
                v = _try_float(_read(n))
                if v is not None:
                    return v
            return None

        fvoci  = _read_any(*_BANK_ALIASES["FVOCI"])
        amort  = _read_any(*_BANK_ALIASES["AMORT"])
        rrepo  = _read_any(*_BANK_ALIASES["REVREPO"])
        plac   = _read_any(*_BANK_ALIASES["PLACEMENTS"])
        fvtpl  = _read_any(*_BANK_ALIASES["FVTPL"])  # display only, not added

        parts_strict_plus = [v for v in (nl_display, fvoci, amort, plac, rrepo) if v is not None]
        ea_calc = sum(parts_strict_plus) if parts_strict_plus else None

        if ea_calc is not None:
            note = "Net Loans + FVOCI + Amortised Cost + Placements + Reverse Repos (FVTPL excluded)"
            rows.append(("Earning Assets (RM)", _fmt_money(ea_calc), note))
      
        # Current values (None-aware) for averages
        gl_cur  = _try_float(_read("Gross Loans"))
        dep_cur = _try_float(_read("Deposits"))
        ea_cur  = _try_float(_read("Earning Assets"))

        # Prior-period lookups for averages (only if current provided)
        if scope == "annual":
            gl_prev  = _prev_annual_value(df, stock_name, year, "Gross Loans")       if gl_cur  is not None else None
            dep_prev = _prev_annual_value(df, stock_name, year, "Deposits")          if dep_cur is not None else None
            ea_prev  = _prev_annual_value(df, stock_name, year, "Earning Assets")    if ea_cur  is not None else None
        else:
            gl_prev  = _prev_quarter_value(df, stock_name, year, quarter, "Q_Gross Loans")    if gl_cur  is not None else None
            dep_prev = _prev_quarter_value(df, stock_name, year, quarter, "Q_Deposits")       if dep_cur is not None else None
            ea_prev  = _prev_quarter_value(df, stock_name, year, quarter, "Q_Earning Assets") if ea_cur  is not None else None

        # Averages (only if current provided)
        avg_gl         = _avg_two(gl_cur,  _try_float(gl_prev))    if gl_cur  is not None else None
        avg_dep        = _avg_two(dep_cur, _try_float(dep_prev))   if dep_cur is not None else None
        avg_ea_display = _avg_two(ea_cur,  _try_float(ea_prev))    if ea_cur  is not None else None

        # Existing helper metrics
        ldr = (gl_cur / dep * 100.0) if (gl_cur is not None and dep not in (None, 0)) else None

        nii = _try_float(_read("NII (incl Islamic)"))
        nim_num = _try_float(_read("TP_Bank_NIM_Num")) or nii

        # NIM denominator logic: override → else average EA (if available)
        avg_ea_for_nim = _avg_two(ea_cur, _try_float(ea_prev)) if (ea_cur is not None or ea_prev is not None) else None
        nim_den = _try_float(_read("TP_Bank_NIM_Den")) or avg_ea_for_nim
        nim = (nim_num / nim_den * 100.0) if (nim_num is not None and nim_den not in (None, 0)) else None

        # NIM denominator logic: override → else average EA (if available)
        avg_ea_for_nim = _avg_two(ea_cur, _try_float(ea_prev)) if (ea_cur is not None or ea_prev is not None) else None
        nim_den = _try_float(_read("TP_Bank_NIM_Den")) or avg_ea_for_nim
        nim = (nim_num / nim_den * 100.0) if (nim_num is not None and nim_den not in (None, 0)) else None

        # (NEW) CIR is disabled for Banking by default — compute only if allowed
        cir = None
        if _cir_allowed(bucket):
            num_override = _try_float(_read("TP_Bank_CIR_Num"))
            op_ex_raw    = _try_float(_read("Operating Expenses"))
            # keep numerator absolute if used
            cir_num      = _abs_or_none(num_override if num_override is not None else op_ex_raw)
            cir_den      = _try_float(_read("TP_Bank_CIR_Den")) or _try_float(_read("Operating Income"))
            cir          = (cir_num / cir_den * 100.0) if (cir_num is not None and cir_den not in (None, 0)) else None

        # Leverage (override-aware)
        lev_num = _try_float(_read("TP_Bank_Lev_Num")) or _try_float(_read("Total Assets"))
        lev_den = _try_float(_read("TP_Bank_Lev_Den")) or _try_float(_read("Equity"))
        leverage = (lev_num / lev_den) if (lev_num is not None and lev_den not in (None, 0)) else None

        # Build helper cards
        rows.extend([
            ("CASA (RM)", _fmt_money(casa_amt), "Demand + Savings"),
            ("CASA Ratio (%)", _fmt_pct(casa_pct), "CASA ÷ Deposits (or D+S+T)"),
            ("CASA (Core, %)", _fmt_pct(casa_core_pct), "Demand + Savings ÷ (Demand + Savings + Fixed)"),
            ("Loan-to-Deposit Ratio (%)", _fmt_pct(ldr), "Gross Loans ÷ Deposits"),
        ])

        # NEW helper cards for averages
        if avg_gl is not None:
            rows.append(("Average Gross Loans (RM)", _fmt_money(avg_gl), "Avg of current & prior"))
        if avg_dep is not None:
            rows.append(("Average Deposits (RM)", _fmt_money(avg_dep), "Avg of current & prior"))
        if avg_ea_display is not None:
            rows.append(("Average Earning Assets (RM)", _fmt_money(avg_ea_display), "Avg of current & prior"))

        rows.append(("NIM (%)", _fmt_pct(nim), "NII ÷ Avg EA"))
        if _cir_allowed(bucket):  # show only if allowed (off for Banking)
            rows.append(("CIR (%)", _fmt_pct(cir), "Operating Expenses ÷ Operating Income"))
        rows.append(("Leverage (x)", "-" if leverage is None else f"{leverage:,.2f}", "Total Assets ÷ Equity"))

    if bucket in ("Utilities", "Telco"):
        capex_to_rev = None
        capex = _abs_or_none(_read("Capex"))
        if capex is not None and rev not in (None, 0):
            capex_to_rev = round(capex / rev * 100.0, 2)
        rows.append(("Capex to Revenue (%)", _fmt_pct(capex_to_rev), "Capex ÷ Revenue") if capex_to_rev is not None else None)

    if bucket == "Tech":
        rnd = _abs_or_none(_read("R&D Expense"))
        val = round(rnd / rev * 100.0, 2) if (rnd is not None and rev not in (None, 0)) else None
        rows.append(("R&D Intensity (%)", _fmt_pct(val), "R&D ÷ Revenue") if val is not None else None)

    if bucket == "Property":
        borrow, cash = _try_float(_read("Total Borrowings")), _try_float(_read("Cash & Cash Equivalents"))
        nd = (borrow - cash) if (borrow is not None and cash is not None) else None
        rows.append(("Net Debt (RM)", _fmt_money(nd), "Borrowings − Cash") if nd is not None else None)

    if bucket == "Transportation/Logistics":
        ask, rpk = _try_float(_read("ASK")), _try_float(_read("RPK"))
        lf = round(rpk / ask * 100.0, 2) if (rpk is not None and ask not in (None, 0)) else None
        ypk = (rev / rpk) if (rev is not None and rpk not in (None, 0)) else None
        rows.append(("Load Factor (%)", _fmt_pct(lf), "RPK ÷ ASK") if lf is not None else None)
        rows.append(("Yield per km (RM)", "-" if ypk is None else f"{ypk:,.4f}", "Revenue ÷ RPK") if ypk is not None else None)

    rows = [r for r in rows if r]
    if not rows:
        return

    with st.expander("📐 Helper (derived metrics)", expanded=False):
        st.caption(f"{bucket} • {'Annual' if scope=='annual' else f'{quarter} {year}'}")

        items = []
        for (label, value_str, note) in rows:
            card = {"label": label, "value": value_str, "note": note}
            if "(%)" in label:
                card["badge"] = "%"
            elif "(RM)" in label:
                card["badge"] = "RM"
            items.append(card)

        render_stat_cards(items, columns=3)

# ---- Infer helpers ----
def _infer_industry_for_stock(df: pd.DataFrame, stock: str, fallback: str = "") -> str:
    if df is None or df.empty:
        return fallback
    s = df.loc[df["Name"] == stock, "Industry"].dropna().astype(str).str.strip()
    if s.empty:
        return fallback
    try:
        m = s.mode()
        return m.iloc[0] if not m.empty else s.iloc[0]
    except Exception:
        return s.iloc[0]


def _infer_bucket_for_stock(df: pd.DataFrame, stock: str, fallback: str = "General") -> str:
    if df is None or df.empty or "IndustryBucket" not in df.columns:
        return fallback
    s = df.loc[df["Name"] == stock, "IndustryBucket"].dropna().astype(str).str.strip()
    if s.empty:
        return fallback
    try:
        m = s.mode()
        chosen = m.iloc[0] if not m.empty else s.iloc[0]
        return chosen if chosen in config.INDUSTRY_BUCKETS else fallback
    except Exception:
        return fallback


# === URL query prefill from Dashboard “Edit” link ===
params = st.query_params
default_stock = params.get("stock_name", [""])[0]

# ---- Load data & normalise core columns ----

# --- Global Settings (applies to all stocks) ---
st.markdown(section(
    "🌐 Global Settings",
    "Used as risk-free rates. These are global, not per stock."
), unsafe_allow_html=True)

cur_fd  = get_fd_eps_rate()
cur_epf = get_epf_rate()

c1, c2 = st.columns([1, 1])
with c1:
    new_fd = calc_number_input(
        "Fixed Deposit (FD) rate (%)",
        value=cur_fd,
        key="fd_rate_global",
        decimals=2,
        placeholder="e.g. 3.50",
    )
with c2:
    new_epf = calc_number_input(
        "EPF rate (%)",
        value=cur_epf,
        key="epf_rate_global",
        decimals=2,
        placeholder="e.g. 5.50",
    )

b1, b2 = st.columns([1, 1])
with b1:
    if st.button("💾 Save settings", use_container_width=True):
        set_fd_eps_rate(new_fd)
        set_epf_rate(new_epf)
        st.session_state["__settings_saved"] = True
        st.success("Saved global FD & EPF rates.")
with b2:
    if st.button("Clear both", type="secondary", use_container_width=True):
        set_fd_eps_rate(None)
        set_epf_rate(None)
        st.session_state["__settings_saved"] = True
        st.success("Cleared FD & EPF rates.")
        st.stop()


df = get_df().copy()

# ensure LastModified & MetaModified columns exist
for _col in ("LastModified", "MetaModified"):
    if _col not in df.columns:
        df[_col] = pd.NA
for col in ("Name", "Industry", "IndustryBucket", "Quarter"):

    if col not in df.columns:
        df[col] = pd.NA
    df[col] = df[col].astype("string")

# ensure IsQuarter exists and is boolean
if "IsQuarter" not in df.columns:
    df["IsQuarter"] = False
else:
    df["IsQuarter"] = df["IsQuarter"].fillna(False).astype(bool)


# Canonicalize casing for consistency across filters/joins
df["Name"] = df["Name"].apply(lambda x: _canon_name(x) if pd.notna(x) else x)
df["Industry"] = df["Industry"].apply(lambda x: _canon_industry(x) if pd.notna(x) else x)
df["IndustryBucket"] = df["IndustryBucket"].apply(
    lambda x: _canonical_bucket(x) if pd.notna(x) else _canonical_bucket("")
)

# Clean “Quarter” too, since we compare later
df["Quarter"] = df["Quarter"].astype("string").str.strip().str.upper()

# Cosmetic only: show annual rows with a blank Quarter in memory.
# (Annual save paths do NOT write Quarter, so this won't be persisted.)
df.loc[df["IsQuarter"] != True, "Quarter"] = ""

# Drop blank names after normalization
df = df[df["Name"].notna() & (df["Name"].str.strip() != "")]

# ---------------- Stock Settings ----------------
st.markdown(section("⚙️ Stock Settings", "Per-stock metadata & current price"), unsafe_allow_html=True)

with st.form("stock_settings_form", clear_on_submit=False):
    stock_name_raw = st.text_input("Stock Name", value=default_stock)
    stock_name = _canon_name(stock_name_raw)

    c1, c2 = st.columns([1, 1])
    with c1:
        industry_text_prefill = (
            _infer_industry_for_stock(df, stock_name, fallback="") if stock_name else ""
        )
        industry = st.text_input("Industry (free text)", value=industry_text_prefill)
    with c2:
        bucket_prefill = (
            _infer_bucket_for_stock(df, stock_name, fallback="General") if stock_name else "General"
        )
        bucket = st.selectbox(
            "Industry Bucket (dropdown)",
            options=list(config.INDUSTRY_BUCKETS),
            index=list(config.INDUSTRY_BUCKETS).index(bucket_prefill),
        )

    # Field preview (annual)
    try:
        preview = _flatten_categories(_get_categories(bucket, quarterly=False))
        labels = [f.get("label") or f.get("key") for f in preview]
        st.caption(
            "Fields for this bucket (annual): " + (", ".join(labels) if labels else "— none —")
        )
    except Exception:
        pass

    # Current price default for this stock (None means "no default shown")
    current_price_default = None
    if stock_name:
        # case-insensitive match to be robust
        mask_stock = df["Name"].astype(str).str.upper() == stock_name.upper()
        if "CurrentPrice" in df.columns and df.loc[mask_stock, "CurrentPrice"].notna().any():
            current_price_default = float(df.loc[mask_stock, "CurrentPrice"].dropna().iloc[0])
        elif "Price" in df.columns and df.loc[mask_stock, "Price"].notna().any():
            current_price_default = float(df.loc[mask_stock, "Price"].dropna().iloc[0])
        elif "SharePrice" in df.columns and df.loc[mask_stock, "SharePrice"].notna().any():
            current_price_default = float(df.loc[mask_stock, "SharePrice"].dropna().iloc[-1])

    # Reseed the paired text widget whenever the picked stock changes,
    # so it shows the last saved value instead of sticking with an old buffer.
    if st.session_state.get("__cp_stock") != stock_name:
        st.session_state.pop("cur_price_stock_txt", None)
        st.session_state["__cp_stock"] = stock_name

    cp = calc_number_input(
        "Current Price (per stock…)",
        value=(current_price_default if current_price_default not in (None, "") else None),
        key="cur_price_stock",
        decimals=4,
        placeholder="leave blank to clear",
    )

    submit_settings = st.form_submit_button("💾 Save stock settings")

if submit_settings:
    if not stock_name:
        st.error("Please enter a Stock Name first.")
    else:
        # Ensure required columns exist (and sane dtypes for ones we touch)
        for col in ("Name", "CurrentPrice", "Industry", "IndustryBucket", "IsQuarter", "Quarter", "Year", "LastModified", "MetaModified"):
            if col not in df.columns:
                df[col] = pd.NA

        # Make sure these two are consistent
        if "IsQuarter" not in df.columns:
            df["IsQuarter"] = False
        else:
            df["IsQuarter"] = df["IsQuarter"].fillna(False).astype(bool)
        df["Quarter"] = df["Quarter"].astype("string")

        # Canonical values
        name_canon = _canon_name(stock_name)
        ind_canon  = _canon_industry(industry)
        buck_canon = _canonical_bucket(bucket)

        # BEFORE snapshot for diff (do this BEFORE any writes)
        try:
            _mask_name = df["Name"].astype(str) == name_canon
            before_row = df.loc[_mask_name].head(1).to_dict(orient="records")
            before_meta = (before_row[0] if before_row else {})
        except Exception:
            before_meta = {}

        # Case-safe match
        mask = df["Name"].astype(str) == name_canon
        now_iso = _now_iso()

        if mask.any():
            # ✅ Update price + meta on ALL rows (do NOT touch row LastModified)
            df.loc[mask, "CurrentPrice"] = (float(cp) if cp is not None else pd.NA)
            if ind_canon:
                df.loc[mask, "Industry"] = ind_canon
            if buck_canon:
                df.loc[mask, "IndustryBucket"] = buck_canon
            # Meta-level timestamp
            df.loc[mask, "MetaModified"] = now_iso
        else:
            # ✅ Create a placeholder row if no rows yet (meta timestamp only)
            base_row = {
                "Name": name_canon,
                "Industry": ind_canon,
                "IndustryBucket": buck_canon,
                "IsQuarter": False,
                "Quarter": "",
                "Year": pd.NA,
                "CurrentPrice": (float(cp) if cp is not None else pd.NA),
                "MetaModified": now_iso,
            }
            df = pd.concat([df, pd.DataFrame([base_row])], ignore_index=True)

        # AFTER snapshot for diff
        try:
            after_row = df.loc[df["Name"].astype(str) == name_canon].head(1).to_dict(orient="records")
            after_meta = (after_row[0] if after_row else {})
        except Exception:
            after_meta = {}

        # Diff & audit
        changes = _diff_dict(before_meta, after_meta, allowed={"CurrentPrice", "Industry", "IndustryBucket"})
        if changes:
            _audit_log_event(
                "update",
                name=name_canon,
                scope="meta",
                changes=changes,
                before=before_meta,
                after=after_meta,
                source="stock_settings",
            )

        # Persist + bust caches
        save_df(df)


        # ✅ sync Quick Edit on rerun
        st.session_state["qeb_pick"] = name_canon        # preselect in Quick Edit
        st.session_state["__open_qe_for"] = name_canon   # auto-expand its expander
        st.session_state.pop(f"cur_price_quick_{name_canon}", None)
        st.session_state.pop("cur_price_stock", None)
        st.session_state.pop("cur_price_stock_txt", None)  # ensure the text widget reseeds next run

        # toast after rerun
        st.session_state["__settings_saved"] = True

        _safe_rerun()

# ---------------- Main Forms (Annual / Quarterly) ----------------
if stock_name:
    tabs_top = st.tabs(["Annual Form", "Quarterly Form"])

    # ---------- Annual ----------
    with tabs_top[0]:
        st.markdown(
            section("📅 Annual Financial Data", "Categories loaded from config presets", "info"),
            unsafe_allow_html=True,
        )

        st.subheader("Annual Financial Data")

        years_for_stock = sorted(
            pd.to_numeric(
                df[(df["Name"] == stock_name) & (df["IsQuarter"] != True)]["Year"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        years = st.multiselect(
            "Years to edit/add (Annual)",
            options=[y for y in range(2000, 2036)],
            default=years_for_stock or [2023],
        )

        cat_dict = _get_categories(bucket, quarterly=False)
        flat_annual = _cats(bucket, quarterly=False)

        tab_annual = st.tabs([f"Year {y}" for y in years]) if years else []
        annual_data = {}

        for i, year in enumerate(years):
            with tab_annual[i]:
                st.markdown(f"#### Year: {year}")
                row = df[
                    (df["Name"] == stock_name)
                    & (df["Year"] == year)
                    & (df["IsQuarter"] != True)
                ]
                prefill = row.iloc[0].to_dict() if not row.empty else {}
                year_data = {}

                # ---- Pretty confirm (Annual clear) ----
                _has_dialog = hasattr(st, "dialog")
                n_fields_annual = len([f for f in flat_annual if f.get("key")])

                def _do_clear_annual(y: int):
                    # actually clear just this year's inputs in session_state (not the saved CSV)
                    for f in flat_annual:
                        k = f.get("key", "")
                        if not k:
                            continue
                        ss = f"{k}_{y}_annual"
                        st.session_state[ss] = None
                        tkey = ss + "_txt"
                        if tkey in st.session_state:
                            st.session_state[tkey] = ""
                    # NEW: prevent immediate re-autofill on rerun
                    mark_autofill_suppressed("annual", y)
                    st.toast(f"Cleared inputs for Year {y} (annual only). Saved data unchanged until you click Save.", icon="🧹")
                    _safe_rerun()

                if _has_dialog:
                    @st.dialog(f"Clear Year {year} (Annual)?")
                    def _confirm_clear_annual_dialog(_y=year, _n=n_fields_annual):
                        # nice KPI cards
                        items = [
                            {"label": "Scope", "value": "Annual", "badge": "Form"},
                            {"label": "Period", "value": str(_y)},
                            {"label": "Fields affected", "value": f"{_n}"},
                        ]
                        render_stat_cards(items, columns=3)
                        st.info("This blanks inputs **on this tab only**. Your saved data isn’t touched until you press **Save All Annual Changes**.", icon="✅")

                        c1, c2 = st.columns(2)
                        with c1:
                            st.button("Cancel", key=f"cancel_clear_{_y}_annual")
                        with c2:
                            if st.button("Yes, clear now", type="primary", key=f"go_clear_{_y}_annual"):
                                _do_clear_annual(_y)

                    # launcher button → opens dialog
                    st.button("🧹 Clear this year’s inputs", key=f"reset_{year}_annual_open", type="secondary", on_click=_confirm_clear_annual_dialog)
                else:
                    # fallback (Streamlit < 1.26): inline confirm
                    if st.button("🧹 Clear this year’s inputs", key=f"reset_{year}_annual_open", type="secondary"):
                        st.session_state[f"show_inline_clear_{year}_annual"] = True

                    if st.session_state.get(f"show_inline_clear_{year}_annual", False):
                        with st.expander(f"Confirm clear: Annual {year}", expanded=True):
                            items = [
                                {"label": "Scope", "value": "Annual", "badge": "Form"},
                                {"label": "Period", "value": str(year)},
                                {"label": "Fields affected", "value": f"{n_fields_annual}"},
                            ]
                            render_stat_cards(items, columns=3)
                            st.info("This blanks inputs **on this tab only**. Your saved data isn’t touched until you press **Save All Annual Changes**.", icon="✅")

                            c1, c2 = st.columns(2)
                            with c1:
                                if st.button("Cancel", key=f"cancel_inline_clear_{year}_annual"):
                                    st.session_state[f"show_inline_clear_{year}_annual"] = False
                                    _safe_rerun()
                            with c2:
                                if st.button("Yes, clear now", type="primary", key=f"go_inline_clear_{year}_annual"):
                                    st.session_state[f"show_inline_clear_{year}_annual"] = False
                                    _do_clear_annual(year)

                # ---- seed session state once (ANNUAL) ----
                seen = set()
                for f in flat_annual:
                    k = f.get("key", "")
                    if not k or k in seen:
                        continue
                    seen.add(k)
                    ssk = f"{k}_{year}_annual"
                    if ssk not in st.session_state:
                        v0 = _try_float(prefill.get(k))
                        st.session_state[ssk] = v0  # may be None (treated as missing)
                        # paired text key for calc_number_input
                        txtk = ssk + "_txt"
                        if txtk not in st.session_state:
                            st.session_state[txtk] = "" if v0 is None else str(v0)

                # ---- NEW: suppression-aware autofill (ANNUAL) ----
                unsuppress_if_user_typed("annual", year, None, flat_annual)

                if not is_autofill_suppressed("annual", year):
                    if bucket == "Banking":
                        _link_bank_tp_overrides_in_state(scope="annual", year=year)

                        # temp map to compute banking helpers without touching session_state first
                        temp = {
                            f.get("key"): st.session_state.get(f"{f.get('key')}_{year}_annual")
                            for f in flat_annual
                        }

                        compute_banking_derivatives_annual(
                            temp,
                            df=df,
                            stock_name=stock_name,
                            year=year,
                            available_keys={f["key"] for f in flat_annual if f.get("key")},
                        )

                        # push back any computed values if allowed & still blank
                        for k, v in temp.items():
                            if v is None:
                                continue
                            _set_if_available_annual(year, k, v, bucket)

                    # common non-banking deriveds
                    autofill_common_fields_annual(df=df, stock_name=stock_name, year=year, bucket=bucket)

                # render sections
                rendered = set()
                for sec in _SECTION_ORDER + [s for s in cat_dict.keys() if s not in _SECTION_ORDER]:
                    fields = cat_dict.get(sec, [])
                    if not fields:
                        continue
                    st.markdown(f"##### {sec}")
                    for f in fields:
                        lbl, key = f.get("label", f.get("key", "")), f.get("key", "")
                        if not key or key in rendered:
                            continue
                        rendered.add(key)
                        unit = f.get("unit")
                        decimals = _decimals_for_unit(unit, key)
                        default_val = _try_float(prefill.get(key))
                        year_data[key] = calc_number_input(
                            _label_with_unit(lbl, unit),
                            value=default_val,
                            key=f"{key}_{year}_annual",
                            decimals=decimals,
                            placeholder=f.get("help")
                            or _placeholder_for(key, bucket, quarterly=False),
                        )
                annual_data[year] = year_data

                render_calc_helper(
                    bucket=bucket,
                    scope="annual",
                    df=df,
                    stock_name=stock_name,
                    year=year,
                    quarter=None,
                    fallback=year_data,
                )

        if st.button("💾 Save All Annual Changes"):
            if not stock_name:
                st.error("Please enter stock name.")
                st.stop()

            available_keys = {f.get("key") for f in flat_annual if f.get("key")}
            for year, inputs_for_year in annual_data.items():
                row_up = {
                    "Name": _canon_name(stock_name),
                    "Industry": _canon_industry(industry),
                    "IndustryBucket": _canonical_bucket(bucket),
                    "Year": int(year),
                    "IsQuarter": False,
                    # no LastModified yet
                }

                # 1) Copy user inputs first
                for k in available_keys:
                    row_up[k] = _try_float(inputs_for_year.get(k))

                # 2) THEN compute derived fields (uses values now in row_up)
                compute_common_derivatives_inplace(row_up, bucket=bucket, prefix="")
                compute_banking_derivatives_annual(
                    row_up,
                    df=df,
                    stock_name=stock_name,
                    year=int(year),
                    available_keys=available_keys,
                )

                # 3) Ensure columns exist
                for c in row_up.keys():
                    if c not in df.columns:
                        df[c] = pd.NA

                # 4) Upsert with robust year match + audit (only set LastModified if changed)
                cond = (
                    (df["Name"] == stock_name)
                    & (~df["IsQuarter"].astype(bool))
                    & (pd.to_numeric(df["Year"], errors="coerce") == int(year))
                )

                # BEFORE
                before = {}
                if cond.any():
                    try:
                        before = df.loc[cond].iloc[0].to_dict()
                    except Exception:
                        before = {}

                # Compute changes EXCLUDING LastModified (and excluding Quarter for annual)
                allowed_keys = set(inputs_for_year.keys()) | _allowed_meta_fields("annual")
                changes = _diff_dict(before, row_up, allowed=allowed_keys)


                if cond.any():
                    if changes:
                        row_up["LastModified"] = _now_iso()
                        df.loc[cond, row_up.keys()] = list(row_up.values())
                        action = "update"
                    else:
                        action = "noop"
                else:
                    row_up["LastModified"] = _now_iso()
                    df = pd.concat([df, pd.DataFrame([row_up])], ignore_index=True)
                    action = "create"

                if action != "noop":
                    try:
                        after = df[
                            (df["Name"] == stock_name)
                            & (~df["IsQuarter"].astype(bool))
                            & (pd.to_numeric(df["Year"], errors="coerce") == int(year))
                        ].iloc[0].to_dict()
                    except Exception:
                        after = row_up

                    _audit_log_event(action, name=stock_name, scope="annual", year=int(year),
                                    changes=changes, before=before, after=after, source="annual_form")

            # Propagate meta
            df.loc[df["Name"] == _canon_name(stock_name), "IndustryBucket"] = _canonical_bucket(bucket)
            if industry:
                df.loc[df["Name"] == _canon_name(stock_name), "Industry"] = _canon_industry(industry)

            save_df(df)
            st.success("Saved annual changes.")
            _safe_rerun()

    # ---------- Quarterly ----------
    with tabs_top[1]:
        st.markdown(
            section("🗓 Quarterly Financial Data", "Categories mirror annual (Q_ keys)", "success"),
            unsafe_allow_html=True,
        )

        st.subheader("Quarterly Financial Data")
        all_quarters = ["Q1", "Q2", "Q3", "Q4"]

        existing_years = sorted(
            set(
                pd.to_numeric(
                    df.loc[df["Name"].astype(str).str.upper() == stock_name.upper(), "Year"],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .tolist()
            )
        )
        default_year = max(existing_years) if existing_years else 2023
        wide_years = list(range(2000, 2036))
        year_options = sorted(set(existing_years + wide_years))
        q_options = [f"{y}-{q}" for y in year_options for q in all_quarters]

        dfq = df[
            (df["Name"].astype(str).str.upper() == stock_name.upper()) & (df["IsQuarter"] == True)
        ].copy()
        dfq["Quarter"] = dfq["Quarter"].astype(str).str.strip().str.upper()
        dfq["Year"] = pd.to_numeric(dfq["Year"], errors="coerce").astype("Int64")
        existing_tokens = [
            f"{int(y)}-{q}"
            for y, q in zip(dfq["Year"], dfq["Quarter"])
            if pd.notna(y) and q in all_quarters
        ]
        q_order = {q: i for i, q in enumerate(all_quarters)}
        existing_tokens = sorted(
            set(existing_tokens), key=lambda t: (int(t.split("-")[0]), q_order[t.split("-")[1]])
        )
        default_tokens = existing_tokens or [f"{default_year}-Q4"]

        sel_tokens = st.multiselect(
            "Select quarters to edit/add",
            options=q_options,
            default=default_tokens,
            key="q_form_tokens",
        )

        parsed, labels = [], []
        for token in sel_tokens:
            try:
                y_str, q = token.split("-", 1)
                y = int(y_str)
                q = q.strip().upper()
                if q in all_quarters:
                    parsed.append((y, q))
                    labels.append(f"{q} {y}")
            except Exception:
                pass

        cats_q = _get_categories(bucket, quarterly=True)
        flat_q = _cats(bucket, quarterly=True)
        tabs_q = st.tabs(labels) if labels else []

        for i, (y, q) in enumerate(parsed):
            with tabs_q[i]:
                st.markdown(f"#### {q} {y}")
                mask_q = (
                    (df["Name"].astype(str).str.upper() == stock_name.upper())
                    & (df["IsQuarter"] == True)
                    & (pd.to_numeric(df["Year"], errors="coerce") == int(y))
                    & (df["Quarter"].astype(str).str.strip().str.upper() == q)
                )
                row_q = df[mask_q]
                prefill_q = row_q.iloc[0].to_dict() if not row_q.empty else {}

                # ---- Pretty confirm (Quarterly clear) ----
                _has_dialog = hasattr(st, "dialog")
                n_fields_q = len([f for f in flat_q if f.get("key")])

                def _do_clear_quarter(y: int, qtr: str):
                    for f in flat_q:
                        k = f.get("key", "")
                        if not k:
                            continue
                        base = f"{k}_{y}_{qtr}_q"
                        st.session_state[base] = None
                        txt = base + "_txt"
                        if txt in st.session_state:
                            st.session_state[txt] = ""
                    # NEW: prevent immediate re-autofill on rerun
                    mark_autofill_suppressed("quarterly", y, qtr)
                    st.toast(f"Cleared inputs for {qtr} {y} (quarter only). Saved data unchanged until you click Save.", icon="🧹")
                    _safe_rerun()

                if _has_dialog:
                    @st.dialog(f"Clear {q} {y} (Quarterly)?")
                    def _confirm_clear_quarter_dialog(_y=y, _q=q, _n=n_fields_q):
                        items = [
                            {"label": "Scope", "value": "Quarterly", "badge": "Form"},
                            {"label": "Period", "value": f"{_q} {_y}"},
                            {"label": "Fields affected", "value": f"{_n}"},
                        ]
                        render_stat_cards(items, columns=3)
                        st.info("This blanks inputs **on this tab only**. Your saved data isn’t touched until you press **Save ALL selected quarters**.", icon="✅")

                        c1, c2 = st.columns(2)
                        with c1:
                            st.button("Cancel", key=f"cancel_clear_{_y}_{_q}_q")
                        with c2:
                            if st.button("Yes, clear now", type="primary", key=f"go_clear_{_y}_{_q}_q"):
                                _do_clear_quarter(_y, _q)

                    st.button("🧹 Clear this quarter’s inputs", key=f"reset_{y}_{q}_q_open", type="secondary", on_click=_confirm_clear_quarter_dialog)
                else:
                    if st.button("🧹 Clear this quarter’s inputs", key=f"reset_{y}_{q}_q_open", type="secondary"):
                        st.session_state[f"show_inline_clear_{y}_{q}_q"] = True

                    if st.session_state.get(f"show_inline_clear_{y}_{q}_q", False):
                        with st.expander(f"Confirm clear: {q} {y}", expanded=True):
                            items = [
                                {"label": "Scope", "value": "Quarterly", "badge": "Form"},
                                {"label": "Period", "value": f"{q} {y}"},
                                {"label": "Fields affected", "value": f"{n_fields_q}"},
                            ]
                            render_stat_cards(items, columns=3)
                            st.info("This blanks inputs **on this tab only**. Your saved data isn’t touched until you press **Save ALL selected quarters**.", icon="✅")

                            c1, c2 = st.columns(2)
                            with c1:
                                if st.button("Cancel", key=f"cancel_inline_clear_{y}_{q}_q"):
                                    st.session_state[f"show_inline_clear_{y}_{q}_q"] = False
                                    _safe_rerun()
                            with c2:
                                if st.button("Yes, clear now", type="primary", key=f"go_inline_clear_{y}_{q}_q"):
                                    st.session_state[f"show_inline_clear_{y}_{q}_q"] = False
                                    _do_clear_quarter(y, q)


                # ---- seed (QUARTERLY) ----
                rendered = set()
                for f in flat_q:
                    key = f.get("key", "")
                    if not key or key in rendered:
                        continue
                    rendered.add(key)
                    wkey = f"{key}_{y}_{q}_q"
                    if wkey not in st.session_state:
                        v0 = _try_float(prefill_q.get(key))
                        st.session_state[wkey] = v0  # may be None (missing)
                        # paired text key for calc_number_input
                        txt = wkey + "_txt"
                        if txt not in st.session_state:
                            st.session_state[txt] = "" if v0 is None else str(v0)

                # ---- NEW: suppression-aware autofill (QUARTERLY) ----
                unsuppress_if_user_typed("quarterly", int(y), q, flat_q)

                if not is_autofill_suppressed("quarterly", int(y), q):
                    if bucket == "Banking":
                        _link_bank_tp_overrides_in_state(scope="quarterly", year=int(y), quarter=q)

                        tmp = {
                            f.get("key"): st.session_state.get(f"{f.get('key')}_{y}_{q}_q")
                            for f in flat_q
                        }

                        compute_banking_derivatives_quarterly(
                            tmp,
                            df=df,
                            stock_name=stock_name,
                            year=int(y),
                            quarter=q,
                            available_keys={f["key"] for f in flat_q if f.get("key")},
                        )

                        for k, v in tmp.items():
                            if v is None:
                                continue
                            _set_if_available_quarterly(int(y), q, k, v, bucket)

                    # common non-banking deriveds
                    autofill_common_fields_quarterly(df=df, stock_name=stock_name, year=int(y), quarter=q, bucket=bucket)

                # render sections
                rendered_keys = set()
                for sec in _SECTION_ORDER + [s for s in cats_q.keys() if s not in _SECTION_ORDER]:
                    fields = cats_q.get(sec, [])
                    if not fields:
                        continue
                    st.markdown(f"##### {sec}")
                    for f in fields:
                        lbl, key = f.get("label", f.get("key", "")), f.get("key", "")
                        if not key or key in rendered_keys:
                            continue
                        rendered_keys.add(key)
                        unit = f.get("unit")
                        decimals = _decimals_for_unit(unit, key)
                        wkey = f"{key}_{y}_{q}_q"
                        default_val = st.session_state.get(wkey, _try_float(prefill_q.get(key)))

                        _ = calc_number_input(
                            _label_with_unit(lbl, unit),
                            value=default_val,
                            key=wkey,
                            decimals=decimals,
                            placeholder=f.get("help")
                            or _placeholder_for(key, bucket, quarterly=True),
                        )

                render_calc_helper(
                    bucket=bucket,
                    scope="quarterly",
                    df=df,
                    stock_name=stock_name,
                    year=int(y),
                    quarter=q,
                    fallback=prefill_q,
                )

                # delete button for this quarter
                dcol, _ = st.columns([1, 3])
                with dcol:
                    if st.button(f"🗑️ Delete {q} {y}", key=f"delete_{y}_{q}_q"):
                        cond = (
                            (df["Name"] == stock_name)
                            & (df["IsQuarter"] == True)
                            & (pd.to_numeric(df["Year"], errors="coerce") == int(y))
                            & (df["Quarter"].astype(str).str.strip().str.upper() == q)
                        )

                        if cond.any():
                            df.drop(df[cond].index, inplace=True)
                            save_df(df)
                            st.warning(f"Deleted {q} {y} for {stock_name}.")
                            _safe_rerun()
                        else:
                            st.info("No row to delete for this selection.")

        st.markdown("---")
        if st.button("💾 Save ALL selected quarters", key="save_all_selected_quarters"):
            saved = 0
            available_q_keys = {f.get("key") for f in flat_q if f.get("key")}

            for (y, q) in parsed:
                new_row = {
                    "Name": _canon_name(stock_name),
                    "Industry": _canon_industry(industry),
                    "IndustryBucket": _canonical_bucket(bucket),
                    "Year": int(y),
                    "IsQuarter": True,
                    "Quarter": q,
                    # no LastModified yet
                }

                # ✅ Preserve blanks as None (do NOT coerce to 0)
                for col in available_q_keys:
                    ss_key = f"{col}_{y}_{q}_q"
                    new_row[col] = _try_float(st.session_state.get(ss_key))
                    
                # Derived fields may fill some Nones using other inputs
                compute_common_derivatives_inplace(new_row, bucket=bucket, prefix="Q_")
                compute_banking_derivatives_quarterly(
                    new_row,
                    df=df,
                    stock_name=stock_name,
                    year=int(y),
                    quarter=q,
                    available_keys=available_q_keys,
                )

                # Ensure columns exist and upsert as before
                for c in new_row.keys():
                    if c not in df.columns:
                        df[c] = pd.NA

                # Upsert + audit
                cond = (
                    (df["Name"] == stock_name)
                    & (df["IsQuarter"] == True)
                    & (df["Year"] == int(y))
                    & (df["Quarter"] == q)
                )

                # BEFORE
                before = {}
                if cond.any():
                    try:
                        before = df.loc[cond].iloc[0].to_dict()
                    except Exception:
                        before = {}

                # Compute changes EXCLUDING LastModified
                allowed_q_keys = set(available_q_keys) | _allowed_meta_fields("quarterly")
                changes = _diff_dict(before, new_row, allowed=allowed_q_keys)


                # UPSERT only stamps LastModified when changed
                if cond.any():
                    if changes:
                        new_row["LastModified"] = _now_iso()
                        df.loc[cond, new_row.keys()] = list(new_row.values())
                        action = "update"
                    else:
                        action = "noop"
                else:
                    new_row["LastModified"] = _now_iso()
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    action = "create"

                if action != "noop":
                    try:
                        after = df[
                            (df["Name"] == _canon_name(stock_name))
                            & (df["IsQuarter"] == True)
                            & (df["Year"] == int(y))
                            & (df["Quarter"] == q)
                        ].iloc[0].to_dict()
                    except Exception:
                        after = new_row

                    _audit_log_event(action, name=_canon_name(stock_name), scope="quarterly",
                                    year=int(y), quarter=q, changes=changes,
                                    before=before, after=after, source="quarter_form")

                saved += 1

            df.loc[df["Name"] == _canon_name(stock_name), "IndustryBucket"] = _canonical_bucket(bucket)
            if industry:
                df.loc[df["Name"] == _canon_name(stock_name), "Industry"] = _canon_industry(industry)

            save_df(df)
            buf_key = f"qeb_quarter_{stock_name}_buf"
            if buf_key in st.session_state:
                del st.session_state[buf_key]
            st.success(f"Saved {saved} quarter(s) for {stock_name}.")
            _safe_rerun()
else:
    st.info("Tip: enter a Stock Name above to add/edit annual & quarterly values.")

# =================================================================
# QUICK EDIT BY STOCK (Annual & Quarterly)
# =================================================================
st.divider()
st.markdown(
    section("🛠 Quick Edit by Stock", "Fast editing for existing rows", "warning"),
    unsafe_allow_html=True,
)

all_rows = df.copy()

# Filters
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    industries = ["All"] + sorted(
        [x for x in all_rows.get("Industry", pd.Series(dtype="string")).dropna().unique()]
    )
    f_industry = st.selectbox(
        "Filter by Industry (free text)", industries, index=0, key="qeb_industry"
    )
with c2:
    buckets = ["All"] + list(config.INDUSTRY_BUCKETS)
    f_bucket = st.selectbox("Filter by Industry Bucket", buckets, index=0, key="qeb_bucket")
with c3:
    f_query = st.text_input("🔎 Search name / industry / bucket", key="qeb_search")

if f_industry != "All":
    all_rows = all_rows[all_rows.get("Industry", "").astype("string") == f_industry]
if f_bucket != "All":
    if "IndustryBucket" not in all_rows.columns:
        all_rows["IndustryBucket"] = ""
    all_rows = all_rows[all_rows["IndustryBucket"] == f_bucket]
if f_query.strip():
    q = f_query.strip().lower()
    if "IndustryBucket" not in all_rows.columns:
        all_rows["IndustryBucket"] = ""
    all_rows = all_rows[
        all_rows["Name"].astype("string").str.lower().str.contains(q, na=False)
        | all_rows["Industry"].astype("string").str.lower().str.contains(q, na=False)
        | all_rows["IndustryBucket"].astype("string").str.lower().str.contains(q, na=False)
    ]


def _empty_editor_frame(all_columns, required_cols):
    cols = list(dict.fromkeys(required_cols + [c for c in all_columns if c not in required_cols]))
    return pd.DataFrame(columns=cols)


if all_rows.empty:
    st.info("No rows for the current filter.")
else:
    names = sorted(
        [n for n in all_rows["Name"].astype("string").dropna().unique() if str(n).strip() != ""]
    )
    name = st.selectbox("Pick a stock to edit", names, key="qeb_pick")

    for name in [name] if name else []:
        st.markdown("---")
        expanded_now = (name == st.session_state.get("__open_qe_for"))
        with st.expander(str(name), expanded=expanded_now):

            # Quick current-price save
            mask_name = df["Name"].astype(str).str.upper() == str(name).strip().upper()
            cur_default = 0.0
            if "CurrentPrice" in df.columns and df.loc[mask_name, "CurrentPrice"].notna().any():
                cur_default = float(df.loc[mask_name, "CurrentPrice"].dropna().iloc[0])
            elif "SharePrice" in df.columns and df.loc[mask_name, "SharePrice"].notna().any():
                cur_default = float(df.loc[mask_name, "SharePrice"].dropna().iloc[-1])

            colcp1, colcp2 = st.columns([1, 1])
            with colcp1:
                cur_price_edit = st.number_input(
                    "Current Price (this stock)",
                    value=float(cur_default),
                    step=0.0001,
                    format="%.4f",
                    key=f"cur_price_quick_{name}",
                )
            with colcp2:
                if st.button("💾 Save current price", key=f"save_cur_price_{name}"):
                    if "CurrentPrice" not in df.columns:
                        df["CurrentPrice"] = pd.NA
                    df.loc[mask_name, "CurrentPrice"] = float(cur_price_edit)
                    df.loc[mask_name, "MetaModified"] = _now_iso()

                    save_df(df)
                    st.success("Current price saved.")
                    bkey = f"qeb_quarter_{name}_buf"
                    if bkey in st.session_state:
                        del st.session_state[bkey]
                    _safe_rerun()

            # Danger zone: delete entire stock (dialog if available)
            st.markdown("**Danger zone**")
            enable_del = st.checkbox(
                f"Tick to enable delete for {name}", key=f"qe_enable_delete_{name}"
            )
            _has_dialog = hasattr(st, "dialog")

            if _has_dialog:

                @st.dialog(f"Delete {name}?")
                def _confirm_delete_dialog_for_name(_target=name):
                    st.write(
                        f"This will permanently delete **all rows** (annual & quarterly) for **{_target}**."
                    )
                    code = st.text_input("Type DELETE to confirm", key=f"qe_del_code_{_target}")
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.button("Cancel", key=f"qe_del_cancel_{_target}")
                    with c2:
                        if st.button("Delete permanently", key=f"qe_del_go_{_target}"):
                            if (code or "").strip().upper() != "DELETE":
                                st.error("Please type DELETE to confirm.")
                                return
                            _mask = (df["Name"].astype(str).str.upper() == str(_target).upper())
                            _removed = int(_mask.sum())
                            if _removed > 0:
                                rows_to_del = df[_mask].copy()
                                df.drop(df[_mask].index, inplace=True)
                                save_df(df)
                                for _, rr in rows_to_del.iterrows():
                                    _audit_log_event(
                                        "delete",
                                        name=str(rr.get("Name","")),
                                        scope=("quarterly" if bool(rr.get("IsQuarter")) else "annual"),
                                        year=(int(rr["Year"]) if pd.notna(rr.get("Year")) else None),
                                        quarter=(str(rr.get("Quarter")) if pd.notna(rr.get("Quarter")) else None),
                                        before=rr.to_dict(),
                                        changes={"__deleted__":[True, False]},
                                        source="delete_entire_stock"
                                    )
                                _bkey = f"qeb_quarter_{_target}_buf"
                                if _bkey in st.session_state:
                                    del st.session_state[_bkey]
                                st.warning(f"Deleted {_removed} row(s) for {_target}.")
                            else:
                                st.info("No rows found for this stock.")

                            _safe_rerun()

                st.button(
                    "🗑️ Delete ENTIRE stock",
                    key=f"qe_del_btn_{name}",
                    disabled=not enable_del,
                    on_click=_confirm_delete_dialog_for_name,
                )
            else:
                if st.button(
                    "🗑️ Delete ENTIRE stock", key=f"qe_del_btn_{name}", disabled=not enable_del
                ):
                    st.session_state[f"qe_show_inline_confirm_{name}"] = True
                if st.session_state.get(f"qe_show_inline_confirm_{name}", False):
                    with st.expander(f"Confirm delete: {name}", expanded=True):
                        code = st.text_input(
                            "Type DELETE to confirm", key=f"qe_del_code_inline_{name}"
                        )
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            if st.button("Cancel", key=f"qe_del_cancel_inline_{name}"):
                                st.session_state[f"qe_show_inline_confirm_{name}"] = False
                                st.rerun()
                        with c2:
                            if st.button("Delete permanently", key=f"qe_del_go_inline_{name}"):
                                if (code or "").strip().upper() != "DELETE":
                                    st.error("Please type DELETE to confirm.")
                                else:
                                    _mask = (df["Name"].astype(str).str.upper() == str(name).upper())
                                    _removed = int(_mask.sum())
                                    if _removed > 0:
                                        rows_to_del = df[_mask].copy()
                                        df.drop(df[_mask].index, inplace=True)
                                        save_df(df)
                                        for _, rr in rows_to_del.iterrows():
                                            _audit_log_event(
                                                "delete",
                                                name=str(rr.get("Name","")),
                                                scope=("quarterly" if bool(rr.get("IsQuarter")) else "annual"),
                                                year=(int(rr["Year"]) if pd.notna(rr.get("Year")) else None),
                                                quarter=(str(rr.get("Quarter")) if pd.notna(rr.get("Quarter")) else None),
                                                before=rr.to_dict(),
                                                changes={"__deleted__":[True, False]},
                                                source="delete_entire_stock"
                                            )
                                        _bkey = f"qeb_quarter_{name}_buf"
                                        if _bkey in st.session_state:
                                            del st.session_state[_bkey]
                                        st.warning(f"Deleted {_removed} row(s) for {name}.")
                                    else:
                                        st.info("No rows found for this stock.")

                                    st.session_state[f"qe_show_inline_confirm_{name}"] = False
                                    _safe_rerun()

            tabs = st.tabs(["Annual", "Quarterly"])
            bucket_default_this = _infer_bucket_for_stock(df, name, fallback="General")
            industry_default_this = _infer_industry_for_stock(df, name, fallback="")

            # ----------------- Annual Quick Edit -----------------
            with tabs[0]:
                av = (
                    df[(df["Name"] == name) & (~df["IsQuarter"].astype(bool))]
                    .sort_values("Year")
                    .reset_index(drop=True)
                    .copy()
                )

                allowed_a_keys = _bucket_allowed_keys(bucket_default_this, quarterly=False)

                if av.empty:
                    av = _empty_editor_frame(
                        df.columns.tolist(),
                        required_cols=["Name", "Industry", "Year", "IsQuarter", "Quarter"],
                    )
                    av.loc[:, "Name"] = name
                    av.loc[:, "Industry"] = industry_default_this
                    av.loc[:, "IndustryBucket"] = bucket_default_this
                    av.loc[:, "Year"] = pd.Series(dtype="Int64")
                    av.loc[:, "IsQuarter"] = False
                    av.loc[:, "Quarter"] = ""
                    av.insert(0, "RowKey", "")
                    av.insert(1, "Delete", False)
                else:
                    av["Year"] = pd.to_numeric(av["Year"], errors="coerce").astype("Int64")

                    def _rk(r):
                        y = r.get("Year")
                        return f"{r['Name']}|{int(y)}|A" if pd.notna(y) else ""

                    av.insert(0, "RowKey", av.apply(_rk, axis=1))
                    av.insert(1, "Delete", False)

                for col in allowed_a_keys:
                    if col not in av.columns:
                        av[col] = pd.NA
                if "IndustryBucket" not in av.columns:
                    av["IndustryBucket"] = bucket_default_this

                base_a = ["Delete", "Name", "Industry", "IndustryBucket", "Year"]
                extra_a = [
                    c
                    for c in av.columns
                    if c in allowed_a_keys
                    and c
                    not in {"IsQuarter", "Quarter", "Name", "Industry", "IndustryBucket", "Year"}
                ]
                allowed_a = list(dict.fromkeys(base_a + extra_a + ["LastModified", "RowKey"]))  # NEW include LastModified

                av_display = av[[c for c in allowed_a if c in av.columns]].copy()
                av_display = av_display.loc[:, ~av_display.columns.duplicated()]
                av_display["Industry"] = av_display.get("Industry", pd.Series(dtype="string")).astype(
                    "string"
                )
                av_display["IndustryBucket"] = av_display.get(
                    "IndustryBucket", pd.Series(dtype="string")
                ).astype("string")
                av_display["Year"] = pd.to_numeric(av_display.get("Year"), errors="coerce").astype(
                    "Int64"
                )
                if "Delete" in av_display.columns:
                    av_display["Delete"] = av_display["Delete"].fillna(False).astype(bool)
                if "RowKey" in av_display.columns:
                    av_display["RowKey"] = av_display["RowKey"].astype("string")

                # ✅ Wrap editor + button in a form so edits commit atomically
                with st.form(f"form_annual_editor_{name}", clear_on_submit=False):
                    edited_a = st.data_editor(
                        av_display,
                        use_container_width=True,
                        height=360,
                        hide_index=True,
                        num_rows="dynamic",
                        column_order=allowed_a,
                        column_config={
                            "RowKey": st.column_config.TextColumn(
                                "RowKey", help="Internal key", disabled=True
                            ),
                            "Delete": st.column_config.CheckboxColumn(
                                "Delete", help="Tick to delete this year"
                            ),
                            "Name": st.column_config.TextColumn("Name", disabled=True),
                            "Industry": st.column_config.TextColumn(
                                "Industry (free text)", help="Optional"
                            ),
                            "IndustryBucket": st.column_config.SelectboxColumn(
                                "Industry Bucket",
                                options=list(config.INDUSTRY_BUCKETS),
                                help="Pick a bucket",
                            ),
                            "Year": st.column_config.NumberColumn("Year", format="%d"),
                            "LastModified": st.column_config.TextColumn("Last Modified", disabled=True),  # NEW
                        },
                        key=f"qeb_annual_{name}",
                    )
                    submit_a = st.form_submit_button(f"💾 Save Annual for {name}")

                if submit_a:
                    if edited_a.empty:
                        del_keys = set()
                        keep = edited_a
                    else:
                        del_col = edited_a.get("Delete")
                        if isinstance(del_col, pd.Series):
                            del_mask = del_col.fillna(False).astype(bool)
                        else:
                            del_mask = pd.Series(False, index=edited_a.index)
                        del_keys = set(
                            edited_a.loc[del_mask, "RowKey"].astype(str).tolist()
                        )
                        keep = edited_a.loc[~del_mask].copy()

                    # Upserts — allow saving rows that only have Year + meta
                    for _, er in keep.iterrows():
                        if pd.isna(er.get("Year")):
                            continue
                        y = int(er["Year"])
                        ind_raw = er.get("Industry")
                        bucket_raw = er.get("IndustryBucket")
                        ind_val = _canon_industry(ind_raw) or industry_default_this
                        bucket_val = _canonical_bucket(bucket_raw, bucket_default_this)

                        row_up = {
                            "Name": name,
                            "Industry": ind_val,
                            "IndustryBucket": bucket_val,
                            "Year": y,
                            "IsQuarter": False,
                            # no LastModified yet
                        }

                        # Include any edited financials that are allowed for this bucket
                        for c in keep.columns:
                            if c in (
                                "RowKey","Delete","Name","Industry","IndustryBucket","Year","IsQuarter","Quarter",
                            ):
                                continue
                            if c in allowed_a_keys:
                                row_up[c] = _try_float(er.get(c))

                        for c in row_up.keys():
                            if c not in df.columns:
                                df[c] = pd.NA

                        compute_common_derivatives_inplace(row_up, bucket=bucket_default_this, prefix="")
                        
                        # Ensure columns exist AFTER computes (covers new derived keys)
                        for c in row_up.keys():
                            if c not in df.columns:
                                df[c] = pd.NA       

                        cond = (
                            (df["Name"] == name)
                            & (pd.to_numeric(df["Year"], errors="coerce") == y)
                            & (~df["IsQuarter"].astype(bool))
                        )

                        # BEFORE
                        before = {}
                        if cond.any():
                            try:
                                before = df.loc[cond].iloc[0].to_dict()
                            except Exception:
                                before = {}

                        # diff EXCLUDING LastModified (and excluding Quarter for annual)
                        allowed = set(allowed_a_keys) | _allowed_meta_fields("annual")

                        changes = _diff_dict(before, row_up, allowed=allowed)

                        # UPSERT only if changed; stamp LastModified on change/create
                        if cond.any():
                            if changes:
                                row_up["LastModified"] = _now_iso()
                                df.loc[cond, row_up.keys()] = list(row_up.values())
                                action = "update"
                            else:
                                action = "noop"
                        else:
                            row_up["LastModified"] = _now_iso()
                            df = pd.concat([df, pd.DataFrame([row_up])], ignore_index=True)
                            action = "create"

                        if action != "noop":
                            try:
                                after = df[
                                    (df["Name"] == name)
                                    & (~df["IsQuarter"].astype(bool))
                                    & (pd.to_numeric(df["Year"], errors="coerce") == y)
                            ].iloc[0].to_dict()
                            except Exception:
                                after = row_up

                            _audit_log_event(action, name=name, scope="annual", year=y,
                                            changes=changes, before=before, after=after, source="quick_annual")

                    # Deletions
                    for key_del in del_keys:
                        try:
                            s_name, s_year, _ = key_del.split("|")
                            mask = (
                                (df["Name"] == s_name)
                                & (~df["IsQuarter"].astype(bool))
                                & (pd.to_numeric(df["Year"], errors="coerce") == int(s_year))
                            )
                            if mask.any():
                                try:
                                    before = df.loc[mask].iloc[0].to_dict()
                                except Exception:
                                    before = {}
                                df.drop(df[mask].index, inplace=True)
                                _audit_log_event("delete", name=s_name, scope="annual",
                                                 year=int(s_year), before=before,
                                                 changes={"__deleted__":[True, False]},
                                                 source="quick_annual")
                        except Exception:
                            pass


                    # Propagate bucket and industry if edited in any row
                    try:
                        nb_series = (
                            keep.get("IndustryBucket", pd.Series(dtype="string"))
                            .astype("string")
                            .str.strip()
                        )
                        nb_series = nb_series[nb_series != ""]
                        if not nb_series.empty:
                            nb_raw = nb_series.mode().iloc[0]
                            new_bucket = _canonical_bucket(nb_raw, bucket_default_this)
                            if new_bucket in config.INDUSTRY_BUCKETS:
                                df.loc[df["Name"] == name, "IndustryBucket"] = new_bucket
                    except Exception:
                        pass
                    try:
                        ni_series = (
                            keep.get("Industry", pd.Series(dtype="string"))
                            .astype("string")
                            .str.strip()
                        )
                        ni_series = ni_series[ni_series != ""]
                        if not ni_series.empty:
                            df.loc[df["Name"] == name, "Industry"] = _canon_industry(
                                ni_series.mode().iloc[0]
                            )
                    except Exception:
                        pass

                    save_df(df)
                    st.success(f"Saved annual changes for {name}.")
                    _safe_rerun()

            # ----------------- Quarterly Quick Edit -----------------
            with tabs[1]:
                quarters = ["—", "Q1", "Q2", "Q3", "Q4"]
                qv = (
                    df[(df["Name"] == name) & (df["IsQuarter"] == True)]
                    .sort_values(["Year", "Quarter"])
                    .reset_index(drop=True)
                    .copy()
                )

                allowed_q_keys = _bucket_allowed_keys(bucket_default_this, quarterly=True)

                if qv.empty:
                    qv = _empty_editor_frame(
                        df.columns.tolist(),
                        required_cols=["Name", "Industry", "Year", "Quarter", "IsQuarter"],
                    )
                    qv.loc[:, "Name"] = name
                    qv.loc[:, "Industry"] = industry_default_this
                    qv.loc[:, "IndustryBucket"] = bucket_default_this
                    qv.loc[:, "Year"] = pd.Series(dtype="Int64")
                    qv.loc[:, "Quarter"] = "—"
                    qv.loc[:, "IsQuarter"] = True
                else:
                    qv["Quarter"] = qv["Quarter"].astype(str).str.strip().str.upper()
                    qv["Quarter"] = qv["Quarter"].where(qv["Quarter"].isin(quarters[1:]), "—")

                for col in allowed_q_keys:
                    if col not in qv.columns:
                        qv[col] = pd.NA
                if "IndustryBucket" not in qv.columns:
                    qv["IndustryBucket"] = bucket_default_this

                base_q = ["Name", "Industry", "IndustryBucket", "Year", "Quarter"]
                extra_q = [
                    c
                    for c in qv.columns
                    if c in allowed_q_keys
                    and c not in {"IsQuarter", "Name", "Industry", "IndustryBucket", "Year", "Quarter"}
                ]
                allowed_q = list(dict.fromkeys(base_q + extra_q + ["LastModified", "Delete", "RowKey"]))  # NEW include LastModified

                state_key = f"qeb_quarter_{name}_buf"

                if state_key not in st.session_state:
                    buf = qv[[c for c in allowed_q if c not in ("RowKey", "Delete")]].copy()
                    st.session_state[state_key] = buf
                else:
                    buf = st.session_state[state_key]

                disp = buf.copy()
                disp["Name"] = name
                ind = (
                    disp["Industry"]
                    .astype("string")
                    .str.strip()
                    .replace({"None": "", "none": "", "NaN": "", "nan": ""})
                )
                disp["Industry"] = ind.where(ind != "", industry_default_this)
                disp["IndustryBucket"] = (
                    disp.get("IndustryBucket", pd.Series(dtype="string"))
                    .astype("string")
                    .str.strip()
                    .where(lambda s: s.notna() & (s != ""), bucket_default_this)
                )
                disp["Quarter"] = (
                    disp["Quarter"]
                    .astype("string")
                    .str.strip()
                    .str.upper()
                    .where(disp["Quarter"].isin(quarters[1:]), "—")
                )
                disp["Year"] = pd.to_numeric(disp["Year"], errors="coerce").astype("Int64")
                disp["RowKey"] = disp.apply(
                    lambda r: f"{name}|{int(r['Year'])}|{r['Quarter']}|Q"
                    if pd.notna(r["Year"]) and r["Quarter"] in quarters[1:]
                    else "",
                    axis=1,
                ).astype("string")

                with st.form(f"form_quarter_editor_{name}", clear_on_submit=False):
                    disp["Delete"] = False
                    edited_q = st.data_editor(
                        disp.loc[:, ~disp.columns.duplicated()],
                        use_container_width=True,
                        height=380,
                        hide_index=True,
                        num_rows="dynamic",
                        column_order=allowed_q,
                        column_config={
                            "RowKey": st.column_config.TextColumn(
                                "RowKey",
                                help="Auto = Name|Year|Quarter|Q",
                                disabled=True,
                                width="large",
                            ),
                            "Delete": st.column_config.CheckboxColumn(
                                "Delete", help="Tick to delete this period"
                            ),
                            "Name": st.column_config.TextColumn("Name", disabled=True),
                            "Industry": st.column_config.TextColumn(
                                "Industry", help="Auto-filled; you can change"
                            ),
                            "IndustryBucket": st.column_config.SelectboxColumn(
                                "Industry Bucket",
                                options=list(config.INDUSTRY_BUCKETS),
                                help="Bucket for industry scoring",
                            ),
                            "Year": st.column_config.NumberColumn("Year", format="%d"),
                            "Quarter": st.column_config.SelectboxColumn(
                                "Quarter", options=quarters
                            ),
                            "LastModified": st.column_config.TextColumn("Last Modified", disabled=True),  # NEW
                        },
                        key=f"qeb_quarter_{name}",
                    )
                    submit_q = st.form_submit_button(f"💾 Save Quarterly for {name}")

                if submit_q:
                    edited = edited_q.copy()

                    # deletions (NA-safe)
                    del_col = edited.get("Delete")
                    if isinstance(del_col, pd.Series):
                        del_mask = del_col.fillna(False).astype(bool)
                    else:
                        del_mask = pd.Series(False, index=edited.index)
                    del_keys = set(
                        edited.loc[del_mask, "RowKey"].dropna().astype(str).tolist()
                    )
                    for key_del in del_keys:
                        try:
                            s_name, s_year, s_quarter, _ = key_del.split("|")
                            mask = (
                                (df["Name"] == s_name)
                                & (df["IsQuarter"] == True)
                                & (pd.to_numeric(df["Year"], errors="coerce") == int(s_year))
                                & (df["Quarter"].astype(str).str.upper() == s_quarter)
                            )
                            if mask.any():
                                try:
                                    before = df.loc[mask].iloc[0].to_dict()
                                except Exception:
                                    before = {}
                                df.drop(df[mask].index, inplace=True)
                                _audit_log_event("delete", name=s_name, scope="quarterly",
                                                 year=int(s_year), quarter=s_quarter, before=before,
                                                 changes={"__deleted__":[True, False]},
                                                 source="quick_quarter")
                        except Exception:
                            pass

                    # keep in buffer (exclude deleted)
                    st.session_state[state_key] = edited.loc[~del_mask].drop(
                        columns=["RowKey", "Delete"], errors="ignore"
                    )

                    # normalise + validate
                    def _normalise_for_save(df_work: pd.DataFrame) -> pd.DataFrame:
                        out = df_work.copy()
                        out["Name"] = _canon_name(name)
                        out["Industry"] = (
                            out["Industry"]
                            .astype("string")
                            .str.strip()
                            .replace({"None": "", "none": "", "NaN": "", "nan": ""})
                            .where(lambda s: s != "", industry_default_this)
                            .map(_canon_industry)
                        )
                        # Map bucket per-row to config canonical values
                        out["IndustryBucket"] = (
                            out.get("IndustryBucket", pd.Series(dtype="string"))
                            .astype("string")
                            .str.strip()
                            .where(lambda s: s.notna() & (s != ""), bucket_default_this)
                            .map(_canonical_bucket)
                        )
                        out["Quarter"] = out["Quarter"].astype("string").str.strip().str.upper()
                        out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
                        return out

                    buf_save = _normalise_for_save(st.session_state[state_key])

                    # 🚫 Guard: year with missing quarter
                    invalid = buf_save[
                        (buf_save["Year"].notna()) & (~buf_save["Quarter"].isin(quarters[1:]))
                    ]
                    if not invalid.empty:
                        st.error(
                            "Please choose a Quarter (Q1–Q4) for rows that have a Year before saving."
                        )
                        st.stop()

                    valid = buf_save[
                        (buf_save["Year"].notna()) & (buf_save["Quarter"].isin(quarters[1:]))
                    ]

                    # upserts (only keys in allowed_q_keys)
                    for _, r in valid.iterrows():
                        y, q = int(r["Year"]), r["Quarter"]

                        # AFTER (candidate row)
                        row = {
                            "Name": name,
                            "Industry": r["Industry"],
                            "IndustryBucket": r["IndustryBucket"],
                            "Year": y,
                            "IsQuarter": True,
                            "Quarter": q,
                            # no LastModified yet
                        }
                        for col in allowed_q_keys:
                            row[col] = _try_float(r.get(col))

                        compute_common_derivatives_inplace(row, bucket=bucket_default_this, prefix="Q_")
                        compute_banking_derivatives_quarterly(
                            row, df=df, stock_name=name, year=y, quarter=q, available_keys=allowed_q_keys
                        )

                        for c in row.keys():
                            if c not in df.columns:
                                df[c] = pd.NA

                        mask = (
                            (df["Name"] == name)
                            & (df["IsQuarter"] == True)
                            & (pd.to_numeric(df["Year"], errors="coerce") == y)
                            & (df["Quarter"].astype(str).str.upper() == q)
                        )

                        # BEFORE
                        before = {}
                        if mask.any():
                            try:
                                before = df.loc[mask].iloc[0].to_dict()
                            except Exception:
                                before = {}

                        # diff EXCLUDING LastModified
                        allowed = set(allowed_q_keys) | _allowed_meta_fields("quarterly")

                        changes = _diff_dict(before, row, allowed=allowed)

                        # UPSERT only if changed; stamp LastModified on change/create
                        if mask.any():
                            if changes:
                                row["LastModified"] = _now_iso()
                                df.loc[mask, row.keys()] = list(row.values())
                                action = "update"
                            else:
                                action = "noop"
                        else:
                            row["LastModified"] = _now_iso()
                            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                            action = "create"

                        if action != "noop":
                            try:
                                after = df[
                                    (df["Name"] == name)
                                    & (df["IsQuarter"] == True)
                                    & (pd.to_numeric(df["Year"], errors="coerce") == y)
                                    & (df["Quarter"].astype(str).str.upper() == q)
                                ].iloc[0].to_dict()
                            except Exception:
                                after = row

                            _audit_log_event(action, name=name, scope="quarterly", year=y, quarter=q,
                                            changes=changes, before=before, after=after, source="quick_quarter")

                    # propagate bucket & industry if edited
                    try:
                        nb_series = (
                            edited.get("IndustryBucket", pd.Series(dtype="string"))
                            .astype("string")
                            .str.strip()
                        )
                        nb_series = nb_series[nb_series != ""]
                        if not nb_series.empty:
                            new_bucket = nb_series.mode().iloc[0]
                            if new_bucket in config.INDUSTRY_BUCKETS:
                                df.loc[df["Name"] == name, "IndustryBucket"] = new_bucket
                    except Exception:
                        pass
                    try:
                        ni_series = (
                            edited.get("Industry", pd.Series(dtype="string"))
                            .astype("string")
                            .str.strip()
                        )
                        ni_series = ni_series[ni_series != ""]
                        if not ni_series.empty:
                            df.loc[df["Name"] == name, "Industry"] = ni_series.mode().iloc[0]
                    except Exception:
                        pass

                    save_df(df)
                    st.success(f"Saved quarterly changes for {name}.")
                    _safe_rerun()
                    
# =================================================================
# PER-STOCK JSON BACKUP / RESTORE (place at bottom, after Quick Edit)
# =================================================================
st.divider()
st.markdown(
    section(
        "📦 Per-stock JSON backup/restore",
        "Download a JSON for one stock (Quarterly or Annual). Upload to fully replace that stock’s data for the chosen scope.",
        "info",
    ),
    unsafe_allow_html=True,
)

if df.empty:
    st.info("No data yet — add a stock above first.")
else:
    # Pick a stock; default to the one currently selected in Quick Edit, if any
    _names_sorted = sorted(df["Name"].dropna().astype(str).unique().tolist())
    default_pick = st.session_state.get("qeb_pick") or ( _names_sorted[0] if _names_sorted else "" )
    stock_pick = st.selectbox("Stock", _names_sorted, index=_names_sorted.index(default_pick) if default_pick in _names_sorted else 0, key="perstock_json_name")

    scope = st.radio("Scope", ["Quarterly", "Annual"], horizontal=True, key="perstock_json_scope")
    _is_q = (scope == "Quarterly")

    # Helper for filename
    import re
    def _safe_name(x: str) -> str:
        return re.sub(r"[^0-9A-Za-z]+", "_", str(x)).strip("_")

    # Slice rows for chosen stock & scope
    sel = df[
        (df["Name"].astype(str) == stock_pick)
        & (df["IsQuarter"].astype(bool) == _is_q)
    ].copy()

    # ---------- Download ----------
    col_dl, col_ul = st.columns([1, 1])
    with col_dl:
        if sel.empty:
            st.caption("No rows yet for this stock/scope.")
        else:
            try:
                payload_bytes = _df_to_json_bytes(sel)
                fname = f"{_safe_name(stock_pick)}_{'quarterly' if _is_q else 'annual'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                st.download_button(
                    "⬇️ Download selected as JSON",
                    payload_bytes,
                    file_name=fname,
                    mime="application/json",
                    use_container_width=True,
                    key="perstock_json_download",
                )
            except Exception as e:
                st.error(f"Could not prepare download: {e}")

    # ---------- Upload (REPLACE this stock+scope) ----------
    with col_ul:
        up = st.file_uploader(
            "Upload JSON to REPLACE this stock+scope",
            type=["json"],
            accept_multiple_files=False,
            key="perstock_json_upload",
        )
        mode_help = "This will **delete all existing rows** for the selected stock & scope, then import the uploaded rows."
        st.caption(mode_help)

        if up is not None:
            try:
                # Parse JSON -> DataFrame
                payload = json.loads(up.read().decode("utf-8"))
                incoming = _json_to_df(payload)
                if incoming is None or incoming.empty:
                    st.error("Uploaded JSON has no rows.")
                else:
                    # Normalise + force to the chosen stock/scope
                    #  - Keep only rows that either match the chosen stock OR have empty name (we will rewrite name)
                    #  - Coerce scope: Quarterly -> IsQuarter True (Q1–Q4); Annual -> IsQuarter False + blank Quarter
                    cleaned_rows = []
                    for _, r in incoming.iterrows():
                        row = _normalize_row_for_upsert(r.to_dict())
                        # Force name to the chosen stock
                        row["Name"] = _canon_name(stock_pick)
                        # Force scope
                        if _is_q:
                            row["IsQuarter"] = True
                            q = str(row.get("Quarter", "") or "").strip().upper()
                            if q not in {"Q1", "Q2", "Q3", "Q4"}:
                                # skip invalid quarter rows for a quarterly import
                                continue
                        else:
                            row["IsQuarter"] = False
                            row["Quarter"] = ""
                        # Year must be int-ish if present
                        try:
                            row["Year"] = int(row["Year"]) if row.get("Year") not in (None, "", "—") else None
                        except Exception:
                            row["Year"] = None

                        # Stamp meta fields we always want sane
                        row["LastModified"] = _now_iso()
                        cleaned_rows.append(row)

                    if not cleaned_rows:
                        st.error("No valid rows for the chosen scope were found in the uploaded file.")
                    else:
                        incoming_df = pd.DataFrame(cleaned_rows)

                        # Ensure all columns exist up front
                        for c in incoming_df.columns:
                            if c not in df.columns:
                                df[c] = pd.NA

                        # DELETE existing rows for this stock & scope
                        cond_del = (df["Name"].astype(str) == stock_pick) & (df["IsQuarter"].astype(bool) == _is_q)
                        rows_to_del = df[cond_del].copy()
                        if not rows_to_del.empty:
                            df.drop(rows_to_del.index, inplace=True)
                            # Audit deletes
                            for _, rr in rows_to_del.iterrows():
                                _audit_log_event(
                                    "delete",
                                    name=str(rr.get("Name","")),
                                    scope=("quarterly" if bool(rr.get("IsQuarter")) else "annual"),
                                    year=(int(rr["Year"]) if pd.notna(rr.get("Year")) else None),
                                    quarter=(str(rr.get("Quarter")) if pd.notna(rr.get("Quarter")) else None),
                                    before=rr.to_dict(),
                                    changes={"__deleted__": [True, False]},
                                    source="perstock_json_replace",
                                )

                        # APPEND new rows (full overwrite semantics)
                        df = pd.concat([df, incoming_df], ignore_index=True)

                        # Audit creates
                        for _, rr in incoming_df.iterrows():
                            _audit_log_event(
                                "create",
                                name=stock_pick,
                                scope=("quarterly" if _is_q else "annual"),
                                year=(int(rr["Year"]) if pd.notna(rr.get("Year")) else None),
                                quarter=(str(rr.get("Quarter")) if pd.notna(rr.get("Quarter")) else None),
                                changes={k: [None, rr.get(k)] for k in rr.keys() if k not in {"LastModified","MetaModified"}},
                                before={},
                                after=rr.to_dict(),
                                source="perstock_json_replace",
                            )

                        # Persist + bust caches
                        save_df(df)
                        st.success(
                            f"Replaced **{len(rows_to_del)}** existing row(s) with **{len(incoming_df)}** new "
                            f"{'quarterly' if _is_q else 'annual'} row(s) for **{stock_pick}**."
                        )
                        _safe_rerun()

            except Exception as e:
                st.error(f"Import failed: {e}")

