# utils/ohlc_resolver.py
from __future__ import annotations

import os, re, unicodedata, difflib
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import pandas as pd

# ---- Optional Streamlit-aware cache ----
try:
    import streamlit as st
    def _cache(func):
        return st.cache_data(show_spinner=False)(func)
except Exception:  # allow import outside Streamlit
    from functools import lru_cache
    def _cache(func):
        return lru_cache(maxsize=8)(func)

# ---------- Paths ----------
def _project_ohlc_dir_from_here() -> Optional[Path]:
    """
    Walk upwards from this file until we find a 'data/ohlc' folder.
    Works both in dev and when packaged.
    """
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        candidate = p / "data" / "ohlc"
        if candidate.exists():
            return candidate
    return None

def default_ohlc_dir() -> Path:
    """
    Priority:
      1. SFD_OHLC_DIR / OHLC_DIR env vars (either name)
      2. <repo>/data/ohlc discovered via _project_ohlc_dir_from_here()
      3. Fallback container path (old deployments)
    """
    for env_name in ("SFD_OHLC_DIR", "OHLC_DIR"):
        env = os.getenv(env_name)
        if env and Path(env).exists():
            return Path(env)

    repo_dir = _project_ohlc_dir_from_here()
    if repo_dir:
        return repo_dir

    # Last resort: historical server path
    return Path("/opt/stock_fundamental_dashboard/stock_fundamental_dashboard/data/ohlc")

# ---------- Text utils ----------
# Corporate/legal + generic words we want to ignore for matching
_STOPWORDS = {
    "berhad","bhd","bhd.","corp","corp.","corporation","company","co","ltd","ltd.","limited","plc",
    "resources","holdings","holding","group","technology","technologies","tech","industries","industry",
    "malaysia","public"
}

def slugify(text: str) -> str:
    """
    ascii-normalize, keep letters+digits, convert non-word to single underscore,
    lowercase.
    """
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[\W]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()

def _tokenize_slug(slug: str) -> List[str]:
    return [t for t in slug.split("_") if t and t not in _STOPWORDS]

def _filename_variants(stem: str) -> List[str]:
    """
    For a filename stem like 'ZHULIAN_CORPORATION_BERHAD', generate
    multiple shapes, all slugified.
    """
    s = str(stem or "")
    variants = {
        s, s.lower(),
        s.replace("_", " "), s.replace("_", ""),
        s.replace("-", " "), s.replace("-", ""),
        s.replace(" ", "_"),
    }
    return sorted({slugify(v) for v in variants if v})

def _strip_suffixes_slug(slug: str) -> str:
    """
    Strip trailing corporate/generic suffixes from a slug.

    e.g.
      'zhulian_corporation_berhad' -> 'zhulian'
      'malayan_banking_berhad'     -> 'malayan_banking'
    """
    parts = [p for p in slug.split("_") if p]
    while parts and parts[-1] in _STOPWORDS:
        parts.pop()
    return "_".join(parts)

def _compact(s: str) -> str:
    """Remove all non-alphanumeric chars."""
    return re.sub(r"[^a-z0-9]+", "", s)

def _add_ticker_keys(index: Dict[str, Path], key: str, path: Path) -> None:
    """
    If the slug contains something that looks like a Bursa-style code,
    index that too so user can type the code directly.

    Examples:
      - 4–5 digits
      - 4–5 digits + 1 letter (e.g. 1234A)
    """
    for match in re.findall(r"(\d{4,5}[a-z]?)", key):
        if match and match not in index:
            index[match] = path

# ---------- Indexing ----------
@_cache
def _build_index(ohlc_dir: str) -> Dict[str, Path]:
    """
    Build a tolerant index mapping many possible keys -> CSV path.

    For each '<stem>.csv', we index:
      - slugified stem
      - variants with/without spaces/_/-
      - compact versions (alnum only)
      - suffix-stripped versions (no Berhad/Bhd/Corp/etc)
      - ticker-like codes seen in the filename

    Any new CSV dropped into data/ohlc is picked up automatically.
    """
    base = Path(ohlc_dir)
    index: Dict[str, Path] = {}
    if not base.exists():
        return index

    for p in base.glob("*.csv"):
        stem = p.stem
        slugs = _filename_variants(stem)

        for s in slugs:
            if not s:
                continue

            # main slug
            index.setdefault(s, p)

            # compact slug
            c = _compact(s)
            if c:
                index.setdefault(c, p)

            # base form (remove corporate/generic suffixes)
            stripped = _strip_suffixes_slug(s)
            if stripped and stripped != s:
                index.setdefault(stripped, p)
                sc = _compact(stripped)
                if sc:
                    index.setdefault(sc, p)

            # add ticker-style keys
            _add_ticker_keys(index, s, p)
            if stripped:
                _add_ticker_keys(index, stripped, p)

    return index

# ---------- Resolve ----------
def resolve_ohlc_path(
    name_or_symbol: str,
    ohlc_dir: Optional[str] = None,
) -> Tuple[Optional[Path], str]:
    """
    Resolve a human/company name or ticker into a CSV path.

    Designed so you NEVER touch code when adding new Malaysia stocks:
      - Just drop a CSV in data/ohlc.
      - Name it with any of:
          * Full name (with/without Berhad/Bhd/Corp/Ltd),
          * Short name,
          * Bursa code (e.g. 5131),
          * Mix (e.g. 'ZHULIAN_5131', 'Zhulian_Corporation_Berhad').

    Returns (Path or None, how), where `how` is:
      - 'code'      : match via ticker-like code in the input
      - 'exact'     : exact slug match
      - 'compact'   : match after stripping non-alnum
      - 'base'      : match after stripping corporate/generic suffixes
      - 'fuzzy:<k>' : fuzzy match (only with meaningful token overlap)
      - 'not_found'
    """
    if not (name_or_symbol or "").strip():
        return None, "not_found"

    base_dir = ohlc_dir or str(default_ohlc_dir())
    index = _build_index(base_dir)
    if not index:
        return None, "not_found"

    raw = str(name_or_symbol).strip()
    key = slugify(raw)
    ckey = _compact(key)
    base_key = _strip_suffixes_slug(key)
    base_ckey = _compact(base_key)

    # 0) direct ticker-style code from raw (e.g. "5131", "5131.KL")
    m = re.search(r"(\d{4,5}[a-z]?)", raw.replace(" ", "").lower())
    code = m.group(1) if m else ""
    if code and code in index:
        return index[code], "code"

    # 1) Exact slug
    if key in index:
        return index[key], "exact"

    # 2) Compact exact
    if ckey in index:
        return index[ckey], "compact"

    # 3) Strip Berhad / Bhd / Corporation / Ltd, etc.
    if base_key and base_key in index:
        return index[base_key], "base"
    if base_ckey and base_ckey in index:
        return index[base_ckey], "base"

    # 4) Fuzzy — require meaningful token overlap (avoid SLP vs SKP collisions)
    def _acceptable_fuzzy(cand_key: str) -> bool:
        a = set(_tokenize_slug(cand_key))
        b = set(_tokenize_slug(key))
        # require at least one shared non-generic token
        return len(a & b) >= 1

    keys = list(index.keys())
    candidates = (
        difflib.get_close_matches(key, keys, n=5, cutoff=0.65)
        or difflib.get_close_matches(base_key, keys, n=5, cutoff=0.65)
        or ([code] if code else [])
    )
    for k in candidates:
        if _acceptable_fuzzy(k):
            return index[k], f"fuzzy:{k}"

    return None, "not_found"

# ---------- Loading ----------

def clean_ohlc_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize + clean OHLC data consistently across the app.

    Guarantees:
      - Date column exists, tz-naive datetime
      - Open/High/Low/Close/Volume are numeric columns (present; filled with NA if missing)
      - rows sorted by Date, duplicates dropped by Date, invalid rows removed
    """
    if df is None:
        return pd.DataFrame()

    df = df.copy()

    # --- Date column ---
    lower_cols = {str(c).strip().lower(): c for c in df.columns}
    date_col = None
    for k in ("date", "timestamp", "time", "datetime"):
        if k in lower_cols:
            date_col = lower_cols[k]
            break
    if date_col is None:
        raise ValueError("No date-like column found in OHLC CSV.")

    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    if hasattr(df["Date"], "dt"):
        try:
            df["Date"] = df["Date"].dt.tz_localize(None)
        except Exception:
            pass

    # Prefer AdjClose if Close missing
    if "Close" not in df.columns and "AdjClose" in df.columns:
        df["Close"] = df["AdjClose"]

    # --- Numeric columns ---
    for c in ("Open", "High", "Low", "Close", "Volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = pd.NA

    # --- Drop / sort / dedupe ---
    df = (
        df.dropna(subset=["Date", "Close"])
          .drop_duplicates(subset=["Date"])
          .sort_values("Date")
          .reset_index(drop=True)
    )

    # Put common columns first
    cols = df.columns.tolist()
    ordered = ["Date", "Open", "High", "Low", "Close", "Volume"]
    present = [c for c in ordered if c in cols]
    rest = [c for c in cols if c not in present]
    return df[present + rest]

def read_ohlc_csv(path: Path) -> pd.DataFrame:
    """Read an OHLC CSV and return a cleaned DataFrame."""
    df = read_ohlc_csv(path)

    return df, path, how

# ---------- Helpers ----------
def unique_key(prefix: str, name_or_symbol: str) -> str:
    """
    Stable, deterministic widget key for Streamlit.
    Same (prefix, name_or_symbol) -> same key on every rerun.
    """
    return f"{slugify(prefix)}__{slugify(name_or_symbol)}"

def list_known(ohlc_dir: Optional[str] = None) -> List[str]:
    """
    List all index keys we recognize (for debugging or dropdowns).
    """
    base_dir = ohlc_dir or str(default_ohlc_dir())
    return sorted(set(_build_index(base_dir).keys()))