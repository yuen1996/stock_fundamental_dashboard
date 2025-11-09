from __future__ import annotations

import os
import re
import unicodedata
import difflib
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


# =====================================================================
# Paths
# =====================================================================

def _project_ohlc_dir_from_here() -> Optional[Path]:
    """
    Walk upwards from this file until we find a data/ohlc directory.
    """
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        candidate = p / "data" / "ohlc"
        if candidate.exists():
            return candidate
    return None


def default_ohlc_dir() -> Path:
    """
    Resolve the default OHLC directory.

    Priority:
    1) Env vars: SFD_OHLC_DIR or OHLC_DIR (if they exist)
    2) <repo_root>/data/ohlc (via walking up from this file)
    3) Fallback to a known container/server path
    """
    for env_name in ("SFD_OHLC_DIR", "OHLC_DIR"):
        env = os.getenv(env_name)
        if env and Path(env).exists():
            return Path(env)

    repo_dir = _project_ohlc_dir_from_here()
    if repo_dir:
        return repo_dir

    # Last resort: adjust this to your deployment
    return Path("/opt/stock_fundamental_dashboard/stock_fundamental_dashboard/data/ohlc")


# =====================================================================
# Text utils
# =====================================================================

def slugify(text: str) -> str:
    """
    ASCII-normalize, keep letters+digits, map separators to single underscore.
    Example:
      "Hup Seng"      -> "hup_seng"
      "GENETEC BHD"   -> "genetec_bhd"
    """
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[\W]+", "_", text)  # non-word → underscore
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()


def _filename_variants(stem: str) -> List[str]:
    """
    Index multiple shapes of a filename stem to be tolerant.
    e.g. Hup_Seng.csv → variants that all slugify to useful keys.
    """
    s = str(stem)
    variants = {
        s,
        s.lower(),
        s.replace("_", " "),
        s.replace("_", ""),
        s.replace("-", " "),
        s.replace("-", ""),
        s.replace(" ", "_"),
    }
    return sorted({slugify(v) for v in variants})


# Words to ignore when matching names to OHLC files
# (keep loosely in sync with utils/name_link.py)
_DROP = {
    "berhad", "bhd", "plc", "limited", "ltd",
    "inc", "inc.", "corp", "corporation",
    "company", "co",
    "holdings", "holding", "group", "resources",
    "technology", "technologies", "tech",
    "malaysia",
    "public", "tbk", "nv", "sa", "ag", "oyj", "ab", "as",
    "sp", "s.p.a", "spa",
    "the",
}


def _compact_key(s: str) -> str:
    """
    Normalised 'compact' key for fuzzy/loose matching:

    - ASCII normalize
    - lower-case
    - split on common separators
    - drop filler/legal words (_DROP)
    - join remaining tokens without spaces

    Examples:
      "Genetec Technology Berhad" -> "genetec"
      "Rhone Ma Bhd"              -> "rhonema"
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = s.encode("ascii", "ignore").decode("ascii").lower()

    # treat these as separators
    for ch in ["_", "-", ".", ",", "&", "(", ")", "[", "]", "/", "|"]:
        s = s.replace(ch, " ")

    tokens = [t for t in s.split() if t and t not in _DROP]
    return "".join(tokens)


# =====================================================================
# Indexing
# =====================================================================

@_cache
def _build_index(ohlc_dir: str) -> Dict[str, Path]:
    """
    Build an index of slug -> Path for all *.csv in the OHLC directory.
    Multiple variants of each filename map to the same path.
    """
    ohlc = Path(ohlc_dir)
    index: Dict[str, Path] = {}

    if not ohlc.exists():
        return index

    for p in ohlc.glob("*.csv"):
        stem = p.stem  # e.g. "Hup_Seng"
        for v in _filename_variants(stem):
            # Later variants for same key just overwrite with same path; OK.
            index[v] = p

    return index


# =====================================================================
# Resolver
# =====================================================================

def resolve_ohlc_path(
    name_or_symbol: str,
    ohlc_dir: Optional[str] = None,
) -> Tuple[Optional[Path], str]:
    """
    Resolve an OHLC CSV path from a human/ticker name.

    Returns (Path or None, how), where how ∈:
      'exact'             - direct slug match
      'compact'           - exact compact-key match (drops Bhd/Berhad/Technology/etc)
      'fuzzy:<key>'       - fuzzy match on slug keys
      'compact_fuzzy:<k>' - fuzzy match on compact keys
      'not_found'
    """
    base_dir = ohlc_dir or str(default_ohlc_dir())
    index = _build_index(base_dir)

    if not name_or_symbol or not index:
        return None, "not_found"

    # Slug and compact forms of the requested name
    key = slugify(name_or_symbol)
    c_key = _compact_key(name_or_symbol)

    # ---- 1) Exact slug match ----
    if key in index:
        return index[key], "exact"

    # ---- 2) Exact compact match ----
    # Build compact index lazily from existing slug keys
    compact_index: Dict[str, Path] = {}
    for k, p in index.items():
        ck = _compact_key(k)
        if ck and ck not in compact_index:
            compact_index[ck] = p

    if c_key and c_key in compact_index:
        return compact_index[c_key], "compact"

    # ---- 3) Fuzzy slug match ----
    keys = list(index.keys())
    if keys:
        match = difflib.get_close_matches(key, keys, n=1, cutoff=0.65)
        if match:
            return index[match[0]], f"fuzzy:{match[0]}"

    # ---- 4) Fuzzy compact match ----
    if c_key and compact_index:
        ckeys = list(compact_index.keys())
        match = difflib.get_close_matches(c_key, ckeys, n=1, cutoff=0.75)
        if match:
            return compact_index[match[0]], f"compact_fuzzy:{match[0]}"

    return None, "not_found"


# =====================================================================
# Loading
# =====================================================================

def load_ohlc(
    name_or_symbol: str,
    ohlc_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, Path, str]:
    """
    Load OHLC CSV by tolerant name.

    Returns:
        (df, path, how)

    Raises:
        FileNotFoundError if nothing suitable is found.
    """
    path, how = resolve_ohlc_path(name_or_symbol, ohlc_dir=ohlc_dir)
    if path is None:
        raise FileNotFoundError(
            f"OHLC file not found for '{name_or_symbol}'. "
            f"Searched in: {ohlc_dir or default_ohlc_dir()}"
        )

    df = pd.read_csv(path)

    # Standardize likely date column
    lower_cols = {c.lower(): c for c in df.columns}
    date_col = next(
        (lower_cols[c] for c in ["date", "timestamp", "time", "datetime"] if c in lower_cols),
        None,
    )
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)

    # Normalize OHLC naming
    rename: Dict[str, str] = {}
    for c in df.columns:
        k = c.strip().lower()
        if k in ("o", "open", "opening"):
            rename[c] = "Open"
        elif k in ("h", "high"):
            rename[c] = "High"
        elif k in ("l", "low"):
            rename[c] = "Low"
        elif k in ("c", "close", "closing", "adj close", "adjusted close"):
            rename[c] = "Close"
        elif k in ("v", "vol", "volume"):
            rename[c] = "Volume"

    if rename:
        df = df.rename(columns=rename)

    return df, path, how


# =====================================================================
# Misc helpers
# =====================================================================

def unique_key(prefix: str, name_or_symbol: str) -> str:
    """
    Stable, deterministic Streamlit widget key.

    Same (prefix, name_or_symbol) -> same key on every rerun.
    """
    return f"{slugify(prefix)}__{slugify(name_or_symbol)}"


def list_known(ohlc_dir: Optional[str] = None) -> List[str]:
    """
    List slugs of all known OHLC files (for debugging or dropdowns).
    """
    base_dir = ohlc_dir or str(default_ohlc_dir())
    return sorted(set(_build_index(base_dir).keys()))
