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
    # Walk upwards until we find /data/ohlc
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        candidate = p / "data" / "ohlc"
        if candidate.exists():
            return candidate
    return None

# ---------- Paths ----------
def default_ohlc_dir() -> Path:
    # Priority: ENVs → repo/data/ohlc → container default path
    # Support BOTH names so every page finds the same folder.
    for env_name in ("SFD_OHLC_DIR", "OHLC_DIR"):
        env = os.getenv(env_name)
        if env and Path(env).exists():
            return Path(env)

    repo_dir = _project_ohlc_dir_from_here()
    if repo_dir:
        return repo_dir

    # Last resort: your server path
    return Path("/opt/stock_fundamental_dashboard/stock_fundamental_dashboard/data/ohlc")


def resolve_ohlc_path(name_or_symbol: str, ohlc_dir: Optional[str] = None) -> Tuple[Optional[Path], str]:
    """Return (Path or None, how) where how ∈ {'exact', 'compact', 'fuzzy:<key>', 'not_found'}."""
    base_dir = ohlc_dir or str(default_ohlc_dir())
    index = _build_index(base_dir)
    key = slugify(name_or_symbol)

    # Build a compact map (remove non-alnum) for exact compact hits
    def _compact(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s)

    compact_index: Dict[str, Path] = {}
    for k, p in index.items():
        compact_index[_compact(k)] = p

    # 1) Exact slug match
    if key in index:
        return index[key], "exact"

    # 2) Compact exact match (e.g., "rhone_ma" → "rhonema")
    ckey = _compact(key)
    if ckey in compact_index:
        return compact_index[ckey], "compact"

    # 3) Fuzzy on the slug keys
    keys = list(index.keys())
    match = difflib.get_close_matches(key, keys, n=1, cutoff=0.65)
    if match:
        return index[match[0]], f"fuzzy:{match[0]}"

    return None, "not_found"


# ---------- Text utils ----------
def slugify(text: str) -> str:
    # ascii-normalize, keep letters+digits, convert spaces/underscores/hyphens to single underscore
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[\W]+", "_", text)  # non-word → underscore
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()

def _filename_variants(stem: str) -> List[str]:
    # Index multiple shapes of a filename stem to be tolerant
    s = stem
    variants = {
        s, s.lower(),
        s.replace("_", " "), s.replace("_", ""),
        s.replace("-", " "), s.replace("-", ""),
        s.replace(" ", "_"),
    }
    return sorted({slugify(v) for v in variants})

# ---------- Indexing ----------
@_cache
def _build_index(ohlc_dir: str) -> Dict[str, Path]:
    ohlc = Path(ohlc_dir)
    index: Dict[str, Path] = {}
    for p in ohlc.glob("*.csv"):
        stem = p.stem  # e.g. Hup_Seng
        for v in _filename_variants(stem):
            index[v] = p
    return index

def resolve_ohlc_path(name_or_symbol: str, ohlc_dir: Optional[str] = None) -> Tuple[Optional[Path], str]:
    """Return (Path or None, how) where how ∈ {'exact', 'fuzzy:<key>', 'not_found'}."""
    base_dir = ohlc_dir or str(default_ohlc_dir())
    index = _build_index(base_dir)
    key = slugify(name_or_symbol)

    # 1) Exact
    if key in index:
        return index[key], "exact"

    # 2) Try fuzzy (close matches among indexed keys)
    keys = list(index.keys())
    match = difflib.get_close_matches(key, keys, n=1, cutoff=0.65)
    if match:
        return index[match[0]], f"fuzzy:{match[0]}"

    return None, "not_found"

# ---------- Loading ----------
def load_ohlc(name_or_symbol: str, ohlc_dir: Optional[str] = None) -> Tuple[pd.DataFrame, Path, str]:
    """Load OHLC CSV by tolerant name. Returns (df, path, how)."""
    path, how = resolve_ohlc_path(name_or_symbol, ohlc_dir=ohlc_dir)
    if path is None:
        raise FileNotFoundError(
            f"OHLC file not found for '{name_or_symbol}'. "
            f"Searched in: {ohlc_dir or default_ohlc_dir()}"
        )

    df = pd.read_csv(path)

    # Standardize likely date column
    lower_cols = {c.lower(): c for c in df.columns}
    date_col = next((lower_cols[c] for c in ["date", "timestamp", "time", "datetime"] if c in lower_cols), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)

    # Normalize OHLC naming
    rename = {}
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

# ---------- Streamlit-safe keys ----------
def unique_key(prefix: str, name_or_symbol: str) -> str:
    """
    Deterministic, duplicate-proof widget key.
    If same (prefix, name) is used multiple times in a run, suffix __2, __3...
    """
    base = f"{slugify(prefix)}__{slugify(name_or_symbol)}"
    try:
        import streamlit as st  # re-import inside function for CLI safety
        counter = st.session_state.setdefault("__key_counter__", {})
        count = counter.get(base, 0) + 1
        counter[base] = count
        st.session_state["__key_counter__"] = counter
        return base if count == 1 else f"{base}__{count}"
    except Exception:
        return base  # non-Streamlit context

# ---------- Helpers ----------
def list_known(ohlc_dir: Optional[str] = None) -> List[str]:
    """List slugs of all known OHLC files (for debugging or dropdowns)."""
    base_dir = ohlc_dir or str(default_ohlc_dir())
    return sorted(set(_build_index(base_dir).keys()))
