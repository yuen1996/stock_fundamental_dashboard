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


def _filename_variants(stem: str) -> List[str]:
    """
    For a filename stem like 'ZHULIAN_CORPORATION_BERHAD', generate
    multiple shapes, all slugified.
    """
    s = str(stem or "")
    variants = {
        s,
        s.lower(),
        s.replace("_", " "),
        s.replace("_", ""),
        s.replace("-", " "),
        s.replace("-", ""),
        s.replace(" ", "_"),
    }
    return sorted({slugify(v) for v in variants if v})


def _strip_suffixes_slug(slug: str) -> str:
    """
    Strip trailing corporate suffixes from a slug.

    e.g.
      'zhulian_corporation_berhad' -> 'zhulian'
      'malayan_banking_berhad'     -> 'malayan_banking'
    """
    suffixes = {
        "berhad", "bhd", "bhd.", "corp", "corp.", "corporation",
        "ltd", "ltd.", "limited",
    }
    parts = [p for p in slug.split("_") if p]
    while parts and parts[-1] in suffixes:
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

            # base form (remove corporate suffixes)
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
      - 'exact'     : exact slug match
      - 'compact'   : match after stripping non-alnum
      - 'base'      : match after stripping corporate suffixes
      - 'code'      : match via ticker-like code
      - 'fuzzy:<k>' : fuzzy match on index key
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

    # 4) Code via slug, just in case
    if code:
        scode = slugify(code)
        if scode in index:
            return index[scode], "code"

    # 5) Fuzzy against all keys
    keys = list(index.keys())
    match = difflib.get_close_matches(key, keys, n=1, cutoff=0.6)
    if not match and base_key:
        match = difflib.get_close_matches(base_key, keys, n=1, cutoff=0.6)
    if not match and code:
        match = difflib.get_close_matches(code, keys, n=1, cutoff=0.6)

    if match:
        k = match[0]
        return index[k], f"fuzzy:{k}"

    return None, "not_found"


# ---------- Loading ----------

def load_ohlc(
    name_or_symbol: str,
    ohlc_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, Path, str]:
    """
    Load OHLC CSV by tolerant name.

    Returns:
        (df, path, how)

    Raises:
        FileNotFoundError if nothing can be resolved.
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
        (lower_cols[c] for c in ("date", "timestamp", "time", "datetime") if c in lower_cols),
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
