# utils/name_link.py
from __future__ import annotations

import os
import re
import unicodedata
from functools import lru_cache

# Words we commonly want to ignore in company names (extend as needed)
_DROP = {
    "berhad", "bhd", "plc", "limited", "ltd", "inc", "inc.", "corp", "corporation", "company", "co",
    "holdings", "holding", "group", "resources", "technology", "technologies", "tech", "malaysia",
    "public", "tbk", "nv", "sa", "ag", "oyj", "ab", "as", "sp", "s.p.a", "spa", "the",
}


def _norm(s: str) -> str:
    """Canonicalize to a loose, comparable form."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("_", " ")
    s = re.sub(r"[^\w\s]", " ", s, flags=re.I)  # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()
    words = [w for w in s.split() if w and w.lower() not in _DROP]
    return " ".join(words).lower()


def _tokens(s: str) -> set[str]:
    return {w for w in re.split(r"\s+", _norm(s)) if len(w) >= 2}


def _score(a: str, b: str, ticker_hint: str | None = None) -> float:
    """
    Heuristic score between normalized names + optional ticker boost.

    Extras:
    - Subset/one-extra-word boost: "kawan" vs "kawan food" is a strong match.
    - Compact-equality boost stays (e.g., "rhonema" == "rhone ma").
    """
    an, bn = _norm(a), _norm(b)
    if not an or not bn:
        return 0.0

    # base similarity
    if an == bn:
        base = 1.0
    elif an.startswith(bn) or bn.startswith(an):
        base = 0.92
    else:
        ta, tb = _tokens(a), _tokens(b)
        if not ta or not tb:
            return 0.0
        jacc = len(ta & tb) / max(1, len(ta | tb))
        base = jacc  # 0..1

        # subset / "one-extra-word" boost
        if (ta <= tb or tb <= ta) and len(ta & tb) >= 1:
            base = max(base, 0.66)

    # Compact equality ("rhonema" == "rhone ma") gets a strong boost
    ac = re.sub(r"[^0-9a-z]+", "", an)
    bc = re.sub(r"[^0-9a-z]+", "", bn)
    if ac and bc and ac == bc:
        base = max(base, 0.94)

    # Ticker/code hint helps disambiguate (e.g., 0272, NATGATE)
    bonus = 0.0
    if ticker_hint:
        t = str(ticker_hint).lower().strip()
        if t:
            bl = b.lower()
            if t in bl:
                bonus += 0.15
            if t.replace(".", "") in bl.replace(".", ""):
                bonus += 0.10

    return min(1.0, base + bonus)


def _safe_root(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _dir_mtime_ns(path: str) -> int:
    """Directory mtime in ns; used as a cache-busting signature."""
    try:
        return int(os.stat(path).st_mtime_ns)
    except Exception:
        return 0


@lru_cache(maxsize=8)
def _index_dir(ohlc_dir: str, _sig: int) -> list[tuple[str, str]]:
    """
    Return list of (root_without_ext, full_path) for .csv files under ohlc_dir.

    Cached for speed, but invalidated automatically when the directory mtime changes
    thanks to the `_sig` parameter (pass _dir_mtime_ns(ohlc_dir) when calling).
    """
    out: list[tuple[str, str]] = []
    if not os.path.isdir(ohlc_dir):
        return out
    for f in os.listdir(ohlc_dir):
        if f.lower().endswith(".csv"):
            full = os.path.join(ohlc_dir, f)
            out.append((_safe_root(full), full))
    return out


# --- Canonical OHLC directory resolver (shared by all pages) ---
def resolve_ohlc_dir() -> str:
    """
    Decide where the OHLC CSVs live.
    Order:
      1) Env override SFD_OHLC_DIR
      2) <repo>/utils/../data/ohlc
      3) <repo>/utils/../stock_fundamental_dashboard/data/ohlc
      4) <cwd>/data/ohlc
      5) <cwd>/stock_fundamental_dashboard/data/ohlc
    If none exist, create #2.
    """
    here = os.path.dirname(__file__)
    env = os.environ.get("SFD_OHLC_DIR")

    candidates = [
        env,
        os.path.abspath(os.path.join(here, "..", "data", "ohlc")),
        os.path.abspath(os.path.join(here, "..", "stock_fundamental_dashboard", "data", "ohlc")),
        os.path.abspath(os.path.join(os.getcwd(), "data", "ohlc")),
        os.path.abspath(os.path.join(os.getcwd(), "stock_fundamental_dashboard", "data", "ohlc")),
    ]

    for d in [c for c in candidates if c]:
        if os.path.isdir(d):
            return d

    # default: create the standard spot next to utils/
    d = os.path.abspath(os.path.join(here, "..", "data", "ohlc"))
    os.makedirs(d, exist_ok=True)
    return d


def find_ohlc_path(stock_name: str, *, ohlc_dir: str, ticker: str | None = None) -> str | None:
    """
    Best-effort resolver: try exact root, then relaxed scoring.
    Returns full path or None if nothing reasonably close (>0.58) is found.
    Also supports single-token containment fallback (e.g., 'apollo' -> Apollo.csv).
    """
    files = _index_dir(ohlc_dir, _dir_mtime_ns(ohlc_dir))  # <-- auto-refreshing index
    if not files:
        return None

    # 1) exact root match on normalized name (tolerant to suffixes)
    want_root = re.sub(r"\s+", "_", _norm(stock_name)).strip("_")
    for root, full in files:
        if re.sub(r"\s+", "_", _norm(root)).strip("_") == want_root:
            return full

    # 2) score all, pick best above threshold
    best_score, best_path = 0.0, None
    for root, full in files:
        sc = _score(stock_name, root, ticker_hint=ticker)
        if sc > best_score:
            best_score, best_path = sc, full
    if best_score >= 0.58:
        return best_path

    # 3) containment fallback — if there’s exactly one file sharing the first token, use it
    toks = list(_tokens(stock_name))
    first = toks[0] if toks else ""
    if first:
        candidates = [full for root, full in files if first in _tokens(root)]
        if len(candidates) == 1:
            return candidates[0]

    return None
