# utils/bus.py
"""A tiny cross-page cache-busting/event bus.

Why it exists:
  - Streamlit caches are per-page + per-process.
  - We need a cheap 'etag(topic)' that changes when underlying data changes,
    so every page can invalidate @st.cache_data consistently.

Design:
  - All bus state lives under SFD_DATA_DIR (or <project>/data by default).
  - Topics are just strings: 'ohlc', 'trades', 'fundamentals', 'queue', 'momentum', etc.
"""

from __future__ import annotations

import os
import json
from pathlib import Path

# Prefer the same DATA_DIR used by io_helpers.py
_BASE = Path(__file__).resolve().parent.parent  # .../utils/ -> project root
_DEFAULT_DATA_DIR = _BASE / "data"
DATA_DIR = Path(os.environ.get("SFD_DATA_DIR", str(_DEFAULT_DATA_DIR))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_data_dir() -> str:
    return str(DATA_DIR)

def _verfile(topic: str) -> Path:
    return DATA_DIR / f".bus.{topic}.version"

def _safe_mtime_ns(p: Path) -> int:
    try:
        return int(p.stat().st_mtime_ns)
    except Exception:
        return 0

def bump(topic: str) -> int:
    """Touch the topic version file and return its mtime_ns."""
    p = _verfile(topic)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch(exist_ok=True)
        os.utime(str(p), None)
    except Exception:
        pass
    return _safe_mtime_ns(p)

def etag(topic: str) -> int:
    """Get current etag for a topic (mtime_ns)."""
    return _safe_mtime_ns(_verfile(topic))

# Momentum snapshot (fast local handoff, no Snowflake wait)
MOMS_SNAPSHOT = DATA_DIR / "momentum_snapshot.json"

def write_momentum_snapshot(d: dict) -> int:
    tmp = Path(str(MOMS_SNAPSHOT) + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    tmp.replace(MOMS_SNAPSHOT)  # atomic
    return bump("momentum")

def read_momentum_snapshot() -> dict:
    try:
        with MOMS_SNAPSHOT.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
