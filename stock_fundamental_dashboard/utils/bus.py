# utils/bus.py
"""Tiny cache/event-bus helpers.

Goal:
- One source of truth for the app data directory (respects SFD_DATA_DIR)
- Consistent cache busting across pages:
  - bump(topic) touches both .bus.<topic>.version AND the legacy .data.version
  - etag(topic) considers BOTH files so any writer (legacy or bus) invalidates caches
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional


def _norm_dir(p: str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(p)))


def data_dir() -> Path:
    """Return the single source of truth for data dir.

    Priority:
      1) SFD_DATA_DIR env
      2) nearest '<parent>/data' next to package (compatible with old layouts)
    """
    env = (os.getenv("SFD_DATA_DIR") or "").strip()
    if env:
        d = _norm_dir(env)
    else:
        # If this file is .../utils/bus.py -> default is .../data
        d = Path(__file__).resolve().parent.parent / "data"
        d = _norm_dir(str(d))
    d.mkdir(parents=True, exist_ok=True)
    return d


# Keep the historical names for backwards compatibility
DATA_DIR = str(data_dir())
BASE = str(Path(DATA_DIR).parent)
DATA = DATA_DIR


def _verfile(topic: str) -> Path:
    return _norm_dir(DATA_DIR) / f".bus.{topic}.version"


def _legacy_file() -> Path:
    # used by some older pages: <DATA_DIR>/.data.version
    return _norm_dir(DATA_DIR) / ".data.version"


def _safe_mtime_ns(p: Path) -> int:
    try:
        return int(p.stat().st_mtime_ns)
    except Exception:
        return 0


def _touch(p: Path) -> int:
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.touch(exist_ok=True)
        os.utime(p, None)
    except Exception:
        # best-effort; still return current mtime if any
        pass
    return _safe_mtime_ns(p)


def bump(topic: str) -> int:
    """Invalidate caches for a given topic.

    Also touches the legacy global version file so older cache keys update too.
    """
    ts_topic = _touch(_verfile(topic))
    _touch(_legacy_file())
    return ts_topic


def etag(topic: str) -> int:
    """Return a cache-busting int.

    We take max(topic-version, legacy-global-version) so:
    - new code that bumps via bus invalidates old caches
    - old code that touches .data.version invalidates bus caches
    """
    return max(_safe_mtime_ns(_verfile(topic)), _safe_mtime_ns(_legacy_file()))


# Momentum snapshot (fast local handoff, no Snowflake wait)
MOMS_SNAPSHOT = str(_norm_dir(DATA_DIR) / "momentum_snapshot.json")


def write_momentum_snapshot(d: dict) -> int:
    tmp = MOMS_SNAPSHOT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    os.replace(tmp, MOMS_SNAPSHOT)  # atomic
    return bump("momentum")


def read_momentum_snapshot() -> dict:
    try:
        with open(MOMS_SNAPSHOT, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
