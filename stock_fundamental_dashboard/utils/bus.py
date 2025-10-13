# utils/bus.py
import os, json

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)

def _verfile(topic: str) -> str:
    return os.path.join(DATA, f".bus.{topic}.version")

def _safe_mtime_ns(p: str) -> int:
    try: return int(os.stat(p).st_mtime_ns)
    except Exception: return 0

def bump(topic: str) -> int:
    p = _verfile(topic)
    open(p, "a").close()
    os.utime(p, None)
    return _safe_mtime_ns(p)

def etag(topic: str) -> int:
    return _safe_mtime_ns(_verfile(topic))

# Momentum snapshot (fast local handoff, no Snowflake wait)
MOMS_SNAPSHOT = os.path.join(DATA, "momentum_snapshot.json")

def write_momentum_snapshot(d: dict) -> int:
    tmp = MOMS_SNAPSHOT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    os.replace(tmp, MOMS_SNAPSHOT)   # atomic
    return bump("momentum")

def read_momentum_snapshot() -> dict:
    try:
        with open(MOMS_SNAPSHOT, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
