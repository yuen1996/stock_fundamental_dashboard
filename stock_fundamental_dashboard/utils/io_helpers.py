# stock_fundamental_dashboard/utils/io_helpers.py
from __future__ import annotations
import os
import pandas as pd
from typing import Optional
import datetime as _dt

# after: import os, import pandas as pd
def _has_value(x) -> bool:
    """True if x is a real scalar value (not None/''/NA)."""
    try:
        return (x is not None) and (x != "") and (not pd.isna(x))
    except Exception:
        return (x is not None) and (x != "")

# ===== Debug/Error surfacing ==================================================
LAST_ERROR: Optional[str] = None
def _set_err(e: Exception, where: str = "") -> None:
    global LAST_ERROR
    where_txt = f" at {where}" if where else ""
    LAST_ERROR = f"{type(e).__name__}{where_txt}: {e}"
    # still print for server logs
    print(f"[io_helpers] ERROR{where_txt}: {e}")

# ===== Package paths ==========================================================
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR  = os.environ.get("SFD_DATA_DIR",  os.path.join(_PKG_ROOT, "data"))
DATA_FILE = os.environ.get("SFD_DATA_FILE", os.path.join(DATA_DIR, "fundamentals.csv"))

PRIMARY_COLUMNS = [
    "Name", "Industry", "IndustryBucket", "Year", "Quarter",
    "IsQuarter", "CurrentPrice"
]

def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=PRIMARY_COLUMNS)

def load_data() -> Optional[pd.DataFrame]:
    """Load the main dataset; return empty DataFrame if file is missing or unreadable."""
    try:
        if os.path.exists(DATA_FILE):
            try:
                df = pd.read_csv(DATA_FILE)
            except pd.errors.EmptyDataError:
                return _empty_df()
            df.columns = [str(c) for c in df.columns]
            for c in PRIMARY_COLUMNS:
                if c not in df.columns:
                    df[c] = pd.NA
            return df
        return _empty_df()
    except Exception as e:
        _set_err(e, "load_data")
        return _empty_df()

def save_data(df: pd.DataFrame) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        for c in PRIMARY_COLUMNS:
            if c not in df.columns:
                df[c] = pd.NA
        df.to_csv(DATA_FILE, index=False)
    except Exception as e:
        _set_err(e, "save_data")
        raise

# ==== Trade Queue paths & schema ==============================================
QUEUE_FILE = os.environ.get("SFD_QUEUE_FILE", os.path.join(DATA_DIR, "trade_queue.csv"))
LIVE_FILE  = os.environ.get("SFD_LIVE_FILE",  os.path.join(DATA_DIR, "trades_live.csv"))
AUDIT_FILE = os.environ.get("SFD_AUDIT_FILE", os.path.join(DATA_DIR, "queue_audit.csv"))

QUEUE_COLUMNS = [
    "Name", "Strategy", "Score", "CurrentPrice",
    "Entry", "Stop", "Take", "Shares", "RR",
    "TP1", "TP2", "TP3",
    "Timestamp", "Reasons",
]

def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

def _read_csv(path: str, columns: list[str]) -> pd.DataFrame:
    """Read CSV with lenient schema (add any missing columns)."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)
    except Exception as e:
        _set_err(e, f"_read_csv({path})")
        return pd.DataFrame(columns=columns)
    for c in columns:
        if c not in df.columns:
            df[c] = pd.NA
    return df[columns]

def _atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    """Write CSV atomically; fall back to a tolerant path if needed."""
    _ensure_data_dir()
    tmp = f"{path}.tmp"
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
        return
    except Exception as e1:
        # Fallback: coerce to object/string-friendly, keep order
        try:
            df2 = df.copy()
            # Replace NA/NaN with empty string so older pandas won't choke on Int64/NA
            df2 = df2.where(pd.notna(df2), "")
            for c in df2.columns:
                # Avoid pandas nullable integer quirks by using object dtype
                df2[c] = df2[c].astype("object")
            df2.to_csv(tmp, index=False)
            os.replace(tmp, path)
            return
        except Exception:
            # Clean up temp file if present and surface the first error
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            _set_err(e1, f"_atomic_write_csv({path})")
            raise

def load_trade_queue() -> pd.DataFrame:
    return _read_csv(QUEUE_FILE, QUEUE_COLUMNS)

def push_trade_candidate(
    name: str,
    strategy: str,
    score: float = 0.0,
    current_price: Optional[float] = None,
    reasons: str = "",
    entry_price: Optional[float] = None,
) -> bool:
    """Append a candidate idea into the Trade Queue. Returns True on success. (Audited: UPSERT)"""
    try:
        q = load_trade_queue()
        row = {
            "Name":         str(name),
            "Strategy":     str(strategy),
            "Score":        float(score) if score is not None else 0.0,
            "CurrentPrice": (float(current_price) if _has_value(current_price) else ""),
            "Entry":        (float(entry_price) if _has_value(entry_price)
                             else (float(current_price) if _has_value(current_price) else "")),
            "Stop":         "",
            "Take":         "",
            "Shares":       "",
            "RR":           "",
            "Timestamp":    _dt.datetime.now().isoformat(timespec="seconds"),
            "Reasons":      reasons or "",
        }
        q = pd.concat([q, pd.DataFrame([row], columns=QUEUE_COLUMNS)], ignore_index=True)
        _atomic_write_csv(q, QUEUE_FILE)

        # --- Audit (queue ADD) ---
        try:
            append_queue_audit("UPSERT", row, audit_reason="push_trade_candidate")
        except Exception:
            pass

        return True
    except Exception as e:
        _set_err(e, "push_trade_candidate")
        return False

# --- Queue Audit Log ----------------------------------------------------------
# File: data/queue_audit.csv  (override via SFD_AUDIT_FILE env or debug_queue_paths())
_AUDIT_COLUMNS = [
    "Timestamp","Event","Name","Strategy","Score","CurrentPrice",
    "Entry","Stop","Take","Shares","RR","TP1","TP2","TP3","Reasons","AuditReason"
]

def _audit_path() -> str:
    # Single source of truth for the audit CSV location
    os.makedirs(DATA_DIR, exist_ok=True)
    return AUDIT_FILE

def load_queue_audit() -> Optional[pd.DataFrame]:
    path = _audit_path()
    if not os.path.exists(path):
        return pd.DataFrame(columns=_AUDIT_COLUMNS)
    try:
        df = pd.read_csv(path)
        # ensure columns exist
        for c in _AUDIT_COLUMNS:
            if c not in df.columns:
                df[c] = None
        return df
    except Exception:
        return pd.DataFrame(columns=_AUDIT_COLUMNS)

def save_queue_audit(df: pd.DataFrame) -> None:
    path = _audit_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # keep only known columns + preserve order
    cols = [c for c in _AUDIT_COLUMNS if c in df.columns]
    out = df[cols].copy()
    tmp = f"{path}.tmp"
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)

def append_queue_audit(event: str, payload: dict | None = None, audit_reason: str = "") -> None:
    """
    Append a single audit row. Safe + non-blocking (best effort).
    Only for queue lifecycle events: UPSERT / UPDATE / DELETE / MARK_LIVE.
    """
    try:
        df = load_queue_audit()
        if df is None or df.empty:
            df = pd.DataFrame(columns=_AUDIT_COLUMNS)
        row = {c: None for c in _AUDIT_COLUMNS}
        row["Timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        row["Event"] = str(event)
        row["AuditReason"] = str(audit_reason or "")
        payload = payload or {}
        # map common fields if present
        for k in ["Name","Strategy","Score","CurrentPrice","Entry","Stop","Take","Shares","RR","TP1","TP2","TP3","Reasons"]:
            if k in payload:
                row[k] = payload[k]
        new = pd.DataFrame([row], columns=_AUDIT_COLUMNS)
        df = pd.concat([df, new], ignore_index=True)
        save_queue_audit(df)
    except Exception as e:
        # swallow errors; audit is non-critical
        _set_err(e, "append_queue_audit")
        pass
# ----------------------------------------------------------------------------- 

def update_trade_candidate(row_index: int, **fields) -> bool:
    """
    Update a queued trade row by index with provided fields, and audit an UPDATE.
    Fields may include: Entry, Stop, Take, Shares, RR, TP1, TP2, TP3, Reasons, etc.
    """
    try:
        q = load_trade_queue()
        if q is None or q.empty or row_index < 0 or row_index >= len(q):
            return False

        # ensure missing columns exist, then update
        for k, v in (fields or {}).items():
            if k not in q.columns:
                q[k] = pd.NA
            q.at[row_index, k] = v

        _atomic_write_csv(q, QUEUE_FILE)

        # audit with the latest row (after update)
        try:
            append_queue_audit("UPDATE", q.iloc[row_index].to_dict(), audit_reason="update_trade_candidate")
        except Exception:
            pass

        return True
    except Exception as e:
        _set_err(e, "update_trade_candidate")
        return False

def mark_live_row(row_id: int) -> bool:
    """
    Move a row (by current index) from Trade Queue to Live list,
    and write an audit record ('MARK_LIVE').
    """
    try:
        q = load_trade_queue()
        if q is None or q.empty or row_id < 0 or row_id >= len(q):
            return False

        row = q.iloc[row_id].to_dict()

        # --- Audit first (best-effort)
        try:
            append_queue_audit("MARK_LIVE", row, audit_reason="mark_live_row")
        except Exception:
            pass

        # --- Move to live/open list (include TP1/TP2/TP3 + OpenDate)
        row_df = q.iloc[[row_id]].copy()

        # ensure TP columns exist (older rows may not have them yet)
        for c in ["TP1", "TP2", "TP3"]:
            if c not in row_df.columns:
                row_df[c] = pd.NA

        # stamp OpenDate on the row we're moving to live
        open_ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if "OpenDate" not in row_df.columns:
            row_df["OpenDate"] = open_ts
        else:
            # fill if empty/NaN
            try:
                mask = row_df["OpenDate"].isna() | (row_df["OpenDate"] == "")
                row_df.loc[mask, "OpenDate"] = open_ts
            except Exception:
                row_df["OpenDate"] = open_ts

        # read/prepare live file and make sure it has needed columns
        live = pd.DataFrame(columns=list(row_df.columns))
        try:
            if os.path.exists(LIVE_FILE):
                live = pd.read_csv(LIVE_FILE)
                for c in row_df.columns:
                    if c not in live.columns:
                        live[c] = None
        except Exception:
            pass

        live = pd.concat([live, row_df], ignore_index=True)
        _atomic_write_csv(live, LIVE_FILE)

        # --- Remove from queue
        q = q.drop(q.index[row_id]).reset_index(drop=True)
        _atomic_write_csv(q, QUEUE_FILE)
        return True
    except Exception as e:
        _set_err(e, "mark_live_row")
        return False


def delete_trade_row(row_id: int, reason: str = "") -> bool:
    """Delete a row (by current index) from Trade Queue and record a DELETE audit."""
    try:
        q = load_trade_queue()
        if q is None or q.empty or row_id < 0 or row_id >= len(q):
            return False

        # capture payload BEFORE deletion for the audit row
        try:
            row_payload = q.iloc[row_id].to_dict()
        except Exception:
            row_payload = {}

        # audit (best-effort, non-blocking)
        try:
            append_queue_audit("DELETE", row_payload, audit_reason=reason or "delete_trade_row")
        except Exception:
            pass

        # drop + save
        q = q.drop(q.index[row_id]).reset_index(drop=True)
        _atomic_write_csv(q, QUEUE_FILE)
        return True
    except Exception as e:
        _set_err(e, "delete_trade_row")
        return False

def delete_trade_rows(row_ids: list[int], reason: str = "delete_trade_rows") -> int:
    """
    Delete multiple rows by their current indices (as shown in Manage Queue),
    writing a DELETE audit for each. Returns the count deleted.
    """
    try:
        q = load_trade_queue()
        if q is None or q.empty:
            return 0

        # unique, in descending order so indices remain valid while dropping
        ids = sorted({int(i) for i in row_ids if isinstance(i, (int,))}, reverse=True)

        deleted = 0
        for rid in ids:
            if rid < 0 or rid >= len(q):
                continue
            try:
                payload = q.iloc[rid].to_dict()
            except Exception:
                payload = {}
            try:
                append_queue_audit("DELETE", payload, audit_reason=reason)
            except Exception:
                pass
            q = q.drop(q.index[rid]).reset_index(drop=True)
            deleted += 1

        if deleted:
            _atomic_write_csv(q, QUEUE_FILE)
        return deleted
    except Exception as e:
        _set_err(e, "delete_trade_rows")
        return 0

def push_trade_candidates(rows: list[dict]) -> int:
    """
    Bulk push PASS candidates into the queue.
    Each dict expects keys: name, strategy, score, current_price, entry_price, reasons.
    Returns the number successfully pushed. (push_trade_candidate already audits each)
    """
    ok = 0
    for r in (rows or []):
        if push_trade_candidate(
            name=str(r.get("name", "")),
            strategy=str(r.get("strategy", "")),
            score=r.get("score", 0.0),
            current_price=r.get("current_price", None),
            entry_price=r.get("entry_price", None),
            reasons=r.get("reasons", ""),
        ):
            ok += 1
    return ok

# Handy for UI diagnostics
def debug_queue_paths() -> dict:
    return {
        "DATA_DIR": DATA_DIR,
        "DATA_FILE": DATA_FILE,
        "QUEUE_FILE": QUEUE_FILE,
        "LIVE_FILE": LIVE_FILE,
        "AUDIT_FILE": AUDIT_FILE,
    }


# ==================== OPEN / CLOSED TRADES (independent, no audit) ====================

# Reuse DATA_DIR, LIVE_FILE and helpers from above.
CLOSED_FILE = os.environ.get("SFD_CLOSED_FILE", os.path.join(DATA_DIR, "trades_closed.csv"))
FRIEND_LIVE_FILE = os.environ.get("SFD_FRIEND_LIVE_FILE", os.path.join(DATA_DIR, "trades_friend_live.csv"))
FRIEND_CLOSED_FILE = os.environ.get("SFD_FRIEND_CLOSED_FILE", os.path.join(DATA_DIR, "trades_friend_closed.csv"))


# Schema for currently-open trades (what 7_Ongoing_Trades.py shows)
OPEN_TRADES_COLUMNS = [
    "Name", "Strategy",
    "Entry", "Stop", "Take", "Shares", "RR", "TP1", "TP2", "TP3",
    "OpenDate",
    "Reasons",
]

# Schema for closed trades archive (independent from queue audit)
CLOSED_TRADES_COLUMNS = [
    "Name", "Strategy",
    "Entry", "Stop", "Take", "Shares", "RR_Init", "TP1", "TP2", "TP3",
    "OpenDate", "CloseDate", "ClosePrice",
    "HoldingDays",
    "PnL", "ReturnPct", "RMultiple",
    "CloseReason", "Reasons",
]

def _ensure_open_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        df = pd.DataFrame(columns=OPEN_TRADES_COLUMNS)
    for c in OPEN_TRADES_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    # keep canonical order first, then any extras (for back-compat)
    known = [c for c in OPEN_TRADES_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in OPEN_TRADES_COLUMNS]
    return df[known + extra]

def _ensure_closed_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        df = pd.DataFrame(columns=CLOSED_TRADES_COLUMNS)
    for c in CLOSED_TRADES_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    known = [c for c in CLOSED_TRADES_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in CLOSED_TRADES_COLUMNS]
    return df[known + extra]

def load_open_trades() -> pd.DataFrame:
    """Read independent list of live positions from LIVE_FILE (no audit involved)."""
    try:
        if not os.path.exists(LIVE_FILE):
            return pd.DataFrame(columns=OPEN_TRADES_COLUMNS)
        df = pd.read_csv(LIVE_FILE)
    except Exception:
        df = pd.DataFrame(columns=OPEN_TRADES_COLUMNS)
    return _ensure_open_schema(df)

def save_open_trades(df: pd.DataFrame) -> None:
    df = _ensure_open_schema(df)
    os.makedirs(os.path.dirname(LIVE_FILE), exist_ok=True)
    _atomic_write_csv(df, LIVE_FILE)

def load_closed_trades() -> pd.DataFrame:
    try:
        if not os.path.exists(CLOSED_FILE):
            return pd.DataFrame(columns=CLOSED_TRADES_COLUMNS)
        df = pd.read_csv(CLOSED_FILE)
    except Exception:
        df = pd.DataFrame(columns=CLOSED_TRADES_COLUMNS)
    return _ensure_closed_schema(df)

def save_closed_trades(df: pd.DataFrame) -> None:
    df = _ensure_closed_schema(df)
    os.makedirs(os.path.dirname(CLOSED_FILE), exist_ok=True)
    _atomic_write_csv(df, CLOSED_FILE)

def close_open_trade_row(rowid: int, close_price: float, close_reason: str) -> bool:
    """
    Close ONE open-trades row by its current CSV index (RowId), and move it to CLOSED_FILE.
    This is fully independent of the Queue Audit Log (no writes there).
    """
    try:
        open_df = load_open_trades()
        if rowid not in open_df.index:
            return False

        row = open_df.loc[rowid].to_dict()

        # Coerce numbers
        entry  = pd.to_numeric(row.get("Entry"),  errors="coerce")
        stop   = pd.to_numeric(row.get("Stop"),   errors="coerce")
        shares = pd.to_numeric(row.get("Shares"), errors="coerce")
        shares = 0.0 if pd.isna(shares) else float(shares)

        # Derived results
        try:
            pnl = (float(close_price) - float(entry)) * shares if pd.notna(entry) else None
            ret_pct = ((float(close_price) / float(entry)) - 1.0) * 100.0 if pd.notna(entry) else None
            r_mult = None
            if pd.notna(entry) and pd.notna(stop) and float(entry) > float(stop):
                r_mult = (float(close_price) - float(entry)) / (float(entry) - float(stop))
        except Exception:
            pnl = ret_pct = r_mult = None

        # Holding days
        try:
            od = pd.to_datetime(row.get("OpenDate"), errors="coerce")
            holding_days = (_dt.datetime.now() - od).days if pd.notna(od) else pd.NA
        except Exception:
            holding_days = pd.NA

        closed_row = {
            "Name":        row.get("Name"),
            "Strategy":    row.get("Strategy"),
            "Entry":       row.get("Entry"),
            "Stop":        row.get("Stop"),
            "Take":        row.get("Take"),
            "Shares":      row.get("Shares"),
            "RR_Init":     row.get("RR"),
            "TP1":         row.get("TP1"),
            "TP2":         row.get("TP2"),
            "TP3":         row.get("TP3"),
            "OpenDate":    row.get("OpenDate"),
            "CloseDate":   _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ClosePrice":  float(close_price),
            "HoldingDays": holding_days,
            "PnL":         pnl,
            "ReturnPct":   ret_pct,
            "RMultiple":   r_mult,
            "CloseReason": close_reason,
            "Reasons":     row.get("Reasons"),
        }

        # Append to closed, remove from open
        closed_df = load_closed_trades()
        closed_df = pd.concat([closed_df, pd.DataFrame([closed_row])], ignore_index=True)
        save_closed_trades(closed_df)

        open_df = open_df.drop(index=rowid).reset_index(drop=True)
        save_open_trades(open_df)

        # NOTE: no append_queue_audit(...) here on purpose (independent).
        return True
    except Exception as e:
        _set_err(e, "close_open_trade_row")
        return False
# ================== END OPEN / CLOSED TRADES (independent) ==================






# ================== FRIEND OPEN / CLOSED TRADES (separate) ===================
def _touch_data_version() -> None:
    """Touch .data.version to invalidate Streamlit caches (best-effort)."""
    try:
        vfile = os.path.join(_PKG_ROOT, ".data.version")
        os.makedirs(os.path.dirname(vfile), exist_ok=True)
        with open(vfile, "a", encoding="utf-8"):
            pass
        os.utime(vfile, None)
    except Exception:
        pass

def load_friend_open_trades() -> pd.DataFrame:
    """Load friend open trades (live positions). Returns empty DataFrame if missing."""
    try:
        if os.path.exists(FRIEND_LIVE_FILE):
            df = pd.read_csv(FRIEND_LIVE_FILE)
            return df if df is not None else pd.DataFrame()
        return pd.DataFrame()
    except Exception as e:
        _set_err(e, "load_friend_open_trades")
        return pd.DataFrame()

def save_friend_open_trades(df: pd.DataFrame) -> None:
    try:
        os.makedirs(os.path.dirname(FRIEND_LIVE_FILE), exist_ok=True)
        _atomic_write_csv(df, FRIEND_LIVE_FILE)
        _touch_data_version()
    except Exception as e:
        _set_err(e, "save_friend_open_trades")

def load_friend_closed_trades() -> pd.DataFrame:
    """Load friend closed trades."""
    try:
        if os.path.exists(FRIEND_CLOSED_FILE):
            df = pd.read_csv(FRIEND_CLOSED_FILE)
            return df if df is not None else pd.DataFrame()
        return pd.DataFrame()
    except Exception as e:
        _set_err(e, "load_friend_closed_trades")
        return pd.DataFrame()

def save_friend_closed_trades(df: pd.DataFrame) -> None:
    try:
        os.makedirs(os.path.dirname(FRIEND_CLOSED_FILE), exist_ok=True)
        _atomic_write_csv(df, FRIEND_CLOSED_FILE)
        _touch_data_version()
    except Exception as e:
        _set_err(e, "save_friend_closed_trades")

def update_friend_open_trades_rows(updates: pd.DataFrame) -> int:
    """
    Batch update friend open trades by RowId (CSV index).
    Expected columns in updates: RowId, FriendName (optional), Shares (optional)
    """
    try:
        if updates is None or updates.empty:
            return 0
        df = load_friend_open_trades()
        if df.empty:
            return 0

        # Ensure columns exist
        if "FriendName" not in df.columns:
            df["FriendName"] = ""
        if "Shares" not in df.columns:
            df["Shares"] = pd.NA

        cnt = 0
        for _, u in updates.iterrows():
            rid = u.get("RowId")
            if pd.isna(rid):
                continue
            rid = int(rid)
            if rid < 0 or rid >= len(df):
                continue
            if "FriendName" in updates.columns and pd.notna(u.get("FriendName")):
                df.at[rid, "FriendName"] = str(u.get("FriendName") or "").strip()
            if "Shares" in updates.columns and pd.notna(u.get("Shares")):
                try:
                    df.at[rid, "Shares"] = int(float(u.get("Shares")))
                except Exception:
                    pass
            cnt += 1

        save_friend_open_trades(df)
        return cnt
    except Exception as e:
        _set_err(e, "update_friend_open_trades_rows")
        return 0

def mark_friend_live_row(row_id: int, friend_name: str) -> bool:
    """
    Move a row (by current index) from Trade Queue to FRIEND live list,
    stamp OpenDate, and write an audit record.
    """
    try:
        q = load_trade_queue()
        if q is None or q.empty or row_id < 0 or row_id >= len(q):
            return False

        row_df = q.iloc[[row_id]].copy()

        # ensure TP columns exist
        for c in ["TP1", "TP2", "TP3"]:
            if c not in row_df.columns:
                row_df[c] = pd.NA

        # stamp OpenDate
        open_ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if "OpenDate" not in row_df.columns:
            row_df["OpenDate"] = open_ts
        else:
            try:
                mask = row_df["OpenDate"].isna() | (row_df["OpenDate"] == "")
                row_df.loc[mask, "OpenDate"] = open_ts
            except Exception:
                row_df["OpenDate"] = open_ts

        # set friend name
        if "FriendName" not in row_df.columns:
            row_df["FriendName"] = str(friend_name or "").strip()
        else:
            row_df["FriendName"] = str(friend_name or "").strip()

        # audit best-effort
        try:
            append_queue_audit("MARK_FRIEND_LIVE", row_df.iloc[0].to_dict(), audit_reason="mark_friend_live_row")
        except Exception:
            pass

        live = pd.DataFrame(columns=list(row_df.columns))
        try:
            if os.path.exists(FRIEND_LIVE_FILE):
                live = pd.read_csv(FRIEND_LIVE_FILE)
                for c in row_df.columns:
                    if c not in live.columns:
                        live[c] = None
        except Exception:
            pass

        live = pd.concat([live, row_df], ignore_index=True)
        save_friend_open_trades(live)

        # remove from queue
        q = q.drop(q.index[row_id]).reset_index(drop=True)
        _atomic_write_csv(q, QUEUE_FILE)
        _touch_data_version()
        return True
    except Exception as e:
        _set_err(e, "mark_friend_live_row")
        return False

def close_friend_open_trade_row(rowid: int, close_price: float, close_reason: str) -> bool:
    """
    Close ONE friend open-trades row by its CSV index (RowId), move to FRIEND_CLOSED_FILE.
    """
    try:
        open_df = load_friend_open_trades()
        if open_df.empty or rowid not in open_df.index:
            return False

        row = open_df.loc[rowid].to_dict()

        entry  = pd.to_numeric(row.get("Entry"),  errors="coerce")
        stop   = pd.to_numeric(row.get("Stop"),   errors="coerce")
        shares = pd.to_numeric(row.get("Shares"), errors="coerce")
        shares = 0.0 if pd.isna(shares) else float(shares)

        try:
            pnl = (float(close_price) - float(entry)) * shares if pd.notna(entry) else None
            ret_pct = ((float(close_price) / float(entry)) - 1.0) * 100.0 if pd.notna(entry) else None
            r_mult = None
            if pd.notna(entry) and pd.notna(stop) and float(entry) > float(stop):
                r_mult = (float(close_price) - float(entry)) / (float(entry) - float(stop))
        except Exception:
            pnl = ret_pct = r_mult = None

        try:
            od = pd.to_datetime(row.get("OpenDate"), errors="coerce")
            holding_days = (_dt.datetime.now() - od).days if pd.notna(od) else pd.NA
        except Exception:
            holding_days = pd.NA

        closed_row = {
            "FriendName":  row.get("FriendName"),
            "Name":        row.get("Name"),
            "Strategy":    row.get("Strategy"),
            "Entry":       row.get("Entry"),
            "Stop":        row.get("Stop"),
            "Take":        row.get("Take"),
            "Shares":      row.get("Shares"),
            "RR_Init":     row.get("RR"),
            "TP1":         row.get("TP1"),
            "TP2":         row.get("TP2"),
            "TP3":         row.get("TP3"),
            "OpenDate":    row.get("OpenDate"),
            "CloseDate":   _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ClosePrice":  float(close_price),
            "HoldingDays": holding_days,
            "PnL":         pnl,
            "ReturnPct":   ret_pct,
            "RMultiple":   r_mult,
            "CloseReason": close_reason,
            "Reasons":     row.get("Reasons"),
        }

        closed_df = load_friend_closed_trades()
        closed_df = pd.concat([closed_df, pd.DataFrame([closed_row])], ignore_index=True)
        save_friend_closed_trades(closed_df)

        open_df = open_df.drop(index=rowid).reset_index(drop=True)
        save_friend_open_trades(open_df)
        return True
    except Exception as e:
        _set_err(e, "close_friend_open_trade_row")
        return False
# ================== END FRIEND OPEN / CLOSED TRADES ==========================

