import os
from datetime import datetime
import pandas as pd
import numpy as np

# ==============================================================
#                      STOCKS DATA (MAIN)
# ==============================================================

DATA_PATH = "data/stocks.csv"

# Unified schema (both Annual and Quarterly).
# Keep this list as the single source of truth to avoid KeyErrors.
ALL_COLUMNS = [
    "Name", "Industry", "Year", "IsQuarter", "Quarter",
    # Annual (income)
    "NetProfit", "GrossProfit", "Revenue", "CostOfSales", "FinanceCosts", "AdminExpenses", "SellDistExpenses",
    # Annual (balance / other)
    "NumShares", "CurrentAsset", "OtherReceivables", "TradeReceivables", "BiologicalAssets", "Inventories", "PrepaidExpenses",
    "IntangibleAsset", "CurrentLiability", "TotalAsset", "TotalLiability", "ShareholderEquity", "Reserves",
    "Dividend", "SharePrice",
    # Independent per-stock current price (used by ratios/TTM)
    "CurrentPrice",
    # Quarterly (prefix Q_)
    "Q_NetProfit", "Q_GrossProfit", "Q_Revenue", "Q_CostOfSales", "Q_FinanceCosts", "Q_AdminExpenses", "Q_SellDistExpenses",
    "Q_NumShares", "Q_CurrentAsset", "Q_OtherReceivables", "Q_TradeReceivables", "Q_BiologicalAssets", "Q_Inventories",
    "Q_PrepaidExpenses", "Q_IntangibleAsset", "Q_CurrentLiability", "Q_TotalAsset", "Q_TotalLiability",
    "Q_ShareholderEquity", "Q_Reserves", "Q_SharePrice", "Q_EndQuarterPrice",
    # Per-row timestamp
    "LastModified",
]


def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ALL_COLUMNS exist; do not drop unknown columns (backward compatible)."""
    if df is None or df.empty:
        df = pd.DataFrame(columns=ALL_COLUMNS)
    for col in ALL_COLUMNS:
        if col not in df.columns:
            if col == "IsQuarter":
                df[col] = False
            elif col == "Quarter":
                df[col] = pd.NA
            else:
                df[col] = pd.NA
    # Make sure dtypes are friendly for downstream usage
    if "IsQuarter" in df.columns:
        df["IsQuarter"] = df["IsQuarter"].fillna(False).astype(bool)
    return df


def load_data() -> pd.DataFrame:
    """Load main stocks file with guaranteed schema."""
    if not os.path.exists(DATA_PATH):
        return ensure_schema(pd.DataFrame(columns=ALL_COLUMNS))
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception:
        df = pd.DataFrame(columns=ALL_COLUMNS)
    return ensure_schema(df)


def save_data(df: pd.DataFrame) -> None:
    """
    Save main stocks data with `LastModified` preservation:
    - If a row (Name, IsQuarter, Year, Quarter) is unchanged vs on-disk row, keep old LastModified.
    - Else stamp now().
    """
    df = ensure_schema(df)

    # Load old for timestamp comparison
    try:
        old = load_data()
    except Exception:
        old = pd.DataFrame(columns=ALL_COLUMNS)
    old = ensure_schema(old)

    key_cols = ["Name", "IsQuarter", "Year", "Quarter"]
    # Ensure key columns exist (they do, from ensure_schema)
    old_keyed = old.set_index(key_cols, drop=False) if not old.empty else old

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_rows = []

    for _, new_row in df.iterrows():
        key = (new_row["Name"], new_row["IsQuarter"], new_row["Year"], new_row["Quarter"])
        if not old_keyed.empty and key in old_keyed.index:
            old_row = old_keyed.loc[key]
            # Compare excluding LastModified
            nr = new_row.copy()
            orow = old_row.copy()
            if "LastModified" in nr:
                nr = nr.drop(labels="LastModified")
            if "LastModified" in orow:
                orow = orow.drop(labels="LastModified")
            # pandas Series.equals can be strict with NaNs; make both strings where needed
            equal = nr.fillna("__NA__").equals(orow.fillna("__NA__"))
            new_row["LastModified"] = old_row.get("LastModified", now_str) if equal else now_str
        else:
            new_row["LastModified"] = now_str
        out_rows.append(new_row)

    out_df = pd.DataFrame(out_rows)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    out_df.to_csv(DATA_PATH, index=False)


# ==============================================================
#                      WATCHLIST HELPERS
# ==============================================================

WATCHLIST_PATH = "data/watchlist.csv"
WATCHLIST_COLUMNS = ["Name", "TargetPrice", "Notes", "Active"]


def _ensure_watchlist_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        df = pd.DataFrame(columns=WATCHLIST_COLUMNS)
    for col in WATCHLIST_COLUMNS:
        if col not in df.columns:
            if col == "Active":
                df[col] = True
            else:
                df[col] = pd.NA
    # Normalize types
    df["Active"] = df["Active"].fillna(True).astype(bool)
    if "TargetPrice" in df.columns:
        df["TargetPrice"] = pd.to_numeric(df["TargetPrice"], errors="coerce")
    if "Notes" in df.columns:
        df["Notes"] = df["Notes"].astype("string").fillna("")
    return df[WATCHLIST_COLUMNS]


def load_watchlist() -> pd.DataFrame:
    if not os.path.exists(WATCHLIST_PATH):
        return pd.DataFrame(columns=WATCHLIST_COLUMNS)
    try:
        df = pd.read_csv(WATCHLIST_PATH)
    except Exception:
        df = pd.DataFrame(columns=WATCHLIST_COLUMNS)
    return _ensure_watchlist_schema(df)


def save_watchlist(df: pd.DataFrame) -> None:
    df = _ensure_watchlist_schema(df)
    os.makedirs(os.path.dirname(WATCHLIST_PATH), exist_ok=True)
    df.to_csv(WATCHLIST_PATH, index=False)


# ==============================================================
#                      TRADE QUEUE HELPERS
# ==============================================================

TRADE_QUEUE_PATH = "data/trade_queue.csv"
# Backward-compatible superset schema (old CSVs will be auto-upgraded)
TRADE_QUEUE_COLUMNS = [
    "Name", "Strategy", "Score", "CurrentPrice",
    # Planning fields
    "Entry", "Stop", "Take", "Shares", "RR",
    "TP1", "TP2", "TP3",
    "Timestamp", "Reasons",
]


def _ensure_trade_queue_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the trade queue has all expected columns (add missing; keep extras)."""
    if df is None or df.empty:
        df = pd.DataFrame(columns=TRADE_QUEUE_COLUMNS)
    for c in TRADE_QUEUE_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    # keep canonical order, but preserve any unknown columns at the end
    known = [c for c in TRADE_QUEUE_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in TRADE_QUEUE_COLUMNS]
    return df[known + extra]


def load_trade_queue() -> pd.DataFrame:
    if not os.path.exists(TRADE_QUEUE_PATH):
        return pd.DataFrame(columns=TRADE_QUEUE_COLUMNS)
    try:
        df = pd.read_csv(TRADE_QUEUE_PATH)
    except Exception:
        df = pd.DataFrame(columns=TRADE_QUEUE_COLUMNS)
    return _ensure_trade_queue_schema(df)


def save_trade_queue(df: pd.DataFrame) -> None:
    df = _ensure_trade_queue_schema(df)
    os.makedirs(os.path.dirname(TRADE_QUEUE_PATH), exist_ok=True)
    df.to_csv(TRADE_QUEUE_PATH, index=False)


# ------------------ Queue Audit (append-only) ------------------

QUEUE_AUDIT_PATH = "data/queue_audit.csv"
QUEUE_AUDIT_COLUMNS = [
    "Timestamp", "Event",       # Event: UPSERT, DELETE
    "Name", "Strategy",
    "Score", "CurrentPrice",
    "Entry", "Stop", "Take", "Shares", "RR",
    "TP1", "TP2", "TP3",
    "Reasons",                  # free-text summary stored in queue row
    "AuditReason",              # chosen reason when deleting/updating
]


def _ensure_queue_audit_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        df = pd.DataFrame(columns=QUEUE_AUDIT_COLUMNS)
    for c in QUEUE_AUDIT_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[QUEUE_AUDIT_COLUMNS]


def load_queue_audit() -> pd.DataFrame:
    if not os.path.exists(QUEUE_AUDIT_PATH):
        return pd.DataFrame(columns=QUEUE_AUDIT_COLUMNS)
    try:
        df = pd.read_csv(QUEUE_AUDIT_PATH)
    except Exception:
        df = pd.DataFrame(columns=QUEUE_AUDIT_COLUMNS)
    return _ensure_queue_audit_schema(df)


def save_queue_audit(df: pd.DataFrame) -> None:
    df = _ensure_queue_audit_schema(df)
    os.makedirs(os.path.dirname(QUEUE_AUDIT_PATH), exist_ok=True)
    df.to_csv(QUEUE_AUDIT_PATH, index=False)


def append_queue_audit(event: str, row: dict, audit_reason: str | None = None) -> None:
    """Append an audit record; `row` can be a queue row dict or Series-like."""
    base = {}
    # read fields from dict-like; missing okay
    for k in ["Name", "Strategy", "Score", "CurrentPrice", "Entry", "Stop", "Take",
              "Shares", "RR", "TP1", "TP2", "TP3", "Reasons"]:
        base[k] = (row.get(k) if isinstance(row, dict) else getattr(row, k, None)) if row is not None else None

    rec = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Event": event,
        **base,
        "AuditReason": audit_reason,
    }
    log_df = load_queue_audit()
    log_df = pd.concat([log_df, pd.DataFrame([rec])], ignore_index=True)
    save_queue_audit(log_df)


# ------------------ Public queue ops (with audit) --------------

def push_trade_candidate(
        name: str,
        strategy: str,
        score: float,
        current_price,
        reasons: str = "",
        *,
        entry:  float | None = None,
        stop:   float | None = None,
        take:   float | None = None,
        shares: int   | None = None,
        rr:     float | None = None,
        tp1:    float | None = None,
        tp2:    float | None = None,
        tp3:    float | None = None,
) -> None:
    """
    Append a *new* idea to the Trade Queue.
    ► No more de-duping on (Name, Strategy) — every call is its own row.
    The extra planning fields (entry/stop/…) are optional; they’re saved
    only if the CSV already has those columns (back-compat safe).
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1) build the row entirely in python dict form
    row = {
        "Name":         name,
        "Strategy":     strategy,
        "Score":        score,
        "CurrentPrice": current_price,
        "Entry":        entry,
        "Stop":         stop,
        "Take":         take,
        "Shares":       shares,
        "RR":           rr,
        "TP1":          tp1,
        "TP2":          tp2,
        "TP3":          tp3,
        "Timestamp":    ts,
        "Reasons":      reasons,
    }

    # 2) load → append → save  (no deduping)
    q = load_trade_queue()
    q = pd.concat([q, pd.DataFrame([row])], ignore_index=True)
    save_trade_queue(q)

    # 3) audit log
    try:
        append_queue_audit("UPSERT", row, audit_reason="push_trade_candidate")
    except Exception:
        pass

# ------------------------------------------------------------------
# NEW helper – lets Risk-Reward Planner update an existing queue row
# ------------------------------------------------------------------
def update_trade_candidate(rowid: int, **updates) -> bool:
    """
    Overwrite selected columns for ONE queue row
    identified by its RowId (the CSV index).

    Returns
    -------
    True  – row was updated and saved  
    False – RowId not found (nothing changed)
    """
    q = load_trade_queue()
    if rowid not in q.index:
        return False

    # Add any new columns that aren't yet in the CSV
    for col in updates:
        if col not in q.columns:
            q[col] = pd.NA

    # Apply changes
    for col, val in updates.items():
        q.at[rowid, col] = val

    save_trade_queue(q)

    # —— audit log ——
    try:
        append_queue_audit("UPDATE", q.loc[rowid].to_dict(),
                           audit_reason="risk_reward_update")
    except Exception:
        pass
    return True
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Helpers that act on ONE queue row by its CSV index  (RowId)
# ------------------------------------------------------------------
def delete_trade_row(rowid: int, audit_reason: str) -> bool:
    """Delete exactly one queue row by index; audit the delete."""
    q = load_trade_queue()
    if rowid not in q.index:            # not found
        return False
    row = q.loc[rowid].to_dict()        # capture for audit
    q = q.drop(index=rowid)
    save_trade_queue(q)
    append_queue_audit("DELETE", row, audit_reason=audit_reason)
    return True


def mark_live_row(rowid: int) -> bool:
    """Move one queue row ➜ open_trades, preserving duplicates."""
    q = load_trade_queue()
    if rowid not in q.index:
        return False
    row = q.loc[rowid].to_dict()

    from datetime import datetime
    entry = row.get("Entry") if pd.notna(row.get("Entry")) else row.get("CurrentPrice")
    open_row = {
        "Name":     row.get("Name"),
        "Strategy": row.get("Strategy"),
        "Entry":    entry,
        "Stop":     row.get("Stop"),
        "Take":     row.get("Take"),
        "Shares":   row.get("Shares"),
        "RR":       row.get("RR"),
        "TP1":      row.get("TP1"),
        "TP2":      row.get("TP2"),
        "TP3":      row.get("TP3"),
        "OpenDate": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Reasons":  row.get("Reasons"),
    }
    open_df = load_open_trades()
    open_df = pd.concat([open_df, pd.DataFrame([open_row])], ignore_index=True)
    save_open_trades(open_df)

    q = q.drop(index=rowid)             # remove only this row
    save_trade_queue(q)
    append_queue_audit("MARK_LIVE", row, audit_reason="mark_live_row")
    return True
# ------------------------------------------------------------------


def delete_trade_candidate(name: str, strategy: str, audit_reason: str) -> bool:
    """
    Delete a queue row by (Name, Strategy). Logs a DELETE audit with `audit_reason`.
    Returns True if deleted, False if not found.
    """
    q = load_trade_queue()
    if q.empty:
        return False

    mask = q["Name"].astype(str).str.lower().eq(str(name).lower()) & \
           q["Strategy"].astype(str).str.lower().eq(str(strategy).lower())

    if not mask.any():
        return False

    # Capture the last matching row for audit before deleting
    row = q.loc[mask].iloc[-1].to_dict()

    # Remove and save
    q = q.loc[~mask]
    save_trade_queue(q)

    # Audit log (DELETE) with provided reason
    try:
        append_queue_audit("DELETE", row, audit_reason=audit_reason)
    except Exception:
        pass

    return True

# ─────────────── OPEN / CLOSED TRADES ───────────────
OPEN_TRADES_PATH    = "data/open_trades.csv"
OPEN_TRADES_COLUMNS = [
    "Name", "Strategy",
    "Entry", "Stop", "Take", "Shares", "RR", "TP1", "TP2", "TP3",
    "OpenDate",
    "Reasons",          # carry from queue
]

CLOSED_TRADES_PATH    = "data/closed_trades.csv"
CLOSED_TRADES_COLUMNS = [
    "Name", "Strategy",
    "Entry", "Stop", "Take", "Shares", "RR_Init", "TP1", "TP2", "TP3",
    "OpenDate", "CloseDate", "ClosePrice",
    "HoldingDays",
    "PnL", "ReturnPct", "RMultiple",    # derived
    "CloseReason", "Reasons",
]

def _ensure_open_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        df = pd.DataFrame(columns=OPEN_TRADES_COLUMNS)
    for c in OPEN_TRADES_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[OPEN_TRADES_COLUMNS]

def _ensure_closed_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        df = pd.DataFrame(columns=CLOSED_TRADES_COLUMNS)
    for c in CLOSED_TRADES_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[CLOSED_TRADES_COLUMNS]

def load_open_trades() -> pd.DataFrame:
    if not os.path.exists(OPEN_TRADES_PATH):
        return pd.DataFrame(columns=OPEN_TRADES_COLUMNS)
    try:
        df = pd.read_csv(OPEN_TRADES_PATH)
    except Exception:
        df = pd.DataFrame(columns=OPEN_TRADES_COLUMNS)
    return _ensure_open_schema(df)

def save_open_trades(df: pd.DataFrame) -> None:
    df = _ensure_open_schema(df)
    os.makedirs(os.path.dirname(OPEN_TRADES_PATH), exist_ok=True)
    df.to_csv(OPEN_TRADES_PATH, index=False)

def load_closed_trades() -> pd.DataFrame:
    if not os.path.exists(CLOSED_TRADES_PATH):
        return pd.DataFrame(columns=CLOSED_TRADES_COLUMNS)
    try:
        df = pd.read_csv(CLOSED_TRADES_PATH)
    except Exception:
        df = pd.DataFrame(columns=CLOSED_TRADES_COLUMNS)
    return _ensure_closed_schema(df)

def save_closed_trades(df: pd.DataFrame) -> None:
    df = _ensure_closed_schema(df)
    os.makedirs(os.path.dirname(CLOSED_TRADES_PATH), exist_ok=True)
    df.to_csv(CLOSED_TRADES_PATH, index=False)

def mark_live_from_queue(
        name: str,
        strategy: str,
        *,
        open_price: float | None = None,
        open_date: str | None = None
) -> bool:
    """
    Move *the most-recent* (Name, Strategy) row from Trade Queue ➜ Open Trades.

    • APPENDS – it no longer removes any existing live rows, so you can
      hold multiple positions in the same stock/strategy at different prices.

    Returns True if moved, False if not found.
    """
    q = load_trade_queue()
    if q.empty:
        return False

    mask = (
        q["Name"].astype(str).str.lower().eq(str(name).lower()) &
        q["Strategy"].astype(str).str.lower().eq(str(strategy).lower())
    )
    if not mask.any():
        return False

    # take the newest matching idea (last row)
    row = q.loc[mask].iloc[-1].to_dict()

    from datetime import datetime
    od = open_date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # prefer queue Entry; fall back to explicit open_price; else CurrentPrice
    entry = row.get("Entry")
    if pd.isna(entry) or entry is None:
        entry = open_price if open_price is not None else row.get("CurrentPrice")

    open_row = {
        "Name":     row.get("Name"),
        "Strategy": row.get("Strategy"),
        "Entry":    entry,
        "Stop":     row.get("Stop"),
        "Take":     row.get("Take"),
        "Shares":   row.get("Shares"),
        "RR":       row.get("RR"),
        "TP1":      row.get("TP1"),
        "TP2":      row.get("TP2"),
        "TP3":      row.get("TP3"),
        "OpenDate": od,
        "Reasons":  row.get("Reasons"),
    }

    # ←-----  **NO more de-dup** – just append
    open_df = load_open_trades()
    open_df = pd.concat([open_df, pd.DataFrame([open_row])], ignore_index=True)
    save_open_trades(open_df)

    # Remove the row we just moved from the queue
    q = q.drop(index=row["_index"] if "_index" in row else mask[mask].index[-1])
    save_trade_queue(q)

    # Audit
    try:
        append_queue_audit("MARK_LIVE", {**row, "Entry": entry}, audit_reason="mark_live_from_queue")
    except Exception:
        pass

    return True

def close_open_trade(name: str, strategy: str, *, close_price: float, close_date: str | None = None, close_reason: str = "") -> bool:
    """
    Close an open trade (Name, Strategy) with a required close_price.
    Moves the row to closed_trades with derived PnL, Return%, RMultiple.
    Returns True if closed, False if not found or invalid.
    """
    open_df = load_open_trades()
    if open_df.empty:
        return False

    mask = open_df["Name"].astype(str).str.lower().eq(str(name).lower()) & \
           open_df["Strategy"].astype(str).str.lower().eq(str(strategy).lower())
    if not mask.any():
        return False

    from datetime import datetime
    row = open_df.loc[mask].iloc[-1].to_dict()

    entry  = pd.to_numeric(pd.Series([row.get("Entry")]), errors="coerce").iloc[0]
    stop   = pd.to_numeric(pd.Series([row.get("Stop")]),  errors="coerce").iloc[0]
    take   = pd.to_numeric(pd.Series([row.get("Take")]),  errors="coerce").iloc[0]
    shares = pd.to_numeric(pd.Series([row.get("Shares")]),errors="coerce").iloc[0]
    rr0    = pd.to_numeric(pd.Series([row.get("RR")]),    errors="coerce").iloc[0]
    tp1    = pd.to_numeric(pd.Series([row.get("TP1")]),   errors="coerce").iloc[0]
    tp2    = pd.to_numeric(pd.Series([row.get("TP2")]),   errors="coerce").iloc[0]
    tp3    = pd.to_numeric(pd.Series([row.get("TP3")]),   errors="coerce").iloc[0]

    if pd.isna(shares) or shares <= 0:
        shares = 0

    cd = close_date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Derived
    pnl = None
    ret = None
    r_mult = None
    if pd.notna(entry) and pd.notna(close_price):
        pnl = (float(close_price) - float(entry)) * float(shares or 0)
        ret = ((float(close_price) / float(entry)) - 1.0) * 100.0
        if pd.notna(stop) and float(entry) > float(stop):
            risk_ps = float(entry) - float(stop)
            if risk_ps > 0:
                r_mult = (float(close_price) - float(entry)) / risk_ps

    # Holding days
    try:
        odt = pd.to_datetime(row.get("OpenDate"), errors="coerce")
        cdt = pd.to_datetime(cd, errors="coerce")
        holding_days = (cdt - odt).days if pd.notna(odt) and pd.notna(cdt) else pd.NA
    except Exception:
        holding_days = pd.NA

    closed_row = {
        "Name": row.get("Name"),
        "Strategy": row.get("Strategy"),
        "Entry": entry, "Stop": stop, "Take": take, "Shares": shares,
        "RR_Init": rr0, "TP1": tp1, "TP2": tp2, "TP3": tp3,
        "OpenDate": row.get("OpenDate"),
        "CloseDate": cd, "ClosePrice": float(close_price),
        "HoldingDays": holding_days,
        "PnL": pnl, "ReturnPct": ret, "RMultiple": r_mult,
        "CloseReason": close_reason,
        "Reasons": row.get("Reasons"),
    }

    # Append to closed
    closed_df = load_closed_trades()
    closed_df = pd.concat([closed_df, pd.DataFrame([closed_row])], ignore_index=True)
    save_closed_trades(closed_df)

    # Remove from open
    open_df = open_df.loc[~mask]
    save_open_trades(open_df)

    # Audit log
    try:
        append_queue_audit("CLOSE", {**row, "ClosePrice": close_price}, audit_reason=close_reason)
    except Exception:
        pass

    return True



# ==============================================================
#                OPTIONAL OHLC / ATR UTILITIES
# ==============================================================

# Supported layouts:
#   A) Per-stock files: data/ohlc/<Name>.csv   (columns: Date, Open, High, Low, Close)
#   B) One combined file: data/ohlc.csv        (same columns + a 'Name' column)
OHLC_DIR = "data/ohlc"
OHLC_COMBINED = "data/ohlc.csv"


def _safe_stock_filename(name: str) -> str:
    """Simple sanitizer for filenames (replace spaces and slashes)."""
    if name is None:
        return ""
    return str(name).strip().replace("/", "_").replace("\\", "_").replace(" ", "_")


def load_ohlc(name: str) -> pd.DataFrame:
    """
    Try to load OHLC for a stock:
      1) data/ohlc/<Name>.csv
      2) data/ohlc.csv filtered by Name (if it has a Name column)
    Returns a DataFrame with [Date, Open, High, Low, Close] (sorted by Date).
    If nothing found, returns empty DataFrame.
    """
    try_paths = []
    if name:
        try_paths.append(os.path.join(OHLC_DIR, _safe_stock_filename(name) + ".csv"))
    try_paths.append(OHLC_COMBINED)

    for path in try_paths:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        # If combined file and Name column exists, filter
        if "Name" in df.columns and name:
            df = df[df["Name"].astype(str).str.lower() == str(name).lower()]

        keep = [c for c in ["Date", "Open", "High", "Low", "Close"] if c in df.columns]
        if len(keep) < 4:  # need Date + at least H/L/C
            continue
        out = df[keep].copy()

        # Parse & clean
        if "Date" in out.columns:
            out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        for c in ["Open", "High", "Low", "Close"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.dropna(subset=["Date", "High", "Low", "Close"]).sort_values("Date")
        return out.reset_index(drop=True)

    return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close"])


def compute_atr(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Wilder's ATR using an RMA-style smoothing:
      TR = max(High-Low, abs(High-PrevClose), abs(Low-PrevClose))
      ATR_t = ATR_{t-1} + (1/period) * (TR_t - ATR_{t-1})
    Returns a pandas Series (same length as ohlc) with ATR values (NaN for first rows).
    """
    if ohlc.empty or any(c not in ohlc.columns for c in ["High", "Low", "Close"]):
        return pd.Series(dtype=float)

    high = ohlc["High"].astype(float)
    low = ohlc["Low"].astype(float)
    close = ohlc["Close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Wilder's smoothing via ewm with alpha=1/period (RMA style)
    atr = tr.ewm(alpha=(1 / period), adjust=False, min_periods=period).mean()
    return atr


def latest_atr(name: str, period: int = 14):
    """
    Convenience: load OHLC for 'name' and return the latest ATR value (float) and date.
    If unavailable, returns (None, None).
    """
    ohlc = load_ohlc(name)
    if ohlc.empty:
        return None, None
    atr_series = compute_atr(ohlc, period=period)
    if atr_series.empty or atr_series.dropna().empty:
        return None, None
    last_idx = atr_series.dropna().index[-1]
    return float(atr_series.loc[last_idx]), ohlc.loc[last_idx, "Date"]



