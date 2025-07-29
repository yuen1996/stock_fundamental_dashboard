import pandas as pd
import os
from datetime import datetime

DATA_PATH = "data/stocks.csv"

# Unified schema (both Annual and Quarterly).
# Keep this list as the single source of truth to avoid KeyErrors.
ALL_COLUMNS = [
    "Name","Industry","Year","IsQuarter","Quarter",
    # Annual (income)
    "NetProfit","GrossProfit","Revenue","CostOfSales","FinanceCosts","AdminExpenses","SellDistExpenses",
    # Annual (balance / other)
    "NumShares","CurrentAsset","OtherReceivables","TradeReceivables","BiologicalAssets","Inventories","PrepaidExpenses",
    "IntangibleAsset","CurrentLiability","TotalAsset","TotalLiability","ShareholderEquity","Reserves",
    "Dividend","SharePrice",
    # Independent per‑stock current price (used by ratios/TTM)
    "CurrentPrice",
    # Quarterly (prefix Q_)
    "Q_NetProfit","Q_GrossProfit","Q_Revenue","Q_CostOfSales","Q_FinanceCosts","Q_AdminExpenses","Q_SellDistExpenses",
    "Q_NumShares","Q_CurrentAsset","Q_OtherReceivables","Q_TradeReceivables","Q_BiologicalAssets","Q_Inventories",
    "Q_PrepaidExpenses","Q_IntangibleAsset","Q_CurrentLiability","Q_TotalAsset","Q_TotalLiability",
    "Q_ShareholderEquity","Q_Reserves","Q_SharePrice","Q_EndQuarterPrice",
    # Per‑row timestamp
    "LastModified",
]

# -------- Watchlist storage --------
WATCHLIST_PATH = "data/watchlist.csv"
WATCHLIST_COLUMNS = ["Name", "TargetPrice", "Notes", "Active"]

# -------- Trade Queue storage --------
TRADE_QUEUE_PATH = "data/trade_queue.csv"
TRADE_QUEUE_COLUMNS = ["Name","Strategy","Score","CurrentPrice","Timestamp","Reasons"]

# -------- Trade Queue archive (when removing with reasons) --------
TRADE_QUEUE_ARCHIVE_PATH = "data/trade_queue_archive.csv"
TRADE_QUEUE_ARCHIVE_COLUMNS = [
    "Name","Strategy","Score","CurrentPrice","Timestamp","Reasons",
    "RemovedAt","RemoveReason"
]

# -------- Positions (planned / entered / exited) --------
POSITIONS_PATH = "data/positions.csv"
POSITIONS_COLUMNS = [
    "Name","Strategy","Status",           # Planned | Entered | Exited
    "EntryPrice","StopPrice","TargetPrice",
    "Shares","Capital","RiskPerTradePct","AccountEquity","RiskAmount","ExpectedR",
    "Industry","PlannedAt","Notes"
]



def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        df = pd.DataFrame(columns=ALL_COLUMNS)
    # Add missing columns; do not drop unknown columns (backward compatibility)
    for col in ALL_COLUMNS:
        if col not in df.columns:
            if col == "IsQuarter":
                df[col] = False
            elif col == "Quarter":
                df[col] = pd.NA
            else:
                df[col] = pd.NA
    return df


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        return ensure_schema(pd.DataFrame(columns=ALL_COLUMNS))
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception:
        # Corrupt or empty file, start fresh with proper schema
        df = pd.DataFrame(columns=ALL_COLUMNS)
    return ensure_schema(df)


def save_data(df: pd.DataFrame) -> None:
    # 1) Ensure schema (so LastModified column exists)
    df = ensure_schema(df)

    # 2) Load the existing on‑disk DataFrame (to compare old timestamps and values)
    try:
        old = load_data()  # this is the same load_data above
    except Exception:
        old = pd.DataFrame(columns=ALL_COLUMNS)

    # 3) Build a lookup on (Name, IsQuarter, Year, Quarter) → old row
    old = old.set_index(["Name", "IsQuarter", "Year", "Quarter"], drop=False)

    # 4) Now walk each row in the new df; if it matches old (ignoring LastModified),
    #    preserve the old LastModified; otherwise stamp with now()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = []
    for _, new_row in df.iterrows():
        key = (new_row["Name"], new_row["IsQuarter"], new_row["Year"], new_row["Quarter"])
        if key in old.index:
            old_row = old.loc[key]
            # compare all columns except LastModified
            # note: old_row may be a Series; drop LastModified from both
            nr = new_row.drop(labels="LastModified")
            orow = old_row.drop(labels="LastModified")
            if nr.equals(orow):
                new_row["LastModified"] = old_row["LastModified"]
            else:
                new_row["LastModified"] = now
        else:
            # brand new record → stamp
            new_row["LastModified"] = now
        out.append(new_row)

    # 5) Reassemble and write back
    df2 = pd.DataFrame(out)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df2.to_csv(DATA_PATH, index=False)



# -------- Watchlist helpers --------
def _ensure_watchlist_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        df = pd.DataFrame(columns=WATCHLIST_COLUMNS)
    for col in WATCHLIST_COLUMNS:
        if col not in df.columns:
            if col == "Active":
                df[col] = True
            else:
                df[col] = pd.NA

    # Type conversions (important for Streamlit editor)
    if "Active" in df.columns:
        df["Active"] = df["Active"].fillna(True).astype(bool)
    if "TargetPrice" in df.columns:
        df["TargetPrice"] = pd.to_numeric(df["TargetPrice"], errors="coerce")
    if "Notes" in df.columns:
        # keep empty strings (not floats) so TextColumn works
        df["Notes"] = df["Notes"].astype("string").fillna("")
    # Return columns in canonical order
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

def load_trade_queue() -> pd.DataFrame:
    if not os.path.exists(TRADE_QUEUE_PATH):
        return pd.DataFrame(columns=TRADE_QUEUE_COLUMNS)
    try:
        df = pd.read_csv(TRADE_QUEUE_PATH)
    except Exception:
        df = pd.DataFrame(columns=TRADE_QUEUE_COLUMNS)
    # ensure columns
    for c in TRADE_QUEUE_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[TRADE_QUEUE_COLUMNS]

def save_trade_queue(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(TRADE_QUEUE_PATH), exist_ok=True)
    df = df[TRADE_QUEUE_COLUMNS]
    df.to_csv(TRADE_QUEUE_PATH, index=False)

def push_trade_candidate(name: str, strategy: str, score: float, current_price, reasons: str = "") -> None:
    from datetime import datetime
    q = load_trade_queue()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "Name": name,
        "Strategy": strategy,
        "Score": score,
        "CurrentPrice": current_price,
        "Timestamp": ts,
        "Reasons": reasons,
    }

    # robust match (works even if there are NaNs or mixed dtypes)
    mask = (
        q["Name"].astype(str).str.lower().eq(str(name).lower()) &
        q["Strategy"].astype(str).str.lower().eq(str(strategy).lower())
    )

    if mask.any():
        # assign per column so it works for one or many matched rows
        for k, v in row.items():
            q.loc[mask, k] = v
    else:
        q = pd.concat([q, pd.DataFrame([row])], ignore_index=True)

    # keep schema + de‑duplicate (keep the latest push)
    q = q[TRADE_QUEUE_COLUMNS]
    q = q.drop_duplicates(subset=["Name", "Strategy"], keep="last")
    save_trade_queue(q)

def load_trade_queue_archive() -> pd.DataFrame:
    if not os.path.exists(TRADE_QUEUE_ARCHIVE_PATH):
        return pd.DataFrame(columns=TRADE_QUEUE_ARCHIVE_COLUMNS)
    try:
        df = pd.read_csv(TRADE_QUEUE_ARCHIVE_PATH)
    except Exception:
        df = pd.DataFrame(columns=TRADE_QUEUE_ARCHIVE_COLUMNS)
    for c in TRADE_QUEUE_ARCHIVE_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[TRADE_QUEUE_ARCHIVE_COLUMNS]

def save_trade_queue_archive(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(TRADE_QUEUE_ARCHIVE_PATH), exist_ok=True)
    df = df[TRADE_QUEUE_ARCHIVE_COLUMNS]
    df.to_csv(TRADE_QUEUE_ARCHIVE_PATH, index=False)

def remove_trade_candidate(name: str, strategy: str, reason: str) -> bool:
    """Move (Name, Strategy) from active queue to archive with RemoveReason."""
    from datetime import datetime
    q = load_trade_queue()
    if q.empty:
        return False

    mask = (
        q["Name"].astype(str).str.lower().eq(str(name).lower()) &
        q["Strategy"].astype(str).str.lower().eq(str(strategy).lower())
    )
    if not mask.any():
        return False

    # rows to archive
    rows = q.loc[mask].copy()
    rows["RemovedAt"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows["RemoveReason"] = reason

    # append to archive
    arch = load_trade_queue_archive()
    arch = pd.concat([arch, rows[TRADE_QUEUE_ARCHIVE_COLUMNS]], ignore_index=True)
    save_trade_queue_archive(arch)

    # drop from active queue
    q = q.loc[~mask].copy()
    save_trade_queue(q)
    return True

def load_positions() -> pd.DataFrame:
    if not os.path.exists(POSITIONS_PATH):
        return pd.DataFrame(columns=POSITIONS_COLUMNS)
    try:
        df = pd.read_csv(POSITIONS_PATH)
    except Exception:
        df = pd.DataFrame(columns=POSITIONS_COLUMNS)
    for c in POSITIONS_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[POSITIONS_COLUMNS]

def save_positions(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(POSITIONS_PATH), exist_ok=True)
    df = df[POSITIONS_COLUMNS]
    df.to_csv(POSITIONS_PATH, index=False)

def upsert_position(
    name: str,
    strategy: str,
    status: str,
    entry: float,
    stop: float,
    target: float | None,
    shares: int,
    capital: float,
    risk_pct: float,
    equity: float,
    risk_amt: float,
    expected_r: float | None,
    industry: str = "",
    notes: str = ""
) -> None:
    from datetime import datetime
    pos = load_positions()
    row = {
        "Name": name,
        "Strategy": strategy,
        "Status": status,
        "EntryPrice": entry,
        "StopPrice": stop,
        "TargetPrice": target,
        "Shares": int(shares),
        "Capital": capital,
        "RiskPerTradePct": risk_pct,
        "AccountEquity": equity,
        "RiskAmount": risk_amt,
        "ExpectedR": expected_r,
        "Industry": industry,
        "PlannedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Notes": notes,
    }
    # unique upsert by (Name, Strategy, Status in Planned/Entered not exited)
    mask = (
        pos["Name"].astype(str).str.lower().eq(str(name).lower()) &
        pos["Strategy"].astype(str).str.lower().eq(str(strategy).lower()) &
        pos["Status"].isin(["Planned","Entered"])
    )
    if mask.any():
        for k, v in row.items():
            pos.loc[mask, k] = v
    else:
        pos = pd.concat([pos, pd.DataFrame([row])], ignore_index=True)

    pos = pos[POSITIONS_COLUMNS]
    save_positions(pos)

