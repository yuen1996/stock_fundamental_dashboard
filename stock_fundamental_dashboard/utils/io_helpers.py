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
