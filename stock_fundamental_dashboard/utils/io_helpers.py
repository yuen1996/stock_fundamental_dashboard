import pandas as pd
import os

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
    # Quarterly (prefix Q_)
    "Q_NetProfit","Q_GrossProfit","Q_Revenue","Q_CostOfSales","Q_FinanceCosts","Q_AdminExpenses","Q_SellDistExpenses",
    "Q_NumShares","Q_CurrentAsset","Q_OtherReceivables","Q_TradeReceivables","Q_BiologicalAssets","Q_Inventories",
    "Q_PrepaidExpenses","Q_IntangibleAsset","Q_CurrentLiability","Q_TotalAsset","Q_TotalLiability",
    "Q_ShareholderEquity","Q_Reserves","Q_SharePrice","Q_EndQuarterPrice",
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
    df = ensure_schema(df)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

