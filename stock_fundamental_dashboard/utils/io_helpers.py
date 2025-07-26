import pandas as pd
import os

DATA_FILE = "data/stocks.csv"
COLS = ["Stock", "Industry", "Price", "FairValue", "EPS", "BVPS", "ROE", "NetMargin",
        "Revenue", "NetProfit", "TTM_Revenue", "TTM_NetProfit", "Shares",
        "PE", "PB", "Score", "ScoreColor"]

def load_stocks():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        return pd.DataFrame(columns=COLS)

def save_stocks(df):
    df.to_csv(DATA_FILE, index=False)
