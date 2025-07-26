import pandas as pd
import os

DATA_PATH = "data/stocks.csv"

def load_data():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame(columns=[
            'Name', 'Industry', 'Year', 'NetProfit', 'Revenue', 'Equity',
            'Asset', 'Liability', 'Dividend', 'ShareOutstanding', 'Price'
        ])
    return pd.read_csv(DATA_PATH)

def save_data(df):
    df.to_csv(DATA_PATH, index=False)
