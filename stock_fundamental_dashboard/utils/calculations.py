import numpy as np

def calc_ratios(row):
    """Calculate all ratios and key metrics for a stock (input: DataFrame row)."""
    ratios = {}
    try:
        ratios['ROE'] = 100 * float(row['NetProfit']) / float(row['Equity']) if float(row['Equity']) else 0
        ratios['Current Ratio'] = float(row['Asset']) / float(row['Liability']) if float(row['Liability']) else 0
        ratios['Dividend Yield'] = 100 * float(row['Dividend']) / float(row['Price']) if float(row['Price']) else 0
        ratios['PE'] = float(row['Price']) / (float(row['NetProfit'])/float(row['ShareOutstanding'])) if float(row['NetProfit']) else np.nan
        ratios['PB'] = float(row['Price']) / (float(row['Equity'])/float(row['ShareOutstanding'])) if float(row['Equity']) else np.nan
        ratios['Debt Asset Ratio'] = 100 * float(row['Liability']) / float(row['Asset']) if float(row['Asset']) else 0
    except Exception:
        for k in ['ROE','Current Ratio','Dividend Yield','PE','PB','Debt Asset Ratio']:
            ratios[k] = np.nan
    return ratios

def calc_cagr(first, last, periods):
    """Compound annual growth rate."""
    try:
        if first <= 0 or last <= 0 or periods <= 0:
            return np.nan
        return 100 * (pow(last/first, 1/periods)-1)
    except Exception:
        return np.nan

def calc_graham_value(eps, bvps, y=4.4, ye=4.4):
    """Benjamin Graham valuation formula (simplified)."""
    try:
        return np.sqrt(22.5 * eps * bvps)
    except Exception:
        return np.nan

def margin_of_safety(price, intrinsic):
    try:
        return 100 * (intrinsic - price) / intrinsic if intrinsic else np.nan
    except Exception:
        return np.nan

def radar_scores(ratios):
    """Return normalized (0-1) dict for radar chart sections: VALUE, DIVIDEND, HEALTH, FUTURE, PAST"""
    return {
        "VALUE": min(1, max(0, (15 - ratios.get("PE", 0)) / 15)),  # lower PE = better value
        "DIVIDEND": min(1, ratios.get("Dividend Yield", 0) / 4),   # above 4% = good
        "HEALTH": min(1, ratios.get("Current Ratio", 0) / 2),      # above 2 = good
        "FUTURE": 0.7,  # Expand with your own logic
        "PAST": 0.7,
    }
