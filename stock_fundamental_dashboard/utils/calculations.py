from math import isfinite

def _to_number(x, default=0):
    try:
        if x is None:
            return default
        if isinstance(x, str):
            x = x.replace(",", "").strip()
            if x == "":
                return default
        return float(x)
    except Exception:
        return default

def pick(row, *names, default=0):
    """Return the first present, non-NaN value among the provided keys."""
    for n in names:
        if n in row:
            val = row[n]
            try:
                # Skip NaN
                if val != val:  # type: ignore[comparison-overlap]
                    continue
            except Exception:
                pass
            return _to_number(val, default)
    return default

def safe_div(a, b):
    a = _to_number(a, None)
    b = _to_number(b, None)
    if a is None or b is None or b == 0:
        return None
    return a / b

def percent(val):
    return None if val is None else val * 100.0

def calc_ratios(row):
    # Income (accept both spaced and camelCase names; and Q_ variants)
    np_ = pick(row, "NetProfit", "Net profit", "Net Profit", "Q_NetProfit", default=0)
    gp  = pick(row, "GrossProfit", "Gross Profit", "Q_GrossProfit", default=0)
    rev = pick(row, "Revenue", "Q_Revenue", default=0)
    cos = pick(row, "CostOfSales", "Cost Of Sales", "Q_CostOfSales", default=0)
    fin = pick(row, "FinanceCosts", "Finance Costs", "Q_FinanceCosts", default=0)
    adm = pick(row, "AdminExpenses", "Administrative Expenses", "Administrative  Expenses", "Q_AdminExpenses", default=0)
    sell= pick(row, "SellDistExpenses", "Selling & Distribution Expenses", "Selling and distribution expenses", "Q_SellDistExpenses", default=0)

    # Balance/other
    shares = pick(row, "NumShares", "Number of Shares", "Number of shares", "ShareOutstanding", "Q_NumShares", default=0)

    # 👉 Prefer CurrentPrice; gracefully fall back to any other price fields
    price  = pick(
        row,
        "CurrentPrice",                 # NEW: per‑stock current price
        "SharePrice",                  # annual end-of-year price
        "Current Share Price",         # older field name
        "End of year share price",     # label form
        "Each end of year share price",
        "Q_SharePrice",                # quarterly current price (if present)
        "Q_EndQuarterPrice",
        "Price",                       # legacy
        default=0
    )

    # Assumed currency per share unless user says otherwise
    div_ps = pick(row, "Dividend", "Dividend pay cent", default=0)

    curr_asset = pick(row, "CurrentAsset", "Current Asset", "Q_CurrentAsset", default=0)
    curr_liab  = pick(row, "CurrentLiability", "Current Liability", "Q_CurrentLiability", default=0)
    inventories= pick(row, "Inventories", "Inventories  (-from current asset)", "Q_Inventories", default=0)

    tot_asset  = pick(row, "TotalAsset", "Total Asset", "Asset", "Q_TotalAsset", default=0)
    tot_liab   = pick(row, "TotalLiability", "Total Liability", "Liability", "Q_TotalLiability", default=0)
    equity     = pick(row, "ShareholderEquity", "Shareholder Equity", "Equity", "Q_ShareholderEquity", default=0)
    intangible = pick(row, "IntangibleAsset", "Intangible asset  (when calculate NTA need to deduct)", "Intangible Asset", "Q_IntangibleAsset", default=0)

    # Per-share
    eps     = safe_div(np_, shares)
    bvps    = safe_div(equity, shares)
    nta_ps  = safe_div(max(tot_asset - intangible - tot_liab, 0), shares)

    # Margins
    gross_margin = percent(safe_div(gp, rev))
    net_margin   = percent(safe_div(np_, rev))

    # Liquidity / leverage
    debt_asset   = percent(safe_div(tot_liab, tot_asset))
    current_ratio= safe_div(curr_asset, curr_liab)
    quick_ratio  = safe_div(max(curr_asset - inventories, 0), curr_liab)

    # Cost structure
    three_fees     = percent(safe_div(adm + fin + sell, rev))
    total_cost_pct = percent(safe_div(cos + adm + fin + sell, rev))

    # Valuation
    pe = safe_div(price, eps) if eps and eps > 0 else None
    pb = safe_div(price, bvps) if bvps and bvps > 0 else None

    # Dividends (assuming div_ps is currency/share)
    div_payout = percent(safe_div(div_ps * shares, np_))
    div_yield  = percent(safe_div(div_ps, price))

    # Profitability
    roe = percent(safe_div(np_, equity))

    return {
        "EPS": eps,
        "BVPS": bvps,
        "ROE (%)": roe,
        "Revenue": rev,
        "NetProfit": np_,
        "Debt-Asset Ratio (%)": debt_asset,
        "Three Fees Ratio (%)": three_fees,
        "Total Cost %": total_cost_pct,
        "Dividend Payout Ratio (%)": div_payout,
        "Dividend Yield (%)": div_yield,
        "Current Ratio": current_ratio,
        "Quick Ratio": quick_ratio,
        "Gross Profit Margin (%)": gross_margin,
        "Net Profit Margin (%)": net_margin,
        "NTA per share": nta_ps,
        "P/E": pe,
        "P/B": pb,
    }

# ==== TTM HELPERS (expanded & with fallbacks) =================================
import math
import pandas as pd
import numpy as np

# Accept more header variants (edit/extend to match your data)
TTM_ALIASES = {
    # Quarterly flow items
    "Q_Revenue"        : ["Q_Revenue", "Revenue", "Sales", "Q_Sales", "Q_TotalRevenue"],
    "Q_GrossProfit"    : ["Q_GrossProfit", "GrossProfit"],
    "Q_OperatingProfit": ["Q_OperatingProfit", "Q_EBIT", "OperatingProfit", "EBIT"],
    "Q_NetProfit"      : ["Q_NetProfit", "Q_Profit", "Q_NetIncome", "NetProfit", "NetIncome"],
    "Q_EPS"            : ["Q_EPS", "EPS", "Basic EPS", "Diluted EPS"],
    "Q_EBITDA"         : ["Q_EBITDA", "EBITDA"],
    "Q_CFO"            : ["Q_CFO", "OperatingCashFlow", "Q_OperatingCashFlow"],
    "Q_CapEx"          : ["Q_CapEx", "CapitalExpenditure", "CapEx"],
    # For fallbacks / valuation
    "SharesOutstanding": ["SharesOutstanding", "ShareOutstanding", "ShareCount", "BasicShares", "NumShares", "Q_NumShares"],
    "TotalDebt"        : ["TotalDebt", "Debt", "Borrowings"],
    "Cash"             : ["Cash", "CashAndEquivalents"],
    "DepAmort"         : ["Q_Depreciation", "Depreciation", "DepAmort", "DepreciationAndAmortization"],
}

def _pick_col(df: pd.DataFrame, names: list[str]) -> str | None:
    """Return the first existing column name from names, else None (searches provided df)."""
    for n in names:
        if n in df.columns:
            return n
    return None

def _pick_any(stock_df: pd.DataFrame, names: list[str]) -> str | None:
    """Return first existing column across the FULL stock_df (not just quarters)."""
    for n in names:
        if n in stock_df.columns:
            return n
    return None

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _q_to_int(q):
    """Convert 1/2/3/4, 'Q1'..'Q4', 'Quarter 1', '1Q' → 1..4."""
    if pd.isna(q): return np.nan
    try:
        qi = int(q)
        return qi if qi in (1,2,3,4) else np.nan
    except Exception:
        s = str(q).strip().upper().replace("QUARTER", "Q").replace(" ", "")
        if s.startswith("Q") and len(s) >= 2 and s[1].isdigit():
            qi = int(s[1]);  return qi if qi in (1,2,3,4) else np.nan
        if s.endswith("Q") and s[0].isdigit():
            qi = int(s[0]);  return qi if qi in (1,2,3,4) else np.nan
        return np.nan

def last_n_quarters(df: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    """Return most recent n quarterly rows sorted by Year, Quarter."""
    q = df[df.get("IsQuarter", False) == True].copy()
    if q.empty: return q
    y  = _to_num(q.get("Year", pd.Series(index=q.index)))
    qi = q.get("Quarter", pd.Series(index=q.index)).map(_q_to_int)
    q = q.assign(_Year=y, _Q=qi).sort_values(by=["_Year", "_Q"], na_position="last")
    q = q.drop(columns=["_Year", "_Q"], errors="ignore")
    return q.tail(n)

def ttm_sum(df_quarters: pd.DataFrame, col_name: str | None, require_4: bool = False) -> float | None:
    """Sum last 4 quarters for a numeric column; returns None if not enough data when require_4=True."""
    if not col_name or col_name not in df_quarters.columns: return None
    s = _to_num(df_quarters[col_name]).dropna()
    if require_4 and len(s.tail(4)) < 4: return None
    if len(s) < 1: return None
    return float(s.tail(4).sum())

def compute_ttm(stock_df: pd.DataFrame, current_price: float | None = None) -> dict:
    """
    Compute common TTM totals and ratios from the latest 4 quarters.
    - Totals: Revenue, GrossProfit, OperatingProfit, NetProfit, EBITDA, CFO, FCF
    - Ratios: TTM margins, P/E (TTM), P/S (TTM), EV/EBITDA (TTM)
    """
    out = {}
    q4 = last_n_quarters(stock_df, 4)
    if q4.empty: return out

    # --- choose columns ---
    col_rev   = _pick_col(q4, TTM_ALIASES["Q_Revenue"])
    col_gp    = _pick_col(q4, TTM_ALIASES["Q_GrossProfit"])
    col_op    = _pick_col(q4, TTM_ALIASES["Q_OperatingProfit"])
    col_np    = _pick_col(q4, TTM_ALIASES["Q_NetProfit"])
    col_eps   = _pick_col(q4, TTM_ALIASES["Q_EPS"])
    col_ebit  = _pick_col(q4, TTM_ALIASES["Q_EBITDA"])
    col_cfo   = _pick_col(q4, TTM_ALIASES["Q_CFO"])
    col_capex = _pick_col(q4, TTM_ALIASES["Q_CapEx"])
    col_dep   = _pick_any(stock_df, TTM_ALIASES["DepAmort"])              # may be quarterly or annual
    col_sh    = _pick_any(stock_df, TTM_ALIASES["SharesOutstanding"])      # often annual
    col_debt  = _pick_any(stock_df, TTM_ALIASES["TotalDebt"])
    col_cash  = _pick_any(stock_df, TTM_ALIASES["Cash"])

    # --- core TTM totals ---
    rev  = ttm_sum(q4, col_rev)
    gp   = ttm_sum(q4, col_gp)
    op   = ttm_sum(q4, col_op)
    np_  = ttm_sum(q4, col_np)
    eps  = ttm_sum(q4, col_eps)  # may be None

    # EPS fallback: use NetProfit / SharesOutstanding (latest non-null shares)
    if eps is None:
        if col_sh:
            sh_series = _to_num(stock_df[col_sh]).dropna()
            if not sh_series.empty and np_ is not None:
                eps = np_ / float(sh_series.iloc[-1])

    # EBITDA fallback: OperatingProfit + Depreciation/Amortization
    ebitda = ttm_sum(q4, col_ebit)
    if ebitda is None and col_op and col_dep:
        # prefer quarterly dep/amort if present; if annual, just use last value as an approximation
        if col_dep in q4.columns:
            dep = ttm_sum(q4, col_dep)
        else:
            dep_series = _to_num(stock_df[col_dep]).dropna()
            dep = float(dep_series.iloc[-1]) if not dep_series.empty else None
        if dep is not None and op is not None:
            ebitda = op + dep

    cfo   = ttm_sum(q4, col_cfo)
    capex = ttm_sum(q4, col_capex)
    fcf   = (cfo - capex) if (cfo is not None and capex is not None) else None

    # --- margins ---
    def pct(a, b):
        try:
            if a is None or b is None or float(b) == 0.0: return None
            return float(a) / float(b) * 100.0
        except Exception:
            return None

    gm = pct(gp, rev)
    om = pct(op, rev)
    nm = pct(np_, rev)

    out.update({
        "TTM Revenue": rev,
        "TTM Gross Profit": gp,
        "TTM Operating Profit": op,
        "TTM Net Profit": np_,
        "TTM EBITDA": ebitda,
        "TTM CFO": cfo,
        "TTM CapEx": capex,
        "TTM FCF": fcf,
        "TTM Gross Margin (%)": gm,
        "TTM Operating Margin (%)": om,
        "TTM Net Margin (%)": nm,
        "TTM EPS": eps,
    })

    # --- valuation multiples ---
    # MarketCap: preferred column else price × shares
    mc = None
    if "MarketCap" in stock_df.columns:
        s = _to_num(stock_df["MarketCap"]).dropna()
        if not s.empty: mc = float(s.iloc[-1])
    if mc is None and (current_price is not None) and col_sh:
        sh_series = _to_num(stock_df[col_sh]).dropna()
        if not sh_series.empty:
            mc = float(current_price) * float(sh_series.iloc[-1])

    if current_price is not None and eps not in (None, 0):

