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

# ==== TTM HELPERS (clean, consolidated) ======================================
import math
import pandas as pd
import numpy as np

# Accept more header variants (extend to match your CSV headers)
TTM_ALIASES = {
    # Quarterly flow items (sum last 4) — added your exact names
    "Q_Revenue"        : ["Q_Revenue", "Revenue", "Sales", "Q_Sales", "Q_TotalRevenue",
                          "Quarterly Revenue"],
    "Q_GrossProfit"    : ["Q_GrossProfit", "GrossProfit", "Quarterly Gross Profit"],
    "Q_OperatingProfit": ["Q_OperatingProfit", "Q_EBIT", "OperatingProfit", "EBIT",
                          "Quarterly Operating Profit"],
    "Q_NetProfit"      : ["Q_NetProfit", "Q_Profit", "Q_NetIncome", "NetProfit", "NetIncome",
                          "Quarterly Net Profit"],
    "Q_EPS"            : ["Q_EPS", "EPS", "Basic EPS", "Diluted EPS",
                          "EPS (Basic)", "EPS (Diluted)", "Quarterly EPS"],
    "Q_EBITDA"         : ["Q_EBITDA", "EBITDA", "Quarterly EBITDA"],
    "Q_CFO"            : ["Q_CFO", "OperatingCashFlow", "Q_OperatingCashFlow",
                          "Quarterly Operating Cash Flow"],
    "Q_CapEx"          : ["Q_CapEx", "CapitalExpenditure", "CapEx",
                          "Quarterly Capital Expenditure"],

    # Depreciation / amortization (for EBITDA fallback)
    "DepAmort"         : ["Q_Depreciation", "Depreciation", "DepAmort",
                          "Depreciation And Amortization", "Depreciation of PPE",
                          "Depreciation expenses", "Quarterly Depreciation"],

    # Shares & EV pieces (quarterly OR annual — pick latest)
    "SharesOutstanding": ["SharesOutstanding", "ShareOutstanding", "ShareCount", "BasicShares",
                          "NumShares", "Number of Shares", "Number of shares", "Q_NumShares"],
    "TotalDebt"        : ["TotalDebt", "Debt", "Borrowings"],
    "Cash"             : ["Cash", "CashAndEquivalents", "Cash & Equivalents"],
}


def _pick_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None

def _pick_any_nonempty(stock_df: pd.DataFrame, names: list[str]) -> str | None:
    """
    Return the first column NAME that both exists AND has at least 1 numeric value.
    If none are non-empty, fall back to the first one that simply exists.
    """
    for n in names:
        if n in stock_df.columns:
            s = _to_num(stock_df[n]).dropna()
            if len(s) > 0:
                return n
    for n in names:  # fallback: just the first present, even if empty
        if n in stock_df.columns:
            return n
    return None


def _to_num(s: pd.Series) -> pd.Series:
    """
    Robust numeric conversion:
    - handles '800,000,000.0000', '1 234', '1.23%', 'RM 1.20', '$1.20'
    - leaves numeric dtypes as-is
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s = (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace(" ", "", regex=False)
         .str.replace("%", "", regex=False)
         .str.replace("RM", "", regex=False)
         .str.replace("$", "", regex=False)
    )
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
    """Sum last 4 quarters; return None if the column is missing or not enough data when require_4=True."""
    if not col_name or col_name not in df_quarters.columns:
        return None
    s = _to_num(df_quarters[col_name]).dropna().tail(4)
    if require_4 and len(s) < 4:
        return None
    if s.empty:
        return None
    return float(s.sum())



def compute_ttm(stock_df: pd.DataFrame, current_price: float | None = None) -> dict:
    """
    Compute common TTM totals and ratios from the latest 4 quarters.
    Totals: Revenue, GrossProfit, OperatingProfit, NetProfit, EBITDA, CFO, FCF
    Ratios: Gross/Operating/Net margin %, TTM EPS, P/E (TTM), P/S (TTM), EV/EBITDA (TTM)
    """
    out: dict[str, float | None] = {}

    q4 = last_n_quarters(stock_df, 4)
    if q4.empty:
        return out

    # --- resolve columns present in this stock’s data ---
    col_rev   = _pick_col(q4, TTM_ALIASES["Q_Revenue"])
    col_gp    = _pick_col(q4, TTM_ALIASES["Q_GrossProfit"])
    col_op    = _pick_col(q4, TTM_ALIASES["Q_OperatingProfit"])
    col_np    = _pick_col(q4, TTM_ALIASES["Q_NetProfit"])
    col_eps   = _pick_col(q4, TTM_ALIASES["Q_EPS"])
    col_ebit  = _pick_col(q4, TTM_ALIASES["Q_EBITDA"])
    col_cfo   = _pick_col(q4, TTM_ALIASES["Q_CFO"])
    col_capex = _pick_col(q4, TTM_ALIASES["Q_CapEx"])

    # May live in quarterly or annual rows; pick from the whole table
    col_dep   = _pick_any(stock_df, TTM_ALIASES["DepAmort"])
    col_sh    = _pick_any_nonempty(stock_df, TTM_ALIASES["SharesOutstanding"])
    col_debt  = _pick_any(stock_df, TTM_ALIASES["TotalDebt"])
    col_cash  = _pick_any(stock_df, TTM_ALIASES["Cash"])

    def valid(v):
        return v is not None and not (isinstance(v, float) and np.isnan(v))
    def tsum(col):
    if not col or col not in q4.columns: 
        return None
    s = _to_num(q4[col]).dropna().tail(4)
    return float(s.sum()) if not s.empty else None

    def pct(a, b):
        if not valid(a) or not valid(b) or float(b) == 0.0: return None
        return float(a) / float(b) * 100.0

    # --- TTM totals ---
    ttm_rev   = tsum(col_rev)
    ttm_gp    = tsum(col_gp)
    ttm_op    = tsum(col_op)
    ttm_np    = tsum(col_np)
    ttm_ebit  = tsum(col_ebit)
    ttm_cfo   = tsum(col_cfo)
    ttm_capex = tsum(col_capex)
    ttm_fcf   = (ttm_cfo - ttm_capex) if (valid(ttm_cfo) and valid(ttm_capex)) else None

    # EBITDA fallback = EBIT + Dep/Amort
    if not valid(ttm_ebit) and valid(ttm_op) and col_dep:
        if col_dep in q4.columns:
            ttm_dep = tsum(col_dep)
        else:
            ttm_dep = _to_num(stock_df[col_dep]).dropna().tail(1).sum() if col_dep in stock_df.columns else None
        if valid(ttm_dep):
            ttm_ebit = float(ttm_op) + float(ttm_dep)

    # EPS TTM: prefer quarterly EPS sum; else NetProfit / average(last 4) shares
    eps_ttm = tsum(col_eps)
    need_eps_fallback = (not valid(eps_ttm)) or (isinstance(eps_ttm, (int, float, np.floating)) and float(eps_ttm) == 0.0)
    if need_eps_fallback and valid(ttm_np) and col_sh:
        shares_series = q4[col_sh] if col_sh in q4.columns else stock_df[col_sh]
        s4 = _to_num(shares_series).dropna().tail(4)
        shares_avg = float(s4.mean()) if not s4.empty else (
            float(_to_num(shares_series).dropna().iloc[-1]) if not _to_num(shares_series).dropna().empty else None
        )
        if shares_avg and shares_avg > 0:
            eps_ttm = float(ttm_np) / shares_avg

    out.update({
        "TTM Revenue": ttm_rev,
        "TTM Gross Profit": ttm_gp,
        "TTM Operating Profit": ttm_op,
        "TTM Net Profit": ttm_np,
        "TTM EBITDA": ttm_ebit,
        "TTM CFO": ttm_cfo,
        "TTM CapEx": ttm_capex,
        "TTM FCF": ttm_fcf,
        "TTM Gross Margin (%)": pct(ttm_gp, ttm_rev),
        "TTM Operating Margin (%)": pct(ttm_op, ttm_rev),
        "TTM Net Margin (%)": pct(ttm_np, ttm_rev),
        "TTM EPS": eps_ttm,
    })

    # --- valuation multiples: use page-passed CURRENT price only ---
    mc = None
    if "MarketCap" in stock_df.columns:
        s_mc = _to_num(stock_df["MarketCap"]).dropna()
        if not s_mc.empty: mc = float(s_mc.iloc[-1])
    if mc is None and (current_price is not None) and col_sh:
        sh = _to_num(stock_df[col_sh]).dropna()
        if not sh.empty:
            mc = float(current_price) * float(sh.iloc[-1])

    if valid(current_price) and valid(eps_ttm) and float(eps_ttm) != 0.0:
        out["P/E (TTM)"] = float(current_price) / float(eps_ttm)
    if (mc is not None) and valid(ttm_rev) and float(ttm_rev) != 0.0:
        out["P/S (TTM)"] = float(mc) / float(ttm_rev)

    debt = _to_num(stock_df[col_debt]).dropna().iloc[-1] if col_debt and col_debt in stock_df.columns and not _to_num(stock_df[col_debt]).dropna().empty else None
    cash = _to_num(stock_df[col_cash]).dropna().iloc[-1] if col_cash and col_cash in stock_df.columns and not _to_num(stock_df[col_cash]).dropna().empty else None
    if (mc is not None) and valid(ttm_ebit) and float(ttm_ebit) != 0.0:
        net_debt = (float(debt) - float(cash)) if (debt is not None and cash is not None) else (float(debt) if debt is not None else 0.0)
        out["EV/EBITDA (TTM)"] = (float(mc) + net_debt) / float(ttm_ebit)

    return out


# ==== END TTM HELPERS =========================================================
