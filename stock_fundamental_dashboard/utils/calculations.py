from math import isfinite

import pandas as pd
import numpy as np


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
    is_quarter = row.get("IsQuarter", False) is True

    # Income (accept both spaced and camelCase names; and Q_ variants)
    np_ = pick(row, "Q_NetProfit", default=0) if is_quarter else pick(row, "NetProfit", "Net profit", "Net Profit", default=0)
    gp  = pick(row, "Q_GrossProfit", default=0) if is_quarter else pick(row, "GrossProfit", "Gross Profit", default=0)
    rev = pick(row, "Q_Revenue", default=0) if is_quarter else pick(row, "Revenue", default=0)
    cos = pick(row, "Q_CostOfSales", default=0) if is_quarter else pick(row, "CostOfSales", "Cost Of Sales", default=0)
    fin = pick(row, "Q_FinanceCosts", default=0) if is_quarter else pick(row, "FinanceCosts", "Finance Costs", default=0)
    adm = pick(row, "Q_AdminExpenses", default=0) if is_quarter else pick(row, "AdminExpenses", "Administrative Expenses", "Administrative  Expenses", default=0)
    sell= pick(row, "Q_SellDistExpenses", default=0) if is_quarter else pick(row, "SellDistExpenses", "Selling & Distribution Expenses", "Selling and distribution expenses", default=0)

    # Balance/other
    shares = pick(row, "Q_NumShares", default=0) if is_quarter else pick(row, "NumShares", "Number of Shares", "Number of shares", "ShareOutstanding", default=0)

    # Correct: For annual use only annual; for quarter only quarter
    if is_quarter:
        price  = pick(row, "Q_EndQuarterPrice", "Q_SharePrice", default=0)
        equity = pick(row, "Q_ShareholderEquity", default=0)
        div_ps = pick(row, "Q_Dividend", default=0)  # Only if you have quarterly dividend per share, else leave 0
        eps    = pick(row, "Q_EPS", default=0)
    else:
        price  = pick(row, "SharePrice", "Current Share Price", "End of year share price", "Each end of year share price", default=0)
        equity = pick(row, "ShareholderEquity", "Shareholder Equity", "Equity", default=0)
        div_ps = pick(row, "Dividend", "Dividend pay cent", default=0)
        eps    = pick(row, "EPS", default=0)

    curr_asset = pick(row, "CurrentAsset", "Current Asset", "Q_CurrentAsset", default=0)
    curr_liab  = pick(row, "CurrentLiability", "Current Liability", "Q_CurrentLiability", default=0)
    inventories= pick(row, "Inventories", "Inventories  (-from current asset)", "Q_Inventories", default=0)

    tot_asset  = pick(row, "TotalAsset", "Total Asset", "Asset", "Q_TotalAsset", default=0)
    tot_liab   = pick(row, "TotalLiability", "Total Liability", "Liability", "Q_TotalLiability", default=0)
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

    # --- Strict period correction for valuation ---
    # P/E: annual rows use annual price/eps; quarterly rows use quarterly price/eps
    # P/B: same
    # Dividend Yield: period dividend per share / period price
    pe = safe_div(price, eps) if eps and eps > 0 else None
    pb = safe_div(price, bvps) if bvps and bvps > 0 else None
    div_payout = percent(safe_div(div_ps * shares, np_)) if np_ else None
    div_yield  = percent(safe_div(div_ps, price)) if price else None
    roe = percent(safe_div(np_, equity)) if equity else None

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

# ==== TTM HELPERS ============================================================
import pandas as pd
import numpy as np
import math

# Accept more header variants (extend to match your CSV headers)
TTM_ALIASES = {
    # Quarterly flow items (sum last 4)
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

    # Shares & EV pieces (quarterly OR annual — pick whichever has data)
    "SharesOutstanding": ["CurrentShares", "SharesOutstanding", "ShareOutstanding", "ShareCount",
                          "BasicShares", "NumShares", "Number of Shares", "Number of shares",
                          "Q_NumShares"],
    "TotalDebt"        : ["TotalDebt", "Debt", "Borrowings"],
    "Cash"             : ["Cash", "CashAndEquivalents", "Cash & Equivalents"],
    "Equity"           : ["ShareholderEquity", "Shareholder Equity", "Equity", "TotalEquity", "Total Equity", "Q_ShareholderEquity"],

}

# Add quarterly finance and tax aliases (used for EBITDA fallback)
TTM_ALIASES["Q_Finance"] = [
    "Q_FinanceCosts", "FinanceCosts", "Finance cost", "Finance costs",
    "InterestExpense", "Interest Expense", "Finance expenses"
]
TTM_ALIASES["Q_Tax"] = [
    "Q_Tax", "IncomeTax", "Income Tax", "Income Tax Expense",
    "Taxation", "Tax expense"
]




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
        if s.startswith("Q") and len(s) >= 2 and s[1].isdigit():  return int(s[1])
        if s.endswith("Q") and s[0].isdigit():                    return int(s[0])
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
    for n in names:  # fallback: first present (even if empty)
        if n in stock_df.columns:
            return n
    return None

def ttm_sum(df_quarters: pd.DataFrame, col_name: str | None, require_4: bool = False) -> float | None:
    """Sum last 4 quarters (NaNs dropped). If require_4=True, need 4 values."""
    if not col_name or col_name not in df_quarters.columns:
        return None
    s = _to_num(df_quarters[col_name]).dropna().tail(4)
    if require_4 and len(s) < 4:
        return None
    if s.empty:
        return None
    return float(s.sum())

def _latest_non_nan(series: pd.Series) -> float | None:
    s = _to_num(series).dropna()
    return float(s.iloc[-1]) if not s.empty else None


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

    # ---------- Data Health ----------
    missing = []
    if out.get("TTM CFO") is None or out.get("TTM CapEx") is None:
        missing.append("CFO or CapEx")
    if out.get("Interest Coverage (EBITDA/Fin)") is None:
        missing.append("Finance cost or EBITDA")
    if out.get("Net Cash (Debt)") is None:
        missing.append("Debt/Cash balance")
    out["DataHealth"] = {"missing": missing, "estimated": []}


    # --- resolve columns present in this stock’s data (quarterly) ---
    col_rev   = _pick_col(q4, TTM_ALIASES["Q_Revenue"])
    col_gp    = _pick_col(q4, TTM_ALIASES["Q_GrossProfit"])
    col_op    = _pick_col(q4, TTM_ALIASES["Q_OperatingProfit"])
    col_np    = _pick_col(q4, TTM_ALIASES["Q_NetProfit"])
    col_eps   = _pick_col(q4, TTM_ALIASES["Q_EPS"])
    col_ebit  = _pick_col(q4, TTM_ALIASES["Q_EBITDA"])
    col_cfo   = _pick_col(q4, TTM_ALIASES["Q_CFO"])
    col_capex = _pick_col(q4, TTM_ALIASES["Q_CapEx"])
    # for EBITDA fallback
    col_fin   = _pick_col(q4, TTM_ALIASES.get("Q_Finance", []))
    col_tax   = _pick_col(q4, TTM_ALIASES.get("Q_Tax", []))

    # These may live in quarterly or annual rows; pick from the whole table
    col_dep   = _pick_any_nonempty(stock_df, TTM_ALIASES["DepAmort"])
    col_sh    = _pick_any_nonempty(stock_df, TTM_ALIASES["SharesOutstanding"])
    col_debt  = _pick_any_nonempty(stock_df, TTM_ALIASES["TotalDebt"])
    col_cash  = _pick_any_nonempty(stock_df, TTM_ALIASES["Cash"])
    col_equity = _pick_any_nonempty(stock_df, TTM_ALIASES["Equity"])

    # Always use the latest YEAR shares if available (prefer annual "Number of Shares")
    shares_latest = None
    for name in [
        "Number of Shares", "Number of shares", "NumShares",  # annual first
        "CurrentShares", "SharesOutstanding", "ShareOutstanding", "ShareCount",
        "BasicShares", "Q_NumShares"                          # other possibilities
    ]:
        if name in stock_df.columns:
            shares_latest = _latest_non_nan(stock_df[name])
            if shares_latest:
                break

    def valid(v):
        return v is not None and not (isinstance(v, float) and np.isnan(v))

    def pct(a, b):
        if not valid(a) or not valid(b) or float(b) == 0.0:
            return None
        return float(a) / float(b) * 100.0

    # --- TTM totals ---
    ttm_rev   = ttm_sum(q4, col_rev)
    ttm_gp    = ttm_sum(q4, col_gp)
    ttm_op    = ttm_sum(q4, col_op)
    ttm_np    = ttm_sum(q4, col_np)
    ttm_ebit  = ttm_sum(q4, col_ebit)
    ttm_cfo   = ttm_sum(q4, col_cfo)
    ttm_capex = ttm_sum(q4, col_capex)
    ttm_fcf   = (ttm_cfo - ttm_capex) if (valid(ttm_cfo) and valid(ttm_capex)) else None
    ttm_fin   = ttm_sum(q4, col_fin)
    ttm_tax   = ttm_sum(q4, col_tax)

    # EBITDA fallback = EBIT + Dep/Amort
    if not valid(ttm_ebit) and valid(ttm_op) and col_dep:
        if col_dep in q4.columns:
            ttm_dep = ttm_sum(q4, col_dep)
        else:
            ttm_dep = _to_num(stock_df[col_dep]).dropna().tail(1).sum() if col_dep in stock_df.columns else None
        if valid(ttm_dep):
            ttm_ebit = float(ttm_op) + float(ttm_dep)

    # ---- (2.5) Stronger EBITDA fallback: NP + Finance + Tax + Dep/Amort ----
    if (ttm_ebit is None or float(ttm_ebit) == 0.0) and (ttm_np is not None):
        add_parts = 0.0
        used_any  = False
        # Dep/Amort from quarterly or whole table
        ttm_dep = None
        if col_dep:
            if col_dep in q4.columns:
                ttm_dep = ttm_sum(q4, col_dep)
            elif col_dep in stock_df.columns:
                ttm_dep = _to_num(stock_df[col_dep]).dropna().tail(4).sum() or None
        if ttm_dep:
            add_parts += float(ttm_dep); used_any = True
        if ttm_fin:
            add_parts += float(ttm_fin); used_any = True
        if ttm_tax:
            add_parts += float(ttm_tax); used_any = True
        if used_any:
            ttm_ebit = float(ttm_np) + add_parts

    # ---- (2.3) EPS TTM: prefer quarterly EPS; else NetProfit / latest shares ----
    eps_ttm = ttm_sum(q4, col_eps)  # may be None or 0.0
    if (eps_ttm is None or float(eps_ttm) == 0.0) and (ttm_np is not None) and shares_latest:
        eps_ttm = float(ttm_np) / float(shares_latest)

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

    # ---- (2.4) Valuation: Market cap from column or CURRENT price × LATEST shares ----
    mc = None
    if "MarketCap" in stock_df.columns:
        s_mc = _to_num(stock_df["MarketCap"]).dropna()
        if not s_mc.empty:
            mc = float(s_mc.iloc[-1])

    if mc is None and (current_price is not None) and shares_latest:
        mc = float(current_price) * float(shares_latest)
        
    # P/B (TTM): MarketCap ÷ latest total equity (book value)
    equity_latest = _latest_non_nan(stock_df[col_equity]) if col_equity else None
    if (mc is not None) and (equity_latest is not None) and float(equity_latest) != 0.0:
        out["P/B (TTM)"] = float(mc) / float(equity_latest)
    

    # Expose primitives INSIDE the function (for diagnostics / Snowflake detail)
    if current_price is not None:
        out["Current Price"] = float(current_price)
    if shares_latest is not None:
        out["Shares"] = float(shares_latest)
    if mc is not None:
        out["MarketCap"] = float(mc)

    # Multiples
    if (current_price is not None) and (eps_ttm is not None) and float(eps_ttm) != 0.0:
        out["P/E (TTM)"] = float(current_price) / float(eps_ttm)
    if (mc is not None) and (ttm_rev is not None) and float(ttm_rev) != 0.0:
        out["P/S (TTM)"] = float(mc) / float(ttm_rev)

    # EV/EBITDA pieces
    debt = _to_num(stock_df[col_debt]).dropna().iloc[-1] if col_debt and col_debt in stock_df.columns and not _to_num(stock_df[col_debt]).dropna().empty else None
    cash = _to_num(stock_df[col_cash]).dropna().iloc[-1] if col_cash and col_cash in stock_df.columns and not _to_num(stock_df[col_cash]).dropna().empty else None
    if (mc is not None) and (ttm_ebit is not None) and float(ttm_ebit) != 0.0:
        net_debt = (float(debt) - float(cash)) if (debt is not None and cash is not None) else (float(debt) if debt is not None else 0.0)
        out["EV/EBITDA (TTM)"] = (float(mc) + net_debt) / float(ttm_ebit)

    # Balance-sheet strength & cash-flow derived metrics
    cash_latest = _latest_non_nan(stock_df[col_cash]) if col_cash else None
    debt_latest = _latest_non_nan(stock_df[col_debt]) if col_debt else None
    net_cash = None
    if (cash_latest is not None) or (debt_latest is not None):
        net_cash = (cash_latest or 0.0) - (debt_latest or 0.0)

    fcf_ps         = (ttm_fcf / shares_latest) if (ttm_fcf is not None and shares_latest) else None
    fcf_yield_pct  = (ttm_fcf / mc * 100.0) if (ttm_fcf is not None and mc) else None
    cash_conv_pct  = (float(ttm_cfo) / float(ttm_np) * 100.0) if (ttm_cfo and ttm_np and float(ttm_np) != 0.0) else None
    int_cov        = (float(ttm_ebit) / abs(float(ttm_fin))) if (ttm_ebit and ttm_fin and float(ttm_fin) != 0.0) else None
    debt_fcf_yrs   = (float(debt_latest) / float(ttm_fcf)) if (debt_latest and ttm_fcf and float(ttm_fcf) > 0.0) else None

    out["Net Cash (Debt)"]          = net_cash
    out["Net Cash / MC (%)"]        = (net_cash / mc * 100.0) if (net_cash is not None and mc) else None
    out["FCF / Share (TTM)"]        = fcf_ps
    out["FCF Yield (TTM) (%)"]      = fcf_yield_pct
    out["Cash Conversion (CFO/NP, %)"] = cash_conv_pct
    out["Interest Coverage (EBITDA/Fin)"] = int_cov
    out["Debt / FCF (yrs)"]         = debt_fcf_yrs

    return out


# --- helper: collect one debug row ------------------------------------------
def _push(details, pillar, label, raw, score, source, components=None):
    details.append({
        "Pillar": pillar,
        "Input":  label,
        "Raw":    None if raw is None else float(raw),
        "Score":  None if score is None else int(score),
        "Source": source,
        "Components": components or {},   # ← NEW
    })


def _score_linear(x, lo, hi, reverse=False):
    """Map x to 0..100 between [lo, hi]. If reverse=True, smaller is better."""
    try:
        if x is None:
            return None
        if reverse:
            # invert axis: high is bad
            x, lo, hi = -x, -lo, -hi
        if hi == lo:
            return None
        frac = (x - lo) / (hi - lo)
        return int(max(0, min(100, round(frac * 100))))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
def compute_factor_scores(stock_name, stock_df, ttm,
                          ohlc_latest=None, industry=None):
    """
    0–100 scores for 5 pillars + per-pillar confidence (0..1) and overall confidence.
    Confidence = how much usable raw signal exists for that pillar.
    """
    scores  = {k: 0 for k in ["Value", "Quality", "Growth", "Cash", "Momentum"]}
    details = []
    conf    = {k: 0.0 for k in ["Value", "Quality", "Growth", "Cash", "Momentum"]}

    # ---------- helpers ----------
    def _present_count(*vals):
        """How many inputs are usable (not None/NaN)."""
        c = 0
        for v in vals:
            if v is None: 
                continue
            if isinstance(v, float):
                if np.isnan(v): 
                    continue
            c += 1
        return c

    # ────────── VALUE ──────────
    pe_ttm = ttm.get("P/E (TTM)")
    ey     = (1.0 / pe_ttm) if (pe_ttm and pe_ttm > 0) else None
    fy_pct = ttm.get("FCF Yield (TTM) (%)")
    fy     = (fy_pct / 100.0) if fy_pct is not None else None

    v1 = _score_linear(ey, 0.03, 0.10) if ey is not None else None
    v2 = _score_linear(fy, 0.02, 0.08) if fy is not None else None

    _push(details, "Value", "Earnings-Yield", ey, v1,
          "NetProfit_TTM ÷ (Price×Shares)",
          {"NetProfit_TTM": ttm.get("TTM Net Profit"),
           "Price": ttm.get("Current Price"),
           "Shares": ttm.get("Shares")})
    _push(details, "Value", "FCF-Yield", fy, v2,
          "FCF_TTM ÷ MarketCap",
          {"FCF_TTM": ttm.get("TTM FCF"),
           "MarketCap": ttm.get("MarketCap")})

    parts = [v for v in (v1, v2) if v is not None]
    scores["Value"] = int(sum(parts)/len(parts)) if parts else 0
    conf["Value"]   = _present_count(ey, fy) / 2.0

    # ────────── QUALITY ────────
    cc = (ttm.get("Cash Conversion (CFO/NP, %)") or 0) / 100.0 if ttm.get("Cash Conversion (CFO/NP, %)") is not None else None
    gm_pct = ttm.get("TTM Gross Margin (%)")
    gm = (gm_pct / 100.0) if gm_pct is not None else None
    if gm is None and ttm.get("TTM Revenue") and ttm.get("TTM Gross Profit"):
        try:
            gm = float(ttm["TTM Gross Profit"]) / float(ttm["TTM Revenue"])
        except Exception:
            gm = None

    q1 = _score_linear(cc, 0.6, 1.3)   if cc is not None else None
    q2 = _score_linear(gm, 0.15, 0.40) if gm is not None else None

    _push(details, "Quality", "Cash-Conversion", cc, q1,
          "CFO_TTM ÷ NetProfit_TTM",
          {"CFO_TTM": ttm.get("TTM CFO"),
           "NetProfit_TTM": ttm.get("TTM Net Profit")})
    _push(details, "Quality", "Gross-Margin", gm, q2,
          "(Revenue–COGS) ÷ Revenue",
          {"GrossProfit_TTM": ttm.get("TTM Gross Profit"),
           "Revenue_TTM": ttm.get("TTM Revenue")})

    parts = [q for q in (q1, q2) if q is not None]
    scores["Quality"] = int(sum(parts)/len(parts)) if parts else 0
    conf["Quality"]   = _present_count(cc, gm) / 2.0

    # ────────── GROWTH ─────────
    # Use last 12 quarters. Revenue YoY% positive frequency + EPS YoY% positive frequency.
    def _to_qnum(q):
        if pd.isna(q): return np.nan
        try:
            qi = int(q);  return qi if qi in (1,2,3,4) else np.nan
        except Exception:
            s = str(q).strip().upper().replace("QUARTER", "Q").replace(" ", "")
            if s.startswith("Q") and len(s) >= 2 and s[1].isdigit():  return int(s[1])
            if s.endswith("Q") and s[0].isdigit():                    return int(s[0])
            return np.nan

    q = stock_df[stock_df.get("IsQuarter", False) == True].copy()

    # ─── Ensure we have a quarter-level EPS series for YoY ───
    if "Q_EPS" not in q.columns and "Q_NetProfit" in q.columns and "Q_NumShares" in q.columns:
        q["Q_EPS"] = pd.to_numeric(q["Q_NetProfit"], errors="coerce") \
                   / pd.to_numeric(q["Q_NumShares"], errors="coerce")

    g1 = g2 = None
    eps_pairs_found = 0
    if not q.empty and {"Year","Quarter"}.issubset(q.columns):
        q = q.assign(_Y=pd.to_numeric(q["Year"], errors="coerce"),
                     _Q=q["Quarter"].apply(_to_qnum))
        q = q.dropna(subset=["_Y","_Q"]).sort_values(["_Y","_Q"]).tail(12)

        def yoy_pos_frac(series_name):
            if series_name not in q.columns: return (None, 0, 0)
            v = pd.to_numeric(q[series_name], errors="coerce").values
            pos = tot = 0
            for i in range(len(v)):
                j = i - 4
                if j >= 0 and pd.notna(v[j]) and v[j] != 0 and pd.notna(v[i]):
                    growth = (v[i] - v[j]) / abs(v[j])
                    if pd.notna(growth):
                        tot += 1
                        if growth > 0: pos += 1
            return ((pos / tot) if tot else None, pos, tot)

        # Revenue YoY+
        r_frac, _, r_tot = yoy_pos_frac("Q_Revenue")
        if r_frac is None:
            r_frac, _, r_tot = yoy_pos_frac("Revenue")
        g1 = _score_linear(r_frac, 0.40, 0.90) if r_frac is not None else None

        # EPS YoY+ (strict: take exactly the latest 8 quarter-vs-q-4 pairs)
        eps_col = "Q_EPS" if "Q_EPS" in q.columns else ("EPS" if "EPS" in q.columns else None)
        e_frac = None
        if eps_col:
            e_frac, e_pos, e_tot = yoy_pos_frac(eps_col)
            # clamp to exactly 8 pairs (the most-recent two years)
            eps_pairs_found = min(e_tot, 8)
            # only score if we have a full 8-pair window
            if eps_pairs_found == 8 and e_frac is not None:
                g2 = _score_linear(e_frac, 0.40, 0.90)


        # record both sub-metrics to the detail table
        _push(details, "Growth", "Rev YoY % positive rate", r_frac, g1,
              "Quarterly Q_Revenue vs q-4", {"pairs_found": r_tot})
        _push(details, "Growth", "EPS YoY % positive rate", e_frac, g2,
              "Quarterly Q_EPS vs q-4", {"pairs_found": eps_pairs_found})

    parts = [p for p in (g1, g2) if p is not None]
    scores["Growth"] = int(sum(parts)/len(parts)) if parts else 0
    # confidence: use what we had; EPS capped to 8 pairs
    eps_conf = min(1.0, eps_pairs_found/8.0) if eps_pairs_found else 0.0
    rev_conf = 1.0 if g1 is not None else 0.0
    conf["Growth"] = (rev_conf + eps_conf) / 2.0

    # ────────── CASH ───────────
    nd_ebitda = None
    try:
        net_cash = ttm.get("Net Cash (Debt)")   # +ve means net cash
        if net_cash is not None:
            net_debt = -float(net_cash)         # convert to net *debt*
        else:
            net_debt = None
        ebitda = ttm.get("TTM EBITDA")
        if (net_debt is not None) and ebitda:
            nd_ebitda = float(net_debt) / float(ebitda)
    except Exception:
        nd_ebitda = None

    icov     = ttm.get("Interest Coverage (EBITDA/Fin)")
    debt_fcf = ttm.get("Debt / FCF (yrs)")

    c1 = _score_linear(nd_ebitda, 4.0, 0.0, reverse=True) if nd_ebitda is not None else None
    c2 = _score_linear(min(icov, 20) if icov is not None else None, 2.0, 12.0) if icov is not None else None
    c3 = _score_linear(debt_fcf, 8.0, 0.0, reverse=True)              if debt_fcf is not None else None

    _push(details, "Cash", "Net-Debt / EBITDA", nd_ebitda, c1,
          "(Debt – Cash) ÷ EBITDA_TTM",
          {"NetDebt": None if ttm.get('Net Cash (Debt)') is None else -float(ttm.get('Net Cash (Debt)')),
           "EBITDA_TTM": ttm.get("TTM EBITDA")})
    _push(details, "Cash", "Coverage", icov, c2,
          "EBITDA_TTM ÷ FinanceCost_TTM",
          {"EBITDA_TTM": ttm.get("TTM EBITDA"),
           "FinanceCost_TTM": None})  # finance cost total is implicit in coverage
    _push(details, "Cash", "Debt / FCF (yrs)", debt_fcf, c3,
          "(Debt – Cash) ÷ FCF_TTM",
          {"Debt": None, "Cash": None, "FCF_TTM": ttm.get("TTM FCF")})

    parts = [c for c in (c1, c2, c3) if c is not None]
    scores["Cash"] = int(sum(parts)/len(parts)) if parts else 0
    conf["Cash"]   = _present_count(nd_ebitda, icov, debt_fcf) / 3.0

    # ────────── MOMENTUM ────────
    # Map –100%…+100% → 0…100 on returns, plus 200-DMA flag, weighted 70/30
    mom = 0
    mom_conf = 0.0
    try:
        if isinstance(ohlc_latest, dict):
            price = ohlc_latest.get("price")
            ma200 = ohlc_latest.get("ma200")
            ret12 = ohlc_latest.get("ret_12m")

            # 12-mo return: stretch from –100%→0 up to +100%→100
            m1 = _score_linear(ret12, -1.0, 1.0) if ret12 is not None else None

            # 200-DMA flag: full 100 if above, 0 if below, None if data missing
            if price is not None and ma200 is not None:
                m2 = 100 if price >= ma200 else 0
            else:
                m2 = None

            # push diagnostics
            _push(details, "Momentum", "12-month return", ret12, m1,
                  "(Price ÷ Price_-252d) – 1",
                  {"price": price, "ret_12m": ret12})
            _push(details, "Momentum", "200-DMA flag",
                  (1.0 if (price is not None and ma200 is not None and price >= ma200) else 0.0) if m2 is not None else None,
                  m2, "Price ≥ 200-DMA",
                  {"price": price, "ma200": ma200})

            # combine
            parts = [v for v in (m1, m2) if v is not None]
            if m1 is not None and m2 is not None:
                mom = int(round(0.70 * m1 + 0.30 * m2))
            elif parts:
                mom = int(round(sum(parts) / len(parts)))
            else:
                mom = 0

            # confidence = sum of weights for inputs we actually had
            mom_conf = (0.70 if m1 is not None else 0.0) + (0.30 if m2 is not None else 0.0)
        else:
            mom, mom_conf = 0, 0.0

    except Exception:
        mom, mom_conf = 0, 0.0

    scores["Momentum"] = mom
    conf["Momentum"]   = mom_conf





    # ---------- pack ----------
    scores["_detail"]        = details
    # overall confidence = mean of available pillar confidences
    used = [conf[k] for k in conf]
    scores["_confidence"]    = conf
    scores["_overall_conf"]  = float(sum(used)) / float(len(used)) if used else 0.0
    return scores

# ─────────────────────────────────────────────────────────────────────────────
# Industry-aware scoring (v1) — cashflow-first, TTM vs LFY, 5y growth, entry valuation, dividend
# ─────────────────────────────────────────────────────────────────────────────
def _lin(x, lo, hi):
    if x is None or any(map(lambda v: v is None, (lo, hi))): return None
    if hi == lo: return 50.0
    v = max(min((x - lo) / (hi - lo), 1.0), 0.0)
    return float(100.0 * v)

def _rev(x, lo, hi):
    s = _lin(x, lo, hi)
    return None if s is None else float(100.0 - s)

def _avg_present(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return (0.0, 0.0)
    return (float(sum(vals))/len(vals), float(len(vals))/float(len(vals)))

def _winsor(x, cap):
    if x is None: return None
    try:
        return float(np.sign(x) * min(abs(float(x)), float(cap)))
    except Exception:
        return None

def _cagr(v0, v1, years):
    try:
        if v0 is None or v1 is None or v0 <= 0 or years <= 0: return None
        return (v1 / v0) ** (1.0 / years) - 1.0
    except Exception:
        return None

def _ppct(a, b):
    if a is None or b is None: return None
    if b == 0: return None
    return (a - b) / abs(b)

def _pp(a, b):  # percentage-point change for margins (already in 0..1)
    if a is None or b is None: return None
    return (a - b)

def _last_fy_five(annual):
    try:
        a = annual[annual["IsQuarter"] != True].sort_values("Year").copy()
        if a.empty: return a, None
        last5 = a.tail(5).copy()
        lfy = last5.iloc[-1] if not last5.empty else None
        return last5, lfy
    except Exception:
        return annual.head(0), None

def compute_industry_scores(stock_name: str,
                            stock_df: pd.DataFrame,
                            bucket: str = "General",
                            entry_price: float | None = None,
                            fd_rate: float = 0.035,
                            ohlc_latest: dict | None = None,   # ← NEW
                            cfg=None):

    """
    Returns a dict:
    {
      "bucket": str,
      "gates": {"data_ok": bool, "cashflow_ok": bool, "notes": [str]},
      "blocks": {...},  # scores per block + detail
      "composite": float,
      "decision": "PASS"|"HOLD"|"REJECT",
      "why": [str],
    }
    """
    # late import to avoid circular when this file is imported from multiple places
    try:
        import config as _cfg
    except Exception:
        from utils import config as _cfg   # type: ignore
    cfg = _cfg if cfg is None else cfg

    weights = cfg.BLOCK_WEIGHTS
    profiles = cfg.BUCKET_PROFILES
    caps = cfg.CAPS
    prof = profiles.get(bucket, profiles["General"])

    notes = []
    gates = {"data_ok": True, "cashflow_ok": True, "notes": notes}

    # ---------- Annual last 5y ----------
    a5, lfy = _last_fy_five(stock_df)
    if a5.empty or lfy is None:
        gates["data_ok"] = False
        notes.append("Missing annual data (need ≥1 LFY, ideally 5y).")

    # Get annual series
    def _col(s, k):
        return pd.to_numeric(s.get(k, pd.Series(dtype="float")), errors="coerce")

    CFO   = _col(a5, "CFO")
    CapEx = _col(a5, "CapEx")
    FCF   = (CFO.fillna(0) - CapEx.fillna(0)) if not CFO.empty else pd.Series(dtype="float")
    NP    = _col(a5, "NetProfit")
    Rev   = _col(a5, "Revenue")
    Eq    = _col(a5, "ShareholderEquity")
    Year  = a5.get("Year", pd.Series(dtype="Int64"))

    years_present = Year.dropna().shape[0]
    if years_present < 3:
        gates["data_ok"] = False
        notes.append("Fewer than 3 of last 5 FY available.")
        
    # ---------- Alerts (Sense layer) ----------
    alerts = []
    try:
        # NPM drops ≥ 3pp YoY (over last 5y)
        npm = (NP / Rev * 100.0).round(1)
        for i in range(1, len(npm)):
            if pd.notna(npm.iloc[i]) and pd.notna(npm.iloc[i-1]):
                d = float(npm.iloc[i] - npm.iloc[i-1])
                if d <= -3.0:
                    yr = int(Year.iloc[i]) if not Year.dropna().empty else None
                    alerts.append({"severity": "warn", "message": f"NPM drop {d:.1f}pp" + (f" in {yr}" if yr else "")})
                    break

        # Spikes ≥ 100% YoY in NP or Revenue (first one we see)
        def _spike(series, label):
            s = series.dropna()
            for j in range(1, len(s)):
                if s.iloc[j-1] != 0:
                    g = (s.iloc[j] - s.iloc[j-1]) / abs(s.iloc[j-1])
                    if abs(g) >= 1.0:
                        alerts.append({"severity": "info", "message": f"{label} jump {g*100:.0f}% YoY"})
                        return
        _spike(NP,  "Net Profit")
        _spike(Rev, "Revenue")

        # Coverage low (latest FY): CFO / NP < 0.6
        if not NP.dropna().empty and not CFO.dropna().empty:
            np_last  = float(NP.dropna().iloc[-1])
            cfo_last = float(CFO.dropna().iloc[-1])
            if np_last != 0 and (cfo_last / np_last) < 0.6:
                alerts.append({"severity": "warn", "message": "Low coverage: CFO/NP < 0.6 (latest FY)"})
    except Exception:
        pass
    

    # ---------- TTM ----------
    cur = entry_price
    if cur is None:
        # fall back to latest stored price in the rows
        try:
            pr = pd.to_numeric(stock_df.get("CurrentPrice", pd.Series([np.nan]))).dropna()
            if not pr.empty: cur = float(pr.iloc[-1])
        except Exception:
            pass
    ttm = compute_ttm(stock_df, current_price=cur) or {}

    # ---------- A) Cash-flow first ----------
    fcf_pos = int((FCF > 0).sum()) if not FCF.empty else 0
    cfo_pos = int((CFO > 0).sum()) if not CFO.empty else 0
    cum_cfo = float(CFO.fillna(0).sum()) if not CFO.empty else 0.0
    cum_np  = float(NP.fillna(0).sum()) if not NP.empty else 0.0
    cfo_np_ratio = (cum_cfo / cum_np) if cum_np not in (0.0, None) else None

    s_fcf = _lin(fcf_pos, 0, 5)
    s_cfo = _lin(cfo_pos, 0, 5)
    s_rat = _lin(cfo_np_ratio, 0.8, 1.2) if cfo_np_ratio is not None else None
    sA, confA = _avg_present([s_fcf, s_cfo, s_rat])

    if fcf_pos < 3 and cfo_pos < 3:
        gates["cashflow_ok"] = False
        notes.append("Cash-flow gate fail: <3 positive years of both CFO and FCF.")
        
    # detail for Cash-flow block
    years_last5 = Year.dropna().astype(int).tolist() if not Year.dropna().empty else []
    detailA = {
        "years": years_last5,
        "CFO_pos_years": int(cfo_pos),
        "FCF_pos_years": int(fcf_pos),
        "cum_CFO": float(cum_cfo) if cum_cfo is not None else None,
        "cum_NP": float(cum_np) if cum_np is not None else None,
        "cum_CFO/NP": float(cfo_np_ratio) if cfo_np_ratio is not None else None,
        "sub_scores": {
            "CFO_pos": float(s_cfo) if s_cfo is not None else None,
            "FCF_pos": float(s_fcf) if s_fcf is not None else None,
            "cum_ratio": float(s_rat) if s_rat is not None else None,
        },
    }
    

    # ---------- B) TTM vs LFY ----------
    # margins as 0..1
    ttm_rev = ttm.get("TTM Revenue")
    ttm_gm  = (ttm.get("TTM Gross Margin (%)") or 0)/100.0 if ttm.get("TTM Gross Margin (%)") is not None else None
    ttm_nm  = (ttm.get("TTM Net Margin (%)") or 0)/100.0   if ttm.get("TTM Net Margin (%)") is not None else None

    lfy_rev = float(lfy["Revenue"]) if lfy is not None and pd.notna(lfy.get("Revenue")) else None
    lfy_gm  = (float(lfy["GrossProfit"])/float(lfy["Revenue"])) if lfy is not None and \
              pd.notna(lfy.get("GrossProfit")) and pd.notna(lfy.get("Revenue")) and float(lfy["Revenue"]) != 0 else None
    lfy_nm  = (float(lfy["NetProfit"])/float(lfy["Revenue"])) if lfy is not None and \
              pd.notna(lfy.get("NetProfit")) and pd.notna(lfy.get("Revenue")) and float(lfy["Revenue"]) != 0 else None

    s_rev = _lin(_ppct(ttm_rev, lfy_rev), -0.05, 0.10) if (ttm_rev is not None and lfy_rev is not None) else None
    s_gm  = _lin(_pp(ttm_gm, lfy_gm),   -0.02, 0.04)    if (ttm_gm  is not None and lfy_gm  is not None) else None
    s_nm  = _lin(_pp(ttm_nm, lfy_nm),   -0.01, 0.03)    if (ttm_nm  is not None and lfy_nm  is not None) else None
    sB, confB = _avg_present([s_rev, s_gm, s_nm])
    
    detailB = {
    "TTM": {
        "Revenue": ttm_rev,
        "GrossMargin": ttm_gm,
        "NetMargin": ttm_nm,
    },
    "LFY": {
        "Revenue": lfy_rev,
        "GrossMargin": lfy_gm,
        "NetMargin": lfy_nm,
    },
    "delta": {
        "rev_pct": _ppct(ttm_rev, lfy_rev),
        "gm_pp": _pp(ttm_gm, lfy_gm),
        "nm_pp": _pp(ttm_nm, lfy_nm),
    },
    "sub_scores": {
        "rev": float(s_rev) if s_rev is not None else None,
        "gm":  float(s_gm)  if s_gm  is not None else None,
        "nm":  float(s_nm)  if s_nm  is not None else None,
    },
}


    # ---------- C) 5y growth & quality ----------
    rev0, rev1 = (float(Rev.dropna().iloc[0]) if not Rev.dropna().empty else None,
                  float(Rev.dropna().iloc[-1]) if not Rev.dropna().empty else None)
    yrs = max(1, int((Year.dropna().iloc[-1] - Year.dropna().iloc[0])) ) if years_present >= 2 else 1
    rev_cagr = _cagr(rev0, rev1, yrs)
    # ROE median & stability if possible
    roe_series = None
    if not NP.dropna().empty and not Eq.dropna().empty:
        try:
            # align by index
            roe_series = (NP / Eq).replace([np.inf, -np.inf], np.nan).dropna()
        except Exception:
            roe_series = None
    roe_med = float(roe_series.median()) if roe_series is not None and not roe_series.empty else None
    roe_std = float(roe_series.std()) if roe_series is not None and roe_series.shape[0] >= 2 else None

    s_cagr = _lin(rev_cagr, 0.00, 0.10) if rev_cagr is not None else None
    s_roem = _lin(roe_med, 0.08, 0.18)  if roe_med  is not None else None
    s_roes = _rev(roe_std, 0.08, 0.02)  if roe_std  is not None else None
    sC, confC = _avg_present([s_cagr, s_roem, s_roes])
    
    detailC = {
    "Revenue": {
        "v0": rev0, "v1": rev1, "years": yrs, "CAGR_5y": rev_cagr,
    },
    "ROE": {
        "median_5y": roe_med,
        "stdev_pp": roe_std,
    },
    "sub_scores": {
        "rev_cagr": float(s_cagr) if s_cagr is not None else None,
        "roe_median": float(s_roem) if s_roem is not None else None,
        "roe_stability": float(s_roes) if s_roes is not None else None,
    },
}


    # ---------- D) Valuation @ entry (bucket-specific) ----------
    entry = entry_price if entry_price is not None else ttm.get("Current Price")
    eps_ttm = ttm.get("TTM EPS")
    pe_entry = (entry / eps_ttm) if (entry not in (None, 0) and eps_ttm and eps_ttm > 0) else None

    mc = ttm.get("MarketCap")
    fcf_ttm = ttm.get("TTM FCF")
    fcf_y = (fcf_ttm / mc) if (fcf_ttm not in (None, 0) and mc and mc > 0) else None

    ebitda = ttm.get("TTM EBITDA")
    fin    = ttm.get("TTM Finance Costs")
    cov    = (ebitda / fin) if (ebitda and fin and fin > 0) else None

    ev_eb = ttm.get("EV/EBITDA (TTM)")

    # winsorize
    pe_entry = _winsor(pe_entry, cfg.CAPS["pe"]) if pe_entry is not None else None
    fcf_y    = _winsor(fcf_y,    cfg.CAPS["yield"]) if fcf_y is not None else None
    cov      = _winsor(cov,      cfg.CAPS["coverage"]) if cov is not None else None
    ev_eb    = _winsor(ev_eb,    cfg.CAPS["ev_ebitda"]) if ev_eb is not None else None

    subs = []
    # which metrics to use at entry
    for m in prof.get("entry", []):
        if m == "pe" and pe_entry is not None:
            lo, hi, rev = prof["value"]["pe"]
            subs.append(_rev(pe_entry, lo, hi) if rev else _lin(pe_entry, lo, hi))
        if m == "pb" and stock_df.get("P/B") is not None:
            try:
                pb = float(pd.to_numeric(stock_df["P/B"].dropna()).iloc[-1])
                pb = _winsor(pb, cfg.CAPS["pb"])
                lo, hi, rev = prof["value"]["pb"]
                subs.append(_rev(pb, lo, hi) if rev else _lin(pb, lo, hi))
            except Exception:
                pass
        if m == "fcf_yield" and fcf_y is not None:
            lo, hi = prof["value"]["fcf_yield"]
            subs.append(_lin(fcf_y, lo, hi))
        if m == "yield":
            # annual dividend / entry price (rough)
            try:
                div = float(pd.to_numeric(stock_df.get("Dividend", pd.Series([0.0]))).dropna().iloc[-1])
                yld = (div / entry) if entry else None
                if yld is not None:
                    lo, hi = prof["value"]["yield"]
                    subs.append(_lin(yld, lo, hi))
            except Exception:
                pass
        if m == "ev_ebitda" and ev_eb is not None:
            lo, hi, rev = prof["value"]["ev_ebitda"]
            subs.append(_rev(ev_eb, lo, hi) if rev else _lin(ev_eb, lo, hi))

    sD, confD = _avg_present(subs)
    # label
    label = "Fair"
    if sD >= 70: label = "Undervalued"
    elif sD < 40: label = "Overvalued"
    
    # try fetch PB latest (used in detail table even if not scored)
    pb_latest = None
    if "P/B" in stock_df.columns:
        try:
            pb_latest = float(pd.to_numeric(stock_df["P/B"], errors="coerce").dropna().iloc[-1])
        except Exception:
            pb_latest = None

    # last annual dividend (for yield-at-entry calc already used above)
    last_div_annual = None
    try:
        sdiv = pd.to_numeric(stock_df.get("Dividend", pd.Series(dtype="float")), errors="coerce").dropna()
        if not sdiv.empty:
            last_div_annual = float(sdiv.iloc[-1])
    except Exception:
        pass

    detailD = {
        "entry_price": entry,
        "metrics": {
            "EPS_TTM": eps_ttm,
            "PE_entry": pe_entry,
            "MarketCap": mc,
            "FCF_TTM": fcf_ttm,
            "FCF_Yield": fcf_y,
            "EV/EBITDA_TTM": ev_eb,
            "PB_latest": pb_latest,
            "Dividend_last_annual": last_div_annual,
        },
        "used_entry_metrics": list(prof.get("entry", [])),
        "bucket_ranges": prof.get("value", {}),
        "label": label,
    }


    # ---------- E) Dividend (simple v1) ----------
    # streak: count last consecutive years with >0 dividend
    div_s = pd.to_numeric(a5.get("Dividend", pd.Series(dtype="float")), errors="coerce").fillna(0.0)
    streak = 0
    for v in reversed(div_s.tolist()):
        if v and v > 0: streak += 1
        else: break
    s_streak = _lin(streak, 0, 5)
    # payout sweet spot 30–70% if we can estimate
    payout = None
    try:
        last_div = float(div_s.iloc[-1]) if not div_s.empty else None
        last_np = float(pd.to_numeric(a5["NetProfit"]).dropna().iloc[-1]) if "NetProfit" in a5.columns else None
        if last_div is not None and last_np and last_np != 0:
            payout = last_div / last_np
    except Exception:
        payout = None
    s_pay = None
    if payout is not None:
        # bell-ish via distance to center 0.5 within [0.3,0.7]
        lo, hi = 0.30, 0.70
        if payout <= lo: s_pay = _lin(payout, 0.0, lo) * 0.5
        elif payout >= hi: s_pay = _rev(payout, hi, 1.5) * 0.5
        else: s_pay = 100.0

    # yield vs FD
    s_fd = None
    try:
        yld = None
        if entry:
            yld = (float(div_s.iloc[-1]) / float(entry)) if not div_s.empty else None
        if yld is not None:
            # map FD -> FD+3pp → 50→100 ; else 0→50 below FD
            if yld >= fd_rate:
                s_fd = _lin(yld, fd_rate, fd_rate + 0.03) * 0.5 + 50.0
                s_fd = min(s_fd, 100.0)
            else:
                s_fd = _lin(yld, 0.0, fd_rate) * 0.5
    except Exception:
        pass
    sE, confE = _avg_present([s_streak, s_pay, s_fd])
    
    detailE = {
    "streak_years": streak,
    "payout": payout,
    "yield_at_entry": (last_div_annual / entry) if (last_div_annual and entry) else None,
    "fd_rate": fd_rate,
    "sub_scores": {
        "streak": float(s_streak) if s_streak is not None else None,
        "payout": float(s_pay) if s_pay is not None else None,
        "yield_vs_fd": float(s_fd) if s_fd is not None else None,
    },
    }


    # ---------- Momentum (optional; needs ohlc_latest) ----------
    mom_score  = 0.0
    mom_conf   = 0.0
    mom_detail = {}
    try:
        if isinstance(ohlc_latest, dict):
            price = ohlc_latest.get("price")
            ma200 = ohlc_latest.get("ma200")
            ret12 = ohlc_latest.get("ret_12m")

            # map −100%…+100% → 0…100 using top-level _lin
            m1 = _lin(ret12, -1.0, 1.0) if ret12 is not None else None
            # 200-DMA flag: 100 if above, 0 if below, None if missing
            m2 = 100.0 if (price is not None and ma200 is not None and price >= ma200) \
                else (0.0 if (price is not None and ma200 is not None) else None)

            parts = [v for v in (m1, m2) if v is not None]
            mom_score = float(sum(parts) / len(parts)) if parts else 0.0
            mom_conf  = (0.70 if m1 is not None else 0.0) + (0.30 if m2 is not None else 0.0)
            mom_detail = {
                "12mReturn": ret12,
                "MA200_Flag": (1.0 if m2 == 100.0 else 0.0) if m2 is not None else None,
                "price": price, "ma200": ma200
            }
    except Exception:
        pass

    # now build the blocks dict (momentum is ready)
    blocks = {
        "cashflow_first": {"score": sA, "conf": confA, "detail": detailA},
        "ttm_vs_lfy":     {"score": sB, "conf": confB, "detail": detailB},
        "growth_quality": {"score": sC, "conf": confC, "detail": detailC},
        "valuation_entry":{"score": sD, "conf": confD, "label": label, "detail": detailD},
        "dividend":       {"score": sE, "conf": confE, "detail": detailE},
        "momentum":       {"score": round(mom_score, 1), "conf": mom_conf, "detail": mom_detail},
    }




    # composite (confidence-weighted)
    num = 0.0
    den = 0.0
    for k, v in blocks.items():
        w = float(weights.get(k, 0))
        if v["conf"] > 0:
            num += w * v["score"] * v["conf"]
            den += w * v["conf"]
    composite = float(num / den) if den > 0 else 0.0

    # decision
    why = []
    if not gates["data_ok"]:      why.append("Data gate failed")
    if not gates["cashflow_ok"]:  why.append("Cash-flow gate failed")

    decision = "REJECT"
    if gates["data_ok"] and gates["cashflow_ok"]:
        if composite >= cfg.MIN_SCORE_INDUSTRY and blocks["valuation_entry"]["score"] >= cfg.MIN_VALUATION_ENTRY:
            decision = "PASS"
        elif composite >= cfg.MIN_SCORE_INDUSTRY - 5 or 40 <= blocks["valuation_entry"]["score"] < cfg.MIN_VALUATION_ENTRY:
            decision = "HOLD"
        else:
            decision = "REJECT"

    return {
    "bucket": bucket,
    "gates": gates,
    "blocks": blocks,
    "composite": round(composite, 1),
    "decision": decision,
    "why": why,
    "alerts": alerts,    # ← NEW
    }


