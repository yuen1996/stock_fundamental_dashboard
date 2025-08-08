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
