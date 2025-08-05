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

    scores  = {k: 0 for k in ["Value", "Quality", "Growth", "Cash", "Momentum"]}
    details = []

    # ── VALUE ────────────────────────────────────────────────
    pe_ttm   = ttm.get("P/E (TTM)")
    ey       = (1/pe_ttm) if pe_ttm else None
    fy       = (ttm.get("FCF Yield (TTM) (%)") or 0)/100

    v1 = _score_linear(ey, 0.03, 0.10)
    v2 = _score_linear(fy, 0.02, 0.08)

    _push(details,"Value","Earnings-Yield",ey,v1,
          "NetProfit_TTM ÷ (Price×Shares)",
          {"NetProfit_TTM": ttm.get("TTM Net Profit"),
           "Price": ttm.get("Current Price"),
           "Shares": ttm.get("Shares")})
    _push(details,"Value","FCF-Yield",fy,v2,
          "FCF_TTM ÷ MarketCap",
          {"FCF_TTM": ttm.get("TTM FCF"),
           "MarketCap": ttm.get("MarketCap")})

    parts = [v for v in (v1,v2) if v is not None]
    scores["Value"] = int(sum(parts)/len(parts)) if parts else 0

    # ── QUALITY ──────────────────────────────────────────────
    cc = (ttm.get("Cash Conversion (CFO/NP, %)") or 0)/100
    gm = (ttm.get("TTM Gross Margin (%)") or 0)/100

    q1 = _score_linear(cc,0.6,1.3)
    q2 = _score_linear(gm,0.15,0.40)

    _push(details,"Quality","Cash-Conversion",cc,q1,
          "CFO_TTM ÷ NetProfit_TTM",
          {"CFO_TTM": ttm.get("TTM CFO"),
           "NetProfit_TTM": ttm.get("TTM Net Profit")})
    _push(details,"Quality","Gross-Margin",gm,q2,
          "(Revenue–COGS) ÷ Revenue",
          {"GrossProfit_TTM": ttm.get("TTM Gross Profit"),
           "Revenue_TTM": ttm.get("TTM Revenue")})

    parts = [q for q in (q1,q2) if q is not None]
    scores["Quality"] = int(sum(parts)/len(parts)) if parts else 0

    # ── GROWTH ───────────────────────────────────────────────
        # ── GROWTH — strict 8-pair EPS YoY using last 12 quarters ─────────────────
    def _to_qnum(q):
        if pd.isna(q): return np.nan
        try:
            qi = int(q)
            return qi if qi in (1,2,3,4) else np.nan
        except Exception:
            s = str(q).strip().upper().replace("QUARTER", "Q").replace(" ", "")
            if s.startswith("Q") and len(s) >= 2 and s[1].isdigit():  return int(s[1])
            if s.endswith("Q") and s[0].isdigit():                    return int(s[0])
            return np.nan

    def _quarterly_df(df):
        q = df[df.get("IsQuarter", False) == True].copy()
        if q.empty: return q
        # normalize Year & Quarter
        q["Year"] = pd.to_numeric(q.get("Year"), errors="coerce").astype("Int64")
        q["Quarter"] = q.get("Quarter")
        q["Quarter"] = q["Quarter"].apply(_to_qnum)
        return q.dropna(subset=["Year","Quarter"])

    def _eps_series(q):
        """Prefer Q_EPS, else EPS, else derive from Q_NetProfit / (Q_NumShares or NumShares)."""
        eps = None
        if "Q_EPS" in q.columns:
            eps = pd.to_numeric(q["Q_EPS"], errors="coerce")
        if (eps is None or eps.isna().all()) and "EPS" in q.columns:
            eps = pd.to_numeric(q["EPS"], errors="coerce")
        if eps is None or eps.isna().all():
            if "Q_NetProfit" in q.columns:
                npq = pd.to_numeric(q["Q_NetProfit"], errors="coerce")
                sh = None
                if "Q_NumShares" in q.columns:
                    sh = pd.to_numeric(q["Q_NumShares"], errors="coerce")
                if (sh is None or sh.isna().all()) and "NumShares" in stock_df.columns:
                    sh = pd.to_numeric(stock_df["NumShares"], errors="coerce")
                if sh is not None:
                    sh = sh.replace(0, np.nan).ffill().bfill()
                    eps = npq / sh
        return eps

    def _build_pairs_same_quarter(q, value_series, need_pairs=8):
        """
        Use only the last 12 quarterly observations (by Year,Quarter).
        Pair (Year,Quarter) with (Year-1,Quarter). Return list of +1/0 for YoY>0.
        """
        qq = q.copy()
        # sort and de-dup
        qq = qq.sort_values(["Year","Quarter"]).drop_duplicates(["Year","Quarter"], keep="last")
        # keep last 12 quarters
        qq = qq.tail(12)
        # index for lookup
        vals = pd.to_numeric(value_series.reindex(qq.index), errors="coerce")
        idx = {(int(r.Year), int(r.Quarter)): float(vals.loc[i]) for i, r in qq.iterrows()}

        pairs = []
        for y, qnum in sorted(idx.keys()):
            prev = (y-1, qnum)
            if prev in idx:
                base = idx[prev]
                cur  = idx[(y, qnum)]
                if base is not None and base != 0 and pd.notna(base) and pd.notna(cur):
                    g = (cur - base) / abs(base)
                    pairs.append(1 if (pd.notna(g) and g > 0) else 0)

        # keep most recent 8
        if len(pairs) > need_pairs:
            pairs = pairs[-need_pairs:]
        return pairs

    q = _quarterly_df(stock_df)

    # Revenue YoY (strict same-quarter pairing over last 12 quarters)
    r_pos = None
    if not q.empty and "Q_Revenue" in q.columns:
        r_pairs = _build_pairs_same_quarter(q, pd.to_numeric(q["Q_Revenue"], errors="coerce"), need_pairs=8)
        if len(r_pairs) == 8:
            r_pos = sum(r_pairs) / 8.0

    # EPS YoY (strict 8-pair rule; refuse to score if < 8)
    e_pos = None
    e_pairs_found = 0
    if not q.empty:
        eps_series = _eps_series(q)
        if eps_series is not None and not eps_series.isna().all():
            e_pairs = _build_pairs_same_quarter(q, eps_series, need_pairs=8)
            e_pairs_found = len(e_pairs)
            if e_pairs_found == 8:
                e_pos = sum(e_pairs) / 8.0

    g1 = _score_linear(r_pos, 0.40, 0.90) if r_pos is not None else None
    g2 = _score_linear(e_pos, 0.40, 0.90) if e_pos is not None else None

    _push(details, "Growth", "Rev YoY % (8 pairs)", r_pos, g1,
          "Q_Revenue: last 12 quarters → 8 same-quarter YoY checks", {})

    if e_pos is None:
        # log clear reason when we don't hit 8/8
        details.append({
            "Pillar": "Growth", "Input": "EPS YoY % (8 pairs, strict)",
            "Raw": None, "Score": None,
            "Source": "Insufficient EPS YoY pairs (<8) or missing EPS/Quarter/Year",
            "Components": {"pairs_found": int(e_pairs_found), "required_pairs": 8}
        })
    else:
        _push(details, "Growth", "EPS YoY % (8 pairs, strict)", e_pos, g2,
              "Quarterly EPS (Q_EPS or NetProfit/Shares); same quarter vs prior year", {"pairs": 8})

    parts = [p for p in (g1, g2) if p is not None]
    scores["Growth"] = int(sum(parts) / len(parts)) if parts else 0


       # ────────── CASH ───────────
    # Safe Net-Debt / EBITDA
    net_cash = ttm.get("Net Cash (Debt)")          # +ve = cash, -ve = debt
    ebitda   = ttm.get("TTM EBITDA")
    nd_ebitda = None
    if net_cash is not None and ebitda not in (None, 0):
        nd_ebitda = -float(net_cash) / float(ebitda)   # invert so “debt” is +ve

    icov     = ttm.get("Interest Coverage (EBITDA/Fin)")
    debt_fcf = ttm.get("Debt / FCF (yrs)")

    c1 = _score_linear(nd_ebitda, 4.0, 0.0, reverse=True) if nd_ebitda is not None else None
    c2 = _score_linear(min(icov, 20) if icov else None, 2.0, 12.0)        if icov is not None else None
    c3 = _score_linear(debt_fcf, 8.0, 0.0, reverse=True)                  if debt_fcf is not None else None

    _push(details, "Cash", "Net-Debt/EBITDA",
          nd_ebitda, c1, "(Debt – Cash) ÷ EBITDA_TTM")
    _push(details, "Cash", "Interest-Coverage",
          icov, c2, "EBITDA_TTM ÷ FinanceCost_TTM")
    _push(details, "Cash", "Debt-to-FCF yrs",
          debt_fcf, c3, "(Debt – Cash) ÷ FCF_TTM")

    parts = [c for c in (c1, c2, c3) if c is not None]
    scores["Cash"] = (
        int(sum(parts) / len(parts)) if len(parts) >= 2
        else int(parts[0] * 0.7) if parts else 0
    )


    # ── MOMENTUM ────────────────────────────────────────────
    if isinstance(ohlc_latest,dict):
        price,ma200,ret12 = (ohlc_latest.get(k) for k in ("price","ma200","ret_12m"))
        m1=_score_linear(ret12,-0.20,0.40)
        m2=100 if price and ma200 and price>=ma200 else 0
        _push(details,"Momentum","12-m Return",ret12,m1,"(Price÷Price_-252d)–1",{})
        _push(details,"Momentum","200-DMA flag",price>=ma200 if price and ma200 else None,
              m2,"Price ≥ 200-DMA",{})
        parts=[m for m in (m1,m2) if m is not None]
        scores["Momentum"]=int(sum(parts)/len(parts)) if parts else 0
    else:
        scores["Momentum"]=0

    scores["_detail"]=details
    return scores
# ─────────────────────────────────────────────────────────────
