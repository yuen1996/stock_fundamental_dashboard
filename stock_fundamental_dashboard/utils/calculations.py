# utils/calculations.py
from __future__ import annotations

import math
import re
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- robust config import (works from root or utils/) ---
try:
    import config  # type: ignore
except Exception:
    try:
        from utils import config  # type: ignore
    except Exception:
        config = None  # type: ignore

# Core numeric helpers (robust, NaN-safe)
def _to_num(x, *, none_ok: bool = True) -> Optional[float]:
    """Convert to float; return None if not parseable (or NaN)."""
    try:
        if x is None:
            return None if none_ok else 0.0
        if isinstance(x, str):
            x = x.replace(",", "").strip()
            if x == "":
                return None if none_ok else 0.0
        v = float(x)
        if math.isfinite(v):
            return v
        return None if none_ok else 0.0
    except Exception:
        return None if none_ok else 0.0

def _safe_div(a, b) -> Optional[float]:
    a = _to_num(a)
    b = _to_num(b)
    if a is None or b in (None, 0):
        return None
    return a / b

# --- TTM helpers: sum/last with None semantics ---
def _sum_or_none(values):
    """Sum numeric values; return None if ALL inputs are None/blank."""
    nums = []
    for v in (values or []):
        try:
            nv = float(v)
        except Exception:
            nv = None
        if nv is not None and (nv == nv):  # not NaN
            nums.append(nv)
    return sum(nums) if nums else None

def _last_or_none(values):
    """Return the last non-None numeric; else None."""
    for v in reversed(values or []):
        try:
            nv = float(v)
        except Exception:
            nv = None
        if nv is not None and (nv == nv):
            return nv
    return None

def _mag(x):
    v = _to_num(x)
    return abs(v) if v is not None else None

def _pct_mag(num, den):
    """Percent with magnitude-normalized denominator and (usually) numerator."""
    v = _safe_div(num, den)
    return v * 100.0 if v is not None else None

def _avg2(a, b) -> Optional[float]:
    a = _to_num(a)
    b = _to_num(b)
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return (a + b) / 2.0

def _sum(vals: Iterable) -> Optional[float]:
    vs = [_to_num(v) for v in vals]
    vs = [v for v in vs if v is not None and math.isfinite(v)]
    if not vs:
        return None
    return float(np.nansum(vs))

def _last(series_like) -> Optional[float]:
    if series_like is None:
        return None
    try:
        s = pd.to_numeric(series_like, errors="coerce")
        if s.size == 0:
            return None
        v = s.iloc[-1]
        return float(v) if pd.notna(v) else None
    except Exception:
        return None

# ======================================================================
# Column aliasing for raw fields (unified access)
# Keys are our internal canonical names; values are possible CSV headers.
# ======================================================================
RAW_ALIASES: Dict[str, List[str]] = {
    # universal income statement
    "revenue": ["Revenue", "Sales", "Total Revenue"],
    "cogs": ["CostOfSales", "Cost of Sales", "COGS"],
    "gross_profit": ["Gross Profit", "GrossProfit"],
    "operating_profit": ["Operating Profit", "EBIT", "OperatingProfit"],
    "ebitda": ["EBITDA"],
    "net_profit": ["Net Profit", "NetProfit", "Profit"],
    "interest_expense": ["Interest Expense", "Finance Cost", "Finance Costs"],
    "income_tax": ["IncomeTax", "Income Tax Expense", "Tax Expense"],
    "selling_expenses": ["Selling & Distribution Expenses", "Selling Expenses", "Selling & Distribution"],
    "admin_expenses":   ["Administrative Expenses", "Admin Expenses"],

    # balance sheet
    "total_assets": ["Total Assets"],
    "total_liabilities": ["Total Liabilities"],
    "equity": ["Equity", "Equity (Book Value)", "ShareholderEquity", "Shareholders' Equity"],
    "current_assets": ["Current Assets"],
    "current_liabilities": ["Current Liabilities"],
    "receivables": ["Accounts Receivable", "Receivables (Avg)", "Receivables"],
    "inventory": ["Inventory (Avg)", "Inventory"],
    "payables": ["Accounts Payable", "Payables (Avg)", "Payables"],
    "intangibles": ["Intangible Assets"],
    "borrowings": ["Total Borrowings", "Borrowings"],
    "cash": ["Cash & Cash Equivalents", "Cash and Cash Equivalents", "Cash"],

    # securities
    "shares": ["Shares", "Shares Outstanding", "Units", "Units Outstanding", "NumShares", "Number of Shares", "Outstanding Shares"],
    "price": ["Price", "Annual Price per Share (RM)", "SharePrice", "CurrentPrice", "End Price", "EndQuarterPrice", "End Quarter Price"],

    # cash flow
    "cfo": [
    "CFO", "Cash Flow from Ops", "Cash from Operations", "Operating Cash Flow",
    "Net Cash from Operating Activities", "Net cash from operating activities",
    "Cash flows from operating activities"
    ],
    "capex": [
        "Capex", "Capital Expenditure", "Capital Expenditures",
        "Purchase of PPE", "Purchases of property, plant and equipment",
        "Payments for property, plant and equipment",
        "Purchase of Property, Plant and Equipment"
    ],

    "dep_amort": [
        "DepAmort",
        "Depreciation & Amortization (Total)",
        "Depreciation and Amortization",
    ],
    "dep_ppe": ["DepPPE", "Depreciation - PPE"],
    "dep_rou": ["DepROU", "Depreciation - ROU Assets", "ROU Depreciation"],
    "dep_invprop": ["DepInvProp", "Depreciation of Investment Properties"],

    # dividends / distributions
    "dps": ["DPS", "Dividend per Share (TTM, RM)", "DividendPS", "Dividend Per Share"],
    "dpu": ["DPU", "Distribution per Unit (TTM, RM)"],

    # ————— Banking extras —————
    "gross_loans": ["Gross Loans", "Gross Loans / Financing (RM)"],
    "deposits": ["Deposits"],
    "demand_dep": ["Demand Deposits"],
    "savings_dep": ["Savings Deposits"],
    "time_dep": ["Time/Fixed Deposits", "Fixed/Time Deposits"],
    "money_mkt_dep": ["Money Market Deposits", "Money Market / NID Deposits (RM)"],
    "other_dep": ["Other Deposits"],
    "nii_incl_islamic": ["NII (incl Islamic)", "Net Interest Income (incl. Islamic, RM)"],
    "fee_income": ["Fee Income", "Fee & Commission Income (RM)"],
    "trading_income": ["Trading Income", "Trading & Investment Income (RM)"],
    "other_op_income": ["Other Operating Income", "Other Operating Income (RM)"],
    "operating_income": ["Operating Income"],
    "operating_expenses": ["Operating Expenses", "Opex"],
    "provisions": ["Provisions", "Provisions / Impairment (RM)"],
    "earning_assets": ["Earning Assets", "Interest Earning Assets", "Earning Assets (RM)"],
    "rwa": ["Risk-Weighted Assets", "Risk Weighted Assets"],
    "npl": ["Non-Performing Loans", "Impaired Financing", "NPA/NPL"],
    "llr": ["Loan Loss Reserve", "Allowance for ECL"],
    "tp_bank_nim_num": [
        "TP_Bank_NIM_Num",
        "TP (Bank) – NIM Numerator: Net Interest Income incl. Islamic (RM)",
    ],
    "tp_bank_nim_den": [
        "TP_Bank_NIM_Den",
        "TP (Bank) – NIM Denominator: Average Earning Assets (RM)",
    ],

    # >>> ADD THIS BLOCK <<<
    "nim_pct": [
        "NIM",
        "Net Interest/Financing Margin (%)",
        "Net Interest Margin (%)",
        "NIM (%)",
    ],
    "lcr_pct":  ["LCR", "LCR (%)"],
    "nsfr_pct": ["NSFR", "NSFR (%)"],

    # --- Banking capital ratios (percent) ---
    "cet1_ratio_pct":           ["CET1 Ratio (%)", "CET1", "CET1 (%)", "CET1 Ratio"],            # ← add "CET1 Ratio"
    "tier1_ratio_pct":          ["Tier 1 Capital Ratio (%)", "Tier 1 Ratio (%)", "Tier 1 (%)", "Tier 1 Ratio"],  # ← add "Tier 1 Ratio"
    "total_capital_ratio_pct":  ["Total Capital Ratio (%)", "Total Capital Ratio", "Total Capital (%)"],

    # --- Banking CASA ratios (percent) — allow direct pass-throughs ---
    "casa_ratio_pct": [
        "CASA Ratio (%)", "CASA Ratio", "CASA (%)"
    ],
    "casa_core_pct": [
        "CASA (Core, %)", "CASA Core (%)", "Core CASA (%)", "CASA Core"
    ],

    # ————— REITs & Property extras —————
    "npi": ["NPI", "Net Property Income"],
    "units": ["Units", "Units Outstanding", "Shares", "Shares Outstanding"],  # alias to shares
    "occupancy": ["Occupancy", "Occupancy (%)"],
    "wale": ["WALE", "WALE (years)", "Weighted Average Lease Expiry"],
    "rental_reversion": ["Rental Reversion", "Rental Reversion (%)"],

    # ————— Telco/Utilities extras —————
    "capex_to_rev_pct": ["Capex to Revenue", "Capex to Revenue (%)"],
    "rab": ["RAB", "Regulated Asset Base (RAB)"],
    "allowed_return_pct": ["Allowed Return", "Allowed Return (%)"],
    "availability_pct": ["Availability Factor", "Availability Factor (%)"],

    # ————— Tech extras —————
    "rnd": ["R&D Expense", "Research & Development"],

    # ————— Healthcare extras —————
    "beds": ["Bed Count"],
    "patient_days": ["Patient Days"],
    "admissions": ["Admissions"],
    "bed_occupancy_pct": ["Bed Occupancy", "Bed Occupancy (%)"],
    "alos_days": ["ALOS", "Average Length of Stay (days)"],

    # ————— Construction / Orderbook style —————
    "orderbook": ["Orderbook", "Order Book"],
    "new_orders": ["New Orders", "New Orders (TTM)"],
    "tender_book": ["Tender Book"],
    "win_rate_pct": ["Win Rate", "Win Rate (%)"],
}

# --- extensions for industry-specific raw fields ---
RAW_ALIASES.update({
    # Retail
    "sssg": ["SSSG", "Same-Store Sales Growth (%)", "Same Store Sales Growth"],

    # Tech
    "ndr_pct": ["Net Dollar Retention (%)", "NDR"],
    "gross_retention_pct": ["Gross Retention (%)"],
    "ltv": ["LTV", "Lifetime Value (LTV)"],
    "cac": ["CAC", "Customer Acquisition Cost (CAC)"],
    "rule_of_40_pct": ["Rule of 40 (%)", "Rule of 40"],

    # Utilities
    "allowed_return_pct": ["Allowed Return", "Allowed Return (%)"],
    "availability_pct": ["Availability Factor", "Availability Factor (%)"],

    # Energy/Materials
    "lifting_cost": ["Lifting Cost", "Lifting Cost (RM/boe)"],
    "average_selling_price": ["Average Selling Price", "Average Selling Price (RM/tonne)", "ASP", "ASP (RM/t)"],
    "strip_ratio": ["Strip Ratio"],
    "head_grade_gpt": ["Head Grade (g/t)", "Head Grade"],

    # Plantation
    # Raw inputs (accept both "key" style and "label-with-units" style headers)
    "cpo_output": ["CPO Output", "CPO Output (t)", "Crude Palm Oil Production (t)", "CPO Production (t)"],
    "ffb_input": ["FFB Input", "FFB Input (t)", "FFB Processed (t)", "FFB Processed"],
    "ffb_harvested": ["FFB Harvested", "FFB Harvested (t)", "FFB Production (t)", "FFB Produced (t)"],
    "planted_hectares": ["Planted Hectares", "Planted Hectares (ha)", "Planted Area (ha)", "Planted Area"],
    "mature_area": ["Mature Area", "Mature Area (ha)", "Mature Hectares (ha)", "Mature Hectares"],
    "unit_cash_cost": ["Unit Cash Cost", "Unit Cash Cost (RM/t)", "Unit Cash Cost (RM/tonne)"],
    "tonnes_sold": ["Tonnes Sold", "Tonnes Sold (t)", "FFB Sales Volume", "FFB Sales Volume (t)", "FFB Sales Volume (tonnes)"],
    "cpo_asp": ["CPO ASP", "CPO ASP (RM/t)", "Average CPO Selling Price (RM/t)", "Average CPO Selling Price", "CPO Average Selling Price (RM/t)"],

    # KPIs
    "oer_pct": ["OER", "OER (CPO/FFB Input, %)", "OER (%)", "Average Oil Extraction Rate (CPO) (%)"],
    "yield_per_hectare": ["FFB Yield (mt/ha)", "FFB Yield", "Yield per Hectare (FFB t/ha)", "Yield per Hectare"],
    "cash_margin_per_ton": ["Cash Margin per ton (RM/t)", "Cash Margin per ton"],
    "ebitda_per_ton": ["EBITDA per ton (RM/t)", "EBITDA per ton"],

    # Telco
    "arpu": ["ARPU", "ARPU (RM)"],
    "churn_rate_pct": ["Churn Rate", "Churn Rate (%)"],
    "avg_data_usage": ["Average Data Usage (GB/user)", "Avg Data Usage", "Average Data Usage"],

    # Transportation/Logistics
    "load_factor_pct": ["Load Factor", "Load Factor (%)", "Load Factor (airlines, %)"],
    "yield_metric": ["Yield per km/parcel", "Yield per km", "Yield", "Yield per km/parcel (RM)"],
    "on_time_pct": ["On-Time Performance", "On-Time Performance (%)"],
    "revenue_per_teu": ["Revenue per TEU", "Revenue per TEU (RM/TEU)"],
    "ask": ["ASK", "ASK (Available Seat Km, m)"],
    "rpk": ["RPK", "RPK (Revenue Passenger Km, m)"],
    "teu_throughput": ["TEU Throughput"],

    # REITs
    "hedged_debt_pct": ["Hedged Debt", "Hedged Debt (%)"],

    # Property
    "unbilled_sales": ["Unbilled Sales"],
    "sales_rate_pct": ["Sales Rate", "Sales Rate (%)", "Take-up Rate (%)", "Take-up Rate"],
    "rnav_ps": ["RNAV per Share"],

    # Leisure/Travel
    "hotel_occupancy_pct": ["Hotel Occupancy", "Hotel Occupancy (%)"],
    "revpar": ["RevPAR", "RevPAR (RM)"],

    # Leases (for lease-adjusted ND/EBITDA)
    "lease_liab_current": ["LeaseLiabCurrent", "Lease Liabilities (Current)"],
    "lease_liab_noncurrent": ["LeaseLiabNonCurrent", "Lease Liabilities (Non-Current)"],
    
    "orderbook": [
        "Orderbook", "Order Book", "Order Backlog", "Unbilled Sales"
    ],
    # capacity utilization (percent values are ok, we take the number)
    "capacity_utilization": [
        "Capacity Utilization", "Capacity Utilization (%)", "Utilization",
        "Capacity Utilization pct", "Capacity Utilization Pct", "Capacity Utilization %"
    ],    
    
})

# add quarterly-prefixed aliases
Q_ALIASES: Dict[str, List[str]] = {
    k: [f"Q_{name}" if not name.startswith("Q_") else name for name in v]
    for k, v in RAW_ALIASES.items()
}

# ensure quarterly synonyms explicitly include these too
Q_ALIASES.update({
    "orderbook": [
        "Q_Orderbook", "Q_Order Book", "Q_Order Backlog", "Q_Unbilled Sales"
    ],
    "capacity_utilization": [
        "Q_Capacity Utilization", "Q_Capacity Utilization (%)", "Q_Utilization",
        "Q_Capacity Utilization pct", "Q_Capacity Utilization %"
    ],

    "cet1_ratio_pct":           ["Q_CET1 Ratio (%)", "Q_CET1", "Q_CET1 (%)", "Q_CET1 Ratio"],                 # ← add
    "tier1_ratio_pct":          ["Q_Tier 1 Capital Ratio (%)", "Q_Tier 1 Ratio (%)", "Q_Tier 1 (%)", "Q_Tier 1 Ratio"],  # ← add
    "total_capital_ratio_pct":  ["Q_Total Capital Ratio (%)", "Q_Total Capital Ratio", "Q_Total Capital (%)"],
})


# convenience: quarterly specific keys that are stocks not flows
Q_STOCK_KEYS = {"price", "shares", "equity", "total_assets", "earning_assets"}

def _pick(row: dict, *canon_keys: str, quarterly: bool = False):
    """
    Pick the first matching column among aliases (case-insensitive).
    `canon_keys` are keys in RAW_ALIASES / Q_ALIASES.
    """
    aliases = Q_ALIASES if quarterly else RAW_ALIASES
    # Map lowercased actual column names -> real key
    key_map = {str(k).strip().lower(): k for k in row.keys()}
    for ck in canon_keys:
        for col in aliases.get(ck, []):
            # exact first
            if col in row:
                v = _to_num(row.get(col))
                if v is not None:
                    return v
            # then case-insensitive
            alt = key_map.get(str(col).strip().lower())
            if alt is not None:
                v = _to_num(row.get(alt))
                if v is not None:
                    return v
    return None

# ======================================================================
# Row-level derivations (Gross Profit, EBITDA, Three Fees, etc.)
# ======================================================================
def _derive_gross_profit(row: dict, *, quarterly: bool = False) -> Optional[float]:
    gp = _pick(row, "gross_profit", quarterly=quarterly)
    if gp is not None:
        return gp
    rev = _pick(row, "revenue", quarterly=quarterly)
    cogs = _pick(row, "cogs", quarterly=quarterly)
    if rev is None or cogs is None:
        return None
    return rev - cogs

def _derive_dep_amort(row: dict, *, quarterly: bool = False) -> Optional[float]:
    v = _pick(row, "dep_amort", quarterly=quarterly)
    if v is not None:
        return v
    ppe = _pick(row, "dep_ppe", quarterly=quarterly)
    rou = _pick(row, "dep_rou", quarterly=quarterly)
    ivp = _pick(row, "dep_invprop", quarterly=quarterly)
    return _sum([ppe, rou, ivp])

def _derive_ebitda(row: dict, *, quarterly: bool = False) -> Optional[float]:
    v = _pick(row, "ebitda", quarterly=quarterly)
    if v is not None:
        return v
    op = _pick(row, "operating_profit", quarterly=quarterly)
    da = _derive_dep_amort(row, quarterly=quarterly)
    if op is None or da is None:
        return None
    return op + da

def _derive_three_fees(row: dict, *, quarterly: bool = False) -> Optional[float]:
    # Selling + Admin + Interest (finance costs) using alias-aware picks
    selling = _pick(row, "selling_expenses", quarterly=quarterly)
    admin   = _pick(row, "admin_expenses",   quarterly=quarterly)
    finance = _pick(row, "interest_expense", quarterly=quarterly)
    return _sum([selling, admin, finance])

def ensure_three_fees_columns(df: pd.DataFrame) -> pd.DataFrame:
    """If 'Three Fees' not present, derive it for each annual row."""
    if df is None or df.empty:
        return df
    if "Three Fees" in df.columns:
        return df
    vals = []
    for _, r in df.iterrows():
        vals.append(_derive_three_fees(r.to_dict(), quarterly=False))
    df = df.copy()
    df["Three Fees"] = vals
    return df

def enrich_three_fees_inplace(row: dict) -> dict:
    """Mutate dict row (raw) by injecting 'Three Fees' if missing."""
    if "Three Fees" not in row or row.get("Three Fees") is None:
        row["Three Fees"] = _derive_three_fees(row, quarterly=False)
    return row

def ensure_auto_raw_columns(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    """
    Derive common raw fields if they are missing:
    - Gross Profit
    - EBITDA
    - Banking 'Operating Income' if not provided
    """
    if df is None or df.empty:
        return df
    out = df.copy()

    if "Gross Profit" not in out.columns:
        out["Gross Profit"] = [
            _derive_gross_profit(r.to_dict(), quarterly=False) for _, r in out.iterrows()
        ]

    if "EBITDA" not in out.columns:
        out["EBITDA"] = [
            _derive_ebitda(r.to_dict(), quarterly=False) for _, r in out.iterrows()
        ]

    if bucket == "Banking":
        if "Operating Income" not in out.columns:
            # sum NII(incl Islamic) + Fee + Trading + Other
            niis = []
            for _, r in out.iterrows():
                rr = r.to_dict()
                nii = _pick(rr, "nii_incl_islamic")
                if nii is None:
                    # derive NII: Interest Income - Interest Expense + Net Islamic Income (if your data has those literal columns)
                    ii = _to_num(rr.get("Interest Income"))
                    ie = _to_num(rr.get("Interest Expense"))
                    isl = _to_num(rr.get("Net Islamic Income"))
                    nii = _sum([(ii - ie if (ii is not None and ie is not None) else None), isl])
                fee = _pick(rr, "fee_income")
                trd = _pick(rr, "trading_income")
                oth = _pick(rr, "other_op_income")
                niis.append(_sum([nii, fee, trd, oth]))
            out["Operating Income"] = niis
    return out

def enrich_auto_raw_row_inplace(row: dict, bucket: str) -> dict:
    """On a dict (e.g., TTM raw row), inject derivatives (GP, EBITDA, Operating Income for banks)."""
    if row.get("Gross Profit") is None:
        row["Gross Profit"] = _derive_gross_profit(row, quarterly=False)
    if row.get("EBITDA") is None:
        row["EBITDA"] = _derive_ebitda(row, quarterly=False)
    if bucket == "Banking" and row.get("Operating Income") is None:
        nii = _pick(row, "nii_incl_islamic")
        if nii is None:
            ii = _to_num(row.get("Interest Income"))
            ie = _to_num(row.get("Interest Expense"))
            isl = _to_num(row.get("Net Islamic Income"))
            nii = _sum([(ii - ie if (ii is not None and ie is not None) else None), isl])
        fee = _pick(row, "fee_income")
        trd = _pick(row, "trading_income")
        oth = _pick(row, "other_op_income")
        row["Operating Income"] = _sum([nii, fee, trd, oth])
    return row

# ======================================================================
# TTM from Quarterly (flows = sum last 4, stocks = last/latest)
# Returned as a dict of raw-like fields you can append to annual.
# ======================================================================

def ttm_raw_row_from_quarters(
    qdf: pd.DataFrame, *, current_price: Optional[float] = None
) -> Optional[dict]:
    if qdf is None or qdf.empty:
        return None
    qdf = qdf.copy()
    if "Year" not in qdf.columns or "Quarter" not in qdf.columns:
        return None

    qdf["__Qnum"] = pd.to_numeric(qdf["Quarter"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    qdf = qdf.dropna(subset=["Year", "__Qnum"]).sort_values(["Year", "__Qnum"])
    last4 = qdf.tail(4)

    def q_sum(canon):
        cols = Q_ALIASES.get(canon, [])
        for col in cols:
            if col in last4.columns:
                vals = pd.to_numeric(last4[col], errors="coerce").tolist()
                return _sum_or_none(vals)     # same None-on-all-missing semantics
        return None

    def q_last(canon):
        cols = Q_ALIASES.get(canon, [])
        for col in cols:
            if col in last4.columns:
                val = _last(last4[col])
                if val is not None:
                    return val
        return None

    ttm: dict = {}
    # flows (sum)
    ttm["Revenue"]          = q_sum("revenue")
    ttm["Gross Profit"]     = q_sum("gross_profit") or None
    ttm["Operating Profit"] = q_sum("operating_profit")
    ttm["EBITDA"]           = q_sum("ebitda")
    ttm["Net Profit"]       = q_sum("net_profit")
    ttm["CFO"]              = q_sum("cfo")
    ttm["Capex"]            = q_sum("capex")
    ttm["DPS"]              = q_sum("dps")
    ttm["DPU"]              = q_sum("dpu")   # allow payout% to work with DPU
    ttm["NII (incl Islamic)"]  = q_sum("nii_incl_islamic")
    ttm["Operating Income"]    = q_sum("operating_income")    
    ttm["Interest Expense"]     = q_sum("interest_expense")
    ttm["Selling Expenses"]     = q_sum("selling_expenses")
    ttm["Administrative Expenses"] = q_sum("admin_expenses")
    # Banking/financial flows needed by view + CIR
    ttm["Operating Expenses"]     = q_sum("operating_expenses")
    ttm["Provisions"]             = q_sum("provisions")
    ttm["Other Operating Income"] = q_sum("other_op_income")
    
    # add COGS to TTM so working-capital day metrics use COGS (with abs) rather than falling back to revenue
    ttm["CostOfSales"] = q_sum("cogs")
    if ttm.get("CostOfSales") is not None:
        ttm["Cost of Sales"] = ttm["CostOfSales"]  # alias to satisfy RAW_ALIASES

    # stocks (carry latest quarter)
    ttm["Equity"]               = q_last("equity")
    ttm["Total Assets"]         = q_last("total_assets")
    ttm["Current Assets"]       = q_last("current_assets")
    ttm["Current Liabilities"]  = q_last("current_liabilities")
    ttm["Receivables"]          = q_last("receivables")
    ttm["Inventory"]            = q_last("inventory")
    ttm["Payables"]             = q_last("payables")
    ttm["Earning Assets"]       = q_last("earning_assets")
    ttm["Shares"]               = q_last("shares")
    ttm["Price"]                = current_price if current_price is not None else q_last("price")
    ttm["Total Borrowings"]     = q_last("borrowings")
    ttm["Cash & Cash Equivalents"] = q_last("cash")
    ttm["NIM"]                  = q_sum("nim_pct")    # sum of the latest 4 quarters per request
    ttm["LCR"]                  = q_last("lcr_pct")
    ttm["NSFR"]                 = q_last("nsfr_pct")
    ttm["CET1 Ratio (%)"]            = q_last("cet1_ratio_pct")
    ttm["Tier 1 Capital Ratio (%)"]  = q_last("tier1_ratio_pct")
    ttm["Total Capital Ratio (%)"]   = q_last("total_capital_ratio_pct")
    ttm["CASA Ratio (%)"]       = q_last("casa_ratio_pct")
    ttm["CASA (Core, %)"]       = q_last("casa_core_pct")  
    ttm["Orderbook"]            = q_last("orderbook")
    ttm["Gross Loans"]          = q_last("gross_loans")
    ttm["Deposits"]             = q_last("deposits")
    ttm["Demand Deposits"]      = q_last("demand_dep")
    ttm["Savings Deposits"]     = q_last("savings_dep")
    ttm["Time/Fixed Deposits"]  = q_last("time_dep")
    ttm["Non-Performing Loans"] = q_last("npl")
    ttm["Loan Loss Reserve"]    = q_last("llr")

    return ttm

# Back-compat (Systematic Decision page expects this)
def compute_ttm(df: pd.DataFrame, name: str, bucket: Optional[str] = None) -> dict:
    """
    Minimal TTM ratios for the Decision page.
    Returns: {"P/E": x, "P/B": y, "ROE%": z, "GM%": a, "NM%": b, "DY": d}
    Accepts optional `bucket` to follow UI selection.
    """
    if df is None or df.empty:
        return {}

    q = df[(df.get("Name") == name) & (df.get("IsQuarter") == True)]
    a = df[(df.get("Name") == name) & (df.get("IsQuarter") != True)]
    if q.empty and a.empty:
        return {}

    # compute raw ttm
    price_hint = _last(a.get("Price")) if isinstance(a, pd.DataFrame) and "Price" in a.columns else None
    raw_ttm = ttm_raw_row_from_quarters(q, current_price=price_hint)
    if not raw_ttm:
        return {}

    # fill derivatives
    enrich_three_fees_inplace(raw_ttm)
    enrich_auto_raw_row_inplace(
        raw_ttm,
        bucket=bucket or _infer_bucket_for_rows(a) or "General"   # <-- changed
    )
    
    # NEW: fallback Shares for TTM if quarterly didn’t provide them (Decision page)
    prev_row = a.dropna(subset=["Year"]).sort_values("Year").iloc[-1].to_dict() if not a.empty else None
    if raw_ttm.get("Shares") in (None, 0) and prev_row:
        raw_ttm["Shares"] = _pick(prev_row, "shares")

    rev = _to_num(raw_ttm.get("Revenue"))
    gp = _to_num(raw_ttm.get("Gross Profit"))
    if gp is None:
        gp = _derive_gross_profit(raw_ttm)

    npf = _to_num(raw_ttm.get("Net Profit"))
    eq  = _to_num(raw_ttm.get("Equity"))
    shr = _to_num(raw_ttm.get("Shares"))
    px  = _to_num(raw_ttm.get("Price"))

    # DPS: prefer TTM; fallback to latest annual DPS if available
    dps = _to_num(raw_ttm.get("DPS"))
    if dps is None and isinstance(a, pd.DataFrame) and "DPS" in a.columns:
        dps = _last(a.get("DPS"))

    eps  = _safe_div(npf, shr)
    bvps = _safe_div(eq,  shr)
    pe   = (px / eps) if (px and eps and eps > 0) else None
    pb   = (px / bvps) if (px and bvps and bvps > 0) else None

    rev_mag = abs(rev) if rev is not None else None
    gm   = (gp / rev_mag * 100.0) if (gp is not None and rev_mag not in (None, 0)) else None
    nm   = (npf / rev_mag * 100.0) if (npf is not None and rev_mag not in (None, 0)) else None

    roe  = (npf / eq * 100.0) if (npf is not None and eq not in (None, 0)) else None
    dy   = ((abs(dps) / px) if (dps is not None and px not in (None, 0)) else None)

    return {"P/E": pe, "P/B": pb, "ROE%": roe, "GM%": gm, "NM%": nm, "DY": dy}


def _infer_bucket_for_rows(annual_rows: pd.DataFrame) -> Optional[str]:
    if annual_rows is None or annual_rows.empty:
        return None
    col = None
    for k in ("IndustryBucket", "Industry"):
        if k in annual_rows.columns:
            col = annual_rows[k]
            break
    if col is not None and col.dropna().size:
        v = str(col.dropna().iloc[-1])
        if config and hasattr(config, "INDUSTRY_BUCKETS"):
            return v if v in getattr(config, "INDUSTRY_BUCKETS") else v
        return v
    return None

# ======================================================================
# Ratio calculators by industry (per-year + TTM)
# Each calculator receives (row_dict, prev_row_dict, context)
# and must return a float or None.
# ======================================================================
Calculator = Callable[[dict, Optional[dict], dict], Optional[float]]

def _ctx_price(row: dict, ctx: dict) -> Optional[float]:
    # prefer row Price; fallback to ctx["price_fallback"]
    px = _to_num(row.get("Price"))
    if px is not None:
        return px
    return _to_num(ctx.get("price_fallback"))

def _common_calcs() -> Dict[str, Calculator]:
    """Ratios broadly applicable to non-bank industries (with avg balances + aliases)."""
    def pct(num, den):
        v = _safe_div(num, den)
        return v * 100.0 if v is not None else None

    def avg_stock(r, p, key):
        return _avg2(_pick(r, key), _pick(p or {}, key))

    d: Dict[str, Calculator] = {
        # Margins
        "Gross Margin (%)":            lambda r, p, c: pct(_derive_gross_profit(r), _pick(r, "revenue")),
        "EBITDA Margin (%)":           lambda r, p, c: pct(_derive_ebitda(r), _pick(r, "revenue")),
        "Operating Profit Margin (%)": lambda r, p, c: pct(_pick(r, "operating_profit"), _pick(r, "revenue")),
        "Net Margin (%)":              lambda r, p, c: pct(_pick(r, "net_profit"), _pick(r, "revenue")),

        # Returns (avg denominators)
        "ROE (%)": lambda r, p, c: pct(_pick(r, "net_profit"), _avg2(_pick(r, "equity"), _pick(p or {}, "equity"))),
        "ROA (%)": lambda r, p, c: pct(_pick(r, "net_profit"), _avg2(_pick(r, "total_assets"), _pick(p or {}, "total_assets"))),

        # Efficiency & Working Capital (use average balances AND |COGS| in denominators)
        "Inventory Turnover (×)": lambda r, p, c: _safe_div(
            _mag(_pick(r, "cogs") or _pick(r, "revenue")),
            _avg2(_pick(r, "inventory"), _pick(p or {}, "inventory"))
        ),
        "Inventory Days (days)": lambda r, p, c: (
            _safe_div(
                _avg2(_pick(r, "inventory"), _pick(p or {}, "inventory")),
                _mag(_pick(r, "cogs") or _pick(r, "revenue"))
            ) * 365.0
            if any(v is not None for v in (_pick(r, "inventory"), _pick(p or {}, "inventory"), _pick(r, "cogs"), _pick(r, "revenue")))
            else None
        ),
        "Receivable Days (days)": lambda r, p, c: (
            _safe_div(_avg2(_pick(r, "receivables"), _pick(p or {}, "receivables")), _mag(_pick(r, "revenue"))) * 365.0
            if _avg2(_pick(r, "receivables"), _pick(p or {}, "receivables")) is not None and _pick(r, "revenue") not in (None, 0)
            else None
        ),
        "Payable Days (days)": lambda r, p, c: (
            _safe_div(_avg2(_pick(r, "payables"), _pick(p or {}, "payables")), _mag(_pick(r, "cogs") or _pick(r, "revenue"))) * 365.0
            if _avg2(_pick(r, "payables"), _pick(p or {}, "payables")) is not None and (_pick(r, "cogs") or _pick(r, "revenue")) not in (None, 0)
            else None
        ),
        "Cash Conversion Cycle (days)": lambda r, p, c: (
            (
                (_safe_div(_avg2(_pick(r, "inventory"), _pick(p or {}, "inventory")), _mag(_pick(r, "cogs") or _pick(r, "revenue"))) or 0.0) * 365.0
                + (_safe_div(_avg2(_pick(r, "receivables"), _pick(p or {}, "receivables")), _mag(_pick(r, "revenue"))) or 0.0) * 365.0
                - (_safe_div(_avg2(_pick(r, "payables"), _pick(p or {}, "payables")), _mag(_pick(r, "cogs") or _pick(r, "revenue"))) or 0.0) * 365.0
            )
            if any(_avg2(_pick(r, k), _pick(p or {}, k)) is not None for k in ("inventory", "receivables", "payables"))
            else None
        ),

        # Liquidity & Leverage
        "Current Ratio (×)":     lambda r, p, c: _safe_div(_pick(r, "current_assets"), _pick(r, "current_liabilities")),
        "Quick Ratio (×)":       lambda r, p, c: _safe_div((_pick(r, "current_assets") or 0.0) - (_pick(r, "inventory") or 0.0), _pick(r, "current_liabilities")),
        "Debt/Equity (×)":       lambda r, p, c: _safe_div(_pick(r, "borrowings"), _pick(r, "equity")),

        # Use abs(interest_expense) to avoid negative coverage from expense sign
        "Interest Coverage (×)": lambda r, p, c: _safe_div(_pick(r, "operating_profit"), _mag(_pick(r, "interest_expense"))),

        "Net Debt / EBITDA (×)": lambda r, p, c: (lambda e, nd:
            _safe_div(nd, e) if (e is not None and e > 0) else None
        )(_derive_ebitda(r), ((_pick(r, "borrowings") or 0.0) - (_pick(r, "cash") or 0.0))),

        # Cash Flow (ratios that should display as positive intensities)
        "Capex to Revenue (%)":  lambda r, p, c: _pct_mag(_mag(_pick(r, "capex")), _mag(_pick(r, "revenue"))),
        "FCF Margin (%)":        lambda r, p, c: _pct_mag(((_pick(r, "cfo") or 0.0) - (_mag(_pick(r, "capex")) or 0.0)), _mag(_pick(r, "revenue"))),
        "CFO/EBITDA (%)":        lambda r, p, c: (_safe_div(_pick(r, "cfo"), _derive_ebitda(r)) * 100.0) if _derive_ebitda(r) not in (None, 0) else None,
        "Cash Conversion (%)":   lambda r, p, c: ( _safe_div(((_pick(r, "cfo") or 0.0) - (_pick(r, "capex") or 0.0)), _pick(r, "net_profit")) * 100.0
                                                   if _pick(r, "net_profit") not in (None, 0) else None ),


        "Three Fees Ratio (%)":  lambda r, p, c: _pct_mag(_mag(_derive_three_fees(r)), _mag(_pick(r, "revenue"))),

        # --- Added metrics (place before Valuation) ---
        # Returns / leverage requested by some buckets
        "Financial Leverage (Assets/Equity)": lambda r, p, c: _safe_div(_pick(r, "total_assets"), _pick(r, "equity")),

        # Income & Efficiency (for 'Financials' bucket in config)
        "Operating Expense Ratio (%)": lambda r, p, c: _pct_mag(_mag(_pick(r, "operating_expenses")), _mag(_pick(r, "revenue"))),

        # Cash Flow (shared)
        "Operating CF Margin (%)": lambda r, p, c: (
            (_pick(r, "cfo") / (_pick(r, "operating_income") or _pick(r, "revenue")) * 100.0)
            if (_pick(r, "cfo") is not None
                and (_pick(r, "operating_income") or _pick(r, "revenue")) not in (None, 0))
            else None
        ),

        # Valuation
        "EPS (RM)":        lambda r, p, c: _safe_div(_pick(r, "net_profit"), _pick(r, "shares")),
        "P/E (×)":         lambda r, p, c: (lambda eps, px: (px / eps) if (px and eps and eps > 0) else None)(
                                _safe_div(_pick(r, "net_profit"), _pick(r, "shares")), _ctx_price(r, c)),
        "P/B (×)":         lambda r, p, c: (lambda bvps, px: (px / bvps) if (px and bvps and bvps > 0) else None)(
                                _safe_div(_pick(r, "equity"), _pick(r, "shares")), _ctx_price(r, c)),
        "EV/EBITDA (×)":   lambda r, p, c: (lambda ebitda, px, shr, debt, cash: (
                                ((px or 0) * (shr or 0) + ((debt or 0) - (cash or 0))) / ebitda
                                if (ebitda and ebitda > 0) else None))(
                                _derive_ebitda(r), _ctx_price(r, c), _pick(r, "shares"), _pick(r, "borrowings"), _pick(r, "cash")),
        "EV/Sales (×)":    lambda r, p, c: (lambda rev, px, shr, debt, cash: (
                                ((px or 0) * (shr or 0) + ((debt or 0) - (cash or 0))) / rev
                                if (rev not in (None, 0)) else None))(
                                _pick(r, "revenue"), _ctx_price(r, c), _pick(r, "shares"), _pick(r, "borrowings"), _pick(r, "cash")),
        "Dividend Yield (%)": lambda r, p, c: (lambda dps, px:
            ((abs(dps) / px) * 100.0) if (dps is not None and px not in (None, 0)) else None
        )(_pick(r, "dps") or _pick(r, "dpu"), _ctx_price(r, c)),

        
        "FCF Yield (%)":   lambda r, p, c: (lambda fcf, px, shr: (
                                fcf / (px * shr) * 100.0) if fcf is not None and px not in (None, 0) and shr not in (None, 0) else None)(
                                ((_pick(r, "cfo") or 0.0) - (_pick(r, "capex") or 0.0)), _ctx_price(r, c), _pick(r, "shares")),
    
        # --- Growth + distributions ---
        "EPS YoY (%)": lambda r, p, ctx: (
            (lambda cur_eps, prev_eps:
                (((cur_eps / prev_eps) - 1.0) * 100.0)
                if (cur_eps is not None and prev_eps not in (None, 0)) else None
            )(
                _safe_div(_pick(r, "net_profit"), _pick(r, "shares")),
                _safe_div(_pick(p or {}, "net_profit"), _pick(p or {}, "shares"))
            )
        ),

        "Payout Ratio (%)": lambda r, p, ctx: (
            (lambda dps_like, eps:
                ((abs(dps_like) / eps) * 100.0)
                if (dps_like is not None and eps is not None and eps > 0) else None
            )(
                (_pick(r, "dps") if _pick(r, "dps") is not None else _pick(r, "dpu")),
                _safe_div(_pick(r, "net_profit"), _pick(r, "shares"))
            )
        ),

    }

    # ------- label aliases to match your UI variants -------
    d["Inventory Days"]    = d["Inventory Days (days)"]
    d["Receivable Days"]   = d["Receivable Days (days)"]
    d["Payable Days"]      = d["Payable Days (days)"]
    d["Capex/Revenue (%)"] = d["Capex to Revenue (%)"]

    # extra label aliases to match config
    d["CapEx Intensity (%)"]             = d["Capex to Revenue (%)"]
    d["Cash Conversion (CFO/EBITDA)"]    = d["CFO/EBITDA (%)"]
    # (you already have the (x) vs (×) loop below — keep it)

    # (x) vs (×) variants
    for base in ["Current Ratio", "Quick Ratio", "Debt/Equity", "Interest Coverage", "Net Debt / EBITDA", "EV/EBITDA", "P/E", "P/B", "Inventory Turnover"]:
        d[f"{base} (x)"] = d.get(f"{base} (×)", d.get(f"{base} (x)"))

    return d

def _banking_calcs() -> Dict[str, Calculator]:
    def pct(num, den):
        v = _safe_div(num, den)
        return v * 100.0 if v is not None else None

    d: Dict[str, Calculator] = {
        # Income & Efficiency
        "NIM (%)": lambda r, p, c: (
            _pick(r, "nim_pct")  # prefer raw NIM (%) if provided
            if _pick(r, "nim_pct") is not None else
            (lambda nii_num, ea_cur, ea_prev, manual_den: (
                (_safe_div(nii_num, (manual_den if manual_den not in (None, 0) else _avg2(ea_cur, ea_prev))) * 100.0)
                if (nii_num is not None and ((manual_den not in (None, 0)) or _avg2(ea_cur, ea_prev) not in (None, 0))) else None
            ))(
                _pick(r, "tp_bank_nim_num") or _pick(r, "nii_incl_islamic"),
                _pick(r, "earning_assets"),
                _pick(p or {}, "earning_assets"),
                _pick(r, "tp_bank_nim_den") or _to_num(r.get("TP_Bank_NIM_Den"))
            )
        ),

        # Cash Flow
        "Operating CF Margin (%)": lambda r, p, c: (
            (lambda cfo, den: (cfo / den * 100.0) if (cfo is not None and den not in (None, 0)) else None)(
                _pick(r, "cfo"),
                # For banks, use Operating Income if available; fall back to Revenue if present.
                _pick(r, "operating_income") or _pick(r, "operating income") or _pick(r, "revenue")
            )
        ),

        # Cost-to-Income with optional overrides
        "Cost-to-Income Ratio (%)": lambda r, p, c: _pct_mag(
            _mag(_to_num(r.get("TP_Bank_CIR_Num")) if _to_num(r.get("TP_Bank_CIR_Num")) is not None else _pick(r, "operating_expenses")),
            (_to_num(r.get("TP_Bank_CIR_Den")) if _to_num(r.get("TP_Bank_CIR_Den")) is not None else
            (_pick(r, "operating_income") or _sum([
                _pick(r, "nii_incl_islamic"),
                _pick(r, "fee_income"),
                _pick(r, "trading_income"),
                _pick(r, "other_op_income"),
            ])))
        ),

        # Leverage with optional overrides
        "Financial Leverage (×)": lambda r, p, c: _safe_div(
            (_to_num(r.get("TP_Bank_Lev_Num")) if _to_num(r.get("TP_Bank_Lev_Num")) is not None else _pick(r, "total_assets")),
            (_to_num(r.get("TP_Bank_Lev_Den")) if _to_num(r.get("TP_Bank_Lev_Den")) is not None else _pick(r, "equity")),
        ),

        # Asset Quality
        "NPL Ratio (%)": lambda r, p, c: pct(_pick(r, "npl"), _pick(r, "gross_loans")),
        "Loan Loss Coverage (×)": lambda r, p, c: _safe_div(_mag(_pick(r, "llr")), _mag(_pick(r, "npl"))),
        "Loan-Loss Coverage (×)": lambda r, p, c: _safe_div(_mag(_pick(r, "llr")), _mag(_pick(r, "npl"))),


        # Capital & Liquidity
        "Loan-to-Deposit Ratio (%)": lambda r, p, c: pct(_pick(r, "gross_loans"), _pick(r, "deposits")),
        "Loan-to-Deposit Ratio (×)": lambda r, p, c: _safe_div(_pick(r, "gross_loans"), _pick(r, "deposits")),
        
        "CASA Ratio (%)": lambda r, p, c: (
            _to_num(r.get("CASA Ratio (%)"))            # literal if present
            or _pick(r, "casa_ratio_pct")               # canonical alias pass-through
            or (
                # computed fallback: (Demand + Savings) / Deposits-or-Time × 100
                (_safe_div(
                    (_pick(r, "demand_dep") or 0.0) + (_pick(r, "savings_dep") or 0.0),
                    (_pick(r, "deposits") or _pick(r, "time_dep") or 0.0)
                ) * 100.0)
                if any(_pick(r, k) is not None for k in ("demand_dep","savings_dep","deposits","time_dep"))
                else None
            )
        ),

        "CASA (Core, %)": lambda r, p, c: (
            _to_num(r.get("CASA (Core, %)"))            # literal if present
            or _pick(r, "casa_core_pct")                # canonical alias pass-through
            or (
                # computed fallback (core) = (Demand + Savings) / Time × 100
                (_safe_div(
                    (_pick(r, "demand_dep") or 0.0) + (_pick(r, "savings_dep") or 0.0),
                    (_pick(r, "time_dep") or 0.0)
                ) * 100.0)
                if any(_pick(r, k) is not None for k in ("demand_dep","savings_dep","time_dep"))
                else None
            )
        ),

        # Returns
        "ROE (%)": lambda r, p, c: pct(_pick(r, "net_profit"), _avg2(_pick(r, "equity"), _pick(p or {}, "equity"))),
        "ROA (%)": lambda r, p, c: pct(_pick(r, "net_profit"), _avg2(_pick(r, "total_assets"), _pick(p or {}, "total_assets"))),

        # Valuation
        "EPS (RM)": lambda r, p, c: _safe_div(_pick(r, "net_profit"), _pick(r, "shares")),
        "P/E (×)": lambda r, p, c: (lambda eps, px: (px / eps) if (px and eps and eps > 0) else None)(
            _safe_div(_pick(r, "net_profit"), _pick(r, "shares")), _ctx_price(r, c)
        ),
        "P/B (×)": lambda r, p, c: (lambda bvps, px: (px / bvps) if (px and bvps and bvps > 0) else None)(
            _safe_div(_pick(r, "equity"), _pick(r, "shares")), _ctx_price(r, c)
        ),
        "Dividend Yield (%)": lambda r, p, c: (lambda dps, px:
            ((abs(dps) / px) * 100.0) if (dps is not None and px not in (None, 0)) else None
        )(_pick(r, "dps"), _ctx_price(r, c)),

        # Direct picks (pass-through)
        "CET1 Ratio (%)":           lambda r, p, c: (_pick(r, "cet1_ratio_pct") 
                                            or _to_num(r.get("CET1 Ratio (%)")) 
                                            or _to_num(r.get("CET1 Ratio"))),
        "Tier 1 Capital Ratio (%)": lambda r, p, c: (_pick(r, "tier1_ratio_pct") 
                                                    or _to_num(r.get("Tier 1 Capital Ratio (%)")) 
                                                    or _to_num(r.get("Tier 1 Ratio"))),
        "Total Capital Ratio (%)":  lambda r, p, c: (_pick(r, "total_capital_ratio_pct") 
                                                    or _to_num(r.get("Total Capital Ratio (%)")) 
                                                    or _to_num(r.get("Total Capital Ratio"))),
        "LCR (%)":  lambda r, p, c: _pick(r, "lcr_pct"),
        "NSFR (%)": lambda r, p, c: _pick(r, "nsfr_pct"),

        # Label variant used in some configs
        "Financial Leverage (Assets/Equity)": lambda r, p, c: _safe_div(_pick(r, "total_assets"), _pick(r, "equity")),
    }

    # unit/glyph aliases so Banking tolerates (x) vs (×)
    d["P/E (x)"]  = d.get("P/E (×)",  d.get("P/E (x)"))
    d["P/B (x)"]  = d.get("P/B (×)",  d.get("P/B (x)"))
    d["Loan-to-Deposit Ratio (x)"] = d.get("Loan-to-Deposit Ratio (×)", d.get("Loan-to-Deposit Ratio (x)"))
    d["Interest Coverage (x)"] = d.get("Interest Coverage (×)", d.get("Interest Coverage (x)"))
    return d

def _reits_calcs() -> Dict[str, Calculator]:
    def pct(num, den):
        v = _safe_div(num, den)
        return v * 100.0 if v is not None else None

    d: Dict[str, Calculator] = {
        # Debt & Hedging
        "Gearing (x)":                  lambda r, p, c: _safe_div(_pick(r, "borrowings"), _pick(r, "equity")),          # Debt/Equity
        "Gearing (Debt/Assets, %)":     lambda r, p, c: pct(_pick(r, "borrowings"), _pick(r, "total_assets")),          # Debt/Assets %
        "Interest Coverage (×)":        lambda r, p, c: _safe_div(_derive_ebitda(r), _mag(_pick(r, "interest_expense"))),


        "Avg Cost of Debt (%)":         lambda r, p, c: _pct_mag(_mag(_pick(r, "interest_expense")), _pick(r, "borrowings")),

        # Portfolio Quality
        "Occupancy (%)":                lambda r, p, c: _pick(r, "occupancy"),
        "WALE (years)":                 lambda r, p, c: _pick(r, "wale"),
        "Rental Reversion (%)":         lambda r, p, c: _pick(r, "rental_reversion"),

        # Distribution & Valuation
        "DPU (RM)":                     lambda r, p, c: _pick(r, "dpu") or _pick(r, "dps"),
        "Distribution Yield (%)":       lambda r, p, c: (lambda dpu, px: ((abs(dpu) / px) * 100.0) if (dpu is not None and px not in (None, 0)) else None)(
                                              _pick(r, "dpu") or _pick(r, "dps"), _ctx_price(r, c)),
        "NAV per Unit":                 lambda r, p, c: _safe_div(_pick(r, "equity"), _pick(r, "shares")),
        "P/NAV (×)":                    lambda r, p, c: (lambda navps, px: (px / navps) if (px and navps and navps > 0) else None)(
                                              _safe_div(_pick(r, "equity"), _pick(r, "shares")), _ctx_price(r, c)),
        
        "NPI Margin (%)":  lambda r, p, c: (_safe_div(_pick(r, "npi"), _pick(r, "revenue")) * 100.0
                                    if (_pick(r, "npi") is not None and _pick(r, "revenue") not in (None, 0)) else None),
        "Hedged Debt (%)": lambda r, p, c: _pick(r, "hedged_debt_pct"),

    }

    # label aliases that config uses
    d["Interest Coverage (x)"]       = d["Interest Coverage (×)"]
    d["P/NAV (x)"]                   = d["P/NAV (×)"]
    d["Average Cost of Debt (%)"]    = d["Avg Cost of Debt (%)"]
    d["Gearing (%)"]                 = d["Gearing (Debt/Assets, %)"]
    return d


def _tech_calcs() -> Dict[str, Calculator]:
    base = _common_calcs()
    base.update({
        "R&D Intensity (%)": lambda r, p, c: (lambda rnd, rev: (rnd / rev * 100.0) if (rnd is not None and rev not in (None, 0)) else None)(
            _pick(r, "rnd"), _pick(r, "revenue")
        ),
        "Net Dollar Retention (%)": lambda r, p, c: _pick(r, "ndr_pct"),
        "Gross Retention (%)":      lambda r, p, c: _pick(r, "gross_retention_pct"),
        "LTV/CAC (x)":              lambda r, p, c: _safe_div(_pick(r, "ltv"), _pick(r, "cac")),
        "Rule of 40 (%)":           lambda r, p, c: _pick(r, "rule_of_40_pct"),
    })
    return base

def _utils_telco_calcs() -> Dict[str, Calculator]:
    base = _common_calcs()
    base.update({
        "Capex to Revenue (%)": base["Capex to Revenue (%)"],  # explicit
        "Operating CF Margin (%)": base["Operating CF Margin (%)"],  # from _common_calcs
        "Dividend Cash Coverage (CFO/Div)": lambda r, p, c: (lambda cfo, dps, shr: (
            cfo / (dps * shr) if (cfo is not None and dps not in (None, 0) and shr not in (None, 0)) else None
        ))(_pick(r, "cfo"), _pick(r, "dps"), _pick(r, "shares")),
        # Utilities-only fields (harmless as pass-throughs)
        "Allowed Return (%)":        lambda r, p, c: _pick(r, "allowed_return_pct"),
        "Availability Factor (%)":   lambda r, p, c: _pick(r, "availability_pct"),
        # Telco-only fields (also harmless elsewhere)
        "ARPU (RM)":                 lambda r, p, c: _pick(r, "arpu"),
        "Churn Rate (%)":            lambda r, p, c: _pick(r, "churn_rate_pct"),

    })
    return base

def _property_calcs() -> Dict[str, Calculator]:
    base = _common_calcs()
    base.update({
        "Net Debt (RM)": lambda r, p, c: (lambda debt, cash: (debt - cash) if (debt is not None and cash is not None) else None)(
            _pick(r, "borrowings"), _pick(r, "cash")
        ),
        "Gearing (x)": lambda r, p, c: _safe_div(_pick(r, "borrowings"), _pick(r, "equity")),
        "Net Gearing (Net Debt/Equity, %)": lambda r, p, c: (lambda netdebt, eq: (
            netdebt / eq * 100.0) if (netdebt is not None and eq not in (None, 0)) else None
        )(((_pick(r, "borrowings") or 0.0) - (_pick(r, "cash") or 0.0)), _pick(r, "equity")),
        
        # Canonical label in config is "Distribution/Dividend Yield (%)"
        "Distribution/Dividend Yield (%)": lambda r, p, c: (lambda dps, px:
            ((abs(dps) / px) * 100.0) if (dps is not None and px not in (None, 0)) else None
        )(_pick(r, "dps") or _pick(r, "dpu"), _ctx_price(r, c)),

        # P/NAV and P/RNAV
        "P/NAV (×)": lambda r, p, c: (lambda navps, px: (px / navps) if (px and navps and navps > 0) else None)(
            _safe_div(_pick(r, "equity"), _pick(r, "shares")), _ctx_price(r, c)),
        "P/RNAV (×)": lambda r, p, c: (lambda rnavps, px: (px / rnavps) if (px and rnavps and rnavps > 0) else None)(
            _pick(r, "rnav_ps"), _ctx_price(r, c)),

        # Unbilled sales coverage
        "Unbilled Sales / Revenue (×)": lambda r, p, c: _safe_div(_pick(r, "unbilled_sales"), _pick(r, "revenue")),
        "Unbilled Cover (months)":      lambda r, p, c: (lambda x: x * 12.0 if x is not None else None)(
            _safe_div(_pick(r, "unbilled_sales"), _pick(r, "revenue"))),

        # Sales rate pass-through
        "Sales Rate (%)": lambda r, p, c: _pick(r, "sales_rate_pct"),
    })
    return base

def _healthcare_calcs() -> Dict[str, Calculator]:
    base = _common_calcs()
    base.update({
        "Bed Occupancy (%)": lambda r, p, c: _pick(r, "bed_occupancy_pct"),
        "ALOS (days)":       lambda r, p, c: _pick(r, "alos_days") or _safe_div(_pick(r, "patient_days"), _pick(r, "admissions")),

        # Ops / per-capacity economics from config
        "Bed Turnover (admissions/bed)": lambda r, p, c: _safe_div(_pick(r, "admissions"), _pick(r, "beds")),
        "Revenue per Bed (RM)":          lambda r, p, c: _safe_div(_pick(r, "revenue"), _pick(r, "beds")),
        "EBITDA per Bed (RM)":           lambda r, p, c: _safe_div(_derive_ebitda(r), _pick(r, "beds")),
        "Revenue per Patient Day (RM)":  lambda r, p, c: _safe_div(_pick(r, "revenue"), _pick(r, "patient_days")),
        "EBITDA per Patient Day (RM)":   lambda r, p, c: _safe_div(_derive_ebitda(r), _pick(r, "patient_days")),
        "R&D Intensity (%)":             lambda r, p, c: (lambda rnd, rev: (rnd / rev * 100.0) if (rnd is not None and rev not in (None, 0)) else None)(
                                                 _pick(r, "rnd"), _pick(r, "revenue")),
    })
    return base

def _construction_calcs() -> Dict[str, Calculator]:
    base = _common_calcs()
    base.update({
        "Win Rate (%)": lambda r, p, c: (lambda newo, tb: (newo / tb * 100.0) if (newo is not None and tb not in (None, 0)) else None)(
            _pick(r, "new_orders"),
            _pick(r, "tender_book"),
        ),
        "Backlog Coverage (×)": lambda r, p, c: _safe_div(_pick(r, "orderbook"), _pick(r, "revenue")),
    })
    return base

BUCKET_CALCS: Dict[str, Dict[str, Calculator]] = {
    "General": _common_calcs(),
    "Manufacturing": _common_calcs(),
    "Retail": _common_calcs(),
    "Financials": _common_calcs(),  # insurers/financials (non-bank)
    "Banking": _banking_calcs(),
    "REITs": _reits_calcs(),
    "Utilities": _utils_telco_calcs(),
    "Energy/Materials": _common_calcs(),
    "Tech": _tech_calcs(),
    "Healthcare": _healthcare_calcs(),
    "Telco": _utils_telco_calcs(),
    "Construction": _construction_calcs(),
    "Plantation": _common_calcs(),
    "Property": _property_calcs(),
    "Transportation/Logistics": _common_calcs(),  # add ASK/RPK later if needed
    "Leisure/Travel": _common_calcs(),
}

def _backlog_coverage(row, prev, ctx):
    ob  = _pick(row, "orderbook")
    rev = _pick(row, "revenue")
    return None if ob is None or rev in (None, 0) else ob / rev

# Add to General so all buckets can use them (others can override)
BUCKET_CALCS.setdefault("General", {}).update({
    # direct passthroughs
    "Order Backlog":        lambda r, p, c: _pick(r, "orderbook"),
    "Capacity Utilization": lambda r, p, c: (
        _pick(r, "capacity_utilization") or _pick(r, "utilization") or _pick(r, "capacity_utilization_pct")
    ),

    # ratio (×)
    "Backlog Coverage (×)": _backlog_coverage,
    "Backlog Coverage (x)": _backlog_coverage,  # glyph variant
})


BUCKET_CALCS["Retail"].update({
    "SSSG (%)": lambda r, p, c: _pick(r, "sssg"),
})

# --- add Leisure/Travel pass-throughs ---
BUCKET_CALCS["Leisure/Travel"].update({
    "Hotel Occupancy (%)": lambda r, p, c: _pick(r, "hotel_occupancy_pct"),
    "RevPAR (RM)":         lambda r, p, c: _pick(r, "revpar"),
    "Win Rate (%)":        lambda r, p, c: _pick(r, "win_rate_pct"),
})

def _transport_calcs() -> Dict[str, Calculator]:
    base = _common_calcs()
    base.update({
        "Load Factor (%)": lambda r, p, c: (
            _pick(r, "load_factor_pct")
            if _pick(r, "load_factor_pct") is not None else
            (_safe_div(_pick(r, "rpk"), _pick(r, "ask")) * 100.0
             if (_pick(r, "rpk") is not None and _pick(r, "ask") not in (None, 0)) else None)
        ),
        "Yield (per km/parcel)": lambda r, p, c: (
            _pick(r, "yield_metric")
            if _pick(r, "yield_metric") is not None else
            _safe_div(_pick(r, "revenue"), _pick(r, "rpk"))
        ),
        "On-Time Performance (%)":  lambda r, p, c: _pick(r, "on_time_pct"),
        "Revenue per TEU (RM/TEU)": lambda r, p, c: (
            _pick(r, "revenue_per_teu")
            if _pick(r, "revenue_per_teu") is not None else
            _safe_div(_pick(r, "revenue"), _pick(r, "teu_throughput"))
        ),
        "Lease-Adj. Net Debt/EBITDA (x)": lambda r, p, c: (lambda e, debt, cash, l1, l2:
            _safe_div(((debt or 0.0) + (l1 or 0.0) + (l2 or 0.0) - (cash or 0.0)), e)
            if (e is not None and e > 0) else None
        )(
            _derive_ebitda(r), _pick(r, "borrowings"), _pick(r, "cash"),
            _pick(r, "lease_liab_current"), _pick(r, "lease_liab_noncurrent")
        ),
    })
    return base

def _energy_materials_calcs() -> Dict[str, Calculator]:
    base = _common_calcs()
    base.update({
        "Lifting Cost (RM/boe)":            lambda r, p, c: _pick(r, "lifting_cost"),
        "Average Selling Price (RM/tonne)": lambda r, p, c: _pick(r, "average_selling_price"),
        "Strip Ratio":                       lambda r, p, c: _pick(r, "strip_ratio"),
        "Head Grade (g/t)":                  lambda r, p, c: _pick(r, "head_grade_gpt"),
        # If you’d like the unitless label to be percent, keep this mapping; otherwise add a (×) variant too:
        "Cash Conversion (CFO/EBITDA)":      _common_calcs()["CFO/EBITDA (%)"],
    })
    return base

def _plantation_calcs() -> Dict[str, Calculator]:
    base = _common_calcs()
    base.update({
        "Gearing (Debt/Assets, %)": lambda r, p, c: _pct_mag(
            _derive_total_debt(r), _pick(r, "total_assets")
        ),

        # --- Operations / Industry KPIs ---
        "FFB Sales Volume (t)": lambda r, p, c: _pick(r, "tonnes_sold"),
        "Unit Cash Cost (RM/t)": lambda r, p, c: _pick(r, "unit_cash_cost"),
        "CPO ASP (RM/t)": lambda r, p, c: _pick(r, "cpo_asp"),

        # Prefer directly-entered KPI; otherwise compute from production inputs (if present)
        "OER (CPO/FFB Input, %)": lambda r, p, c: (
            _pick(r, "oer_pct") or _pct_mag(_pick(r, "cpo_output"), _pick(r, "ffb_input"))
        ),

        # Prefer directly-entered yield; otherwise compute from harvested ÷ (mature area first, else planted)
        "Yield per Hectare (FFB t/ha)": lambda r, p, c: (
            _pick(r, "yield_per_hectare")
            or _safe_div(_pick(r, "ffb_harvested"), (_pick(r, "mature_area") or _pick(r, "planted_hectares")))
        ),

        # Cash margin is sometimes entered directly; optional fallback if both ASP & cash cost are available
        "Cash Margin per ton (RM/t)": lambda r, p, c: (
            _pick(r, "cash_margin_per_ton")
            or (
                (_to_num(_pick(r, "cpo_asp")) - _to_num(_pick(r, "unit_cash_cost")))
                if (_to_num(_pick(r, "cpo_asp")) is not None and _to_num(_pick(r, "unit_cash_cost")) is not None)
                else None
            )
        ),

        # Prefer directly-entered KPI; otherwise compute EBITDA ÷ FFB sales volume (tonnes sold)
        "EBITDA per ton (RM/t)": lambda r, p, c: (
            _pick(r, "ebitda_per_ton") or _safe_div(_derive_ebitda(r), _pick(r, "tonnes_sold"))
        ),
    })
    return base

# --- registry overrides (ensure these run after the function defs above) ---
BUCKET_CALCS["Transportation/Logistics"] = _transport_calcs()
BUCKET_CALCS["Energy/Materials"] = _energy_materials_calcs()
BUCKET_CALCS["Plantation"] = _plantation_calcs()


# Public: build per-year summary table for View page
def _syn_index_for_bucket(bucket: str) -> Dict[str, str]:
    """
    Resolve label synonyms using a centralized, inherited index if available:
      - Prefer config.summary_syn_index(bucket)  (which should merge General + bucket)
      - Fallback to local logic that MERGES General + bucket maps (bucket overrides)
    Returns: dict mapping LOWERCASED label variants -> canonical label.
    """
    idx: Dict[str, str] = {}

    # 1) Preferred: centralized helper that already handles inheritance
    if config is not None and hasattr(config, "summary_syn_index"):
        try:
            merged = config.summary_syn_index(bucket)
            if isinstance(merged, dict):
                # normalize keys to lowercase
                return {str(k).strip().lower(): str(v) for k, v in merged.items()}
        except Exception:
            # fall through to local fallback
            pass

    # 2) Fallback: build a merged map from INDUSTRY_SUMMARY_RATIOS_CATEGORIES
    if config is None:
        return idx

    cats_cfg = (getattr(config, "INDUSTRY_SUMMARY_RATIOS_CATEGORIES", {}) or {})
    general = cats_cfg.get("General") or {}
    bucket_map = cats_cfg.get(bucket) or {}

    def add_items(src: dict):
        # src is {CategoryName: {CanonicalLabel: [synonyms...]}}
        for _cat, items in (src or {}).items():
            if not isinstance(items, dict):
                continue
            for canonical, syns in items.items():
                canon = str(canonical).strip()
                idx[canon.lower()] = canon

                # strip any trailing unit token in parentheses, e.g. (RM), (RM/t), (years)
                base = re.sub(r"\s*\([^)]*\)$", "", canon).strip()
                idx[base.lower()] = canon

                # tolerate (%) and (x)/(×) on either the base or canon
                variants = {
                    base.replace(" (%)", ""),  base + " (%)",
                    base.replace(" (x)",  ""),  base + " (x)",
                    base.replace(" (×)", ""),  base + " (×)",
                }
                for v in variants:
                    idx[str(v).strip().lower()] = canon


                for s in (syns or []):
                    idx[str(s).strip().lower()] = canon

    # merge: General first, then bucket overrides
    add_items(general)
    add_items(bucket_map)

    # extra hard-coded fallbacks
    extras = {
        "p/e": "P/E (×)", "p/b": "P/B (×)", "eps": "EPS (RM)",
        "dividend yield": "Dividend Yield (%)",
        "receivables days": "Receivable Days", "payables days": "Payable Days",
        "capex/revenue": "Capex to Revenue (%)",
        "capex intensity": "CapEx Intensity (%)",
        "net debt / ebitda": "Net Debt / EBITDA (×)",
        "operating cash flow": "CFO",
        "free cash flow": "FCF",
        "orderbook": "Orderbook",
        "nav per unit": "NAV per Unit",
    }
    for k, v in extras.items():
        idx[k] = v

    return idx

def _canon_metric(label: str, bucket: str) -> Optional[str]:
    idx = _syn_index_for_bucket(bucket)
    if not label:
        return None
    s = str(label).strip().lower()
    if s in idx:
        return idx[s]
    for v in [s.replace(" (%)",""), s + " (%)", s.replace(" (x)",""), s + " (x)", s.replace(" (×)",""), s + " (×)"]:
        if v in idx:
            return idx[v]
    return None

def _lookup_calc(calcs: Dict[str, "Calculator"], label: str) -> Optional["Calculator"]:
    """Find a calculator by label; tolerant to (x) vs (×)."""
    if not label:
        return None
    fn = calcs.get(label)
    if fn:
        return fn
    # swap (x) <-> (×)
    if "(x)" in label:
        fn = calcs.get(label.replace("(x)", "(×)"))
        if fn:
            return fn
    if "(\u00d7)" in label or "(×)" in label:
        fn = calcs.get(label.replace("(×)", "(x)"))
        if fn:
            return fn
    return None

# --- Extra helpers -------------------------------------------------------
def epf_yoy_pct(epf_now: float | None, epf_prev: float | None) -> float | None:
    """
    EPF YoY (%) = (epf_now/epf_prev - 1)*100
    Returns None if missing or zero/invalid.
    """
    try:
        if epf_now is None or epf_prev in (None, 0):
            return None
        epf_now = float(epf_now); epf_prev = float(epf_prev)
        if not (math.isfinite(epf_now) and math.isfinite(epf_prev)) or epf_prev == 0:
            return None
        return (epf_now/epf_prev - 1.0) * 100.0
    except Exception:
        return None

def _append_eps_yoy_and_payout_rows(pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Adds/updates TTM-only rows inside the Summary table:
      • EPS YoY (%)        = (EPS_TTM / EPS_lastFY - 1) * 100
      • Revenue YoY (%)    = (Revenue_TTM / Revenue_lastFY - 1) * 100
      • Payout Ratio (%)   = Dividend Yield (%)_TTM × P/E (×)_TTM
    If inputs are missing, leaves things untouched.
    """
    df = pivot.copy()
    if df is None or df.empty or "Metric" not in df.columns:
        return df

    # Locate TTM col (e.g. 'TTM 2025') and the last FY col (max int)
    ttm_col = next((c for c in reversed(df.columns)
                    if isinstance(c, str) and c.upper().startswith("TTM")), None)
    fy_cols = [c for c in df.columns if isinstance(c, (int, np.integer))]
    fy_col  = max(fy_cols) if fy_cols else None
    if not ttm_col:
        return df

    def _ensure_row(metric: str, category: str):
        mask = df["Metric"].astype(str) == metric
        if mask.any():
            return mask
        new = {c: np.nan for c in df.columns}
        new["Category"] = category
        new["Metric"]   = metric
        df.loc[len(df)] = new
        return df["Metric"].astype(str) == metric

    # ---- EPS YoY (%) ----
    eps_yoy = np.nan
    try:
        eps_row = df[df["Metric"].astype(str).str.lower() == "eps (rm)"]
        eps_ttm = pd.to_numeric(eps_row[ttm_col], errors="coerce").iloc[0] if not eps_row.empty else np.nan
        eps_fy  = pd.to_numeric(eps_row[fy_col],  errors="coerce").iloc[0] if (fy_col is not None and not eps_row.empty) else np.nan
        if pd.notna(eps_ttm) and pd.notna(eps_fy) and float(eps_fy) != 0:
            eps_yoy = (float(eps_ttm)/float(eps_fy) - 1.0) * 100.0
    except Exception:
        pass
    if pd.notna(eps_yoy):
        mask = _ensure_row("EPS YoY (%)", "Returns")
        df.loc[mask, ttm_col] = float(eps_yoy)

    # ---- Revenue YoY (%) ----
    # Looks for a base Revenue row (common labels: Revenue/Sales/Total Revenue).
    rev_yoy = np.nan
    try:
        rev_row = df[
            df["Metric"].astype(str).str.lower().isin(
                ["revenue", "sales", "total revenue"]
            )
        ]
        rev_ttm = pd.to_numeric(rev_row[ttm_col], errors="coerce").iloc[0] if not rev_row.empty else np.nan
        rev_fy  = pd.to_numeric(rev_row[fy_col],  errors="coerce").iloc[0] if (fy_col is not None and not rev_row.empty) else np.nan
        if pd.notna(rev_ttm) and pd.notna(rev_fy) and float(rev_fy) != 0:
            rev_yoy = (float(rev_ttm) / float(rev_fy) - 1.0) * 100.0
    except Exception:
        pass
    if pd.notna(rev_yoy):
        mask = _ensure_row("Revenue YoY (%)", "Returns")  # change to "Growth" if preferred
        df.loc[mask, ttm_col] = float(rev_yoy)

    # ---- Payout Ratio (%) (TTM) = DY% × P/E ----
    payout = np.nan
    try:
        dy_row = df[df["Metric"].astype(str).str.lower() == "dividend yield (%)"]
        pe_row = df[df["Metric"].astype(str).str.lower().isin(["p/e (×)", "p/e (x)"])]
        dy = pd.to_numeric(dy_row[ttm_col], errors="coerce").iloc[0] if not dy_row.empty else np.nan
        pe = pd.to_numeric(pe_row[ttm_col], errors="coerce").iloc[0] if not pe_row.empty else np.nan
        if pd.notna(dy) and pd.notna(pe):
            payout = float(dy) * float(pe)
    except Exception:
        pass
    if pd.notna(payout):
        mask = _ensure_row("Payout Ratio (%)", "Valuation")
        df.loc[mask, ttm_col] = float(payout)

    return df

def build_summary_table(
    annual_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    *,
    bucket: str,
    include_ttm: bool = True,
    price_fallback: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute all summary metrics for the given bucket.

    Returns a DataFrame like:
        columns = ["Category", "Metric", <year1>, <year2>, ..., "TTM?"]
        values = floats or NaN

    The metric list + order should follow your View page's category layout.
    """
    if annual_df is None:
        annual_df = pd.DataFrame()
    if quarterly_df is None:
        quarterly_df = pd.DataFrame()

    # Select calculators for bucket
    calcs = BUCKET_CALCS.get(bucket, _common_calcs())

    # Establish metric list & preferred order from config if available
    ordered_metrics: List[Tuple[str, str]] = []  # (Category, Metric)
    if config is not None and hasattr(config, "INDUSTRY_SUMMARY_RATIOS_CATEGORIES"):
        cat_map = config.INDUSTRY_SUMMARY_RATIOS_CATEGORIES.get(bucket) or config.INDUSTRY_SUMMARY_RATIOS_CATEGORIES.get("General", {})
        # flatten with your preferred order (SUMMARY_RATIO_CATEGORY_ORDER in your code base)
        order = getattr(config, "SUMMARY_RATIO_CATEGORY_ORDER", [])
        seen = set()
        for cat in (order or list(cat_map.keys())):
            items = cat_map.get(cat, [])
            for metric in items:
                label = metric if isinstance(metric, str) else str(metric)
                if label not in seen:
                    ordered_metrics.append((cat, label))
                    seen.add(label)
        # also include any calculators not in config (as a safety net)
        for m in calcs.keys():
            if m not in seen:
                ordered_metrics.append(("Other", m))
                seen.add(m)
    else:
        # no config fallback: just dump calculators under "General"
        ordered_metrics = [("General", m) for m in calcs.keys()]

    # Make sure annual is sorted by Year and drop rows without a valid Year value
    if not annual_df.empty and "Year" in annual_df.columns:
        annual_df = annual_df.copy()
        annual_df["Year"] = pd.to_numeric(annual_df["Year"], errors="coerce")
        annual_df = annual_df.dropna(subset=["Year"])
        annual_df["Year"] = annual_df["Year"].astype(int)        
        annual_df = annual_df.drop_duplicates(subset=["Year"], keep="last").sort_values("Year")
    years: List[int] = []
    if "Year" in annual_df.columns and not annual_df.empty:
        years = annual_df["Year"].astype(int).tolist()

    # Prepare per-year rows
    records: List[dict] = []

    # For each year, compute metrics
    ctx = {"price_fallback": price_fallback}
    for idx, (i, r) in enumerate(annual_df.iterrows()):
        year_val = r.get("Year")
        try:
            year_int = int(year_val)
        except (TypeError, ValueError):
            continue  # skip rows without a concrete financial year        
        row = r.to_dict()
        prev_row = annual_df.iloc[idx - 1].to_dict() if idx > 0 else None

        # ensure common derivatives are present
        row = enrich_auto_raw_row_inplace(row, bucket=bucket)

        for cat, metric in ordered_metrics:
            canon = _canon_metric(metric, bucket) or metric
            fn = _lookup_calc(calcs, canon) or _lookup_calc(calcs, metric)
            if fn is None:
                continue
            val = fn(row, prev_row, ctx)
            records.append({"Category": cat, "Metric": metric, "Year": year_int, "Value": val})

    out = pd.DataFrame.from_records(records)
    if out.empty:
        out = pd.DataFrame(columns=["Category", "Metric"] + ([*years] if years else []))
        return out

    # Pivot to wide (years across)
    pivot = out.pivot_table(index=["Category", "Metric"], columns="Year", values="Value", aggfunc="last")

    # Optionally compute TTM column from quarterly
    if include_ttm and not quarterly_df.empty:
        # try to build a raw TTM row (flows summed)
        raw_ttm = ttm_raw_row_from_quarters(quarterly_df, current_price=price_fallback)
        if raw_ttm:
            enrich_three_fees_inplace(raw_ttm)
            enrich_auto_raw_row_inplace(raw_ttm, bucket=bucket)
            # Small synthetic prev for averages (use last annual)
            prev_row = annual_df.iloc[-1].to_dict() if not annual_df.empty else None
            # NEW: fallback Shares for TTM if quarterly didn’t provide them
            if raw_ttm.get("Shares") in (None, 0) and prev_row:
                raw_ttm["Shares"] = _pick(prev_row, "shares")

        # --- BACK-FILL / OVERRIDE TTM USING LATEST ANNUAL WHEN IT'S NEWER THAN QUARTERLIES ---
        if prev_row:
            # Determine "freshness": compare latest Quarterly year vs latest Annual year
            try:
                last_q_year = int(pd.to_numeric(quarterly_df.get("Year"), errors="coerce").dropna().iloc[-1])
            except Exception:
                last_q_year = None
            try:
                last_a_year = int(_to_num(prev_row.get("Year")) or int(annual_df["Year"].dropna().astype(int).iloc[-1]))
            except Exception:
                last_a_year = None

            # 1) Normal back-fill for missing fields we expect on UI (no override here)
            for k in [
                # banking flows we show on the TTM cards
                "NII (incl Islamic)", "Operating Income", "Operating Expenses", "Provisions",
                # inputs useful for NIM fallback
                "Interest Income", "Interest Expense", "Net Islamic Income", "Earning Assets",
            ]:
                if raw_ttm.get(k) is None and (k in prev_row) and (_to_num(prev_row.get(k)) is not None):
                    raw_ttm[k] = _to_num(prev_row.get(k))

            # 2) Capital ratios: override quarterly values if Annual is newer (e.g., FY2025 vs Q4'24)
            _cap_keys = [("CET1 Ratio", "CET1 Ratio (%)"),
                         ("Tier 1 Ratio", "Tier 1 Capital Ratio (%)"),
                         ("Total Capital Ratio", "Total Capital Ratio (%)")]

            for k_plain, k_label in _cap_keys:
                ann_v = _to_num(prev_row.get(k_plain))
                if ann_v is None:
                    continue
                has_q_val = (_to_num(raw_ttm.get(k_label)) is not None) or (_to_num(raw_ttm.get(k_plain)) is not None)

                # If quarterly has no value -> adopt annual
                if not has_q_val:
                    raw_ttm[k_plain] = ann_v
                    continue

                # Quarterly exists but is older than annual -> prefer newer Annual
                if (last_a_year is not None) and (last_q_year is not None) and (last_a_year > last_q_year):
                    raw_ttm[k_plain] = ann_v

            # (keep going — next lines already in your file)
            ctx = {"price_fallback": price_fallback or _last(annual_df.get("Price"))}
            ttm_vals = {}
            for cat, metric in ordered_metrics:
                canon = _canon_metric(metric, bucket) or metric
                fn = _lookup_calc(calcs, canon) or _lookup_calc(calcs, metric)
                if fn is None:
                    continue
                ttm_vals[(cat, metric)] = fn(raw_ttm, prev_row, ctx)

            # inject as "TTM" column
            ttm_series = pd.Series(ttm_vals)
            ttm_series.index = pd.MultiIndex.from_tuples(ttm_series.index, names=["Category", "Metric"])
            pivot["TTM"] = ttm_series


    pivot = pivot.sort_index(level=[0, 1])
    pivot = pivot.reset_index()

    # Order columns as Category, Metric, years..., TTM
    cols = ["Category", "Metric"] + [c for c in pivot.columns if isinstance(c, (int, np.integer))]
    if "TTM" in pivot.columns:
        cols = cols + ["TTM"]
    pivot = pivot.loc[:, cols]

    # NEW: enrich with EPS YoY (TTM vs last FY) and Payout Ratio (TTM)
    pivot = _append_eps_yoy_and_payout_rows(pivot)

    return pivot


# ======================================================================
# Small utilities often used by pages
# ======================================================================
def last_price_from_annual(annual_df: pd.DataFrame) -> Optional[float]:
    if annual_df is None or annual_df.empty or "Price" not in annual_df.columns:
        return None
    return _last(annual_df["Price"])

def last_year(annual_df: pd.DataFrame) -> Optional[int]:
    if annual_df is None or annual_df.empty or "Year" not in annual_df.columns:
        return None
    ys = pd.to_numeric(annual_df["Year"], errors="coerce").dropna()
    return int(ys.max()) if not ys.empty else None

# ======================================================================
# Extra KPIs: CAGR helpers (FY/TTM), PEG (Graham), Margin of Safety, CF KPIs
# ======================================================================

def _row_from_summary(sum_df: pd.DataFrame, metric_label: str) -> Optional[pd.Series]:
    if sum_df is None or sum_df.empty:
        return None
    hit = sum_df[sum_df["Metric"].astype(str).str.lower() == str(metric_label).lower()]
    return hit.iloc[0] if not hit.empty else None


def _years_in_summary_df(sum_df: pd.DataFrame) -> List[int]:
    if sum_df is None or sum_df.empty:
        return []
    ys = [c for c in sum_df.columns if isinstance(c, (int, np.integer))]
    return sorted(ys)

def cagr_from_summary(
    sum_df: pd.DataFrame,
    metric_label: str,
    years_back: int = 5,
    *,
    end_basis: str = "TTM",  # "TTM" or "FY"
) -> Optional[float]:
    """
    CAGR (decimal) for a metric present in the summary table.
    If end_basis="TTM" and TTM exists, uses TTM as the end point; otherwise last FY.
    FY window:     start = years[-(N+1)]  (needs N+1 FY columns)
    TTM window:    start = years[-N]      (needs N FY columns)
    """
    row = _row_from_summary(sum_df, metric_label)
    if row is None:
        return None

    years = _years_in_summary_df(sum_df)
    if len(years) < 1:
        return None

    # Use TTM as end if requested and available
    is_ttm = (end_basis or "TTM").upper() == "TTM" and ("TTM" in row.index) and (_to_num(row.get("TTM")) is not None)
    v_end = _to_num(row.get("TTM")) if is_ttm else _to_num(row.get(years[-1]))

    if years_back < 1:
        return None

    # Start-year rules differ for FY vs TTM
    if is_ttm:
        # need at least N FY columns, start at years[-N]
        if len(years) < years_back:
            return None
        y0 = years[-years_back]
    else:
        # need at least N+1 FY columns, start at years[-(N+1)]
        if len(years) <= years_back:
            return None
        y0 = years[-(years_back + 1)]

    v0 = _to_num(row.get(y0))

    if v0 in (None, 0) or v_end in (None, 0) or v0 <= 0 or v_end <= 0:
        return None

    return (v_end / v0) ** (1.0 / years_back) - 1.0

def graham_peg(
    sum_df: pd.DataFrame,
    *,
    growth_source_metric: str = "EPS (RM)",
    pe_metric: str = "P/E (×)",
    years_back: int = 5,
    end_basis: str = "TTM",
) -> Optional[float]:
    """
    PEG (Graham flavor): PEG = P/E ÷ (2 × g%), where g% = EPS CAGR in percent.
    Uses EPS CAGR over the requested window (FY or TTM-anchored) and P/E (TTM if available).
    """
    # growth (decimal) then to percent
    g = cagr_from_summary(sum_df, growth_source_metric, years_back, end_basis=end_basis)
    if g is None or g <= 0:
        return None
    g_pct = g * 100.0

    # P/E: prefer TTM if present
    pe_row = _row_from_summary(sum_df, pe_metric)
    if pe_row is None:
        return None
    pe = None
    if "TTM" in pe_row.index and _to_num(pe_row.get("TTM")) is not None:
        pe = _to_num(pe_row.get("TTM"))
    else:
        years = _years_in_summary_df(sum_df)
        if not years:
            return None
        pe = _to_num(pe_row.get(years[-1]))

    if pe in (None, 0):
        return None
    return pe / (2.0 * g_pct) if g_pct > 0 else None


def graham_intrinsic_value_per_share(
    sum_df: pd.DataFrame,
    *,
    eps_metric: str = "EPS (RM)",
    years_back: int = 5,
    end_basis: str = "TTM",
    bond_yield_now: Optional[float] = None,
    base_yield: float = 4.4,
) -> Optional[float]:
    """
    Graham intrinsic value per share:
        IV = EPS × (8.5 + 2g) × (base_yield / bond_yield_now)
    If bond_yield_now is None, uses the classic simplification: IV = EPS × (8.5 + 2g)
    where g = EPS CAGR over the window (g in %).
    """
    eps_row = _row_from_summary(sum_df, eps_metric)
    if eps_row is None:
        return None
    eps = None
    if "TTM" in eps_row.index and _to_num(eps_row.get("TTM")) is not None:
        eps = _to_num(eps_row.get("TTM"))
    else:
        years = _years_in_summary_df(sum_df)
        if not years:
            return None
        eps = _to_num(eps_row.get(years[-1]))

    g = cagr_from_summary(sum_df, eps_metric, years_back, end_basis=end_basis)
    if eps in (None, 0) or g is None:
        return None
    g_pct = g * 100.0

    iv = eps * (8.5 + 2.0 * g_pct)
    if bond_yield_now and bond_yield_now > 0:
        iv *= (base_yield / bond_yield_now)  # full Graham adjustment
    return iv


def price_per_share_from_summary(sum_df: pd.DataFrame) -> Optional[float]:
    """Derive price from P/E × EPS if both TTM available, else None."""
    eps_row = _row_from_summary(sum_df, "EPS (RM)")
    pe_row  = _row_from_summary(sum_df, "P/E (×)")
    if eps_row is None or pe_row is None:
        return None
    eps = _to_num(eps_row.get("TTM")) if "TTM" in eps_row.index else None
    pe  = _to_num(pe_row.get("TTM")) if "TTM" in pe_row.index else None
    if eps not in (None, 0) and pe not in (None, 0):
        return eps * pe
    return None


def margin_of_safety_pct(
    sum_df: pd.DataFrame,
    *,
    years_back: int = 5,
    end_basis: str = "TTM",
    price_fallback: Optional[float] = None,
    bond_yield_now: Optional[float] = None,
) -> Optional[float]:
    """
    Margin of Safety (%) = (IV − Price) / IV × 100
    IV from graham_intrinsic_value_per_share(); Price from P/E×EPS TTM or fallback arg.
    """
    iv = graham_intrinsic_value_per_share(
        sum_df, years_back=years_back, end_basis=end_basis, bond_yield_now=bond_yield_now
    )
    if iv in (None, 0):
        return None

    px = price_per_share_from_summary(sum_df)
    if px in (None, 0):
        px = _to_num(price_fallback)
    if px in (None, 0):
        return None

    return (iv - px) / iv * 100.0

def compute_cashflow_kpis(
    annual_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    *,
    basis: str = "TTM",                 # "TTM" or "FY"
    price_fallback: Optional[float] = None,
    bucket: Optional[str] = None
) -> Dict[str, Optional[float]]:
    """
    Returns CFO, Capex, FCF and common CF ratios.
    Bucket controls EBITDA/Operating Income derivations via enrich_* helpers.
    """
    basis = (basis or "TTM").upper()

    # ---- pick a raw row (TTM from quarters, or latest FY) ----
    if basis == "TTM":
        raw = ttm_raw_row_from_quarters(quarterly_df, current_price=price_fallback) or {}
    else:
        if annual_df is None or annual_df.empty:
            raw = {}
        else:
            a = annual_df.dropna(subset=["Year"]).sort_values("Year").iloc[-1].to_dict()
            raw = dict(a)

    # ---- fill derivatives using the right industry bucket ----
    bucket_guess = bucket or _infer_bucket_for_rows(annual_df) or "General"
    enrich_three_fees_inplace(raw)
    enrich_auto_raw_row_inplace(raw, bucket=bucket_guess)

    # ---- pull the bits we need ----
    cfo     = _to_num(raw.get("CFO"))
    capex   = _to_num(raw.get("Capex"))
    rev     = _to_num(raw.get("Revenue"))
    npf     = _to_num(raw.get("Net Profit"))
    ebitda  = _to_num(raw.get("EBITDA"))
    price   = _to_num(raw.get("Price")) or _to_num(price_fallback)
    shares  = _to_num(raw.get("Shares"))
    mktcap  = (price * shares) if (price and shares) else None

    # ---- compute KPIs ----
    fcf = None
    if cfo is not None or capex is not None:
        fcf = (cfo or 0.0) - abs(capex or 0.0)

    out = {
        "CFO": cfo,
        "Capex": capex,
        "FCF": fcf,
        "FCF Margin (%)":      _pct_mag(fcf, rev),
        "FCF Yield (%)":       _pct_mag(fcf, mktcap),
        "Capex to Revenue (%)": _pct_mag(abs(capex) if capex is not None else None, rev),
        "CFO/EBITDA (%)":      (_safe_div(cfo, ebitda) * 100.0) if (cfo is not None and ebitda not in (None, 0)) else None,
        "Cash Conversion (%)": (_safe_div(fcf, npf) * 100.0) if (fcf is not None and npf not in (None, 0)) else None,
    }
    return out


__all__ = [
    # TTM / raw helpers
    "ttm_raw_row_from_quarters",
    "compute_ttm",
    "ensure_three_fees_columns",
    "ensure_auto_raw_columns",
    "enrich_three_fees_inplace",
    "enrich_auto_raw_row_inplace",

    # Soldier Worm scanners
    "build_soldier_worm_report",
    "build_soldier_worm_ttm_kpis",
    "build_soldier_worm_cagr_kpis",
    "build_soldier_worm_cashflow_kpis",
    "build_soldier_worm_calc_trace",
    "build_soldier_worm_ttm_kpis_trace",
    "build_soldier_worm_cagr_kpis_trace",
    "build_soldier_worm_cashflow_kpis_trace",
    "build_soldier_worm_cagr_calc_trace",

    # Summary table
    "build_summary_table",

    # Extra KPIs & valuation helpers
    "cagr_from_summary",
    "graham_peg",
    "graham_intrinsic_value_per_share",
    "margin_of_safety_pct",
    "compute_cashflow_kpis",

    # Misc
    "last_price_from_annual",
    "last_year",

]

# --- Back-compat: Soldier Worm moved to utils/soldier_worm.py ---
def build_soldier_worm_report(
    annual_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    *,
    bucket: str,
    include_ttm: bool = True,
    price_fallback: Optional[float] = None,
) -> pd.DataFrame:
    from utils import soldier_worm as _sw
    return _sw.build_soldier_worm_report(
        annual_df, quarterly_df, bucket=bucket, include_ttm=include_ttm, price_fallback=price_fallback
    )

def build_soldier_worm_ttm_kpis(
    sum_df: pd.DataFrame,
    annual_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    *,
    bucket: str,
    labels: List[str],
    price_fallback: Optional[float] = None,
) -> pd.DataFrame:
    from utils import soldier_worm as _sw
    return _sw.build_soldier_worm_ttm_kpis(
        sum_df, annual_df, quarterly_df, bucket=bucket, labels=labels, price_fallback=price_fallback
    )

def build_soldier_worm_cagr_kpis(
    sum_df: pd.DataFrame,
    annual_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    *,
    bucket: str,
    labels: List[str],
    years_back: int = 5,
    end_basis: str = "TTM",
    price_fallback: Optional[float] = None,
) -> pd.DataFrame:
    from utils import soldier_worm as _sw
    return _sw.build_soldier_worm_cagr_kpis(
        sum_df, annual_df, quarterly_df, bucket=bucket, labels=labels, years_back=years_back,
        end_basis=end_basis, price_fallback=price_fallback
    )

def build_soldier_worm_cashflow_kpis(
    annual_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    *,
    basis: str = "TTM",
    price_fallback: Optional[float] = None,
) -> pd.DataFrame:
    from utils import soldier_worm as _sw
    return _sw.build_soldier_worm_cashflow_kpis(
        annual_df, quarterly_df, basis=basis, price_fallback=price_fallback
    )

# --- Back-compat: Soldier Worm moved to utils/soldier_worm.py ---
def build_soldier_worm_calc_trace(
    annual_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    *,
    bucket: str,
    include_ttm: bool = True,
    price_fallback: Optional[float] = None,
) -> pd.DataFrame:
    from utils import soldier_worm as _sw
    return _sw.build_soldier_worm_calc_trace(
        annual_df, quarterly_df, bucket=bucket, include_ttm=include_ttm, price_fallback=price_fallback
    )

def build_soldier_worm_ttm_kpis_trace(sum_df, annual_df, quarterly_df, *, bucket, labels, price_fallback=None):
    from utils import soldier_worm as _sw
    return _sw.build_soldier_worm_ttm_kpis_trace(
        sum_df, annual_df, quarterly_df, bucket=bucket, labels=labels, price_fallback=price_fallback
    )

def build_soldier_worm_cashflow_kpis_trace(annual_df, quarterly_df, *, basis="TTM", price_fallback=None):
    from utils import soldier_worm as _sw
    return _sw.build_soldier_worm_cashflow_kpis_trace(
        annual_df, quarterly_df, basis=basis, price_fallback=price_fallback
    )

def build_soldier_worm_cagr_kpis_trace(sum_df, annual_df, quarterly_df, *, bucket, labels, years_back=5, end_basis="TTM", price_fallback=None):
    from utils import soldier_worm as _sw
    return _sw.build_soldier_worm_cagr_kpis_trace(
        sum_df, annual_df, quarterly_df, bucket=bucket, labels=labels,
        years_back=years_back, end_basis=end_basis, price_fallback=price_fallback
    )
    
def build_soldier_worm_cagr_calc_trace(
    sum_df: pd.DataFrame,
    annual_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    *,
    bucket: str,
    labels: list[str],
    years_back: int = 5,
    end_basis: str = "TTM",
    price_fallback: Optional[float] = None,
) -> pd.DataFrame:
    from utils import soldier_worm as _sw
    return _sw.build_soldier_worm_cagr_calc_trace(
        sum_df, annual_df, quarterly_df,
        bucket=bucket, labels=labels, years_back=years_back, end_basis=end_basis,
        price_fallback=price_fallback
    )    