# utils/soldier_worm.py
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- robust config import (root or utils/) ---
try:
    import config  # type: ignore
except Exception:
    try:
        from utils import config  # type: ignore
    except Exception:
        config = None  # type: ignore

# We import the raw helpers / alias maps / calculators from calculations.py
from utils.calculations import (
    RAW_ALIASES,
    Q_ALIASES,
    BUCKET_CALCS,
    _pick,
    _to_num,
    _safe_div,
    _avg2,
    _sum,
    _last,
    _ctx_price,
    _canon_metric,     # keep this in calculations.py so summary can use it too
    _lookup_calc,      # tiny helper added in calculations.py (x vs ×)
    ttm_raw_row_from_quarters,
    enrich_three_fees_inplace,
    enrich_auto_raw_row_inplace,
    _infer_bucket_for_rows,
    cagr_from_summary,
    price_per_share_from_summary,
)

# ========================= Soldier Worm: Missing-Input Diagnostics =========================

def _has(row: dict | None, canon_key: str) -> bool:
    """True if 'canon_key' (canonical or literal) is present in row (number + finite)."""
    if row is None:
        return False
    # previous-year token, e.g. 'equity_prev'
    if canon_key.endswith("_prev"):
        base = canon_key[:-5]
        return _pick(row, base) is not None
    # not a known canonical alias? treat as literal column header (case-insensitive)
    if canon_key not in RAW_ALIASES and canon_key not in Q_ALIASES:
        target = str(canon_key).strip().lower()
        for k in row.keys():
            if str(k).strip().lower() == target:
                return _to_num(row.get(k)) is not None
        return False
    # canonical lookup
    return _pick(row, canon_key) is not None

def _present_ctx_price(row: dict, ctx: dict) -> bool:
    return _ctx_price(row, ctx) is not None

def _canon_label(tok: str) -> str:
    mapping = {
        "equity_prev": "Equity (prev FY)",
        "total_assets_prev": "Total Assets (prev FY)",
        "earning_assets_prev": "Average Earning Assets (prev FY)",
        "price_ctx": "Price (row or fallback)",
    }
    return mapping.get(tok, tok.replace("_", " ").title())

def _missing_from_paths(
    row: dict,
    prev_row: dict | None,
    ctx: dict,
    *,
    paths: list[list[str]],
    is_ttm: bool = False
) -> list[str]:
    """
    Given alternative requirement paths, return missing tokens from the *best* path (fewest missing).
    Tokens can be:
      • canonical keys (RAW_ALIASES/Q_ALIASES)
      • '<key>_prev' (previous FY value)
      • 'price_ctx' (row['Price'] or ctx['price_fallback'])
      • 'a|b' (either a or b acceptable)
    """
    def _has_token(t: str) -> bool:
        if "|" in t:
            alts = [x.strip() for x in t.split("|")]
            return any(_has(row, a) for a in alts)
        if t == "price_ctx":
            return _present_ctx_price(row, ctx)
        if t.endswith("_prev"):
            base = t[:-5]
            return _has(prev_row, base)
        return _has(row, t)

    best_missing: Optional[List[str]] = None
    for path in (paths or []):
        missing: List[str] = []
        for tok in path:
            if not _has_token(tok):
                lab = _canon_label(tok)
                # add gentle hints for TTM dependencies
                if is_ttm and lab.lower() in {
                    "revenue", "gross profit", "operating profit", "ebitda", "net profit",
                    "cfo", "capex", "dps", "nii (incl islamic)", "operating income"
                }:
                    lab += " — need last 4 quarters"
                if is_ttm and lab.lower() in {
                    "equity","total assets","current assets","current liabilities","receivables","inventory",
                    "payables","earning assets","shares","price","total borrowings","cash & cash equivalents",
                    "orderbook", "capacity utilization" 
                }:
                    lab += " — need latest quarter"
                missing.append(lab)
        if best_missing is None or len(missing) < len(best_missing):
            best_missing = missing
    return best_missing or []

def _requirements_catalog(bucket: str) -> dict[str, list[list[str]]]:
    """
    Map Metric Label -> list of acceptable requirement paths (each path is a list of tokens).
    Tokens are canonical raw keys from RAW_ALIASES (e.g., 'revenue', 'equity', 'cfo', etc.),
    or special tokens: '..._prev', 'price_ctx', and OR groups like 'cogs|revenue'.
    """
    common = {
        # Margins (unchanged)
        "Gross Margin (%)":            [["gross_profit","revenue"], ["revenue","cogs"]],
        "EBITDA Margin (%)":           [["ebitda","revenue"], ["operating_profit","dep_amort","revenue"]],
        "Operating Profit Margin (%)": [["operating_profit","revenue"]],
        "Net Margin (%)":              [["net_profit","revenue"]],

        # Returns — prefer average denominator, allow current-only fallback
        "ROE (%)": [["net_profit","equity","equity_prev"],
                    ["net_profit","equity"]],
        "ROA (%)": [["net_profit","total_assets","total_assets_prev"],
                    ["net_profit","total_assets"]],

        # Working-capital / efficiency — prefer average, allow current-only
        "Inventory Turnover (×)": [["cogs|revenue","inventory","inventory_prev"],
                                   ["cogs|revenue","inventory"]],
        "Inventory Days (days)":  [["cogs|revenue","inventory","inventory_prev"],
                                   ["cogs|revenue","inventory"]],
        "Receivable Days (days)":[["revenue","receivables","receivables_prev"],
                                  ["revenue","receivables"]],
        "Payable Days (days)":   [["cogs|revenue","payables","payables_prev"],
                                  ["cogs|revenue","payables"]],
        "Cash Conversion Cycle (days)":[
            ["cogs|revenue","inventory","inventory_prev","receivables","receivables_prev","payables","payables_prev"],
            ["cogs|revenue","inventory","receivables","payables"],
        ],

        # Liquidity & leverage (unchanged)
        "Current Ratio (×)":           [["current_assets","current_liabilities"]],
        "Quick Ratio (×)":             [["current_assets","inventory","current_liabilities"]],
        "Debt/Equity (×)":             [["borrowings","equity"]],
        "Interest Coverage (×)":       [["operating_profit","interest_expense"]],
        "Net Debt / EBITDA (×)":       [["borrowings","cash","ebitda"], ["borrowings","cash","operating_profit","dep_amort"]],
        "Financial Leverage (Assets/Equity)":[["total_assets","equity"]],

        # Cash flow (unchanged)
        "Capex to Revenue (%)":        [["capex","revenue"]],
        "FCF Margin (%)":              [["cfo","capex","revenue"]],
        "CFO/EBITDA (%)":              [["cfo","ebitda"], ["cfo","operating_profit","dep_amort"]],
        "Cash Conversion (%)":         [["cfo","capex","net_profit"]],
        "Three Fees Ratio (%)":        [["revenue","interest_expense"]],
        "Operating CF Margin (%)":     [["cfo","revenue"]],

        # Valuation (unchanged)
        "EPS (RM)":                    [["net_profit","shares"]],
        "P/E (×)":                     [["net_profit","shares","price_ctx"]],
        "P/B (×)":                     [["equity","shares","price_ctx"]],
        "EV/EBITDA (×)":               [["ebitda","shares","price_ctx","borrowings","cash"], ["operating_profit","dep_amort","shares","price_ctx","borrowings","cash"]],
        "EV/Sales (×)":                [["revenue","shares","price_ctx","borrowings","cash"]],
        "Dividend Yield (%)":          [["dps|dpu","price_ctx"]],
        "FCF Yield (%)":               [["cfo","capex","shares","price_ctx"]],
        
        # --- Simple passthrough raw KPIs used in the TTM strip ---
        "Revenue":             [["revenue"]],
        "Gross Profit":        [["gross_profit"]],
        "Operating Profit":    [["operating_profit"]],
        "EBITDA":              [["ebitda"]],
        "Net Profit":          [["net_profit"]],
        "EPS":                 [["net_profit","shares"]],   # simple EPS (TTM)
        "DPS":                 [["dps|dpu"]],

        # Label variants used in config — add the same fallbacks
        "Inventory Days":              [["cogs|revenue","inventory","inventory_prev"],
                                        ["cogs|revenue","inventory"]],
        "Receivables Days (days)":     [["revenue","receivables","receivables_prev"],
                                        ["revenue","receivables"]],
        "Receivables Days":            [["revenue","receivables","receivables_prev"],
                                        ["revenue","receivables"]],
        "Payables Days (days)":        [["cogs|revenue","payables","payables_prev"],
                                        ["cogs|revenue","payables"]],
        "Payables Days":               [["cogs|revenue","payables","payables_prev"],
                                        ["cogs|revenue","payables"]],
        "CapEx Intensity (%)":         [["capex","revenue"]],
        "Cash Conversion (CFO/EBITDA)":[["cfo","ebitda"], ["cfo","operating_profit","dep_amort"]],
        "Capex/Revenue (%)":           [["capex","revenue"]],
        # --- Backlog & capacity (used by multiple buckets / TTM strip) ---
        "Order Backlog":                [["orderbook"]],                    # stock -> latest quarter
        "Backlog Coverage (×)":         [["orderbook","revenue"]],          # coverage = orderbook / revenue
        "Backlog Coverage":             [["orderbook","revenue"]],          # label variant without the × glyph
        "Capacity Utilization":         [["capacity_utilization|utilization|capacity_utilization_pct"]],
        # Growth + distributions
        "EPS YoY (%)":         [["net_profit","shares","net_profit_prev","shares_prev"]],
        "Payout Ratio (%)":    [["dps|dpu","net_profit","shares"]],
    }

    banking = {
        # Income & efficiency

        # --- TTM-strip passthroughs (raw picks) ---
        "NII (incl Islamic)":  [["nii_incl_islamic"]],
        "Operating Income":    [["operating_income"]],
        "Operating Expenses":  [["operating_expenses"]],
        "Provisions":          [["provisions"]],

        "NIM (%)": [
            ["nii_incl_islamic","earning_assets","earning_assets_prev"],
            ["tp_bank_nim_num|nii_incl_islamic","tp_bank_nim_den|earning_assets"],
            ["nim_pct"]  # <— NEW: if a direct NIM% column exists
        ],

        "Operating CF Margin (%)": [
            ["cfo", "operating_income"],  # primary, matches Banking calc
            ["cfo", "nii_incl_islamic", "fee_income", "trading_income", "other_op_income"],  # derive OI if needed
            ["cfo", "revenue"],  # last-resort fallback
        ],

        "Cost-to-Income Ratio (%)":    [["operating_expenses","operating_income"],
                                        ["operating_expenses","nii_incl_islamic","fee_income","trading_income","other_op_income"]],
        "Financial Leverage (×)":      [["total_assets","equity"]],
        # Asset quality
        "NPL Ratio (%)":               [["npl","gross_loans"]],
        "Loan Loss Coverage (×)":      [["llr","npl"]],
        # Capital & liquidity
        "Loan-to-Deposit Ratio (%)":   [["gross_loans","deposits"]],
        "CASA Ratio (%)": [
            ["casa_ratio_pct"],                                # direct override if present
            ["demand_dep","savings_dep","deposits|time_dep"],  # computed fallback
        ],
        "CASA (Core, %)": [
            ["casa_core_pct"],                                 # direct override if present
            ["demand_dep","savings_dep","time_dep"],           # computed fallback
        ],

        # Returns — same fallback idea
        "ROE (%)":                     [["net_profit","equity","equity_prev"],
                                        ["net_profit","equity"]],
        "ROA (%)":                     [["net_profit","total_assets","total_assets_prev"],
                                        ["net_profit","total_assets"]],
        # Valuation
        "P/E (×)":                     [["net_profit","shares","price_ctx"]],
        "P/B (×)":                     [["equity","shares","price_ctx"]],
        "Dividend Yield (%)":          [["dps","price_ctx"]],
        # Direct picks (passthrough)
        "CET1 Ratio (%)":              [["cet1_ratio_pct"]],
        "Tier 1 Capital Ratio (%)":    [["tier1_ratio_pct"]],
        "Total Capital Ratio (%)":     [["total_capital_ratio_pct"]],
        "LCR (%)":                     [["lcr_pct"]],
        "NSFR (%)":                    [["nsfr_pct"]],
        "Loan-to-Deposit Ratio (×)":   [["gross_loans","deposits"]],
        "Loan-Loss Coverage (×)":      [["llr","npl"]],
        "Financial Leverage (Assets/Equity)":[["total_assets","equity"]],
    }

    # Merge by bucket (unchanged)
    if bucket == "Banking":
        base = {**common, **banking}
    elif bucket == "REITs":
        base = {**common, **{
            "Gearing (x)":                 [["borrowings","equity"]],
            "Gearing (Debt/Assets, %)":    [["borrowings","total_assets"]],
            "Interest Coverage (×)":       [["ebitda","interest_expense"], ["operating_profit","dep_amort","interest_expense"]],
            "Average Cost of Debt (%)": [["interest_expense","borrowings"]],
            "Occupancy (%)":               [["occupancy"]],
            "WALE (years)":                [["wale"]],
            "Rental Reversion (%)":        [["rental_reversion"]],
            "DPU (RM)":                    [["dpu|dps"]],
            "Distribution Yield (%)":      [["dpu|dps","price_ctx"]],
            "NAV per Unit (RM)":           [["equity","shares"]],
            "P/NAV (×)":                   [["equity","shares","price_ctx"]],
        }}
    elif bucket in ("Utilities","Telco"):
        base = {**common, **{"Dividend Cash Coverage (CFO/Div)":[["cfo","dps","shares"]]}}
    elif bucket == "Healthcare":
        base = {**common, **{
            "Bed Occupancy (%)":           [["bed_occupancy_pct"]],
            "ALOS (days)":                 [["alos_days"], ["patient_days","admissions"]],
            "Bed Turnover (admissions/bed)":[["admissions","beds"]],
            "Revenue per Bed (RM)":        [["revenue","beds"]],
            "EBITDA per Bed (RM)":         [["ebitda","beds"], ["operating_profit","dep_amort","beds"]],
            "Revenue per Patient Day (RM)":[["revenue","patient_days"]],
            "EBITDA per Patient Day (RM)":[["ebitda","patient_days"], ["operating_profit","dep_amort","patient_days"]],
            "R&D Intensity (%)":           [["rnd","revenue"]],
        }}
    elif bucket == "Construction":
        base = {**common, **{
            "Win Rate (%)":                [["new_orders","tender_book"]],
            "Backlog Coverage (×)":        [["orderbook","revenue"]],
        }}
    else:
        base = dict(common)

    # Add (×)/(x) label variants (unchanged)
    for k in list(base.keys()):
        if "(×)" in k and k.replace("(×)","(x)") not in base:
            base[k.replace("(×)","(x)")] = base[k]
        if "(x)" in k and k.replace("(x)","(×)") not in base:
            base[k.replace("(x)","(×)")] = base[k]
    return base

def build_soldier_worm_report(
    annual_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    *,
    bucket: str,
    include_ttm: bool = True,
    price_fallback: Optional[float] = None,
) -> pd.DataFrame:
    """
    Crawl Annual rows (by FY) + optional TTM (from quarters) and
    list missing raw inputs for metrics that failed to compute.
    Returns: ['Category','Metric','Period','Missing Inputs'].
    """
    if annual_df is None:
        annual_df = pd.DataFrame()
    if quarterly_df is None:
        quarterly_df = pd.DataFrame()

    calcs = BUCKET_CALCS.get(bucket, BUCKET_CALCS.get("General", {}))

    # Metric order mirrors the View summary categories from config (plus safety-net extras)
    ordered_metrics: List[Tuple[str, str]] = []
    if config is not None and hasattr(config, "INDUSTRY_SUMMARY_RATIOS_CATEGORIES"):
        cat_map = config.INDUSTRY_SUMMARY_RATIOS_CATEGORIES.get(bucket) or \
                  config.INDUSTRY_SUMMARY_RATIOS_CATEGORIES.get("General", {})
        order = getattr(config, "SUMMARY_RATIO_CATEGORY_ORDER", [])
        seen = set()
        for cat in (order or list(cat_map.keys())):
            items = cat_map.get(cat, [])
            for metric in items:
                label = metric if isinstance(metric, str) else str(metric)
                if label not in seen:
                    ordered_metrics.append((cat, label))
                    seen.add(label)
        # ---- control whether to include "extras" not in your config ----
        include_extras = bool(getattr(config, "SOLDIER_WORM_INCLUDE_EXTRAS", False))
        exclude = set(getattr(config, "SOLDIER_WORM_EXCLUDE", []))
        if include_extras:
            for m in calcs.keys():
                if m not in seen and m not in exclude:
                    ordered_metrics.append(("Other", m))
                    seen.add(m)
        # If include_extras == False (default), we DON'T add unknown metrics.

    else:
        ordered_metrics = [("General", m) for m in calcs.keys()]

    # --- Ensure critical Banking direct-picks are scanned even if config omits them ---
    if bucket == "Banking":
        must_haves = [
            ("Capital & Liquidity", "NSFR (%)"),
            ("Capital & Liquidity", "LCR (%)"),
            ("Capital & Liquidity", "CET1 Ratio (%)"),
            ("Capital & Liquidity", "Tier 1 Capital Ratio (%)"),
            ("Capital & Liquidity", "Total Capital Ratio (%)"),
        ]
        present = {(c, m) for (c, m) in ordered_metrics}
        for item in must_haves:
            if item not in present:
                ordered_metrics.append(item)

    reqs = _requirements_catalog(bucket)
    rows: List[dict] = []

    # ---------- Annual FY scan ----------
    if not annual_df.empty and "Year" in annual_df.columns:
        A = annual_df.drop_duplicates(subset=["Year"], keep="last").sort_values("Year")
        ctx = {"price_fallback": price_fallback}
        for idx, (_, r) in enumerate(A.iterrows()):
            row = enrich_auto_raw_row_inplace(r.to_dict(), bucket=bucket)
            prev = A.iloc[idx-1].to_dict() if idx > 0 else None
            for cat, metric in ordered_metrics:
                canon = _canon_metric(metric, bucket) or metric
                fn = _lookup_calc(calcs, canon) or _lookup_calc(calcs, metric)
                if fn is None:
                    # No calculator? still report what's missing (from requirements)
                    paths = reqs.get(canon, []) or reqs.get(metric, [])
                    if paths:
                        miss = _missing_from_paths(row, prev, ctx, paths=paths, is_ttm=False)
                        if miss:
                            rows.append({
                                "Category": cat,
                                "Metric": metric,
                                "Period": int(r.get("Year")),
                                "Missing Inputs": ", ".join(sorted(dict.fromkeys(miss))),
                            })
                    continue
                try:
                    val = fn(row, prev, ctx)
                except Exception:
                    val = None
                if val is None:
                    paths = reqs.get(canon, []) or reqs.get(metric, [])
                    miss = _missing_from_paths(row, prev, ctx, paths=paths, is_ttm=False)
                    if miss:
                        rows.append({
                            "Category": cat,
                            "Metric": metric,
                            "Period": int(r.get("Year")),
                            "Missing Inputs": ", ".join(sorted(dict.fromkeys(miss))),
                        })

    # ---------- TTM scan (from quarters) ----------
    if include_ttm and not quarterly_df.empty:
        raw_ttm = ttm_raw_row_from_quarters(quarterly_df, current_price=price_fallback) or {}
        if raw_ttm:
            enrich_three_fees_inplace(raw_ttm)
            enrich_auto_raw_row_inplace(raw_ttm, bucket=bucket)
            prev = (annual_df.dropna(subset=["Year"]).sort_values("Year").iloc[-1].to_dict()
                    if ("Year" in annual_df.columns and not annual_df.empty) else None)
            ctx = {"price_fallback": price_fallback}
            for cat, metric in ordered_metrics:
                canon = _canon_metric(metric, bucket) or metric
                fn = _lookup_calc(calcs, canon) or _lookup_calc(calcs, metric)
                if fn is None:
                    paths = reqs.get(canon, []) or reqs.get(metric, [])
                    if paths:
                        miss = _missing_from_paths(raw_ttm, prev, ctx, paths=paths, is_ttm=True)
                        if miss:
                            rows.append({
                                "Category": cat,
                                "Metric": metric,
                                "Period": "TTM",
                                "Missing Inputs": ", ".join(sorted(dict.fromkeys(miss))),
                            })
                    continue
                try:
                    val = fn(raw_ttm, prev, ctx)
                except Exception:
                    val = None
                if val is None:
                    paths = reqs.get(canon, []) or reqs.get(metric, [])
                    miss = _missing_from_paths(raw_ttm, prev, ctx, paths=paths, is_ttm=True)
                    if miss:
                        rows.append({
                            "Category": cat,
                            "Metric": metric,
                            "Period": "TTM",
                            "Missing Inputs": ", ".join(sorted(dict.fromkeys(miss))),
                        })

    out = pd.DataFrame(rows, columns=["Category","Metric","Period","Missing Inputs"])
    if out.empty:
        return pd.DataFrame(columns=["Category","Metric","Period","Missing Inputs"])

    def _period_key(p):
        if isinstance(p, (int, np.integer)):
            return (0, int(p))
        return (1, 999999)  # TTM after FYs

    out = out.sort_values(by=["Category","Metric","Period"], key=lambda s: s.map(_period_key))
    return out.reset_index(drop=True)

# ========================= Soldier Worm: KPI diagnostics (TTM, CAGR, Cash Flow) =========================

def build_soldier_worm_ttm_kpis(
    sum_df: pd.DataFrame,
    annual_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    *,
    bucket: str,
    labels: List[str],
    price_fallback: Optional[float] = None,
) -> pd.DataFrame:
    """
    For each TTM KPI label, report missing raw inputs required to compute it (TTM context).
    Uses the metric requirements catalog + ttm_raw_row_from_quarters.
    """
    reqs = _requirements_catalog(bucket)
    rows: List[dict] = []

    # Raw TTM row (flows summed, stocks last)
    raw_ttm = ttm_raw_row_from_quarters(quarterly_df, current_price=price_fallback) or {}
    if not raw_ttm:
        for lab in (labels or []):
            rows.append({"Metric": lab, "Period": "TTM", "Missing Inputs": "Quarterly data (last 4 quarters) not available"})
        return pd.DataFrame(rows, columns=["Metric","Period","Missing Inputs"])

    enrich_three_fees_inplace(raw_ttm)
    enrich_auto_raw_row_inplace(raw_ttm, bucket=bucket)
    prev = (annual_df.dropna(subset=["Year"]).sort_values("Year").iloc[-1].to_dict()
            if (annual_df is not None and not annual_df.empty and "Year" in annual_df.columns) else None)
    ctx  = {"price_fallback": price_fallback}

    for lab in (labels or []):
        # normalize a couple of common UI slips before canonicalizing
        _norm = str(lab).strip().lower()
        if _norm == "operating":
            lab = "Operating Profit"
        elif _norm == "ebidta":   # common typo
            lab = "EBITDA"

        # be tolerant: try canonical map, then synonym resolver, then raw label
        canon = _canon_metric(lab, bucket) or _canon_from_syn(lab, bucket) or lab
        paths = reqs.get(canon, []) or reqs.get(lab, [])


        # NEW: Surface KPIs we forgot to define requirements for
        if not paths:
            rows.append({
                "Metric": lab,
                "Period": "TTM",
                "Missing Inputs": "No path available",
            })
            continue

        miss = _missing_from_paths(raw_ttm, prev, ctx, paths=paths, is_ttm=True)

        if miss:
            rows.append({
                "Metric": lab,
                "Period": "TTM",
                "Missing Inputs": ", ".join(sorted(dict.fromkeys(miss))),
            })
    return pd.DataFrame(rows, columns=["Metric","Period","Missing Inputs"])

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
    """
    Align CAGR diagnostics with the UI:
      • Try summary row first;
      • If not available, fall back to annual raw series (Revenue/EBITDA/Net Profit/CFO/FCF/Orderbook);
      • For CFO/FCF, use annual series with a TTM end override when end_basis='TTM';
      • Emit missing messages only when inputs are truly absent/≤0, and show specific TTM input hints.
    """
    end_basis = (end_basis or "TTM").upper()
    rows: List[dict] = []
    reqs = _requirements_catalog(bucket)

    # --- helpers for summary table ---
    def _ttm_col_name(df: pd.DataFrame) -> Optional[str]:
        for c in (df.columns if isinstance(df, pd.DataFrame) else []):
            if isinstance(c, str) and c.upper().startswith("TTM"):
                return c
        return "TTM" if (isinstance(df, pd.DataFrame) and "TTM" in df.columns) else None

    def _years_in_summary(df: pd.DataFrame) -> List[int]:
        if df is None or df.empty: return []
        ys = [c for c in df.columns if isinstance(c, (int, np.integer))]
        return sorted(ys)

    def _find_row(label: str) -> Optional[pd.Series]:
        if sum_df is None or sum_df.empty: return None
        hit = sum_df[sum_df["Metric"].astype(str).str.lower() == str(label).lower()]
        return hit.iloc[0] if not hit.empty else None

    # Map "<Metric> CAGR" base labels to the summary/raw labels you actually use
    def _base_label_for_cagr(name: str) -> str:
        base = str(name).replace(" CAGR","").strip()
        mapping = {
            "Operating Cash Flow": "CFO",
            "Free Cash Flow": "FCF",
            "NAV per Unit": "NAV per Unit",
            "Orderbook": "Orderbook",
        }
        # also honor the bucket canonicalizer
        canon = _canon_metric(base, bucket)
        return mapping.get(base, canon or base)

    # --- helpers for using annual raw series ---
    def _series_from_annual_for_canon(canon_key: str) -> Optional[pd.Series]:
        A = annual_df
        if A is None or A.empty or "Year" not in A.columns:
            return None
        cols = RAW_ALIASES.get(canon_key, [])
        for col in cols:
            if col in A.columns:
                s = pd.to_numeric(A[col], errors="coerce")
                yrs = pd.to_numeric(A["Year"], errors="coerce")
                if s.notna().sum() >= 2 and yrs.notna().sum() >= 2:
                    return pd.Series({int(y): float(v) for y, v in zip(yrs, s)})
        return None

    def _series_from_annual(label: str) -> Optional[pd.Series]:
        # label is UI/base label (Revenue/EBITDA/Net Profit/CFO/FCF/Orderbook)
        if label == "FCF":
            A = annual_df
            if A is None or A.empty or "Year" not in A.columns:
                return None
            cfo = None
            for k in RAW_ALIASES.get("cfo", []):
                if k in A.columns:
                    cfo = pd.to_numeric(A[k], errors="coerce")
                    break
            capex = None
            for k in RAW_ALIASES.get("capex", []):
                if k in A.columns:
                    capex = pd.to_numeric(A[k], errors="coerce")
                    break
            if cfo is None or capex is None:
                return None
            yrs = pd.to_numeric(A["Year"], errors="coerce")
            if cfo.notna().sum() >= 2 and capex.notna().sum() >= 2 and yrs.notna().sum() >= 2:
                fcf = cfo - capex.abs()  # use magnitude of capex
                return pd.Series({int(y): float(v) for y, v in zip(yrs, fcf)})

            return None

        canon_map = {
            "Revenue": "revenue",
            "EBITDA": "ebitda",
            "Net Profit": "net_profit",
            "CFO": "cfo",
            "Orderbook": "orderbook",
            "Gross Loans": "gross_loans",
            "Deposits": "deposits",
        }
        canon = canon_map.get(label)
        if not canon:
            return None
        return _series_from_annual_for_canon(canon)

    def _cagr_from_summary_row(row: pd.Series, years: List[int], N: int,
                                ttm_col: Optional[str]) -> Optional[float]:
        if row is None or not years or len(years) < 1:
            return None

        # availability: TTM needs N FY cols, FY needs N+1 FY cols
        need = (len(years) < N) if end_basis == "TTM" else (len(years) <= N)
        if need:
            return None

        # end value
        is_ttm = (end_basis == "TTM" and ttm_col and ttm_col in row and _to_num(row.get(ttm_col)) is not None)
        v_end = _to_num(row.get(ttm_col)) if is_ttm else _to_num(row.get(years[-1]))

        # start value (different anchor for TTM vs FY)
        y0 = years[-N] if is_ttm else years[-(N + 1)]
        v0 = _to_num(row.get(y0))

        if v0 in (None, 0) or v_end in (None, 0) or v0 <= 0 or v_end <= 0:
            return None
        return (v_end / v0) ** (1.0 / N) - 1.0

    def _cagr_from_annual_series(label: str, N: int, ttm_override: Optional[float]) -> Optional[float]:
        s = _series_from_annual(label)
        if s is None or s.dropna().size < 2:
            return None

        yrs = sorted([int(x) for x in s.index if isinstance(x, (int, np.integer))])

        # availability: TTM needs N FY points, FY needs N+1 FY points
        need = (len(yrs) < N) if end_basis == "TTM" else (len(yrs) <= N)
        if need:
            return None

        # end value
        v_end = float(ttm_override) if (end_basis == "TTM" and ttm_override is not None) \
                else float(s.get(yrs[-1], np.nan))

        # start value (different anchor)
        y0 = yrs[-N] if end_basis == "TTM" else yrs[-(N + 1)]
        v0 = float(s.get(y0, np.nan))

        if not (math.isfinite(v_end) and math.isfinite(v0)) or v0 <= 0 or v_end <= 0:
            return None
        return (v_end / v0) ** (1.0 / N) - 1.0

    # --- build TTM raw for diagnostics / overrides ---
    raw_ttm = ttm_raw_row_from_quarters(quarterly_df, current_price=price_fallback) or {}
    if raw_ttm:
        enrich_three_fees_inplace(raw_ttm)
        enrich_auto_raw_row_inplace(raw_ttm, bucket=bucket)
    prev = (annual_df.dropna(subset=["Year"]).sort_values("Year").iloc[-1].to_dict()
            if (annual_df is not None and not annual_df.empty and "Year" in annual_df.columns) else None)
    ctx = {"price_fallback": price_fallback}

    # handy TTM overrides
    ttm_cfo   = _to_num(raw_ttm.get("CFO")) if raw_ttm else None
    ttm_capex = _to_num(raw_ttm.get("Capex")) if raw_ttm else None
    ttm_fcf   = (
        None
        if (ttm_cfo is None and ttm_capex is None)
        else ((ttm_cfo or 0.0) - (abs(ttm_capex) if ttm_capex is not None else 0.0))
    )


    years = _years_in_summary(sum_df) if (sum_df is not None and not sum_df.empty) else []
    ttm_col = _ttm_col_name(sum_df) if (sum_df is not None and not sum_df.empty) else None
    N = int(years_back)
    
    # map UI base label -> canonical raw key (for alias-aware _pick)
    def _canon_for_base(base_label: str) -> Optional[str]:
        m = {
            "Revenue": "revenue",
            "EBITDA": "ebitda",
            "Net Profit": "net_profit",
            "CFO": "cfo",
            "Orderbook": "orderbook",
            "Gross Loans": "gross_loans",
            "Deposits": "deposits",
            "FCF": None,  # handled separately
            "Free Cash Flow": None,
        }
        return m.get(base_label)
  
    # requirements helper for simple raw tokens (for neat TTM messages)
    def _paths_for_base_label(base_label: str) -> List[List[str]]:
        # use RAW tokens so _missing_from_paths can add "— need last 4 quarters"/"— need latest quarter"
        tok = {
            "Revenue": "revenue",
            "EBITDA": "ebitda",
            "Net Profit": "net_profit",
            "CFO": "cfo",
            "Orderbook": "orderbook",
            # NEW — balances (stock items) → latest quarter
            "Gross Loans": "gross_loans",
            "Deposits": "deposits",
        }.get(base_label)
        if base_label in ("FCF", "Free Cash Flow"):
            return [["cfo", "capex"]]
        return [[tok]] if tok else []


    def _emit(name: str, msg: str):
        rows.append({"Metric": name, "Period": f"{N}y {end_basis}", "Missing Inputs": msg})

    # ---------------- main scan ----------------
    for lab in (labels or []):
        L = (lab or "").strip()
        L_low = L.lower()

        # PEG (Graham)
        if "peg" in L_low:
            g = cagr_from_summary(sum_df, "EPS (RM)", N, end_basis=end_basis)
            if g is None or g <= 0:
                need_cols = N if end_basis == "TTM" else (N + 1)
                _emit("PEG (Graham)", f"Need EPS series over {need_cols} FYs and valid end value ({end_basis}).")

            pe_paths = reqs.get("P/E (×)", [])
            if end_basis == "TTM" and raw_ttm:
                miss = _missing_from_paths(raw_ttm, prev, ctx, paths=pe_paths, is_ttm=True)
                if miss:
                    _emit("PEG (Graham)", "P/E inputs missing: " + ", ".join(sorted(dict.fromkeys(miss))))
            continue

        # Margin of Safety
        if "margin of safety" in L_low:
            g = cagr_from_summary(sum_df, "EPS (RM)", N, end_basis=end_basis)
            eps_row = _find_row("EPS (RM)")
            eps_ok = False
            if eps_row is not None:
                v_end = (eps_row.get(ttm_col) if (end_basis=="TTM" and ttm_col and ttm_col in eps_row)
                         else eps_row.get(years[-1] if years else None))
                eps_ok = (_to_num(v_end) not in (None, 0))
            if g is None or not eps_ok:
                need_cols = N if end_basis == "TTM" else (N + 1)
                _emit("Margin of Safety", f"Need EPS and EPS CAGR over {need_cols} FYs (end={end_basis}).")

            if (price_per_share_from_summary(sum_df) in (None, 0)) and _to_num(price_fallback) in (None, 0):
                _emit("Margin of Safety", "Price missing — provide current Price or ensure P/E and EPS (TTM) exist.")
            continue

        # only "<metric> CAGR"
        if not L_low.endswith(" cagr"):
            continue

        base = _base_label_for_cagr(L)  # map to CFO/FCF/etc where needed

        # 1) try summary row (if that metric exists in summary)
        g = None
        r = _find_row(base)
        if isinstance(r, pd.Series) and years:
            g = _cagr_from_summary_row(r, years, N, ttm_col)
            
        # --- NEW: TTM end overrides to mirror View Stock (alias-aware) ---
        ttm_end = None
        if end_basis == "TTM":
            # flows / orderbook: pick by canonical key from TTM raw
            if base in ("Revenue", "EBITDA", "Net Profit", "CFO", "Orderbook"):
                canon = _canon_for_base(base)
                if raw_ttm and canon:
                    ttm_end = _to_num(_pick(raw_ttm, canon))
                if base == "CFO" and ttm_cfo is not None:
                    ttm_end = ttm_cfo  # keep explicit CFO override

            # banking balances — latest quarter (stock), alias-aware; else last FY
            if base in ("Gross Loans", "Deposits"):
                canon = _canon_for_base(base)
                if raw_ttm and canon:
                    ttm_end = _to_num(_pick(raw_ttm, canon))
                if ttm_end is None:
                    s = _series_from_annual(base)
                    if s is not None and not s.dropna().empty:
                        fy_cols = sorted(int(y) for y in s.index if isinstance(y, (int, np.integer)))
                        ttm_end = float(s.get(fy_cols[-1])) if fy_cols else None

            # special: Free Cash Flow derived from flows (CFO − |Capex|)
            if base in ("FCF", "Free Cash Flow"):
                ttm_end = ttm_fcf

        # 2) fallback to annual raw series (now with TTM end override where applicable)
        if g is None:
            g = _cagr_from_annual_series(base, N, ttm_end)

        # 3) still None -> explain what’s actually missing
        if g is None:
            # First: check FY availability given end-basis
            if not years:
                need_cols = N if end_basis == "TTM" else (N + 1)
                _emit(L, f"Need at least {need_cols} FY columns for '{base}'.")
                continue

            have_ok = (len(years) >= N) if end_basis == "TTM" else (len(years) >= N + 1)
            if not have_ok:
                need_cols = N if end_basis == "TTM" else (N + 1)
                _emit(L, f"Need at least {need_cols} FY columns for '{base}'.")
                continue

            # start FY for the window (differs for TTM vs FY)
            y0 = years[-N] if end_basis == "TTM" else years[-(N + 1)]

            # If the TTM end is the issue, show quarter-specific guidance
            if end_basis == "TTM":
                paths = _paths_for_base_label(base)
                if raw_ttm and paths:
                    miss = _missing_from_paths(raw_ttm, prev, ctx, paths=paths, is_ttm=True)
                    if miss:
                        _emit(L, f"TTM '{base}' missing inputs: " + ", ".join(sorted(dict.fromkeys(miss))))
                        continue
                elif not raw_ttm:
                    _emit(L, f"TTM '{base}' missing (need latest 4 quarters).")
                    continue

            # Otherwise it’s likely the start FY value
            _emit(L, f"'{base}' is missing or ≤0 in FY {y0}.")

    return pd.DataFrame(rows, columns=["Metric","Period","Missing Inputs"])

def build_soldier_worm_cashflow_kpis(
    annual_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    *,
    basis: str = "TTM",
    price_fallback: Optional[float] = None,
) -> pd.DataFrame:
    """
    Report missing raw inputs for Cash Flow KPI cards (CFO, Capex, FCF, margins/yields, CFO/EBITDA, Cash Conversion).
    """
    basis = (basis or "TTM").upper()
    rows: List[dict] = []

    # Build raw dict similar to compute_cashflow_kpis()
    if basis == "TTM":
        raw = ttm_raw_row_from_quarters(quarterly_df, current_price=price_fallback) or {}
    else:
        if annual_df is None or annual_df.empty:
            raw = {}
        else:
            a = annual_df.dropna(subset=["Year"]).sort_values("Year").iloc[-1].to_dict()
            raw = dict(a)

    bucket_guess = _infer_bucket_for_rows(annual_df) or "General"
    enrich_auto_raw_row_inplace(raw, bucket=bucket_guess)

    def need(metric, *fields):
        miss: List[str] = []
        for f in fields:
            v = _to_num(raw.get(f))
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                miss.append(f)
        if miss:
            rows.append({"Metric": metric, "Period": basis, "Missing Inputs": ", ".join(sorted(dict.fromkeys(miss)))})

    # CFO, Capex, FCF
    need("CFO", "CFO")
    if _to_num(raw.get("Capex")) is None and _to_num(raw.get("Capital Expenditure")) is None:
        rows.append({"Metric": "Capex", "Period": basis, "Missing Inputs": "Capex or Capital Expenditure"})

    # FCF Margin (%)
    if any(_to_num(raw.get(k)) is None for k in ("CFO",)) or all(_to_num(raw.get(k)) is None for k in ("Capex","Capital Expenditure")):
        rows.append({"Metric": "FCF Margin", "Period": basis, "Missing Inputs": "CFO and Capex"})
    if _to_num(raw.get("Revenue")) in (None, 0):
        rows.append({"Metric": "FCF Margin", "Period": basis, "Missing Inputs": "Revenue"})

    # FCF Yield (%)
    px  = _to_num(raw.get("Price")) or _to_num(price_fallback)
    shr = _to_num(raw.get("Shares"))
    if px in (None, 0) or shr in (None, 0):
        miss = []
        if px in (None, 0):  miss.append("Price")
        if shr in (None, 0): miss.append("Shares")
        rows.append({"Metric": "FCF Yield", "Period": basis, "Missing Inputs": ", ".join(miss)})

    # Capex to Revenue (%)
    if all(_to_num(raw.get(k)) is None for k in ("Capex","Capital Expenditure")):
        rows.append({"Metric": "Capex to Revenue", "Period": basis, "Missing Inputs": "Capex"})
    if _to_num(raw.get("Revenue")) in (None, 0):
        rows.append({"Metric": "Capex to Revenue", "Period": basis, "Missing Inputs": "Revenue"})

    # CFO/EBITDA (%)
    if _to_num(raw.get("CFO")) is None:
        rows.append({"Metric": "CFO/EBITDA", "Period": basis, "Missing Inputs": "CFO"})
    if _to_num(raw.get("EBITDA")) in (None, 0):
        rows.append({"Metric": "CFO/EBITDA", "Period": basis, "Missing Inputs": "EBITDA"})

    # Cash Conversion (%)
    if any(_to_num(raw.get(k)) is None for k in ("CFO",)) or all(_to_num(raw.get(k)) is None for k in ("Capex","Capital Expenditure")):
        rows.append({"Metric": "Cash Conversion", "Period": basis, "Missing Inputs": "CFO and Capex"})
    if _to_num(raw.get("Net Profit")) in (None, 0):
        rows.append({"Metric": "Cash Conversion", "Period": basis, "Missing Inputs": "Net Profit"})

    out = pd.DataFrame(rows, columns=["Metric","Period","Missing Inputs"])
    if not out.empty:
        out = (
            out.groupby(["Metric","Period"], as_index=False)["Missing Inputs"]
               .apply(lambda s: ", ".join(sorted(dict.fromkeys((", ".join(s)).split(", ")))))
               .sort_values(["Metric","Period"])
               .reset_index(drop=True)
        )
    return out


__all__ = [
    "build_soldier_worm_report",
    "build_soldier_worm_ttm_kpis",
    "build_soldier_worm_cagr_kpis",
    "build_soldier_worm_cashflow_kpis",
]


# --- NEW: Soldier Worm — verbose calculation trace --------------------
def _ordered_metrics_for_bucket(bucket: str) -> list[tuple[str, str]]:
    """
    Returns [("Category","Metric"), ...] honoring your config's category order.
    """
    out: list[tuple[str, str]] = []
    cats = (getattr(config, "INDUSTRY_SUMMARY_RATIOS_CATEGORIES", {}) or {}).get(bucket) \
        or (getattr(config, "INDUSTRY_SUMMARY_RATIOS_CATEGORIES", {}) or {}).get("General", {})
    order = getattr(config, "SUMMARY_RATIO_CATEGORY_ORDER", list(cats.keys()))
    seen = set()
    for cat in order:
        items = cats.get(cat, {})
        if not isinstance(items, dict):  # safety
            continue
        for metric in items.keys():
            m = str(metric)
            if m not in seen:
                out.append((cat, m))
                seen.add(m)
    return out

def _select_satisfied_path(
    row: dict,
    prev_row: dict | None,
    ctx: dict,
    paths: list[list[str]],
) -> tuple[list[str] | None, dict[str, tuple[float | object, str]]]:
    """
    Pick the first requirement path that is fully satisfied for this row.
    Returns (resolved_path_tokens, inputs_used{name -> (value, source)}),
    where source ∈ {"row","prev","ctx"}.
    """
    if not paths:
        return None, {}

    def _resolve_one(token: str) -> tuple[str | None, float | object | None, str | None]:
        tok = str(token).strip()

        # OR group: a|b (prefer the first available in the current row)
        if "|" in tok:
            for cand in [t.strip() for t in tok.split("|")]:
                v = _pick(row, cand)
                if v is not None:
                    return cand, v, "row"
            return None, None, None

        # explicit previous-year requirement
        if tok.endswith("_prev"):
            base = tok[:-5]
            v = _pick(prev_row or {}, base)
            return (base, v, "prev") if v is not None else (None, None, None)

        # contextual price: prefer row price, else fallback ctx
        if tok == "price_ctx":
            px_row = _to_num(row.get("Price"))
            if px_row is not None:
                return "price", px_row, "row"
            px_fb = _to_num(ctx.get("price_fallback"))
            return ("price_fallback", px_fb, "ctx") if px_fb is not None else (None, None, None)

        # normal canonical token from current row (via alias map)
        v = _pick(row, tok)
        if v is not None:
            return tok, v, "row"

        # very last resort: literal, case-insensitive header lookup
        t_low = tok.lower()
        for k in row.keys():
            if str(k).strip().lower() == t_low:
                vv = _to_num(row.get(k))
                if vv is not None:
                    return k, vv, "row"
        return None, None, None

    for p in paths:
        # only accept fully satisfied paths (no missing tokens)
        if _missing_from_paths(row, prev_row, ctx, paths=[p], is_ttm=False):
            continue

        used: dict[str, tuple[float | object, str]] = {}
        resolved_tokens: list[str] = []
        ok = True
        for token in p:
            name, val, src = _resolve_one(token)
            if name is None:
                ok = False
                break
            resolved_tokens.append(name)
            used[name] = (val, src or "row")
        if ok:
            return resolved_tokens, used

    return None, {}


def build_soldier_worm_calc_trace(
    annual_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    *,
    bucket: str,
    include_ttm: bool = True,
    price_fallback: float | None = None,
) -> pd.DataFrame:
    """
    Verbose per-metric calculation trace, aligned to Summary sections:
      Category | Metric | Period | Value | Path Used | Inputs (name=value [source]) | Flags | Note
    """
    if annual_df is None:
        annual_df = pd.DataFrame()
    if quarterly_df is None:
        quarterly_df = pd.DataFrame()

    calcs = BUCKET_CALCS.get(bucket, BUCKET_CALCS.get("General", {}))
    reqs  = _requirements_catalog(bucket)
    ordered = _ordered_metrics_for_bucket(bucket)

    if not annual_df.empty and "Year" in annual_df.columns:
        annual_df = annual_df.drop_duplicates(subset=["Year"], keep="last").sort_values("Year")

    DENOM_HINTS = {"revenue","equity","assets","ebitda","operating_profit","operating_income","nav","shares"}
    def _is_percentish(metric: str) -> bool:
        m = str(metric).lower()
        return "(%)" in m or "margin" in m or "yield" in m or " ratio" in m

    def _fmt_num(v):
        try:
            f = float(v)
            return None if not np.isfinite(f) else f
        except Exception:
            return None

    def _nm_reason(metric: str, inputs: dict[str, tuple[float, str]]) -> str:
        m = str(metric).lower()

        def v(name: str):
            t = inputs.get(name)
            try:
                f = float(t[0]) if t else None
                return f if np.isfinite(f) else None
            except Exception:
                return None

        # P/E → EPS must be > 0. If EPS not provided, derive from net_profit/shares if available
        if "p/e" in m:
            eps = v("eps")
            if eps is None:
                npat = v("net_profit"); sh = v("shares")
                eps = (npat / sh) if (npat not in (None,0) and sh not in (None,0)) else None
            if eps is not None and eps <= 0:
                return "Not meaningful (EPS ≤ 0)"

        # P/B or P/NAV → BVPS must be > 0
        if "p/b" in m or "p/nav" in m:
            eq = v("equity"); sh = v("shares")
            bvps = (eq / sh) if (eq not in (None,0) and sh not in (None,0)) else None
            if bvps is not None and bvps <= 0:
                return "Not meaningful (BVPS ≤ 0)"

        # EV/EBITDA or Net Debt / EBITDA → EBITDA must be > 0
        if "ebitda" in m and (("ev/ebitda" in m) or ("net debt / ebitda" in m) or ("net debt/ebitda" in m)):
            e = v("ebitda")
            if e is not None and e <= 0:
                return "Not meaningful (EBITDA ≤ 0)"

        # CFO/EBITDA → EBITDA must be > 0 (CFO can be negative)
        if "cfo/ebitda" in m:
            e = v("ebitda")
            if e is not None and e <= 0:
                return "Not meaningful (EBITDA ≤ 0)"
            
        # Capex to Revenue → a literal 0 capex is usually not meaningful for intensity
        if "capex to revenue" in m or "capex/revenue" in m or "capex intensity" in m:
            cap = v("capex")
            if cap is not None and cap == 0:
                return "Not meaningful (Capex = 0)"

        # CAGR rows are handled in the CAGR trace, not here
        return ""
    
    def _pretty_formula(metric: str, path_tokens: list[str] | None) -> str:
        # --- Banking polish: mirror fallback for Operating CF Margin (%)
        # If OI exists (explicit or derived), show "cfo / operating_income × 100";
        # otherwise show "cfo / revenue × 100". Keeps trace aligned with the calc.
        if str(metric).strip().lower() == "operating cf margin (%)" and path_tokens:
            toks = [t.lower() for t in path_tokens]
            uses_oi = (
                ("operating_income" in toks) or
                any(t in toks for t in ["nii_incl_islamic","fee_income","trading_income","other_op_income"])
            )
            if "cfo" in toks and uses_oi:
                return "cfo / operating_income × 100"
            if "cfo" in toks and "revenue" in toks:
                return "cfo / revenue × 100"
        # --- default pretty-printer below ---

        if not path_tokens:
            return "—"
        toks = list(path_tokens)

        # if dep/amort present, show a summed numerator explicitly
        if any(("dep" in t) or ("amort" in t) for t in toks) and len(toks) >= 2:
            num_tokens = [t for t in toks if ("dep" in t) or ("amort" in t) or (t in {"operating_profit","ebitda","gross_profit"})]
            den_tokens = [t for t in toks if t not in num_tokens]
            num = " + ".join(num_tokens) if num_tokens else (toks[0] if toks else "…")
            den = den_tokens[-1] if den_tokens else "…"
            base = f"({num}) / {den}"
        elif len(toks) >= 2:
            base = f"{toks[0]} / {toks[1]}"
        else:
            base = toks[0]

        return base + (" × 100" if _is_percentish(metric) else "")

    def _eval_row(row: dict, prev_row: dict | None, period_label: str, out_rows: list[dict]):
        ctx = {"price_fallback": price_fallback}
        enrich_auto_raw_row_inplace(row, bucket=bucket)
        for cat, metric in ordered:
            canon = _canon_metric(metric, bucket) or metric
            fn = _lookup_calc(calcs, canon) or _lookup_calc(calcs, metric)
            val = None
            try:
                if fn is not None:
                    val = fn(row, prev_row, ctx)
            except Exception:
                val = None

            paths = reqs.get(canon) or reqs.get(metric) or []
            path_used, inputs_used = _select_satisfied_path(row, prev_row, ctx, paths)

            # Build Inputs string with sources, collect flags
            ivals, flags = [], set()
            denom_zero = False
            for k, (v, src) in (inputs_used or {}).items():
                num = _fmt_num(v)
                num_str = "—" if num is None else f"{num:,.4f}"
                tag = " [prev FY]" if src == "prev" else (" [fallback]" if src == "ctx" else "")
                if src == "prev":     flags.add("used prev FY")
                if src == "ctx":      flags.add("used fallback")
                if k.lower() in DENOM_HINTS and (num is None or num == 0):
                    denom_zero = True
                ivals.append(f"{k}={num_str}{tag}")

            # Outlier checks on final value
            vnum = _fmt_num(val)
            if _is_percentish(metric):
                if vnum is not None and abs(vnum) > 300:
                    flags.add("outlier %")
            else:
                # generic negative/huge checks are noisy; keep it minimal
                pass
            if denom_zero:
                flags.add("denominator 0/blank")

            fmla = _pretty_formula(metric, path_used)
            note = "" if path_used else "No path fully satisfied (value may be blank)"

            # If we had a path / calc but result is NaN, try to explain with sign logic
            if (path_used or fn is not None or fmla != "—") and (vnum is None):
                nm = _nm_reason(metric, inputs_used or {})
                if nm:
                    note = nm

            out_rows.append({
                "Category": cat,
                "Metric": metric,
                "Period": period_label,
                "Value": (vnum if vnum is not None else np.nan),
                "Path Used": " | ".join(path_used) if path_used else "—",
                "Formula": fmla,
                "Inputs": ", ".join(ivals) if ivals else "—",
                "Flags": ", ".join(sorted(flags)) if flags else "",
                "Note": note,
            })


    rows_out: list[dict] = []
    prev = None
    for _, r in annual_df.iterrows():
        cur = r.to_dict()
        lab = (str(int(r["Year"])) if pd.notna(r.get("Year")) else "FY")
        _eval_row(cur, prev, lab, rows_out)
        prev = cur

    if include_ttm and not quarterly_df.empty:
        raw_ttm = ttm_raw_row_from_quarters(quarterly_df, current_price=price_fallback)
        if raw_ttm:
            enrich_three_fees_inplace(raw_ttm)
            _eval_row(raw_ttm, (annual_df.iloc[-1].to_dict() if not annual_df.empty else None), "TTM", rows_out)

    if not rows_out:
        return pd.DataFrame(columns=["Category","Metric","Period","Value","Path Used","Formula","Inputs","Flags","Note"])

    out = pd.DataFrame.from_records(rows_out)

    # keep sections grouped and metric order stable
    # Build explicit order maps for categories and (category, metric) pairs
    cat_order = {}
    met_order = {}
    for c, m in ordered:
        if c not in cat_order:
            cat_order[c] = len(cat_order)
        met_order.setdefault(c, [])
        if m not in met_order[c]:
            met_order[c].append(m)

    out["__cat_order"] = out["Category"].map(lambda c: cat_order.get(c, 9999))
    out["__met_order"] = out.apply(
        lambda r: (met_order.get(r["Category"], []).index(r["Metric"])
                if r["Metric"] in met_order.get(r["Category"], []) else 9999),
        axis=1
    )
    out = out.sort_values(["__cat_order","__met_order","Period"]).drop(columns=["__cat_order","__met_order"])
    return out

# --- EXTRA: per-section calc traces for cards ---------------------------------

def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None

def _fmt_inputs(d: dict[str, float]) -> str:
    out = []
    for k, v in (d or {}).items():
        try:
            out.append(f"{k}={float(v):,.4f}")
        except Exception:
            out.append(f"{k}={v}")
    return ", ".join(out) if out else "—"

def _canon_from_syn(label: str, bucket: str) -> str | None:
    """Resolve menu labels to Summary canonical labels using your config synonyms (tolerant)."""
    cats = (getattr(config, "INDUSTRY_SUMMARY_RATIOS_CATEGORIES", {}) or {}).get(bucket) \
        or (getattr(config, "INDUSTRY_SUMMARY_RATIOS_CATEGORIES", {}) or {}).get("General", {})
    idx = {}
    for _cat, mapping in (cats or {}).items():
        if isinstance(mapping, dict):
            for canon, syns in mapping.items():
                key = str(canon).strip().lower()
                idx[key] = canon
                for s in (syns or []):
                    idx[str(s).strip().lower()] = canon

    # normalizer
    def _norm(s: str) -> str:
        s = str(s).strip().lower()
        # unit variants
        s = s.replace(" (x)", " (×)")
        s = s.replace("(x)", "(×)")
        # day unit noise
        s = s.replace(" (days)", "").replace("(days)", "")
        # plural to singular for a few well-known offenders
        s = s.replace("receivables days", "receivable days")
        s = s.replace("payables days", "payable days")
        # spacing cleanups
        s = " ".join(s.split())
        return s

    s = _norm(label)
    if s in idx: 
        return idx[s]

    # a few more tolerant tries
    for v in [s.replace(" (%)", ""), s + " (%)",
              s.replace(" (×)", ""), s + " (×)"]:
        if v in idx:
            return idx[v]

    return None

def build_soldier_worm_ttm_kpis_trace(
    sum_df: pd.DataFrame,
    annual_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    *,
    bucket: str,
    labels: list[str],
    price_fallback: float | None = None,
) -> pd.DataFrame:
    """
    Trace for the small TTM KPI strip (Revenue, EBITDA, EPS, P/E, DY ...).
    Shows explicit 'Not meaningful' notes for sign-problem cases (e.g., EPS ≤ 0).
    """
    rows = []
    if quarterly_df is None or quarterly_df.empty:
        return pd.DataFrame(columns=["Category","Metric","Period","Value","Path Used","Formula","Inputs","Flags","Note"])

    # TTM raw (flows=sum last 4 qtrs; stocks=last)
    ttm = ttm_raw_row_from_quarters(quarterly_df, current_price=price_fallback) or {}
    if not ttm:
        return pd.DataFrame(columns=["Category","Metric","Period","Value","Path Used","Formula","Inputs","Flags","Note"])
    enrich_three_fees_inplace(ttm)                  # gp/ebitda, etc.
    enrich_auto_raw_row_inplace(ttm, bucket=bucket) # ensure EPS/DPS and aliases exist

    prev = annual_df.iloc[-1].to_dict() if (annual_df is not None and not annual_df.empty) else None

    calcs = BUCKET_CALCS.get(bucket, BUCKET_CALCS.get("General", {}))
    reqs  = _requirements_catalog(bucket)

    # helper for robust Summary lookups
    def _summary_colname(df):
        if df is None or df.empty: return None
        if "Metric" in df.columns: return "Metric"
        if "Ratio"  in df.columns: return "Ratio"
        return None

    for lbl in (labels or []):
        canon = _canon_from_syn(lbl, bucket) or lbl
        fn = _lookup_calc(calcs, canon) or _lookup_calc(calcs, lbl)

        # Try Summary-style calc first (with requirements → inputs)
        val = None; path = []; inputs = {}; note = ""; formula = ""
        if fn is not None:
            try:
                ctx = {"price_fallback": price_fallback}
                val = fn(ttm, prev, ctx)
            except Exception:
                val = None
            p = reqs.get(canon) or reqs.get(lbl)
            if p:
                path, inputs = _select_satisfied_path(ttm, prev, {"price_fallback": price_fallback}, p)

        # If still none, handle a few common TTM card formulas explicitly
        L = str(lbl).strip().lower()
        if _num(val) is None:
            if L in ("revenue","gross profit","operating profit","ebitda","net profit","eps","dps"):
                val = _num(ttm.get(lbl))  # direct pull by label
                formula = f"{lbl} (ttm raw)"
                path    = [lbl.lower()]
                inputs  = {lbl: val}

            elif L in ("dividend yield","dividend yield (%)"):
                dps   = _num(ttm.get("DPS"))
                price = _num(ttm.get("Price")) or _num(price_fallback)
                if dps is not None and price not in (None, 0):
                    val = (dps / price) * 100.0
                    formula = "dps / price × 100"
                    path    = ["dps", "price|price_fallback"]
                    inputs  = {"dps": dps, ("price" if _num(ttm.get("Price")) else "price_fallback"): price}

            elif L in ("p/e","p/e (×)","p/e (x)"):
                # P/E only meaningful if EPS > 0
                eps   = _num(ttm.get("EPS"))
                if eps is None:
                    npat  = _num(ttm.get("Net Profit"))
                    sh    = _num(ttm.get("Shares"))
                    eps   = (npat / sh) if (npat not in (None,0) and sh not in (None,0)) else None
                price = _num(ttm.get("Price")) or _num(price_fallback)
                if eps is not None and eps > 0 and price is not None:
                    val = price / eps
                    formula = "price / eps"
                    path    = ["price|price_fallback", "eps|net_profit/shares"]
                    inputs  = {"price" if _num(ttm.get("Price")) else "price_fallback": price, "eps": eps}
                elif eps is not None and price is not None and eps <= 0:
                    formula = "price / eps"
                    path    = ["price|price_fallback", "eps|net_profit/shares"]
                    inputs  = {"price" if _num(ttm.get("Price")) else "price_fallback": price, "eps": eps}
                    note    = "Not meaningful (EPS ≤ 0)"

            # --- EXTRA explicit formulas ----------------------------------

            elif L in ("eps yoy", "eps yoy (%)", "eps yoy%"):
                # EPS YoY (%) = ((EPS_TTM / EPS_lastFY) - 1) × 100
                eps_ttm = _num(ttm.get("EPS"))
                if eps_ttm is None:
                    npat_ttm = _num(ttm.get("Net Profit"))
                    sh_ttm   = _num(ttm.get("Shares"))
                    eps_ttm  = (npat_ttm / sh_ttm) if (npat_ttm not in (None,0) and sh_ttm not in (None,0)) else None

                eps_prev = None
                if isinstance(annual_df, pd.DataFrame) and not annual_df.empty:
                    prev_row = annual_df.iloc[-1].to_dict()
                    eps_prev = _num(prev_row.get("EPS") or prev_row.get("EPS (RM)") or prev_row.get("Earnings per Share (RM)"))
                    if eps_prev is None:
                        np_prev = _num(prev_row.get("Net Profit") or prev_row.get("Net Income") or prev_row.get("Profit After Tax") or prev_row.get("PAT"))
                        sh_prev = _num(prev_row.get("Shares") or prev_row.get("Shares Outstanding"))
                        eps_prev = (np_prev / sh_prev) if (np_prev not in (None,0) and sh_prev not in (None,0)) else None

                val, note = None, ""
                if eps_ttm is not None and eps_prev not in (None, 0):
                    val = ((float(eps_ttm) / float(eps_prev)) - 1.0) * 100.0
                elif eps_prev == 0:
                    note = "Not meaningful (prev EPS = 0)"
                else:
                    note = "Missing EPS_TTM or prev FY EPS"

                rows.append({
                    "Category": "TTM KPIs",
                    "Metric": lbl,
                    "Period": "TTM",
                    "Value": (None if val is None else float(val)),
                    "Path Used": "eps_ttm|net_profit/shares  |  eps_lastFY|annual_eps",
                    "Formula": "(EPS_TTM / EPS_lastFY) − 1 × 100",
                    "Inputs": _fmt_inputs({"eps_ttm": eps_ttm, "eps_lastFY": eps_prev}),
                    "Flags": "",
                    "Note": note,
                })
                continue  # <--- IMPORTANT: prevents the extra generic row

            elif L in ("payout ratio", "payout ratio (%)", "dividend payout ratio", "dividend payout ratio (%)"):
                # Payout Ratio (%) = DPS / EPS × 100
                pr_val = None
                try:
                    if isinstance(sum_df, pd.DataFrame) and not sum_df.empty:
                        ttm_col = next((c for c in sum_df.columns if isinstance(c, str) and c.upper().startswith("TTM")), None)
                        name_col = _summary_colname(sum_df)
                        if ttm_col and name_col:
                            mask = sum_df[name_col].astype(str).str.strip().str.lower() == "payout ratio (%)"
                            pr_row = sum_df[mask]
                            if not pr_row.empty and pd.notna(pr_row.iloc[0].get(ttm_col)):
                                pr_val = float(pr_row.iloc[0].get(ttm_col))
                except Exception:
                    pr_val = None

                dps = _num(ttm.get("DPS") or ttm.get("Dividend per Share (TTM, RM)") or ttm.get("Dividend per Share (RM)"))
                eps = _num(ttm.get("EPS"))
                if eps is None:
                    npat = _num(ttm.get("Net Profit"))
                    sh   = _num(ttm.get("Shares"))
                    eps  = (npat / sh) if (npat not in (None,0) and sh not in (None,0)) else None

                val = (pr_val if (pr_val is not None and np.isfinite(pr_val))
                       else ((float(dps) / float(eps)) * 100.0 if (dps is not None and eps not in (None, 0)) else None))

                note = ""
                if pr_val is None and (eps in (None, 0) or dps is None):
                    note = "Missing DPS or EPS (TTM)"
                elif pr_val is None and eps == 0:
                    note = "Not meaningful (EPS = 0)"

                rows.append({
                    "Category": "TTM KPIs",
                    "Metric": lbl,
                    "Period": "TTM",
                    "Value": (None if val is None else float(val)),
                    "Path Used": "dps|summary/ttm_raw  |  eps|summary/ttm_raw or net_profit/shares",
                    "Formula": "DPS / EPS × 100  (= DY% × P/E)",
                    "Inputs": _fmt_inputs({"dps": dps, "eps": eps}),
                    "Flags": "",
                    "Note": note,
                })
                continue  # same reason as above

        # --- NEW: requirements-path fallback for passthrough KPIs ---
        if _num(val) is None:
            p = reqs.get(canon) or reqs.get(lbl)
            if p:
                path2, inputs2 = _select_satisfied_path(
                    ttm, prev, {"price_fallback": price_fallback}, p
                )
                if path2:
                    path = path2
                    inputs = inputs2
                    if len(path2) == 1:
                        k = path2[0]
                        val = _num((inputs2.get(k) or (None, ""))[0])
                        formula = f"{k} (ttm raw)"

        # Generic row (for everything else)
        rows.append({
            "Category": "TTM KPIs",
            "Metric": lbl,
            "Period": "TTM",
            "Value": (float(val) if _num(val) is not None else np.nan),
            "Path Used": (" | ".join(path) if path else "—"),
            "Formula": (formula or (getattr(fn, "__name__", "") if fn else "")),
            "Inputs": _fmt_inputs(inputs),
            "Flags": "",
            "Note": note if note else ("" if (path or fn or formula) else "No path available"),
        })

    return pd.DataFrame(rows)


def build_soldier_worm_cashflow_kpis_trace(
    annual_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    *,
    basis: str = "TTM",
    price_fallback: float | None = None,
) -> pd.DataFrame:
    """Trace for the Cash-Flow KPI card strip (CFO/Capex/FCF, ratios, etc.)."""
    rows = []
    if basis.upper() == "TTM":
        raw = ttm_raw_row_from_quarters(quarterly_df, current_price=price_fallback) or {}
    else:
        if annual_df is None or annual_df.empty:
            return pd.DataFrame()
        raw = annual_df.dropna(subset=["Year"]).sort_values("Year").iloc[-1].to_dict()

    # helpers
    cfo        = _num(raw.get("CFO") or raw.get("Cash Flow from Ops") or raw.get("Operating Cash Flow"))
    capex_raw  = _num(raw.get("Capex") or raw.get("Capital Expenditure"))
    capex_mag  = (abs(capex_raw) if capex_raw is not None else None)   # <-- use magnitude for downstream calcs
    fcf        = (None if (cfo is None and capex_mag is None) else ((cfo or 0.0) - (capex_mag or 0.0)))
    rev        = _num(raw.get("Revenue") or raw.get("Total Revenue") or raw.get("TotalRevenue") or raw.get("Sales"))
    ebitda     = _num(raw.get("EBITDA"))
    npat       = _num(raw.get("Net Profit") or raw.get("Net Income") or raw.get("NPAT") or raw.get("Profit After Tax"))
    shares     = _num(raw.get("Shares") or raw.get("Units"))
    price      = _num(raw.get("Price")) or _num(price_fallback)

    # value builders
    def add(metric, value, formula, inputs, path):
        rows.append({
            "Category": "Cash Flow",
            "Metric": metric,
            "Period": basis.upper(),
            "Value": (float(value) if _num(value) is not None else np.nan),
            "Path Used": " | ".join(path) if path else "—",
            "Formula": formula,
            "Inputs": _fmt_inputs(inputs),
            "Flags": "",
            "Note": "",
        })

    # show raw Capex line as-is
    add("CFO", cfo, "cfo", {"cfo": cfo}, ["cfo|cash_flow_from_ops|operating_cash_flow"])
    add("Capex", capex_raw, "capex", {"capex": capex_raw}, ["capex|capital_expenditure"])

    # but compute FCF/ratios with |Capex|
    add("Free Cash Flow",
        fcf,
        "cfo − abs(capex)",
        {"cfo": cfo, "abs_capex": capex_mag},
        ["cfo", "abs(capex)"])

    add("FCF Margin (%)",
        (None if fcf is None or rev in (None, 0) else (fcf / rev) * 100),
        "fcf / revenue × 100 (with capex as |capex|)",
        {"fcf": fcf, "revenue": rev},
        ["fcf", "revenue"])
    
    # NEW: Operating CF Margin (CFO / Revenue × 100)
    add("Operating CF Margin (%)",
        (None if cfo is None or rev in (None, 0) else (cfo / rev) * 100),
        "cfo / revenue × 100",
        {"cfo": cfo, "revenue": rev},
        ["cfo", "revenue"])

    fcf_ps = (None if fcf is None or shares in (None, 0) else fcf / shares)
    add("FCF Yield (%)",
        (None if fcf_ps is None or price in (None, 0) else (fcf_ps / price) * 100),
        "(fcf/shares) / price × 100 (with capex as |capex|)",
        {"fcf": fcf, "shares": shares, "price": price},
        ["fcf", "shares", "price|price_fallback"])

    add("Capex to Revenue (%)",
        (None if capex_mag in (None, 0) or rev in (None, 0) else (capex_mag / rev) * 100),
        "abs(capex) / revenue × 100",
        {"abs_capex": capex_mag, "revenue": rev},
        ["abs(capex)", "revenue"])

    add("CFO/EBITDA (%)",
        (None if cfo in (None, 0) or ebitda in (None, 0) else (cfo / ebitda) * 100),
        "cfo / ebitda × 100",
        {"cfo": cfo, "ebitda": ebitda},
        ["cfo", "ebitda"])

    add("Cash Conversion (%)",
        (None if cfo in (None, 0) or npat in (None, 0) else (cfo / npat) * 100),
        "cfo / net_profit × 100",
        {"cfo": cfo, "net_profit": npat},
        ["cfo", "net_profit"])


    return pd.DataFrame(rows)

def build_soldier_worm_cagr_kpis_trace(
    sum_df: pd.DataFrame,
    annual_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    *,
    bucket: str,
    labels: list[str],
    years_back: int = 5,
    end_basis: str = "TTM",
    price_fallback: float | None = None,
) -> pd.DataFrame:
    """Trace for “… CAGR” cards: shows start/end values, N and formula, with sign-aware notes."""
    if sum_df is None or sum_df.empty:
        return pd.DataFrame(columns=["Category","Metric","Period","Value","Path Used","Formula","Inputs","Flags","Note"])

    # pick the TTM column name you already use
    def _ttm_col(df):
        for c in df.columns:
            if isinstance(c, str) and c.upper().startswith("TTM"): return c
        return "TTM" if "TTM" in df.columns else None

    ttm_col = _ttm_col(sum_df)
    years   = sorted([c for c in sum_df.columns if isinstance(c, (int, np.integer))])

    out = []
    for item in (labels or []):
        if not str(item).lower().endswith(" cagr"):
            continue
        base = item.replace(" CAGR","").strip()
        # find the base metric row (tolerate synonyms)
        canon = _canon_from_syn(base, bucket) or base
        row = None
        hit = sum_df[sum_df["Metric"].astype(str).str.lower() == str(canon).lower()]
        if not hit.empty:
            row = hit.iloc[0]

        use_ttm = (str(end_basis).upper() == "TTM")
        if row is None or not years:
            out.append({"Category":"CAGR","Metric":item,"Period":end_basis,"Value":np.nan,
                        "Path Used":"—","Formula":"(end/start)^(1/N) − 1",
                        "Inputs":"—","Flags":"", "Note":"base metric not found"})
            continue

        # window length vs data (requested_N is what the UI asked for)
        requested_N = int(years_back)
        N = min(requested_N, max(1, len(years) - (1 if use_ttm else 0)))

        end_label = (ttm_col if (use_ttm and ttm_col and pd.notna(row.get(ttm_col))) else years[-1])
        start_year = (years[-N] if use_ttm else years[-(N+1)])

        v_end = _num(row.get(end_label))
        v0    = _num(row.get(start_year))

        g, note = None, ""
        if v_end in (None, 0) or v0 in (None, 0):
            note = "Insufficient data (start/end missing or = 0)"
        elif v_end <= 0 or v0 <= 0:
            # Align with core CAGR engine which requires strictly positive values
            note = "Not meaningful (start/end ≤ 0)"
        else:
            try:
                g = (v_end / v0) ** (1.0 / N) - 1.0
            except Exception:
                g = None
                note = "Computation failed (check signs/values)"

        out.append({
            "Category": "CAGR",
            "Metric": item,
            "Period": str(end_basis).upper(),
            "Value": (float(g*100.0) if g is not None else np.nan),
            "Path Used": f"start={start_year} → end={end_label}",
            "Formula": "(end / start)^(1/N) − 1",
            "Inputs": _fmt_inputs({"start": v0, "end": v_end, "N": N}),
            "Flags": "",
            "Note": note,
        })

    return pd.DataFrame(out)

# --- NEW: explicit CAGR calc trace (TTM/FY) ----------------------------------
def build_soldier_worm_cagr_calc_trace(
    sum_df: pd.DataFrame,
    annual_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    *,
    bucket: str,
    labels: list[str],
    years_back: int = 5,
    end_basis: str = "TTM",
    price_fallback: float | None = None,
) -> pd.DataFrame:
    """
    Trace table for “… CAGR” cards:
      Category | Metric | Period | Value | Path Used | Formula | Inputs | Flags | Note

    - Picks the same row as the Summary table (synonym tolerant)
    - Computes with the *actual* N used when history is short and shows it
    - Falls back to annual raw series (with TTM overrides for flows) when needed
    """
    import math, numpy as np, pandas as pd

    # empty shell early
    cols = ["Category","Metric","Period","Value","Path Used","Formula","Inputs","Flags","Note"]
    if sum_df is None or sum_df.empty:
        return pd.DataFrame(columns=cols)

    EB = str(end_basis).upper()
    requested_N = int(years_back)

    # helpers --------------------------------------------------------------
    def _ttm_col(df: pd.DataFrame) -> str | None:
        if df is None or df.empty: 
            return None
        for c in df.columns:
            if isinstance(c, str) and c.upper().startswith("TTM"):
                return c
        return None

    def _years_in(df: pd.DataFrame) -> list[int]:
        if df is None or df.empty: 
            return []
        return sorted([int(c) for c in df.columns if isinstance(c, (int, np.integer))])

    def _canonical_base(lbl: str) -> str:
        base = str(lbl).replace("CAGR", "").strip()
        m = {
            "operating cash flow": "CFO",
            "free cash flow": "Free Cash Flow",
            "net profit": "Net Profit",
            "orderbook": "Orderbook",
            "revenue": "Revenue",
            "ebitda": "EBITDA",
        }
        return m.get(base.lower(), base)

    def _summary_row_for(label: str) -> pd.Series | None:
        hit = sum_df[sum_df["Metric"].astype(str).str.lower() == str(label).lower()]
        return hit.iloc[0] if not hit.empty else None

    def _annual_series_for(label: str) -> pd.Series | None:
        """Build a Year->value series from ANNUAL raw columns (FCF = CFO − |Capex|)."""
        if annual_df is None or annual_df.empty or "Year" not in annual_df.columns:
            return None

        years = pd.to_numeric(annual_df["Year"], errors="coerce")
        cmap = {
            "Revenue": ["Revenue","Total Revenue","TotalRevenue","Sales"],
            "EBITDA":  ["EBITDA"],
            "Net Profit": ["Net Profit","NPAT","Profit After Tax","Profit attributable to owners","Net Income","PATAMI"],
            "CFO": ["CFO","Cash Flow from Ops","Cash from Operations","Operating Cash Flow"],
            "Capex": ["Capex","Capital Expenditure","Capital Expenditures",
                      "Purchase of PPE","Purchases of property, plant and equipment",
                      "Payments for property, plant and equipment",
                      "Purchase of Property, Plant and Equipment"],
            "Orderbook": ["Orderbook","Order Book","Unbilled Sales"],
        }

        if label == "Free Cash Flow" or label == "FCF":
            s_cfo = _annual_series_for("CFO")
            s_cap = _annual_series_for("Capex")
            if s_cfo is None or s_cap is None:
                return None
            idx = sorted(set(s_cfo.index) & set(s_cap.index))
            if not idx:
                return None
            return pd.Series({int(y): float(s_cfo.get(y, np.nan)) - abs(float(s_cap.get(y, np.nan))) for y in idx})

        for col in (cmap.get(label) or [label]):
            if col in annual_df.columns:
                s = pd.to_numeric(annual_df[col], errors="coerce")
                good = (~s.isna()) & (~years.isna())
                if good.sum() >= 2:
                    return pd.Series({int(y): float(v) for y, v in zip(years[good], s[good])})
        return None

    def _ttm_override_for_any(base: str):
        """
        TTM end overrides from quarters:
          • Flows (Revenue/EBITDA/Net Profit/CFO/FCF): sum last 4 quarters
          • Balances (Gross Loans/Deposits/Orderbook): take the latest quarter
        """
        if quarterly_df is None or quarterly_df.empty:
            return None

        q = quarterly_df.copy()
        # Try to infer quarter order
        if "Qnum" not in q.columns and "Quarter" in q.columns:
            q["Qnum"] = pd.to_numeric(q["Quarter"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
        if "Qnum" in q.columns:
            q = q.dropna(subset=["Year","Qnum"]).sort_values(["Year","Qnum"])
        else:
            q = q.dropna(subset=["Year"]).sort_values(["Year"])

        # helpers
        def _have(col: str) -> bool:
            return any(c.strip().lower() == col.strip().lower() for c in q.columns)

        def _col(col: str) -> Optional[str]:
            # case-insensitive match
            for c in q.columns:
                if c.strip().lower() == col.strip().lower():
                    return c
            return None

        def qsum(cands):
            tail = q.tail(min(4, len(q)))
            for cand in cands:
                c = _col(cand)
                if c:
                    return pd.to_numeric(tail[c], errors="coerce").sum(skipna=True)
            return None

        def qlast(cands):
            last = q.tail(1)
            for cand in cands:
                c = _col(cand)
                if c and pd.notna(last[c]).any():
                    return float(pd.to_numeric(last[c], errors="coerce").iloc[-1])
            return None

        # flows
        if base in ("Revenue",):
            return qsum(["Q_Revenue","Q_Total Revenue","Q_TotalRevenue","Q_Sales"])
        if base in ("EBITDA",):
            return qsum(["Q_EBITDA"])
        if base in ("Net Profit",):
            return qsum(["Q_Net Profit","Q_NPAT","Q_Profit After Tax","Q_Profit attributable to owners"])
        if base in ("CFO",):
            return qsum(["Q_CFO","Q_Cash Flow from Ops","Q_Cash from Operations","Q_Operating Cash Flow"])
        if base in ("Free Cash Flow","FCF"):
            ocf = qsum(["Q_CFO","Q_Cash Flow from Ops","Q_Cash from Operations","Q_Operating Cash Flow"])
            cap = qsum(["Q_Capex","Q_Capital Expenditure"])
            return ((ocf or 0.0) - (abs(cap) if cap is not None else 0.0)) if (ocf is not None or cap is not None) else None

        # balances (latest quarter, alias-friendly)
        if base in ("Orderbook",):
            return qlast(["Q_Orderbook","Q_Order Book","Q_Unbilled Sales"])
        if base in ("Gross Loans",):
            return qlast(["Q_Gross Loans","Q_gross_loans","Gross Loans","gross_loans"])
        if base in ("Deposits",):
            return qlast(["Q_Deposits","Q_deposits","Deposits","deposits"])

        return None

    def _fmt_inputs(end_v, start_v, N):
        try:
            e = float(end_v); s = float(start_v)
            return f"end={e:,.4f}, start={s:,.4f}, N={int(N)}"
        except Exception:
            return "—"

    # working --------------------------------------------------------------
    ttm_col = _ttm_col(sum_df)
    years   = _years_in(sum_df)

    out = []
    for lbl in (labels or []):
        if not str(lbl).lower().endswith(" cagr"):
            continue

        # (a) synonym tolerant look-up against the Summary row
        base  = _canonical_base(lbl)
        canon = _canon_from_syn(base, bucket) or base
        row   = _summary_row_for(canon)

        path  = []
        flags = []
        note  = ""
        val   = np.nan

        # determine the maximum usable window given available FY columns
        if not years:
            out.append({
                "Category":"CAGR","Metric":lbl,"Period":f"{requested_N}y {EB}",
                "Value":np.nan,"Path Used":"—","Formula":"(end/start)^(1/N) − 1",
                "Inputs":"—","Flags":"","Note":"base metric not found in Summary"
            })
            continue

        N_avail = max(1, len(years) - (1 if EB == "TTM" else 0))
        N = min(requested_N, N_avail)
        if N != requested_N:
            flags.append(f"short history → used {N}y")

        # 1) Try Summary row values first (keeps in sync with displayed table)
        end_v = start_v = None
        if isinstance(row, pd.Series):
            if EB == "TTM":
                if ttm_col and (ttm_col in row) and pd.notna(row.get(ttm_col)):
                    end_v = _num(row.get(ttm_col))
                    start_year = years[-N]
                    start_v = _num(row.get(start_year))
                    path.append("summary|TTM")
                # else leave for fallback
            else:
                last_fy = years[-1]
                end_v = _num(row.get(last_fy))
                start_year = years[-(N+1)]
                start_v = _num(row.get(start_year))
                path.append("summary|FY")

        # 2) Fall back to annual raw series (+ TTM override for flows) if needed
        if (end_v is None or start_v is None):
            s = _annual_series_for(canon)
            if s is not None and s.dropna().size >= 2:
                yrs = sorted([int(i) for i in s.index if pd.notna(s.get(i))])
                if EB == "TTM":
                    override = _ttm_override_for_any(canon)
                    if override is not None:
                        end_v = float(override); path.append("annual_raw|TTM")
                    else:
                        end_v = float(s.get(yrs[-1])); path.append("annual_raw|FY-as-end")
                        flags.append("TTM end missing → used FY end")
                    if len(yrs) >= N:
                        start_v = float(s.get(yrs[-N]))
                else:
                    end_v = float(s.get(yrs[-1])); path.append("annual_raw|FY")
                    if len(yrs) >= N + 1:
                        start_v = float(s.get(yrs[-(N+1)]))

        # 3) Compute CAGR using the effective N we actually used
        if end_v in (None, 0) or start_v in (None, 0):
            note = "Insufficient data (start/end missing or = 0)"
        elif (end_v is not None and start_v is not None) and (end_v <= 0 or start_v <= 0):
            note = "Not meaningful (start/end ≤ 0)"
        else:
            try:
                g = (float(end_v) / float(start_v)) ** (1.0 / float(N)) - 1.0
                val = float(g * 100.0)
            except Exception:
                note = "Computation failed (check signs/values)"

        out.append({
            "Category": "CAGR",
            "Metric": lbl,
            # (b) show the actual N and basis used
            "Period": f"{N}y {EB}",
            "Value": (val if (isinstance(val, (int,float)) and math.isfinite(val)) else np.nan),
            "Path Used": (" | ".join(path) if path else "—"),
            "Formula": "(end / start)^(1/N) − 1 (shown as %)",
            "Inputs": _fmt_inputs(end_v, start_v, N),
            "Flags": ", ".join(flags),
            "Note": note,
        })

    return pd.DataFrame(out, columns=cols)
