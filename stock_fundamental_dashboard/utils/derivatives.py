# utils/derivatives.py
# ----------------------------------------------------
# Lightweight helpers to compute “Derivatives Health”
# ----------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


__all__ = ["DerivInputs", "compute_deriv_health", "traffic_light"]


@dataclass
class DerivInputs:
    """
    Inputs pulled from the derivatives note / financial statements.

    All values should be in the SAME currency (e.g., RM).
    Leave any unknowns as None.
    """
    total_notional: float | int | None = None          # Sum of all derivative notionals (gross or delta-adj per your policy)
    equity: float | int | None = None                  # Shareholders' equity
    deriv_pnl_ttm: float | int | None = None           # Derivatives P&L (TTM): realized + fair value changes
    ebit_ttm: float | int | None = None                # EBIT (TTM)
    collateral_posted: float | int | None = None       # Margin/restricted cash posted for derivatives
    top1_counterparty_notional: float | int | None = None  # Notional with largest counterparty
    level3_fv: float | int | None = None               # Level-3 fair value (derivatives only)
    total_deriv_fv: float | int | None = None          # Total derivatives fair value (Level 1+2+3, net or gross by your policy)


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Divide with None/zero protection, returning None if not computable."""
    try:
        if a is None or b is None:
            return None
        b = float(b)
        if b == 0.0:
            return None
        return float(a) / b
    except Exception:
        return None


def traffic_light(
    value: Optional[float],
    *,
    yellow: float,
    red: float,
    higher_is_risk: bool = True
) -> str:
    """
    Classify a value into 🟢 / 🟡 / 🔴.

    If higher_is_risk=True:
        value <= yellow → 🟢
        value <= red    → 🟡
        else            → 🔴
    If higher_is_risk=False, the logic is inverted (lower is riskier).
    """
    if value is None:
        return "⚪"  # unknown / not available

    v = float(value)
    if higher_is_risk:
        if v <= yellow:
            return "🟢"
        if v <= red:
            return "🟡"
        return "🔴"
    else:
        if v >= red:
            return "🟢"
        if v >= yellow:
            return "🟡"
        return "🔴"


def compute_deriv_health(inp: DerivInputs) -> Dict[str, Any]:
    """
    Compute five simple risk checks and an overall light.

    Metrics:
      • Notional / Equity (×)
      • Deriv P&L / EBIT (TTM)
      • Collateral / Equity
      • Top Counterparty Share
      • Level-3 FV Share

    Returns:
      {
        "overall": "🟢" | "🟡" | "🔴",
        "metrics": [
          (label: str, value: float|None, light: str, unit_suffix: str),
          ...
        ],
        "raw": { key: value_or_None, ... }
      }
    """
    # Ratios
    notional_to_equity = _safe_div(inp.total_notional, inp.equity)
    deriv_pnl_share = _safe_div(abs(inp.deriv_pnl_ttm) if inp.deriv_pnl_ttm is not None else None, inp.ebit_ttm)
    collateral_ratio = _safe_div(inp.collateral_posted, inp.equity)
    counterparty_conc = _safe_div(inp.top1_counterparty_notional, inp.total_notional)
    level3_share = _safe_div(inp.level3_fv, inp.total_deriv_fv)

    # Thresholds (rules of thumb; adjust to your policy)
    tl_notional_to_equity = traffic_light(notional_to_equity, yellow=3.0, red=5.0, higher_is_risk=True)
    tl_deriv_pnl_share = traffic_light(deriv_pnl_share, yellow=0.10, red=0.20, higher_is_risk=True)
    tl_collateral_ratio = traffic_light(collateral_ratio, yellow=0.10, red=0.20, higher_is_risk=True)
    tl_cp_conc = traffic_light(counterparty_conc, yellow=0.40, red=0.50, higher_is_risk=True)
    tl_level3_share = traffic_light(level3_share, yellow=0.30, red=0.60, higher_is_risk=True)

    items = [
        ("Notional / Equity", notional_to_equity, tl_notional_to_equity, "×"),
        ("Deriv P&L / EBIT (TTM)", deriv_pnl_share, tl_deriv_pnl_share, ""),   # already a ratio
        ("Collateral / Equity", collateral_ratio, tl_collateral_ratio, ""),    # ratio
        ("Top Counterparty Share", counterparty_conc, tl_cp_conc, ""),         # ratio
        ("Level-3 FV Share", level3_share, tl_level3_share, ""),               # ratio
    ]

    # Overall light: any red → 🔴; else any yellow → 🟡; else 🟢 (unknowns don't force red)
    lights = [tl for _, _, tl, _ in items]
    overall = "🟢"
    if "🔴" in lights:
        overall = "🔴"
    elif "🟡" in lights:
        overall = "🟡"

    return {
        "overall": overall,
        "metrics": items,
        "raw": {
            "notional_to_equity": notional_to_equity,
            "deriv_pnl_share": deriv_pnl_share,
            "collateral_ratio": collateral_ratio,
            "counterparty_concentration": counterparty_conc,
            "level3_share": level3_share,
        },
    }
