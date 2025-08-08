# utils/rules.py
from typing import Dict, List, Tuple

# ─────────────── STRATEGY DEFINITIONS ────────────────
RULESETS = {
    "Quality-Value": {
        "mandatory": [
            ("Positive EPS",           lambda m: (m.get("EPS") or 0) > 0,                     "EPS must be > 0"),
            ("ROE ≥ 12%",              lambda m: (m.get("ROE (%)") or 0) >= 12,               "ROE ≥ 12%"),
            ("Debt-Asset ≤ 50%",       lambda m: (m.get("Debt-Asset Ratio (%)") or 100) <= 50,"Debt/Asset ≤ 50%"),
        ],
        "scored": [  # label · fn · weight
            ("P/E ≤ 15",               lambda m: (m.get("P/E")  or 1e9) <= 15,                     20),
            ("P/B ≤ 2",                lambda m: (m.get("P/B")  or 1e9) <= 2,                      20),
            ("Gross Margin ≥ 20%",     lambda m: (m.get("Gross Profit Margin (%)") or 0) >= 20,    20),
            ("Current Ratio ≥ 2",      lambda m: (m.get("Current Ratio") or 0) >= 2,               20),
            ("Dividend Yield ≥ 4%",    lambda m: (m.get("Dividend Yield (%)") or 0) >= 4,          20),
        ],
    },
    "Dividend": {
        "mandatory": [
            ("Dividend Yield ≥ 3%",    lambda m: (m.get("Dividend Yield (%)") or 0) >= 3,     "Yield ≥ 3%"),
            ("Payout ≤ 80%",           lambda m: (m.get("Dividend Payout Ratio (%)") or 100) <= 80,"Payout ≤ 80%"),
        ],
        "scored": [
            ("ROE ≥ 10%",              lambda m: (m.get("ROE (%)") or 0) >= 10,                    25),
            ("Debt-Asset ≤ 50%",       lambda m: (m.get("Debt-Asset Ratio (%)") or 100) <= 50,     25),
            ("P/E ≤ 18",               lambda m: (m.get("P/E") or 1e9) <= 18,                      25),
            ("Current Ratio ≥ 1.5",    lambda m: (m.get("Current Ratio") or 0) >= 1.5,             25),
        ],
    },
}

# ─────────────── VQGM (Value+Quality+Growth with solvency gates) ───────────────
RULESETS["VQGM"] = {
    "mandatory": [
        ("EPS TTM > 0",                         lambda m: (m.get("TTM EPS") or 0) > 0,                             "TTM EPS must be > 0"),
        ("InterestCoverage ≥ 3",                lambda m: (m.get("Interest Coverage (EBITDA/Fin)") or 0) >= 3,     "Interest Coverage < 3"),
        ("Debt/FCF ≤ 5  or FCF>0",              lambda m: ((m.get("Debt / FCF (yrs)") or 0) <= 5) or ((m.get("TTM FCF") or 0) > 0), "Debt/FCF too high and FCF ≤ 0"),
    ],
    "scored": [
        # Value
        ("P/E (TTM) ≤ 15",                      lambda m: (m.get("P/E (TTM)") or 1e9) <= 15,                        20),
        ("FCF Yield (TTM) ≥ 5%",                lambda m: (m.get("FCF Yield (TTM) (%)") or 0) >= 5,                 20),
        # Quality
        ("Cash Conversion ≥ 80%",               lambda m: (m.get("Cash Conversion (CFO/NP, %)") or 0) >= 80,        20),
        ("TTM Gross Margin ≥ 20%",              lambda m: (m.get("TTM Gross Margin (%)") or 0) >= 20,               20),
        # Growth (light guard if you lack YoY growth series here)
        ("TTM Net Margin ≥ 5%",                 lambda m: (m.get("TTM Net Margin (%)") or 0) >= 5,                  20),
    ],
}


# Strict global cut-off
MIN_SCORE = 60

def evaluate(metrics: Dict[str, float], ruleset_key: str):
    """Return dict with pass/fail, score %, and detail arrays."""
    cfg = RULESETS[ruleset_key]

    # Mandatory gate
    mand, all_ok = [], True
    for label, fn, reason in cfg["mandatory"]:
        ok = bool(fn(metrics))
        mand.append((label, ok, "" if ok else reason))
        all_ok &= ok

    # Scored section
    scored, pts = [], 0
    total_pts = sum(w for *_ , w in cfg["scored"])
    for label, fn, weight in cfg["scored"]:
        ok = bool(fn(metrics))
        if ok: pts += weight
        scored.append((label, ok, weight))

    score_pct = round(100 * pts / total_pts, 1) if total_pts else 0.0
    return {
        "mandatory": mand,
        "scored":    scored,
        "score":     score_pct,
        "pass":      bool(all_ok and score_pct >= MIN_SCORE),
        "reasons":   [r for _, ok, r in mand if not ok],
    }
