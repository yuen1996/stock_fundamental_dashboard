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
