# utils/rules.py
from typing import Dict, List, Tuple

# Two ready-to-use strategies. You can add more later.
RULESETS = {
    "Quality-Value": {
        "mandatory": [
            ("Positive EPS", lambda m: (m.get("EPS") or 0) > 0, "EPS must be > 0"),
            ("ROE ≥ 12%", lambda m: (m.get("ROE (%)") or 0) >= 12, "ROE ≥ 12%"),
            ("Debt-Asset ≤ 50%", lambda m: (m.get("Debt-Asset Ratio (%)") or 100) <= 50, "Debt/Asset ≤ 50%"),
        ],
        "scored": [
            ("P/E ≤ 15", lambda m: (m.get("P/E") or 1e9) <= 15, 20),
            ("P/B ≤ 2", lambda m: (m.get("P/B") or 1e9) <= 2, 20),
            ("Gross Margin ≥ 20%", lambda m: (m.get("Gross Profit Margin (%)") or 0) >= 20, 20),
            ("Current Ratio ≥ 2", lambda m: (m.get("Current Ratio") or 0) >= 2, 20),
            ("Dividend Yield ≥ 4%", lambda m: (m.get("Dividend Yield (%)") or 0) >= 4, 20),
        ],
    },
    "Dividend": {
        "mandatory": [
            ("Dividend Yield ≥ 3%", lambda m: (m.get("Dividend Yield (%)") or 0) >= 3, "Dividend Yield ≥ 3%"),
            ("Payout ≤ 80%", lambda m: (m.get("Dividend Payout Ratio (%)") or 100) <= 80, "Payout ≤ 80%"),
        ],
        "scored": [
            ("ROE ≥ 10%", lambda m: (m.get("ROE (%)") or 0) >= 10, 25),
            ("Debt-Asset ≤ 50%", lambda m: (m.get("Debt-Asset Ratio (%)") or 100) <= 50, 25),
            ("P/E ≤ 18", lambda m: (m.get("P/E") or 1e9) <= 18, 25),
            ("Current Ratio ≥ 1.5", lambda m: (m.get("Current Ratio") or 0) >= 1.5, 25),
        ],
    },
}

def evaluate_ratios(metrics: Dict[str, float], ruleset_key: str, min_score: int = 60):
    cfg = RULESETS[ruleset_key]
    # Mandatory gate
    mand: List[Tuple[str, bool, str]] = []
    all_mand_ok = True
    for label, fn, reason in cfg["mandatory"]:
        ok = bool(fn(metrics))
        mand.append((label, ok, "" if ok else reason))
        if not ok:
            all_mand_ok = False

    # Scored checks
    scored = []
    score_pts = 0
    max_pts = sum(w for _, _, w in cfg["scored"])
    for label, fn, weight in cfg["scored"]:
        ok = bool(fn(metrics))
        if ok:
            score_pts += weight
        scored.append((label, ok, weight))

    score_pct = round(100 * score_pts / max_pts, 1) if max_pts else 0.0
    decision = bool(all_mand_ok and (score_pct >= min_score))
    reasons = [r for _, ok, r in mand if not ok]

    return {
        "mandatory": mand,          # [(label, ok, reason_if_fail)]
        "scored": scored,           # [(label, ok, weight)]
        "score": score_pct,         # 0..100
        "pass": decision,           # True/False
        "reasons": reasons,         # unmet mandatory reasons
    }
