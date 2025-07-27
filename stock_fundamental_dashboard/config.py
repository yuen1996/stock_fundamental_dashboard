# Central config for thresholds keyed EXACTLY as ratios return from utils/calculations.py
RATIO_THRESHOLDS = {
    "ROE (%)": 12,                # good if >= 12
    "Current Ratio": 2,           # good if >= 2
    "Dividend Yield (%)": 4,      # good if >= 4%
    "P/E": 15,                    # good if <= 15
    "P/B": 2,                     # good if <= 2
    "Debt-Asset Ratio (%)": 50,   # good if <= 50
}
