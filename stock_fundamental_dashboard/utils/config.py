# Central config for fields, ratios, and thresholds

STOCK_FIELDS = [
    'Name', 'Industry', 'Year', 'NetProfit', 'Revenue', 'Equity', 'Asset',
    'Liability', 'Dividend', 'ShareOutstanding', 'Price'
]

RATIO_THRESHOLDS = {
    'ROE': 12,           # percent, good if >= 12
    'Current Ratio': 2,  # good if >= 2
    'Dividend Yield': 4, # percent
    'PE': 15,            # good if <= 15
    'PB': 2,             # good if <= 2
    'Debt Asset Ratio': 50, # percent
    # add more as you like
}
