def safe_div(a, b):
    try:
        if b == 0 or b is None: return None
        return a / b
    except Exception:
        return None

def percent(val):
    return round(val*100, 2) if val is not None else None

def calc_ratios(row):
    # Fetch all needed fields, handle missing gracefully
    np = row.get('Net profit', 0)
    gp = row.get('Gross Profit', 0)
    rev = row.get('Revenue', 0)
    cos = row.get('Cost Of Sales', 0)
    shares = row.get('Number of shares', 0)
    div = row.get('Dividend pay cent', 0)
    price = row.get('Each end of year share price', row.get('Current Share Price', 0))
    cur_asset = row.get('Current Asset', 0)
    cur_liab = row.get('Current Liability', 0)
    trade_rec = row.get('Trade Receivables (for calculate billing period)', 0)
    admin_exp = row.get('Administrative  Expenses', 0)
    sell_exp = row.get('Selling and distribution expenses', 0)
    total_asset = row.get('Total Asset', 0)
    total_liab = row.get('Total Liability', 0)
    sh_equity = row.get('Shareholder Equity', 0)
    reserves = row.get('Reserves', 0)
    inventories = row.get('Inventories  (-from current asset)', 0)
    intangibles = row.get('Intangible asset  (when calculate NTA need to deduct)', 0)
    current_price = row.get('Current Share Price', price)
    
    # EPS
    eps = safe_div(np, shares)
    # Billing Period (days)
    billing_period = safe_div(trade_rec, rev / 365) if rev else None
    # Debt-Asset Ratio
    debt_asset = percent(safe_div(total_liab, total_asset))
    # Three Fees Ratio
    three_fees = percent(safe_div(admin_exp + sell_exp, rev))
    # Total Cost %
    total_cost_pct = percent(safe_div(cos + admin_exp + sell_exp, rev))
    # Dividend payout ratio
    div_payout = percent(safe_div(div * shares, np))  # Assumes dividend in currency per share
    # Dividend yield
    div_yield = percent(safe_div(div, price)) if price else None
    # Current ratio
    curr_ratio = safe_div(cur_asset, cur_liab)
    # Quick ratio
    quick_ratio = safe_div(cur_asset - inventories, cur_liab) if cur_liab else None
    # Gross profit margin
    gross_margin = percent(safe_div(gp, rev))
    # Net profit margin
    net_margin = percent(safe_div(np, rev))
    # NTA per share
    nta = safe_div((total_asset - intangibles - total_liab), shares) if shares else None
    # P/E
    pe = safe_div(price, eps) if eps else None
    # P/B
    pb = safe_div(price, nta) if nta else None
    # ROE
    roe = percent(safe_div(np, sh_equity))
    
    # Put all into dictionary
    return {
        "EPS": eps,
        "Billing Period (days)": billing_period,
        "Debt-Asset Ratio (%)": debt_asset,
        "Three Fees Ratio (%)": three_fees,
        "Total Cost %": total_cost_pct,
        "Dividend Payout Ratio (%)": div_payout,
        "Dividend Yield (%)": div_yield,
        "Current Ratio": curr_ratio,
        "Quick Ratio": quick_ratio,
        "Gross Profit Margin (%)": gross_margin,
        "Net Profit Margin (%)": net_margin,
        "NTA per share": nta,
        "P/E": pe,
        "P/B": pb,
        "ROE (%)": roe,
    }
