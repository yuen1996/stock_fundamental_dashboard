# ---- config helpers & globals (place at TOP of config.py) ----

def _f(label: str, key: str, *, unit: str | None = None,
       type: str = "float", help: str | None = None) -> dict:
    """Factory for a single field definition used by INDUSTRY_FORM_CATEGORIES."""
    return {"label": label, "key": key, "unit": unit, "type": type, "help": help}

_SECTION_ORDER = ["Income Statement", "Balance Sheet", "Cash Flow", "Other", "Industry KPIs"]

def _flatten_categories(cats: dict) -> list[dict]:
    out: list[dict] = []
    # keep a nice order, then any extra sections
    for sec in _SECTION_ORDER:
        out.extend(cats.get(sec, []))
    for sec, items in cats.items():
        if sec not in _SECTION_ORDER:
            out.extend(items)
    return out

def _prefix_quarter(cats: dict) -> dict:
    """Return a copy of `cats` with each field key prefixed with 'Q_' for quarterly forms."""
    out = {}
    for sec, items in cats.items():
        q_items = []
        for f in items:
            k = f.get("key")
            if k:
                q_items.append({**f, "key": k if k.startswith("Q_") else f"Q_{k}"})
            else:
                q_items.append(f)
        out[sec] = q_items
    return out

# ----------------------------
# TTM aggregation policy (ALL)
# ----------------------------

# Normalize keys: strip "Q_" and compare lowercase
def _norm_key(key: str) -> str:
    k = (key or "").strip()
    if k.upper().startswith("Q_"):
        k = k[2:]
    return k.lower()

# --- 1) Explicit MEAN list (true period-averages only)
_MEAN_KEYS = {
    # Banking – Other (Averages)
    "average earning assets",
    "average gross loans",
    "average deposits",
    # Working cap (explicit "(Avg)" keys across buckets)
    "accounts receivable",   # Receivables (Avg)
    "inventory",             # Inventory (Avg)
    "accounts payable",      # Payables (Avg)
}

# --- 2) Explicit SUM list (flows across all buckets)
_SUM_KEYS = {
    # Income Statement (all industries)
    "revenue", "costofsales", "gross profit",
    "operating profit", "ebitda",
    "net profit", "interest expense",
    "selling expenses", "administrative expenses",
    "three fees", "incometax", "income tax expense",  # both spellings caught
    "dps",  # dividend per share (quarterly payout)
    # Cash Flow (all industries)
    "cfo", "capex", "depamort", "depppe", "deprou", "depinvprop",
    # Financials – Industry KPIs
    "gross written premiums", "net claims incurred",
    "insurance operating expenses", "net fee income",
    "net interest income",
    # Banking – Income Statement
    "interest income", "interest expense",
    "net islamic income", "nii (incl islamic)",
    "fee income", "trading income", "other operating income",
    "operating income", "operating expenses", "provisions",
    "incometax",
    # Banking – Cash Flow
    "cfo (cash flow from ops, rm)",
    # Banking – TP overrides (NIM/CIR numerators & CIR denominator)
    "tp_bank_nim_num", "tp_bank_cir_num", "tp_bank_cir_den",
    # REITs – per unit payout in Other
    "dpu",
    # Energy/Materials – flows
    "annual production",
    # Construction – flows
    "new orders",
    # Plantation – flows
    "cpo output", "ffb input", "ffb harvested", "tonnes sold", "pko output",
    # Transport/Logistics – flows
    "teu throughput", "ask", "rpk",
    # Healthcare – flows
    "patient days", "admissions",
    # Leisure/Travel – flows
    "visits",
    # Telco – special flow
    "spectrum fees",
}

# --- 3) Explicit LAST list (levels, ratios, rates, counts)
_LAST_KEYS = {
    # Balance Sheet (ALL buckets)
    "total assets", "total liabilities", "equity", "equity (book value)",
    "current assets", "current liabilities", "intangible assets",
    "biologicalassets", "total borrowings",
    "cash & cash equivalents", "cash & cash equivalents (rm)",  # robust
    "shares", "price",
    # Retail/Telco – lease liabilities
    "leaseliabcurrent", "leaseliabnoncurrent",
    # Property/REITs
    "nav per unit", "units outstanding", "rnav per share",
    "net debt", "gdv", "landbank",
    # Banking – Balance Sheet (expanded)
    "gross loans", "net loans",
    "deposits", "demand deposits", "savings deposits",
    "fixed/time deposits", "money market deposits", "other deposits",
    "fvoci investments", "amortised cost investments",
    "reverse repos", "fvtpl investments", "fvtpl derivatives",
    "earning assets",
    "non-performing loans", "loan loss reserve",
    "risk-weighted assets", "regulatory reserve",
    "total assets (rm)", "total liabilities (rm)", "shareholders’ equity (rm)",
    # Banking – capital & liquidity ratios (Other)
    "cet1 ratio", "tier 1 ratio", "total capital ratio",
    "casa ratio", "casa (core, %)", "loan-to-deposit ratio",
    "nim", "lcr", "nsfr",
    # Banking – TP leverage (use latest levels)
    "tp_bank_lev_num", "tp_bank_lev_den",
    # Utilities – Other
    "rab", "allowed return", "availability factor", "capex to revenue",
    # Energy/Materials – Other (levels/rates)
    "lifting cost", "proven reserves", "strip ratio",
    "head grade", "average selling price",
    # Tech – Other
    "r&d expense", "r&d intensity", "arr", "ndr", "gross retention",
    "cac", "ltv", "mau", "rule of 40",
    # Retail – KPIs
    "sssg", "average basket size", "store count",
    # Manufacturing – KPIs
    "capacity utilization", "order backlog", "backlog coverage",
    # Healthcare – Other
    "bed count", "bed occupancy", "alos",  # ALOS is a rate (days)
    # Telco – Other
    "arpu", "subscribers", "churn rate", "average data usage",
    # Construction – Other
    "orderbook", "tender book", "win rate",
    # Plantation – Other
    "planted hectares", "unit cash cost", "cpo asp",
    "mature area", "avg tree age",
    # Transport/Logistics – Other
    "load factor", "yield per km", "on-time performance", "fleet size",
    # Leisure/Travel – Other
    "hotel occupancy", "revpar", "win rate", "average spend", "room count",
}

# For robustness, also treat anything that *looks like* a ratio/percent/rate as "last".
def _looks_like_ratio_or_rate(label_or_key: str) -> bool:
    s = (label_or_key or "").lower()
    return any(t in s for t in [
        "%", " ratio", "margin", "yield", " rate", "coverage", "turnover",
        "roe", "roa", "nim", "c/i", "cir", "ldr", "casa", "lcr", "nsfr",
        "gearing", "interest coverage", "net debt/ebitda", "capex/revenue",
        "capex to revenue", "cash conversion", "conversion cycle",
        "inventory days", "receivable days", "payable days", "ccc",
        "occupancy", "wale", "reversion", "rule of 40", "ndr", "retention",
        "churn", "load factor", "on-time performance", "sales rate",
    ])

def _ttm_agg_for(category: str, key: str, label: str) -> str:
    cat    = (category or "").strip()
    k_norm = _norm_key(key)
    l_norm = (label or key or "").strip().lower()

    # 1) Explicit MEAN wins first
    if k_norm in _MEAN_KEYS or l_norm in _MEAN_KEYS:
        return "mean"

    # 2) Explicit LAST (levels, counts, prices) — BEFORE category defaults
    if k_norm in _LAST_KEYS or l_norm in _LAST_KEYS:
        return "last"

    # 3) Explicit SUM
    if k_norm in _SUM_KEYS or l_norm in _SUM_KEYS:
        return "sum"

    # 4) Category defaults
    if cat in ("Income Statement", "Cash Flow"):
        return "sum"
    if cat == "Balance Sheet":
        return "last"

    # 5) Ratios/rates/percentages → last
    if _looks_like_ratio_or_rate(l_norm) or _looks_like_ratio_or_rate(k_norm):
        return "last"

    # 6) Conservative fallback
    return "last"

# Buckets used by the UI dropdowns etc.
INDUSTRY_BUCKETS = (
    "General",
    "Manufacturing", "Retail", "Financials", "Banking", "REITs",
    "Utilities", "Energy/Materials", "Tech", "Healthcare",
    "Telco", "Construction", "Plantation", "Property",
    "Transportation/Logistics", "Leisure/Travel",
)

# --------- Category definitions per bucket (ANNUAL) ----------
# Banking stays one list (see at end)

INDUSTRY_FORM_CATEGORIES = {
    "Manufacturing": {
        "Income Statement": [
            _f("Revenue (TTM/FY)", "Revenue", unit="RM"),
            _f("Cost of Sales", "CostOfSales", unit="RM"),
            _f("Gross Profit", "Gross Profit", unit="RM",
                help="Optional. If left blank, engine computes: Revenue − Cost of Sales"),
            _f("Operating Profit", "Operating Profit", unit="RM"),
            _f("EBITDA", "EBITDA", unit="RM",
                help="Optional. If blank, engine computes: Operating Profit + Depreciation & Amortization (Total). If DepAmort is blank, it sums DepPPE + DepROU + DepInvProp."),
            _f("Net Profit", "Net Profit", unit="RM"),
            _f("Interest Expense", "Interest Expense", unit="RM"),
            _f("Selling & Distribution Expenses", "Selling Expenses", unit="RM"),
            _f("Administrative Expenses", "Administrative Expenses", unit="RM"),
            _f("Three Fees (Selling+Admin+Finance, RM)", "Three Fees", unit="RM",
               help="Optional. If left blank, engine sums: Selling & Distribution Expenses + Administrative Expenses + Interest Expense"),
            _f("Income Tax Expense", "IncomeTax", unit="RM"),
            _f("Dividend per Share (TTM, RM)", "DPS", unit="RM"),
        ],
        # --- Banking → Balance Sheet (expanded) ---
        "Balance Sheet": [
            _f("Total Assets", "Total Assets", unit="RM"),
            _f("Total Liabilities", "Total Liabilities", unit="RM"),
            _f("Equity (Book Value)", "Equity", unit="RM"),
            _f("Current Assets", "Current Assets", unit="RM"),
            _f("Current Liabilities", "Current Liabilities", unit="RM"),
            _f("Receivables (Avg)", "Accounts Receivable", unit="RM"),
            _f("Inventory (Avg)", "Inventory", unit="RM"),
            _f("Payables (Avg)", "Accounts Payable", unit="RM"),
            _f("Intangible Assets", "Intangible Assets", unit="RM"),
            _f("Biological Assets", "BiologicalAssets", unit="RM"),
            _f("Total Borrowings", "Total Borrowings", unit="RM"),
            _f("Cash & Equivalents", "Cash & Cash Equivalents", unit="RM"),
            _f("Shares Outstanding", "Shares", type="int"),
            _f("Annual Price per Share (RM)", "Price", unit="RM"),
        ],
        "Cash Flow": [
            _f("CFO (Cash Flow from Ops)", "CFO", unit="RM"),
            _f("Capex", "Capex", unit="RM"),
            _f("Depreciation & Amortization (Total)", "DepAmort", unit="RM",
                help="Optional. If blank, engine sums Depreciation - PPE + Depreciation - ROU Assets + Depreciation of Investment Properties (ignores missing parts)."),
            _f("Depreciation - PPE", "DepPPE", unit="RM"),
            _f("Depreciation - ROU Assets", "DepROU", unit="RM"),
            _f("Depreciation of Investment Properties", "DepInvProp", unit="RM"),  # ← added
        ],
        "Other": [],
        "Industry KPIs": [
            _f("Capacity Utilization (%)", "Capacity Utilization", unit="%"),
            _f("Order Backlog", "Order Backlog", unit="RM"),
            _f("Backlog Coverage (x)", "Backlog Coverage"),
        ],
    },

    "Retail": {
        "Income Statement": [
            _f("Revenue (TTM/FY)", "Revenue", unit="RM"),
            _f("Cost of Sales", "CostOfSales", unit="RM"),
            _f("Gross Profit", "Gross Profit", unit="RM",
                help="Optional. If left blank, engine computes: Revenue − Cost of Sales"),
            _f("Operating Profit", "Operating Profit", unit="RM"),
            _f("EBITDA", "EBITDA", unit="RM",
                help="Optional. If blank, engine computes: Operating Profit + Depreciation & Amortization (Total). If DepAmort is blank, it sums DepPPE + DepROU + DepInvProp."),
            _f("Net Profit", "Net Profit", unit="RM"),
            _f("Interest Expense", "Interest Expense", unit="RM"),
            _f("Selling & Distribution Expenses", "Selling Expenses", unit="RM"),
            _f("Administrative Expenses", "Administrative Expenses", unit="RM"),
            _f("Three Fees (Selling+Admin+Finance, RM)", "Three Fees", unit="RM",
               help="Optional. If left blank, engine sums: Selling & Distribution Expenses + Administrative Expenses + Interest Expense"),
            _f("Income Tax Expense", "IncomeTax", unit="RM"),
            _f("Dividend per Share (TTM, RM)", "DPS", unit="RM"),
        ],
        "Balance Sheet": [
            _f("Total Assets", "Total Assets", unit="RM"),
            _f("Total Liabilities", "Total Liabilities", unit="RM"),
            _f("Equity (Book Value)", "Equity", unit="RM"),
            _f("Current Assets", "Current Assets", unit="RM"),
            _f("Current Liabilities", "Current Liabilities", unit="RM"),
            _f("Receivables (Avg)", "Accounts Receivable", unit="RM"),
            _f("Inventory (Avg)", "Inventory", unit="RM"),
            _f("Payables (Avg)", "Accounts Payable", unit="RM"),
            _f("Intangible Assets", "Intangible Assets", unit="RM"),
            _f("Lease Liabilities (Current)", "LeaseLiabCurrent", unit="RM"),
            _f("Lease Liabilities (Non-Current)", "LeaseLiabNonCurrent", unit="RM"),
            _f("Total Borrowings", "Total Borrowings", unit="RM"),
            _f("Cash & Equivalents", "Cash & Cash Equivalents", unit="RM"),
            _f("Shares Outstanding", "Shares", type="int"),
            _f("Annual Price per Share (RM)", "Price", unit="RM"),
        ],
        "Cash Flow": [
            _f("CFO (Cash Flow from Ops)", "CFO", unit="RM"),
            _f("Capex", "Capex", unit="RM"),
            _f("Depreciation & Amortization (Total)", "DepAmort", unit="RM",
                help="Optional. If blank, engine sums Depreciation - PPE + Depreciation - ROU Assets + Depreciation of Investment Properties (ignores missing parts)."),
            _f("Depreciation - PPE", "DepPPE", unit="RM"),
            _f("Depreciation - ROU Assets", "DepROU", unit="RM"),
            _f("Depreciation of Investment Properties", "DepInvProp", unit="RM"),  # ← added
        ],
        "Other": [],
        "Industry KPIs": [
            _f("Same-Store Sales Growth (%)", "SSSG", unit="%"),
            _f("Average Basket Size (RM)", "Average Basket Size", unit="RM"),
            _f("Store Count", "Store Count", type="int"),
        ],
    },

    "Financials": {
        "Income Statement": [
            _f("Revenue (TTM/FY)", "Revenue", unit="RM"),
            _f("Gross Profit", "Gross Profit", unit="RM",
                help="Optional. If left blank, engine computes: Revenue − Cost of Sales"),
            _f("Operating Profit", "Operating Profit", unit="RM"),
            _f("EBITDA", "EBITDA", unit="RM",
                help="Optional. If blank, engine computes: Operating Profit + Depreciation & Amortization (Total). If DepAmort is blank, it sums DepPPE + DepROU + DepInvProp."),
            _f("Net Profit", "Net Profit", unit="RM"),
            _f("Interest Expense", "Interest Expense", unit="RM"),
            _f("Administrative Expenses", "Administrative Expenses", unit="RM"),
            _f("Income Tax Expense", "IncomeTax", unit="RM"),
            _f("Dividend per Share (TTM, RM)", "DPS", unit="RM"),
        ],
        "Balance Sheet": [
            _f("Total Assets", "Total Assets", unit="RM"),
            _f("Total Liabilities", "Total Liabilities", unit="RM"),
            _f("Equity (Book Value)", "Equity", unit="RM"),
            _f("Current Assets", "Current Assets", unit="RM"),
            _f("Current Liabilities", "Current Liabilities", unit="RM"),
            _f("Intangible Assets", "Intangible Assets", unit="RM"),
            _f("Total Borrowings", "Total Borrowings", unit="RM"),
            _f("Cash & Equivalents", "Cash & Cash Equivalents", unit="RM"),
            _f("Shares Outstanding", "Shares", type="int"),
            _f("Annual Price per Share (RM)", "Price", unit="RM"),
        ],
        "Cash Flow": [
            _f("CFO (Cash Flow from Ops)", "CFO", unit="RM"),
            _f("Capex", "Capex", unit="RM"),
            _f("Depreciation & Amortization (Total)", "DepAmort", unit="RM",
                help="Optional. If blank, engine sums Depreciation - PPE + Depreciation - ROU Assets + Depreciation of Investment Properties (ignores missing parts)."),
            _f("Depreciation of Investment Properties", "DepInvProp", unit="RM"),  # ← added
        ],
        "Other": [],
        "Industry KPIs": [
            _f("Gross Written Premiums", "Gross Written Premiums", unit="RM"),
            _f("Net Claims Incurred", "Net Claims Incurred", unit="RM"),
            _f("Insurance Operating Expenses", "Insurance Operating Expenses", unit="RM"),
            _f("Assets Under Management (AUM)", "AUM", unit="RM"),
            _f("Net Fee Income", "Net Fee Income", unit="RM"),
            _f("Finance Receivables", "Finance Receivables", unit="RM"),
            _f("Net Interest Income", "Net Interest Income", unit="RM"),
        ],
    },

    # Banking — single list (lean and aligned)
    "Banking": {
        "Income Statement": [
            _f("Dividend per Share (TTM, RM)", "DPS", unit="RM"),
            _f("Interest Income (RM)", "Interest Income", unit="RM"),
            _f("Interest Expense (RM)", "Interest Expense", unit="RM"),
            _f("Net Income from Islamic Banking (RM)", "Net Islamic Income", unit="RM"),
            _f("Net Interest Income (incl. Islamic, RM)", "NII (incl Islamic)", unit="RM",
            help="Optional. If left blank, engine computes: (Interest Income − Interest Expense) + Net Islamic Income"),
            _f("Fee & Commission Income (RM)", "Fee Income", unit="RM"),
            _f("Trading & Investment Income (RM)", "Trading Income", unit="RM"),
            _f("Other Operating Income (RM)", "Other Operating Income", unit="RM"),
            _f("Operating Income (Total, RM)", "Operating Income", unit="RM",
            help="Optional. If blank, engine sums NII(incl Islamic) + Fee + Trading + Other"),
            _f("Operating Expenses (RM)", "Operating Expenses", unit="RM"),
            _f("Provisions / Impairment (RM)", "Provisions", unit="RM"),
            _f("Income Tax Expense (RM)", "IncomeTax", unit="RM"),
            _f("Net Profit (RM)", "Net Profit", unit="RM"),
        ],
        # --- Banking → Balance Sheet (expanded) ---
        "Balance Sheet": [
            _f("Gross Loans / Financing (RM)", "Gross Loans", unit="RM"),
            _f("Deposits (RM)", "Deposits", unit="RM"),
            _f("Demand Deposits (RM)", "Demand Deposits", unit="RM"),
            _f("Savings Deposits (RM)", "Savings Deposits", unit="RM"),
            _f("Fixed/Time Deposits (RM)", "Fixed/Time Deposits", unit="RM"),
            _f("Money Market / NID Deposits (RM)", "Money Market Deposits", unit="RM"),
            _f("Other Deposits (RM)", "Other Deposits", unit="RM"),

            # --- IEA components (Strict) ---
            _f("Net Loans/Financing (RM)", "Net Loans", unit="RM",
            help="Group A11: use the TOTAL 'Net loans, advances and financing' (customers + financial institutions). If left blank, engine uses: Gross Loans − Loan Loss Reserve."),
            _f("Financial Investments at FVOCI (RM)", "FVOCI Investments", unit="RM",
            help="Group A10: 'Financial investments measured at FVOCI' — use the TOTAL."),
            _f("Financial Investments at Amortised Cost (RM)", "Amortised Cost Investments", unit="RM",
            help="Group A10: 'Financial investments measured at amortised cost' — use the TOTAL."),
            _f("Financial Assets Purchased under Resale/Reverse Repo (RM)", "Reverse Repos", unit="RM",
            help="Balance sheet line: 'Financial assets purchased under resale agreements' (reverse-repo assets)."),

            # --- FVTPL add-ons (StrictPlus) ---
            _f("Financial Assets at FVTPL (RM)", "FVTPL Investments", unit="RM",
            help="Include ONLY interest-bearing FVTPL: government/treasury bills & bonds (e.g., MGS/MGII/BNM bills/foreign gov), money-market instruments, Cagamas, corporate bonds/sukuk, structured deposits. EXCLUDE equities, unit trusts/mutual funds, and ALL derivative assets."),
            _f("Derivative Assets at FVTPL (RM)", "FVTPL Derivatives", unit="RM",
            help="Derivative assets at fair value (trading/hedging). Always EXCLUDED from IEA (both Strict and StrictPlus); tracked here for completeness."),

            # Single IEA field (kept for convenience / averaging)
            _f("Earning Assets (RM)", "Earning Assets", unit="RM",
            help="Manual override if you prefer to type the denominator directly. Formulas — Strict IEA = Net Loans + FVOCI + Amortised Cost + Reverse Repos. StrictPlus (add interest-bearing FVTPL) = Strict IEA + FVTPL Investments (interest-bearing only). Do NOT add derivatives, cash & short-term funds, or interbank placements unless your policy requires it."),

            _f("Total Borrowings (RM)", "Total Borrowings", unit="RM"),
            _f("Non-Performing Loans / Impaired Financing (RM)", "Non-Performing Loans", unit="RM"),
            _f("Loan Loss Reserves / Allowance for ECL (RM)", "Loan Loss Reserve", unit="RM"),
            _f("Risk-Weighted Assets (RM)", "Risk-Weighted Assets", unit="RM"),
            _f("Cash & Cash Equivalents (RM)", "Cash & Cash Equivalents", unit="RM"),
            _f("Total Assets (RM)", "Total Assets", unit="RM"),
            _f("Total Liabilities (RM)", "Total Liabilities", unit="RM"),
            _f("Shareholders’ Equity (RM)", "Equity", unit="RM"),
            _f("Regulatory Reserve (RM)", "Regulatory Reserve", unit="RM"),
            _f("Shares Outstanding", "Shares", type="int"),
            _f("Annual Price per Share (RM)", "Price", unit="RM"),
        ],
        "Cash Flow": [
            _f("CFO (Cash Flow from Ops, RM)", "CFO", unit="RM"),
        ],
        "Other": [
            _f("CET1 Ratio (%)", "CET1 Ratio", unit="%"),
            _f("Tier 1 Capital Ratio (%)", "Tier 1 Ratio", unit="%"),
            _f("Total Capital Ratio (%)", "Total Capital Ratio", unit="%"),
            _f("CASA Ratio (%)", "CASA Ratio", unit="%"),
            _f("CASA (Core, %)", "CASA (Core, %)", unit="%"),
            _f("Loan-to-Deposit Ratio (%)", "Loan-to-Deposit Ratio", unit="%"),
            _f("Net Interest/Financing Margin (%)", "NIM", unit="%"),
            _f("Liquidity Coverage Ratio (LCR, %)", "LCR", unit="%"),
            _f("Net Stable Funding Ratio (NSFR, %)", "NSFR", unit="%"),
            _f("Average Earning Assets (RM)", "Average Earning Assets", unit="RM"),
            _f("Average Gross Loans / Financing (RM)", "Average Gross Loans", unit="RM"),
            _f("Average Deposits (RM)", "Average Deposits", unit="RM"),

            # Three-proportion (bank) optional raw overrides (match code’s canonical keys)
            _f("TP (Bank) – NIM Numerator: Net Interest Income incl. Islamic (RM)",
            "TP_Bank_NIM_Num", unit="RM",
            help="Optional override; if blank, engine uses 'NII (incl Islamic)' from Income Statement."),
            _f("TP (Bank) – NIM Denominator: Average Earning Assets (RM)",
            "TP_Bank_NIM_Den", unit="RM",
            help="Optional override; if blank, engine uses 'Average Earning Assets'."),
            _f("TP (Bank) – CIR Numerator: Operating Expenses (RM)",
            "TP_Bank_CIR_Num", unit="RM",
            help="Optional override; if blank, engine uses 'Operating Expenses'."),
            _f("TP (Bank) – CIR Denominator: Operating Income (RM)",
            "TP_Bank_CIR_Den", unit="RM",
            help="Optional override; if blank, engine uses 'Operating Income'."),
            _f("TP (Bank) – Leverage Numerator: Total Assets (RM)",
            "TP_Bank_Lev_Num", unit="RM",
            help="Optional override; if blank, engine uses 'Total Assets'."),
            _f("TP (Bank) – Leverage Denominator: Equity (RM)",
            "TP_Bank_Lev_Den", unit="RM",
            help="Optional override; if blank, engine uses 'Equity'."),
        ],
        "Industry KPIs": [],
    },

    "REITs": {
        "Income Statement": [
            _f("Revenue (TTM/FY)", "Revenue", unit="RM"),
            _f("Gross Profit", "Gross Profit", unit="RM",
                help="Optional. If left blank, engine computes: Revenue − Cost of Sales"),
            _f("Operating Profit", "Operating Profit", unit="RM"),
            _f("EBITDA", "EBITDA", unit="RM",
                help="Optional. If blank, engine computes: Operating Profit + Depreciation & Amortization (Total). If DepAmort is blank, it sums DepPPE + DepROU + DepInvProp."),
            _f("Net Profit", "Net Profit", unit="RM"),
            _f("Dividend per Share (TTM, RM)", "DPS", unit="RM"),
            _f("Interest Expense", "Interest Expense", unit="RM"),
            _f("Income Tax Expense", "IncomeTax", unit="RM"),
        ],
        "Balance Sheet": [
            _f("Total Assets", "Total Assets", unit="RM"),
            _f("Total Liabilities", "Total Liabilities", unit="RM"),
            _f("Equity (Book Value)", "Equity", unit="RM"),
            _f("Current Assets", "Current Assets", unit="RM"),
            _f("Current Liabilities", "Current Liabilities", unit="RM"),
            _f("Receivables (Avg)", "Accounts Receivable", unit="RM"),
            _f("Payables (Avg)", "Accounts Payable", unit="RM"),
            _f("Intangible Assets", "Intangible Assets", unit="RM"),
            _f("Total Borrowings", "Total Borrowings", unit="RM"),
            _f("Cash & Equivalents", "Cash & Cash Equivalents", unit="RM"),
            _f("Annual Price per Share (RM)", "Price", unit="RM"),
            _f("Units Outstanding", "Units Outstanding", type="int"),
        ],
        "Cash Flow": [
            _f("CFO (Cash Flow from Ops)", "CFO", unit="RM"),
            _f("Capex", "Capex", unit="RM"),
            _f("Depreciation & Amortization (Total)", "DepAmort", unit="RM",
                help="Optional. If blank, engine sums Depreciation - PPE + Depreciation - ROU Assets + Depreciation of Investment Properties (ignores missing parts)."),
            _f("Depreciation - PPE", "DepPPE", unit="RM"),
            _f("Depreciation - ROU Assets", "DepROU", unit="RM"),
            _f("Depreciation - Investment Property", "DepInvProp", unit="RM"),
        ],
        "Other": [
            _f("NAV per Unit", "NAV per Unit", unit="RM",
                help="Optional. If blank, engine computes: Equity ÷ Units Outstanding."),
            _f("Occupancy (%)", "Occupancy", unit="%"),
            _f("WALE (years)", "WALE", unit="years"),
            _f("Rental Reversion (%)", "Rental Reversion", unit="%"),
            _f("DPU (RM)", "DPU", unit="RM"),
            _f("Average Cost of Debt (%)", "Avg Cost of Debt", unit="%",
                help="Optional. If blank, engine computes: Interest Expense ÷ Total Borrowings × 100."),
            _f("Hedged Debt (%)", "Hedged Debt", unit="%"),
        ],
        "Industry KPIs": [],
    },

    "Utilities": {
        "Income Statement": [
            _f("Revenue (TTM/FY)", "Revenue", unit="RM"),
            _f("Cost of Sales", "CostOfSales", unit="RM"),
            _f("Gross Profit", "Gross Profit", unit="RM",
                help="Optional. If left blank, engine computes: Revenue − Cost of Sales"),
            _f("Operating Profit", "Operating Profit", unit="RM"),
            _f("EBITDA", "EBITDA", unit="RM",
                help="Optional. If blank, engine computes: Operating Profit + Depreciation & Amortization (Total). If DepAmort is blank, it sums DepPPE + DepROU + DepInvProp."),
            _f("Net Profit", "Net Profit", unit="RM"),
            _f("Interest Expense", "Interest Expense", unit="RM"),
            _f("Selling & Distribution Expenses", "Selling Expenses", unit="RM"),
            _f("Administrative Expenses", "Administrative Expenses", unit="RM"),
            _f("Three Fees (Selling+Admin+Finance, RM)", "Three Fees", unit="RM",
               help="Optional. If left blank, engine sums: Selling & Distribution Expenses + Administrative Expenses + Interest Expense"),
            _f("Income Tax Expense", "IncomeTax", unit="RM"),
            _f("Dividend per Share (TTM, RM)", "DPS", unit="RM"),
        ],
        "Balance Sheet": [
            _f("Total Assets", "Total Assets", unit="RM"),
            _f("Total Liabilities", "Total Liabilities", unit="RM"),
            _f("Equity (Book Value)", "Equity", unit="RM"),
            _f("Current Assets", "Current Assets", unit="RM"),
            _f("Current Liabilities", "Current Liabilities", unit="RM"),
            _f("Receivables (Avg)", "Accounts Receivable", unit="RM"),
            _f("Inventory (Avg)", "Inventory", unit="RM"),
            _f("Payables (Avg)", "Accounts Payable", unit="RM"),
            _f("Intangible Assets", "Intangible Assets", unit="RM"),
            _f("Total Borrowings", "Total Borrowings", unit="RM"),
            _f("Cash & Equivalents", "Cash & Cash Equivalents", unit="RM"),
            _f("Shares Outstanding", "Shares", type="int"),
            _f("Annual Price per Share (RM)", "Price", unit="RM"),
        ],
        "Cash Flow": [
            _f("CFO (Cash Flow from Ops)", "CFO", unit="RM"),
            _f("Capex", "Capex", unit="RM"),
            _f("Depreciation & Amortization (Total)", "DepAmort", unit="RM",
                help="Optional. If blank, engine sums Depreciation - PPE + Depreciation - ROU Assets + Depreciation of Investment Properties (ignores missing parts)."),
            _f("Depreciation - PPE", "DepPPE", unit="RM"),
            _f("Depreciation - ROU Assets", "DepROU", unit="RM"),
            _f("Depreciation of Investment Properties", "DepInvProp", unit="RM"),  # ← added
        ],
        "Other": [
            _f("Regulated Asset Base (RAB)", "RAB", unit="RM"),
            _f("Allowed Return (%)", "Allowed Return", unit="%"),
            _f("Availability Factor (%)", "Availability Factor", unit="%"),
            _f("Capex to Revenue (%)", "Capex to Revenue", unit="%",
                help="Optional. If blank, engine computes: Capex ÷ Revenue × 100."),
        ],
        "Industry KPIs": [],
    },

    "Energy/Materials": {
        "Income Statement": [
            _f("Revenue (TTM/FY)", "Revenue", unit="RM"),
            _f("Cost of Sales", "CostOfSales", unit="RM"),
            _f("Gross Profit", "Gross Profit", unit="RM",
                help="Optional. If left blank, engine computes: Revenue − Cost of Sales"),
            _f("Operating Profit", "Operating Profit", unit="RM"),
            _f("EBITDA", "EBITDA", unit="RM",
                help="Optional. If blank, engine computes: Operating Profit + Depreciation & Amortization (Total). If DepAmort is blank, it sums DepPPE + DepROU + DepInvProp."),
            _f("Net Profit", "Net Profit", unit="RM"),
            _f("Interest Expense", "Interest Expense", unit="RM"),
            _f("Selling & Distribution Expenses", "Selling Expenses", unit="RM"),
            _f("Administrative Expenses", "Administrative Expenses", unit="RM"),
            _f("Three Fees (Selling+Admin+Finance, RM)", "Three Fees", unit="RM",
               help="Optional. If left blank, engine sums: Selling & Distribution Expenses + Administrative Expenses + Interest Expense"),
            _f("Income Tax Expense", "IncomeTax", unit="RM"),
            _f("Dividend per Share (TTM, RM)", "DPS", unit="RM"),
        ],
        "Balance Sheet": [
            _f("Total Assets", "Total Assets", unit="RM"),
            _f("Total Liabilities", "Total Liabilities", unit="RM"),
            _f("Equity (Book Value)", "Equity", unit="RM"),
            _f("Current Assets", "Current Assets", unit="RM"),
            _f("Current Liabilities", "Current Liabilities", unit="RM"),
            _f("Receivables (Avg)", "Accounts Receivable", unit="RM"),
            _f("Inventory (Avg)", "Inventory", unit="RM"),
            _f("Payables (Avg)", "Accounts Payable", unit="RM"),
            _f("Intangible Assets", "Intangible Assets", unit="RM"),
            _f("Total Borrowings", "Total Borrowings", unit="RM"),
            _f("Cash & Equivalents", "Cash & Cash Equivalents", unit="RM"),
            _f("Shares Outstanding", "Shares", type="int"),
            _f("Annual Price per Share (RM)", "Price", unit="RM"),
        ],
        "Cash Flow": [
            _f("CFO (Cash Flow from Ops)", "CFO", unit="RM"),
            _f("Capex", "Capex", unit="RM"),
            _f("Depreciation & Amortization (Total)", "DepAmort", unit="RM",
                help="Optional. If blank, engine sums Depreciation - PPE + Depreciation - ROU Assets + Depreciation of Investment Properties (ignores missing parts)."),
            _f("Depreciation - PPE", "DepPPE", unit="RM"),
            _f("Depreciation - ROU Assets", "DepROU", unit="RM"),
            _f("Depreciation of Investment Properties", "DepInvProp", unit="RM"),  # ← added
        ],
        "Other": [
            _f("Lifting Cost (RM/boe)", "Lifting Cost", unit="RM/boe"),
            _f("Proven Reserves (boe)", "Proven Reserves", unit="boe"),
            _f("Annual Production (boe)", "Annual Production", unit="boe"),
            _f("Strip Ratio (mining)", "Strip Ratio"),
            _f("Head Grade (g/t)", "Head Grade"),
            _f("Average Selling Price (RM/tonne)", "Average Selling Price", unit="RM"),
        ],
        "Industry KPIs": [],
    },

    "Tech": {
        "Income Statement": [
            _f("Revenue (TTM/FY)", "Revenue", unit="RM"),
            _f("Cost of Sales", "CostOfSales", unit="RM"),
            _f("Gross Profit", "Gross Profit", unit="RM",
                help="Optional. If left blank, engine computes: Revenue − Cost of Sales"),
            _f("Operating Profit", "Operating Profit", unit="RM"),
            _f("EBITDA", "EBITDA", unit="RM",
                help="Optional. If blank, engine computes: Operating Profit + Depreciation & Amortization (Total). If DepAmort is blank, it sums DepPPE + DepROU + DepInvProp."),
            _f("Net Profit", "Net Profit", unit="RM"),
            _f("Interest Expense", "Interest Expense", unit="RM"),
            _f("Selling & Distribution Expenses", "Selling Expenses", unit="RM"),
            _f("Administrative Expenses", "Administrative Expenses", unit="RM"),
            _f("Three Fees (Selling+Admin+Finance, RM)", "Three Fees", unit="RM",
               help="Optional. If left blank, engine sums: Selling & Distribution Expenses + Administrative Expenses + Interest Expense"),
            _f("Income Tax Expense", "IncomeTax", unit="RM"),
            _f("Dividend per Share (TTM, RM)", "DPS", unit="RM"),
        ],
        "Balance Sheet": [
            _f("Total Assets", "Total Assets", unit="RM"),
            _f("Total Liabilities", "Total Liabilities", unit="RM"),
            _f("Equity (Book Value)", "Equity", unit="RM"),
            _f("Current Assets", "Current Assets", unit="RM"),
            _f("Current Liabilities", "Current Liabilities", unit="RM"),
            _f("Receivables (Avg)", "Accounts Receivable", unit="RM"),
            _f("Inventory (Avg)", "Inventory", unit="RM"),
            _f("Payables (Avg)", "Accounts Payable", unit="RM"),
            _f("Intangible Assets", "Intangible Assets", unit="RM"),
            _f("Total Borrowings", "Total Borrowings", unit="RM"),
            _f("Cash & Equivalents", "Cash & Cash Equivalents", unit="RM"),
            _f("Shares Outstanding", "Shares", type="int"),
            _f("Annual Price per Share (RM)", "Price", unit="RM"),
        ],
        "Cash Flow": [
            _f("CFO (Cash Flow from Ops)", "CFO", unit="RM"),
            _f("Capex", "Capex", unit="RM"),
            _f("Depreciation & Amortization (Total)", "DepAmort", unit="RM",
                help="Optional. If blank, engine sums Depreciation - PPE + Depreciation - ROU Assets + Depreciation of Investment Properties (ignores missing parts)."),
            _f("Depreciation - PPE", "DepPPE", unit="RM"),
            _f("Depreciation - ROU Assets", "DepROU", unit="RM"),
            _f("Depreciation of Investment Properties", "DepInvProp", unit="RM"),  # ← added
        ],
        "Other": [
            _f("R&D Expense", "R&D Expense", unit="RM"),
            _f("R&D Intensity (%)", "R&D Intensity", unit="%",
                help="Optional. If blank, engine computes: R&D Expense ÷ Revenue × 100."),
            _f("Annual Recurring Revenue (ARR)", "ARR", unit="RM"),
            _f("Net Dollar Retention (%)", "NDR", unit="%"),
            _f("Gross Retention (%)", "Gross Retention", unit="%"),
            _f("Customer Acquisition Cost (CAC)", "CAC", unit="RM"),
            _f("Lifetime Value (LTV)", "LTV", unit="RM"),
            _f("Monthly Active Users (MAU)", "MAU", type="int"),
            _f("Rule of 40 (%)", "Rule of 40", unit="%"),
        ],
        "Industry KPIs": [],
    },

    "Healthcare": {
        "Income Statement": [
            _f("Revenue (TTM/FY)", "Revenue", unit="RM"),
            _f("Cost of Sales", "CostOfSales", unit="RM"),
            _f("Gross Profit", "Gross Profit", unit="RM",
                help="Optional. If left blank, engine computes: Revenue − Cost of Sales"),
            _f("Operating Profit", "Operating Profit", unit="RM"),
            _f("EBITDA", "EBITDA", unit="RM",
                help="Optional. If blank, engine computes: Operating Profit + Depreciation & Amortization (Total). If DepAmort is blank, it sums DepPPE + DepROU + DepInvProp."),
            _f("Net Profit", "Net Profit", unit="RM"),
            _f("Interest Expense", "Interest Expense", unit="RM"),
            _f("Selling & Distribution Expenses", "Selling Expenses", unit="RM"),
            _f("Administrative Expenses", "Administrative Expenses", unit="RM"),
            _f("Three Fees (Selling+Admin+Finance, RM)", "Three Fees", unit="RM",
               help="Optional. If left blank, engine sums: Selling & Distribution Expenses + Administrative Expenses + Interest Expense"),
            _f("Income Tax Expense", "IncomeTax", unit="RM"),
            _f("Dividend per Share (TTM, RM)", "DPS", unit="RM"),
        ],
        "Balance Sheet": [
            _f("Total Assets", "Total Assets", unit="RM"),
            _f("Total Liabilities", "Total Liabilities", unit="RM"),
            _f("Equity (Book Value)", "Equity", unit="RM"),
            _f("Current Assets", "Current Assets", unit="RM"),
            _f("Current Liabilities", "Current Liabilities", unit="RM"),
            _f("Receivables (Avg)", "Accounts Receivable", unit="RM"),
            _f("Inventory (Avg)", "Inventory", unit="RM"),
            _f("Payables (Avg)", "Accounts Payable", unit="RM"),
            _f("Intangible Assets", "Intangible Assets", unit="RM"),
            _f("Total Borrowings", "Total Borrowings", unit="RM"),
            _f("Cash & Equivalents", "Cash & Cash Equivalents", unit="RM"),
            _f("Shares Outstanding", "Shares", type="int"),
            _f("Annual Price per Share (RM)", "Price", unit="RM"),
        ],
        "Cash Flow": [
            _f("CFO (Cash Flow from Ops)", "CFO", unit="RM"),
            _f("Capex", "Capex", unit="RM"),
            _f("Depreciation & Amortization (Total)", "DepAmort", unit="RM",
                help="Optional. If blank, engine sums Depreciation - PPE + Depreciation - ROU Assets + Depreciation of Investment Properties (ignores missing parts)."),
            _f("Depreciation - PPE", "DepPPE", unit="RM"),
            _f("Depreciation - ROU Assets", "DepROU", unit="RM"),
            _f("Depreciation of Investment Properties", "DepInvProp", unit="RM"),  # ← added
        ],
        "Other": [
            _f("Bed Count", "Bed Count", type="int"),
            _f("Patient Days", "Patient Days", type="int"),
            _f("Admissions", "Admissions", type="int"),
            _f("Bed Occupancy (%)", "Bed Occupancy", unit="%"),
            _f("Average Length of Stay (days)", "ALOS", unit="days",
                help="Optional. If blank, engine computes: Patient Days ÷ Admissions."),
            _f("R&D Expense", "R&D Expense", unit="RM"),
        ],
        "Industry KPIs": [],
    },

    "Telco": {
        "Income Statement": [
            _f("Revenue (TTM/FY)", "Revenue", unit="RM"),
            _f("Cost of Sales", "CostOfSales", unit="RM"),
            _f("Gross Profit", "Gross Profit", unit="RM",
                help="Optional. If left blank, engine computes: Revenue − Cost of Sales"),
            _f("Operating Profit", "Operating Profit", unit="RM"),
            _f("EBITDA", "EBITDA", unit="RM",
                help="Optional. If blank, engine computes: Operating Profit + Depreciation & Amortization (Total). If DepAmort is blank, it sums DepPPE + DepROU + DepInvProp."),
            _f("Net Profit", "Net Profit", unit="RM"),
            _f("Interest Expense", "Interest Expense", unit="RM"),
            _f("Selling & Distribution Expenses", "Selling Expenses", unit="RM"),
            _f("Administrative Expenses", "Administrative Expenses", unit="RM"),
            _f("Three Fees (Selling+Admin+Finance, RM)", "Three Fees", unit="RM",
               help="Optional. If left blank, engine sums: Selling & Distribution Expenses + Administrative Expenses + Interest Expense"),
            _f("Income Tax Expense", "IncomeTax", unit="RM"),
            _f("Dividend per Share (TTM, RM)", "DPS", unit="RM"),
        ],
        "Balance Sheet": [
            _f("Total Assets", "Total Assets", unit="RM"),
            _f("Total Liabilities", "Total Liabilities", unit="RM"),
            _f("Equity (Book Value)", "Equity", unit="RM"),
            _f("Current Assets", "Current Assets", unit="RM"),
            _f("Current Liabilities", "Current Liabilities", unit="RM"),
            _f("Receivables (Avg)", "Accounts Receivable", unit="RM"),
            _f("Inventory (Avg)", "Inventory", unit="RM"),
            _f("Payables (Avg)", "Accounts Payable", unit="RM"),
            _f("Intangible Assets", "Intangible Assets", unit="RM"),
            _f("Lease Liabilities (Current)", "LeaseLiabCurrent", unit="RM"),
            _f("Lease Liabilities (Non-Current)", "LeaseLiabNonCurrent", unit="RM"),
            _f("Total Borrowings", "Total Borrowings", unit="RM"),
            _f("Cash & Equivalents", "Cash & Cash Equivalents", unit="RM"),
            _f("Shares Outstanding", "Shares", type="int"),
            _f("Annual Price per Share (RM)", "Price", unit="RM"),
        ],
        "Cash Flow": [
            _f("CFO (Cash Flow from Ops)", "CFO", unit="RM"),
            _f("Capex", "Capex", unit="RM"),
            _f("Depreciation & Amortization (Total)", "DepAmort", unit="RM",
                help="Optional. If blank, engine sums Depreciation - PPE + Depreciation - ROU Assets + Depreciation of Investment Properties (ignores missing parts)."),
            _f("Depreciation - PPE", "DepPPE", unit="RM"),
            _f("Depreciation - ROU Assets", "DepROU", unit="RM"),
            _f("Depreciation of Investment Properties", "DepInvProp", unit="RM"),  # ← added
        ],
        "Other": [
            _f("ARPU (RM)", "ARPU", unit="RM"),
            _f("Subscribers", "Subscribers", type="int"),
            _f("Churn Rate (%)", "Churn Rate", unit="%"),
            _f("Capex to Revenue (%)", "Capex to Revenue", unit="%"),
            _f("Average Data Usage (GB/user)", "Avg Data Usage", unit="GB/user"),
            _f("Spectrum Fees (RM)", "Spectrum Fees", unit="RM"),
        ],
        "Industry KPIs": [],
    },

    "Construction": {
        "Income Statement": [
            _f("Revenue (TTM/FY)", "Revenue", unit="RM"),
            _f("Cost of Sales", "CostOfSales", unit="RM"),
            _f("Gross Profit", "Gross Profit", unit="RM",
                help="Optional. If left blank, engine computes: Revenue − Cost of Sales"),
            _f("Operating Profit", "Operating Profit", unit="RM"),
            _f("EBITDA", "EBITDA", unit="RM",
                help="Optional. If blank, engine computes: Operating Profit + Depreciation & Amortization (Total). If DepAmort is blank, it sums DepPPE + DepROU + DepInvProp."),
            _f("Net Profit", "Net Profit", unit="RM"),
            _f("Interest Expense", "Interest Expense", unit="RM"),
            _f("Selling & Distribution Expenses", "Selling Expenses", unit="RM"),
            _f("Administrative Expenses", "Administrative Expenses", unit="RM"),
            _f("Three Fees (Selling+Admin+Finance, RM)", "Three Fees", unit="RM",
               help="Optional. If left blank, engine sums: Selling & Distribution Expenses + Administrative Expenses + Interest Expense"),
            _f("Income Tax Expense", "IncomeTax", unit="RM"),
            _f("Dividend per Share (TTM, RM)", "DPS", unit="RM"),
        ],
        "Balance Sheet": [
            _f("Total Assets", "Total Assets", unit="RM"),
            _f("Total Liabilities", "Total Liabilities", unit="RM"),
            _f("Equity (Book Value)", "Equity", unit="RM"),
            _f("Current Assets", "Current Assets", unit="RM"),
            _f("Current Liabilities", "Current Liabilities", unit="RM"),
            _f("Receivables (Avg)", "Accounts Receivable", unit="RM"),
            _f("Inventory (Avg)", "Inventory", unit="RM"),
            _f("Payables (Avg)", "Accounts Payable", unit="RM"),
            _f("Intangible Assets", "Intangible Assets", unit="RM"),
            _f("Total Borrowings", "Total Borrowings", unit="RM"),
            _f("Cash & Equivalents", "Cash & Cash Equivalents", unit="RM"),
            _f("Shares Outstanding", "Shares", type="int"),
            _f("Annual Price per Share (RM)", "Price", unit="RM"),
        ],
        "Cash Flow": [
            _f("CFO (Cash Flow from Ops)", "CFO", unit="RM"),
            _f("Capex", "Capex", unit="RM"),
            _f("Depreciation & Amortization (Total)", "DepAmort", unit="RM",
                help="Optional. If blank, engine sums Depreciation - PPE + Depreciation - ROU Assets + Depreciation of Investment Properties (ignores missing parts)."),
            _f("Depreciation - PPE", "DepPPE", unit="RM"),
            _f("Depreciation - ROU Assets", "DepROU", unit="RM"),
            _f("Depreciation of Investment Properties", "DepInvProp", unit="RM"),  # ← added
        ],
        "Other": [
            _f("Orderbook", "Orderbook", unit="RM"),
            _f("New Orders (TTM)", "New Orders", unit="RM"),
            _f("Tender Book", "Tender Book", unit="RM"),
            _f("Win Rate (%)", "Win Rate", unit="%",
                help="Optional. If blank, engine computes: New Orders ÷ Tender Book × 100."),
        ],
        "Industry KPIs": [],
    },

    "Plantation": {
        "Income Statement": [
            _f("Revenue (TTM/FY)", "Revenue", unit="RM"),
            _f("Cost of Sales", "CostOfSales", unit="RM"),
            _f("Gross Profit", "Gross Profit", unit="RM",
                help="Optional. If left blank, engine computes: Revenue − Cost of Sales"),
            _f("Operating Profit", "Operating Profit", unit="RM"),
            _f("EBITDA", "EBITDA", unit="RM",
                help="Optional. If blank, engine computes: Operating Profit + Depreciation & Amortization (Total). If DepAmort is blank, it sums DepPPE + DepROU + DepInvProp."),
            _f("Net Profit", "Net Profit", unit="RM"),
            _f("Interest Expense", "Interest Expense", unit="RM"),
            _f("Selling & Distribution Expenses", "Selling Expenses", unit="RM"),
            _f("Administrative Expenses", "Administrative Expenses", unit="RM"),
            _f("Three Fees (Selling+Admin+Finance, RM)", "Three Fees", unit="RM",
               help="Optional. If left blank, engine sums: Selling & Distribution Expenses + Administrative Expenses + Interest Expense"),
            _f("Income Tax Expense", "IncomeTax", unit="RM"),
            _f("Dividend per Share (TTM, RM)", "DPS", unit="RM"),
        ],
        "Balance Sheet": [
            _f("Total Assets", "Total Assets", unit="RM"),
            _f("Total Liabilities", "Total Liabilities", unit="RM"),
            _f("Equity (Book Value)", "Equity", unit="RM"),
            _f("Current Assets", "Current Assets", unit="RM"),
            _f("Current Liabilities", "Current Liabilities", unit="RM"),
            _f("Receivables (Avg)", "Accounts Receivable", unit="RM"),
            _f("Inventory (Avg)", "Inventory", unit="RM"),
            _f("Payables (Avg)", "Accounts Payable", unit="RM"),
            _f("Intangible Assets", "Intangible Assets", unit="RM"),
            _f("Biological Assets", "BiologicalAssets", unit="RM"),
            _f("Total Borrowings", "Total Borrowings", unit="RM"),
            _f("Cash & Equivalents", "Cash & Cash Equivalents", unit="RM"),
            _f("Shares Outstanding", "Shares", type="int"),
            _f("Annual Price per Share (RM)", "Price", unit="RM"),
        ],
        "Cash Flow": [
            _f("CFO (Cash Flow from Ops)", "CFO", unit="RM"),
            _f("Capex", "Capex", unit="RM"),
            _f("Depreciation & Amortization (Total)", "DepAmort", unit="RM",
                help="Optional. If blank, engine sums Depreciation - PPE + Depreciation - ROU Assets + Depreciation of Investment Properties (ignores missing parts)."),
            _f("Depreciation - PPE", "DepPPE", unit="RM"),
            _f("Depreciation - ROU Assets", "DepROU", unit="RM"),
            _f("Depreciation of Investment Properties", "DepInvProp", unit="RM"),  # ← added
        ],
        "Other": [
            _f("CPO Output (t)", "CPO Output", unit="t"),
            _f("FFB Input (t)", "FFB Input", unit="t"),
            _f("FFB Harvested (t)", "FFB Harvested", unit="t"),
            _f("Planted Hectares (ha)", "Planted Hectares", unit="ha"),
            _f("Unit Cash Cost (RM/t)", "Unit Cash Cost", unit="RM/t"),
            _f("Tonnes Sold (t)", "Tonnes Sold", unit="t"),
            _f("Average CPO Selling Price (RM/t)", "CPO ASP", unit="RM/t"),
            _f("Palm Kernel Output (t)", "PKO Output", unit="t"),
            _f("Mature Area (ha)", "Mature Area", unit="ha"),
            _f("Average Tree Age (years)", "Avg Tree Age", unit="years"),
        ],
        "Industry KPIs": [],
    },

    "Property": {
        "Income Statement": [
            _f("Revenue (TTM/FY)", "Revenue", unit="RM"),
            _f("Cost of Sales", "CostOfSales", unit="RM"),
            _f("Gross Profit", "Gross Profit", unit="RM",
                help="Optional. If left blank, engine computes: Revenue − Cost of Sales"),
            _f("Operating Profit", "Operating Profit", unit="RM"),
            _f("EBITDA", "EBITDA", unit="RM",
                help="Optional. If blank, engine computes: Operating Profit + Depreciation & Amortization (Total). If DepAmort is blank, it sums DepPPE + DepROU + DepInvProp."),
            _f("Net Profit", "Net Profit", unit="RM"),
            _f("Interest Expense", "Interest Expense", unit="RM"),
            _f("Selling & Distribution Expenses", "Selling Expenses", unit="RM"),
            _f("Administrative Expenses", "Administrative Expenses", unit="RM"),
            _f("Three Fees (Selling+Admin+Finance, RM)", "Three Fees", unit="RM",
               help="Optional. If left blank, engine sums: Selling & Distribution Expenses + Administrative Expenses + Interest Expense"),
            _f("Income Tax Expense", "IncomeTax", unit="RM"),
            _f("Dividend per Share (TTM, RM)", "DPS", unit="RM"),
        ],
        "Balance Sheet": [
            _f("Total Assets", "Total Assets", unit="RM"),
            _f("Total Liabilities", "Total Liabilities", unit="RM"),
            _f("Equity (Book Value)", "Equity", unit="RM"),
            _f("Current Assets", "Current Assets", unit="RM"),
            _f("Current Liabilities", "Current Liabilities", unit="RM"),
            _f("Receivables (Avg)", "Accounts Receivable", unit="RM"),
            _f("Inventory (Avg)", "Inventory", unit="RM"),
            _f("Payables (Avg)", "Accounts Payable", unit="RM"),
            _f("Intangible Assets", "Intangible Assets", unit="RM"),
            _f("Total Borrowings", "Total Borrowings", unit="RM"),
            _f("Cash & Equivalents", "Cash & Cash Equivalents", unit="RM"),
            _f("Shares Outstanding", "Shares", type="int"),
            _f("Annual Price per Share (RM)", "Price", unit="RM"),
        ],
        "Cash Flow": [
            _f("CFO (Cash Flow from Ops)", "CFO", unit="RM"),
            _f("Capex", "Capex", unit="RM"),
            _f("Depreciation & Amortization (Total)", "DepAmort", unit="RM",
                help="Optional. If blank, engine sums Depreciation - PPE + Depreciation - ROU Assets + Depreciation of Investment Properties (ignores missing parts)."),
            _f("Depreciation - PPE", "DepPPE", unit="RM"),
            _f("Depreciation - ROU Assets", "DepROU", unit="RM"),
            _f("Depreciation - Investment Property", "DepInvProp", unit="RM"),
        ],
        "Other": [
            _f("Unbilled Sales", "Unbilled Sales", unit="RM"),
            _f("Units Sold", "Units Sold", type="int"),
            _f("Units Launched", "Units Launched", type="int"),
            _f("RNAV per Share", "RNAV per Share", unit="RM"),
            _f("Net Debt (optional)", "Net Debt", unit="RM",
                help="Optional. If blank, engine computes: Total Borrowings − Cash & Cash Equivalents."),
            _f("Gross Development Value (GDV)", "GDV", unit="RM"),
            _f("Landbank (acres)", "Landbank", unit="acres"),
            _f("Sales Rate (%)", "Sales Rate", unit="%"),
        ],
        "Industry KPIs": [],
    },

    "Transportation/Logistics": {
        "Income Statement": [
            _f("Revenue (TTM/FY)", "Revenue", unit="RM"),
            _f("Cost of Sales", "CostOfSales", unit="RM"),
            _f("Gross Profit", "Gross Profit", unit="RM",
                help="Optional. If left blank, engine computes: Revenue − Cost of Sales"),
            _f("Operating Profit", "Operating Profit", unit="RM"),
            _f("EBITDA", "EBITDA", unit="RM",
                help="Optional. If blank, engine computes: Operating Profit + Depreciation & Amortization (Total). If DepAmort is blank, it sums DepPPE + DepROU + DepInvProp."),
            _f("Net Profit", "Net Profit", unit="RM"),
            _f("Interest Expense", "Interest Expense", unit="RM"),
            _f("Selling & Distribution Expenses", "Selling Expenses", unit="RM"),
            _f("Administrative Expenses", "Administrative Expenses", unit="RM"),
            _f("Three Fees (Selling+Admin+Finance, RM)", "Three Fees", unit="RM",
               help="Optional. If left blank, engine sums: Selling & Distribution Expenses + Administrative Expenses + Interest Expense"),
            _f("Income Tax Expense", "IncomeTax", unit="RM"),
            _f("Dividend per Share (TTM, RM)", "DPS", unit="RM"),
        ],
        "Balance Sheet": [
            _f("Total Assets", "Total Assets", unit="RM"),
            _f("Total Liabilities", "Total Liabilities", unit="RM"),
            _f("Equity (Book Value)", "Equity", unit="RM"),
            _f("Current Assets", "Current Assets", unit="RM"),
            _f("Current Liabilities", "Current Liabilities", unit="RM"),
            _f("Receivables (Avg)", "Accounts Receivable", unit="RM"),
            _f("Inventory (Avg)", "Inventory", unit="RM"),
            _f("Payables (Avg)", "Accounts Payable", unit="RM"),
            _f("Intangible Assets", "Intangible Assets", unit="RM"),
            _f("Lease Liabilities (Current)", "LeaseLiabCurrent", unit="RM"),
            _f("Lease Liabilities (Non-Current)", "LeaseLiabNonCurrent", unit="RM"),
            _f("Total Borrowings", "Total Borrowings", unit="RM"),
            _f("Cash & Equivalents", "Cash & Cash Equivalents", unit="RM"),
            _f("Shares Outstanding", "Shares", type="int"),
            _f("Annual Price per Share (RM)", "Price", unit="RM"),
        ],
        "Cash Flow": [
            _f("CFO (Cash Flow from Ops)", "CFO", unit="RM"),
            _f("Capex", "Capex", unit="RM"),
            _f("Depreciation & Amortization (Total)", "DepAmort", unit="RM",
                help="Optional. If blank, engine sums Depreciation - PPE + Depreciation - ROU Assets + Depreciation of Investment Properties (ignores missing parts)."),
            _f("Depreciation - PPE", "DepPPE", unit="RM"),
            _f("Depreciation - ROU Assets", "DepROU", unit="RM"),
            _f("Depreciation of Investment Properties", "DepInvProp", unit="RM"),  # ← added
        ],
        "Other": [
            _f("TEU Throughput (ports)", "TEU Throughput", type="int"),
            _f("Load Factor (airlines, %)", "Load Factor", unit="%",
                help="Optional. If blank, engine computes: RPK ÷ ASK × 100."),
            _f("Yield per km/parcel", "Yield per km", unit="RM",
                help="Optional. If blank, engine computes: Revenue ÷ RPK."),
            _f("ASK (Available Seat Km, m)", "ASK", unit="m km"),
            _f("RPK (Revenue Passenger Km, m)", "RPK", unit="m km"),
            _f("On-Time Performance (%)", "On-Time Performance", unit="%"),
            _f("Fleet Size", "Fleet Size", type="int"),
        ],
        "Industry KPIs": [],
    },

    "Leisure/Travel": {
        "Income Statement": [
            _f("Revenue (TTM/FY)", "Revenue", unit="RM"),
            _f("Cost of Sales", "CostOfSales", unit="RM"),
            _f("Gross Profit", "Gross Profit", unit="RM",
                help="Optional. If left blank, engine computes: Revenue − Cost of Sales"),
            _f("Operating Profit", "Operating Profit", unit="RM"),
            _f("EBITDA", "EBITDA", unit="RM",
                help="Optional. If blank, engine computes: Operating Profit + Depreciation & Amortization (Total). If DepAmort is blank, it sums DepPPE + DepROU + DepInvProp."),
            _f("Net Profit", "Net Profit", unit="RM"),
            _f("Interest Expense", "Interest Expense", unit="RM"),
            _f("Selling & Distribution Expenses", "Selling Expenses", unit="RM"),
            _f("Administrative Expenses", "Administrative Expenses", unit="RM"),
            _f("Three Fees (Selling+Admin+Finance, RM)", "Three Fees", unit="RM",
               help="Optional. If left blank, engine sums: Selling & Distribution Expenses + Administrative Expenses + Interest Expense"),
            _f("Income Tax Expense", "IncomeTax", unit="RM"),
            _f("Dividend per Share (TTM, RM)", "DPS", unit="RM"),
        ],
        "Balance Sheet": [
            _f("Total Assets", "Total Assets", unit="RM"),
            _f("Total Liabilities", "Total Liabilities", unit="RM"),
            _f("Equity (Book Value)", "Equity", unit="RM"),
            _f("Current Assets", "Current Assets", unit="RM"),
            _f("Current Liabilities", "Current Liabilities", unit="RM"),
            _f("Receivables (Avg)", "Accounts Receivable", unit="RM"),
            _f("Inventory (Avg)", "Inventory", unit="RM"),
            _f("Payables (Avg)", "Accounts Payable", unit="RM"),
            _f("Intangible Assets", "Intangible Assets", unit="RM"),
            _f("Total Borrowings", "Total Borrowings", unit="RM"),
            _f("Cash & Equivalents", "Cash & Cash Equivalents", unit="RM"),
            _f("Shares Outstanding", "Shares", type="int"),
            _f("Annual Price per Share (RM)", "Price", unit="RM"),
        ],
        "Cash Flow": [
            _f("CFO (Cash Flow from Ops)", "CFO", unit="RM"),
            _f("Capex", "Capex", unit="RM"),
            _f("Depreciation & Amortization (Total)", "DepAmort", unit="RM",
                help="Optional. If blank, engine sums Depreciation - PPE + Depreciation - ROU Assets + Depreciation of Investment Properties (ignores missing parts)."),
            _f("Depreciation - PPE", "DepPPE", unit="RM"),
            _f("Depreciation - ROU Assets", "DepROU", unit="RM"),
            _f("Depreciation of Investment Properties", "DepInvProp", unit="RM"),  # ← added
        ],
        "Other": [
            _f("Hotel Occupancy (%)", "Hotel Occupancy", unit="%"),
            _f("RevPAR (RM)", "RevPAR", unit="RM"),
            _f("Gaming Win Rate (%)", "Win Rate", unit="%"),
            _f("Visits", "Visits", type="int"),
            _f("Average Spend (RM)", "Average Spend", unit="RM"),
            _f("Room Count", "Room Count", type="int"),
        ],
        "Industry KPIs": [],
    },

    "General": {
        "Income Statement": [
            _f("Revenue (TTM/FY)", "Revenue", unit="RM"),
            _f("Cost of Sales", "CostOfSales", unit="RM"),
            _f("Gross Profit", "Gross Profit", unit="RM",
                help="Optional. If left blank, engine computes: Revenue − Cost of Sales"),
            _f("Operating Profit", "Operating Profit", unit="RM"),
            _f("EBITDA", "EBITDA", unit="RM",
                help="Optional. If blank, engine computes: Operating Profit + Depreciation & Amortization (Total). If DepAmort is blank, it sums DepPPE + DepROU + DepInvProp."),
            _f("Net Profit", "Net Profit", unit="RM"),
            _f("Interest Expense", "Interest Expense", unit="RM"),
            _f("Selling & Distribution Expenses", "Selling Expenses", unit="RM"),
            _f("Administrative Expenses", "Administrative Expenses", unit="RM"),
            _f("Three Fees (Selling+Admin+Finance, RM)", "Three Fees", unit="RM",
               help="Optional. If left blank, engine sums: Selling & Distribution Expenses + Administrative Expenses + Interest Expense"),
            _f("Income Tax Expense", "IncomeTax", unit="RM"),
            _f("Dividend per Share (TTM, RM)", "DPS", unit="RM"),
        ],
        "Balance Sheet": [
            _f("Total Assets", "Total Assets", unit="RM"),
            _f("Total Liabilities", "Total Liabilities", unit="RM"),
            _f("Equity (Book Value)", "Equity", unit="RM"),
            _f("Current Assets", "Current Assets", unit="RM"),
            _f("Current Liabilities", "Current Liabilities", unit="RM"),
            _f("Receivables (Avg)", "Accounts Receivable", unit="RM"),
            _f("Inventory (Avg)", "Inventory", unit="RM"),
            _f("Payables (Avg)", "Accounts Payable", unit="RM"),
            _f("Intangible Assets", "Intangible Assets", unit="RM"),
            _f("Total Borrowings", "Total Borrowings", unit="RM"),
            _f("Cash & Equivalents", "Cash & Cash Equivalents", unit="RM"),
            _f("Shares Outstanding", "Shares", type="int"),
            _f("Annual Price per Share (RM)", "Price", unit="RM"),
        ],
        "Cash Flow": [
            _f("CFO (Cash Flow from Ops)", "CFO", unit="RM"),
            _f("Capex", "Capex", unit="RM"),
            _f("Depreciation & Amortization (Total)", "DepAmort", unit="RM",
                help="Optional. If blank, engine sums Depreciation - PPE + Depreciation - ROU Assets + Depreciation of Investment Properties (ignores missing parts)."),
            _f("Depreciation - PPE", "DepPPE", unit="RM"),
            _f("Depreciation - ROU Assets", "DepROU", unit="RM"),
            _f("Depreciation of Investment Properties", "DepInvProp", unit="RM"),  # ← added
        ],
        "Other": [],
        "Industry KPIs": [],
    },
}

# --------- Quarterly categories (keys prefixed Q_) ----------
INDUSTRY_FORM_CATEGORIES_Q = {b: _prefix_quarter(cats) for b, cats in INDUSTRY_FORM_CATEGORIES.items()}

# --------- Back-compat flattened lists (what Add/Edit currently reads) ----------
# Annual (flattened)
INDUSTRY_FORM_FIELDS = {b: _flatten_categories(cats) for b, cats in INDUSTRY_FORM_CATEGORIES.items()}

# Quarterly (flattened; useful if/when you wire config-driven quarterly)
INDUSTRY_FORM_FIELDS_Q = {b: _flatten_categories(cats) for b, cats in INDUSTRY_FORM_CATEGORIES_Q.items()}

# =============================================
# Summary table – per-industry ratios (grouped)
# =============================================

# How to order categories when flattened into a single list:
SUMMARY_RATIO_CATEGORY_ORDER = [
    "Margins",
    "Returns",
    "Efficiency & Working Capital",
    "Liquidity & Leverage",
    "Cash Flow",
    "Operations / Industry KPIs",
    "Income & Efficiency",            # (Non-financials: holds Three Fees Ratio)
    "Three-Proportion (Banking)",     # 👈 NEW group for banks (NIM/C-I/Leverage)
    "Asset Quality",                  # (Banking)
    "Capital & Liquidity",            # (Banking)
    "Debt & Hedging",                 # (REITs)
    "Portfolio Quality",              # (REITs)
    "Distribution & Valuation",       # (REITs/Property)
    "Valuation",
]


def _flatten_ratio_categories(cat_dict: dict) -> dict:
    """Flatten a category->(label->keys) dict into a single ordered dict."""
    out = {}
    # first in our preferred order
    for sec in SUMMARY_RATIO_CATEGORY_ORDER:
        for label, keys in cat_dict.get(sec, {}).items():
            out[label] = keys
    # then any categories not in the preferred order
    for sec, items in cat_dict.items():
        if sec not in SUMMARY_RATIO_CATEGORY_ORDER:
            for label, keys in items.items():
                out[label] = keys
    return out

INDUSTRY_SUMMARY_RATIOS_CATEGORIES = {
    "Manufacturing": {
        "Margins": {
            "Gross Margin (%)": ["Gross Margin (%)", "Gross Margin", "Gross Profit Margin"],
            "EBITDA Margin (%)": ["EBITDA Margin (%)", "EBITDA Margin", "EBITDA/Revenue"],
            "Operating Profit Margin (%)": ["Operating Profit Margin (%)", "Operating Profit Margin", "EBIT Margin (%)"],
            "Net Margin (%)": ["Net Margin (%)", "Net Profit Margin", "Net Profit/Revenue"],
        },
        "Returns": {
            "ROE (%)": ["ROE (%)", "ROE", "Return on Equity"],
            "ROA (%)": ["ROA (%)", "ROA", "Return on Assets"],
        },
        "Efficiency & Working Capital": {
            "Inventory Days (days)": ["Inventory Days (days)", "Inventory Days", "DIO", "Days Inventory Outstanding"],
            "Receivable Days (days)": ["Receivable Days (days)", "Receivable Days", "DSO", "Days Sales Outstanding"],
            "Payable Days (days)": ["Payable Days (days)", "Payable Days", "DPO", "Days Payables Outstanding"],
            "Cash Conversion Cycle (days)": ["Cash Conversion Cycle (days)", "CCC", "Cash Conversion Cycle"],
            # added from your list:
            "Inventory Turnover (×)": ["Inventory Turnover (×)", "Inventory Turnover"],
        },
        "Income & Efficiency": {
            "Three Fees Ratio (%)": [
                "Three Fees Ratio (%)",
                "Three Fees (%)",
                "SG&A+Fin / Revenue (%)",
                "Selling+Admin+Finance / Revenue (%)",
            ],
        },
        "Liquidity & Leverage": {
            "Current Ratio (x)": ["Current Ratio (x)", "Current Ratio", "Current Assets / Current Liabilities"],
            "Quick Ratio (x)": ["Quick Ratio (x)", "Quick Ratio", "Acid Test Ratio"],
            "Debt/Equity (x)": ["Debt/Equity (x)", "Debt/Equity", "D/E"],
            "Interest Coverage (x)": ["Interest Coverage (x)", "Interest Coverage", "EBIT/Interest", "Interest Coverage (×)"],
            "Net Debt / EBITDA (×)": ["Net Debt / EBITDA (×)", "Net Debt/EBITDA (×)", "Net Debt/EBITDA"],
        },
        "Cash Flow": {
            "Capex/Revenue (%)": ["Capex/Revenue (%)", "Capex to Revenue", "Capex Intensity (%)", "CapEx Intensity (%)"],
            "FCF Margin (%)": ["FCF Margin (%)", "Free Cash Flow Margin"],
        },
        "Valuation": {
            "Dividend Yield (%)": ["Dividend Yield (%)", "Dividend Yield", "DY"],
            "P/E (x)": ["P/E (x)", "P/E", "PE"],
            "P/B (x)": ["P/B (x)", "P/B", "PB"],
            "EV/EBITDA (x)": ["EV/EBITDA (x)", "EV/EBITDA"],
            "EPS (RM)": ["EPS (RM)", "EPS", "Earnings per Share", "Basic EPS (RM)", "Diluted EPS (RM)", "Basic EPS", "Diluted EPS", "EPS (sen)"],

        },
    },

    "Retail": {
        "Margins": {
            "Gross Margin (%)": ["Gross Margin (%)", "Gross Margin", "Gross Profit Margin"],
            "EBITDA Margin (%)": ["EBITDA Margin (%)", "EBITDA Margin"],
            "Operating Profit Margin (%)": ["Operating Profit Margin (%)", "Operating Profit Margin", "EBIT Margin (%)"],
            "Net Margin (%)": ["Net Margin (%)", "Net Profit Margin"],
        },
        "Operations / Industry KPIs": {
            "SSSG (%)": ["Same-Store Sales Growth (%)", "SSSG", "Same Store Sales Growth"],
        },
        "Efficiency & Working Capital": {
            "Inventory Days (days)": ["Inventory Days (days)", "DIO", "Days Inventory Outstanding"],
            "Receivable Days (days)": ["Receivable Days (days)", "DSO"],
            "Payable Days (days)": ["Payable Days (days)", "DPO"],
            "Cash Conversion Cycle (days)": ["Cash Conversion Cycle (days)", "CCC"],
            "Inventory Turnover (×)": ["Inventory Turnover (×)", "Inventory Turnover"],
        },
        "Income & Efficiency": {
            "Three Fees Ratio (%)": [
                "Three Fees Ratio (%)",
                "Three Fees (%)",
                "SG&A+Fin / Revenue (%)",
                "Selling+Admin+Finance / Revenue (%)",
            ],
        },
        "Liquidity & Leverage": {
            "Debt/Equity (x)": ["Debt/Equity (x)", "Debt/Equity", "D/E"],
            "Interest Coverage (x)": ["Interest Coverage (x)", "EBIT/Interest", "Interest Coverage (×)"],
        },
        "Cash Flow": {
            "Capex/Revenue (%)": ["Capex/Revenue (%)", "Capex to Revenue", "Capex Intensity (%)", "CapEx Intensity (%)"],
            "Operating CF Margin (%)": ["Operating CF Margin (%)", "Operating CF Margin", "OCF Margin (%)"],
        },
        "Valuation": {
            "Dividend Yield (%)": ["Dividend Yield (%)", "Dividend Yield", "DY"],
            "P/E (x)": ["P/E (x)", "P/E", "PE"],
            "P/B (x)": ["P/B (x)", "P/B", "PB"],
            "EV/EBITDA (x)": ["EV/EBITDA (x)", "EV/EBITDA"],
            "EPS (RM)": ["EPS (RM)", "EPS", "Earnings per Share", "Basic EPS (RM)", "Diluted EPS (RM)", "Basic EPS", "Diluted EPS", "EPS (sen)"],
        },
    },

    "Financials": {
        "Margins": {
            "Gross Margin (%)": ["Gross Margin (%)", "Gross Margin"],
            "EBITDA Margin (%)": ["EBITDA Margin (%)", "EBITDA Margin"],
            "Operating Profit Margin (%)": ["Operating Profit Margin (%)", "Operating Profit Margin", "EBIT Margin (%)"],
            "Net Margin (%)": ["Net Margin (%)", "Net Profit Margin"],
        },
        "Income & Efficiency": {
            "Operating Expense Ratio (%)": ["Operating Expense Ratio (%)", "Operating Expense Ratio"],
        },
        "Returns": {
            "ROE (%)": ["ROE (%)", "ROE", "Return on Equity"],
            "ROA (%)": ["ROA (%)", "ROA", "Return on Assets"],
        },
        "Liquidity & Leverage": {
            "Debt/Equity (x)": ["Debt/Equity (x)", "D/E", "Debt/Equity"],
            "Interest Coverage (x)": ["Interest Coverage (x)", "EBIT/Interest", "Interest Coverage (×)"],
            "Financial Leverage (Assets/Equity)": ["Financial Leverage (Assets/Equity)", "Financial Leverage"],
        },
        "Cash Flow": {
            "Capex/Revenue (%)": ["Capex/Revenue (%)", "Capex to Revenue"],
        },
        "Valuation": {
            "Dividend Yield (%)": ["Dividend Yield (%)", "Dividend Yield", "DY"],
            "P/E (x)": ["P/E (x)", "P/E", "PE"],
            "P/B (x)": ["P/B (x)", "P/B", "PB"],
            "EPS (RM)": ["EPS (RM)", "EPS", "Earnings per Share", "Basic EPS (RM)", "Diluted EPS (RM)", "Basic EPS", "Diluted EPS", "EPS (sen)"],
        },
    },

    "Banking": {
        "Income & Efficiency": {
            "NIM (%)": ["NIM (%)", "NIM", "Net Interest/Financing Margin (%)", "Net Interest Margin"],
            "Cost-to-Income Ratio (%)": ["Cost-to-Income Ratio (%)", "Cost-to-Income", "CIR", "Cost to Income", "C/I (%)"],
            "Financial Leverage (Assets/Equity)": ["Financial Leverage (Assets/Equity)", "Financial Leverage"],
        },

        "Cash Flow": {
            "Operating CF Margin (%)": [
                "Operating CF Margin (%)",
                "Operating CF Margin",
                "OCF Margin (%)",
                "CFO Margin (%)",
                "CFO/Revenue (%)",
                "CFO / Operating Income (%)"
            ],
        },        

        "Asset Quality": {
            "NPL Ratio (%)": ["NPL Ratio (%)", "NPL (%)", "NPL Ratio"],
            "Loan-Loss Coverage (×)": ["Loan-Loss Coverage (×)", "Coverage Ratio (×)", "Loan Loss Coverage (×)", "Coverage Ratio (%)"],
        },
        "Capital & Liquidity": {
            "CASA Ratio (%)": ["CASA Ratio (%)", "CASA Ratio", "CASA"],
            # CORE CASA = (Demand+Savings) / (Demand+Savings+Fixed)
            "CASA (Core, %)": ["CASA (Core, %)", "CASA Core (%)", "Core CASA (%)", "CASA Core", "Core CASA"],
            "Loan-to-Deposit Ratio (%)": [
                "Loan-to-Deposit Ratio (%)",
                "Loan-to-Deposit Ratio (×)",
                "Loan-to-Deposit Ratio",
                "LDR (×)",
                "LDR (%)",
            ],
            "CET1 Ratio (%)": ["CET1 Ratio (%)", "CET1 Ratio", "CET1"],
            "Tier 1 Capital Ratio (%)": ["Tier 1 Capital Ratio (%)", "Tier 1 Ratio"],
            "Total Capital Ratio (%)": ["Total Capital Ratio (%)", "Total Capital Ratio"],
            "LCR (%)": ["Liquidity Coverage Ratio (LCR, %)", "LCR"],
            "NSFR (%)": ["Net Stable Funding Ratio (NSFR, %)", "NSFR"],
        },
        "Returns": {
            "ROE (%)": ["ROE (%)", "ROE", "Return on Equity"],
            "ROA (%)": ["ROA (%)", "ROA", "Return on Assets"],
        },
        "Valuation": {
            "Dividend Yield (%)": ["Dividend Yield (%)", "Dividend Yield", "DY"],
            "P/B (x)": ["P/B (x)", "P/B", "PB"],
            "P/E (x)": ["P/E (x)", "P/E", "PE"],
            "EPS (RM)": ["EPS (RM)", "EPS", "Earnings per Share", "Basic EPS (RM)", "Diluted EPS (RM)", "Basic EPS", "Diluted EPS", "EPS (sen)"],
        },
    },

    "REITs": {
        "Distribution & Valuation": {
            "Distribution Yield (%)": ["Distribution Yield (%)", "Dividend Yield (%)", "DPU Yield", "DY"],
            "P/NAV (x)": ["P/NAV (x)", "P/NAV", "P/B (x)", "PB", "Price/NAV"],
            "EPS (RM)": ["EPS (RM)", "EPS", "Earnings per Share", "Basic EPS (RM)", "Diluted EPS (RM)", "Basic EPS", "Diluted EPS", "EPS (sen)"],
        },
        "Portfolio Quality": {
            "Occupancy (%)": ["Occupancy (%)", "Occupancy"],
            "WALE (years)": ["WALE (years)", "WALE"],
            "Rental Reversion (%)": ["Rental Reversion (%)", "Rental Reversion"],
        },
        "Debt & Hedging": {
            "Gearing (x)": ["Gearing (x)", "Debt/Assets (x)", "Debt/Equity (x)", "D/E"],
            "Gearing (Debt/Assets, %)": ["Gearing (%)", "Debt/Assets (%)", "Debt/Assets (x)", "Gearing (%)"],
            "Interest Coverage (x)": ["Interest Coverage (x)", "EBIT/Interest", "Interest Coverage (×)", "ICR"],
            "Hedged Debt (%)": ["Hedged Debt (%)", "Hedged Debt"],
            "Average Cost of Debt (%)": ["Average Cost of Debt (%)", "Avg Cost of Debt (%)", "Avg Cost of Debt"],
        },
        "Margins": {
            "NPI Margin (%)": ["NPI Margin (%)", "NPI Margin"],
            "EBITDA Margin (%)": ["EBITDA Margin (%)", "EBITDA Margin"],
            "Net Margin (%)": ["Net Margin (%)", "Net Profit Margin"],
        },
    },

    "Utilities": {
        "Margins": {
            "EBITDA Margin (%)": ["EBITDA Margin (%)", "EBITDA Margin"],
            "Operating Profit Margin (%)": ["Operating Profit Margin (%)", "Operating Profit Margin", "EBIT Margin (%)"],
            "Net Margin (%)": ["Net Margin (%)", "Net Profit Margin"],
        },
        "Returns": {
            "ROE (%)": ["ROE (%)", "ROE", "Return on Equity"],
            "ROA (%)": ["ROA (%)", "ROA", "Return on Assets"],
        },
        "Operations / Industry KPIs": {
            "Availability Factor (%)": ["Availability Factor (%)", "Availability Factor"],
            "Allowed Return (%)": ["Allowed Return (%)", "Allowed Return"],
        },
        "Liquidity & Leverage": {
            "Debt/Equity (x)": ["Debt/Equity (x)", "D/E", "Debt/Equity"],
            "Interest Coverage (x)": ["Interest Coverage (x)", "EBIT/Interest", "Interest Coverage (×)"],
            "Net Debt / EBITDA (×)": ["Net Debt / EBITDA (×)", "Net Debt/EBITDA (×)", "Net Debt/EBITDA"],
        },
        "Cash Flow": {
            "Capex/Revenue (%)": ["Capex/Revenue (%)", "Capex to Revenue"],
            "CapEx Intensity (%)": ["CapEx Intensity (%)", "Capex Intensity (%)", "Capex/Revenue (%)", "Capex to Revenue"],
            "Dividend Cash Coverage (CFO/Div)": ["Dividend Cash Coverage (CFO/Div)", "Dividend Cash Coverage"],
        },
        "Valuation": {
            "Dividend Yield (%)": ["Dividend Yield (%)", "Dividend Yield", "DY"],
            "P/E (x)": ["P/E (x)", "P/E", "PE"],
            "P/B (x)": ["P/B (x)", "P/B", "PB"],
            "EV/EBITDA (x)": ["EV/EBITDA (x)", "EV/EBITDA"],
            "EPS (RM)": ["EPS (RM)", "EPS", "Earnings per Share", "Basic EPS (RM)", "Diluted EPS (RM)", "Basic EPS", "Diluted EPS", "EPS (sen)"],
        },
    },

    "Energy/Materials": {
        "Margins": {
            "Gross Margin (%)": ["Gross Margin (%)", "Gross Margin"],
            "EBITDA Margin (%)": ["EBITDA Margin (%)", "EBITDA Margin"],
            "Operating Profit Margin (%)": ["Operating Profit Margin (%)", "Operating Profit Margin", "EBIT Margin (%)"],
            "Net Margin (%)": ["Net Margin (%)", "Net Profit Margin"],
        },
        "Returns": {
            "ROE (%)": ["ROE (%)", "ROE"],
            "ROA (%)": ["ROA (%)", "ROA"],
        },
        "Operations / Industry KPIs": {
            "Lifting Cost (RM/boe)": ["Lifting Cost (RM/boe)", "Lifting Cost"],
            "Average Selling Price (RM/tonne)": ["Average Selling Price (RM/tonne)", "Average Selling Price", "ASP", "ASP (RM/t)"],
            "Strip Ratio": ["Strip Ratio"],
            "Head Grade (g/t)": ["Head Grade (g/t)", "Head Grade"],
        },
        "Income & Efficiency": {
            "Three Fees Ratio (%)": [
                "Three Fees Ratio (%)",
                "Three Fees (%)",
                "SG&A+Fin / Revenue (%)",
                "Selling+Admin+Finance / Revenue (%)",
            ],
        },
        "Liquidity & Leverage": {
            "Debt/Equity (x)": ["Debt/Equity (x)", "D/E"],
            "Interest Coverage (x)": ["Interest Coverage (x)", "EBIT/Interest", "Interest Coverage (×)"],
            "Net Debt / EBITDA (×)": ["Net Debt / EBITDA (×)", "Net Debt/EBITDA (×)", "Net Debt/EBITDA"],
        },
        "Cash Flow": {
            "CapEx Intensity (%)": ["CapEx Intensity (%)", "Capex Intensity (%)", "Capex/Revenue (%)", "Capex to Revenue"],
            "Cash Conversion (CFO/EBITDA)": ["Cash Conversion (CFO/EBITDA)", "Cash Conversion"],
        },
        "Valuation": {
            "Dividend Yield (%)": ["Dividend Yield (%)", "Dividend Yield", "DY"],
            "P/E (x)": ["P/E (x)", "P/E", "PE"],
            "P/B (x)": ["P/B (x)", "P/B", "PB"],
            "EV/EBITDA (x)": ["EV/EBITDA (x)", "EV/EBITDA"],
            "EPS (RM)": ["EPS (RM)", "EPS", "Earnings per Share", "Basic EPS (RM)", "Diluted EPS (RM)", "Basic EPS", "Diluted EPS", "EPS (sen)"],
        },
    },

    "Tech": {
        "Margins": {
            "Gross Margin (%)": ["Gross Margin (%)", "Gross Margin"],
            "EBITDA Margin (%)": ["EBITDA Margin (%)", "EBITDA Margin"],
            "Operating Profit Margin (%)": ["Operating Profit Margin (%)", "Operating Profit Margin", "EBIT Margin (%)"],
            "Net Margin (%)": ["Net Margin (%)", "Net Profit Margin"],
        },
        "Operations / Industry KPIs": {
            "R&D Intensity (%)": ["R&D Intensity (%)", "R&D Intensity"],
            "Rule of 40 (%)": ["Rule of 40 (%)", "Rule of 40"],
            "Net Dollar Retention (%)": ["Net Dollar Retention (%)", "NDR"],
            "Gross Retention (%)": ["Gross Retention (%)", "Gross Retention"],
            "LTV/CAC (x)": ["LTV/CAC (x)", "LTV/CAC", "LTV to CAC"],
        },
        "Income & Efficiency": {
            "Three Fees Ratio (%)": [
                "Three Fees Ratio (%)",
                "Three Fees (%)",
                "SG&A+Fin / Revenue (%)",
                "Selling+Admin+Finance / Revenue (%)",
            ],
        },
        "Liquidity & Leverage": {
            "Debt/Equity (x)": ["Debt/Equity (x)", "D/E"],
            "Interest Coverage (x)": ["Interest Coverage (x)", "EBIT/Interest", "Interest Coverage (×)"],
        },
        "Cash Flow": {
            "FCF Margin (%)": ["FCF Margin (%)", "FCF Margin"],
        },
        "Valuation": {
            "Dividend Yield (%)": ["Dividend Yield (%)", "Dividend Yield", "DY"],
            "P/E (x)": ["P/E (x)", "P/E", "PE"],
            "P/B (x)": ["P/B (x)", "P/B", "PB"],
            "EV/EBITDA (x)": ["EV/EBITDA (x)", "EV/EBITDA"],
            "EV/Sales (×)": ["EV/Sales (×)", "EV/Sales"],
            "EPS (RM)": ["EPS (RM)", "EPS", "Earnings per Share", "Basic EPS (RM)", "Diluted EPS (RM)", "Basic EPS", "Diluted EPS", "EPS (sen)"],
        },
    },

    "Healthcare": {
        "Margins": {
            "Gross Margin (%)": ["Gross Margin (%)", "Gross Margin"],
            "EBITDA Margin (%)": ["EBITDA Margin (%)", "EBITDA Margin"],
            "Operating Profit Margin (%)": ["Operating Profit Margin (%)", "Operating Profit Margin", "EBIT Margin (%)"],
            "Net Margin (%)": ["Net Margin (%)", "Net Profit Margin"],
        },
        "Operations / Industry KPIs": {
            "Bed Occupancy (%)": ["Bed Occupancy (%)", "Bed Occupancy"],
            "ALOS (days)": ["Average Length of Stay (days)", "ALOS", "Average Length of Stay"],
            "Bed Turnover (admissions/bed)": ["Bed Turnover (admissions/bed)", "Bed Turnover"],
            "R&D Intensity (%)": ["R&D Intensity (%)", "R&D Intensity"],
            # Per-capacity economics:
            "Revenue per Bed (RM)": ["Revenue per Bed (RM)", "Revenue per Bed"],
            "EBITDA per Bed (RM)": ["EBITDA per Bed (RM)"],
            "Revenue per Patient Day (RM)": ["Revenue per Patient Day (RM)"],
            "EBITDA per Patient Day (RM)": ["EBITDA per Patient Day (RM)"],
        },
        "Income & Efficiency": {
            "Three Fees Ratio (%)": [
                "Three Fees Ratio (%)",
                "Three Fees (%)",
                "SG&A+Fin / Revenue (%)",
                "Selling+Admin+Finance / Revenue (%)",
            ],
        },
        "Returns": {
            "ROE (%)": ["ROE (%)", "ROE"],
            "ROA (%)": ["ROA (%)", "ROA"],
        },
        "Liquidity & Leverage": {
            "Debt/Equity (x)": ["Debt/Equity (x)", "D/E"],
            "Interest Coverage (x)": ["Interest Coverage (x)", "EBIT/Interest", "Interest Coverage (×)"],
        },
        "Valuation": {
            "Dividend Yield (%)": ["Dividend Yield (%)", "Dividend Yield", "DY"],
            "P/E (x)": ["P/E (x)", "P/E", "PE"],
            "P/B (x)": ["P/B (x)", "P/B", "PB"],
            "EV/EBITDA (x)": ["EV/EBITDA (x)", "EV/EBITDA"],
            "EPS (RM)": ["EPS (RM)", "EPS", "Earnings per Share", "Basic EPS (RM)", "Diluted EPS (RM)", "Basic EPS", "Diluted EPS", "EPS (sen)"],
        },
    },

    "Telco": {
        "Margins": {
            "EBITDA Margin (%)": ["EBITDA Margin (%)", "EBITDA Margin"],
            "Operating Profit Margin (%)": ["Operating Profit Margin (%)", "Operating Profit Margin", "EBIT Margin (%)"],
            "Net Margin (%)": ["Net Margin (%)", "Net Profit Margin"],
        },
        "Operations / Industry KPIs": {
            "ARPU (RM)": ["ARPU (RM)", "ARPU"],
            "Churn Rate (%)": ["Churn Rate (%)", "Churn Rate"],
        },
        "Income & Efficiency": {
            "Three Fees Ratio (%)": [
                "Three Fees Ratio (%)",
                "Three Fees (%)",
                "SG&A+Fin / Revenue (%)",
                "Selling+Admin+Finance / Revenue (%)",
            ],
        },
        "Cash Flow": {
            "Capex/Revenue (%)": ["Capex/Revenue (%)", "Capex to Revenue", "Capex Intensity (%)", "CapEx Intensity (%)"],
            "Operating CF Margin (%)": ["Operating CF Margin (%)", "Operating CF Margin", "OCF Margin (%)"],
        },
        "Liquidity & Leverage": {
            "Debt/Equity (x)": ["Debt/Equity (x)", "D/E"],
            # keep this legacy label for UI back-compat, plus the canonical one:
            "Net Debt/EBITDA (x)": ["Net Debt/EBITDA (x)", "Net Debt/EBITDA"],
            "Net Debt / EBITDA (×)": ["Net Debt / EBITDA (×)", "Net Debt/EBITDA (×)", "Net Debt/EBITDA"],
            "Interest Coverage (x)": ["Interest Coverage (x)", "EBIT/Interest", "Interest Coverage (×)"],
        },
        "Valuation": {
            "Dividend Yield (%)": ["Dividend Yield (%)", "Dividend Yield", "DY"],
            "P/E (x)": ["P/E (x)", "P/E", "PE"],
            "P/B (x)": ["P/B (x)", "P/B", "PB"],
            "EV/EBITDA (x)": ["EV/EBITDA (x)", "EV/EBITDA"],
            "EPS (RM)": ["EPS (RM)", "EPS", "Earnings per Share", "Basic EPS (RM)", "Diluted EPS (RM)", "Basic EPS", "Diluted EPS", "EPS (sen)"],
        },
    },

    "Construction": {
        "Margins": {
            "Gross Margin (%)": ["Gross Margin (%)", "Gross Margin"],
            "EBITDA Margin (%)": ["EBITDA Margin (%)", "EBITDA Margin"],
            "Operating Profit Margin (%)": ["Operating Profit Margin (%)", "Operating Profit Margin", "EBIT Margin (%)"],
            "Net Margin (%)": ["Net Margin (%)", "Net Profit Margin"],
        },
        "Operations / Industry KPIs": {
            "Backlog Coverage (x)": ["Backlog Coverage (x)", "Backlog Coverage"],
            "Win Rate (%)": ["Win Rate (%)", "Win Rate"],
        },
        "Efficiency & Working Capital": {
            "DIO (days)": ["DIO (days)", "DIO"],
            "DSO (days)": ["DSO (days)", "DSO"],
            "DPO (days)": ["DPO (days)", "DPO"],
            "Cash Conversion Cycle (days)": ["Cash Conversion Cycle (days)", "Cash Conversion Cycle"],
        },
        "Income & Efficiency": {
            "Three Fees Ratio (%)": [
                "Three Fees Ratio (%)",
                "Three Fees (%)",
                "SG&A+Fin / Revenue (%)",
                "Selling+Admin+Finance / Revenue (%)",
            ],
        },
        "Liquidity & Leverage": {
            "Debt/Equity (x)": ["Debt/Equity (x)", "D/E"],
            "Interest Coverage (x)": ["Interest Coverage (x)", "EBIT/Interest", "Interest Coverage (×)"],
            "Net Debt / EBITDA (×)": ["Net Debt / EBITDA (×)", "Net Debt/EBITDA (×)", "Net Debt/EBITDA"],
        },
        "Cash Flow": {
            "Capex/Revenue (%)": ["Capex/Revenue (%)", "Capex to Revenue"],
        },
        "Valuation": {
            "Dividend Yield (%)": ["Dividend Yield (%)", "Dividend Yield", "DY"],
            "P/E (x)": ["P/E (x)", "P/E", "PE"],
            "P/B (x)": ["P/B (x)", "P/B", "PB"],
            "EV/EBITDA (x)": ["EV/EBITDA (x)", "EV/EBITDA"],
            # disambiguate common GP naming:
            "Gross Profit Margin (%)": ["Gross Profit Margin (%)", "Gross Margin (%)", "GP Margin (%)"],
            "EPS (RM)": ["EPS (RM)", "EPS", "Earnings per Share", "Basic EPS (RM)", "Diluted EPS (RM)", "Basic EPS", "Diluted EPS", "EPS (sen)"],
        },
    },

    "Plantation": {
        "Margins": {
            "Gross Margin (%)": ["Gross Margin (%)", "Gross Margin"],
            "EBITDA Margin (%)": ["EBITDA Margin (%)", "EBITDA Margin"],
            "Operating Profit Margin (%)": ["Operating Profit Margin (%)", "Operating Profit Margin", "EBIT Margin (%)"],
            "Net Margin (%)": ["Net Margin (%)", "Net Profit Margin"],
        },
        "Operations / Industry KPIs": {
            "Unit Cash Cost (RM/t)": ["Unit Cash Cost (RM/t)", "Unit Cash Cost"],
            "CPO ASP (RM/t)": ["Average CPO Selling Price (RM/t)", "CPO ASP", "Average CPO Selling Price", "ASP (RM/t)"],
            "OER (CPO/FFB Input, %)": ["OER (CPO/FFB Input, %)", "OER (%)"],
            "Yield per Hectare (FFB t/ha)": ["Yield per Hectare (FFB t/ha)", "Yield per Hectare"],
            "Cash Margin per ton (RM/t)": ["Cash Margin per ton (RM/t)", "Cash Margin per ton"],
            "EBITDA per ton (RM/t)": ["EBITDA per ton (RM/t)", "EBITDA per ton"],
        },
        "Income & Efficiency": {
            "Three Fees Ratio (%)": [
                "Three Fees Ratio (%)",
                "Three Fees (%)",
                "SG&A+Fin / Revenue (%)",
                "Selling+Admin+Finance / Revenue (%)",
            ],
        },
        "Liquidity & Leverage": {
            "Debt/Equity (x)": ["Debt/Equity (x)", "D/E"],
            "Interest Coverage (x)": ["Interest Coverage (x)", "EBIT/Interest", "Interest Coverage (×)"],
            "Gearing (Debt/Assets, %)": ["Gearing (%)", "Debt/Assets (%)", "Debt/Assets (x)", "Gearing (x)"],
        },
        "Valuation": {
            "Dividend Yield (%)": ["Dividend Yield (%)", "Dividend Yield", "DY"],
            "P/E (x)": ["P/E (x)", "P/E", "PE"],
            "P/B (x)": ["P/B (x)", "P/B", "PB"],
            "EV/EBITDA (x)": ["EV/EBITDA (x)", "EV/EBITDA"],
            "EPS (RM)": ["EPS (RM)", "EPS", "Earnings per Share", "Basic EPS (RM)", "Diluted EPS (RM)", "Basic EPS", "Diluted EPS", "EPS (sen)"],
        },
    },

    "Property": {
        "Margins": {
            "Gross Margin (%)": ["Gross Margin (%)", "Gross Margin"],
            "EBITDA Margin (%)": ["EBITDA Margin (%)", "EBITDA Margin"],
            "Operating Profit Margin (%)": ["Operating Profit Margin (%)", "Operating Profit Margin", "EBIT Margin (%)"],
            "Net Margin (%)": ["Net Margin (%)", "Net Profit Margin"],
        },
        "Operations / Industry KPIs": {
            "Sales Rate (%)": ["Sales Rate (%)", "Sales Rate", "Take-up Rate (%)", "Take-up Rate"],
            "Unbilled Sales / Revenue (×)": ["Unbilled Sales / Revenue (×)", "Unbilled Sales/Revenue (×)"],
            "Unbilled Cover (months)": ["Unbilled Cover (months)", "Unbilled Cover"],
        },
        "Income & Efficiency": {
            "Three Fees Ratio (%)": [
                "Three Fees Ratio (%)",
                "Three Fees (%)",
                "SG&A+Fin / Revenue (%)",
                "Selling+Admin+Finance / Revenue (%)",
            ],
        },
        "Liquidity & Leverage": {
            "Gearing (x)": ["Debt/Equity (x)", "D/E", "Gearing (x)"],
            "Net Gearing (Net Debt/Equity, %)": ["Net Gearing (Net Debt/Equity, %)", "Net Gearing (%)"],
            "Interest Coverage (x)": ["Interest Coverage (x)", "EBIT/Interest", "Interest Coverage (×)"],
        },
        "Cash Flow": {
            "Capex/Revenue (%)": ["Capex/Revenue (%)", "Capex to Revenue"],
        },
        "Distribution & Valuation": {
            "Distribution/Dividend Yield (%)": ["Dividend Yield (%)", "Distribution Yield (%)", "DY"],
            "P/NAV (x)": ["P/NAV (x)", "Price/NAV", "P/NAV", "P/B (x)", "PB"],
            "P/RNAV (×)": ["P/RNAV (×)", "P/RNAV"],
            "P/E (x)": ["P/E (x)", "P/E", "PE"],
            "EV/EBITDA (x)": ["EV/EBITDA (x)", "EV/EBITDA"],
            "EPS (RM)": ["EPS (RM)", "EPS", "Earnings per Share", "Basic EPS (RM)", "Diluted EPS (RM)", "Basic EPS", "Diluted EPS", "EPS (sen)"],
        },
    },

    "Transportation/Logistics": {
        "Margins": {
            "EBITDA Margin (%)": ["EBITDA Margin (%)", "EBITDA Margin"],
            "Operating Profit Margin (%)": ["Operating Profit Margin (%)", "Operating Profit Margin", "EBIT Margin (%)"],
            "Net Margin (%)": ["Net Margin (%)", "Net Profit Margin"],
        },
        "Operations / Industry KPIs": {
            "Load Factor (%)": ["Load Factor (airlines, %)", "Load Factor"],
            "Yield (per km/parcel)": ["Yield per km/parcel", "Yield per km", "Yield", "Yield per km/parcel (RM)"],
            "On-Time Performance (%)": ["On-Time Performance (%)", "On-Time Performance"],
            "Revenue per TEU (RM/TEU)": ["Revenue per TEU (RM/TEU)", "Revenue per TEU"],
        },
        "Income & Efficiency": {
            "Three Fees Ratio (%)": [
                "Three Fees Ratio (%)",
                "Three Fees (%)",
                "SG&A+Fin / Revenue (%)",
                "Selling+Admin+Finance / Revenue (%)",
            ],
        },
        "Liquidity & Leverage": {
            "Debt/Equity (x)": ["Debt/Equity (x)", "D/E"],
            "Lease-Adj. Net Debt/EBITDA (x)": ["Net Debt/EBITDA (x)", "Net Debt/EBITDA"],
            "Net Debt / EBITDA (×)": ["Net Debt / EBITDA (×)", "Net Debt/EBITDA (×)", "Net Debt/EBITDA"],
            "Interest Coverage (x)": ["Interest Coverage (x)", "EBIT/Interest", "Interest Coverage (×)"],
        },
        "Cash Flow": {
            "Capex/Revenue (%)": ["Capex/Revenue (%)", "Capex to Revenue"],
        },
        "Valuation": {
            "Dividend Yield (%)": ["Dividend Yield (%)", "Dividend Yield", "DY"],
            "P/E (x)": ["P/E (x)", "P/E", "PE"],
            "EV/EBITDA (x)": ["EV/EBITDA (x)", "EV/EBITDA"],
            "EPS (RM)": ["EPS (RM)", "EPS", "Earnings per Share", "Basic EPS (RM)", "Diluted EPS (RM)", "Basic EPS", "Diluted EPS", "EPS (sen)"],
        },
    },

    "Leisure/Travel": {
        "Margins": {
            "EBITDA Margin (%)": ["EBITDA Margin (%)", "EBITDA Margin"],
            "Operating Profit Margin (%)": ["Operating Profit Margin (%)", "Operating Profit Margin", "EBIT Margin (%)"],
            "Net Margin (%)": ["Net Margin (%)", "Net Profit Margin"],
        },
        "Operations / Industry KPIs": {
            "Hotel Occupancy (%)": ["Hotel Occupancy (%)", "Hotel Occupancy"],
            "RevPAR (RM)": ["RevPAR (RM)", "RevPAR"],
            "Win Rate (%)": ["Gaming Win Rate (%)", "Win Rate"],
        },
        "Income & Efficiency": {
            "Three Fees Ratio (%)": [
                "Three Fees Ratio (%)",
                "Three Fees (%)",
                "SG&A+Fin / Revenue (%)",
                "Selling+Admin+Finance / Revenue (%)",
            ],
        },
        "Liquidity & Leverage": {
            "Debt/Equity (x)": ["Debt/Equity (x)", "D/E"],
            "Interest Coverage (x)": ["Interest Coverage (x)", "EBIT/Interest", "Interest Coverage (×)"],
        },
        "Cash Flow": {
            "Capex/Revenue (%)": ["Capex/Revenue (%)", "Capex to Revenue"],
        },
        "Valuation": {
            "Dividend Yield (%)": ["Dividend Yield (%)", "Dividend Yield", "DY"],
            "P/E (x)": ["P/E (x)", "P/E", "PE"],
            "EV/EBITDA (x)": ["EV/EBITDA (x)", "EV/EBITDA"],
            "EPS (RM)": ["EPS (RM)", "EPS", "Earnings per Share", "Basic EPS (RM)", "Diluted EPS (RM)", "Basic EPS", "Diluted EPS", "EPS (sen)"],
        },
    },

    "General": {
        "Margins": {
            "Gross Margin (%)": ["Gross Margin (%)", "Gross Margin"],
            "EBITDA Margin (%)": ["EBITDA Margin (%)", "EBITDA Margin"],
            "Operating Profit Margin (%)": ["Operating Profit Margin (%)", "Operating Profit Margin", "EBIT Margin (%)"],
            "Net Margin (%)": ["Net Margin (%)", "Net Profit Margin"],
        },
        "Returns": {
            "ROE (%)": ["ROE (%)", "ROE", "Return on Equity"],
            "ROA (%)": ["ROA (%)", "ROA", "Return on Assets"],
        },
        "Income & Efficiency": {
            "Three Fees Ratio (%)": [
                "Three Fees Ratio (%)",
                "Three Fees (%)",
                "SG&A+Fin / Revenue (%)",
                "Selling+Admin+Finance / Revenue (%)",
            ],
        },
        "Liquidity & Leverage": {
            "Current Ratio (x)": ["Current Ratio (x)", "Current Ratio"],
            "Debt/Equity (x)": ["Debt/Equity (x)", "D/E", "Debt/Equity"],
            # expose both x/× glyphs:
            "Interest Coverage (x)": ["Interest Coverage (x)", "EBIT/Interest", "Interest Coverage (×)"],
            "Net Debt / EBITDA (×)": ["Net Debt / EBITDA (×)", "Net Debt/EBITDA (×)", "Net Debt/EBITDA"],
        },
        "Cash Flow": {
            "Capex/Revenue (%)": ["Capex/Revenue (%)", "Capex to Revenue"],
        },
        "Valuation": {
            "Dividend Yield (%)": ["Dividend Yield (%)", "Dividend Yield", "DY"],
            "P/E (x)": ["P/E (x)", "P/E", "PE"],
            "P/B (x)": ["P/B (x)", "P/B", "PB"],
            "EV/EBITDA (x)": ["EV/EBITDA (x)", "EV/EBITDA"],
            "EPS (RM)": ["EPS (RM)", "EPS", "Earnings per Share", "Basic EPS (RM)", "Diluted EPS (RM)", "Basic EPS", "Diluted EPS", "EPS (sen)"],
        },
    },
}

# -------------- Back-compat (flat) --------------
INDUSTRY_SUMMARY_RATIOS = {
    bucket: _flatten_ratio_categories(cats)
    for bucket, cats in INDUSTRY_SUMMARY_RATIOS_CATEGORIES.items()
}

SHOW_RATIO_TABLE = True


# --- bucket-aware summary helpers (inherit "General" + override) ---

def summary_cfg_for(bucket: str) -> dict:
    base = INDUSTRY_SUMMARY_RATIOS_CATEGORIES.get("General", {})
    spec = INDUSTRY_SUMMARY_RATIOS_CATEGORIES.get(bucket, {})
    out = {sec: dict(base.get(sec, {})) for sec in SUMMARY_RATIO_CATEGORY_ORDER}
    for sec, items in spec.items():
        out.setdefault(sec, {})
        out[sec].update(items or {})
    return out

def summary_syn_index(bucket: str) -> dict:
    """lowercased label/synonym -> canonical label (tolerant to (%) and (x)/(×))"""
    idx = {}
    cfg = summary_cfg_for(bucket)
    for sec_items in cfg.values():
        for canon, syns in (sec_items or {}).items():
            canon = str(canon).strip()
            idx[canon.lower()] = canon
            variants = {
                canon.replace(" (%)",""), canon + " (%)",
                canon.replace(" (x)",""), canon + " (x)",
                canon.replace(" (×)",""), canon + " (×)",
            }
            for v in variants:
                idx[str(v).strip().lower()] = canon
            for s in (syns or []):
                idx[str(s).strip().lower()] = canon
    return idx

def summary_label_order(bucket: str) -> list[tuple[str, list[str]]]:
    """[(category, [ordered canonical labels...]), ...]"""
    cfg = summary_cfg_for(bucket)
    return [(sec, list(cfg.get(sec, {}).keys())) for sec in SUMMARY_RATIO_CATEGORY_ORDER]

# --- generic list helper: inherit General list + append bucket-specific uniques ---
def menu_for_bucket(bucket: str, mapping: dict[str, list[str]], base: str = "General") -> list[str]:
    base_list = list(mapping.get(base, []))
    extra = [x for x in mapping.get(bucket, []) if x not in base_list]
    return base_list + extra


# ================================
# Metric Menus by Industry (names only) — aligned to your config
# ================================

# ---------- Shared building blocks ----------
_BASE_TTM_NONFIN = [
    "Revenue",
    "Gross Profit",
    "Operating Profit",
    "EBITDA",
    "Net Profit",
    "Gross Margin",
    "Net Margin",
    "EPS",
    "EPS YoY",   
    "DPS",
    "Payout Ratio",  
    "Dividend Yield",
    "ROE",
    "ROA",
    "P/E",
    "P/B",
    "Net Debt / EBITDA",
    "Interest Coverage",
]

_BASE_CF_NONBANK = [
    "CFO",
    "Capex",
    "FCF",
    "FCF Margin",
    "FCF Yield",
    "Capex to Revenue",
    "CFO/EBITDA",
    "Cash Conversion",
]

# ---------- TTM menus (names only) ----------
TTM_METRICS_BY_BUCKET = {
    "Banking": [
        "Net Profit",
        "NII (incl Islamic)",
        "Operating Income",
        "Operating Expenses",
        "Provisions",
        "NIM",
        "Cost-to-Income Ratio",
        "CASA Ratio",
        "CASA (Core, %)",
        "Loan-to-Deposit Ratio",
        "CET1 Ratio",
        "Tier 1 Ratio",
        "Total Capital Ratio",
        "ROE",
        "ROA",
        "DPS",
        "Payout Ratio",
        "Dividend Yield",
        "EPS",
        "EPS YoY", 
        "P/E",
        "P/B",
    ],

    "REITs": _BASE_TTM_NONFIN + [
        "DPU",
        "Distribution Yield",
        "P/NAV",
        "Occupancy",
        "WALE",
        "Rental Reversion",
    ],

    "Manufacturing": _BASE_TTM_NONFIN + [
        "Inventory Days",
        "Receivables Days",
        "Payables Days",
        "Capacity Utilization",
        "Order Backlog",
        "Backlog Coverage",
    ],

    "Retail": _BASE_TTM_NONFIN + [
        "SSSG",
        "Average Basket Size",
        "Store Count",
        "Inventory Days",
        "Receivables Days",
        "Payables Days",
    ],

    "Financials": _BASE_TTM_NONFIN + [
        "Gross Written Premiums",
        "Net Claims Incurred",
        "Insurance Operating Expenses",
        "AUM",
        "Net Fee Income",
        "Finance Receivables",
        "Net Interest Income",
    ],

    "Utilities": _BASE_TTM_NONFIN + [
        "RAB",
        "Allowed Return",
        "Availability Factor",
        "Capex to Revenue",
    ],

    "Energy/Materials": _BASE_TTM_NONFIN + [
        "Lifting Cost",
        "Proven Reserves",
        "Annual Production",
        "Strip Ratio",
        "Head Grade",
        "Average Selling Price",
    ],

    "Tech": _BASE_TTM_NONFIN + [
        "R&D Expense",
        "R&D Intensity",
        "ARR",
        "Net Dollar Retention",
        "Gross Retention",
        "CAC",
        "LTV",
        "MAU",
        "Rule of 40",
    ],

    "Healthcare": _BASE_TTM_NONFIN + [
        "Bed Occupancy",
        "ALOS",
        "Patient Days",
        "Admissions",
        "R&D Expense",
    ],

    "Telco": _BASE_TTM_NONFIN + [
        "ARPU",
        "Subscribers",
        "Churn Rate",
        "Capex to Revenue",
        "Average Data Usage",
        "Spectrum Fees",
    ],

    "Construction": _BASE_TTM_NONFIN + [
        "Orderbook",
        "New Orders",
        "Tender Book",
        "Win Rate",
    ],

    "Plantation": _BASE_TTM_NONFIN + [
        "CPO Output",
        "FFB Input",
        "FFB Harvested",
        "Unit Cash Cost",
        "Tonnes Sold",
        "CPO ASP",
        "Planted Hectares",
        "Mature Area",
        "Average Tree Age",
    ],

    "Property": _BASE_TTM_NONFIN + [
        "Unbilled Sales",
        "Units Sold",
        "Units Launched",
        "RNAV per Share",
        "Net Debt",
        "GDV",
        "Sales Rate",
    ],

    "Transportation/Logistics": _BASE_TTM_NONFIN + [
        "TEU Throughput",
        "Load Factor",
        "Yield",
        "ASK",
        "RPK",
        "On-Time Performance",
        "Fleet Size",
    ],

    "Leisure/Travel": _BASE_TTM_NONFIN + [
        "Hotel Occupancy",
        "RevPAR",
        "Gaming Win Rate",
        "Visits",
        "Average Spend",
        "Room Count",
    ],

    "General": _BASE_TTM_NONFIN[:],
}

# ---------- Cash-Flow menus (names only) ----------
CF_METRICS_BY_BUCKET = {
    "Banking": ["CFO"],

    "REITs": _BASE_CF_NONBANK[:],
    "Manufacturing": _BASE_CF_NONBANK[:],
    "Retail": _BASE_CF_NONBANK[:],
    "Financials": _BASE_CF_NONBANK[:],
    "Utilities": _BASE_CF_NONBANK[:],
    "Energy/Materials": _BASE_CF_NONBANK[:],
    "Tech": _BASE_CF_NONBANK[:],
    "Healthcare": _BASE_CF_NONBANK[:],
    "Telco": _BASE_CF_NONBANK[:],
    "Construction": _BASE_CF_NONBANK[:],
    "Plantation": _BASE_CF_NONBANK[:],
    "Property": _BASE_CF_NONBANK[:],
    "Transportation/Logistics": _BASE_CF_NONBANK[:],
    "Leisure/Travel": _BASE_CF_NONBANK[:],
    "General": _BASE_CF_NONBANK[:],
}

# ---------- CAGR menus (names only) ----------
# (Built directly from your CAGR_BY_BUCKET; adds PEG (Graham) & Margin of Safety)
CAGR_METRICS_BY_BUCKET = {
    "General": [
        "Revenue CAGR",
        "Net Profit CAGR",
        "EPS CAGR",
        "BVPS CAGR",
        "PEG (Graham)",
        "Margin of Safety",
    ],

    "Manufacturing": [
        "Revenue CAGR",
        "EBITDA CAGR",
        "Net Profit CAGR",
        "Operating Cash Flow CAGR",
        "Free Cash Flow CAGR",
        "Orderbook CAGR",
        "PEG (Graham)",
        "Margin of Safety",
    ],

    "Retail": [
        "Revenue CAGR",
        "Net Profit CAGR",
        "EPS CAGR",
        "Store Count CAGR",
        "PEG (Graham)",
        "Margin of Safety",
    ],

    "Financials": [
        "Net Profit CAGR",
        "EPS CAGR",
        "AUM CAGR",
        "PEG (Graham)",
        "Margin of Safety",
    ],

    # NOTE: per your rule, Banking NEVER uses Revenue in CAGR.
    "Banking": [
        "Gross Loans CAGR",
        "Deposits CAGR",
        "Net Profit CAGR",
        "EPS CAGR",
        "PEG (Graham)",
        "Margin of Safety",
    ],

    "REITs": [
        "NPI CAGR",
        "DPU CAGR",
        "NAV per Unit CAGR",
        "PEG (Graham)",
        "Margin of Safety",
    ],

    "Utilities": [
        "Revenue CAGR",
        "RAB CAGR",
        "PEG (Graham)",
        "Margin of Safety",
    ],

    "Energy/Materials": [
        "Revenue CAGR",
        "EBITDA CAGR",
        "Net Profit CAGR",
        "Annual Production CAGR",
        "PEG (Graham)",
        "Margin of Safety",
    ],

    "Tech": [
        "Revenue CAGR",
        "ARR CAGR",
        "Net Profit CAGR",
        "EPS CAGR",
        "PEG (Graham)",
        "Margin of Safety",
    ],

    "Healthcare": [
        "Revenue CAGR",
        "Net Profit CAGR",
        "Bed Count CAGR",
        "PEG (Graham)",
        "Margin of Safety",
    ],

    "Telco": [
        "Revenue CAGR",
        "Subscribers CAGR",
        "ARPU CAGR",
        "PEG (Graham)",
        "Margin of Safety",
    ],

    "Construction": [
        "Revenue CAGR",
        "Net Profit CAGR",
        "Orderbook CAGR",
        "PEG (Graham)",
        "Margin of Safety",
    ],

    "Plantation": [
        "Revenue CAGR",
        "Net Profit CAGR",
        "Tonnes Sold CAGR",
        "PEG (Graham)",
        "Margin of Safety",
    ],

    "Property": [
        "Revenue CAGR",
        "Net Profit CAGR",
        "Unbilled Sales CAGR",
        "PEG (Graham)",
        "Margin of Safety",
    ],

    "Transportation/Logistics": [
        "Revenue CAGR",
        "EBITDA CAGR",
        "Net Profit CAGR",
        "TEU Throughput CAGR",
        "PEG (Graham)",
        "Margin of Safety",
    ],

    "Leisure/Travel": [
        "Revenue CAGR",
        "Net Profit CAGR",
        "Visits CAGR",
        "PEG (Graham)",
        "Margin of Safety",
    ],
}
