# utils/dictionary.py
from __future__ import annotations
import re
import pandas as pd
import streamlit as st

# --- pull your config + TTM policy ---
from config import (
    INDUSTRY_FORM_CATEGORIES,
    INDUSTRY_SUMMARY_RATIOS_CATEGORIES,
    _ttm_agg_for,
)

# ---------------- CSS (chips, sticky panel) ----------------
_CSS = """
<style>
.dict-sticky { position: sticky; top: 72px; }
.dict-card  { background:#fff; border:1px solid #e5e7eb; border-radius:14px;
              box-shadow:0 8px 24px rgba(15,23,42,.06); padding:12px 14px; }
.dict-ctl   { display:flex; gap:8px; align-items:center; }
.dict-chip  { display:inline-block; padding:.18rem .5rem; border-radius:999px;
              font-weight:800; font-size:.75rem; border:1px solid #e5e7eb; }
.dict-chip.good { background:#ecfdf5; color:#065f46; border-color:#a7f3d0; }
.dict-chip.ok   { background:#eff6ff; color:#1e3a8a; border-color:#bfdbfe; }
.dict-chip.warn { background:#fffbeb; color:#92400e; border-color:#fde68a; }
.dict-chip.bad  { background:#fef2f2; color:#991b1b; border-color:#fecaca; }
.dict-small { color:#64748b; font-size:.85rem; }
</style>
"""

def _inject_css_once():
    if "_dict_css" not in st.session_state:
        st.markdown(_CSS, unsafe_allow_html=True)
        st.session_state["_dict_css"] = True

# ---------------- Banking/REIT quick glossary + formulas ----------------
GLOSSARY = {
    # Banking assets
    "FVTPL Investments": {
        "what": "Debt securities at fair value through P/L (trading/designated). Include only interest-bearing (gov bills/bonds, money-market, Cagamas, corporate bonds/sukuk, structured deposits). Exclude equities, unit trusts and all derivatives.",
        "where": "BS/Notes: 'Financial assets at FVTPL' — use total; for IEA we include only the interest-bearing portion (StrictPlus).",
        "why": "Can be part of IEA when interest-bearing.",
    },
    "FVTPL Derivatives": {
        "what": "Derivative assets at fair value (trading/hedging).",
        "where": "BS/Notes: 'Derivative financial assets'.",
        "why": "Tracked but excluded from IEA and NIM denominator.",
    },
    "FVOCI Investments": {
        "what": "Debt securities at fair value through OCI.",
        "where": "BS: 'Financial investments measured at FVOCI' (total).",
        "why": "Included in IEA (Strict & StrictPlus).",
    },
    "Amortised Cost Investments": {
        "what": "Debt securities measured at amortised cost (hold-to-collect).",
        "where": "BS: 'Financial investments measured at amortised cost' (total).",
        "why": "Included in IEA.",
    },
    "Reverse Repos": {
        "what": "Financial assets purchased under resale (reverse-repo).",
        "where": "BS: 'Financial assets purchased under resale agreements'.",
        "why": "Included in IEA.",
    },
    "Net Loans": {
        "what": "Gross loans minus loan-loss allowances (ECL).",
        "where": "BS: 'Net loans/financing' (TOTAL for customers + FIs).",
        "why": "In IEA Strict; app computes Net = Gross − LLR if blank.",
    },
    "Earning Assets": {
        "what": "Denominator for NIM. Strict = Net Loans + FVOCI + Amortised Cost + Reverse Repos. StrictPlus = Strict + interest-bearing FVTPL.",
        "where": "Computed; or key-in override.",
        "why": "Normalizes NII to assets that earn interest.",
    },

    # Banking ratios
    "NIM (%)": {"what": "Net Interest/Financing Margin.", "where": "Computed: NII(incl Islamic)/Avg Earning Assets×100"},
    "Cost-to-Income Ratio (%)": {"what": "Cost efficiency.", "where": "Operating Expenses / Operating Income×100"},
    "CASA Ratio (%)": {"what": "Share of deposits in current & savings.", "where": "(Demand+Savings)/Deposits×100"},
    "CASA (Core, %)": {"what": "Core CASA = (Demand+Savings)/(Demand+Savings+Fixed)×100."},
    "Loan-to-Deposit Ratio (%)": {"what": "Loans funded by deposits.", "where": "Gross Loans/Deposits×100"},
    "NPL Ratio (%)": {"what": "Non-performers / Gross loans.", "where": "NPL/Gross Loans×100"},
    "Loan-Loss Coverage (×)": {"what": "Allowances covering NPLs.", "where": "Loan Loss Reserve/NPL"},

    # REIT quicks
    "NPI Margin (%)": {"what": "Net property income margin.", "where": "NPI/Revenue×100"},
    "WALE (years)": {"what": "Weighted Avg Lease Expiry (yrs)."},
    "Rental Reversion (%)": {"what": "% change on renewals vs prior rent."},
    "P/NAV (x)": {"what": "Price to NAV per unit.", "where": "Price per unit/NAV per unit"},
    "DPU": {"what": "Distribution per Unit."},
}

FORMULAS = {
    "Gross Margin (%)": "Gross Profit ÷ Revenue × 100",
    "EBITDA Margin (%)": "EBITDA ÷ Revenue × 100",
    "Operating Profit Margin (%)": "Operating Profit ÷ Revenue × 100",
    "Net Margin (%)": "Net Profit ÷ Revenue × 100",
    "Current Ratio (x)": "Current Assets ÷ Current Liabilities",
    "Quick Ratio (x)": "(Current Assets − Inventory) ÷ Current Liabilities",
    "Debt/Equity (x)": "Total Borrowings ÷ Equity",
    "Interest Coverage (x)": "EBIT ÷ Interest Expense",
    "Net Debt / EBITDA (×)": "(Total Borrowings − Cash & Equivalents) ÷ EBITDA",
    "Capex/Revenue (%)": "Capex ÷ Revenue × 100",
    "ROE (%)": "Net Profit ÷ Equity × 100",
    "ROA (%)": "Net Profit ÷ Total Assets × 100",
    "Distribution Yield (%)": "DPU (TTM) ÷ Price per Unit × 100",
    "P/NAV (x)": "Price per Unit ÷ NAV per Unit",
    "NIM (%)": "NII (incl Islamic) ÷ Average Earning Assets × 100",
    "Cost-to-Income Ratio (%)": "Operating Expenses ÷ Operating Income × 100",
}

# ---------------- rule-of-thumb bands (non-advice heuristics) ----------------
GUIDES = {
    "General": {
        "P/B (x)": {"good":"≤1.0", "ok":"1.0–2.0", "watch":">2.0"},
        "P/E (x)": {"good":"≤12–15", "ok":"15–25", "watch":">25"},
        "Dividend Yield (%)": {"good":"≥4–5%", "ok":"2–4%", "watch":"<2%"},
        "Interest Coverage (x)": {"good":"≥6×", "ok":"3–6×", "watch":"<3×"},
    },
    "Banking": {
        "NIM (%)": {"good":"≥3%", "ok":"2–3%", "watch":"<2%"},
        "Cost-to-Income (%)": {"good":"<45%", "ok":"45–55%", "watch":">55%"},
        "Loan-to-Deposit (%)": {"good":"85–100%", "ok":"70–85% or 100–105%", "watch":">105%"},
        "NPL Ratio (%)": {"good":"<1.5%", "ok":"1.5–3%", "watch":">3%"},
        "Loan-Loss Coverage (×)": {"good":">1.0×", "ok":"0.7–1.0×", "watch":"<0.7×"},
        "CET1 Ratio (%)": {"good":"≥13%", "ok":"11–13%", "watch":"<11%"},
    },
    "REITs": {
        "Gearing (Debt/Assets, %)": {"good":"<35%", "ok":"35–45%", "watch":">45%"},
        "Interest Coverage (x)": {"good":">=3.0×", "ok":"2.0–3.0×", "watch":"<2.0×"},
        "Occupancy (%)": {"good":">=95%", "ok":"90–95%", "watch":"<90%"},
        "WALE (years)": {"good":">=3–4y", "ok":"2–3y", "watch":"<2y"},
        "Distribution Yield (%)": {"good":"≥5–6%", "ok":"3–5%", "watch":"<3%"},
    },
}
# (Heuristics vary by market/cycle; treat as rough context only.)

def _norm(s:str) -> str:
    return re.sub(r"\s+"," ", (s or "").strip()).lower()

# ---------------- build dictionary dataframe from your config ----------------
@st.cache_data(show_spinner=False)
def build_dictionary_df() -> pd.DataFrame:
    rows = []
    # raw fields
    for bucket, sections in INDUSTRY_FORM_CATEGORIES.items():
        for sec, fields in sections.items():
            for f in fields:
                label = f.get("label","")
                key   = f.get("key","")
                unit  = f.get("unit") or ""
                ftype = f.get("type","float")
                helpx = f.get("help") or ""
                ttm   = _ttm_agg_for(sec, key, label)
                g     = GLOSSARY.get(label) or GLOSSARY.get(key) or {}
                rows.append({
                    "Term": label, "Key": key, "Industry": bucket, "Section": sec,
                    "Unit": unit, "Type": ftype, "TTM policy": ttm,
                    "What": g.get("what",""),
                    "Where": g.get("where", (helpx or f"Look for “{label}” in {sec}.")),
                    "Why": g.get("why",""),
                    "Formula": g.get("formula",""),
                    "Source": "Annual Form Field",
                })
    # ratios
    for bucket, groups in INDUSTRY_SUMMARY_RATIOS_CATEGORIES.items():
        for sec, items in (groups or {}).items():
            for canon, _ in (items or {}).items():
                rows.append({
                    "Term": canon, "Key": "", "Industry": bucket, "Section": f"Summary — {sec}",
                    "Unit": "%", "Type": "ratio", "TTM policy": "derived",
                    "What": GLOSSARY.get(canon,{}).get("what",""),
                    "Where": "Derived by app (see formula).",
                    "Why": GLOSSARY.get(canon,{}).get("why",""),
                    "Formula": FORMULAS.get(canon, GLOSSARY.get(canon,{}).get("formula","")),
                    "Source": "Summary Ratio",
                })
    return pd.DataFrame(rows)

# ---------------- sidebar renderer ----------------
class DictLayout:
    """Returned to the caller so they can write their content in .main."""
    def __init__(self, main, aside): self.main, self.aside = main, aside

def right_dictionary_sidebar(*, key_prefix: str = "dict", opened: bool=False,
                             main_width: int = 7, aside_width: int = 3) -> DictLayout:
    """
    Call near the top of any page, then write the rest of your page inside:
        lay = right_dictionary_sidebar(key_prefix="addedit")
        with lay.main:
            ... (your original page content) ...
    """
    _inject_css_once()

    open_key = f"{key_prefix}_open"
    if open_key not in st.session_state: st.session_state[open_key] = opened

    # Toggle lives on the top-right as a tiny control row
    c = st.columns([1,1,1,1,1,1,1,1,1,1,1,1])
    with c[-1]:
        st.toggle("📚 Dictionary", key=open_key)

    if st.session_state[open_key]:
        main, aside = st.columns([main_width, aside_width], gap="large")
        with aside:
            with st.container():
                st.markdown("<div class='dict-sticky'>", unsafe_allow_html=True)
                _render_dictionary_panel(key_prefix)
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        main, aside = st.container(), None

    return DictLayout(main, aside)

def _render_dictionary_panel(key_prefix: str):
    df = build_dictionary_df()
    st.markdown("<div class='dict-card'>", unsafe_allow_html=True)
    st.markdown("**Investor Dictionary**", unsafe_allow_html=True)

    # Controls
    q = st.text_input("Search", key=f"{key_prefix}_q", placeholder="term / key / formula / why …")
    bkt = st.selectbox("Industry", ["All"] + sorted(df["Industry"].unique()), index=0, key=f"{key_prefix}_b")
    sec = st.selectbox("Section", ["All"] + sorted(df["Section"].unique()), index=0, key=f"{key_prefix}_s")

    mask = pd.Series(True, index=df.index)
    if q:
        ql = q.lower()
        mask &= df.apply(lambda r: any(ql in str(v).lower() for v in
                   [r["Term"], r["Key"], r["What"], r["Where"], r["Why"], r["Formula"], r["Industry"], r["Section"]]), axis=1)
    if bkt != "All": mask &= (df["Industry"] == bkt)
    if sec != "All": mask &= (df["Section"] == sec)

    out = df.loc[mask].sort_values(["Industry","Section","Term"]).head(200)
    st.dataframe(
        out[["Term","Key","Industry","Section","Unit","TTM policy","What","Where","Why","Formula"]],
        hide_index=True, use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Guides (rule-of-thumb bands)
    st.markdown("<div class='dict-card' style='margin-top:10px'>", unsafe_allow_html=True)
    st.markdown("**Quick Guides (rules-of-thumb)**", unsafe_allow_html=True)
    sel = bkt if bkt != "All" else "General"
    guide = GUIDES.get(sel, GUIDES["General"])
    for label, bands in guide.items():
        st.markdown(
            f"<div class='dict-ctl'><div style='min-width:150px;font-weight:700'>{label}</div>"
            f"<span class='dict-chip good'>good: {bands.get('good','—')}</span>"
            f"<span class='dict-chip ok'>ok: {bands.get('ok','—')}</span>"
            f"<span class='dict-chip warn'>watch: {bands.get('watch','—')}</span></div>",
            unsafe_allow_html=True,
        )
    st.markdown("<div class='dict-small'>Heuristics vary by market/cycle; use as rough context only.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📌 Banking IEA policy cheat-sheet"):
        st.markdown("""
- **Strict IEA** = Net Loans + FVOCI + Amortised Cost + Reverse Repos  
- **StrictPlus** = Strict IEA **+ interest-bearing FVTPL Investments**  
- Exclude: derivative assets, equities/unit trusts, cash & short-term funds (unless your policy differs)  
- **NIM** uses **NII (incl Islamic)** as numerator by default; override keys available in config.  
- **CIR** = Operating Expenses / Operating Income × 100; overrides available.  
- **Core CASA** = (Demand + Savings) / (Demand + Savings + Fixed) × 100.
        """)
