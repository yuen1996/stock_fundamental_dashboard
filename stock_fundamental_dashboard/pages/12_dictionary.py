# === INLINE INVESTOR DICTIONARY (self-contained; no utils.dictionary import) ===
from __future__ import annotations
import re
import pandas as pd
import streamlit as st
from config import (
    INDUSTRY_FORM_CATEGORIES,
    INDUSTRY_SUMMARY_RATIOS_CATEGORIES,
    _ttm_agg_for,
)

# ---------- CSS (chips, cards, flyout) ----------
def _dd_inject_css_once(width_px: int = 440):
    key = f"_dd_css_{width_px}"
    if st.session_state.get(key):  # already injected
        return
    st.session_state[key] = True

    st.markdown(f"""
<style>
/* panel cards + chips (harmonized with utils/ui look) */
.dict-sticky {{ position: sticky; top: 72px; }}
.dict-card  {{ background:#fff; border:1px solid #e5e7eb; border-radius:14px;
              box-shadow:0 8px 24px rgba(15,23,42,.06); padding:12px 14px; }}
.dict-ctl   {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
.dict-chip  {{ display:inline-block; padding:.18rem .5rem; border-radius:999px;
              font-weight:800; font-size:.75rem; border:1px solid #e5e7eb; }}
.dict-chip.good {{ background:#ecfdf5; color:#065f46; border-color:#a7f3d0; }}
.dict-chip.ok   {{ background:#eff6ff; color:#1e3a8a; border-color:#bfdbfe; }}
.dict-chip.warn {{ background:#fffbeb; color:#92400e; border-color:#fde68a; }}
.dict-chip.bad  {{ background:#fef2f2; color:#991b1b; border-color:#fecaca; }}
.dict-small {{ color:#64748b; font-size:.85rem; }}

/* flyout: make Streamlit dialog behave like a right drawer */
:root {{ --dict-flyout-width: {int(width_px)}px; }}
div[role="dialog"] > div:first-child {{
  margin-left:auto !important; margin-right:0 !important;
  width:var(--dict-flyout-width) !important; max-width:var(--dict-flyout-width) !important;
  height:100vh !important; border-radius:14px 0 0 14px !important;
  overflow:hidden !important; box-shadow:0 12px 40px rgba(15,23,42,.28) !important;
  border:1px solid #e5e7eb !important; padding:0 !important; background:#fff !important;
}}
div[role="dialog"] ~ div {{ background: rgba(2,6,23,.28) !important; }} /* overlay tint */

.dict-flyout-head {{
  display:flex; align-items:center; justify-content:space-between;
  padding:12px 14px; border-bottom:1px solid #e5e7eb; background:#fff;
}}
.dict-flyout-body {{ padding:10px 12px 14px 12px; height:calc(100vh - 56px); overflow:auto; background:#fff; }}
.dict-close {{ border:1px solid #e5e7eb; background:#fff; border-radius:10px; padding:6px 10px; cursor:pointer; }}
.dict-close:hover {{ background:#f8fafc; }}
</style>
""", unsafe_allow_html=True)

# ---------- Content: glossary / formulas / guides ----------
DD_GLOSSARY = {
    # Banking assets
    "FVTPL Investments": {
        "what": "Debt securities at fair value through P/L (trading/designated). Include only interest-bearing (gov bills/bonds, money-market, Cagamas, corporate bonds/sukuk, structured deposits). Exclude equities, unit trusts and all derivatives.",
        "where": "BS/Notes: 'Financial assets at FVTPL' — use total; for IEA we include only the interest-bearing portion (StrictPlus).",
        "why": "Can be part of IEA when interest-bearing.",
    },
    "FVTPL Derivatives": {"what": "Derivative assets at fair value (trading/hedging).", "where": "BS/Notes: 'Derivative financial assets'.", "why": "Tracked but excluded from IEA and NIM denominator."},
    "FVOCI Investments": {"what": "Debt securities at fair value through OCI.", "where": "BS: 'Financial investments measured at FVOCI' (total).", "why": "Included in IEA (Strict & StrictPlus)."},
    "Amortised Cost Investments": {"what": "Debt securities measured at amortised cost (hold-to-collect).", "where": "BS: 'Financial investments measured at amortised cost' (total).", "why": "Included in IEA."},
    "Reverse Repos": {"what": "Financial assets purchased under resale (reverse-repo).", "where": "BS: 'Financial assets purchased under resale agreements'.", "why": "Included in IEA."},
    "Net Loans": {"what": "Gross loans minus loan-loss allowances (ECL).", "where": "BS: 'Net loans/financing' (TOTAL for customers + FIs).", "why": "In IEA Strict; app computes Net = Gross − LLR if blank."},
    "Earning Assets": {"what": "Denominator for NIM. Strict = Net Loans + FVOCI + Amortised Cost + Reverse Repos. StrictPlus = Strict + interest-bearing FVTPL.", "where": "Computed; or key-in override.", "why": "Normalizes NII to assets that earn interest."},

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

DD_FORMULAS = {
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

DD_GUIDES = {
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
# -----------------------------------------------------------------------------

def _dd_norm(s: str) -> str:
    return re.sub(r"\s+"," ", (s or "").strip()).lower()

@st.cache_data(show_spinner=False)
def dd_build_dictionary_df() -> pd.DataFrame:
    """Build one combined dictionary table from your config + glossary."""
    rows = []
    # Raw fields from INDUSTRY_FORM_CATEGORIES
    for bucket, sections in (INDUSTRY_FORM_CATEGORIES or {}).items():
        for sec, fields in (sections or {}).items():
            for f in (fields or []):
                label = f.get("label","")
                key   = f.get("key","")
                unit  = f.get("unit") or ""
                ftype = f.get("type","float")
                helpx = f.get("help") or ""
                ttm   = _ttm_agg_for(sec, key, label)
                g     = DD_GLOSSARY.get(label) or DD_GLOSSARY.get(key) or {}
                rows.append({
                    "Term": label, "Key": key, "Industry": bucket, "Section": sec,
                    "Unit": unit, "Type": ftype, "TTM policy": ttm,
                    "What": g.get("what",""),
                    "Where": g.get("where", (helpx or f"Look for “{label}” in {sec}.")),
                    "Why": g.get("why",""),
                    "Formula": g.get("formula",""),
                    "Source": "Annual Form Field",
                })
    # Ratios from INDUSTRY_SUMMARY_RATIOS_CATEGORIES
    for bucket, groups in (INDUSTRY_SUMMARY_RATIOS_CATEGORIES or {}).items():
        for sec, items in (groups or {}).items():
            for canon, _ in (items or {}).items():
                rows.append({
                    "Term": canon, "Key": "", "Industry": bucket, "Section": f"Summary — {sec}",
                    "Unit": "%", "Type": "ratio", "TTM policy": "derived",
                    "What": DD_GLOSSARY.get(canon,{}).get("what",""),
                    "Where": "Derived by app (see formula).",
                    "Why": DD_GLOSSARY.get(canon,{}).get("why",""),
                    "Formula": DD_FORMULAS.get(canon, DD_GLOSSARY.get(canon,{}).get("formula","")),
                    "Source": "Summary Ratio",
                })
    return pd.DataFrame(rows)

def _dd_render_panel(key_prefix: str):
    df = dd_build_dictionary_df()
    st.markdown("<div class='dict-card'>", unsafe_allow_html=True)
    st.markdown("**Investor Dictionary**", unsafe_allow_html=True)

    # Controls
    q = st.text_input("Search", key=f"{key_prefix}_q", placeholder="term / key / formula / why …")
    bkt_opts = ["All"] + sorted(df["Industry"].dropna().unique().tolist())
    sec_opts = ["All"] + sorted(df["Section"].dropna().unique().tolist())
    bkt = st.selectbox("Industry", bkt_opts, index=0, key=f"{key_prefix}_b")
    sec = st.selectbox("Section", sec_opts, index=0, key=f"{key_prefix}_s")

    # Filter
    mask = pd.Series(True, index=df.index)
    if q:
        ql = q.lower()
        cols = ["Term","Key","What","Where","Why","Formula","Industry","Section"]
        mask &= df[cols].astype(str).apply(lambda s: s.str.lower().str.contains(ql, na=False)).any(axis=1)
    if bkt != "All":
        mask &= (df["Industry"] == bkt)
    if sec != "All":
        mask &= (df["Section"] == sec)

    out = df.loc[mask].sort_values(["Industry","Section","Term"]).head(200)
    st.dataframe(
        out[["Term","Key","Industry","Section","Unit","TTM policy","What","Where","Why","Formula"]],
        hide_index=True, use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Guides
    st.markdown("<div class='dict-card' style='margin-top:10px'>", unsafe_allow_html=True)
    st.markdown("**Quick Guides (rules-of-thumb)**", unsafe_allow_html=True)
    sel = bkt if bkt != "All" else "General"
    guide = DD_GUIDES.get(sel, DD_GUIDES["General"])
    for label, bands in (guide or {}).items():
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

def dd_dictionary_flyout(*, key_prefix: str = "dict", width_px: int = 440,
                         button_label: str = "📚 Dictionary", open_default: bool = False,
                         show_pin: bool = True) -> None:
    """
    Top-right button → right-side overlay flyout.
    Usage in page:
        dd_dictionary_flyout(key_prefix="viewstock", width_px=440)
    """
    _dd_inject_css_once(width_px)

    open_key = f"{key_prefix}_flyout_open"
    pin_key  = f"{key_prefix}_flyout_pin"
    st.session_state.setdefault(open_key, open_default)
    st.session_state.setdefault(pin_key, False)

    # Tiny launcher button (top-right)
    cols = st.columns([1,1,1,1,1,1,1,1,1,1,1,1])
    with cols[-1]:
        if st.button(button_label, key=f"{key_prefix}_flyout_btn", use_container_width=True):
            st.session_state[open_key] = True
            st.rerun()

    # Flyout content
    if st.session_state[open_key] or st.session_state[pin_key]:
        @st.dialog("Investor Dictionary")
        def _show():
            st.markdown(
                "<div class='dict-flyout-head'>"
                "<div style='font-weight:800;font-size:1rem'>Investor Dictionary</div>"
                "<div style='display:flex;gap:8px;align-items:center'></div>"
                "</div>",
                unsafe_allow_html=True,
            )
            top_cols = st.columns([1,1,8,1])
            with top_cols[0]:
                if show_pin:
                    st.session_state[pin_key] = st.checkbox("Pin", value=st.session_state[pin_key], help="Keep flyout open after reruns.")
            with top_cols[-1]:
                if st.button("✕", key=f"{key_prefix}_close", help="Close", use_container_width=True):
                    st.session_state[open_key] = False
                    st.rerun()

            st.markdown("<div class='dict-flyout-body'>", unsafe_allow_html=True)
            _dd_render_panel(key_prefix)
            st.markdown("</div>", unsafe_allow_html=True)

        _show()
# === END INLINE INVESTOR DICTIONARY ===
