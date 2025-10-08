# pages/12_Dictionary.py
from __future__ import annotations
import os, re, inspect
import pandas as pd
import streamlit as st

# ---- Shared app UI (your utils/ui.py) ---------------------------------------
try:
    from utils.ui import setup_page, render_page_title, section
except Exception:
    # Soft fallback so the page still renders if utils/ui.py is not available
    def setup_page(title: str, page_id: str | None = None): st.set_page_config(page_title=title, layout="wide")
    def render_page_title(page_name: str): st.title(f"📊 Fundamentals Dashboard — {page_name}")
    def section(title: str, desc: str = "", tone: str = "") -> str:
        return f'<div style="padding:.6rem .8rem;border:1px solid #e5e7eb;border-radius:12px;background:#fff;font-weight:800;margin:.75rem 0">{title} <span style="font-weight:400;color:#64748b">{desc}</span></div>'

# ---- Pull config (robust to import errors) ----------------------------------
try:
    from config import (
        INDUSTRY_FORM_CATEGORIES,
        INDUSTRY_SUMMARY_RATIOS_CATEGORIES,
        _ttm_agg_for,
    )
except Exception as _cfg_err:
    INDUSTRY_FORM_CATEGORIES = {}
    INDUSTRY_SUMMARY_RATIOS_CATEGORIES = {}
    def _ttm_agg_for(*_a, **_k): return ""
    st.warning(f"Config not loaded: {getattr(_cfg_err, 'msg', _cfg_err)}")

# =============================================================================
#                                   CSS
# =============================================================================
_DD_CSS = """
<style>
:root{
  --bg:#f6f7fb; --surface:#ffffff; --text:#0f172a; --muted:#475569; --border:#e5e7eb;
  --shadow:0 8px 24px rgba(15,23,42,.06);
  --primary:#4f46e5; --info:#0ea5e9; --success:#10b981; --warning:#f59e0b; --danger:#ef4444;
}

/* Card + sticky */
.dict-sticky { position: sticky; top: 72px; }
.dict-card  { background:#fff; border:1px solid var(--border); border-radius:14px;
              box-shadow:var(--shadow); padding:12px 14px; }

/* Controls and chips */
.dict-ctl   { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.dict-chip  { display:inline-block; padding:.18rem .5rem; border-radius:999px;
              font-weight:800; font-size:.75rem; border:1px solid var(--border); }
.dict-chip.good { background:#ecfdf5; color:#065f46; border-color:#a7f3d0; }
.dict-chip.ok   { background:#eff6ff; color:#1e3a8a; border-color:#bfdbfe; }
.dict-chip.warn { background:#fffbeb; color:#92400e; border-color:#fde68a; }
.dict-chip.bad  { background:#fef2f2; color:#991b1b; border-color:#fecaca; }
.dict-small { color:#64748b; font-size:.85rem; }

/* Download row */
.dict-actions { display:flex; align-items:center; gap:10px; justify-content:flex-end; }
</style>
"""

_FLYOUT_CSS_TEMPLATE = """
<style>
:root { --dict-flyout-width: %(width_px)spx; }

/* Make Streamlit dialog look like a right flyout */
div[role="dialog"] > div:first-child {
  margin-left: auto !important;
  margin-right: 0 !important;
  width: var(--dict-flyout-width) !important;
  max-width: var(--dict-flyout-width) !important;
  height: 100vh !important;
  border-radius: 14px 0 0 14px !important;
  overflow: hidden !important;
  box-shadow: 0 12px 40px rgba(15,23,42,.28) !important;
  border: 1px solid #e5e7eb !important;
  padding: 0 !important;
}

/* Soften overlay */
div[role="dialog"] ~ div { background: rgba(2, 6, 23, 0.28) !important; }

/* Body */
div[role="dialog"] h1, div[role="dialog"] h2, div[role="dialog"] h3 { margin: 0 0 8px 0; }
.dict-flyout-head {
  display:flex; align-items:center; justify-content:space-between;
  padding: 12px 14px; border-bottom:1px solid #e5e7eb; background:#fff;
}
.dict-flyout-body {
  padding: 10px 12px 14px 12px; height: calc(100vh - 56px); overflow:auto; background:#fff;
}
.dict-close {
  border:1px solid #e5e7eb; background:#fff; border-radius:10px; padding:6px 10px; cursor:pointer;
}
.dict-close:hover { background:#f8fafc; }
</style>
"""

def _dd_inject_css_once():
    if "_dd_css" not in st.session_state:
        st.markdown(_DD_CSS, unsafe_allow_html=True)
        st.session_state["_dd_css"] = True

def _dd_inject_flyout_css_once(width_px: int = 420):
    key = f"_dd_flyout_css_{int(width_px)}"
    if key not in st.session_state:
        st.markdown(_FLYOUT_CSS_TEMPLATE % {"width_px": int(width_px)}, unsafe_allow_html=True)
        st.session_state[key] = True

# =============================================================================
#                            Domain dictionaries
# =============================================================================
GLOSSARY = {
    # Banking assets
    "FVTPL Investments": {
        "what": "Debt securities at fair value through P/L (trading/designated). Include only interest-bearing. Exclude equities, unit trusts and all derivatives.",
        "where": "BS/Notes: 'Financial assets at FVTPL' — use total; for IEA include only interest-bearing portion (StrictPlus).",
        "why": "Can be part of IEA when interest-bearing.",
    },
    "FVTPL Derivatives": {"what": "Derivative assets at fair value (trading/hedging).", "where": "BS: 'Derivative financial assets'.", "why": "Tracked but excluded from IEA and NIM denominator."},
    "FVOCI Investments": {"what": "Debt securities at fair value through OCI.", "where": "BS: 'Financial investments measured at FVOCI' (total).", "why": "Included in IEA (Strict & StrictPlus)."},
    "Amortised Cost Investments": {"what": "Debt securities at amortised cost (hold-to-collect).", "where": "BS: 'Financial investments measured at amortised cost' (total).", "why": "Included in IEA."},
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

# =============================================================================
#                         Builder: dictionary dataframe
# =============================================================================
def _norm(s: str) -> str:
    return re.sub(r"\s+"," ", (s or "").strip()).lower()

@st.cache_data(show_spinner=False)
def dd_build_dictionary_df() -> pd.DataFrame:
    rows = []
    # Raw fields from config
    for bucket, sections in (INDUSTRY_FORM_CATEGORIES or {}).items():
        for sec, fields in (sections or {}).items():
            for f in (fields or []):
                label = f.get("label","")
                key   = f.get("key","")
                unit  = f.get("unit") or ""
                ftype = f.get("type","float")
                helpx = f.get("help") or ""
                ttm   = _ttm_agg_for(sec, key, label) if callable(_ttm_agg_for) else ""
                g     = GLOSSARY.get(label) or GLOSSARY.get(key) or {}
                rows.append({
                    "Term": label,
                    "Key": key,
                    "Industry": bucket,
                    "Section": sec,
                    "Unit": unit,
                    "Type": ftype,
                    "TTM policy": ttm or "",
                    "What": g.get("what",""),
                    "Where": g.get("where", (helpx or f"Look for “{label}” in {sec}.")),
                    "Why": g.get("why",""),
                    "Formula": g.get("formula",""),
                    "Source": "Annual Form Field",
                })

    # Summary ratio blocks from config
    for bucket, groups in (INDUSTRY_SUMMARY_RATIOS_CATEGORIES or {}).items():
        for sec, items in (groups or {}).items():
            for canon, _ in (items or {}).items():
                rows.append({
                    "Term": canon,
                    "Key": "",
                    "Industry": bucket,
                    "Section": f"Summary — {sec}",
                    "Unit": "%",
                    "Type": "ratio",
                    "TTM policy": "derived",
                    "What": GLOSSARY.get(canon,{}).get("what",""),
                    "Where": "Derived by app (see formula).",
                    "Why": GLOSSARY.get(canon,{}).get("why",""),
                    "Formula": FORMULAS.get(canon, GLOSSARY.get(canon,{}).get("formula","")),
                    "Source": "Summary Ratio",
                })
    return pd.DataFrame(rows)

# =============================================================================
#                    Reusable panel (inline or flyout)
# =============================================================================
def _dd_render_panel(key_prefix: str):
    df = dd_build_dictionary_df()

    st.markdown("<div class='dict-card'>", unsafe_allow_html=True)
    st.markdown("**Investor Dictionary**", unsafe_allow_html=True)

    # Controls
    q = st.text_input("Search", key=f"{key_prefix}_q", placeholder="term / key / formula / why …")
    bkt_options = ["All"] + (sorted(df["Industry"].dropna().unique()) if not df.empty else [])
    sec_options = ["All"] + (sorted(df["Section"].dropna().unique()) if not df.empty else [])
    bkt = st.selectbox("Industry", bkt_options, index=0, key=f"{key_prefix}_b")
    sec = st.selectbox("Section",  sec_options, index=0, key=f"{key_prefix}_s")

    # Filter
    if df.empty:
        st.info("No rows from config. You can still use the Quick Guides below.")
        out = pd.DataFrame(columns=["Term","Key","Industry","Section","Unit","TTM policy","What","Where","Why","Formula"])
    else:
        mask = pd.Series(True, index=df.index)
        if q:
            ql = q.lower()
            mask &= df.apply(lambda r: any(ql in str(v).lower() for v in
                        [r["Term"], r["Key"], r["What"], r["Where"], r["Why"], r["Formula"], r["Industry"], r["Section"]]), axis=1)
        if bkt != "All": mask &= (df["Industry"] == bkt)
        if sec != "All": mask &= (df["Section"] == sec)
        out = df.loc[mask].sort_values(["Industry","Section","Term"]).head(500)
        out = out[["Term","Key","Industry","Section","Unit","TTM policy","What","Where","Why","Formula"]]

    # Actions: download CSV
    csv = out.to_csv(index=False).encode("utf-8")
    st.markdown(
        "<div class='dict-actions'>",
        unsafe_allow_html=True,
    )
    st.download_button(
        "⬇️ Download table (CSV)",
        data=csv,
        file_name="investor_dictionary.csv",
        mime="text/csv",
        use_container_width=False,
        key=f"{key_prefix}_dl",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.dataframe(out, hide_index=True, use_container_width=True)
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

# Right-side flyout (to reuse on any page)
def dd_dictionary_flyout(*, key_prefix: str = "dict", width_px: int = 420,
                         button_label: str = "📚 Dictionary", open_default: bool = False,
                         show_pin: bool = True) -> None:
    _dd_inject_css_once()
    _dd_inject_flyout_css_once(width_px)

    open_key = f"{key_prefix}_flyout_open"
    pin_key  = f"{key_prefix}_flyout_pin"
    if open_key not in st.session_state:
        st.session_state[open_key] = open_default
    if pin_key not in st.session_state:
        st.session_state[pin_key] = False

    cols = st.columns([1,1,1,1,1,1,1,1,1,1,1,1])
    with cols[-1]:
        if st.button(button_label, key=f"{key_prefix}_flyout_btn", use_container_width=True):
            st.session_state[open_key] = True
            st.rerun()

    if st.session_state[open_key] or st.session_state[pin_key]:
        @st.dialog("Investor Dictionary")
        def _show_flyout():
            st.markdown(
                "<div class='dict-flyout-head'>"
                "<div style='font-weight:800;font-size:1rem'>Investor Dictionary</div>"
                f"<div style='display:flex;gap:8px;align-items:center'></div></div>",
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
        _show_flyout()

# =============================================================================
#                                 PAGE RENDER
# =============================================================================
# Configure the page and render inline dictionary
setup_page("Dictionary", page_id="12_Dictionary")
_dd_inject_css_once()
render_page_title("Dictionary")
st.markdown(section("Investor Dictionary", "Glossary • formulas • quick guides", "info"), unsafe_allow_html=True)

# Optional top-right flyout button even on this page (handy on mobile)
dd_dictionary_flyout(key_prefix="dictpage", width_px=440, button_label="Open as Flyout", open_default=False)

# Main inline panel
_dd_render_panel("dictpage")
