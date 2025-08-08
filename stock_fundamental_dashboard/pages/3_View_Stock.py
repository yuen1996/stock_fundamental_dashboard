# --- make project root importable so we can import io_helpers/calculations/rules ---
import os, sys
_THIS = os.path.dirname(__file__)
_PARENT = os.path.abspath(os.path.join(_THIS, ".."))         # .../stock_fundamental_dashboard
_GRANDP = os.path.abspath(os.path.join(_THIS, "..", ".."))   # repo root (one level above)

for p in (_PARENT, _GRANDP):
    if p not in sys.path:
        sys.path.insert(0, p)

# now these imports will work whether files are in the package root or repo root
try:
    import io_helpers, calculations, rules
except ModuleNotFoundError:
    from utils import io_helpers, calculations, rules  # fallback if you move them under utils/


# --- path patch so this page can import from project root ---
import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))  # project root (parent of /pages)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import re


# --- robust imports: prefer package (utils), fall back to top-level ---
try:
    from utils import io_helpers, calculations, rules
except Exception:
    import io_helpers
    import calculations
    import rules

# ------- Drag-and-drop imports (prefer stylable fallback) -------
DRAG_LIB = None
try:
    from st_draggable_list import DraggableList  # pip install st-draggable-list
    DRAG_LIB = "draggable-list"
except Exception:
    try:
        from streamlit_sortables import sort_items  # pip install streamlit-sortables
        DRAG_LIB = "sortables"
    except Exception:
        DRAG_LIB = None

# ---------- Force Wide Layout on Page ----------
st.set_page_config(layout="wide")

# ---------- Unified Page CSS (same as Dashboard) ----------
BASE_CSS = """
<style>
:root{
  --bg:#f6f7fb;               /* app background */
  --surface:#ffffff;          /* cards/tables background */
  --text:#0f172a;             /* main text */
  --muted:#475569;            /* secondary text */
  --border:#e5e7eb;           /* card & table borders */
  --shadow:0 8px 24px rgba(15, 23, 42, .06);

  /* accent colors for section stripes */
  --primary:#4f46e5;          /* indigo */
  --info:#0ea5e9;             /* sky   */
  --success:#10b981;          /* green */
  --warning:#f59e0b;          /* amber */
  --danger:#ef4444;           /* red   */
}
html, body, [class*="css"]{
  font-size:16px !important; color:var(--text);
}
.stApp{
  background: radial-gradient(1000px 500px at 10% -10%, #f0f3fb 0%, var(--bg) 60%), var(--bg);
}
h1, h2, h3, h4{
  color:var(--text) !important; font-weight:800 !important; letter-spacing:.2px;
}

/* Section header card (visual separators) */
.sec{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:14px;
  box-shadow:var(--shadow);
  padding:.65rem .9rem;
  margin:1rem 0 .6rem 0;
  display:flex; align-items:center; gap:.6rem;
}
.sec .t{ font-size:1.05rem; font-weight:800; margin:0; padding:0; }
.sec .d{ color:var(--muted); font-size:.95rem; margin-left:.25rem; }
.sec::before{
  content:""; display:inline-block;
  width:8px; height:26px; border-radius:6px; background:var(--primary);
}
.sec.info::before    { background:var(--info); }
.sec.success::before { background:var(--success); }
.sec.warning::before { background:var(--warning); }
.sec.danger::before  { background:var(--danger); }

/* Tables / editors */
.stDataFrame, .stDataEditor{ font-size:15px !important; }
div[data-testid="stDataFrame"] table{ border-collapse:separate !important; border-spacing:0; }
div[data-testid="stDataFrame"] table tbody tr:hover td{ background:#f8fafc !important; }
div[data-testid="stDataFrame"] td{ border-bottom:1px solid var(--border) !important; }

/* → enable horizontal scrolling on ANY dataframe */
div[data-testid="stDataFrame"] {
  overflow-x: auto;
}

/* → collapse the grid into single borders, like Raw Data */
div[data-testid="stDataFrame"] table {
  border-collapse: collapse !important;
  width: max-content;      /* allow it to be wider than the container */
  min-width: 100%;         /* but at least fill the width */
}

/* → draw 1px solid borders on every cell */
div[data-testid="stDataFrame"] table th,
div[data-testid="stDataFrame"] table td {
  border: 1px solid var(--border) !important;
}

/* → keep the header row pinned when you scroll */
div[data-testid="stDataFrame"] table thead th {
  position: sticky !important;
  top: 0 !important;
  background: #f9fafb !important;
  z-index: 10 !important;
}


/* Inputs */
div[data-baseweb="input"] input, textarea, .stNumberInput input{ font-size:15px !important; }
.stSlider > div [data-baseweb="slider"]{ margin-top:.25rem; }

/* Buttons */
.stButton>button{ border-radius:12px !important; padding:.55rem 1.1rem !important; font-weight:700; }

/* Tabs */
.stTabs [role="tab"]{ font-size:15px !important; font-weight:600 !important; }

/* Sidebar theme (dark, same as Dashboard) */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg, #0b1220 0%, #1f2937 100%) !important;
}
[data-testid="stSidebar"] *{ color:#e5e7eb !important; }
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

# Extra CSS for the drag chips (kept minimal & matching Dashboard scale)
DRAG_CSS = """
<style>
.sort-help { color:#475569; font-size:.90rem; margin:.15rem 0 .25rem 0; }
.sort-box {
  border:1px dashed #cbd5e1; background:#f8fafc; border-radius:12px;
  padding:.15rem .30rem; max-height:90px; overflow:auto;
}
.sort-box [draggable="true"]{
  display:inline-flex !important; width:auto !important; max-width:100% !important;
  border-radius:9999px !important; background:#eef2ff !important; border:1px solid #c7d2fe !important;
  color:#0f172a !important; font-size:10px !important; padding:1px 6px !important; line-height:1.15 !important; margin:2px !important;
}
.sort-box > div{ display:flex; flex-wrap:wrap; gap:4px; align-items:center; }
.sdl-item, .sdl-item *{
  display:inline-flex !important; width:auto !important; max-width:100% !important;
  border-radius:9999px !important; background:#eef2ff !important; color:#0f172a !important; border:1px solid #c7d2fe !important;
  font-size:10px !important; padding:1px 6px !important; margin:2px !important;
}
.sdl-wrapper, .sdl-container, .sdl-list{ display:flex !important; flex-wrap:wrap !important; gap:4px !important; }
</style>
"""
st.markdown(DRAG_CSS, unsafe_allow_html=True)

st.header("🔍 View Stock")

df = io_helpers.load_data()
if df is None or df.empty or "Name" not in df.columns:
    st.warning("No data.")
    st.stop()

# Ensure compatibility
if "IsQuarter" not in df.columns:
    df["IsQuarter"] = False
if "Quarter" not in df.columns:
    df["Quarter"] = pd.NA

stocks = sorted([s for s in df["Name"].dropna().unique()])

# ---------- Field definitions ----------
ANNUAL_SECTIONS = [
    ("Income Statement", [
        ("Net Profit", "NetProfit"),
        ("Gross Profit", "GrossProfit"),
        ("Revenue", "Revenue"),
        ("Cost Of Sales", "CostOfSales"),
        ("Finance Costs", "FinanceCosts"),
        ("Administrative Expenses", "AdminExpenses"),
        ("Selling & Distribution Expenses", "SellDistExpenses"),
    ]),
    ("Balance Sheet", [
        ("Number of Shares", "NumShares"),
        ("Current Asset", "CurrentAsset"),
        ("Other Receivables", "OtherReceivables"),
        ("Trade Receivables", "TradeReceivables"),
        ("Biological Assets", "BiologicalAssets"),
        ("Inventories", "Inventories"),
        ("Prepaid Expenses", "PrepaidExpenses"),
        ("Intangible Asset", "IntangibleAsset"),
        ("Current Liability", "CurrentLiability"),
        ("Total Asset", "TotalAsset"),
        ("Total Liability", "TotalLiability"),
        ("Shareholder Equity", "ShareholderEquity"),
        ("Reserves", "Reserves"),
        ("Cash & Cash Equivalents", "Cash"),
        ("Total Debt / Borrowings", "TotalDebt"),
    ]),
    ("Other Data", [
        ("Dividend pay cent", "Dividend"),
        ("End of year share price", "SharePrice"),
    ]),
    # NEW: show annual cash-flow inputs
    ("Cash Flow Statement", [
        ("Operating Cash Flow", "CFO"),
        ("Capital Expenditure", "CapEx"),
        ("Income Tax Expense", "IncomeTax"),
        ("Depreciation & Amortization", "DepAmort"),
        ("EBITDA (optional)", "EBITDA"),
    ]),
]



QUARTERLY_SECTIONS = [
    ("Quarterly Income Statement", [
        ("Quarterly Net Profit", "Q_NetProfit"),
        ("Quarterly Gross Profit", "Q_GrossProfit"),
        ("Quarterly Revenue", "Q_Revenue"),
        ("Quarterly Cost Of Sales", "Q_CostOfSales"),
        ("Quarterly Finance Costs", "Q_FinanceCosts"),
        ("Quarterly Administrative Expenses", "Q_AdminExpenses"),
        ("Quarterly Selling & Distribution Expenses", "Q_SellDistExpenses"),
    ]),
    ("Quarterly Balance Sheet", [
        ("Number of Shares", "Q_NumShares"),
        ("Current Asset", "Q_CurrentAsset"),
        ("Other Receivables", "Q_OtherReceivables"),
        ("Trade Receivables", "Q_TradeReceivables"),
        ("Biological Assets", "Q_BiologicalAssets"),
        ("Inventories", "Q_Inventories"),
        ("Prepaid Expenses", "Q_PrepaidExpenses"),
        ("Intangible Asset", "Q_IntangibleAsset"),
        ("Current Liability", "Q_CurrentLiability"),
        ("Total Asset", "Q_TotalAsset"),
        ("Total Liability", "Q_TotalLiability"),
        ("Shareholder Equity", "Q_ShareholderEquity"),
        ("Reserves", "Q_Reserves"),
    ]),

    ("Quarterly Cash Flow Statement", [
        ("Quarterly Operating Cash Flow", "Q_CFO"),
        ("Quarterly Capital Expenditure", "Q_CapEx"),
        ("Quarterly Income Tax", "Q_Tax"),
        ("Quarterly Depreciation & Amortization", "Q_DepAmort"),
        ("Quarterly EBITDA (optional)", "Q_EBITDA"),
    ]),


    ("Quarterly Other Data", [
    ("Each end per every quarter price", "Q_EndQuarterPrice"),
]),

]

# ---------- Helpers ----------
def _to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.floating, np.integer)):
        return float(x)
    try:
        s = str(x).replace(",", "").strip()
        if s == "":
            return np.nan
        return float(s)
    except Exception:
        return np.nan

def _format_4(x):
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):,.4f}"
    except Exception:
        return ""

def _reorder_empty_last(df_numeric: pd.DataFrame) -> pd.DataFrame:
    if df_numeric.empty:
        return df_numeric
    mask = df_numeric.isna().all(axis=1)
    return pd.concat([df_numeric[~mask], df_numeric[mask]], axis=0)

def quarter_key_to_num(q):
    if pd.isna(q):
        return np.nan
    s = str(q).upper().strip()
    m = re.search(r"(\d+)", s)
    if not m:
        return np.nan
    try:
        n = int(m.group(1))
        return n if 1 <= n <= 4 else np.nan
    except Exception:
        return np.nan

def nice_label(col):
    if isinstance(col, tuple) and len(col) == 2:
        return f"{col[0]} • {col[1]}"
    return str(col)

# (Kept for fallback visual only)
PILL_STYLE = {
    "list": {"backgroundColor": "transparent", "display": "flex", "flexWrap": "wrap", "gap": "8px", "padding": "4px"},
    "item": {"backgroundColor": "#eef2ff", "borderRadius": "9999px", "padding": "4px 10px",
             "border": "1px solid #c7d2fe", "color": "#1e293b", "fontSize": "13px"},
    "itemLabel": {"color": "#1e293b"},
}

def _drag_labels(labels, key_suffix):
    if DRAG_LIB == "sortables":
        from streamlit_sortables import sort_items
        with st.container():
            st.markdown('<div class="sort-box">', unsafe_allow_html=True)
            new_labels = sort_items(items=labels, key=f"sort_{key_suffix}")
            st.markdown('</div>', unsafe_allow_html=True)
        return new_labels if isinstance(new_labels, list) else labels

    if DRAG_LIB == "draggable-list":
        from st_draggable_list import DraggableList
        data = [{"id": str(i), "order": i, "name": lab} for i, lab in enumerate(labels)]
        with st.container():
            st.markdown('<div class="sort-box">', unsafe_allow_html=True)
            result = DraggableList(data, key=f"drag_{key_suffix}")
            st.markdown('</div>', unsafe_allow_html=True)
        if isinstance(result, list):
            if result and isinstance(result[0], dict) and "name" in result[0]:
                return [d["name"] for d in result]
            return [str(x) for x in result]
        return labels

    st.info("Install a drag component to enable reordering: `pip install streamlit-sortables` "
            "or `pip install streamlit-draggable-list`.")
    return labels

def drag_reorder(columns, key_suffix, help_text="Drag to set column order. Left-most = first."):
    labels = [nice_label(c) for c in columns]
    state_key = f"reorder_saved_{key_suffix}"
    saved = st.session_state.get(state_key)
    if isinstance(saved, list):
        base = [l for l in saved if l in labels] + [l for l in labels if l not in saved]
    else:
        base = labels

    show = st.checkbox("🔧 Reorder (drag) — compact", value=False, key=f"reorder_toggle_{key_suffix}")
    if show:
        st.markdown(f'<div class="sort-help">{help_text}</div>', unsafe_allow_html=True)
        current_order = _drag_labels(base, key_suffix)
        if isinstance(current_order, list) and current_order:
            st.session_state[state_key] = current_order
    else:
        current_order = st.session_state.get(state_key, base)

    label_to_col = {nice_label(c): c for c in columns}
    ordered_cols = [label_to_col[l] for l in current_order if l in label_to_col]
    for c in columns:
        if c not in ordered_cols:
            ordered_cols.append(c)
    return ordered_cols

# ---------- Ratio thresholds + styling (green / blue / red with tooltips) ----------
THRESHOLDS = {
    "ROE (%)":                 ("ge", 15,   10,   "Higher is better"),
    "Debt-Asset Ratio (%)":    ("le", 40,   60,   "Lower is better"),
    "P/E":                     ("le", 15,   20,   "Lower is better"),
    "P/B":                     ("le", 2.0,  3.0,  "Lower is better"),
    "Gross Profit Margin (%)": ("ge", 30,   20,   "Higher is better"),
    "Net Profit Margin (%)":   ("ge", 15,   8,    "Higher is better"),
    "Current Ratio":           ("ge", 2.0,  1.5,  "Higher is better"),
    "Quick Ratio":             ("ge", 1.5,  1.0,  "Higher is better"),
    "Dividend Yield (%)":      ("ge", 5.0,  3.0,  "Higher is better"),
    "Three Fees Ratio (%)":    ("le", 30,   40,   "Lower is better"),
    "Total Cost %":            ("le", 70,   80,   "Lower is better"),
    "EPS":                     ("ge", 0.0,  0.0,  "Positive is better"),
    "NTA per share":           ("ge", 0.0,  0.0,  "Positive is better"),
    "BVPS":                    ("ge", 0.0,  0.0,  "Positive is better"),
}
COLORS = {"good": "#16a34a22", "ok": "#3b82f622", "bad": "#ef444422"}

def _rate(value, rule):
    if rule is None or value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None, ""
    direction, great, ok, note = rule
    try:
        v = float(value)
    except Exception:
        return None, ""
    if direction == "ge":
        state = "good" if v >= great else ("ok" if v >= ok else "bad")
        expl = f"value {v:,.4f} → {'≥' if state!='bad' else '<'} thresholds; {note}"
    else:
        state = "good" if v <= great else ("ok" if v <= ok else "bad")
        expl = f"value {v:,.4f} → {'≤' if state!='bad' else '>'} thresholds; {note}"
    return state, expl

def style_ratio_table(df_in: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """
    Apply green/blue/red background based on THRESHOLDS whether metrics are
    in the columns *or* in the index (works for both orientations).
    Also builds tooltips explaining the rating.
    """
    df = df_in.copy()
    df_num = df.apply(pd.to_numeric, errors="coerce")

    # Which axis has the metric names?
    cols_match = [c for c in df.columns if c in THRESHOLDS]
    idx_match  = [i for i in df.index   if i in THRESHOLDS]

    sty = df.style.format(lambda x: "" if pd.isna(x) else f"{x:,.4f}")

    if len(cols_match) >= len(idx_match):
        # ---- Metrics are in COLUMNS (our default layout)
        for col in df.columns:
            rule = THRESHOLDS.get(col)
            if not rule:
                continue
            def _col_styles(s, _rule=rule):
                out = []
                for v in s.values:
                    state, _ = _rate(v, _rule)
                    out.append("" if not state else f"background-color: {COLORS[state]};")
                return out
            sty = sty.apply(_col_styles, subset=[col], axis=0)

            # tooltips for this column
            tips = []
            for v in df_num[col].values:
                state, expl = _rate(v, rule)
                direction, great, ok, note = rule
                bound = f"≥ {great}" if direction == "ge" else f"≤ {great}"
                mid   = f"≥ {ok}"    if direction == "ge" else f"≤ {ok}"
                try:
                    vtxt = "–" if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else f"{float(v):,.4f}"
                except Exception:
                    vtxt = str(v)
                tips.append(f"{col}: {vtxt} → {'' if not state else state.upper()} (good {bound}, OK {mid}). {note}")
            sty = sty.set_tooltips(pd.DataFrame({col: tips}, index=df.index))

    else:
        # ---- Metrics are in INDEX (the transposed layout)
        for idx in df.index:
            rule = THRESHOLDS.get(idx)
            if not rule:
                continue
            def _row_styles(s, _rule=rule):
                out = []
                for v in s.values:
                    state, _ = _rate(v, _rule)
                    out.append("" if not state else f"background-color: {COLORS[state]};")
                return out
            sty = sty.apply(_row_styles, subset=pd.IndexSlice[idx, :], axis=1)

            # tooltips for this row
            tips = []
            for v in df_num.loc[idx, :].values:
                state, expl = _rate(v, rule)
                direction, great, ok, note = rule
                bound = f"≥ {great}" if direction == "ge" else f"≤ {great}"
                mid   = f"≥ {ok}"    if direction == "ge" else f"≤ {ok}"
                try:
                    vtxt = "–" if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else f"{float(v):,.4f}"
                except Exception:
                    vtxt = str(v)
                tips.append(f"{idx}: {vtxt} → {'' if not state else state.upper()} (good {bound}, OK {mid}). {note}")
            sty = sty.set_tooltips(pd.DataFrame([tips], index=[idx], columns=df.columns))

    return sty

def _show_styled(styler, height=420):
    """
    Render a pandas Styler in-place so that:
      • it fills 100% of the Streamlit column width,
      • the table’s min-width is 100% (and can overflow horizontally via scroll),
      • the header row and first column stay pinned,
      • your THRESHOLDS background-colours and tooltips all still work.
    """
    try:
        html_table = styler.to_html()
    except Exception:
        st.dataframe(getattr(styler, "data", styler),
                     use_container_width=True,
                     height=height)
        return

    wrapper = f"""
<style>
  :root {{
    --border: #e5e7eb;
    --hover: #f8fafc;
    --shadow: 0 8px 24px rgba(15,23,42,.06);
  }}
  .wrap {{
    border:1px solid var(--border);
    border-radius:12px;
    box-shadow:var(--shadow);
    background:#fff;
    overflow:hidden;
    width:100%;
  }}
  .scroll {{
    overflow-x:auto;
    overflow-y:auto;
    max-height:{int(height)}px;
  }}
  .wrap table {{
    border-collapse: collapse;
    width: max-content;
    min-width: 100%;
  }}
  .wrap thead th {{
    position:sticky; top:0; z-index:5;
    background:#f9fafb;
    border-bottom:1px solid var(--border);
  }}
  .wrap th, .wrap td {{
    border:1px solid var(--border);
    padding:8px 10px;
    white-space:nowrap;
  }}
  .wrap tbody tr:hover td,
  .wrap tbody tr:hover th {{
    background:var(--hover);
  }}
  .wrap tbody tr:nth-child(2n) td,
  .wrap tbody tr:nth-child(2n) th {{
    background:#fcfcfd;
  }}
  .wrap tbody th {{
    position:sticky; left:0; z-index:4;
    background:#fff;
    border-right:1px solid var(--border);
  }}
  .wrap thead th:first-child {{
    position:sticky; left:0; z-index:6;
    background:#f9fafb;
    border-right:1px solid var(--border);
  }}
  .pd-styler {{
    width: max-content;
  }}
</style>
<div class="wrap"><div class="scroll">{html_table}</div></div>
"""
    st.markdown(wrapper, unsafe_allow_html=True)






# ---------- Builders ----------
def build_annual_raw_numeric(annual_df: pd.DataFrame) -> pd.DataFrame:
    if annual_df.empty:
        return pd.DataFrame()
    years = sorted([int(y) for y in annual_df["Year"].dropna().unique()])
    rows = []
    for sec, items in ANNUAL_SECTIONS:
        for label, key in items:
            rows.append((sec, label, key))
    idx = pd.MultiIndex.from_tuples([(r[0], r[1]) for r in rows], names=["Section", "Field"])
    out = pd.DataFrame(index=idx, columns=[str(y) for y in years], dtype=float)
    ann_by_year = {int(r["Year"]): r for _, r in annual_df.iterrows()}
    for (sec, label, key), (i_sec, i_field) in zip(rows, out.index):
        for y in years:
            val = np.nan
            row = ann_by_year.get(y)
            if row is not None and key in row:
                val = _to_float(row[key])
            out.loc[(i_sec, i_field), str(y)] = val
    return out

def build_quarter_raw_numeric(quarter_df: pd.DataFrame) -> pd.DataFrame:
    if quarter_df.empty:
        return pd.DataFrame()
    q = quarter_df.copy()
    q["Qnum"] = q["Quarter"].map(quarter_key_to_num)
    q = q.dropna(subset=["Year", "Qnum"])
    q["Year"] = q["Year"].astype(int)
    q = q.sort_values(["Year", "Qnum"])
    periods = [f"{int(r['Year'])} Q{int(r['Qnum'])}" for _, r in q.iterrows()]
    seen, cols, row_by_period = set(), [], {}
    for period, (_, r) in zip(periods, q.iterrows()):
        if period in seen:
            continue
        seen.add(period)
        cols.append(period)
        row_by_period[period] = r
    rows = []
    for sec, items in QUARTERLY_SECTIONS:
        for label, key in items:
            rows.append((sec, label, key))
    idx = pd.MultiIndex.from_tuples([(r[0], r[1]) for r in rows], names=["Section", "Field"])
    out = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for (sec, label, key), (i_sec, i_field) in zip(rows, out.index):
        for c in cols:
            row = row_by_period.get(c)
            val = np.nan
            if row is not None and key in row:
                val = _to_float(row[key])
            out.loc[(i_sec, i_field), c] = val
    return out

# ---------- Chart helpers ----------
def field_options(sections):
    opts = []
    for sec, items in sections:
        for lbl, _ in items:
            opts.append((f"{sec} • {lbl}", (sec, lbl)))
    return opts

def plot_single_series(x_values, y_values, title, yname, height=320):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines+markers", name=yname))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), title=title, xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

def multi_panel_charts(count, options, x_labels, series_getter, key_prefix, chart_height=320):
    count = max(1, min(4, int(count)))
    option_labels = [o[0] for o in options]
    if not option_labels:
        st.info("No series available to chart.")
        return
    row1 = st.columns(2)
    row2 = st.columns(2) if count > 2 else (None, None)

    def render_cell(col_container, i):
        with col_container:
            default_idx = i if i < len(option_labels) else 0
            sel = st.selectbox(
                f"Chart {i+1} – pick a series",
                options=option_labels,
                index=default_idx,
                key=f"{key_prefix}_sel_{i}",
            )
            payload = dict(options)[sel]
            y = series_getter(payload)
            if y is None or (pd.isna(pd.Series(y)).all()):
                st.info("No data for this selection.")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_labels, y=y, mode="lines+markers", name=sel))
                fig.update_layout(height=chart_height, margin=dict(l=10, r=10, t=40, b=10), title=sel, xaxis_title="", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

    render_cell(row1[0], 0)
    if count >= 2: render_cell(row1[1], 1)
    if count >= 3 and row2[0] is not None: render_cell(row2[0], 2)
    if count >= 4 and row2[1] is not None: render_cell(row2[1], 3)

# ---------- UI ----------
for stock_name in stocks:
    with st.expander(stock_name, expanded=False):
        stock = df[df["Name"] == stock_name].sort_values(["Year"])
        annual = stock[stock["IsQuarter"] != True].copy()
        quarterly = stock[stock["IsQuarter"] == True].copy()

        # ---- Current Price metric (STRICT: CurrentPrice only)
        cur_val = 0.0
        if "CurrentPrice" in stock.columns:
            s = stock["CurrentPrice"].dropna()
            if not s.empty:
                cur_val = float(s.iloc[-1])
                st.metric("Current Price", f"{cur_val:,.4f}")

        # Quick price debug (temporary) — stays inside the expander
        if st.checkbox("Debug: show chosen price", key=f"price_dbg_{stock_name}"):
            st.write("Chosen price for TTM multiples:", cur_val)

        # ---- Section header before the tabs
        st.markdown(
            '<div class="sec"><div class="t">📚 Financial Reports</div>'
            '<div class="d">Annual & Quarterly data, ratios & charts</div></div>',
            unsafe_allow_html=True
        )

        tabs = st.tabs(["Annual Report", "Quarterly Report"])

        # =========================
        # ANNUAL
        # =========================
        with tabs[0]:
            st.subheader(f"{stock_name} - Annual Financial Data")

            # ---- Annual — Raw Data
            st.markdown('<div class="sec info"><div class="t">🧾 Annual — Raw Data</div></div>', unsafe_allow_html=True)
            st.markdown("#### Raw Data")
            ann_numeric = build_annual_raw_numeric(annual)
            ann_raw_layout = st.radio(
                "Raw data layout (annual)",
                ["Fields → columns (Year rows)", "Years → columns (Field rows)"],
                horizontal=True,
                key=f"annual_raw_layout_{stock_name}"
            )

            if ann_numeric.empty:
                st.info("No annual raw data available.")
            else:
                if ann_raw_layout.startswith("Years"):
                    disp_num = _reorder_empty_last(ann_numeric)
                    new_cols = drag_reorder(
                        disp_num.columns.tolist(),
                        key_suffix=f"ann_raw_yearcols_{stock_name}",
                        help_text="Drag to reorder **Year** columns."
                    )
                    disp_num = disp_num[new_cols]
                    st.session_state[f"ann_raw_x_{stock_name}"] = [str(x) for x in new_cols]
                else:
                    disp_num = _reorder_empty_last(ann_numeric.T)
                    field_cols = disp_num.columns.tolist()
                    new_cols = drag_reorder(
                        field_cols,
                        key_suffix=f"ann_raw_fieldcols_{stock_name}",
                        help_text="Drag to reorder **Field** columns."
                    )
                    disp_num = disp_num[new_cols]
                    st.session_state[f"ann_raw_x_{stock_name}"] = [str(x) for x in disp_num.index.tolist()]

                disp_fmt = disp_num.applymap(_format_4)
                if isinstance(disp_fmt.columns, pd.MultiIndex):
                    disp_fmt.columns = pd.Index([nice_label(c) for c in disp_fmt.columns])
                st.dataframe(disp_fmt, use_container_width=True, height=420)

                    
                        # ---- TTM (last 4 quarters) — shown ABOVE CAGR
            ttm = calculations.compute_ttm(stock, current_price=cur_val)

            # === TTM deep debug (temporary) ===
            # Turn this on in the UI to see what columns are found, last 4 values, and the computed TTM pieces.
            if st.checkbox("TTM deep debug", key=f"ttm_deep_dbg_{stock_name}"):
                import pandas as pd, numpy as np
                revenue_candidates = ["Quarterly Revenue","Q_Revenue","Revenue","Sales","Q_Sales","Q_TotalRevenue"]
                np_candidates      = ["Quarterly Net Profit","Q_NetProfit","Q_Profit","Q_NetIncome","NetProfit","NetIncome"]
                eps_candidates     = ["Quarterly EPS","Q_EPS","EPS","Basic EPS","Diluted EPS","EPS (Basic)","EPS (Diluted)"]
                shares_candidates  = ["Number of Shares","SharesOutstanding","ShareOutstanding","ShareCount",
                                      "BasicShares","NumShares","Number of shares","Q_NumShares"]
                price_candidates   = ["CurrentPrice","Q_EndQuarterPrice","Q_SharePrice","Each end per every quarter price","Price"]

                def first_present(cands):
                    return next((c for c in cands if c in stock.columns), None)

                rev_col  = first_present(revenue_candidates)
                np_col   = first_present(np_candidates)
                eps_col  = first_present(eps_candidates)
                sh_col   = first_present(shares_candidates)
                p_col    = first_present(price_candidates)

                st.write("Resolved columns by presence:", {
                    "Revenue": rev_col, "NetProfit": np_col, "EPS": eps_col, "Shares": sh_col, "Any price col": p_col
                })

                q = stock[stock.get("IsQuarter", False) == True].copy()
                cols_to_show = ["Year", "Quarter"] + [c for c in [rev_col, np_col, eps_col, sh_col, p_col] if c]
                st.write("Quarterly (tail 6):")
                st.dataframe(q[cols_to_show].tail(6), use_container_width=True)

                def to_num(s):
                    return calculations._to_num(s) if s is not None else pd.Series(dtype="float64")

                rev_ttm = to_num(q.get(rev_col)).tail(4).sum() if rev_col else None
                np_ttm  = to_num(q.get(np_col)).tail(4).sum()  if np_col  else None
                sh4     = to_num(q.get(sh_col)).tail(4).dropna() if sh_col else pd.Series(dtype="float64")
                sh_avg  = sh4.mean() if not sh4.empty else None
                eps_sum = to_num(q.get(eps_col)).tail(4).sum() if eps_col else None

                eps_fallback = (eps_sum is None) or (isinstance(eps_sum, (int,float,np.floating)) and float(eps_sum) == 0.0)
                eps_ttm_dbg = eps_sum
                if eps_fallback and (np_ttm is not None) and (sh_avg and sh_avg > 0):
                    eps_ttm_dbg = float(np_ttm) / float(sh_avg)

                st.write({
                    "DEBUG rev_ttm(sum last 4)": rev_ttm,
                    "DEBUG np_ttm(sum last 4)":  np_ttm,
                    "DEBUG shares last4":        sh4.tolist() if not sh4.empty else [],
                    "DEBUG shares_avg":          sh_avg,
                    "DEBUG eps_ttm (sum or fallback)": eps_ttm_dbg,
                    "Chosen CurrentPrice":       cur_val,
                    "compute_ttm() output":      ttm
                })

            if ttm:
                st.markdown(
                    '<div class="sec success"><div class="t">🕒 TTM (Trailing 12 Months)</div>'
                    '<div class="d">Sums of the latest 4 quarters; ratios derived from those totals</div></div>',
                    unsafe_allow_html=True
                )
                c1, c2, c3, c4 = st.columns(4)

                def _fmt(v, d=2):
                    try:
                        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                            return "–"
                        return f"{float(v):,.{d}f}"
                    except Exception:
                        return "–"

                with c1:
                    st.metric("TTM Revenue", _fmt(ttm.get("TTM Revenue"), 0))
                    st.metric("TTM Net Profit", _fmt(ttm.get("TTM Net Profit"), 0))
                with c2:
                    gm = ttm.get("TTM Gross Margin (%)")
                    nm = ttm.get("TTM Net Margin (%)")
                    st.metric("TTM Gross Margin", f"{_fmt(gm, 2)}%" if gm is not None else "–")
                    st.metric("TTM Net Margin", f"{_fmt(nm, 2)}%" if nm is not None else "–")
                with c3:
                    st.metric("TTM EPS", _fmt(ttm.get("TTM EPS"), 4))
                    st.metric("P/E (TTM)", _fmt(ttm.get("P/E (TTM)"), 2))
                with c4:
                    st.metric("P/S (TTM)", _fmt(ttm.get("P/S (TTM)"), 2))
                    st.metric("EV/EBITDA (TTM)", _fmt(ttm.get("EV/EBITDA (TTM)"), 2))

                      # ---- 💧 Cash Flow Wealth (clear, separate section)
            st.markdown(
                '<div class="sec warning"><div class="t">💧 Cash Flow Wealth</div>'
                '<div class="d">Cash generation, reinvestment, and balance-sheet strength</div></div>',
                unsafe_allow_html=True
            )

            # Toggle instead of nested expander (nested expanders are not allowed)
            show_cf = st.checkbox("Show Cash Flow Wealth details", value=True, key=f"cfw_{stock_name}")
            if show_cf:
                # Small helper for formatting in this section
                def _fmt2(v, d=2):
                    try:
                        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                            return "–"
                        return f"{float(v):,.{d}f}"
                    except Exception:
                        return "–"

                r1c1, r1c2, r1c3, r1c4 = st.columns(4)
                with r1c1:
                    st.metric("TTM CFO", _fmt2(ttm.get("TTM CFO"), 0))
                    st.metric("Cash Conversion (CFO/NP)", f"{_fmt2(ttm.get('Cash Conversion (CFO/NP, %)'), 2)}%")
                with r1c2:
                    st.metric("TTM CapEx", _fmt2(ttm.get("TTM CapEx"), 0))
                    st.metric("Debt / FCF (yrs)", _fmt2(ttm.get("Debt / FCF (yrs)"), 2))
                with r1c3:
                    st.metric("TTM FCF", _fmt2(ttm.get("TTM FCF"), 0))
                    st.metric("FCF / Share (TTM)", _fmt2(ttm.get("FCF / Share (TTM)"), 4))
                with r1c4:
                    st.metric("FCF Margin (TTM)", f"{_fmt2((ttm.get('TTM FCF') or 0) / (ttm.get('TTM Revenue') or 1) * 100.0, 2)}%")
                    st.metric("FCF Yield (TTM)", f"{_fmt2(ttm.get('FCF Yield (TTM) (%)'), 2)}%")

                r2c1, r2c2, r2c3, r2c4 = st.columns(4)
                with r2c1:
                    st.metric("Net Cash (Debt)", _fmt2(ttm.get("Net Cash (Debt)"), 0))
                with r2c2:
                    st.metric("Net Cash / MC", f"{_fmt2(ttm.get('Net Cash / MC (%)'), 2)}%")
                with r2c3:
                    st.metric("Interest Coverage", _fmt2(ttm.get("Interest Coverage (EBITDA/Fin)"), 2))
                with r2c4:
                    st.metric("Market Cap (est.)", _fmt2((ttm.get("P/S (TTM)") or 0) * (ttm.get("TTM Revenue") or 0), 0))


            # === Snowflake & Momentum (inside the same Annual tab) ===

                        # === Momentum inputs from uploaded price file (trading-view CSV) ===
            ohlc_latest = None
            try:
                # load_ohlc reads /data/ohlc/<Name>.csv (set by the Momentum page)
                try:
                    import io_helpers as _ioh
                except Exception:
                    from utils import io_helpers as _ioh

                oh = _ioh.load_ohlc(stock_name)
                if oh is not None and not oh.empty:
                    df = oh.copy()
                    if "Date" in df.columns:
                        df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
                    df["Close"] = pd.to_numeric(df.get("Close"), errors="coerce")
                    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

                    if not df.empty:
                        close = df["Close"]
                        price = float(close.iloc[-1])

                        # 200-DMA needs >=200 trading rows
                        ma200 = float(close.rolling(200, min_periods=200).mean().iloc[-1]) if len(close) >= 200 else None

                        # 12-month return: prefer ~252 trading days; fallback to 365-day window
                        ret_12m = None
                        if len(close) >= 252 and pd.notna(close.iloc[-252]) and float(close.iloc[-252]) != 0.0:
                            base = float(close.iloc[-252])
                            ret_12m = float(price / base - 1.0)
                        else:
                            cutoff = df["Date"].iloc[-1] - pd.Timedelta(days=365)
                            win = df[df["Date"] >= cutoff]
                            if len(win) >= 2:
                                base = float(win["Close"].iloc[0])
                                if base != 0.0:
                                    ret_12m = float(price / base - 1.0)

                        ohlc_latest = {"price": price, "ma200": ma200, "ret_12m": ret_12m}
            except Exception:
                ohlc_latest = None

            # Friendly messages so you know why Momentum may not score fully
            if ohlc_latest is None:
                st.info("Momentum: no price file found for this stock (upload daily CSV in the Momentum page).")
            else:
                msgs = []
                if ohlc_latest.get("ma200") is None:
                    msgs.append("need ≥200 daily rows for 200-DMA")
                if ohlc_latest.get("ret_12m") is None:
                    msgs.append("need ≥252 daily rows for 12-month return")
                if msgs:
                    st.warning("Momentum not fully scored — " + " & ".join(msgs))



            # ---- ❄️ Snowflake — Overall Performance --------------------
            try:
                scores = calculations.compute_factor_scores(
                    stock_name, stock, ttm, ohlc_latest=ohlc_latest, industry=None
                )
                base   = ["Value", "Quality", "Growth", "Cash", "Momentum"]
                labels = ["Future Value", "Earnings Quality", "Growth Consistency",
                        "Cash Strength", "Momentum"]
                vals   = [int(scores.get(k, 0) or 0) for k in base]
            except Exception:
                scores = {}  # keep defined for detail table below
                labels = ["Future Value", "Earnings Quality", "Growth Consistency",
                        "Cash Strength", "Momentum"]
                vals   = [0, 0, 0, 0, 0]

            # Warn if EPS YoY didn’t hit 8/8 (strict rule on last 12 quarters)
            det = scores.get("_detail", [])
            if isinstance(det, list) and any(d.get("Input","").startswith("EPS YoY %") and d.get("Score") is None for d in det):
                miss = [d for d in det if d.get("Input","").startswith("EPS YoY %")]
                pf = miss[0].get("Components", {}).get("pairs_found", 0) if miss else 0
                st.warning(f"EPS YoY not scored — only {pf}/8 pairs available in the last 12 quarters.")

            # ---- Radar --------------------------------------------------
            lab2 = labels + [labels[0]]
            val2 = vals   + [vals[0]]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=val2, theta=lab2, fill="toself", name=stock_name,
                hovertemplate="<b>%{theta}</b>: %{r}/100<extra></extra>"
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False, margin=dict(l=20, r=20, t=30, b=20), height=420
            )
            st.markdown("#### Snowflake — Overall Performance")
            st.plotly_chart(fig, use_container_width=True)

            # ---- Composite & pillar mini-cards --------------------------
            composite = int(round(sum(vals) / max(1, len(vals))))
            st.metric("Composite Score", f"{composite}/100")
            _cols = st.columns(5)
            for i, (lab, v) in enumerate(zip(labels, vals)):
                with _cols[i]:
                    st.metric(lab, f"{v}/100")

            # ---- Data Health (avoid nested expanders) -------------------
            dh = (ttm or {}).get("DataHealth", {}) if ttm else {}
            ok = not dh.get("missing")
            st.markdown(f"**Data Health:** {'✅ OK' if ok else '⚠️ Incomplete'}")
            if dh.get("missing"):
                if st.checkbox("Show missing inputs affecting metrics", key=f"dh_missing_{stock_name}"):
                    for m in dh["missing"]:
                        st.write("-", m)

            # ---- Show per-metric breakdown (raw, score, source) ---------
            if st.checkbox("ℹ️  Show Snowflake calculation detail", key=f"sf_detail_{stock_name}"):
                detail = scores.get("_detail", [])
                if detail:
                    df = pd.DataFrame(detail)
                    df["Raw"] = df["Raw"].apply(
                        lambda x: "–" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{float(x):,.4f}"
                    )
                    df["Score"] = df["Score"].apply(
                        lambda x: "–" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{int(round(float(x))):d}"
                    )
                    st.dataframe(
                        df[["Pillar", "Input", "Raw", "Score", "Source"]],
                        use_container_width=True, height=300
                    )
                    # component drill-down (HTML <details> avoids nested expander error)
                    for _, row in df.iterrows():
                        comp = row.get("Components")
                        if comp:
                            st.markdown(
                                f"<details><summary><b>{row['Pillar']} · {row['Input']} components</b></summary>",
                                unsafe_allow_html=True,
                            )
                            for k, v in comp.items():
                                if v is None or (isinstance(v, float) and np.isnan(v)):
                                    st.markdown(f"- **{k}**: –")
                                else:
                                    st.markdown(f"- **{k}**: {float(v):,.4f}")
                            st.markdown("</details>", unsafe_allow_html=True)
                else:
                    st.info("Diagnostic data not available.")





            # ---- Annual — Calculated Ratios (styled)
            st.markdown('<div class="sec success"><div class="t">📐 Annual — Calculated Ratios</div></div>', unsafe_allow_html=True)
            st.markdown("#### Calculated Ratios")

            ratios = []
            for _, row in annual.iterrows():
                r = calculations.calc_ratios(row)
                r["Year"] = row["Year"]
                ratios.append(r)
            ratio_df = pd.DataFrame(ratios).set_index("Year").round(4)

            if ratio_df.empty:
                st.info("No ratio data available.")
            else:
                ratio_layout = st.radio(
                    "Calculated ratios table layout (annual)",
                    ["Metrics → columns (Year rows)", "Years → columns (Metric rows)"],
                    horizontal=True,
                    key=f"annual_ratio_layout_{stock_name}"
                )

                if ratio_layout.startswith("Years"):
                    disp = ratio_df.T
                    new_cols_r = drag_reorder(
                        disp.columns.tolist(),
                        key_suffix=f"ann_ratio_yearcols_{stock_name}",
                        help_text="Drag to reorder **Year** columns."
                    )
                    disp = disp[new_cols_r]
                    styled = style_ratio_table(disp)
                    _show_styled(styled, height=360)
                    st.caption("Legend: 🟩 Great  •  🟦 OK  •  🟥 Fails threshold")

                    st.session_state[f"ann_ratio_x_{stock_name}"] = [str(x) for x in new_cols_r]
                    st.session_state[f"ann_ratio_metrics_{stock_name}"] = disp.index.tolist()
                else:
                    disp = ratio_df
                    new_cols_r = drag_reorder(
                        disp.columns.tolist(),
                        key_suffix=f"ann_ratio_metriccols_{stock_name}",
                        help_text="Drag to reorder **Metric** columns."
                    )
                    disp = disp[new_cols_r]
                    styled = style_ratio_table(disp)
                    _show_styled(styled, height=360)
                    st.caption("Legend: 🟩 Great  •  🟦 OK  •  🟥 Fails threshold")

                    st.session_state[f"ann_ratio_x_{stock_name}"] = [str(x) for x in disp.index.astype(str).tolist()]
                    st.session_state[f"ann_ratio_metrics_{stock_name}"] = new_cols_r


            # ---- Undervalue/Overvalue Bar
            st.markdown("#### Undervalue/Overvalue Bar")
            pe = None
            if not ratio_df.empty:
                pe = ratio_df.iloc[-1].get("P/E", None)
            if pe is not None and pe == pe:
                if pe < 15:
                    st.success(f"P/E = {pe:.2f} (Undervalued)")
                elif pe < 25:
                    st.info(f"P/E = {pe:.2f} (Fair Value)")
                else:
                    st.error(f"P/E = {pe:.2f} (Overvalued)")
                st.progress(min(max((25 - pe) / 25, 0), 1))
            else:
                st.info("Not enough data for value bar.")

            # ---- Advanced Growth & Valuation Metrics (CAGR, PEG, Graham, MOS)
            years = sorted(annual["Year"].dropna().astype(int).unique())
            if len(years) >= 2:
                first_year, last_year = years[0], years[-1]
                rev_first = ann_numeric.loc[("Income Statement", "Revenue"), str(first_year)]
                rev_last  = ann_numeric.loc[("Income Statement", "Revenue"), str(last_year)]
                np_first  = ann_numeric.loc[("Income Statement", "Net Profit"), str(first_year)]
                np_last   = ann_numeric.loc[("Income Statement", "Net Profit"), str(last_year)]
                eps_first = ratio_df.loc[first_year, "EPS"] if first_year in ratio_df.index else None
                eps_last  = ratio_df.loc[last_year,  "EPS"] if last_year  in ratio_df.index else None

                period = last_year - first_year
                cagr_rev = ((rev_last / rev_first) ** (1/period) - 1) * 100 if rev_first and rev_first > 0 else None
                cagr_np  = ((np_last  / np_first)  ** (1/period) - 1) * 100 if np_first  and np_first  > 0 else None
                if eps_first is not None and not pd.isna(eps_first) and eps_first > 0:
                    cagr_eps = ((eps_last / eps_first) ** (1/period) - 1) * 100
                else:
                    cagr_eps = None

                if (eps_first is not None and not pd.isna(eps_first)
                        and eps_last is not None and not pd.isna(eps_last)
                        and period > 0):
                    avg_eps_growth = (eps_last - eps_first) / period
                    est_eps_3y     = eps_last + avg_eps_growth * 3
                else:
                    avg_eps_growth = None
                    est_eps_3y     = None

                last_pe = ratio_df.iloc[-1].get("P/E") if not ratio_df.empty else None
                peg     = last_pe / cagr_eps if (last_pe and cagr_eps and cagr_eps > 0) else None

                if eps_last is not None and not pd.isna(eps_last):
                    graham_val = eps_last * (8.5 + 2 * (cagr_eps or 0))
                else:
                    graham_val = None

                price_latest = cur_val
                mos = ((graham_val - price_latest) / graham_val * 100) if (graham_val and graham_val > 0) else None

                st.markdown("#### 🚀 Growth & Valuation Metrics")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rev CAGR (%)", f"{cagr_rev:.2f}%" if cagr_rev is not None else "N/A")
                c1.metric("NP  CAGR (%)", f"{cagr_np:.2f}%" if cagr_np is not None else "N/A")
                c2.metric("Est EPS (3-yr)", f"{est_eps_3y:,.4f}" if est_eps_3y is not None else "N/A")
                c2.metric("P/E Ratio", f"{last_pe:.2f}" if last_pe is not None else "N/A")
                c3.metric("EPS CAGR (%)", f"{cagr_eps:.2f}%" if cagr_eps is not None else "N/A")
                c3.metric("PEG Ratio", f"{peg:.2f}" if peg is not None else "N/A")
                c4.metric("Graham Value", f"{graham_val:,.2f}" if graham_val is not None else "N/A")
                c4.metric("Margin of Safety", f"{mos:,.2f}%" if mos is not None else "N/A")




            # ---- Annual comparison charts …
            st.markdown('<div class="sec warning"><div class="t">📊 Annual — Comparison Charts</div></div>', unsafe_allow_html=True)
            st.markdown("### 📊 Annual Comparison Charts (up to 4)")

            # A) Raw Data comparisons (YoY)
            st.markdown("##### Raw Data (select up to 4 series to compare across Years)")
            ann_opts = field_options(ANNUAL_SECTIONS)
            years_order = st.session_state.get(
                f"ann_raw_x_{stock_name}",
                [str(int(y)) for y in sorted(annual['Year'].dropna().unique())]
            )
            def ann_series_getter(sec_lbl):
                if ann_numeric.empty:
                    return None
                sec, lbl = sec_lbl
                if (sec, lbl) not in ann_numeric.index:
                    return None
                y = ann_numeric.loc[(sec, lbl), :]
                return pd.Series(y).reindex([str(x) for x in years_order]).values

            ann_count = st.slider("Number of raw-data charts", 1, 4, 2, key=f"ann_raw_chartcount_{stock_name}")
            multi_panel_charts(
                ann_count, ann_opts, years_order, ann_series_getter,
                key_prefix=f"annual_raw_chart_{stock_name}", chart_height=320
            )

            # B) Calculated Ratios comparisons (YoY)
            st.markdown("##### Calculated Ratios (select up to 4 ratios to compare across Years)")
            ratio_x = st.session_state.get(
                f"ann_ratio_x_{stock_name}",
                [str(int(y)) for y in ratio_df.index.tolist()]
            )
            ratio_metrics = st.session_state.get(
                f"ann_ratio_metrics_{stock_name}",
                list(ratio_df.columns)
            )
            def ratio_series_getter(metric_name):
                if metric_name not in ratio_df.columns:
                    return None
                y = ratio_df[metric_name]
                y = pd.Series(y.values, index=y.index.astype(str))
                return y.reindex([str(x) for x in ratio_x]).values

            ratio_count = st.slider("Number of ratio charts", 1, 4, 2, key=f"ann_ratio_chartcount_{stock_name}")
            multi_panel_charts(
                ratio_count, [(m, m) for m in ratio_metrics], ratio_x, ratio_series_getter,
                key_prefix=f"annual_ratio_chart_{stock_name}", chart_height=320
            )

            # ---- Systematic Decision (read-only)
            st.markdown(
                '<div class="sec danger"><div class="t">🚦 Systematic Decision</div>'
                '<div class="d">Read-only evaluation for latest annual data</div></div>',
                unsafe_allow_html=True
            )
            st.markdown("### 🚦 Systematic Decision")

            annual_rows_for_eval = annual.sort_values("Year")
            if annual_rows_for_eval.empty:
                st.info("No annual data to evaluate.")
            else:
                latest_row = annual_rows_for_eval.iloc[-1]
                metrics = calculations.calc_ratios(latest_row)
                ev = rules.evaluate(metrics, "Quality-Value")  # or "Dividend"
                cA, cB = st.columns([1, 3])
                with cA:
                    st.metric("Score", f"{ev['score']}%")
                    st.markdown("**Status:** " + ("✅ PASS" if ev["pass"] else "❌ REJECT"))
                with cB:
                    if not ev["pass"] and ev["reasons"]:
                        st.warning("Unmet (mandatory): " + "; ".join(ev["reasons"]))
                    show_checks = st.checkbox("Show checks", value=False, key=f"sys_checks_{stock_name}")
                    if show_checks:
                        st.markdown("**Mandatory checks**")
                        for label, ok, reason in ev["mandatory"]:
                            st.write(("✅ " if ok else "❌ ") + label + ("" if ok else f" — {reason}"))
                        st.markdown("**Scored checks**")
                        for label, ok, weight in ev["scored"]:
                            st.write(("✅ " if ok else "❌ ") + f"{label} ({weight} pts)")

        # =========================
        # QUARTERLY
        # =========================
        with tabs[1]:
            st.subheader(f"{stock_name} - Quarterly Report")

            st.markdown('<div class="sec info"><div class="t">📄 Quarterly — Raw Data</div></div>', unsafe_allow_html=True)
            st.markdown("#### Raw Quarterly Data")
            q_numeric = build_quarter_raw_numeric(quarterly)
            q_raw_layout = st.radio(
                "Raw data layout (quarterly)",
                ["Fields → columns (Period rows)", "Periods → columns (Field rows)"],
                horizontal=True,
                key=f"quarter_raw_layout_{stock_name}"
            )
            if q_numeric.empty:
                st.info("No quarterly raw data available.")
            else:
                if q_raw_layout.startswith("Periods"):
                    disp_qnum = _reorder_empty_last(q_numeric)
                    new_cols = drag_reorder(
                        disp_qnum.columns.tolist(),
                        key_suffix=f"q_raw_periodcols_{stock_name}",
                        help_text="Drag to reorder **Period** columns."
                    )
                    disp_qnum = disp_qnum[new_cols]
                else:
                    disp_qnum = _reorder_empty_last(q_numeric.T)
                    field_cols = disp_qnum.columns.tolist()
                    new_cols = drag_reorder(
                        field_cols,
                        key_suffix=f"q_raw_fieldcols_{stock_name}",
                        help_text="Drag to reorder **Field** columns."
                    )
                    disp_qnum = disp_qnum[new_cols]

                disp_qfmt = disp_qnum.applymap(_format_4)
                if isinstance(disp_qfmt.columns, pd.MultiIndex):
                    disp_qfmt.columns = pd.Index([nice_label(c) for c in disp_qfmt.columns])
                st.dataframe(disp_qfmt, use_container_width=True, height=420)


              # ---- Quarterly Calculated Ratios (STYLED)
    st.markdown('<div class="sec success"><div class="t">📐 Quarterly — Calculated Ratios</div></div>', unsafe_allow_html=True)
    st.markdown("#### Quarterly Calculated Ratios")

    qratios = []
    for _, row in quarterly.iterrows():
        r = calculations.calc_ratios(row)
        r["Year"] = row.get("Year")
        r["Quarter"] = row.get("Quarter")
        qratios.append(r)
    qratio_df = pd.DataFrame(qratios)

    # Normalize to Period index (e.g., 2024 Q3)
    if "Quarter" in qratio_df.columns:
        qratio_df["Qnum"] = qratio_df["Quarter"].map(quarter_key_to_num)
        qratio_df = qratio_df.dropna(subset=["Year", "Qnum"])
        qratio_df["Year"] = qratio_df["Year"].astype(int)
        qratio_df = qratio_df.sort_values(["Year", "Qnum"])
        qratio_df["Period"] = qratio_df["Year"].astype(str) + " Q" + qratio_df["Qnum"].astype(int).astype(str)
        qratio_df = qratio_df.drop(columns=["Qnum"]).set_index("Period")
    elif "Year" in qratio_df.columns:
        qratio_df = qratio_df.set_index("Year")

    if qratio_df.empty:
        st.info("No quarterly ratio data available.")
    else:
        qratio_df = qratio_df.apply(pd.to_numeric, errors="ignore").round(4)

        qratio_layout = st.radio(
            "Calculated ratios table layout (quarterly)",
            ["Metrics → columns (Period rows)", "Periods → columns (Metric rows)"],
            horizontal=True,
            key=f"quarter_ratio_layout_{stock_name}"
        )

        if qratio_layout.startswith("Periods"):
            # Periods → columns (metrics as rows)
            disp_qratio = qratio_df.T
            new_cols = drag_reorder(
                disp_qratio.columns.tolist(),
                key_suffix=f"q_ratio_periodcols_{stock_name}",
                help_text="Drag to reorder **Period** columns."
            )
            disp_qratio = disp_qratio[new_cols]
            styled = style_ratio_table(disp_qratio)
            _show_styled(styled, height=360)
            st.caption("Legend: 🟩 Great  •  🟦 OK  •  🟥 Fails threshold")
        else:
            # Metrics → columns (periods as rows)
            disp_qratio = qratio_df
            new_cols = drag_reorder(
                disp_qratio.columns.tolist(),
                key_suffix=f"q_ratio_metriccols_{stock_name}",
                help_text="Drag to reorder **Metric** columns."
            )
            disp_qratio = disp_qratio[new_cols]
            styled = style_ratio_table(disp_qratio)
            _show_styled(styled, height=360)
            st.caption("Legend: 🟩 Great  •  🟦 OK  •  🟥 Fails threshold")


            # ===== Quarterly comparison charts under the ratio table =====
            st.markdown('<div class="sec warning"><div class="t">📊 Quarterly — Comparison Charts</div></div>', unsafe_allow_html=True)
            st.markdown("### 📊 Quarterly Comparison Charts (up to 4)")

            # A) Raw Quarterly comparisons
            st.markdown("##### Raw Quarterly Data (select up to 4 series to compare across Periods)")
            q_opts = field_options(QUARTERLY_SECTIONS)
            period_labels = list(q_numeric.columns)
            def q_series_getter(sec_lbl):
                if q_numeric.empty:
                    return None
                sec, lbl = sec_lbl
                if (sec, lbl) not in q_numeric.index:
                    return None
                y = q_numeric.loc[(sec, lbl), :]
                return pd.Series(y).values

            q_raw_count = st.slider("Number of raw-data charts", 1, 4, 2, key=f"q_raw_chartcount_{stock_name}")
            multi_panel_charts(
                q_raw_count, q_opts, period_labels, q_series_getter,
                key_prefix=f"quarter_raw_chart_{stock_name}", chart_height=320
            )

            # B) Ratio comparisons
            st.markdown("##### Calculated Ratios (select up to 4 ratios to compare across Periods)")
            ratio_opts = [(m, m) for m in qratio_df.columns.tolist()]
            period_ratio = qratio_df.index.tolist()
            def q_ratio_series_getter(metric_name):
                if metric_name not in qratio_df.columns:
                    return None
                y = qratio_df[metric_name]
                return pd.Series(y.values, index=period_ratio).values

            q_ratio_count = st.slider("Number of ratio charts", 1, 4, 2, key=f"q_ratio_chartcount_{stock_name}")
            multi_panel_charts(
                q_ratio_count, ratio_opts, period_ratio, q_ratio_series_getter,
                key_prefix=f"quarter_ratio_chart_{stock_name}", chart_height=320
            )

st.caption("Drag chips fixed: style args removed for streamlit-sortables to prevent React errors. Fallback drag list keeps pill styling. Charts & tables unchanged.")


