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
    # Prefer: st-draggable-list (CSS-stylable pills)
    from st_draggable_list import DraggableList  # pip install st-draggable-list
    DRAG_LIB = "draggable-list"
except Exception:
    try:
        # Fallback: streamlit-sortables (works but not CSS-stylable)
        from streamlit_sortables import sort_items  # pip install streamlit-sortables
        DRAG_LIB = "sortables"
    except Exception:
        DRAG_LIB = None

# ---------- Force Wide Layout on Page ----------
st.set_page_config(layout="wide")

# ---------- Page CSS ----------
BASE_CSS = """
<style>
html, body, [class*="css"] { font-size: 16px !important; }
h1, h2, h3, h4 { color:#0f172a !important; font-weight:800 !important; letter-spacing:.2px; }
.stApp { background: radial-gradient(1100px 600px at 18% -10%, #f7fbff 0%, #eef4ff 45%, #ffffff 100%); }
.stDataFrame { font-size: 15px !important; }

/* Drag area container (tight & capped height) */
.sort-help { color:#475569; font-size:.90rem; margin:.15rem 0 .25rem 0; }
.sort-box {
  border:1px dashed #cbd5e1; background:#f8fafc; border-radius:12px;
  padding:.15rem .30rem;              /* tighter padding */
  max-height:90px; overflow:auto;     /* smaller and scrollable */
}

/* ---------- Compact pill styling ---------- */
/* streamlit-sortables generic draggable */
.sort-box [draggable="true"]{
  display:inline-flex !important;
  width:auto !important; max-width:100% !important;
  border-radius:9999px !important;
  background:#eef2ff !important;              /* soft indigo */
  border:1px solid #c7d2fe !important;
  color:#0f172a !important;
  font-size:10px !important;                  /* smaller text */
  padding:1px 6px !important;                 /* smaller chip */
  line-height:1.15 !important;
  margin:2px !important;
}
.sort-box > div{ display:flex; flex-wrap:wrap; gap:4px; align-items:center; }

/* st-draggable-list pills */
.sdl-item, .sdl-item * {
  display:inline-flex !important;
  width:auto !important; max-width:100% !important;
  border-radius:9999px !important;
  background:#eef2ff !important;
  color:#0f172a !important;
  border:1px solid #c7d2fe !important;
  font-size:10px !important;
  padding:1px 6px !important;
  margin:2px !important;
}
/* ensure list wraps, not full-width lines */
.sdl-wrapper, .sdl-container, .sdl-list {
  display:flex !important; flex-wrap:wrap !important; gap:4px !important;
}
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)

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
    ]),
    ("Other Data", [
        ("Dividend pay cent", "Dividend"),
        ("End of year share price", "SharePrice"),
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
    ("Quarterly Other Data", [
        ("Current Share Price", "Q_SharePrice"),
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
    """
    Return labels in new order using whichever drag lib is available.
    **Important:** For streamlit-sortables we no longer send style props to avoid React component errors.
    """
    if DRAG_LIB == "sortables":
        from streamlit_sortables import sort_items
        with st.container():
            st.markdown('<div class="sort-box">', unsafe_allow_html=True)
            new_labels = sort_items(items=labels, key=f"sort_{key_suffix}")  # no style args
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

    # 1) restore last saved order for this list (if any)
    state_key = f"reorder_saved_{key_suffix}"
    saved = st.session_state.get(state_key)
    if isinstance(saved, list):
        # keep only labels that still exist, then append any new ones
        base = [l for l in saved if l in labels] + [l for l in labels if l not in saved]
    else:
        base = labels

    # 2) toggle to open the drag UI (compact)
    show = st.checkbox("🔧 Reorder (drag) — compact", value=False, key=f"reorder_toggle_{key_suffix}")
    if show:
        st.markdown(f'<div class="sort-help">{help_text}</div>', unsafe_allow_html=True)
        # pass the current base order to the drag widget so it shows the *saved* order
        current_order = _drag_labels(base, key_suffix)
        # 3) persist the new order so it stays when you close the toggle
        if isinstance(current_order, list) and current_order:
            st.session_state[state_key] = current_order
    else:
        # when closed, keep using the last saved order (or the original if none)
        current_order = st.session_state.get(state_key, base)

    # 4) map labels back to the real column objects
    label_to_col = {nice_label(c): c for c in columns}
    ordered_cols = [label_to_col[l] for l in current_order if l in label_to_col]
    # include any columns not present (safety)
    for c in columns:
        if c not in ordered_cols:
            ordered_cols.append(c)
    return ordered_cols



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
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        title=title, xaxis_title="", yaxis_title=""
    )
    st.plotly_chart(fig, use_container_width=True)

def multi_panel_charts(count, options, x_labels, series_getter, key_prefix, chart_height=320):
    # clamp chart count
    count = max(1, min(4, int(count)))
    option_labels = [o[0] for o in options]

    # if there are no options at all, show a single “no data” message and return
    if not option_labels:
        st.info("No series available to chart.")
        return

    # prepare grid of up to 4 cells
    row1 = st.columns(2)
    row2 = st.columns(2) if count > 2 else (None, None)

    def render_cell(col_container, i):
        with col_container:
            # pick a default index in-bounds
            default_idx = i if i < len(option_labels) else 0

            sel = st.selectbox(
                f"Chart {i+1} – pick a series",
                options=option_labels,
                index=default_idx,
                key=f"{key_prefix}_sel_{i}",
            )

            # now hook up the series and draw
            payload = dict(options)[sel]
            y = series_getter(payload)
            if y is None or (pd.isna(pd.Series(y)).all()):
                st.info("No data for this selection.")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_labels, y=y, mode="lines+markers", name=sel))
                fig.update_layout(
                    height=chart_height,
                    margin=dict(l=10, r=10, t=40, b=10),
                    title=sel, xaxis_title="", yaxis_title=""
                )
                st.plotly_chart(fig, use_container_width=True)

    # render up to `count` cells
    render_cell(row1[0], 0)
    if count >= 2:
        render_cell(row1[1], 1)
    if count >= 3 and row2[0] is not None:
        render_cell(row2[0], 2)
    if count >= 4 and row2[1] is not None:
        render_cell(row2[1], 3)


# ---------- UI ----------
for stock_name in stocks:
    with st.expander(stock_name, expanded=False):
        stock = df[df["Name"] == stock_name].sort_values(["Year"])
        annual = stock[stock["IsQuarter"] != True].copy()
        quarterly = stock[stock["IsQuarter"] == True].copy()

        # ---- Current Price metric (prefer CurrentPrice; fallback to latest SharePrice)
        cur_val = None
        if "CurrentPrice" in stock.columns:
            s = stock["CurrentPrice"].dropna()
            if not s.empty:
                cur_val = s.iloc[-1]

        if cur_val is None:
            if "SharePrice" in annual.columns:
                s2 = annual.sort_values("Year")["SharePrice"].dropna()
                if not s2.empty:
                    cur_val = s2.iloc[-1]

        cur_val = float(cur_val) if cur_val is not None else 0.0
        st.metric("Current Price", f"{cur_val:,.4f}")


        tabs = st.tabs(["Annual Report", "Quarterly Report"])


        # =========================
        # ANNUAL
        # =========================
        with tabs[0]:
            st.subheader(f"{stock_name} - Annual Financial Data")

            # ---- Raw Data table
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

            # ---- Calculated Ratios table
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
                    st.dataframe(disp, use_container_width=True, height=360)
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
                    st.dataframe(disp, use_container_width=True, height=360)
                    st.session_state[f"ann_ratio_x_{stock_name}"] = [str(x) for x in disp.index.astype(str).tolist()]
                    st.session_state[f"ann_ratio_metrics_{stock_name}"] = new_cols_r

                # Radar (snowflake)
                st.markdown("#### Financial Snowflake (Radar)")
                metrics_radar = ["Net Profit Margin (%)", "ROE (%)", "Current Ratio", "Debt-Asset Ratio (%)", "Dividend Yield (%)"]
                categories = [m.replace(" (%)", "") for m in metrics_radar]
                last = ratio_df.iloc[-1]
                vals = [last.get(m, 0) or 0 for m in metrics_radar]
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=vals, theta=categories, fill='toself', name='Latest Year'))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max(vals + [1])])), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                # Undervalue/Overvalue Bar
                st.markdown("#### Undervalue/Overvalue Bar")
                pe = last.get("P/E", None)
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
                    
                # ==== Advanced Growth & Valuation Metrics ====
                # only if we have at least two annual points
                years = sorted(annual["Year"].dropna().astype(int).unique())
                if len(years) >= 2:
                    first_year, last_year = years[0], years[-1]
                    # raw values from the numeric table
                    rev_first = ann_numeric.loc[("Income Statement", "Revenue"), str(first_year)]
                    rev_last  = ann_numeric.loc[("Income Statement", "Revenue"), str(last_year)]
                    np_first  = ann_numeric.loc[("Income Statement", "Net Profit"), str(first_year)]
                    np_last   = ann_numeric.loc[("Income Statement", "Net Profit"), str(last_year)]
                    # EPS from your ratio_df
                    eps_first = ratio_df.loc[first_year, "EPS"]
                    eps_last  = ratio_df.loc[last_year,  "EPS"]

                    period = last_year - first_year
                    # CAGR formulas
                    cagr_rev = ((rev_last / rev_first) ** (1/period) - 1) * 100 if rev_first > 0 else None
                    cagr_np  = ((np_last  / np_first)  ** (1/period) - 1) * 100 if np_first > 0 else None
                     # only compute CAGR if eps_first is a real, positive number
                    if eps_first is not None and not pd.isna(eps_first) and eps_first > 0:
                        cagr_eps = ((eps_last / eps_first) ** (1/period) - 1) * 100
                    else:
                        cagr_eps = None

                    # Safely compute EPS growth & 3‑year estimate only if we have two real EPS values and a positive period
                    if (eps_first is not None and not pd.isna(eps_first)
                            and eps_last is not None and not pd.isna(eps_last)
                            and period > 0):
                        avg_eps_growth = (eps_last - eps_first) / period
                        est_eps_3y     = eps_last + avg_eps_growth * 3
                    else:
                        avg_eps_growth = None
                        est_eps_3y     = None

                    # PE & PEG (unchanged)
                    last_pe = last.get("P/E")
                    peg     = last_pe / cagr_eps if (last_pe and cagr_eps and cagr_eps > 0) else None

                    # Benjamin Graham intrinsic value & margin of safety
                    if eps_last is not None and not pd.isna(eps_last):
                        graham_val = eps_last * (8.5 + 2 * (cagr_eps or 0))
                    else:
                        graham_val = None

                    price_latest = cur_val
                    mos = (
                        (graham_val - price_latest) / graham_val * 100
                        if (graham_val is not None and graham_val > 0)
                        else None
                    )


                    # ==== Growth & Valuation Metrics ====
                    st.markdown("#### 🚀 Growth & Valuation Metrics")
                    c1, c2, c3, c4 = st.columns(4)

                    c1.metric(
                        "Rev CAGR (%)",
                        f"{cagr_rev:.2f}%" if cagr_rev is not None else "N/A"
                    )
                    c1.metric(
                        "NP  CAGR (%)",
                        f"{cagr_np:.2f}%" if cagr_np is not None else "N/A"
                    )

                    c2.metric(
                        "Est EPS (3‑yr)",
                        f"{est_eps_3y:,.4f}" if est_eps_3y is not None else "N/A"
                    )
                    c2.metric(
                        "P/E Ratio",
                        f"{last_pe:.2f}" if last_pe is not None else "N/A"
                    )

                    c3.metric(
                        "EPS CAGR (%)",
                        f"{cagr_eps:.2f}%" if cagr_eps is not None else "N/A"
                    )
                    c3.metric(
                        "PEG Ratio",
                        f"{peg:.2f}" if peg is not None else "N/A"
                    )

                    c4.metric(
                        "Graham Value",
                        f"{graham_val:,.2f}" if graham_val is not None else "N/A"
                    )
                    c4.metric(
                        "Margin of Safety",
                        f"{mos:.2f}%" if mos is not None else "N/A"
                    )

   

                         # ===== Annual comparison charts under the bar =====
            st.markdown("### 📊 Annual Comparison Charts (up to 4)")

            # ---- A) Raw Data comparisons (YoY)
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
                key_prefix=f"annual_raw_chart_{stock_name}",
                chart_height=320
            )

            # ---- B) Calculated Ratios comparisons (YoY)
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
                y = pd.Series(y.values, index=y.index.astype(str))  # normalize index to str
                return y.reindex([str(x) for x in ratio_x]).values

            ratio_count = st.slider("Number of ratio charts", 1, 4, 2, key=f"ann_ratio_chartcount_{stock_name}")
            multi_panel_charts(
                ratio_count, [(m, m) for m in ratio_metrics], ratio_x, ratio_series_getter,
                key_prefix=f"annual_ratio_chart_{stock_name}",
                chart_height=320
            )

            # === Systematic Decision (strict, read-only) – shown below annual charts ===
            st.markdown("### 🚦 Systematic Decision")

            annual_rows_for_eval = annual.sort_values("Year")
            if annual_rows_for_eval.empty:
                st.info("No annual data to evaluate.")
            else:
                latest_row = annual_rows_for_eval.iloc[-1]
                metrics = calculations.calc_ratios(latest_row)

                ev = rules.evaluate(metrics, "Quality-Value")  # or "Dividend"

                # Compact layout: status + score on the left; details on the right
                cA, cB = st.columns([1, 3])

                with cA:
                    st.metric("Score", f"{ev['score']}%")
                    st.markdown("**Status:** " + ("✅ PASS" if ev["pass"] else "❌ REJECT"))

                with cB:
                    # Show unmet mandatory reasons first if any
                    if not ev["pass"] and ev["reasons"]:
                        st.warning("Unmet (mandatory): " + "; ".join(ev["reasons"]))

                    # Toggle to reveal detailed checks (avoids nested expanders)
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

            st.markdown("#### Quarterly Calculated Ratios")
            qratios = []
            for _, row in quarterly.iterrows():
                r = calculations.calc_ratios(row)
                r["Year"] = row["Year"]
                r["Quarter"] = row["Quarter"] if "Quarter" in row else None
                qratios.append(r)
            qratio_df = pd.DataFrame(qratios)

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
                qratio_layout = st.radio(
                    "Calculated ratios table layout (quarterly)",
                    ["Metrics → columns (Period rows)", "Periods → columns (Metric rows)"],
                    horizontal=True,
                    key=f"quarter_ratio_layout_{stock_name}"
                )

                if qratio_layout.startswith("Periods"):
                    disp_qratio = qratio_df.round(4).T
                    new_cols = drag_reorder(
                        disp_qratio.columns.tolist(),
                        key_suffix=f"q_ratio_periodcols_{stock_name}",
                        help_text="Drag to reorder **Period** columns."
                    )
                    disp_qratio = disp_qratio[new_cols]
                    st.dataframe(disp_qratio, use_container_width=True, height=360)
                else:
                    disp_qratio = qratio_df.round(4)
                    new_cols = drag_reorder(
                        disp_qratio.columns.tolist(),
                        key_suffix=f"q_ratio_metriccols_{stock_name}",
                        help_text="Drag to reorder **Metric** columns."
                    )
                    disp_qratio = disp_qratio[new_cols]
                    st.dataframe(disp_qratio, use_container_width=True, height=360)

                    
                      # ===== Quarterly comparison charts under the ratio table =====
            st.markdown("### 📊 Quarterly Comparison Charts (up to 4)")

            # ---- A) Raw Quarterly Data comparisons (select up to 4 series) ----
            st.markdown("##### Raw Quarterly Data (select up to 4 series to compare across Periods)")
            q_opts = field_options(QUARTERLY_SECTIONS)
            # x‑axis labels are the period columns from build_quarter_raw_numeric
            period_labels = list(q_numeric.columns)
            def q_series_getter(sec_lbl):
                if q_numeric.empty:
                    return None
                sec, lbl = sec_lbl
                if (sec, lbl) not in q_numeric.index:
                    return None
                y = q_numeric.loc[(sec, lbl), :]
                return pd.Series(y).values

            q_raw_count = st.slider(
                "Number of raw-data charts",
                min_value=1, max_value=4, value=2,
                key=f"q_raw_chartcount_{stock_name}"
            )
            multi_panel_charts(
                q_raw_count, q_opts, period_labels, q_series_getter,
                key_prefix=f"quarter_raw_chart_{stock_name}",
                chart_height=320
            )

            # ---- B) Calculated Ratios comparisons (select up to 4 ratios) ----
            st.markdown("##### Calculated Ratios (select up to 4 ratios to compare across Periods)")
            # payloads are just metric names
            ratio_opts = [(m, m) for m in qratio_df.columns.tolist()]
            period_ratio = qratio_df.index.tolist()
            def q_ratio_series_getter(metric_name):
                if metric_name not in qratio_df.columns:
                    return None
                y = qratio_df[metric_name]
                return pd.Series(y.values, index=period_ratio).values

            q_ratio_count = st.slider(
                "Number of ratio charts",
                min_value=1, max_value=4, value=2,
                key=f"q_ratio_chartcount_{stock_name}"
            )
            multi_panel_charts(
                q_ratio_count, ratio_opts, period_ratio, q_ratio_series_getter,
                key_prefix=f"quarter_ratio_chart_{stock_name}",
                chart_height=320
            )
          

st.caption("Drag chips fixed: style args removed for streamlit-sortables to prevent React errors. Fallback drag list keeps pill styling. Charts & tables unchanged.")
