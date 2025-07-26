import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from utils import io_helpers, calculations
import re

# ------- Drag-and-drop imports (robust) -------
DRAG_LIB = None
try:
    # Preferred: streamlit-sortables
    from streamlit_sortables import sort_items  # pip install streamlit-sortables
    DRAG_LIB = "sortables"
except Exception:
    try:
        # Fallback: streamlit-draggable-list
        from st_draggable_list import DraggableList  # pip install streamlit-draggable-list
        DRAG_LIB = "draggable-list"
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
.sort-help { color:#475569; font-size:.90rem; margin:.25rem 0 .5rem 0; }
.sort-box { border:1px dashed #cbd5e1; padding:.35rem .5rem; border-radius:12px; background:#f8fafc; }

/* Fallback visual tweaks for draggable-list */
.sdl-item, .sdl-item * {
  border-radius:9999px !important;
  background:#eef2ff !important;
  color:#1e293b !important;
  border:1px solid #c7d2fe !important;
  font-size:13px !important;
  padding:4px 10px !important;
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

# Soft pill style for streamlit-sortables
PILL_STYLE = {
    "list": {
        "backgroundColor": "transparent",
        "display": "flex",
        "flexWrap": "wrap",
        "gap": "8px",
        "padding": "4px",
    },
    "item": {
        "backgroundColor": "#eef2ff",
        "borderRadius": "9999px",
        "padding": "4px 10px",
        "border": "1px solid #c7d2fe",
        "color": "#1e293b",
        "fontSize": "13px",
    },
    "itemLabel": {"color": "#1e293b"},
}

def _drag_labels(labels, key_suffix):
    """Return labels in new order using whichever drag lib is available (unique key prevents duplicates).
       Compatible with multiple streamlit-sortables versions."""
    if DRAG_LIB == "sortables":
        from streamlit_sortables import sort_items
        with st.container():
            st.markdown('<div class="sort-box">', unsafe_allow_html=True)
            # Try different parameter names across versions; fall back to no style.
            try:
                new_labels = sort_items(items=labels, key=f"sort_{key_suffix}", customStyle=PILL_STYLE)
            except TypeError:
                try:
                    new_labels = sort_items(items=labels, key=f"sort_{key_suffix}", styles=PILL_STYLE)
                except TypeError:
                    try:
                        new_labels = sort_items(items=labels, key=f"sort_{key_suffix}", custom_style=PILL_STYLE)
                    except TypeError:
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
    st.markdown(f'<div class="sort-help">{help_text}</div>', unsafe_allow_html=True)
    new_label_order = _drag_labels(labels, key_suffix)

    label_to_col = {nice_label(c): c for c in columns}
    ordered_cols = [label_to_col[l] for l in new_label_order if l in label_to_col]
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
    count = max(1, min(4, int(count)))
    option_labels = [o[0] for o in options]
    row1 = st.columns(2)
    row2 = st.columns(2) if count > 2 else (None, None)

    def render_cell(col_container, i):
        with col_container:
            default_idx = i if i < len(option_labels) else 0
            sel = st.selectbox(
                f"Chart {i+1} – pick a series",
                options=option_labels,
                index=default_idx if option_labels else None,
                key=f"{key_prefix}_sel_{i}",
            )
            if options:
                payload = dict(options)[sel]
                y = series_getter(payload)
                if y is None or (pd.isna(pd.Series(y)).all()):
                    st.info("No data for this selection.")
                else:
                    plot_single_series(x_labels, y, sel, sel, height=chart_height)

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
                    # Convert index to string before reindex to avoid all-NaN
                    y = pd.Series(y.values, index=y.index.astype(str))
                    return y.reindex([str(x) for x in ratio_x]).values

                ratio_count = st.slider("Number of ratio charts", 1, 4, 2, key=f"ann_ratio_chartcount_{stock_name}")
                multi_panel_charts(
                    ratio_count, [(m, m) for m in ratio_metrics], ratio_x, ratio_series_getter,
                    key_prefix=f"annual_ratio_chart_{stock_name}",
                    chart_height=320
                )

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

st.caption("Drag chips now work across streamlit-sortables versions (style arg fallback). Charts & tables unchanged.")
