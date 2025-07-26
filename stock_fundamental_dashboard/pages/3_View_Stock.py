import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from utils import io_helpers, calculations
import re

# ---------- Force Wide Layout on Page ----------
st.set_page_config(layout="wide")

# ---------- Page CSS ----------
BASE_CSS = """
<style>
html, body, [class*="css"] { font-size: 16px !important; }
h1, h2, h3, h4 { color:#0f172a !important; font-weight:800 !important; letter-spacing:.2px; }
.stApp { background: radial-gradient(1100px 600px at 18% -10%, #f7fbff 0%, #eef4ff 45%, #ffffff 100%); }
.stDataFrame { font-size: 15px !important; }
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

# ---------- Field definitions (order matters) ----------
ANNUAL_SECTIONS = [
    ("Income Statement", [
        ("Net Profit",                     "NetProfit"),
        ("Gross Profit",                   "GrossProfit"),
        ("Revenue",                        "Revenue"),
        ("Cost Of Sales",                  "CostOfSales"),
        ("Finance Costs",                  "FinanceCosts"),
        ("Administrative Expenses",        "AdminExpenses"),
        ("Selling & Distribution Expenses","SellDistExpenses"),
    ]),
    ("Balance Sheet", [
        ("Number of Shares",   "NumShares"),
        ("Current Asset",      "CurrentAsset"),
        ("Other Receivables",  "OtherReceivables"),
        ("Trade Receivables",  "TradeReceivables"),
        ("Biological Assets",  "BiologicalAssets"),
        ("Inventories",        "Inventories"),
        ("Prepaid Expenses",   "PrepaidExpenses"),
        ("Intangible Asset",   "IntangibleAsset"),
        ("Current Liability",  "CurrentLiability"),
        ("Total Asset",        "TotalAsset"),
        ("Total Liability",    "TotalLiability"),
        ("Shareholder Equity", "ShareholderEquity"),
        ("Reserves",           "Reserves"),
    ]),
    ("Other Data", [
        ("Dividend pay cent",         "Dividend"),
        ("End of year share price",   "SharePrice"),
    ]),
]

QUARTERLY_SECTIONS = [
    ("Quarterly Income Statement", [
        ("Quarterly Net Profit",                     "Q_NetProfit"),
        ("Quarterly Gross Profit",                   "Q_GrossProfit"),
        ("Quarterly Revenue",                        "Q_Revenue"),
        ("Quarterly Cost Of Sales",                  "Q_CostOfSales"),
        ("Quarterly Finance Costs",                  "Q_FinanceCosts"),
        ("Quarterly Administrative Expenses",        "Q_AdminExpenses"),
        ("Quarterly Selling & Distribution Expenses","Q_SellDistExpenses"),
    ]),
    ("Quarterly Balance Sheet", [
        ("Number of Shares",   "Q_NumShares"),
        ("Current Asset",      "Q_CurrentAsset"),
        ("Other Receivables",  "Q_OtherReceivables"),
        ("Trade Receivables",  "Q_TradeReceivables"),
        ("Biological Assets",  "Q_BiologicalAssets"),
        ("Inventories",        "Q_Inventories"),
        ("Prepaid Expenses",   "Q_PrepaidExpenses"),
        ("Intangible Asset",   "Q_IntangibleAsset"),
        ("Current Liability",  "Q_CurrentLiability"),
        ("Total Asset",        "Q_TotalAsset"),
        ("Total Liability",    "Q_TotalLiability"),
        ("Shareholder Equity", "Q_ShareholderEquity"),
        ("Reserves",           "Q_Reserves"),
    ]),
    ("Quarterly Other Data", [
        ("Current Share Price",                 "Q_SharePrice"),
        ("Each end per every quarter price",    "Q_EndQuarterPrice"),
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
    # keep blanks for NaN; otherwise 4 decimal with thousands
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):,.4f}"
    except Exception:
        return ""

def _reorder_empty_last(df_numeric: pd.DataFrame) -> pd.DataFrame:
    """Move rows with all-NaN to the bottom; keep original order for non-empty rows."""
    if df_numeric.empty:
        return df_numeric
    mask = df_numeric.isna().all(axis=1)
    return pd.concat([df_numeric[~mask], df_numeric[mask]], axis=0)

def quarter_key_to_num(q):
    """'Q1' -> 1, '1' -> 1, else NaN"""
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

# ---------- Builders (return NUMERIC tables; formatting + transpose happens later) ----------
def build_annual_raw_numeric(annual_df: pd.DataFrame) -> pd.DataFrame:
    """Rows = fields, Cols = years (numeric, NaNs)."""
    if annual_df.empty:
        return pd.DataFrame()

    years = sorted([int(y) for y in annual_df["Year"].dropna().unique()])
    # Build row index (MultiIndex: Section, Field)
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
    """Rows = fields, Cols = 'Year Q#' (numeric, NaNs)."""
    if quarter_df.empty:
        return pd.DataFrame()

    q = quarter_df.copy()
    q["Qnum"] = q["Quarter"].map(quarter_key_to_num)
    q = q.dropna(subset=["Year", "Qnum"])
    q["Year"] = q["Year"].astype(int)
    q = q.sort_values(["Year", "Qnum"])

    periods = [f"{int(r['Year'])} Q{int(r['Qnum'])}" for _, r in q.iterrows()]
    seen = set()
    cols = []
    row_by_period = {}
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

            # ---- Raw Data with layout toggle + empties-last awareness
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
                    # Years on columns -> transpose numeric -> rows=Fields; then empties-last by rows
                    disp_num = ann_numeric  # rows=Fields, cols=Years (already)
                    disp_num = _reorder_empty_last(disp_num)
                    # Now transpose for Years as columns? Wait: option says "Years → columns (Field rows)" -> that's the current numeric orientation
                    # so do NOT transpose here.
                else:
                    # "Fields → columns (Year rows)" -> years rows, fields columns
                    disp_num = ann_numeric.T  # rows=Years, cols=Fields
                    disp_num = _reorder_empty_last(disp_num)

                # Format
                disp_fmt = disp_num.copy()
                disp_fmt = disp_fmt.applymap(_format_4)

                # If columns are MultiIndex (when fields are columns), give nice names
                if isinstance(disp_fmt.columns, pd.MultiIndex):
                    disp_fmt.columns = pd.Index([f"{sec} • {fld}" for sec, fld in disp_fmt.columns])

                st.dataframe(disp_fmt, use_container_width=True, height=420)

            # ---- Calculated Ratios with layout toggle (unchanged logic)
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
                    disp_ratio = ratio_df.T
                    disp_ratio.index.name = "Metric"
                    st.dataframe(disp_ratio, use_container_width=True, height=360)
                else:
                    st.dataframe(ratio_df, use_container_width=True, height=360)

                # Radar (snowflake) – latest year
                st.markdown("#### Financial Snowflake (Radar)")
                metrics = ["Net Profit Margin (%)", "ROE (%)", "Current Ratio", "Debt-Asset Ratio (%)", "Dividend Yield (%)"]
                categories = [m.replace(" (%)", "") for m in metrics]
                last = ratio_df.iloc[-1]
                vals = [last.get(m, 0) or 0 for m in metrics]
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=vals, theta=categories, fill='toself', name='Latest Year'))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max(vals + [1])])), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                # Value bar based on P/E
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

                # Trends
                st.markdown("#### Ratio Trends (Yearly)")
                for col in ratio_df.columns:
                    st.line_chart(ratio_df[[col]])

        # =========================
        # QUARTERLY
        # =========================
        with tabs[1]:
            st.subheader(f"{stock_name} - Quarterly Report")

            # ---- Raw Quarterly Data with layout toggle + empties-last awareness
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
                    # Periods → columns (Field rows): current numeric orientation (rows=Fields, cols=Periods)
                    disp_qnum = q_numeric
                    disp_qnum = _reorder_empty_last(disp_qnum)
                else:
                    # Fields → columns (Period rows): transpose
                    disp_qnum = q_numeric.T  # rows=Periods, cols=Fields
                    disp_qnum = _reorder_empty_last(disp_qnum)

                disp_qfmt = disp_qnum.applymap(_format_4)
                if isinstance(disp_qfmt.columns, pd.MultiIndex):
                    disp_qfmt.columns = pd.Index([f"{sec} • {fld}" for sec, fld in disp_qfmt.columns])
                st.dataframe(disp_qfmt, use_container_width=True, height=420)

            # ---- Quarterly ratios with layout toggle
            st.markdown("#### Quarterly Calculated Ratios")
            qratios = []
            for _, row in quarterly.iterrows():
                r = calculations.calc_ratios(row)
                r["Year"] = row["Year"]
                r["Quarter"] = row["Quarter"] if "Quarter" in row else None
                qratios.append(r)
            qratio_df = pd.DataFrame(qratios)

            if "Quarter" in qratio_df.columns:
                # Build Period label "Year Q#"
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
                    disp_qratio.index.name = "Metric"
                    st.dataframe(disp_qratio, use_container_width=True, height=360)
                else:
                    st.dataframe(qratio_df.round(4), use_container_width=True, height=360)

            # (Charts for quarterly remain optional/unmodified)

st.caption("Raw Data: switch layouts freely. Rows with all blank values are automatically moved to the bottom.")
