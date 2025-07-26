import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from utils import io_helpers, calculations

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

# ---------- DEFINE RAW USER INPUT COLUMNS ----------
user_input_columns_annual = [
    # Add ALL fields users can key in for annual
    "FairValue", "EPS", "BVPS", "Revenue", "NetProfit", "Price", "Industry",
    # Add more user-input columns as needed
]
user_input_columns_quarterly = [
    # Add ALL fields users can key in for quarterly
    "FairValue", "EPS", "BVPS", "Revenue", "NetProfit", "Price", "Industry", "Quarter",
    # Add more user-input columns as needed
]

for stock_name in stocks:
    with st.expander(stock_name, expanded=False):
        stock = df[df["Name"] == stock_name].sort_values(["Year"])
        # Split annual vs quarterly (prefer IsQuarter)
        annual = stock[stock["IsQuarter"] != True].copy()
        quarterly = stock[stock["IsQuarter"] == True].copy()

        tabs = st.tabs(["Annual Report", "Quarterly Report"])

        # --- Annual ---
        with tabs[0]:
            st.subheader(f"{stock_name} - Annual Financial Data")
            if annual.empty:
                st.info("No annual data available.")
            else:
                # --------- RAW DATA (ANNUAL) ---------
                st.markdown("#### Raw Data")
                ann_raw = annual.copy()
                if "Year" in ann_raw.columns:
                    ann_raw = ann_raw.set_index("Year")
                existing_annual_cols = [c for c in user_input_columns_annual if c in ann_raw.columns]
                if not existing_annual_cols:
                    st.info("No raw fields found for this stock.")
                else:
                    ann_raw = ann_raw[existing_annual_cols]
                    ann_raw_layout = st.radio(
                        "Raw data layout (annual)",
                        ["Fields → columns (Year rows)", "Years → columns (Field rows)"],
                        horizontal=True,
                        key=f"annual_raw_layout_{stock_name}"
                    )
                    if ann_raw_layout.startswith("Years"):
                        disp_raw = ann_raw.T
                        disp_raw.index.name = "Field"
                        st.dataframe(disp_raw, use_container_width=True, height=360)
                    else:
                        st.dataframe(ann_raw, use_container_width=True, height=360)

                # --------- CALCULATED RATIOS (ANNUAL) ---------
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
                    table_layout = st.radio(
                        "Table layout",
                        ["Metrics → columns (Year rows)", "Years → columns (Metric rows)"],
                        horizontal=True,
                        key=f"annual_ratio_layout_{stock_name}"
                    )
                    if table_layout.startswith("Years"):
                        disp = ratio_df.T
                        disp.index.name = "Metric"
                        st.dataframe(disp, use_container_width=True, height=360)
                    else:
                        st.dataframe(ratio_df, use_container_width=True, height=360)

                # Radar (snowflake)
                st.markdown("#### Financial Snowflake (Radar)")
                if not ratio_df.empty:
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
                pe = last.get("P/E", None) if not ratio_df.empty else None
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

        # --- Quarterly ---
        with tabs[1]:
            st.subheader(f"{stock_name} - Quarterly Report")
            if quarterly.empty:
                st.info("No quarterly data available.")
            else:
                # --------- RAW DATA (QUARTERLY) ---------
                st.markdown("#### Raw Quarterly Data")
                qdf = quarterly.copy()
                if "Year" in qdf.columns and "Quarter" in qdf.columns:
                    qdf["Period"] = qdf["Year"].astype(str) + " " + qdf["Quarter"].astype(str)
                    qdf = qdf.set_index("Period")
                existing_q_cols = [c for c in user_input_columns_quarterly if c in qdf.columns]
                if not existing_q_cols:
                    st.info("No raw fields found for this stock.")
                else:
                    qdf = qdf[existing_q_cols]
                    q_raw_layout = st.radio(
                        "Raw data layout (quarterly)",
                        ["Fields → columns (Quarter rows)", "Quarters → columns (Field rows)"],
                        horizontal=True,
                        key=f"quarterly_raw_layout_{stock_name}"
                    )
                    if q_raw_layout.startswith("Quarters"):
                        disp_qraw = qdf.T
                        disp_qraw.index.name = "Field"
                        st.dataframe(disp_qraw, use_container_width=True, height=360)
                    else:
                        st.dataframe(qdf, use_container_width=True, height=360)

                # --------- CALCULATED RATIOS (QUARTERLY) ---------
                st.markdown("#### Quarterly Calculated Ratios")
                qratios = []
                for _, row in quarterly.iterrows():
                    r = calculations.calc_ratios(row)
                    r["Year"] = row["Year"]
                    r["Quarter"] = row["Quarter"] if "Quarter" in row else None
                    qratios.append(r)
                qratio_df = pd.DataFrame(qratios)
                if "Year" in qratio_df.columns and "Quarter" in qratio_df.columns:
                    qratio_df["Period"] = qratio_df["Year"].astype(str) + " " + qratio_df["Quarter"].astype(str)
                    qratio_df = qratio_df.set_index("Period")
                elif "Year" in qratio_df.columns:
                    qratio_df = qratio_df.set_index("Year")
                if qratio_df.empty:
                    st.info("No quarterly ratio data available.")
                else:
                    st.dataframe(qratio_df.round(4), use_container_width=True, height=400)
                # No charts for quarterly

st.caption("Click a stock name to see all years, quarters and charts.")
