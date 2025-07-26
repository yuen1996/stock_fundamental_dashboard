import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from utils import io_helpers, calculations

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
                st.markdown("#### Raw Data")
                st.dataframe(annual.set_index("Year"), use_container_width=True)

                ratios = []
                for _, row in annual.iterrows():
                    r = calculations.calc_ratios(row)
                    r["Year"] = row["Year"]
                    ratios.append(r)
                ratio_df = pd.DataFrame(ratios).set_index("Year").round(4)

                st.markdown("#### Calculated Ratios")
                st.dataframe(ratio_df, use_container_width=True)

                # Radar (snowflake)
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

        # --- Quarterly ---
        with tabs[1]:
            st.subheader(f"{stock_name} - Quarterly Report")
            if quarterly.empty:
                st.info("No quarterly data available.")
            else:
                st.markdown("#### Raw Quarterly Data")
                qdf = quarterly.copy()
                if "Quarter" in qdf.columns:
                    qvals = qdf["Quarter"].astype(str)
                    # If already looks like Q#, keep; else prefix Q
                    qvals = qvals.where(qvals.str.match(r"^Q\\d$", na=False), "Q" + qvals.str.replace(r"^[Qq]", "", regex=True))
                    qdf["Period"] = qdf["Year"].astype(str) + " " + qvals
                    qdf = qdf.set_index("Period")
                st.dataframe(qdf, use_container_width=True)

                # Calculate ratios for quarters
                qratios = []
                for _, row in quarterly.iterrows():
                    r = calculations.calc_ratios(row)
                    r["Year"] = row["Year"]
                    r["Quarter"] = row["Quarter"] if "Quarter" in row else None
                    qratios.append(r)
                qratio_df = pd.DataFrame(qratios)
                if "Quarter" in qratio_df.columns:
                    qvals = qratio_df["Quarter"].astype(str)
                    qvals = qvals.where(qvals.str.match(r"^Q\\d$", na=False), "Q" + qvals.str.replace(r"^[Qq]", "", regex=True))
                    qratio_df["Period"] = qratio_df["Year"].astype(str) + " " + qvals
                    qratio_df = qratio_df.set_index("Period")
                else:
                    qratio_df = qratio_df.set_index("Year")

                st.markdown("#### Quarterly Calculated Ratios")
                st.dataframe(qratio_df.round(4), use_container_width=True)
                # No charts for quarterly
st.caption("Click a stock name to see all years, quarters and charts.")

