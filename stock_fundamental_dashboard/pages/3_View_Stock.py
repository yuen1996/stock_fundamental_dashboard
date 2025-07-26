import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from utils import io_helpers, calculations

st.header("🔍 View Stock")
df = io_helpers.load_data()
if df is None or df.empty:
    st.warning("No data.")
    st.stop()

stocks = sorted(df["Name"].unique())

for stock_name in stocks:
    with st.expander(stock_name):
        stock = df[df["Name"] == stock_name].sort_values(["Year", "Quarter" if "Quarter" in df.columns else "Year"])
        # Prepare annual and quarterly DataFrames
        annual = stock[stock['Type'] == 'Annual'] if 'Type' in stock.columns else stock[stock['Quarter'].isnull()]
        quarterly = stock[stock['Type'] == 'Quarterly'] if 'Type' in stock.columns else stock[stock['Quarter'].notnull()]

        # Tab navigation inside expander
        tabs = st.tabs(["Annual Report", "Quarterly Report"])
        
        # --- Annual Report Tab ---
        with tabs[0]:
            st.subheader(f"{stock_name} - Annual Financial Data")
            if annual.empty:
                st.info("No annual data available.")
            else:
                st.markdown("#### Raw Data")
                st.dataframe(annual.set_index("Year"))
                
                # Calculate ratios for all years
                ratios = []
                for _, row in annual.iterrows():
                    r = calculations.calc_ratios(row)
                    r["Year"] = row["Year"]
                    ratios.append(r)
                ratio_df = pd.DataFrame(ratios).set_index("Year").round(4)

                st.markdown("#### Calculated Ratios")
                st.dataframe(ratio_df)

                # Radar chart
                st.markdown("#### Financial Snowflake (Radar)")
                metrics = [
                    "Net Profit Margin (%)", "ROE (%)", "Current Ratio", "Debt-Asset Ratio (%)", "Dividend Yield (%)"
                ]
                categories = [m.replace(" (%)", "") for m in metrics]
                last = ratio_df.iloc[-1]
                vals = [last.get(m, 0) or 0 for m in metrics]
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=categories,
                    fill='toself',
                    name='Latest Year'
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, max(vals + [1])])),
                    showlegend=False
                )
                st.plotly_chart(fig)

                # Undervalue/overvalue bar (P/E)
                st.markdown("#### Undervalue/Overvalue Bar")
                pe = last.get("P/E", None)
                if pe is not None and pe == pe:  # not NaN
                    if pe < 15:
                        st.success(f"P/E = {pe:.2f} (Undervalued)")
                    elif pe < 25:
                        st.info(f"P/E = {pe:.2f} (Fair Value)")
                    else:
                        st.error(f"P/E = {pe:.2f} (Overvalued)")
                    st.progress(min(max((25-pe)/25,0),1))
                else:
                    st.info("Not enough data for value bar.")

                # Ratio trend charts
                st.markdown("#### Ratio Trends (Yearly)")
                for col in ratio_df.columns:
                    st.line_chart(ratio_df[[col]])

        # --- Quarterly Tab ---
        with tabs[1]:
            st.subheader(f"{stock_name} - Quarterly Report")
            if quarterly.empty:
                st.info("No quarterly data available.")
            else:
                st.markdown("#### Raw Quarterly Data")
                # Combine year/quarter to show as index
                qdf = quarterly.copy()
                if "Quarter" in qdf.columns:
                    qdf["Period"] = qdf["Year"].astype(str) + " Q" + qdf["Quarter"].astype(str)
                    qdf = qdf.set_index("Period")
                st.dataframe(qdf)

                # Calculate ratios for all quarters
                qratios = []
                for _, row in quarterly.iterrows():
                    r = calculations.calc_ratios(row)
                    r["Year"] = row["Year"]
                    r["Quarter"] = row["Quarter"] if "Quarter" in row else None
                    qratios.append(r)
                qratio_df = pd.DataFrame(qratios)
                # Combine to show period
                if "Quarter" in qratio_df.columns:
                    qratio_df["Period"] = qratio_df["Year"].astype(str) + " Q" + qratio_df["Quarter"].astype(str)
                    qratio_df = qratio_df.set_index("Period")
                else:
                    qratio_df = qratio_df.set_index("Year")
                st.markdown("#### Quarterly Calculated Ratios")
                st.dataframe(qratio_df.round(4))
                # No chart for quarterly as requested

st.caption("Click a stock name to see all years, quarters and charts.")

