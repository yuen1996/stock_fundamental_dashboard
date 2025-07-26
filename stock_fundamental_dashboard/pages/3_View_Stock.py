import streamlit as st
import pandas as pd
from utils import io_helpers, calculations
import plotly.graph_objects as go

st.header("📈 View Stock")

df = io_helpers.load_data()
if df.empty:
    st.warning("No data to show.")
    st.stop()

names = df['Name'].unique().tolist()
stock_name = st.selectbox("Select stock", names)
stock_df = df[df['Name'] == stock_name].sort_values(by='Year')

if stock_df.empty:
    st.info("No data for selected stock.")
    st.stop()

st.subheader(f"{stock_name} - Financial Ratios")

for idx, row in stock_df.iterrows():
    st.markdown(f"### Year: {int(row['Year'])}")
    ratios = calculations.calc_ratios(row)
    st.json(ratios)

    eps = float(row['NetProfit']) / float(row['ShareOutstanding']) if float(row['ShareOutstanding']) else 0
    bvps = float(row['Equity']) / float(row['ShareOutstanding']) if float(row['ShareOutstanding']) else 0
    intrinsic = calculations.calc_graham_value(eps, bvps)
    mos = calculations.margin_of_safety(float(row['Price']), intrinsic)
    st.write(f"**Benjamin Graham value:** {intrinsic:.2f} | **Margin of safety:** {mos:.2f}%")

    radar = calculations.radar_scores(ratios)
    categories = list(radar.keys())
    values = list(radar.values())
    fig = go.Figure(data=go.Scatterpolar(
        r=values + values[:1], theta=categories + categories[:1],
        fill='toself', name=stock_name, line_color="gold"
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False, title="Snowflake Financial Chart"
    )
    st.plotly_chart(fig, use_container_width=True)

    bar_val = (intrinsic - float(row['Price'])) / intrinsic if intrinsic else 0
    st.progress(min(1, max(0, bar_val+0.5)))

    st.divider()
