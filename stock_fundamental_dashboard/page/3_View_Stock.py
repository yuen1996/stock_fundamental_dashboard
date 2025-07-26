import streamlit as st
from utils.io_helpers import load_stocks
from utils.calculations import get_radar_chart_data, get_historic_chart

st.title("View Stock Details")

df = load_stocks()
if df.empty:
    st.warning("No data available. Please add stocks first.")
    st.stop()

stock = st.selectbox("Select stock to view", df['Stock'].unique())
row = df[df['Stock'] == stock].iloc[0]

st.header(f"{row['Stock']} ({row['Industry']})")
st.write("**All Ratios & Details:**")
st.json(row.to_dict())

st.subheader("Radar Chart (Fundamentals)")
radar_data, metrics = get_radar_chart_data(row)
import plotly.graph_objs as go
fig = go.Figure(
    data=[
        go.Scatterpolar(
            r=radar_data['values'],
            theta=radar_data['metrics'],
            fill='toself'
        )
    ]
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("History/TTM/CAGR (Demo)")
fig_hist = get_historic_chart(row)
st.plotly_chart(fig_hist, use_container_width=True)
