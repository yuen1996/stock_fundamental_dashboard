import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from utils.calculations import get_radar_chart_data
from utils.io_helpers import load_stocks

st.title("Dashboard: All Stocks")
df = load_stocks()

if df.empty:
    st.info("No data. Go to Add/Edit page to add stocks.")
    st.stop()

# Filtering
industries = df['Industry'].unique()
selected_industries = st.multiselect("Filter by Industry", industries, default=industries)
show_only = st.selectbox("Show only", ["All", "Achievers (Green)", "Moderate (Blue)", "Not Achieved (Red)"])

df = df[df['Industry'].isin(selected_industries)]

if show_only == "Achievers (Green)":
    df = df[df['ScoreColor'] == 'green']
elif show_only == "Moderate (Blue)":
    df = df[df['ScoreColor'] == 'blue']
elif show_only == "Not Achieved (Red)":
    df = df[df['ScoreColor'] == 'red']

# Sorting
sort_col = st.selectbox("Sort by", df.columns, index=0)
sort_asc = st.radio("Sort order", ["Ascending", "Descending"]) == "Ascending"
df = df.sort_values(by=sort_col, ascending=sort_asc)

# Color styling
def color_row(row):
    color = row['ScoreColor']
    if color == 'green':
        return ['background-color: #b7eb8f'] * len(row)
    elif color == 'blue':
        return ['background-color: #91d5ff'] * len(row)
    elif color == 'red':
        return ['background-color: #ffa39e'] * len(row)
    else:
        return [''] * len(row)

st.dataframe(df.style.apply(color_row, axis=1), use_container_width=True)

# Comparison: Show radar for selected stock
compare = st.selectbox("Select stock for radar/graphic view", df['Stock'].unique())
row = df[df['Stock'] == compare].iloc[0]
radar_data, metrics = get_radar_chart_data(row)
fig = go.Figure(
    data=[
        go.Scatterpolar(
            r=radar_data['values'],
            theta=radar_data['metrics'],
            fill='toself',
            name=compare
        )
    ],
    layout=go.Layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False
    )
)
st.plotly_chart(fig, use_container_width=True)

# Value bar
st.subheader("Undervalue / Overvalue Bar")
fair_value = row.get('FairValue', None)
price = row.get('Price', None)
if price and fair_value:
    fig2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=price,
        gauge={
            'axis': {'range': [None, fair_value*2]},
            'bar': {'color': "green" if price < fair_value else "red"},
            'steps': [
                {'range': [0, fair_value], 'color': "#b7eb8f"},
                {'range': [fair_value, fair_value*2], 'color': "#ffa39e"}
            ],
            'threshold': {'line': {'color': "blue", 'width': 4}, 'thickness': 0.75, 'value': fair_value}
        },
        title={'text': "Stock Price vs. Fair Value"}
    ))
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Please ensure Price and FairValue fields are filled.")

if st.button("Refresh data"):
    st.experimental_rerun()
