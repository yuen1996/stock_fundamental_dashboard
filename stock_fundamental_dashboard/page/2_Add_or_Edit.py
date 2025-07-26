import streamlit as st
import pandas as pd
from utils.io_helpers import load_stocks, save_stocks
from utils.calculations import calc_all_ratios

st.title("Add or Edit Stock Data")

df = load_stocks()
stock_names = df['Stock'].tolist() if not df.empty else []
editing = st.selectbox("Edit existing stock?", ["Add New"] + stock_names)

if editing != "Add New":
    stock_row = df[df['Stock'] == editing].iloc[0]
else:
    stock_row = pd.Series(dtype='object')

with st.form("stock_form"):
    stock = st.text_input("Stock", value=stock_row.get('Stock', ''))
    industry = st.text_input("Industry", value=stock_row.get('Industry', ''))
    price = st.number_input("Current Price", value=float(stock_row.get('Price', 0)), min_value=0.0)
    fair_value = st.number_input("Fair Value Estimate", value=float(stock_row.get('FairValue', 0)), min_value=0.0)
    eps = st.number_input("EPS (Trailing 12M)", value=float(stock_row.get('EPS', 0)), min_value=0.0)
    bvps = st.number_input("Book Value per Share", value=float(stock_row.get('BVPS', 0)), min_value=0.0)
    roe = st.number_input("ROE (%)", value=float(stock_row.get('ROE', 0)), min_value=0.0)
    net_margin = st.number_input("Net Margin (%)", value=float(stock_row.get('NetMargin', 0)), min_value=0.0)
    revenue = st.number_input("Revenue (Latest)", value=float(stock_row.get('Revenue', 0)), min_value=0.0)
    net_profit = st.number_input("Net Profit (Latest)", value=float(stock_row.get('NetProfit', 0)), min_value=0.0)
    ttm_revenue = st.number_input("TTM Revenue", value=float(stock_row.get('TTM_Revenue', 0)), min_value=0.0)
    ttm_net_profit = st.number_input("TTM Net Profit", value=float(stock_row.get('TTM_NetProfit', 0)), min_value=0.0)
    shares = st.number_input("Shares Outstanding", value=float(stock_row.get('Shares', 0)), min_value=0.0)
    submit = st.form_submit_button("Save Stock")

if submit:
    # Calculate all ratios & scoring
    input_data = {
        "Stock": stock, "Industry": industry, "Price": price, "FairValue": fair_value,
        "EPS": eps, "BVPS": bvps, "ROE": roe, "NetMargin": net_margin,
        "Revenue": revenue, "NetProfit": net_profit, "TTM_Revenue": ttm_revenue, "TTM_NetProfit": ttm_net_profit,
        "Shares": shares
    }
    result = calc_all_ratios(input_data)
    # Update df and save
    if editing != "Add New":
        df.loc[df['Stock'] == editing] = result
    else:
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    save_stocks(df)
    st.success(f"Saved: {stock}")
    st.experimental_rerun()
