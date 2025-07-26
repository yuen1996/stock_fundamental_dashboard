import streamlit as st
import pandas as pd
from utils import io_helpers, config

st.header("➕ Add / Edit Stock")

df = io_helpers.load_data()

with st.form("add_edit_form"):
    cols = st.columns(3)
    data = {}
    for idx, field in enumerate(config.STOCK_FIELDS):
        data[field] = cols[idx % 3].text_input(field, key=field)
    submitted = st.form_submit_button("Add/Update Stock")
    if submitted:
        try:
            data['Year'] = int(data['Year'])
            data['NetProfit'] = float(data['NetProfit'])
            data['Revenue'] = float(data['Revenue'])
            data['Equity'] = float(data['Equity'])
            data['Asset'] = float(data['Asset'])
            data['Liability'] = float(data['Liability'])
            data['Dividend'] = float(data['Dividend'])
            data['ShareOutstanding'] = float(data['ShareOutstanding'])
            data['Price'] = float(data['Price'])
        except:
            st.error("Please enter valid numeric values.")
            st.stop()
        exists = ((df['Name'] == data['Name']) & (df['Year'] == data['Year'])).any() if not df.empty else False
        row_df = pd.DataFrame([data])
        if exists:
            df.loc[(df['Name'] == data['Name']) & (df['Year'] == data['Year']), :] = data
        else:
            df = pd.concat([df, row_df], ignore_index=True)
        io_helpers.save_data(df)
        st.success(f"Saved: {data['Name']} ({data['Year']})")
