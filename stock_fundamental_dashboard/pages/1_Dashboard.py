import streamlit as st
import pandas as pd
from utils import io_helpers, calculations, config

st.header("📊 Stock Dashboard")

df = io_helpers.load_data()

if df.empty:
    st.info("No stock data yet. Please add stock in the 'Add/Edit' page.")
    st.stop()

ratio_df = df.apply(calculations.calc_ratios, axis=1, result_type='expand')
df_ratios = pd.concat([df, ratio_df], axis=1)

industry = st.selectbox("Filter by Industry", ["All"] + sorted(df['Industry'].dropna().unique().tolist()))
if industry != "All":
    df_ratios = df_ratios[df_ratios["Industry"] == industry]

sort_by = st.selectbox("Sort by", ["Name"] + list(ratio_df.columns))
df_ratios = df_ratios.sort_values(by=sort_by, ascending=False if sort_by!="Name" else True)

def color_row(row):
    color = []
    for col in ratio_df.columns:
        val = row[col]
        if pd.isnull(val): color.append("")
        elif col in config.RATIO_THRESHOLDS:
            threshold = config.RATIO_THRESHOLDS[col]
            if col in ["PE", "PB", "Debt Asset Ratio"]:
                if val <= threshold:
                    color.append("background-color: #b6fcb6")
                elif val <= 1.5 * threshold:
                    color.append("background-color: #c8e4ff")
                else:
                    color.append("background-color: #ffc2b3")
            else:
                if val >= threshold:
                    color.append("background-color: #b6fcb6")
                elif val >= 0.8 * threshold:
                    color.append("background-color: #c8e4ff")
                else:
                    color.append("background-color: #ffc2b3")
        else:
            color.append("")
    return color

st.dataframe(df_ratios.style.apply(color_row, axis=1, subset=ratio_df.columns), use_container_width=True)

