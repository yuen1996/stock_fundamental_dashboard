import streamlit as st
import pandas as pd
from utils import io_helpers, calculations

# ---------- Force Wide Layout on Page ----------
st.set_page_config(layout="wide")

# ---------- Page CSS (for larger font, bigger table, nicer look) ----------
BASE_CSS = """
<style>
html, body, [class*="css"] { font-size: 17px !important; }
h1, h2, h3, h4 { color:#0f172a !important; font-weight:800 !important; letter-spacing:.2px; }
.stApp { background: radial-gradient(1100px 600px at 18% -10%, #f7fbff 0%, #eef4ff 45%, #ffffff 100%); }
.stDataEditor, .stDataFrame { font-size: 16px !important; }
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

st.header("📊 Dashboard: All Stocks & Ratios")

# --- Load Data ---
df = io_helpers.load_data()
if df is None or df.empty or "Name" not in df.columns:
    st.warning("No stock data found. Please add data in 'Add or Edit'.")
    st.stop()

if "IsQuarter" not in df.columns:
    df["IsQuarter"] = False

annual = df[df["IsQuarter"] != True].copy()

# --- FILTERS: Industry, Search, Sort ---
industries = ["All"] + sorted([x for x in annual["Industry"].dropna().unique()])
industry_sel = st.selectbox("Filter by Industry", industries, index=0)

search_text = st.text_input("🔍 Search Stock Name or Industry", placeholder="Type to filter...")

view = annual if industry_sel == "All" else annual[annual["Industry"] == industry_sel]

if search_text.strip():
    search_lower = search_text.strip().lower()
    view = view[
        view["Name"].str.lower().str.contains(search_lower, na=False) |
        view["Industry"].str.lower().str.contains(search_lower, na=False)
    ]

# --- Sort Selector ---
sort_col = st.selectbox("Sort by column", view.columns, index=0)
sort_asc = st.radio("Sort order", ["Ascending", "Descending"], horizontal=True) == "Ascending"
view = view.sort_values(by=sort_col, ascending=sort_asc).reset_index(drop=True)

# --- Quick Edit/Delete Table (Annual) ---
st.subheader("✏️ Quick Edit (Annual Records)")

# Add internal key columns for stable editing
view.insert(0, "RowKey", view.apply(lambda r: f"{r['Name']}|{int(r['Year'])}|A", axis=1))
view.insert(1, "Delete", False)

# Show the table wider and taller
edited = st.data_editor(
    view,
    use_container_width=True,
    height=600,  # Bigger!
    hide_index=True,
    num_rows="dynamic",
    column_config={
        "RowKey": st.column_config.TextColumn("RowKey", help="Internal key", disabled=True, width="small"),
        "Delete": st.column_config.CheckboxColumn("Delete", help="Tick to delete this row"),
        "Name": st.column_config.TextColumn("Name", disabled=True),
        "Year": st.column_config.NumberColumn("Year", disabled=True, format="%d"),
    },
)

# Save button to persist edits/deletes
if st.button("💾 Save edits & deletes"):
    # 1) Deletions
    del_keys = set(edited.loc[edited["Delete"] == True, "RowKey"].tolist())
    for key in del_keys:
        name, year, _ = key.split("|")
        mask = (df["Name"] == name) & (df["Year"] == int(year)) & (df["IsQuarter"] != True)
        df = df[~mask]

    # 2) Edits (non-deleted rows)
    keep = edited[edited["Delete"] != True].copy()
    for _, row in keep.iterrows():
        name, year, _ = row["RowKey"].split("|")
        mask = (df["Name"] == name) & (df["Year"] == int(year)) & (df["IsQuarter"] != True)
        # columns to update: everything in df except RowKey/Delete (not in df anyway)
        update_cols = [c for c in keep.columns if c not in ("RowKey", "Delete")]
        for c in update_cols:
            if c in df.columns:
                df.loc[mask, c] = row[c]

    io_helpers.save_data(df)
    st.success("Changes saved.")
    st.experimental_rerun()

# -------- Read-only ratio summaries per stock --------
st.subheader("📈 Annual Ratios by Stock")
df_view = df[df["IsQuarter"] != True].copy()
if industry_sel != "All":
    df_view = df_view[df_view["Industry"] == industry_sel]
if search_text.strip():
    search_lower = search_text.strip().lower()
    df_view = df_view[
        df_view["Name"].str.lower().str.contains(search_lower, na=False) |
        df_view["Industry"].str.lower().str.contains(search_lower, na=False)
    ]
if df_view.empty:
    st.info("No annual rows to display for this filter.")
    st.stop()

for name in df_view["Name"].dropna().unique():
    st.markdown(f"### {name}")
    stock = df_view[df_view["Name"] == name].sort_values("Year")
    if stock.empty:
        continue
    ratios = []
    for _, row in stock.iterrows():
        r = calculations.calc_ratios(row)
        r["Year"] = row["Year"]
        ratios.append(r)
    r_df = pd.DataFrame(ratios).set_index("Year").round(4)
    st.dataframe(r_df, use_container_width=True)

