import streamlit as st
import pandas as pd
from utils import io_helpers, calculations

# ---------- Force Wide Layout on Page ----------
st.set_page_config(layout="wide")

# ---------- Page CSS (larger font / clean background) ----------
BASE_CSS = """
<style>
html, body, [class*="css"] { font-size: 17px !important; }
h1, h2, h3, h4 { color:#0f172a !important; font-weight:800 !important; letter-spacing:.2px; }
.stApp { background: radial-gradient(1100px 600px at 18% -10%, #f7fbff 0%, #eef4ff 45%, #ffffff 100%); }
.stDataEditor, .stDataFrame { font-size: 16px !important; }
.stButton>button { border-radius: 12px; padding: .6rem 1.1rem; font-weight: 700; }
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

# Work with annual rows only on dashboard
annual = df[df["IsQuarter"] != True].copy()

# --- FILTERS: Industry, Search ---
colf1, colf2 = st.columns([1, 2])
with colf1:
    industries = ["All"] + sorted([x for x in annual["Industry"].dropna().unique()])
    industry_sel = st.selectbox("Filter by Industry", industries, index=0)
with colf2:
    search_text = st.text_input("🔍 Search Stock Name or Industry", placeholder="Type to filter...")

view = annual if industry_sel == "All" else annual[annual["Industry"] == industry_sel]
if search_text.strip():
    q = search_text.strip().lower()
    view = view[
        view["Name"].str.lower().str.contains(q, na=False) |
        view["Industry"].str.lower().str.contains(q, na=False)
    ]

# --- View mode toggle ---
view_mode = st.radio(
    "View mode",
    ["Group by Stock (recommended)", "Flat (all years)"],
    horizontal=True
)

# ===============================================================
#  A) SUMMARY (one row per stock — latest year only)
# ===============================================================
st.subheader("📌 Latest-year Summary (one row per stock)")

if view.empty:
    st.info("No rows to show for the current filter.")
else:
    # Latest row per stock
    latest = view.sort_values(["Name", "Year"]).groupby("Name", as_index=False).tail(1).copy()

    # Compute key ratios for each latest row
    rows = []
    for _, r in latest.iterrows():
        ratio = calculations.calc_ratios(r)
        rows.append({
            "Stock": r["Name"],
            "Industry": r.get("Industry", None),
            "Year": r.get("Year", None),
            "Price": r.get("SharePrice", None),
            "Revenue": ratio.get("Revenue"),
            "NetProfit": ratio.get("NetProfit"),
            "EPS": ratio.get("EPS"),
            "ROE (%)": ratio.get("ROE (%)"),
            "P/E": ratio.get("P/E"),
            "P/B": ratio.get("P/B"),
            "Net Profit Margin (%)": ratio.get("Net Profit Margin (%)"),
            "Dividend Yield (%)": ratio.get("Dividend Yield (%)"),
        })

    summary = pd.DataFrame(rows).sort_values(["Stock"]).reset_index(drop=True)
    st.dataframe(summary, use_container_width=True, height=320)

# ===============================================================
#  B) GROUPED EDIT — one expander per stock
# ===============================================================
if view_mode.startswith("Group"):
    st.subheader("✏️ Edit by Stock (expand to edit/delete annual rows)")
    if view.empty:
        st.info("No rows to edit for the current filter.")
    else:
        for name in sorted(view["Name"].dropna().unique()):
            sv = view[view["Name"] == name].sort_values("Year").reset_index(drop=True).copy()
            if sv.empty:
                continue

            yr_min = int(sv["Year"].min()) if pd.notna(sv["Year"]).any() else "-"
            yr_max = int(sv["Year"].max()) if pd.notna(sv["Year"]).any() else "-"
            with st.expander(f"{name}  •  Years {yr_min}–{yr_max}", expanded=False):
                # Add key columns
                sv.insert(0, "RowKey", sv.apply(lambda r: f"{r['Name']}|{int(r['Year'])}|A", axis=1))
                sv.insert(1, "Delete", False)

                edited = st.data_editor(
                    sv,
                    use_container_width=True,
                    height=360,
                    hide_index=True,
                    num_rows="fixed",   # keep to existing rows inside each stock
                    column_config={
                        "RowKey": st.column_config.TextColumn("RowKey", help="Internal key", disabled=True, width="small"),
                        "Delete": st.column_config.CheckboxColumn("Delete", help="Tick to delete this year"),
                        "Name": st.column_config.TextColumn("Name", disabled=True),
                        "Year": st.column_config.NumberColumn("Year", disabled=True, format="%d"),
                    },
                    key=f"editor_{name}",
                )

                # Save only this stock's edits/deletes
                if st.button(f"💾 Save changes for {name}", key=f"save_{name}"):
                    # Delete selected
                    del_keys = set(edited.loc[edited["Delete"] == True, "RowKey"].tolist())
                    for key in del_keys:
                        s_name, s_year, _ = key.split("|")
                        mask = (df["Name"] == s_name) & (df["Year"] == int(s_year)) & (df["IsQuarter"] != True)
                        df.drop(df[mask].index, inplace=True)

                    # Update remaining rows
                    keep = edited[edited["Delete"] != True].copy()
                    for _, er in keep.iterrows():
                        s_name, s_year, _ = er["RowKey"].split("|")
                        mask = (df["Name"] == s_name) & (df["Year"] == int(s_year)) & (df["IsQuarter"] != True)

                        update_cols = [c for c in keep.columns if c not in ("RowKey", "Delete")]
                        for c in update_cols:
                            if c in df.columns:
                                df.loc[mask, c] = er[c]

                    io_helpers.save_data(df)
                    st.success(f"Saved changes for {name}.")
                    st.experimental_rerun()

# ===============================================================
#  C) Flat view (optional)
# ===============================================================
else:
    st.subheader("📃 Flat Table (all annual rows)")
    # Add internal key columns for stable editing
    flat = view.sort_values(["Name", "Year"]).reset_index(drop=True).copy()
    flat.insert(0, "RowKey", flat.apply(lambda r: f"{r['Name']}|{int(r['Year'])}|A", axis=1))
    flat.insert(1, "Delete", False)

    edited = st.data_editor(
        flat,
        use_container_width=True,
        height=600,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "RowKey": st.column_config.TextColumn("RowKey", help="Internal key", disabled=True, width="small"),
            "Delete": st.column_config.CheckboxColumn("Delete", help="Tick to delete this row"),
            "Name": st.column_config.TextColumn("Name", disabled=True),
            "Year": st.column_config.NumberColumn("Year", disabled=True, format="%d"),
        },
        key="flat_editor",
    )

    # Save button to persist edits/deletes for the flat view
    if st.button("💾 Save edits & deletes (flat view)", key="save_flat"):
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
            update_cols = [c for c in keep.columns if c not in ("RowKey", "Delete")]
            for c in update_cols:
                if c in df.columns:
                    df.loc[mask, c] = row[c]

        io_helpers.save_data(df)
        st.success("Changes saved.")
        st.experimental_rerun()
