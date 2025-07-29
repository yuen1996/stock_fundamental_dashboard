import streamlit as st
import pandas as pd
import numpy as np
import os
import streamlit.components.v1 as components
import math
from datetime import datetime, timedelta
from utils import io_helpers, calculations
from utils import rules as rules_engine

# ---------- Force Wide Layout on Page ----------
st.set_page_config(layout="wide")

# ---------- Page CSS ----------
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

st.header("📊 Dashboard")

# --- Load Data ---
df = io_helpers.load_data()
if df is None or df.empty or "Name" not in df.columns:
    st.warning("No stock data found. Please add data in 'Add or Edit'.")
    st.stop()

# Ensure proper types
if "IsQuarter" not in df.columns:
    df["IsQuarter"] = False
df["IsQuarter"] = df["IsQuarter"].astype(bool)

# Get data file last modified timestamp
DATA_PATH = io_helpers.DATA_PATH            # ← from utils/io_helpers.py
if os.path.exists(DATA_PATH):
    file_ts  = os.path.getmtime(DATA_PATH)
    last_mod = datetime.fromtimestamp(file_ts)
else:
    last_mod = None

# Work with annual rows only for summary
annual = df[~df["IsQuarter"]].copy()

# --- FILTERS: Industry, Search, Date ---
colf1, colf2, colf3 = st.columns([1, 2, 1])
with colf1:
    industries = ["All"] + sorted(annual["Industry"].dropna().unique())
    industry_sel = st.selectbox("Filter by Industry", industries, index=0)
with colf2:
    search_text = st.text_input("🔍 Search Stock Name or Industry", placeholder="Type to filter...")
with colf3:
    date_opts = ["Any", "> 7 days", "> 14 days", "> 1 month", "> 3 months"]
    date_sel = st.selectbox("Filter by last update", date_opts, index=0)

# Apply filters
df_view = annual.copy()
if industry_sel != "All":
    df_view = df_view[df_view["Industry"] == industry_sel]
if search_text.strip():
    q = search_text.lower()
    df_view = df_view[
        df_view["Name"].str.lower().str.contains(q, na=False) |
        df_view["Industry"].str.lower().str.contains(q, na=False)
    ]
if last_mod and date_sel != "Any":
    age = datetime.now() - last_mod
    if date_sel == "> 7 days":
        thresh = timedelta(days=7)
    elif date_sel == "> 14 days":
        thresh = timedelta(days=14)
    elif date_sel == "> 1 month":
        thresh = timedelta(days=30)
    else:
        thresh = timedelta(days=90)
    if age <= thresh:
        df_view = df_view.iloc[0:0]  # no rows pass

# ===============================================================
#  A) SUMMARY (one row per stock — latest year only)
# ===============================================================
st.subheader("📌 Latest-year Summary")
if df_view.empty:
    st.info("No rows to show for the current filter.")
else:
    latest = (
        df_view
        .sort_values(["Name", "Year"])
        .groupby("Name", as_index=False)
        .tail(1)
    )

# ---- Systematic Decision (beta) ----
with st.expander("🚦 Systematic Decision (beta)", expanded=False):
    strategy = st.selectbox("Strategy", list(rules_engine.RULESETS.keys()), key="sys_strategy")
    min_score = st.slider("Minimum score to PASS", 0, 100, 60, 5, key="sys_min_score")
    show_only_pass = st.checkbox("Show only PASS", True, key="sys_only_pass")

    dec_rows = []
    for _, r in latest.iterrows():
        metrics = calculations.calc_ratios(r)
        cur = r.get("CurrentPrice", np.nan)
        if pd.isna(cur):
            cur = r.get("SharePrice", np.nan)
        ev = rules_engine.evaluate_ratios(metrics, strategy, min_score)
        dec_rows.append({
            "Stock": r["Name"],
            "Industry": r["Industry"],
            "Year": r["Year"],
            "Current Price": cur,
            "Score": ev["score"],
            "Decision": "PASS" if ev["pass"] else "REJECT",
            "Unmet (mandatory)": "; ".join(ev["reasons"]) if ev["reasons"] else "",
        })

    dec_df = pd.DataFrame(dec_rows).sort_values(
        ["Decision","Score","Stock"], ascending=[True, False, True]
    )
    view_df = dec_df[dec_df["Decision"].eq("PASS")].reset_index(drop=True) if show_only_pass else dec_df.reset_index(drop=True)
    st.dataframe(view_df, use_container_width=True, height=280)

    if not view_df.empty:
        st.markdown("**Push PASS stocks to Trade Queue:**")
        cols = st.columns(min(4, len(view_df)))
        for i, row in view_df.iterrows():
            with cols[i % len(cols)]:
                if st.button(f"Push {row['Stock']}", key=f"push_{strategy}_{row['Stock']}"):
                    price_val = float(row["Current Price"]) if pd.notna(row["Current Price"]) else np.nan
                    io_helpers.push_trade_candidate(
                        name=row["Stock"],
                        strategy=strategy,
                        score=float(row["Score"]),
                        current_price=price_val,
                        reasons=row["Unmet (mandatory)"],
                    )
                    st.success(f"Pushed {row['Stock']} to Trade Queue")

    st.markdown("---")
    st.subheader("📋 Trade Queue")
    q = io_helpers.load_trade_queue()
    st.dataframe(q, use_container_width=True, height=240)

# ===== Exposure Summary & Planner =====
st.markdown("### ⚖️ Exposure Summary & Planner")

# Inputs for equity & caps
c1, c2, c3, c4 = st.columns(4)
with c1:
    equity = st.number_input("Account Equity", min_value=0.0, value=100000.0, step=1000.0, format="%.2f", key="acct_equity")
with c2:
    risk_pct = st.number_input("Risk % per trade", min_value=0.1, max_value=10.0, value=1.0, step=0.1, format="%.1f", key="risk_pct")
with c3:
    max_active = st.number_input("Max active names", min_value=1, max_value=100, value=10, step=1, key="max_active")
with c4:
    max_pos_pct = st.number_input("Max % equity per position", min_value=1.0, max_value=100.0, value=15.0, step=0.5, format="%.1f", key="max_pos_pct")

max_sector_pct = st.number_input("Max % equity per sector/industry", min_value=5.0, max_value=100.0, value=30.0, step=1.0, format="%.1f", key="max_sector_pct")

# Load positions (active) & compute exposures
pos = io_helpers.load_positions()
active_pos = pos[pos["Status"].isin(["Planned","Entered"])].copy()

total_capital = float(active_pos["Capital"].fillna(0).sum()) if not active_pos.empty else 0.0
active_count = int(len(active_pos))
pos_pct_total = (total_capital / equity * 100.0) if equity else 0.0

st.write(f"**Active positions:** {active_count} / {int(max_active)} | **Capital deployed:** {total_capital:,.2f} ({pos_pct_total:.2f}% of equity)")

# Sector/Industry exposure
if not active_pos.empty and equity:
    expo = (active_pos.groupby("Industry", dropna=False)["Capital"].sum() / equity * 100.0).sort_values(ascending=False)
    st.dataframe(expo.reset_index().rename(columns={"Industry":"Industry/Sector","Capital":"% of Equity"}),
                 use_container_width=True, height=180)

st.markdown("---")

# ===== Planner =====
st.markdown("### 📐 Plan a Position from Trade Queue")
if q.empty:
    st.info("Trade Queue is empty. Push PASS stocks above, then plan positions here.")
else:
    # map Name -> latest Industry from your data
    df_all = io_helpers.load_data()  # safe to reload here
    latest_ind = (df_all.sort_values(["Name","IsQuarter","Year","Quarter"])
                        .drop_duplicates("Name", keep="last")[["Name","Industry"]]
                        .set_index("Name")["Industry"].to_dict())

    # select queue item
    options = [f"{row['Name']}  —  {row['Strategy']}" for _, row in q.iterrows()]
    idx = st.selectbox("Choose a queued candidate", list(range(len(options))), format_func=lambda i: options[i], key="plan_pick")

    row = q.iloc[int(idx)]
    name = str(row["Name"])
    strategy = str(row["Strategy"])
    cur_price = float(row["CurrentPrice"]) if pd.notna(row["CurrentPrice"]) else float("nan")
    industry = latest_ind.get(name, "")

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        entry = st.number_input("Entry Price", min_value=0.0, value=(cur_price if pd.notna(cur_price) and cur_price>0 else 1.00), step=0.01, format="%.4f", key="plan_entry")
    with cc2:
        stop = st.number_input("Stop Price", min_value=0.0, value=max(0.01, round(entry*0.92, 4)), step=0.01, format="%.4f", key="plan_stop")
    with cc3:
        target = st.number_input("Target Price (optional)", min_value=0.0, value=round(entry*1.15, 4), step=0.01, format="%.4f", key="plan_target")

    risk_amt = round(equity * (risk_pct/100.0), 2)
    if stop >= entry:
        st.error("Stop must be **below** Entry for a long position.")
        can_save = False
        shares = 0
        capital = 0.0
        exp_r = None
    else:
        risk_per_share = entry - stop
        shares = int(math.floor(risk_amt / risk_per_share)) if risk_per_share > 0 else 0
        capital = round(shares * entry, 2)
        exp_r = round((target - entry) / (entry - stop), 2) if target and (entry - stop) > 0 else None
        can_save = shares > 0

    st.write(f"**Risk amount:** {risk_amt:,.2f}  |  **Shares:** {shares}  |  **Capital:** {capital:,.2f}" + (f"  |  **Expected R:** {exp_r}x" if exp_r is not None else ""))

    # Caps enforcement
    pos_pct = (capital / equity * 100.0) if equity else 0.0
    sector_now = 0.0
    if industry and not active_pos.empty and equity:
        sector_now = float(active_pos.loc[active_pos["Industry"].astype(str)==str(industry), "Capital"].sum()) / equity * 100.0
    sector_after = sector_now + pos_pct

    if active_count >= max_active:
        st.warning(f"Max active names reached: {active_count} / {int(max_active)}")
        can_save = False
    if pos_pct > max_pos_pct:
        st.warning(f"Position size would be {pos_pct:.2f}% of equity (cap {max_pos_pct:.1f}%).")
        can_save = False
    if industry and sector_after > max_sector_pct:
        st.warning(f"{industry} exposure would be {sector_after:.2f}% (cap {max_sector_pct:.1f}%).")
        can_save = False

    remove_when_saved = st.checkbox("Remove from Trade Queue after saving plan", value=True, key="plan_remove_queue")

    if st.button("💾 Save Plan", type="primary", disabled=not can_save, key=f"save_plan_{name}_{strategy}"):
        io_helpers.upsert_position(
            name=name,
            strategy=strategy,
            status="Planned",
            entry=float(entry),
            stop=float(stop),
            target=float(target) if target else None,
            shares=int(shares),
            capital=float(capital),
            risk_pct=float(risk_pct),
            equity=float(equity),
            risk_amt=float(risk_amt),
            expected_r=float(exp_r) if exp_r is not None else None,
            industry=str(industry),
            notes=""
        )
        if remove_when_saved:
            io_helpers.remove_trade_candidate(name=name, strategy=strategy, reason="Planned → Portfolio")
        st.success(f"Planned {name} ({strategy}).")

st.markdown("---")

# ===== Manage Queue: remove with reason =====
st.markdown("### 🧹 Manage Queue (remove with reason)")
if q.empty:
    st.info("No items to remove.")
else:
    REASONS_CANCEL = [
        "Fundamentals changed",
        "Valuation no longer attractive",
        "Technical breakdown",
        "Price ran away / missed entry",
        "Earnings / corporate action risk",
        "Liquidity / volume concern",
        "Risk constraints / exposure",
        "Better opportunity found",
        "Duplicate / already in portfolio",
        "Manual cancel",
    ]
    # Render per-row controls in a grid (up to 4 per row)
    cols = st.columns(min(4, len(q)))
    for i, (_, r) in enumerate(q.iterrows()):
        with cols[i % len(cols)]:
            st.write(f"**{r['Name']}**\n\n_{r['Strategy']}_")
            reason = st.selectbox("Reason", REASONS_CANCEL, key=f"remove_reason_{i}")
            if st.button("Remove from Queue", key=f"remove_btn_{i}"):
                ok = io_helpers.remove_trade_candidate(str(r["Name"]), str(r["Strategy"]), reason)
                if ok:
                    st.success("Removed.")
                else:
                    st.error("Could not remove (not found).")



    rows = []
    for _, r in latest.iterrows():
        ratio = calculations.calc_ratios(r)
        cur   = r.get("CurrentPrice", np.nan)
        if pd.isna(cur):
            cur = r.get("SharePrice", np.nan)
        rows.append({
            "Stock": r["Name"],
            "Industry": r["Industry"],
            "Year": r["Year"],
            "Current Price": cur,
            "Revenue": ratio.get("Revenue"),
            "NetProfit": ratio.get("NetProfit"),
            "EPS": ratio.get("EPS"),
            "ROE (%)": ratio.get("ROE (%)"),
            "P/E": ratio.get("P/E"),
            "P/B": ratio.get("P/B"),
            "Net Profit Margin (%)": ratio.get("Net Profit Margin (%)"),
            "Dividend Yield (%)": ratio.get("Dividend Yield (%)"),
            "LastModified": r.get("LastModified", "N/A")
        })
    summary = pd.DataFrame(rows).sort_values("Stock").reset_index(drop=True)
    # render centered header and plain table (no Edit column)
        # centered heading
    st.markdown(
        "<h4 style='text-align:center'>📌 Latest-year Summary</h4>",
        unsafe_allow_html=True
    )
    # Streamlit’s native DataFrame style
    st.dataframe(summary, use_container_width=True, height=320)



# ===============================================================
#  📊 Latest-quarter Summary (one row per stock)
# ===============================================================
# centered heading
st.markdown(
    "<h4 style='text-align:center'>📌 Latest-quarter Summary</h4>",
    unsafe_allow_html=True
)

quarterly = df[df["IsQuarter"]].copy()
if industry_sel != "All":
    quarterly = quarterly[quarterly["Industry"] == industry_sel]
if search_text.strip():
    lowq = search_text.lower()
    quarterly = quarterly[
        quarterly["Name"].str.lower().str.contains(lowq, na=False) |
        quarterly["Industry"].str.lower().str.contains(lowq, na=False)
    ]

latest_q = (
    quarterly
    .sort_values(["Name", "Year", "Quarter"])
    .groupby("Name", as_index=False)
    .tail(1)
)

qrows = []
for _, r in latest_q.iterrows():
    ratio = calculations.calc_ratios(r)
    cur = r.get("CurrentPrice", np.nan)
    if pd.isna(cur):
        cur = r.get("Q_EndQuarterPrice", np.nan)
    qrows.append({
        "Stock": r["Name"],
        "Industry": r["Industry"],
        "Year": r["Year"],
        "Quarter": r["Quarter"],
        "Current Price": cur,
        "Revenue": ratio.get("Revenue"),
        "NetProfit": ratio.get("NetProfit"),
        "EPS": ratio.get("EPS"),
        "ROE (%)": ratio.get("ROE (%)"),
        "P/E": ratio.get("P/E"),
        "P/B": ratio.get("P/B"),
        "Net Profit Margin (%)": ratio.get("Net Profit Margin (%)"),
        "Dividend Yield (%)": ratio.get("Dividend Yield (%)"),
        "LastModified": r["LastModified"]  # now per‑row
    })

qsummary = pd.DataFrame(qrows).sort_values("Stock").reset_index(drop=True)
st.dataframe(qsummary, use_container_width=True, height=320)




# ===============================================================
#  B) 🎯 Watchlist & Target Price
# ===============================================================
st.subheader("🎯 Watchlist & Target Price")

# Latest current price per stock: prefer CurrentPrice anywhere in df,
# otherwise fall back to latest annual SharePrice.
pref_current = (
    df.dropna(subset=["CurrentPrice"])
      .sort_values(["Name", "Year"])
      .groupby("Name", as_index=False)
      .tail(1)[["Name", "CurrentPrice"]]
      .set_index("Name")
)

fallback_from_annual = (
    annual.sort_values(["Name", "Year"])
          .groupby("Name", as_index=False)
          .tail(1)[["Name", "SharePrice"]]
          .rename(columns={"SharePrice": "CurrentPrice"})
          .set_index("Name")
)

latest_prices = fallback_from_annual.copy()
for n, row in pref_current.iterrows():
    latest_prices.loc[n, "CurrentPrice"] = row["CurrentPrice"]

# ---- Add / Update form
with st.form("watchlist_add_form", clear_on_submit=True):
    c1, c2 = st.columns([2,1])
    with c1:
        name_input = st.text_input(
            "Stock to watch",
            placeholder="Type a name (matches existing stock names or add custom)",
            help="Type an existing stock name to link the current price, or a custom one."
        )
    with c2:
        target_input = st.number_input("Target price", min_value=0.0, step=0.0001, format="%.4f")
    notes_input = st.text_input("Notes (optional)", placeholder="e.g., buy zone, catalyst, etc.")
    submitted = st.form_submit_button("Add / Update")

if submitted:
    wl = io_helpers.load_watchlist()
    if not name_input:
        st.error("Please enter a stock name for the watchlist.")
    else:
        # Upsert
        exists = wl["Name"].str.lower().eq(name_input.lower()).any()
        if exists:
            wl.loc[wl["Name"].str.lower() == name_input.lower(), ["TargetPrice", "Notes", "Active"]] = [target_input, notes_input, True]
        else:
            wl = pd.concat([
                wl,
                pd.DataFrame([{"Name": name_input, "TargetPrice": target_input, "Notes": notes_input, "Active": True}])
            ], ignore_index=True)
        io_helpers.save_watchlist(wl)
        st.success(f"Watchlist updated for '{name_input}'.")
        st.experimental_rerun()  # show the updated list immediately

# ---- Editor
wl = io_helpers.load_watchlist()
if wl.empty:
    st.info("No items in your watchlist yet. Use the form above to add one.")
else:
    # Enrich with computed cols (ensure proper dtypes for editor)
    disp = wl.copy()
    disp.insert(0, "Delete", False)

    # Ensure text dtype for Notes to avoid Streamlit type mismatch
    disp["Notes"] = disp["Notes"].astype("string").fillna("")

    # Map current price from latest_prices; keep NaN if not available
    disp["CurrentPrice"] = disp["Name"].map(lambda n: latest_prices.loc[n, "CurrentPrice"] if n in latest_prices.index else np.nan)

    # numeric conversions
    disp["TargetPrice"]   = pd.to_numeric(disp["TargetPrice"], errors="coerce")
    disp["CurrentPrice"]  = pd.to_numeric(disp["CurrentPrice"], errors="coerce")

    # upside (avoid division by zero)
    disp["Upside %"] = np.where(
        disp["CurrentPrice"] > 0,
        (disp["TargetPrice"] - disp["CurrentPrice"]) / disp["CurrentPrice"] * 100.0,
        np.nan
    ).round(2)

    edited = st.data_editor(
        disp,
        use_container_width=True,
        height=420,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "Delete":       st.column_config.CheckboxColumn("Delete"),
            "Name":         st.column_config.TextColumn("Name"),
            "TargetPrice":  st.column_config.NumberColumn("Target Price", format="%.4f"),
            "CurrentPrice": st.column_config.NumberColumn("Current Price", disabled=True, format="%.4f"),
            "Upside %":     st.column_config.NumberColumn("Upside %", disabled=True, help="(Target - Current) / Current * 100"),
            "Notes":        st.column_config.TextColumn("Notes"),
            "Active":       st.column_config.CheckboxColumn("Active"),
        },
        key="watchlist_editor",
    )

    if st.button("💾 Save Watchlist"):
        keep = edited[edited["Delete"] != True].copy()
        # Normalize types before saving
        keep["Notes"]       = keep["Notes"].astype("string").fillna("")
        keep["TargetPrice"] = pd.to_numeric(keep["TargetPrice"], errors="coerce")
        keep["Active"]      = keep["Active"].fillna(True).astype(bool)
        keep = keep[["Name","TargetPrice","Notes","Active"]]
        io_helpers.save_watchlist(keep)
        st.success("Watchlist saved.")
        st.experimental_rerun()
