# 4_Systematic_Decision.py

# --- path patch: allow imports from both package root and repo root ---
import os, sys
PACKAGE_ROOT = os.path.dirname(os.path.dirname(__file__))      # parent of /pages
REPO_ROOT    = os.path.dirname(PACKAGE_ROOT)                   # parent of package
for p in (PACKAGE_ROOT, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
# ---------------------------------------------------------------------

import streamlit as st, pandas as pd, numpy as np

# --- robust imports: prefer package (utils), fall back to top-level ---
try:
    from utils import calculations, io_helpers, rules
except Exception:
    import calculations
    import io_helpers
    import rules

st.set_page_config(page_title="Systematic Decision", layout="wide")

st.title("🚦 Systematic Decision Engine")
st.caption("PASS only when all mandatory checks succeed and score ≥ "
           f"{rules.MIN_SCORE}% (per selected ruleset).")

# 1) Load latest ANNUAL row per stock
df = io_helpers.load_data()
if df is None or df.empty or "Name" not in df.columns:
    st.warning("No data found. Please add stocks in **Add/Edit** first.")
    st.stop()

annual_only = df[df["IsQuarter"] != True].copy()
if annual_only.empty:
    st.info("No annual rows available.")
    st.stop()

latest = (annual_only
          .sort_values(["Name", "Year"])
          .groupby("Name", as_index=False)
          .tail(1)
          .reset_index(drop=True))

# Strategy choice
strategy = st.selectbox("Strategy / Playbook", list(rules.RULESETS), index=0)
st.divider()

# 2) Evaluate
rows = []
for _, row in latest.iterrows():
    metrics = calculations.calc_ratios(row)
    cur = row.get("CurrentPrice", np.nan)
    if pd.isna(cur):
        cur = row.get("SharePrice", np.nan)
    ev = rules.evaluate(metrics, strategy)
    rows.append({
        "Name":         row["Name"],
        "Industry":     row.get("Industry", ""),
        "Year":         int(row["Year"]),
        "CurrentPrice": cur,
        "Score":        ev["score"],
        "Decision":     "PASS" if ev["pass"] else "REJECT",
        "Unmet":        "; ".join(ev["reasons"]),
    })

dec_df = (pd.DataFrame(rows)
          .sort_values(["Decision", "Score", "Name"], ascending=[True, False, True])
          .reset_index(drop=True))

st.subheader("Evaluation Result")
st.dataframe(dec_df, use_container_width=True, height=380)

# 3) Actions for PASS candidates
pass_df = dec_df[dec_df["Decision"].eq("PASS")]
if not pass_df.empty:
    st.markdown("### Actions for PASS Stocks")

    # lay out in responsive cards
    cols = st.columns(min(3, max(1, len(pass_df))))
    for i, r in pass_df.reset_index(drop=True).iterrows():
        with cols[i % len(cols)]:
            st.markdown(f"**{r['Name']}**  \nScore: **{r['Score']}%**")
            price_str = f"{float(r['CurrentPrice']):,.4f}" if pd.notna(r["CurrentPrice"]) else "N/A"
            st.caption(f"Current Price: {price_str}")

            # Buttons: Push to Queue now OR Plan Trade (open Planner pre-filled)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Push Queue", key=f"push_{r['Name']}"):
                    io_helpers.push_trade_candidate(
                        name=r["Name"],
                        strategy=strategy,
                        score=float(r["Score"]),
                        current_price=float(r["CurrentPrice"]) if pd.notna(r["CurrentPrice"]) else None,
                        reasons=r["Unmet"]
                    )
                    st.success(f"Added **{r['Name']}** to Trade Queue")

            with c2:
                if st.button("Plan Trade", key=f"plan_{r['Name']}"):
                    # Build query params to prefill Planner
                    params = {
                        "stock":    r["Name"],
                        "strategy": strategy,
                        "entry":    float(r["CurrentPrice"]) if pd.notna(r["CurrentPrice"]) else "",
                        "r":        2.0,    # default R multiple
                        "riskpct":  1.0,    # default risk per trade (%)
                        "lot":      100,    # default lot size
                    }
                    # Update query params and navigate to Planner
                    try:
                        st.query_params.clear()
                        st.query_params.update(params)
                    except Exception:
                        st.experimental_set_query_params(**params)
                    try:
                        st.switch_page("pages/5_Risk_Reward_Planner.py")
                    except Exception:
                        st.info("Open **Risk / Reward Planner** from the sidebar — it is pre-filled via URL parameters.")

st.divider()
st.subheader("📋 Current Trade Queue")
st.dataframe(io_helpers.load_trade_queue(), use_container_width=True, height=300)

