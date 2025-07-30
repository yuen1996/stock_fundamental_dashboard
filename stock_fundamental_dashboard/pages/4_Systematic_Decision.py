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
st.caption("Strict rules – PASS only when all mandatory checks succeed and score ≥ "
           f"{rules.MIN_SCORE}%.")

# 1) Load latest ANNUAL row per stock
df = io_helpers.load_data()
latest = (df[~df["IsQuarter"]]
          .sort_values(["Name", "Year"])
          .groupby("Name", as_index=False)
          .tail(1)
          .reset_index(drop=True))

# Strategy choice (no other knobs)
strategy = st.selectbox("Strategy / Playbook", list(rules.RULESETS), index=0)
st.divider()

# 2) Evaluate
rows = []
for _, row in latest.iterrows():
    metrics = calculations.calc_ratios(row)
    cur = row.get("CurrentPrice", np.nan)
    cur = cur if pd.notna(cur) else row.get("SharePrice", np.nan)
    ev = rules.evaluate(metrics, strategy)
    rows.append({
        "Name":         row["Name"],
        "Industry":     row["Industry"],
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
st.dataframe(dec_df, use_container_width=True, height=340)

# 3) Push PASS candidates only
pass_df = dec_df[dec_df["Decision"].eq("PASS")]
if not pass_df.empty:
    st.markdown("### Push PASS stocks to Trade Queue")
    cols = st.columns(min(4, len(pass_df)))
    for i, r in pass_df.iterrows():
        with cols[i % len(cols)]:
            if st.button(f"Push {r['Name']}", key=f"push_{r['Name']}"):
                io_helpers.push_trade_candidate(
                    name=r["Name"],
                    strategy=strategy,
                    score=float(r["Score"]),
                    current_price=r["CurrentPrice"],
                    reasons=r["Unmet"]
                )
                st.success(f"Added **{r['Name']}** to Trade Queue")

st.divider()
st.subheader("📋 Current Trade Queue")
st.dataframe(io_helpers.load_trade_queue(), use_container_width=True, height=280)
