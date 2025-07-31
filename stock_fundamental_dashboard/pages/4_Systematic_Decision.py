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
                    params = {
                        "stock":    r["Name"],
                        "strategy": strategy,
                        "entry":    float(r["CurrentPrice"]) if pd.notna(r["CurrentPrice"]) else "",
                        "r":        2.0,     # default target R
                        "riskpct":  1.0,     # default risk per trade
                        "lot":      100      # default lot size
                    }
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
tq = io_helpers.load_trade_queue()
st.dataframe(tq, use_container_width=True, height=300)

st.markdown("### 🔧 Manage Queue (Mark Live / Delete with reason)")
if tq.empty:
    st.info("Queue is empty.")
else:
    DELETE_REASONS = [
        "Duplicate idea",
        "Fails rules on recheck",
        "R:R below threshold",
        "Market conditions changed",
        "Wrong symbol / data error",
        "Moved to Watchlist",
        "Other (specify)",
    ]
    st.caption("To **open** a position, use **Mark Live**. To **delete** a queued plan, choose a reason and confirm. All changes are audited.")

    for i, row in tq.reset_index(drop=True).iterrows():
        with st.container():
            c1, c2, c3, c4, c5, c6 = st.columns([2, 1.2, 1.2, 1.6, 1.4, 1.4])
            name      = row.get("Name")
            strat     = row.get("Strategy")
            entry     = row.get("Entry")
            rr        = row.get("RR")
            ts        = row.get("Timestamp")

            c1.markdown(f"**{name}**  \n_Strategy:_ {strat}")
            c2.write(f"Entry: {entry:,.4f}" if pd.notna(entry) else "Entry: N/A")
            c3.write(f"RR: {rr:.2f}" if pd.notna(rr) else "RR: N/A")
            c4.write(f"Added: {ts}")

            # ── Mark Live
            if c5.button("✅ Mark Live", key=f"mlive_{i}_{name}_{strat}"):
                ok = io_helpers.mark_live_from_queue(name=name, strategy=strat)
                if ok:
                    st.success(f"Moved to **Ongoing Trades**: {name} ({strat})")
                    try: st.rerun()
                    except Exception: st.experimental_rerun()
                else:
                    st.error("Could not mark live (row not found).")

            # ── Delete with reason
            with c6:
                reason_sel = st.selectbox(
                    "Reason",
                    DELETE_REASONS,
                    index=2 if (pd.notna(rr) and float(rr) < 1.5) else 0,
                    key=f"del_reason_{i}_{name}_{strat}"
                )
                detail_needed = reason_sel == "Other (specify)"
                detail = ""
                if detail_needed:
                    detail = st.text_input("Detail", key=f"del_detail_{i}_{name}_{strat}")
                can_delete = (reason_sel and (not detail_needed or (detail.strip())))
                if st.button("🗑️ Delete", key=f"btn_del_{i}_{name}_{strat}", disabled=not can_delete):
                    audit_reason = reason_sel if not detail_needed else f"{reason_sel}: {detail.strip()}"
                    ok = io_helpers.delete_trade_candidate(name=name, strategy=strat, audit_reason=audit_reason)
                    if ok:
                        st.success(f"Deleted from queue: {name} ({strat}) — reason recorded.")
                        try: st.rerun()
                        except Exception: st.experimental_rerun()
                    else:
                        st.error("Row not found (maybe already deleted).")


