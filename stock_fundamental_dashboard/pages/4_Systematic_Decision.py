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

# 3) Actions for PASS candidates — TABLE VERSION
pass_df = dec_df[dec_df["Decision"].eq("PASS")].copy()
if not pass_df.empty:
    st.markdown("### Actions for PASS Stocks (Table)")
    # Prepare table with action columns
    pass_df = pass_df[["Name", "Industry", "Year", "CurrentPrice", "Score", "Unmet"]].copy()
    pass_df.insert(0, "SelectPush", False)
    pass_df.insert(1, "SelectPlan", False)
    pass_df["Strategy"] = strategy  # allow override per-row if you like later

    edited_pass = st.data_editor(
        pass_df,
        use_container_width=True,
        height=320,
        hide_index=True,
        column_config={
            "SelectPush":  st.column_config.CheckboxColumn("Push"),
            "SelectPlan":  st.column_config.CheckboxColumn("Plan"),
            "CurrentPrice": st.column_config.NumberColumn("Current Price", format="%.4f", disabled=True),
            "Score":        st.column_config.NumberColumn("Score", format="%.0f", disabled=True),
            "Unmet":        st.column_config.TextColumn("Reasons (auto from evaluation)", disabled=True),
            "Strategy":     st.column_config.TextColumn("Strategy", disabled=True),
        },
        key="pass_actions_editor",
    )

    c1, c2, c3 = st.columns([1.2, 1.2, 3])
    with c1:
        if st.button("➕ Push selected to Queue"):
            pushed = 0
            for _, r in edited_pass.iterrows():
                if r.get("SelectPush"):
                    io_helpers.push_trade_candidate(
                        name=r["Name"],
                        strategy=strategy,
                        score=float(r["Score"]),
                        current_price=float(r["CurrentPrice"]) if pd.notna(r["CurrentPrice"]) else None,
                        reasons=str(r.get("Unmet") or "")
                    )
                    pushed += 1
            st.success(f"Pushed {pushed} stock(s) to Trade Queue.")
            try: st.rerun()
            except Exception: st.experimental_rerun()
    with c2:
        if st.button("📐 Plan selected (first)"):
            chosen = edited_pass[edited_pass["SelectPlan"] == True]
            if chosen.empty:
                st.warning("Select one row to plan.")
            else:
                r = chosen.iloc[0]
                params = {
                    "stock":    r["Name"],
                    "strategy": strategy,
                    "entry":    float(r["CurrentPrice"]) if pd.notna(r["CurrentPrice"]) else "",
                    "r":        2.0,
                    "riskpct":  1.0,
                    "lot":      100,
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
tq = io_helpers.load_trade_queue().copy()

# 4) Manage Queue — TABLE VERSION (bulk actions)
st.markdown("### 🔧 Manage Queue (table: Mark Live / Delete with reason)")
if tq.empty:
    st.info("Queue is empty.")
else:
    # Prepare table
    tq["RR"]    = pd.to_numeric(tq.get("RR"), errors="coerce")
    tq["Entry"] = pd.to_numeric(tq.get("Entry"), errors="coerce")

    # Default reason heuristic
    def _default_reason(rr):
        try:
            rr = float(rr)
            return "R:R below threshold" if rr < 1.5 else "Duplicate idea"
        except Exception:
            return "Duplicate idea"

    DELETE_REASONS = [
        "Duplicate idea",
        "Fails rules on recheck",
        "R:R below threshold",
        "Market conditions changed",
        "Wrong symbol / data error",
        "Moved to Watchlist",
        "Other (specify)",
    ]

    table = tq[["Name","Strategy","Entry","RR","Timestamp","Reasons"]].copy()
    table.insert(0, "Select", False)
    table["Reason"] = [ _default_reason(x) for x in table["RR"] ]
    table["Detail"] = ""

    edited_q = st.data_editor(
        table,
        use_container_width=True,
        height=340,
        hide_index=True,
        column_config={
            "Select":   st.column_config.CheckboxColumn("Select"),
            "Entry":    st.column_config.NumberColumn("Entry", format="%.4f", disabled=True),
            "RR":       st.column_config.NumberColumn("RR", format="%.2f", disabled=True),
            "Timestamp":st.column_config.TextColumn("Added", disabled=True),
            "Reasons":  st.column_config.TextColumn("Notes/Reasons", disabled=True),
            "Reason":   st.column_config.SelectboxColumn("Delete Reason", options=DELETE_REASONS),
            "Detail":   st.column_config.TextColumn("Detail (if Other)"),
        },
        key="queue_manage_editor",
    )

    c1, c2, c3 = st.columns([1.2, 1.2, 3])
    with c1:
        if st.button("✅ Mark Live selected"):
            moved = 0
            for _, r in edited_q.iterrows():
                if r.get("Select"):
                    ok = io_helpers.mark_live_from_queue(name=r["Name"], strategy=r["Strategy"])
                    if ok: moved += 1
            st.success(f"Marked live: {moved} row(s).")
            try: st.rerun()
            except Exception: st.experimental_rerun()

    with c2:
        if st.button("🗑️ Delete selected"):
            deleted, missing, invalid = 0, 0, 0
            for _, r in edited_q.iterrows():
                if not r.get("Select"): 
                    continue
                reason = r.get("Reason") or ""
                det    = r.get("Detail") or ""
                if reason == "Other (specify)" and not det.strip():
                    invalid += 1
                    continue
                audit_reason = reason if reason != "Other (specify)" else f"{reason}: {det.strip()}"
                ok = io_helpers.delete_trade_candidate(name=r["Name"], strategy=r["Strategy"], audit_reason=audit_reason)
                if ok: 
                    deleted += 1
                else:
                    missing += 1
            msg = f"Deleted {deleted} row(s)."
            if missing: msg += f" {missing} not found."
            if invalid: msg += f" {invalid} skipped (missing detail for 'Other')."
            st.success(msg)
            try: st.rerun()
            except Exception: st.experimental_rerun()
