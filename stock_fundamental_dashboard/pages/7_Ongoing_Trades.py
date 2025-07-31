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
