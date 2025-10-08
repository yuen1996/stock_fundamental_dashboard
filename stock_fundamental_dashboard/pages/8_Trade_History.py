# pages/8_Trade_History.py
from __future__ import annotations

from auth_gate import require_auth
require_auth()

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# UI helpers (reuse the same CSS as Ongoing Trades)
try:
    from utils.ui import (
        setup_page,
        section,
        register_ongoing_trades_css,
        render_stat_cards,
        render_page_title,
    )
except Exception:
    from ui import (
        setup_page,
        section,
        register_ongoing_trades_css,
        render_stat_cards,
    )  # type: ignore
    try:
        from ui import render_page_title  # type: ignore
    except Exception:
        def render_page_title(page_name: str) -> None:
            st.title(f"📊 Fundamentals Dashboard — {page_name}")

# IO helpers
try:
    from utils import io_helpers as ioh
except Exception:
    import io_helpers as ioh  # type: ignore


# ─────────────────────────────────────────
# Page setup + CSS
# ─────────────────────────────────────────
setup_page("Trade History (Closed)")
render_page_title("Trade History (Closed)")
try:
    register_ongoing_trades_css()
except Exception:
    pass


# --- Pretty styles for the Inspect dialog (same as Ongoing Trades) ---
st.markdown(
    """
<style>
.inspect { padding-top:.25rem; }
.inspect .muted { color:#6b7280;font-size:.9rem;margin:0 0 .5rem 0; }
.inspect .kpi { text-align:center; padding:.35rem .25rem .8rem; }
.inspect .kpi h4 { margin:.15rem 0 .25rem 0; font-weight:600; color:#6b7280; }
.inspect .kpi .v  { font-size:2rem; font-weight:700; letter-spacing:.2px; }
.inspect .pos { color:#10b981; }   /* green */
.inspect .neg { color:#ef4444; }   /* red   */
.inspect .pill { display:inline-block; padding:.15rem .5rem; border-radius:999px; font-weight:600; font-size:.85rem; }
.inspect .pill.neutral { background:#e5e7eb; color:#111827; }
.inspect .pill.good { background:#d1fae5; color:#065f46; }
.inspect .pill.bad  { background:#fee2e2; color:#7f1d1d; }
.inspect .subtle { color:#6b7280; font-size:.85rem; }
</style>
""",
    unsafe_allow_html=True,
)


def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass


# ─────────────────────────────────────────
# Load closed trades and add visible RowId
# ─────────────────────────────────────────
hist = ioh.load_closed_trades()
if hist is None or hist.empty:
    st.info("No closed trades yet. Close some positions from **Ongoing Trades** to see them here.")
    st.stop()

hist = hist.reset_index().rename(columns={"index": "RowId"})

# Coerce some numeric columns early (robust to missing cols)
to_num = lambda c: pd.to_numeric(hist.get(c), errors="coerce")
hist["PnL"]         = to_num("PnL")
hist["ReturnPct"]   = to_num("ReturnPct")
hist["RMultiple"]   = to_num("RMultiple")
hist["HoldingDays"] = to_num("HoldingDays")
hist["ClosePrice"]  = to_num("ClosePrice")
hist["Entry"]       = to_num("Entry")
hist["Stop"]        = to_num("Stop")
hist["Shares"]      = to_num("Shares")
hist["RR_Init"]     = to_num("RR_Init")


# ─────────────────────────────────────────
# Filters
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec"><div class="t">🔎 Filters</div>'
    '<div class="d">Narrow the history to analyze performance</div></div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns([2, 1, 1, 1.2])
with c1:
    q = st.text_input("Search name", placeholder="Type part of a stock name…").strip().lower()
with c2:
    strat_opts = ["All"] + sorted([s for s in hist.get("Strategy", pd.Series(dtype=str)).dropna().unique()])
    strat = st.selectbox("Strategy", strat_opts, index=0)
with c3:
    outcome = st.selectbox("Outcome", ["All", "Winners (>0)", "Losers (≤0)"], index=0)
with c4:
    period = st.selectbox("Closed in", ["Any", "Last 7 days", "Last 14 days", "Last 1 month", "Last 3 months"], index=0)

filtered = hist.copy()

if q:
    filtered = filtered[filtered["Name"].astype(str).str.lower().str.contains(q, na=False)]
if strat != "All":
    filtered = filtered[filtered["Strategy"] == strat]
if outcome != "All":
    pnl = pd.to_numeric(filtered.get("PnL"), errors="coerce")
    ret = pd.to_numeric(filtered.get("ReturnPct"), errors="coerce")
    wins = (pnl > 0) | ((pnl.isna()) & (ret > 0))
    filtered = filtered[wins.fillna(False)] if outcome == "Winners (>0)" else filtered[~wins.fillna(False)]

if period != "Any":
    now = datetime.now()
    if period == "Last 7 days":
        cutoff = now - timedelta(days=7)
    elif period == "Last 14 days":
        cutoff = now - timedelta(days=14)
    elif period == "Last 1 month":
        cutoff = now - timedelta(days=30)
    else:
        cutoff = now - timedelta(days=90)
    dt = pd.to_datetime(filtered.get("CloseDate"), errors="coerce")
    filtered = filtered[dt >= cutoff]

if filtered.empty:
    st.info("No rows match the current filters.")
    st.stop()


# ─────────────────────────────────────────
# Overview KPI cards
# ─────────────────────────────────────────
st.markdown(section("📊 Overview", "Performance for the filtered set"), unsafe_allow_html=True)

pnl_sum  = float(pd.to_numeric(filtered["PnL"], errors="coerce").sum(skipna=True))
ret_mean = pd.to_numeric(filtered["ReturnPct"], errors="coerce")
ret_mean = float(ret_mean.dropna().mean()) if ret_mean.notna().any() else None
r_mean   = pd.to_numeric(filtered["RMultiple"], errors="coerce")
r_mean   = float(r_mean.dropna().mean()) if r_mean.notna().any() else None
hold_avg = pd.to_numeric(filtered["HoldingDays"], errors="coerce")
hold_avg = int(hold_avg.dropna().mean()) if hold_avg.notna().any() else None

wins = (pd.to_numeric(filtered["PnL"], errors="coerce") > 0) | (
    pd.to_numeric(filtered["PnL"], errors="coerce").isna() & (pd.to_numeric(filtered["ReturnPct"], errors="coerce") > 0)
)
win_rate = float(100.0 * wins.mean()) if not filtered.empty else None

def _tone_pnl(v):
    if v > 0: return "good"
    if v < 0: return "bad"
    return "neutral"

render_stat_cards(
    [
        {"label": "Trades", "value": f"{len(filtered):,}", "badge": "Closed"},
        {"label": "Gross P&L (RM)", "value": f"{pnl_sum:,.2f}", "badge": "Total", "tone": _tone_pnl(pnl_sum)},
        {"label": "Win rate", "value": f"{win_rate:.1f}%" if win_rate is not None else "—", "badge": "Hit %"},
        {"label": "Avg Return (%)", "value": f"{ret_mean:.2f}" if ret_mean is not None else "—", "badge": "Mean"},
        {"label": "Avg R multiple", "value": f"{r_mean:.2f}" if r_mean is not None else "—", "badge": "Mean"},
        {"label": "Avg Holding (d)", "value": f"{hold_avg:,}" if hold_avg is not None else "—", "badge": "Duration"},
    ],
    columns=3,
)


# ─────────────────────────────────────────
# History table
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec success"><div class="t">🧾 Closed Trades</div>'
    '<div class="d">Browse, export, or remove selected rows from history</div></div>',
    unsafe_allow_html=True,
)

table = filtered.copy()
table.insert(0, "Select", False)

edited = st.data_editor(
    table,
    use_container_width=True,
    height=560,
    hide_index=True,
    column_config={
        "Select":      st.column_config.CheckboxColumn("Sel"),

        "RowId":       st.column_config.NumberColumn("RowId", disabled=True),
        "Name":        st.column_config.TextColumn("Name", disabled=True),
        "Strategy":    st.column_config.TextColumn("Strategy", disabled=True),

        "Entry":       st.column_config.NumberColumn("Entry", format="%.4f", disabled=True),
        "Stop":        st.column_config.NumberColumn("Stop",  format="%.4f", disabled=True),
        "Take":        st.column_config.NumberColumn("Take",  format="%.4f", disabled=True),
        "Shares":      st.column_config.NumberColumn("Shares", format="%d",   disabled=True),
        "RR_Init":     st.column_config.NumberColumn("RR Init", format="%.2f", disabled=True),

        "TP1":         st.column_config.NumberColumn("TP1", format="%.4f", disabled=True),
        "TP2":         st.column_config.NumberColumn("TP2", format="%.4f", disabled=True),
        "TP3":         st.column_config.NumberColumn("TP3", format="%.4f", disabled=True),

        "OpenDate":    st.column_config.TextColumn("Open Date",  disabled=True),
        "CloseDate":   st.column_config.TextColumn("Close Date", disabled=True),
        "ClosePrice":  st.column_config.NumberColumn("Close Price", format="%.4f", disabled=True),
        "HoldingDays": st.column_config.NumberColumn("Days", format="%d", disabled=True),

        "PnL":         st.column_config.NumberColumn("P&L (RM)", format="%.2f", disabled=True),
        "ReturnPct":   st.column_config.NumberColumn("Return (%)", format="%.2f", disabled=True),
        "RMultiple":   st.column_config.NumberColumn("R multiple", format="%.2f", disabled=True),

        "CloseReason": st.column_config.TextColumn("Close Reason", disabled=True),
        "Reasons":     st.column_config.TextColumn("Notes", disabled=True),
    },
    column_order=[
        "Select", "RowId", "Name", "Strategy",
        "Entry", "Stop", "Take", "Shares", "RR_Init",
        "TP1", "TP2", "TP3",
        "OpenDate", "CloseDate", "ClosePrice", "HoldingDays",
        "PnL", "ReturnPct", "RMultiple",
        "CloseReason", "Reasons",
    ],
    key="closed_trades_editor",
)


# ─────────────────────────────────────────
# Quick inspect dialog for first selected row (styled like Ongoing)
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec info"><div class="t">🔍 Inspect</div>'
    '<div class="d">Quick breakdown for the first selected row</div></div>',
    unsafe_allow_html=True,
)

def _open_inspect_dialog(row: pd.Series):
    title = f"{row.get('Name')} — {row.get('Strategy')}"
    # Values
    entry = row.get("Entry")
    close = row.get("ClosePrice")
    rm    = row.get("RMultiple")
    pnl   = row.get("PnL")
    ret   = row.get("ReturnPct")
    days  = int(row.get("HoldingDays") or 0)
    tp1   = row.get("TP1")
    tp2   = row.get("TP2")
    tp3   = row.get("TP3")
    sl    = row.get("Stop")

    pill_tone = "neutral"
    if pd.notna(rm):
        pill_tone = "good" if rm >= 1.0 else ("neutral" if rm >= 0 else "bad")
    pnl_cls = "pos" if (isinstance(pnl, (int,float)) and pnl >= 0) else "neg"

    if hasattr(st, "dialog"):
        @st.dialog(title)
        def _dlg():
            st.markdown(
                f"<div class='inspect'><div class='muted'>Opened: {row.get('OpenDate','—')} • Closed: {row.get('CloseDate','—')}</div></div>",
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    "<div class='inspect kpi'><h4>Entry</h4>"
                    f"<div class='v'>{('—' if pd.isna(entry) else f'{entry:.4f}')}</div></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    "<div class='inspect kpi'><h4>Close</h4>"
                    f"<div class='v'>{('—' if pd.isna(close) else f'{close:.4f}')}</div></div>",
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    "<div class='inspect kpi'><h4>R multiple</h4>"
                    f"<div class='v'><span class='pill {pill_tone}'>{'—' if pd.isna(rm) else f'{rm:.2f}'}</span></div></div>",
                    unsafe_allow_html=True,
                )

            c4, c5, c6 = st.columns(3)
            with c4:
                st.markdown(
                    "<div class='inspect kpi'><h4>P&L (RM)</h4>"
                    f"<div class='v {pnl_cls}'>{'—' if pd.isna(pnl) else f'{pnl:,.2f}'}</div></div>",
                    unsafe_allow_html=True,
                )
            with c5:
                st.markdown(
                    "<div class='inspect kpi'><h4>Return (%)</h4>"
                    f"<div class='v'>{'—' if pd.isna(ret) else f'{ret:.2f}'}</div></div>",
                    unsafe_allow_html=True,
                )
            with c6:
                st.markdown(
                    "<div class='inspect kpi'><h4>Days</h4>"
                    f"<div class='v'>{days}</div></div>",
                    unsafe_allow_html=True,
                )

            st.caption(
                f"TP1 {tp1 if pd.notna(tp1) else '—'} | "
                f"TP2 {tp2 if pd.notna(tp2) else '—'} | "
                f"TP3 {tp3 if pd.notna(tp3) else '—'} | SL {sl if pd.notna(sl) else '—'}"
            )

            reason = str(row.get("CloseReason", "") or "").strip()
            notes  = str(row.get("Reasons", "") or "").strip()
            if reason or notes:
                st.divider()
                st.write(f"**Reason:** {reason or '—'}")
                if notes:
                    st.write(notes)
        _dlg()
    else:
        # Fallback if dialog isn't available
        st.info("Update Streamlit (≥1.31) to show dialogs. Showing inline:")
        cols = ["Name","Strategy","Entry","ClosePrice","PnL","ReturnPct","RMultiple","HoldingDays","CloseReason","Reasons"]
        st.write(row[cols])


if st.button("📌 Inspect first selected"):
    sel = edited[edited.get("Select") == True]
    if sel.empty:
        st.info("Tick a row first.")
    else:
        _open_inspect_dialog(sel.iloc[0])


# ─────────────────────────────────────────
# Actions: Export / Delete
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec warning"><div class="t">⬇️ Export / 🗑️ Delete</div>'
    '<div class="d">Download filtered rows or remove selected rows from history</div></div>',
    unsafe_allow_html=True,
)

c1, c2 = st.columns([1.4, 1.4])

with c1:
    csv = filtered.drop(columns=["Select"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered (CSV)",
        data=csv,
        file_name="trade_history_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

with c2:
    if st.button("🗑️ Delete selected from history", use_container_width=True):
        sel_ids = [int(r.RowId) for _, r in edited.iterrows() if r.Select]
        if not sel_ids:
            st.info("Nothing selected.")
        else:
            df_now = ioh.load_closed_trades()
            if df_now is None or df_now.empty:
                st.warning("History is already empty.")
            else:
                keep = df_now.drop(index=[i for i in sel_ids if 0 <= i < len(df_now)], errors="ignore")
                ioh.save_closed_trades(keep.reset_index(drop=True))
                st.success(f"Deleted {len(sel_ids)} row(s) from history.")
                _safe_rerun()
