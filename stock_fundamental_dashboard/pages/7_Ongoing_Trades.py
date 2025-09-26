# pages/7_Ongoing_Trades.py
from __future__ import annotations

from auth_gate import require_auth
require_auth()

import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Shared UI helpers
try:
    from utils.ui import (
        setup_page,
        section,
        register_ongoing_trades_css,
        render_stat_cards,
        render_page_title,
    )
except Exception:
    # fallback if ui.py is at project root
    from ui import (
        setup_page,
        section,
        register_ongoing_trades_css,
        render_stat_cards,  # make sure we bring this too
    )  # type: ignore
    try:
        from ui import render_page_title  # type: ignore
    except Exception:
        def render_page_title(page_name: str) -> None:
            st.title(f"📊 Fundamentals Dashboard — {page_name}")

# Robust import for IO helpers
try:
    from utils import io_helpers
except Exception:
    import io_helpers  # type: ignore

try:
    from utils.dividend_helpers import (
        dividend_data_version,
        get_latest_dividend_yield,
    )
except Exception:  # pragma: no cover - fallback for legacy entrypoints
    from dividend_helpers import (  # type: ignore
        dividend_data_version,
        get_latest_dividend_yield,
    )

# ---------- Page setup ----------
setup_page("Ongoing Trades")
render_page_title("Ongoing Trades")
try:
    register_ongoing_trades_css()
except Exception:
    pass

# --- Dialog styles: Inspect + Close Wizard ---
st.markdown("""
<style>
/* ===== Inspect dialog ===== */
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

/* ===== Close Wizard dialog ===== */
.wiz { padding:.25rem 0 .25rem; }
.wiz .hdr { display:flex; align-items:center; justify-content:space-between; margin-bottom:.35rem; }
.wiz .title { font-weight:700; font-size:1.05rem; letter-spacing:.2px; }
.wiz .step { font-size:.9rem; color:#6b7280; }
.wiz .meta { display:flex; gap:.4rem; flex-wrap:wrap; margin:.25rem 0 .5rem; }
.wiz .chip { background:#f3f4f6; color:#111827; border-radius:999px; padding:.15rem .6rem; font-size:.82rem; font-weight:600; }
.wiz .chip.good { background:#d1fae5; color:#065f46; }
.wiz .chip.bad  { background:#fee2e2; color:#7f1d1d; }
.wiz .chip.neutral { background:#e5e7eb; color:#111827; }
.wiz .hint { color:#6b7280; font-size:.85rem; margin-top:.25rem; }
.wiz .actions { display:flex; gap:.5rem; justify-content:flex-end; margin-top:.5rem; }
</style>
""", unsafe_allow_html=True)

st.session_state.pop("_wiz_rendered_this_run", None)

# Small helper for clean reruns after actions
def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass


# ─────────────────────────────────────────
# Load live positions & give each a RowId (row-exact actions even after filters)
# ─────────────────────────────────────────
open_df = io_helpers.load_open_trades()
if open_df.empty:
    st.info(
        "No ongoing trades. Use **Systematic Decision → Manage Queue → Mark Live** to open a position."
    )
    st.stop()

open_df = open_df.reset_index().rename(columns={"index": "RowId"})


# === Latest price helper (syncs from Add/Edit page data) =====================
def _latest_price_for(name: str) -> float | None:
    """Read the most recent price saved on Add/Edit (CurrentPrice -> Price -> SharePrice)."""
    try:
        df = io_helpers.load_data()
        if df is None or df.empty:
            return None
        rows = df[df["Name"].astype(str).str.upper() == str(name).upper()]
        if rows.empty:
            return None
        for col in ("CurrentPrice", "Price", "SharePrice"):
            if col in rows.columns:
                s = pd.to_numeric(rows[col], errors="coerce").dropna()
                if not s.empty:
                    return float(s.iloc[-1])
        return None
    except Exception:
        return None


# ─────────────────────────────────────────
# Filters (search by name, strategy, recent period)
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec"><div class="t">🔎 Filters</div>'
    '<div class="d">Narrow the list before taking action</div></div>',
    unsafe_allow_html=True,
)

f1, f2, f3 = st.columns([2, 1, 1])
with f1:
    q = st.text_input("Search name", placeholder="Type part of a stock name…")
with f2:
    strategies = ["All"] + sorted([s for s in open_df["Strategy"].dropna().unique()])
    strat_sel = st.selectbox("Strategy", strategies, index=0)
with f3:
    period = st.selectbox(
        "Opened in",
        ["Any", "Last 7 days", "Last 14 days", "Last 1 month", "Last 3 months"],
        index=0,
    )

filtered = open_df.copy()

if q.strip():
    qq = q.lower()
    filtered = filtered[filtered["Name"].str.lower().str.contains(qq, na=False)]
if strat_sel != "All":
    filtered = filtered[filtered["Strategy"] == strat_sel]
if period != "Any" and "OpenDate" in filtered.columns:
    now = datetime.now()
    if period == "Last 7 days":
        cutoff = now - timedelta(days=7)
    elif period == "Last 14 days":
        cutoff = now - timedelta(days=14)
    elif period == "Last 1 month":
        cutoff = now - timedelta(days=30)
    else:
        cutoff = now - timedelta(days=90)
    dt = pd.to_datetime(filtered["OpenDate"], errors="coerce")
    filtered = filtered[dt >= cutoff]

if filtered.empty:
    st.info("No rows match the current filters.")
    st.stop()


# === Live price + P&L columns (synced to Add/Edit → CurrentPrice) ============
to_num = lambda c: pd.to_numeric(filtered.get(c), errors="coerce")
shares = to_num("Shares").fillna(0)
entry = to_num("Entry").fillna(0)
stop = to_num("Stop")
tp1 = to_num("TP1")
tp2 = to_num("TP2")
tp3 = to_num("TP3")

# Latest price from Add/Edit
filtered["Current"] = filtered["Name"].map(_latest_price_for).astype("float")
cur = pd.to_numeric(filtered.get("Current"), errors="coerce")

# Dividend yield + income estimates for dividend-focused long trades
strategy_series = filtered.get("Strategy")
if strategy_series is None:
    strategy_series = pd.Series(index=filtered.index, dtype="object")
strategy_series = strategy_series.astype(str)
longtrade_mask = strategy_series.str.contains("LongTrade", case=False, na=False) & strategy_series.str.contains("DY", case=False, na=False)

dividend_version = dividend_data_version()
div_yield = filtered["Name"].astype(str).map(
    lambda n: get_latest_dividend_yield(n, data_version=dividend_version) if n else None
)
div_yield = pd.to_numeric(div_yield, errors="coerce")
filtered["Dividend Yield (%)"] = div_yield.where(longtrade_mask).round(2)

shares_for_div = pd.to_numeric(filtered.get("Shares"), errors="coerce")
entry_for_div = pd.to_numeric(filtered.get("Entry"), errors="coerce")
valid_inputs = longtrade_mask & shares_for_div.notna() & entry_for_div.notna() & div_yield.notna()
annual_div = (shares_for_div * entry_for_div * (div_yield / 100.0)).where(valid_inputs)
filtered["Est. Dividend (Annual)"] = annual_div.round(2)
filtered["Est. Dividend (Half-Year)"] = (annual_div / 2.0).round(2)

# P&L metrics
filtered["Value (RM)"] = (shares * cur).round(2)
filtered["Cost (RM)"] = (shares * entry).round(2)
filtered["Unrealized PnL (RM)"] = (shares * (cur - entry)).round(2)
filtered["Return (%)"] = ((cur / entry - 1.0) * 100.0).where(entry > 0).round(2)

# Live R multiple for long trades (Entry > Stop)
risk_ps = (entry - stop)
filtered["R live"] = ((cur - entry) / risk_ps).where((risk_ps > 0), other=pd.NA).round(2)

# Distances to targets / SL
def _delta_pct(target, cur_px):
    if pd.isna(target) or pd.isna(cur_px) or target == 0:
        return None
    return (target - cur_px) / target * 100.0

filtered["Δ to TP1 (RM)"] = (tp1 - cur).round(4)
filtered["Δ to TP2 (RM)"] = (tp2 - cur).round(4)
filtered["Δ to TP3 (RM)"] = (tp3 - cur).round(4)
filtered["Δ to TP1 (%)"] = filtered.apply(
    lambda r: round(_delta_pct(r.get("TP1"), r.get("Current")), 2)
    if pd.notna(r.get("TP1"))
    else None,
    axis=1,
)
filtered["Δ to TP2 (%)"] = filtered.apply(
    lambda r: round(_delta_pct(r.get("TP2"), r.get("Current")), 2)
    if pd.notna(r.get("TP2"))
    else None,
    axis=1,
)
filtered["Δ to TP3 (%)"] = filtered.apply(
    lambda r: round(_delta_pct(r.get("TP3"), r.get("Current")), 2)
    if pd.notna(r.get("TP3"))
    else None,
    axis=1,
)
filtered["Δ to SL (RM)"] = (cur - stop).round(4)  # negative = above SL (safe)
filtered["SL Breach?"] = (cur <= stop).where(stop.notna())


# ─────────────────────────────────────────
# Overview (KPI cards)
# ─────────────────────────────────────────
st.markdown(
    section("📊 Overview", "Open positions & exposure (filtered)"),
    unsafe_allow_html=True,
)

total_cost = float((shares * entry).sum())
cur_value = float((shares * cur).sum(skipna=True))
open_pnl = float((shares * (cur - entry)).sum(skipna=True))

rr_init = pd.to_numeric(filtered.get("RR"), errors="coerce")
avg_rr = float(rr_init.dropna().mean()) if rr_init.notna().any() else None

open_dates = pd.to_datetime(filtered.get("OpenDate"), errors="coerce")
avg_hold_days = (
    int((pd.Timestamp.now() - open_dates).dt.days.dropna().mean())
    if open_dates.notna().any()
    else None
)

def _tone_rr(v):
    if v is None:
        return ""
    if v >= 2.0:
        return "good"
    if v >= 1.5:
        return "warn"
    return "bad"

def _tone_pnl(v):
    if v > 0:
        return "good"
    if v < 0:
        return "bad"
    return "neutral"

render_stat_cards(
    [
        {"label": "Open Positions", "value": f"{len(filtered):,}", "badge": "Shown"},
        {"label": "Total Cost (RM)", "value": f"{total_cost:,.2f}", "badge": "Exposure"},
        {"label": "Cur Value (RM)", "value": f"{cur_value:,.2f}", "badge": "Marked"},
        {
            "label": "Unrealized P&L",
            "value": f"{open_pnl:,.2f}",
            "badge": "Live",
            "tone": _tone_pnl(open_pnl),
        },
        {
            "label": "Avg RR Init",
            "value": (f"{avg_rr:.2f}×" if avg_rr is not None else "—"),
            "badge": "Quality",
            "tone": _tone_rr(avg_rr),
        },
        {
            "label": "Avg Holding (d)",
            "value": (f"{avg_hold_days:,}" if avg_hold_days is not None else "—"),
            "badge": "Duration",
        },
    ],
    columns=3,
)


# ─────────────────────────────────────────
# Dividend-oriented overview
# ─────────────────────────────────────────
st.markdown(
    section("💸 Dividend Outlook", "Projected income from dividend-focused long trades"),
    unsafe_allow_html=True,
)

yield_series = pd.to_numeric(filtered.get("Dividend Yield (%)"), errors="coerce").where(longtrade_mask)
annual_series = pd.to_numeric(filtered.get("Est. Dividend (Annual)"), errors="coerce").where(longtrade_mask)
half_series = pd.to_numeric(filtered.get("Est. Dividend (Half-Year)"), errors="coerce").where(longtrade_mask)

if longtrade_mask.any() and (
    yield_series.notna().any() or annual_series.notna().any() or half_series.notna().any()
):
    total_annual = float(annual_series.dropna().sum()) if annual_series.notna().any() else 0.0
    total_half = float(half_series.dropna().sum()) if half_series.notna().any() else 0.0
    avg_yield = (
        float(yield_series.dropna().mean()) if yield_series.notna().any() else None
    )
    render_stat_cards(
        [
            {"label": "Dividend Longs", "value": f"{int(longtrade_mask.sum()):,}", "badge": "Positions"},
            {"label": "Est. Annual Div (RM)", "value": f"{total_annual:,.2f}", "badge": "Projected"},
            {"label": "Est. Half-Year (RM)", "value": f"{total_half:,.2f}", "badge": "Projected"},
            {
                "label": "Avg Yield on Cost",
                "value": f"{avg_yield:.2f}%" if avg_yield is not None else "—",
                "badge": "Yield",
            },
        ],
        columns=4,
    )
else:
    st.info("No dividend-oriented LongTrade positions in the current filter.")


# ─────────────────────────────────────────
# Editor table (filtered set)
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec success"><div class="t">🧾 Open Positions</div>'
    '<div class="d">Specify Close Price &amp; Reason, then tick rows to close</div></div>',
    unsafe_allow_html=True,
)

CLOSE_REASONS = [
    "Target hit",
    "Stop hit",
    "Trailing stop",
    "Time stop",
    "Thesis changed",
    "Portfolio rebalance",
    "Other (specify)",
]

table = filtered.copy()
table.insert(0, "Select", False)
table["ClosePrice"] = 0.0
table["CloseReason"] = CLOSE_REASONS[0]
table["Detail"] = ""

edited = st.data_editor(
    table,
    use_container_width=True,
    height=520,
    hide_index=True,
    column_config={
        "Select": st.column_config.CheckboxColumn("Sel"),
        "RowId": st.column_config.NumberColumn("RowId", disabled=True),
        "Name": st.column_config.TextColumn("Name", disabled=True),
        "Strategy": st.column_config.TextColumn("Strategy", disabled=True),

        "Entry": st.column_config.NumberColumn("Entry", format="%.4f", disabled=True),
        "Stop": st.column_config.NumberColumn("Stop", format="%.4f", disabled=True),
        "TP1": st.column_config.NumberColumn("TP1", format="%.4f", disabled=True),
        "TP2": st.column_config.NumberColumn("TP2", format="%.4f", disabled=True),
        "TP3": st.column_config.NumberColumn("TP3", format="%.4f", disabled=True),
        "Take": st.column_config.NumberColumn("Take", format="%.4f", disabled=True),

        "Shares": st.column_config.NumberColumn("Shares", format="%d", disabled=True),
        "RR": st.column_config.NumberColumn("RR Init", format="%.2f", disabled=True),
        "OpenDate": st.column_config.TextColumn("Open Date", disabled=True),
        "Reasons": st.column_config.TextColumn("Notes", disabled=True),

        # Live metrics (read-only)
        "Current": st.column_config.NumberColumn("Current", format="%.4f", disabled=True),
        "Dividend Yield (%)": st.column_config.NumberColumn(
            "Dividend Yield (%)", format="%.2f%%", disabled=True
        ),
        "Est. Dividend (Annual)": st.column_config.NumberColumn(
            "Est. Dividend (Annual)", format="RM %.2f", disabled=True
        ),
        "Est. Dividend (Half-Year)": st.column_config.NumberColumn(
            "Est. Dividend (Half-Year)", format="RM %.2f", disabled=True
        ),
        "R live": st.column_config.NumberColumn("R live", format="%.2f", disabled=True),
        "Return (%)": st.column_config.NumberColumn("Return (%)", format="%.2f", disabled=True),
        "Unrealized PnL (RM)": st.column_config.NumberColumn("P&L (RM)", format="%.2f", disabled=True),
        "Δ to TP1 (RM)": st.column_config.NumberColumn("Δ to TP1 (RM)", format="%.4f", disabled=True),
        "Δ to TP2 (RM)": st.column_config.NumberColumn("Δ to TP2 (RM)", format="%.4f", disabled=True),
        "Δ to TP3 (RM)": st.column_config.NumberColumn("Δ to TP3 (RM)", format="%.4f", disabled=True),
        "Δ to TP1 (%)": st.column_config.NumberColumn("Δ to TP1 (%)", format="%.2f", disabled=True),
        "Δ to TP2 (%)": st.column_config.NumberColumn("Δ to TP2 (%)", format="%.2f", disabled=True),
        "Δ to TP3 (%)": st.column_config.NumberColumn("Δ to TP3 (%)", format="%.2f", disabled=True),
        "Δ to SL (RM)": st.column_config.NumberColumn("Δ to SL (RM)", format="%.4f", disabled=True),
        "SL Breach?": st.column_config.CheckboxColumn("SL Breach?", disabled=True),

        # Close controls
        "ClosePrice": st.column_config.NumberColumn("Close Price", format="%.4f"),
        "CloseReason": st.column_config.SelectboxColumn(
            "Close Reason", options=CLOSE_REASONS
        ),
        "Detail": st.column_config.TextColumn("Detail (if Other)"),
    },
    column_order=[
        "Select",
        "RowId",
        "Name",
        "Strategy",
        "Entry",
        "Stop",
        "TP1",
        "TP2",
        "TP3",
        "Take",
        "Shares",
        "RR",
        "OpenDate",
        "Current",
        "Dividend Yield (%)",
        "Est. Dividend (Annual)",
        "Est. Dividend (Half-Year)",
        "R live",
        "Return (%)",
        "Unrealized PnL (RM)",
        "Δ to TP1 (RM)",
        "Δ to TP2 (RM)",
        "Δ to TP3 (RM)",
        "Δ to SL (RM)",
        "SL Breach?",
        "Reasons",
        "ClosePrice",
        "CloseReason",
        "Detail",
    ],
    key="open_trades_editor",
)


# ─────────────────────────────────────────
# Optional: quick Inspect dialog
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec info"><div class="t">🔍 Inspect</div>'
    '<div class="d">Quick P&L breakdown for the first selected row</div></div>',
    unsafe_allow_html=True,
)

def _open_inspect_dialog(row: pd.Series):
    title = f"{row.get('Name')} — {row.get('Strategy')}"
    entry = row.get("Entry")
    cur   = row.get("Current")
    stop  = row.get("Stop")
    tp1   = row.get("TP1")
    tp2   = row.get("TP2")
    tp3   = row.get("TP3")
    shares = int(row.get("Shares") or 0)
    pnl   = row.get("Unrealized PnL (RM)")
    ret   = row.get("Return (%)")
    rlive = row.get("R live")

    def _pct_to_target(tp):
        try:
            if tp is None or np.isnan(tp) or entry is None or np.isnan(entry) or cur is None or np.isnan(cur):
                return None
            denom = float(tp) - float(entry)
            if denom <= 0:  # not a meaningful upward target
                return None
            return float(np.clip((float(cur) - float(entry)) / denom, 0.0, 1.0))
        except Exception:
            return None

    def _sl_buffer():
        try:
            if stop is None or np.isnan(stop) or entry is None or np.isnan(entry) or cur is None or np.isnan(cur):
                return None
            risk = float(entry) - float(stop)
            if risk <= 0:
                return None
            return float(np.clip((float(cur) - float(stop)) / risk, 0.0, 1.0))
        except Exception:
            return None

    p1 = _pct_to_target(tp1)
    p2 = _pct_to_target(tp2)
    p3 = _pct_to_target(tp3)
    ps = _sl_buffer()
    sl_breached = (stop is not None and not pd.isna(stop) and cur is not None and cur <= stop)

    # Tones
    pill_tone = "neutral"
    if isinstance(rlive, (int, float)) and np.isfinite(rlive):
        pill_tone = "good" if rlive >= 1.0 else ("neutral" if rlive >= 0 else "bad")
    pnl_cls = "pos" if (isinstance(pnl, (int, float)) and pnl >= 0) else "neg"

    if hasattr(st, "dialog"):
        @st.dialog(title)
        def _dlg():
            st.markdown(
                f"<div class='inspect'><div class='muted'>Opened: {row.get('OpenDate','—')}</div></div>",
                unsafe_allow_html=True
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    "<div class='inspect kpi'><h4>Entry</h4>"
                    f"<div class='v'>{entry:.4f}</div></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    "<div class='inspect kpi'><h4>Current</h4>"
                    f"<div class='v'>{(f'{cur:.4f}' if pd.notna(cur) else '—')}</div></div>",
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    "<div class='inspect kpi'><h4>R live</h4>"
                    f"<div class='v'><span class='pill {pill_tone}'>{'—' if pd.isna(rlive) else f'{rlive:.2f}'}</span></div></div>",
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
                    "<div class='inspect kpi'><h4>Shares</h4>"
                    f"<div class='v'>{shares:,d}</div></div>",
                    unsafe_allow_html=True,
                )

            st.divider()
            st.markdown("**Targets & SL**")

            def _bar(lbl, pct, tp_val):
                if pct is None:
                    st.caption(f"{lbl}: —")
                else:
                    st.progress(pct, text=f"{lbl}: {pct*100:.0f}%  (TP {tp_val:.4f})")

            _bar("TP1 progress", p1, tp1)
            _bar("TP2 progress", p2, tp2)
            _bar("TP3 progress", p3, tp3)

            if sl_breached:
                st.error(f"Stop-loss breached (SL {stop:.4f} ≥ Current {cur:.4f}).")
            elif ps is None:
                st.caption("SL buffer: —")
            else:
                st.progress(ps, text=f"SL buffer: {ps*100:.0f}% away from stop  (SL {stop:.4f})")

            st.caption(
                f"**Plan** — TP1 {tp1 if pd.notna(tp1) else '—'} | "
                f"TP2 {tp2 if pd.notna(tp2) else '—'} | "
                f"TP3 {tp3 if pd.notna(tp3) else '—'} | SL {stop if pd.notna(stop) else '—'}"
            )
        _dlg()
    else:
        # Fallback for older Streamlit
        st.info("Update Streamlit (≥1.31) to show dialogs. Showing inline:")
        st.write(
            row[
                [
                    "Name",
                    "Strategy",
                    "Entry",
                    "Current",
                    "R live",
                    "Unrealized PnL (RM)",
                    "Return (%)",
                    "TP1",
                    "TP2",
                    "TP3",
                    "Stop",
                ]
            ]
        )

if st.button("📌 Inspect first selected"):
    sel = edited[edited.get("Select") == True]
    if sel.empty:
        st.info("Tick a row first.")
    else:
        _open_inspect_dialog(sel.iloc[0])

# ─────────────────────────────────────────
# Close Wizard (pop-up, one-by-one)
# ─────────────────────────────────────────
def _start_close_wizard(rows_df: pd.DataFrame):
    """Seed session state with selected rows to process one-by-one."""
    st.session_state["close_wizard_rows"] = rows_df.to_dict(orient="records")
    st.session_state["close_wizard_idx"] = 0
    st.session_state["close_wizard_active"] = True

def _clear_close_wizard_state():
    for k in ("close_wizard_rows", "close_wizard_idx", "close_wizard_active"):
        st.session_state.pop(k, None)

def _show_close_wizard_dialog():
    """Render a minimal dialog to complete missing close info, one by one."""
    if not hasattr(st, "dialog"):
        st.warning("Update Streamlit (≥1.31) to use the pop-up wizard. Use inline inputs instead.")
        _clear_close_wizard_state()
        return

    # pull from session
    rows = st.session_state.get("close_wizard_rows", []) or []
    idx  = int(st.session_state.get("close_wizard_idx", 0))
    total = len(rows)

    @st.dialog("Complete close details")
    def _wiz():
        rows = st.session_state.get("close_wizard_rows", []) or []
        idx  = int(st.session_state.get("close_wizard_idx", 0))
        total = len(rows)

        # Finished → clear + rerun (auto close dialog)
        if total == 0 or idx >= total:
            _clear_close_wizard_state()
            _safe_rerun()
            return

        r = rows[idx]
        name    = r.get("Name", "—")
        strat   = r.get("Strategy", "—")
        entry   = r.get("Entry")
        current = r.get("Current")
        shares  = int(r.get("Shares") or 0)
        rlive   = r.get("R live")
        pnl     = r.get("Unrealized PnL (RM)")
        retpct  = r.get("Return (%)")
        opened  = r.get("OpenDate", "—")

        # Prefill price with Current (fallback Entry)
        default_price = current if (current is not None and not pd.isna(current)) else entry

        def _fmt4(x): 
            return "—" if (x is None or (isinstance(x,(int,float)) and pd.isna(x))) else f"{float(x):.4f}"
        def _fmt2(x): 
            return "—" if (x is None or (isinstance(x,(int,float)) and pd.isna(x))) else f"{float(x):.2f}"

        # Small header + progress (clean look)
        st.markdown(
            f"""
<div class="wiz">
  <div class="hdr">
    <div class="title">Close <strong>{name}</strong> — {strat}</div>
    <div class="step">Step {idx+1} of {total}</div>
  </div>
  <div class="meta">
    <span class="chip">Opened: {opened}</span>
    <span class="chip">Entry: {_fmt4(entry)}</span>
    <span class="chip">Current: {_fmt4(current)}</span>
    <span class="chip">Shares: {shares:,d}</span>
    <span class="chip">R live: {_fmt2(rlive)}</span>
    <span class="chip">P&L: {"—" if pd.isna(pnl) else f"{float(pnl):,.2f}"}</span>
    <span class="chip">Return: {_fmt2(retpct)}%</span>
  </div>
</div>
""", unsafe_allow_html=True)
        st.progress(idx / max(total, 1), text=f"Progress {idx}/{total}")

        # Form (Enter to submit)
        with st.form(key=f"wiz_form_{idx}", clear_on_submit=False, border=True):
            price = st.number_input(
                "Close Price", value=float(default_price or 0.0),
                min_value=0.0, step=0.001, format="%.4f", key=f"wiz_price_{idx}"
            )
            reason = st.selectbox("Close Reason", CLOSE_REASONS, index=0, key=f"wiz_reason_{idx}")
            need_detail = (reason == "Other (specify)")
            detail = st.text_input("Detail (if Other)", key=f"wiz_detail_{idx}",
                                   placeholder="e.g., exit due to earnings risk")

            submit = st.form_submit_button("Close & Next ➜", type="primary", use_container_width=True)

            if submit:
                if price <= 0:
                    st.error("Close Price must be > 0.")
                    st.stop()
                if need_detail and not (detail or "").strip():
                    st.error("Please specify detail for 'Other'.")
                    st.stop()
                reason_txt = reason if not need_detail else f"{reason}: {detail.strip()}"
                ok = io_helpers.close_open_trade_row(int(r["RowId"]), float(price), reason_txt)
                if ok:
                    st.toast(f"Closed {name}", icon="✅")
                st.session_state["close_wizard_idx"] = idx + 1
                st.rerun()

    _wiz()


# ─────────────────────────────────────────
# Actions (auto wizard for missing info)
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec warning"><div class="t">🔒 Actions</div>'
    '<div class="d">Click Close selected — any row missing Close Price / required detail will open a pop-up</div></div>',
    unsafe_allow_html=True,
)

# Selected rows from editor
selected_mask = edited.get("Select") == True
selected_rows = edited[selected_mask] if "Select" in edited.columns else edited.iloc[0:0]
st.caption(f"Selected: **{len(selected_rows)}** row(s)")

def _needs_wizard(row: pd.Series) -> bool:
    px = float(row.get("ClosePrice") or 0)
    reason = (row.get("CloseReason") or "").strip()
    detail = (row.get("Detail") or "").strip()
    if px <= 0:
        return True
    if reason == "Other (specify)" and not detail:
        return True
    return False

def _close_row_now(row: pd.Series) -> bool:
    try:
        px = float(row.get("ClosePrice") or 0)
        reason = (row.get("CloseReason") or "").strip()
        detail = (row.get("Detail") or "").strip()
        reason_txt = reason if reason != "Other (specify)" else f"{reason}: {detail}"
        return bool(io_helpers.close_open_trade_row(int(row["RowId"]), px, reason_txt))
    except Exception:
        return False

if st.button("🔒 Close selected", type="primary"):
    if selected_rows.empty:
        st.info("Tick at least one row first.")
    else:
        # Partition into valid vs missing
        needs = selected_rows[selected_rows.apply(_needs_wizard, axis=1)]
        valids = selected_rows.drop(needs.index, errors="ignore")

        # Close all valid rows immediately
        closed = 0
        for _, r in valids.iterrows():
            if _close_row_now(r):
                closed += 1

        if closed:
            st.toast(f"Closed {closed} trade(s).", icon="✅")

        if not needs.empty:
            _start_close_wizard(needs)
            # Don't open the dialog here — the single call below will handle it
        else:
            # All done, just refresh the table
            _safe_rerun()

# Open the wizard in exactly one place per run
if st.session_state.get("close_wizard_active", False) and not st.session_state.get("_wiz_rendered_this_run", False):
    st.session_state["_wiz_rendered_this_run"] = True
    _show_close_wizard_dialog()

