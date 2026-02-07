# pages/7z_Friend_Trades.py
from __future__ import annotations

from auth_gate import require_auth
require_auth()

import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import re

# --- Version etag to invalidate caches when data changes ---
_THIS = os.path.dirname(__file__)
_GRANDP = os.path.abspath(os.path.join(_THIS, "..", ".."))
_VERSION_FILE = os.path.join(_GRANDP, ".data.version")
def _data_etag() -> int:
    try:
        return int(os.stat(_VERSION_FILE).st_mtime_ns)
    except Exception:
        return 0

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


# ---------- Page setup ----------
setup_page("Friend Trades")
render_page_title("Friend Trades")
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
open_df = io_helpers.load_friend_open_trades()
if open_df.empty:
    st.info(
        "No friend trades yet. In **Systematic Decision → Manage Queue**, tick **Friend** and Mark Live."
    )
    st.stop()

open_df = open_df.reset_index().rename(columns={"index": "RowId"})

# >>> PATCH START: tolerant price index by Name or Code/Ticker
@st.cache_data(show_spinner=False)
def _build_price_index(_etag: int) -> dict[str, float]:
    """
    Build a case/format-tolerant price index keyed by:
      - UPPER(Name)
      - UPPER(Name) with non-alphanumerics stripped (so 'RHONE MA' -> 'RHONEMA')
      - UPPER(Code/Ticker/Symbol) if such a column exists
    Uses the most recent non-null price from CurrentPrice (preferred) or Price/SharePrice.
    """
    df = io_helpers.load_data()
    if df is None or df.empty:
        return {}

    # Choose best available price column
    price_col = None
    for c in ("CurrentPrice", "Price", "SharePrice"):
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        return {}

    work = df.copy()
    work[price_col] = pd.to_numeric(work[price_col], errors="coerce")

    idx: dict[str, float] = {}

    # --- Index by Name (case-insensitive) ---
    if "Name" in work.columns:
        name_key = work["Name"].astype(str).str.upper().str.strip()
        s_name = work.dropna(subset=[price_col]).groupby(name_key)[price_col].last()
        for k, v in s_name.items():
            idx[k] = float(v)
        # Add no-punct/spaces variants (e.g., 'RHONE MA' -> 'RHONEMA')
        extra: dict[str, float] = {}
        for k, v in idx.items():
            k2 = re.sub(r"[^A-Z0-9]+", "", k)
            if k2 and k2 not in idx:
                extra[k2] = v
        idx.update(extra)

    # --- Also index by Code/Ticker/Symbol, if present ---
    code_col = None
    for c in ("Code", "Ticker", "StockCode", "Symbol"):
        if c in work.columns:
            code_col = c
            break
    if code_col:
        codes = work[[code_col, price_col]].dropna(subset=[code_col, price_col]).copy()
        code_key = codes[code_col].astype(str).str.upper().str.strip()
        s_code = codes.groupby(code_key)[price_col].last()
        for k, v in s_code.items():
            k_str = str(k)
            idx[k_str] = float(v)
            idx[re.sub(r"[^A-Z0-9]+", "", k_str)] = float(v)

    return idx


def _lookup_latest_price(name: str | None, code: str | None) -> float | None:
    """
    Try name, then code/ticker, using both raw UPPER and space/punct-stripped variants.
    """
    idx = _build_price_index(_data_etag())
    for key in (name, code):
        if not key:
            continue
        k1 = str(key).upper().strip()
        if k1 in idx:
            return idx[k1]
        k2 = re.sub(r"[^A-Z0-9]+", "", k1)
        if k2 in idx:
            return idx[k2]
    return None
# <<< PATCH END

# ─────────────────────────────────────────
# Filters (search by name, strategy, recent period)
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec"><div class="t">🔎 Filters</div>'
    '<div class="d">Narrow the list before taking action</div></div>',
    unsafe_allow_html=True,
)

f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
with f1:
    q = st.text_input("Search name", placeholder="Type part of a stock name…")
with f2:
    strategies = ["All"] + sorted([s for s in open_df["Strategy"].dropna().unique()])
    strat_sel = st.selectbox("Strategy", strategies, index=0)
with f3:
    friend_options = ["All"]
    if "FriendName" in open_df.columns:
        _friends = [x for x in open_df["FriendName"].dropna().astype(str).unique().tolist() if x.strip()]
        friend_options = ["All"] + sorted(_friends)
    friend_sel = st.selectbox("Friend", friend_options, index=0)
with f4:
    period = st.selectbox(
        "Opened in",
        ["Any", "Last 7 days", "Last 14 days", "Last 1 month", "Last 3 months"],
        index=0,
    )


with st.expander("💸 Dividend settings (optional)", expanded=False):
    default_tax_pct = st.number_input(
        "Default Dividend Tax %",
        min_value=0.0,
        max_value=60.0,
        value=0.0,
        step=0.5,
        help="Used for NET dividend estimates. If a row has TaxPct, it overrides this default.",
    )

# ─────────────────────────────────────────
# Dividend helpers (TTM DPS) — aligned with Long Trade Dividends logic
# ─────────────────────────────────────────
def _ttm_dps_from_session(name: str) -> float | None:
    """Prefer DPS from View Stock page's TTM KPI sync cache (exactly what you saw there)."""
    try:
        cache = st.session_state.get("TTM_KPI_SYNC", {})
        rec = cache.get(str(name)) or {}
        vals = rec.get("values") or {}
        v = vals.get("DPS")
        if v is None:
            return None
        v = float(v)
        return v if np.isfinite(v) and v >= 0 else None
    except Exception:
        return None

def _find_q_dps_col(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    cands = []
    for c in df.columns:
        s = str(c).lower()
        if ("dps" in s) or ("dividend per share" in s) or ("dpu" in s):
            cands.append(c)
    for c in cands:
        if str(c).lower().startswith("q_"):
            return c
    return cands[0] if cands else None

@st.cache_data(show_spinner=False)
def _ttm_dps_from_quarters_cached(_etag: int, name: str) -> float | None:
    """Fallback: sum the latest up to 4 quarters of DPS from io_helpers.load_data()."""
    try:
        df = io_helpers.load_data()
        if df is None or df.empty or "Name" not in df.columns:
            return None
        sub = df[df["Name"].astype(str).str.upper() == str(name).upper()].copy()
        if sub.empty:
            return None
        q = sub[sub.get("IsQuarter") == True].copy()
        if q.empty:
            return None
        if "Qnum" not in q.columns and "Quarter" in q.columns:
            q["Qnum"] = q["Quarter"].astype(str).str.extract(r"(\d+)").astype(float)
        q["Year"] = pd.to_numeric(q.get("Year"), errors="coerce")
        q = q.dropna(subset=["Year", "Qnum"]).sort_values(["Year", "Qnum"])
        col = _find_q_dps_col(q)
        if not col:
            return None
        s = pd.to_numeric(q[col], errors="coerce").dropna()
        if s.empty:
            return None
        v = float(s.tail(min(4, len(s))).sum())
        return v if np.isfinite(v) and v >= 0 else None
    except Exception:
        return None

def get_ttm_dps(name: str) -> float | None:
    v = _ttm_dps_from_session(name)
    if v is None:
        v = _ttm_dps_from_quarters_cached(_data_etag(), str(name))
    return v

def _row_tax_pct(row, default_tax: float) -> float:
    """Per-row tax override if TaxPct exists, else use default."""
    try:
        v = float(row.get("TaxPct"))
        if np.isfinite(v):
            return max(0.0, min(60.0, v))
    except Exception:
        pass
    return float(default_tax or 0.0)

# ─────────────────────────────────────────
# Enrich ALL rows once (price, P&L, dividend), then filter for display
# ─────────────────────────────────────────
all_df = open_df.copy()

# numeric conversions
all_shares = pd.to_numeric(all_df.get("Shares"), errors="coerce").fillna(0.0)
all_entry  = pd.to_numeric(all_df.get("Entry"), errors="coerce").fillna(0.0)
all_stop   = pd.to_numeric(all_df.get("Stop"), errors="coerce")
all_tp1    = pd.to_numeric(all_df.get("TP1"), errors="coerce")
all_tp2    = pd.to_numeric(all_df.get("TP2"), errors="coerce")
all_tp3    = pd.to_numeric(all_df.get("TP3"), errors="coerce")

# Latest price from Add/Edit (tolerant name/code match)
all_df["Current"] = all_df.apply(
    lambda r: _lookup_latest_price(
        r.get("Name"),
        r.get("Code") or r.get("Ticker") or r.get("Symbol")
    ),
    axis=1
).astype("float")
all_cur = pd.to_numeric(all_df.get("Current"), errors="coerce")

# P&L metrics
all_df["Value (RM)"] = (all_shares * all_cur).round(2)
all_df["Cost (RM)"] = (all_shares * all_entry).round(2)
all_df["Unrealized PnL (RM)"] = (all_shares * (all_cur - all_entry)).round(2)
all_df["Return (%)"] = ((all_cur / all_entry - 1.0) * 100.0).where(all_entry > 0).round(2)

# Live R multiple (Entry > Stop)
risk_ps = (all_entry - all_stop)
all_df["R live"] = ((all_cur - all_entry) / risk_ps).where((risk_ps > 0), other=pd.NA).round(2)

# Distances to targets / SL
def _delta_pct(target, cur_px):
    if pd.isna(target) or pd.isna(cur_px) or target == 0:
        return None
    return (target - cur_px) / target * 100.0

all_df["Δ to TP1 (RM)"] = (all_tp1 - all_cur).round(4)
all_df["Δ to TP2 (RM)"] = (all_tp2 - all_cur).round(4)
all_df["Δ to TP3 (RM)"] = (all_tp3 - all_cur).round(4)
all_df["Δ to TP1 (%)"] = all_df.apply(
    lambda r: round(_delta_pct(r.get("TP1"), r.get("Current")), 2) if pd.notna(r.get("TP1")) else None,
    axis=1,
)
all_df["Δ to TP2 (%)"] = all_df.apply(
    lambda r: round(_delta_pct(r.get("TP2"), r.get("Current")), 2) if pd.notna(r.get("TP2")) else None,
    axis=1,
)
all_df["Δ to TP3 (%)"] = all_df.apply(
    lambda r: round(_delta_pct(r.get("TP3"), r.get("Current")), 2) if pd.notna(r.get("TP3")) else None,
    axis=1,
)
all_df["Δ to SL (RM)"] = (all_cur - all_stop).round(4)  # negative = above SL (safe)
all_df["SL Breach?"] = (all_cur <= all_stop).where(all_stop.notna())

# Dividend metrics (TTM DPS × Shares)
all_df["TTM DPS (RM)"] = all_df["Name"].map(lambda n: get_ttm_dps(str(n))).astype("float")
all_df["Dividend (Gross RM)"] = (all_shares * pd.to_numeric(all_df["TTM DPS (RM)"], errors="coerce").fillna(0)).round(2)
all_df["Tax %"] = all_df.apply(lambda r: _row_tax_pct(r, default_tax_pct), axis=1)
all_df["Dividend (Net RM)"] = (all_df["Dividend (Gross RM)"] * (1.0 - all_df["Tax %"] / 100.0)).round(2)
all_df["Dividend Yield (TTM, %)"] = ((pd.to_numeric(all_df["TTM DPS (RM)"], errors="coerce") / all_cur) * 100.0).where(all_cur > 0).round(2)

# Apply filters on enriched data
filtered = all_df.copy()

if q.strip():
    qq = q.lower()
    filtered = filtered[filtered["Name"].str.lower().str.contains(qq, na=False)]
if strat_sel != "All":
    filtered = filtered[filtered["Strategy"] == strat_sel]
if friend_sel != "All" and "FriendName" in filtered.columns:
    filtered = filtered[filtered["FriendName"].astype(str) == str(friend_sel)]
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


# ─────────────────────────────────────────
# Overview (KPI cards)
# ─────────────────────────────────────────
st.markdown(
    section("📊 Overview", "Open positions & exposure (filtered)"),
    unsafe_allow_html=True,
)

_shares = pd.to_numeric(filtered.get("Shares"), errors="coerce").fillna(0.0)
_entry  = pd.to_numeric(filtered.get("Entry"), errors="coerce").fillna(0.0)
_cur    = pd.to_numeric(filtered.get("Current"), errors="coerce").fillna(0.0)

total_cost = float(pd.to_numeric(filtered.get("Cost (RM)"), errors="coerce").fillna(0.0).sum())
cur_value  = float(pd.to_numeric(filtered.get("Value (RM)"), errors="coerce").fillna(0.0).sum())
open_pnl   = float(pd.to_numeric(filtered.get("Unrealized PnL (RM)"), errors="coerce").fillna(0.0).sum())

div_net_total = float(pd.to_numeric(filtered.get("Dividend (Net RM)"), errors="coerce").fillna(0.0).sum())
div_gross_total = float(pd.to_numeric(filtered.get("Dividend (Gross RM)"), errors="coerce").fillna(0.0).sum())
div_yield_net_wgt = (div_net_total / cur_value * 100.0) if cur_value > 0 else None

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
            "label": "Dividend Yield",
            "value": (f"{div_yield_net_wgt:.2f}%" if div_yield_net_wgt is not None else "—"),
            "badge": "TTM NET",
        },
        {
            "label": "Proj Dividend (RM)",
            "value": f"{div_net_total:,.2f}",
            "badge": "TTM NET",
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
    columns=4,
)

# ─────────────────────────────────────────
# Friend Accounts (grouped view)
# ─────────────────────────────────────────
st.markdown(
    section("👥 Friend Accounts", "Performance grouped by friend (live positions)"),
    unsafe_allow_html=True,
)

use_filters_for_accounts = st.checkbox(
    "Use current filters for accounts summary",
    value=bool((q and q.strip()) or strat_sel != "All" or friend_sel != "All" or period != "Any"),
    key="friend_accounts_use_filters",
    help="If OFF, totals include all open friend trades. If ON, it uses the same filters above.",
)

accounts_base = filtered if use_filters_for_accounts else all_df

if "FriendName" in accounts_base.columns:
    base = accounts_base.copy()
    base["FriendName"] = base["FriendName"].fillna("Unknown").astype(str).str.strip()
    base.loc[base["FriendName"] == "", "FriendName"] = "Unknown"

    # Optional realized P&L from friend closed trades
    realized_by_friend = {}
    try:
        load_closed = getattr(io_helpers, "load_friend_closed_trades", None)
        if callable(load_closed):
            cdf = load_closed()
            if cdf is not None and not cdf.empty and "FriendName" in cdf.columns:
                cdf = cdf.copy()
                cdf["FriendName"] = cdf["FriendName"].fillna("Unknown").astype(str).str.strip()
                cdf.loc[cdf["FriendName"] == "", "FriendName"] = "Unknown"
                if "PnL" in cdf.columns:
                    cdf["_pnl"] = pd.to_numeric(cdf["PnL"], errors="coerce")
                else:
                    sh = pd.to_numeric(cdf.get("Shares"), errors="coerce").fillna(0.0)
                    en = pd.to_numeric(cdf.get("Entry"), errors="coerce").fillna(0.0)
                    cp = pd.to_numeric(cdf.get("ClosePrice"), errors="coerce").fillna(0.0)
                    cdf["_pnl"] = sh * (cp - en)
                realized_by_friend = (
                    cdf.groupby("FriendName")["_pnl"].sum(min_count=1).fillna(0.0).to_dict()
                )
    except Exception:
        realized_by_friend = {}

    rows = []
    for fname, g in base.groupby("FriendName", dropna=False):
        cost = float(pd.to_numeric(g.get("Cost (RM)"), errors="coerce").fillna(0.0).sum())
        val  = float(pd.to_numeric(g.get("Value (RM)"), errors="coerce").fillna(0.0).sum())
        upnl = float(pd.to_numeric(g.get("Unrealized PnL (RM)"), errors="coerce").fillna(0.0).sum())
        rpn  = float(realized_by_friend.get(fname, 0.0) or 0.0)
        tpnl = upnl + rpn
        ret  = (val / cost - 1.0) * 100.0 if cost > 0 else None
        div_net = float(pd.to_numeric(g.get("Dividend (Net RM)"), errors="coerce").fillna(0.0).sum())
        dy_net  = (div_net / val * 100.0) if val > 0 else None
        rows.append(
            {
                "Friend": fname,
                "Open": int(len(g)),
                "Invested (RM)": round(cost, 2),
                "Cur Value (RM)": round(val, 2),
                "Unrealized P&L (RM)": round(upnl, 2),
                "Realized P&L (RM)": round(rpn, 2),
                "Total P&L (RM)": round(tpnl, 2),
                "Return (%)": (round(ret, 2) if ret is not None else None),
                "Proj Dividend NET (RM)": round(div_net, 2),
                "Avg Div Yield NET (%)": (round(dy_net, 2) if dy_net is not None else None),
            }
        )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["Total P&L (RM)", "Cur Value (RM)"], ascending=[False, False])

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Per-friend drilldown
    for _, row in summary_df.iterrows():
        fname = row["Friend"]
        sub = base[base["FriendName"] == fname].copy()
        pnl_total = float(row.get("Total P&L (RM)") or 0.0)
        ret = row.get("Return (%)")
        title = f"👤 {fname} — {pnl_total:,.2f} RM"
        if ret is not None:
            title += f"  |  {ret:+.2f}%"

        expanded = (friend_sel != "All" and str(friend_sel) == str(fname))
        with st.expander(title, expanded=expanded):
            # small KPI set
            render_stat_cards(
                [
                    {"label": "Open Positions", "value": f"{int(row['Open']):,}", "badge": "Live"},
                    {"label": "Invested (RM)", "value": f"{float(row['Invested (RM)']):,.2f}", "badge": "Cost"},
                    {"label": "Cur Value (RM)", "value": f"{float(row['Cur Value (RM)']):,.2f}", "badge": "Marked"},
                    {
                        "label": "Unrealized P&L",
                        "value": f"{float(row['Unrealized P&L (RM)']):,.2f}",
                        "badge": "Live",
                        "tone": _tone_pnl(float(row['Unrealized P&L (RM)'])),
                    },
                    {
                        "label": "Total P&L",
                        "value": f"{float(row['Total P&L (RM)']):,.2f}",
                        "badge": "All",
                        "tone": _tone_pnl(float(row['Total P&L (RM)'])),
                    },
                    {
                        "label": "Dividend Yield",
                        "value": (f"{float(row['Avg Div Yield NET (%)']):.2f}%" if row.get('Avg Div Yield NET (%)') is not None else "—"),
                        "badge": "TTM NET",
                    },
                    {
                        "label": "Proj Dividend (RM)",
                        "value": f"{float(row['Proj Dividend NET (RM)']):,.2f}",
                        "badge": "TTM NET",
                    },
                ],
                columns=4,
            )

            show_cols = [
                "Name",
                "Strategy",
                "OpenDate",
                "Entry",
                "Current",
                "Shares",
                "Cost (RM)",
                "Value (RM)",
                "Unrealized PnL (RM)",
                "Return (%)",
                "R live",
                "Dividend Yield (TTM, %)",
                "Dividend (Net RM)",
            ]
            show_cols = [c for c in show_cols if c in sub.columns]
            show = sub[show_cols].copy()
            st.dataframe(show, use_container_width=True, hide_index=True)
else:
    st.info("No FriendName column found in the Friend Trades file yet.")




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

table = filtered.drop(columns=["TTM DPS (RM)", "Dividend (Gross RM)", "Tax %", "Dividend (Net RM)", "Dividend Yield (TTM, %)"], errors="ignore").copy()
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
                ok = io_helpers.close_friend_open_trade_row(int(r["RowId"]), float(price), reason_txt)
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
        return bool(io_helpers.close_friend_open_trade_row(int(row["RowId"]), px, reason_txt))
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

