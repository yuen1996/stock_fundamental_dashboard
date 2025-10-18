# pages/12_Long_Trade_Dividends.py
from __future__ import annotations

from auth_gate import require_auth
require_auth()

import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os, sys

# ---------------- UI helpers (robust like your Ongoing page) ----------------
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
    from ui import (  # type: ignore
        setup_page,
        section,
        register_ongoing_trades_css,
        render_stat_cards,
    )
    try:
        from ui import render_page_title  # type: ignore
    except Exception:
        def render_page_title(page_name: str) -> None:
            st.title(f"📊 Fundamentals Dashboard — {page_name}")

# ---------------- IO helpers (robust) ----------------
try:
    from utils import io_helpers
except Exception:
    import io_helpers  # type: ignore
    
# NEW: Momentum filename resolver + optional OHLC etag bus
from utils import name_link
try:
    from utils import bus
except Exception:
    bus = None  # graceful fallback if utils.bus isn't available
   
# ---------- Page setup ----------
setup_page("Long Trade — Dividends")
render_page_title("Long Trade — Dividends")
try:
    register_ongoing_trades_css()
except Exception:
    pass

# --- Dialog styles: reuse same CSS for Inspect + Close Wizard ---
st.markdown("""
<style>
.inspect { padding-top:.25rem; }
.inspect .muted { color:#6b7280;font-size:.9rem;margin:0 0 .5rem 0; }
.inspect .kpi { text-align:center; padding:.35rem .25rem .8rem; }
.inspect .kpi h4 { margin:.15rem 0 .25rem 0; font-weight:600; color:#6b7280; }
.inspect .kpi .v  { font-size:2rem; font-weight:700; letter-spacing:.2px; }
.inspect .pos { color:#10b981; } .inspect .neg { color:#ef4444; }
.inspect .pill { display:inline-block; padding:.15rem .5rem; border-radius:999px; font-weight:600; font-size:.85rem; }
.inspect .pill.neutral { background:#e5e7eb; color:#111827; }
.inspect .pill.good { background:#d1fae5; color:#065f46; }
.inspect .pill.bad  { background:#fee2e2; color:#7f1d1d; }
.inspect .subtle { color:#6b7280; font-size:.85rem; }

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

def _safe_rerun():
    try: st.rerun()
    except Exception:
        try: st.experimental_rerun()
        except Exception: pass
        
# --- Version etag to invalidate caches when fundamentals / OHLC change ---
_THIS = os.path.dirname(__file__)
_GRANDP = os.path.abspath(os.path.join(_THIS, "..", ".."))
_VERSION_FILE = os.path.join(_GRANDP, ".data.version")

def _data_etag() -> int:
    """Bumps whenever Add/Edit writes data (used to key caches)."""
    try:
        return int(os.stat(_VERSION_FILE).st_mtime_ns)
    except Exception:
        return 0

def _resolve_ohlc_dir() -> str:
    """Use the same search logic as Momentum to find data/ohlc."""
    _PARENT = os.path.abspath(os.path.join(_THIS, ".."))
    candidates = [
        os.path.abspath(os.path.join(_PARENT, "data", "ohlc")),   # canonical
        os.path.abspath(os.path.join(os.getcwd(), "data", "ohlc")),
        os.path.abspath(os.path.join(_THIS, "..", "data", "ohlc")),
        os.path.abspath(os.path.join(_GRANDP, "data", "ohlc")),
    ]
    for d in candidates:
        if os.path.isdir(d):
            return d
    return candidates[0]

def _ohlc_dir_etag() -> int:
    """Cache-buster for OHLC changes (prefer bus.etag if available)."""
    try:
        return int(bus.etag("ohlc")) if bus else int(os.stat(_resolve_ohlc_dir()).st_mtime_ns)
    except Exception:
        return 0      

# ─────────────────────────────────────────
# Load open positions & scope to LongTrade
# ─────────────────────────────────────────
open_df = io_helpers.load_open_trades()
if open_df.empty:
    st.info("No ongoing trades. Use **Systematic Decision → PASS → Push selected (Long trade — DY > EPF)**, then **Manage Queue → Mark Live**.", )
    st.stop()

open_df = open_df.reset_index().rename(columns={"index": "RowId"})

def _is_longtrade(x) -> bool:
    # exact match "LongTrade" from Systematic Decision push; accept "long" substring as safety
    s = str(x or "").strip().lower()
    return s == "longtrade" or "long" in s

long_df = open_df[open_df["Strategy"].apply(_is_longtrade)].copy()
if long_df.empty:
    st.info("No **LongTrade** positions found. Push from Systematic Decision with the Long trade button first.", )
    st.stop()

# === Live price (AUTO) — Momentum OHLC (close before today) → fallback Add/Edit ===
@st.cache_data(show_spinner=False)
def _load_ohlc_for_name(stock_name: str, _etag: int, *, ticker: str | None = None) -> pd.DataFrame | None:
    path = name_link.find_ohlc_path(stock_name, ohlc_dir=_resolve_ohlc_dir(), ticker=ticker)
    if not path or not os.path.exists(path):
        return None
    dfp = pd.read_csv(path)
    dfp["Date"] = pd.to_datetime(dfp["Date"], errors="coerce")
    if "Close" not in dfp.columns and "Adj Close" in dfp.columns:
        dfp["Close"] = pd.to_numeric(dfp["Adj Close"], errors="coerce")
    for c in ["Open","High","Low","Close","Volume"]:
        if c in dfp.columns:
            dfp[c] = pd.to_numeric(dfp[c], errors="coerce")
    dfp = (
        dfp.dropna(subset=["Date","Close"])
           .drop_duplicates(subset=["Date"])
           .sort_values("Date")
           .reset_index(drop=True)
    )
    return dfp

def _current_price_from_addedit(name: str) -> float | None:
    df = io_helpers.load_data()
    if df is None or df.empty:
        return None
    rows = df[df["Name"].astype(str).str.upper() == str(name).upper()]
    if rows.empty:
        return None
    for col in ("CurrentPrice", "EndQuarterPrice", "Price", "SharePrice"):
        if col in rows.columns:
            s = pd.to_numeric(rows[col], errors="coerce").dropna()
            if not s.empty:
                v = float(s.iloc[-1])
                if np.isfinite(v):
                    return v
    return None

def _momentum_close_before_today(name: str, stock_rows: pd.DataFrame | None = None) -> float | None:
    # optional ticker/code hint (helps match file names)
    ticker_hint = None
    sr = stock_rows if stock_rows is not None else io_helpers.load_data()
    try:
        sub = sr[sr["Name"].astype(str).str.upper() == str(name).upper()] if isinstance(sr, pd.DataFrame) else None
        if sub is not None and not sub.empty:
            for c in ("Ticker", "Code", "Symbol"):
                if c in sub.columns:
                    v = str(sub[c].dropna().iloc[-1]).strip()
                    if v:
                        ticker_hint = v
                        break
    except Exception:
        pass

    dfp = _load_ohlc_for_name(name, _ohlc_dir_etag(), ticker=ticker_hint)
    if dfp is None or dfp.empty:
        return None

    today = pd.Timestamp.today().normalize()
    prior = dfp[dfp["Date"] < today]  # avoid today’s partial bar
    if prior.empty:
        return None

    px = float(prior.iloc[-1]["Close"])
    return px if np.isfinite(px) else None

@st.cache_data(show_spinner=False)
def _live_price_for_cached(name: str, _etag_data: int, _etag_ohlc: int) -> float | None:
    # 1) Momentum OHLC close before today
    mom = _momentum_close_before_today(name)
    if mom is not None:
        return mom
    # 2) Fallback to Add/Edit current
    return _current_price_from_addedit(name)

def _live_price_for(name: str) -> float | None:
    return _live_price_for_cached(name, _data_etag(), _ohlc_dir_etag())

# ─────────────────────────────────────────
# Filters (same UX, but Strategy fixed = LongTrade)
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec"><div class="t">🔎 Filters</div>'
    '<div class="d">Narrow the list of LongTrade holdings</div></div>',
    unsafe_allow_html=True,
)

f1, f2 = st.columns([2, 1])
with f1:
    q = st.text_input("Search name", placeholder="Type part of a stock name…")
with f2:
    period = st.selectbox(
        "Opened in",
        ["Any", "Last 7 days", "Last 14 days", "Last 1 month", "Last 3 months"],
        index=0,
    )

filtered = long_df.copy()
if q.strip():
    qq = q.lower()
    filtered = filtered[filtered["Name"].str.lower().str.contains(qq, na=False)]
if period != "Any" and "OpenDate" in filtered.columns:
    now = datetime.now()
    cutoff = now - timedelta(days=7 if period=="Last 7 days" else 14 if period=="Last 14 days" else 30 if period=="Last 1 month" else 90)
    dt = pd.to_datetime(filtered["OpenDate"], errors="coerce")
    filtered = filtered[dt >= cutoff]

if filtered.empty:
    st.info("No rows match the current filters.")
    st.stop()

# === Live columns (identical to Ongoing) =====================================
to_num = lambda c: pd.to_numeric(filtered.get(c), errors="coerce")
shares = to_num("Shares").fillna(0)
entry  = to_num("Entry").fillna(0)
stop   = to_num("Stop")
tp1    = to_num("TP1")
tp2    = to_num("TP2")
tp3    = to_num("TP3")

# Momentum “Auto” like View Stock: OHLC close (yesterday) → Add/Edit fallback
filtered["Current"] = filtered["Name"].map(_live_price_for).astype("float")
cur = pd.to_numeric(filtered.get("Current"), errors="coerce")

filtered["Value (RM)"]       = (shares * cur).round(2)
filtered["Cost (RM)"]        = (shares * entry).round(2)
filtered["Unrealized PnL (RM)"] = (shares * (cur - entry)).round(2)
filtered["Return (%)"]       = ((cur / entry - 1.0) * 100.0).where(entry > 0).round(2)

risk_ps = (entry - stop)
filtered["R live"] = ((cur - entry) / risk_ps).where((risk_ps > 0), other=pd.NA).round(2)

def _delta_pct(target, cur_px):
    if pd.isna(target) or pd.isna(cur_px) or target == 0: return None
    return (target - cur_px) / target * 100.0

filtered["Δ to TP1 (RM)"] = (tp1 - cur).round(4)
filtered["Δ to TP2 (RM)"] = (tp2 - cur).round(4)
filtered["Δ to TP3 (RM)"] = (tp3 - cur).round(4)
filtered["Δ to TP1 (%)"]  = filtered.apply(lambda r: round(_delta_pct(r.get("TP1"), r.get("Current")), 2) if pd.notna(r.get("TP1")) else None, axis=1)
filtered["Δ to TP2 (%)"]  = filtered.apply(lambda r: round(_delta_pct(r.get("TP2"), r.get("Current")), 2) if pd.notna(r.get("TP2")) else None, axis=1)
filtered["Δ to TP3 (%)"]  = filtered.apply(lambda r: round(_delta_pct(r.get("TP3"), r.get("Current")), 2) if pd.notna(r.get("TP3")) else None, axis=1)
filtered["Δ to SL (RM)"]  = (cur - stop).round(4)
filtered["SL Breach?"]    = (cur <= stop).where(stop.notna())

# ─────────────────────────────────────────
# Dividend Tracker (entry-aware, tax-aware) — NEW
# ─────────────────────────────────────────
st.markdown(
    section("📮 Dividend Tracker", "Projected dividends per year from DPS × Shares × held-fraction (from your **OpenDate**).", "info"),
    unsafe_allow_html=True,
)

s1, s2 = st.columns([1,1])
with s1:
    default_tax_pct = st.number_input("Default Dividend Tax %", min_value=0.0, max_value=60.0, value=0.0, step=0.5)
with s2:
    prorate_mode = st.selectbox("Prorate method", ["Daily (precise)", "Monthly (simple)"], index=0)

# === TTM DPS helpers =========================================================
def _ttm_dps_from_session(name: str) -> float | None:
    """Prefer DPS from the View Stock page's TTM KPI cache (exactly what you saw there)."""
    try:
        cache = st.session_state.get("TTM_KPI_SYNC", {})
        rec   = cache.get(str(name)) or {}
        vals  = rec.get("values") or {}
        v     = vals.get("DPS")
        if v is None:
            return None
        v = float(v)
        return v if np.isfinite(v) and v >= 0 else None
    except Exception:
        return None

def _find_q_dps_col(df: pd.DataFrame) -> str | None:
    """Choose a quarterly DPS column (Q_* preferred) if present."""
    if df is None or df.empty:
        return None
    cands = []
    for c in df.columns:
        s = str(c).lower()
        if ("dps" in s or "dividend per share" in s or "dpu" in s):
            cands.append(c)
    for c in cands:
        if str(c).lower().startswith("q_"):
            return c
    return cands[0] if cands else None

def _ttm_dps_from_quarters(name: str) -> float | None:
    """Fallback: sum the latest up to 4 quarters of DPS from io_helpers.load_data()."""
    try:
        df = io_helpers.load_data()
        if df is None or df.empty:
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
    """Unified accessor: View-Stock cache → quarterly sum → None."""
    v = _ttm_dps_from_session(name)
    if v is None:
        v = _ttm_dps_from_quarters(name)
    return v
# ============================================================================

# --- Per-trade TTM DPS + Dividend (Gross/Net) -------------------------------
# Sync per-row Tax% (if present), else default from the Dividend Tracker section
tax_default = float(default_tax_pct or 0.0)

filtered["TTM DPS (RM)"] = filtered["Name"].map(lambda n: get_ttm_dps(str(n))).astype("float")
filtered["Dividend (Gross RM)"] = (
    pd.to_numeric(filtered["Shares"], errors="coerce").fillna(0)
    * pd.to_numeric(filtered["TTM DPS (RM)"], errors="coerce").fillna(0)
).round(2)

def _row_tax_pct(row) -> float:
    try:
        v = float(row.get("TaxPct"))
        if np.isfinite(v):
            return max(0.0, min(60.0, v))
    except Exception:
        pass
    return tax_default

filtered["Tax %"] = filtered.apply(_row_tax_pct, axis=1)
filtered["Dividend (Net RM)"] = (
    filtered["Dividend (Gross RM)"] * (1.0 - filtered["Tax %"] / 100.0)
).round(2)

# KPI value: use sum of per-trade NET dividends
latest_div = f'{float(pd.to_numeric(filtered["Dividend (Net RM)"], errors="coerce").sum()):,.2f}'


def _year_bounds(y: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=y, month=1, day=1)
    end   = pd.Timestamp(year=y, month=12, day=31, hour=23, minute=59, second=59)
    return start, end

def _days_in_year(y: int) -> int:
    s, e = _year_bounds(y); return (e - s).days + 1

def _held_fraction_for_year(open_dt: pd.Timestamp, year: int, today: pd.Timestamp, mode: str) -> float:
    y_start, y_end = _year_bounds(year)
    start = max(open_dt.normalize(), y_start)
    end   = min(today.normalize(), y_end)
    if end < y_start or start > y_end or end < start: return 0.0
    if mode.startswith("Daily"):
        return ((end - start).days + 1) / _days_in_year(year)
    # monthly (simple)
    sm = max(1, start.month) if start.year == year else 1
    em = min(12, end.month)  if end.year == year else 12
    return max(0.0, min(1.0, (em - sm + 1) / 12.0))

def _pick_dps_col(df: pd.DataFrame) -> str | None:
    for c in ["DPS", "Dividend per Share (TTM, RM)", "DPU"]:
        if c in df.columns: return c
    for c in df.columns:
        cc = str(c).lower()
        if ("dps" in cc) or ("dividend per share" in cc) or ("dpu" in cc):
            return c
    return None

today = pd.Timestamp.now()
lp = filtered[["Name","Shares","OpenDate"]].copy()
lp["Shares"]   = pd.to_numeric(lp["Shares"], errors="coerce").fillna(0)
lp["OpenDate"] = pd.to_datetime(lp["OpenDate"], errors="coerce")
lp["TaxPct"]   = pd.to_numeric(filtered.get("TaxPct"), errors="coerce") if "TaxPct" in filtered.columns else np.nan

src_df = io_helpers.load_data()
dps_by_year = pd.DataFrame(columns=["Year","Total Dividend (Net RM)"])
breakdown  = pd.DataFrame(columns=["Year","Name","Dividend (Net RM)","Dividend (Gross RM)","Held %","Tax %"])

if src_df is not None and not src_df.empty and "Name" in src_df.columns:
    df_all = src_df.copy()
    if "IsQuarter" in df_all.columns:
        df_all = df_all[df_all["IsQuarter"] != True]
    if "Year" in df_all.columns:
        df_all["Year"] = pd.to_numeric(df_all["Year"], errors="coerce")

    dps_col = _pick_dps_col(df_all)
    if dps_col:
        annual = df_all[["Name","Year",dps_col]].dropna(subset=["Year"]).copy()
        annual["Year"] = pd.to_numeric(annual["Year"], errors="coerce").astype("Int64")
        annual[dps_col] = pd.to_numeric(annual[dps_col], errors="coerce")

        merged = annual.merge(lp, on="Name", how="inner")

        merged["Held %"] = merged.apply(
            lambda r: (
                0.0 if pd.isna(r["OpenDate"]) or pd.isna(r["Year"])
                else _held_fraction_for_year(pd.Timestamp(r["OpenDate"]), int(r["Year"]), today, prorate_mode) * 100.0
            ),
            axis=1
        )

        merged["Dividend (Gross RM)"] = (merged[dps_col] * merged["Shares"] * (merged["Held %"]/100.0)).round(2)

        def _tax_pct(row) -> float:
            try:
                v = float(row.get("TaxPct"))
                if np.isfinite(v): return max(0.0, min(60.0, v))
            except Exception:
                pass
            return float(default_tax_pct or 0.0)

        merged["Tax %"] = merged.apply(_tax_pct, axis=1)
        merged["Dividend (Net RM)"] = (merged["Dividend (Gross RM)"] * (1.0 - merged["Tax %"]/100.0)).round(2)

        dps_by_year = (
            merged.groupby("Year", as_index=False)["Dividend (Net RM)"].sum()
                  .rename(columns={"Dividend (Net RM)":"Total Dividend (Net RM)"})
                  .sort_values("Year")
                  .reset_index(drop=True)
        )

        breakdown = (
            merged.groupby(["Year","Name"], as_index=False)
                  .agg(**{
                      "Dividend (Net RM)":   ("Dividend (Net RM)","sum"),
                      "Dividend (Gross RM)": ("Dividend (Gross RM)","sum"),
                      "Held %":              ("Held %","mean"),
                      "Tax %":               ("Tax %","mean"),
                  })
                  .sort_values(["Year","Name"])
                  .reset_index(drop=True)
        )

# ─────────────────────────────────────────
# Overview (KPI cards) — same as Ongoing + extra Dividend KPI
# ─────────────────────────────────────────
st.markdown(
    section("📊 Overview", "Open LongTrade positions & exposure (filtered)"),
    unsafe_allow_html=True,
)

total_cost = float((shares * entry).sum())
cur_value  = float((shares * cur).sum(skipna=True))
open_pnl   = float((shares * (cur - entry)).sum(skipna=True))

rr_init = pd.to_numeric(filtered.get("RR"), errors="coerce")
avg_rr  = float(rr_init.dropna().mean()) if rr_init.notna().any() else None

open_dates = pd.to_datetime(filtered.get("OpenDate"), errors="coerce")
avg_hold_days = int((pd.Timestamp.now() - open_dates).dt.days.dropna().mean()) if open_dates.notna().any() else None

def _tone_rr(v):
    if v is None: return ""
    if v >= 2.0: return "good"
    if v >= 1.5: return "warn"
    return "bad"

def _tone_pnl(v):
    if v > 0: return "good"
    if v < 0: return "bad"
    return "neutral"

render_stat_cards(
    [
        {"label": "Open Positions",     "value": f"{len(filtered):,}", "badge": "Shown"},
        {"label": "Total Cost (RM)",    "value": f"{total_cost:,.2f}", "badge": "Exposure"},
        {"label": "Cur Value (RM)",     "value": f"{cur_value:,.2f}",  "badge": "Marked"},
        {"label": "Unrealized P&L",     "value": f"{open_pnl:,.2f}",   "badge": "Live", "tone": _tone_pnl(open_pnl)},
        {"label": "Avg RR Init",        "value": (f"{avg_rr:.2f}×" if avg_rr is not None else "—"), "badge": "Quality", "tone": _tone_rr(avg_rr)},
        {"label": "Avg Holding (d)",    "value": (f"{avg_hold_days:,}" if avg_hold_days is not None else "—"), "badge": "Duration"},
        {"label": "Projected Dividend (NET)", "value": latest_div, "badge": "Latest Year RM", "tone": "good"},
    ],
    columns=3,
)

# Compact note: we now use TTM DPS per trade (no yearly tables)
st.caption(
    "Dividends per position use **TTM DPS × Shares**. "
    "Net = Gross × (1 − Tax%). TTM DPS is taken from **View Stock → TTM KPI (DPS)** if available; "
    "else it sums the latest 4 quarters’ DPS."
)

# ─────────────────────────────────────────
# Editor table (filtered set) — same as Ongoing
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec success"><div class="t">🧾 Open LongTrade Positions</div>'
    '<div class="d">Specify Close Price &amp; Reason, then tick rows to close</div></div>',
    unsafe_allow_html=True,
)

CLOSE_REASONS = [
    "Target hit","Stop hit","Trailing stop","Time stop","Thesis changed","Portfolio rebalance","Other (specify)",
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
        "Stop":  st.column_config.NumberColumn("Stop",  format="%.4f", disabled=True),
        "TP1":   st.column_config.NumberColumn("TP1",   format="%.4f", disabled=True),
        "TP2":   st.column_config.NumberColumn("TP2",   format="%.4f", disabled=True),
        "TP3":   st.column_config.NumberColumn("TP3",   format="%.4f", disabled=True),
        "Take":  st.column_config.NumberColumn("Take",  format="%.4f", disabled=True),

        "Shares": st.column_config.NumberColumn("Shares", format="%d", disabled=True),
        "RR":     st.column_config.NumberColumn("RR Init", format="%.2f", disabled=True),
        "OpenDate": st.column_config.TextColumn("Open Date", disabled=True),
        "Reasons":  st.column_config.TextColumn("Notes", disabled=True),

        "TTM DPS (RM)":        st.column_config.NumberColumn("TTM DPS (RM)", format="%.4f", disabled=True),
        "Dividend (Gross RM)": st.column_config.NumberColumn("Dividend (Gross RM)", format="%.2f", disabled=True),
        "Tax %":               st.column_config.NumberColumn("Tax %", format="%.2f", disabled=True),
        "Dividend (Net RM)":   st.column_config.NumberColumn("Dividend (Net RM)", format="%.2f", disabled=True),

        "Current": st.column_config.NumberColumn("Current", format="%.4f", disabled=True),
        "R live":  st.column_config.NumberColumn("R live", format="%.2f", disabled=True),
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
        "ClosePrice":  st.column_config.NumberColumn("Close Price", format="%.4f"),
        "CloseReason": st.column_config.SelectboxColumn("Close Reason", options=CLOSE_REASONS),
        "Detail":      st.column_config.TextColumn("Detail (if Other)"),
    },
    column_order=[
        "Select","RowId","Name","Strategy","Entry","Stop","TP1","TP2","TP3","Take",
        "Shares","RR","OpenDate","Current","R live","Return (%)","Unrealized PnL (RM)",
        "Δ to TP1 (RM)","Δ to TP2 (RM)","Δ to TP3 (RM)","Δ to SL (RM)","SL Breach?","Reasons",
        "ClosePrice","CloseReason","Detail",
        "TTM DPS (RM)","Dividend (Gross RM)","Tax %","Dividend (Net RM)",
    ],
    key="lt_open_trades_editor",
)

# ─────────────────────────────────────────
# Inspect dialog — same
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
    tp1   = row.get("TP1"); tp2 = row.get("TP2"); tp3 = row.get("TP3")
    shares = int(row.get("Shares") or 0)
    pnl   = row.get("Unrealized PnL (RM)")
    ret   = row.get("Return (%)")
    rlive = row.get("R live")

    def _pct_to_target(tp):
        try:
            if tp is None or np.isnan(tp) or entry is None or np.isnan(entry) or cur is None or np.isnan(cur): return None
            denom = float(tp) - float(entry)
            if denom <= 0: return None
            return float(np.clip((float(cur) - float(entry)) / denom, 0.0, 1.0))
        except Exception:
            return None

    def _sl_buffer():
        try:
            if stop is None or np.isnan(stop) or entry is None or np.isnan(entry) or cur is None or np.isnan(cur): return None
            risk = float(entry) - float(stop)
            if risk <= 0: return None
            return float(np.clip((float(cur) - float(stop)) / risk, 0.0, 1.0))
        except Exception:
            return None

    p1 = _pct_to_target(tp1); p2 = _pct_to_target(tp2); p3 = _pct_to_target(tp3)
    ps = _sl_buffer()
    sl_breached = (stop is not None and not pd.isna(stop) and cur is not None and cur <= stop)

    pill_tone = "neutral"
    if isinstance(rlive, (int, float)) and np.isfinite(rlive):
        pill_tone = "good" if rlive >= 1.0 else ("neutral" if rlive >= 0 else "bad")
    pnl_cls = "pos" if (isinstance(pnl, (int, float)) and pnl >= 0) else "neg"

    if hasattr(st, "dialog"):
        @st.dialog(title)
        def _dlg():
            st.markdown(f"<div class='inspect'><div class='muted'>Opened: {row.get('OpenDate','—')}</div></div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("<div class='inspect kpi'><h4>Entry</h4>"  f"<div class='v'>{entry:.4f}</div></div>", unsafe_allow_html=True)
            with c2:
                st.markdown("<div class='inspect kpi'><h4>Current</h4>"f"<div class='v'>{(f'{cur:.4f}' if pd.notna(cur) else '—')}</div></div>", unsafe_allow_html=True)
            with c3:
                st.markdown("<div class='inspect kpi'><h4>R live</h4>" f"<div class='v'><span class='pill {pill_tone}'>{'—' if pd.isna(rlive) else f'{rlive:.2f}'}</span></div></div>", unsafe_allow_html=True)

            c4, c5, c6 = st.columns(3)
            with c4:
                st.markdown("<div class='inspect kpi'><h4>P&L (RM)</h4>" f"<div class='v {pnl_cls}'>{'—' if pd.isna(pnl) else f'{pnl:,.2f}'}</div></div>", unsafe_allow_html=True)
            with c5:
                st.markdown("<div class='inspect kpi'><h4>Return (%)</h4>" f"<div class='v'>{'—' if pd.isna(ret) else f'{ret:.2f}'}</div></div>", unsafe_allow_html=True)
            with c6:
                st.markdown("<div class='inspect kpi'><h4>Shares</h4>"    f"<div class='v'>{shares:,d}</div></div>", unsafe_allow_html=True)

            st.divider()
            st.markdown("**Targets & SL**")
            def _bar(lbl, pct, tp_val):
                if pct is None: st.caption(f"{lbl}: —")
                else:           st.progress(pct, text=f"{lbl}: {pct*100:.0f}%  (TP {tp_val:.4f})")
            _bar("TP1 progress", p1, tp1); _bar("TP2 progress", p2, tp2); _bar("TP3 progress", p3, tp3)

            if sl_breached: st.error(f"Stop-loss breached (SL {stop:.4f} ≥ Current {cur:.4f}).")
            elif ps is None: st.caption("SL buffer: —")
            else: st.progress(ps, text=f"SL buffer: {ps*100:.0f}% away from stop  (SL {stop:.4f})")

            st.caption(f"**Plan** — TP1 {tp1 if pd.notna(tp1) else '—'} | TP2 {tp2 if pd.notna(tp2) else '—'} | TP3 {tp3 if pd.notna(tp3) else '—'} | SL {stop if pd.notna(stop) else '—'}")
        _dlg()
    else:
        st.info("Update Streamlit (≥1.31) to show dialogs. Showing inline:")
        st.write(row[["Name","Strategy","Entry","Current","R live","Unrealized PnL (RM)","Return (%)","TP1","TP2","TP3","Stop"]])

if st.button("📌 Inspect first selected"):
    sel = edited[edited.get("Select") == True]
    if sel.empty: st.info("Tick a row first.")
    else: _open_inspect_dialog(sel.iloc[0])

# ─────────────────────────────────────────
# Close Wizard (identical behavior)
# ─────────────────────────────────────────
def _start_close_wizard(rows_df: pd.DataFrame):
    st.session_state["close_wizard_rows"] = rows_df.to_dict(orient="records")
    st.session_state["close_wizard_idx"] = 0
    st.session_state["close_wizard_active"] = True

def _clear_close_wizard_state():
    for k in ("close_wizard_rows","close_wizard_idx","close_wizard_active"):
        st.session_state.pop(k, None)

def _show_close_wizard_dialog():
    if not hasattr(st, "dialog"):
        st.warning("Update Streamlit (≥1.31) to use the pop-up wizard. Use inline inputs instead.")
        _clear_close_wizard_state()
        return

    @st.dialog("Complete close details")
    def _wiz():
        rows = st.session_state.get("close_wizard_rows", []) or []
        idx  = int(st.session_state.get("close_wizard_idx", 0))
        total = len(rows)
        if total == 0 or idx >= total:
            _clear_close_wizard_state()
            _safe_rerun()
            return

        r = rows[idx]
        name    = r.get("Name","—"); strat = r.get("Strategy","—")
        entry   = r.get("Entry"); current = r.get("Current")
        shares  = int(r.get("Shares") or 0)
        rlive   = r.get("R live"); pnl = r.get("Unrealized PnL (RM)")
        retpct  = r.get("Return (%)"); opened = r.get("OpenDate","—")
        default_price = current if (current is not None and not pd.isna(current)) else entry

        def _fmt4(x): return "—" if (x is None or (isinstance(x,(int,float)) and pd.isna(x))) else f"{float(x):.4f}"
        def _fmt2(x): return "—" if (x is None or (isinstance(x,(int,float)) and pd.isna(x))) else f"{float(x):.2f}"

        st.markdown(f"""
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

        CLOSE_REASONS = [
            "Target hit","Stop hit","Trailing stop","Time stop","Thesis changed","Portfolio rebalance","Other (specify)",
        ]
        with st.form(key=f"wiz_form_{idx}", clear_on_submit=False, border=True):
            price = st.number_input("Close Price", value=float(default_price or 0.0), min_value=0.0, step=0.001, format="%.4f", key=f"wiz_price_{idx}")
            reason = st.selectbox("Close Reason", CLOSE_REASONS, index=0, key=f"wiz_reason_{idx}")
            need_detail = (reason == "Other (specify)")
            detail = st.text_input("Detail (if Other)", key=f"wiz_detail_{idx}", placeholder="e.g., exit due to earnings risk")
            submit = st.form_submit_button("Close & Next ➜", type="primary", use_container_width=True)
            if submit:
                if price <= 0: st.error("Close Price must be > 0."); st.stop()
                if need_detail and not (detail or "").strip(): st.error("Please specify detail for 'Other'."); st.stop()
                reason_txt = reason if not need_detail else f"{reason}: {detail.strip()}"
                ok = io_helpers.close_open_trade_row(int(r["RowId"]), float(price), reason_txt)
                if ok: st.toast(f"Closed {name}", icon="✅")
                st.session_state["close_wizard_idx"] = idx + 1
                st.rerun()
    _wiz()

st.markdown(
    '<div class="sec warning"><div class="t">🔒 Actions</div>'
    '<div class="d">Click Close selected — any row missing Close Price / required detail will open a pop-up</div></div>',
    unsafe_allow_html=True,
)

selected_mask = edited.get("Select") == True
selected_rows = edited[selected_mask] if "Select" in edited.columns else edited.iloc[0:0]
st.caption(f"Selected: **{len(selected_rows)}** row(s)")

def _needs_wizard(row: pd.Series) -> bool:
    px = float(row.get("ClosePrice") or 0)
    reason = (row.get("CloseReason") or "").strip()
    detail = (row.get("Detail") or "").strip()
    if px <= 0: return True
    if reason == "Other (specify)" and not detail: return True
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
        needs = selected_rows[selected_rows.apply(_needs_wizard, axis=1)]
        valids = selected_rows.drop(needs.index, errors="ignore")

        closed = 0
        for _, r in valids.iterrows():
            if _close_row_now(r): closed += 1
        if closed: st.toast(f"Closed {closed} trade(s).", icon="✅")

        if not needs.empty:
            _start_close_wizard(needs)
        else:
            _safe_rerun()

if st.session_state.get("close_wizard_active", False) and not st.session_state.get("_wiz_rendered_this_run", False):
    st.session_state["_wiz_rendered_this_run"] = True
    _show_close_wizard_dialog()

