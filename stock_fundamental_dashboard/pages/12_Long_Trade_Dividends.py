# pages/12_Long_Trade_Dividends.py
from __future__ import annotations

from auth_gate import require_auth
require_auth()

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# ---- UI helpers (robust) ----
try:
    from utils.ui import setup_page, section, render_stat_cards
except Exception:
    from ui import setup_page, section, render_stat_cards  # type: ignore

# ---- IO helpers (robust) ----
import os, sys
_THIS   = os.path.dirname(__file__)
_PARENT = os.path.abspath(os.path.join(_THIS, ".."))
_GRANDP = os.path.abspath(os.path.join(_PARENT, ".."))
for p in (_PARENT, _GRANDP):
    if p not in sys.path:
        sys.path.insert(0, p)
try:
    from utils import io_helpers
except Exception:
    import io_helpers  # type: ignore

# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------
setup_page("Long Trade — Dividends")

st.markdown(
    section(
        "📮 Long Trade — Dividend Tracker",
        "A focused view of your long-term holdings and their projected dividends per year.",
    ),
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────
# Load open positions & filter to LONG
# ─────────────────────────────────────────
open_df = io_helpers.load_open_trades()
if open_df is None or open_df.empty:
    st.info("You have no open positions yet.")
    st.stop()

def _is_long_strategy(s: str | None) -> bool:
    return isinstance(s, str) and ("long" in s.lower())

long_df = open_df[open_df["Strategy"].apply(_is_long_strategy)].copy()
if long_df.empty:
    st.info("No Long Trade positions found.")
    st.stop()

# ─────────────────────────────────────────
# Filters (search by name + recent period)
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec"><div class="t">🔎 Filters</div>'
    '<div class="d">Narrow the list of Long Trade holdings</div></div>',
    unsafe_allow_html=True,
)

f1, f2 = st.columns([2, 1])
with f1:
    q = st.text_input("Search name", placeholder="Type part of a stock name…")
with f2:
    period = st.selectbox(
        "Opened within",
        ["All", "Last 7 days", "Last 14 days", "Last 1 month", "Last 3 months"],
        index=0,
    )

filtered = long_df.copy()
if q:
    filtered = filtered[filtered["Name"].astype(str).str.contains(q, case=False, na=False)]

if period != "All" and "OpenDate" in filtered.columns:
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
# Helper: latest price from Add/Edit dataset
# ─────────────────────────────────────────
def _latest_price_for(name: str) -> float | None:
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

# add a simple live P/L preview
filtered = filtered.copy()
filtered["Entry"]  = pd.to_numeric(filtered.get("Entry"), errors="coerce")
filtered["Shares"] = pd.to_numeric(filtered.get("Shares"), errors="coerce")
filtered["Price"]  = filtered["Name"].apply(_latest_price_for)
filtered["Price"]  = pd.to_numeric(filtered["Price"], errors="coerce")
filtered["Unrealized P/L (RM)"] = ((filtered["Price"] - filtered["Entry"]) * filtered["Shares"]).round(2)

# ─────────────────────────────────────────
# Dividend Projection: DPS × Shares → totals by Year
# ─────────────────────────────────────────
src_df = io_helpers.load_data()
dps_by_year = pd.DataFrame(columns=["Year", "Total Dividend (RM)"])
breakdown  = pd.DataFrame(columns=["Year", "Name", "Dividend (RM)"])

if src_df is not None and not src_df.empty and "Name" in src_df.columns:
    df_all = src_df.copy()
    if "IsQuarter" in df_all.columns:
        df_all = df_all[df_all["IsQuarter"] != True]
    if "Year" in df_all.columns:
        df_all["Year"] = pd.to_numeric(df_all["Year"], errors="coerce")

    DPS_CANDIDATES = ["DPS", "Dividend per Share (TTM, RM)", "DPU"]
    def _pick_dps_col(df: pd.DataFrame) -> str | None:
        for c in DPS_CANDIDATES:
            if c in df.columns:
                return c
        for c in df.columns:
            cc = str(c).lower()
            if ("dps" in cc) or ("dividend per share" in cc) or ("dpu" in cc):
                return c
        return None

    dps_col = _pick_dps_col(df_all)
    if dps_col:
        lp = filtered[["Name", "Shares"]].copy()
        lp["Shares"] = pd.to_numeric(lp["Shares"], errors="coerce").fillna(0)

        annual = df_all[["Name", "Year", dps_col]].copy()
        annual[dps_col] = pd.to_numeric(annual[dps_col], errors="coerce")

        merged = (
            annual.merge(lp, on="Name", how="inner")
                  .dropna(subset=["Year", dps_col])
        )
        merged["Dividend (RM)"] = (merged[dps_col] * merged["Shares"]).round(2)

        dps_by_year = (
            merged.groupby("Year", as_index=False)["Dividend (RM)"].sum()
                  .rename(columns={"Dividend (RM)": "Total Dividend (RM)"})
                  .sort_values("Year")
        )
        breakdown = (
            merged.groupby(["Year", "Name"], as_index=False)["Dividend (RM)"].sum()
                  .sort_values(["Year", "Name"])
        )

# KPIs
latest_val = "—"
if not dps_by_year.empty:
    now_year = pd.Timestamp.now().year
    row = dps_by_year[dps_by_year["Year"] == now_year]
    if row.empty:
        row = dps_by_year.tail(1)
    if not row.empty:
        latest_val = f'{float(row["Total Dividend (RM)"].iloc[0]):,.2f}'

render_stat_cards(
    [
        {"label": "Projected Dividend (latest year)", "value": latest_val, "badge": "RM", "tone": "good"},
        {"label": "Long Positions", "value": f"{len(filtered):,}", "badge": "Count"},
    ],
    columns=2,
)

c1, c2 = st.columns([2, 3])
with c1:
    st.dataframe(
        dps_by_year,
        use_container_width=True,
        height=min(260, 72 + 28*max(1, min(len(dps_by_year), 8))),
    )
with c2:
    st.dataframe(
        breakdown,
        use_container_width=True,
        height=min(260, 72 + 28*max(1, min(len(breakdown), 8))),
    )

st.caption("Projection uses **DPS × Shares** per year from your master dataset. Update DPS/Year in Add/Edit to refine this.")

# ─────────────────────────────────────────
# Long Trade holdings table (read-only)
# ─────────────────────────────────────────
show_cols = [c for c in ["Name","Strategy","OpenDate","Entry","Shares","Price","Unrealized P/L (RM)","TP1","TP2","TP3","Stop","RR"] if c in filtered.columns]
st.markdown(
    '<div class="sec"><div class="t">📋 Long Trade Holdings (read-only)</div>'
    '<div class="d">This mirrors Ongoing Trades but scoped to Long Trade only.</div></div>',
    unsafe_allow_html=True,
)
st.dataframe(
    filtered[show_cols].sort_values("Name"),
    use_container_width=True,
    height=min(420, 96 + 28*max(4, min(len(filtered), 12))),
)
