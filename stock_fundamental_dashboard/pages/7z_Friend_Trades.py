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
    from ui import setup_page, section, register_ongoing_trades_css, render_stat_cards, render_page_title  # type: ignore

from utils import io_helpers

setup_page("Friend Trades")
register_ongoing_trades_css()
render_page_title("Friend Trades")

# --- Version etag to invalidate caches when data changes ---
_THIS = os.path.dirname(__file__)
_GRANDP = os.path.abspath(os.path.join(_THIS, "..", ".."))
_VERSION_FILE = os.path.join(_GRANDP, ".data.version")

def _data_etag() -> int:
    try:
        return int(os.stat(_VERSION_FILE).st_mtime_ns)
    except Exception:
        return 0

def _safe_key(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name or "")).lower()


# --- Dividend helpers (shared with Ongoing Trades) ---
def _norm_name(x: str) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip()).upper()

def _safe_name(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x or "").strip().lower()).strip("_")

def _ttm_div_yield_pct_from_session(name: str) -> float | None:
    """
    Try to read TTM Dividend Yield (%) from View Stock's session caches.
    This is the most reliable source (computed on View Stock, then synced).
    """
    try:
        cache = st.session_state.get("TTM_KPI_SYNC", {}) or {}
        if not cache:
            return None

        # locate record by tolerant name matching
        rec = cache.get(str(name))
        if not rec:
            target = _norm_name(name)
            for k, v in cache.items():
                if _norm_name(k) == target:
                    rec = v
                    break
        rec = rec or {}
        vals = rec.get("values") or {}

        # direct common key
        for key in ("Dividend Yield (%)", "Dividend Yield %", "Dividend Yield", "Div Yield (%)", "Div Yield %"):
            if key in vals:
                v = vals.get(key)
                try:
                    v = float(v)
                    return v if np.isfinite(v) and v >= 0 else None
                except Exception:
                    pass

        # fuzzy: label contains both 'dividend' and 'yield'
        for k, vv in vals.items():
            kl = str(k).lower()
            if ("dividend" in kl) and ("yield" in kl):
                try:
                    v = float(vv)
                    return v if np.isfinite(v) and v >= 0 else None
                except Exception:
                    continue

        # fallback: consolidated ttm_dict_{safe}
        safe = _safe_name(name)
        ttm_dict = (
            st.session_state.get(f"ttm_dict_{safe}")
            or st.session_state.get(f"ttm_dict_{safe.lower()}")
            or {}
        )
        if isinstance(ttm_dict, dict):
            for key in ("Dividend Yield (%)", "Dividend Yield %", "Dividend Yield"):
                if key in ttm_dict:
                    try:
                        v = float(ttm_dict.get(key))
                        return v if np.isfinite(v) and v >= 0 else None
                    except Exception:
                        pass
            for k, vv in ttm_dict.items():
                kl = str(k).lower()
                if ("dividend" in kl) and ("yield" in kl):
                    try:
                        v = float(vv)
                        return v if np.isfinite(v) and v >= 0 else None
                    except Exception:
                        continue

        return None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def _ttm_div_yield_pct_from_data(_etag: int, name: str) -> float | None:
    """
    Read TTM Dividend Yield (%) from the fundamentals dataset (row where Year == 'TTM').
    This works even after a full app refresh, before View Stock is opened.
    """
    try:
        df = io_helpers.load_data()
        if df is None or df.empty or "Name" not in df.columns or "Year" not in df.columns:
            return None

        sub = df[df["Name"].astype(str).str.upper() == str(name).upper()].copy()
        if sub.empty:
            return None

        year_s = sub["Year"].astype(str).str.upper().str.strip()
        ttm = sub[year_s == "TTM"].copy()
        if ttm.empty:
            # sometimes stored as IsQuarter==False with Quarter=='TTM' (rare), try that too
            if "Quarter" in sub.columns:
                q_s = sub["Quarter"].astype(str).str.upper().str.strip()
                ttm = sub[q_s == "TTM"].copy()
        if ttm.empty:
            return None

        # pick a dividend-yield-like column
        col = None
        for c in ttm.columns:
            cl = str(c).lower()
            if ("dividend" in cl) and ("yield" in cl):
                col = c
                # prefer the standard label
                if cl.strip() in ("dividend yield (%)", "dividend yield %"):
                    break
        if not col:
            return None

        v = pd.to_numeric(ttm[col], errors="coerce").dropna()
        if v.empty:
            return None

        out = float(v.iloc[-1])
        return out if np.isfinite(out) and out >= 0 else None
    except Exception:
        return None

def get_ttm_div_yield_pct(name: str) -> float | None:
    v = _ttm_div_yield_pct_from_session(name)
    if v is None:
        v = _ttm_div_yield_pct_from_data(_data_etag(), str(name))
    return v

def _ttm_dps_from_session(name: str) -> float | None:
    """Read TTM DPS from View Stock's session caches (robust to label/name variants)."""
    try:
        cache = st.session_state.get("TTM_KPI_SYNC", {}) or {}
        if not cache:
            return None

        # locate record by tolerant name matching
        rec = cache.get(str(name))
        if not rec:
            target = _norm_name(name)
            for k, v in cache.items():
                if _norm_name(k) == target:
                    rec = v
                    break
        rec = rec or {}
        vals = rec.get("values") or {}

        # 1) direct "DPS"
        v = vals.get("DPS")

        # 2) fallback: any label containing DPS / dividend per share
        if v is None:
            for k, vv in vals.items():
                kl = str(k).lower()
                if ("dps" in kl) or ("dividend per share" in kl) or ("dividend/share" in kl):
                    v = vv
                    break

        # 3) fallback: View Stock also stores a consolidated ttm_dict_{safe} dict
        if v is None:
            safe = _safe_name(name)
            ttm_dict = (
                st.session_state.get(f"ttm_dict_{safe}")
                or st.session_state.get(f"ttm_dict_{safe.lower()}")
                or {}
            )
            if isinstance(ttm_dict, dict):
                v = ttm_dict.get("DPS")
                if v is None:
                    for k, vv in ttm_dict.items():
                        kl = str(k).lower()
                        if ("dps" in kl) or ("dividend per share" in kl):
                            v = vv
                            break

        if v is None:
            return None
        v = float(v)
        return v if np.isfinite(v) and v >= 0 else None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def _ttm_dps_from_quarters_cached(_etag: int, name: str) -> float | None:
    """
    Quarter-sum DPS fallback (only used if Dividend Yield % is not available).
    """
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
        # find a quarterly DPS column
        dps_col = None
        for c in q.columns:
            s = str(c).lower()
            if ("dps" in s) or ("dividend per share" in s) or ("dpu" in s):
                dps_col = c
                if s.startswith("q_"):
                    break
        if not dps_col:
            return None
        s = pd.to_numeric(q[dps_col], errors="coerce").dropna()
        if s.empty:
            return None
        v = float(s.tail(min(4, len(s))).sum())
        return v if np.isfinite(v) and v >= 0 else None
    except Exception:
        return None

def get_ttm_dps(name: str) -> float | None:
    """
    Priority:
      1) Dividend Yield % (session → data) => implied DPS = CurPrice * Yield%
      2) DPS from session (if available)
      3) Quarter-sum DPS (if dataset has quarterly DPS)
    This avoids the common bug where a 'DPS' field gets accidentally mapped to EPS after a full app refresh.
    """
    y = get_ttm_div_yield_pct(name)
    if y is not None:
        return None  # caller will compute implied DPS using its own current price
    v = _ttm_dps_from_session(name)
    if v is None:
        v = _ttm_dps_from_quarters_cached(_data_etag(), str(name))
    return v

@st.cache_data(show_spinner=False)
def _build_price_index(_etag: int) -> dict[str, float]:
    """Build a case/format-tolerant current price index from your fundamentals.csv."""
    df = io_helpers.load_data()
    if df is None or df.empty:
        return {}
    # candidates for price
    price_cols = [c for c in df.columns if c.lower() in ("currentprice", "price", "last", "close")]
    if not price_cols:
        return {}
    pcol = price_cols[0]
    out = {}
    for _, r in df.iterrows():
        name = str(r.get("Name") or "").strip()
        code = str(r.get("Code") or r.get("Ticker") or r.get("Symbol") or "").strip()
        try:
            px = float(pd.to_numeric(r.get(pcol), errors="coerce"))
        except Exception:
            px = np.nan
        if not name or not np.isfinite(px):
            continue
        for key in (name, code):
            if not key:
                continue
            k1 = str(key).upper().strip()
            out[k1] = float(px)
            out[re.sub(r"[^A-Z0-9]+", "", k1)] = float(px)
    return out

def _lookup_latest_price(name: str | None, code: str | None) -> float | None:
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

@st.cache_data(show_spinner=False)
def _build_ttm_dps_index(_etag: int) -> dict[str, float]:
    """Map stock Name -> TTM Dividend Per Share (RM)."""
    df = io_helpers.load_data()
    if df is None or df.empty or "Name" not in df.columns:
        return {}
    # pick best DPS column
    cols = list(df.columns)
    cand = [
        "Dividend per Share (TTM, RM)",
        "Dividend per Share (TTM)",
        "DPS (TTM)",
        "TTM DPS",
        "TTM_DPS",
        "DPS_TTM",
        "DividendPS_TTM",
    ]
    dps_col = None
    for c in cand:
        if c in cols:
            dps_col = c
            break
    # fallback: anything that looks like TTM DPS
    if dps_col is None:
        for c in cols:
            cl = c.lower()
            if ("dividend" in cl or "dps" in cl) and "ttm" in cl:
                dps_col = c
                break

    out: dict[str, float] = {}
    if dps_col is not None:
        for name, g in df.groupby("Name"):
            # prefer latest annual/non-quarter row if available
            gg = g.copy()
            # sort by Year, Quarter if exist
            if "Year" in gg.columns:
                gg["__y"] = pd.to_numeric(gg["Year"], errors="coerce")
            else:
                gg["__y"] = np.nan
            if "Quarter" in gg.columns:
                gg["__q"] = pd.to_numeric(gg["Quarter"], errors="coerce")
            else:
                gg["__q"] = np.nan
            gg = gg.sort_values(["__y", "__q"], ascending=[True, True])
            val = pd.to_numeric(gg[dps_col], errors="coerce").dropna()
            if not val.empty:
                out[str(name)] = float(val.iloc[-1])
        return out

    # If no TTM DPS column exists, try sum last 4 quarter DPS
    qcol = None
    for c in cols:
        cl = c.lower()
        if ("dividend per share" in cl or cl == "dps" or "dps" in cl) and "ttm" not in cl:
            qcol = c
            break
    if qcol is None:
        return {}
    for name, g in df.groupby("Name"):
        gg = g.copy()
        if "IsQuarter" in gg.columns:
            gg = gg[gg["IsQuarter"].astype(str).isin(["1","True","true","YES","Yes"])]
        if "Year" in gg.columns:
            gg["__y"] = pd.to_numeric(gg["Year"], errors="coerce")
        else:
            gg["__y"] = np.nan
        if "Quarter" in gg.columns:
            gg["__q"] = pd.to_numeric(gg["Quarter"], errors="coerce")
        else:
            gg["__q"] = np.nan
        gg = gg.sort_values(["__y","__q"], ascending=[True,True])
        vals = pd.to_numeric(gg[qcol], errors="coerce").dropna().tail(4)
        if not vals.empty:
            out[str(name)] = float(vals.sum())
    return out

# ─────────────────────────────────────────
# Load live positions & give each a RowId
# ─────────────────────────────────────────
open_df = io_helpers.load_friend_open_trades()
if open_df is None or open_df.empty:
    st.info("No friend trades yet. In **Systematic Decision → Manage Queue**, tick **Friend** and Mark Live.")
    st.stop()

open_df = open_df.reset_index().rename(columns={"index": "RowId"})

# Ensure columns
for c, default in [("FriendName",""), ("Shares", 0), ("Entry", np.nan), ("Stop", np.nan), ("Take", np.nan), ("Strategy","")]:
    if c not in open_df.columns:
        open_df[c] = default

# ─────────────────────────────────────────
# Filters (Search / Strategy / Opened in)
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec"><div class="t">🔎 Filters</div><div class="d">Filter across all friends (tables below update)</div></div>',
    unsafe_allow_html=True,
)
f1, f2, f3 = st.columns([2,1,1])
with f1:
    q = st.text_input("Search name", placeholder="Type part of a stock name…")
with f2:
    strategies = ["All"] + sorted([s for s in open_df["Strategy"].dropna().unique()])
    strat_sel = st.selectbox("Strategy", strategies, index=0)
with f3:
    period = st.selectbox("Opened in", ["Any","Last 7 days","Last 14 days","Last 1 month","Last 3 months"], index=0)

filtered = open_df.copy()
if q.strip():
    qq = q.lower()
    filtered = filtered[filtered["Name"].astype(str).str.lower().str.contains(qq, na=False)]
if strat_sel != "All":
    filtered = filtered[filtered["Strategy"] == strat_sel]
if period != "Any" and "OpenDate" in filtered.columns:
    now = datetime.now()
    cutoff = now - timedelta(days={"Last 7 days":7,"Last 14 days":14,"Last 1 month":30,"Last 3 months":90}[period])
    dt = pd.to_datetime(filtered["OpenDate"], errors="coerce")
    filtered = filtered[dt >= cutoff]

if filtered.empty:
    st.info("No rows match the current filters.")
    st.stop()

# ─────────────────────────────────────────
# Enrich with current price, P&L, and TTM dividends
# ─────────────────────────────────────────
to_num = lambda s: pd.to_numeric(filtered.get(s), errors="coerce")
filtered["Shares"] = pd.to_numeric(filtered.get("Shares"), errors="coerce").fillna(0).astype(int)
filtered["Entry"] = to_num("Entry").fillna(0.0)
filtered["Stop"]  = to_num("Stop")
filtered["Take"]  = to_num("Take")

filtered["Current"] = filtered.apply(
    lambda r: _lookup_latest_price(r.get("Name"), r.get("Code") or r.get("Ticker") or r.get("Symbol")),
    axis=1,
).astype("float")

cur = pd.to_numeric(filtered["Current"], errors="coerce")
shares = pd.to_numeric(filtered["Shares"], errors="coerce").fillna(0)
entry = pd.to_numeric(filtered["Entry"], errors="coerce").fillna(0)

filtered["Cost (RM)"] = (shares * entry).astype(float)
filtered["Value (RM)"] = (shares * cur).astype(float)
filtered["Unreal PnL (RM)"] = (shares * (cur - entry)).astype(float)
filtered["Return %"] = np.where(entry > 0, ((cur / entry) - 1.0) * 100.0, np.nan)

# --- Dividend Yield % (best source) + implied DPS (same as Ongoing Trades) ---
ttm_div_yield_pct = pd.to_numeric(filtered["Name"].map(lambda n: get_ttm_div_yield_pct(str(n))), errors="coerce")
ttm_dps_raw = pd.to_numeric(filtered["Name"].map(lambda n: get_ttm_dps(str(n))), errors="coerce")
ttm_dps_implied = np.where(cur > 0, (cur * ttm_div_yield_pct / 100.0), np.nan)
ttm_dps = pd.Series(ttm_dps_implied, index=filtered.index)
ttm_dps = ttm_dps.where(~pd.isna(ttm_div_yield_pct), ttm_dps_raw)
filtered["TTM DPS (RM)"] = pd.to_numeric(ttm_dps, errors="coerce").round(4)
filtered["Proj Div (RM)"] = (pd.to_numeric(shares, errors="coerce").fillna(0.0) * pd.to_numeric(ttm_dps, errors="coerce").fillna(0.0)).round(2)
filtered["Div Yield %"] = np.where(
    ~pd.isna(ttm_div_yield_pct),
    ttm_div_yield_pct,
    np.where(cur > 0, (pd.to_numeric(ttm_dps, errors='coerce') / cur) * 100.0, np.nan),
).round(2)
# Normalize friend names
filtered["FriendName"] = filtered["FriendName"].fillna("").astype(str).str.strip()
filtered.loc[filtered["FriendName"] == "", "FriendName"] = "Unassigned"

friends = sorted(filtered["FriendName"].unique().tolist())

# ─────────────────────────────────────────
# Friend Accounts summary (top)
# ─────────────────────────────────────────
st.markdown(section("👥 Friend Accounts", "Grouped overview (based on current filters)"), unsafe_allow_html=True)

summary_rows = []
for fn in friends:
    df_f = filtered[filtered["FriendName"] == fn]
    cost = float(df_f["Cost (RM)"].sum())
    val  = float(df_f["Value (RM)"].sum(skipna=True))
    pnl  = float(df_f["Unreal PnL (RM)"].sum(skipna=True))
    div  = float(df_f["Proj Div (RM)"].sum(skipna=True))
    yld  = (div / val * 100.0) if val > 0 else np.nan
    avg_simple = float(df_f.drop_duplicates("Name")["Div Yield %"].dropna().mean()) if (not df_f.empty and df_f.drop_duplicates("Name")["Div Yield %"].dropna().size>0) else np.nan
    avg_simple = float(df_f.drop_duplicates("Name")["Div Yield %"].dropna().mean()) if (not df_f.empty and df_f.drop_duplicates("Name")["Div Yield %"].dropna().size>0) else np.nan
    summary_rows.append({
        "Friend": fn,
        "Positions": int(len(df_f)),
        "Total Cost (RM)": round(cost, 2),
        "Cur Value (RM)": round(val, 2),
        "Unreal P&L (RM)": round(pnl, 2),
        "Proj Div (RM)": round(div, 2),
        "Div Yield %": (round(yld, 2) if np.isfinite(yld) else np.nan),

        "Avg Stock Div % (simple)": (round(float(df_f.drop_duplicates("Name")["Div Yield %"].dropna().mean()), 2) if not df_f.empty and df_f.drop_duplicates("Name")["Div Yield %"].dropna().size > 0 else np.nan),
    })

summary_df = pd.DataFrame(summary_rows).sort_values(["Friend"])
st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.caption("Avg Stock Div % (simple) = simple average of each stock's TTM dividend yield (not weighted). Useful to compare with FD rates.")

# ─────────────────────────────────────────
# Per-friend sections
# ─────────────────────────────────────────
st.markdown(section("🧾 Open Positions", "Edit Friend / Shares, then Save. Tick rows to close."), unsafe_allow_html=True)

CLOSE_REASONS = [
    "Target hit",
    "Stop hit",
    "Trailing stop",
    "Time stop",
    "Thesis changed",
    "Portfolio rebalance",
    "Other (specify)",
]

def _apply_edits_and_save(edited: pd.DataFrame, original: pd.DataFrame) -> int:
    """Detect FriendName/Shares changes and persist them."""
    if edited is None or edited.empty:
        return 0
    # compare on RowId
    o = original[["RowId","FriendName","Shares"]].copy()
    e = edited[["RowId","FriendName","Shares"]].copy()
    o["RowId"] = pd.to_numeric(o["RowId"], errors="coerce")
    e["RowId"] = pd.to_numeric(e["RowId"], errors="coerce")
    merged = e.merge(o, on="RowId", how="left", suffixes=("_new","_old"))
    changed = merged[
        (merged["FriendName_new"].astype(str) != merged["FriendName_old"].astype(str)) |
        (pd.to_numeric(merged["Shares_new"], errors="coerce").fillna(0).astype(int) != pd.to_numeric(merged["Shares_old"], errors="coerce").fillna(0).astype(int))
    ].copy()
    if changed.empty:
        return 0
    upd = pd.DataFrame({
        "RowId": changed["RowId"].astype(int),
        "FriendName": changed["FriendName_new"].astype(str),
        "Shares": pd.to_numeric(changed["Shares_new"], errors="coerce").fillna(0).astype(int),
    })
    return int(io_helpers.update_friend_open_trades_rows(upd))

for fn in friends:
    df_f = filtered[filtered["FriendName"] == fn].copy()
    if df_f.empty:
        continue

    # KPIs
    cost = float(df_f["Cost (RM)"].sum())
    val  = float(df_f["Value (RM)"].sum(skipna=True))
    pnl  = float(df_f["Unreal PnL (RM)"].sum(skipna=True))
    div  = float(df_f["Proj Div (RM)"].sum(skipna=True))
    yld  = (div / val * 100.0) if val > 0 else np.nan
    avg_simple = float(df_f.drop_duplicates("Name")["Div Yield %"].dropna().mean()) if (not df_f.empty and df_f.drop_duplicates("Name")["Div Yield %"].dropna().size>0) else np.nan

    label = f"{fn} — {len(df_f)} pos | P&L {pnl:,.0f} | Cost {cost:,.0f}"
    with st.expander(label, expanded=(fn != "Unassigned")):
        render_stat_cards([
            {"label": "Open Positions", "value": f"{len(df_f):,}", "badge": "Shown"},
            {"label": "Total Cost (RM)", "value": f"{cost:,.2f}", "badge": "Exposure"},
            {"label": "Cur Value (RM)", "value": f"{val:,.2f}", "badge": "Marked"},
            {"label": "Unreal P&L (RM)", "value": f"{pnl:,.2f}", "badge": "Live", "tone": ("good" if pnl>0 else "bad" if pnl<0 else "neutral")},
            {"label": "Proj Dividend (RM)", "value": f"{div:,.2f}", "badge": "TTM"},
            {"label": "Div Yield %", "value": (f"{yld:,.2f}%" if np.isfinite(yld) else "—"), "badge": "TTM"},
        {"label": "Avg Stock Div % (simple)", "value": (f"{avg_simple:.2f}%" if np.isfinite(avg_simple) else "—"), "badge": "TTM"},
            ])

        table = df_f.copy()
        table.insert(0, "SelectClose", False)
        table["ClosePrice"] = 0.0
        table["CloseReason"] = CLOSE_REASONS[0]
        table["Detail"] = ""

        editor_key = f"friend_open_editor_{_safe_key(fn)}"
        edited = st.data_editor(
            table,
            use_container_width=True,
            height=420,
            hide_index=True,
            key=editor_key,
            column_config={
                "SelectClose": st.column_config.CheckboxColumn("Close"),
                "RowId": st.column_config.NumberColumn("RowId", disabled=True),
                "Name": st.column_config.TextColumn("Name", disabled=True),
                "Strategy": st.column_config.TextColumn("Strategy", disabled=True),
                "FriendName": st.column_config.TextColumn("Friend", help="Edit to move trade to another friend"),
                "Shares": st.column_config.NumberColumn("Shares", format="%d", help="Edit quantity"),

                "Entry": st.column_config.NumberColumn("Entry", format="%.4f", disabled=True),
                "Stop": st.column_config.NumberColumn("Stop", format="%.4f", disabled=True),
                "Take": st.column_config.NumberColumn("Take", format="%.4f", disabled=True),
                "Current": st.column_config.NumberColumn("Current", format="%.4f", disabled=True),

                "ClosePrice": st.column_config.NumberColumn("Close Px", format="%.4f"),
                "CloseReason": st.column_config.SelectboxColumn("Reason", options=CLOSE_REASONS),
                "Detail": st.column_config.TextColumn("Detail"),
                "Cost (RM)": st.column_config.NumberColumn("Cost", format="%.2f", disabled=True),
                "Value (RM)": st.column_config.NumberColumn("Value", format="%.2f", disabled=True),
                "Unreal PnL (RM)": st.column_config.NumberColumn("P&L", format="%.2f", disabled=True),
                "Return %": st.column_config.NumberColumn("Ret %", format="%.2f", disabled=True),
                "Proj Div (RM)": st.column_config.NumberColumn("Proj Div", format="%.2f", disabled=True),
                "Div Yield %": st.column_config.NumberColumn("Div %", format="%.2f", disabled=True),
                "TTM DPS (RM)": st.column_config.NumberColumn("TTM DPS", format="%.4f", disabled=True),
            },
        )

        b1, b2 = st.columns([1.2, 1.8])
        with b1:
            if st.button("💾 Save Friend / Shares changes", use_container_width=True, key=f"save_{_safe_key(fn)}"):
                updated = _apply_edits_and_save(edited, df_f)
                if updated:
                    st.success(f"Saved {updated} row(s).")
                    (getattr(st, "rerun", None) or getattr(st, "experimental_rerun", lambda: None))()
                else:
                    st.info("No changes detected.")

        with b2:
            if st.button("🔒 Close selected", type="primary", use_container_width=True, key=f"close_{_safe_key(fn)}"):
                sel = edited[edited["SelectClose"] == True] if "SelectClose" in edited.columns else edited.iloc[0:0]
                if sel.empty:
                    st.info("Tick at least one row to close.")
                else:
                    # commit edits for selected rows first (so Shares/FriendName is correct)
                    _apply_edits_and_save(edited, df_f)

                    bad = []
                    closed = 0
                    for _, r in sel.iterrows():
                        rid = int(r["RowId"])
                        px = float(r.get("ClosePrice") or 0.0)
                        reason = str(r.get("CloseReason") or "").strip()
                        detail = str(r.get("Detail") or "").strip()
                        if px <= 0:
                            bad.append(f"RowId {rid}: ClosePrice <= 0")
                            continue
                        if reason == "Other (specify)" and not detail:
                            bad.append(f"RowId {rid}: Detail required")
                            continue
                        reason_txt = reason if reason != "Other (specify)" else f"{reason}: {detail}"
                        if io_helpers.close_friend_open_trade_row(rid, px, reason_txt):
                            closed += 1

                    if bad:
                        st.warning("Some rows not closed:\n- " + "\n- ".join(bad))
                    if closed:
                        st.success(f"Closed {closed} trade(s).")
                        (getattr(st, "rerun", None) or getattr(st, "experimental_rerun", lambda: None))()
