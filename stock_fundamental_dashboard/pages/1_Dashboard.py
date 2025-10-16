# pages/1_Dashboard.py
from __future__ import annotations

# ---- Auth ----
from auth_gate import require_auth
require_auth()

# ---- Stdlib / 3rd-party ----
import os
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import re, json

# ---- UI helpers (robust imports) ----
try:
    from utils.ui import setup_page, section, render_stat_cards, render_page_title
except Exception:  # pragma: no cover - fallback for old layouts
    from ui import setup_page, section, render_stat_cards  # type: ignore
    try:
        from ui import render_page_title  # type: ignore
    except Exception:
        def render_page_title(page_name: str) -> None:
            st.title(f"📊 Fundamentals Dashboard — {page_name}")

# ---- Shared helpers (robust imports) ----
try:  # pragma: no cover
    from utils import io_helpers, calculations, rules
except Exception:  # pragma: no cover
    import io_helpers  # type: ignore
    import calculations  # type: ignore
    import rules  # type: ignore
    
# ---- Extra utils for momentum OHLC fallback ----
try:
    from utils import name_link  # locate OHLC CSV path by name/ticker
except Exception:
    import name_link  # fallback if utils.name_link isn't namespaced

try:
    from utils import bus  # optional etag cache “version”
except Exception:
    class bus:  # tiny fallback; safe if utils.bus doesn’t exist
        @staticmethod
        def etag(_): return 0
    

# -----------------------------------------------------------------------------
# Page header
# -----------------------------------------------------------------------------
setup_page("Dashboard")
render_page_title("Dashboard")

# -----------------------------------------------------------------------------
# Small utils
# -----------------------------------------------------------------------------
_PRICE_KEYS = ("CurrentPrice", "Price", "SharePrice", "Annual Price per Share (RM)")

def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str):
            v = value.replace(",", "").strip()
            if v == "":
                return None
            value = float(v)
        else:
            value = float(value)
        if math.isfinite(value):
            return float(value)
    except Exception:
        return None
    return None

def _first_numeric(source: Mapping[str, Any] | pd.Series, keys: Iterable[str]) -> float | None:
    for key in keys:
        if key in source:
            try:
                val = _to_float(source.get(key))  # type: ignore[arg-type]
                if val is not None and pd.notna(val):
                    return val
            except Exception:
                continue
    return None

def _resolve_bucket(row: Mapping[str, Any]) -> str:
    raw = row.get("IndustryBucket") or row.get("Bucket") or row.get("Industry")
    bucket = str(raw).strip() if raw not in (None, "", float("nan")) else "General"
    if bucket not in calculations.BUCKET_CALCS:
        return "General"
    return bucket

def _evaluate_stock(
    *, name: str, stock_df: pd.DataFrame, bucket: str, current_price: float | None
) -> dict:
    evaluator = getattr(rules, "evaluate", None)
    if callable(evaluator):
        try:
            return evaluator(
                name=name,
                stock_df=stock_df,
                bucket=bucket,
                current_price=current_price,
            )
        except TypeError:
            pass
    return rules.funnel_rule(
        name=name,
        stock_df=stock_df,
        bucket=bucket,
        current_price=current_price,
    )

def _segmented(label: str, options: list[str], default: str) -> str:
    """Prefer segmented_control; fallback to radio; always return a valid option."""
    try:
        v = st.segmented_control(label, options=options, default=default)  # type: ignore[attr-defined]
    except Exception:
        idx = max(0, options.index(default)) if default in options else 0
        v = st.radio(label, options, index=idx, horizontal=True)
    # hard guard — some streamlit builds can return None before first render
    if not isinstance(v, str):
        return default
    v = v.strip()
    return v if v in options else default


# -----------------------------------------------------------------------------
# Load dataset
# -----------------------------------------------------------------------------
fund_df = io_helpers.load_data()
if fund_df is None or fund_df.empty:
    st.info("No fundamentals have been uploaded yet. Use **Add / Edit** to start populating the dataset.")
    st.stop()

fund_df = fund_df.copy()
fund_df["Name"] = fund_df["Name"].astype(str)

# Split annual vs quarter for later use (Activity fallback, etc.)
if "IsQuarter" in fund_df.columns:
    mask_quarter = fund_df["IsQuarter"].fillna(False).astype(bool)
else:
    mask_quarter = pd.Series(False, index=fund_df.index)

annual_df = fund_df[~mask_quarter].copy()
quarter_df = fund_df[mask_quarter].copy()

# -----------------------------------------------------------------------------
# Company selection (no Filters UI) — consider all companies in the dataset
# -----------------------------------------------------------------------------
filtered = annual_df.copy()
if filtered.empty:
    st.warning("No companies found in the dataset.")
    st.stop()

selected_names = sorted(filtered["Name"].dropna().unique())

# Latest row & bucket per name (used later by Trade Readiness)
name_to_bucket: dict[str, str] = {}
latest_rows: dict[str, pd.Series] = {}
for name, group in filtered.groupby("Name"):
    ordered = group.copy()
    ordered["_Year"] = pd.to_numeric(ordered.get("Year"), errors="coerce")
    ordered = ordered.sort_values("_Year")
    if ordered.empty:
        continue
    latest = ordered.iloc[-1]
    latest_rows[name] = latest
    name_to_bucket[name] = _resolve_bucket(latest)

def _resolve_ohlc_dir() -> str:
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "ohlc")),
        os.path.abspath(os.path.join(os.getcwd(), "data", "ohlc")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "ohlc")),
    ]
    for d in candidates:
        if os.path.isdir(d):
            return d
    return candidates[0]

@st.cache_data(show_spinner=False)
def _load_ohlc_for_name(stock_name: str, _etag: int, *, ticker: str | None = None) -> pd.DataFrame | None:
    ohlc_dir = _resolve_ohlc_dir()
    path = name_link.find_ohlc_path(stock_name, ohlc_dir=ohlc_dir, ticker=ticker)
    if not path or not os.path.exists(path):
        return None
    dfp = pd.read_csv(path)
    dfp["Date"] = pd.to_datetime(dfp["Date"], errors="coerce")
    if "Close" not in dfp.columns and "Adj Close" in dfp.columns:
        dfp["Close"] = pd.to_numeric(dfp["Adj Close"], errors="coerce")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in dfp.columns:
            dfp[c] = pd.to_numeric(dfp[c], errors="coerce")
    dfp = (
        dfp.dropna(subset=["Date", "Close"])
           .drop_duplicates(subset=["Date"])
           .sort_values("Date")
           .reset_index(drop=True)
    )
    return dfp

def _pct_change_from_ohlc(name: str, latest_row: Mapping[str, Any], *, sessions: int = 1) -> float | None:
    """
    Percent change over `sessions` trading days (1=1D, 5=1W, 21=1M approx).
    Uses OHLC CSV written by Momentum. Excludes today's partial bar if present.
    """
    # try to pass a ticker/code hint to match filename
    ticker_hint = None
    for col in ("Ticker", "Code", "Symbol"):
        if col in latest_row and isinstance(latest_row[col], str) and latest_row[col].strip():
            ticker_hint = latest_row[col].strip()
            break

    dfp = _load_ohlc_for_name(name, int(getattr(bus, "etag", lambda *_: 0)("ohlc")), ticker=ticker_hint)
    if dfp is None or dfp.empty:
        return None

    today = pd.Timestamp.today().normalize()
    prior = dfp[dfp["Date"] < today]  # avoid today's partial candle

    series = prior["Close"] if len(prior) >= sessions + 1 else (
        dfp["Close"] if len(dfp) >= sessions + 1 else None
    )
    if series is None:
        return None

    a = float(series.iloc[-(sessions + 1)])
    b = float(series.iloc[-1])
    if a and np.isfinite(a) and a != 0 and np.isfinite(b):
        return (b / a - 1.0) * 100.0
    return None

def _safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_")

def _load_yf_map() -> dict:
    """Read the same prefs file Momentum uses to store default Yahoo tickers."""
    prefs_path = os.path.join(os.path.expanduser("~"), ".app_state", "momentum_yf_prefs.json")
    try:
        with open(prefs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            m = data.get("map", {}) if isinstance(data, dict) else {}
            return m if isinstance(m, dict) else {}
    except Exception:
        return {}

def _lookup_yf_ticker(stock_name: str, latest_row: Mapping[str, Any]) -> str | None:
    m = _load_yf_map()
    # try direct + “safe” key (this is how 9_Momentum_Data saves keys)
    for k in (stock_name, _safe_name(stock_name).lower()):
        if k in m and str(m[k]).strip():
            return str(m[k]).strip()
    # fallback: a column that already looks like a Yahoo-style ticker (has a dot)
    for col in ("YahooTicker", "Ticker", "Code", "Symbol"):
        v = latest_row.get(col)
        if isinstance(v, str) and "." in v:
            return v.strip()
    return None


# -----------------------------------------------------------------------------
# Momentum heatmap (green = up, red = down; size = |% change|)
# Sits above the Activity section. Syncs with your data by:
#   1) Trying common "% change" columns written by your momentum page
#   2) Falling back to (CurrentPrice vs PrevClose) if available
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Momentum heatmap (green = up, red = down; size = |% change|)
# Choose 1D / 1W / 1M; pulls from dataset % columns first, then OHLC CSVs.
# -----------------------------------------------------------------------------
st.markdown(
    section("🟩 Momentum heatmap", "Green = up, red = down. Tile size = |% change|. Label shows Code/Name and % change."),
    unsafe_allow_html=True,
)

# UI: pick the change window (robust to None/unknown)
WIN_OPTIONS = ("1D", "1W", "1M")
win = _segmented("Change window", list(WIN_OPTIONS), "1D") or "1D"
if not isinstance(win, str):
    win = "1D"
win = win.strip().upper()
if win not in WIN_OPTIONS:
    win = "1D"
SESS = {"1D": 1, "1W": 5, "1M": 21}.get(win, 1)


def _first_str(source: Mapping[str, Any] | pd.Series, keys: Iterable[str]) -> str | None:
    for k in keys:
        if k in source:
            v = source.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None

def _find_pct_change(row: Mapping[str, Any], *, sessions: int) -> float | None:
    # 1) Check dataset-saved columns first (allow 1D/1W/1M flavors)
    CAND_BASE = ["ChangePct", "Change %", "Price Change %", "DailyChange%", "MomentumChangePct",
                 "ChgPct", "% Change", "Momentum % Change"]
    by_window_suffix = {
        1:  ["1D", "D", "Day"],
        5:  ["1W", "W", "Week"],
        21: ["1M", "M", "Month"],
    }[sessions]

    # exact window-specific names first, then generic
    candidates = [f"{c} {sfx}" for c in CAND_BASE for sfx in by_window_suffix] + CAND_BASE
    for c in candidates:
        if c in row and pd.notna(row[c]):
            v = _to_float(row[c])
            if v is not None:
                return float(v)

    # 2) Fallback to OHLC CSVs (sessions back)
    return _pct_change_from_ohlc(row.get("Name") if "Name" in row else "", row, sessions=sessions)

# Build heatmap tiles from the latest rows
tiles: list[dict[str, Any]] = []
for name, latest in latest_rows.items():
    pct = _find_pct_change(latest, sessions=SESS)
    if pct is None:
        continue
    code = _first_str(latest, ("Code", "Ticker", "Symbol"))
    label = code or name  # show code if present; else name

    yf_t = _lookup_yf_ticker(name, latest)
    yf_url = f"https://finance.yahoo.com/quote/{yf_t}" if yf_t else None

    tiles.append({
        "Name": name,
        "Label": label,
        "ChangePct": float(pct),
        "AbsChange": abs(float(pct)),
        "YahooURL": yf_url,
    })

if not tiles:
    st.info("No momentum % change found yet. Save Yahoo OHLC on the Momentum page or add a '% Change' column.")
else:
    hdf = pd.DataFrame(tiles)
    
    # ---- FIX: avoid zero-weight treemap tiles ----
    EPS = 1e-6  # tiny positive area
    # make sure numbers are valid and strictly positive for Plotly's weighting
    hdf["AbsChange"] = pd.to_numeric(hdf["AbsChange"], errors="coerce").fillna(0.0)
    hdf.loc[hdf["AbsChange"] <= 0, "AbsChange"] = EPS
    # (optional) ensure color source is numeric too
    hdf["ChangePct"] = pd.to_numeric(hdf["ChangePct"], errors="coerce").fillna(0.0)

    fig = px.treemap(
        hdf,
        path=[px.Constant("All"), "Label"],
        values="AbsChange",
        color="ChangePct",
        custom_data=["ChangePct"],
        color_continuous_scale=["#b71c1c", "#ffebee", "#e8f5e9", "#1b5e20"],  # red -> green
        color_continuous_midpoint=0,
    )
    fig.update_traces(
        texttemplate="%{label}<br>%{customdata[0]:+.2f}%",   # 2 decimals, signed
        hovertemplate="<b>%{label}</b><br>%{customdata[0]:+.2f}% change<extra></extra>",
        root_color="rgba(0,0,0,0)",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title="%Δ"),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Yahoo link list (top movers first)
    links = hdf.dropna(subset=["YahooURL"]).copy()
    if not links.empty:
        links = links.sort_values("AbsChange", ascending=False).head(12)
        st.caption("Yahoo links (top movers)")
        cols = st.columns(4)
        for i, r in enumerate(links.itertuples()):
            with cols[i % 4]:
                st.markdown(f"[{r.Label}]({r.YahooURL})  ({r.ChangePct:+.2f}%)")

# -----------------------------------------------------------------------------
# Activity (edits & deletes) — synced with Add/Edit audit log
# -----------------------------------------------------------------------------
st.markdown(
    section("📝 Activity (edits & deletes)", "Last changes from Add/Edit, split by Annual & Quarterly."),
    unsafe_allow_html=True,
)

# locate repo root (two levels above /pages)
_THIS = os.path.dirname(__file__)
_GRANDP = os.path.abspath(os.path.join(_THIS, "..", ".."))
_AUDIT_FILE = os.path.join(_GRANDP, "data", "audit_log.jsonl")

# Retain only the last N days of audit events (hard limit)
RETAIN_DAYS = 30

def _prune_audit_file(retain_days: int = RETAIN_DAYS) -> tuple[int, int]:
    """
    Physically rewrite data/audit_log.jsonl keeping only events within retain_days.
    Returns (kept, pruned). Safe no-op if file absent or unreadable lines.
    """
    try:
        import json as _json
        if not os.path.exists(_AUDIT_FILE):
            return (0, 0)

        cutoff_utc = datetime.now(timezone.utc) - timedelta(days=retain_days)
        keep_lines: list[str] = []
        pruned = 0
        kept = 0

        with open(_AUDIT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    ev = _json.loads(raw)
                    ts = pd.to_datetime(ev.get("ts"), utc=True, errors="coerce")
                    if ts is not None and not pd.isna(ts) and ts.tzinfo is not None:
                        if ts.to_pydatetime() >= cutoff_utc:
                            keep_lines.append(raw + "\n")
                            kept += 1
                        else:
                            pruned += 1
                    else:
                        # no/invalid ts → drop it (counts as pruned)
                        pruned += 1
                except Exception:
                    pruned += 1

        # Only rewrite if something changed
        if pruned > 0:
            tmp_path = _AUDIT_FILE + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as w:
                w.writelines(keep_lines)
            os.replace(tmp_path, _AUDIT_FILE)

        return (kept, pruned)
    except Exception:
        # best-effort; never break dashboard
        return (0, 0)


def _load_audit_events() -> pd.DataFrame:
    if not os.path.exists(_AUDIT_FILE):
        return pd.DataFrame(columns=["ts","action","scope","name","year","quarter","changes","source"])
    rows = []
    with open(_AUDIT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = pd.json.loads(line)  # type: ignore[attr-defined]
            except Exception:
                try:
                    import json as _json
                    ev = _json.loads(line)
                except Exception:
                    continue
            rows.append(ev)
    if not rows:
        return pd.DataFrame(columns=["ts","action","scope","name","year","quarter","changes","source"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df.get("ts"), errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts", ascending=False)
    df["Action"] = df.get("action", "").astype(str).str.title()
    df["Scope"] = df.get("scope", "").astype(str).str.title()
    df["Name"] = df.get("name", "").astype(str)
    df["Year"] = pd.to_numeric(df.get("year"), errors="coerce").astype("Int64")
    df["Quarter"] = df.get("quarter", "").astype(str)
    # pretty list of changed fields
    def _field_list(x):
        if isinstance(x, dict):
            try:
                return ", ".join(sorted(list(x.keys())))
            except Exception:
                return ""
        return ""
    df["Changed Fields"] = df.get("changes").apply(_field_list)
    df["When"] = df["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    # expose source in a consistent column
    df["Source"] = df.get("source", "").astype(str)
    return df

def _fallback_recent_from_dataset(base: pd.DataFrame, *, is_quarter: bool) -> pd.DataFrame:
    """If no audit log yet, show updates inferred from LastModified (no deletes)."""
    if base is None or base.empty or "LastModified" not in base.columns:
        return pd.DataFrame(columns=["When","Action","Scope","Name","Year","Quarter","Changed Fields","Source"])
    dfx = base.copy()
    dfx["ts"] = pd.to_datetime(dfx["LastModified"], errors="coerce")
    dfx = dfx.dropna(subset=["ts"]).sort_values("ts", ascending=False)
    dfx["Action"] = "Update"
    dfx["Scope"] = "Quarterly" if is_quarter else "Annual"
    dfx["Name"] = dfx["Name"].astype(str)
    dfx["Year"] = pd.to_numeric(dfx.get("Year"), errors="coerce").astype("Int64")
    dfx["Quarter"] = dfx.get("Quarter", "").astype(str)
    dfx["Changed Fields"] = ""
    dfx["Source"] = "fallback_last_modified"
    dfx["When"] = dfx["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return dfx[["When","Action","Scope","Name","Year","Quarter","Changed Fields","Source"]]

# Time window selector (max 30 days allowed)
window = _segmented("Show activity from", ["7 days", "30 days"], "7 days")

# Enforce hard retention on disk BEFORE loading (physically removes old lines)
try:
    _ = _prune_audit_file(RETAIN_DAYS)
except Exception:
    pass

now = datetime.now(timezone.utc).astimezone()
if window == "7 days":
    cutoff = now - timedelta(days=7)
else:  # "30 days"
    cutoff = now - timedelta(days=30)

# For string comparisons in fallback (When column is string)
cutoff_naive = cutoff.replace(tzinfo=None)

# Load events or fallback
ev_df = _load_audit_events()
if ev_df.empty and "LastModified" in fund_df.columns:
    annual_recent = _fallback_recent_from_dataset(annual_df, is_quarter=False)
    quarter_recent = _fallback_recent_from_dataset(quarter_df, is_quarter=True)

    # 'When' is a string -> tz-naive datetime64; compare with cutoff_naive
    if cutoff_naive is not None:
        annual_recent = annual_recent[pd.to_datetime(annual_recent["When"]) >= cutoff_naive]
        quarter_recent = quarter_recent[pd.to_datetime(quarter_recent["When"]) >= cutoff_naive]
else:
    dfw = ev_df.copy()
    # 'ts' is tz-aware; compare with tz-aware cutoff
    if cutoff is not None:
        dfw = dfw[dfw["ts"] >= cutoff]
    annual_recent = dfw[dfw["Scope"].str.lower() == "annual"][["When","Action","Name","Year","Changed Fields","Source"]]
    quarter_recent = dfw[dfw["Scope"].str.lower() == "quarterly"][["When","Action","Name","Year","Quarter","Changed Fields","Source"]]

tabs = st.tabs(["Annual activity", "Quarterly activity"])

with tabs[0]:
    if annual_recent.empty:
        st.info("No annual activity in this window.")
    else:
        st.dataframe(
            annual_recent,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Year": st.column_config.NumberColumn("Year", format="%d"),
            },
        )

with tabs[1]:
    if quarter_recent.empty:
        st.info("No quarterly activity in this window.")
    else:
        st.dataframe(
            quarter_recent,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Year": st.column_config.NumberColumn("Year", format="%d"),
            },
        )

# -----------------------------------------------------------------------------
# Trade readiness evaluation (unchanged)
# -----------------------------------------------------------------------------
st.markdown(
    section("✅ Trade readiness", "Scores from the rule engine combined with queue risk/reward inputs."),
    unsafe_allow_html=True,
)

queue_df = io_helpers.load_trade_queue()
queue_df = queue_df.copy() if isinstance(queue_df, pd.DataFrame) else pd.DataFrame()
queue_df["Name"] = queue_df.get("Name", pd.Series(dtype=str)).astype(str)

queue_lookup = {n.upper(): row for n, row in queue_df.set_index("Name").iterrows()} if not queue_df.empty else {}

trade_records: list[dict[str, Any]] = []
for name in selected_names:
    stock_rows = fund_df[fund_df["Name"] == name].copy()
    if stock_rows.empty:
        continue
    bucket = name_to_bucket.get(name, "General")
    current_price = _first_numeric(latest_rows.get(name, {}), _PRICE_KEYS)
    evaluation = _evaluate_stock(name=name, stock_df=stock_rows, bucket=bucket, current_price=current_price)

    blocks = evaluation.get("blocks", {}) if isinstance(evaluation, dict) else {}
    valuation_block = blocks.get("valuation_entry", {}) if isinstance(blocks, dict) else {}
    score = float(evaluation.get("composite", 0.0)) if isinstance(evaluation, dict) else 0.0
    val_score = float(valuation_block.get("score", 0.0)) if isinstance(valuation_block, dict) else 0.0
    val_label = valuation_block.get("label", "—") if isinstance(valuation_block, dict) else "—"

    try:
        min_score, min_val = rules.min_thresholds_for(bucket)
    except Exception:
        min_score, min_val = 65, 50
    is_viable = (score >= float(min_score)) and (val_score >= float(min_val))

    queue_row = queue_lookup.get(name.upper()) if queue_lookup else None
    entry = _to_float(queue_row.get("Entry")) if queue_row is not None else None
    stop = _to_float(queue_row.get("Stop")) if queue_row is not None else None
    take = _to_float(queue_row.get("Take")) if queue_row is not None else None

    risk = None
    reward = None
    rr = None
    if entry is not None and stop is not None and take is not None:
        if entry > stop:
            risk = entry - stop
        if take != entry:
            reward = take - entry
        if risk not in (None, 0) and reward is not None:
            rr = reward / risk
    if rr is None and queue_row is not None:
        rr = _to_float(queue_row.get("RR"))

    trade_records.append(
        {
            "Name": name,
            "Bucket": bucket,
            "Composite Score": score,
            "Cash Flow Score": blocks.get("cashflow_first", {}).get("score") if isinstance(blocks.get("cashflow_first"), dict) else None,
            "TTM vs LFY Score": blocks.get("ttm_vs_lfy", {}).get("score") if isinstance(blocks.get("ttm_vs_lfy"), dict) else None,
            "Valuation Score": val_score,
            "Valuation Label": val_label,
            "Dividend Score": blocks.get("dividend", {}).get("score") if isinstance(blocks.get("dividend"), dict) else None,
            "Momentum Score": blocks.get("momentum", {}).get("score") if isinstance(blocks.get("momentum"), dict) else None,
            "Strategy": queue_row.get("Strategy") if queue_row is not None else None,
            "Entry": entry,
            "Stop": stop,
            "Take": take,
            "Risk": risk,
            "Reward": reward,
            "R/R": rr,
            "Viable": is_viable,
        }
    )

trade_df = pd.DataFrame(trade_records)
if trade_df.empty:
    st.info("No trade evaluations available for the selected companies.")
else:
    ready_count = int(trade_df["Viable"].sum())
    avg_score = float(trade_df["Composite Score"].mean()) if not trade_df["Composite Score"].isna().all() else 0.0
    avg_rr = float(trade_df["R/R"].dropna().mean()) if not trade_df["R/R"].dropna().empty else None

    cards = [
        {
            "label": "Ready to Trade",
            "value": f"{ready_count} / {len(trade_df)}",
            "note": "Meet score and valuation thresholds",
            "tone": "good" if ready_count else "",
        },
        {
            "label": "Average Composite",
            "value": f"{avg_score:.1f}",
            "note": "Rule-engine composite (0-100)",
        },
        {
            "label": "Avg R/R",
            "value": (f"{avg_rr:.2f}×" if avg_rr is not None else "—"),
            "note": "Calculated from queue entries",
        },
    ]
    render_stat_cards(cards, columns=3)

    st.dataframe(
        trade_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Composite Score": st.column_config.NumberColumn("Composite Score", format="%.1f"),
            "Cash Flow Score": st.column_config.NumberColumn("Cash Flow Score", format="%.1f"),
            "TTM vs LFY Score": st.column_config.NumberColumn("TTM vs LFY Score", format="%.1f"),
            "Valuation Score": st.column_config.NumberColumn("Valuation Score", format="%.1f"),
            "Dividend Score": st.column_config.NumberColumn("Dividend Score", format="%.1f"),
            "Momentum Score": st.column_config.NumberColumn("Momentum Score", format="%.1f"),
            "Entry": st.column_config.NumberColumn("Entry", format="%.2f"),
            "Stop": st.column_config.NumberColumn("Stop", format="%.2f"),
            "Take": st.column_config.NumberColumn("Take", format="%.2f"),
            "Risk": st.column_config.NumberColumn("Risk", format="%.2f"),
            "Reward": st.column_config.NumberColumn("Reward", format="%.2f"),
            "R/R": st.column_config.NumberColumn("R/R", format="%.2f"),
        },
    )

# -----------------------------------------------------------------------------
# Ongoing trades snapshot (unchanged)
# -----------------------------------------------------------------------------
st.markdown(
    section("📈 Ongoing trades", "Live positions pulled from the trades ledger."),
    unsafe_allow_html=True,
)

open_trades = io_helpers.load_open_trades()
if open_trades.empty:
    st.info("No ongoing trades are being tracked right now.")
else:
    display_cols = [
        c
        for c in [
            "Name",
            "Strategy",
            "Entry",
            "Stop",
            "Take",
            "Shares",
            "OpenDate",
            "Pnl",
            "ReturnPct",
        ]
        if c in open_trades.columns
    ]
    st.dataframe(open_trades[display_cols], use_container_width=True, hide_index=True)
