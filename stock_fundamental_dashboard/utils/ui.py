# utils/ui.py
"""
Shared UI helpers + CSS for all Streamlit pages.

Usage (per page):
    from utils.ui import setup_view_stock_page, render_kpi_text_grid, section
    setup_view_stock_page("View Stock")

Available:
    setup_add_or_edit_page(title="Add / Edit Stock")
    setup_view_stock_page(title="View Stock")
    setup_decision_page(title="Systematic Decision")
    render_kpi_text_grid(pairs, columns=3)
    section(title, desc="", tone="")  # tone: "", "info", "warning", "danger", "success"
"""

from __future__ import annotations
import os, sys, inspect
import streamlit as st

__all__ = (
    "setup_page",
    "setup_add_or_edit_page",
    "setup_view_stock_page",
    "setup_decision_page",
    "render_page_title",
    "render_kpi_text_grid",
    "render_stat_cards",
    "section",
    "register_page_css",
    "get_page_id",
    "render_compare_header",
    "render_soldier_worm_diagnostics",
    "register_ongoing_trades_css",
)

# --------------------------- Core helpers ---------------------------

def render_stat_cards(items, columns: int = 3, caption: str | None = None):
    """Render KPI-style stat cards (HTML, no Markdown code blocks)."""
    from html import escape
    tiles = []
    for it in items:
        label = escape(str(it.get("label", "")))
        value = escape(str(it.get("value", "")))
        note  = it.get("note", "")
        note  = f"<div class='stat-note'>{escape(str(note))}</div>" if note else ""
        tone  = it.get("tone", "")
        badge = it.get("badge", "")
        tone_cls = f" tone-{tone}" if tone else ""
        badge_html = f"<span class='badge'>{escape(str(badge))}</span>" if badge else ""
        tiles.append(
            "<div class='stat-card{tone}'>"
            "<div class='stat-head'>{badge}<span class='stat-title'>{label}</span></div>"
            "<div class='stat-value'>{value}</div>"
            "{note}"
            "</div>".format(tone=tone_cls, badge=badge_html, label=label, value=value, note=note)
        )

    cap = f"<div class='stat-caption'>{escape(str(caption))}</div>" if caption else ""
    html = cap + f"<div class='stat-grid cols-{columns}'>" + "".join(tiles) + "</div>"
    st.markdown(html, unsafe_allow_html=True)

def setup_page(title: str, page_id: str | None = None) -> None:
    """Set page config + inject base CSS + page-specific CSS (if any)."""
    st.set_page_config(page_title=title, layout="wide")
    pid = page_id or get_page_id()
    css = BASE_CSS + HOTFIX_CSS + PAGE_CSS.get(pid, "")
    st.markdown(css, unsafe_allow_html=True)

def render_page_title(page_name: str) -> None:
    """Render the shared dashboard title for every page."""
    st.title(f"📊 Fundamentals Dashboard — {page_name}")

def setup_add_or_edit_page(title: str = "Add / Edit Stock") -> None:
    setup_page(title, "2_Add_or_Edit")

def setup_view_stock_page(title: str = "View Stock") -> None:
    setup_page(title, "3_View_Stock")

def setup_decision_page(title: str = "Systematic Decision") -> None:
    setup_page(title, "4_Systematic_Decision")

def register_page_css(page_id: str, css: str) -> None:
    """Optional: dynamically add/override a page's CSS at runtime."""
    PAGE_CSS[page_id] = f"\n<style>\n{css.strip()}\n</style>\n"

def get_page_id() -> str:
    """
    Resolve page id from /pages/<file>.py (e.g. '3_View_Stock').
    Falls back to argv if outside /pages.
    """
    f = inspect.currentframe()
    while f:
        m = inspect.getmodule(f)
        if m and getattr(m, "__file__", ""):
            path = m.__file__.replace("\\", "/")
            if "/pages/" in path:
                return os.path.splitext(os.path.basename(path))[0]
        f = f.f_back
    return os.path.splitext(os.path.basename(sys.argv[0]))[0]

# --------------------------- Tiny UI primitives ---------------------------

def render_kpi_text_grid(pairs, columns: int = 3):
    """
    Render compact KPI tiles.
    pairs: list[tuple[str, str]]  -> [(label, value_text)]
    """
    tiles = []
    for label, value_text in pairs:
        tiles.append(
            f"<div class='kpi'><div class='k'>{label}</div><div class='v'>{value_text}</div></div>"
        )
    st.markdown(
        f"<div class='kpi-grid cols-{columns}'>" + "".join(tiles) + "</div>",
        unsafe_allow_html=True,
    )

def render_compare_header(
    left_title: str,
    right_title: str,
    left_pairs: list[tuple[str, str]],
    right_pairs: list[tuple[str, str]],
    *,
    ttm_left: str | None = None,
    ttm_right: str | None = None,
    columns: int = 2,
):
    """
    Pretty two-column compare header with optional TTM badges.

    Usage:
        render_compare_header(
            a, b,
            facts_a, facts_b,
            ttm_left="TTM 2025", ttm_right="TTM 2025",
            columns=2
        )
    """
    from html import escape

    def _title_block(title: str, ttm: str | None):
        chip = f"<span class='ttm-chip'>{escape(ttm)}</span>" if ttm else ""
        return (
            "<div class='cmp-title'>"
            f"<h4 class='cmp-h'>{escape(title)}</h4>{chip}"
            "</div>"
        )

    st.markdown("<div class='cmp-head'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(_title_block(left_title, ttm_left), unsafe_allow_html=True)
        render_kpi_text_grid(left_pairs, columns=columns)
    with c2:
        st.markdown(_title_block(right_title, ttm_right), unsafe_allow_html=True)
        render_kpi_text_grid(right_pairs, columns=columns)
    st.markdown("</div>", unsafe_allow_html=True)



def render_soldier_worm_diagnostics(warn_df, trace_df, *, key_prefix: str = "sw"):
    """Premium diagnostics panel for Soldier Worm (Missing Inputs + Calc Trace)."""
    import pandas as pd
    from collections import Counter

    warn_df = warn_df if isinstance(warn_df, pd.DataFrame) else pd.DataFrame()
    trace_df = trace_df if isinstance(trace_df, pd.DataFrame) else pd.DataFrame()

    miss_cnt = int(len(warn_df)) if not warn_df.empty else 0
    trace_cnt = int(len(trace_df)) if not trace_df.empty else 0

    # ---- overview stats ----
    total_checks = 0
    if not trace_df.empty and {"Metric","Period"}.issubset(trace_df.columns):
        try:
            total_checks = int(len(trace_df[["Metric","Period"]].drop_duplicates()))
        except Exception:
            total_checks = 0

    health = None
    if total_checks > 0:
        health = max(0, min(100, round(100 * (1 - (miss_cnt / total_checks)))))

    cats = int(warn_df["Category"].nunique()) if (not warn_df.empty and "Category" in warn_df.columns) else 0

    top_missing = "—"
    if not warn_df.empty and "Missing Inputs" in warn_df.columns:
        toks = []
        for s in warn_df["Missing Inputs"].astype(str).tolist():
            for t in s.split(","):
                t = t.strip()
                if not t:
                    continue
                toks.append(t.split("—")[0].strip())
        if toks:
            top_missing = Counter(toks).most_common(1)[0][0]

    cards = [
        {"label": "Data health", "value": f"{health}%" if health is not None else "—", "note": "checks vs missing"},
        {"label": "Missing checks", "value": f"{miss_cnt}", "note": "needs inputs"},
        {"label": "Categories affected", "value": f"{cats}", "note": "summary sections"},
        {"label": "Top missing field", "value": top_missing, "note": "most frequent"},
    ]
    render_stat_cards(cards, columns=4)

    def _h(n, row_h=28, base=140, max_h=620):
        try:
            n = int(n)
        except Exception:
            n = 0
        return min(max_h, base + row_h * max(1, min(n, 14)))

    tab1, tab2, tab3 = st.tabs([
        "Overview",
        f"Missing ({miss_cnt})",
        f"Trace ({trace_cnt})",
    ])

    # -------------------- Overview --------------------
    with tab1:
        if warn_df.empty:
            st.success("✅ No missing inputs detected for the current setup.")
        else:
            st.markdown("#### Focus list")
            focus = warn_df.copy()
            if "Missing Count" in focus.columns:
                try:
                    focus = focus.sort_values(["Missing Count"], ascending=False)
                except Exception:
                    pass
            focus = focus.head(12)
            show_cols = [c for c in ["Category","Metric","Period","Missing Count","Missing Inputs"] if c in focus.columns]
            st.dataframe(focus[show_cols], use_container_width=True, height=_h(len(focus), row_h=26, base=120, max_h=520))

            try:
                csv_bytes = warn_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download missing list (CSV)",
                    data=csv_bytes,
                    file_name="soldier_worm_missing_inputs.csv",
                    mime="text/csv",
                    key=f"{key_prefix}_dl_missing",
                )
            except Exception:
                pass

    # -------------------- Missing Inputs --------------------
    with tab2:
        col1, col2, col3, col4 = st.columns([1.1, 1.1, 1.2, 1.6])
        with col1:
            scope = st.selectbox("Scope", ["All", "Only TTM", "Only FY"], index=0, key=f"{key_prefix}_scope")
        with col2:
            mode = st.selectbox("View", ["Grouped (recommended)", "Table"], index=0, key=f"{key_prefix}_mode")
        with col3:
            cats_sel = []
            if not warn_df.empty and "Category" in warn_df.columns:
                opts = ["All"] + sorted([c for c in warn_df["Category"].dropna().unique().tolist()])
                pick = st.selectbox("Category", opts, index=0, key=f"{key_prefix}_cat")
                if pick != "All":
                    cats_sel = [pick]
        with col4:
            q = st.text_input("Search (metric / missing fields)…", key=f"{key_prefix}_q")

        view_w = warn_df.copy()
        if not view_w.empty:
            if scope == "Only TTM":
                view_w = view_w[view_w["Period"].astype(str).str.upper() == "TTM"]
            elif scope == "Only FY":
                view_w = view_w[view_w["Period"].astype(str).str.upper() != "TTM"]
            if cats_sel:
                view_w = view_w[view_w["Category"].isin(cats_sel)]
            if q.strip():
                s = q.strip().lower()
                view_w = view_w[
                    view_w["Metric"].astype(str).str.lower().str.contains(s)
                    | view_w["Missing Inputs"].astype(str).str.lower().str.contains(s)
                ]

        if view_w.empty:
            st.caption("No missing inputs in this view.")
        else:
            if mode.startswith("Grouped"):
                for cat in view_w["Category"].dropna().unique().tolist():
                    cat_df = view_w[view_w["Category"] == cat]
                    with st.expander(f"{cat}  ·  {len(cat_df)}", expanded=False):
                        for met in cat_df["Metric"].dropna().unique().tolist():
                            mdf = cat_df[cat_df["Metric"] == met]
                            st.markdown(f"**{met}**")
                            for _, rr in mdf.iterrows():
                                per = rr.get("Period", "")
                                per_lab = f"FY {int(per)}" if isinstance(per, (int, float)) and str(per).isdigit() else str(per)
                                miss = rr.get("Missing Inputs", "")
                                miss_cnt2 = rr.get("Missing Count", None)
                                chip = ""
                                if miss_cnt2 is not None:
                                    try:
                                        chip = f"<span class='badge'>{int(miss_cnt2)} missing</span>"
                                    except Exception:
                                        chip = ""
                                st.markdown(
                                    f"<div style='padding:.25rem 0 .65rem;border-bottom:1px solid rgba(148,163,184,.25)'>"
                                    f"<span class='badge'>{per_lab}</span> {chip}<br/>"
                                    f"<span style='color:var(--muted); font-size:.93rem'>{miss}</span>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                            st.markdown("")
            else:
                show_cols = [c for c in ["Category","Metric","Period","Missing Count","Missing Inputs","Fix Tip"] if c in view_w.columns]
                try:
                    sty = view_w[show_cols].style.hide(axis="index")
                except Exception:
                    sty = view_w[show_cols].style
                st.dataframe(sty, use_container_width=True, height=_h(len(view_w), row_h=26, base=120, max_h=620))

    # -------------------- Calc Trace --------------------
    with tab3:
        col1, col2, col3, col4 = st.columns([1.1, 1.2, 1.2, 1.6])
        with col1:
            scope = st.selectbox("Scope", ["All", "Only TTM", "Only FY"], index=0, key=f"{key_prefix}_t_scope")
        with col2:
            only_sus = st.toggle("Suspicious only", value=False, key=f"{key_prefix}_t_sus")
        with col3:
            met = None
            if not trace_df.empty and "Metric" in trace_df.columns:
                opts = ["All"] + sorted(trace_df["Metric"].dropna().astype(str).unique().tolist())
                met = st.selectbox("Metric", opts, index=0, key=f"{key_prefix}_t_met")
        with col4:
            q = st.text_input("Search (metric / inputs / path / flags)…", key=f"{key_prefix}_t_q")

        view_t = trace_df.copy()
        if not view_t.empty:
            if scope == "Only TTM":
                view_t = view_t[view_t["Period"].astype(str).str.upper() == "TTM"]
            elif scope == "Only FY":
                p = view_t["Period"].astype(str).str.upper()
                view_t = view_t[(p == "FY") | (p.str.startswith("FY "))]
            if met and met != "All":
                view_t = view_t[view_t["Metric"].astype(str) == str(met)]
            if only_sus and {"Flags","Note"}.issubset(view_t.columns):
                sus = (view_t["Flags"].astype(str).str.strip() != "") | (view_t["Note"].astype(str).str.strip() != "")
                view_t = view_t[sus]
            if q.strip():
                s = q.strip().lower()
                cols = [c for c in ["Metric","Inputs","Path Used","Formula","Flags","Note"] if c in view_t.columns]
                if cols:
                    mask = False
                    for c in cols:
                        mask = mask | view_t[c].astype(str).str.lower().str.contains(s)
                    view_t = view_t[mask]

        if view_t.empty:
            st.caption("No calc trace yet (not enough data).")
        else:
            if "Value" in view_t.columns and "Metric" in view_t.columns:
                pct_mask = view_t["Metric"].astype(str).str.contains(r"\(%\)|margin|yield| ratio", case=False, regex=True)
                try:
                    sty = view_t.style.hide(axis="index")
                except Exception:
                    sty = view_t.style

                try:
                    sty = sty.format(lambda v: "—" if pd.isna(v) else f"{float(v):,.2f}%", subset=pd.IndexSlice[pct_mask, ["Value"]])
                    sty = sty.format(lambda v: "—" if pd.isna(v) else f"{float(v):,.2f}",  subset=pd.IndexSlice[~pct_mask, ["Value"]])
                except Exception:
                    pass

                def _row_style(r):
                    base = [""] * len(r)
                    has_note = str(r.get("Note","")).strip() != ""
                    flags    = str(r.get("Flags","")).lower()
                    if "fallback" in flags:
                        return ["background-color:#fff7ed;"] * len(r)
                    if "prev fy" in flags:
                        return ["background-color:#f1f5f9;"] * len(r)
                    if "outlier" in flags:
                        return ["background-color:#fee2e2;"] * len(r)
                    if has_note:
                        return ["background-color:#fef9c3;"] * len(r)
                    return base

                try:
                    sty = sty.apply(_row_style, axis=1)
                except Exception:
                    pass

                st.dataframe(sty, use_container_width=True, height=_h(len(view_t), row_h=26, base=120, max_h=650))
            else:
                st.dataframe(view_t, use_container_width=True, height=_h(len(view_t), row_h=26, base=120, max_h=650))


def section(title: str, desc: str = "", tone: str = "") -> str:
    """
    Re-usable pill header. tone in {"", "info", "warning", "danger", "success"}.
    Usage: st.markdown(section("Title", "Optional description", "info"), unsafe_allow_html=True)
    """
    tone_class = f" {tone}" if tone else ""
    return (
        f'<div class="sec{tone_class}"><div class="t">{title}</div>'
        f'<div class="d">{desc}</div></div>'
    )


def register_ongoing_trades_css():
    """Injects the Ongoing Trades CSS theme on demand."""
    import streamlit as st
    st.markdown(ONGOING_TRADES_CSS, unsafe_allow_html=True)

# --------------------------- Base CSS (all pages) ---------------------------

BASE_CSS = """
<style>
:root{
  --bg:#f6f7fb; --surface:#ffffff; --text:#0f172a; --muted:#475569; --border:#e5e7eb;
  --shadow:0 8px 24px rgba(15, 23, 42, .06);
  --primary:#4f46e5; --info:#0ea5e9; --success:#10b981; --warning:#f59e0b; --danger:#ef4444;
}

/* App + typography */
[data-testid="stAppViewContainer"]{ background:var(--bg) !important; }
html, body, [data-testid="stAppViewContainer"] *{ letter-spacing:.1px; color:var(--text); }
h1, h2, h3, h4, h5, h6{ font-weight:800 !important; letter-spacing:.2px; }

/* Sidebar theme */
[data-testid="stSidebar"]{ background:linear-gradient(180deg, #0b1220 0%, #1f2937 100%) !important; }
[data-testid="stSidebar"] *{ color:#e5e7eb !important; }

/* Section header pill */
.sec{
  background:var(--surface); border:1px solid var(--border); border-radius:14px;
  box-shadow:var(--shadow); padding:.65rem .9rem; margin:1rem 0 .6rem;
  display:flex; align-items:center; gap:.6rem;
}
.sec .t{ font-size:1.05rem; font-weight:800; margin:0; }
.sec .d{ color:var(--muted); font-size:.95rem; margin-left:.25rem; }
.sec::before{ content:""; width:8px; height:26px; border-radius:6px; background:var(--primary); display:inline-block; }
.sec.info::before{background:var(--info);} .sec.success::before{background:var(--success);}
.sec.warning::before{background:var(--warning);} .sec.danger::before{background:var(--danger);}

/* KPI tiles */
.kpi-grid{ display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:14px; margin:.5rem 0 1rem; }
.kpi-grid.cols-1{ grid-template-columns:1fr; }
.kpi-grid.cols-2{ grid-template-columns:repeat(2, minmax(0,1fr)); }
.kpi-grid.cols-3{ grid-template-columns:repeat(3, minmax(0,1fr)); }
.kpi-grid.cols-4{ grid-template-columns:repeat(4, minmax(0,1fr)); }
@media (max-width:1100px){ .kpi-grid{ grid-template-columns:repeat(2, minmax(0,1fr)); } }
@media (max-width:700px){ .kpi-grid{ grid-template-columns:1fr; } }

.kpi{
  background:var(--surface); border:1px solid var(--border); border-radius:14px;
  box-shadow:var(--shadow); padding:12px 14px; min-height:96px;
  display:flex; flex-direction:column; justify-content:center;
}
.kpi .k{ color:var(--muted); font-weight:700; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
.kpi .v{ font-weight:800; font-size:26px; line-height:1; margin-top:6px; }

/* --- KPI Stat Cards --- */
.stat-grid{display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; margin:.35rem 0 1rem;}
.stat-grid.cols-1{grid-template-columns:1fr;}
.stat-grid.cols-2{grid-template-columns:repeat(2,minmax(0,1fr));}
.stat-grid.cols-3{grid-template-columns:repeat(3,minmax(0,1fr));}
.stat-grid.cols-4{grid-template-columns:repeat(4,minmax(0,1fr));}
@media (max-width:1100px){ .stat-grid{ grid-template-columns:repeat(2,minmax(0,1fr)); } }
@media (max-width:700px){ .stat-grid{ grid-template-columns:1fr; } }

.stat-card{
  position:relative; overflow:hidden;
  background:var(--surface); border:1px solid var(--border); border-radius:14px; box-shadow:var(--shadow);
  padding:.7rem .85rem;
}
.stat-card::after{
  content:""; position:absolute; inset:0;
  background:radial-gradient(120% 120% at 110% -10%, rgba(79,70,229,.06), transparent 55%);
  pointer-events:none;
}
.stat-head{display:flex; align-items:center; gap:.45rem; color:var(--muted);}
.stat-title{font-weight:800; font-size:.9rem; text-transform:uppercase; letter-spacing:.06em;}
.badge{font-weight:800; font-size:.7rem; padding:.15rem .4rem; border-radius:999px; background:rgba(79,70,229,.12); color:#4338CA;}

.stat-value{font-weight:900; font-size:1.35rem; line-height:1.15; margin-top:.25rem; font-variant-numeric:tabular-nums;}
.stat-note{color:var(--muted); font-size:.9rem; margin-top:.15rem;}

.stat-card.tone-good{border-color:rgba(16,185,129,.28);}
.stat-card.tone-good .badge{background:rgba(16,185,129,.12); color:#047857;}

.stat-card.tone-warn{border-color:rgba(245,158,11,.30);}
.stat-card.tone-warn .badge{background:rgba(245,158,11,.14); color:#92400e;}

.stat-card.tone-bad{border-color:rgba(239,68,68,.30);}
.stat-card.tone-bad .badge{background:rgba(239,68,68,.14); color:#991b1b;}

/* Tables: polish + sticky headers */
div[data-testid="stDataFrame"] table tbody tr:hover td,
div[data-testid="stDataEditor"] table tbody tr:hover td{ background:#f8fafc !important; }
div[data-testid="stDataFrame"] td, div[data-testid="stDataEditor"] td{ border-bottom:1px solid var(--border) !important; }

[data-testid="stDataFrame"] thead tr:nth-child(1) th,
[data-testid="stTable"]     thead tr:nth-child(1) th { position: sticky; top: 0;  z-index: 3; background:#f8fafc; }
[data-testid="stDataFrame"] thead tr:nth-child(2) th,
[data-testid="stTable"]     thead tr:nth-child(2) th { position: sticky; top: 36px; z-index: 3; background:#f8fafc; }

/* Inputs / Buttons / Tabs */
.stButton>button{ border-radius:12px !important; padding:.55rem 1.1rem !important; font-weight:700; }
.stTabs [role="tab"]{ font-size:15px !important; font-weight:600 !important; }
</style>
"""

HOTFIX_CSS = """
<style id="ipad-hotfix">
/* ---------- iPad / iOS Safari hotfix ---------- */

/* 1) Tell Safari to use LIGHT built-in controls (stops auto-darkening). */
:root{ color-scheme: light; }
html{ -webkit-text-size-adjust:100%; }

/* 2) Force light backgrounds + dark text on all inputs/selects/areas. */
input, select, textarea, button,
.stTextInput input, .stNumberInput input,
.stTextArea textarea,
[data-baseweb="select"] [role="combobox"],
[data-baseweb="select"] input {
  -webkit-appearance: none;
  appearance: none;
  background-color: #ffffff !important;
  color: #0A112A !important;
  border-color: #E5E7EB !important;
}

/* 3) Fix iOS autofill painting yellow/black. */
input:-webkit-autofill,
select:-webkit-autofill,
textarea:-webkit-autofill {
  -webkit-text-fill-color: #0A112A !important;
  box-shadow: 0 0 0 1000px #ffffff inset !important;
}

/* 4) Streamlit dataframe/table → keep body light with readable text. */
div[data-testid="stDataFrame"] {
  background: #ffffff !important;
  color: #0A112A !important;
}
div[data-testid="stDataFrame"] [role="gridcell"],
div[data-testid="stDataFrame"] [role="row"],
div[data-testid="stDataFrame"] [data-testid="cell"] {
  background: #ffffff !important;
  color: #0A112A !important;
}

/* Header can stay dark if you like; ensure header text is readable. */
div[data-testid="stDataFrame"] [role="columnheader"]{
  background: #111827 !important;  /* dark header bar */
  color: #E5E7EB !important;       /* light text */
}

/* 5) Extra: Streamlit select dropdown menu items. */
div[data-baseweb="menu"] {
  background: #ffffff !important;
  color: #0A112A !important;
}

/* 6) Focus ring (visible on iPad). */
input:focus, select:focus, textarea:focus {
  outline: 2px solid #2563EB !important;
  box-shadow: none !important;
}
</style>
"""

# --------------------------- Page-specific CSS ---------------------------
# Keys must match page file names in /pages (without .py)
PAGE_CSS = {
    # 2) Add / Edit page
    "2_Add_or_Edit": """
<style>
/* Helper cards used by render_calc_helper (moved from page) */
.helper-grid{display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:.65rem;}
@media (max-width:1200px){.helper-grid{grid-template-columns: repeat(2, minmax(0,1fr));}}
@media (max-width:800px){.helper-grid{grid-template-columns:1fr;}}
.helper-card{background:#fff;border:1px solid var(--border);border-radius:12px;padding:.6rem .8rem;box-shadow:var(--shadow);}
.helper-title{font-weight:800;font-size:.95rem;margin:0 0 .15rem 0;}
.helper-value{font-weight:800;font-size:1.05rem;margin:.1rem 0;}
.calc-note{color:var(--muted);font-size:.9rem;margin:.1rem 0 0 0;}
</style>
""",

    # 3) View Stock page (optional overrides)
    "3_View_Stock": "",

    # 4) Systematic Decision page
    "4_Systematic_Decision": """
<style>
/* Example grid for decision rules */
.rules-grid{display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:12px;}
@media (max-width:900px){ .rules-grid{ grid-template-columns: 1fr; } }
.rule-card{background:#fff;border:1px solid var(--border);border-radius:12px;padding:.7rem .9rem;box-shadow:var(--shadow);}
.rule-title{font-weight:800;margin:0 0 .25rem 0;}
.rule-desc{color:var(--muted);font-size:.95rem;margin:0;}
</style>
""",

    # 5) Risk / Reward Planner
    "5_Risk_Reward_Planner": """
<style>
/* Planner page add-ons (builds on BASE_CSS) */

/* Area card: wraps a section with a soft container */
.area{
  background:var(--surface); border:1px solid var(--border); border-radius:16px;
  box-shadow:var(--shadow); padding:14px 16px; margin:.75rem 0 1rem;
}
.area .area-title{font-weight:800; font-size:1.05rem; margin:0 0 .15rem 0;}
.area .area-desc{color:var(--muted); font-size:.95rem; margin:0 0 .4rem 0;}

/* Simple responsive grids for inputs / content alignment */
.grid-2{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }
.grid-3{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }
.grid-4{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; }
@media (max-width:1100px){ .grid-4{grid-template-columns:repeat(2,minmax(0,1fr));} .grid-3{grid-template-columns:repeat(2,minmax(0,1fr));} }
@media (max-width:740px){  .grid-4,.grid-3,.grid-2{grid-template-columns:1fr;} }

/* Optional pill badges (used by R:R band when needed) */
.chip{
  display:inline-block; padding:.28rem .6rem; border-radius:999px; font-weight:800; font-size:.8rem; color:white;
}
.chip.good{ background:var(--success); }
.chip.warn{ background:var(--warning); }
.chip.bad{  background:var(--danger);  }

/* Light polish on inputs in dense cards */
.stNumberInput input, .stTextInput input, textarea{ font-size:15px !important; }
</style>
""",
}

# —— CSS used by Ongoing Trades page (cards, sidebar, grid fonts) ——
ONGOING_TRADES_CSS = """
<style>
:root{
  --bg:#f6f7fb; --surface:#ffffff; --text:#0f172a; --muted:#475569;
  --border:#e5e7eb; --shadow:0 8px 24px rgba(15,23,42,.06);
  --primary:#4f46e5; --info:#0ea5e9; --success:#10b981; --warning:#f59e0b; --danger:#ef4444;
}
html, body, [class*="css"]{ font-size:16px !important; color:var(--text); }
.stApp{ background: radial-gradient(1000px 500px at 10% -10%, #f0f3fb 0%, var(--bg) 60%), var(--bg); }
h1,h2,h3,h4{ color:var(--text) !important; font-weight:800 !important; letter-spacing:.2px; }
.sec{ background:var(--surface); border:1px solid var(--border); border-radius:14px;
      box-shadow:var(--shadow); padding:.65rem .9rem; margin:1rem 0 .6rem 0; display:flex; align-items:center; gap:.6rem; }
.sec .t{ font-size:1.05rem; font-weight:800; margin:0; padding:0; }
.sec .d{ color:var(--muted); font-size:.95rem; margin-left:.25rem; }
.sec::before{ content:""; display:inline-block; width:8px; height:26px; border-radius:6px; background:var(--primary); }
.sec.info::before{ background:var(--info); } .sec.success::before{ background:var(--success); }
.sec.warning::before{ background:var(--warning); } .sec.danger::before{ background:var(--danger); }
.stDataFrame, .stDataEditor{ font-size:15px !important; }
div[data-testid="stDataFrame"] td, div[data-testid="stDataEditor"] td{ border-bottom:1px solid var(--border) !important; }
div[data-baseweb="input"] input, textarea, .stNumberInput input{ font-size:15px !important; }
.stButton>button{ border-radius:12px !important; padding:.55rem 1.1rem !important; font-weight:700; }
.stTabs [role="tab"]{ font-size:15px !important; font-weight:600 !important; }
[data-testid="stSidebar"]{ background:linear-gradient(180deg,#0b1220 0%,#1f2937 100%) !important; }
[data-testid="stSidebar"] *{ color:#e5e7eb !important; }
</style>
"""
