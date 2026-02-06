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


def _render_sidebar_shell(active_page_id: str | None = None) -> None:
    """
    Premium sidebar shell. Pure-UI only: safe to call on every page.

    - Adds branding + quick navigation (best-effort; falls back gracefully)
    - Adds quick actions (clear cache, rerun)
    """
    try:
        with st.sidebar:
            st.markdown(
                "<div class='sb-brand'>"
                "<div class='sb-title'>📈 Stock Dashboard</div>"
                "<div class='sb-sub'>Fundamentals • Momentum • Quant</div>"
                "</div>",
                unsafe_allow_html=True,
            )

            # Best-effort navigation (works on newer Streamlit; safe fallback otherwise)
            nav = [
                ("🏠 Dashboard", "pages/1_Dashboard.py"),
                ("✍️ Add / Edit", "pages/2_Add_or_Edit.py"),
                ("🔎 View Stock", "pages/3_View_Stock.py"),
                ("🧭 Systematic Decision", "pages/4_Systematic_Decision.py"),
                ("🎯 Risk / Reward", "pages/5_Risk_Reward_Planner.py"),
                ("🧾 Queue Audit Log", "pages/6_Queue_Audit_Log.py"),
                ("🟣 Ongoing Trades", "pages/7_Ongoing_Trades.py"),
                ("📚 Trade History", "pages/8_Trade_History.py"),
                ("💰 Dividends", "pages/8_Long_Trade_Dividends.py"),
                ("⚡ Momentum Data", "pages/9_Momentum_Data.py"),
                ("📈 Quant Tech Charts", "pages/10_Quant_Tech_Charts.py"),
                ("🤖 AI Analyst", "pages/11_AI_Analyst.py"),
                ("📖 Dictionary", "pages/12_Dictionary.py"),
            ]

            # Show nav links if the app is using Streamlit multipage
            if hasattr(st, "page_link"):
                for label, path in nav:
                    try:
                        st.page_link(path, label=label, use_container_width=True)
                    except Exception:
                        # In case page_link exists but app isn't in multipage mode
                        pass
            else:
                st.caption("Navigation (upgrade Streamlit for clickable links):")
                for label, _ in nav:
                    st.markdown(f"- {label}")

            st.markdown("<div class='sb-divider'></div>", unsafe_allow_html=True)

            with st.expander("⚙️ Quick actions", expanded=False):
                def _safe_rerun():
                    try:
                        st.rerun()
                    except Exception:
                        try:
                            st.experimental_rerun()
                        except Exception:
                            pass
                # Density toggle (CSS-only)
                density = st.selectbox("UI density", ["Comfort", "Compact"], index=0, key="ui_density")
                if density == "Compact":
                    st.markdown(
                        "<style>section.main > div.block-container{padding-top:.7rem;padding-bottom:1.4rem}</style>",
                        unsafe_allow_html=True,
                    )

                if st.button("🔄 Rerun", use_container_width=True, key="ui_rerun_btn"):
                    _safe_rerun()

                if st.button("🧹 Clear cache", use_container_width=True, key="ui_clear_cache_btn"):
                    try:
                        st.cache_data.clear()
                    except Exception:
                        pass
                    try:
                        st.cache_resource.clear()
                    except Exception:
                        pass
                    _safe_rerun()
    except Exception:
        # Never break the app due to sidebar polish.
        return


def setup_page(title: str, page_id: str | None = None) -> None:
    """Set page config + inject base CSS + page-specific CSS (if any)."""
    st.set_page_config(page_title=title, layout="wide")
    pid = page_id or get_page_id()
    css = BASE_CSS + HOTFIX_CSS + PAGE_CSS.get(pid, "")
    st.markdown(css, unsafe_allow_html=True)

    # Optional: consistent premium sidebar shell (safe; won't break existing sidebars)
    _render_sidebar_shell(active_page_id=pid)


def render_page_title(page_name: str) -> None:
    """Render the shared dashboard title for every page (premium hero header)."""
    from html import escape
    from datetime import datetime

    user = st.session_state.get("auth_user") or ""
    now = datetime.now().strftime("%d %b %Y, %H:%M")
    badge_user = f"<span class='pill'>👤 {escape(str(user))}</span>" if user else ""
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-left">
            <div class="brand">Stooper • Fundamentals</div>
            <div class="page">{escape(page_name)}</div>
            <div class="sub">Last refreshed: {now}</div>
          </div>
          <div class="hero-right">
            <span class="pill muted">📌 {escape(get_page_id())}</span>
            {badge_user}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

:root{
  --bg:#f6f7fb; --surface:#ffffff; --text:#0f172a; --muted:#475569; --border:#e5e7eb;
  --shadow:0 8px 24px rgba(15,23,42,.06);
  --shadow-lg:0 18px 55px rgba(15,23,42,.10);
  --primary:#4f46e5; --primary-2:#7c3aed; --info:#0ea5e9; --success:#10b981; --warning:#f59e0b; --danger:#ef4444;
}

/* App background + typography */
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 700px at 18% -10%, rgba(79,70,229,.10) 0%, rgba(124,58,237,.06) 35%, transparent 65%),
    radial-gradient(900px 500px at 85% 8%, rgba(14,165,233,.09) 0%, transparent 55%),
    var(--bg) !important;
}
html, body{
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
  letter-spacing:.1px;
  color:var(--text);
}
code, pre, kbd{
  font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}
h1, h2, h3, h4, h5, h6{ font-weight:900 !important; letter-spacing:.2px; }
p, li, label{ color:var(--text); }

/* Main container spacing */
section.main > div.block-container{
  padding-top: 1.1rem;
  padding-bottom: 2.0rem;
  max-width: 1500px;
}

/* Hide Streamlit chrome */
header[data-testid="stHeader"]{ background: transparent; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stToolbar"]{ visibility:hidden; }

/* Sidebar theme */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #0b1220 0%, #1f2937 100%) !important;
}
[data-testid="stSidebar"] *{ color:#e5e7eb !important; }
section[data-testid="stSidebarNav"]{ opacity: .95; }
section[data-testid="stSidebarNav"] a{ border-radius:12px; }
section[data-testid="stSidebarNav"] a:hover{ background: rgba(255,255,255,.06) !important; }

/* Sidebar shell */
.sb-brand{
  margin: .25rem 0 .8rem;
  padding: .8rem .85rem;
  border-radius: 16px;
  background: rgba(255,255,255,.06);
  border: 1px solid rgba(255,255,255,.10);
}
.sb-brand .sb-title{ font-weight: 900; font-size: 1.05rem; line-height:1.1; }
.sb-brand .sb-sub{ font-size: .85rem; color: rgba(229,231,235,.78); margin-top:.25rem; }
.sb-divider{ height:1px; background: rgba(255,255,255,.10); margin: .65rem 0; }

/* Hero header (render_page_title) */
.hero{
  position:relative;
  background: linear-gradient(135deg, rgba(255,255,255,.92) 0%, rgba(255,255,255,.80) 100%);
  border: 1px solid rgba(229,231,235,.90);
  border-radius: 20px;
  box-shadow: var(--shadow-lg);
  padding: 16px 18px;
  margin: .15rem 0 1.1rem;
  display:flex; align-items:center; justify-content:space-between; gap: 14px;
  overflow:hidden;
}
.hero::before{
  content:"";
  position:absolute; inset:-2px;
  background:
    radial-gradient(700px 280px at 15% 0%, rgba(79,70,229,.14), transparent 60%),
    radial-gradient(520px 260px at 90% 10%, rgba(14,165,233,.12), transparent 55%),
    radial-gradient(500px 240px at 70% 115%, rgba(124,58,237,.10), transparent 55%);
  pointer-events:none;
}
.hero-left, .hero-right{ position:relative; z-index:1; }
.hero .brand{
  font-weight: 900;
  text-transform: uppercase;
  letter-spacing: .14em;
  font-size: .72rem;
  color: var(--muted);
}
.hero .page{
  font-size: 2.05rem;
  font-weight: 950;
  letter-spacing: -0.02em;
  line-height: 1.05;
  margin-top: .25rem;
  background: linear-gradient(90deg, #0f172a, var(--primary));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.hero .sub{ color: var(--muted); font-size: .95rem; margin-top: .35rem; }
.hero-right{ display:flex; align-items:center; gap:.5rem; flex-wrap:wrap; justify-content:flex-end; }

.pill{
  display:inline-flex; align-items:center; gap:.35rem;
  padding: .26rem .62rem;
  border-radius: 999px;
  font-weight: 900;
  font-size: .78rem;
  border: 1px solid rgba(79,70,229,.22);
  background: rgba(79,70,229,.12);
  color: #4338CA;
}
.pill.muted{
  border-color: rgba(15,23,42,.10);
  background: rgba(15,23,42,.05);
  color: var(--muted);
}

/* Section header pill */
.sec{
  background:var(--surface); border:1px solid var(--border); border-radius:16px;
  box-shadow:var(--shadow); padding:.70rem .95rem; margin:1.05rem 0 .65rem;
  display:flex; align-items:center; gap:.65rem;
}
.sec .t{ font-size:1.05rem; font-weight:900; margin:0; }
.sec .d{ color:var(--muted); font-size:.95rem; margin-left:.25rem; }
.sec::before{ content:""; width:9px; height:28px; border-radius:7px; background:var(--primary); display:inline-block; }
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
  position:relative;
  background:var(--surface); border:1px solid rgba(229,231,235,.95); border-radius:16px;
  box-shadow:var(--shadow); padding:12px 14px; min-height:98px;
  display:flex; flex-direction:column; justify-content:center;
  transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
}
.kpi:hover{ transform: translateY(-2px); box-shadow: var(--shadow-lg); border-color: rgba(79,70,229,.22); }
.kpi .k{ color:var(--muted); font-weight:800; font-size:12px; text-transform:uppercase; letter-spacing:.06em; }
.kpi .v{ font-weight:950; font-size:26px; line-height:1; margin-top:6px; font-variant-numeric: tabular-nums; }

/* Stat Cards */
.stat-grid{display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; margin:.35rem 0 1rem;}
.stat-grid.cols-1{grid-template-columns:1fr;}
.stat-grid.cols-2{grid-template-columns:repeat(2,minmax(0,1fr));}
.stat-grid.cols-3{grid-template-columns:repeat(3,minmax(0,1fr));}
.stat-grid.cols-4{grid-template-columns:repeat(4,minmax(0,1fr));}
@media (max-width:1100px){ .stat-grid{ grid-template-columns:repeat(2,minmax(0,1fr)); } }
@media (max-width:700px){ .stat-grid{ grid-template-columns:1fr; } }

.stat-card{
  position:relative; overflow:hidden;
  background:var(--surface); border:1px solid rgba(229,231,235,.95); border-radius:16px; box-shadow:var(--shadow);
  padding:.75rem .90rem;
  transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
}
.stat-card:hover{ transform: translateY(-2px); box-shadow: var(--shadow-lg); border-color: rgba(79,70,229,.18); }
.stat-card::after{
  content:""; position:absolute; inset:0;
  background: radial-gradient(120% 120% at 110% -10%, rgba(79,70,229,.07), transparent 55%);
  pointer-events:none;
}
.stat-head{display:flex; align-items:center; gap:.45rem; color:var(--muted);}
.stat-title{font-weight:900; font-size:.85rem; text-transform:uppercase; letter-spacing:.08em;}
.badge{font-weight:900; font-size:.70rem; padding:.14rem .42rem; border-radius:999px; background:rgba(79,70,229,.12); color:#4338CA; border:1px solid rgba(79,70,229,.18);}

.stat-value{font-weight:950; font-size:1.38rem; line-height:1.15; margin-top:.25rem; font-variant-numeric:tabular-nums;}
.stat-note{color:var(--muted); font-size:.92rem; margin-top:.15rem;}

/* Tone borders */
.stat-card.tone-good{border-color:rgba(16,185,129,.30);}
.stat-card.tone-good .badge{background:rgba(16,185,129,.12); border-color:rgba(16,185,129,.18); color:#047857;}
.stat-card.tone-warn{border-color:rgba(245,158,11,.32);}
.stat-card.tone-warn .badge{background:rgba(245,158,11,.14); border-color:rgba(245,158,11,.18); color:#92400e;}
.stat-card.tone-bad{border-color:rgba(239,68,68,.32);}
.stat-card.tone-bad .badge{background:rgba(239,68,68,.14); border-color:rgba(239,68,68,.18); color:#991b1b;}

/* DataFrame & chart containers */
div[data-testid="stDataFrame"], div[data-testid="stDataEditor"]{
  background: var(--surface) !important;
  border: 1px solid rgba(229,231,235,.95) !important;
  border-radius: 16px !important;
  overflow: hidden !important;
  box-shadow: var(--shadow) !important;
}
div[data-testid="stDataFrame"] table tbody tr:hover td,
div[data-testid="stDataEditor"] table tbody tr:hover td{ background:#f8fafc !important; }
div[data-testid="stDataFrame"] td, div[data-testid="stDataEditor"] td{ border-bottom:1px solid var(--border) !important; }

/* Sticky table headers */
[data-testid="stDataFrame"] thead tr:nth-child(1) th,
[data-testid="stTable"]     thead tr:nth-child(1) th { position: sticky; top: 0;  z-index: 3; background:#f8fafc; }
[data-testid="stDataFrame"] thead tr:nth-child(2) th,
[data-testid="stTable"]     thead tr:nth-child(2) th { position: sticky; top: 36px; z-index: 3; background:#f8fafc; }

/* Inputs polish */
div[data-baseweb="input"] input, textarea,
[data-baseweb="select"] [role="combobox"]{
  border-radius: 12px !important;
}
.stTextInput input, .stNumberInput input, .stTextArea textarea{
  font-size: 15px !important;
}

/* Buttons */
.stButton>button{
  border-radius:12px !important;
  padding:.60rem 1.10rem !important;
  font-weight:900 !important;
}
.stButton>button:hover{ filter: brightness(1.02); transform: translateY(-1px); }
.stButton>button:active{ transform: translateY(0px); }

/* Tabs */
.stTabs [role="tab"]{ font-size:15px !important; font-weight:800 !important; }
.stTabs [role="tab"][aria-selected="true"]{ color: var(--primary) !important; }
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
