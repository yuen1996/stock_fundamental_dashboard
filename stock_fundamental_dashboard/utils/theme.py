# utils/theme.py
import streamlit as st

# ---- Color tokens (WCAG-friendly) ----
PALETTE = {
    "bg":        "#f6f7fb",   # neutral app background
    "surface":   "#ffffff",
    "text":      "#0f172a",
    "muted":     "#475569",
    "border":    "#e5e7eb",
    "shadow":    "0 8px 24px rgba(15, 23, 42, .06)",

    # variants
    "primary":   "#4f46e5",   # indigo
    "info":      "#0ea5e9",   # sky
    "success":   "#10b981",   # emerald
    "warning":   "#f59e0b",   # amber
    "danger":    "#ef4444",   # red
}

def _css():
    return f"""
<style>
:root {{
  --bg:        {PALETTE["bg"]};
  --surface:   {PALETTE["surface"]};
  --text:      {PALETTE["text"]};
  --muted:     {PALETTE["muted"]};
  --border:    {PALETTE["border"]};
  --shadow:    {PALETTE["shadow"]};

  --primary:   {PALETTE["primary"]};
  --info:      {PALETTE["info"]};
  --success:   {PALETTE["success"]};
  --warning:   {PALETTE["warning"]};
  --danger:    {PALETTE["danger"]};
}}

html, body, [class*="css"] {{
  font-size: 16px !important;
  color: var(--text);
}}
.stApp {{
  background:
    radial-gradient(1000px 500px at 10% -10%, #f0f3fb 0%, var(--bg) 60%),
    var(--bg);
}}

/* Headings */
h1, h2, h3, h4 {{
  color: var(--text) !important;
  font-weight: 800 !important;
  letter-spacing: .2px;
}}
.page-title {{
  font-size: 1.8rem; font-weight: 800; margin: .2rem 0 1rem 0;
}}

/* Cards / surfaces (used by section headers) */
.section {{
  background: var(--surface);
  border-radius: 14px;
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  padding: .75rem .9rem;
  margin: 1rem 0 .5rem 0;
  display: flex; align-items: center; gap: .6rem;
}}
.section .title {{
  margin: 0; padding: 0;
  font-size: 1.05rem; font-weight: 800; letter-spacing: .2px;
}}
.section .desc {{ color: var(--muted); font-size: .95rem; margin-left: .25rem; }}

/* colored stripe */
.section::before {{
  content: "";
  display: inline-block;
  width: 8px; height: 26px;
  border-radius: 6px;
  background: var(--primary);
}}
.section.info::before    {{ background: var(--info);    }}
.section.success::before {{ background: var(--success); }}
.section.warning::before {{ background: var(--warning); }}
.section.danger::before  {{ background: var(--danger);  }}

/* Buttons */
.stButton>button {{
  border-radius: 12px !important;
  padding: .55rem 1.1rem !important;
  font-weight: 700;
}}

/* Tabs */
.stTabs [role="tab"] {{
  font-size: 15px !important; font-weight: 600 !important;
}}

/* Tables (dataframe / data_editor) */
.stDataFrame, .stDataEditor {{
  font-size: 15px !important;
}}
div[data-testid="stDataFrame"] table tbody tr:hover td,
div[data-testid="stDataEditor"] table tbody tr:hover td {{
  background: #f8fafc !important;
}}

/* Sidebar theme */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #0b1220 0%, #1f2937 100%) !important;
}}
[data-testid="stSidebar"] * {{ color: #e5e7eb !important; }}
</style>
"""

def inject():
    """Inject global CSS. Call once per page."""
    st.markdown(_css(), unsafe_allow_html=True)

def page_setup(page_title: str = "", icon: str | None = None, layout: str = "wide"):
    """
    One call per page to standardize look & feel.
    - Sets layout
    - Injects CSS
    - Optionally renders a large page title (emoji-safe)
    """
    try:
        st.set_page_config(page_title=page_title or "Dashboard", layout=layout, page_icon=icon)
    except Exception:
        # set_page_config can only be called once; ignore if already set
        pass
    inject()
    if page_title:
        # You can still keep your existing st.title()/st.header(); this is optional.
        st.markdown(f'<div class="page-title">{(icon or "")} {page_title}</div>', unsafe_allow_html=True)

def section_header(title: str, variant: str = "primary", description: str | None = None):
    """
    Render a consistent section header with a colored stripe.
    variant: primary | info | success | warning | danger
    """
    variant = variant if variant in {"primary","info","success","warning","danger"} else "primary"
    html = f'<div class="section {variant}"><div class="title">{title}</div>'
    if description:
        html += f'<div class="desc">{description}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
