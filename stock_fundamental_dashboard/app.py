import streamlit as st

st.set_page_config(page_title="Stock Fundamental Dashboard", layout="wide", page_icon="📈")

# ---------- Global CSS ----------
BASE_CSS = """
<style>
/* Base typography */
html, body, [class*="css"] {
  font-size: 16px !important;
}
h1, h2, h3, h4 {
  color: #0f172a !important;       /* slate-900 */
  font-weight: 800 !important;
  letter-spacing: .2px;
}
p, label, span, div {
  color: #0b132a;                   /* strong readable black-blue */
}

/* App background + cards */
.stApp {
  background: radial-gradient(1100px 600px at 18% -10%, #f7fbff 0%, #eef4ff 45%, #ffffff 100%);
}

/* Sidebar dark */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b1220 0%, #1f2937 100%) !important;
}
[data-testid="stSidebar"] * {
  color: #e5e7eb !important;        /* slate-200 text in sidebar */
}

/* Tabs */
.stTabs [role="tab"] {
  font-size: 15px !important;
  font-weight: 600 !important;
}

/* Editors & tables */
.stDataFrame, .stDataEditor, .dataframe {
  font-size: 15px !important;
}
div[data-baseweb="input"] input, textarea, .stNumberInput input {
  font-size: 15px !important;
}

/* Buttons */
.stButton>button {
  border-radius: 12px !important;
  padding: .6rem 1rem !important;
  font-weight: 700 !important;
  border: 1px solid #0ea5e9 !important;   /* sky-500 */
  background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%) !important;
  color: #0b132a !important;
}
.stButton>button:hover {
  box-shadow: 0 6px 22px rgba(14,165,233,.25) !important;
}

/* Nice pill badge utility (optional) */
.badge {
  display:inline-block; padding:.25rem .55rem; border-radius:999px;
  background:#111827; color:#fff; font-size:.85rem;
}
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

# ---------- App header ----------
st.title("Stock Fundamental Dashboard")

st.markdown(
    """
    Use the left sidebar to navigate:

    - **Dashboard**: summary, inline edit & delete (annual rows).
    - **Add or Edit**: manage annual & quarterly data.
    - **View Stock**: deep‑dive per stock with ratios and charts.
    """
)
