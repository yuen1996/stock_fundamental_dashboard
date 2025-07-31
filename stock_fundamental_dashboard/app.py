import streamlit as st

st.set_page_config(page_title="Stock Fundamental Dashboard", layout="wide", page_icon="📈")

# ---------- Global CSS ----------
BASE_CSS = """
<style>
html, body, [class*="css"] {
  font-size: 16px !important;
}
h1, h2, h3, h4 {
  color: #0f172a !important;
  font-weight: 800 !important;
  letter-spacing: .2px;
}
p, label, span, div {
  color: #0b132a;
}
.stApp {
  background: radial-gradient(1100px 600px at 18% -10%, #f7fbff 0%, #eef4ff 45%, #ffffff 100%);
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b1220 0%, #1f2937 100%) !important;
}
[data-testid="stSidebar"] * {
  color: #e5e7eb !important;
}
.stTabs [role="tab"] {
  font-size: 15px !important;
  font-weight: 600 !important;
}
.stDataFrame, .stDataEditor, .dataframe {
  font-size: 15px !important;
}
div[data-baseweb="input"] input, textarea, .stNumberInput input {
  font-size: 15px !important;
}
.stButton>button {
  border-radius: 12px !important;
  padding: .6rem 1.5rem !important;
  font-weight: 600;
}
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

st.title("📈 Stock Fundamental Dashboard")

st.markdown(
    """
Welcome to the **Stock Fundamental Dashboard**!  
Manage, analyze and visualize your stock financial data.

- Go to the **Dashboard** to view and sort all stocks.
- Use **Add/Edit** to enter or update stock information.
- Select **View Stock** for detailed annual and quarterly financials, ratios, and charts.
- Check **Systematic Decision** to evaluate strategies.
- Plan trades with **Risk / Reward Planner** (entry, stop, take, R:R, position size).

---
"""
)
st.info("Choose a page from the sidebar to begin.")

# ---------- Sidebar Navigation ----------
st.sidebar.title("Navigation")

# Prefer Streamlit's built-in page links (Streamlit >= 1.22). Fallback to markdown links.
try:
    st.sidebar.page_link("pages/1_Dashboard.py", label="📊 Dashboard")
    st.sidebar.page_link("pages/2_Add_or_Edit.py", label="✏️ Add / Edit")
    st.sidebar.page_link("pages/3_View_Stock.py", label="🔍 View Stock")
    st.sidebar.page_link("pages/4_Systematic_Decision.py", label="🧭 Systematic Decision")
    st.sidebar.page_link("pages/5_Risk_Reward_Planner.py", label="📐 Risk / Reward Planner")
except Exception:
    st.sidebar.markdown(
        """
- [📊 Dashboard](./pages/1_Dashboard.py)
- [✏️ Add / Edit](./pages/2_Add_or_Edit.py)
- [🔍 View Stock](./pages/3_View_Stock.py)
- [🧭 Systematic Decision](./pages/4_Systematic_Decision.py)
- [📐 Risk / Reward Planner](./pages/5_Risk_Reward_Planner.py)
"""
    )

