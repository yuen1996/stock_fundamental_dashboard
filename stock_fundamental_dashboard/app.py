import os, time, base64, hmac, hashlib
from datetime import datetime
import streamlit as st

# =============================================================================
# Page setup
# =============================================================================
st.set_page_config(page_title="Stock Fundamental Dashboard", layout="wide", page_icon="📈")

# ---------- Minimal, safe CSS ----------
st.markdown("""
<style>
:root { color-scheme: light !important; }
.stApp { background: radial-gradient(1100px 600px at 18% -10%, #f7fbff 0%, #eef4ff 45%, #ffffff 100%); }
html, body, [class*="css"] { font-size: 16px !important; }
h1, h2, h3, h4 { color: #0f172a !important; font-weight: 800 !important; letter-spacing: .2px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0b1220 0%, #1f2937 100%) !important; }
[data-testid="stSidebar"] * { color: #e5e7eb !important; }
.stButton>button { border-radius: 12px !important; padding: .6rem 1.5rem !important; font-weight: 600; }
section[data-testid="stSidebarNav"]{ display: none !important; } /* hide default nav */
.kpi {display:flex;gap:.6rem;align-items:center;background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:12px 16px;box-shadow:0 1px 2px rgba(16,24,40,.06)}
.kpi .v{font-weight:800;color:#0f172a;font-size:20px}
.kpi .l{color:#475569;font-size:12px;margin-top:2px}
.quick a{display:flex;align-items:center;gap:.6rem;background:#fff;border-radius:14px;border:1px solid #e5e7eb;padding:14px 16px;text-decoration:none;color:#0f172a}
.quick a:hover{border-color:#94a3b8}
.footer{color:#64748b;font-size:12px}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Security helpers (PBKDF2-HMAC + rate limit + idle timeout)
# =============================================================================
LOCKOUT_THRESHOLD = 5
BASE_LOCK_SECONDS = 30
IDLE_TIMEOUT_SECONDS = 30 * 60  # 30 minutes

def _pbkdf2_verify(stored: str, password: str) -> bool:
    """stored: 'pbkdf2$<iterations>$<salt_b64>$<hash_b64>'"""
    try:
        algo, iters_s, salt_b64, hash_b64 = stored.split("$", 3)
        if algo != "pbkdf2": return False
        iters = int(iters_s)
        salt = base64.b64decode(salt_b64.encode())
        want = base64.b64decode(hash_b64.encode())
        got = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iters)
        return hmac.compare_digest(got, want)
    except Exception:
        return False

def _load_users_from_secrets() -> dict:
    users = {}
    try:
        for k, v in st.secrets.get("auth", {}).items():
            users[str(k).strip().upper()] = str(v)
    except Exception:
        pass
    return users

def _first_run_defaults() -> dict:
    # First run only. Please move to PBKDF2 in .streamlit/secrets.toml
    return {"ADMIN": "admin"}

def _users_and_first_run():
    users = _load_users_from_secrets()
    return (users, False) if users else (_first_run_defaults(), True)

def _rate_limit_check(uid: str) -> tuple[bool, str]:
    now = time.time()
    store = st.session_state.setdefault("login_failures", {})
    rec = store.get(uid, {"count": 0, "locked_until": 0.0})
    if now < rec.get("locked_until", 0.0):
        wait = int(rec["locked_until"] - now)
        return False, f"Too many attempts. Try again in {wait}s."
    return True, ""

def _rate_limit_note_failure(uid: str):
    now = time.time()
    store = st.session_state.setdefault("login_failures", {})
    rec = store.get(uid, {"count": 0, "locked_until": 0.0})
    rec["count"] = rec.get("count", 0) + 1
    if rec["count"] >= LOCKOUT_THRESHOLD:
        extra = rec["count"] - LOCKOUT_THRESHOLD
        rec["locked_until"] = now + BASE_LOCK_SECONDS * (2 ** extra)
    store[uid] = rec

def _rate_limit_reset(uid: str):
    store = st.session_state.setdefault("login_failures", {})
    if uid in store: del store[uid]

def _do_rerun():
    try: st.rerun()
    except Exception:
        try: st.experimental_rerun()
        except Exception: pass

def _logout():
    st.session_state.pop("auth_user", None)
    st.session_state.pop("last_active", None)
    _do_rerun()

# =============================================================================
# Auth state + idle timeout
# =============================================================================
users, first_run = _users_and_first_run()
if "auth_user" not in st.session_state:
    st.session_state["auth_user"] = None

now = time.time()
last = st.session_state.get("last_active")
if st.session_state["auth_user"] and last and (now - last > IDLE_TIMEOUT_SECONDS):
    st.warning("Session expired due to inactivity.")
    _logout()
st.session_state["last_active"] = now

# =============================================================================
# Modal login (st.dialog) or inline fallback
# =============================================================================
_HAS_DIALOG = hasattr(st, "dialog")
def _login_form_body():
    if first_run:
        st.warning("First-run defaults active (ADMIN / admin). Please set hashed credentials in `.streamlit/secrets.toml`.")
        st.caption("PBKDF2 format: `pbkdf2$200000$<salt_b64>$<hash_b64>`")

    uid = st.text_input("User ID").strip().upper()
    pwd = st.text_input("Password", type="password")

    colA, colB = st.columns([1, 1])
    with colA:
        submit = st.button("Log in", type="primary", use_container_width=True)
    with colB:
        st.button("Cancel", use_container_width=True, key="login_cancel")

    if submit:
        if not uid or not pwd:
            st.error("Please enter both User ID and Password.")
            return
        ok, msg = _rate_limit_check(uid)
        if not ok:
            st.error(msg); return
        stored = users.get(uid)
        if stored is None:
            _rate_limit_note_failure(uid); st.error("Unknown user ID."); return
        authed = _pbkdf2_verify(stored, pwd) if stored.startswith("pbkdf2$") else (pwd == stored)
        if not stored.startswith("pbkdf2$"):
            st.info("This user uses a plain-text password in secrets. Please switch to PBKDF2 format.")
        if not authed:
            _rate_limit_note_failure(uid); st.error("Incorrect password."); return
        _rate_limit_reset(uid)
        st.session_state["auth_user"] = uid
        st.session_state["last_active"] = time.time()
        _do_rerun()

if st.session_state["auth_user"] is None:
    if _HAS_DIALOG:
        @st.dialog("🔐 Sign in")
        def _login_dialog(): _login_form_body()
        _login_dialog(); st.stop()
    else:
        st.title("🔐 Sign in"); _login_form_body(); st.stop()

# =============================================================================
# Authenticated app below
# =============================================================================

# ---------- sidebar navigation (role-aware) ----------
uid = st.session_state["auth_user"]
is_admin = (uid == "ADMIN")

def _switch(path: str):
    try:
        st.switch_page(path)
    except Exception:
        # Fallback note (Streamlit < 1.32, or when running on Cloud with older router)
        st.info(f"Open the page from the sidebar: **{path}**")

st.sidebar.title("Navigation")
st.sidebar.caption(f"Signed in as **{uid}**")

# You have these pages in your app; keep them all here:
PAGES_FUND = [
    ("pages/1_Dashboard.py",            "📊 Dashboard"),
    ("pages/2_Add_or_Edit.py",          "✏️ Add or Edit"),
    ("pages/3_View_Stock.py",           "🔍 View Stock"),
    ("pages/4_Systematic_Decision.py",  "🧭 Systematic Decision"),
]
PAGES_TRADING = [
    ("pages/5_Risk_Reward_Planner.py",  "📐 Risk Reward Planner"),
    ("pages/6_Queue_Audit_Log.py",      "🧾 Queue Audit Log"),
    ("pages/7_Ongoing_Trades.py",       "📈 Ongoing Trades"),
    ("pages/8_Trade_History.py",        "📘 Trade History"),
    ("pages/Long_Trade_Dividends.py",   "💸 Long Trade Dividends"),  # new in list
]
PAGES_QUANT = [
    ("pages/9_Momentum_Data.py",        "⚡ Momentum Data"),
    ("pages/10_Quant_Tech_Charts.py",   "🧪 Quant Tech Charts"),
    ("pages/AI_Analyst.py",             "🤖 AI Analyst"),            # new in list
]

# ADMIN can get the audit log up top for convenience
if is_admin:
    st.sidebar.subheader("Operations")
    st.sidebar.page_link("pages/6_Queue_Audit_Log.py", label="🧾 Queue Audit Log")

st.sidebar.subheader("Fundamentals")
for p, lbl in PAGES_FUND:
    st.sidebar.page_link(p, label=lbl)

st.sidebar.subheader("Trading & Logs")
for p, lbl in PAGES_TRADING:
    st.sidebar.page_link(p, label=lbl)

st.sidebar.subheader("Momentum & Quant")
for p, lbl in PAGES_QUANT:
    st.sidebar.page_link(p, label=lbl)

st.sidebar.divider()
st.sidebar.button("🚪 Log out", on_click=_logout, use_container_width=True)

# ---------- command palette (quick jump) ----------
st.markdown("### ")
with st.container():
    c1, c2 = st.columns([2.2, 1])
    with c1:
        options = (
            [f"{lbl} — Fundamentals" for _, lbl in PAGES_FUND] +
            [f"{lbl} — Trading & Logs" for _, lbl in PAGES_TRADING] +
            [f"{lbl} — Momentum & Quant" for _, lbl in PAGES_QUANT]
        )
        sel = st.selectbox("⌨️ Quick jump (Command-Palette)", ["—"] + options, index=0)
        if sel != "—":
            label = sel.split(" — ")[0]
            lookup = {lbl: path for group in (PAGES_FUND + PAGES_TRADING + PAGES_QUANT) for path, lbl in [group]}
            _switch(lookup[label])
    with c2:
        st.write("")  # spacing
        st.write("")

# ---------- hero ----------
st.title("📈 Stock Fundamental Dashboard")
st.caption(f"Welcome, **{uid}** — your all-in-one workspace for fundamentals, trading records, and quantitative signals.")
st.divider()

# ---------- small KPIs (no DB call; purely cosmetic placeholders) ----------
c1, c2, c3, c4 = st.columns(4)
def kpi(icon, value, label):
    with st.container():
        st.markdown(f"""
        <div class="kpi"><div class="i">{icon}</div>
        <div><div class="v">{value}</div><div class="l">{label}</div></div></div>
        """, unsafe_allow_html=True)

kpi("🧾", "Clean audit trail", "All edits/dels are logged")
with c2: kpi("💾", "Local JSON I/O", "Backup & import on the Edit page")
with c3: kpi("🧮", "Auto-derived metrics", "NIM, CASA, EBITDA & more")
with c4: kpi("🔒", "Secure login", "PBKDF2 + idle timeout")

# ---------- quick launch grid ----------
st.markdown("#### Quick launch")
ql_col1, ql_col2, ql_col3 = st.columns(3)
def quick_link(path, label, desc):
    st.markdown(f'<div class="quick"><a href="#" onclick="return false;">{label}</a></div>', unsafe_allow_html=True)
    if st.button(label, use_container_width=True, key=f"ql_{path}"):
        _switch(path)
    st.caption(desc)

with ql_col1:
    quick_link("pages/1_Dashboard.py",           "📊 Dashboard", "Overview cards & recent changes")
    quick_link("pages/3_View_Stock.py",          "🔍 View Stock", "Deep-dive a single name")
with ql_col2:
    quick_link("pages/2_Add_or_Edit.py",         "✏️ Add or Edit", "Annual/quarterly inputs + JSON import")
    quick_link("pages/5_Risk_Reward_Planner.py", "📐 R/R Planner", "Stops, targets, sizing")
with ql_col3:
    quick_link("pages/9_Momentum_Data.py",       "⚡ Momentum Data", "CSV OHLC loader + mapping")
    quick_link("pages/10_Quant_Tech_Charts.py",  "🧪 Quant Tech Charts", "Candles, indicators, signals")

st.divider()

# ---------- About / Help ----------
with st.expander("ℹ️ About this workspace"):
    st.markdown("""
- **Fundamentals**: configurable industry buckets; auto-casing & robust quarterly/annual upserts.
- **Banking helpers**: CASA, NIM, leverage, CIR; override-aware.
- **Quick Edit**: fast table editor with per-row audit and safe LastModified stamping.
- **Audit log**: every create/update/delete is written to `data/audit_log.jsonl`.
- **Momentum & Quant**: attach OHLC, visualize, and export signal CSVs.
    """)

# ---------- tiny footer ----------
def _data_version_info():
    try:
        ts = os.stat(os.path.join(os.path.dirname(__file__), ".data.version")).st_mtime
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "—"

st.markdown(
    f"""<div class="footer">Data etag: <b>{_data_version_info()}</b> · 
    Idle timeout: <b>{IDLE_TIMEOUT_SECONDS//60} min</b> · 
    User: <b>{uid}</b></div>""",
    unsafe_allow_html=True,
)
