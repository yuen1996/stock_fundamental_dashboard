import os, time, base64, hmac, hashlib
import streamlit as st

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Stock Fundamental Dashboard", layout="wide", page_icon="📈")

# ---------- Minimal, safe CSS (REPLACE all previous CSS blocks) ----------
st.markdown("""
<style>
/* Force a light color scheme */
:root { color-scheme: light !important; }

/* Subtle app background + clean typography */
.stApp { background: radial-gradient(1100px 600px at 18% -10%, #f7fbff 0%, #eef4ff 45%, #ffffff 100%); }
html, body, [class*="css"] { font-size: 16px !important; }
h1, h2, h3, h4 { color: #0f172a !important; font-weight: 800 !important; letter-spacing: .2px; }

/* DO NOT restyle inputs/selects — let Streamlit's theme handle it */

/* Sidebar dark style */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0b1220 0%, #1f2937 100%) !important; }
[data-testid="stSidebar"] * { color: #e5e7eb !important; }

/* Nice buttons */
.stButton>button { border-radius: 12px !important; padding: .6rem 1.5rem !important; font-weight: 600; }

/* Hide default sidebar nav (we use page_link) */
section[data-testid="stSidebarNav"]{ display: none !important; }
</style>
""", unsafe_allow_html=True)



# -----------------------------------------------------------------------------
# Security helpers (PBKDF2-HMAC + rate limit + idle timeout)
# -----------------------------------------------------------------------------
LOCKOUT_THRESHOLD = 5           # attempts before locking
BASE_LOCK_SECONDS = 30          # first lock = 30s; exponential backoff after that
IDLE_TIMEOUT_SECONDS = 30 * 60  # auto logout after 30 minutes idle

def _pbkdf2_verify(stored: str, password: str) -> bool:
    """
    stored format: 'pbkdf2$<iterations>$<salt_b64>$<hash_b64>'
    """
    try:
        algo, iters_s, salt_b64, hash_b64 = stored.split("$", 3)
        if algo != "pbkdf2":
            return False
        iters = int(iters_s)
        salt = base64.b64decode(salt_b64.encode())
        want = base64.b64decode(hash_b64.encode())
        got = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iters)
        return hmac.compare_digest(got, want)
    except Exception:
        return False

def _load_users_from_secrets() -> dict:
    """
    Expect st.secrets['auth'] mapping like:
      [auth]
      ADMIN = "pbkdf2$200000$<salt_b64>$<hash_b64>"
      ANALYST = "pbkdf2$200000$<salt_b64>$<hash_b64>"
    (Plaintext values will still work but are discouraged.)
    """
    users = {}
    try:
        for k, v in st.secrets.get("auth", {}).items():
            users[str(k).strip().upper()] = str(v)
    except Exception:
        pass
    return users

def _first_run_defaults() -> dict:
    # fallback single user for first run ONLY — replace with PBKDF2 in secrets asap
    return {"ADMIN": "admin"}

def _users_and_first_run():
    users = _load_users_from_secrets()
    if users:
        return users, False
    return _first_run_defaults(), True

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
    if uid in store:
        del store[uid]

def _do_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

def _logout():
    st.session_state.pop("auth_user", None)
    st.session_state.pop("last_active", None)
    _do_rerun()

# -----------------------------------------------------------------------------
# Auth state + idle timeout
# -----------------------------------------------------------------------------
users, first_run = _users_and_first_run()
if "auth_user" not in st.session_state:
    st.session_state["auth_user"] = None

now = time.time()
last = st.session_state.get("last_active")
if st.session_state["auth_user"] and last and (now - last > IDLE_TIMEOUT_SECONDS):
    st.warning("Session expired due to inactivity.")
    _logout()
st.session_state["last_active"] = now

# -----------------------------------------------------------------------------
# Modal login (st.dialog) or inline fallback
# -----------------------------------------------------------------------------
_HAS_DIALOG = hasattr(st, "dialog")

def _login_form_body():
    if first_run:
        st.warning("First-run defaults active (ADMIN / admin). Please set hashed credentials in `.streamlit/secrets.toml`.")
        st.caption("Use PBKDF2 format: `pbkdf2$200000$<salt_b64>$<hash_b64>`")

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
            st.error(msg)
            return

        stored = users.get(uid)
        if stored is None:
            _rate_limit_note_failure(uid)
            st.error("Unknown user ID.")
            return

        authed = False
        if stored.startswith("pbkdf2$"):
            authed = _pbkdf2_verify(stored, pwd)
        else:
            authed = (pwd == stored)  # discouraged
            st.info("This user uses a plain-text password in secrets. Please switch to PBKDF2 format.")

        if not authed:
            _rate_limit_note_failure(uid)
            st.error("Incorrect password.")
            return

        _rate_limit_reset(uid)
        st.session_state["auth_user"] = uid
        st.session_state["last_active"] = time.time()
        _do_rerun()

# Show modal when not authenticated
if st.session_state["auth_user"] is None:
    if _HAS_DIALOG:
        @st.dialog("🔐 Sign in")
        def _login_dialog():
            _login_form_body()
        _login_dialog()  # open modal
        st.stop()
    else:
        st.title("🔐 Sign in")
        _login_form_body()
        st.stop()

# -----------------------------------------------------------------------------
# Authenticated app below
# -----------------------------------------------------------------------------
st.title("📈 Stock Fundamental Dashboard")
st.caption(f"Logged in as **{st.session_state['auth_user']}**")
st.divider()

st.markdown("""
Manage, analyze and visualize your stock financial data.

- **📊 Dashboard** — overview, latest annual/quarter summaries.  
- **✏️ Add / Edit** — enter annual & quarterly fundamentals.  
- **🔍 View Stock** — deep-dive ratios, charts, details per stock.  
- **🧭 Systematic Decision** — evaluate/value-investing funnel.  
- **📐 Risk / Reward Planner** — entries, stops, targets, size.  
- **🧾 Queue Audit Log** — every add/update/delete.  
- **📈 Ongoing Trades** & **📘 Trade History** — manage PnL.  
- **⚡ Momentum Data** — import/attach OHLC CSVs.  
- **🧪 Quant Tech & Signals** — candlesticks + indicators + signal panel & CSV export.
""")

# ---------- Sidebar navigation ----------
st.sidebar.title("Navigation")
st.sidebar.caption(f"Signed in as **{st.session_state['auth_user']}**")

try:
    st.sidebar.subheader("Fundamentals")
    st.sidebar.page_link("pages/1_Dashboard.py",           label="📊 Dashboard")
    st.sidebar.page_link("pages/2_Add_or_Edit.py",         label="✏️ Add / Edit")
    st.sidebar.page_link("pages/3_View_Stock.py",          label="🔍 View Stock")
    st.sidebar.page_link("pages/4_Systematic_Decision.py", label="🧭 Systematic Decision")
    st.sidebar.divider()

    st.sidebar.subheader("Trading & Logs")
    st.sidebar.page_link("pages/5_Risk_Reward_Planner.py", label="📐 Risk / Reward Planner")
    st.sidebar.page_link("pages/6_Queue_Audit_Log.py",     label="🧾 Queue Audit Log")
    st.sidebar.page_link("pages/7_Ongoing_Trades.py",      label="📈 Ongoing Trades")
    st.sidebar.page_link("pages/8_Trade_History.py",       label="📘 Trade History")
    st.sidebar.divider()

    st.sidebar.subheader("Momentum & Quant")
    st.sidebar.page_link("pages/9_Momentum_Data.py",       label="⚡ Momentum Data")
    st.sidebar.page_link("pages/10_Quant_Tech_Charts.py",  label="🧪 Quant Tech & Signals")
except Exception:
    st.sidebar.markdown("""
**Fundamentals**
- [📊 Dashboard](./pages/1_Dashboard.py)  
- [✏️ Add / Edit](./pages/2_Add_or_Edit.py)  
- [🔍 View Stock](./pages/3_View_Stock.py)  
- [🧭 Systematic Decision](./pages/4_Systematic_Decision.py)

**Trading & Logs**
- [📐 Risk / Reward Planner](./pages/5_Risk_Reward_Planner.py)  
- [🧾 Queue Audit Log](./pages/6_Queue_Audit_Log.py)  
- [📈 Ongoing Trades](./pages/7_Ongoing_Trades.py)  
- [📘 Trade History](./pages/8_Trade_History.py)

**Momentum & Quant**
- [⚡ Momentum Data](./pages/9_Momentum_Data.py)  
- [🧪 Quant Tech & Signals](./pages/10_Quant_Tech_Charts.py)
""")

st.sidebar.button("🚪 Log out", on_click=lambda: _logout(), use_container_width=True)

st.info("Pick a page from the sidebar to get started.")
