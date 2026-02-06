# auth_gate.py
import time
import streamlit as st

# Keep this in sync with app.py
IDLE_TIMEOUT_SECONDS = 30 * 60  # 30 minutes


def _redirect_to_login() -> None:
    """
    Send the user back to the main app page (app.py) to log in.

    Uses st.switch_page when available (Streamlit >= 1.30).
    Falls back to the old warning if not.
    """
    # Newer Streamlit: use built-in page switching
    if hasattr(st, "switch_page"):
        try:
            # If your main script is app.py, this is the correct target.
            st.switch_page("app.py")
            return  # switch_page triggers a rerun, so normally this line isn't reached
        except Exception:
            # If anything goes wrong, fall back to warning below
            pass

    # Fallback for older versions / errors
    st.warning("Please sign in on the main app page to use this screen.")
    st.stop()


def require_auth() -> None:
    """
    Used at the top of every protected /pages/*.py file.

    - Trusts app.py to own the login UI.
    - Here we just:
        * enforce idle timeout
        * block the page if there's no logged-in user
        * redirect to app.py when login is needed
    """

    # Make sure the key exists (same pattern as app.py)
    if "auth_user" not in st.session_state:
        st.session_state["auth_user"] = None

    user = st.session_state.get("auth_user")

    # ----- Idle timeout (mirror app.py behaviour) -----
    now = time.time()
    last = st.session_state.get("last_active")

    if user and last and (now - last > IDLE_TIMEOUT_SECONDS):
        # Session expired -> soft logout and send user back to login page
        st.session_state["auth_user"] = None
        st.session_state["last_active"] = now
        _redirect_to_login()
        st.stop()

    # Still active → refresh last_active
    st.session_state["last_active"] = now

    # ----- Not logged in at all? -> redirect to app.py -----
    if not st.session_state["auth_user"]:
        _redirect_to_login()
        st.stop()

