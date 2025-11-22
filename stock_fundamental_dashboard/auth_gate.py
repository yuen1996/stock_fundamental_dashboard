# auth_gate.py
import time
import streamlit as st

# Keep this in sync with app.py
IDLE_TIMEOUT_SECONDS = 30 * 60  # 30 minutes


def require_auth() -> None:
    """
    Used at the top of every protected /pages/*.py file.

    - Trusts app.py to own the login UI.
    - Here we just:
        * enforce idle timeout
        * block the page if there's no logged-in user
    """

    # Make sure the key exists (same pattern as app.py)
    if "auth_user" not in st.session_state:
        st.session_state["auth_user"] = None

    user = st.session_state.get("auth_user")

    # ----- Idle timeout (mirror app.py behaviour) -----
    now = time.time()
    last = st.session_state.get("last_active")

    if user and last and (now - last > IDLE_TIMEOUT_SECONDS):
        # Session expired -> soft logout and ask user to sign in again
        st.session_state["auth_user"] = None
        st.session_state["last_active"] = now
        st.warning("Session expired due to inactivity. Please sign in again from the main page.")
        st.stop()

    # Still active → refresh last_active
    st.session_state["last_active"] = now

    # ----- Not logged in at all? -----
    if not st.session_state["auth_user"]:
        # IMPORTANT: do *not* show another login form here.
        # app.py already owns the login dialog.
        st.warning("Please sign in on the main app page to use this screen.")
        st.stop()
