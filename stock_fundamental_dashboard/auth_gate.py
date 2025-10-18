# auth_gate.py
import time
import streamlit as st

IDLE_TIMEOUT_SECONDS = 30 * 60  # keep in sync with app.py

def require_auth():
    user = st.session_state.get("auth_user")
    # idle logout
    now = time.time()
    last = st.session_state.get("last_active")
    if user and last and (now - last > IDLE_TIMEOUT_SECONDS):
        st.session_state.pop("auth_user", None)
        st.session_state.pop("last_active", None)
        user = None
    st.session_state["last_active"] = now

    if not user:
        # Bounce to app root (shows modal login)
        if hasattr(st, "switch_page"):
            try:
                st.switch_page("app.py")
            except Exception:
                pass
        st.stop()
