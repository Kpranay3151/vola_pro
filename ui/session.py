"""Session state management for the Streamlit app."""

import streamlit as st


def init_session_state():
    """Initialize all session state keys if not already present."""
    defaults = {
        "current_user_id": None,       # set on first sidebar render
        "chat_histories": {},           # {user_id: [{"role", "content", "metadata"}]}
        "pending_query": None,          # set by example-query buttons
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def get_chat_history(user_id: str) -> list:
    return st.session_state["chat_histories"].get(user_id, [])


def append_message(user_id: str, role: str, content: str, metadata: dict = None):
    if user_id not in st.session_state["chat_histories"]:
        st.session_state["chat_histories"][user_id] = []
    st.session_state["chat_histories"][user_id].append({
        "role": role,
        "content": content,
        "metadata": metadata or {},
    })


def clear_chat_history(user_id: str):
    st.session_state["chat_histories"][user_id] = []
