"""Chat interface — message rendering, pipeline calls, inline chart display."""

import os
import streamlit as st

from ui.session import get_chat_history, append_message
from ui.metadata import render_metadata_panel


def render_chat(pipeline):
    """Render the chat area: history + input + pipeline execution."""
    user_id = st.session_state["current_user_id"]

    # Header
    profile = pipeline.cache.get_profile(user_id)
    user_name = profile.get("user_name", user_id) if profile else user_id
    st.header(f"Chat with {user_name}'s Financial Data")

    # ── Render message history ──────────────────────────
    for msg in get_chat_history(user_id):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                meta = msg.get("metadata", {})
                if meta.get("visualizations"):
                    _render_charts(meta["visualizations"])
                if meta:
                    render_metadata_panel(meta)

    # ── Get user input (from example button OR chat box) ──
    pending = st.session_state.get("pending_query")
    if pending:
        st.session_state["pending_query"] = None

    user_input = st.chat_input("Ask about your finances...")
    prompt = pending or user_input

    if not prompt:
        return

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    append_message(user_id, "user", prompt)

    # ── Call pipeline ───────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your transactions..."):
            try:
                result = pipeline.run(user_id=user_id, prompt=prompt)
            except Exception as e:
                result = {
                    "response": f"An unexpected error occurred: {e}",
                    "visualizations": [],
                    "cache_hit": False,
                    "latency_ms": 0,
                    "guardrail_flags": ["SYSTEM_ERROR"],
                    "data_summary": {},
                    "user_name": user_name,
                }

        # Handle error vs normal response
        if "error" in result:
            st.error(result["error"])
            response_text = result["error"]
            metadata = {
                "latency_ms": result.get("latency_ms", 0),
                "guardrail_flags": [],
                "visualizations": [],
                "cache_hit": False,
            }
        else:
            response_text = result.get("response", "")
            st.markdown(response_text)

            if result.get("visualizations"):
                _render_charts(result["visualizations"])

            metadata = {
                "latency_ms": result.get("latency_ms", 0),
                "cache_hit": result.get("cache_hit", False),
                "guardrail_flags": result.get("guardrail_flags", []),
                "visualizations": result.get("visualizations", []),
            }

        render_metadata_panel(metadata)

    append_message(user_id, "assistant", response_text, metadata)
    st.rerun()


def _render_charts(chart_paths: list):
    """Display chart PNGs inline, 2 per row."""
    cols = st.columns(min(len(chart_paths), 2))
    for i, path in enumerate(chart_paths):
        with cols[i % 2]:
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.warning(f"Chart not found: {path}")
