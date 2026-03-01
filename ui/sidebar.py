"""Sidebar — user selector, profile card, and example query buttons."""

import streamlit as st
import pandas as pd

from ui.config import EXAMPLE_QUERIES
from ui.session import clear_chat_history


def _format_currency(amount: float) -> str:
    """Format a number as currency string."""
    return f"${abs(amount):,.2f}"


def render_sidebar(pipeline, df: pd.DataFrame):
    """Render the full sidebar: user picker, profile, example queries."""
    with st.sidebar:
        st.title("Vola Pro")
        st.caption("Financial Transaction Analysis")
        st.divider()

        # ── User selection ──────────────────────────────
        st.subheader("Select User")

        # Build user options dynamically from the DataFrame
        user_roster = (
            df[["user_id", "user_name"]]
            .drop_duplicates()
            .sort_values("user_name")
        )
        user_map = {
            f"{row.user_name} ({row.user_id})": row.user_id
            for row in user_roster.itertuples()
        }

        # Default to first user if session hasn't set one yet
        if st.session_state["current_user_id"] is None:
            st.session_state["current_user_id"] = user_roster.iloc[0]["user_id"]

        current = st.session_state["current_user_id"]
        current_label = next(
            (label for label, uid in user_map.items() if uid == current),
            list(user_map.keys())[0],
        )

        selected_label = st.selectbox(
            "Active User",
            options=list(user_map.keys()),
            index=list(user_map.keys()).index(current_label),
            label_visibility="collapsed",
        )
        new_user_id = user_map[selected_label]

        if new_user_id != current:
            st.session_state["current_user_id"] = new_user_id
            st.rerun()

        # ── Profile card ────────────────────────────────
        st.divider()
        st.subheader("User Profile")
        _render_profile(pipeline, new_user_id, df)

        # ── Example queries ─────────────────────────────
        st.divider()
        st.subheader("Try These Queries")
        _render_example_queries()

        # ── Actions ─────────────────────────────────────
        st.divider()
        if st.button("Clear Conversation", use_container_width=True):
            clear_chat_history(st.session_state["current_user_id"])
            st.rerun()


def _render_profile(pipeline, user_id: str, df: pd.DataFrame):
    """Show the user's computed financial profile."""
    # Ensure profile is cached
    if not pipeline.cache.has(user_id, "profile"):
        user_df = df[df["user_id"] == user_id].copy()
        pipeline.cache.compute_and_cache_profile(user_id, user_df)

    profile = pipeline.cache.get_profile(user_id)
    if not profile:
        st.warning("No profile data available.")
        return

    st.metric("Total Transactions", profile["total_transactions"])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Income", _format_currency(profile["total_income"]))
    with col2:
        st.metric("Expenses", _format_currency(profile["total_expense"]))

    st.metric("Avg Monthly Spend", _format_currency(profile["avg_monthly_spend"]))

    date_range = profile.get("date_range", {})
    st.caption(f"Data: {date_range.get('start', '?')} to {date_range.get('end', '?')}")

    if profile.get("top_categories"):
        st.markdown("**Top Categories:**")
        for cat in profile["top_categories"][:5]:
            st.markdown(f"- {cat['category']}: {_format_currency(cat['total'])}")


def _render_example_queries():
    """Render grouped example query buttons."""
    for category, queries in EXAMPLE_QUERIES.items():
        is_first = category == list(EXAMPLE_QUERIES.keys())[0]
        with st.expander(category, expanded=is_first):
            for query in queries:
                if st.button(query, key=f"eq_{hash(query)}", use_container_width=True):
                    st.session_state["pending_query"] = query
                    st.rerun()
