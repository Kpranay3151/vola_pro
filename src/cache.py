"""User-specific KV Cache Manager for the Transaction RAG Pipeline.

Stores per-user profiles, query histories, and visualization state in memory.
"""

import pandas as pd
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.utils import get_parent_category, format_currency


class UserCacheManager:
    """In-memory key-value cache scoped per user.
    
    Cache structure:
        user:{id}:profile      → dict with name, date_range, top_categories, avg_monthly_spend
        user:{id}:query_history → deque of (prompt, operation_desc, result_summary) tuples
        user:{id}:viz_state    → dict with last_chart_type, axes, filters
    """

    MAX_QUERY_HISTORY = 5

    def __init__(self):
        self._store: Dict[str, Any] = {}

    # ──────────────────────────────────────────────
    # Generic accessors
    # ──────────────────────────────────────────────

    def _key(self, user_id: str, namespace: str) -> str:
        return f"user:{user_id}:{namespace}"

    def get(self, user_id: str, namespace: str) -> Optional[Any]:
        return self._store.get(self._key(user_id, namespace))

    def set(self, user_id: str, namespace: str, value: Any) -> None:
        self._store[self._key(user_id, namespace)] = value

    def has(self, user_id: str, namespace: str) -> bool:
        return self._key(user_id, namespace) in self._store

    def clear_user(self, user_id: str) -> None:
        """Remove all cached data for a user."""
        keys_to_remove = [k for k in self._store if k.startswith(f"user:{user_id}:")]
        for k in keys_to_remove:
            del self._store[k]

    def clear_all(self) -> None:
        """Clear the entire cache."""
        self._store.clear()

    # ──────────────────────────────────────────────
    # Profile cache
    # ──────────────────────────────────────────────

    def get_profile(self, user_id: str) -> Optional[dict]:
        return self.get(user_id, "profile")

    def compute_and_cache_profile(self, user_id: str, user_df: pd.DataFrame) -> dict:
        """Compute user profile from their transaction data and cache it.
        
        Args:
            user_id: The user identifier.
            user_df: DataFrame filtered to this user's transactions only.
            
        Returns:
            Profile dict with summary statistics.
        """
        if user_df.empty:
            profile = {
                "user_name": "Unknown",
                "date_range": {"start": None, "end": None},
                "total_transactions": 0,
                "top_categories": [],
                "avg_monthly_spend": 0.0,
                "total_income": 0.0,
                "total_expense": 0.0,
            }
            self.set(user_id, "profile", profile)
            return profile

        user_name = user_df["user_name"].iloc[0]
        date_min = user_df["transaction_date"].min()
        date_max = user_df["transaction_date"].max()

        # Expenses are positive amounts, income is negative
        expenses = user_df[user_df["transaction_amount"] > 0]
        income = user_df[user_df["transaction_amount"] < 0]

        # Top categories by total spend
        if not expenses.empty:
            category_spend = (
                expenses.assign(parent_cat=expenses["transaction_category_detail"].apply(get_parent_category))
                .groupby("parent_cat")["transaction_amount"]
                .sum()
                .sort_values(ascending=False)
            )
            top_categories = [
                {"category": cat, "total": round(float(amt), 2)}
                for cat, amt in category_spend.head(5).items()
            ]
        else:
            top_categories = []

        # Average monthly spend
        if not expenses.empty:
            expenses_monthly = expenses.set_index("transaction_date").resample("ME")["transaction_amount"].sum()
            avg_monthly_spend = round(float(expenses_monthly.mean()), 2)
        else:
            avg_monthly_spend = 0.0

        total_income = round(float(income["transaction_amount"].sum()), 2)  # will be negative
        total_expense = round(float(expenses["transaction_amount"].sum()), 2)

        profile = {
            "user_name": user_name,
            "date_range": {
                "start": date_min.strftime("%Y-%m-%d"),
                "end": date_max.strftime("%Y-%m-%d"),
            },
            "total_transactions": len(user_df),
            "top_categories": top_categories,
            "avg_monthly_spend": avg_monthly_spend,
            "total_income": total_income,
            "total_expense": total_expense,
        }

        self.set(user_id, "profile", profile)
        return profile

    # ──────────────────────────────────────────────
    # Query history cache
    # ──────────────────────────────────────────────

    def get_query_history(self, user_id: str) -> List[Tuple[str, str, str]]:
        history = self.get(user_id, "query_history")
        if history is None:
            return []
        return list(history)

    def append_query(self, user_id: str, prompt: str, operation: str, result_summary: str) -> None:
        """Append a query interaction to the user's history (FIFO, max N)."""
        if not self.has(user_id, "query_history"):
            self.set(user_id, "query_history", deque(maxlen=self.MAX_QUERY_HISTORY))
        self.get(user_id, "query_history").append((prompt, operation, result_summary))

    # ──────────────────────────────────────────────
    # Visualization state cache
    # ──────────────────────────────────────────────

    def get_viz_state(self, user_id: str) -> Optional[dict]:
        return self.get(user_id, "viz_state")

    def update_viz_state(self, user_id: str, chart_type: str, axes: dict, filters: dict) -> None:
        """Update the last visualization state for continuity across turns."""
        self.set(user_id, "viz_state", {
            "last_chart_type": chart_type,
            "axes": axes,
            "filters": filters,
            "updated_at": datetime.now().isoformat(),
        })

    # ──────────────────────────────────────────────
    # Profile summary for prompt injection
    # ──────────────────────────────────────────────

    def get_profile_summary_text(self, user_id: str) -> str:
        """Return a human-readable profile summary for LLM context injection."""
        profile = self.get_profile(user_id)
        if not profile:
            return "No cached profile available for this user."

        lines = [
            f"User: {profile['user_name']}",
            f"Data range: {profile['date_range']['start']} to {profile['date_range']['end']}",
            f"Total transactions: {profile['total_transactions']}",
            f"Average monthly spending: {format_currency(profile['avg_monthly_spend'])}",
            f"Total income: {format_currency(abs(profile['total_income']))}",
            f"Total expenses: {format_currency(profile['total_expense'])}",
        ]

        if profile["top_categories"]:
            cats = ", ".join(
                f"{c['category']} ({format_currency(c['total'])})"
                for c in profile["top_categories"]
            )
            lines.append(f"Top spending categories: {cats}")

        return "\n".join(lines)
