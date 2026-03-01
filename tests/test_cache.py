"""Tests for the UserCacheManager."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest
from src.cache import UserCacheManager


@pytest.fixture
def sample_df():
    """Create a small test DataFrame."""
    data = {
        "user_id": ["usr_001"] * 6 + ["usr_002"] * 2,
        "user_name": ["Alice"] * 6 + ["Bob"] * 2,
        "transaction_date": pd.to_datetime([
            "2025-01-01", "2025-01-15", "2025-02-01",
            "2025-02-15", "2025-03-01", "2025-03-15",
            "2025-01-10", "2025-02-10",
        ]),
        "transaction_amount": [100, -5000, 200, 50, 1500, -5000, 300, -4000],
        "transaction_category_detail": [
            "RENT_HOUSING", "SALARY_INCOME", "GROCERIES_FOOD",
            "COFFEE_FOOD", "RENT_HOUSING", "SALARY_INCOME",
            "GYM_HEALTH", "SALARY_INCOME",
        ],
        "merchant_name": [
            "AvalonBay", "Employer", "Trader Joe's",
            "Starbucks", "AvalonBay", "Employer",
            "LA Fitness", "Employer",
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def cache():
    return UserCacheManager()


class TestProfileCache:
    def test_compute_and_cache_profile(self, cache, sample_df):
        user_df = sample_df[sample_df["user_id"] == "usr_001"]
        profile = cache.compute_and_cache_profile("usr_001", user_df)

        assert profile["user_name"] == "Alice"
        assert profile["total_transactions"] == 6
        assert profile["date_range"]["start"] == "2025-01-01"
        assert profile["date_range"]["end"] == "2025-03-15"
        assert profile["total_expense"] > 0
        assert profile["total_income"] < 0  # income is stored as negative
        assert len(profile["top_categories"]) > 0

    def test_cache_hit(self, cache, sample_df):
        user_df = sample_df[sample_df["user_id"] == "usr_001"]

        assert not cache.has("usr_001", "profile")
        cache.compute_and_cache_profile("usr_001", user_df)
        assert cache.has("usr_001", "profile")

        profile = cache.get_profile("usr_001")
        assert profile is not None
        assert profile["user_name"] == "Alice"

    def test_empty_user_df(self, cache):
        empty_df = pd.DataFrame(columns=[
            "user_id", "user_name", "transaction_date",
            "transaction_amount", "transaction_category_detail", "merchant_name"
        ])
        profile = cache.compute_and_cache_profile("usr_none", empty_df)
        assert profile["user_name"] == "Unknown"
        assert profile["total_transactions"] == 0

    def test_profile_summary_text(self, cache, sample_df):
        user_df = sample_df[sample_df["user_id"] == "usr_001"]
        cache.compute_and_cache_profile("usr_001", user_df)
        text = cache.get_profile_summary_text("usr_001")
        assert "Alice" in text
        assert "$" in text


class TestQueryHistory:
    def test_append_and_retrieve(self, cache):
        cache.append_query("usr_001", "test prompt", "analyzed", "result summary")
        history = cache.get_query_history("usr_001")
        assert len(history) == 1
        assert history[0][0] == "test prompt"

    def test_history_max_size(self, cache):
        for i in range(10):
            cache.append_query("usr_001", f"prompt_{i}", f"op_{i}", f"result_{i}")
        history = cache.get_query_history("usr_001")
        assert len(history) == UserCacheManager.MAX_QUERY_HISTORY
        # Should have the most recent 5
        assert history[-1][0] == "prompt_9"

    def test_empty_history(self, cache):
        history = cache.get_query_history("nonexistent")
        assert history == []


class TestVizState:
    def test_update_and_retrieve(self, cache):
        cache.update_viz_state("usr_001", "monthly_trend", {"x": "month"}, {"months": 6})
        state = cache.get_viz_state("usr_001")
        assert state["last_chart_type"] == "monthly_trend"
        assert state["axes"]["x"] == "month"

    def test_no_viz_state(self, cache):
        assert cache.get_viz_state("usr_001") is None


class TestCacheClear:
    def test_clear_user(self, cache, sample_df):
        user_df = sample_df[sample_df["user_id"] == "usr_001"]
        cache.compute_and_cache_profile("usr_001", user_df)
        cache.append_query("usr_001", "test", "op", "result")

        cache.clear_user("usr_001")
        assert not cache.has("usr_001", "profile")
        assert cache.get_query_history("usr_001") == []

    def test_clear_all(self, cache, sample_df):
        user_df = sample_df[sample_df["user_id"] == "usr_001"]
        cache.compute_and_cache_profile("usr_001", user_df)
        cache.clear_all()
        assert not cache.has("usr_001", "profile")
