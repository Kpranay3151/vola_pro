"""Tests for visualization functions."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest
from src.visualizations import (
    plot_monthly_spending_trend,
    plot_category_breakdown,
    plot_income_vs_expense,
)


@pytest.fixture
def user_df():
    """Create a test DataFrame for one user spanning several months."""
    data = {
        "user_id": ["usr_test"] * 12,
        "user_name": ["Test User"] * 12,
        "transaction_date": pd.to_datetime([
            "2025-01-05", "2025-01-15", "2025-01-20",
            "2025-02-01", "2025-02-15", "2025-02-28",
            "2025-03-01", "2025-03-10", "2025-03-20",
            "2025-04-01", "2025-04-15", "2025-04-20",
        ]),
        "transaction_amount": [
            1500, -5000, 100,
            200, 50, -5000,
            1500, 300, 75,
            1500, -5000, 200,
        ],
        "transaction_category_detail": [
            "RENT_HOUSING", "SALARY_INCOME", "GROCERIES_FOOD",
            "GROCERIES_FOOD", "COFFEE_FOOD", "SALARY_INCOME",
            "RENT_HOUSING", "GYM_HEALTH", "STREAMING_ENTERTAINMENT",
            "RENT_HOUSING", "SALARY_INCOME", "GROCERIES_FOOD",
        ],
        "merchant_name": [
            "AvalonBay", "Employer", "Trader Joe's",
            "Whole Foods", "Starbucks", "Employer",
            "AvalonBay", "Equinox", "Netflix",
            "AvalonBay", "Employer", "Safeway",
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture(autouse=True)
def cleanup_output():
    """Ensure output directory exists and clean up after tests."""
    os.makedirs("./output", exist_ok=True)
    yield
    # Optional: clean up generated test files
    # for f in os.listdir("./output"):
    #     if f.startswith("usr_test_"):
    #         os.remove(os.path.join("./output", f))


class TestMonthlySpendingTrend:
    def test_generates_chart(self, user_df):
        path = plot_monthly_spending_trend(user_df, "usr_test", months=6)
        assert os.path.exists(path)
        assert path.endswith(".png")

    def test_with_category_filter(self, user_df):
        path = plot_monthly_spending_trend(user_df, "usr_test", months=6, category_filter="FOOD")
        assert os.path.exists(path)

    def test_empty_data(self):
        empty_df = pd.DataFrame(columns=[
            "user_id", "user_name", "transaction_date",
            "transaction_amount", "transaction_category_detail", "merchant_name"
        ])
        empty_df["transaction_date"] = pd.to_datetime(empty_df["transaction_date"])
        path = plot_monthly_spending_trend(empty_df, "usr_test", months=6)
        assert os.path.exists(path)


class TestCategoryBreakdown:
    def test_generates_chart(self, user_df):
        path = plot_category_breakdown(user_df, "usr_test", period="all")
        assert os.path.exists(path)
        assert path.endswith(".png")

    def test_top_n_limit(self, user_df):
        path = plot_category_breakdown(user_df, "usr_test", period="all", top_n=2)
        assert os.path.exists(path)

    def test_last_month_period(self, user_df):
        path = plot_category_breakdown(user_df, "usr_test", period="last_1_month")
        assert os.path.exists(path)


class TestIncomeVsExpense:
    def test_generates_chart(self, user_df):
        path = plot_income_vs_expense(user_df, "usr_test", months=6)
        assert os.path.exists(path)
        assert path.endswith(".png")

    def test_without_net_line(self, user_df):
        path = plot_income_vs_expense(user_df, "usr_test", months=6, show_net_line=False)
        assert os.path.exists(path)

    def test_single_month(self, user_df):
        path = plot_income_vs_expense(user_df, "usr_test", months=1)
        assert os.path.exists(path)
