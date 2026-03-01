"""Utility helpers for data loading, category parsing, and date handling."""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


def load_transaction_data(csv_path: str) -> pd.DataFrame:
    """Load and validate the transactions CSV into a DataFrame.
    
    Args:
        csv_path: Path to the CSV file.
        
    Returns:
        DataFrame with parsed dates and validated schema.
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
        ValueError: If required columns are missing.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Transaction data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = [
        "user_id", "user_name", "transaction_date",
        "transaction_amount", "transaction_category_detail", "merchant_name"
    ]
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse dates
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    return df


def parse_category(category_detail: str) -> dict:
    """Parse hierarchical category string like 'SUBCATEGORY_PARENT' into components.
    
    Examples:
        'RENT_HOUSING' → {'subcategory': 'RENT', 'parent': 'HOUSING'}
        'COFFEE_FOOD' → {'subcategory': 'COFFEE', 'parent': 'FOOD'}
        'STREAMING_ENTERTAINMENT' → {'subcategory': 'STREAMING', 'parent': 'ENTERTAINMENT'}
    """
    parts = category_detail.rsplit("_", 1)
    if len(parts) == 2:
        return {"subcategory": parts[0], "parent": parts[1]}
    return {"subcategory": category_detail, "parent": "OTHER"}


def get_parent_category(category_detail: str) -> str:
    """Extract the parent category from a detail string."""
    return parse_category(category_detail)["parent"]


def get_subcategory(category_detail: str) -> str:
    """Extract the subcategory from a detail string."""
    return parse_category(category_detail)["subcategory"]


def filter_by_months(df: pd.DataFrame, months: int, reference_date: Optional[datetime] = None) -> pd.DataFrame:
    """Filter DataFrame to rows within the last N months from reference date.
    
    Args:
        df: DataFrame with 'transaction_date' column.
        months: Number of months to look back.
        reference_date: The date to count back from (defaults to max date in df).
    """
    if reference_date is None:
        reference_date = df["transaction_date"].max()

    cutoff = reference_date - pd.DateOffset(months=months)
    return df[df["transaction_date"] >= cutoff].copy()


def format_currency(amount: float) -> str:
    """Format a number as USD currency string."""
    if amount < 0:
        return f"-${abs(amount):,.2f}"
    return f"${amount:,.2f}"


def hash_user_id(user_id: str) -> str:
    """Hash a user_id for audit logging (no raw PII)."""
    import hashlib
    return hashlib.sha256(user_id.encode()).hexdigest()[:12]


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars / 4 heuristic)."""
    return max(1, len(text) // 4)
