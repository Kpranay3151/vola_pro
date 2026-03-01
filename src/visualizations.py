"""Visualization functions for financial transaction charts.

Each function takes user transaction data and produces a styled matplotlib chart
saved as PNG in ./output/. The LLM invokes these via tool calls.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from typing import Optional, List
from datetime import datetime

from src.utils import get_parent_category, filter_by_months, format_currency

# ── Chart styling ─────────────────────────────────────
DARK_BG = "#1a1a2e"
CARD_BG = "#16213e"
ACCENT_1 = "#e94560"
ACCENT_2 = "#0f3460"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2a4a"

COLORS_PALETTE = [
    "#e94560", "#0f3460", "#533483", "#48c9b0", "#f39c12",
    "#3498db", "#e74c3c", "#1abc9c", "#9b59b6", "#2ecc71",
]

OUTPUT_DIR = "./output"


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _apply_dark_style(ax, fig):
    """Apply consistent dark theme styling to a chart."""
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, alpha=0.3, linestyle="--")


def plot_monthly_spending_trend(
    user_df: pd.DataFrame,
    user_id: str,
    months: int = 1,
    category_filter: Optional[str] = None,
) -> str:
    """Line chart showing monthly spending totals with a rolling average overlay.
    
    Args:
        user_df: DataFrame filtered to the target user.
        user_id: User identifier (for filename).
        months: Lookback period in months.
        category_filter: Optional parent category to filter by.
        
    Returns:
        Path to the saved PNG file.
    """
    _ensure_output_dir()

    df = filter_by_months(user_df, months)

    # Only expenses (positive amounts)
    df = df[df["transaction_amount"] > 0].copy()

    if category_filter:
        df = df[df["transaction_category_detail"].apply(get_parent_category) == category_filter.upper()]

    if df.empty:
        # Create a simple "no data" chart
        fig, ax = plt.subplots(figsize=(10, 5))
        _apply_dark_style(ax, fig)
        ax.text(0.5, 0.5, "No transaction data for this period",
                ha="center", va="center", fontsize=14, color=TEXT_COLOR,
                transform=ax.transAxes)
        ax.set_title("Monthly Spending Trend", fontsize=14, fontweight="bold", pad=15)
        filename = f"{user_id}_trend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        return filepath

    # Aggregate by month
    monthly = df.set_index("transaction_date").resample("ME")["transaction_amount"].sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_style(ax, fig)

    # Main line
    x_labels = monthly.index.strftime("%b %Y")
    ax.plot(x_labels, monthly.values, color=ACCENT_1, marker="o",
            linewidth=2.5, markersize=8, label="Monthly Total", zorder=3)

    # Rolling average (3-month) if enough data
    if len(monthly) >= 3:
        rolling = monthly.rolling(window=3, min_periods=1).mean()
        ax.plot(x_labels, rolling.values, color="#48c9b0",
                linewidth=2, linestyle="--", alpha=0.8, label="3-Month Avg", zorder=2)

    # Annotations on data points
    for i, (label, val) in enumerate(zip(x_labels, monthly.values)):
        ax.annotate(format_currency(val), (label, val),
                    textcoords="offset points", xytext=(0, 12),
                    fontsize=8, color=TEXT_COLOR, ha="center")

    title = "Monthly Spending Trend"
    if category_filter:
        title += f" — {category_filter.title()}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Amount ($)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    plt.xticks(rotation=45, ha="right")

    filename = f"{user_id}_trend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return filepath


def plot_category_breakdown(
    user_df: pd.DataFrame,
    user_id: str,
    period: str = "last_3_months",
    top_n: int = 7,
) -> str:
    """Donut chart showing proportional spending by category.
    
    Args:
        user_df: DataFrame filtered to the target user.
        user_id: User identifier (for filename).
        period: Time window — 'last_1_month', 'last_3_months', 'last_6_months', 'all'.
        top_n: Number of top categories; rest grouped as 'Other'.
        
    Returns:
        Path to the saved PNG file.
    """
    _ensure_output_dir()

    # Parse period
    period_map = {
        "last_1_month": 1, "last_3_months": 3,
        "last_6_months": 6, "all": 999,
    }
    months = period_map.get(period, 3)
    df = filter_by_months(user_df, months)

    # Only expenses
    df = df[df["transaction_amount"] > 0].copy()

    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 8))
        _apply_dark_style(ax, fig)
        ax.text(0.5, 0.5, "No spending data for this period",
                ha="center", va="center", fontsize=14, color=TEXT_COLOR,
                transform=ax.transAxes)
        filename = f"{user_id}_categories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        return filepath

    # Category aggregation
    df["parent_category"] = df["transaction_category_detail"].apply(get_parent_category)
    cat_totals = df.groupby("parent_category")["transaction_amount"].sum().sort_values(ascending=False)

    # Group beyond top_n into "Other"
    if len(cat_totals) > top_n:
        top = cat_totals.head(top_n)
        other = pd.Series({"Other": cat_totals.iloc[top_n:].sum()})
        cat_totals = pd.concat([top, other])

    total_spend = cat_totals.sum()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(DARK_BG)

    colors = COLORS_PALETTE[:len(cat_totals)]
    percentages = cat_totals.values / total_spend * 100

    # Draw donut without inline labels (avoids overlap on small slices)
    wedges, _ = ax.pie(
        cat_totals.values,
        labels=[""] * len(cat_totals),  # no inline labels
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.4, edgecolor=DARK_BG, linewidth=2),
    )

    # Only annotate percentage on slices large enough (>= 5%)
    for i, (wedge, pct, val) in enumerate(zip(wedges, percentages, cat_totals.values)):
        if pct >= 5:
            ang = (wedge.theta2 + wedge.theta1) / 2
            x = 0.78 * np.cos(np.radians(ang))
            y = 0.78 * np.sin(np.radians(ang))
            ax.text(x, y, f"{pct:.1f}%\n{format_currency(val)}",
                    ha="center", va="center", fontsize=8, color=TEXT_COLOR)

    # Legend with category name, percentage, and amount
    legend_labels = [
        f"{name}  ({pct:.1f}%, {format_currency(val)})"
        for name, pct, val in zip(cat_totals.index, percentages, cat_totals.values)
    ]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
              fontsize=9, facecolor=CARD_BG, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, framealpha=0.9)

    # Center text
    ax.text(0, 0, f"Total\n{format_currency(total_spend)}",
            ha="center", va="center", fontsize=16, fontweight="bold", color=TEXT_COLOR)

    ax.set_title("Spending by Category", fontsize=14, fontweight="bold",
                 color=TEXT_COLOR, pad=20)

    filename = f"{user_id}_categories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return filepath


def plot_income_vs_expense(
    user_df: pd.DataFrame,
    user_id: str,
    months: int = 6,
    show_net_line: bool = True,
) -> str:
    """Grouped bar chart comparing monthly income (green) vs expenses (red),
    with an optional net savings line overlay.
    
    Args:
        user_df: DataFrame filtered to the target user.
        user_id: User identifier (for filename).
        months: Lookback period.
        show_net_line: Whether to overlay the net savings line.
        
    Returns:
        Path to the saved PNG file.
    """
    _ensure_output_dir()

    df = filter_by_months(user_df, months)

    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        _apply_dark_style(ax, fig)
        ax.text(0.5, 0.5, "No transaction data for this period",
                ha="center", va="center", fontsize=14, color=TEXT_COLOR,
                transform=ax.transAxes)
        filename = f"{user_id}_income_expense_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        return filepath

    # Separate income (negative amounts) and expenses (positive amounts)
    df_monthly = df.set_index("transaction_date").resample("ME").apply(
        lambda g: pd.Series({
            "income": abs(g[g["transaction_amount"] < 0]["transaction_amount"].sum()),
            "expense": g[g["transaction_amount"] > 0]["transaction_amount"].sum(),
        })
    )

    x_labels = df_monthly.index.strftime("%b %Y")
    x = np.arange(len(x_labels))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_dark_style(ax, fig)

    bars_income = ax.bar(x - bar_width / 2, df_monthly["income"], bar_width,
                         label="Income", color="#2ecc71", alpha=0.85, zorder=3)
    bars_expense = ax.bar(x + bar_width / 2, df_monthly["expense"], bar_width,
                          label="Expense", color=ACCENT_1, alpha=0.85, zorder=3)

    # Value labels on bars
    for bar in bars_income:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f"${height:,.0f}", (bar.get_x() + bar.get_width() / 2, height),
                        textcoords="offset points", xytext=(0, 5),
                        fontsize=7, color="#2ecc71", ha="center")

    for bar in bars_expense:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f"${height:,.0f}", (bar.get_x() + bar.get_width() / 2, height),
                        textcoords="offset points", xytext=(0, 5),
                        fontsize=7, color=ACCENT_1, ha="center")

    # Net savings line
    if show_net_line:
        net = df_monthly["income"] - df_monthly["expense"]
        ax2 = ax.twinx()
        ax2.plot(x, net.values, color="#f39c12", marker="D", linewidth=2,
                 markersize=6, label="Net Savings", zorder=4)
        ax2.set_ylabel("Net Savings ($)", fontsize=11, color="#f39c12")
        ax2.tick_params(colors="#f39c12", labelsize=9)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax2.spines["right"].set_color("#f39c12")

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc="upper left", facecolor=CARD_BG, edgecolor=GRID_COLOR,
                  labelcolor=TEXT_COLOR)
    else:
        ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("Amount ($)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_title("Income vs. Expenses", fontsize=14, fontweight="bold", pad=15)

    filename = f"{user_id}_income_expense_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return filepath
