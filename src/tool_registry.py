"""Tool Registry — JSON tool schemas and dispatch logic for LLM tool calling."""

import json
import pandas as pd
from typing import Any, Callable, Dict, List, Optional

from src.visualizations import (
    plot_monthly_spending_trend,
    plot_category_breakdown,
    plot_income_vs_expense,
)


# ════════════════════════════════════════════════
# Tool schema definitions (OpenAI function-calling format)
# ════════════════════════════════════════════════

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "plot_monthly_spending_trend",
            "description": (
                "Generate a line chart showing how the user's spending has changed over time. "
                "Shows monthly totals with a 3-month rolling average overlay. "
                "Use when the user asks about spending trends, patterns over time, or how their "
                "spending has changed. Example prompts: 'Show me my spending trend', "
                "'How has my spending changed?', 'What does my spending look like over time?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user ID (auto-injected, do not set).",
                    },
                    "months": {
                        "type": "integer",
                        "description": "Number of months to look back. Default 1. Use larger values for trend analysis.",
                        "default": 1,
                    },
                    "category_filter": {
                        "type": "string",
                        "description": (
                            "Optional parent category to filter by (e.g., 'FOOD', 'HOUSING', 'TRAVEL', "
                            "'ENTERTAINMENT', 'HEALTH', 'FINANCE', 'TRANSPORT', 'SHOPPING', 'EDUCATION', 'PETS'). "
                            "Leave empty for all categories."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_category_breakdown",
            "description": (
                "Generate a donut chart showing where the user's money is going, with proportional "
                "spending by category. Total spend displayed in the center. "
                "Use when the user asks about spending categories, where money goes, or wants a "
                "breakdown. Example prompts: 'What did I spend the most on?', "
                "'Where is my money going?', 'Show me a category breakdown'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user ID (auto-injected, do not set).",
                    },
                    "period": {
                        "type": "string",
                        "description": "Time window: 'last_1_month', 'last_3_months', 'last_6_months', or 'all'.",
                        "default": "last_3_months",
                        "enum": ["last_1_month", "last_3_months", "last_6_months", "all"],
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top categories to show. Rest grouped as 'Other'.",
                        "default": 7,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_income_vs_expense",
            "description": (
                "Generate a grouped bar chart comparing monthly income (green) vs expenses (red) "
                "with an optional net savings line. Use when the user asks about saving, "
                "financial health, income vs spending, or net balance. "
                "Example prompts: 'Am I saving money?', 'How am I doing financially?', "
                "'Show income vs expenses', 'Am I spending more than I earn?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user ID (auto-injected, do not set).",
                    },
                    "months": {
                        "type": "integer",
                        "description": "Number of months to look back. Default 6.",
                        "default": 6,
                    },
                    "show_net_line": {
                        "type": "boolean",
                        "description": "Whether to overlay a net savings trend line. Default true.",
                        "default": True,
                    },
                },
                "required": [],
            },
        },
    },
]


class ToolRegistry:
    """Registry that maps tool names to executable functions and handles dispatch."""

    def __init__(self, user_df: pd.DataFrame, user_id: str):
        self.user_df = user_df
        self.user_id = user_id

        # Map tool names to handler functions
        self._tools: Dict[str, Callable] = {
            "plot_monthly_spending_trend": self._handle_monthly_trend,
            "plot_category_breakdown": self._handle_category_breakdown,
            "plot_income_vs_expense": self._handle_income_vs_expense,
        }

    def get_schemas(self) -> List[dict]:
        """Return the tool schemas for the LLM."""
        return TOOL_SCHEMAS

    def dispatch(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call and return the result.
        
        Args:
            tool_name: Name of the tool to execute.
            arguments: Dict of arguments from the LLM's tool call.
            
        Returns:
            Dict with 'success', 'chart_path', and 'chart_type' keys.
        """
        handler = self._tools.get(tool_name)
        if not handler:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
                "chart_path": None,
                "chart_type": None,
            }

        try:
            result = handler(**arguments)
            return result
        except Exception as e:
            return {
                "success": False,
                "error": f"Tool execution error: {str(e)}",
                "chart_path": None,
                "chart_type": None,
            }

    def _handle_monthly_trend(self, months: int = 1, category_filter: Optional[str] = None, **kwargs) -> dict:
        chart_path = plot_monthly_spending_trend(
            self.user_df, self.user_id, months=months, category_filter=category_filter
        )
        return {
            "success": True,
            "chart_path": chart_path,
            "chart_type": "monthly_spending_trend",
            "description": f"Monthly spending trend chart generated for the last {months} month(s).",
        }

    def _handle_category_breakdown(self, period: str = "last_3_months", top_n: int = 7, **kwargs) -> dict:
        chart_path = plot_category_breakdown(
            self.user_df, self.user_id, period=period, top_n=top_n
        )
        return {
            "success": True,
            "chart_path": chart_path,
            "chart_type": "category_breakdown",
            "description": f"Category breakdown donut chart generated for {period.replace('_', ' ')}.",
        }

    def _handle_income_vs_expense(self, months: int = 6, show_net_line: bool = True, **kwargs) -> dict:
        chart_path = plot_income_vs_expense(
            self.user_df, self.user_id, months=months, show_net_line=show_net_line
        )
        return {
            "success": True,
            "chart_path": chart_path,
            "chart_type": "income_vs_expense",
            "description": f"Income vs expense chart generated for the last {months} month(s).",
        }
