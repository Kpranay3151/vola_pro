"""TransactionRAGPipeline — Main orchestrator for the financial transaction AI pipeline.

Connects all stages: input validation, context assembly, LLM reasoning,
tool dispatch, guardrails, caching, and response composition.
"""

import time
import json
import pandas as pd
from typing import Any, Dict, List, Optional

from src.cache import UserCacheManager
from src.context import build_context, build_user_df_summary
from src.llm_client import LLMClient
from src.tool_registry import ToolRegistry
from src.guardrails import InputGuardrails, OutputGuardrails, OperationalGuardrails
from src.audit import AuditLogger
from src.utils import format_currency, get_parent_category


class TransactionRAGPipeline:
    """Production-grade AI pipeline for financial transaction analysis.
    
    Usage:
        pipeline = TransactionRAGPipeline(df=transactions_df)
        result = pipeline.run(user_id="usr_a1b2c3d4", prompt="What did I spend the most on?")
    """

    def __init__(self, df: pd.DataFrame, api_key: Optional[str] = None):
        """Initialize the pipeline with a transactions DataFrame.
        
        Args:
            df: DataFrame with columns: user_id, user_name, transaction_date,
                transaction_amount, transaction_category_detail, merchant_name.
            api_key: Optional API key for OpenRouter (falls back to env var).
                     Set GEMINI_API_KEY in .env for Gemini (preferred provider).
        """
        self.df = df.copy()
        self.df["transaction_date"] = pd.to_datetime(self.df["transaction_date"])

        self.cache = UserCacheManager()
        self.audit = AuditLogger()
        self.operational = OperationalGuardrails()

        # LLM client — may fail if no API key; handled gracefully at runtime
        self._llm_client = None
        self._api_key = api_key

    def _get_llm_client(self) -> LLMClient:
        """Lazy-initialize the LLM client with coupled timeout."""
        if self._llm_client is None:
            self._llm_client = LLMClient(
                api_key=self._api_key,
                timeout=self.operational.TIMEOUT_SECONDS,
            )
        return self._llm_client

    def run(self, user_id: str, prompt: str) -> Dict[str, Any]:
        """Execute the full pipeline for a user query.
        
        Args:
            user_id: Unique user identifier.
            prompt: Natural language query.
            
        Returns:
            Structured result dict with keys: user_name, response, data_summary,
            visualizations, cache_hit, latency_ms, guardrail_flags.
        """
        start_time = time.time()
        guardrail_flags = []
        visualizations = []

        # ─── Stage 1: Input Validation & User Data Fetch ───

        # 1a. Check if user exists
        if user_id not in self.df["user_id"].values:
            available = self.df["user_id"].unique().tolist()
            return self._error_response(
                f"User '{user_id}' not found.",
                start_time,
                error_detail={"available_users": available},
            )

        # 1b. Run input guardrails
        input_result = InputGuardrails.run_all(prompt, user_id)
        if not input_result.passed:
            guardrail_flags.extend(input_result.flags)
            return self._guarded_response(
                input_result.message, user_id, start_time, guardrail_flags, prompt
            )

        # Apply any modifications (e.g., truncation)
        if input_result.modified_text:
            prompt = input_result.modified_text
        guardrail_flags.extend(input_result.flags)

        # 1c. Filter DataFrame to user
        user_df = self.df[self.df["user_id"] == user_id].copy()

        # 1d. Cache check / profile computation
        cache_hit = self.cache.has(user_id, "profile")
        if not cache_hit:
            self.cache.compute_and_cache_profile(user_id, user_df)

        profile = self.cache.get_profile(user_id)
        user_name = profile.get("user_name", "Unknown")

        # 1e. Empty-result detection with suggestions (Gap 3)
        if user_df.empty:
            date_range = profile.get("date_range", {})
            return self._empty_result_response(
                user_name, date_range, start_time, guardrail_flags, prompt, user_id
            )

        # ─── Stage 2: Context Assembly ───

        user_df_summary = build_user_df_summary(user_df)
        messages = build_context(prompt, self.cache, user_id, user_df_summary)

        # Operational: token budget check (Gap 1 — apply truncation to messages)
        full_context = " ".join(m["content"] for m in messages)
        full_context, token_flags = self.operational.check_token_budget(full_context)
        guardrail_flags.extend(token_flags)

        if token_flags:
            # Rebuild messages with truncated content — trim the longest message
            messages = self._trim_messages_to_budget(messages)

        # ─── Stage 3: LLM Reasoning + Tool Dispatch ───

        tool_registry = ToolRegistry(user_df, user_id)
        llm_text = ""
        data_summary = self._compute_data_summary(user_df, prompt)

        # Check circuit breaker
        used_fallback = False
        if self.operational.is_circuit_open():
            guardrail_flags.append("CIRCUIT_BREAKER_OPEN")
            llm_text = self._fallback_response(user_df, profile, prompt)
            used_fallback = True
        else:
            try:
                llm_client = self._get_llm_client()
                llm_response = llm_client.chat(
                    messages=messages,
                    tools=tool_registry.get_schemas(),
                )
                self.operational.record_success()

                # R2: Retry once if malformed tool-call JSON was detected
                if llm_response.has_malformed_tool_calls and not llm_response.has_tool_calls:
                    guardrail_flags.append("MALFORMED_TOOL_CALL_RETRY")
                    retry_messages = messages + [{
                        "role": "system",
                        "content": (
                            "Your previous response contained malformed tool-call arguments. "
                            "Please try again. If you want to call a tool, ensure the arguments "
                            "are valid JSON. Otherwise, answer the question in plain text."
                        ),
                    }]
                    try:
                        llm_response = llm_client.chat(
                            messages=retry_messages,
                            tools=tool_registry.get_schemas(),
                        )
                    except Exception:
                        pass  # fall through with original (text-only) response

                llm_text = llm_response.text

                # Execute tool calls
                if llm_response.has_tool_calls:
                    for tc in llm_response.tool_calls:
                        tool_result = tool_registry.dispatch(tc["name"], tc["arguments"])
                        if tool_result["success"] and tool_result.get("chart_path"):
                            visualizations.append(tool_result["chart_path"])
                            # Update viz cache
                            self.cache.update_viz_state(
                                user_id,
                                chart_type=tool_result.get("chart_type", "unknown"),
                                axes=tc["arguments"],
                                filters=tc["arguments"],
                            )

                    # If LLM only made tool calls and no text, provide a summary
                    if not llm_text.strip() and visualizations:
                        llm_text = self._generate_chart_summary(visualizations, profile)

            except (RuntimeError, ValueError) as e:
                # LLM unreachable — graceful degradation
                tripped = self.operational.record_failure()
                guardrail_flags.append("LLM_UNAVAILABLE")
                if tripped:
                    guardrail_flags.append("CIRCUIT_BREAKER_TRIPPED")
                llm_text = self._fallback_response(user_df, profile, prompt)
                used_fallback = True

        # ─── Stage 4: Response Composition ───

        # Output guardrails (skip hallucination check on fallback — numbers are self-computed)
        output_result = OutputGuardrails.run_all(
            llm_text, data_summary, len(user_df),
            skip_hallucination=used_fallback,
        )
        guardrail_flags.extend(output_result.flags)

        if not output_result.passed:
            llm_text = output_result.message

        if output_result.message and output_result.passed:
            llm_text += f"\n\n⚠️ {output_result.message}"

        # Update query history cache (Gap 8 — store descriptive Pandas operations)
        operation_desc = self._infer_pandas_operation(prompt, visualizations)
        result_summary = llm_text[:200] if llm_text else "No response generated"
        self.cache.append_query(user_id, prompt, operation_desc, result_summary)

        latency_ms = (time.time() - start_time) * 1000

        result = {
            "user_name": user_name,
            "response": llm_text,
            "data_summary": data_summary,
            "visualizations": visualizations,
            "cache_hit": cache_hit,
            "latency_ms": round(latency_ms, 2),
            "guardrail_flags": guardrail_flags,
        }

        # Audit log
        self.audit.log(
            user_id=user_id,
            prompt=prompt,
            response_length=len(llm_text),
            latency_ms=latency_ms,
            guardrail_flags=guardrail_flags,
            cache_hit=cache_hit,
            visualizations=visualizations,
        )

        return result

    # ──────────────────────────────────────────────
    # Helper methods
    # ──────────────────────────────────────────────

    def _compute_data_summary(self, user_df: pd.DataFrame, prompt: str) -> Dict:
        """Compute verified data summary for hallucination checking."""
        expenses = user_df[user_df["transaction_amount"] > 0]
        income = user_df[user_df["transaction_amount"] < 0]

        summary = {
            "total_expense": round(float(expenses["transaction_amount"].sum()), 2),
            "total_income": round(float(abs(income["transaction_amount"].sum())), 2),
            "transaction_count": len(user_df),
            "avg_expense": round(float(expenses["transaction_amount"].mean()), 2) if not expenses.empty else 0,
        }

        # Date range for hallucination date verification (Gap 5)
        if not user_df.empty:
            summary["date_range_start"] = str(user_df["transaction_date"].min().date())
            summary["date_range_end"] = str(user_df["transaction_date"].max().date())

        # Add category totals
        if not expenses.empty:
            cats = (
                expenses.assign(cat=expenses["transaction_category_detail"].apply(get_parent_category))
                .groupby("cat")["transaction_amount"].sum()
            )
            for cat, amt in cats.items():
                summary[f"category_{cat}"] = round(float(amt), 2)

        # Monthly totals
        if not expenses.empty:
            monthly = expenses.set_index("transaction_date").resample("ME")["transaction_amount"].sum()
            for date, amt in monthly.items():
                summary[f"month_{date.strftime('%Y_%m')}"] = round(float(amt), 2)

        # Top merchants by spend (for hallucination verification on merchant queries)
        if not expenses.empty and "merchant_name" in expenses.columns:
            top_merchants = (
                expenses.groupby("merchant_name")["transaction_amount"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )
            for merchant, amt in top_merchants.items():
                summary[f"merchant_{merchant}"] = round(float(amt), 2)

        return summary

    def _fallback_response(self, user_df: pd.DataFrame, profile: dict, prompt: str) -> str:
        """Generate a response using only cached data and DataFrame stats when LLM is unavailable."""
        expenses = user_df[user_df["transaction_amount"] > 0]
        income = user_df[user_df["transaction_amount"] < 0]

        total_expense = expenses["transaction_amount"].sum()
        total_income = abs(income["transaction_amount"].sum())
        net = total_income - total_expense

        top_cats = profile.get("top_categories", [])
        cat_text = ", ".join(f"{c['category']} ({format_currency(c['total'])})" for c in top_cats[:3])

        response = (
            f"⚠️ I'm currently experiencing connectivity issues with the AI service, "
            f"but here's what I can tell you from your data:\n\n"
            f"**Financial Overview for {profile.get('user_name', 'you')}**\n"
            f"• Period: {profile['date_range']['start']} to {profile['date_range']['end']}\n"
            f"• Total Income: {format_currency(total_income)}\n"
            f"• Total Expenses: {format_currency(total_expense)}\n"
            f"• Net Savings: {format_currency(net)}\n"
            f"• Top Spending: {cat_text}\n"
            f"• Average Monthly Spend: {format_currency(profile.get('avg_monthly_spend', 0))}\n\n"
            f"Please try again shortly for a more detailed analysis."
        )
        return response

    def _generate_chart_summary(self, chart_paths: List[str], profile: dict) -> str:
        """Generate a brief text summary when the LLM only produced tool calls."""
        n = len(chart_paths)
        charts_text = f"I've generated {n} chart{'s' if n > 1 else ''} for your analysis."
        return f"Here's your financial analysis, {profile.get('user_name', '')}! {charts_text} Check the visualizations below for detailed insights."

    def _trim_messages_to_budget(self, messages: List[Dict]) -> List[Dict]:
        """Trim messages to fit within token budget using priority-based removal.

        Priority order (drop first → last):
          1. Few-shot history (system message with 'PREVIOUS INTERACTIONS')
          2. Data summary portion of user context (truncate)
          3. System prompt (truncate only as last resort)
        The user query is never trimmed (already capped at 500 chars).
        """
        from src.utils import estimate_tokens

        max_tokens = self.operational.MAX_INPUT_TOKENS

        def _total_tokens(msgs):
            return sum(estimate_tokens(m["content"]) for m in msgs)

        trimmed = list(messages)

        # Step 1: Drop few-shot history messages
        if _total_tokens(trimmed) > max_tokens:
            trimmed = [
                m for m in trimmed
                if "PREVIOUS INTERACTIONS" not in m.get("content", "")
            ]

        # Step 2: Truncate data summary (system message with 'DATA SUMMARY')
        if _total_tokens(trimmed) > max_tokens:
            for i, m in enumerate(trimmed):
                if m["role"] == "system" and "DATA SUMMARY" in m.get("content", ""):
                    # Keep the profile part, truncate the data summary
                    content = m["content"]
                    summary_marker = "DATA SUMMARY:"
                    idx = content.find(summary_marker)
                    if idx > 0:
                        # Keep first 500 chars of data summary
                        profile_part = content[:idx + len(summary_marker)]
                        summary_part = content[idx + len(summary_marker):]
                        trimmed[i] = {
                            "role": "system",
                            "content": profile_part + summary_part[:500] + "\n[...truncated for token budget]",
                        }
                    break

        # Step 3: Last resort — truncate system prompt itself
        if _total_tokens(trimmed) > max_tokens:
            for i, m in enumerate(trimmed):
                if m["role"] == "system" and "financial analyst" in m.get("content", "").lower():
                    max_chars = self.operational.MAX_INPUT_TOKENS * 2  # ~half budget
                    trimmed[i] = {
                        "role": "system",
                        "content": m["content"][:max_chars] + "\n[...truncated]",
                    }
                    break

        return trimmed

    def _empty_result_response(
        self, user_name: str, date_range: dict, start_time: float,
        flags: List[str], prompt: str, user_id: str
    ) -> dict:
        """Response when user has no transactions matching the query."""
        start = date_range.get("start", "unknown")
        end = date_range.get("end", "unknown")

        response = (
            f"No transactions found for your query, {user_name}. "
            f"Your transaction history spans {start} to {end}. "
            f"Here are some suggestions:\n"
            f"• Try asking about a different time period (e.g., 'last 3 months')\n"
            f"• Ask for a general overview: 'What did I spend the most on?'\n"
            f"• Check a specific category: 'How much did I spend on food?'"
        )
        flags.append("NO_DATA_FOR_QUERY")

        latency_ms = (time.time() - start_time) * 1000
        self.audit.log(
            user_id=user_id, prompt=prompt, response_length=len(response),
            latency_ms=latency_ms, guardrail_flags=flags,
            cache_hit=False, visualizations=[],
        )

        return {
            "user_name": user_name,
            "response": response,
            "data_summary": {},
            "visualizations": [],
            "cache_hit": False,
            "latency_ms": round(latency_ms, 2),
            "guardrail_flags": flags,
        }

    def _infer_pandas_operation(self, prompt: str, visualizations: List[str]) -> str:
        """Infer a descriptive Pandas operation from the prompt and results."""
        prompt_lower = prompt.lower()

        if visualizations:
            return "df.groupby('category')['transaction_amount'].sum().plot()"
        if any(kw in prompt_lower for kw in ["category", "most", "breakdown", "top"]):
            return "df.groupby(parent_category)['transaction_amount'].sum().sort_values(ascending=False)"
        if any(kw in prompt_lower for kw in ["trend", "month", "over time", "pattern"]):
            return "df.resample('ME', on='transaction_date')['transaction_amount'].sum()"
        if any(kw in prompt_lower for kw in ["income", "expense", "saving", "financial", "net"]):
            return "df.groupby(df['transaction_amount'] > 0)['transaction_amount'].agg(['sum', 'count'])"
        if any(kw in prompt_lower for kw in ["total", "sum", "how much"]):
            return "df['transaction_amount'].sum()"
        return f"df.query(\"user_id == '{{user_id}}'\").describe()"

    def _error_response(self, message: str, start_time: float, error_detail: dict = None) -> dict:
        """Build a structured error response."""
        latency_ms = (time.time() - start_time) * 1000
        result = {
            "error": message,
            "latency_ms": round(latency_ms, 2),
        }
        if error_detail:
            result.update(error_detail)
        return result

    def _guarded_response(
        self, message: str, user_id: str, start_time: float,
        flags: List[str], prompt: str
    ) -> dict:
        """Build a response when input guardrails block the request."""
        latency_ms = (time.time() - start_time) * 1000
        profile = self.cache.get_profile(user_id)
        user_name = profile.get("user_name", "Unknown") if profile else "Unknown"

        self.audit.log(
            user_id=user_id, prompt=prompt, response_length=len(message),
            latency_ms=latency_ms, guardrail_flags=flags,
            cache_hit=False, visualizations=[],
        )

        return {
            "user_name": user_name,
            "response": message,
            "data_summary": {},
            "visualizations": [],
            "cache_hit": False,
            "latency_ms": round(latency_ms, 2),
            "guardrail_flags": flags,
        }
