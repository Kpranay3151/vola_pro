"""LLM Guardrails — Input, Output, and Operational safety checks.

Protects against prompt injection, data leakage, hallucination, and misuse.
"""

import re
import time
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from src.utils import estimate_tokens


# ════════════════════════════════════════════════
# Input Guardrails
# ════════════════════════════════════════════════

# Patterns that indicate prompt injection attempts
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)?\s*(instructions?|prompts?|rules?|guidelines?)",
    r"(reveal|show|display|output|print|repeat)\s+(\w+\s+)*(the\s+)?(system\s+)?(prompt|instructions?|rules?)",
    r"you\s+are\s+now\s+(a|an|my)\s+",
    r"act\s+as\s+(a|an|if)\s+",
    r"forget\s+(everything|all|your)\s+",
    r"override\s+(your|the|all)\s+",
    r"disregard\s+(your|the|all|previous)\s+",
    r"new\s+instructions?\s*:",
    r"system\s*:\s*",
    r"<\s*system\s*>",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"bypass\s+(the\s+)?(filter|guard|safety|restriction)",
    r"pretend\s+(you\s+)?(are|to\s+be)\s+",
]

# Financial / transaction related keywords for scope enforcement
FINANCIAL_KEYWORDS = [
    "spend", "spent", "spending", "expense", "expenses", "income", "salary",
    "saving", "savings", "budget", "money", "financial", "transaction",
    "category", "merchant", "payment", "cost", "charge", "bill",
    "trend", "month", "monthly", "weekly", "daily", "report", "summary",
    "total", "average", "breakdown", "chart", "graph", "plot", "show",
    "how much", "where", "what", "when", "top", "most", "least",
    "food", "rent", "housing", "travel", "insurance", "gym", "health",
    "transport", "entertainment", "shopping", "groceries", "restaurant",
    "subscription", "streaming", "fuel", "utilities", "internet",
    "refund", "cashback", "freelance",
]

# Off-topic indicators
OFF_TOPIC_PATTERNS = [
    r"(write|compose|create)\s+(\w+\s+)*(poem|song|story|essay|code|script)",
    r"(what|who)\s+is\s+the\s+(president|capital|population)",
    r"(translate|convert)\s+.+\s+(to|into)\s+(french|spanish|german|japanese)",
    r"(tell|say)\s+(me\s+)?a\s+joke",
    r"(how|what)\s+(do|does|is|are)\s+.*(weather|news|sports|politics|recipe|cook)",
    r"(play|sing|dance|draw|paint)\s+",
]

# Maximum prompt length (characters)
MAX_PROMPT_LENGTH = 500

# Toxicity keyword blocklist
TOXICITY_KEYWORDS = [
    "fuck", "shit", "damn", "bitch", "ass", "bastard", "crap",
    "idiot", "stupid", "moron", "dumb", "retard", "kill", "die",
    "hate", "racist", "sexist",
]

# Hedging phrases indicating low confidence
HEDGING_PHRASES = [
    "i'm not sure", "i am not sure", "i think", "might be", "could be",
    "it seems", "not certain", "unclear",
    "hard to say", "difficult to determine", "insufficient data",
    "not enough information", "cannot determine",
    # "perhaps" and "possibly" removed — commonly used as polite suggestions
    # ("or perhaps a trend?"), not genuine uncertainty — caused false positives.
]


class GuardrailResult:
    """Result of a guardrail check."""

    def __init__(self, passed: bool, flags: List[str] = None, modified_text: str = None, message: str = None):
        self.passed = passed
        self.flags = flags or []
        self.modified_text = modified_text
        self.message = message

    def __repr__(self):
        return f"GuardrailResult(passed={self.passed}, flags={self.flags})"


class InputGuardrails:
    """Checks applied to user prompts before they reach the LLM."""

    @staticmethod
    def check_prompt_injection(prompt: str) -> GuardrailResult:
        """Detect and reject prompt injection attempts."""
        prompt_lower = prompt.lower().strip()

        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, prompt_lower):
                return GuardrailResult(
                    passed=False,
                    flags=["PROMPT_INJECTION_DETECTED"],
                    message=(
                        "I appreciate your curiosity, but I can only help with questions about "
                        "your financial transactions. I'm unable to modify my instructions or "
                        "share system details. Please ask me about your spending, income, or "
                        "financial trends instead!"
                    ),
                )

        return GuardrailResult(passed=True)

    @staticmethod
    def check_scope(prompt: str) -> GuardrailResult:
        """Reject prompts that are clearly off-topic (not financial)."""
        prompt_lower = prompt.lower().strip()

        # Check if prompt matches off-topic patterns
        is_off_topic = any(re.search(p, prompt_lower) for p in OFF_TOPIC_PATTERNS)

        # Check if prompt has any financial relevance
        has_financial_context = any(kw in prompt_lower for kw in FINANCIAL_KEYWORDS)

        if is_off_topic and not has_financial_context:
            return GuardrailResult(
                passed=False,
                flags=["OFF_TOPIC"],
                message=(
                    "I'm your financial transaction assistant and can only help with questions "
                    "about your spending, income, savings, and transaction history. "
                    "Try asking something like 'What did I spend the most on last month?' or "
                    "'Show me my spending trend.'"
                ),
            )

        return GuardrailResult(passed=True)

    @staticmethod
    def check_cross_user(prompt: str, current_user_id: str) -> GuardrailResult:
        """Detect attempts to access another user's data."""
        prompt_lower = prompt.lower()

        # Look for user ID patterns that don't match the current user
        user_id_pattern = r"usr_[a-z0-9]+"
        mentioned_ids = re.findall(user_id_pattern, prompt_lower)

        for uid in mentioned_ids:
            if uid != current_user_id.lower():
                return GuardrailResult(
                    passed=False,
                    flags=["CROSS_USER_LEAKAGE_ATTEMPT"],
                    message=(
                        "For privacy and security, I can only provide information about your own "
                        "financial transactions. I'm unable to share details about other users' data."
                    ),
                )

        # Also check for generic patterns like "user xyz's" or "another user"
        cross_user_patterns = [
            r"(another|other|different)\s+user",
            r"user[_\s]?\w+('s|s)\s+(spending|data|transaction|income|expense)",
            r"(show|tell|give)\s+me\s+.*(someone|somebody)\s*(else)?",
        ]
        for pattern in cross_user_patterns:
            if re.search(pattern, prompt_lower):
                return GuardrailResult(
                    passed=False,
                    flags=["CROSS_USER_LEAKAGE_ATTEMPT"],
                    message=(
                        "For privacy and security, I can only provide information about your own "
                        "financial transactions. I'm unable to share details about other users' data."
                    ),
                )

        return GuardrailResult(passed=True)

    @staticmethod
    def check_length(prompt: str) -> GuardrailResult:
        """Enforce maximum prompt length with graceful truncation."""
        if len(prompt) > MAX_PROMPT_LENGTH:
            truncated = prompt[:MAX_PROMPT_LENGTH]
            return GuardrailResult(
                passed=True,  # still passes, but truncated
                flags=["INPUT_TRUNCATED"],
                modified_text=truncated,
                message=f"Your prompt was truncated to {MAX_PROMPT_LENGTH} characters.",
            )
        return GuardrailResult(passed=True)

    @classmethod
    def run_all(cls, prompt: str, current_user_id: str) -> GuardrailResult:
        """Run all input guardrails in sequence. Returns first failure or pass."""
        # 1. Prompt injection
        result = cls.check_prompt_injection(prompt)
        if not result.passed:
            return result

        # 2. Cross-user leakage
        result = cls.check_cross_user(prompt, current_user_id)
        if not result.passed:
            return result

        # 3. Scope enforcement
        result = cls.check_scope(prompt)
        if not result.passed:
            return result

        # 4. Length check (applies truncation but doesn't block)
        result = cls.check_length(prompt)
        return result


class OutputGuardrails:
    """Checks applied to LLM responses before returning to the user."""

    @staticmethod
    def check_hallucination(response_text: str, actual_data: Dict) -> GuardrailResult:
        """Cross-reference numbers AND dates in LLM response against actual data.

        Args:
            response_text: The LLM's text response.
            actual_data: Dict of verified numbers from DataFrame computation.
                         Keys are descriptive labels, values are numbers.
                         Should include 'date_range_start' and 'date_range_end' keys.
        """
        flags = []

        # ── Numeric verification ──
        dollar_amounts = re.findall(r'\$[\d,]+\.?\d*', response_text)
        plain_numbers = re.findall(r'(?<!\w)(\d{2,}(?:,\d{3})*(?:\.\d+)?)(?!\w)', response_text)

        extracted_numbers = []
        for amt in dollar_amounts:
            try:
                num = float(amt.replace("$", "").replace(",", ""))
                extracted_numbers.append(num)
            except ValueError:
                continue

        for num_str in plain_numbers:
            try:
                num = float(num_str.replace(",", ""))
                if num > 1:  # skip trivial numbers
                    extracted_numbers.append(num)
            except ValueError:
                continue

        if extracted_numbers and actual_data:
            actual_values = list(actual_data.values())
            for num in extracted_numbers:
                is_grounded = any(
                    abs(num - actual_val) / max(abs(actual_val), 1) < 0.05
                    for actual_val in actual_values
                    if isinstance(actual_val, (int, float))
                )
                if not is_grounded:
                    flags.append(f"UNGROUNDED_NUMBER:{num}")

        # ── Date verification (Gap 5) ──
        date_start = actual_data.get("date_range_start")
        date_end = actual_data.get("date_range_end")
        if date_start and date_end:
            MONTH_NAMES = {
                "january": 1, "february": 2, "march": 3, "april": 4,
                "may": 5, "june": 6, "july": 7, "august": 8,
                "september": 9, "october": 10, "november": 11, "december": 12,
                "jan": 1, "feb": 2, "mar": 3, "apr": 4,
                "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
            }
            text_lower = response_text.lower()
            # Match patterns like "March 2025", "Jan 2025"
            month_year_matches = re.findall(
                r'(january|february|march|april|may|june|july|august|september|october|november|december|'
                r'jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+(\d{4})', text_lower
            )
            for month_name, year_str in month_year_matches:
                try:
                    month_num = MONTH_NAMES.get(month_name)
                    year = int(year_str)
                    if month_num:
                        # Check if this month/year falls within actual data range
                        from datetime import datetime as dt
                        mentioned_start = dt(year, month_num, 1)
                        actual_start = pd.Timestamp(date_start).to_pydatetime()
                        actual_end = pd.Timestamp(date_end).to_pydatetime()
                        if mentioned_start < actual_start.replace(day=1) or mentioned_start > actual_end:
                            flags.append(f"UNGROUNDED_DATE:{month_name.title()} {year_str}")
                except (ValueError, TypeError):
                    continue

        if flags:
            return GuardrailResult(
                passed=True,  # don't block, but flag
                flags=["POTENTIAL_HALLUCINATION"] + flags[:3],
                message="Note: Some figures or dates in the response could not be verified against your actual data.",
            )

        return GuardrailResult(passed=True)

    @staticmethod
    def check_toxicity(response_text: str) -> GuardrailResult:
        """Simple keyword-based toxicity filter."""
        text_lower = response_text.lower()
        found = [kw for kw in TOXICITY_KEYWORDS if kw in text_lower]

        if found:
            return GuardrailResult(
                passed=False,
                flags=["TOXIC_CONTENT_DETECTED"],
                message="The response contained inappropriate content and has been filtered.",
            )

        return GuardrailResult(passed=True)

    @staticmethod
    def check_confidence(response_text: str, data_row_count: int) -> GuardrailResult:
        """Add a disclaimer if the LLM expresses uncertainty or data is sparse."""
        text_lower = response_text.lower()
        is_hedging = any(phrase in text_lower for phrase in HEDGING_PHRASES)
        is_sparse = data_row_count < 5

        if is_hedging or is_sparse:
            disclaimer = ""
            flags = []

            if is_hedging:
                flags.append("LOW_CONFIDENCE")
                disclaimer = " The analysis expresses some uncertainty."

            if is_sparse:
                flags.append("SPARSE_DATA")
                disclaimer += f" Note: This analysis is based on only {data_row_count} transaction(s), which may limit accuracy."

            return GuardrailResult(
                passed=True,
                flags=flags,
                message=disclaimer.strip(),
            )

        return GuardrailResult(passed=True)

    @classmethod
    def run_all(cls, response_text: str, actual_data: Dict, data_row_count: int, skip_hallucination: bool = False) -> GuardrailResult:
        """Run all output guardrails. Aggregates flags."""
        all_flags = []
        messages = []

        # 1. Toxicity
        result = cls.check_toxicity(response_text)
        if not result.passed:
            return result  # hard block
        all_flags.extend(result.flags)

        # 2. Hallucination (skip for fallback responses — numbers are self-computed)
        if not skip_hallucination:
            result = cls.check_hallucination(response_text, actual_data)
            all_flags.extend(result.flags)
            if result.message:
                messages.append(result.message)

        # 3. Confidence
        result = cls.check_confidence(response_text, data_row_count)
        all_flags.extend(result.flags)
        if result.message:
            messages.append(result.message)

        return GuardrailResult(
            passed=True,
            flags=all_flags,
            message=" ".join(messages) if messages else None,
        )


class OperationalGuardrails:
    """Token budgets, timeouts, and circuit breaker logic."""

    MAX_INPUT_TOKENS = 6000
    MAX_OUTPUT_TOKENS = 2000
    TIMEOUT_SECONDS = 30
    CIRCUIT_BREAKER_THRESHOLD = 3
    CIRCUIT_BREAKER_COOLDOWN = 60  # seconds

    def __init__(self):
        self._consecutive_failures = 0
        self._circuit_open_until: Optional[datetime] = None

    def check_token_budget(self, context_text: str) -> Tuple[str, List[str]]:
        """Truncate context if it exceeds the token budget.
        
        Returns:
            Tuple of (possibly_truncated_text, flags_list).
        """
        tokens = estimate_tokens(context_text)
        flags = []

        if tokens > self.MAX_INPUT_TOKENS:
            # Truncate to fit budget (approximate)
            max_chars = self.MAX_INPUT_TOKENS * 4
            context_text = context_text[:max_chars]
            flags.append(f"CONTEXT_TRUNCATED_FROM_{tokens}_TOKENS")

        return context_text, flags

    def is_circuit_open(self) -> bool:
        """Check if the circuit breaker is tripped."""
        if self._circuit_open_until and datetime.now() < self._circuit_open_until:
            return True
        if self._circuit_open_until and datetime.now() >= self._circuit_open_until:
            # Reset after cooldown
            self._circuit_open_until = None
            self._consecutive_failures = 0
        return False

    def record_success(self) -> None:
        """Record a successful LLM call, resetting the failure counter."""
        self._consecutive_failures = 0

    def record_failure(self) -> bool:
        """Record a failed LLM call. Returns True if circuit is now tripped."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.CIRCUIT_BREAKER_THRESHOLD:
            self._circuit_open_until = datetime.now() + timedelta(seconds=self.CIRCUIT_BREAKER_COOLDOWN)
            return True
        return False
