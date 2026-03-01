"""LLM Client — Multi-provider HTTP client with retry, model fallback, and tool calling support.

Supports:
  1. Google Gemini (via OpenAI-compatible endpoint) — primary if GEMINI_API_KEY is set
  2. OpenRouter (multi-model free tier) — fallback or standalone if only OPENROUTER_API_KEY is set
"""

import os
import json
import time
import requests
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Provider configurations ──────────────────────────────────────

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

OPENROUTER_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-4b:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
]
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Retry config
MAX_RETRIES = 3
BACKOFF_BASE = 4  # seconds (gives rate-limited free tiers time to recover)


class LLMResponse:
    """Structured response from the LLM."""

    def __init__(
        self,
        text: str = "",
        tool_calls: List[Dict] = None,
        model_used: str = "",
        usage: Dict = None,
        raw: Dict = None,
        malformed_tool_call_count: int = 0,
    ):
        self.text = text
        self.tool_calls = tool_calls or []
        self.model_used = model_used
        self.usage = usage or {}
        self.raw = raw or {}
        self.malformed_tool_call_count = malformed_tool_call_count

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def has_malformed_tool_calls(self) -> bool:
        return self.malformed_tool_call_count > 0


class LLMClient:
    """Multi-provider LLM client (Gemini primary, OpenRouter fallback).

    Features:
        - Automatic retry with exponential backoff
        - Provider + model fallback chain
        - Tool/function calling support
        - Configurable timeout
    """

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self.timeout = timeout

        # Build provider chain: Gemini first (if available), then OpenRouter
        self._providers: List[Dict] = []

        gemini_key = os.getenv("GEMINI_API_KEY", "")
        openrouter_key = api_key or os.getenv("OPENROUTER_API_KEY", "")

        if gemini_key:
            self._providers.append({
                "name": "Gemini",
                "api_key": gemini_key,
                "api_url": GEMINI_API_URL,
                "models": list(GEMINI_MODELS),
                "headers_extra": {},
            })

        if openrouter_key:
            self._providers.append({
                "name": "OpenRouter",
                "api_key": openrouter_key,
                "api_url": OPENROUTER_API_URL,
                "models": list(OPENROUTER_MODELS),
                "headers_extra": {
                    "HTTP-Referer": "https://github.com/vola-pro-pipeline",
                    "X-Title": "Transaction RAG Pipeline",
                },
            })

        if not self._providers:
            raise ValueError(
                "No LLM API key found. Set GEMINI_API_KEY or OPENROUTER_API_KEY "
                "in your .env file."
            )

        provider_names = [p["name"] for p in self._providers]
        logger.info(f"LLM providers configured: {provider_names}")

    def chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        """Send a chat completion request with optional tool schemas.

        Tries each provider + model in the fallback chain on failure.

        Args:
            messages: Chat messages in OpenAI format.
            tools: Optional tool schemas for function calling.
            model: Override model (skips fallback chain).
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.

        Returns:
            LLMResponse with text and/or tool calls.

        Raises:
            RuntimeError: If all providers, models, and retries are exhausted.
        """
        last_error = None

        for provider in self._providers:
            models_to_try = [model] if model else provider["models"]
            provider_name = provider["name"]

            for model_name in models_to_try:
                for attempt in range(MAX_RETRIES):
                    try:
                        response = self._make_request(
                            provider, model_name, messages, tools, temperature, max_tokens
                        )
                        return self._parse_response(response, model_name)
                    except requests.HTTPError as e:
                        last_error = e
                        # Smart 429 handling: parse wait time from response
                        wait = self._get_retry_wait(e, attempt)
                        print(f"  ⚠ LLM call failed ({provider_name}/{model_name}, attempt={attempt+1}/{MAX_RETRIES}): {e}")
                        print(f"    Retrying in {wait}s...")
                        time.sleep(wait)
                    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
                        last_error = e
                        wait = BACKOFF_BASE * (2 ** attempt)
                        print(f"  ⚠ LLM call failed ({provider_name}/{model_name}, attempt={attempt+1}/{MAX_RETRIES}): {e}")
                        print(f"    Retrying in {wait}s...")
                        time.sleep(wait)

                print(f"  ✗ {provider_name}/{model_name} exhausted all retries, trying next...")

            print(f"  ✗ Provider {provider_name} exhausted, trying next provider...")

        raise RuntimeError(
            f"All LLM providers and models exhausted after retries. Last error: {last_error}"
        )

    def _make_request(
        self,
        provider: Dict,
        model: str,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        temperature: float,
        max_tokens: int,
    ) -> Dict:
        """Make a single HTTP request to the provider's API."""
        headers = {
            "Authorization": f"Bearer {provider['api_key']}",
            "Content-Type": "application/json",
            **provider["headers_extra"],
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        response = requests.post(
            provider["api_url"],
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()
        return response.json()

    @staticmethod
    def _get_retry_wait(error: requests.HTTPError, attempt: int) -> int:
        """Extract wait time from 429 response, or use exponential backoff."""
        import re
        if hasattr(error, 'response') and error.response is not None:
            status = error.response.status_code
            if status == 429:
                # Try to parse "retry in X.Xs" from Gemini responses
                try:
                    body = error.response.text
                    match = re.search(r'retry in (\d+\.?\d*)', body, re.IGNORECASE)
                    if match:
                        return int(float(match.group(1))) + 2  # add buffer
                except Exception:
                    pass
                # 429 without parseable wait — use generous fixed wait
                return 30
            elif status == 404:
                return 1  # no point waiting long for 404
        # Default exponential backoff
        return BACKOFF_BASE * (2 ** attempt)

    def _parse_response(self, raw: Dict, model_name: str) -> LLMResponse:
        """Parse the API response into an LLMResponse."""
        choice = raw.get("choices", [{}])[0]
        message = choice.get("message", {})

        text = message.get("content", "") or ""

        # Parse tool calls (flag and skip malformed JSON instead of silent {})
        tool_calls = []
        malformed_count = 0
        raw_tool_calls = message.get("tool_calls", [])
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            args_str = func.get("arguments", "{}")

            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                logger.warning(
                    f"Malformed tool-call arguments for '{name}': {args_str!r}. Skipping tool call."
                )
                malformed_count += 1
                continue  # skip this tool call rather than invoking with empty args

            tool_calls.append({
                "id": tc.get("id", ""),
                "name": name,
                "arguments": args,
            })

        usage = raw.get("usage", {})

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            model_used=model_name,
            usage=usage,
            raw=raw,
            malformed_tool_call_count=malformed_count,
        )
