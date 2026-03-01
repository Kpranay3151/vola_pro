"""OpenRouter LLM Client — HTTP client with retry, model fallback, and tool calling support."""

import os
import json
import time
import requests
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Free models on OpenRouter that support tool/function calling
# (ordered by preference)
FREE_MODELS = [
    "meta-llama/llama-4-maverick:free",
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Retry config
MAX_RETRIES = 3
BACKOFF_BASE = 1  # seconds


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
    """Client for the OpenRouter chat completions API.
    
    Features:
        - Automatic retry with exponential backoff
        - Model fallback chain (tries multiple free models)
        - Tool/function calling support
        - Configurable timeout
    """

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.timeout = timeout
        self.models = list(FREE_MODELS)

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Provide it as an argument or set the "
                "OPENROUTER_API_KEY environment variable."
            )

    def chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        """Send a chat completion request with optional tool schemas.
        
        Tries each model in the fallback chain on failure.
        
        Args:
            messages: Chat messages in OpenAI format.
            tools: Optional tool schemas for function calling.
            model: Override model (skips fallback chain).
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.
            
        Returns:
            LLMResponse with text and/or tool calls.
            
        Raises:
            RuntimeError: If all models and retries are exhausted.
        """
        models_to_try = [model] if model else self.models
        last_error = None

        for model_name in models_to_try:
            for attempt in range(MAX_RETRIES):
                try:
                    response = self._make_request(
                        model_name, messages, tools, temperature, max_tokens
                    )
                    return self._parse_response(response, model_name)
                except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
                    last_error = e
                    wait = BACKOFF_BASE * (2 ** attempt)
                    print(f"  ⚠ LLM call failed (model={model_name}, attempt={attempt+1}/{MAX_RETRIES}): {e}")
                    print(f"    Retrying in {wait}s...")
                    time.sleep(wait)

            print(f"  ✗ Model {model_name} exhausted all retries, trying next model...")

        raise RuntimeError(
            f"All LLM models exhausted after retries. Last error: {last_error}"
        )

    def _make_request(
        self,
        model: str,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        temperature: float,
        max_tokens: int,
    ) -> Dict:
        """Make a single HTTP request to OpenRouter."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/vola-pro-pipeline",
            "X-Title": "Transaction RAG Pipeline",
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
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()
        return response.json()

    def _parse_response(self, raw: Dict, model_name: str) -> LLMResponse:
        """Parse the OpenRouter API response into an LLMResponse."""
        choice = raw.get("choices", [{}])[0]
        message = choice.get("message", {})

        text = message.get("content", "") or ""

        # Parse tool calls (Gap 2 — flag and skip malformed JSON instead of silent {})
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
