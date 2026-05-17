import threading
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import AIMessage


# Approximate pricing per 1M tokens: (input_usd, output_usd)
# Sources: provider pricing pages as of 2025. Marked with ~ where estimated.
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-6": (15.00, 75.00),
    "claude-opus-4-5": (15.00, 75.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    # OpenAI
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o3": (10.00, 40.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    # Speculative/future OpenAI IDs used in catalog
    "gpt-5.4": (2.50, 10.00),
    "gpt-5.4-mini": (0.40, 1.60),
    "gpt-5.4-nano": (0.10, 0.40),
    "gpt-5.4-pro": (30.00, 180.00),
    "gpt-5.2": (2.00, 8.00),
    # Google
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.0-flash": (0.10, 0.40),
    # Speculative/future Google IDs used in catalog
    "gemini-3.1-pro-preview": (1.25, 10.00),
    "gemini-3-flash-preview": (0.30, 2.50),
    "gemini-3.1-flash-lite-preview": (0.10, 0.40),
    # xAI
    "grok-3": (3.00, 15.00),
    "grok-3-mini": (0.30, 0.50),
    "grok-4-0709": (3.00, 15.00),
    "grok-4-1-fast-reasoning": (2.00, 10.00),
    "grok-4-fast-reasoning": (2.00, 10.00),
    "grok-4-1-fast-non-reasoning": (1.00, 5.00),
    "grok-4-fast-non-reasoning": (1.00, 5.00),
    # DeepSeek
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    "deepseek-v4-pro": (0.50, 2.00),
    "deepseek-v4-flash": (0.10, 0.30),
    # Qwen
    "qwen-plus": (0.30, 1.00),
    "qwen3-max": (0.80, 2.40),
    "qwen3.5-flash": (0.05, 0.20),
    "qwen3.5-plus": (0.30, 1.00),
    "qwen3.6-plus": (0.40, 1.20),
    # GLM
    "glm-4.7": (0.40, 1.50),
    "glm-5": (0.80, 3.00),
    "glm-5.1": (1.00, 4.00),
}


def get_model_cost(model_name: str, tokens_in: int, tokens_out: int) -> float:
    """Return USD cost for the given token counts and model. Returns 0.0 if model unknown."""
    key = model_name.lower().strip()
    if key not in MODEL_PRICING:
        # Try prefix match (e.g. "claude-sonnet-4-6-20250514" → "claude-sonnet-4-6")
        for catalog_key in MODEL_PRICING:
            if key.startswith(catalog_key) or catalog_key.startswith(key):
                key = catalog_key
                break
        else:
            return 0.0
    price_in, price_out = MODEL_PRICING[key]
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


class StatsCallbackHandler(BaseCallbackHandler):
    """Callback handler that tracks LLM calls, tool calls, token usage, and cost."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.total_cost_usd = 0.0
        self._run_models: Dict[str, str] = {}  # run_id -> model_name
        self._has_pricing = False  # True once at least one model with known pricing fires

    def _extract_model_name(self, serialized: Dict[str, Any]) -> str:
        """Pull model name out of the serialized LLM dict."""
        kw = serialized.get("kwargs", {})
        return (
            kw.get("model_name")
            or kw.get("model")
            or serialized.get("name", "")
            or ""
        )

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when an LLM starts."""
        model = self._extract_model_name(serialized)
        with self._lock:
            self.llm_calls += 1
            if run_id is not None:
                self._run_models[str(run_id)] = model

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when a chat model starts."""
        model = self._extract_model_name(serialized)
        with self._lock:
            self.llm_calls += 1
            if run_id is not None:
                self._run_models[str(run_id)] = model

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Extract token usage from LLM response and accumulate cost."""
        try:
            generation = response.generations[0][0]
        except (IndexError, TypeError):
            return

        usage_metadata = None
        if hasattr(generation, "message"):
            message = generation.message
            if isinstance(message, AIMessage) and hasattr(message, "usage_metadata"):
                usage_metadata = message.usage_metadata

        if usage_metadata:
            t_in = usage_metadata.get("input_tokens", 0)
            t_out = usage_metadata.get("output_tokens", 0)
            model = ""
            if run_id is not None:
                with self._lock:
                    model = self._run_models.pop(str(run_id), "")
            cost = get_model_cost(model, t_in, t_out)
            with self._lock:
                self.tokens_in += t_in
                self.tokens_out += t_out
                self.total_cost_usd += cost
                if cost > 0:
                    self._has_pricing = True

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Increment tool call counter when a tool starts."""
        with self._lock:
            self.tool_calls += 1

    def get_stats(self) -> Dict[str, Any]:
        """Return current statistics."""
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "total_cost_usd": self.total_cost_usd,
                "has_pricing": self._has_pricing,
            }
