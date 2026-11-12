"""
agent/llm_client.py — Unified LLM client supporting multiple free backends.

Supported backends (set FIXION_BACKEND in .env):
  - "gemini"      : Google Gemini via google-genai SDK (default)
  - "groq"        : Groq (free tier, Llama 3.3 70B / DeepSeek R1)
  - "openrouter"  : OpenRouter (free models: DeepSeek R1, Llama 3.3 70B)

Groq and OpenRouter use the OpenAI-compatible API, so we use openai SDK for them.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

# ── Default models per backend ─────────────────────────────────────────────────

BACKEND_DEFAULTS = {
    "gemini": "gemini-3.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "openrouter": "cohere/north-mini-code:free",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


_ACTIVE_GEMINI_MODEL: str | None = None


# ── Simple message/response wrappers ──────────────────────────────────────────

class ToolCall:
    def __init__(self, name: str, args: dict[str, Any], call_id: str = ""):
        self.name = name
        self.args = args
        self.call_id = call_id


class LLMResponse:
    def __init__(
        self,
        text: str,
        tool_calls: list[ToolCall],
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ):
        self.text = text
        self.tool_calls = tool_calls
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


# ── Backend implementations ────────────────────────────────────────────────────

def _call_gemini(
    history: list[Any],
    system_prompt: str,
    tools: list[Any],
) -> tuple[LLMResponse, Any]:
    """Call via google-genai SDK with automatic model fallback and bounded retry."""
    from google import genai
    from google.genai import types as genai_types
    from google.genai.errors import APIError

    global _ACTIVE_GEMINI_MODEL

    client = genai.Client(
        api_key=settings.gemini_api_key,
        http_options=genai_types.HttpOptions(timeout=60000),
    )

    # Candidate models in preference order (fallback if primary experiences spikes or 503s)
    base_candidates = [settings.model, "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash"]
    candidate_models: list[str] = []
    if _ACTIVE_GEMINI_MODEL and _ACTIVE_GEMINI_MODEL in base_candidates:
        candidate_models.append(_ACTIVE_GEMINI_MODEL)
    for alt in base_candidates:
        if alt not in candidate_models:
            candidate_models.append(alt)

    response = None
    last_error: Exception | None = None

    for model_name in candidate_models:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=history,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        tools=tools,
                        tool_config=genai_types.ToolConfig(
                            function_calling_config=genai_types.FunctionCallingConfig(
                                mode="AUTO"
                            )
                        ),
                        temperature=0.2,
                        max_output_tokens=8192,
                    ),
                )
                _ACTIVE_GEMINI_MODEL = model_name
                break
            except APIError as e:
                last_error = e
                err_str = str(e).lower()
                code = getattr(e, "code", 503)
                if "503" in err_str or code == 503:
                    logger.warning("Model %s returned 503 (high demand spike). Flipping to alternate model...", model_name)
                    break
                elif "429" in err_str or code == 429:
                    logger.warning("Gemini API rate limit on %s. Flipping to alternate model...", model_name)
                    break
                elif "404" in err_str or code == 404:
                    logger.warning("Model %s not found on API. Trying next model...", model_name)
                    break
                else:
                    raise
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                err_name = type(e).__name__.lower()
                if any(
                    k in err_str or k in err_name
                    for k in (
                        "timeout",
                        "timed out",
                        "connect",
                        "connection",
                        "closed",
                        "reset",
                        "network",
                        "socket",
                        "10054",
                        "10053",
                        "remote",
                        "eof",
                        "protocol",
                    )
                ):
                    logger.warning(
                        "Network/timeout issue on %s (%s: %s). Attempt %d/2...",
                        model_name,
                        type(e).__name__,
                        e,
                        attempt + 1,
                    )
                    time.sleep(1)
                else:
                    raise

        if response is not None:
            break

    if response is None:
        raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")

    candidate = response.candidates[0]
    content = candidate.content

    tool_calls = []
    text_parts = []
    for part in content.parts:
        if part.function_call is not None:
            tool_calls.append(ToolCall(
                name=part.function_call.name,
                args=dict(part.function_call.args) if part.function_call.args else {},
            ))
        if part.text:
            text_parts.append(part.text)

    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
    completion_tokens = getattr(usage, "candidates_token_count", 0) or 0

    return (
        LLMResponse(
            text=" ".join(text_parts),
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        content,
    )


def _call_openai_compat(
    history: list[dict],
    system_prompt: str,
    tool_schemas: list[dict],
    base_url: str,
    api_key: str,
    model: str,
) -> LLMResponse:
    """Call via OpenAI-compatible API (Groq or OpenRouter)."""
    try:
        from openai import OpenAI
    except ImportError as err:
        raise RuntimeError(
            "openai package not installed. Run: pip install openai"
        ) from err

    client = OpenAI(api_key=api_key, base_url=base_url)

    messages = [{"role": "system", "content": system_prompt}] + history
    response = None

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
                tool_choice="auto" if tool_schemas else None,
                temperature=0.2,
                max_tokens=8192,
            )
            break
        except Exception as e:
            err_str = str(e)
            if ("429" in err_str or "503" in err_str or "rate" in err_str.lower()) and attempt < 2:
                logger.warning("API rate limit (%s). Retrying in %ds...", base_url, (attempt + 1) * 3)
                time.sleep((attempt + 1) * 3)
            else:
                raise

    if response is None:
        raise RuntimeError("No response received from OpenAI-compatible endpoint.")

    msg = response.choices[0].message

    tool_calls = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            tool_calls.append(ToolCall(
                name=tc.function.name,
                args=args,
                call_id=tc.id,
            ))

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    return LLMResponse(
        text=msg.content or "",
        tool_calls=tool_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


# ── Tool schema converters ─────────────────────────────────────────────────────

def gemini_tools_to_openai(declarations: list[dict]) -> list[dict]:
    """Convert Fixion tool declaration dicts to OpenAI tool schema format."""
    tools = []
    for d in declarations:
        schema = {
            "type": "function",
            "function": {
                "name": d["name"],
                "description": d["description"],
            },
        }
        if "parameters" in d:
            schema["function"]["parameters"] = d["parameters"]
        tools.append(schema)
    return tools


# ── History converters ─────────────────────────────────────────────────────────

def openai_history_append_assistant(history: list[dict], response: LLMResponse, raw_tool_calls: Any = None) -> None:
    """Append assistant turn (with optional tool_calls) to OpenAI-format history."""
    msg: dict[str, Any] = {"role": "assistant", "content": response.text or None}
    if response.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.call_id or f"call_{i}",
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
            }
            for i, tc in enumerate(response.tool_calls)
        ]
    history.append(msg)


def openai_history_append_tool_results(history: list[dict], tool_calls: list[ToolCall], results: list[Any]) -> None:
    """Append tool results to OpenAI-format history."""
    for tc, result in zip(tool_calls, results, strict=False):
        history.append({
            "role": "tool",
            "tool_call_id": tc.call_id or "call_0",
            "content": json.dumps(result, ensure_ascii=False),
        })
