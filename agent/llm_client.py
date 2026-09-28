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
    "gemini": "gemini-3.8-flash",
    "groq": "llama-3.3-70b-versatile",
    "openrouter": "deepseek/deepseek-r1",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


# ── Simple message/response wrappers ──────────────────────────────────────────

class ToolCall:
    def __init__(self, name: str, args: dict[str, Any], call_id: str = ""):
        self.name = name
        self.args = args
        self.call_id = call_id


class LLMResponse:
    def __init__(self, text: str, tool_calls: list[ToolCall]):
        self.text = text
        self.tool_calls = tool_calls

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


# ── Backend implementations ────────────────────────────────────────────────────

def _call_gemini(
    history: list[Any],
    system_prompt: str,
    tools: list[Any],
) -> LLMResponse:
    """Call via google-genai SDK."""
    from google import genai
    from google.genai import types as genai_types
    from google.genai.errors import APIError

    client = genai.Client(api_key=settings.gemini_api_key)
    model = settings.model if settings.model != BACKEND_DEFAULTS["gemini"] else BACKEND_DEFAULTS["gemini"]

    while True:
        try:
            response = client.models.generate_content(
                model=model,
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
            break
        except APIError as e:
            code = getattr(e, "code", 503)
            if code in (429, 503) or "503" in str(e) or "429" in str(e):
                logger.warning("Gemini API error (%s). Retrying in 30s...", code)
                time.sleep(30)
            else:
                raise
        except Exception as e:
            # Network-level errors (DNS failure, connection reset, timeout)
            err = str(e)
            if any(k in err for k in ("getaddrinfo", "ConnectError", "ConnectionError", "timeout", "RemoteDisconnected")):
                logger.warning("Network error (%s). Retrying in 15s...", type(e).__name__)
                time.sleep(15)
            else:
                raise

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

    return LLMResponse(text=" ".join(text_parts), tool_calls=tool_calls), content


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
    except ImportError:
        raise RuntimeError(
            "openai package not installed. Run: pip install openai"
        )

    client = OpenAI(api_key=api_key, base_url=base_url)

    messages = [{"role": "system", "content": system_prompt}] + history

    while True:
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
            if "429" in err_str or "503" in err_str or "rate" in err_str.lower():
                logger.warning("API rate limit (%s). Retrying in 20s...", base_url)
                time.sleep(20)
            else:
                raise

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

    return LLMResponse(text=msg.content or "", tool_calls=tool_calls)


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
    for tc, result in zip(tool_calls, results):
        history.append({
            "role": "tool",
            "tool_call_id": tc.call_id or f"call_0",
            "content": json.dumps(result, ensure_ascii=False),
        })
