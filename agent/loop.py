"""
agent/loop.py — The main tool-calling agent loop.

This is the heart of Fixion. It orchestrates:
  1. Issue understanding
  2. Localization (reproduce → traceback → symbol graph → RAG)
  3. Patch generation (unified diff)
  4. Verification (lint → test → retry)

The loop runs until:
  - Tests pass (success)
  - Max retries exceeded (give up, return best attempt)
  - A fatal error occurs (sandbox crash, etc.)

Architecture:
  - Uses google-genai SDK with tool_config for function calling
  - Tools are dispatched by name in a central router (_dispatch_tool)
  - Each iteration of the fix loop is a full conversation turn
  - Tool results are fed back as user messages (tool_result content)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types as genai_types

from agent.tools import apply_patch as ap_mod
from agent.tools import find_references as ref_mod
from agent.tools import git_ops as git_mod
from agent.tools import read_file as rf_mod
from agent.tools import run_linter as lint_mod
from agent.tools import run_tests as rt_mod
from agent.tools import search_code as sc_mod
from config import settings
from ingest.embed_index import EmbedIndex
from ingest.symbol_graph import SymbolGraph
from sandbox.docker_runner import DockerSandbox

logger = logging.getLogger(__name__)

# ── Data structures ────────────────────────────────────────────────────────────


from typing import Any, Callable

@dataclass
class AgentContext:
    """Everything the agent needs for a single fix run."""

    repo_root: Path
    repo_url: str
    issue_text: str
    failing_tests: list[str]        # test node IDs to focus on
    embed_index: EmbedIndex
    symbol_graph: SymbolGraph
    sandbox: DockerSandbox
    max_retries: int = field(default_factory=lambda: settings.max_retries)
    log_callback: Callable[[str], None] | None = None



@dataclass
class AgentResult:
    """Outcome of one complete agent run."""

    success: bool
    patch: str                       # final unified diff (may be empty if failed)
    root_cause: str
    fix_explanation: str
    tests_passed: list[str]
    tests_failed: list[str]
    iterations: int
    final_traceback: str
    confidence: str                  # "high" | "medium" | "low"
    error_message: str = ""


# ── Tool dispatch ──────────────────────────────────────────────────────────────


def _dispatch_tool(
    tool_name: str,
    tool_args: dict[str, Any],
    ctx: AgentContext,
) -> Any:
    """
    Central router: map a tool name + args to the correct Python function.
    Returns a JSON-serializable result.
    """
    logger.debug("Tool call: %s(%s)", tool_name, json.dumps(tool_args)[:200])

    # ── Code search ──
    if tool_name == "search_code":
        return sc_mod.search_code(
            query=tool_args["query"],
            embed_index=ctx.embed_index,
            top_k=min(int(tool_args.get("top_k", 8)), 20),
        )

    # ── File reading ──
    if tool_name == "read_file":
        return rf_mod.read_file(
            file_path=tool_args["file_path"],
            repo_root=ctx.repo_root,
            start_line=tool_args.get("start_line"),
            end_line=tool_args.get("end_line"),
            context_lines=tool_args.get("context_lines", 5),
        )

    # ── Symbol graph ──
    if tool_name == "find_definition":
        return ref_mod.find_definition(tool_args["symbol_name"], ctx.symbol_graph)
    if tool_name == "find_callers":
        return ref_mod.find_callers(tool_args["symbol_name"], ctx.symbol_graph)
    if tool_name == "find_callees":
        return ref_mod.find_callees(tool_args["symbol_name"], ctx.symbol_graph)
    if tool_name == "find_neighborhood":
        return ref_mod.find_neighborhood(
            tool_args["symbol_name"], ctx.symbol_graph, radius=tool_args.get("radius", 2)
        )
    if tool_name == "get_file_symbols":
        return ref_mod.get_file_symbols(tool_args["file_path"], ctx.symbol_graph)

    # ── Git ops ──
    if tool_name == "git_blame":
        return git_mod.git_blame(
            tool_args["file_path"], ctx.repo_root,
            int(tool_args["start_line"]), int(tool_args["end_line"])
        )
    if tool_name == "git_log":
        return git_mod.git_log(
            tool_args["file_path"], ctx.repo_root, n=int(tool_args.get("n", 10))
        )
    if tool_name == "git_diff":
        return git_mod.git_diff(ctx.repo_root)

    # ── Patch ops ──
    if tool_name == "validate_patch":
        return ap_mod.validate_patch(tool_args["patch_text"], ctx.repo_root)
    if tool_name == "apply_patch":
        return ap_mod.apply_patch(tool_args["patch_text"], ctx.repo_root)

    # ── Test/lint ──
    if tool_name == "run_linter":
        return lint_mod.run_linter(ctx.repo_root, ctx.sandbox)
    if tool_name == "run_tests":
        test_paths = tool_args.get("test_paths") or ctx.failing_tests or None
        extra_args = tool_args.get("extra_args", [])
        return rt_mod.run_tests(ctx.repo_root, ctx.sandbox, test_paths, extra_args=extra_args)

    return {"error": f"Unknown tool: {tool_name}"}


# ── Tool declarations (all tools combined) ─────────────────────────────────────

def _build_tool_list() -> list[genai_types.Tool]:
    """Build the list of Gemini Tool objects from all module declarations."""
    declarations = [
        sc_mod.TOOL_DECLARATION,
        rf_mod.TOOL_DECLARATION,
        *ref_mod.TOOL_DECLARATIONS,
        *git_mod.TOOL_DECLARATIONS,
        *ap_mod.TOOL_DECLARATIONS,
        lint_mod.TOOL_DECLARATION,
        rt_mod.TOOL_DECLARATION,
    ]
    functions = [
        genai_types.FunctionDeclaration(
            name=d["name"],
            description=d["description"],
            parameters=genai_types.Schema(**d["parameters"]) if "parameters" in d else None,
        )
        for d in declarations
    ]
    return [genai_types.Tool(function_declarations=functions)]


# ── System prompt loader ───────────────────────────────────────────────────────

def _load_prompt(name: str) -> str:
    prompts_dir = Path(__file__).parent / "prompts"
    return (prompts_dir / f"{name}.txt").read_text(encoding="utf-8")


# ── Main loop ──────────────────────────────────────────────────────────────────


def run_agent(ctx: AgentContext) -> AgentResult:
    """
    Run the full Fixion agent loop for one issue.

    Supports multiple backends: gemini | groq | openrouter
    Set FIXION_BACKEND in .env to switch.
    """
    import time
    from agent.llm_client import (
        BACKEND_DEFAULTS, GROQ_BASE_URL, OPENROUTER_BASE_URL,
        ToolCall, LLMResponse,
        _call_gemini, _call_openai_compat,
        gemini_tools_to_openai,
        openai_history_append_assistant, openai_history_append_tool_results,
    )

    backend = settings.backend.lower()
    logger.info("Using backend: %s", backend)

    # Tool declarations (raw dicts for all backends)
    raw_declarations = [
        sc_mod.TOOL_DECLARATION,
        rf_mod.TOOL_DECLARATION,
        *ref_mod.TOOL_DECLARATIONS,
        *git_mod.TOOL_DECLARATIONS,
        *ap_mod.TOOL_DECLARATIONS,
        lint_mod.TOOL_DECLARATION,
        rt_mod.TOOL_DECLARATION,
    ]

    system_prompt = _load_prompt("system")

    test_list = "\n".join(f"  - {t}" for t in ctx.failing_tests) if ctx.failing_tests else "  (unknown — run the full suite)"
    user_message = f"""## Issue
{ctx.issue_text}

## Repository
{ctx.repo_url}

## Failing Tests (if known)
{test_list}

Reproduce the bug, localize the root cause, generate a patch, and verify it passes tests.
Follow your mandatory 4-phase workflow. You MUST write a patch by iteration 2.
"""

    iterations = 0
    last_patch = ""
    last_traceback = ""
    final_answer = ""

    logger.info("Starting agent loop for issue (max_retries=%d)", ctx.max_retries)

    # ── Gemini backend ─────────────────────────────────────────────────────────
    if backend == "gemini":
        from google import genai
        from google.genai import types as genai_types

        gemini_tools = _build_tool_list()
        history: list[genai_types.Content] = [
            genai_types.Content(role="user", parts=[genai_types.Part(text=user_message)])
        ]

        while iterations <= ctx.max_retries:
            iterations += 1
            logger.info("Agent iteration %d/%d", iterations, ctx.max_retries + 1)

            llm_response, raw_content = _call_gemini(history, system_prompt, gemini_tools)
            history.append(raw_content)

            if not llm_response.has_tool_calls:
                final_answer = llm_response.text
                logger.info("Agent produced final answer after %d iterations.", iterations)
                break

            tool_results: list[genai_types.Part] = []
            for tc in llm_response.tool_calls:
                result = _dispatch_tool(tc.name, tc.args, ctx)
                if tc.name == "apply_patch" and result.get("applied"):
                    last_patch = tc.args.get("patch_text", "")
                if tc.name == "run_tests" and not result.get("passed", True):
                    last_traceback = result.get("traceback", "")

                tool_results.append(
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=tc.name,
                            response={"result": json.dumps(result, ensure_ascii=False, indent=2)},
                        )
                    )
                )

            history.append(genai_types.Content(role="user", parts=tool_results))

        else:
            logger.warning("Max retries (%d) exceeded. Returning best attempt.", ctx.max_retries)
            final_answer = "Max retries exceeded."

    # ── Groq / OpenRouter / Ollama backend ────────────────────────────────────
    else:
        if backend == "groq":
            api_key = settings.groq_api_key
            base_url = GROQ_BASE_URL
            model = settings.model if settings.model not in BACKEND_DEFAULTS.values() else BACKEND_DEFAULTS.get("groq", "")
        elif backend == "openrouter":
            api_key = settings.openrouter_api_key
            base_url = OPENROUTER_BASE_URL
            model = settings.model if settings.model not in BACKEND_DEFAULTS.values() else BACKEND_DEFAULTS.get("openrouter", "")
        elif backend == "ollama":
            api_key = "ollama"  # not needed
            base_url = "http://localhost:11434/v1"
            model = settings.model  # e.g., "qwen2.5-coder:7b"
        else:
            raise ValueError(f"Unknown backend: {backend}")

        if not api_key:
            raise RuntimeError(
                f"No API key for backend '{backend}'. "
                f"Set API keys in your .env file."
            )

        openai_tools = gemini_tools_to_openai(raw_declarations)
        oa_history: list[dict] = [{"role": "user", "content": user_message}]

        while iterations <= ctx.max_retries:
            iterations += 1
            logger.info("Agent iteration %d/%d [%s/%s]", iterations, ctx.max_retries + 1, backend, model)

            llm_response = _call_openai_compat(
                oa_history, system_prompt, openai_tools, base_url, api_key, model
            )
            openai_history_append_assistant(oa_history, llm_response)

            if not llm_response.has_tool_calls:
                final_answer = llm_response.text
                logger.info("Agent produced final answer after %d iterations.", iterations)
                break

            results = []
            for tc in llm_response.tool_calls:
                result = _dispatch_tool(tc.name, tc.args, ctx)
                if tc.name == "apply_patch" and result.get("applied"):
                    last_patch = tc.args.get("patch_text", "")
                if tc.name == "run_tests" and not result.get("passed", True):
                    last_traceback = result.get("traceback", "")
                results.append(result)

            openai_history_append_tool_results(oa_history, llm_response.tool_calls, results)

        else:
            logger.warning("Max retries (%d) exceeded. Returning best attempt.", ctx.max_retries)
            final_answer = "Max retries exceeded."

    return _parse_final_answer(
        final_answer,
        last_patch=last_patch,
        last_traceback=last_traceback,
        iterations=iterations,
        ctx=ctx,
    )



def _parse_final_answer(
    text: str,
    *,
    last_patch: str,
    last_traceback: str,
    iterations: int,
    ctx: AgentContext,
) -> AgentResult:
    """
    Extract structured fields from the model's final text response.
    Falls back to safe defaults if the model didn't use the expected format.
    """
    # Try to extract labelled sections
    def _extract(label: str) -> str:
        for line in text.splitlines():
            if line.upper().startswith(label.upper() + ":"):
                return line[len(label) + 1:].strip()
        return ""

    root_cause = _extract("ROOT CAUSE") or "See full output."
    fix_explanation = _extract("FIX") or ""
    confidence = _extract("CONFIDENCE") or "low"

    # Determine success by running final test check
    success = False
    tests_passed: list[str] = []
    tests_failed: list[str] = []

    if last_patch:
        final_test = rt_mod.run_tests(ctx.repo_root, ctx.sandbox, ctx.failing_tests or None)
        tests_passed = final_test.get("passed_tests", [])
        tests_failed = final_test.get("failed_tests", [])
        success = final_test.get("passed", False)
    else:
        tests_failed = ctx.failing_tests

    return AgentResult(
        success=success,
        patch=last_patch,
        root_cause=root_cause,
        fix_explanation=fix_explanation,
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        iterations=iterations,
        final_traceback=last_traceback,
        confidence=confidence if success else "low",
    )
