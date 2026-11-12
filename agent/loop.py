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

import contextlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google.genai import types as genai_types

from agent.llm_client import (
    BACKEND_DEFAULTS,
    GROQ_BASE_URL,
    OPENROUTER_BASE_URL,
    _call_gemini,
    _call_openai_compat,
    gemini_tools_to_openai,
    openai_history_append_assistant,
    openai_history_append_tool_results,
)
from agent.tools import apply_patch as ap_mod
from agent.tools import evaluate_candidates as eval_cand_mod
from agent.tools import find_references as ref_mod
from agent.tools import git_ops as git_mod
from agent.tools import read_file as rf_mod
from agent.tools import reproduce_issue as repro_mod
from agent.tools import run_linter as lint_mod
from agent.tools import run_tests as rt_mod
from agent.tools import search_code as sc_mod
from config import settings
from ingest.embed_index import EmbedIndex
from ingest.symbol_graph import SymbolGraph
from ingest.traceback_parser import parse_traceback
from sandbox.docker_runner import DockerSandbox

logger = logging.getLogger(__name__)

# ── Data structures ────────────────────────────────────────────────────────────


@dataclass
class AgentContext:
    """Everything the agent needs for a single fix run."""

    repo_root: Path
    repo_url: str
    issue_text: str
    failing_tests: list[str]        # test node IDs to focus on
    embed_index: EmbedIndex | None = None
    symbol_graph: SymbolGraph | None = None
    sandbox: DockerSandbox | None = None
    max_retries: int = field(default_factory=lambda: settings.max_retries)
    log_callback: Callable[[str], None] | None = None
    use_symbol_graph: bool = field(default_factory=lambda: settings.use_symbol_graph)
    use_semantic_search: bool = field(default_factory=lambda: settings.use_semantic_search)
    require_reproduction: bool = True
    reproduction_confirmed: bool = False
    reproduction_test_path: str | None = None
    baseline_passed_tests: set[str] = field(default_factory=set)
    num_candidates: int = field(default_factory=lambda: settings.num_candidates)
    approval_callback: Callable[[str], bool] | None = None


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
    metrics: dict[str, Any] = field(default_factory=dict)


# ── Reflection & Localization helpers ─────────────────────────────────────────


def build_failure_reflection(
    last_patch: str,
    traceback_text: str,
    failed_tests: list[str],
    repo_root: Path | str | None = None,
    retry_num: int = 1,
) -> dict[str, Any]:
    """
    Build a structured failure reflection on a failed retry,
    feeding what was tried, the diff, and the failing assertion instead of the raw log.
    """
    parsed = parse_traceback(traceback_text, repo_root=repo_root)

    lines = [
        f"### ⚠️ Test Execution Failed (Retry #{retry_num})",
        f"- **Failed Tests**: {', '.join(failed_tests) if failed_tests else 'Unknown'}",
    ]
    if parsed.error_type:
        lines.append(f"- **Error Type**: `{parsed.error_type}`")
    if parsed.error_message:
        lines.append(f"- **Error Message**: {parsed.error_message}")
    if parsed.failing_assertion:
        lines.append(f"- **Failing Assertion**: `{parsed.failing_assertion}`")
    if parsed.primary_frame:
        lines.append(
            f"- **Failure Location**: `{parsed.primary_frame.file_path}:{parsed.primary_frame.line_number}` in `{parsed.primary_frame.function_name}`"
        )
        if parsed.primary_frame.source_line:
            lines.append(f"- **Failing Code Line**: `{parsed.primary_frame.source_line}`")

    if last_patch:
        lines.append("\n**Attempted Patch (Diff):**")
        lines.append(f"```diff\n{last_patch.strip()}\n```")

    lines.append("\n**Reflection Guidance**:")
    lines.append("The patch applied previously failed the test suite.")
    lines.append("Do NOT repeat the exact same change. Reflect on why the assertion failed above and fix the logic.")

    reflection_message = "\n".join(lines)

    reflection_dict: dict[str, Any] = {
        "passed": False,
        "retry_number": retry_num,
        "error_type": parsed.error_type,
        "error_message": parsed.error_message,
        "failing_assertion": parsed.failing_assertion,
        "failed_tests": failed_tests,
        "attempted_patch": last_patch,
        "failure_reflection": reflection_message,
    }

    if parsed.primary_frame:
        reflection_dict["seed_suggestions"] = {
            "file_path": parsed.primary_frame.file_path,
            "line_number": parsed.primary_frame.line_number,
            "function_name": parsed.primary_frame.function_name,
            "read_file_args": {
                "file_path": parsed.primary_frame.file_path,
                "start_line": max(1, parsed.primary_frame.line_number - 15),
                "end_line": parsed.primary_frame.line_number + 15,
            },
            "find_references_args": (
                {"symbol_name": parsed.primary_frame.function_name}
                if parsed.primary_frame.function_name and parsed.primary_frame.function_name != "<module>"
                else None
            ),
            "hint": (
                f"Failure localized to {parsed.primary_frame.file_path}:{parsed.primary_frame.line_number} "
                f"in {parsed.primary_frame.function_name}. Suggested next steps: call read_file or find_references."
            ),
        }

    return reflection_dict


def _process_test_result(
    result: dict[str, Any],
    last_patch: str,
    iterations: int,
    retries: int,
    ctx: AgentContext,
) -> tuple[dict[str, Any], str, int]:
    """
    Process a run_tests result:
    - If pre-patch run: record baseline passing tests, attach seed suggestions on failure.
    - If post-patch target test failed: return structured failure reflection (no raw log).
    - If post-patch target test passed: run full test suite (regression guard).
      If regressions occur, automatically reject and revert patch.
    Returns: (processed_result, last_traceback, updated_retries)
    """
    # ── Case 1: Pre-patch run (no patch applied yet) ──
    if not last_patch:
        if result.get("passed_tests"):
            ctx.baseline_passed_tests.update(result["passed_tests"])

        if result.get("passed", False):
            return result, "", retries

        raw_tb = result.get("traceback", "") or result.get("stdout_preview", "")
        processed = dict(result)
        parsed = parse_traceback(raw_tb, repo_root=ctx.repo_root)
        if parsed.primary_frame:
            processed["seed_suggestions"] = {
                "file_path": parsed.primary_frame.file_path,
                "line_number": parsed.primary_frame.line_number,
                "function_name": parsed.primary_frame.function_name,
                "read_file_args": {
                    "file_path": parsed.primary_frame.file_path,
                    "start_line": max(1, parsed.primary_frame.line_number - 15),
                    "end_line": parsed.primary_frame.line_number + 15,
                },
                "find_references_args": (
                    {"symbol_name": parsed.primary_frame.function_name}
                    if parsed.primary_frame.function_name and parsed.primary_frame.function_name != "<module>"
                    else None
                ),
                "hint": (
                    f"Failure localized to {parsed.primary_frame.file_path}:{parsed.primary_frame.line_number} "
                    f"in {parsed.primary_frame.function_name}. Suggested next steps: call read_file or find_references."
                ),
            }
        return processed, raw_tb, retries

    # ── Case 2: Post-patch run, target test failed ──
    if not result.get("passed", False):
        raw_tb = result.get("traceback", "") or result.get("stdout_preview", "")
        if iterations > 1:
            retries += 1
        processed = build_failure_reflection(
            last_patch=last_patch,
            traceback_text=raw_tb,
            failed_tests=result.get("failed_tests", []),
            repo_root=ctx.repo_root,
            retry_num=retries or 1,
        )
        return processed, raw_tb, retries

    # ── Case 3: Post-patch run, target test PASSED -> REGRESSION GUARD ──
    logger.info("Target test passed! Running full test suite as regression guard...")
    full_suite = rt_mod.run_tests(ctx.repo_root, ctx.sandbox, test_paths=None)

    suite_failed = set(full_suite.get("failed_tests", []))
    regressions = [t for t in ctx.baseline_passed_tests if t in suite_failed]

    if not regressions and not full_suite.get("passed", False):
        targets = set(ctx.failing_tests)
        if ctx.reproduction_test_path:
            targets.add(ctx.reproduction_test_path)
        other_failed = [t for t in suite_failed if not any(tgt in t for tgt in targets)]
        if other_failed:
            regressions = other_failed
        elif not full_suite.get("passed", False):
            regressions = list(suite_failed) or ["full_suite_execution_failure"]

    if regressions:
        logger.warning("Regression detected in full test suite: %s. Reverting patch.", regressions)
        import subprocess

        # Preserve reproduction test file if present
        repro_content = None
        repro_p = None
        if ctx.reproduction_test_path:
            p = ctx.repo_root / ctx.reproduction_test_path
            if p.exists():
                repro_content = p.read_text(encoding="utf-8")
                repro_p = p

        subprocess.run(["git", "checkout", "--", "."], cwd=str(ctx.repo_root), capture_output=True)

        if repro_content and repro_p:
            repro_p.write_text(repro_content, encoding="utf-8")

        rejection_result = {
            "passed": False,
            "target_test_passed": True,
            "regression_detected": True,
            "regressed_tests": regressions,
            "error_message": (
                f"Patch rejected: target test passed, but {len(regressions)} regression(s) "
                f"detected in full test suite: {', '.join(regressions)}. "
                "The patch has been automatically reverted. Devise a fix that does NOT break existing tests."
            ),
            "failed_tests": regressions,
            "traceback": full_suite.get("traceback", ""),
            "failure_reflection": (
                f"### ⚠️ Patch Rejected Due to Regressions\n"
                f"- **Target Test**: PASSED\n"
                f"- **Regressed Tests**: {', '.join(regressions)}\n"
                f"- **Attempted Patch Diff**:\n```diff\n{last_patch.strip()}\n```\n"
                f"- **Action Required**: The patch fixed the target test but broke previously passing tests. "
                f"The patch has been reverted. Analyze side effects on the regressed tests and try a more targeted change."
            ),
        }
        return rejection_result, full_suite.get("traceback", ""), retries + 1

    # Full suite passed cleanly!
    accepted_result = dict(result)
    accepted_result["full_suite_passed"] = True
    accepted_result["passed_tests"] = full_suite.get("passed_tests", [])
    accepted_result["message"] = "Target tests AND full test suite passed cleanly with no regressions!"
    return accepted_result, "", retries


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
        if not getattr(ctx, "use_semantic_search", True) or ctx.embed_index is None:
            return {"results": [], "warning": "Semantic search is disabled for ablation."}
        return sc_mod.search_code(
            query=tool_args["query"],
            embed_index=ctx.embed_index,
            top_k=min(int(tool_args.get("top_k", 8)), 20),
            mode=tool_args.get("mode", "hybrid"),
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
    if tool_name in ("find_definition", "find_callers", "find_callees", "find_neighborhood", "get_file_symbols"):
        if not getattr(ctx, "use_symbol_graph", True) or ctx.symbol_graph is None:
            return {"results": [], "warning": "Symbol graph is disabled for ablation."}
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

    # ── Reproduction tool ──
    if tool_name == "reproduce_issue":
        res = repro_mod.reproduce_issue(
            test_code=tool_args["test_code"],
            repo_root=ctx.repo_root,
            sandbox=ctx.sandbox,
            test_file_path=tool_args.get("test_file_path", "tests/test_reproduce_issue.py"),
        )
        if res.get("reproduced"):
            ctx.reproduction_confirmed = True
            ctx.reproduction_test_path = res.get("test_file_path")
            if res.get("test_file_path") and res.get("test_file_path") not in ctx.failing_tests:
                ctx.failing_tests.append(res["test_file_path"])
        return res

    # ── Multi-candidate evaluation ──
    if tool_name == "evaluate_candidates":
        if getattr(ctx, "require_reproduction", True) and not getattr(ctx, "reproduction_confirmed", False):
            return {
                "winner_found": False,
                "winner_applied": False,
                "error": (
                    "Reproduction guard: You must first write a minimal failing test using "
                    "'reproduce_issue' and confirm it fails before evaluating candidate patches."
                ),
            }
        return eval_cand_mod.evaluate_candidates(
            candidates=tool_args["candidates"],
            repo_root=ctx.repo_root,
            sandbox=ctx.sandbox,
            target_tests=ctx.failing_tests or None,
            baseline_passed_tests=ctx.baseline_passed_tests or None,
            apply_winner=tool_args.get("apply_winner", True),
        )

    # ── Patch ops ──
    if tool_name == "validate_patch":
        return ap_mod.validate_patch(tool_args["patch_text"], ctx.repo_root)
    if tool_name == "apply_patch":
        if getattr(ctx, "require_reproduction", True) and not getattr(ctx, "reproduction_confirmed", False):
            return {
                "applied": False,
                "error": (
                    "Reproduction guard: You must first write a minimal failing test using "
                    "'reproduce_issue' and confirm it fails before you can apply a patch."
                ),
                "files_changed": [],
            }
        # Human-in-the-loop review guard
        if getattr(ctx, "approval_callback", None) is not None:
            approved = ctx.approval_callback(tool_args["patch_text"])
            if not approved:
                return {
                    "applied": False,
                    "error": (
                        "Patch rejected by human reviewer. Please formulate an alternative "
                        "patch hypothesis addressing reviewer concerns."
                    ),
                    "files_changed": [],
                }
        return ap_mod.apply_patch(tool_args["patch_text"], ctx.repo_root)

    # ── Git PR ops ──
    if tool_name == "create_pull_request":
        return git_mod.create_pull_request(
            repo_root=ctx.repo_root,
            title=tool_args["title"],
            root_cause=tool_args["root_cause"],
            fix_explanation=tool_args["fix_explanation"],
            branch_name=tool_args.get("branch_name"),
            repo_url=ctx.repo_url,
        )

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
        repro_mod.TOOL_DECLARATION,
        eval_cand_mod.TOOL_DECLARATION,
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


def _notify(ctx: AgentContext, msg: str) -> None:
    """Send log event to logger and to ctx.log_callback if available."""
    logger.info("%s", msg)
    if ctx.log_callback:
        try:
            ctx.log_callback(msg)
        except UnicodeEncodeError:
            safe_msg = msg.encode("ascii", errors="replace").decode("ascii")
            with contextlib.suppress(Exception):
                ctx.log_callback(safe_msg)
        except Exception:
            pass


def run_agent(ctx: AgentContext) -> AgentResult:
    """
    Run the full Fixion agent loop for one issue.

    Supports multiple backends: gemini | groq | openrouter
    Set FIXION_BACKEND in .env to switch.
    """
    backend = settings.backend.lower()
    logger.info("Using backend: %s", backend)

    # Tool declarations (raw dicts for all backends)
    raw_declarations = [
        repro_mod.TOOL_DECLARATION,
        eval_cand_mod.TOOL_DECLARATION,
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
    tokens_in = 0
    tokens_out = 0
    tool_call_count = 0
    retries = 0
    start_time = time.monotonic()

    logger.info("Starting agent loop for issue (max_retries=%d)", ctx.max_retries)

    # ── Gemini backend ─────────────────────────────────────────────────────────
    gemini_failed = False
    if backend == "gemini":
        from google.genai import types as genai_types

        gemini_tools = _build_tool_list()
        history: list[genai_types.Content] = [
            genai_types.Content(role="user", parts=[genai_types.Part(text=user_message)])
        ]

        while iterations <= ctx.max_retries:
            iterations += 1
            _notify(ctx, f"🧠 Reasoning iteration {iterations}/{ctx.max_retries + 1} [{backend}]...")

            try:
                llm_response, raw_content = _call_gemini(history, system_prompt, gemini_tools)
            except Exception as exc:
                if settings.openrouter_api_key or settings.groq_api_key:
                    fallback_b = "openrouter" if settings.openrouter_api_key else "groq"
                    _notify(ctx, f"⚠️ Gemini API issue ({exc}). Failing over to {fallback_b}...")
                    gemini_failed = True
                    backend = fallback_b
                    iterations = 0
                    break
                else:
                    _notify(ctx, f"❌ LLM error: {exc}")
                    final_answer = f"Agent failed due to LLM error: {exc}"
                    break

            history.append(raw_content)

            tokens_in += getattr(llm_response, "prompt_tokens", 0)
            tokens_out += getattr(llm_response, "completion_tokens", 0)
            tool_call_count += len(llm_response.tool_calls)

            if not llm_response.has_tool_calls:
                final_answer = llm_response.text
                _notify(ctx, f"🏁 Agent produced final resolution after {iterations} iteration(s).")
                break

            tool_results: list[genai_types.Part] = []
            for tc in llm_response.tool_calls:
                _notify(ctx, f"🔧 Tool: {tc.name}...")
                result = _dispatch_tool(tc.name, tc.args, ctx)
                if tc.name == "reproduce_issue":
                    if result.get("reproduced"):
                        _notify(ctx, "✅ Bug reproduced: test confirmed failing on unpatched codebase.")
                    else:
                        _notify(ctx, f"ℹ️ Reproduction status: {result.get('status')}")
                elif tc.name == "apply_patch" and result.get("applied"):
                    last_patch = tc.args.get("patch_text", "")
                    _notify(ctx, "📝 Candidate patch applied cleanly to workspace.")
                elif tc.name == "evaluate_candidates" and result.get("winner_applied"):
                    last_patch = result.get("winner_patch", "")
                    _notify(ctx, f"🏆 Evaluated {result.get('total_candidates', 0)} candidates — winning patch applied.")
                elif tc.name == "run_tests":
                    if result.get("passed"):
                        _notify(ctx, "✅ Verification tests PASSED!")
                    else:
                        failed = result.get("failed_tests", [])
                        _notify(ctx, f"❌ Tests FAILED ({len(failed)} test(s): {', '.join(failed[:2])})")
                    result, last_traceback, retries = _process_test_result(
                        result, last_patch, iterations, retries, ctx
                    )
                    if result.get("regression_detected"):
                        last_patch = ""
                        _notify(ctx, "⚠️ Regression detected! Auto-reverting patch.")

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
            if not gemini_failed:
                _notify(ctx, f"⚠️ Max retries ({ctx.max_retries}) reached.")
                final_answer = "Max retries exceeded."

    # ── Groq / OpenRouter / Ollama backend ────────────────────────────────────
    if backend != "gemini" or gemini_failed:
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
            _notify(ctx, f"🧠 Reasoning iteration {iterations}/{ctx.max_retries + 1} [{backend}/{model}]...")

            llm_response = _call_openai_compat(
                oa_history, system_prompt, openai_tools, base_url, api_key, model
            )
            openai_history_append_assistant(oa_history, llm_response)

            tokens_in += getattr(llm_response, "prompt_tokens", 0)
            tokens_out += getattr(llm_response, "completion_tokens", 0)
            tool_call_count += len(llm_response.tool_calls)

            if not llm_response.has_tool_calls:
                final_answer = llm_response.text
                _notify(ctx, f"🏁 Agent produced final resolution after {iterations} iteration(s).")
                break

            results = []
            for tc in llm_response.tool_calls:
                _notify(ctx, f"🔧 Tool: {tc.name}...")
                result = _dispatch_tool(tc.name, tc.args, ctx)
                if tc.name == "reproduce_issue":
                    if result.get("reproduced"):
                        _notify(ctx, "✅ Bug reproduced: test confirmed failing on unpatched codebase.")
                    else:
                        _notify(ctx, f"ℹ️ Reproduction status: {result.get('status')}")
                elif tc.name == "apply_patch" and result.get("applied"):
                    last_patch = tc.args.get("patch_text", "")
                    _notify(ctx, "📝 Candidate patch applied cleanly to workspace.")
                elif tc.name == "evaluate_candidates" and result.get("winner_applied"):
                    last_patch = result.get("winner_patch", "")
                    _notify(ctx, f"🏆 Evaluated {result.get('total_candidates', 0)} candidates — winning patch applied.")
                elif tc.name == "run_tests":
                    if result.get("passed"):
                        _notify(ctx, "✅ Verification tests PASSED!")
                    else:
                        failed = result.get("failed_tests", [])
                        _notify(ctx, f"❌ Tests FAILED ({len(failed)} test(s): {', '.join(failed[:2])})")
                    result, last_traceback, retries = _process_test_result(
                        result, last_patch, iterations, retries, ctx
                    )
                    if result.get("regression_detected"):
                        last_patch = ""
                        _notify(ctx, "⚠️ Regression detected! Auto-reverting patch.")
                results.append(result)

            openai_history_append_tool_results(oa_history, llm_response.tool_calls, results)

        else:
            _notify(ctx, f"⚠️ Max retries ({ctx.max_retries}) reached.")
            final_answer = "Max retries exceeded."

    return _parse_final_answer(
        final_answer,
        last_patch=last_patch,
        last_traceback=last_traceback,
        iterations=iterations,
        ctx=ctx,
        start_time=start_time,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        tool_call_count=tool_call_count,
        retries=retries,
    )



def _parse_final_answer(
    text: str,
    *,
    last_patch: str,
    last_traceback: str,
    iterations: int,
    ctx: AgentContext,
    start_time: float,
    tokens_in: int = 0,
    tokens_out: int = 0,
    tool_call_count: int = 0,
    retries: int = 0,
) -> AgentResult:
    """
    Extract structured fields from the model's final text response.
    Falls back to safe defaults if the model didn't use the expected format.
    """
    wall_time = max(0.0, time.monotonic() - start_time)

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

    if text == "Max retries exceeded.":
        status = "max_retries_exceeded"
    elif success:
        status = "resolved"
    else:
        status = "unresolved"

    metrics = {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "total_tokens": tokens_in + tokens_out,
        "tool_call_count": tool_call_count,
        "iterations": iterations,
        "retries": retries,
        "wall_time": round(wall_time, 2),
        "status": status,
    }

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
        metrics=metrics,
    )
