"""
agent/tools/run_linter.py — Run ruff + mypy in the sandbox as a cheap pre-filter.

Called before burning LLM retry budget on test execution.
If the patch introduces syntax errors or obvious type mismatches, this
catches them in seconds rather than waiting for pytest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sandbox.docker_runner import DockerSandbox


def run_linter(repo_path: Path, sandbox: DockerSandbox) -> dict[str, Any]:
    """
    Run ruff (style/error check) and mypy (type check) on the repo.

    Returns
    -------
    dict with:
        - clean (bool): True if no errors detected
        - output (str): combined linter output
    """
    result = sandbox.run_linter(repo_path)
    return {
        "clean": result.get("passed", False),
        "output": result.get("output", "")[:3000],
    }


# ── Gemini tool declaration ────────────────────────────────────────────────────

TOOL_DECLARATION = {
    "name": "run_linter",
    "description": (
        "Run ruff (syntax/style errors) and mypy (type errors) on the current repo state. "
        "Use immediately after applying a patch and before running tests. "
        "If this fails, revise the patch instead of wasting time running tests."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}
