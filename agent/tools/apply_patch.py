"""
agent/tools/apply_patch.py — Validate and apply a unified diff to the repo.

Two-phase approach:
1. ``validate_patch``: run ``git apply --check`` — zero cost, catches malformed diffs
2. ``apply_patch``: actually write the changes if validation passes

The agent always calls validate first. If validation fails, it revises the
diff without touching the filesystem.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _run_git_apply(
    patch_text: str,
    repo_root: Path,
    check_only: bool = False,
) -> dict[str, Any]:
    """
    Run ``git apply [--check]`` on *patch_text*.

    Returns a dict with:
        - valid (bool)
        - error (str) — git apply stderr if failed
    """
    args = ["git", "apply", "--whitespace=fix"]
    if check_only:
        args.append("--check")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".patch", delete=False, encoding="utf-8"
    ) as f:
        f.write(patch_text)
        patch_file = f.name

    try:
        result = subprocess.run(
            [*args, patch_file],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        success = result.returncode == 0
        return {
            "valid": success,
            "error": (result.stderr or result.stdout).strip() if not success else "",
        }
    finally:
        Path(patch_file).unlink(missing_ok=True)


def validate_patch(patch_text: str, repo_root: Path) -> dict[str, Any]:
    """
    Check whether *patch_text* is a valid unified diff that can be applied cleanly.

    Does NOT modify any files. Safe to call on every LLM-generated diff.

    Returns
    -------
    dict with:
        - valid (bool): True if the patch would apply cleanly
        - error (str): git apply error message if invalid
        - line_count (int): number of lines in the diff
    """
    result = _run_git_apply(patch_text, repo_root, check_only=True)
    result["line_count"] = len(patch_text.splitlines())
    return result


def apply_patch(patch_text: str, repo_root: Path) -> dict[str, Any]:
    """
    Apply *patch_text* to the repo working tree.

    Always runs validate first. Only writes if validation passes.

    Returns
    -------
    dict with:
        - applied (bool): True if patch was successfully applied
        - error (str): error message if application failed
        - files_changed (list[str]): files touched by the patch
    """
    # Phase 1: validate
    validation = _run_git_apply(patch_text, repo_root, check_only=True)
    if not validation["valid"]:
        return {
            "applied": False,
            "error": f"Patch validation failed: {validation['error']}",
            "files_changed": [],
        }

    # Phase 2: apply
    result = _run_git_apply(patch_text, repo_root, check_only=False)
    if not result["valid"]:
        return {
            "applied": False,
            "error": result["error"],
            "files_changed": [],
        }

    # Extract changed files from diff header
    files_changed = []
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            files_changed.append(line[6:])

    return {
        "applied": True,
        "error": "",
        "files_changed": files_changed,
    }


# ── Gemini tool declarations ───────────────────────────────────────────────────

TOOL_DECLARATIONS = [
    {
        "name": "validate_patch",
        "description": (
            "Validate a unified diff patch WITHOUT applying it to the filesystem. "
            "ALWAYS call this before apply_patch. "
            "If invalid, revise the diff format and try again."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "patch_text": {
                    "type": "string",
                    "description": "The complete unified diff text (starting with --- a/... +++ b/...).",
                }
            },
            "required": ["patch_text"],
        },
    },
    {
        "name": "apply_patch",
        "description": (
            "Apply a validated unified diff patch to the repository. "
            "Call validate_patch first. "
            "After applying, immediately run run_linter then run_tests."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "patch_text": {
                    "type": "string",
                    "description": "The complete unified diff text to apply.",
                }
            },
            "required": ["patch_text"],
        },
    },
]
