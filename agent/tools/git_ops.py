"""
agent/tools/git_ops.py — Git operations for context gathering.

The agent uses these to understand repo history around a bug:
- git blame (who last touched a line)
- git log (recent commits to a file)
- git diff (current working-tree changes)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _run_git(args: list[str], cwd: Path) -> str:
    """Run a git command and return stdout as a string."""
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout + result.stderr


def git_blame(
    file_path: str,
    repo_root: Path,
    start_line: int,
    end_line: int,
) -> dict[str, Any]:
    """
    Run git blame on a line range in a file.

    Returns author, commit SHA, date, and line content for each line.
    """
    output = _run_git(
        ["blame", "-L", f"{start_line},{end_line}", "--porcelain", file_path],
        cwd=repo_root,
    )
    lines = []
    current: dict[str, str] = {}
    for line in output.splitlines():
        if line.startswith("\t"):
            current["line_content"] = line[1:]
            lines.append(current)
            current = {}
        elif line.startswith("author "):
            current["author"] = line[7:]
        elif line.startswith("author-time "):
            current["timestamp"] = line[12:]
        elif line.startswith("summary "):
            current["commit_message"] = line[8:]
        elif len(line) == 40 and all(c in "0123456789abcdef" for c in line):
            current["sha"] = line

    return {
        "file_path": file_path,
        "start_line": start_line,
        "end_line": end_line,
        "blame": lines,
    }


def git_log(
    file_path: str,
    repo_root: Path,
    n: int = 10,
) -> dict[str, Any]:
    """
    Return the last *n* commits that touched *file_path*.
    """
    output = _run_git(
        ["log", f"-{n}", "--pretty=format:%H|%an|%ad|%s", "--date=short", "--", file_path],
        cwd=repo_root,
    )
    commits = []
    for line in output.strip().splitlines():
        parts = line.split("|", 3)
        if len(parts) == 4:
            commits.append({
                "sha": parts[0],
                "author": parts[1],
                "date": parts[2],
                "message": parts[3],
            })
    return {"file_path": file_path, "commits": commits}


def git_diff(repo_root: Path) -> dict[str, Any]:
    """Return the current working-tree diff (staged + unstaged)."""
    diff = _run_git(["diff", "HEAD"], cwd=repo_root)
    return {"diff": diff, "has_changes": bool(diff.strip())}


# ── Gemini tool declarations ───────────────────────────────────────────────────

TOOL_DECLARATIONS = [
    {
        "name": "git_blame",
        "description": (
            "Show who last modified each line in a file and when. "
            "Use to understand ownership or find the commit that introduced a bug."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "File path relative to repo root."},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            "required": ["file_path", "start_line", "end_line"],
        },
    },
    {
        "name": "git_log",
        "description": "Show recent commits that touched a specific file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "n": {"type": "integer", "description": "Number of commits to show (default 10).", "default": 10},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "git_diff",
        "description": "Show the current working-tree diff (any uncommitted changes, including applied patches).",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]
