"""
agent/tools/git_ops.py — Git operations for context gathering.

The agent uses these to understand repo history around a bug:
- git blame (who last touched a line)
- git log (recent commits to a file)
- git diff (current working-tree changes)
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


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


def create_pull_request(
    repo_root: Path,
    title: str,
    root_cause: str,
    fix_explanation: str,
    branch_name: str | None = None,
    base_branch: str = "main",
    repo_url: str | None = None,
    github_token: str | None = None,
) -> dict[str, Any]:
    """
    Push working-tree changes to a new git branch and open a Pull Request via GitHub API.

    Parameters
    ----------
    repo_root:
        Local root directory of the repository.
    title:
        Pull request and commit title.
    root_cause:
        Detailed analysis of the root cause for PR description.
    fix_explanation:
        Explanation of the fix and verification for PR description.
    branch_name:
        Optional branch name (default: fixion/fix-<timestamp>).
    base_branch:
        Target base branch to merge into (default: main).
    repo_url:
        GitHub repository URL (e.g. https://github.com/owner/repo).
    github_token:
        GitHub API token for authentication.
    """
    token = github_token or settings.github_token
    diff_status = git_diff(repo_root)
    if not diff_status["has_changes"]:
        return {"success": False, "error": "No working tree changes detected to create a pull request."}

    branch = branch_name or f"fixion/fix-{int(time.time())}"

    # 1. Create branch, stage and commit
    try:
        subprocess.run(["git", "checkout", "-b", branch], cwd=str(repo_root), check=True, capture_output=True, text=True)
        subprocess.run(["git", "add", "."], cwd=str(repo_root), check=True, capture_output=True, text=True)
        subprocess.run(["git", "commit", "-m", title], cwd=str(repo_root), check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return {"success": False, "error": f"Git commit failed: {e.stderr or e.stdout}"}

    # 2. Parse repo owner and name
    owner = ""
    repo_name = ""
    if repo_url:
        match = re.search(r"github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)", repo_url)
        if match:
            owner = match.group("owner")
            repo_name = match.group("repo").removesuffix(".git")

    if not owner or not repo_name:
        remote_out = _run_git(["remote", "get-url", "origin"], repo_root)
        match = re.search(r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/]+)", remote_out)
        if match:
            owner = match.group("owner")
            repo_name = match.group("repo").removesuffix(".git")

    if not owner or not repo_name:
        return {
            "success": False,
            "branch": branch,
            "error": "Could not determine GitHub repository owner/name from URL or origin remote.",
        }

    # 3. Push branch
    push_remote = "origin"
    if token:
        push_remote = f"https://x-access-token:{token}@github.com/{owner}/{repo_name}.git"

    push_res = subprocess.run(["git", "push", "-u", push_remote, branch], cwd=str(repo_root), capture_output=True, text=True)
    if push_res.returncode != 0 and "already exists" not in push_res.stderr:
        logger.warning("Git push warning: %s", push_res.stderr)

    if not token:
        return {
            "success": True,
            "branch": branch,
            "pr_url": "",
            "message": f"Branch '{branch}' created and committed locally. GITHUB_TOKEN not configured to create remote PR via API.",
        }

    # 4. Open PR via GitHub API
    api_url = f"https://api.github.com/repos/{owner}/{repo_name}/pulls"
    pr_body = (
        f"## Root Cause\n{root_cause}\n\n"
        f"## Fix Explanation\n{fix_explanation}\n\n"
        "---\n*Automated PR created by [Fixion Autonomous SWE Agent](https://github.com/Fixion)*"
    )
    payload = json.dumps({
        "title": title,
        "head": branch,
        "base": base_branch,
        "body": pr_body,
    }).encode("utf-8")

    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={
            "User-Agent": "Fixion-SWE-Agent",
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {
                "success": True,
                "branch": branch,
                "pr_number": data.get("number"),
                "pr_url": data.get("html_url", ""),
                "title": data.get("title", ""),
                "message": f"Successfully created PR #{data.get('number')}: {data.get('html_url')}",
            }
    except Exception as exc:
        logger.error("Failed to create GitHub PR: %s", exc)
        return {
            "success": False,
            "branch": branch,
            "error": f"Failed creating PR via GitHub API: {exc}",
        }


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
    {
        "name": "create_pull_request",
        "description": "Commit changes, push to a new branch, and open a Pull Request via GitHub API.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "PR title"},
                "root_cause": {"type": "string", "description": "Root cause summary"},
                "fix_explanation": {"type": "string", "description": "Fix explanation"},
                "branch_name": {"type": "string", "description": "Branch name (optional)"},
            },
            "required": ["title", "root_cause", "fix_explanation"],
        },
    },
]
