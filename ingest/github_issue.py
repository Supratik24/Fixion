"""
ingest/github_issue.py — Parse and fetch GitHub issue titles and descriptions.

Supports:
- Extracting owner, repo, and issue number from issue URLs
- Fetching issue title, body, and metadata via GitHub REST API
- Providing fallback formatting if unauthenticated or offline
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

_ISSUE_URL_RE = re.compile(
    r"https?://github\.com/(?P<owner>[a-zA-Z0-9_\-\.]+)/(?P<repo>[a-zA-Z0-9_\-\.]+)/issues/(?P<number>\d+)",
    re.IGNORECASE,
)


def parse_issue_url(url: str) -> tuple[str, str, int] | None:
    """
    Parse a GitHub issue URL into (owner, repo, issue_number).

    Example:
    >>> parse_issue_url("https://github.com/psf/requests/issues/1234")
    ('psf', 'requests', 1234)
    """
    if not url or not isinstance(url, str):
        return None
    match = _ISSUE_URL_RE.search(url.strip())
    if not match:
        return None
    return match.group("owner"), match.group("repo"), int(match.group("number"))


def fetch_github_issue(
    url_or_number: str,
    token: str | None = None,
    repo_url: str | None = None,
) -> dict[str, Any]:
    """
    Fetch issue title, body, and metadata from the GitHub REST API.

    Parameters
    ----------
    url_or_number:
        Either a full GitHub issue URL or issue number.
    token:
        Optional GitHub personal access token (falls back to settings.github_token).
    repo_url:
        Optional repository URL if url_or_number is just a number.

    Returns
    -------
    dict with:
        - owner (str)
        - repo (str)
        - number (int)
        - title (str)
        - body (str)
        - issue_text (str)
        - repo_url (str)
        - html_url (str)
    """
    parsed = parse_issue_url(url_or_number)
    if parsed:
        owner, repo, number = parsed
    elif repo_url and url_or_number.isdigit():
        repo_parsed = re.search(r"github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)", repo_url)
        if not repo_parsed:
            raise ValueError(f"Could not parse repo_url: {repo_url}")
        owner = repo_parsed.group("owner")
        repo = repo_parsed.group("repo").removesuffix(".git")
        number = int(url_or_number)
    else:
        raise ValueError(
            f"Invalid GitHub issue format: '{url_or_number}'. "
            "Expected format: https://github.com/<owner>/<repo>/issues/<number>"
        )

    api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{number}"
    req = urllib.request.Request(
        api_url,
        headers={
            "User-Agent": "Fixion-SWE-Agent",
            "Accept": "application/vnd.github.v3+json",
        },
    )

    auth_token = token or settings.github_token
    if auth_token:
        req.add_header("Authorization", f"Bearer {auth_token}")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        title = data.get("title", "").strip()
        body = (data.get("body") or "").strip()
        issue_text = f"## {title}\n\n{body}" if body else title

        return {
            "owner": owner,
            "repo": repo,
            "number": number,
            "title": title,
            "body": body,
            "issue_text": issue_text,
            "repo_url": f"https://github.com/{owner}/{repo}",
            "html_url": data.get("html_url", f"https://github.com/{owner}/{repo}/issues/{number}"),
            "state": data.get("state", "open"),
            "author": data.get("user", {}).get("login", ""),
        }
    except urllib.error.HTTPError as err:
        logger.error("GitHub API error (%d): %s", err.code, err.reason)
        raise RuntimeError(f"GitHub API error ({err.code}): {err.reason} for {api_url}") from err
    except Exception as exc:
        logger.error("Failed fetching GitHub issue: %s", exc)
        raise RuntimeError(f"Failed fetching GitHub issue from {api_url}: {exc}") from exc
