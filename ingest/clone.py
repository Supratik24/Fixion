"""
ingest/clone.py — Clone a GitHub repository into a local workspace directory.

Features:
- Supports public repos (no auth) and private repos (via GITHUB_TOKEN)
- Returns the local path and the HEAD commit SHA
- Idempotent: if the workspace already exists at the same SHA, skips re-clone
- Creates a fresh branch for patch application so main/master is never dirtied
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import git

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class CloneResult:
    """Result of a successful repo clone operation."""

    repo_url: str
    local_path: Path
    head_sha: str
    branch: str  # working branch created for this run


def _workspace_path(repo_url: str) -> Path:
    """Deterministic local path derived from the repo URL."""
    slug = hashlib.sha1(repo_url.encode()).hexdigest()[:12]
    # e.g. ".fixion_cache/workspaces/owner__repo__a1b2c3d4e5f6"
    name = repo_url.rstrip("/").split("/")[-2:]
    label = "__".join(name) + "__" + slug
    return settings.workspace_dir / label


def _authenticated_url(repo_url: str) -> str:
    """Inject GitHub PAT into HTTPS URL if token is configured."""
    token = settings.github_token
    if token and repo_url.startswith("https://github.com/"):
        return repo_url.replace("https://", f"https://{token}@")
    return repo_url


def clone_repo(
    repo_url: str,
    *,
    force_reclone: bool = False,
    branch_prefix: str = "fixion/patch",
) -> CloneResult:
    """
    Clone *repo_url* into the workspace directory and return a CloneResult.

    Parameters
    ----------
    repo_url:
        HTTPS GitHub URL, e.g. ``"https://github.com/psf/requests"``.
    force_reclone:
        If True, delete and re-clone even if the workspace already exists.
    branch_prefix:
        Prefix for the throwaway working branch created for this run.
    """
    settings.workspace_dir.mkdir(parents=True, exist_ok=True)
    local_path = _workspace_path(repo_url)

    if local_path.exists() and not force_reclone:
        logger.info("Workspace already exists at %s — pulling latest.", local_path)
        repo = git.Repo(local_path)
        origin = repo.remotes.origin
        origin.fetch()
        repo.git.checkout(repo.active_branch.name)
        repo.git.pull("--ff-only")
    else:
        if local_path.exists():
            import shutil
            shutil.rmtree(local_path)
        logger.info("Cloning %s → %s", repo_url, local_path)
        auth_url = _authenticated_url(repo_url)
        git.Repo.clone_from(auth_url, local_path, depth=50)

    repo = git.Repo(local_path)
    head_sha = repo.head.commit.hexsha

    # Create a fresh working branch for this run so we never dirty default branch
    working_branch = f"{branch_prefix}-{head_sha[:8]}"
    existing_branches = [b.name for b in repo.branches]  # type: ignore[attr-defined]
    if working_branch not in existing_branches:
        repo.git.checkout("-b", working_branch)
        logger.info("Created working branch: %s", working_branch)
    else:
        repo.git.checkout(working_branch)

    return CloneResult(
        repo_url=repo_url,
        local_path=local_path,
        head_sha=head_sha,
        branch=working_branch,
    )


def reset_to_head(local_path: Path) -> None:
    """
    Discard any uncommitted changes in the workspace.
    Call this between retry iterations to start fresh.
    """
    repo = git.Repo(local_path)
    repo.git.checkout("--", ".")
    repo.git.clean("-fd")
    logger.debug("Workspace reset to HEAD: %s", local_path)
