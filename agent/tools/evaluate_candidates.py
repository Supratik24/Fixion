"""
agent/tools/evaluate_candidates.py — Competitive multi-candidate patch evaluation tool.

The agent calls this tool to submit N candidate patches.
Each patch is tested in its own container against target tests, full suite, and linter.
The cleanest patch with the smallest diff is selected as the winner.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent.candidate_evaluator import evaluate_candidates_in_containers
from agent.tools import apply_patch as ap_mod
from sandbox.docker_runner import DockerSandbox

logger = logging.getLogger(__name__)


def evaluate_candidates(
    candidates: list[str],
    repo_root: Path,
    sandbox: DockerSandbox | None = None,
    target_tests: list[str] | None = None,
    baseline_passed_tests: set[str] | None = None,
    apply_winner: bool = True,
) -> dict[str, Any]:
    """
    Test N candidate patches in parallel/isolated containers and select the winning patch.

    Parameters
    ----------
    candidates:
        List of candidate unified diff strings to evaluate.
    repo_root:
        Local root directory of the repository.
    sandbox:
        DockerSandbox instance or None.
    target_tests:
        Specific pytest node IDs / paths that must pass.
    baseline_passed_tests:
        Set of known baseline passing tests for regression detection.
    apply_winner:
        If True and a winner is found, automatically applies the winning patch to the repo.

    Returns
    -------
    dict with:
        - winner_found (bool): True if at least one candidate passed all verification steps.
        - winner_id (int | None): 1-based index of the winning candidate.
        - winner_patch (str): Winning unified diff string (or empty if none).
        - winner_diff_size (int): Size of the winning diff in lines.
        - winner_applied (bool): Whether the winner was applied to repo_root.
        - candidates (list[dict]): Full per-candidate results including acceptance/rejection reasons.
        - summary_table (str): Markdown comparison table.
        - message (str): Human- and LLM-readable summary.
    """
    if not candidates:
        return {
            "winner_found": False,
            "winner_id": None,
            "winner_patch": "",
            "winner_diff_size": 0,
            "winner_applied": False,
            "candidates": [],
            "summary_table": "No candidates provided.",
            "message": "No candidate patches were provided for evaluation.",
        }

    report = evaluate_candidates_in_containers(
        candidates=candidates,
        repo_root=repo_root,
        sandbox=sandbox,
        target_tests=target_tests,
        baseline_passed_tests=baseline_passed_tests,
    )

    winner_applied = False
    if report.winner and apply_winner:
        apply_res = ap_mod.apply_patch(report.winner.patch, repo_root)
        winner_applied = apply_res.get("applied", False)

    if report.winner:
        msg = (
            f"Candidate #{report.winner.candidate_id} selected as WINNER ({report.winner.diff_size} lines changed).\n"
            f"All checks passed: target tests passed, full suite passed (no regressions), lint clean.\n"
            f"{'Applied to repository working tree.' if winner_applied else ''}\n\n"
            f"{report.summary_table}"
        )
    else:
        msg = (
            f"No candidate passed all verification steps out of {report.total_candidates} evaluated.\n\n"
            f"{report.summary_table}"
        )

    return {
        "winner_found": report.winner is not None,
        "winner_id": report.winner.candidate_id if report.winner else None,
        "winner_patch": report.winner.patch if report.winner else "",
        "winner_diff_size": report.winner.diff_size if report.winner else 0,
        "winner_applied": winner_applied,
        "candidates": [c.to_dict() for c in report.candidates],
        "summary_table": report.summary_table,
        "message": msg,
    }


# ── Tool declaration ──────────────────────────────────────────────────────────

TOOL_DECLARATION = {
    "name": "evaluate_candidates",
    "description": (
        "Evaluate multiple candidate patches in parallel, testing each in its own container. "
        "Selects the winner based on: passes target tests -> passes full suite -> lint-clean -> smallest diff. "
        "Automatically applies the winning patch to the repository and logs acceptance/rejection reasons for all candidates."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "A list of candidate unified diffs (e.g. 2 or 3 distinct patch hypotheses) "
                    "to test and compare."
                ),
            },
            "apply_winner": {
                "type": "boolean",
                "description": "Whether to automatically apply the winning patch to the repository (default true).",
            },
        },
        "required": ["candidates"],
    },
}
