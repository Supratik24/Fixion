"""
agent/candidate_evaluator.py — Multi-candidate patch generation & evaluation.

Evaluates N candidate patches, each tested in its own container:
1. Applies candidate patch to an isolated workspace copy.
2. Runs target tests in the container.
3. If target tests pass, runs the full test suite (regression guard).
4. Runs linter (ruff + mypy).
5. Selects winner: passes tests -> passes full suite -> lint-clean -> smallest diff.
6. Logs reasons for acceptance / rejection of every candidate.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent.tools import apply_patch as ap_mod
from agent.tools import run_linter as lint_mod
from agent.tools import run_tests as rt_mod
from sandbox.docker_runner import DockerSandbox

logger = logging.getLogger(__name__)


def calculate_diff_size(patch_text: str) -> int:
    """Calculate the number of added or removed lines in a unified diff."""
    if not patch_text or not patch_text.strip():
        return 0
    count = 0
    for line in patch_text.splitlines():
        if (line.startswith("+") and not line.startswith("+++")) or (
            line.startswith("-") and not line.startswith("---")
        ):
            count += 1
    return count


@dataclass
class CandidateResult:
    """Evaluation result for an individual candidate patch."""

    candidate_id: int
    patch: str
    diff_size: int
    applied_cleanly: bool = False
    passes_target_tests: bool = False
    passes_full_suite: bool = False
    lint_clean: bool = False
    status: str = "rejected"  # "accepted" | "rejected"
    reason: str = ""
    failed_tests: list[str] = field(default_factory=list)
    passed_tests: list[str] = field(default_factory=list)
    linter_output: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Summary of competitive multi-candidate evaluation."""

    candidates: list[CandidateResult]
    winner: CandidateResult | None
    total_candidates: int
    qualifying_candidates: int
    summary_table: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_candidates": self.total_candidates,
            "qualifying_candidates": self.qualifying_candidates,
            "winner": self.winner.to_dict() if self.winner else None,
            "candidates": [c.to_dict() for c in self.candidates],
            "summary_table": self.summary_table,
        }


def _create_isolated_copy(repo_root: Path, target_dir: Path) -> None:
    """Create an isolated copy of the repository for containerized candidate testing."""
    shutil.copytree(
        repo_root,
        target_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            ".git",
            ".fixion_cache",
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
        ),
    )


def evaluate_single_candidate(
    candidate_id: int,
    patch_text: str,
    repo_root: Path,
    sandbox: DockerSandbox | None = None,
    target_tests: list[str] | None = None,
    baseline_passed_tests: set[str] | None = None,
) -> CandidateResult:
    """Evaluate one candidate patch in an isolated container/directory."""
    start_time = time.monotonic()
    diff_size = calculate_diff_size(patch_text)

    with tempfile.TemporaryDirectory() as tmp_dir:
        isolated_root = Path(tmp_dir)
        _create_isolated_copy(repo_root, isolated_root)

        # 1. Apply patch
        apply_res = ap_mod.apply_patch(patch_text, isolated_root)
        if not apply_res.get("applied", False):
            err = apply_res.get("error", "Failed to apply diff")
            return CandidateResult(
                candidate_id=candidate_id,
                patch=patch_text,
                diff_size=diff_size,
                applied_cleanly=False,
                status="rejected",
                reason=f"Rejected: patch failed to apply cleanly ({err})",
                duration_seconds=round(time.monotonic() - start_time, 2),
            )

        # 2. Run target tests in container
        target_res = rt_mod.run_tests(isolated_root, sandbox, test_paths=target_tests)
        passes_target = target_res.get("passed", False)
        failed_tests = target_res.get("failed_tests", [])
        passed_tests = target_res.get("passed_tests", [])

        if not passes_target:
            fail_str = ", ".join(failed_tests) if failed_tests else "unknown test failure"
            return CandidateResult(
                candidate_id=candidate_id,
                patch=patch_text,
                diff_size=diff_size,
                applied_cleanly=True,
                passes_target_tests=False,
                status="rejected",
                reason=f"Rejected: failed target test(s) ({fail_str})",
                failed_tests=failed_tests,
                passed_tests=passed_tests,
                duration_seconds=round(time.monotonic() - start_time, 2),
            )

        # 3. Run full test suite in container (regression guard)
        full_res = rt_mod.run_tests(isolated_root, sandbox, test_paths=None)
        suite_failed = set(full_res.get("failed_tests", []))
        regressions: list[str] = []

        if baseline_passed_tests:
            regressions = [t for t in baseline_passed_tests if t in suite_failed]
        if not regressions and not full_res.get("passed", False):
            # Check any failures not in target tests
            targets = set(target_tests or [])
            other_failed = [t for t in suite_failed if not any(tgt in t for tgt in targets)]
            if other_failed:
                regressions = other_failed
            elif not full_res.get("passed", False):
                regressions = list(suite_failed) or ["full_suite_execution_failure"]

        if regressions:
            reg_str = ", ".join(regressions)
            return CandidateResult(
                candidate_id=candidate_id,
                patch=patch_text,
                diff_size=diff_size,
                applied_cleanly=True,
                passes_target_tests=True,
                passes_full_suite=False,
                status="rejected",
                reason=f"Rejected: caused {len(regressions)} regression(s) in full test suite ({reg_str})",
                failed_tests=regressions,
                passed_tests=full_res.get("passed_tests", []),
                duration_seconds=round(time.monotonic() - start_time, 2),
            )

        # 4. Run linter in container
        lint_res = lint_mod.run_linter(isolated_root, sandbox)
        lint_clean = lint_res.get("clean", False)
        linter_output = lint_res.get("output", "")

        if not lint_clean:
            return CandidateResult(
                candidate_id=candidate_id,
                patch=patch_text,
                diff_size=diff_size,
                applied_cleanly=True,
                passes_target_tests=True,
                passes_full_suite=True,
                lint_clean=False,
                status="rejected",
                reason="Rejected: linter (ruff/mypy) errors detected in patched code",
                linter_output=linter_output[:500],
                passed_tests=full_res.get("passed_tests", []),
                duration_seconds=round(time.monotonic() - start_time, 2),
            )

        # Candidate passed all checks!
        return CandidateResult(
            candidate_id=candidate_id,
            patch=patch_text,
            diff_size=diff_size,
            applied_cleanly=True,
            passes_target_tests=True,
            passes_full_suite=True,
            lint_clean=True,
            status="qualifying",
            reason="Passed target tests, full suite, and linter",
            passed_tests=full_res.get("passed_tests", []),
            duration_seconds=round(time.monotonic() - start_time, 2),
        )


def evaluate_candidates_in_containers(
    candidates: list[str],
    repo_root: Path,
    sandbox: DockerSandbox | None = None,
    target_tests: list[str] | None = None,
    baseline_passed_tests: set[str] | None = None,
) -> EvaluationReport:
    """
    Evaluate N candidate patches in parallel/isolated containers and select the winner.

    Selection criteria:
    1. Must pass target tests
    2. Must pass full test suite (0 regressions)
    3. Must be lint-clean
    4. Smallest diff (fewest modified lines)
    """
    results: list[CandidateResult] = []

    logger.info("Evaluating %d candidate patch(es) in isolated containers...", len(candidates))
    for idx, patch_text in enumerate(candidates, start=1):
        logger.info("Evaluating candidate %d/%d...", idx, len(candidates))
        res = evaluate_single_candidate(
            candidate_id=idx,
            patch_text=patch_text,
            repo_root=repo_root,
            sandbox=sandbox,
            target_tests=target_tests,
            baseline_passed_tests=baseline_passed_tests,
        )
        results.append(res)

    # Filter qualifying candidates
    qualifying = [c for c in results if c.status == "qualifying"]

    winner: CandidateResult | None = None
    if qualifying:
        # Sort by smallest diff, then candidate_id
        qualifying.sort(key=lambda c: (c.diff_size, c.candidate_id))
        winner = qualifying[0]
        winner.status = "accepted"
        winner.reason = (
            f"Accepted (Winner): passes target tests, passes full suite, "
            f"lint-clean, smallest diff ({winner.diff_size} lines changed)"
        )

        # Mark other qualifying candidates as tie-break rejected
        for other in qualifying[1:]:
            other.status = "rejected"
            other.reason = (
                f"Rejected (Tie-break): valid and clean, but diff ({other.diff_size} lines) "
                f"is larger than winner Candidate #{winner.candidate_id} ({winner.diff_size} lines)"
            )

    # Build Markdown summary table
    table_lines = [
        "| Candidate | Target Tests | Full Suite | Lint Clean | Diff Size | Status | Reason |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in results:
        t_icon = "✅" if c.passes_target_tests else "❌"
        s_icon = "✅" if c.passes_full_suite else "❌"
        l_icon = "✅" if c.lint_clean else "❌"
        st_icon = "🏆 WINNER" if c.status == "accepted" else "❌ REJECTED"
        table_lines.append(
            f"| #{c.candidate_id} | {t_icon} | {s_icon} | {l_icon} | {c.diff_size} lines | {st_icon} | {c.reason} |"
        )
    summary_table = "\n".join(table_lines)

    logger.info(
        "Candidate evaluation complete: %d/%d qualifying. Winner: %s",
        len(qualifying),
        len(results),
        f"Candidate #{winner.candidate_id}" if winner else "None",
    )

    return EvaluationReport(
        candidates=results,
        winner=winner,
        total_candidates=len(results),
        qualifying_candidates=len(qualifying),
        summary_table=summary_table,
    )
