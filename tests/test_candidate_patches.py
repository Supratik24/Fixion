"""
tests/test_candidate_patches.py — Tests for Phase 4:
- calculate_diff_size
- Competitive multi-candidate evaluation in isolated containers
- Selection ranking: passes target tests -> passes full suite -> lint-clean -> smallest diff
- Acceptance and rejection reason logging for each candidate
- evaluate_candidates tool execution and winner application
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

from agent.candidate_evaluator import (
    calculate_diff_size,
    evaluate_candidates_in_containers,
    evaluate_single_candidate,
)
from agent.loop import AgentContext, _dispatch_tool
from agent.tools.evaluate_candidates import evaluate_candidates


def test_calculate_diff_size():
    diff = """--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,4 @@
 def foo():
-    return 1
+    # fix
+    return 2
"""
    # 1 removed line (- return 1), 2 added lines (+ # fix, + return 2) = 3
    assert calculate_diff_size(diff) == 3


def test_evaluate_single_candidate_rejected_on_bad_diff():
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        bad_diff = "MALFORMED DIFF NOT UNIFIED FORMAT"
        res = evaluate_single_candidate(1, bad_diff, repo_root)
        assert res.status == "rejected"
        assert "failed to apply" in res.reason


def test_evaluate_candidates_selection_smallest_diff():
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        # Initialize a git repo
        subprocess.run(["git", "init"], cwd=str(repo_root), capture_output=True)
        subprocess.run(["git", "config", "user.name", "FixionBot"], cwd=str(repo_root), capture_output=True)
        subprocess.run(["git", "config", "user.email", "bot@fixion.dev"], cwd=str(repo_root), capture_output=True)
        (repo_root / "calc.py").write_text("def calc(): return 0\n", encoding="utf-8")
        subprocess.run(["git", "add", "calc.py"], cwd=str(repo_root), capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo_root), capture_output=True)

        cand1_fails_test = """--- a/calc.py
+++ b/calc.py
@@ -1 +1 @@
-def calc(): return 0
+def calc(): return -1
"""
        cand2_large_diff = """--- a/calc.py
+++ b/calc.py
@@ -1 +1,5 @@
-def calc(): return 0
+# refactor whole function
+def calc():
+    # comment
+    x = 10
+    return x
"""
        cand3_minimal_diff = """--- a/calc.py
+++ b/calc.py
@@ -1 +1 @@
-def calc(): return 0
+def calc(): return 10
"""

        def mock_run_tests(repo_path, sandbox, test_paths=None, **kwargs):
            content = (repo_path / "calc.py").read_text(encoding="utf-8")
            if "return -1" in content:
                # Cand 1 fails target test
                return {"passed": False, "failed_tests": ["test_calc"], "passed_tests": []}
            # Cand 2 and 3 pass target tests and full suite
            return {"passed": True, "failed_tests": [], "passed_tests": ["test_calc", "test_other"]}

        with patch("agent.candidate_evaluator.rt_mod.run_tests", side_effect=mock_run_tests), \
             patch("agent.candidate_evaluator.lint_mod.run_linter", return_value={"clean": True, "output": ""}):

            report = evaluate_candidates_in_containers(
                candidates=[cand1_fails_test, cand2_large_diff, cand3_minimal_diff],
                repo_root=repo_root,
                sandbox=None,
                target_tests=["test_calc"],
                baseline_passed_tests={"test_other"},
            )

        assert report.total_candidates == 3
        assert report.qualifying_candidates == 2  # Cand 2 and Cand 3 qualified
        assert report.winner is not None
        assert report.winner.candidate_id == 3  # Cand 3 wins because smallest diff!
        assert "Winner" in report.winner.reason

        # Cand 1 rejected because failed test
        c1 = report.candidates[0]
        assert c1.status == "rejected"
        assert "failed target test" in c1.reason

        # Cand 2 rejected by tie-break because larger diff
        c2 = report.candidates[1]
        assert c2.status == "rejected"
        assert "Tie-break" in c2.reason
        assert "larger than winner Candidate #3" in c2.reason

        # Verify Markdown summary table contains all candidates
        assert "#1" in report.summary_table
        assert "#2" in report.summary_table
        assert "#3" in report.summary_table
        assert "WINNER" in report.summary_table


def test_evaluate_candidates_tool_applies_winner():
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        subprocess.run(["git", "init"], cwd=str(repo_root), capture_output=True)
        subprocess.run(["git", "config", "user.name", "FixionBot"], cwd=str(repo_root), capture_output=True)
        subprocess.run(["git", "config", "user.email", "bot@fixion.dev"], cwd=str(repo_root), capture_output=True)
        (repo_root / "calc.py").write_text("def calc(): return 0\n", encoding="utf-8")
        subprocess.run(["git", "add", "calc.py"], cwd=str(repo_root), capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo_root), capture_output=True)

        patch_valid = """--- a/calc.py
+++ b/calc.py
@@ -1 +1 @@
-def calc(): return 0
+def calc(): return 42
"""

        with patch("agent.candidate_evaluator.rt_mod.run_tests", return_value={"passed": True, "failed_tests": [], "passed_tests": ["t1"]}), \
             patch("agent.candidate_evaluator.lint_mod.run_linter", return_value={"clean": True, "output": ""}):

            res = evaluate_candidates(
                candidates=[patch_valid],
                repo_root=repo_root,
                sandbox=None,
                apply_winner=True,
            )

        assert res["winner_found"] is True
        assert res["winner_applied"] is True
        assert res["winner_id"] == 1
        assert (repo_root / "calc.py").read_text(encoding="utf-8") == "def calc(): return 42\n"


def test_evaluate_candidates_reproduction_guard():
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        ctx = AgentContext(
            repo_root=repo_root,
            repo_url="https://github.com/example/repo",
            issue_text="Issue",
            failing_tests=[],
            require_reproduction=True,
            reproduction_confirmed=False,  # Unconfirmed
        )

        res = _dispatch_tool(
            "evaluate_candidates",
            {"candidates": ["--- diff ---"]},
            ctx,
        )

        assert res["winner_found"] is False
        assert "Reproduction guard" in res["error"]
