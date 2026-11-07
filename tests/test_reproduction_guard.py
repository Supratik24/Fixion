"""
tests/test_reproduction_guard.py — Tests for Phase 3:
- reproduce_issue tool (confirmed failing vs unexpectedly passed)
- Pre-patch reproduction guard (blocks apply_patch until reproduction confirmed)
- Regression guard (full suite run, patch rejection and git reversion)
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

from agent.loop import AgentContext, _dispatch_tool, _process_test_result
from agent.tools.reproduce_issue import reproduce_issue

# ── 1. reproduce_issue Tool Tests ──────────────────────────────────────────────


def test_reproduce_issue_confirmed_failing():
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        test_code = "def test_fail():\n    assert 1 == 2\n"

        # Mock run_tests returning failure
        with patch("agent.tools.reproduce_issue.run_tests") as mock_rt:
            mock_rt.return_value = {
                "passed": False,
                "exit_code": 1,
                "failed_tests": ["tests/test_reproduce_issue.py::test_fail"],
                "passed_tests": [],
                "traceback": "AssertionError: assert 1 == 2",
            }
            res = reproduce_issue(
                test_code=test_code,
                repo_root=repo_root,
                sandbox=None,
                test_file_path="tests/test_reproduce_issue.py",
            )

        assert res["reproduced"] is True
        assert res["status"] == "confirmed_failing"
        assert (repo_root / "tests" / "test_reproduce_issue.py").exists()
        assert "AssertionError" in res["traceback"]


def test_reproduce_issue_unexpectedly_passed():
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        test_code = "def test_pass():\n    assert 1 == 1\n"

        with patch("agent.tools.reproduce_issue.run_tests") as mock_rt:
            mock_rt.return_value = {
                "passed": True,
                "exit_code": 0,
                "failed_tests": [],
                "passed_tests": ["tests/test_reproduce_issue.py::test_pass"],
                "traceback": "",
            }
            res = reproduce_issue(
                test_code=test_code,
                repo_root=repo_root,
                sandbox=None,
                test_file_path="tests/test_reproduce_issue.py",
            )

        assert res["reproduced"] is False
        assert res["status"] == "unexpectedly_passed"
        assert "PASSED unexpectedly" in res["message"]


def test_reproduce_issue_collection_error():
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        test_code = "def helper_function():\n    pass\n"

        with patch("agent.tools.reproduce_issue.run_tests") as mock_rt:
            mock_rt.return_value = {
                "passed": False,
                "exit_code": 5,
                "failed_tests": [],
                "passed_tests": [],
                "traceback": "",
            }
            res = reproduce_issue(
                test_code=test_code,
                repo_root=repo_root,
                sandbox=None,
                test_file_path="tests/test_reproduce_issue.py",
            )

        assert res["reproduced"] is False
        assert res["status"] == "collection_error"
        assert "No tests were collected" in res["message"]


# ── 2. Pre-Patch Reproduction Guard Tests ─────────────────────────────────────


def test_apply_patch_blocked_without_reproduction():
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        ctx = AgentContext(
            repo_root=repo_root,
            repo_url="https://github.com/example/repo",
            issue_text="Issue text",
            failing_tests=[],
            require_reproduction=True,
            reproduction_confirmed=False,
        )

        patch_text = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-a\n+b\n"
        res = _dispatch_tool("apply_patch", {"patch_text": patch_text}, ctx)

        assert res["applied"] is False
        assert "Reproduction guard" in res["error"]


def test_apply_patch_allowed_after_reproduction():
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        # Initialize a git repo so git apply can work
        subprocess.run(["git", "init"], cwd=str(repo_root), capture_output=True)
        subprocess.run(["git", "config", "user.name", "FixionBot"], cwd=str(repo_root), capture_output=True)
        subprocess.run(["git", "config", "user.email", "bot@fixion.dev"], cwd=str(repo_root), capture_output=True)
        (repo_root / "foo.py").write_text("a = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "foo.py"], cwd=str(repo_root), capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo_root), capture_output=True)

        ctx = AgentContext(
            repo_root=repo_root,
            repo_url="https://github.com/example/repo",
            issue_text="Issue text",
            failing_tests=["tests/test_foo.py"],
            require_reproduction=True,
            reproduction_confirmed=True,  # Confirmed!
        )

        patch_text = """--- a/foo.py
+++ b/foo.py
@@ -1 +1 @@
-a = 1
+a = 2
"""
        res = _dispatch_tool("apply_patch", {"patch_text": patch_text}, ctx)
        assert res["applied"] is True
        assert (repo_root / "foo.py").read_text(encoding="utf-8") == "a = 2\n"


def test_dispatch_reproduce_issue_sets_context_flag():
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        ctx = AgentContext(
            repo_root=repo_root,
            repo_url="https://github.com/example/repo",
            issue_text="Issue text",
            failing_tests=[],
            require_reproduction=True,
            reproduction_confirmed=False,
        )

        with patch("agent.tools.reproduce_issue.run_tests") as mock_rt:
            mock_rt.return_value = {
                "passed": False,
                "exit_code": 1,
                "failed_tests": ["tests/test_reproduce_issue.py::test_fail"],
                "passed_tests": [],
                "traceback": "Error",
            }
            res = _dispatch_tool(
                "reproduce_issue",
                {"test_code": "def test_fail(): assert False", "test_file_path": "tests/test_reproduce_issue.py"},
                ctx,
            )

        assert res["reproduced"] is True
        assert ctx.reproduction_confirmed is True
        assert ctx.reproduction_test_path == "tests/test_reproduce_issue.py"
        assert "tests/test_reproduce_issue.py" in ctx.failing_tests


# ── 3. Regression Guard Tests ──────────────────────────────────────────────────


def test_regression_guard_rejects_and_reverts():
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        # Initialize git repo
        subprocess.run(["git", "init"], cwd=str(repo_root), capture_output=True)
        subprocess.run(["git", "config", "user.name", "FixionBot"], cwd=str(repo_root), capture_output=True)
        subprocess.run(["git", "config", "user.email", "bot@fixion.dev"], cwd=str(repo_root), capture_output=True)
        (repo_root / "code.py").write_text("orig = True\n", encoding="utf-8")
        subprocess.run(["git", "add", "code.py"], cwd=str(repo_root), capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo_root), capture_output=True)

        ctx = AgentContext(
            repo_root=repo_root,
            repo_url="https://github.com/example/repo",
            issue_text="Bug",
            failing_tests=["tests/test_target.py::test_target"],
            baseline_passed_tests={"tests/test_existing.py::test_clean"},
            reproduction_confirmed=True,
            reproduction_test_path="tests/test_reproduce.py",
        )

        # Create reproduction test file
        (repo_root / "tests").mkdir(parents=True, exist_ok=True)
        (repo_root / "tests" / "test_reproduce.py").write_text("def test_repro(): pass\n", encoding="utf-8")

        # Simulate applied patch that modified code.py
        (repo_root / "code.py").write_text("modified = True\n", encoding="utf-8")
        last_patch = "--- a/code.py\n+++ b/code.py\n@@ -1 +1 @@\n-orig = True\n+modified = True\n"

        target_test_result = {
            "passed": True,
            "failed_tests": [],
            "passed_tests": ["tests/test_target.py::test_target"],
        }

        # Mock full suite run returning regression on test_clean
        with patch("agent.tools.run_tests.run_tests") as mock_full_suite:
            mock_full_suite.return_value = {
                "passed": False,
                "failed_tests": ["tests/test_existing.py::test_clean"],
                "passed_tests": ["tests/test_target.py::test_target"],
                "traceback": "Regression error in existing test",
            }

            processed, _, retries = _process_test_result(
                target_test_result, last_patch=last_patch, iterations=2, retries=0, ctx=ctx
            )

        assert processed["passed"] is False
        assert processed["regression_detected"] is True
        assert "tests/test_existing.py::test_clean" in processed["regressed_tests"]
        assert retries == 1

        # Verify git working tree was reverted: code.py should be back to "orig = True"
        assert (repo_root / "code.py").read_text(encoding="utf-8") == "orig = True\n"
        # Verify reproduction test was preserved!
        assert (repo_root / "tests" / "test_reproduce.py").exists()


def test_regression_guard_accepts_clean_run():
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        ctx = AgentContext(
            repo_root=repo_root,
            repo_url="https://github.com/example/repo",
            issue_text="Bug",
            failing_tests=["tests/test_target.py::test_target"],
            baseline_passed_tests={"tests/test_existing.py::test_clean"},
            reproduction_confirmed=True,
        )

        target_test_result = {
            "passed": True,
            "failed_tests": [],
            "passed_tests": ["tests/test_target.py::test_target"],
        }

        with patch("agent.tools.run_tests.run_tests") as mock_full_suite:
            mock_full_suite.return_value = {
                "passed": True,
                "failed_tests": [],
                "passed_tests": ["tests/test_target.py::test_target", "tests/test_existing.py::test_clean"],
            }

            processed, _, retries = _process_test_result(
                target_test_result, last_patch="--- diff ---", iterations=2, retries=0, ctx=ctx
            )

        assert processed["passed"] is True
        assert processed["full_suite_passed"] is True
        assert "tests/test_existing.py::test_clean" in processed["passed_tests"]
        assert retries == 0
