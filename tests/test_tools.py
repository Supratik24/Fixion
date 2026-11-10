"""
tests/test_tools.py — Unit tests for individual Fixion agent tools:
- read_file
- find_references (definition & callers)
- apply_patch (validate and apply unified diffs)
- run_linter
- git_ops (blame, log, diff)
- run_tests
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from agent.tools import apply_patch as ap_mod
from agent.tools import find_references as fr_mod
from agent.tools import git_ops as git_mod
from agent.tools import read_file as rf_mod
from agent.tools import run_linter as rl_mod
from agent.tools import run_tests as rt_mod
from ingest.ast_parser import CodeChunk
from ingest.symbol_graph import SymbolGraph
from sandbox.docker_runner import TestResult as DockerTestResult

# -----------------------------------------------------------------------------
# 1. read_file tool tests
# -----------------------------------------------------------------------------

def test_read_file_full_and_slice():
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        test_file = repo_root / "src" / "sample.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"line {i}\n" for i in range(1, 21)]
        test_file.write_text("".join(lines), encoding="utf-8")

        # 1. Slice lines 5 to 10 with 0 extra context
        res = rf_mod.read_file("src/sample.py", repo_root, start_line=5, end_line=10, context_lines=0)
        assert "error" not in res
        assert res["total_lines"] == 20
        assert res["start_line"] == 5
        assert res["end_line"] == 10
        assert "line 5" in str(res["content"])
        assert "line 10" in str(res["content"])
        content_lines = str(res["content"]).splitlines()
        assert not any("line 1\n" in line or line.strip().endswith("line 1") for line in content_lines)

        # 2. Non-existent file
        err_res = rf_mod.read_file("missing.py", repo_root)
        assert "error" in err_res
        assert "not found" in str(err_res["error"]).lower()


# -----------------------------------------------------------------------------
# 2. find_references tool tests
# -----------------------------------------------------------------------------

def test_find_references_and_definition():
    graph = SymbolGraph()
    chunk1 = CodeChunk(
        file_path="math_mod.py",
        symbol_name="calc_total",
        symbol_type="function",
        start_line=1,
        end_line=5,
        source="def calc_total(items):\n    return sum(items)\n",
        docstring="Sum items.",
        parent_class="",
    )
    chunk2 = CodeChunk(
        file_path="main.py",
        symbol_name="run_app",
        symbol_type="function",
        start_line=10,
        end_line=15,
        source="def run_app():\n    return calc_total([1, 2])\n",
        docstring="Main.",
        parent_class="",
    )
    graph.build_from_chunks([chunk1, chunk2])

    # 1. find_definition
    defn = fr_mod.find_definition("calc_total", graph)
    assert defn.get("symbol_name") == "calc_total"
    assert defn.get("file_path") == "math_mod.py"
    assert defn.get("start_line") == 1

    # 2. find_callers
    callers = fr_mod.find_callers("calc_total", graph)
    assert callers.get("symbol") == "calc_total"
    caller_names = [c["symbol_name"] for c in callers.get("callers", [])]
    assert "run_app" in caller_names

    # 3. Missing symbol
    missing = fr_mod.find_definition("unknown_func", graph)
    assert "error" in missing


# -----------------------------------------------------------------------------
# 3. apply_patch & validate_patch tool tests
# -----------------------------------------------------------------------------

def test_apply_and_validate_patch():
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        subprocess.run(["git", "init"], cwd=str(repo_root), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "FixionBot"], cwd=str(repo_root), check=True)
        subprocess.run(["git", "config", "user.email", "bot@fixion.dev"], cwd=str(repo_root), check=True)

        target = repo_root / "hello.py"
        target.write_text("def hello():\n    return 'bad'\n", encoding="utf-8")
        subprocess.run(["git", "add", "hello.py"], cwd=str(repo_root), check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo_root), check=True)

        clean_diff = """--- a/hello.py
+++ b/hello.py
@@ -1,2 +1,2 @@
 def hello():
-    return 'bad'
+    return 'good'
"""
        # Validate patch
        val = ap_mod.validate_patch(clean_diff, repo_root)
        assert val["valid"]

        # Apply patch
        app_res = ap_mod.apply_patch(clean_diff, repo_root)
        assert app_res["applied"]
        assert "hello.py" in app_res["files_changed"]
        assert "return 'good'" in target.read_text(encoding="utf-8")

        # Invalid diff rejected
        bad_val = ap_mod.validate_patch("MALFORMED DIFF", repo_root)
        assert not bad_val["valid"]


# -----------------------------------------------------------------------------
# 4. run_linter tool tests
# -----------------------------------------------------------------------------

def test_run_linter():
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        good_code = repo_root / "clean.py"
        good_code.write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")

        # Local ruff check fallback
        res = rl_mod.run_linter(repo_root, sandbox=None)
        assert "clean" in res
        assert "output" in res

        # Mocked sandbox check
        mock_sandbox = MagicMock()
        mock_sandbox.run_linter.return_value = {"passed": True, "output": "All clean"}
        sandbox_res = rl_mod.run_linter(repo_root, sandbox=mock_sandbox)
        assert sandbox_res["clean"] is True
        assert sandbox_res["output"] == "All clean"


# -----------------------------------------------------------------------------
# 5. git_ops tool tests (blame, log, diff)
# -----------------------------------------------------------------------------

def test_git_ops_blame_log_diff():
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        subprocess.run(["git", "init"], cwd=str(repo_root), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "FixionBot"], cwd=str(repo_root), check=True)
        subprocess.run(["git", "config", "user.email", "bot@fixion.dev"], cwd=str(repo_root), check=True)

        tracked = repo_root / "tracked.py"
        tracked.write_text("line 1\nline 2\n", encoding="utf-8")
        subprocess.run(["git", "add", "tracked.py"], cwd=str(repo_root), check=True)
        subprocess.run(["git", "commit", "-m", "first commit"], cwd=str(repo_root), check=True)

        # 1. git_log
        log_res = git_mod.git_log("tracked.py", repo_root, n=5)
        assert len(log_res["commits"]) >= 1
        assert log_res["commits"][0]["message"] == "first commit"

        # 2. git_blame
        blame_res = git_mod.git_blame("tracked.py", repo_root, start_line=1, end_line=2)
        assert blame_res["file_path"] == "tracked.py"
        assert len(blame_res["blame"]) >= 1

        # 3. git_diff
        tracked.write_text("line 1\nmodified line 2\n", encoding="utf-8")
        diff_res = git_mod.git_diff(repo_root)
        assert diff_res["has_changes"] is True
        assert "modified line 2" in diff_res["diff"]


# -----------------------------------------------------------------------------
# 6. run_tests tool tests
# -----------------------------------------------------------------------------

def test_run_tests_with_mock_sandbox():
    mock_sandbox = MagicMock()
    mock_sandbox.run_tests.return_value = DockerTestResult(
        passed=True,
        exit_code=0,
        stdout="1 passed in 0.05s",
        stderr="",
        traceback="",
        failed_tests=[],
        passed_tests=["tests/test_foo.py::test_bar"],
        duration_seconds=0.05,
    )

    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        res = rt_mod.run_tests(repo_root, sandbox=mock_sandbox, test_paths=["tests/test_foo.py::test_bar"])
        assert res["passed"] is True
        assert res["exit_code"] == 0
        assert res["passed_tests"] == ["tests/test_foo.py::test_bar"]
        assert mock_sandbox.run_tests.called
