"""
tests/test_local_sandbox.py — Tests for the LocalSandbox fallback runner.
"""

from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

from sandbox.docker_runner import DockerSandbox, LocalSandbox


def test_local_sandbox_run_tests_success_and_failure():
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        src_dir = repo_root / "sample_pkg"
        tests_dir = repo_root / "tests"
        src_dir.mkdir(parents=True)
        tests_dir.mkdir(parents=True)

        (src_dir / "__init__.py").write_text("", encoding="utf-8")
        (src_dir / "calc.py").write_text(
            textwrap.dedent("""\
                def add(a, b):
                    return a + b

                def buggy_sub(a, b):
                    return a + b  # deliberate bug
            """),
            encoding="utf-8",
        )

        (tests_dir / "test_calc.py").write_text(
            textwrap.dedent("""\
                from sample_pkg.calc import add, buggy_sub

                def test_add():
                    assert add(2, 3) == 5

                def test_sub():
                    assert buggy_sub(5, 3) == 2
            """),
            encoding="utf-8",
        )

        sandbox = LocalSandbox(timeout=30)
        sandbox.build_image()

        # Run full suite -> should fail on test_sub
        res = sandbox.run_tests(repo_root)
        assert res.passed is False
        assert res.exit_code != 0
        assert any("test_sub" in f for f in res.failed_tests)
        assert any("test_add" in p for p in res.passed_tests)
        assert len(res.traceback) > 0

        # Run specific passing test
        res_passing = sandbox.run_tests(repo_root, test_paths=["tests/test_calc.py::test_add"])
        assert res_passing.passed is True
        assert res_passing.exit_code == 0
        assert any("test_add" in p for p in res_passing.passed_tests)


def test_docker_sandbox_fallback_when_daemon_unavailable():
    # Instantiating DockerSandbox without a running docker daemon should fall back to LocalSandbox
    sb = DockerSandbox(fallback_to_local=True)
    # If docker daemon happens to be running on the machine, is_docker_available might be True or False,
    # but build_image and run_tests must not raise an unhandled DockerException.
    sb.build_image()

    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        (repo_root / "test_simple.py").write_text("def test_ok(): assert 1 == 1\n", encoding="utf-8")
        res = sb.run_tests(repo_root)
        assert res.passed is True
        assert res.exit_code == 0


def test_local_sandbox_run_linter():
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        (repo_root / "good.py").write_text("x = 1\n", encoding="utf-8")
        sandbox = LocalSandbox()
        res = sandbox.run_linter(repo_root)
        assert "passed" in res
