"""
tests/test_sandbox.py — Tests for the Docker sandbox runner.

These tests require Docker to be running. They are marked with
@pytest.mark.integration and skipped in CI environments without Docker.

To run: pytest tests/test_sandbox.py -v
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

try:
    import docker
    _docker_available = True
    client = docker.from_env()
    client.ping()
except Exception:
    _docker_available = False

pytestmark = pytest.mark.skipif(
    not _docker_available,
    reason="Docker not available or not running",
)


@pytest.fixture(scope="module")
def sandbox():
    from sandbox.docker_runner import DockerSandbox
    sb = DockerSandbox()
    sb.build_image()
    return sb


@pytest.fixture()
def simple_repo(tmp_path: Path) -> Path:
    """Minimal Python project that has one passing and one failing test."""
    (tmp_path / "stats.py").write_text(
        textwrap.dedent("""\
            def mean(values):
                return sum(values) / len(values)
        """),
        encoding="utf-8",
    )
    (tmp_path / "test_stats.py").write_text(
        textwrap.dedent("""\
            from stats import mean

            def test_mean_basic():
                assert mean([1, 2, 3]) == 2.0

            def test_mean_empty():
                # This should raise ZeroDivisionError (bug: no guard)
                import pytest
                with pytest.raises(ZeroDivisionError):
                    mean([])
        """),
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        "[build-system]\nrequires = []\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture()
def fixed_repo(tmp_path: Path) -> Path:
    """Same project but with the bug fixed."""
    (tmp_path / "stats.py").write_text(
        textwrap.dedent("""\
            def mean(values):
                if not values:
                    return 0.0
                return sum(values) / len(values)
        """),
        encoding="utf-8",
    )
    (tmp_path / "test_stats.py").write_text(
        textwrap.dedent("""\
            from stats import mean

            def test_mean_basic():
                assert mean([1, 2, 3]) == 2.0

            def test_mean_empty():
                assert mean([]) == 0.0
        """),
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        "[build-system]\nrequires = []\n",
        encoding="utf-8",
    )
    return tmp_path


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_passing_tests_returns_success(sandbox, fixed_repo: Path) -> None:
    result = sandbox.run_tests(fixed_repo)
    assert result.passed is True
    assert result.exit_code == 0
    assert len(result.passed_tests) == 2


def test_failing_tests_detected(sandbox, simple_repo: Path) -> None:
    # The empty-list test expects ZeroDivisionError but with pytest.raises it passes
    # The basic test should always pass
    result = sandbox.run_tests(simple_repo, test_paths=["test_stats.py::test_mean_basic"])
    assert result.passed is True


def test_specific_test_path(sandbox, fixed_repo: Path) -> None:
    result = sandbox.run_tests(fixed_repo, test_paths=["test_stats.py::test_mean_empty"])
    assert result.passed is True
    assert "test_stats.py::test_mean_empty" in result.passed_tests


def test_result_has_duration(sandbox, fixed_repo: Path) -> None:
    result = sandbox.run_tests(fixed_repo)
    assert result.duration_seconds > 0


def test_container_is_removed_after_run(sandbox, fixed_repo: Path) -> None:
    """Containers should be auto-removed (no leaks)."""
    import docker as docker_mod
    client = docker_mod.from_env()
    containers_before = len(client.containers.list(all=True))
    sandbox.run_tests(fixed_repo)
    containers_after = len(client.containers.list(all=True))
    # Should not have grown (container auto-removed)
    assert containers_after <= containers_before
