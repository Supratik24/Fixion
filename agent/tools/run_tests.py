"""
agent/tools/run_tests.py — Run the repo's test suite in the Docker sandbox.

The agent calls this to:
1. Reproduce a bug (run a failing test to get a real traceback)
2. Verify a patch (run tests after applying the diff)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sandbox.docker_runner import DockerSandbox, TestResult


def run_tests(
    repo_path: Path,
    sandbox: DockerSandbox,
    test_paths: list[str] | None = None,
    *,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run the test suite (or specific tests) inside the sandbox.

    Parameters
    ----------
    repo_path:
        Local path to the repo (with any patches already applied).
    sandbox:
        DockerSandbox instance to use.
    test_paths:
        Specific pytest node IDs to run, e.g.
        ``["tests/test_stats.py::test_division"]``.
        If None, runs the full suite (slow — prefer specific tests).
    extra_args:
        Additional pytest arguments, e.g. ``["-x", "--no-header"]``.

    Returns
    -------
    dict with:
        - passed (bool)
        - exit_code (int)
        - traceback (str) — last traceback if tests failed
        - failed_tests (list[str])
        - passed_tests (list[str])
        - duration_seconds (float)
        - error_message (str) — set if sandbox itself failed
    """
    result: TestResult = sandbox.run_tests(
        repo_path,
        test_paths=test_paths,
        extra_pytest_args=extra_args,
    )
    return {
        "passed": result.passed,
        "exit_code": result.exit_code,
        "traceback": result.traceback,
        "failed_tests": result.failed_tests,
        "passed_tests": result.passed_tests,
        "duration_seconds": round(result.duration_seconds, 2),
        "error_message": result.error_message,
        "stdout_preview": result.stdout[:2000],  # first 2k chars for context
    }


# ── Gemini tool declaration ────────────────────────────────────────────────────

TOOL_DECLARATION = {
    "name": "run_tests",
    "description": (
        "Run the repository's test suite (or specific tests) inside a Docker sandbox. "
        "Use FIRST to reproduce the failing test and get a real traceback. "
        "Use AFTER applying a patch to verify it passes. "
        "Specify test_paths to run only relevant tests (much faster than the full suite)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "test_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Pytest node IDs to run, e.g. ['tests/test_foo.py::TestClass::test_bar']. "
                    "Leave empty to run the full suite (slow)."
                ),
            },
            "extra_args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Extra pytest arguments, e.g. ['-x', '-v'].",
            },
        },
        "required": [],
    },
}
