"""
agent/tools/reproduce_issue.py — Write and run a minimal reproduction test.

The agent calls this tool in Phase 1 before attempting any patch:
1. Writes a minimal failing test to the repository (e.g. tests/test_reproduce_issue.py).
2. Executes the test inside the sandbox.
3. Confirms that the test fails on the unpatched codebase.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent.tools.run_tests import run_tests
from sandbox.docker_runner import DockerSandbox

logger = logging.getLogger(__name__)


def reproduce_issue(
    test_code: str,
    repo_root: Path,
    sandbox: DockerSandbox | None = None,
    test_file_path: str = "tests/test_reproduce_issue.py",
) -> dict[str, Any]:
    """
    Write a minimal reproduction test and verify that it fails against the unpatched codebase.

    Parameters
    ----------
    test_code:
        Complete Python test code (must contain test functions starting with `test_`).
    repo_root:
        Local root directory of the repository.
    sandbox:
        DockerSandbox instance or None.
    test_file_path:
        Relative path where the reproduction test should be saved (default: "tests/test_reproduce_issue.py").

    Returns
    -------
    dict with:
        - reproduced (bool): True if test executed and failed as expected.
        - status (str): "confirmed_failing" | "unexpectedly_passed" | "collection_error" | "error"
        - test_file_path (str): Relative path to written test.
        - failed_tests (list[str]): List of failing test node IDs.
        - passed_tests (list[str]): List of passing test node IDs.
        - traceback (str): Traceback output if test failed.
        - message (str): Human- and LLM-readable summary of the outcome.
    """
    # 1. Normalize path
    target = Path(test_file_path)
    if target.is_absolute():
        try:
            rel_path = str(target.relative_to(repo_root)).replace("\\", "/")
        except ValueError:
            rel_path = target.name
    else:
        rel_path = str(test_file_path).replace("\\", "/")

    abs_path = repo_root / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Write the test code
    try:
        abs_path.write_text(test_code, encoding="utf-8")
    except Exception as e:
        return {
            "reproduced": False,
            "status": "error",
            "test_file_path": rel_path,
            "error_message": f"Failed writing reproduction test file: {e}",
            "message": f"Could not write reproduction test: {e}",
        }

    # 3. Execute the reproduction test
    test_res = run_tests(repo_root, sandbox, test_paths=[rel_path])

    exit_code = test_res.get("exit_code", 1)
    passed = test_res.get("passed", False)
    failed_tests = test_res.get("failed_tests", [])
    passed_tests = test_res.get("passed_tests", [])
    traceback = test_res.get("traceback", "")

    # Exit code 5 in pytest means no tests collected
    if exit_code == 5 or (not passed and not failed_tests and not traceback):
        return {
            "reproduced": False,
            "status": "collection_error",
            "test_file_path": rel_path,
            "exit_code": exit_code,
            "failed_tests": [],
            "passed_tests": passed_tests,
            "traceback": traceback,
            "message": (
                f"No tests were collected from {rel_path}. "
                "Ensure your test functions are named with the 'test_' prefix."
            ),
        }

    # If the test passed, it failed to reproduce the bug!
    if passed:
        return {
            "reproduced": False,
            "status": "unexpectedly_passed",
            "test_file_path": rel_path,
            "exit_code": exit_code,
            "failed_tests": [],
            "passed_tests": passed_tests,
            "message": (
                f"Reproduction test at {rel_path} PASSED unexpectedly! "
                "A reproduction test MUST fail on the unpatched codebase to prove the bug exists. "
                "Please refine the test assertions or inputs to expose the defect."
            ),
        }

    # The test failed! Bug reproduced successfully
    return {
        "reproduced": True,
        "status": "confirmed_failing",
        "test_file_path": rel_path,
        "exit_code": exit_code,
        "failed_tests": failed_tests or [rel_path],
        "passed_tests": passed_tests,
        "traceback": traceback,
        "message": (
            f"Issue successfully reproduced at {rel_path}! "
            "The test failed on the unpatched codebase as expected. "
            "You may now proceed to localize the root cause and apply a patch."
        ),
    }


# ── Tool declaration ──────────────────────────────────────────────────────────

TOOL_DECLARATION = {
    "name": "reproduce_issue",
    "description": (
        "Write a minimal reproduction test file and run it inside the sandbox. "
        "MANDATORY first step: you must confirm the test FAILS on the unpatched codebase "
        "before attempting any patch. "
        "Returns reproduced=True when the test fails as expected."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "test_code": {
                "type": "string",
                "description": (
                    "Full Python test code containing one or more test functions (e.g. def test_bug(): ...) "
                    "that assert the expected behavior and fail due to the reported issue."
                ),
            },
            "test_file_path": {
                "type": "string",
                "description": (
                    "Relative path where the reproduction test file should be saved "
                    "(default: 'tests/test_reproduce_issue.py')."
                ),
            },
        },
        "required": ["test_code"],
    },
}
