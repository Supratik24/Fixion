"""
tests/test_agent_loop_mock.py — End-to-end agent loop execution with a mock LLM.

Simulates a complete 5-turn SWE agent session:
1. Turn 1: Calls reproduce_issue -> reproduction confirmed failing
2. Turn 2: Calls read_file -> inspects buggy source code
3. Turn 3: Calls apply_patch -> applies unified diff
4. Turn 4: Calls run_tests -> test suite passes
5. Turn 5: Emits final resolution text with root cause and fix explanation
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from google.genai import types as genai_types

from agent.llm_client import LLMResponse, ToolCall
from agent.loop import AgentContext, run_agent
from sandbox.docker_runner import TestResult as DockerTestResult


def test_agent_loop_end_to_end_mock():
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        subprocess.run(["git", "init"], cwd=str(repo_root), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "FixionBot"], cwd=str(repo_root), check=True)
        subprocess.run(["git", "config", "user.email", "bot@fixion.dev"], cwd=str(repo_root), check=True)

        src_file = repo_root / "calc.py"
        src_file.write_text(
            "def safe_div(a, b):\n    if b == 0:\n        return 0\n    return a / b\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "add", "calc.py"], cwd=str(repo_root), check=True)
        subprocess.run(["git", "commit", "-m", "init calc"], cwd=str(repo_root), check=True)

        diff = """--- a/calc.py
+++ b/calc.py
@@ -1,4 +1,4 @@
 def safe_div(a, b):
     if b == 0:
-        return 0
+        raise ZeroDivisionError('zero denominator')
     return a / b
"""

        # Turn sequence for mock LLM
        turns = [
            # Turn 1: reproduce
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        name="reproduce_issue",
                        args={
                            "test_code": "import pytest\nfrom calc import safe_div\ndef test_div_zero():\n    with pytest.raises(ZeroDivisionError):\n        safe_div(1, 0)\n",
                            "test_file_path": "tests/test_repro.py",
                        },
                    )
                ],
                prompt_tokens=150,
                completion_tokens=40,
            ),
            # Turn 2: read_file
            LLMResponse(
                text="",
                tool_calls=[ToolCall(name="read_file", args={"file_path": "calc.py"})],
                prompt_tokens=180,
                completion_tokens=30,
            ),
            # Turn 3: apply_patch
            LLMResponse(
                text="",
                tool_calls=[ToolCall(name="apply_patch", args={"patch_text": diff})],
                prompt_tokens=220,
                completion_tokens=50,
            ),
            # Turn 4: run_tests
            LLMResponse(
                text="",
                tool_calls=[ToolCall(name="run_tests", args={})],
                prompt_tokens=260,
                completion_tokens=25,
            ),
            # Turn 5: final resolution
            LLMResponse(
                text=(
                    "ROOT CAUSE: safe_div silently returned 0 on zero division.\n"
                    "FIX EXPLANATION: Raised ZeroDivisionError on b=0."
                ),
                tool_calls=[],
                prompt_tokens=300,
                completion_tokens=60,
            ),
        ]

        turn_index = 0

        def fake_call_gemini(history, system_prompt, gemini_tools):
            nonlocal turn_index
            resp = turns[min(turn_index, len(turns) - 1)]
            turn_index += 1
            raw_content = genai_types.Content(
                role="model",
                parts=[genai_types.Part(text=resp.text or "calling tool...")],
            )
            return resp, raw_content

        # Mock Sandbox to simulate reproduction fail then patch pass
        mock_sandbox = MagicMock()
        sandbox_call_count = 0

        def fake_sandbox_run_tests(*args, **kwargs):
            nonlocal sandbox_call_count
            sandbox_call_count += 1
            if sandbox_call_count == 1:
                # Call 1: reproduce_issue verifies the bug reproduces (fails)
                return DockerTestResult(
                    passed=False,
                    exit_code=1,
                    stdout="1 failed in 0.02s",
                    stderr="",
                    traceback="ZeroDivisionError not raised",
                    failed_tests=["tests/test_repro.py::test_div_zero"],
                    passed_tests=[],
                    duration_seconds=0.02,
                )
            # Subsequent calls (regression baseline, post-patch test, regression check, final parse) pass
            return DockerTestResult(
                passed=True,
                exit_code=0,
                stdout="1 passed in 0.02s",
                stderr="",
                traceback="",
                failed_tests=[],
                passed_tests=["tests/test_repro.py::test_div_zero"],
                duration_seconds=0.02,
            )

        mock_sandbox.run_tests.side_effect = fake_sandbox_run_tests

        ctx = AgentContext(
            repo_root=repo_root,
            repo_url="https://github.com/test-org/calc-repo",
            issue_text="safe_div(1, 0) should raise ZeroDivisionError, not return 0",
            failing_tests=["tests/test_repro.py::test_div_zero"],
            sandbox=mock_sandbox,
            max_retries=5,
            require_reproduction=True,
        )

        with patch("agent.loop._call_gemini", side_effect=fake_call_gemini):
            result = run_agent(ctx)

            # Assertions
            assert result.success is True
            assert "raise ZeroDivisionError" in result.patch
            assert "safe_div" in result.root_cause or "safe_div" in result.fix_explanation
            assert result.iterations >= 4
            assert result.metrics["tokens_in"] > 0
            assert result.metrics["tokens_out"] > 0
            assert result.metrics["tool_call_count"] >= 4
            assert result.metrics["wall_time"] >= 0
