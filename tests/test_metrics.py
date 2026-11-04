"""
tests/test_metrics.py — Unit tests for metrics tracking, ablation flags, and evaluation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.llm_client import LLMResponse, ToolCall
from agent.loop import AgentContext, AgentResult, _dispatch_tool, run_agent
from eval.metrics import EvalMetrics, IssueMetrics, estimate_cost


def test_estimate_cost():
    # Gemini pricing
    cost = estimate_cost("gemini-3.8-flash", "gemini-3.8-flash", 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.50, rel=1e-3)

    # Groq pricing
    cost_groq = estimate_cost("groq", "llama-3.3-70b-versatile", 1_000_000, 1_000_000)
    assert cost_groq == pytest.approx(1.38, rel=1e-3)

    # Ollama is free
    cost_ollama = estimate_cost("ollama", "qwen2.5-coder:7b", 1_000_000, 1_000_000)
    assert cost_ollama == 0.0


def test_issue_metrics_initialization():
    issue = IssueMetrics(
        instance_id="test__issue-1",
        resolved=True,
        patch_valid=True,
        iterations=2,
        correct_localization=True,
        duration_seconds=12.5,
        tokens_in=1000,
        tokens_out=250,
        tool_call_count=3,
        retries=1,
        status="resolved",
        backend="gemini-3.8-flash",
    )
    assert issue.wall_time == 12.5
    assert issue.cost_usd > 0.0
    d = issue.to_dict()
    assert d["instance_id"] == "test__issue-1"
    assert d["tokens_in"] == 1000
    assert d["status"] == "resolved"


def test_eval_metrics_aggregation():
    metrics = EvalMetrics()

    m1 = IssueMetrics(
        instance_id="inst-1",
        resolved=True,
        patch_valid=True,
        iterations=2,
        correct_localization=True,
        duration_seconds=10.0,
        tokens_in=500,
        tokens_out=200,
        tool_call_count=2,
        retries=0,
        status="resolved",
        backend="gemini",
    )
    m2 = IssueMetrics(
        instance_id="inst-2",
        resolved=False,
        patch_valid=True,
        iterations=4,
        correct_localization=True,
        duration_seconds=20.0,
        tokens_in=1500,
        tokens_out=600,
        tool_call_count=5,
        retries=2,
        status="unresolved",
        backend="gemini",
    )

    metrics.add(m1)
    metrics.add(m2)

    assert metrics.total == 2
    assert metrics.resolved == 1
    assert metrics.resolve_rate == 0.5
    assert metrics.patch_validity_rate == 1.0
    assert metrics.avg_iterations == 3.0
    assert metrics.avg_duration == 15.0
    assert metrics.avg_tokens_in == 1000.0
    assert metrics.avg_tokens_out == 400.0
    assert metrics.avg_tool_calls == 3.5
    assert metrics.avg_retries == 1.0

    summary = metrics.summary()
    assert summary["total_issues"] == 2
    assert summary["resolve_rate"] == "50.0%"

    md = metrics.markdown_table(backend="gemini")
    assert "| Backend | Total | Resolved |" in md
    assert "gemini" in md


def test_ablation_flags_in_tool_dispatch():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir)
        ctx = AgentContext(
            repo_root=repo_root,
            repo_url="https://github.com/example/repo",
            issue_text="Test issue",
            failing_tests=[],
            embed_index=None,
            symbol_graph=None,
            sandbox=None,
            use_symbol_graph=False,
            use_semantic_search=False,
        )

        # search_code should return ablation warning
        res_search = _dispatch_tool("search_code", {"query": "parse"}, ctx)
        assert "warning" in res_search
        assert "Semantic search is disabled" in res_search["warning"]

        # find_definition should return ablation warning
        res_def = _dispatch_tool("find_definition", {"symbol_name": "foo"}, ctx)
        assert "warning" in res_def
        assert "Symbol graph is disabled" in res_def["warning"]


def test_agent_loop_metrics_with_mock_llm():
    """Verify that run_agent tracks tokens, tool calls, iterations, retries, and wall time."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir)

        ctx = AgentContext(
            repo_root=repo_root,
            repo_url="https://github.com/example/repo",
            issue_text="ZeroDivisionError bug",
            failing_tests=["test_foo"],
            embed_index=None,
            symbol_graph=None,
            sandbox=None,
            max_retries=2,
            use_symbol_graph=False,
            use_semantic_search=False,
        )

        # Mock LLM responses: 1 turn with tool call, then final answer
        resp1 = LLMResponse(
            text="Let's inspect the file",
            tool_calls=[ToolCall(name="read_file", args={"file_path": "foo.py"})],
            prompt_tokens=120,
            completion_tokens=40,
        )
        resp2 = LLMResponse(
            text="ROOT CAUSE: bad divide\nFIX: raise error\nCONFIDENCE: high",
            tool_calls=[],
            prompt_tokens=200,
            completion_tokens=50,
        )

        with patch("agent.loop._call_gemini", side_effect=[(resp1, MagicMock()), (resp2, MagicMock())]), \
             patch("agent.loop._dispatch_tool", return_value={"content": "x = 1"}), \
             patch("agent.tools.run_tests.run_tests", return_value={"passed": True, "passed_tests": ["test_foo"]}):

            result = run_agent(ctx)

            assert isinstance(result, AgentResult)
            assert result.iterations == 2
            assert "metrics" in result.__dict__ or hasattr(result, "metrics")

            m = result.metrics
            assert m["tokens_in"] == 320
            assert m["tokens_out"] == 90
            assert m["total_tokens"] == 410
            assert m["tool_call_count"] == 1
            assert m["iterations"] == 2
            assert m["wall_time"] >= 0.0
            assert m["status"] in ("resolved", "unresolved")
