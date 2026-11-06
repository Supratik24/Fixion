"""
tests/test_localization.py — Unit tests for smarter localization:
- Traceback parser (Python & pytest)
- BM25 lexical search & code tokenization
- Hybrid retrieval with Reciprocal Rank Fusion (RRF)
- Structured failure reflection on failed retries
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from agent.loop import AgentContext, _process_test_result, build_failure_reflection
from agent.tools.search_code import search_code
from ingest.ast_parser import CodeChunk
from ingest.bm25 import BM25Index, tokenize_code
from ingest.embed_index import EmbedIndex
from ingest.traceback_parser import parse_traceback

# ── 1. Traceback Parser Tests ──────────────────────────────────────────────────


def test_parse_traceback_standard_python():
    tb_text = """
Traceback (most recent call last):
  File "/app/repo/src/stats.py", line 42, in calculate_average
    return sum(nums) / len(nums)
ZeroDivisionError: division by zero
"""
    parsed = parse_traceback(tb_text, repo_root="/app/repo")
    assert parsed.error_type == "ZeroDivisionError"
    assert "division by zero" in parsed.error_message
    assert len(parsed.frames) >= 1
    assert parsed.primary_frame is not None
    assert parsed.primary_frame.file_path == "src/stats.py"
    assert parsed.primary_frame.line_number == 42
    assert parsed.primary_frame.function_name == "calculate_average"
    assert parsed.primary_frame.source_line == "return sum(nums) / len(nums)"


def test_parse_traceback_pytest_format():
    tb_text = """
_____________________________ test_divide_zero _____________________________
tests/test_math.py:15: in test_divide_zero
    assert divide(10, 0) == 0
buggy_math/math_utils.py:12: in divide
    return a / b
E   ZeroDivisionError: division by zero
"""
    parsed = parse_traceback(tb_text)
    assert parsed.error_type == "ZeroDivisionError"
    assert parsed.primary_frame is not None
    # Primary frame should prefer the implementation over the test file
    assert parsed.primary_frame.file_path == "buggy_math/math_utils.py"
    assert parsed.primary_frame.line_number == 12
    assert parsed.primary_frame.function_name == "divide"


def test_parse_traceback_assertion_error():
    tb_text = """
tests/test_calc.py:20: in test_add
    assert add(2, 2) == 5
E   AssertionError: assert 4 == 5
"""
    parsed = parse_traceback(tb_text)
    assert parsed.error_type == "AssertionError"
    assert parsed.failing_assertion == "assert 4 == 5"


def test_parse_traceback_filters_framework_frames():
    tb_text = """
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/_pytest/runner.py", line 120, in pytest_runtest_call
    item.runtest()
  File "/app/repo/tests/test_foo.py", line 18, in test_something
    foo()
  File "/app/repo/foo/core.py", line 25, in foo
    raise ValueError("Invalid argument")
ValueError: Invalid argument
"""
    parsed = parse_traceback(tb_text, repo_root="/app/repo")
    assert parsed.primary_frame is not None
    assert parsed.primary_frame.file_path == "foo/core.py"
    assert parsed.primary_frame.line_number == 25
    assert parsed.primary_frame.function_name == "foo"
    assert not parsed.primary_frame.is_framework


# ── 2. BM25 & Tokenization Tests ──────────────────────────────────────────────


def test_bm25_tokenize_code():
    tokens = tokenize_code("def calculateMedianValue(input_numbers: list[int]): return 0")
    # Sub-word splitting for camelCase and snake_case
    assert "calculate" in tokens
    assert "median" in tokens
    assert "value" in tokens
    assert "input" in tokens
    assert "numbers" in tokens
    assert "list" in tokens
    assert "int" in tokens


def test_bm25_index_search():
    chunks = [
        CodeChunk(
            file_path="math/stats.py",
            symbol_name="calculate_median",
            symbol_type="function",
            start_line=10,
            end_line=25,
            source="def calculate_median(values):\n    sorted_v = sorted(values)\n    return sorted_v[len(sorted_v)//2]",
            docstring="Compute median of numeric sequence.",
            parent_class="",
        ),
        CodeChunk(
            file_path="math/arithmetic.py",
            symbol_name="divide_safe",
            symbol_type="function",
            start_line=1,
            end_line=15,
            source="def divide_safe(a, b):\n    if b == 0:\n        raise ZeroDivisionError('cannot divide by zero')\n    return a / b",
            docstring="Safe division handling zero denominator.",
            parent_class="",
        ),
        CodeChunk(
            file_path="core/parser.py",
            symbol_name="parse_json_payload",
            symbol_type="function",
            start_line=5,
            end_line=20,
            source="def parse_json_payload(raw_str):\n    import json\n    return json.loads(raw_str)",
            docstring="Parse raw json string.",
            parent_class="",
        ),
    ]

    index = BM25Index.build(chunks)

    # Search for division
    results_div = index.search("divide zero denominator", top_k=2)
    assert len(results_div) > 0
    assert results_div[0]["symbol_name"] == "divide_safe"

    # Search for median
    results_med = index.search("calculate_median sorted", top_k=2)
    assert len(results_med) > 0
    assert results_med[0]["symbol_name"] == "calculate_median"


# ── 3. Hybrid Retrieval & Reciprocal Rank Fusion (RRF) ─────────────────────────


def test_hybrid_search_rrf():
    mock_bm25 = MagicMock()
    mock_bm25.search.return_value = [
        {"file_path": "a.py", "symbol_name": "func_a", "start_line": 10, "end_line": 20, "score": 2.5, "source": "def func_a(): pass"},
        {"file_path": "b.py", "symbol_name": "func_b", "start_line": 1, "end_line": 10, "score": 1.8, "source": "def func_b(): pass"},
    ]

    index = EmbedIndex(table=MagicMock(), table_name="test_index", bm25=mock_bm25)
    # Mock vector search
    index.search_vector = MagicMock(return_value=[
        {"file_path": "b.py", "symbol_name": "func_b", "start_line": 1, "end_line": 10, "_distance": 0.1, "source": "def func_b(): pass"},
        {"file_path": "c.py", "symbol_name": "func_c", "start_line": 5, "end_line": 15, "_distance": 0.2, "source": "def func_c(): pass"},
    ])

    hybrid_results = index.search_hybrid("query text", top_k=3, rrf_k=60)
    assert len(hybrid_results) == 3

    # func_b appears in both BM25 and vector, so its reciprocal rank score should be highest!
    assert hybrid_results[0]["symbol_name"] == "func_b"
    assert "rrf_score" in hybrid_results[0]
    assert hybrid_results[0]["rrf_score"] > 0


def test_search_code_tool_modes():
    mock_bm25 = MagicMock()
    mock_bm25.search.return_value = [
        {"file_path": "a.py", "symbol_name": "func_bm25", "symbol_type": "function", "start_line": 1, "end_line": 10, "score": 3.0, "source": "..."}
    ]
    index = EmbedIndex(table=MagicMock(), table_name="test_index", bm25=mock_bm25)
    index.search_vector = MagicMock(return_value=[
        {"file_path": "b.py", "symbol_name": "func_vec", "symbol_type": "function", "start_line": 1, "end_line": 10, "_distance": 0.1, "source": "..."}
    ])

    # Mode = bm25
    res_bm25 = search_code("test query", embed_index=index, mode="bm25")
    assert len(res_bm25) == 1
    assert res_bm25[0]["symbol_name"] == "func_bm25"

    # Mode = vector
    res_vec = search_code("test query", embed_index=index, mode="vector")
    assert len(res_vec) == 1
    assert res_vec[0]["symbol_name"] == "func_vec"

    # Mode = hybrid
    res_hybrid = search_code("test query", embed_index=index, mode="hybrid")
    assert len(res_hybrid) == 2


# ── 4. Failure Reflection & Seed Suggestions ───────────────────────────────────


def test_build_failure_reflection():
    diff = """--- a/math_utils.py
+++ b/math_utils.py
@@ -12,2 +12,2 @@
-    return a / b
+    return a / (b or 1)
"""
    tb = """
tests/test_math.py:10: in test_divide
    assert divide(10, 0) is None
E   AssertionError: assert 10.0 is None
"""
    reflection = build_failure_reflection(
        last_patch=diff,
        traceback_text=tb,
        failed_tests=["tests/test_math.py::test_divide"],
        retry_num=2,
    )

    assert not reflection["passed"]
    assert reflection["retry_number"] == 2
    assert reflection["error_type"] == "AssertionError"
    assert reflection["failing_assertion"] == "assert 10.0 is None"
    assert reflection["attempted_patch"] == diff
    assert "Attempted Patch (Diff):" in reflection["failure_reflection"]
    assert "AssertionError" in reflection["failure_reflection"]
    assert "assert 10.0 is None" in reflection["failure_reflection"]
    assert "traceback" not in reflection  # Raw log excluded per requirement


def test_process_test_result_initial_run_provides_seeds():
    ctx = AgentContext(
        repo_root=Path("/fake/repo"),
        repo_url="https://github.com/fake/repo",
        issue_text="ZeroDivision bug",
        failing_tests=["test_math.py::test_div"],
    )

    raw_test_result = {
        "passed": False,
        "traceback": """tests/test_math.py:5: in test_div\n    div(1, 0)\nmath.py:12: in div\n    return a / b\nE   ZeroDivisionError: division by zero""",
        "failed_tests": ["test_math.py::test_div"],
    }

    # Iteration 1 without previous patch
    processed, _, retries = _process_test_result(
        raw_test_result, last_patch="", iterations=1, retries=0, ctx=ctx
    )

    assert not processed["passed"]
    assert retries == 0
    assert "seed_suggestions" in processed
    seeds = processed["seed_suggestions"]
    assert seeds["file_path"] == "math.py"
    assert seeds["line_number"] == 12
    assert seeds["function_name"] == "div"
    assert seeds["read_file_args"]["file_path"] == "math.py"
    assert seeds["read_file_args"]["start_line"] == 1
    assert seeds["read_file_args"]["end_line"] == 27
    assert seeds["find_references_args"]["symbol_name"] == "div"


def test_process_test_result_retry_uses_failure_reflection():
    ctx = AgentContext(
        repo_root=Path("/fake/repo"),
        repo_url="https://github.com/fake/repo",
        issue_text="ZeroDivision bug",
        failing_tests=["test_math.py::test_div"],
    )

    raw_test_result = {
        "passed": False,
        "traceback": """math.py:12: in div\n    assert False\nE   AssertionError: assert False""",
        "stdout_preview": "RAW TERMINAL LOG SPAM THAT SHOULD NOT BE FED TO THE MODEL",
        "failed_tests": ["test_math.py::test_div"],
    }

    # Iteration 2 with attempted patch
    processed, _, retries = _process_test_result(
        raw_test_result, last_patch="--- a/math.py\n+++ b/math.py\n...", iterations=2, retries=0, ctx=ctx
    )

    assert not processed["passed"]
    assert retries == 1
    # Raw log spam must be stripped on retry
    assert "stdout_preview" not in processed
    assert "failure_reflection" in processed
    assert processed["attempted_patch"].startswith("--- a/math.py")
    assert processed["failing_assertion"] == "assert False"
