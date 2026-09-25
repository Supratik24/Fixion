"""
tests/test_ast_parser.py — Unit tests for the tree-sitter AST chunker.
"""

from __future__ import annotations

import textwrap
import tempfile
from pathlib import Path

import pytest

from ingest.ast_parser import CodeChunk, parse_file, parse_repo


# ── Fixtures ───────────────────────────────────────────────────────────────────

SIMPLE_SOURCE = textwrap.dedent("""\
    import os
    from pathlib import Path

    def top_level_func(x: int) -> int:
        \"\"\"Double the input.\"\"\"
        return x * 2

    class MyClass:
        \"\"\"A simple class.\"\"\"

        def method_one(self) -> None:
            pass

        def method_two(self, value: str) -> str:
            return value.upper()

    async def async_func() -> None:
        \"\"\"An async function.\"\"\"
        pass
""")


@pytest.fixture()
def temp_repo(tmp_path: Path) -> Path:
    """Create a minimal temp repo with one Python file."""
    (tmp_path / "module.py").write_text(SIMPLE_SOURCE, encoding="utf-8")
    return tmp_path


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_parse_file_returns_chunks(temp_repo: Path) -> None:
    chunks = parse_file(temp_repo / "module.py", temp_repo)
    assert len(chunks) > 0


def test_top_level_function_detected(temp_repo: Path) -> None:
    chunks = parse_file(temp_repo / "module.py", temp_repo)
    names = [c.symbol_name for c in chunks]
    assert "top_level_func" in names


def test_class_detected(temp_repo: Path) -> None:
    chunks = parse_file(temp_repo / "module.py", temp_repo)
    names = [c.symbol_name for c in chunks]
    assert "MyClass" in names


def test_methods_detected(temp_repo: Path) -> None:
    chunks = parse_file(temp_repo / "module.py", temp_repo)
    names = [c.symbol_name for c in chunks]
    assert "MyClass.method_one" in names
    assert "MyClass.method_two" in names


def test_async_function_detected(temp_repo: Path) -> None:
    chunks = parse_file(temp_repo / "module.py", temp_repo)
    types = {c.symbol_name: c.symbol_type for c in chunks}
    assert types.get("async_func") == "async_function"


def test_docstring_extraction(temp_repo: Path) -> None:
    chunks = parse_file(temp_repo / "module.py", temp_repo)
    func_chunks = {c.symbol_name: c for c in chunks}
    assert "Double the input." in func_chunks["top_level_func"].docstring


def test_line_numbers_nonzero(temp_repo: Path) -> None:
    chunks = parse_file(temp_repo / "module.py", temp_repo)
    for chunk in chunks:
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line


def test_relative_file_path(temp_repo: Path) -> None:
    chunks = parse_file(temp_repo / "module.py", temp_repo)
    for chunk in chunks:
        assert not chunk.file_path.startswith("/")
        assert "module.py" in chunk.file_path


def test_parse_repo_walks_all_files(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def func_a(): pass", encoding="utf-8")
    (tmp_path / "b.py").write_text("def func_b(): pass", encoding="utf-8")
    subdir = tmp_path / "sub"
    subdir.mkdir()
    (subdir / "c.py").write_text("def func_c(): pass", encoding="utf-8")

    chunks = parse_repo(tmp_path)
    names = [c.symbol_name for c in chunks]
    assert "func_a" in names
    assert "func_b" in names
    assert "func_c" in names


def test_parse_repo_excludes_pycache(tmp_path: Path) -> None:
    (tmp_path / "real.py").write_text("def good(): pass", encoding="utf-8")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "cached.py").write_text("def bad(): pass", encoding="utf-8")

    chunks = parse_repo(tmp_path)
    names = [c.symbol_name for c in chunks]
    assert "good" in names
    assert "bad" not in names


def test_empty_file_returns_no_chunks(tmp_path: Path) -> None:
    (tmp_path / "empty.py").write_text("", encoding="utf-8")
    chunks = parse_file(tmp_path / "empty.py", tmp_path)
    assert chunks == []


def test_imports_captured(temp_repo: Path) -> None:
    chunks = parse_file(temp_repo / "module.py", temp_repo)
    assert len(chunks) > 0
    # All chunks should carry the module's imports
    assert any("import os" in imp for chunk in chunks for imp in chunk.imports)
