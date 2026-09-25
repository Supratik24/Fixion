"""
tests/test_symbol_graph.py — Unit tests for the symbol graph.
"""

from __future__ import annotations

import pytest

from ingest.ast_parser import CodeChunk
from ingest.symbol_graph import SymbolGraph, build_symbol_graph


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_chunk(
    symbol: str,
    file_path: str = "module.py",
    start: int = 1,
    end: int = 10,
    symbol_type: str = "function",
    parent: str = "",
    source: str = "",
) -> CodeChunk:
    return CodeChunk(
        file_path=file_path,
        symbol_name=symbol,
        symbol_type=symbol_type,
        start_line=start,
        end_line=end,
        source=source or f"def {symbol}(): pass",
        docstring="",
        parent_class=parent,
    )


@pytest.fixture()
def sample_chunks() -> list[CodeChunk]:
    return [
        _make_chunk("parse_input", file_path="parser.py", start=1, end=10,
                    source="def parse_input(s): return validate(s)"),
        _make_chunk("validate", file_path="validator.py", start=1, end=5),
        _make_chunk("MyClass", file_path="models.py", symbol_type="class"),
        _make_chunk("MyClass.run", file_path="models.py", parent="MyClass",
                    source="def run(self): return parse_input('x')"),
    ]


@pytest.fixture()
def graph(sample_chunks: list[CodeChunk]) -> SymbolGraph:
    return build_symbol_graph(sample_chunks)


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_symbols_are_nodes(graph: SymbolGraph) -> None:
    defn = graph.get_definition("validate")
    assert defn is not None
    assert defn.file_path == "validator.py"


def test_class_node_exists(graph: SymbolGraph) -> None:
    defn = graph.get_definition("MyClass")
    assert defn is not None
    assert defn.symbol_type == "class"


def test_get_file_symbols(graph: SymbolGraph) -> None:
    syms = graph.get_file_symbols("models.py")
    assert "MyClass" in syms
    assert "MyClass.run" in syms


def test_call_edge_detected(graph: SymbolGraph) -> None:
    # parse_input calls validate (because 'validate' appears in its source)
    callees = graph.get_callees("parse_input")
    assert "validate" in callees


def test_caller_edge_inverse(graph: SymbolGraph) -> None:
    callers = graph.get_callers("validate")
    assert "parse_input" in callers


def test_bare_name_lookup(graph: SymbolGraph) -> None:
    candidates = graph.find_symbols_by_name("validate")
    assert "validate" in candidates


def test_subgraph_around(graph: SymbolGraph) -> None:
    neighbors = graph.subgraph_around("parse_input", radius=1)
    assert "validate" in neighbors


def test_unknown_symbol_returns_none(graph: SymbolGraph) -> None:
    assert graph.get_definition("nonexistent_symbol_xyz") is None


def test_save_and_load(graph: SymbolGraph, tmp_path) -> None:
    path = tmp_path / "graph.json"
    graph.save(path)

    loaded = SymbolGraph.load(path)
    defn = loaded.get_definition("validate")
    assert defn is not None
    assert defn.file_path == "validator.py"


def test_import_graph_populated(sample_chunks) -> None:
    # Add import info to a chunk
    chunk = _make_chunk("importer", file_path="main.py")
    chunk.imports = ["import os", "from pathlib import Path"]
    chunks = sample_chunks + [chunk]
    g = build_symbol_graph(chunks)
    ig = g.get_import_graph()
    assert "main.py" in ig
