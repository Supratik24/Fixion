"""
ingest/symbol_graph.py — Build an import/call graph from tree-sitter AST data.

The graph is a directed NetworkX DiGraph where nodes are fully-qualified symbol
names and edges represent:
  - ``imports``: module A imports module B
  - ``calls``:   function A calls function B
  - ``defines``: file F defines symbol S

Query interface:
  - get_definition(symbol)  → file_path, start_line, end_line
  - get_callers(symbol)     → list of symbols that call this one
  - get_callees(symbol)     → list of symbols this one calls
  - get_file_symbols(file)  → all symbols defined in a file
  - get_import_graph()      → file-level import adjacency
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from ingest.ast_parser import CodeChunk

logger = logging.getLogger(__name__)

# Edge type attribute key
EDGE_TYPE = "rel"


@dataclass
class SymbolDef:
    """Location metadata for a symbol."""

    symbol_name: str
    file_path: str
    start_line: int
    end_line: int
    symbol_type: str
    parent_class: str


class SymbolGraph:
    """
    Directed graph of symbols and their relationships.

    Nodes: symbol names (str), with ``def`` attribute = SymbolDef dict
    Edges: (source, target, rel=<type>)
    """

    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()
        # Secondary lookup: file_path → [symbol_name, ...]
        self._file_index: dict[str, list[str]] = {}

    # ── Build ──────────────────────────────────────────────────────────────────

    def build_from_chunks(self, chunks: list[CodeChunk]) -> None:
        """
        Populate the graph from the list of CodeChunks produced by ast_parser.

        This is a two-pass process:
          1. Add all symbol nodes (defines edges from file → symbol)
          2. Add call/import edges by scanning source text heuristically
        """
        logger.info("Building symbol graph from %d chunks…", len(chunks))

        # Pass 1: nodes
        for chunk in chunks:
            defn = SymbolDef(
                symbol_name=chunk.symbol_name,
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                symbol_type=chunk.symbol_type,
                parent_class=chunk.parent_class,
            )
            self._g.add_node(chunk.symbol_name, defn=asdict(defn))
            file_node = f"file:{chunk.file_path}"
            self._g.add_node(file_node, defn={"file_path": chunk.file_path})
            self._g.add_edge(file_node, chunk.symbol_name, rel="defines")
            self._file_index.setdefault(chunk.file_path, []).append(chunk.symbol_name)

            # File-level imports → file import edges
            for imp in chunk.imports:
                self._g.add_edge(file_node, f"import:{imp}", rel="imports")

        # Pass 2: heuristic call edges (best-effort without a full type-checker)
        known_symbols = {s for s in self._g.nodes if not s.startswith(("file:", "import:"))}
        for chunk in chunks:
            self._add_call_edges(chunk, known_symbols)

        logger.info(
            "Symbol graph: %d nodes, %d edges",
            self._g.number_of_nodes(),
            self._g.number_of_edges(),
        )

    def _add_call_edges(self, chunk: CodeChunk, known_symbols: set[str]) -> None:
        """
        Scan chunk.source for references to known symbols and add call edges.
        This is a lightweight textual heuristic — not a full type resolver.
        """
        source = chunk.source
        for sym in known_symbols:
            if sym == chunk.symbol_name:
                continue
            # Match bare name or method call pattern
            bare_name = sym.split(".")[-1]
            if bare_name and bare_name in source:
                self._g.add_edge(chunk.symbol_name, sym, rel="calls")

    # ── Query API ──────────────────────────────────────────────────────────────

    def get_definition(self, symbol: str) -> SymbolDef | None:
        """Return the SymbolDef for *symbol*, or None if not in graph."""
        node_data = self._g.nodes.get(symbol)
        if node_data and "defn" in node_data:
            d = node_data["defn"]
            return SymbolDef(**d)
        return None

    def get_callers(self, symbol: str) -> list[str]:
        """Return all symbols that have a ``calls`` edge pointing to *symbol*."""
        return [
            src
            for src, dst, data in self._g.in_edges(symbol, data=True)
            if data.get(EDGE_TYPE) == "calls"
        ]

    def get_callees(self, symbol: str) -> list[str]:
        """Return all symbols that *symbol* calls."""
        return [
            dst
            for src, dst, data in self._g.out_edges(symbol, data=True)
            if data.get(EDGE_TYPE) == "calls"
        ]

    def get_file_symbols(self, file_path: str) -> list[str]:
        """Return all symbols defined in *file_path*."""
        return list(self._file_index.get(file_path, []))

    def get_import_graph(self) -> dict[str, list[str]]:
        """Return file-level import adjacency as a plain dict."""
        result: dict[str, list[str]] = {}
        for src, dst, data in self._g.edges(data=True):
            if data.get(EDGE_TYPE) == "imports" and src.startswith("file:"):
                file = src[len("file:"):]
                result.setdefault(file, []).append(dst)
        return result

    def find_symbols_by_name(self, name: str) -> list[str]:
        """
        Find fully-qualified symbol names whose bare name matches *name*.
        Useful when a stack trace mentions ``parse_file`` but you need the FQN.
        """
        return [sym for sym in self._g.nodes if sym.split(".")[-1] == name]

    def subgraph_around(self, symbol: str, radius: int = 2) -> list[str]:
        """
        Return all symbols within *radius* hops of *symbol* (undirected BFS).
        Useful for finding the neighborhood of a suspicious function.
        """
        if symbol not in self._g:
            return []
        ego = nx.ego_graph(self._g.to_undirected(), symbol, radius=radius)
        return [n for n in ego.nodes if not n.startswith(("file:", "import:"))]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Serialize the graph to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "nodes": [
                {"id": n, **self._g.nodes[n]} for n in self._g.nodes
            ],
            "edges": [
                {"src": u, "dst": v, **d} for u, v, d in self._g.edges(data=True)
            ],
            "file_index": self._file_index,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Symbol graph saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "SymbolGraph":
        """Deserialize a symbol graph from a JSON file produced by save()."""
        data = json.loads(path.read_text(encoding="utf-8"))
        sg = cls()
        for node in data["nodes"]:
            nid = node.pop("id")
            sg._g.add_node(nid, **node)
        for edge in data["edges"]:
            sg._g.add_edge(edge["src"], edge["dst"], rel=edge.get("rel", ""))
        sg._file_index = data.get("file_index", {})
        logger.info("Symbol graph loaded from %s", path)
        return sg


def build_symbol_graph(chunks: list[CodeChunk]) -> SymbolGraph:
    """Convenience function: build and return a SymbolGraph from chunks."""
    sg = SymbolGraph()
    sg.build_from_chunks(chunks)
    return sg
