"""
agent/tools/find_references.py — Symbol graph traversal for localization.

The agent uses this to explore the call graph around a suspicious symbol:
who calls it, what does it call, what's nearby in the dependency graph.
"""

from __future__ import annotations

from ingest.symbol_graph import SymbolGraph


def find_definition(symbol_name: str, graph: SymbolGraph) -> dict[str, object]:
    """
    Find where a symbol is defined in the repository.

    Parameters
    ----------
    symbol_name:
        Bare name (e.g. ``"parse_file"``) or fully-qualified
        (e.g. ``"AstParser.parse_file"``). Bare names are looked up
        by suffix match if an exact match isn't found.
    graph:
        The SymbolGraph for the current repo.
    """
    defn = graph.get_definition(symbol_name)
    if defn is None:
        # Try bare-name lookup
        candidates = graph.find_symbols_by_name(symbol_name.split(".")[-1])
        if not candidates:
            return {"error": f"Symbol '{symbol_name}' not found in graph."}
        if len(candidates) == 1:
            defn = graph.get_definition(candidates[0])
        else:
            return {
                "ambiguous": True,
                "candidates": candidates,
                "message": f"Multiple definitions found for '{symbol_name}'. Specify one.",
            }
    return {
        "symbol_name": defn.symbol_name,
        "file_path": defn.file_path,
        "start_line": defn.start_line,
        "end_line": defn.end_line,
        "symbol_type": defn.symbol_type,
        "parent_class": defn.parent_class,
    }


def find_callers(symbol_name: str, graph: SymbolGraph) -> dict[str, object]:
    """
    Return all symbols that call *symbol_name*.

    Useful for understanding the blast radius of a bug: if you change X,
    who else is affected?
    """
    callers = graph.get_callers(symbol_name)
    caller_defs = []
    for caller in callers[:20]:  # cap at 20 to avoid huge responses
        defn = graph.get_definition(caller)
        if defn:
            caller_defs.append({
                "symbol_name": defn.symbol_name,
                "file_path": defn.file_path,
                "start_line": defn.start_line,
            })
    return {"symbol": symbol_name, "caller_count": len(callers), "callers": caller_defs}


def find_callees(symbol_name: str, graph: SymbolGraph) -> dict[str, object]:
    """Return all symbols that *symbol_name* calls."""
    callees = graph.get_callees(symbol_name)
    callee_defs = []
    for callee in callees[:20]:
        defn = graph.get_definition(callee)
        if defn:
            callee_defs.append({
                "symbol_name": defn.symbol_name,
                "file_path": defn.file_path,
                "start_line": defn.start_line,
            })
    return {"symbol": symbol_name, "callee_count": len(callees), "callees": callee_defs}


def find_neighborhood(symbol_name: str, graph: SymbolGraph, radius: int = 2) -> dict[str, object]:
    """
    Return all symbols within *radius* hops of *symbol_name* in the call graph.
    Useful for understanding the local context of a suspicious function.
    """
    neighbors = graph.subgraph_around(symbol_name, radius=radius)
    result = []
    for sym in neighbors:
        defn = graph.get_definition(sym)
        if defn:
            result.append({"symbol_name": sym, "file_path": defn.file_path})
    return {"center": symbol_name, "radius": radius, "neighbors": result}


def get_file_symbols(file_path: str, graph: SymbolGraph) -> dict[str, object]:
    """Return all symbols defined in a given file."""
    symbols = graph.get_file_symbols(file_path)
    defs = []
    for sym in symbols:
        defn = graph.get_definition(sym)
        if defn:
            defs.append({
                "symbol_name": defn.symbol_name,
                "symbol_type": defn.symbol_type,
                "start_line": defn.start_line,
                "end_line": defn.end_line,
            })
    return {"file_path": file_path, "symbol_count": len(defs), "symbols": defs}


# ── Gemini tool declarations ───────────────────────────────────────────────────

TOOL_DECLARATIONS = [
    {
        "name": "find_definition",
        "description": (
            "Find where a symbol (function, class, method) is defined. "
            "Returns the file path and line range so you can read it with read_file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol_name": {
                    "type": "string",
                    "description": "The symbol name to look up (bare or fully-qualified).",
                }
            },
            "required": ["symbol_name"],
        },
    },
    {
        "name": "find_callers",
        "description": "Find all code locations that call a given symbol. Use to assess blast radius.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol_name": {"type": "string", "description": "The symbol to find callers of."}
            },
            "required": ["symbol_name"],
        },
    },
    {
        "name": "find_callees",
        "description": "Find all symbols that a given function/method calls internally.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol_name": {"type": "string", "description": "The symbol to inspect."}
            },
            "required": ["symbol_name"],
        },
    },
    {
        "name": "find_neighborhood",
        "description": (
            "Return all symbols within N hops of a symbol in the call graph. "
            "Use to understand local context around a suspicious function."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol_name": {"type": "string"},
                "radius": {
                    "type": "integer",
                    "description": "Number of hops (default 2, max 3).",
                    "default": 2,
                },
            },
            "required": ["symbol_name"],
        },
    },
    {
        "name": "get_file_symbols",
        "description": "List all functions and classes defined in a specific file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "File path relative to the repository root.",
                }
            },
            "required": ["file_path"],
        },
    },
]
