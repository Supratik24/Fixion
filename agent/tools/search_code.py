"""
agent/tools/search_code.py — Semantic search over the embedded code index.

Exposed as a Gemini tool: the agent calls this to find relevant code chunks
given a natural-language query (e.g. "where is division handled in stats module").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ingest.embed_index import EmbedIndex


def search_code(
    query: str,
    embed_index: EmbedIndex,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """
    Search the code index semantically.

    Parameters
    ----------
    query:
        Natural-language description of what you're looking for,
        e.g. ``"function that parses user input and raises ValueError"``.
    embed_index:
        Pre-built EmbedIndex instance for the current repo.
    top_k:
        Number of results to return (default 8).

    Returns
    -------
    list of dicts with keys: file_path, symbol_name, symbol_type,
    start_line, end_line, docstring, source, _distance
    """
    results = embed_index.search(query, top_k=top_k)
    # Return a clean subset of fields (drop the raw vector)
    return [
        {
            "file_path": r["file_path"],
            "symbol_name": r["symbol_name"],
            "symbol_type": r["symbol_type"],
            "start_line": r["start_line"],
            "end_line": r["end_line"],
            "docstring": r.get("docstring", ""),
            "source_preview": r.get("source", "")[:300],
            "relevance_score": round(1.0 - r.get("_distance", 0.5), 3),
        }
        for r in results
    ]


# ── Gemini tool declaration ────────────────────────────────────────────────────

TOOL_DECLARATION = {
    "name": "search_code",
    "description": (
        "Semantically search the repository's code index. "
        "Use this to find functions, classes, or files related to a concept, "
        "error message, or feature without knowing the exact symbol name. "
        "Returns ranked code chunks with file paths and line numbers."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language description of what you're looking for.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default 8, max 20).",
                "default": 8,
            },
        },
        "required": ["query"],
    },
}
