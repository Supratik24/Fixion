"""
agent/tools/search_code.py — Hybrid code search (BM25 + Dense Vectors + RRF).

Exposed as an agent tool: finds relevant code chunks using natural-language queries
or exact identifier names, combining lexical keyword matching and semantic vector search.
"""

from __future__ import annotations

from typing import Any

from ingest.embed_index import EmbedIndex


def search_code(
    query: str,
    embed_index: EmbedIndex,
    top_k: int = 8,
    mode: str = "hybrid",
) -> list[dict[str, Any]]:
    """
    Search the code index using hybrid retrieval (BM25 + Vector + Reciprocal Rank Fusion).

    Parameters
    ----------
    query:
        Natural-language description or exact identifier/symbol name to look for.
    embed_index:
        Pre-built EmbedIndex instance for the current repo.
    top_k:
        Number of results to return (default 8, max 20).
    mode:
        Retrieval mode: "hybrid" (default), "bm25", or "vector".

    Returns
    -------
    list of dicts with keys: file_path, symbol_name, symbol_type,
    start_line, end_line, docstring, source_preview, relevance_score
    """
    if embed_index is None:
        return []

    mode_lower = mode.lower()
    if mode_lower == "bm25" and getattr(embed_index, "bm25", None) is not None:
        results = embed_index.bm25.search(query, top_k=top_k)
    elif mode_lower == "vector" and hasattr(embed_index, "search_vector"):
        results = embed_index.search_vector(query, top_k=top_k)
    elif hasattr(embed_index, "search_hybrid") and getattr(embed_index, "bm25", None) is not None:
        results = embed_index.search_hybrid(query, top_k=top_k)
    else:
        results = embed_index.search(query, top_k=top_k)

    formatted: list[dict[str, Any]] = []
    for r in results:
        # Determine best available relevance metric
        if "rrf_score" in r:
            rel = r["rrf_score"]
        elif "_distance" in r:
            rel = round(1.0 - float(r["_distance"]), 3)
        elif "score" in r:
            rel = round(float(r["score"]), 3)
        else:
            rel = 1.0

        formatted.append(
            {
                "file_path": r["file_path"],
                "symbol_name": r["symbol_name"],
                "symbol_type": r["symbol_type"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "docstring": r.get("docstring", ""),
                "source_preview": r.get("source", "")[:300],
                "relevance_score": rel,
            }
        )
    return formatted


# ── Tool declaration ──────────────────────────────────────────────────────────

TOOL_DECLARATION = {
    "name": "search_code",
    "description": (
        "Search the repository's code index using hybrid retrieval (BM25 keyword search "
        "+ semantic embeddings merged with Reciprocal Rank Fusion). "
        "Finds functions, classes, error messages, or concepts with ranked code locations."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language description or identifier name to search.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default 8, max 20).",
                "default": 8,
            },
            "mode": {
                "type": "string",
                "enum": ["hybrid", "bm25", "vector"],
                "description": "Retrieval mode (default: hybrid).",
                "default": "hybrid",
            },
        },
        "required": ["query"],
    },
}
