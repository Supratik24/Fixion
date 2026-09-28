"""
ingest/embed_index.py — Gemini text-embedding-004 + LanceDB vector index.

Embeds CodeChunks and stores them in a persistent LanceDB table.
Semantic search returns the top-k most similar chunks to a query string.

Index is keyed by (repo_url, head_sha) so it's automatically invalidated
when the repo changes.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
from google import genai
from google.genai import types as genai_types

from config import settings
from ingest.ast_parser import CodeChunk

logger = logging.getLogger(__name__)

# LanceDB table schema (PyArrow)
_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("file_path", pa.string()),
        pa.field("symbol_name", pa.string()),
        pa.field("symbol_type", pa.string()),
        pa.field("start_line", pa.int32()),
        pa.field("end_line", pa.int32()),
        pa.field("parent_class", pa.string()),
        pa.field("docstring", pa.string()),
        pa.field("source", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 3072)),
    ]
)

_EMBED_BATCH_SIZE = 100  # Max limit for Gemini batchEmbedContents


def _make_client() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


def _table_name(repo_url: str, head_sha: str) -> str:
    slug = hashlib.sha1(f"{repo_url}:{head_sha}".encode()).hexdigest()[:16]
    return f"index_{slug}"


def _chunk_to_embed_text(chunk: CodeChunk) -> str:
    """Compose the string that gets embedded for a chunk."""
    parts = [
        f"File: {chunk.file_path}",
        f"Symbol: {chunk.symbol_name} ({chunk.symbol_type})",
    ]
    if chunk.docstring:
        parts.append(f"Docstring: {chunk.docstring}")
    parts.append(chunk.source[:2000])  # cap at 2k chars to stay in token budget
    return "\n".join(parts)


def _embed_batch(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using the Gemini embedding model."""
    response = client.models.embed_content(
        model=settings.embed_model,
        contents=texts,
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return [list(e.values) for e in response.embeddings]


class EmbedIndex:
    """
    Persistent vector index for a single repo at a specific commit SHA.

    Usage::

        index = EmbedIndex.build(chunks, repo_url, head_sha)
        results = index.search("division by zero in stats", top_k=5)
    """

    def __init__(self, table: Any, table_name: str) -> None:
        self._table = table
        self._table_name = table_name

    # ── Build ──────────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        chunks: list[CodeChunk],
        repo_url: str,
        head_sha: str,
        index_dir: Path | None = None,
    ) -> "EmbedIndex":
        """
        Embed *chunks* and persist them in LanceDB.

        If an index for (repo_url, head_sha) already exists, it is reused
        without re-embedding (fast path). Force a rebuild by deleting the
        index directory.
        """
        index_dir = index_dir or settings.index_dir
        index_dir.mkdir(parents=True, exist_ok=True)

        db = lancedb.connect(str(index_dir))
        tname = _table_name(repo_url, head_sha)

        if tname in db.table_names():
            logger.info("Reusing existing index '%s' (cached)", tname)
            return cls(db.open_table(tname), tname)

        logger.info("Building embedding index for %d chunks…", len(chunks))
        client = _make_client()

        rows: list[dict[str, Any]] = []
        texts = [_chunk_to_embed_text(c) for c in chunks]

        from google.genai.errors import ClientError
        import time
        # Embed in batches
        vectors: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[i : i + _EMBED_BATCH_SIZE]
            logger.debug("Embedding batch %d/%d…", i // _EMBED_BATCH_SIZE + 1,
                         (len(texts) + _EMBED_BATCH_SIZE - 1) // _EMBED_BATCH_SIZE)
            
            while True:
                try:
                    vectors.extend(_embed_batch(client, batch))
                    break
                except ClientError as e:
                    if "429" in str(e):
                        logger.warning("Hit 429 Rate Limit. Sleeping for 30s...")
                        time.sleep(30)
                    else:
                        raise
            
            if i + _EMBED_BATCH_SIZE < len(texts):
                time.sleep(5)

        for chunk, vec in zip(chunks, vectors):
            chunk_id = hashlib.sha1(
                f"{chunk.file_path}:{chunk.symbol_name}:{chunk.start_line}".encode()
            ).hexdigest()
            rows.append(
                {
                    "id": chunk_id,
                    "file_path": chunk.file_path,
                    "symbol_name": chunk.symbol_name,
                    "symbol_type": chunk.symbol_type,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "parent_class": chunk.parent_class,
                    "docstring": chunk.docstring,
                    "source": chunk.source,
                    "vector": [float(v) for v in vec],
                }
            )

        table = db.create_table(tname, data=rows, schema=_SCHEMA)
        logger.info("Index '%s' created with %d entries.", tname, len(rows))
        return cls(table, tname)

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """
        Return the top-k chunks most semantically similar to *query*.

        Each result is a dict with all chunk metadata fields plus a ``_distance``
        float (lower = more similar for L2; for cosine, lower = more similar).
        """
        client = _make_client()
        query_vec = _embed_batch(client, [query])[0]

        results = (
            self._table.search(query_vec, vector_column_name="vector")
            .limit(top_k)
            .to_list()
        )
        return results

    @classmethod
    def load(cls, repo_url: str, head_sha: str, index_dir: Path | None = None) -> "EmbedIndex":
        """Load an existing index without re-building."""
        index_dir = index_dir or settings.index_dir
        db = lancedb.connect(str(index_dir))
        tname = _table_name(repo_url, head_sha)
        if tname not in db.table_names():
            raise FileNotFoundError(
                f"No index found for repo={repo_url} sha={head_sha}. Run build() first."
            )
        return cls(db.open_table(tname), tname)
