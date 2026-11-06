"""
ingest/bm25.py — BM25 keyword search index for source code chunks.

Provides fast lexical retrieval with identifier-aware tokenization
(splitting snake_case and camelCase) to complement dense vector embeddings.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ingest.ast_parser import CodeChunk

_IDENTIFIER_SPLIT_RE = re.compile(r"[_\s\-\./\\:;,()\[\]{}'\"=<>!+*?&|]+")
_CAMEL_CASE_RE = re.compile(r"([a-z])([A-Z])")


def tokenize_code(text: str) -> list[str]:
    """
    Split text into normalized tokens, splitting camelCase and snake_case identifiers.
    """
    if not text:
        return []

    # Insert space before capital letters for camelCase
    spaced = _CAMEL_CASE_RE.sub(r"\1 \2", text)
    raw_tokens = _IDENTIFIER_SPLIT_RE.split(spaced.lower())

    tokens: list[str] = []
    for tok in raw_tokens:
        tok = tok.strip()
        if len(tok) >= 2:
            tokens.append(tok)
    return tokens


@dataclass
class BM25Document:
    """Document representation in BM25 index."""

    chunk: CodeChunk
    tokens: list[str]
    doc_len: int
    term_freqs: dict[str, int]


class BM25Index:
    """
    Pure Python BM25Okapi implementation tailored for source code chunks.
    Zero heavy dependencies, deterministic, and fast.
    """

    def __init__(
        self,
        docs: list[BM25Document],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.num_docs = len(docs)
        self.avg_doc_len = (
            sum(d.doc_len for d in docs) / self.num_docs if self.num_docs > 0 else 0.0
        )

        # Calculate document frequencies
        self.doc_freqs: Counter[str] = Counter()
        for doc in docs:
            for term in doc.term_freqs:
                self.doc_freqs[term] += 1

        # Calculate IDF table
        self.idf: dict[str, float] = {}
        for term, df in self.doc_freqs.items():
            # Standard Okapi IDF with floor at 0.0
            val = math.log(((self.num_docs - df + 0.5) / (df + 0.5)) + 1.0)
            self.idf[term] = max(val, 0.0)

    @classmethod
    def build(cls, chunks: list[CodeChunk]) -> BM25Index:
        """Build BM25 index from a list of CodeChunks."""
        docs: list[BM25Document] = []
        for chunk in chunks:
            # Combine symbol name, docstring, file path, and source for lexical indexing
            text = (
                f"{chunk.file_path} {chunk.symbol_name} "
                f"{chunk.docstring or ''} {chunk.source}"
            )
            tokens = tokenize_code(text)
            term_freqs = dict(Counter(tokens))
            docs.append(
                BM25Document(
                    chunk=chunk,
                    tokens=tokens,
                    doc_len=len(tokens),
                    term_freqs=term_freqs,
                )
            )
        return cls(docs)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """
        Search the BM25 index for query keywords.
        Returns ranked chunks with score and rank.
        """
        query_tokens = tokenize_code(query)
        if not query_tokens or self.num_docs == 0:
            return []

        scores: list[tuple[float, BM25Document]] = []
        for doc in self.docs:
            score = 0.0
            for term in query_tokens:
                if term not in doc.term_freqs:
                    continue
                tf = doc.term_freqs[term]
                idf = self.idf.get(term, 0.0)

                # BM25 formula
                numerator = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc.doc_len / self.avg_doc_len))
                score += idf * (numerator / denom)

            if score > 0.0:
                scores.append((score, doc))

        scores.sort(key=lambda x: x[0], reverse=True)

        results: list[dict[str, Any]] = []
        for rank, (score, doc) in enumerate(scores[:top_k], start=1):
            chunk = doc.chunk
            results.append(
                {
                    "file_path": chunk.file_path,
                    "symbol_name": chunk.symbol_name,
                    "symbol_type": chunk.symbol_type,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "docstring": chunk.docstring or "",
                    "source": chunk.source,
                    "source_preview": chunk.source[:300],
                    "score": round(score, 4),
                    "rank": rank,
                }
            )
        return results
