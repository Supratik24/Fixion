"""
ingest/ast_parser.py — Tree-sitter-based AST chunking for Python files.

Produces language-aware chunks at function/class boundaries, never arbitrary
character splits. Each chunk carries enough metadata for precise localization
and embedding.

Chunk hierarchy: File → Class → Function/Method
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

logger = logging.getLogger(__name__)

PY_LANGUAGE = Language(tspython.language())
_PARSER = Parser(PY_LANGUAGE)

# Node types that define a "chunk" boundary
CHUNK_NODE_TYPES = frozenset(
    ["function_definition", "async_function_definition", "class_definition"]
)


@dataclass
class CodeChunk:
    """A contiguous unit of source code extracted from the AST."""

    file_path: str           # relative to repo root
    symbol_name: str         # fully qualified: ClassName.method_name
    symbol_type: str         # "function" | "async_function" | "class"
    start_line: int          # 1-indexed, inclusive
    end_line: int            # 1-indexed, inclusive
    source: str              # raw source text of the chunk
    docstring: str           # extracted docstring (empty string if none)
    parent_class: str        # enclosing class name, or "" for module-level
    imports: list[str] = field(default_factory=list)  # top-level import lines


# ── Internal helpers ──────────────────────────────────────────────────────────


def _extract_docstring(node: Node, source_bytes: bytes) -> str:
    """Return the docstring of a function/class node if present."""
    # Body is the last child of function_definition / class_definition
    for child in node.children:
        if child.type == "block":
            for stmt in child.children:
                if stmt.type == "expression_statement":
                    for inner in stmt.children:
                        if inner.type == "string":
                            raw = source_bytes[inner.start_byte : inner.end_byte].decode(
                                "utf-8", errors="replace"
                            )
                            # Strip quotes
                            return raw.strip("'\"").strip()
    return ""


def _get_name(node: Node, source_bytes: bytes) -> str:
    """Extract the identifier name from a definition node."""
    for child in node.children:
        if child.type == "identifier":
            return source_bytes[child.start_byte : child.end_byte].decode("utf-8")
    return "<anonymous>"


def _extract_imports(root: Node, source_bytes: bytes) -> list[str]:
    """Collect all top-level import lines from the module."""
    imports: list[str] = []
    for child in root.children:
        if child.type in ("import_statement", "import_from_statement"):
            imports.append(
                source_bytes[child.start_byte : child.end_byte]
                .decode("utf-8", errors="replace")
                .strip()
            )
    return imports


def _is_async(node: Node) -> bool:
    """Return True if a function_definition node has 'async' as its first child."""
    children = list(node.children)
    return bool(children) and children[0].type == "async"


def _walk_chunks(
    node: Node,
    source_bytes: bytes,
    file_path: str,
    imports: list[str],
    parent_class: str = "",
) -> Iterator[CodeChunk]:
    """Recursively yield CodeChunks from function/class definition nodes."""
    for child in node.children:
        if child.type == "function_definition":
            # In tree-sitter-python ≥0.23 an async function is represented as
            # function_definition with an 'async' keyword sibling, not a separate
            # async_function_definition node type.
            is_async = _is_async(child)
            name = _get_name(child, source_bytes)
            full_name = f"{parent_class}.{name}" if parent_class else name
            source = source_bytes[child.start_byte : child.end_byte].decode(
                "utf-8", errors="replace"
            )
            yield CodeChunk(
                file_path=file_path,
                symbol_name=full_name,
                symbol_type="async_function" if is_async else "function",
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                source=source,
                docstring=_extract_docstring(child, source_bytes),
                parent_class=parent_class,
                imports=imports,
            )
            # Recurse into nested functions (closures) via the function's block
            for grandchild in child.children:
                if grandchild.type == "block":
                    yield from _walk_chunks(grandchild, source_bytes, file_path, imports, full_name)

        elif child.type == "class_definition":
            class_name = _get_name(child, source_bytes)
            full_class = f"{parent_class}.{class_name}" if parent_class else class_name
            class_source = source_bytes[child.start_byte : child.end_byte].decode(
                "utf-8", errors="replace"
            )
            yield CodeChunk(
                file_path=file_path,
                symbol_name=full_class,
                symbol_type="class",
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                source=class_source,
                docstring=_extract_docstring(child, source_bytes),
                parent_class=parent_class,
                imports=imports,
            )
            # Recurse into the class body (methods live inside the 'block' child)
            for grandchild in child.children:
                if grandchild.type == "block":
                    yield from _walk_chunks(grandchild, source_bytes, file_path, imports, full_class)

        elif child.type == "block":
            # Top-level decorated or nested blocks — pass through
            yield from _walk_chunks(child, source_bytes, file_path, imports, parent_class)



# ── Public API ────────────────────────────────────────────────────────────────


def parse_file(file_path: Path, repo_root: Path) -> list[CodeChunk]:
    """
    Parse a single Python file and return all CodeChunks found in it.

    Parameters
    ----------
    file_path:
        Absolute path to the `.py` file.
    repo_root:
        Absolute path to the repo root (used to compute relative path).
    """
    try:
        source_bytes = file_path.read_bytes()
    except (OSError, PermissionError) as exc:
        logger.warning("Cannot read %s: %s", file_path, exc)
        return []

    tree = _PARSER.parse(source_bytes)
    rel_path = str(file_path.relative_to(repo_root))
    imports = _extract_imports(tree.root_node, source_bytes)
    return list(_walk_chunks(tree.root_node, source_bytes, rel_path, imports))


def parse_repo(repo_root: Path, *, exclude_dirs: set[str] | None = None) -> list[CodeChunk]:
    """
    Walk every `.py` file in *repo_root* and return all CodeChunks.

    Parameters
    ----------
    repo_root:
        Root directory of the cloned repository.
    exclude_dirs:
        Directory names to skip (default: ``{".git", "__pycache__", ".venv",
        "node_modules", "dist", "build"}``)
    """
    if exclude_dirs is None:
        exclude_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules", "dist", "build"}

    chunks: list[CodeChunk] = []
    py_files = [
        p
        for p in repo_root.rglob("*.py")
        if not any(part in exclude_dirs for part in p.parts)
    ]
    logger.info("Parsing %d Python files in %s", len(py_files), repo_root)

    for py_file in py_files:
        chunks.extend(parse_file(py_file, repo_root))

    logger.info("Extracted %d code chunks total", len(chunks))
    return chunks
