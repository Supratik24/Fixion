"""
agent/tools/read_file.py — Read a slice of a file from the local repo.

The agent uses this to inspect specific lines after localization, before
generating a patch. Line ranges are 1-indexed and inclusive.
"""

from __future__ import annotations

from pathlib import Path


def read_file(
    file_path: str,
    repo_root: Path,
    start_line: int | None = None,
    end_line: int | None = None,
    context_lines: int = 5,
) -> dict[str, object]:
    """
    Read a file (or a slice of it) from the repo.

    Parameters
    ----------
    file_path:
        Path relative to repo root (e.g. ``"src/stats.py"``).
    repo_root:
        Absolute path to the cloned repository.
    start_line:
        First line to return (1-indexed). If None, starts from line 1.
    end_line:
        Last line to return (1-indexed, inclusive). If None, reads to EOF.
    context_lines:
        Extra lines of context to include before/after the requested range.

    Returns
    -------
    dict with:
        - ``content``: the requested lines as a string
        - ``total_lines``: total line count of the file
        - ``start_line``: actual start line returned
        - ``end_line``: actual end line returned
        - ``file_path``: the relative file path
    """
    abs_path = repo_root / file_path
    if not abs_path.exists():
        return {
            "error": f"File not found: {file_path}",
            "file_path": file_path,
        }

    try:
        all_lines = abs_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    except OSError as exc:
        return {"error": str(exc), "file_path": file_path}

    total = len(all_lines)

    # Apply context padding
    actual_start = max(1, (start_line or 1) - context_lines)
    actual_end = min(total, (end_line or total) + context_lines)

    selected = all_lines[actual_start - 1 : actual_end]

    # Annotate with line numbers
    annotated = "".join(
        f"{actual_start + i:5d} | {line}"
        for i, line in enumerate(selected)
    )

    return {
        "file_path": file_path,
        "content": annotated,
        "total_lines": total,
        "start_line": actual_start,
        "end_line": actual_end,
    }


# ── Gemini tool declaration ────────────────────────────────────────────────────

TOOL_DECLARATION = {
    "name": "read_file",
    "description": (
        "Read the contents of a specific file in the repository. "
        "Optionally specify a line range to read only the relevant section. "
        "Lines are annotated with their numbers for easy reference when writing a patch."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file relative to the repository root.",
            },
            "start_line": {
                "type": "integer",
                "description": "First line to read (1-indexed). Omit to start from line 1.",
            },
            "end_line": {
                "type": "integer",
                "description": "Last line to read (1-indexed, inclusive). Omit to read to EOF.",
            },
            "context_lines": {
                "type": "integer",
                "description": "Extra lines of context to include before and after the range.",
                "default": 5,
            },
        },
        "required": ["file_path"],
    },
}
