"""
ingest/traceback_parser.py — Parse pytest and Python tracebacks into structured frames.

Extracts:
  - Error type (e.g. AssertionError, ZeroDivisionError, TypeError)
  - Error message and failing assertion details
  - Structured frames (file path, line number, function name, source line)
  - Identifies the primary repository frame (excluding site-packages / test runner internals)
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ParsedFrame:
    """A single frame in a stack trace."""

    file_path: str
    line_number: int
    function_name: str
    source_line: str = ""
    is_test_file: bool = False
    is_framework: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParsedTraceback:
    """Structured breakdown of a test or execution traceback."""

    error_type: str = ""
    error_message: str = ""
    failing_assertion: str = ""
    frames: list[ParsedFrame] = None  # type: ignore[assignment]
    primary_frame: ParsedFrame | None = None
    raw_traceback: str = ""

    def __post_init__(self) -> None:
        if self.frames is None:
            self.frames = []
        if self.primary_frame is None and self.frames:
            # Pick the best non-framework frame, preferring non-test source files
            repo_frames = [f for f in self.frames if not f.is_framework]
            src_frames = [f for f in repo_frames if not f.is_test_file]
            if src_frames:
                self.primary_frame = src_frames[-1]
            elif repo_frames:
                self.primary_frame = repo_frames[-1]
            else:
                self.primary_frame = self.frames[-1]

    def to_dict(self) -> dict:
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "failing_assertion": self.failing_assertion,
            "primary_frame": self.primary_frame.to_dict() if self.primary_frame else None,
            "frames": [f.to_dict() for f in self.frames],
        }


# Regex patterns for different Python / pytest traceback formats
# Pattern 1: Standard Python traceback: File "path/to/file.py", line 42, in func_name
_STD_FRAME_RE = re.compile(
    r'File\s+["\'](?P<file>[^"\']+)["\'],\s+line\s+(?P<line>\d+)(?:,\s+in\s+(?P<func>\w+))?',
    re.IGNORECASE,
)

# Pattern 2: Pytest short traceback: path/to/file.py:42: in func_name OR path/to/file.py:42: ErrorType
_PYTEST_FRAME_RE = re.compile(
    r'^(?P<file>[a-zA-Z0-9_\-\./\\]+\.py):(?P<line>\d+)(?::\s+(?:in\s+(?P<func>\w+)|(?P<msg>.+)))?$',
    re.MULTILINE,
)

# Pattern 3: Pytest failure line marker: E   AssertionError: ... or E       assert 0 == 1
_PYTEST_ERROR_LINE_RE = re.compile(
    r"^E\s+(?:(?P<err_type>[A-Za-z_]\w*Error|[A-Za-z_]\w*Exception):\s*(?P<err_msg>.*)|(?P<assertion>assert\s+.*))$",
    re.MULTILINE,
)


def _is_framework_path(path_str: str) -> bool:
    norm = path_str.replace("\\", "/").lower()
    framework_markers = [
        "site-packages",
        "dist-packages",
        "/_pytest/",
        "/pytest/",
        "<frozen ",
        "lib/python",
        "python3.",
    ]
    return any(marker in norm for marker in framework_markers)


def _is_test_path(path_str: str) -> bool:
    norm = path_str.replace("\\", "/").lower()
    return "test_" in norm or "_test.py" in norm or "/tests/" in norm or "/test/" in norm


def parse_traceback(text: str, repo_root: Path | str | None = None) -> ParsedTraceback:
    """
    Parse a pytest or Python error traceback string into structured frames and assertion details.
    """
    if not text or not text.strip():
        return ParsedTraceback(raw_traceback=text)

    frames: list[ParsedFrame] = []
    error_type = ""
    error_message = ""
    failing_assertion = ""

    lines = text.splitlines()

    # Pass 1: Parse frames
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check standard Python traceback frame
        std_match = _STD_FRAME_RE.search(line)
        if std_match:
            f_path = std_match.group("file")
            l_num = int(std_match.group("line"))
            f_name = std_match.group("func") or "<module>"
            src_line = ""
            if i + 1 < len(lines) and not lines[i + 1].strip().startswith("File "):
                src_line = lines[i + 1].strip()

            frames.append(
                ParsedFrame(
                    file_path=f_path,
                    line_number=l_num,
                    function_name=f_name,
                    source_line=src_line,
                    is_test_file=_is_test_path(f_path),
                    is_framework=_is_framework_path(f_path),
                )
            )
            i += 1
            continue

        # Check pytest short format frame
        pytest_match = _PYTEST_FRAME_RE.match(line)
        if pytest_match:
            f_path = pytest_match.group("file")
            l_num = int(pytest_match.group("line"))
            f_name = pytest_match.group("func") or ""
            msg = pytest_match.group("msg") or ""

            src_line = ""
            if i + 1 < len(lines) and not lines[i + 1].startswith("E "):
                src_line = lines[i + 1].strip()

            frames.append(
                ParsedFrame(
                    file_path=f_path,
                    line_number=l_num,
                    function_name=f_name,
                    source_line=src_line,
                    is_test_file=_is_test_path(f_path),
                    is_framework=_is_framework_path(f_path),
                )
            )

            # Sometimes error type is in pytest short header
            if msg and ("Error" in msg or "Exception" in msg):
                parts = msg.split(":", 1)
                error_type = parts[0].strip()
                if len(parts) > 1:
                    error_message = parts[1].strip()

        i += 1

    # Pass 2: Extract error types & assertions from E lines or trailing lines
    fallback_assert = ""
    for line in lines:
        line_clean = line.strip()

        # Match pytest error prefix: E ...
        e_match = _PYTEST_ERROR_LINE_RE.match(line_clean)
        if e_match:
            if e_match.group("err_type"):
                error_type = e_match.group("err_type")
                error_message = e_match.group("err_msg") or ""
                if error_message.startswith("assert "):
                    failing_assertion = error_message
            elif e_match.group("assertion"):
                failing_assertion = e_match.group("assertion")

        # Fallback for standard Python exception line at the end: TypeError: ...
        if not error_type and (":" in line_clean) and ("Error" in line_clean or "Exception" in line_clean):
            parts = line_clean.split(":", 1)
            candidate_type = parts[0].strip()
            if candidate_type.endswith(("Error", "Exception")) and " " not in candidate_type:
                error_type = candidate_type
                error_message = parts[1].strip()

        if "assert " in line_clean and not fallback_assert and not line_clean.startswith("E "):
            fallback_assert = line_clean

    if not failing_assertion:
        failing_assertion = fallback_assert

    # Clean relative repo paths if repo_root is provided
    if repo_root:
        root_str = str(repo_root).replace("\\", "/").rstrip("/")
        for f in frames:
            norm_p = f.file_path.replace("\\", "/")
            if norm_p.startswith(root_str):
                f.file_path = norm_p[len(root_str):].lstrip("/")

    return ParsedTraceback(
        error_type=error_type,
        error_message=error_message,
        failing_assertion=failing_assertion,
        frames=frames,
        raw_traceback=text,
    )
