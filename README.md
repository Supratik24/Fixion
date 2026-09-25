# Fixion 🔧

**SWE-bench-style autonomous coding agent** — read → localize → debug → fix → test → verify.

Fixion is a closed-loop AI agent that:
1. Clones a GitHub repository and builds a code index (AST chunks + symbol graph + embeddings)
2. Understands a bug report or failing test
3. Reproduces the bug in a Docker sandbox to get a real traceback
4. Localizes the root cause (exact file + function + line)
5. Generates a minimal unified diff patch
6. Verifies the fix passes tests — and retries if it doesn't

## Quick Start

### Prerequisites

- Python 3.11+
- Docker Desktop (running)
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

### Install

```bash
git clone https://github.com/Supratik24/Fixion
cd fixion

pip install -e ".[dev]"

cp .env.example .env
# Edit .env and set GEMINI_API_KEY
```

### Build the sandbox image (one-time)

```bash
fixion build-sandbox
```

### Fix a bug

```bash
fixion fix \
  --repo https://github.com/psf/requests \
  --issue "ConnectionError raised when response has no Content-Length header" \
  --test "tests/test_adapters.py::test_stream_response"
```

The agent will:
- Reproduce the failing test
- Localize the root cause using symbol graph + semantic search
- Generate and apply a patch
- Re-run tests, retry up to 5 times if needed
- Print the verified diff and root-cause explanation

### Pre-index a repo (for faster fixes)

```bash
fixion index --repo https://github.com/psf/requests
```

### Run SWE-bench evaluation

```bash
# Quick smoke test: 10 instances
fixion eval --n 10

# Full SWE-bench Lite (300 instances)
fixion eval --n 300 --output-dir eval_results/
```

## Architecture

```
Fixion/
├── ingest/             # M1 — Repo ingestion & indexing
│   ├── clone.py        # GitPython repo cloning
│   ├── ast_parser.py   # tree-sitter AST chunking
│   ├── symbol_graph.py # NetworkX import/call graph
│   └── embed_index.py  # Gemini embeddings + LanceDB
│
├── sandbox/            # M2 — Safe execution environment
│   ├── docker_runner.py
│   └── Dockerfile.base
│
├── agent/              # M3-M5 — Reasoning & tool-calling loop
│   ├── loop.py         # Main agent loop (Gemini tool calling)
│   ├── tools/          # 14 tools: search, read, find, patch, test…
│   └── prompts/        # System, localize, patch prompts
│
├── eval/               # M6 — SWE-bench evaluation harness
│   ├── swebench_harness.py
│   └── metrics.py
│
├── tests/              # Unit + integration tests for Fixion itself
├── cli.py              # Click CLI entrypoint
└── config.py           # Centralized settings (pydantic-settings)
```

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Language | Python 3.11 | Ecosystem fit |
| AST Parsing | `tree-sitter` | Language-aware chunks |
| Symbol Graph | `networkx` | Import/call graph |
| Embeddings | Gemini `text-embedding-004` | Native, no extra key |
| Vector DB | `LanceDB` | Embedded, fast, no server |
| LLM Agent | Gemini 2.5 Pro | Strong code reasoning |
| Sandbox | Docker (`docker-py`) | Safe execution |
| Test runner | `pytest --json-report` | Structured output |
| Static analysis | `ruff` + `mypy` | Cheap pre-filter |
| Git ops | `gitpython` + `subprocess` | Clone, diff, apply |
| CLI | `click` + `rich` | Clean UX |
| Eval | SWE-bench Lite | Industry benchmark |

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✓ | Google Gemini API key |
| `GITHUB_TOKEN` | Optional | For private repos + PR creation |
| `FIXION_MODEL` | Optional | Gemini model (default: gemini-2.5-pro) |
| `FIXION_MAX_RETRIES` | Optional | Max fix iterations (default: 5) |
| `FIXION_SANDBOX_TIMEOUT` | Optional | Test timeout seconds (default: 300) |

## Running Tests

```bash
# Unit tests (no Docker required)
pytest tests/test_ast_parser.py tests/test_symbol_graph.py -v

# Integration tests (requires Docker)
pytest tests/test_sandbox.py -v

# All tests
pytest -v
```

## Key Design Decisions

**Why unified diffs, not full file rewrites?**
Smaller surface area, easier to validate with `git apply --check`, closer to how real PRs look.

**Why reproduce first, before reasoning?**
Real tracebacks beat issue text. The localization accuracy difference is significant.

**Why max 5 retries?**
Each retry burns ~2-4 LLM calls. Cost control and avoiding infinite loops.

**Why LanceDB over Chroma?**
Embedded (no server process), stores to a directory, fast for <1M vectors.

## Evaluation

Fixion is designed to be evaluated on [SWE-bench Lite](https://www.swebench.com/) — 300 curated real GitHub issues with gold test patches.

Key metric: **resolve_rate** = percentage of issues where Fixion's patch passes the gold test suite.

```bash
python -m eval.swebench_harness --n 30 --output-dir eval_results/
```

## License

MIT
