"""
cli.py — Fixion command-line interface.

Usage examples:

  # Fix a bug in a GitHub repo
  fixion fix --repo https://github.com/psf/requests \
             --issue "ConnectionError when response has no content-length" \
             --test "tests/test_adapters.py::test_empty_response"

  # Index a repo (for fast subsequent fixes)
  fixion index --repo https://github.com/psf/requests

  # Run SWE-bench evaluation
  fixion eval --n 30

  # Build the Docker sandbox image
  fixion build-sandbox
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


@click.group()
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging.")
def main(debug: bool) -> None:
    """Fixion — autonomous software engineering agent."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ── fix ────────────────────────────────────────────────────────────────────────


@main.command()
@click.option("--repo", required=True, help="GitHub repository URL (HTTPS).")
@click.option("--issue", required=True, help="Issue description or bug report text.")
@click.option(
    "--test",
    "failing_tests",
    multiple=True,
    help="Failing test node ID(s) to focus on. Repeatable.",
)
@click.option(
    "--output",
    default=None,
    help="Write the patch to this file (default: print to stdout).",
)
@click.option(
    "--max-retries",
    default=None,
    type=int,
    help="Override the max patch-fix-test iterations.",
)
def fix(
    repo: str,
    issue: str,
    failing_tests: tuple[str, ...],
    output: str | None,
    max_retries: int | None,
) -> None:
    """Reproduce, localize, patch, and verify a bug in a GitHub repository."""
    from agent.loop import AgentContext, run_agent
    from config import settings
    from ingest.ast_parser import parse_repo
    from ingest.clone import clone_repo
    from ingest.embed_index import EmbedIndex
    from ingest.symbol_graph import build_symbol_graph
    from sandbox.docker_runner import DockerSandbox

    console.print(Panel.fit(f"[bold cyan]Fixion[/] — fixing: [yellow]{repo}[/]"))

    with console.status("[bold green]Cloning repository…"):
        clone_result = clone_repo(repo)

    console.print(f"[OK] Cloned to [dim]{clone_result.local_path}[/] @ [cyan]{clone_result.head_sha[:8]}[/]")

    with console.status("[bold green]Parsing and indexing…"):
        chunks = parse_repo(clone_result.local_path)
        sg = build_symbol_graph(chunks)
        embed_idx = EmbedIndex.build(chunks, repo, clone_result.head_sha)

    console.print(f"[OK] Indexed [cyan]{len(chunks)}[/] code chunks")

    with console.status("[bold green]Building sandbox…"):
        sandbox = DockerSandbox()
        sandbox.build_image()

    console.print("✓ Sandbox ready")

    ctx = AgentContext(
        repo_root=clone_result.local_path,
        repo_url=repo,
        issue_text=issue,
        failing_tests=list(failing_tests),
        embed_index=embed_idx,
        symbol_graph=sg,
        sandbox=sandbox,
        max_retries=max_retries or settings.max_retries,
    )

    console.print("\n[bold]Starting agent loop…[/]\n")
    result = run_agent(ctx)

    # Print result
    status = "[bold green]✓ RESOLVED[/]" if result.success else "[bold red]✗ UNRESOLVED[/]"
    console.print(Panel(f"{status}  |  iterations: {result.iterations}  |  confidence: {result.confidence}"))

    console.print(f"\n[bold]Root Cause:[/]\n{result.root_cause}")
    console.print(f"\n[bold]Fix:[/]\n{result.fix_explanation}")

    if result.patch:
        console.print("\n[bold]Patch:[/]")
        console.print(Syntax(result.patch, "diff", theme="monokai", line_numbers=False))

        if output:
            Path(output).write_text(result.patch, encoding="utf-8")
            console.print(f"\n✓ Patch written to [cyan]{output}[/]")
    else:
        console.print("\n[dim]No patch generated.[/]")

    if result.tests_passed:
        console.print(f"\n[green]Passing tests:[/] {', '.join(result.tests_passed[:5])}")
    if result.tests_failed:
        console.print(f"[red]Still failing:[/] {', '.join(result.tests_failed[:5])}")

    sys.exit(0 if result.success else 1)


# ── index ──────────────────────────────────────────────────────────────────────


@main.command()
@click.option("--repo", required=True, help="GitHub repository URL (HTTPS).")
def index(repo: str) -> None:
    """Pre-index a repository for faster subsequent fix runs."""
    from ingest.ast_parser import parse_repo
    from ingest.clone import clone_repo
    from ingest.embed_index import EmbedIndex
    from ingest.symbol_graph import build_symbol_graph

    with console.status(f"Cloning {repo}…"):
        clone_result = clone_repo(repo)
    with console.status("Parsing…"):
        chunks = parse_repo(clone_result.local_path)
    with console.status("Building symbol graph…"):
        sg = build_symbol_graph(chunks)
    with console.status("Embedding and indexing…"):
        EmbedIndex.build(chunks, repo, clone_result.head_sha)

    console.print(f"✓ Indexed [cyan]{len(chunks)}[/] chunks for [yellow]{repo}[/]")


# ── build-sandbox ──────────────────────────────────────────────────────────────


@main.command("build-sandbox")
def build_sandbox() -> None:
    """Build the Docker sandbox image (run once before using fix/eval)."""
    from sandbox.docker_runner import DockerSandbox

    with console.status("Building Docker image…"):
        sandbox = DockerSandbox()
        sandbox.build_image(force=True)

    console.print("[green]✓ Sandbox image built successfully.[/]")


# ── eval ───────────────────────────────────────────────────────────────────────


@main.command()
@click.option("--n", default=10, show_default=True, help="Number of SWE-bench Lite instances to evaluate.")
@click.option("--instance-id", default=None, help="Evaluate a single specific instance.")
@click.option("--output-dir", default="eval_results", show_default=True)
@click.option("--summary-file", default="eval_summary.json", show_default=True)
@click.pass_context
def eval(ctx: click.Context, n: int, instance_id: str | None, output_dir: str, summary_file: str) -> None:
    """Run Fixion against SWE-bench Lite and report resolve-rate."""
    from eval.swebench_harness import main as eval_main

    ctx.invoke(
        eval_main,
        n=n,
        instance_id=instance_id,
        split="test",
        output_dir=output_dir,
        summary_file=summary_file,
    )


if __name__ == "__main__":
    main()
