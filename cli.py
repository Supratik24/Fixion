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

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

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
@click.option("--repo", default=None, help="GitHub repository URL (HTTPS). Auto-detected if issue URL is provided.")
@click.option("--issue", default=None, help="Issue description or GitHub issue URL.")
@click.option("--issue-url", default=None, help="GitHub issue URL to fetch title and description from.")
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
@click.option(
    "--candidates",
    "-n",
    default=None,
    type=int,
    help="Number of candidate patches to generate/evaluate (default: 3).",
)
@click.option(
    "--pr",
    "--create-pr",
    "create_pr",
    is_flag=True,
    default=False,
    help="Open a Pull Request on GitHub upon success.",
)
@click.option(
    "--no-symbol-graph",
    is_flag=True,
    default=False,
    help="Disable symbol graph construction and navigation tools (ablation).",
)
@click.option(
    "--no-semantic-search",
    is_flag=True,
    default=False,
    help="Disable semantic vector search (ablation).",
)
def fix(
    repo: str | None,
    issue: str | None,
    issue_url: str | None,
    failing_tests: tuple[str, ...],
    output: str | None,
    max_retries: int | None,
    candidates: int | None,
    create_pr: bool,
    no_symbol_graph: bool,
    no_semantic_search: bool,
) -> None:
    """Reproduce, localize, patch, and verify a bug in a GitHub repository."""
    from agent.loop import AgentContext, run_agent
    from agent.tools import git_ops as git_mod
    from config import settings
    from ingest.ast_parser import parse_repo
    from ingest.clone import clone_repo
    from ingest.embed_index import EmbedIndex
    from ingest.github_issue import fetch_github_issue
    from ingest.symbol_graph import build_symbol_graph
    from sandbox.docker_runner import DockerSandbox

    # Resolve issue URL if provided directly or passed as --issue
    target_url = issue_url or (issue if issue and "github.com" in issue and "/issues/" in issue else None)
    if target_url:
        console.print(f"[bold cyan]Fetching issue details from GitHub:[/] {target_url}")
        try:
            issue_info = fetch_github_issue(target_url)
            issue = issue_info["issue_text"]
            if not repo:
                repo = issue_info["repo_url"]
            console.print(f"[bold green]✓ Loaded Issue:[/] {issue_info['title']}\n")
        except Exception as e:
            console.print(f"[bold red]Failed fetching GitHub issue:[/] {e}")
            sys.exit(1)

    if not repo:
        console.print("[bold red]Error:[/] Missing repository. Provide --repo or a valid GitHub --issue URL.")
        sys.exit(1)
    if not issue:
        console.print("[bold red]Error:[/] Missing issue description. Provide --issue or --issue-url.")
        sys.exit(1)

    console.print(Panel.fit(f"[bold cyan]Fixion[/] — fixing: [yellow]{repo}[/]"))

    with console.status("[bold green]Cloning repository…"):
        clone_result = clone_repo(repo)

    console.print(f"[OK] Cloned to [dim]{clone_result.local_path}[/] @ [cyan]{clone_result.head_sha[:8]}[/]")

    with console.status("[bold green]Parsing repository AST…"):
        chunks = parse_repo(clone_result.local_path)

    if not no_symbol_graph:
        with console.status("[bold green]Building symbol graph…"):
            sg = build_symbol_graph(chunks)
        console.print("[OK] Symbol graph built")
    else:
        sg = None
        console.print("[yellow]! Symbol graph disabled for ablation[/]")

    if not no_semantic_search:
        with console.status("[bold green]Indexing embeddings in LanceDB…"):
            embed_idx = EmbedIndex.build(chunks, repo, clone_result.head_sha)
        console.print(f"[OK] Indexed [cyan]{len(chunks)}[/] code chunks")
    else:
        embed_idx = None
        console.print("[yellow]! Semantic search disabled for ablation[/]")

    with console.status("[bold green]Setting up execution sandbox…"):
        sandbox = DockerSandbox()
        if sandbox.is_docker_available():
            sandbox.build_image()
            console.print("✓ Docker sandbox ready")
        else:
            console.print("[yellow]! Docker unavailable; running in local sandbox mode[/]")

    ctx = AgentContext(
        repo_root=clone_result.local_path,
        repo_url=repo,
        issue_text=issue,
        failing_tests=list(failing_tests),
        embed_index=embed_idx,
        symbol_graph=sg,
        sandbox=sandbox,
        max_retries=max_retries or settings.max_retries,
        num_candidates=candidates or settings.num_candidates,
        use_symbol_graph=not no_symbol_graph,
        use_semantic_search=not no_semantic_search,
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

    # Pull Request creation if opt-in enabled
    if create_pr and result.success and result.patch:
        console.print("\n[bold cyan]Creating GitHub Pull Request…[/]")
        first_line = issue.splitlines()[0][:60]
        pr_res = git_mod.create_pull_request(
            repo_root=clone_result.local_path,
            title=f"Fix: {first_line}",
            root_cause=result.root_cause,
            fix_explanation=result.fix_explanation,
            repo_url=repo,
        )
        if pr_res.get("success") and pr_res.get("pr_url"):
            console.print(f"[bold green]✓ Pull Request opened:[/] [cyan underline]{pr_res['pr_url']}[/]")
        else:
            console.print(f"[yellow]! PR creation note:[/] {pr_res.get('message') or pr_res.get('error')}")

    # Metrics Table
    m = result.metrics or {}
    table = Table(title="Fixion Run Metrics", show_header=True, header_style="bold cyan")
    table.add_column("Tokens (In / Out)", justify="center")
    table.add_column("Tool Calls", justify="center")
    table.add_column("Iterations", justify="center")
    table.add_column("Retries", justify="center")
    table.add_column("Wall Time", justify="center")
    table.add_column("Status", justify="center")

    tok_str = f"{m.get('tokens_in', 0):,} / {m.get('tokens_out', 0):,}"
    status_label = m.get("status", "resolved" if result.success else "unresolved")
    table.add_row(
        tok_str,
        str(m.get("tool_call_count", 0)),
        str(m.get("iterations", result.iterations)),
        str(m.get("retries", 0)),
        f"{m.get('wall_time', 0.0):.2f}s",
        status_label,
    )
    console.print("\n", table)

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
        build_symbol_graph(chunks)
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
@click.option("--run-id", default=None, help="Identifier for saving to results/<run_id>.json and .md.")
@click.option("--no-symbol-graph", is_flag=True, default=False, help="Disable symbol graph (ablation).")
@click.option("--no-semantic-search", is_flag=True, default=False, help="Disable semantic search (ablation).")
@click.pass_context
def eval(
    ctx: click.Context,
    n: int,
    instance_id: str | None,
    output_dir: str,
    summary_file: str,
    run_id: str | None,
    no_symbol_graph: bool,
    no_semantic_search: bool,
) -> None:
    """Run Fixion against SWE-bench Lite and report resolve-rate."""
    from eval.swebench_harness import main as eval_main

    ctx.invoke(
        eval_main,
        n=n,
        instance_id=instance_id,
        split="test",
        output_dir=output_dir,
        summary_file=summary_file,
        run_id=run_id,
        no_symbol_graph=no_symbol_graph,
        no_semantic_search=no_semantic_search,
    )


if __name__ == "__main__":
    main()
