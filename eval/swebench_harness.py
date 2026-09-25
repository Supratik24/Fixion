"""
eval/swebench_harness.py — Run Fixion against SWE-bench Lite instances.

SWE-bench Lite: 300 curated Python GitHub issues from popular repos.
Dataset: princeton-nlp/SWE-bench_Lite (HuggingFace datasets)

Each instance has:
  - instance_id: unique ID
  - repo: "owner/repo" on GitHub
  - issue_text / problem_statement: the bug description
  - test_patch: the gold test that should pass after the fix
  - FAIL_TO_PASS: test node IDs that currently fail (should pass after fix)
  - PASS_TO_PASS: tests that should remain passing

Usage:
  python -m eval.swebench_harness --n 10 --output results.json

Or run a single instance:
  python -m eval.swebench_harness --instance-id django__django-12345
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import click
from datasets import load_dataset

from agent.loop import AgentContext, AgentResult, run_agent
from eval.metrics import EvalMetrics, IssueMetrics
from ingest.ast_parser import parse_repo
from ingest.clone import clone_repo, reset_to_head
from ingest.embed_index import EmbedIndex
from ingest.symbol_graph import build_symbol_graph
from sandbox.docker_runner import DockerSandbox

logger = logging.getLogger(__name__)

DATASET_NAME = "princeton-nlp/SWE-bench_Lite"
GITHUB_BASE = "https://github.com/"


def _load_instances(
    n: int | None = None,
    instance_id: str | None = None,
    split: str = "test",
) -> list[dict[str, Any]]:
    """Load SWE-bench Lite instances from HuggingFace."""
    logger.info("Loading SWE-bench Lite dataset (split=%s)…", split)
    ds = load_dataset(DATASET_NAME, split=split)

    if instance_id:
        instances = [i for i in ds if i["instance_id"] == instance_id]
        if not instances:
            raise ValueError(f"Instance '{instance_id}' not found in dataset.")
        return instances

    instances = list(ds)
    if n:
        instances = instances[:n]
    return instances


def _run_single_instance(
    instance: dict[str, Any],
    sandbox: DockerSandbox,
    output_dir: Path,
) -> IssueMetrics:
    """Process one SWE-bench instance end-to-end."""
    instance_id = instance["instance_id"]
    repo_slug = instance["repo"]                # e.g. "psf/requests"
    repo_url = GITHUB_BASE + repo_slug
    issue_text = instance.get("problem_statement", instance.get("issue_text", ""))
    fail_to_pass: list[str] = json.loads(instance.get("FAIL_TO_PASS", "[]"))

    logger.info("Processing instance: %s", instance_id)
    start = time.monotonic()

    try:
        # 1. Clone
        clone_result = clone_repo(repo_url)

        # 2. Index
        chunks = parse_repo(clone_result.local_path)
        sg = build_symbol_graph(chunks)
        embed_idx = EmbedIndex.build(chunks, repo_url, clone_result.head_sha)

        # 3. Run agent
        ctx = AgentContext(
            repo_root=clone_result.local_path,
            repo_url=repo_url,
            issue_text=issue_text,
            failing_tests=fail_to_pass,
            embed_index=embed_idx,
            symbol_graph=sg,
            sandbox=sandbox,
        )
        result: AgentResult = run_agent(ctx)

        # 4. Save output
        out_file = output_dir / f"{instance_id}.json"
        out_file.write_text(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "resolved": result.success,
                    "patch": result.patch,
                    "root_cause": result.root_cause,
                    "fix_explanation": result.fix_explanation,
                    "iterations": result.iterations,
                    "tests_passed": result.tests_passed,
                    "tests_failed": result.tests_failed,
                    "confidence": result.confidence,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # 5. Reset workspace for next run
        reset_to_head(clone_result.local_path)

        duration = time.monotonic() - start
        return IssueMetrics(
            instance_id=instance_id,
            resolved=result.success,
            patch_valid=bool(result.patch),
            iterations=result.iterations,
            correct_localization=result.success,  # proxy: resolved → correct localization
            duration_seconds=duration,
        )

    except Exception as exc:
        logger.exception("Instance %s failed: %s", instance_id, exc)
        duration = time.monotonic() - start
        return IssueMetrics(
            instance_id=instance_id,
            resolved=False,
            patch_valid=False,
            iterations=0,
            correct_localization=False,
            duration_seconds=duration,
            error=str(exc),
        )


@click.command()
@click.option("--n", default=None, type=int, help="Number of instances to evaluate (default: all).")
@click.option("--instance-id", default=None, help="Run a single specific instance.")
@click.option("--split", default="test", show_default=True, help="Dataset split to use.")
@click.option("--output-dir", default="eval_results", show_default=True, help="Directory for per-instance JSON outputs.")
@click.option("--summary-file", default="eval_summary.json", show_default=True, help="Path for aggregate metrics JSON.")
def main(
    n: int | None,
    instance_id: str | None,
    split: str,
    output_dir: str,
    summary_file: str,
) -> None:
    """Run Fixion against SWE-bench Lite and report resolve-rate."""
    logging.basicConfig(level=logging.INFO)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sandbox = DockerSandbox()
    sandbox.build_image()

    instances = _load_instances(n=n, instance_id=instance_id, split=split)
    logger.info("Evaluating %d instances…", len(instances))

    metrics = EvalMetrics()
    for inst in instances:
        m = _run_single_instance(inst, sandbox, out_dir)
        metrics.add(m)
        logger.info(
            "[%s] resolved=%s iterations=%d duration=%.1fs",
            m.instance_id,
            m.resolved,
            m.iterations,
            m.duration_seconds,
        )

    metrics.print_report()

    # Save summary
    Path(summary_file).write_text(
        json.dumps(metrics.summary(), indent=2), encoding="utf-8"
    )
    logger.info("Summary saved to %s", summary_file)


if __name__ == "__main__":
    main()
