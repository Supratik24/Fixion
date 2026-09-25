"""
eval/metrics.py — Metrics for evaluating the Fixion agent.

Key metrics:
  - resolve_rate: % of issues where the patch passes the gold test suite
  - localization_accuracy: % where we correctly identified the root-cause file
  - patch_validity_rate: % where the diff applied without error
  - avg_iterations: mean number of fix-test cycles per issue
  - cost_per_issue: estimated token cost (requires tracking)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IssueMetrics:
    """Metrics for a single issue run."""

    instance_id: str
    resolved: bool              # patch passed gold test suite
    patch_valid: bool           # diff applied cleanly
    iterations: int             # number of fix-test cycles
    correct_localization: bool  # root-cause file correctly identified
    duration_seconds: float
    error: str = ""             # set if agent crashed


@dataclass
class EvalMetrics:
    """Aggregate metrics across all evaluated issues."""

    total: int = 0
    resolved: int = 0
    patch_valid: int = 0
    correct_localization: int = 0
    total_iterations: int = 0
    total_duration: float = 0.0
    issues: list[IssueMetrics] = field(default_factory=list)

    def add(self, m: IssueMetrics) -> None:
        self.total += 1
        self.issues.append(m)
        if m.resolved:
            self.resolved += 1
        if m.patch_valid:
            self.patch_valid += 1
        if m.correct_localization:
            self.correct_localization += 1
        self.total_iterations += m.iterations
        self.total_duration += m.duration_seconds

    @property
    def resolve_rate(self) -> float:
        return self.resolved / self.total if self.total else 0.0

    @property
    def patch_validity_rate(self) -> float:
        return self.patch_valid / self.total if self.total else 0.0

    @property
    def localization_accuracy(self) -> float:
        return self.correct_localization / self.total if self.total else 0.0

    @property
    def avg_iterations(self) -> float:
        return self.total_iterations / self.total if self.total else 0.0

    @property
    def avg_duration(self) -> float:
        return self.total_duration / self.total if self.total else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "total_issues": self.total,
            "resolved": self.resolved,
            "resolve_rate": f"{self.resolve_rate:.1%}",
            "patch_validity_rate": f"{self.patch_validity_rate:.1%}",
            "localization_accuracy": f"{self.localization_accuracy:.1%}",
            "avg_iterations": round(self.avg_iterations, 2),
            "avg_duration_seconds": round(self.avg_duration, 1),
        }

    def print_report(self) -> None:
        s = self.summary()
        print("\n" + "=" * 60)
        print("FIXION EVALUATION REPORT")
        print("=" * 60)
        for k, v in s.items():
            print(f"  {k:<30} {v}")
        print("=" * 60)

        # Per-issue breakdown
        print("\nPer-Issue Results:")
        print(f"  {'Instance ID':<40} {'Resolved':<10} {'Iters':<6} {'Time(s)'}")
        print("  " + "-" * 70)
        for m in self.issues:
            status = "✓" if m.resolved else "✗"
            print(f"  {m.instance_id:<40} {status:<10} {m.iterations:<6} {m.duration_seconds:.1f}")
        print()
