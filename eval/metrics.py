"""
eval/metrics.py — Metrics for evaluating the Fixion agent.

Key metrics recorded per job:
  - tokens in / out
  - tool-call count
  - iterations & retries
  - wall time
  - final status (resolved / unresolved / max_retries_exceeded / error)
  - estimated cost (USD)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Pricing estimates per million tokens ($/1M tokens: input, output)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_1m, output_per_1m)
    "gemini": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 5.00),
    "gemini-3.8-flash": (0.10, 0.40),
    "groq": (0.59, 0.79),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "openrouter": (0.55, 2.19),
    "deepseek/deepseek-r1": (0.55, 2.19),
    "ollama": (0.0, 0.0),
}


def estimate_cost(backend: str, model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimate total inference cost in USD based on model or backend rates."""
    b_lower = backend.lower()
    m_lower = model.lower()

    rates = (
        MODEL_PRICING.get(m_lower)
        or MODEL_PRICING.get(b_lower)
        or (0.20, 0.80)
    )
    in_rate, out_rate = rates
    cost = (tokens_in / 1_000_000.0) * in_rate + (tokens_out / 1_000_000.0) * out_rate
    return round(cost, 6)


@dataclass
class IssueMetrics:
    """Metrics for a single issue run."""

    instance_id: str
    resolved: bool              # patch passed gold test suite
    patch_valid: bool           # diff applied cleanly
    iterations: int             # number of fix-test cycles
    correct_localization: bool  # root-cause file correctly identified
    duration_seconds: float
    tokens_in: int = 0
    tokens_out: int = 0
    tool_call_count: int = 0
    retries: int = 0
    wall_time: float = 0.0
    status: str = "unresolved"  # "resolved" | "unresolved" | "max_retries_exceeded" | "error"
    backend: str = "gemini"
    model: str = ""
    cost_usd: float = 0.0
    error: str = ""             # set if agent crashed

    def __post_init__(self) -> None:
        if self.wall_time <= 0.0 and self.duration_seconds > 0.0:
            self.wall_time = self.duration_seconds
        if self.duration_seconds <= 0.0 and self.wall_time > 0.0:
            self.duration_seconds = self.wall_time
        if self.cost_usd == 0.0 and (self.tokens_in > 0 or self.tokens_out > 0):
            self.cost_usd = estimate_cost(self.backend, self.model, self.tokens_in, self.tokens_out)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalMetrics:
    """Aggregate metrics across all evaluated issues."""

    total: int = 0
    resolved: int = 0
    patch_valid: int = 0
    correct_localization: int = 0
    total_iterations: int = 0
    total_duration: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_tool_calls: int = 0
    total_retries: int = 0
    total_cost_usd: float = 0.0
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
        self.total_duration += m.wall_time or m.duration_seconds
        self.total_tokens_in += m.tokens_in
        self.total_tokens_out += m.tokens_out
        self.total_tool_calls += m.tool_call_count
        self.total_retries += m.retries
        self.total_cost_usd += m.cost_usd

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

    @property
    def avg_tokens_in(self) -> float:
        return self.total_tokens_in / self.total if self.total else 0.0

    @property
    def avg_tokens_out(self) -> float:
        return self.total_tokens_out / self.total if self.total else 0.0

    @property
    def avg_tool_calls(self) -> float:
        return self.total_tool_calls / self.total if self.total else 0.0

    @property
    def avg_retries(self) -> float:
        return self.total_retries / self.total if self.total else 0.0

    @property
    def avg_cost_usd(self) -> float:
        return self.total_cost_usd / self.total if self.total else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "total_issues": self.total,
            "resolved": self.resolved,
            "resolve_rate": f"{self.resolve_rate:.1%}",
            "patch_validity_rate": f"{self.patch_validity_rate:.1%}",
            "localization_accuracy": f"{self.localization_accuracy:.1%}",
            "avg_iterations": round(self.avg_iterations, 2),
            "avg_duration_seconds": round(self.avg_duration, 1),
            "avg_tokens_in": round(self.avg_tokens_in, 1),
            "avg_tokens_out": round(self.avg_tokens_out, 1),
            "avg_tool_calls": round(self.avg_tool_calls, 1),
            "avg_retries": round(self.avg_retries, 2),
            "avg_cost_usd": f"${self.avg_cost_usd:.4f}",
            "total_cost_usd": f"${self.total_cost_usd:.4f}",
        }

    def markdown_table(self, backend: str | None = None) -> str:
        """Generate a Markdown summary table broken down by backend."""
        # Group issues by backend
        groups: dict[str, list[IssueMetrics]] = {}
        for issue in self.issues:
            b = issue.backend or backend or "default"
            groups.setdefault(b, []).append(issue)

        if not groups:
            b_name = backend or "default"
            groups[b_name] = []

        headers = [
            "Backend",
            "Total",
            "Resolved",
            "Resolve Rate",
            "Avg Tokens (In/Out)",
            "Avg Cost",
            "Avg Time (s)",
        ]
        divider = ["---"] * len(headers)

        rows = [
            f"| {' | '.join(headers)} |",
            f"| {' | '.join(divider)} |",
        ]

        for b_name, b_issues in groups.items():
            b_total = len(b_issues)
            b_resolved = sum(1 for x in b_issues if x.resolved)
            b_rate = (b_resolved / b_total) if b_total else 0.0
            avg_in = (sum(x.tokens_in for x in b_issues) / b_total) if b_total else 0.0
            avg_out = (sum(x.tokens_out for x in b_issues) / b_total) if b_total else 0.0
            avg_cost = (sum(x.cost_usd for x in b_issues) / b_total) if b_total else 0.0
            avg_time = (sum(x.wall_time for x in b_issues) / b_total) if b_total else 0.0

            rows.append(
                f"| {b_name} | {b_total} | {b_resolved} | {b_rate:.1%} | "
                f"{avg_in:.0f} / {avg_out:.0f} | ${avg_cost:.4f} | {avg_time:.1f}s |"
            )

        return "\n".join(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "issues": [m.to_dict() for m in self.issues],
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
        print(f"  {'Instance ID':<35} {'Status':<10} {'Iters':<6} {'Tokens(In/Out)':<18} {'Cost':<8} {'Time(s)'}")
        print("  " + "-" * 90)
        for m in self.issues:
            status = "✓ Pass" if m.resolved else "✗ Fail"
            tok_str = f"{m.tokens_in}/{m.tokens_out}"
            print(f"  {m.instance_id:<35} {status:<10} {m.iterations:<6} {tok_str:<18} ${m.cost_usd:<7.4f} {m.wall_time:.1f}")
        print()
        print("\nMarkdown Table Summary:")
        print(self.markdown_table())
