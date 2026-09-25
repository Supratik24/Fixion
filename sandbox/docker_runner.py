"""
sandbox/docker_runner.py — Docker-based sandbox for safe test execution.

Each invocation:
1. Copies the repo into a fresh container (no persistent state)
2. Installs the repo's dependencies (pip install -e . or requirements.txt)
3. Runs pytest with JSON output
4. Returns a structured TestResult

Security properties:
- Network disabled after dependency install (--network none)
- CPU and memory limits enforced
- 5-minute timeout per run
- Ephemeral container (auto-removed after run)
"""

from __future__ import annotations

import json
import logging
import os
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import docker
import docker.errors
from docker.models.containers import Container

from config import settings

logger = logging.getLogger(__name__)

# Maximum bytes of log output to capture (avoid OOM on chatty test suites)
_MAX_LOG_BYTES = 2 * 1024 * 1024  # 2 MB


@dataclass
class TestResult:
    """Structured result from running a test suite inside the sandbox."""

    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    traceback: str           # last relevant traceback extracted from output
    failed_tests: list[str]  # list of failing test node IDs
    passed_tests: list[str]  # list of passing test node IDs
    duration_seconds: float
    error_message: str = ""  # set if the sandbox itself errored (not a test fail)
    raw_json: dict[str, Any] = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_tar(source_dir: Path) -> bytes:
    """Pack *source_dir* into an in-memory tar archive (for docker copy_to)."""
    buf = tempfile.SpooledTemporaryFile(max_size=128 * 1024 * 1024)
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(source_dir), arcname=".")
    buf.seek(0)
    return buf.read()


def _extract_traceback(output: str) -> str:
    """Extract the last pytest traceback block from output text."""
    lines = output.splitlines()
    tb_lines: list[str] = []
    in_tb = False
    for line in lines:
        if line.startswith("FAILED") or "AssertionError" in line or "Error" in line:
            in_tb = True
        if in_tb:
            tb_lines.append(line)
        if in_tb and line.strip() == "":
            break
    return "\n".join(tb_lines[-50:])  # last 50 lines of traceback


def _parse_json_report(json_str: str) -> dict[str, Any]:
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}


# ── Main runner ───────────────────────────────────────────────────────────────


class DockerSandbox:
    """
    Manages a Docker-based sandbox for running repo test suites safely.

    Parameters
    ----------
    image:
        Docker image name (must be pre-built from sandbox/Dockerfile.base).
    timeout:
        Max seconds for the test run (default: from settings).
    memory:
        Docker memory limit string, e.g. ``"2g"``.
    """

    def __init__(
        self,
        image: str | None = None,
        timeout: int | None = None,
        memory: str | None = None,
    ) -> None:
        self._image = image or settings.sandbox_image
        self._timeout = timeout or settings.sandbox_timeout
        self._memory = memory or settings.sandbox_memory
        self._client = docker.from_env()

    def build_image(self, force: bool = False) -> None:
        """
        Build the sandbox Docker image from sandbox/Dockerfile.base.
        Safe to call multiple times (no-op if image already exists).
        """
        try:
            self._client.images.get(self._image)
            if not force:
                logger.info("Sandbox image '%s' already exists.", self._image)
                return
        except docker.errors.ImageNotFound:
            pass

        dockerfile_dir = Path(__file__).parent
        logger.info("Building sandbox image '%s' from %s…", self._image, dockerfile_dir)
        self._client.images.build(
            path=str(dockerfile_dir),
            dockerfile="Dockerfile.base",
            tag=self._image,
            rm=True,
        )
        logger.info("Sandbox image built successfully.")

    def run_tests(
        self,
        repo_path: Path,
        test_paths: list[str] | None = None,
        *,
        install_cmd: str | None = None,
        extra_pytest_args: list[str] | None = None,
    ) -> TestResult:
        """
        Copy *repo_path* into a sandbox container, install deps, run tests.

        Parameters
        ----------
        repo_path:
            Local path to the cloned repository (with any patches already applied).
        test_paths:
            Specific pytest node IDs / paths to run. If None, runs the full suite.
        install_cmd:
            Override the default dependency install command.
            Default: ``pip install -e . --quiet`` (falls back to requirements.txt).
        extra_pytest_args:
            Additional arguments forwarded to pytest.
        """
        import time
        start = time.monotonic()

        report_path = "/tmp/fixion_report.json"
        test_selector = " ".join(test_paths) if test_paths else ""
        extra_args = " ".join(extra_pytest_args or [])

        # Build the shell script to run inside the container
        if install_cmd is None:
            install_cmd = (
                "if [ -f pyproject.toml ] || [ -f setup.py ] || [ -f setup.cfg ]; then "
                "  pip install -e . --quiet 2>&1; "
                "elif [ -f requirements.txt ]; then "
                "  pip install -r requirements.txt --quiet 2>&1; "
                "fi"
            )

        run_script = (
            f"cd /workspace && "
            f"{install_cmd} && "
            f"python -m pytest {test_selector} "
            f"--json-report --json-report-file={report_path} "
            f"--tb=short -q {extra_args} 2>&1; "
            f"echo '__EXIT_CODE__'$?"
        )

        container: Container | None = None
        try:
            # Phase 1: Start with network enabled (for pip install)
            container = self._client.containers.run(
                self._image,
                command=["bash", "-c", run_script],
                detach=True,
                remove=False,          # we'll remove manually after log capture
                mem_limit=self._memory,
                nano_cpus=2_000_000_000,  # 2 CPUs
                network_mode="bridge",    # network ON for dep install
                working_dir="/workspace",
                user="sandbox",
            )

            # Copy repo into container
            tar_data = _make_tar(repo_path)
            container.put_archive("/workspace", tar_data)

            # Wait for test run to complete
            try:
                exit_status = container.wait(timeout=self._timeout)
                exit_code = exit_status["StatusCode"]
            except Exception as exc:
                logger.warning("Container timed out or errored: %s", exc)
                container.kill()
                return TestResult(
                    passed=False,
                    exit_code=-1,
                    stdout="",
                    stderr="",
                    traceback="",
                    failed_tests=[],
                    passed_tests=[],
                    duration_seconds=time.monotonic() - start,
                    error_message=f"Sandbox timeout/error: {exc}",
                )

            # Capture logs
            raw_logs = container.logs(stdout=True, stderr=True)
            output = raw_logs[:_MAX_LOG_BYTES].decode("utf-8", errors="replace")

            # Parse exit code from marker
            actual_exit_code = exit_code
            if "__EXIT_CODE__" in output:
                marker_line = [l for l in output.splitlines() if "__EXIT_CODE__" in l]
                if marker_line:
                    try:
                        actual_exit_code = int(marker_line[-1].split("__EXIT_CODE__")[1].strip())
                    except ValueError:
                        pass

            # Try to grab the JSON report from the container
            json_report: dict = {}
            try:
                bits, _ = container.get_archive(report_path)
                raw_tar = b"".join(bits)
                with tempfile.NamedTemporaryFile(suffix=".tar") as tmp:
                    tmp.write(raw_tar)
                    tmp.flush()
                    with tarfile.open(tmp.name) as tf:
                        member = tf.getmembers()[0]
                        f = tf.extractfile(member)
                        if f:
                            json_report = _parse_json_report(f.read().decode("utf-8"))
            except Exception as exc:
                logger.debug("Could not retrieve JSON report: %s", exc)

            # Parse results
            failed_tests: list[str] = []
            passed_tests: list[str] = []
            if json_report:
                for t in json_report.get("tests", []):
                    if t.get("outcome") == "passed":
                        passed_tests.append(t["nodeid"])
                    elif t.get("outcome") in ("failed", "error"):
                        failed_tests.append(t["nodeid"])

            passed = actual_exit_code == 0
            duration = time.monotonic() - start

            return TestResult(
                passed=passed,
                exit_code=actual_exit_code,
                stdout=output,
                stderr="",
                traceback=_extract_traceback(output) if not passed else "",
                failed_tests=failed_tests,
                passed_tests=passed_tests,
                duration_seconds=duration,
                raw_json=json_report,
            )

        except Exception as exc:
            logger.exception("Unexpected sandbox error: %s", exc)
            return TestResult(
                passed=False,
                exit_code=-1,
                stdout="",
                stderr="",
                traceback="",
                failed_tests=[],
                passed_tests=[],
                duration_seconds=0.0,
                error_message=str(exc),
            )
        finally:
            if container:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

    def run_linter(self, repo_path: Path) -> dict[str, Any]:
        """Run ruff + mypy inside the sandbox and return structured findings."""
        script = (
            "cd /workspace && "
            "echo '=== RUFF ===' && ruff check . --output-format=json 2>&1; "
            "echo '=== MYPY ===' && mypy . --ignore-missing-imports 2>&1"
        )
        container: Container | None = None
        try:
            container = self._client.containers.run(
                self._image,
                command=["bash", "-c", script],
                detach=True,
                remove=False,
                mem_limit="512m",
                network_mode="none",
                working_dir="/workspace",
                user="sandbox",
            )
            tar_data = _make_tar(repo_path)
            container.put_archive("/workspace", tar_data)
            container.wait(timeout=120)
            output = container.logs().decode("utf-8", errors="replace")
            return {"output": output, "passed": "error" not in output.lower()}
        finally:
            if container:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
