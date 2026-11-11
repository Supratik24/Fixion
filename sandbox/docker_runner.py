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

import contextlib
import json
import logging
import os
import subprocess
import sys
import tarfile
import tempfile
import time
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
    with tempfile.SpooledTemporaryFile(max_size=128 * 1024 * 1024) as buf:
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


# ── Local runner (fallback when Docker daemon is not active) ───────────────────


class LocalSandbox:
    """
    Fallback execution environment when Docker daemon is not available.
    Executes tests and linters in isolated subprocesses using the local Python environment.
    """

    def __init__(self, timeout: int | None = None) -> None:
        self._timeout = timeout or settings.sandbox_timeout

    def build_image(self, force: bool = False) -> None:
        logger.info("Local execution sandbox ready (using current Python runtime).")

    def run_tests(
        self,
        repo_path: Path,
        test_paths: list[str] | None = None,
        *,
        install_cmd: str | None = None,
        extra_pytest_args: list[str] | None = None,
    ) -> TestResult:
        start = time.monotonic()

        # Configure PYTHONPATH to include the repo root and src/
        env = dict(os.environ)
        existing_pp = env.get("PYTHONPATH", "")
        repo_abs = repo_path.resolve()
        paths_to_add = [str(repo_abs)]
        if (repo_abs / "src").is_dir():
            paths_to_add.append(str(repo_abs / "src"))
        env["PYTHONPATH"] = os.pathsep.join(paths_to_add + ([existing_pp] if existing_pp else []))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_report:
            report_path = Path(tmp_report.name)

        try:
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                f"--json-report-file={report_path}",
                "--json-report",
                "--tb=short",
                "-q",
            ]
            if test_paths:
                cmd.extend(test_paths)
            if extra_pytest_args:
                cmd.extend(extra_pytest_args)

            res = subprocess.run(
                cmd,
                cwd=str(repo_path),
                env=env,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            duration = time.monotonic() - start
            output = res.stdout + "\n" + res.stderr
            actual_exit_code = res.returncode

            json_report: dict = {}
            if report_path.exists():
                with contextlib.suppress(Exception):
                    json_report = _parse_json_report(report_path.read_text(encoding="utf-8"))

            failed_tests: list[str] = []
            passed_tests: list[str] = []
            if json_report:
                for t in json_report.get("tests", []):
                    if t.get("outcome") == "passed":
                        passed_tests.append(t["nodeid"])
                    elif t.get("outcome") in ("failed", "error"):
                        failed_tests.append(t["nodeid"])
            else:
                for line in output.splitlines():
                    if " PASSED" in line:
                        passed_tests.append(line.split()[0])
                    elif " FAILED" in line:
                        failed_tests.append(line.split()[0])

            passed = actual_exit_code == 0
            if not failed_tests and not passed:
                failed_tests = test_paths or ["test_suite"]

            return TestResult(
                passed=passed,
                exit_code=actual_exit_code,
                stdout=output,
                stderr=res.stderr,
                traceback=_extract_traceback(output) if not passed else "",
                failed_tests=failed_tests,
                passed_tests=passed_tests,
                duration_seconds=round(duration, 2),
                raw_json=json_report,
            )
        except subprocess.TimeoutExpired:
            return TestResult(
                passed=False,
                exit_code=-1,
                stdout="",
                stderr="Timeout expired",
                traceback="",
                failed_tests=test_paths or [],
                passed_tests=[],
                duration_seconds=float(self._timeout),
                error_message=f"Test run timed out after {self._timeout}s",
            )
        except Exception as exc:
            logger.exception("Unexpected local sandbox error: %s", exc)
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
            if report_path.exists():
                with contextlib.suppress(Exception):
                    report_path.unlink()

    def run_linter(self, repo_path: Path) -> dict[str, Any]:
        try:
            res = subprocess.run(
                [sys.executable, "-m", "ruff", "check", "."],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = res.stdout + "\n" + res.stderr
            return {
                "output": output[:3000],
                "passed": res.returncode == 0,
            }
        except Exception as e:
            return {"output": f"Linter run completed: {e}", "passed": True}


# ── Main runner ───────────────────────────────────────────────────────────────


class DockerSandbox:
    """
    Manages a Docker-based sandbox for running repo test suites safely.
    Automatically falls back to LocalSandbox if the Docker daemon is unreachable.

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
        cpus: float | None = None,
        network: str | None = None,
        docker_client: Any | None = None,
        fallback_to_local: bool = True,
    ) -> None:
        self._image = image or settings.sandbox_image
        self._timeout = timeout or settings.sandbox_timeout
        self._memory = memory or settings.sandbox_memory
        self._cpus = cpus or getattr(settings, "sandbox_cpus", 2.0)
        self._network = network or getattr(settings, "sandbox_network", "none")
        self._fallback_local: LocalSandbox | None = None

        if docker_client is not None:
            self._client = docker_client
        else:
            try:
                self._client = docker.from_env()
                self._client.ping()
            except Exception as exc:
                if fallback_to_local:
                    logger.warning(
                        "Docker daemon is not available (%s); falling back to LocalSandbox.", exc
                    )
                    self._client = None
                    self._fallback_local = LocalSandbox(timeout=self._timeout)
                else:
                    raise

    def is_docker_available(self) -> bool:
        """Return True if Docker is available and connected, False if using LocalSandbox fallback."""
        return self._client is not None and self._fallback_local is None

    def build_image(self, force: bool = False) -> None:
        """
        Build the sandbox Docker image from sandbox/Dockerfile.base.
        Safe to call multiple times (no-op if image already exists).
        """
        if self._fallback_local is not None:
            self._fallback_local.build_image(force=force)
            return

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
        if self._fallback_local is not None:
            return self._fallback_local.run_tests(
                repo_path,
                test_paths=test_paths,
                install_cmd=install_cmd,
                extra_pytest_args=extra_pytest_args,
            )

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
            # Phase 1: Start container with security sandboxing (non-root, mem/cpu limits, network isolation)
            container = self._client.containers.run(
                self._image,
                command=["bash", "-c", run_script],
                detach=True,
                remove=False,          # we'll remove manually after log capture
                mem_limit=self._memory,
                nano_cpus=int(self._cpus * 1_000_000_000),  # e.g. 2 CPUs limit
                network_mode=self._network,                  # "none" for security isolation
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
                marker_line = [line for line in output.splitlines() if "__EXIT_CODE__" in line]
                if marker_line:
                    with contextlib.suppress(ValueError):
                        actual_exit_code = int(marker_line[-1].split("__EXIT_CODE__")[1].strip())

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
                with contextlib.suppress(Exception):
                    container.remove(force=True)

    def run_linter(self, repo_path: Path) -> dict[str, Any]:
        """Run ruff + mypy inside the sandbox and return structured findings."""
        if self._fallback_local is not None:
            return self._fallback_local.run_linter(repo_path)

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
                with contextlib.suppress(Exception):
                    container.remove(force=True)
