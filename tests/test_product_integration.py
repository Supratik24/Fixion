"""
tests/test_product_integration.py — Unit tests for Phase 5:
- GitHub issue URL parser and API fetcher (ingest/github_issue.py)
- Pull request creation and git push logic (agent/tools/git_ops.py)
- SQLite database persistence and querying (storage/db.py)
- Human-in-the-loop (HITL) approval / rejection lifecycle in loop and API
- FastAPI endpoints for GitHub issue fetching, jobs history, and HITL decisioning
"""

from __future__ import annotations

import io
import json
import queue
import subprocess
import tempfile
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agent.loop import AgentContext, _dispatch_tool
from agent.tools import git_ops as git_mod
from api import _make_approval_callback, app, approvals, jobs
from ingest.github_issue import fetch_github_issue, parse_issue_url
from storage import db as db_mod

# -----------------------------------------------------------------------------
# 1. GitHub Issue Ingestion Tests
# -----------------------------------------------------------------------------

def test_parse_issue_url_valid():
    parsed = parse_issue_url("https://github.com/psf/requests/issues/1234")
    assert parsed == ("psf", "requests", 1234)

    parsed = parse_issue_url("http://github.com/fastapi/fastapi/issues/42/")
    assert parsed == ("fastapi", "fastapi", 42)


def test_parse_issue_url_invalid():
    assert parse_issue_url("https://gitlab.com/owner/repo/issues/123") is None
    assert parse_issue_url("https://github.com/owner/repo") is None
    assert parse_issue_url("https://github.com/owner/repo/pull/123") is None
    assert parse_issue_url("") is None


def test_fetch_github_issue_success():
    mock_payload = {
        "title": "Fix division by zero",
        "body": "When b=0 divide() should raise ZeroDivisionError.",
        "number": 101,
    }
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(mock_payload).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        data = fetch_github_issue("https://github.com/test-org/test-repo/issues/101")
        assert data["title"] == "Fix division by zero"
        assert data["number"] == 101
        assert "ZeroDivisionError" in data["body"]
        assert data["repo_url"] == "https://github.com/test-org/test-repo"
        assert data["owner"] == "test-org"
        assert data["repo"] == "test-repo"


def test_fetch_github_issue_http_error():
    err = urllib.error.HTTPError(
        url="https://api.github.com/repos/o/r/issues/999",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=io.BytesIO(b'{"message": "Not Found"}'),
    )
    with (
        patch("urllib.request.urlopen", side_effect=err),
        pytest.raises(RuntimeError, match="GitHub API error"),
    ):
        fetch_github_issue("https://github.com/o/r/issues/999")


# -----------------------------------------------------------------------------
# 2. Pull Request Creation Tests
# -----------------------------------------------------------------------------

def test_create_pull_request_no_changes():
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        with patch("agent.tools.git_ops.git_diff", return_value={"diff": "", "has_changes": False}):
            res = git_mod.create_pull_request(
                repo_root=repo_root,
                title="Fix issue",
                root_cause="Bad logic",
                fix_explanation="Corrected logic",
            )
            assert not res["success"]
            assert "No working tree changes" in res["error"]


def test_create_pull_request_no_token():
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        with (
            patch("agent.tools.git_ops.git_diff", return_value={"diff": "some diff", "has_changes": True}),
            patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout="", stderr="")),
            patch("agent.tools.git_ops._run_git", return_value="https://github.com/test-org/test-repo.git"),
            patch.object(git_mod.settings, "github_token", ""),
        ):
            res = git_mod.create_pull_request(
                repo_root=repo_root,
                title="Fix issue",
                root_cause="Bad logic",
                fix_explanation="Corrected logic",
                branch_name="fixion/test-branch",
                github_token="",
            )
            assert res["success"]
            assert res["branch"] == "fixion/test-branch"
            assert "GITHUB_TOKEN not configured" in res["message"]


def test_create_pull_request_success():
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)

        mock_api_res = MagicMock()
        mock_api_res.read.return_value = json.dumps({
            "html_url": "https://github.com/test-org/sample-repo/pull/77",
            "number": 77,
        }).encode("utf-8")
        mock_api_res.__enter__.return_value = mock_api_res

        with (
            patch("agent.tools.git_ops.git_diff", return_value={"diff": "diff content", "has_changes": True}),
            patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout="", stderr="")),
            patch("agent.tools.git_ops._run_git", return_value="https://github.com/test-org/sample-repo.git"),
            patch("urllib.request.urlopen", return_value=mock_api_res),
        ):
            res = git_mod.create_pull_request(
                repo_root=repo_root,
                title="Fix buggy calculation",
                root_cause="Zero denominator ignored",
                fix_explanation="Added guard check",
                branch_name="fixion/fix-77",
                github_token="fake_gh_token",
            )
            assert res["success"]
            assert res["pr_url"] == "https://github.com/test-org/sample-repo/pull/77"
            assert res["branch"] == "fixion/fix-77"


# -----------------------------------------------------------------------------
# 3. SQLite Database Persistence Tests
# -----------------------------------------------------------------------------

def test_sqlite_db_lifecycle():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db_path = Path(tmp) / "fixion_test.db"
        with patch.object(db_mod, "DB_PATH", db_path):
            db_mod.init_db()

            # 1. Save Job
            db_mod.save_job(
                job_id="job-1234",
                repo="https://github.com/org/repo.git",
                issue="Test bug report",
                status="running",
            )

            job = db_mod.get_job("job-1234")
            assert job is not None
            assert job["id"] == "job-1234"
            assert job["status"] == "running"
            assert job["repo"] == "https://github.com/org/repo.git"
            assert job["issue"] == "Test bug report"

            # 2. Append Logs
            db_mod.append_job_log("job-1234", "Starting run...")
            db_mod.append_job_log("job-1234", "Executing step 1")

            updated_job = db_mod.get_job("job-1234")
            assert "Starting run..." in updated_job["logs"]
            assert "Executing step 1" in updated_job["logs"]

            # 3. Update Job
            metrics_dict = {"tokens_in": 120, "tokens_out": 45, "wall_time_seconds": 3.2}
            db_mod.update_job(
                job_id="job-1234",
                status="completed",
                patch="--- a/file.py\n+++ b/file.py\n",
                metrics=metrics_dict,
                pr_url="https://github.com/org/repo/pull/1",
                root_cause="Missing check",
                fix_explanation="Added validation",
            )

            job_done = db_mod.get_job("job-1234")
            assert job_done["status"] == "completed"
            assert job_done["pr_url"] == "https://github.com/org/repo/pull/1"
            assert job_done["root_cause"] == "Missing check"
            assert job_done["metrics"]["tokens_in"] == 120

            # 4. List Jobs
            all_jobs = db_mod.list_jobs(limit=10)
            assert len(all_jobs) == 1
            assert all_jobs[0]["id"] == "job-1234"


# -----------------------------------------------------------------------------
# 4. Human-in-the-Loop (HITL) Workflow Tests
# -----------------------------------------------------------------------------

def test_hitl_approval_callback_approved():
    log_q = queue.Queue()
    job_id = "test-job-approval"
    jobs[job_id] = {"status": "running"}

    callback = _make_approval_callback(job_id, log_q)

    # Simulate callback running in worker thread and user approving
    import threading
    import time

    result_container = []

    def run_callback():
        res = callback("--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n")
        result_container.append(res)

    worker = threading.Thread(target=run_callback)
    worker.start()

    # Wait for approval state to be registered
    for _ in range(50):
        if job_id in approvals and approvals[job_id]["status"] == "pending":
            break
        time.sleep(0.02)

    assert approvals[job_id]["status"] == "pending"
    assert jobs[job_id]["status"] == "awaiting_approval"

    # User approves
    approvals[job_id]["status"] = "approved"
    approvals[job_id]["event"].set()

    worker.join(timeout=3)
    assert not worker.is_alive()
    assert result_container == [True]
    assert jobs[job_id]["status"] == "running"


def test_hitl_approval_callback_rejected():
    log_q = queue.Queue()
    job_id = "test-job-rejection"
    jobs[job_id] = {"status": "running"}

    callback = _make_approval_callback(job_id, log_q)

    import threading
    import time

    result_container = []

    def run_callback():
        res = callback("--- a/bar.py\n+++ b/bar.py\n")
        result_container.append(res)

    worker = threading.Thread(target=run_callback)
    worker.start()

    for _ in range(50):
        if job_id in approvals and approvals[job_id]["status"] == "pending":
            break
        time.sleep(0.02)

    # User rejects
    approvals[job_id]["status"] = "rejected"
    approvals[job_id]["event"].set()

    worker.join(timeout=3)
    assert not worker.is_alive()
    assert result_container == [False]


def test_hitl_dispatch_tool_rejection_prevents_file_change():
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        src_file = repo_root / "math_app.py"
        src_file.write_text("def sub(a, b): return a + b\n", encoding="utf-8")

        diff = """--- a/math_app.py
+++ b/math_app.py
@@ -1 +1 @@
-def sub(a, b): return a + b
+def sub(a, b): return a - b
"""
        # Callback that rejects the patch
        def rejection_callback(_p):
            return False

        ctx = AgentContext(
            repo_root=repo_root,
            repo_url="https://github.com/org/repo.git",
            issue_text="sub should subtract",
            failing_tests=[],
            require_reproduction=False,
            reproduction_confirmed=True,
            approval_callback=rejection_callback,
        )

        res = _dispatch_tool("apply_patch", {"patch_text": diff}, ctx)
        assert not res.get("applied")
        assert "rejected by human reviewer" in res.get("error", "").lower()
        # File should remain unedited
        assert "return a + b" in src_file.read_text(encoding="utf-8")


# -----------------------------------------------------------------------------
# 5. FastAPI Endpoints Integration Tests
# -----------------------------------------------------------------------------

client = TestClient(app)


def test_api_github_issue_endpoint():
    mock_payload = {
        "title": "Bug in parser",
        "body": "Parsing failed on line 10",
        "number": 55,
    }
    mock_res = MagicMock()
    mock_res.read.return_value = json.dumps(mock_payload).encode("utf-8")
    mock_res.__enter__.return_value = mock_res

    with patch("urllib.request.urlopen", return_value=mock_res):
        resp = client.post("/api/github/issue", json={"url": "https://github.com/org/repo/issues/55"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"]
        assert data["data"]["title"] == "Bug in parser"
        assert data["data"]["number"] == 55


def test_api_github_issue_invalid_url():
    resp = client.post("/api/github/issue", json={"url": "not_a_valid_url"})
    assert resp.status_code == 200
    data = resp.json()
    assert not data["success"]
    assert "Invalid GitHub issue" in data["error"]


def test_api_jobs_list_endpoint():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db_path = Path(tmp) / "fixion_endpoint.db"
        with patch.object(db_mod, "DB_PATH", db_path):
            db_mod.init_db()
            db_mod.save_job("test-job-99", "https://github.com/o/r.git", "issue desc")

            resp = client.get("/api/jobs")
            assert resp.status_code == 200
            data = resp.json()
            assert "jobs" in data
            assert any(j["id"] == "test-job-99" for j in data["jobs"])


def test_api_approval_endpoints():
    import threading

    job_id = "test-job-endpoint-approval"
    event = threading.Event()
    approvals[job_id] = {
        "event": event,
        "status": "pending",
        "patch": "--- diff ---",
    }

    # 1. Check approval status
    resp = client.get(f"/api/job/{job_id}/approval_status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "pending"
    assert resp.json()["patch"] == "--- diff ---"

    # 2. Approve
    resp = client.post(f"/api/job/{job_id}/approve")
    assert resp.status_code == 200
    assert resp.json()["success"]
    assert approvals[job_id]["status"] == "approved"
    assert event.is_set()
