import contextlib
import logging
import queue
import threading
import uuid
from collections.abc import Callable
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.loop import AgentContext, AgentResult, run_agent
from agent.tools import git_ops as git_mod
from ingest import github_issue as gh_issue_mod
from storage import db as db_mod

# Initialize SQLite database schema
db_mod.init_db()

app = FastAPI(title="Fixion Web API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

# In-memory job tracking & approval synchronizers
jobs: dict[str, dict[str, Any]] = {}
approvals: dict[str, dict[str, Any]] = {}


class JobLogHandler(logging.Handler):
    """Custom logging handler to route logs to a specific job's queue and SQLite database."""
    def __init__(self, job_id: str, log_queue: queue.Queue):
        super().__init__()
        self.job_id = job_id
        self.log_queue = log_queue
        self.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S'))

    def emit(self, record):
        if threading.current_thread().name == f"job-{self.job_id}":
            log_entry = self.format(record)
            self.log_queue.put(log_entry)
            with contextlib.suppress(Exception):
                db_mod.append_job_log(self.job_id, log_entry)


class FixRequest(BaseModel):
    repo_url: str = ""
    issue_text: str = ""
    issue_url: str = ""
    test_path: str = ""
    create_pr: bool = False
    human_in_the_loop: bool = False
    no_symbol_graph: bool = False
    no_semantic_search: bool = False


class GitHubIssueRequest(BaseModel):
    url: str


def _make_approval_callback(job_id: str, log_queue: queue.Queue) -> Callable[[str], bool]:
    """Create a blocking human-in-the-loop callback that waits for user approval via the UI."""
    def callback(patch_text: str) -> bool:
        event = threading.Event()
        approvals[job_id] = {
            "event": event,
            "status": "pending",
            "patch": patch_text,
        }
        if job_id in jobs:
            jobs[job_id]["status"] = "awaiting_approval"
            jobs[job_id]["proposed_patch"] = patch_text
        db_mod.update_job(job_id, status="awaiting_approval", patch=patch_text)

        log_queue.put("⏸️ [HUMAN REVIEW REQUIRED] Proposed patch generated. Awaiting approval...")
        log_queue.put(f"PATCH_PROPOSED:\n{patch_text}")

        # Wait up to 10 minutes (600s) for user decision
        signaled = event.wait(timeout=600)
        if not signaled:
            log_queue.put("⚠️ Approval timed out after 10 minutes. Rejecting patch.")
            db_mod.update_job(job_id, status="failed")
            return False

        decision = approvals[job_id].get("status", "rejected")
        if decision == "approved":
            if job_id in jobs:
                jobs[job_id]["status"] = "running"
            db_mod.update_job(job_id, status="running")
            log_queue.put("✅ Patch APPROVED by human reviewer. Applying to repository...")
            return True
        else:
            if job_id in jobs:
                jobs[job_id]["status"] = "running"
            db_mod.update_job(job_id, status="running")
            log_queue.put("❌ Patch REJECTED by human reviewer. Formulating alternative fix...")
            return False

    return callback


def _setup_fix_run(
    repo_url: str,
    issue_text: str,
    test_path: str,
    log_queue: queue.Queue,
    no_symbol_graph: bool = False,
    no_semantic_search: bool = False,
    approval_callback: Callable[[str], bool] | None = None,
) -> AgentContext:
    from ingest.ast_parser import parse_repo
    from ingest.clone import clone_repo
    from ingest.embed_index import EmbedIndex
    from ingest.symbol_graph import build_symbol_graph
    from sandbox.docker_runner import DockerSandbox
    
    log_queue.put(f"🚀 Cloning repository {repo_url}...")
    clone_result = clone_repo(repo_url)
    log_queue.put(f"✅ Cloned to {clone_result.local_path} @ {clone_result.head_sha[:8]}")
    
    log_queue.put("🔍 Parsing repository AST...")
    chunks = parse_repo(clone_result.local_path)

    if not no_symbol_graph:
        sg = build_symbol_graph(chunks)
        log_queue.put("✅ Built symbol call graph")
    else:
        sg = None
        log_queue.put("⚠️ Symbol graph disabled for ablation")

    if not no_semantic_search:
        embed_idx = EmbedIndex.build(chunks, repo_url, clone_result.head_sha)
        log_queue.put(f"✅ Indexed {len(chunks)} code chunks")
    else:
        embed_idx = None
        log_queue.put("⚠️ Semantic search disabled for ablation")
    
    sandbox = DockerSandbox()
    log_queue.put("🐳 Ensuring Docker sandbox image exists...")
    sandbox.build_image()
    log_queue.put("✅ Sandbox ready")
    
    failing_tests = [test_path] if test_path else []
    
    return AgentContext(
        repo_root=clone_result.local_path,
        repo_url=repo_url,
        issue_text=issue_text,
        failing_tests=failing_tests,
        embed_index=embed_idx,
        symbol_graph=sg,
        sandbox=sandbox,
        use_symbol_graph=not no_symbol_graph,
        use_semantic_search=not no_semantic_search,
        approval_callback=approval_callback,
    )

def run_job_sync(job_id: str, req: FixRequest, log_queue: queue.Queue):
    """Background thread function that runs the Fixion agent."""
    threading.current_thread().name = f"job-{job_id}"
    
    handler = JobLogHandler(job_id, log_queue)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    
    try:
        # 1. Resolve issue URL if provided
        if req.issue_url and not req.issue_text:
            log_queue.put(f"🌐 Fetching GitHub issue details from {req.issue_url}...")
            issue_info = gh_issue_mod.fetch_github_issue(req.issue_url)
            req.issue_text = issue_info["issue_text"]
            if not req.repo_url:
                req.repo_url = issue_info["repo_url"]
            log_queue.put(f"✅ Fetched issue: {issue_info['title']}")

        # Persist job record in SQLite
        db_mod.save_job(
            job_id=job_id,
            repo=req.repo_url,
            issue=req.issue_text,
            status="running",
        )

        approval_cb = _make_approval_callback(job_id, log_queue) if req.human_in_the_loop else None

        ctx = _setup_fix_run(
            repo_url=req.repo_url,
            issue_text=req.issue_text,
            test_path=req.test_path,
            log_queue=log_queue,
            no_symbol_graph=req.no_symbol_graph,
            no_semantic_search=req.no_semantic_search,
            approval_callback=approval_cb,
        )
        
        log_queue.put("🤖 Starting Agent Loop...")
        result: AgentResult = run_agent(ctx)
        
        # Pull Request creation if opt-in enabled
        pr_url = ""
        if req.create_pr and result.success and result.patch:
            log_queue.put("🐙 Creating GitHub Pull Request...")
            pr_res = git_mod.create_pull_request(
                repo_root=ctx.repo_root,
                title=f"Fix: {req.issue_text.splitlines()[0][:60]}",
                root_cause=result.root_cause,
                fix_explanation=result.fix_explanation,
                repo_url=req.repo_url,
            )
            if pr_res.get("success") and pr_res.get("pr_url"):
                pr_url = pr_res["pr_url"]
                log_queue.put(f"🎉 Pull Request opened: {pr_url}")
            else:
                log_queue.put(f"⚠️ PR creation note: {pr_res.get('message') or pr_res.get('error')}")

        # Update in-memory job state
        jobs[job_id]["status"] = "completed" if result.success else "unresolved"
        jobs[job_id]["result"] = {
            "success": result.success,
            "patch": result.patch,
            "root_cause": result.root_cause,
            "fix_explanation": result.fix_explanation,
            "tests_passed": result.tests_passed,
            "tests_failed": result.tests_failed,
            "metrics": result.metrics,
            "pr_url": pr_url,
        }

        # Persist final results to SQLite
        db_mod.update_job(
            job_id,
            status="completed" if result.success else "unresolved",
            patch=result.patch,
            root_cause=result.root_cause,
            fix_explanation=result.fix_explanation,
            metrics=result.metrics,
            pr_url=pr_url,
        )
        log_queue.put(f"✅ Job Finished! Success: {result.success}")
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        db_mod.update_job(job_id, status="failed")
        log_queue.put(f"❌ Fatal Error: {str(e)}")
        logging.exception("Job failed")
    finally:
        root_logger.removeHandler(handler)
        log_queue.put("EOF")


@app.post("/api/fix")
async def start_fix(req: FixRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    log_queue = queue.Queue()
    
    jobs[job_id] = {
        "status": "running",
        "queue": log_queue,
        "result": None,
        "error": None,
        "proposed_patch": "",
    }
    
    thread = threading.Thread(
        target=run_job_sync,
        args=(job_id, req, log_queue),
        daemon=True,
    )
    thread.start()
    
    return {"job_id": job_id}


@app.get("/api/jobs")
async def get_jobs():
    """Retrieve all persisted jobs from SQLite."""
    job_list = db_mod.list_jobs(limit=100)
    return {"jobs": job_list}


@app.post("/api/github/issue")
async def get_github_issue(req: GitHubIssueRequest):
    """Fetch GitHub issue details from a URL."""
    try:
        data = gh_issue_mod.fetch_github_issue(req.url)
        return {"success": True, "data": data}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/job/{job_id}/approve")
async def approve_patch(job_id: str):
    """Approve a proposed patch in human-in-the-loop mode."""
    if job_id in approvals:
        approvals[job_id]["status"] = "approved"
        approvals[job_id]["event"].set()
        return {"success": True, "message": "Patch approved."}
    return {"error": "Job not in approval state"}


@app.post("/api/job/{job_id}/reject")
async def reject_patch(job_id: str):
    """Reject a proposed patch in human-in-the-loop mode."""
    if job_id in approvals:
        approvals[job_id]["status"] = "rejected"
        approvals[job_id]["event"].set()
        return {"success": True, "message": "Patch rejected."}
    return {"error": "Job not in approval state"}


@app.get("/api/job/{job_id}/approval_status")
async def get_approval_status(job_id: str):
    """Check approval status of a job."""
    if job_id in approvals:
        return {
            "status": approvals[job_id]["status"],
            "patch": approvals[job_id].get("patch", ""),
        }
    return {"status": "none", "patch": ""}


@app.post("/api/cancel/{job_id}")
async def cancel_job(job_id: str):
    if job_id in jobs:
        jobs[job_id]["status"] = "cancelled"
        db_mod.update_job(job_id, status="cancelled")
        jobs[job_id]["queue"].put("⚠️ Job cancelled by user.")
        jobs[job_id]["queue"].put("EOF")
        return {"success": True}
    return {"error": "Job not found"}


@app.get("/api/logs/{job_id}")
async def stream_logs(job_id: str, request: Request):
    import asyncio
    if job_id not in jobs:
        return {"error": "Job not found"}
        
    log_queue = jobs[job_id]["queue"]

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
                
            if jobs.get(job_id, {}).get("status") == "cancelled":
                yield "data: ⚠️ Job cancelled by user.\n\n"
                yield "data: [DONE]\n\n"
                break
                
            try:
                msg = log_queue.get_nowait()
                if msg == "EOF":
                    yield "data: [DONE]\n\n"
                    break
                safe_msg = msg.replace("\n", "\\n")
                yield f"data: {safe_msg}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"
                await asyncio.sleep(0.1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    if job_id in jobs:
        result_data = jobs[job_id]["result"]
        metrics_data = result_data.get("metrics") if isinstance(result_data, dict) else None
        return {
            "status": jobs[job_id]["status"],
            "result": result_data,
            "metrics": metrics_data,
            "error": jobs[job_id]["error"],
        }
    # Fallback to SQLite DB
    db_job = db_mod.get_job(job_id)
    if db_job:
        return {
            "status": db_job["status"],
            "result": {
                "success": db_job["status"] == "completed",
                "patch": db_job["patch"],
                "root_cause": db_job["root_cause"],
                "fix_explanation": db_job["fix_explanation"],
                "pr_url": db_job.get("pr_url", ""),
            },
            "metrics": db_job.get("metrics"),
            "error": None,
        }
    return {"error": "Job not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
