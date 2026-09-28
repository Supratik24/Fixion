import logging
import queue
import threading
import uuid
from typing import Dict, Any, Generator

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.loop import AgentContext, run_agent, AgentResult

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

# In-memory job tracking
jobs: Dict[str, Dict[str, Any]] = {}


class JobLogHandler(logging.Handler):
    """Custom logging handler to route logs to a specific job's queue."""
    def __init__(self, job_id: str, log_queue: queue.Queue):
        super().__init__()
        self.job_id = job_id
        self.log_queue = log_queue
        self.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S'))

    def emit(self, record):
        # We check thread ID to ensure we only capture logs from the background thread running this job
        if threading.current_thread().name == f"job-{self.job_id}":
            log_entry = self.format(record)
            self.log_queue.put(log_entry)


class FixRequest(BaseModel):
    repo_url: str
    issue_text: str
    test_path: str = ""

def _setup_fix_run(repo_url: str, issue_text: str, test_path: str, log_queue: queue.Queue) -> AgentContext:
    from config import settings
    from ingest.ast_parser import parse_repo
    from ingest.clone import clone_repo
    from ingest.embed_index import EmbedIndex
    from ingest.symbol_graph import build_symbol_graph
    from sandbox.docker_runner import DockerSandbox
    
    log_queue.put(f"🚀 Cloning repository {repo_url}...")
    clone_result = clone_repo(repo_url)
    log_queue.put(f"✅ Cloned to {clone_result.local_path} @ {clone_result.head_sha[:8]}")
    
    log_queue.put(f"🔍 Parsing and indexing...")
    chunks = parse_repo(clone_result.local_path)
    sg = build_symbol_graph(chunks)
    embed_idx = EmbedIndex.build(chunks, repo_url, clone_result.head_sha)
    log_queue.put(f"✅ Indexed {len(chunks)} code chunks")
    
    sandbox = DockerSandbox()
    log_queue.put(f"🐳 Ensuring Docker sandbox image exists...")
    sandbox.build_image()
    log_queue.put(f"✅ Sandbox ready")
    
    failing_tests = [test_path] if test_path else []
    
    return AgentContext(
        repo_root=clone_result.local_path,
        repo_url=repo_url,
        issue_text=issue_text,
        failing_tests=failing_tests,
        embed_index=embed_idx,
        symbol_graph=sg,
        sandbox=sandbox,
    )

def run_job_sync(job_id: str, req: FixRequest, log_queue: queue.Queue):
    """Background thread function that runs the Fixion agent."""
    threading.current_thread().name = f"job-{job_id}"
    
    # Attach custom handler to the root logger for this thread
    handler = JobLogHandler(job_id, log_queue)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    
    try:
        ctx = _setup_fix_run(
            repo_url=req.repo_url,
            issue_text=req.issue_text,
            test_path=req.test_path,
            log_queue=log_queue
        )
        
        log_queue.put(f"🤖 Starting Agent Loop...")
        result: AgentResult = run_agent(ctx)
        
        # Save result
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = {
            "success": result.success,
            "patch": result.patch,
            "root_cause": result.root_cause,
            "fix_explanation": result.fix_explanation,
            "tests_passed": result.tests_passed,
            "tests_failed": result.tests_failed,
        }
        log_queue.put(f"✅ Job Finished! Success: {result.success}")
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
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
        "error": None
    }
    
    # Spawn real OS thread to not block FastAPI async event loop
    thread = threading.Thread(
        target=run_job_sync,
        args=(job_id, req, log_queue),
        daemon=True
    )
    thread.start()
    
    return {"job_id": job_id}


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
                
            try:
                # Non-blocking get in async loop
                msg = log_queue.get_nowait()
                if msg == "EOF":
                    yield f"data: [DONE]\n\n"
                    break
                safe_msg = msg.replace("\n", "\\n")
                yield f"data: {safe_msg}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"
                await asyncio.sleep(0.1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    if job_id not in jobs:
        return {"error": "Job not found"}
    return {
        "status": jobs[job_id]["status"],
        "result": jobs[job_id]["result"],
        "error": jobs[job_id]["error"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
