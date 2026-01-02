"""Job management utilities."""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException

from .state import jobs, job_manager, MAIN_LOOP

router = APIRouter(tags=["Jobs"])


def create_job(job_type: str, prompt: str = None, model: str = None, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a new job entry with optional metadata."""
    job_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    job = {
        "job_id": job_id,
        "type": job_type,
        "status": "pending",
        "progress": 0,
        "phase": "queued",
        "message": "Job queued",
        "result_path": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
        # Optional metadata for display
        "prompt": prompt,
        "model": model,
        "params": params or {},
    }
    jobs[job_id] = job
    
    # Broadcast new job
    coro = job_manager.broadcast({
        "type": "job_update",
        "job": job
    })
    
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        from .state import MAIN_LOOP
        if MAIN_LOOP and not MAIN_LOOP.is_closed():
            asyncio.run_coroutine_threadsafe(coro, MAIN_LOOP)
        else:
            coro.close()
    
    return job


def update_job(job_id: str, event_loop: asyncio.AbstractEventLoop = None, **kwargs):
    """Update job status and broadcast via WebSocket."""
    if job_id in jobs:
        jobs[job_id].update(kwargs)
        jobs[job_id]["updated_at"] = datetime.now().isoformat()
        
        coro = job_manager.broadcast({
            "type": "job_update",
            "job": jobs[job_id]
        })
        
        # Use provided event loop or try to get/use main loop
        loop = event_loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                from .state import MAIN_LOOP
                loop = MAIN_LOOP
        
        if loop and not loop.is_closed():
            asyncio.run_coroutine_threadsafe(coro, loop)
        else:
            coro.close()


def is_job_cancelled(job_id: str) -> bool:
    """Check if a job has been cancelled."""
    if job_id in jobs:
        return jobs[job_id].get("status") == "cancelled"
    return False


# --- Routes ---

@router.get("/api/jobs")
async def list_jobs():
    """List all jobs."""
    return list(jobs.values())


@router.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@router.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Use update_job to broadcast cancellation via WebSocket
    update_job(job_id, status="cancelled", message="Job cancelled by user")
    print(f"🛑 Job {job_id[:8]}... cancelled")
    
    return {"message": "Job cancelled"}
