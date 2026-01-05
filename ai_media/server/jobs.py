"""Job management utilities."""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException

from .state import jobs, job_manager, MAIN_LOOP

router = APIRouter(tags=["Jobs"])


def create_job(job_type: str, prompt: str = None, model: str = None, params: Dict[str, Any] = None, job_id: str = None) -> Dict[str, Any]:
    """Create a new job entry with optional metadata."""
    job_id = job_id or str(uuid.uuid4())
    now = datetime.now().isoformat()
    job = {
        "job_id": job_id,
        "type": job_type,
        "status": "pending",
        "progress": 0,
        "phase": "queued",
        "message": "Job queued",
        "logs": [],  # Accumulates log messages for display
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
        # Prevent reviving a cancelled job unless we're explicitly setting it to cancelled/failed
        current_status = jobs[job_id].get("status")
        new_status = kwargs.get("status")
        
        if current_status == "cancelled" and new_status != "cancelled" and new_status != "failed":
            # Ignore updates for cancelled jobs to stop background tasks from overwriting state
            return

        # Append message to logs if provided
        # Append message or log_line to logs
        log_entry = kwargs.get("log_line") or kwargs.get("message")
        
        # Only add message to logs if it's not just a status update
        # If log_line is present, it's explicitly a log
        if log_entry:
            if "logs" not in jobs[job_id]:
                jobs[job_id]["logs"] = []
            
            # Avoid duplicate last line if possible
            if not jobs[job_id]["logs"] or jobs[job_id]["logs"][-1] != log_entry:
                jobs[job_id]["logs"].append(log_entry)
            
            # Keep last 100 lines
            if len(jobs[job_id]["logs"]) > 100:
                jobs[job_id]["logs"] = jobs[job_id]["logs"][-100:]
        
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
    
    job = jobs[job_id]
    
    # Use update_job to broadcast cancellation via WebSocket
    update_job(job_id, status="cancelled", message="Job cancelled by user")
    print(f"🛑 Job {job_id[:8]}... cancelled")
    
    # Terminate the child process if running (multiprocessing approach)
    from .process_manager import terminate_job_process
    terminated = terminate_job_process(job_id)
    
    if terminated:
        print(f"✅ Process terminated for job {job_id[:8]}")
    else:
        # Fallback: try to unload model from cache (old threading approach)
        from .cache import model_cache
        
        # Map job types to cache categories
        type_map = {
            "article": "text",
            "code": "text", 
            "chat": "text",
            "image": "image",
            "audio": "audio",
            "video": "video",
            "transform": "transform",
            "upscale": "upscale",
            "vision": "vision"
        }
        
        category = type_map.get(job["type"])
        if category:
            generator = model_cache.get(category, job.get("model"))
            if generator and hasattr(generator, "stop"):
                generator.stop()
                
            print(f"🧹 Unloading {category} model due to cancellation...")
            model_cache.unload(category)
    
    return {"message": "Job cancelled"}
