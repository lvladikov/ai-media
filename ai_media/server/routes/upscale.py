"""Upscale route."""

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks

from ..models import UpscaleRequest
from ..jobs import create_job
from ..tasks import upscale as upscale_tasks

router = APIRouter(tags=["Upscale"])


@router.post("/api/upscale")
async def upscale_media(request: UpscaleRequest, background_tasks: BackgroundTasks):
    """Start an upscale job."""
    job = create_job(
        "upscale",
        prompt=None,
        model=request.method,
        params={"factor": request.factor, "strength": request.strength}
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = Path(request.input_path).suffix
    filename = request.output_filename or f"upscaled_{timestamp}{ext}"
    output_path = f"output/{filename}"
    
    background_tasks.add_task(
        upscale_tasks.run_upscale,
        job["job_id"],
        request.input_path,
        output_path,
        request.factor,
        request.method,
        request.strength,
    )
    
    return {"job_id": job["job_id"], "status": "pending", "output_path": output_path}
