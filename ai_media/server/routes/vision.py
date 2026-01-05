"""Vision and description routes."""

from datetime import datetime
import os

from fastapi import APIRouter

from ..models import VisionRequest
from ..config import CONFIG
from ..jobs import create_job
from ..process_manager import spawn_job_process
from ..tasks import description as vision_tasks

router = APIRouter(tags=["Vision"])

@router.post("/api/vision/describe")
async def generate_description(request: VisionRequest):
    """Start a vision/description job."""
    job = create_job(
        "vision",
        model=request.model,
        params={"input": request.input_path, "force": request.force},
        job_id=request.job_id
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = request.output_filename or f"description_{timestamp}.txt"
    output_path = os.path.join(CONFIG["paths"]["media_output"], filename)
    
    spawn_job_process(
        job["job_id"],
        vision_tasks.run_description,
        (
            job["job_id"],
            request.input_path,
            request.model,
            output_path,
            request.force,
            True, # Always bypass warning in server mode
        ),
    )
    
    return {"job_id": job["job_id"], "status": "pending", "output_path": output_path}
