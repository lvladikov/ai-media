"""Transform route."""

import os
from datetime import datetime

from fastapi import APIRouter

from ..models import TransformRequest
from ..config import CONFIG
from ..jobs import create_job
from ..process_manager import spawn_job_process
from ..tasks import transform as transform_tasks

router = APIRouter(tags=["Transform"])


@router.post("/api/transform")
async def transform_image(request: TransformRequest):
    """Start an image transformation job."""
    job = create_job(
        "transform",
        prompt=request.instruction,
        model=request.model,
        params={"input_path": request.input_path, "bypass_warning": request.bypass_warning}
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = request.output_filename or f"transform_{timestamp}.png"
    output_path = os.path.join(CONFIG["paths"]["media_output"], filename)
    
    spawn_job_process(
        job["job_id"],
        transform_tasks.run_transform,
        (
            job["job_id"],
            request.input_path,
            request.instruction,
            output_path,
            request.model,
            True, # Always bypass warning in server mode
        ),
    )
    
    return {"job_id": job["job_id"], "status": "pending", "output_path": output_path}

