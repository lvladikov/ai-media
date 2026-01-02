"""Transform route."""

from datetime import datetime

from fastapi import APIRouter, BackgroundTasks

from ..models import TransformRequest
from ..jobs import create_job
from ..tasks import transform as transform_tasks

router = APIRouter(tags=["Transform"])


@router.post("/api/transform")
async def transform_image(request: TransformRequest, background_tasks: BackgroundTasks):
    """Start an image transformation job."""
    job = create_job(
        "transform",
        prompt=request.instruction,
        model=request.model,
        params={"input_path": request.input_path}
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = request.output_filename or f"transform_{timestamp}.png"
    output_path = f"output/{filename}"
    
    background_tasks.add_task(
        transform_tasks.run_transform,
        job["job_id"],
        request.input_path,
        request.instruction,
        output_path,
        request.model,
    )
    
    return {"job_id": job["job_id"], "status": "pending", "output_path": output_path}
