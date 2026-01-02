"""Convert route."""

from datetime import datetime

from fastapi import APIRouter, BackgroundTasks

from ..models import ConvertRequest
from ..jobs import create_job
from ..tasks import convert as convert_tasks

router = APIRouter(tags=["Convert"])


@router.post("/api/convert")
async def convert_media(request: ConvertRequest, background_tasks: BackgroundTasks):
    """Start a media conversion job."""
    job = create_job(
        "convert",
        prompt=None,
        model=None,
        params={"input_path": request.input_path, "target_format": request.target_format}
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = request.output_filename or f"converted_{timestamp}.{request.target_format}"
    output_path = f"output/{filename}"
    
    background_tasks.add_task(
        convert_tasks.run_convert,
        job["job_id"],
        request.input_path,
        request.target_format,
        output_path,
    )
    
    return {"job_id": job["job_id"], "status": "pending", "output_path": output_path}
