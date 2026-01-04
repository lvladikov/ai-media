"""Convert route."""

from datetime import datetime

from fastapi import APIRouter

from ..models import ConvertRequest
from ..jobs import create_job
from ..process_manager import spawn_job_process
from ..tasks import convert as convert_tasks

from ..config import CONFIG
import os

router = APIRouter(tags=["Convert"])


@router.post("/api/convert")
async def convert_media(request: ConvertRequest):
    """Start a media conversion job."""
    job = create_job(
        "convert",
        prompt=None,
        model=None,
        params={"input_path": request.input_path, "target_format": request.target_format}
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = request.output_filename or f"converted_{timestamp}.{request.target_format}"
    output_path = os.path.join(CONFIG["paths"]["media_output"], filename)
    
    spawn_job_process(
        job["job_id"],
        convert_tasks.run_convert,
        (
            job["job_id"],
            request.input_path,
            request.target_format,
            output_path,
            request.ocr_enabled,
            request.ocr_model,
        ),
    )
    
    return {"job_id": job["job_id"], "status": "pending", "output_path": output_path}

