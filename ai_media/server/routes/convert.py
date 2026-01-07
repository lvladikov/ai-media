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
        params={
            "input_path": request.input_path, 
            "target_format": request.target_format,
            "translate": request.translate,
            "target_language": request.target_language
        }
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if request.output_filename:
        filename = request.output_filename
    else:
        # Determine components based on request type
        prefix = "translated" if request.translate else "converted"
        
        # Check for direct input
        content_label = "_direct_input" if request.is_direct_text else ""
        
        # Append language code if translation is enabled
        lang_suffix = f".{request.target_language}" if request.translate and request.target_language else ""
        
        filename = f"{prefix}{content_label}_{timestamp}{lang_suffix}.{request.target_format}"
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
            request.translate,
            request.target_language,
            request.translation_model,
            request.render_method,
            request.is_direct_text,
        ),
    )
    
    return {"job_id": job["job_id"], "status": "pending", "output_path": output_path}

