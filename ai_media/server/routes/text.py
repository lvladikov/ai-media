"""Article and code generation routes."""

import os
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..models import ArticleGenerateRequest, CodeGenerateRequest, TextExportRequest
from ..config import CONFIG
from ..jobs import create_job
from ..process_manager import spawn_job_process
from ..tasks import text as text_tasks

router = APIRouter(tags=["Text Generation"])


@router.post("/api/generate/article")
async def generate_article(request: ArticleGenerateRequest):
    """Start an article generation job."""
    job = create_job(
        "article",
        prompt=request.topic,
        model=request.model,
        params={
            "format": request.format,
            "length": request.length,
            "online": request.online,
            "bypass_warning": request.bypass_warning,
        }
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = request.output_filename or f"article_{timestamp}.{request.format}"
    output_path = os.path.join(CONFIG["paths"]["media_output"], filename)
    
    spawn_job_process(
        job["job_id"],
        text_tasks.run_article_generation,
        (
            job["job_id"],
            request.topic,
            output_path,
            request.model,
            request.format,
            request.length,
            request.online,
            request.research_iterations,
            request.max_images,
            True, # Always bypass warning in server mode
        ),
    )
    
    return {"job_id": job["job_id"], "status": "pending", "output_path": output_path}


@router.post("/api/generate/code")
async def generate_code(request: CodeGenerateRequest):
    """Start a code generation job."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use output_name if provided, otherwise generate timestamped folder
    output_name = request.output_name.strip() if request.output_name else f"code_{timestamp}"
    
    # Always use folder-based output - generate_code handles single vs multi-file
    output_path = os.path.join(CONFIG["paths"]["media_output"], output_name)
    
    job = create_job(
        "code",
        prompt=request.prompt,
        model=request.model,
        params={"output_name": output_name, "bypass_warning": request.bypass_warning}
    )
    
    spawn_job_process(
        job["job_id"],
        text_tasks.run_code_generation,
        (
            job["job_id"],
            request.prompt,
            output_path,
            request.model,
            True, # Always bypass warning in server mode
        ),
    )
    
    return {"job_id": job["job_id"], "status": "pending", "output_path": output_path}


@router.post("/api/text/export")
async def export_text(request: TextExportRequest):
    """Export text to a specific format and return as a file download."""
    from ai_media.utils.text_conversion import convert_text
    import io

    content_bytes = convert_text(request.content, request.format, request.filename)
    
    media_types = {
        "md": "text/markdown",
        "txt": "text/plain",
        "html": "text/html",
        "json": "application/json",
        "pdf": "application/pdf",
        "rtf": "application/rtf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    
    media_type = media_types.get(request.format.lower(), "application/octet-stream")
    
    return StreamingResponse(
        io.BytesIO(content_bytes),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={request.filename}"}
    )
