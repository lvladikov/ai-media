"""File upload and download routes."""

import os
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

from ..config import CONFIG

router = APIRouter(tags=["Files"])


@router.get("/api/files/{file_path:path}")
async def get_file(file_path: str):
    """Serve generated files for preview/download."""
    # Handle both absolute and relative paths
    if file_path.startswith("/"):
        full_path = file_path
    else:
        full_path = os.path.join(CONFIG["paths"]["media_output"], file_path)
    
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(full_path)


@router.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file for use in generation (e.g., img2vid, transform)."""
    upload_dir = os.path.join(CONFIG["paths"]["media_output"], "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = Path(file.filename).suffix if file.filename else ""
    filename = f"upload_{timestamp}{ext}"
    file_path = os.path.join(upload_dir, filename)
    
    # Save file
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    return {
        "filename": filename,
        "path": file_path,
        "size": len(content),
    }
