"""File upload and download routes."""

import io
import os
import zipfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse, StreamingResponse

from ..config import CONFIG

router = APIRouter(tags=["Files"])


@router.get("/api/files/zip")
async def zip_folder(path: str = Query(..., description="Path to folder to zip")):
    """Create a zip archive from a folder and return as download."""
    # Handle both absolute and relative paths
    if path.startswith("/"):
        full_path = path
    else:
        full_path = os.path.join(CONFIG["paths"]["media_output"], path)
    
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Path not found")
    
    if not os.path.isdir(full_path):
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    # Create zip in memory
    zip_buffer = io.BytesIO()
    folder_name = os.path.basename(full_path.rstrip("/"))
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(full_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Use relative path within zip
                arcname = os.path.relpath(file_path, full_path)
                zf.write(file_path, arcname)
    
    zip_buffer.seek(0)
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={folder_name}.zip"}
    )


@router.get("/api/files/{file_path:path}")
async def get_file(file_path: str, download: bool = Query(False, description="Force download instead of inline preview")):
    """Serve generated files for preview/download."""
    # Handle both absolute and relative paths
    if file_path.startswith("/"):
        full_path = file_path
    else:
        full_path = os.path.join(CONFIG["paths"]["media_output"], file_path)
    
    
    if not os.path.exists(full_path):
        # Fallback: if path is absolute but leading slash was stripped (common in some API calls)
        # try adding it back if the original didn't start with /
        if not file_path.startswith("/") and os.path.exists("/" + file_path):
            full_path = "/" + file_path
        else:
            raise HTTPException(status_code=404, detail="File not found")
    
    filename = os.path.basename(full_path)
    
    if download:
        return FileResponse(
            full_path,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
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
