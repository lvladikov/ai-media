"""Cleanup routes for managing cached models and output folders."""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
from threading import Thread

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import CONFIG
from ..jobs import create_job, update_job

router = APIRouter(tags=["Cleanup"])


class CleanupJobRequest(BaseModel):
    """Request to start a cleanup job."""
    action: str  # clear-data-output, clear-media-output, clear-all-outputs, clear-hub-model
    folder_name: str = None  # Required for clear-hub-model


def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    elif size_bytes < 1024 ** 4:
        return f"{size_bytes / (1024 ** 3):.1f} GB"
    else:
        return f"{size_bytes / (1024 ** 4):.1f} TB"


def get_folder_size(path: str) -> int:
    """Recursively calculate the total size of a folder."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
    except (OSError, PermissionError):
        pass
    return total_size

def get_folder_stats(path: str) -> Dict[str, Any]:
    """Get file count and total size for a folder."""
    if not os.path.exists(path):
        return {"file_count": 0, "size": 0, "size_formatted": "0 B", "path": path}
    
    file_count = 0
    total_size = 0
    try:
        for item in os.listdir(path):
            if item.startswith("."):
                continue
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path):
                file_count += 1
                total_size += os.path.getsize(item_path)
            elif os.path.isdir(item_path):
                file_count += 1
                total_size += get_folder_size(item_path)
    except OSError:
        pass
    
    return {
        "file_count": file_count,
        "size": total_size,
        "size_formatted": format_size(total_size),
        "path": path
    }


@router.get("/api/cleanup/output-stats")
async def get_output_stats() -> Dict[str, Any]:
    """Get stats for testing and media output folders."""
    script_dir = Path(__file__).parent.parent.parent
    
    testing_path = str(script_dir / "testing" / "data" / "outputs")
    media_path = CONFIG["paths"].get("media_output", "output")
    
    return {
        "testing_output": get_folder_stats(testing_path),
        "media_output": get_folder_stats(media_path)
    }


@router.get("/api/cleanup/hub-models")
async def get_hub_models() -> Dict[str, Any]:
    """List all cached hub models with their sizes."""
    hf_home = CONFIG["paths"].get("hf_home")
    if not hf_home:
        raise HTTPException(status_code=400, detail="hf_home not configured")
    
    hub_path = os.path.join(hf_home, "hub")
    if not os.path.exists(hub_path):
        return []
    
    models = []
    try:
        for item in os.listdir(hub_path):
            # Skip hidden folders like .locks
            if item.startswith("."):
                continue
            item_path = os.path.join(hub_path, item)
            if os.path.isdir(item_path):
                size = get_folder_size(item_path)
                models.append({
                    "name": item,
                    "size": size,
                    "size_formatted": format_size(size),
                    "path": item_path
                })
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Error scanning hub folder: {e}")
    
    # Sort by size (largest first)
    models.sort(key=lambda x: x["size"], reverse=True)
    
    # Calculate total
    total_size = sum(m["size"] for m in models)
    
    return {
        "models": models,
        "total_size": total_size,
        "total_size_formatted": format_size(total_size),
        "hub_path": hub_path
    }


@router.post("/api/cleanup/job")
async def start_cleanup_job(request: CleanupJobRequest) -> Dict[str, Any]:
    """Start a cleanup job and return job_id for tracking."""
    
    valid_actions = ["clear-data-output", "clear-media-output", "clear-all-outputs", "clear-hub-model"]
    if request.action not in valid_actions:
        raise HTTPException(status_code=400, detail=f"Invalid action. Must be one of: {valid_actions}")
    
    if request.action == "clear-hub-model" and not request.folder_name:
        raise HTTPException(status_code=400, detail="folder_name required for clear-hub-model action")
    
    # Create job
    job = create_job(
        job_type="cleanup",
        prompt=request.action,
        params={"action": request.action, "folder_name": request.folder_name}
    )
    job_id = job["job_id"]
    
    # Run cleanup in background thread
    def run_cleanup():
        try:
            update_job(job_id, status="generating", phase="cleanup", message="Starting cleanup...")
            
            script_dir = Path(__file__).parent.parent.parent
            
            if request.action == "clear-data-output":
                path = script_dir / "testing" / "data" / "outputs"
                deleted = _clear_directory(str(path), job_id)
                update_job(job_id, status="complete", progress=100, 
                          message=f"Cleared {len(deleted)} items from testing/data/outputs")
                
            elif request.action == "clear-media-output":
                path = CONFIG["paths"].get("media_output", "output")
                deleted = _clear_directory(str(path), job_id)
                update_job(job_id, status="complete", progress=100,
                          message=f"Cleared {len(deleted)} items from media_output")
                
            elif request.action == "clear-all-outputs":
                # Clear both
                path1 = script_dir / "testing" / "data" / "outputs"
                deleted1 = _clear_directory(str(path1), job_id)
                update_job(job_id, progress=50, message=f"Cleared testing/data/outputs ({len(deleted1)} items)")
                
                path2 = CONFIG["paths"].get("media_output", "output")
                deleted2 = _clear_directory(str(path2), job_id)
                update_job(job_id, status="complete", progress=100,
                          message=f"Cleared both folders ({len(deleted1) + len(deleted2)} items total)")
                
            elif request.action == "clear-hub-model":
                hf_home = CONFIG["paths"].get("hf_home")
                if not hf_home:
                    update_job(job_id, status="failed", error="hf_home not configured")
                    return
                
                folder_path = os.path.join(hf_home, "hub", request.folder_name)
                if not os.path.exists(folder_path):
                    update_job(job_id, status="failed", error=f"Folder not found: {request.folder_name}")
                    return
                
                size = get_folder_size(folder_path)
                update_job(job_id, log_line=f"🗑️ Deleting {request.folder_name} ({format_size(size)})...")
                
                try:
                    shutil.rmtree(folder_path)
                    update_job(job_id, status="complete", progress=100,
                              message=f"Successfully deleted {request.folder_name} ({format_size(size)})")
                except Exception as e:
                    update_job(job_id, status="failed", error=str(e))
                    
        except Exception as e:
            update_job(job_id, status="failed", error=str(e))
    
    thread = Thread(target=run_cleanup, daemon=True)
    thread.start()
    
    return {"job_id": job_id, "status": "started"}


def _clear_directory(path: str, job_id: str) -> List[str]:
    """Clear directory contents, logging progress to job."""
    if not os.path.exists(path):
        update_job(job_id, log_line=f"📁 Directory does not exist: {path}")
        return []
    
    deleted_items = []
    try:
        items = os.listdir(path)
    except OSError:
        return []
    
    for item in items:
        # Skip hidden files
        if item.startswith("."):
            continue
        
        item_path = os.path.join(path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
                deleted_items.append(item)
                update_job(job_id, log_line=f"🗑️ Deleted: {item}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                deleted_items.append(f"{item}/")
                update_job(job_id, log_line=f"🗑️ Deleted: {item}/")
        except Exception as e:
            update_job(job_id, log_line=f"❌ Failed to delete {item}: {e}")
    
    return deleted_items
