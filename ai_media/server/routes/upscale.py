"""Upscale route."""

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter

from ..models import UpscaleRequest
from ..jobs import create_job
from ..process_manager import spawn_job_process
from ..tasks import upscale as upscale_tasks

from ..config import CONFIG
import os

router = APIRouter(tags=["Upscale"])


@router.post("/api/upscale")
async def upscale_media(request: UpscaleRequest):
    """Start an upscale job."""
    job = create_job(
        "upscale",
        prompt=None,
        model=request.method,
        params={"factor": request.factor, "strength": request.strength}
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = Path(request.input_path).suffix
    prefix = "downscaled" if float(request.factor) < 1.0 else "upscaled"
    filename = request.output_filename or f"{prefix}_{timestamp}{ext}"
    output_path = os.path.join(CONFIG["paths"]["media_output"], filename)
    
    spawn_job_process(
        job["job_id"],
        upscale_tasks.run_upscale,
        (
            job["job_id"],
            request.input_path,
            output_path,
            request.factor,
            request.method,
            request.strength,
        ),
    )
    
    return {"job_id": job["job_id"], "status": "pending", "output_path": output_path}


@router.post("/api/upscale/validate")
async def validate_upscale(request: UpscaleRequest):
    """Validate upscale request and check resources."""
    if not os.path.exists(request.input_path):
        return {"status": "error", "message": "Input file not found"}

    try:
        import psutil
        from PIL import Image
        import math
        
        # Get image dimensions
        # For video, we'd need to probe, but for now we'll support image validation mostly
        # or treat video as single frame calculation (which is accurate enough for RAM per frame)
        suffix = Path(request.input_path).suffix.lower()
        is_video = suffix in ['.mp4', '.mov', '.avi', '.mkv', '.webm']
        
        width, height = 0, 0
        
        if is_video:
            # Simple probe for video
            import subprocess
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0:s=x",
                request.input_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                width, height = map(int, result.stdout.strip().split('x'))
        else:
            with Image.open(request.input_path) as img:
                width, height = img.size

        if width == 0 or height == 0:
            return {"status": "ok", "message": "Could not determine dimensions"}

        factor = float(request.factor)
        target_w = int(width * factor)
        target_h = int(height * factor)
        megapixels = (target_w * target_h) / 1_000_000
        
        # Estimate RAM (logic matches CLI)
        # Assuming CPU for web server for safety margin, or simple heuristic
        # CLI uses: (megapixels * 0.8) if dev == "cpu" else (megapixels * 0.4)
        # We'll use a conservative estimate
        estimated_ram_gb = megapixels * 0.8
        
        vm = psutil.virtual_memory()
        available_gb = vm.available / (1024**3)
        
        # Thresholds
        is_huge = megapixels > 25
        
        # Only apply strict RAM check and resolution warnings for 'ai' method (Latent Upscale)
        # Real-ESRGAN (fast) is much more efficient and handles tiling better
        if request.method == 'ai':
             is_tight = estimated_ram_gb > (available_gb * 0.85)
             
             if is_tight:
                warning = "This upscale will consume nearly all available RAM (System freeze risk)."
                warning_type = "critical"
             elif is_huge:
                warning = "This resolution is extremely high (Billboard size)."
                warning_type = "warning"
        else:
             # For Fast/Simple, we generally trust the user and system swap
             # No warnings for purely resolution based checks
             pass

            
        return {
            "status": "warning" if warning else "ok",
            "warning": warning,
            "warning_type": warning_type,
            "details": {
                "input_resolution": f"{width}x{height}",
                "target_resolution": f"{target_w}x{target_h}",
                "megapixels": round(megapixels, 1),
                "estimated_ram_gb": round(estimated_ram_gb, 1),
                "available_ram_gb": round(available_gb, 1)
            }
        }
        
    except Exception as e:
        print(f"Validation error: {e}")
        return {"status": "ok"} # Fail open if we can't check

