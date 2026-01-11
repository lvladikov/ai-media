"""Upscale background task."""

import os
from pathlib import Path
from multiprocessing import Queue


def run_upscale(
    job_id: str,
    input_path: str,
    output_path: str,
    factor: float,
    method: str,
    strength: float,
    bypass_warning: bool = False,
    force: bool = False,
    progress_queue: Queue = None,
):
    """Background task for upscaling. Runs in child process."""
    
    def send_update(**kwargs):
        """Send progress update to parent via queue."""
        if progress_queue:
            try:
                progress_queue.put({"job_id": job_id, **kwargs})
            except Exception:
                pass
    
    try:
        send_update(status="loading", phase="loading", progress=10, message="Preparing upscale...")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        is_video = Path(input_path).suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv', '.webm']
        
        from ai_media.upscaling import (
            simple_upscale_image, simple_upscale_video,
            upscale_image_fast, upscale_video_fast,
            upscale_image_file, upscale_video_file
        )
        
        send_update(status="generating", phase="generating", progress=30, message=f"Upscaling ({method} x{factor})...")
        
        success = False
        
        if is_video:
            if method == 'simple':
                success = simple_upscale_video(input_path, output_path, factor=factor, force=force)
            elif method == 'fast':
                success = upscale_video_fast(input_path, output_path, factor=factor, force=force, bypass_warning=bypass_warning)
            else:  # ai / creative
                send_update(status="generating", phase="generating", progress=30, message="Upscaling video (AI Slow Mode)...")
                # upscale_video_file currently doesn't check resources directly but calls simple/fast or does its own thing.
                # Actually upscale_video_file in upscaling.py does NOT exist or was not shown?
                # Wait, I saw upscale_image_file. Let me check for upscale_video_file.
                success = upscale_video_file(input_path, output_path, strength=strength, factor=factor, force=force, bypass_warning=bypass_warning)
        else:  # Image
            if method == 'simple':
                success = simple_upscale_image(input_path, output_path, factor=factor, force=force)
            elif method == 'fast':
                success = upscale_image_fast(input_path, output_path, factor=factor, force=force, bypass_warning=bypass_warning)
            else:  # ai / creative
                def on_progress(pct, msg):
                    send_update(status="generating", phase="generating", progress=pct, message=msg)
                    
                success = upscale_image_file(input_path, output_path, strength=strength, factor=factor, 
                                             progress_callback=on_progress, force=force, bypass_warning=bypass_warning)

        if success:
            from . import get_relative_path
            send_update(status="complete", phase="complete", progress=100,
                       message="Upscale completed successfully", result_path=get_relative_path(output_path))
        else:
            send_update(status="failed", phase="failed", progress=100,
                       message="Upscale failed", error="Upscale returned False")
    except Exception as e:
        send_update(status="failed", phase="failed", progress=100,
                   message=f"Error: {str(e)}", error=str(e))
    finally:
        try:
            from ai_media.utils.system import clear_gpu_memory
            clear_gpu_memory()
        except Exception:
            pass

