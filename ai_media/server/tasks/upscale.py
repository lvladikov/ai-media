"""Upscale background task."""

import os
from pathlib import Path

from ..jobs import update_job, is_job_cancelled


def run_upscale(
    job_id: str,
    input_path: str,
    output_path: str,
    factor: float,
    method: str,
    strength: float,
):
    """Background task for upscaling."""
    try:
        # Check for cancellation before starting
        if is_job_cancelled(job_id):
            return
            
        update_job(job_id, status="loading", phase="loading", progress=10, message="Preparing upscale...")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        is_video = Path(input_path).suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv', '.webm']
        
        from ai_media.upscaling import (
            simple_upscale_image, simple_upscale_video,
            upscale_image_fast, upscale_video_fast,
            upscale_image_file, upscale_video_file
        )
        
        # Check for cancellation before upscaling
        if is_job_cancelled(job_id):
            print(f"🛑 Upscale cancelled for job {job_id[:8]}...")
            return
        
        update_job(job_id, status="generating", phase="generating", progress=30, message=f"Upscaling ({method} x{factor})...")
        
        success = False
        
        if is_video:
            if method == 'simple':
                success = simple_upscale_video(input_path, output_path, factor=factor, force=True)
            elif method == 'fast':
                success = upscale_video_fast(input_path, output_path, factor=factor)
            else:  # ai / creative
                # AI video upscale is very slow, warn user
                update_job(job_id, status="generating", phase="generating", progress=30, message="Upscaling video (AI Slow Mode)...")
                success = upscale_video_file(input_path, output_path, strength=strength, factor=factor)
        else:  # Image
            if method == 'simple':
                success = simple_upscale_image(input_path, output_path, factor=factor, force=True)
            elif method == 'fast':
                success = upscale_image_fast(input_path, output_path, factor=factor)
            else:  # ai / creative
                success = upscale_image_file(input_path, output_path, strength=strength, factor=factor)

        # Check for cancellation after upscaling
        if is_job_cancelled(job_id):
            print(f"🛑 Upscale cancelled for job {job_id[:8]}...")
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except Exception:
                pass
            return

        if success:
            update_job(job_id, status="complete", phase="complete", progress=100,
                      message="Upscale completed successfully", result_path=output_path)
        else:
            update_job(job_id, status="failed", phase="failed", progress=100,
                      message="Upscale failed", error="Upscale returned False")
    except Exception as e:
        if not is_job_cancelled(job_id):
            update_job(job_id, status="failed", phase="failed", progress=100,
                      message=f"Error: {str(e)}", error=str(e))
    finally:
        try:
            from ai_media.utils.system import clear_gpu_memory
            clear_gpu_memory()
        except Exception:
            pass
