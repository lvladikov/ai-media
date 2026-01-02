"""Video generation background task."""

import os
from pathlib import Path
from typing import Optional

from ..jobs import update_job, is_job_cancelled


def run_video_generation(
    job_id: str,
    prompt: str,
    output_path: str,
    width: int,
    height: int,
    duration: float,
    fps: int,
    model: str,
    input_image: Optional[str],
):
    """Background task for video generation."""
    try:
        # Check for cancellation before starting
        if is_job_cancelled(job_id):
            return
            
        update_job(job_id, status="loading", phase="loading", progress=10, message="Loading video model...")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        from ai_media.generators.video import generate_video as gen_video
        
        # Check for cancellation before generation
        if is_job_cancelled(job_id):
            print(f"🛑 Video generation cancelled for job {job_id[:8]}...")
            return
        
        update_job(job_id, status="generating", phase="generating", progress=30, message="Generating video...")
        
        success = gen_video(
            prompt=prompt,
            output_file=output_path,
            width=width,
            height=height,
            duration=duration,
            fps=fps,
            model_name=model,
            input_image=input_image,
        )
        
        # Check for cancellation after generation
        if is_job_cancelled(job_id):
            print(f"🛑 Video generation cancelled for job {job_id[:8]}...")
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except Exception:
                pass
            return
        
        if success:
            update_job(job_id, status="complete", phase="complete", progress=100, 
                      message="Video generated successfully", result_path=output_path)
        else:
            update_job(job_id, status="failed", phase="failed", progress=100,
                      message="Generation failed", error="Video generation returned False")
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
