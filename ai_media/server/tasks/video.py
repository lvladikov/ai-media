"""Video generation background task."""

import os
from pathlib import Path
from typing import Optional
from multiprocessing import Queue


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
    progress_queue: Queue = None,
):
    """Background task for video generation. Runs in child process."""
    
    def send_update(**kwargs):
        """Send progress update to parent via queue."""
        if progress_queue:
            try:
                progress_queue.put({"job_id": job_id, **kwargs})
            except Exception:
                pass
    
    try:
        send_update(status="loading", phase="loading", progress=10, message="Loading video model...")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        from ai_media.generators.video import generate_video as gen_video
        
        send_update(status="generating", phase="generating", progress=30, message="Generating video...")
        
        success = gen_video(
            prompt=prompt,
            output_path=output_path,
            duration=duration,
            width=width,
            height=height,
            model_name=model,
            image_input=input_image,
        )
        
        if success:
            send_update(status="complete", phase="complete", progress=100, 
                       message="Video generated successfully", result_path=output_path)
        else:
            send_update(status="failed", phase="failed", progress=100,
                       message="Generation failed", error="Video generation returned False")
    except Exception as e:
        send_update(status="failed", phase="failed", progress=100,
                   message=f"Error: {str(e)}", error=str(e))
    finally:
        try:
            from ai_media.utils.system import clear_gpu_memory
            clear_gpu_memory()
        except Exception:
            pass

