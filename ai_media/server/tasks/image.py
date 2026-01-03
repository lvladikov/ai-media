"""Image generation background task."""

import os
from pathlib import Path
from multiprocessing import Queue


def run_image_generation(
    job_id: str,
    prompt: str,
    output_path: str,
    width: int,
    height: int,
    model: str,
    steps: int,
    guidance_scale: float,
    force: bool = False,
    progress_queue: Queue = None,
):
    """Background task for image generation. Runs in child process."""
    
    def send_update(**kwargs):
        """Send progress update to parent via queue."""
        if progress_queue:
            try:
                progress_queue.put({"job_id": job_id, **kwargs})
            except Exception:
                pass
    
    try:
        send_update(status="loading", phase="loading", progress=10, message="Loading model...")
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Import and run the existing generator
        from ai_media.generators.image import generate_image as gen_image
        
        send_update(status="generating", phase="generating", progress=30, message="Generating image...")
        
        success = gen_image(
            prompt=prompt,
            output_file=output_path,
            width=width,
            height=height,
            model_name=model,
            steps=steps,
            guidance_scale=guidance_scale,
            force=force,
        )
        
        if success:
            send_update(
                status="complete",
                phase="complete",
                progress=100,
                message="Image generated successfully",
                result_path=output_path,
            )
        else:
            send_update(
                status="failed",
                phase="failed",
                progress=100,
                message="Generation failed",
                error="Image generation returned False",
            )
    except Exception as e:
        send_update(
            status="failed",
            phase="failed",
            progress=100,
            message=f"Error: {str(e)}",
            error=str(e),
        )
    finally:
        # Clear GPU memory
        try:
            from ai_media.utils.system import clear_gpu_memory
            clear_gpu_memory()
        except Exception:
            pass

