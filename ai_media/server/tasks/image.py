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
    negative_prompt: str = "",
    force: bool = False,
    bypass_warning: bool = False,
    framework: str = None,
    precision: str = None,
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
        send_update(status="loading", phase="loading", progress=0, message="Loading model...")
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Import and run the existing generator
        from ai_media.generators.image import ImageGenerator
        
        # Define callback for progress updates
        def on_progress(percent, message):
            send_update(
                status="generating",
                phase="generating",
                progress=percent,
                message=message
            )

        # Initialize generator with explicit framework/precision if provided
        generator = ImageGenerator(
            model_id=model, 
            use_mlx=(framework == "mlx") if framework else None,
            precision=precision,
            framework=framework 
        )
        
        outputs = generator.generate(
            prompt=prompt,
            output_file=output_path,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            force=force,
            bypass_warning=bypass_warning,
            progress_callback=on_progress,
        )
        
        success = len(outputs) > 0
        
        if success:
            from . import get_relative_path
            send_update(
                status="complete",
                phase="complete",
                progress=100,
                message="Image generated successfully",
                result_path=get_relative_path(output_path),
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

