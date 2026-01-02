"""Image generation background task."""

import os
from pathlib import Path

from ..jobs import update_job, is_job_cancelled


def run_image_generation(
    job_id: str,
    prompt: str,
    output_path: str,
    width: int,
    height: int,
    model: str,
    steps: int,
    guidance_scale: float,
):
    """Background task for image generation."""
    try:
        # Check for cancellation before starting
        if is_job_cancelled(job_id):
            return
            
        update_job(job_id, status="loading", phase="loading", progress=10, message="Loading model...")
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Import and run the existing generator
        from ai_media.generators.image import generate_image as gen_image
        
        # Check for cancellation before generation
        if is_job_cancelled(job_id):
            print(f"🛑 Image generation cancelled for job {job_id[:8]}...")
            return
        
        update_job(job_id, status="generating", phase="generating", progress=30, message="Generating image...")
        
        success = gen_image(
            prompt=prompt,
            output_file=output_path,
            width=width,
            height=height,
            model_name=model,
            steps=steps,
            guidance_scale=guidance_scale,
        )
        
        # Check for cancellation after generation (cleanup if cancelled mid-generation)
        if is_job_cancelled(job_id):
            print(f"🛑 Image generation cancelled for job {job_id[:8]}...")
            # Clean up partial output if exists
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except Exception:
                pass
            return
        
        if success:
            update_job(
                job_id,
                status="complete",
                phase="complete",
                progress=100,
                message="Image generated successfully",
                result_path=output_path,
            )
        else:
            update_job(
                job_id,
                status="failed",
                phase="failed",
                progress=100,
                message="Generation failed",
                error="Image generation returned False",
            )
    except Exception as e:
        if not is_job_cancelled(job_id):
            update_job(
                job_id,
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
