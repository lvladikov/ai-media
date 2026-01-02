"""Transform background task."""

import os
from pathlib import Path

from ..jobs import update_job, is_job_cancelled


def run_transform(
    job_id: str,
    input_path: str,
    instruction: str,
    output_path: str,
    model: str,
):
    """Background task for image transformation."""
    try:
        # Check for cancellation before starting
        if is_job_cancelled(job_id):
            return
            
        update_job(job_id, status="loading", phase="loading", progress=10, message="Loading transform model...")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Check for cancellation before generation
        if is_job_cancelled(job_id):
            print(f"🛑 Transform cancelled for job {job_id[:8]}...")
            return
        
        # Determine which transform function to use
        if model == "remove-bg" or instruction.lower() == "remove-bg":
            from ai_media.transform import remove_background
            update_job(job_id, status="generating", phase="generating", progress=30, message="Removing background...")
            success = remove_background(input_path, output_path)
        else:
            from ai_media.transform import transform_image as trans_img
            update_job(job_id, status="generating", phase="generating", progress=30, message="Transforming image...")
            success = trans_img(
                input_file=input_path,
                output_file=output_path,
                instruction=instruction,
                model_name=model,
            )
        
        # Check for cancellation after generation
        if is_job_cancelled(job_id):
            print(f"🛑 Transform cancelled for job {job_id[:8]}...")
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except Exception:
                pass
            return
        
        if success:
            update_job(job_id, status="complete", phase="complete", progress=100,
                      message="Transform completed successfully", result_path=output_path)
        else:
            update_job(job_id, status="failed", phase="failed", progress=100,
                      message="Transform failed", error="Transform returned False")
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
