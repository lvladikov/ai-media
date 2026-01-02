"""Audio generation background task."""

import os
from pathlib import Path

from ..jobs import update_job, is_job_cancelled


def run_audio_generation(
    job_id: str,
    prompt: str,
    output_path: str,
    duration: float,
    model: str,
):
    """Background task for audio generation."""
    try:
        # Check for cancellation before starting
        if is_job_cancelled(job_id):
            return
            
        update_job(job_id, status="loading", phase="loading", progress=10, message="Loading audio model...")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        from ai_media.generators.audio import generate_audio as gen_audio
        
        # Check for cancellation before generation
        if is_job_cancelled(job_id):
            print(f"🛑 Audio generation cancelled for job {job_id[:8]}...")
            return
        
        update_job(job_id, status="generating", phase="generating", progress=30, message="Generating audio...")
        
        success = gen_audio(
            prompt=prompt,
            output_file=output_path,
            duration=duration,
            model_name=model,
        )
        
        # Check for cancellation after generation
        if is_job_cancelled(job_id):
            print(f"🛑 Audio generation cancelled for job {job_id[:8]}...")
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except Exception:
                pass
            return
        
        if success:
            update_job(job_id, status="complete", phase="complete", progress=100,
                      message="Audio generated successfully", result_path=output_path)
        else:
            update_job(job_id, status="failed", phase="failed", progress=100,
                      message="Generation failed", error="Audio generation returned False")
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
