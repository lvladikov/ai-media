"""Audio generation background task."""

import os
from pathlib import Path
from multiprocessing import Queue


def run_audio_generation(
    job_id: str,
    prompt: str,
    output_path: str,
    duration: float,
    model: str,
    force: bool = False,
    bypass_warning: bool = False,
    progress_queue: Queue = None,
):
    """Background task for audio generation. Runs in child process."""
    
    def send_update(**kwargs):
        """Send progress update to parent via queue."""
        if progress_queue:
            try:
                progress_queue.put({"job_id": job_id, **kwargs})
            except Exception:
                pass
    
    try:
        send_update(status="loading", phase="loading", progress=0, message="Loading audio model...")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        from ai_media.generators.audio import generate_audio as gen_audio
        
        send_update(status="generating", phase="generating", progress=30, message="Generating audio...")
        
        success = gen_audio(
            prompt=prompt,
            output_path=output_path,
            duration=duration,
            sampling_rate=44100,  # Default sampling rate
            model_name=model,
            force=force,
            bypass_warning=bypass_warning,
        )
        
        if success:
            send_update(status="complete", phase="complete", progress=100,
                       message="Audio generated successfully", result_path=output_path)
        else:
            send_update(status="failed", phase="failed", progress=100,
                       message="Generation failed", error="Audio generation returned False")
    except Exception as e:
        send_update(status="failed", phase="failed", progress=100,
                   message=f"Error: {str(e)}", error=str(e))
    finally:
        try:
            from ai_media.utils.system import clear_gpu_memory
            clear_gpu_memory()
        except Exception:
            pass

