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
    from ai_media.generators.audio import generate_audio as gen_audio
    
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
        
        import datetime
        generation_start_sent = False
        
        def on_progress(pct, msg):
            nonlocal generation_start_sent
            # Defer status change and timer until we see a "Synthesizing" or similar message
            is_generating = any(k in msg.lower() for k in ["synthesizing", "generating"])
            
            update_kwargs = {
                "status": "generating" if is_generating else "loading",
                "phase": "generating" if is_generating else "loading",
                "progress": pct,
                "message": msg
            }
            if is_generating and not generation_start_sent:
                update_kwargs["generation_started_at"] = datetime.datetime.utcnow().isoformat()
                generation_start_sent = True
            send_update(**update_kwargs)

        success = gen_audio(
            prompt=prompt,
            output_path=output_path,
            duration=duration,
            sampling_rate=44100,  # Default sampling rate
            model_name=model,
            force=force,
            bypass_warning=bypass_warning,
            progress_callback=on_progress
        )
        
        if success:
            from . import get_relative_path
            send_update(status="complete", phase="complete", progress=100,
                       message="Audio generated successfully", result_path=get_relative_path(output_path))
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

