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
    force: bool = False,
    bypass_warning: bool = False,
    framework: str = None,
    precision: str = None,
    progress_queue: Queue = None,
):
    """Background task for video generation. Runs in child process."""
    from ai_media.generators.video import generate_video as gen_video
    
    def send_update(**kwargs):
        """Send progress update to parent via queue."""
        if progress_queue:
            try:
                progress_queue.put({"job_id": job_id, **kwargs})
            except Exception:
                pass
    
    try:
        send_update(status="loading", phase="loading", progress=0, message="Loading video model...")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        import datetime
        generation_start_sent = False
        
        def on_progress(status_or_pct, pct_or_msg=None, msg=None, terminal=False):
            """Progress callback supporting both 2-arg and 4-arg formats."""
            nonlocal generation_start_sent
            
            # Handle both (pct, msg) and (status, pct, msg, terminal) signatures
            if isinstance(status_or_pct, str) and pct_or_msg is not None:
                # 4-arg format: (status, pct, msg, terminal)
                status = status_or_pct
                pct = pct_or_msg if isinstance(pct_or_msg, (int, float)) else 0
                message = msg if msg else str(pct_or_msg)
            else:
                # 2-arg format: (pct, msg)
                status = None
                pct = status_or_pct if isinstance(status_or_pct, (int, float)) else 0
                message = str(pct_or_msg) if pct_or_msg else ""
            
            # Detect generating phase from message content
            is_generating = any(k in message.lower() for k in [
                "rendering", "generating", "stitching", "upscaling", "muxing",
                "steps", "sampling", "processing"
            ]) or status == "generating"
            
            update_kwargs = {
                "status": "generating" if is_generating else "loading",
                "phase": "generating" if is_generating else "loading",
                "progress": pct,
                "message": message
            }
            if is_generating and not generation_start_sent:
                update_kwargs["generation_started_at"] = datetime.datetime.utcnow().isoformat()
                generation_start_sent = True
            send_update(**update_kwargs)


        success = gen_video(
            prompt=prompt,
            output_path=output_path,
            duration=duration,
            width=width,
            height=height,
            model_name=model,
            image_input=input_image,
            force=force,
            bypass_warning=bypass_warning,
            progress_callback=on_progress
        )
        
        if success:
            from . import get_relative_path
            send_update(status="complete", phase="complete", progress=100, 
                       message="Video generated successfully", result_path=get_relative_path(output_path))
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

