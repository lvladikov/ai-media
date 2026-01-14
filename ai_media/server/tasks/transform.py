"""Transform background task."""

import os
from pathlib import Path
from multiprocessing import Queue


def run_transform(
    job_id: str,
    input_path: str,
    instruction: str,
    output_path: str,
    model: str,
    guidance_scale: float = 7.5,
    image_guidance_scale: float = 1.5,
    silhouette: bool = False,
    bypass_warning: bool = False,
    force: bool = False,
    progress_queue: Queue = None,
    use_mlx: bool = None,
    precision: str = None,
):
    """Background task for image transformation. Runs in child process."""
    from ai_media.generators.transform import remove_background, generate_edit
    
    def send_update(**kwargs):
        """Send progress update to parent via queue."""
        if progress_queue:
            try:
                progress_queue.put({"job_id": job_id, **kwargs})
            except Exception:
                pass
    
    try:
        send_update(status="loading", phase="loading", progress=10, message="Loading transform model...")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Determine which transform function to use
        if model == "remove-bg" or instruction.lower() == "remove-bg":
            import datetime
            send_update(
                status="generating", 
                phase="generating", 
                progress=30, 
                message="Removing background...",
                generation_started_at=datetime.datetime.utcnow().isoformat()
            )
            success = remove_background(input_path, output_path, silhouette=silhouette, force=force, bypass_warning=bypass_warning)
        else:
            import datetime
            generation_start_sent = False
            
            def on_progress(pct, msg):
                nonlocal generation_start_sent
                
                # Determine if we are actually generating or still loading/sharding
                is_actually_generating = "Generating" in msg or "Editing" in msg or "Applying edits" in msg
                
                update_kwargs = {
                    "status": "generating" if is_actually_generating else "loading",
                    "phase": "generating" if is_actually_generating else "loading",
                    "progress": pct,
                    "message": msg
                }
                
                if is_actually_generating and not generation_start_sent:
                    update_kwargs["generation_started_at"] = datetime.datetime.utcnow().isoformat()
                    generation_start_sent = True
                    
                send_update(**update_kwargs)
            success = generate_edit(
                input_path=input_path,
                prompt=instruction,
                output_path=output_path,
                model_name=model,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                progress_callback=on_progress,
                force=force,
                bypass_warning=bypass_warning,
                use_mlx=use_mlx,
                precision=precision,
            )
        
        if success:
            from . import get_relative_path
            send_update(status="complete", phase="complete", progress=100,
                       message="Transform completed successfully", result_path=get_relative_path(output_path))
        else:
            send_update(status="failed", phase="failed", progress=100,
                       message="Transform failed", error="Transform returned False")
    except Exception as e:
        send_update(status="failed", phase="failed", progress=100,
                   message=f"Error: {str(e)}", error=str(e))
    finally:
        try:
            from ai_media.utils.system import clear_gpu_memory
            clear_gpu_memory()
        except Exception:
            pass

