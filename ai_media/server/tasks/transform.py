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
    bypass_warning: bool = False,
    force: bool = False,
    progress_queue: Queue = None,
):
    """Background task for image transformation. Runs in child process."""
    
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
            from ai_media.generators.transform import remove_background
            send_update(status="generating", phase="generating", progress=30, message="Removing background...")
            success = remove_background(input_path, output_path, force=force, bypass_warning=bypass_warning)
        else:
            from ai_media.generators.transform import generate_edit
            
            def on_progress(pct, msg):
                send_update(status="generating", phase="generating", progress=pct, message=msg)
                
            send_update(status="generating", phase="generating", progress=30, message="Transforming image...")
            success = generate_edit(
                input_path=input_path,
                prompt=instruction,
                output_path=output_path,
                model_name=model,
                progress_callback=on_progress,
                force=force,
                bypass_warning=bypass_warning,
            )
        
        if success:
            send_update(status="complete", phase="complete", progress=100,
                       message="Transform completed successfully", result_path=output_path)
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

