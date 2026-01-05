"""Description/caption generation task."""

import os
import torch
from ...generators.description import generate_caption
from ...utils.system import clear_gpu_memory

def run_description(job_id, input_path, model_name, output_path=None, force=False, bypass_warning=False, progress_queue=None):
    """
    Run vision (description generation) in a separate process.
    
    Args:
        job_id: Unique job ID
        input_path: Path to image or video
        model_name: Model name or ID
        output_path: Path to save description (optional)
        force: Force execution
        bypass_warning: Skip resource warnings
        progress_queue: Progress queue
    """
    def send_update(**kwargs):
        if progress_queue:
            try:
                progress_queue.put({"job_id": job_id, **kwargs})
            except Exception:
                pass

    try:
        send_update(
            status="loading",
            progress=10,
            phase="Initializing",
            message=f"Loading vision model: {model_name}..."
        )
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        send_update(
            status="generating",
            progress=30,
            phase="Processing",
            message="Generating description..."
        )
        
        # Generate caption
        caption = generate_caption(input_path, device, quiet=False, model_type=model_name)
        
        if caption:
            # Save to file if output_path provided
            if output_path:
                try:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(caption)
                except Exception as e:
                    print(f"Error saving vision result to {output_path}: {e}")

            send_update(
                status="complete",
                progress=100,
                phase="Complete",
                message="Description generated successfully",
                result=caption,
                result_path=output_path
            )
        else:
            send_update(
                status="failed",
                progress=100,
                phase="Failed",
                error="Failed to generate description"
            )
            
    except Exception as e:
        send_update(
            status="failed",
            progress=100,
            phase="Error",
            error=str(e)
        )
    finally:
        clear_gpu_memory()
