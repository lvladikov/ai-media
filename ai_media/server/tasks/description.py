"""Description/caption generation task."""

import os
import torch
from ...generators.description import generate_caption
from ...utils.system import clear_gpu_memory

def run_description(job_id, input_path, model_name, output_path=None, force=False, bypass_warning=False, progress_queue=None):
    """
    Run analysis (description generation) in a separate process.
    
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
            message=f"Loading analysis model: {model_name}..."
        )
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        import datetime
        send_update(
            status="generating",
            progress=30,
            phase="Processing",
            message=f"Running {model_name}...",
            generation_started_at=datetime.datetime.utcnow().isoformat()
        )
        
        caption = ""
        output_file = output_path
        
        if model_name == "auto-subtitles":
            from ...generators.subtitles import SubtitlesGenerator
            gen = SubtitlesGenerator(device=str(device))
            
            # Run generator (creates files side-by-side with input by default)
            # We want to capture the SRT content to show in UI
            gen.run(input_path, model_size="medium")
            
            # Predict output path to read it back
            # SubtitlesGenerator behavior: input.srt (if no lang specific)
            # Actually run() generates .srt for detected/source lang.
            # Let's try to find the generated .srt file
            possible_srt = Path(input_path).with_suffix(".srt")
            if possible_srt.exists():
                with open(possible_srt, "r", encoding="utf-8") as f:
                    caption = f.read()
                output_file = str(possible_srt)
            else:
                caption = "Subtitles generated but file path could not be resolved for preview."
                
        elif model_name == "transcription":
            from ...generators.transcription import TranscriptionGenerator
            gen = TranscriptionGenerator(device=str(device))
            caption = gen.run(input_path, output_format="markdown")
            
            # For transcription, we might want to save a .md file
            if not output_file:
                output_file = str(Path(input_path).with_suffix(".md"))
                
        else:
            # Standard Analysis Models (Description)
            caption = generate_caption(input_path, device, quiet=False, model_type=model_name)
        
        if caption:
            # Save to file if output_path provided
            if output_path:
                try:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(caption)
                except Exception as e:
                    print(f"Error saving analysis result to {output_path}: {e}")

            from . import get_relative_path
            send_update(
                status="complete",
                progress=100,
                phase="Complete",
                message="Description generated successfully",
                result=caption,
                result_path=get_relative_path(output_path) if output_path else None
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
