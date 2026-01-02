"""Convert background task."""

import os
from pathlib import Path
from multiprocessing import Queue


def run_convert(
    job_id: str,
    input_path: str,
    target_format: str,
    output_path: str,
    progress_queue: Queue = None,
):
    """Background task for media conversion. Runs in child process."""
    
    def send_update(**kwargs):
        """Send progress update to parent via queue."""
        if progress_queue:
            try:
                progress_queue.put({"job_id": job_id, **kwargs})
            except Exception:
                pass
    
    try:
        send_update(status="loading", phase="loading", progress=10, message="Preparing for conversion...")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        input_ext = Path(input_path).suffix.lower().lstrip(".")
        target_format = target_format.lower()
        
        send_update(status="generating", phase="generating", progress=30, message=f"Converting {input_ext} to {target_format}...")
        
        from ai_media.conversion import (
            convert_image, convert_video, convert_audio, convert_document
        )
        
        success = False
        
        # Determine conversion type
        if target_format in ['jpg', 'png', 'webp']:
            success = convert_image(input_path, output_path)
        elif target_format in ['mp4', 'gif', 'mov', 'webm']:
            success = convert_video(input_path, output_path)
        elif target_format in ['mp3', 'wav', 'aac', 'flac']:
            success = convert_audio(input_path, output_path)
        elif target_format in ['md', 'html', 'pdf', 'docx', 'txt']:
            success = convert_document(input_path, output_path)
        else:
            raise ValueError(f"Unsupported target format: {target_format}")

        if success:
            send_update(status="complete", phase="complete", progress=100,
                       message="Conversion completed successfully", result_path=output_path)
        else:
            send_update(status="failed", phase="failed", progress=100,
                       message="Conversion failed", error="Conversion returned False")
    except Exception as e:
        send_update(status="failed", phase="failed", progress=100,
                   message=f"Error: {str(e)}", error=str(e))

