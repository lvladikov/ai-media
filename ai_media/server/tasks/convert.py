"""Convert background task."""

import os
from pathlib import Path

from ..jobs import update_job, is_job_cancelled


def run_convert(
    job_id: str,
    input_path: str,
    target_format: str,
    output_path: str,
):
    """Background task for media conversion."""
    try:
        # Check for cancellation before starting
        if is_job_cancelled(job_id):
            return
            
        update_job(job_id, status="loading", phase="loading", progress=10, message="Preparing for conversion...")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        input_ext = Path(input_path).suffix.lower().lstrip(".")
        target_format = target_format.lower()
        
        # Check for cancellation before conversion
        if is_job_cancelled(job_id):
            print(f"🛑 Conversion cancelled for job {job_id[:8]}...")
            return
        
        update_job(job_id, status="generating", phase="generating", progress=30, message=f"Converting {input_ext} to {target_format}...")
        
        from ai_media.conversion import (
            convert_image, convert_video, convert_audio, convert_document
        )
        
        success = False
        
        # Determine conversion type
        if target_format in ['jpg', 'png', 'webp']:
            success = convert_image(input_path, output_path, target_format)
        elif target_format in ['mp4', 'gif', 'mov', 'webm']:
            success = convert_video(input_path, output_path, target_format)
        elif target_format in ['mp3', 'wav', 'aac', 'flac']:
            success = convert_audio(input_path, output_path, target_format)
        elif target_format in ['md', 'html', 'pdf', 'docx', 'txt']:
            # Document conversion
            success = convert_document(input_path, output_path, target_format)
        else:
            raise ValueError(f"Unsupported target format: {target_format}")

        # Check for cancellation after conversion
        if is_job_cancelled(job_id):
            print(f"🛑 Conversion cancelled for job {job_id[:8]}...")
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except Exception:
                pass
            return

        if success:
            update_job(job_id, status="complete", phase="complete", progress=100,
                      message="Conversion completed successfully", result_path=output_path)
        else:
            update_job(job_id, status="failed", phase="failed", progress=100,
                      message="Conversion failed", error="Conversion returned False")
    except Exception as e:
        if not is_job_cancelled(job_id):
            update_job(job_id, status="failed", phase="failed", progress=100,
                      message=f"Error: {str(e)}", error=str(e))
