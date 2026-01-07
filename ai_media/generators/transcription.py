
from typing import Optional, Dict, List, Union
import json
import os
from pathlib import Path
from .subtitles import SubtitlesGenerator

class TranscriptionGenerator:
    """
    Generator for plain text/JSON transcription of audio/video using faster-whisper.
    Wraps SubtitlesGenerator for core transcription logic.
    """
    def __init__(self, device: str = "cuda"):
        self.subtitles_gen = SubtitlesGenerator(device=device)

    def run(self, input_path: str, output_format: str = "markdown") -> str:
        """
        Run transcription and return formatted string.
        
        Args:
            input_path: Path to audio/video file.
            output_format: 'markdown' (default) or 'json'.
            
        Returns:
            The transcribed content as a string.
        """
        # 1. Extract Audio
        audio_path = self.subtitles_gen.extract_audio(input_path)
        if not audio_path:
            raise RuntimeError("Failed to extract audio")

        try:
            # 2. Transcribe
            # default to "large-v3" or "medium" for better quality in analysis mode? 
            # SubtitlesGenerator defaults to "small". Let's use "medium" for Analysis if not specified.
            # actually let's stick to defaults or allow config.
            segments, _, _ = self.subtitles_gen.transcribe_audio(audio_path, model_size="medium")
            
            if not segments:
                return "No speech detected."

            # 3. Format Output
            if output_format == "json":
                return json.dumps(segments, indent=2)
            else:
                # Markdown format
                lines = []
                for s in segments:
                    # Time in [MM:SS]
                    start_m = int(s['start'] // 60)
                    start_s = int(s['start'] % 60)
                    timestamp = f"[{start_m:02d}:{start_s:02d}]"
                    lines.append(f"**{timestamp}** {s['text']}")
                return "\n\n".join(lines)

        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
