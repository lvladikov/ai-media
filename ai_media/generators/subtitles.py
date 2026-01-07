
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import torch


from rich.console import Console
console = Console()


# NLLB Language Codes
NLLB_LANGUAGE_CODES = {
    "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "ru": "rus_Cyrl", "zh": "zho_Hans",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "ar": "arb_Arab", "hi": "hin_Deva",
    "nl": "nld_Latn", "pl": "pol_Latn", "tr": "tur_Latn", "sv": "swe_Latn",
    "no": "nob_Latn", "da": "dan_Latn", "fi": "fin_Latn", "el": "ell_Grek",
    "he": "heb_Hebr", "th": "tha_Thai", "vi": "vie_Latn", "id": "ind_Latn",
    "bg": "bul_Cyrl", "uk": "ukr_Cyrl", "cs": "ces_Latn", "ro": "ron_Latn",
    "hu": "hun_Latn", "sk": "slk_Latn", "hr": "hrv_Latn", "sr": "srp_Cyrl",
    "sl": "slv_Latn", "et": "est_Latn", "lv": "lvs_Latn", "lt": "lit_Latn",
    "mk": "mkd_Cyrl", "sq": "als_Latn", "bs": "bos_Latn", "mt": "mlt_Latn",
    "is": "isl_Latn", "ga": "gle_Latn", "cy": "cym_Latn", "af": "afr_Latn",
    "sw": "swh_Latn", "bn": "ben_Beng", "ta": "tam_Taml", "te": "tel_Telu",
    "ml": "mal_Mlym", "kn": "kan_Knda", "mr": "mar_Deva", "gu": "guj_Gujr",
    "pa": "pan_Guru", "ur": "urd_Arab", "fa": "pes_Arab", "ms": "zsm_Latn",
    "tl": "tgl_Latn", "my": "mya_Mymr", "km": "khm_Khmr", "lo": "lao_Laoo",
    "ne": "npi_Deva", "si": "sin_Sinh", "ka": "kat_Geor", "hy": "hye_Armn",
    "az": "azj_Latn", "kk": "kaz_Cyrl", "uz": "uzn_Latn", "mn": "khk_Cyrl",
}

LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "hi": "Hindi",
    "nl": "Dutch", "pl": "Polish", "tr": "Turkish", "sv": "Swedish",
    "no": "Norwegian", "da": "Danish", "fi": "Finnish", "el": "Greek",
    "he": "Hebrew", "th": "Thai", "vi": "Vietnamese", "id": "Indonesian",
    "bg": "Bulgarian", "uk": "Ukrainian", "cs": "Czech", "ro": "Romanian",
    "hu": "Hungarian", "sk": "Slovak", "hr": "Croatian", "sr": "Serbian",
    "sl": "Slovenian", "et": "Estonian", "lv": "Latvian", "lt": "Lithuanian",
    "mk": "Macedonian", "sq": "Albanian", "bs": "Bosnian", "mt": "Maltese",
    "is": "Icelandic", "ga": "Irish", "cy": "Welsh", "af": "Afrikaans",
    "sw": "Swahili", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
    "ml": "Malayalam", "kn": "Kannada", "mr": "Marathi", "gu": "Gujarati",
    "pa": "Punjabi", "ur": "Urdu", "fa": "Persian", "ms": "Malay",
    "tl": "Filipino", "my": "Myanmar", "km": "Khmer", "lo": "Lao",
    "ne": "Nepali", "si": "Sinhala", "ka": "Georgian", "hy": "Armenian",
    "az": "Azerbaijani", "kk": "Kazakh", "uz": "Uzbek", "mn": "Mongolian",
}


class SubtitlesGenerator:
    """
    Generator for creating subtitles using faster-whisper and NLLB-200.
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.compute_type = "float16" if device == "cuda" else "int8"
        
        # Verify faster-whisper installation
        try:
            import faster_whisper
        except ImportError:
            console.print("[bold red]Error: fast-whisper is not installed.[/bold red]")
            console.print("Please install it with: pip install faster-whisper")
            sys.exit(1)

    def extract_audio(self, video_path: str) -> Optional[str]:
        """Extract audio from video file to a temporary wav file."""
        video_path_obj = Path(video_path)
        audio_path = video_path_obj.with_suffix(".tmp.wav")
        
        console.print(f"[cyan]Extracting audio from:[/cyan] {video_path}")
        
        try:
            command = [
                "ffmpeg",
                "-i", str(video_path),
                "-vn",                      # No video
                "-acodec", "pcm_s16le",     # PCM 16-bit little-endian
                "-ar", "16000",             # 16kHz sample rate
                "-ac", "1",                 # Mono channel
                "-y",                       # Overwrite output file
                str(audio_path)
            ]
            
            # Using subprocess directly as imageio-ffmpeg wrapper can be limited
            subprocess.run(command, check=True, stderr=subprocess.PIPE)
            console.print("[green]Audio extraction complete.[/green]")
            return str(audio_path)
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]FFmpeg failed to extract audio:[/bold red] {e.stderr.decode() if e.stderr else str(e)}")
            return None
        except Exception as e:
            console.print(f"[bold red]Error extracting audio:[/bold red] {e}")
            return None

    def transcribe_audio(self, 
                         audio_path: str, 
                         model_size: str = "small", 
                         language: str = None,
                         vad_min_silence_duration_ms: int = 2000,
                         vad_threshold: float = 0.5,
                         condition_on_previous_text: bool = True,
                         no_speech_threshold: float = 0.6,
                         **kwargs) -> Tuple[List[Dict], float, float]:
        """
        Transcribe audio using faster-whisper.
        
        Args:
            audio_path: Path to audio file
            model_size: Whisper model size
            language: Source language code (auto-detected if None)
            vad_min_silence_duration_ms: Min silence duration (ms) to split segments
            vad_threshold: Speech probability threshold (0.0-1.0)
            condition_on_previous_text: Use previous segment as context
            no_speech_threshold: Threshold for skipping silent segments
            **kwargs: Additional whisper parameters
        
        Returns:
            Tuple containing:
            - List of segments (dict with start, end, text)
            - Audio duration in seconds
            - Transcription time in seconds
        """
        from faster_whisper import WhisperModel
        
        console.print(f"[bold cyan]Loading Whisper model ({model_size}) on {self.device}...[/bold cyan]")
        model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)
        
        start_time = time.time()
        console.print(f"Transcribing audio...")
        
        try:
            # beam_size=5 is standard recommendation for accuracy
            segments_generator, info = model.transcribe(
                audio_path, 
                beam_size=5,
                language=language,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=vad_min_silence_duration_ms,
                    threshold=vad_threshold,
                ),
                condition_on_previous_text=condition_on_previous_text,
                no_speech_threshold=no_speech_threshold, 
            )
            
            # Consume generator to get all segments
            segments = []
            for segment in segments_generator:
                segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                })
                # Optional: Live feedback could go here if verbose
            
            transcription_time = time.time() - start_time
            audio_duration = info.duration
            
            console.print(f"[green]Transcription complete.[/green] ({len(segments)} segments)")
            console.print(f"Time: {transcription_time:.2f}s (Audio Duration: {audio_duration:.2f}s)")
            
            return segments, audio_duration, transcription_time

            
        except Exception as e:
            console.print(f"[bold red]Transcription failed:[/bold red] {e}")
            return [], 0.0, 0.0

    def translate_segments(self, 
                           segments: List[Dict], 
                           source_lang: str, 
                           target_lang: str, 
                           model_size: str = "facebook/nllb-200-distilled-600M") -> List[Dict]:
        """
        Translate segments using NLLB-200.
        """
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        except ImportError:
            console.print("[bold red]Error: transformers is not installed.[/bold red]") 
            return []

        src_nllb = NLLB_LANGUAGE_CODES.get(source_lang)
        tgt_nllb = NLLB_LANGUAGE_CODES.get(target_lang)
        
        if not src_nllb:
            # Fallback to English if unknown, but better to be explicit
            src_nllb = "eng_Latn"
            console.print(f"[yellow]Warning: Unknown source language '{source_lang}', assuming English ({src_nllb})[/yellow]")
        
        if not tgt_nllb:
            console.print(f"[bold red]Error: Unknown target language '{target_lang}'[/bold red]")
            return []

        console.print(f"[bold cyan]Loading NLLB model ({model_size})...[/bold cyan]")
        console.print(f"Translating: {source_lang} ({src_nllb}) -> {target_lang} ({tgt_nllb})")
        
        # Setup model loading args
        model_kwargs = {}
        if self.device == "cuda":
            # Optimization: Use bfloat16 on CUDA as requested
            model_kwargs["torch_dtype"] = torch.bfloat16
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_size)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_size, **model_kwargs).to(self.device)
            
            translator = pipeline(
                "translation",
                model=model,
                tokenizer=tokenizer,
                src_lang=src_nllb,
                tgt_lang=tgt_nllb,
                max_length=512,
                device=0 if self.device == "cuda" else -1
            )
            
            console.print(f"Translating {len(segments)} segments...")
            
            translated_segments = []
            
            # TODO: Batch processing would be faster, but per-segment is safer for alignment
            # For simplicity and porting logic, we do loop.
            
            # Using rich progress bar
            from rich.progress import track
            
            for segment in track(segments, description="Translating segments..."):
                result = translator(segment["text"])
                translated_text = result[0]["translation_text"]
                
                translated_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": translated_text
                })
                
            return translated_segments
            
        except Exception as e:
            console.print(f"[bold red]Translation failed:[/bold red] {e}")
            import traceback
            traceback.print_exc()
            return []

    def generate_srt(self, segments: List[Dict], output_path: str):
        """Generate SRT file from segments."""
        def format_timestamp(seconds: float) -> str:
            milliseconds = int((seconds % 1) * 1000)
            seconds = int(seconds)
            minutes = seconds // 60
            hours = minutes // 60
            minutes = minutes % 60
            seconds = seconds % 60
            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments, start=1):
                    start_str = format_timestamp(segment["start"])
                    end_str = format_timestamp(segment["end"])
                    text = segment["text"].strip()
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_str} --> {end_str}\n")
                    f.write(f"{text}\n\n")
            
            console.print(f"[green]Saved SRT:[/green] {output_path}")
        except Exception as e:
            console.print(f"[bold red]Error saving SRT:[/bold red] {e}")

    def generate_vtt(self, segments: List[Dict], output_path: str):
        """Generate WebVTT file from segments."""
        def format_timestamp(seconds: float) -> str:
            milliseconds = int((seconds % 1) * 1000)
            seconds = int(seconds)
            minutes = seconds // 60
            hours = minutes // 60
            minutes = minutes % 60
            seconds = seconds % 60
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")
                for i, segment in enumerate(segments, start=1):
                    start_str = format_timestamp(segment["start"])
                    end_str = format_timestamp(segment["end"])
                    text = segment["text"].strip()
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_str} --> {end_str}\n")
                    f.write(f"{text}\n\n")
            
            console.print(f"[green]Saved VTT:[/green] {output_path}")
        except Exception as e:
            console.print(f"[bold red]Error saving VTT:[/bold red] {e}")

    def generate_ass(self, segments: List[Dict], output_path: str):
        """Generate ASS (Advanced SubStation Alpha) file from segments."""
        def format_timestamp(seconds: float) -> str:
            centiseconds = int((seconds % 1) * 100)
            seconds = int(seconds)
            minutes = seconds // 60
            hours = minutes // 60
            minutes = minutes % 60
            seconds = seconds % 60
            return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                # ASS Header
                f.write("[Script Info]\n")
                f.write("Title: AI-Media Generated Subtitles\n")
                f.write("ScriptType: v4.00+\n")
                f.write("Collisions: Normal\n")
                f.write("PlayDepth: 0\n\n")
                
                # Styles
                f.write("[V4+ Styles]\n")
                f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
                f.write("Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1\n\n")
                
                # Events
                f.write("[Events]\n")
                f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
                
                for segment in segments:
                    start_str = format_timestamp(segment["start"])
                    end_str = format_timestamp(segment["end"])
                    text = segment["text"].strip().replace("\n", "\\N")
                    f.write(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}\n")
            
            console.print(f"[green]Saved ASS:[/green] {output_path}")
        except Exception as e:
            console.print(f"[bold red]Error saving ASS:[/bold red] {e}")

    def generate_sub(self, segments: List[Dict], output_path: str, fps: float = 25.0):
        """Generate SUB (MicroDVD) file from segments."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for segment in segments:
                    start_frame = int(segment["start"] * fps)
                    end_frame = int(segment["end"] * fps)
                    text = segment["text"].strip().replace("\n", "|")
                    f.write(f"{{{start_frame}}}{{{end_frame}}}{text}\n")
            
            console.print(f"[green]Saved SUB:[/green] {output_path}")
        except Exception as e:
            console.print(f"[bold red]Error saving SUB:[/bold red] {e}")

    def generate_txt(self, segments: List[Dict], output_path: str):
        """Generate plain text transcript (no timestamps)."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for segment in segments:
                    f.write(segment["text"].strip() + "\n")
            
            console.print(f"[green]Saved TXT:[/green] {output_path}")
        except Exception as e:
            console.print(f"[bold red]Error saving TXT:[/bold red] {e}")

    def generate_json(self, segments: List[Dict], output_path: str):
        """Generate JSON file with full segment data."""
        import json
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"segments": segments}, f, indent=2, ensure_ascii=False)
            
            console.print(f"[green]Saved JSON:[/green] {output_path}")
        except Exception as e:
            console.print(f"[bold red]Error saving JSON:[/bold red] {e}")

    def save_subtitles(self, segments: List[Dict], output_path: str, format: str = "srt", fps: float = 25.0):
        """Save subtitles in the specified format."""
        format = format.lower()
        if format == "srt":
            self.generate_srt(segments, output_path)
        elif format == "vtt":
            self.generate_vtt(segments, output_path)
        elif format == "ass":
            self.generate_ass(segments, output_path)
        elif format == "sub":
            self.generate_sub(segments, output_path, fps)
        elif format == "txt":
            self.generate_txt(segments, output_path)
        elif format == "json":
            self.generate_json(segments, output_path)
        else:
            console.print(f"[bold red]Unsupported format:[/bold red] {format}")
            console.print("Supported formats: srt, vtt, ass, sub, txt, json")

    def run(self, 
            input_path: str, 
            output_context: str = None, # Not used directly but kept for interface consistency
            model_size: str = "small",
            source_lang: str = None, 
            target_langs: List[str] = None,
            output_format: str = "srt",
            fps: float = 25.0,
            **vad_kwargs):
        """
        Main execution method.
        
        Args:
            input_path: Path to input video/audio file
            output_context: Not used (kept for interface consistency)
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
            source_lang: Source language code (auto-detected if None)
            target_langs: List of target language codes for translation
            output_format: Output subtitle format (srt, vtt, ass, sub, txt, json)
            fps: Frames per second for SUB format
            **vad_kwargs: VAD parameters (vad_min_silence_duration_ms, vad_threshold, etc.)
        """
        input_file = Path(input_path)
        if not input_file.exists():
            console.print(f"[bold red]Error: Input file not found:[/bold red] {input_path}")
            return

        # 1. Extract Audio (handles both video and audio inputs)
        audio_path = self.extract_audio(input_path)
        if not audio_path:
            return

        try:
            # 2. Transcribe with VAD params
            segments, _, _ = self.transcribe_audio(
                audio_path, 
                model_size=model_size, 
                language=source_lang,
                **vad_kwargs
            )
            if not segments:
                console.print("[yellow]No speech detected.[/yellow]")
                return

            # Save original transcription in requested format
            out_ext = f".{output_format}"
            base_output = input_file.with_suffix(out_ext)
            self.save_subtitles(segments, str(base_output), format=output_format, fps=fps)

            # 3. Translate if requested
            if target_langs:
                # If source lang wasn't provided, default to 'en' for translation
                src_lang_code = source_lang if source_lang else "en"
                
                for tgt in target_langs:
                    translated = self.translate_segments(
                        segments, 
                        source_lang=src_lang_code,
                        target_lang=tgt
                    )
                    if translated:
                         out_path = input_file.with_suffix(f".{tgt}{out_ext}")
                         self.save_subtitles(translated, str(out_path), format=output_format, fps=fps)

        finally:
            # Cleanup temp audio if extracted from video
            if audio_path and Path(audio_path).exists() and audio_path != input_path:
                try:
                    Path(audio_path).unlink()
                except:
                    pass

            # Cleanup temp audio
            if os.path.exists(audio_path):
                os.remove(audio_path)
