"""Convert background task."""

import os
import datetime
from pathlib import Path
from multiprocessing import Queue


def run_convert(
    job_id: str,
    input_path: str,
    target_format: str,
    output_path: str,
    ocr_enabled: bool = False,
    ocr_model: str = "florence",
    translate: bool = False,
    target_language: str = "eng_Latn",
    translation_model: str = "nllb-200-3.3b",
    render_method: str = "smart",
    is_direct_text: bool = False,
    progress_queue: Queue = None,
    **kwargs
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
        
        if translate and input_ext in ['jpg', 'jpeg', 'png', 'webp'] and target_format in ['jpg', 'jpeg', 'png', 'webp']:
             msg = f"Translating Image ({target_language})..."
        elif translate:
             if is_direct_text:
                 msg = "Translating Direct Text Input..."
             else:
                 msg = f"Translating Document ({input_ext.upper()})..."
        else:
             msg = f"Converting {input_ext} to {target_format}..."
             
        import datetime
        send_update(
            status="generating", 
            phase="generating", 
            progress=30, 
            message=msg,
            generation_started_at=datetime.datetime.utcnow().isoformat()
        )
        
        from ai_media.conversion import (
            convert_image, convert_video, convert_audio, convert_document
        )
        # Imports for Speech-to-Speech
        from ai_media.generators.transcription import TranscriptionGenerator
        from ai_media.generators.audio import generate_audio
        import ai_media.generators.text as gen_text
        from ai_media.generators.translation import TranslationGenerator

        success = False
        
        translate_model = translation_model
        
        # Callback to mark when actual translation work starts (after model loading)
        generation_started = [False]  # Use list to allow modification in nested scope
        def on_translation_ready():
            if not generation_started[0] and translate:
                generation_started[0] = True
                send_update(
                    status="generating",
                    phase="translating",
                    progress=50,
                    message="Translating content...",
                    generation_started_at=datetime.datetime.utcnow().isoformat()
                )
        
        # Speech-to-Speech (Seamless or Pipeline)
        input_is_av = input_ext in ['mp3', 'wav', 'aac', 'flac', 'm4a', 'ogg', 'webm', 'mp4', 'mov', 'avi', 'mkv']
        target_is_audio = target_format in ['mp3', 'wav', 'aac', 'flac']
        
        if translate and input_is_av and target_is_audio:
            send_update(status="generating", phase="init", progress=10, message=f"Starting translation ({translate_model})...")
            
            # Check if using SeamlessM4T
            if "seamless" in translate_model:
                send_update(phase="generating", progress=30, message="Running SeamlessM4T (S2ST)...")
                translator = TranslationGenerator()
                translator.load_model(translate_model)
                on_translation_ready()  # Model loaded, start timing
                success = translator.run(
                    input_path=input_path, 
                    target_lang=kwargs.get("target_language", "eng_Latn"),
                    task="s2st",
                    output_path=output_path
                )
                if success:
                     success = True # run returns path or None, verify? 
                     # run returns output_path on success based on my code
            
            else:
                # Fallback to Pipeline (Transcribe -> Text Translate -> TTS)
                send_update(phase="transcribing", progress=40, message="Transcribing speech...")
                transcriber = TranscriptionGenerator()
                source_text = transcriber.run(input_path, output_format="markdown") 
                
                # Check for empty transcription
                if not source_text or len(source_text.strip()) == 0:
                     raise ValueError("Transcription failed or audio was empty.")

                clean_text = source_text
                import re
                clean_text = re.sub(r'\[\d{2}:\d{2}\]', '', clean_text)
                clean_text = clean_text.replace('**', '').strip()
                
                send_update(phase="translating", progress=60, message="Translating text...")
                target_lang = kwargs.get("target_language", "eng_Latn")
                
                # Use specified text model if applicable, or generic default
                # gen_text.translate_text might need updates to accept model_id if we want Qwen/NLLB choice here
                # For now, it uses NLLB-200 by default. 
                # TODO: Pass translate_model to translate_text if it's a text model
                text_gen = gen_text.ArticleGenerator()
                translated_text = text_gen.translate_text(
                    clean_text, 
                    target_lang=target_lang, 
                    source_lang="auto", 
                    model_id=translate_model, 
                    keep_loaded=False,
                    on_ready=on_translation_ready
                )
                
                if translated_text:
                    send_update(phase="synthesizing", progress=80, message=f"Synthesizing speech ({target_lang})...")
                    success = generate_audio(
                        prompt=translated_text,
                        output_path=output_path,
                        duration=0,
                        sampling_rate=24000,
                        model_name="bark", # Pipeline default
                        force=True
                    )
                else:
                    raise ValueError("Translation returned empty text.")

        # Determine conversion type (Standard Fallback)
        elif target_format in ['jpg', 'png', 'webp', 'gif', 'tiff', 'bmp']:
            if translate:
                success = convert_document(
                    input_path,
                    output_path,
                    target_format=target_format,
                    ocr_enabled=ocr_enabled,
                    ocr_model=ocr_model,
                    translate=translate,
                    target_language=target_language,
                    translation_model=translation_model,
                    render_method=render_method,
                    on_translation_ready=on_translation_ready
                )
            else:
                success = convert_image(input_path, output_path)
        elif target_format in ['mp4', 'mov', 'webm', 'avi']:
            success = convert_video(input_path, output_path)
        elif target_format in ['mp3', 'wav', 'aac', 'flac']:
            success = convert_audio(input_path, output_path)
        elif target_format in ['md', 'html', 'pdf', 'docx', 'txt', 'rtf', 'json', 'xhtml']:
            success = convert_document(
                input_path, 
                output_path, 
                ocr_enabled=ocr_enabled, 
                ocr_model=ocr_model,
                translate=translate,
                target_language=target_language,
                translation_model=translation_model,
                render_method=render_method,
                on_translation_ready=on_translation_ready
            )
        else:
            raise ValueError(f"Unsupported target format: {target_format}")

        if success:
            from . import get_relative_path
            msg = "Translation completed successfully" if translate else "Conversion completed successfully"
            send_update(status="complete", phase="complete", progress=100,
                       message=msg, result_path=get_relative_path(output_path))
        else:
            msg = "Translation failed" if translate else "Conversion failed"
            send_update(status="failed", phase="failed", progress=100,
                       message=msg, error="Operation returned False")
    except Exception as e:
        send_update(status="failed", phase="failed", progress=100,
                   message=f"Error: {str(e)}", error=str(e))


