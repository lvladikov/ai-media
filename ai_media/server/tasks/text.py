"""Article and code generation background tasks."""

import asyncio
import datetime
from pathlib import Path

from ..jobs import update_job, is_job_cancelled
from ..cache import model_cache
from ...generators.text import ArticleGenerator


def run_article_generation(
    job_id: str,
    topic: str,
    output_path: str,
    model: str,
    fmt: str,
    length: str,
    online: bool,
    research_iterations: int,
    event_loop: asyncio.AbstractEventLoop = None,
):
    """Background task for article generation."""
    try:
        # Check for cancellation before starting
        if is_job_cancelled(job_id):
            return
            
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Define progress callback wrapper
        def progress_callback(status, progress, message):
            update_job(job_id, event_loop=event_loop, status=status, progress=progress, message=message, phase=status)

        # Use cached model if same, otherwise load new
        generator = model_cache.get("text", model)
        if generator is None:
            # Init with callback
            generator = ArticleGenerator(model_name=model, progress_callback=progress_callback)
            model_cache.set("text", model, generator)
        else:
            # Update existing generator's callback
            generator.progress_callback = progress_callback
            progress_callback("loading", 10, "Using cached model...")

        # Update job with resolved model name for display
        update_job(job_id, event_loop=event_loop, model=generator.model_name)
        
        # Check for cancellation before generation
        if is_job_cancelled(job_id):
            print(f"🛑 Article generation cancelled for job {job_id[:8]}...")
            return

        # Explicitly load model first to avoid timer counting during loading
        # The generator's _load_model() uses progress_callback("loading", ...)
        generator._load_model()
        
        # Check cancellation again after expensive load
        if is_job_cancelled(job_id):
            return

        update_job(
            job_id, 
            event_loop=event_loop, 
            status="generating", 
            phase="generating", 
            progress=30, 
            message="Generating article...",
            generation_started_at=datetime.datetime.utcnow().isoformat()
        )
        
        success = generator.generate_article(
            topic=topic,
            output_file=output_path,
            format=fmt,
            online=online,
            research_iter=research_iterations if online else 0,
            length=length,
        )
        
        # Check for cancellation after generation
        if is_job_cancelled(job_id):
            print(f"🛑 Article generation cancelled for job {job_id[:8]}...")
            return
        
        if success:
            update_job(job_id, event_loop=event_loop, status="complete", phase="complete", progress=100,
                      message="Article generated successfully", result_path=output_path,
                      reasoning=generator.last_reasoning)
        else:
            update_job(job_id, event_loop=event_loop, status="failed", phase="failed", progress=100,
                      message="Generation failed", error="Article generation returned False")
    except Exception as e:
        if not is_job_cancelled(job_id):
            update_job(job_id, event_loop=event_loop, status="failed", phase="failed", progress=100,
                      message=f"Error: {str(e)}", error=str(e))
    # Note: No GPU clear here - model stays cached for reuse


def run_code_generation(
    job_id: str,
    prompt: str,
    output_path: str,
    model: str,
    event_loop: asyncio.AbstractEventLoop = None,
):
    """Background task for code generation."""
    try:
        # Check for cancellation before starting
        if is_job_cancelled(job_id):
            return
            
        # Ensure output directory exists
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        from ai_media.generators.text import ArticleGenerator
        
        # Define progress callback wrapper (same as article generation)
        def progress_callback(status, progress, message):
            update_job(job_id, event_loop=event_loop, status=status, progress=progress, message=message, phase=status)

        # Use cached model if same, otherwise load new
        generator = model_cache.get("text", model)
        if generator is None:
            # Init with callback so loading progress is shown
            generator = ArticleGenerator(model_name=model, progress_callback=progress_callback)
            model_cache.set("text", model, generator)
        else:
            # Update existing generator's callback
            generator.progress_callback = progress_callback
            progress_callback("loading", 10, "Using cached model...")

        # Update job with resolved model name for display
        update_job(job_id, event_loop=event_loop, model=generator.model_name)
        
        # Check for cancellation before expensive loading
        if is_job_cancelled(job_id):
            print(f"🛑 Code generation cancelled for job {job_id[:8]} before loading...")
            generator.is_cancelled = True
            return

        # Explicitly load model first
        generator._load_model()
        
        # Check for cancellation before generation (again)
        if is_job_cancelled(job_id):
            print(f"🛑 Code generation cancelled for job {job_id[:8]}...")
            # CRITICAL: If we cancelled DURING load, the model is now in memory but the job is dead.
            # We must explicitly unload it here, because the cache might have already tried to unload 
            # the *previous* state (empty) or missed it.
            if hasattr(generator, '_unload_model'):
                print("🧹 Cleaning up model loaded during cancellation...")
                generator._unload_model()
            return
        
        update_job(
            job_id, 
            event_loop=event_loop, 
            status="generating", 
            phase="generating", 
            progress=30, 
            message="Generating code...",
            generation_started_at=datetime.datetime.utcnow().isoformat()
        )
        
        # Generate code using the generator (output_path is the folder)
        generator.generate_code(
            prompt=prompt,
            output_file=output_path,
        )
        
        # Check for cancellation after generation
        if is_job_cancelled(job_id):
            print(f"🛑 Code generation cancelled for job {job_id[:8]}...")
            return
        
        # Detect what was generated - count files in output folder
        import os
        generated_files = []
        output_dir = Path(output_path)
        
        if output_dir.is_dir():
            # Collect all files recursively
            for root, dirs, files in os.walk(output_path):
                for f in files:
                    generated_files.append(os.path.join(root, f))
            
            # If directory is empty, check if a file was created as a sibling with extension
            # (This happens when generate_code appends extension to output_path)
            if not generated_files:
                parent = output_dir.parent
                name = output_dir.name
                for f in parent.iterdir():
                    # Check for file starting with name, but explicitly NOT the directory itself
                    if f.is_file() and f.name.startswith(name) and f.name != name:
                        generated_files.append(str(f))
                        output_path = str(f)
                        # Detect if we should treat this as single file output and remove the empty dir
                        try:
                            os.rmdir(output_dir)
                        except:
                            pass
                        break

        elif output_dir.is_file():
            # Single file was written directly with that name
            generated_files.append(output_path)
        else:
            # Check if a file was created with an extension added
            parent = output_dir.parent
            name = output_dir.name
            for f in parent.iterdir():
                if f.is_file() and f.name.startswith(name):
                    generated_files.append(str(f))
                    output_path = str(f)  # Update to actual file path
                    break
        
        is_multi_file = len(generated_files) > 1
        
        if generated_files:
            # For single file, use the actual file path
            if not is_multi_file and len(generated_files) == 1:
                output_path = generated_files[0]
            
            # For multi-file, extract relative paths from output_path for display
            file_list = []
            if is_multi_file:
                base_dir = Path(output_path)
                for f in generated_files:
                    try:
                        rel_path = Path(f).relative_to(base_dir)
                        file_list.append(str(rel_path))
                    except ValueError:
                        file_list.append(Path(f).name)
            
            update_job(job_id, event_loop=event_loop, status="complete", phase="complete", progress=100,
                      message=f"Generated {len(generated_files)} file(s)" if is_multi_file else "Code generated successfully",
                      result_path=output_path, is_multi_file=is_multi_file, generated_files=file_list if is_multi_file else [],
                      reasoning=generator.last_reasoning)
        else:
            update_job(job_id, event_loop=event_loop, status="failed", phase="failed", progress=100,
                      message="Generation failed", error="No files were generated")
                      
        # UNLOAD MODEL IMMEDIATELY to save resources as per user request
        if hasattr(generator, 'stop'):
            generator.stop()

        # Explicitly call the aggressive unload method we just wrote
        if hasattr(generator, '_unload_model'):
            generator._unload_model()
            
        # Also clean up from cache system so it doesn't think it's still there
        model_cache.unload("text")

        print("🔌 Code Generator disconnected - aggressive cleanup complete")
            
    except Exception as e:
        if not is_job_cancelled(job_id):
            update_job(job_id, event_loop=event_loop, status="failed", phase="failed", progress=100,
                      message=f"Error: {str(e)}", error=str(e))
    # Note: No GPU clear here - model stays cached for reuse
