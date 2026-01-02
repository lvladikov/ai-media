"""Article and code generation background tasks."""

import asyncio
from pathlib import Path

from ..jobs import update_job, is_job_cancelled
from ..cache import model_cache


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
        
        from ai_media.generators.text import ArticleGenerator
        
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
        
        # Check for cancellation before generation
        if is_job_cancelled(job_id):
            print(f"🛑 Article generation cancelled for job {job_id[:8]}...")
            return
        
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
                      message="Article generated successfully", result_path=output_path)
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
    language: str,
):
    """Background task for code generation."""
    try:
        # Check for cancellation before starting
        if is_job_cancelled(job_id):
            return
            
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        from ai_media.generators.text import ArticleGenerator
        
        # Use cached model if same, otherwise load new
        generator = model_cache.get("text", model)
        if generator is None:
            update_job(job_id, status="loading", phase="loading", progress=10, message="Loading language model...")
            generator = ArticleGenerator(model_name=model)
            model_cache.set("text", model, generator)
        else:
            update_job(job_id, status="loading", phase="loading", progress=10, message="Using cached model...")
        
        # Check for cancellation before generation
        if is_job_cancelled(job_id):
            print(f"🛑 Code generation cancelled for job {job_id[:8]}...")
            return
        
        update_job(job_id, status="generating", phase="generating", progress=30, message="Generating code...")
        
        # Generate code using the generator
        code = generator.generate_code(
            prompt=prompt,
            filename=output_path,
        )
        
        # Check for cancellation after generation
        if is_job_cancelled(job_id):
            print(f"🛑 Code generation cancelled for job {job_id[:8]}...")
            return
        
        if code:
            update_job(job_id, status="complete", phase="complete", progress=100,
                      message="Code generated successfully", result_path=output_path)
        else:
            update_job(job_id, status="failed", phase="failed", progress=100,
                      message="Generation failed", error="Code generation returned None")
    except Exception as e:
        if not is_job_cancelled(job_id):
            update_job(job_id, status="failed", phase="failed", progress=100,
                      message=f"Error: {str(e)}", error=str(e))
    # Note: No GPU clear here - model stays cached for reuse
