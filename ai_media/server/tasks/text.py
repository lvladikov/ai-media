"""Article and code generation background tasks."""

import datetime
import os
from pathlib import Path
from multiprocessing import Queue

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
    max_images: int = 0,
    progress_queue: Queue = None,
):
    """Background task for article generation. Runs in child process."""
    
    def send_update(**kwargs):
        """Send progress update to parent via queue."""
        if progress_queue:
            try:
                progress_queue.put({"job_id": job_id, **kwargs})
            except Exception:
                pass
    
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Define progress callback wrapper that uses IPC
        def progress_callback(status, progress, message):
            send_update(status=status, progress=progress, message=message, phase=status)
        
        # Create generator fresh (no caching in multiprocessing)
        generator = ArticleGenerator(model_name=model, progress_callback=progress_callback)
        
        # Update job with resolved model name  
        send_update(model=generator.model_name)
        
        # Load model (progress_callback handles updates)
        generator._load_model()
        
        # Use progress=None to trigger pulsating animation in UI
        send_update(
            status="generating", 
            phase="generating", 
            progress=None, 
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
            max_images=max_images,
        )
        
        if success:
            send_update(status="complete", phase="complete", progress=100,
                       message="Article generated successfully", result_path=output_path,
                       reasoning=generator.last_reasoning)
        else:
            send_update(status="failed", phase="failed", progress=100,
                       message="Generation failed", error="Article generation returned False")
    except Exception as e:
        send_update(status="failed", phase="failed", progress=100,
                   message=f"Error: {str(e)}", error=str(e))
    finally:
        # Clear GPU memory
        try:
            from ai_media.utils.system import clear_gpu_memory
            clear_gpu_memory()
        except Exception:
            pass


def run_code_generation(
    job_id: str,
    prompt: str,
    output_path: str,
    model: str,
    progress_queue: Queue = None,
):
    """Background task for code generation. Runs in child process."""
    
    def send_update(**kwargs):
        """Send progress update to parent via queue."""
        if progress_queue:
            try:
                progress_queue.put({"job_id": job_id, **kwargs})
            except Exception:
                pass
    
    try:
        # Ensure output directory exists
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        from ai_media.generators.text import ArticleGenerator
        
        # Define progress callback wrapper
        def progress_callback(status, progress, message):
            send_update(status=status, progress=progress, message=message, phase=status)
        
        # Create generator fresh (no caching in multiprocessing)
        generator = ArticleGenerator(model_name=model, progress_callback=progress_callback)
        
        # Update job with resolved model name
        send_update(model=generator.model_name)
        
        # Load model
        generator._load_model()
        
        send_update(
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
        
        # Detect what was generated - count files in output folder
        generated_files = []
        output_dir = Path(output_path)
        
        if output_dir.is_dir():
            # Collect all files recursively
            for root, dirs, files in os.walk(output_path):
                for f in files:
                    generated_files.append(os.path.join(root, f))
            
            # If directory is empty, check if a file was created as a sibling with extension
            if not generated_files:
                parent = output_dir.parent
                name = output_dir.name
                for f in parent.iterdir():
                    if f.is_file() and f.name.startswith(name) and f.name != name:
                        generated_files.append(str(f))
                        output_path = str(f)
                        try:
                            os.rmdir(output_dir)
                        except:
                            pass
                        break

        elif output_dir.is_file():
            generated_files.append(output_path)
        else:
            parent = output_dir.parent
            name = output_dir.name
            for f in parent.iterdir():
                if f.is_file() and f.name.startswith(name):
                    generated_files.append(str(f))
                    output_path = str(f)
                    break
        
        is_multi_file = len(generated_files) > 1
        
        if generated_files:
            if not is_multi_file and len(generated_files) == 1:
                output_path = generated_files[0]
            
            file_list = []
            if is_multi_file:
                base_dir = Path(output_path)
                for f in generated_files:
                    try:
                        rel_path = Path(f).relative_to(base_dir)
                        file_list.append(str(rel_path))
                    except ValueError:
                        file_list.append(Path(f).name)
            
            send_update(status="complete", phase="complete", progress=100,
                       message=f"Generated {len(generated_files)} file(s)" if is_multi_file else "Code generated successfully",
                       result_path=output_path, is_multi_file=is_multi_file, generated_files=file_list if is_multi_file else [],
                       reasoning=generator.last_reasoning)
        else:
            send_update(status="failed", phase="failed", progress=100,
                       message="Generation failed", error="No files were generated")
                       
    except Exception as e:
        send_update(status="failed", phase="failed", progress=100,
                   message=f"Error: {str(e)}", error=str(e))
                       
    finally:
        # Clear GPU memory
        try:
            from ai_media.utils.system import clear_gpu_memory
            clear_gpu_memory()
        except Exception:
            pass


