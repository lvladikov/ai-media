"""OpenAI-compatible API routes for text and image generation."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel
import time
import uuid
import json
import asyncio

import threading
import re
import torch


from ..cache import model_cache
from ..config import CONFIG # Import config to check verbosity
# Lazy import ArticleGenerator and TEXT_MODELS to avoid heavy startup cost
# from ai_media.generators.text import ArticleGenerator
# from ai_media.models import TEXT_MODELS

router = APIRouter(prefix="/v1", tags=["OpenAI API"])

# Pydantic models (Simplified OpenAI Spec)
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0

class ModelList(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Dict[str, int]

@router.get("/models", response_model=ModelList)
async def list_models():
    """List available text models."""
    from ai_media.models import TEXT_MODELS, IMAGE_MODELS
    models_data = []
    
    # Add text models
    for model_id, hf_id in TEXT_MODELS.items():
        if model_id == "default": continue
        
        models_data.append({
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "ai-media",
        })

    # Add image models
    for model_id, hf_id in IMAGE_MODELS.items():
        if model_id == "default": continue
        
        models_data.append({
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "ai-media",
            "permission": [],
            "root": hf_id,
        })
        
    return ModelList(data=models_data)


def _handle_shutdown(model_name, stream=False):
    """Handle the shutdown command."""
    print("\n🛑 Received 'stop inference server' command via chat. Shutting down...\n")
    
    # Helper to perform delayed shutdown
    def shutdown_later():
        import time, os, signal
        time.sleep(2.0) # Give time for response to be sent cleaning
        os.kill(os.getpid(), signal.SIGINT)

    import threading
    threading.Thread(target=shutdown_later).start()
    
    content = "🛑 **AI-Media Server Stopped**\n\nThe server has been successfully shut down. To continue this chat or start a new one, you will need to restart the server manually."

    # Handle Streaming Response
    if stream:
        from fastapi.responses import StreamingResponse
        async def stop_stream():
            request_id = f"chatcmpl-{uuid.uuid4()}"
            created = int(time.time())
            chunk = {
                "id": request_id, 
                "object": "chat.completion.chunk", 
                "created": created, 
                "model": model_name, 
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.1)
            
            final_chunk = {
                "id": request_id, 
                "object": "chat.completion.chunk", 
                "created": created, 
                "model": model_name, 
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stop_stream(), media_type="text/event-stream")

    # Handle Standard Response
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

def _handle_memory_command(model_name, action, stream=False):
    """Handle memory management commands (unload/flush)."""
    # Use global cache to handle unloading safely
    from ..cache import model_cache
    from ai_media.utils.system import clear_gpu_memory
    import gc
    
    msg = ""
    if action == "unload":
        # Unload via cache (checks existence first)
        if model_cache.is_loaded("text"):
             print(f"Server: Unloading text model...")
             model_cache.unload("text")
             msg = f"Model {model_name} unloaded and memory flushed."
        elif model_cache.is_loaded("image"):
             print(f"Server: Unloading image model...")
             model_cache.unload("image")
             msg = f"Image model unloaded and memory flushed."
        else:
             print("Server: No models were loaded.")
             msg = "No models were loaded."
             
        # Always do a final sweep
        clear_gpu_memory()
        
    elif action == "flush":
        print(f"Server: Flushing memory...")
        # Force unload everything
        model_cache.unload_all()
        clear_gpu_memory()
        msg = "Memory flushed (GC + Cache Empty)."

    if stream:
        created_time = int(time.time())
        request_id = f"chatcmpl-{uuid.uuid4()}"
        
        def make_chunk(content):
            return {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": None
                    }
                ]
            }

        async def stop_stream():
            yield f"data: {json.dumps(make_chunk(msg))}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stop_stream(), media_type="text/event-stream")
    else:
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": msg
                },
                "finish_reason": "stop"
            }],
            "usage": None
        }

async def _handle_image_generation(request: Request, prompt: str, model_name: str, stream: bool = False, response_prefix: str = None):
    """Handle image generation requests."""
    from ai_media.generators.image import ImageGenerator
    from ai_media.server.config import load_config
    
    # 1. Clean prompt of URLs to avoid context pollution (CLIP token warnings)
    original_prompt = prompt
    # Remove http/https URLs
    prompt = re.sub(r'http[s]?://\S+', '', prompt)
    # Remove file paths (common triggers from previous outputs, e.g. /Volumes/... or /api/files/...)
    prompt = re.sub(r'(/Volumes|/Users|/home|/mnt|/api/files)\S+', '', prompt)
    # Clean up extra whitespace
    prompt = " ".join(prompt.split())
    
    if prompt != original_prompt:
        print(f"Server: Cleaned prompt: '{prompt}' (Original had URLs/Paths)")
    else:
        print(f"Server: Generating image with {model_name}...")
    
    request_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())

    def make_chunk(content, finish_reason=None):
        return {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason
            }]
        }

    # Init Generator
    # Init Generator (Cached)
    # Ensure Policy: Unload non-image models first (if not done by ensure_only logic in wrapper, but redundant check safe)
    model_cache.ensure_only("image")
    
    img_generator = model_cache.get("image", model_name)
    if not img_generator:
        print(f"Loading new ImageGenerator for {model_name}")
        from ai_media.generators.image import ImageGenerator
        img_generator = ImageGenerator(model_id=model_name)
        model_cache.set("image", model_name, img_generator)
    else:
        print(f"Using cached ImageGenerator for {model_name}")
    
    # Run Generation (Async wrapper)
    loop = asyncio.get_running_loop()
    
    if stream:
        async def image_stream_generator():
            if response_prefix:
                # Yield the prefix as the "assistant" message content (before the thinking block)
                # This allows the user to see the random prompt immediately
                yield "data: " + json.dumps(make_chunk(response_prefix + "\n\n")) + "\n\n"

            # Use asyncio.Queue to bridge sync callback -> async generator
            progress_queue = asyncio.Queue()
            
            last_msg = [None]
            def on_progress(percent, message):
                # De-duplicate
                if message and message.strip() == last_msg[0]:
                    return
                last_msg[0] = message.strip() if message else None
                
                # Thread-safe put into the async queue
                loop.call_soon_threadsafe(progress_queue.put_nowait, ("PROGRESS", percent, message))
            
            def run_generation():
                try:
                    # Thread-safe generation
                    with img_generator._lock:
                        result = img_generator.generate(
                            prompt=prompt, 
                            progress_callback=on_progress,
                            bypass_warning=True  # Skip resource confirmation prompts
                        )
                    loop.call_soon_threadsafe(progress_queue.put_nowait, ("DONE", result, None))
                except BaseException as e:
                    # Catch KeyboardInterrupt and other system exits
                    loop.call_soon_threadsafe(progress_queue.put_nowait, ("ERROR", None, str(e)))
            
            # Yield Start of Thinking Block
            yield "data: " + json.dumps(make_chunk('<think>\n')) + "\n\n"
            yield "data: " + json.dumps(make_chunk(f'Loading {model_name}...\n\n')) + "\n\n"

            # Force flush and allow Client UI to initialize the Thinking container
            # This is critical when `response_prefix` was used, as the client needs to switch context.
            await asyncio.sleep(0.5)

            # Start generation in background thread AFTER header is sent to prevent progress lag
            loop.run_in_executor(None, run_generation)
            
            # Stream progress updates as they arrive
            result = None
            last_progress = {} # prefix -> last_percent
            
            try:
                while True:
                    item = await progress_queue.get()
                    
                    # Conflation: If the queue has backed up, skip intermediate PROGRESS updates
                    # and jump straight to the latest status. This fixes the "lag" where the UI
                    # is displaying "Loading..." while the server is already "Generating: 50%".
                    while not progress_queue.empty():
                        try:
                            next_item = progress_queue.get_nowait()
                            # Always prioritize DONE/ERROR signals.
                            # If we have [Prog 20%, Prog 21%, DONE], we effectively jump to DONE.
                            # If we have [Prog 20%, Prog 21%], we jump to 21%.
                            item = next_item
                        except asyncio.QueueEmpty:
                            break
                    
                    msg_type = item[0]
                    
                    if msg_type == "DONE":
                        # Auto-complete any remaining 100%s
                        for prefix, percent in last_progress.items():
                            if percent < 100:
                                yield "data: " + json.dumps(make_chunk(f'{prefix}{100}%\n\n')) + "\n\n"
                        result = item[1]
                        break
                    elif msg_type == "ERROR":
                        error_msg = item[2]
                        # Check if error was a cancellation
                        if "cancelled" in str(error_msg).lower():
                             yield "data: " + json.dumps(make_chunk(f'🛑 Generation Cancelled.\n')) + "\n\n"
                        else:
                             yield "data: " + json.dumps(make_chunk(f'❌ Error: {error_msg}\n')) + "\n\n"
                        break
                    elif msg_type == "PROGRESS":
                        percent = item[1]
                        message = item[2]
                        
                        # Detect prefix (e.g. "Loading: 50%" or "Generating: 32%, Eta...")
                        import re
                        match = re.search(r'^(.*?)\d+%', message.strip())
                        if match:
                            prefix = match.group(1)
                            # Remove incorrect auto-complete for interleaved bars
                            last_progress[prefix] = percent
                        
                        yield "data: " + json.dumps(make_chunk(f'{message}\n\n')) + "\n\n"
                        # Force flush buffer by sending a keep-alive comment
                        yield ":\n\n"
                
    
                # Yield End of Thinking Block
                yield "data: " + json.dumps(make_chunk('</think>\n\n')) + "\n\n"
                
                if result and len(result) > 0:
                    abs_path = result[0]
                    
                    # Load Config for consistent URL construction
                    try:
                        config = load_config()
                        server_host = config.get("server", {}).get("host") 
                        server_port = config.get("server", {}).get("port")
                        
                        if not server_host or not server_port:
                            raise ValueError("Missing 'server.host' or 'server.port' in config.json")
                            
                    except Exception as e:
                        yield "data: " + json.dumps(make_chunk(f'⚠️ Config Error: {str(e)}\n')) + "\n\n"
                        server_host = "ERROR_MISSING_CONFIG" 
                        server_port = "ERROR"
    
                    # Construct URL using Config
                    scheme = request.url.scheme
                    clean_path = str(abs_path).lstrip('/')
                    image_url = f"{scheme}://{server_host}:{server_port}/api/files/{clean_path}"
                    
                    # Standard Markdown URL
                    final_content = f"\n![Generated Image]({image_url})"
                    
                    yield "data: " + json.dumps(make_chunk(final_content)) + "\n\n"
                    
                    # Yield Finish
                    yield "data: " + json.dumps(make_chunk(None, finish_reason='stop')) + "\n\n"
                    yield "data: [DONE]\n\n"
                elif not result: # Only yield failure if no result and NO error (e.g. cancelled handled above)
                     # If we had an ERROR map, we broke. If we just have no result but no ERROR (shouldn't happen), assume fail.
                     pass
                     
            except asyncio.CancelledError:
                print(f"Client disconnected. Stopping generation for {model_name}...")
                img_generator.stop()
                raise
            finally:
                # Ensure we stop if we leave the loop for ANY reason (error, done, disconnect)
                # But only if it's still running? stop() is safe to call multiple times.
                img_generator.stop()

        return StreamingResponse(image_stream_generator(), media_type="text/event-stream")

    # Non-Streaming (Standard Logic)
    def _run_gen_safe():
        with img_generator._lock:
             return img_generator.generate(prompt=prompt, bypass_warning=True)

    outputs = await loop.run_in_executor(None, _run_gen_safe)
    
    if outputs and len(outputs) > 0:
        abs_path = outputs[0]
        # Load Config for consistent URL construction
        try:
            config = load_config()
            server_host = config.get("server", {}).get("host")
            server_port = config.get("server", {}).get("port")
            
            if not server_host or not server_port:
                 # In non-streaming, we can just let it fail or default to None which breaks URL
                 # But sticking to "no hardcoded defaults"
                 raise ValueError("Missing 'server.host' or 'server.port' in config.json")
        except Exception as e:
            # Fallback for non-streaming response text
            return {
                "id": request_id, 
                "object": "chat.completion", 
                "created": created_time, 
                "model": model_name, 
                "choices": [{"index": 0, "message": {"role": "assistant", "content": f"Failed to generate image URL: {str(e)}"}}], 
                "usage": None
            }
            
        # Construct URL
        scheme = request.url.scheme
        # Strip leading slash for the route parameter
        clean_path = str(abs_path).lstrip('/')
        image_url = f"{scheme}://{server_host}:{server_port}/api/files/{clean_path}"
        
        response_content = f"![Generated Image]({image_url})"
        if response_prefix:
            response_content = response_prefix + response_content
    else:
        response_content = "Failed to generate image."
        if response_prefix:
            response_content = response_prefix + "\n" + response_content

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created_time,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_content
            },
            "finish_reason": "stop"
        }],
        "usage": None
    }
    
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        created=int(time.time()),
        model=model_name,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_content),
                finish_reason="stop"
            )
        ],
        usage={"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1}
    )

def log_response_with_reasoning(model_name, text):
    """Log response parsing out <think> tags if present."""
    thinking = ""
    response = text
    
    # Extract reasoning if present
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        response = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    if CONFIG["server"].get("verbose_inference"):
        print() # spacing
        if thinking:
            # Grey color for thinking
            print(f"\033[90m💭 Reasoning:\n{thinking}\033[0m")
            
        print(f"📤 Inference Response ({model_name}): {response[:300]}...")
        print() # spacing


async def generate_response_stream(generator, prompt, model_name, request_id, chat_max_tokens, temperature, top_p):
    """Yields Server-Sent Events (SSE) for streaming responses."""
    from transformers import TextIteratorStreamer, StoppingCriteriaList
    from ai_media.generators.text import CancelStopCriteria
    
    import queue
    # Use a Queue to pass the streamer from thread to main loop
    streamer_queue = queue.Queue()
    
    # Track full response for logging
    full_response = []
    
    
    # Args for thread
    # Filter prompt to remove 'stop inference server' exchanges to prevent model confusion hallucinations
    filtered_prompt = []
    stop_triggered = False
    stop_triggered = False
    for m in prompt:
         content_lower = m.content.strip().lower()
         
         # Filter commands from history so model doesn't see them
         if content_lower in ["stop inference server", "unload model", "flush memory"]:
             stop_triggered = True
             continue
             
         if stop_triggered and m.role == "assistant":
             # Skip the response to the stop command too
             stop_triggered = False
             continue
         filtered_prompt.append(m)

    thread_args = {
        'prompt_data': filtered_prompt, # Raw messages now
        'max_new_tokens': chat_max_tokens,
        'temperature': temperature,
        'top_p': top_p,
    }
    
    created_time = int(time.time())
    
    # Helper to yield chunks
    def make_chunk(content):
        return {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": None
                }
            ]
        }

    # Run generation in a separate thread
    def thread_target():
        with generator._lock:
            # Load model
            generator._load_model()
            
            if generator.pipeline and generator.tokenizer:
                # Create streamer NOW with valid tokenizer
                local_streamer = TextIteratorStreamer(generator.tokenizer, skip_prompt=True, skip_special_tokens=True)
                streamer_queue.put(local_streamer)
                
                # Prepare Prompt
                conversation = [{"role": m.role, "content": m.content} for m in thread_args['prompt_data']]
                
                # INJECT SYSTEM INSTRUCTION: Suppress proactive tool usage for standard chat.
                # We specifically want to prevent "hello" triggering "read_file".
                if conversation and conversation[0]['role'] == 'system':
                    conversation[0]['content'] += (
                        "\n\nIMPORTANT: Do not proactively use tools to read files or context (like 'read_currently_open_file') "
                        "unless the user explicitly asks you to examine specific files or the codebase. "
                        "For greetings and general questions, simply reply in text."
                    )
                
                real_prompt = generator.tokenizer.apply_chat_template(
                    conversation, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                gen_kwargs = dict(
                    text_inputs=real_prompt,
                    max_new_tokens=thread_args['max_new_tokens'],
                    do_sample=thread_args['temperature'] > 0,
                    temperature=thread_args['temperature'],
                    top_p=thread_args['top_p'],
                    return_full_text=False, # Optimization: Only return new tokens
                    streamer=local_streamer,
                    stopping_criteria=StoppingCriteriaList([CancelStopCriteria(generator)])
                )
                
                # Reset cancellation state before starting
                generator.is_cancelled = False
                
                # IMPORTANT: Run inference in no_grad() context to avoid OOM
                # This prevents PyTorch from building the computation graph
                with torch.no_grad():
                    generator.pipeline(**gen_kwargs)
            else:
                streamer_queue.put(None) # Signal failure

            
    thread = threading.Thread(target=thread_target)
    
    # 1. Start Thread
    thread.start()
    
    # 2. Capture Loading Logs (Hack: Capture global stdout/stderr filtered by thread? 
    # Or just yield a static message? User wants actual logs.)
    # Since we can't easily capture only one thread's C-level stdout (from tqdm),
    # we'll emit a "Thinking" block with a static loading message primarily, 
    # or try to capture python print statements.
    
    # Send <think> start
    # Determine device for logging
    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = "CUDA"
    elif torch.backends.mps.is_available():
        device_name = "MPS"

    # Dynamic log buffer for real-time loading feedback
    # We populate this from the thread via generator.progress_callback
    logs_to_show = []
    
    # Callback to receive TQDM/Loading updates from ArticleGenerator
    last_msg = [None] # Use list for closure mutability
    def loading_callback(status, percent, message):
         if message and message.strip():
             # Strip \r (carriage return) which TQDM uses to overwrite lines in terminal
             # We want them as distinct lines in the Thinking block
             clean_msg = message.replace("\r", "").strip()
             if clean_msg and clean_msg != last_msg[0]:
                 logs_to_show.append(clean_msg)
                 last_msg[0] = clean_msg

    # 0. Set callback BEFORE starting thread to catch immediate "Checking resources" etc.
    generator.progress_callback = loading_callback
    
    # Run generation in a separate thread
    def thread_target():
        with generator._lock:
            # Re-ensure callback (for safety)
            generator.progress_callback = loading_callback
            try:
                # Load model
                generator._load_model()
                
                if generator.pipeline and generator.tokenizer:
                    # Create streamer NOW with valid tokenizer
                    local_streamer = TextIteratorStreamer(generator.tokenizer, skip_prompt=True, skip_special_tokens=True)
                    streamer_queue.put(local_streamer)
                    
                    # Prepare Prompt
                    conversation = [{"role": m.role, "content": m.content} for m in thread_args['prompt_data']]
                    
                    # INJECT SYSTEM INSTRUCTION
                    if conversation and conversation[0]['role'] == 'system':
                        conversation[0]['content'] += (
                            "\n\nIMPORTANT: Do not proactively use tools to read files or context (like 'read_currently_open_file') "
                            "unless the user explicitly asks you to examine specific files or the codebase. "
                            "For greetings and general questions, simply reply in text."
                        )
                    
                    real_prompt = generator.tokenizer.apply_chat_template(
                        conversation, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    
                    gen_kwargs = dict(
                        text_inputs=real_prompt,
                        max_new_tokens=thread_args['max_new_tokens'],
                        do_sample=thread_args['temperature'] > 0,
                        temperature=thread_args['temperature'],
                        top_p=thread_args['top_p'],
                        return_full_text=False, # Optimization: Only return new tokens
                        streamer=local_streamer,
                        stopping_criteria=StoppingCriteriaList([CancelStopCriteria(generator)])
                    )
                    
                    # Reset cancellation state before starting
                    generator.is_cancelled = False
                    
                    # IMPORTANT: Run inference in no_grad() context to avoid OOM
                    with torch.no_grad():
                        generator.pipeline(**gen_kwargs)
                else:
                    streamer_queue.put(None) # Signal failure
            except Exception as e:
                 print(f"Generate Thread Error: {e}")
                 import traceback
                 traceback.print_exc()
                 streamer_queue.put(None)
            finally:
                 # Detach callback
                 generator.progress_callback = None

            
    thread = threading.Thread(target=thread_target)
    
    # 1. Start Thread
    thread.start()
    
    # Send <think> start and initial message
    yield f"data: {json.dumps(make_chunk('<think>\n'))}\n\n"
    yield f"data: {json.dumps(make_chunk(f'Server: Loading model {model_name}...\n\n'))}\n\n"
    yield f"data: {json.dumps(make_chunk(f'Moving to device ({device_name})...\n\n'))}\n\n"
    
    # Loop while thread is alive AND streamer has no tokens yet
    last_progress = {} # prefix -> last_percent
    
    def complete_progress(current_msg=None):
        """Helper to yield 100% for any previous prefixes if they finished abruptly."""
        import re
        prefix = None
        if current_msg:
            # Detect prefix (e.g. "Loading: 50%" -> "Loading: ")
            match = re.search(r'^(.*?)\d+%$', current_msg.strip())
            if match:
                prefix = match.group(1)
        
        updates = []
        for old_prefix, old_percent in list(last_progress.items()):
            if (prefix is None or old_prefix != prefix) and old_percent < 100:
                updates.append(f"data: {json.dumps(make_chunk(f'{old_prefix}100%' + '\n\n'))}\n\n")
                last_progress[old_prefix] = 100
        
        # Update current prefix
        if prefix:
             last_progress[prefix] = int(re.search(r'(\d+)%$', current_msg).group(1))
        return updates

    while thread.is_alive() and not generator.pipeline:
        while logs_to_show:
            msg = logs_to_show.pop(0)
            for completion in complete_progress(msg):
                yield completion
            yield f"data: {json.dumps(make_chunk(msg + '\n\n'))}\n\n"
        await asyncio.sleep(0.1) 
        
    # Wait for streamer to be available
    streamer = None
    while thread.is_alive():
        try:
            streamer = streamer_queue.get_nowait()
            break
        except queue.Empty:
            if logs_to_show:
                msg = logs_to_show.pop(0)
                for completion in complete_progress(msg):
                    yield completion
                yield f"data: {json.dumps(make_chunk(msg + '\n'))}\n\n"
            await asyncio.sleep(0.5)

    # Final completion before closing thinking block
    for completion in complete_progress():
        yield completion
            
    if not streamer:
        yield f"data: {json.dumps(make_chunk('Failed to load model or tokenizer init failed.\n'))}\n\n"
        return

    # Signal end of thinking

        
    # If we exited loop, model is loaded (or thread died)
    if not generator.pipeline:
        yield f"data: {json.dumps(make_chunk('Failed to load model.\n'))}\n\n"
    else:
        yield f"data: {json.dumps(make_chunk('Model loaded.\n</think>\n'))}\n\n"

    try:
        for new_text in streamer:
            if not new_text: continue
            
            full_response.append(new_text)
            yield f"data: {json.dumps(make_chunk(new_text))}\n\n"
            await asyncio.sleep(0)
            
        # End of stream
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        

        
        # Log final response
        log_response_with_reasoning(model_name, ''.join(full_response))

        
    except Exception as e:
        print(f"Error during streaming: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        # Signal generator to stop (interrupts the thread)
        generator.stop()
        generator._cleanup_memory()

# Helper for non-streaming generation
def run_generation_sync(generator, prompt, kwargs):
    with generator._lock:
        return generator.pipeline(prompt, **kwargs)

@router.post("/chat/completions")
async def chat_completions(raw_request: Request, body: ChatCompletionRequest):
    """Handle chat completion request."""
    # Alias body to request for backward compat where we used 'request' var name
    request = body 
    try:
        from ai_media.generators.text import ArticleGenerator
        from ai_media.models import TEXT_MODELS, IMAGE_MODELS
        
        model_name = request.model
        
        # Get last message content
        last_msg = request.messages[-1].content if request.messages else ""
        
        # Log incoming request (only if verbose)
        if CONFIG["server"].get("verbose_inference"):
            # Cyan color for request, extra spacing
            print(f"\n\033[96m📥 Inference Request ({model_name}): {last_msg[:200]}...\033[0m\n")

        # Check for text commands
        last_msg_content = ""
        if request.messages:
            last_msg_content = request.messages[-1].content.strip().lower()

        # 1. STOP COMMAND
        if last_msg_content == "stop inference server":
            return _handle_shutdown(model_name, stream=request.stream)
            
        # 2. UNLOAD COMMAND
        if last_msg_content == "unload model":
            print(f"User requested Unload Model.")
            return _handle_memory_command(model_name, "unload", request.stream)

        # 3. FLUSH MEMORY COMMAND
        if last_msg_content == "flush memory":
            print(f"User requested Flush Memory.")
            return _handle_memory_command(model_name, "flush", request.stream)

        # 4. RANDOM PROMPT COMMAND
        from ai_media.utils.prompts import is_random_prompt_trigger, get_random_prompt
        response_prefix = None
        
        if is_random_prompt_trigger(last_msg_content):
            # Determine prompt type based on model
            is_image = model_name in IMAGE_MODELS or model_name in IMAGE_MODELS.values()
            prompt_type = "image" if is_image else "code"
            random_prompt = get_random_prompt(prompt_type)
            print(f"Random Prompt ({prompt_type}): {random_prompt}")
            
            if is_image:
                 # IF IMAGE MODEL: Don't return text, but PROCEED to generate image with this prompt
                 last_msg = random_prompt
                 response_prefix = f"🎲 **Random Prompt**\n\n{random_prompt}\n\n"
                 # Fallthrough to IMAGE_GENERATION_HANDLING below...
            else:
                # IF TEXT/CODE: Return the random prompt (not execute it, just show it)
                if request.stream:
    
                    async def random_stream():
                        request_id = f"chatcmpl-{uuid.uuid4()}"
                        created = int(time.time())
                        content = f"🎲 **Random Prompt**\n\n{random_prompt}"
                        chunk = {"id": request_id, "object": "chat.completion.chunk", "created": created, 
                                 "model": model_name, "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}
                        yield f"data: {json.dumps(chunk)}\n\n"
                        final_chunk = {"id": request_id, "object": "chat.completion.chunk", "created": created,
                                       "model": model_name, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                    return StreamingResponse(random_stream(), media_type="text/event-stream")
                else:
                    return {
                        "id": f"chatcmpl-{uuid.uuid4()}", "object": "chat.completion", "created": int(time.time()),
                        "model": model_name,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": f"🎲 **Random Prompt**\n\n{random_prompt}"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    }

        # --- IMAGE GENERATION HANDLING ---
        if model_name in IMAGE_MODELS or model_name in IMAGE_MODELS.values():
             # Enforce single model policy (Unload text models if any)
             model_cache.ensure_only("image")
             
             # Intercept Auto-Title Generation prompts (from clients like Continue)
             if "reply with a title" in last_msg.lower() and "chat" in last_msg.lower():
                 print("Server: Detected Auto-Title Prompt on Image Model. Returning static title.")
                 content = "Image Chat"
                 
                 chunk = {"id": f"chatcmpl-{uuid.uuid4()}", "object": "chat.completion.chunk", "created": int(time.time()), 
                          "model": model_name, "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}
                 final_chunk = chunk.copy()
                 final_chunk["choices"][0]["delta"] = {}
                 final_chunk["choices"][0]["finish_reason"] = "stop"
                 
                 if request.stream:

                     async def title_stream():
                         yield f"data: {json.dumps(chunk)}\n\n"
                         yield f"data: {json.dumps(final_chunk)}\n\n"
                         yield "data: [DONE]\n\n"
                     return StreamingResponse(title_stream(), media_type="text/event-stream")
                 else:
                     return {
                         "id": f"chatcmpl-{uuid.uuid4()}", "object": "chat.completion", "created": int(time.time()),
                         "model": model_name,
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                         "usage": None
                     }

             return await _handle_image_generation(raw_request, last_msg, model_name, stream=request.stream, response_prefix=response_prefix)

        # --- TEXT GENERATION HANDLING (Existing Logic) ---
        # Enforce single model policy (Unload image models if any)
        model_cache.ensure_only("text")
        
        # 1. Get/Load Generator
        # Check cache first
        generator = model_cache.get("text", model_name)
        if generator is None:
            # Check if model exists in definitions
            from ai_media.models import TEXT_MODELS
            if model_name not in TEXT_MODELS and request.model not in TEXT_MODELS.values():
                 # Try default fallback to 'default' if no match?
                 # ideally we strictly match available models
                 pass

            print(f"Server: Loading model {model_name} for API request...")
            # Initialize generator (loads model)
            # bypass_warning=True because API requests are usually automated/intentional
            generator = ArticleGenerator(model_name=model_name, bypass_warning=True)
            
            # Load model INSIDE the stream generator for streaming requests
            # For non-streaming, we must load it here
            loaded = True
            if not request.stream:
                # Load in executor to avoid blocking
                loop = asyncio.get_running_loop()
                loaded = await loop.run_in_executor(None, generator._load_model)
            
                if not loaded:
                    raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}")
            
            model_cache.set("text", model_name, generator)
        
        # 2. Prepare Messages
        if not request.stream:
             # Check pipeline only if not streaming (streaming loads it lazily)
             if not generator.pipeline or not generator.tokenizer:
                  raise HTTPException(status_code=500, detail="Model pipeline not initialized")
        else:
             # For streaming, we need tokenizer to be loaded to apply template
             pass 

        # We need the tokenizer to update the progress stream with tokens? No
        # We need the tokenizer to format the prompt.
        
        # Simple fix: We pass the raw messages to the generator stream function, 
        # and let IT handle loading -> tokenizing -> generating.
        
        request_id = f"chatcmpl-{uuid.uuid4()}"
        chat_max_tokens = request.max_tokens if request.max_tokens else 2048
        
        # 3. STREAMING Response
        if request.stream:
            return StreamingResponse(
                generate_response_stream(
                    generator, 
                    request.messages,  # Pass raw messages
                    model_name, 
                    request_id, 
                    chat_max_tokens, 
                    request.temperature, 
                    request.top_p
                ),
                media_type="text/event-stream"
            )

        # 4. NON-STREAMING Response (Async wrapper)
        # Ensure we have tokenizer now
        if not generator.tokenizer:
             generator._load_model()
             
        # Convert pydantic models to dicts
        conversation = [{"role": m.role, "content": m.content} for m in request.messages]
        prompt = generator.tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )

        kwargs = dict(
            max_new_tokens=chat_max_tokens,
            do_sample=request.temperature > 0,
            temperature=request.temperature,
            top_p=request.top_p,
            return_full_text=False
        )
        
        # Run in thread pool to prevent blocking
        loop = asyncio.get_running_loop()
        outputs = await loop.run_in_executor(None, run_generation_sync, generator, prompt, kwargs)
            
        response_text = outputs[0]['generated_text'].strip()
        log_response_with_reasoning(model_name, response_text)
        
        # Cleanup
        generator._cleanup_memory()
        
        # Format Response
        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": -1, 
                "completion_tokens": -1,
                "total_tokens": -1
            }
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
