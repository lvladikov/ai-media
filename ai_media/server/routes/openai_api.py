"""OpenAI-compatible API routes for text and image generation."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel
import time
import uuid
import json
import os
import asyncio

import threading
import re
import torch
import random
from pathlib import Path

PROMPTS_FILE = Path(__file__).parent.parent.parent / "data" / "prompts.json"


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

# OpenAI Responses API models (newer API used by Continue in agent mode)
class ResponsesInputMessage(BaseModel):
    role: str
    content: str

class ResponsesRequest(BaseModel):
    """OpenAI Responses API request format."""
    model: str
    input: Union[str, List[ResponsesInputMessage]]  # Can be string or message array
    instructions: Optional[str] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_output_tokens: Optional[int] = None
    stream: Optional[bool] = False
    # Additional fields Continue might send
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None
    reasoning: Optional[Dict[str, Any]] = None


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


async def _handle_image_generation(request: Request, prompt: str, model_name: str, stream: bool = False, response_prefix: str = None, framework: str = None, precision: str = None):
    """Handle image generation requests."""
    # Apply Transformers v5 patch before importing diffusers-dependent modules
    from ai_media.utils.transformers_patch import ensure_patch_applied
    ensure_patch_applied()
    
    from ai_media.generators.image import ImageGenerator
    from ai_media.server.config import load_config
    from ai_media.utils.parsers import extract_prompt_parameters
    
    # 0. Extract Parameters from Prompt
    prompt, extracted_params = extract_prompt_parameters(prompt)
    if extracted_params:
        print(f"Server: Extracted parameters: {extracted_params}")

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

    # Init Generator (Cached)
    # Ensure Policy: Unload non-image models first (if not done by ensure_only logic in wrapper, but redundant check safe)
    model_cache.ensure_only("image")
    
    img_generator = model_cache.get("image", model_name)
    if not img_generator:
        print(f"Loading new ImageGenerator for {model_name}")

        if not framework:
            from ai_media.server.config import CONFIG
            if "generation" in CONFIG:
                framework = CONFIG["generation"].get("ml_framework")

        use_mlx_arg = "mlx" if framework == "mlx" else None
        img_generator = ImageGenerator(model_id=model_name, use_mlx=use_mlx_arg, precision=precision)
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
                            bypass_warning=True,  # Skip resource confirmation prompts
                            **extracted_params
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
            
            def complete_progress(current_msg=None):
                """Helper to yield 100% for any previous prefixes if they finished abruptly."""
                import re
                prefix = None
                if current_msg:
                    # Detect ONLY simple progress bar lines like:
                    #   "Loading: 50%" or "Generating: 75%"
                    # NOT informational lines like:
                    #   "⏱️ Estimated Resources: ... | GPU: 32.5%"
                    stripped = current_msg.strip()
                    if '|' not in stripped:  # Skip multi-value info lines
                        match = re.match(r'^([A-Za-z\.\s]+[:\s]+)\d+%', stripped)
                        if match:
                            prefix = match.group(1)
                
                updates = []
                for old_prefix, old_percent in list(last_progress.items()):
                    # If we switched to a new prefix, OR we finished (current_msg=None)
                    if (prefix is None or old_prefix != prefix) and old_percent < 100:
                        updates.append(f"data: {json.dumps(make_chunk(f'{old_prefix}100%' + '\n\n'))}\n\n")
                        last_progress[old_prefix] = 100
                
                # Update current prefix tracking
                if prefix:
                     match = re.search(r'(\d+)%', current_msg)
                     if match:
                        last_progress[prefix] = int(match.group(1))
                return updates


            try:
                while True:
                    item = await progress_queue.get()
                    
                    # Conflation: If the queue has backed up, skip intermediate PROGRESS updates
                    # BUT only if they are for the same "phase" (same prefix).
                    # Never skip headers, errors, or DONE signals.
                    while not progress_queue.empty():
                        try:
                            next_item = progress_queue.get_nowait()
                            
                            # Stop conflating if we see a non-progress item
                            if next_item[0] != "PROGRESS":
                                # Put it back? No, we can't put back easily.
                                # Strategy: Peek? Queue doesn't support peek.
                                # Better Strategy: Only skip if next_item is also PROGRESS AND matches context?
                                # Simple safe fix: 
                                # If next is DONE/ERROR, take it and discard current `item` (jump to end/fail).
                                # If next is PROGRESS, take it (jump ahead).
                                item = next_item
                                if item[0] != "PROGRESS":
                                    break
                            else:
                                item = next_item
                        except asyncio.QueueEmpty:
                            break
                    
                    msg_type = item[0]
                    
                    if msg_type == "DONE":
                        # Auto-complete any remaining 100%s BEFORE yielding result
                        for update in complete_progress(None):
                            yield update
                        result = item[1]
                        break
                    elif msg_type == "ERROR":
                        # Auto-complete pending bars BEFORE showing error
                        for update in complete_progress(None):
                            yield update
                            
                        error_msg = item[2]
                        if "cancelled" in str(error_msg).lower():
                             yield "data: " + json.dumps(make_chunk(f'🛑 Generation Cancelled.\n')) + "\n\n"
                        else:
                             yield "data: " + json.dumps(make_chunk(f'❌ Error: {error_msg}\n')) + "\n\n"
                        break
                    elif msg_type == "PROGRESS":
                        percent = item[1]
                        message = item[2]
                        
                        # Yield completions for previous phases first
                        for update in complete_progress(message):
                            yield update
                        
                        yield "data: " + json.dumps(make_chunk(f'{message}\n\n')) + "\n\n"
                        yield ":\n\n" # keep-alive
                
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
    
                    # Construct URL using Config - use just filename for cleaner URLs
                    scheme = request.url.scheme
                    filename = os.path.basename(abs_path)
                    image_url = f"{scheme}://{server_host}:{server_port}/api/files/{filename}"
                    
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
            
        # Construct URL - use just filename for cleaner URLs
        scheme = request.url.scheme
        filename = os.path.basename(abs_path)
        image_url = f"{scheme}://{server_host}:{server_port}/api/files/{filename}"
        
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


async def generate_response_stream(generator, prompt, model_name, request_id, chat_max_tokens, temperature, top_p, response_prefix=None):
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

    # Determine device for logging
    device_name = "CPU"
    if getattr(generator, 'use_mlx', False):
         device_name = "MLX"
    elif torch.cuda.is_available():
        device_name = "CUDA"
    elif torch.backends.mps.is_available():
        device_name = "MPS"

    # Dynamic log buffer for real-time loading feedback
    # We populate this from the thread via generator.progress_callback
    logs_to_show = []
    
    # Callback to receive TQDM/Loading updates from ArticleGenerator
    last_msg = [None] # Use list for closure mutability
    last_percent_logged = {} # prefix -> percent
    
    def loading_callback(status, percent, message):
         if message and message.strip():
             # Strip \r (carriage return) which TQDM uses to overwrite lines in terminal
             clean_msg = message.replace("\r", "").strip()
             
             # Throttling Logic for Percentages (Reduce spam in Chat UI)
             # content e.g. "Generating: 15%"
             import re
             match = re.search(r'^([A-Za-z\.\s]+[:\s]+)(\d+)%$', clean_msg)
             if match:
                 prefix = match.group(1)
                 curr_pct = int(match.group(2))
                 
                 last_pct = last_percent_logged.get(prefix, -1)
                 
                 # Only show if:
                 # 1. First time seeing this prefix
                 # 2. Significant jump (>20%)
                 # 3. Completion (100%)
                 # 4. Error status
                 if (last_pct == -1 or 
                     (curr_pct - last_pct) >= 20 or 
                     curr_pct == 100 or 
                     status == "error"):
                     
                     if clean_msg != last_msg[0]:
                        logs_to_show.append(clean_msg)
                        last_msg[0] = clean_msg
                        last_percent_logged[prefix] = curr_pct
             else:
                 # Non-percentage message (e.g. text logs), show always if new
                 if clean_msg != last_msg[0]:
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
                
                # Check for MLX Generator first
                if getattr(generator, 'use_mlx', False) and getattr(generator, 'mlx_generator', None):
                    # MLX Path
                    messages = [{"role": m.role, "content": m.content} for m in thread_args['prompt_data']]
                    
                     # INJECT SYSTEM INSTRUCTION
                    if messages and messages[0]['role'] == 'system':
                        messages[0]['content'] += (
                            "\n\nIMPORTANT: Do not proactively use tools to read files or context (like 'read_currently_open_file') "
                            "unless the user explicitly asks you to examine specific files or the codebase. "
                            "For greetings and general questions, simply reply in text."
                        )

                    iterator = generator.mlx_generator.chat(
                        messages=messages,
                        max_tokens=thread_args['max_new_tokens'],
                        temperature=thread_args['temperature'],
                        top_p=thread_args['top_p'],
                        stream=True
                    )
                    
                    # Reset cancellation state
                    generator.is_cancelled = False
                    
                    streamer_queue.put(iterator)
                
                elif generator.pipeline and generator.tokenizer:
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
    # IMPORTANT: Send response_prefix FIRST if we have one (e.g., random prompt info)
    # This ensures clients like Continue see the expanded prompt
    if response_prefix:
        yield f"data: {json.dumps(make_chunk(response_prefix))}\n\n"
    yield f"data: {json.dumps(make_chunk('<think>\n'))}\n\n"
    
    # Check if model is already loaded to avoid spamming "Loading..." logs on every message
    # We check if:
    # 1. Generator has a loaded model (pipeline or mlx_generator)
    # 2. The loaded model's name matches the requested model_name
    is_already_loaded = False
    try:
        if generator.model_name == model_name:
             if getattr(generator, 'use_mlx', False) and getattr(generator, 'mlx_generator', None):
                 is_already_loaded = True
             elif generator.pipeline:
                 is_already_loaded = True
    except:
        pass

    if not is_already_loaded:
        yield f"data: {json.dumps(make_chunk(f'Server: Loading model {model_name}...\n\n'))}\n\n"
        yield f"data: {json.dumps(make_chunk(f'Moving to device ({device_name})...\n\n'))}\n\n"

    # Loop while thread is alive AND streamer has no tokens yet
    last_progress = {} # prefix -> last_percent

    def complete_progress(current_msg=None):
        """Helper to yield 100% for any previous prefixes if they finished abruptly."""
        import re
        prefix = None
        if current_msg:
            # Detect ONLY simple progress bar lines like:
            #   "Loading: 50%" or "Generating: 75%"
            # NOT informational lines like:
            #   "⏱️ Estimated Resources: ... | GPU: 32.5%"
            # Pattern: Must be a SHORT word/phrase followed by colon/space and ONLY a percentage.
            #          e.g. "Loading: 50%", "Generating... 75%"
            #          If the line contains multiple values (e.g. "| CPU: 13%"), skip it.
            stripped = current_msg.strip()
            if '|' not in stripped:  # Skip multi-value info lines
                match = re.match(r'^([A-Za-z\.\s]+[:\s]+)\d+%$', stripped)
                if match:
                    prefix = match.group(1)
        
        updates = []
        for old_prefix, old_percent in list(last_progress.items()):
            if (prefix is None or old_prefix != prefix) and old_percent < 100:
                updates.append(f"data: {json.dumps(make_chunk(f'{old_prefix}100%' + '\n\n'))}\n\n")
                last_progress[old_prefix] = 100
        
        # Update current prefix
        if prefix:
             match = re.search(r'(\d+)%$', current_msg)
             if match:
                 last_progress[prefix] = int(match.group(1))
        return updates


    while thread.is_alive() and not (generator.pipeline or (getattr(generator, 'use_mlx', False) and getattr(generator, 'mlx_generator', None))):
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
            
    # CRITICAL FIX for MLX: The thread might finish BEFORE we grab the streamer (since chat() returns generator instantly)
    # So if streamer is None, check queue ONE LAST TIME
    if not streamer:
        try:
            streamer = streamer_queue.get_nowait()
        except queue.Empty:
            pass

    if not streamer:
        yield f"data: {json.dumps(make_chunk('Failed to load model or tokenizer init failed.\n'))}\n\n"
        return

    # If we exited loop, model is loaded (or thread died)
    if not generator.pipeline and not (getattr(generator, 'use_mlx', False) and getattr(generator, 'mlx_generator', None)):
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

        # 4. RANDOM PROMPT COMMAND (Unified)
        # Use shared trigger utility from single source of truth
        from ai_media.utils.prompts import is_random_prompt_trigger, RANDOM_PROMPT_TRIGGERS
        
        response_prefix = None
        if is_random_prompt_trigger(last_msg_content):
            try:
                # Determine prompt type based on model
                is_image = model_name in IMAGE_MODELS or model_name in IMAGE_MODELS.values()
                
                # Load prompts from single source of truth
                if PROMPTS_FILE.exists():
                     with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        
                     prompt_text = ""
                     
                     if is_image:
                         # Image models use 'image' category
                         if "image" in data:
                             prompt_text = random.choice(data["image"])
                     else:
                         # Text models get unified 'code' + 'article' pool
                         pool = []
                         if "code" in data: pool.extend(data["code"])
                         if "article" in data: pool.extend(data["article"])
                         if pool:
                             prompt_text = random.choice(pool)
                             
                     if prompt_text:
                         print(f"🎲 Random Prompt Selected ({'Image' if is_image else 'Unified'}): {prompt_text}")
                         
                         if is_image:
                              # For image models, replace prompt and add prefix to response
                              last_msg = prompt_text
                              response_prefix = f"🎲 **Random Prompt**\n\n{prompt_text}\n\n"
                         else:
                              # For text models, replace user input and add prefix to response
                              request.messages[-1].content = prompt_text
                              print(f"Server: Replaced user input with random prompt.")
                              response_prefix = f"🎲 **Random Prompt**\n\n{prompt_text}\n\n"
                     else:
                        print("⚠️ Prompts pool empty.")
            except Exception as e:
                print(f"❌ Error getting random prompt: {e}")


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

             # Extract optional parameters for image generation
             framework = None
             precision = None
             try:
                 request_json = await raw_request.json()
                 framework = request_json.get("framework")
                 precision = request_json.get("precision")
             except:
                 pass

             return await _handle_image_generation(raw_request, last_msg, model_name, stream=request.stream, 
                                                   response_prefix=response_prefix, 
                                                   framework=framework, precision=precision)

        # --- TEXT GENERATION HANDLING (Existing Logic) ---
        # Enforce single model policy (Unload image models if any)
        model_cache.ensure_only("text")
        
        # Parse model:precision:framework syntax (e.g., llama-3.1-8b:int4:mlx)
        from ai_media.utils.precision import parse_model_precision_framework
        base_model_name, precision_suffix, framework_suffix = parse_model_precision_framework(model_name)
        
        # Fallback to Global Config keys if not specified in model suffix
        if not framework_suffix:
             framework_suffix = CONFIG.get("generation", {}).get("ml_framework")
        
        if not precision_suffix:
             precision_suffix = CONFIG.get("generation", {}).get("precision_force")
        
        # Use base model name for cache key and model lookup
        cache_key = f"{base_model_name}:{precision_suffix or ''}:{framework_suffix or ''}"
        cache_key = cache_key.rstrip(":") # Clean up trailing colons
        
        # 1. Get/Load Generator
        # Check cache first
        generator = model_cache.get("text", cache_key)
        if generator is None:
            # Check if model exists in definitions
            from ai_media.models import TEXT_MODELS
            if base_model_name not in TEXT_MODELS and request.model not in TEXT_MODELS.values():
                 # Try default fallback to 'default' if no match?
                 # ideally we strictly match available models
                 pass

            print(f"Server: Loading model {base_model_name} for API request{'(precision: ' + precision_suffix + ')' if precision_suffix else ''}...")
            # Initialize generator (loads model)
            # bypass_warning=True because API requests are usually automated/intentional
            generator = ArticleGenerator(
                model_name=base_model_name, 
                bypass_warning=True,
                precision_force=precision_suffix,  # Pass parsed precision
                framework_force=framework_suffix    # Pass parsed framework
            )
            
            # Load model INSIDE the stream generator for streaming requests
            # For non-streaming, we must load it here
            loaded = True
            if not request.stream:
                # Load in executor to avoid blocking
                loop = asyncio.get_running_loop()
                loaded = await loop.run_in_executor(None, generator._load_model)
            
                if not loaded:
                    raise HTTPException(status_code=500, detail=f"Failed to load model {base_model_name}")
            
            model_cache.set("text", cache_key, generator)
        
        # 2. Prepare Messages
        if not request.stream:
             # Check for EITHER PyTorch pipeline OR MLX generator
             has_pytorch = generator.pipeline and generator.tokenizer
             has_mlx = getattr(generator, 'use_mlx', False) and getattr(generator, 'mlx_generator', None)
             
             if not has_pytorch and not has_mlx:
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
                    request.top_p,
                    response_prefix=response_prefix
                ),
                media_type="text/event-stream"
            )

        # 4. NON-STREAMING Response
        loop = asyncio.get_running_loop()
        conversation = [{"role": m.role, "content": m.content} for m in request.messages]
        
        # Check if using MLX
        if getattr(generator, 'use_mlx', False) and getattr(generator, 'mlx_generator', None):
            # MLX path: use mlx_generator.chat()
            def run_mlx_generation():
                tokens = []
                for token in generator.mlx_generator.chat(
                    messages=conversation,
                    max_tokens=chat_max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stream=True
                ):
                    tokens.append(token)
                return "".join(tokens)
            
            response_text = await loop.run_in_executor(None, run_mlx_generation)
        else:
            # PyTorch path: ensure tokenizer loaded
            if not generator.tokenizer:
                 generator._load_model()
                 
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


async def generate_responses_stream(
    generator, 
    messages: List[dict], 
    model_name: str, 
    response_id: str,
    max_tokens: int,
    temperature: float,
    top_p: float
):
    """
    Generate SSE stream in OpenAI Responses API format.
    Uses event: prefixes required by Continue in agent mode.
    """
    import time
    
    created_time = int(time.time())
    output_index = 0
    content_index = 0
    
    # Create the response object structure
    def make_response_event(event_type: str, data: dict):
        """Format as proper SSE with event: prefix"""
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    
    # 1. response.created event
    yield make_response_event("response.created", {
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created_time,
            "status": "in_progress",
            "model": model_name,
            "output": []
        }
    })
    
    # 2. response.in_progress event
    yield make_response_event("response.in_progress", {
        "type": "response.in_progress",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created_time,
            "status": "in_progress",
            "model": model_name,
            "output": []
        }
    })
    
    # 3. response.output_item.added - announcing new text output
    yield make_response_event("response.output_item.added", {
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": {
            "type": "message",
            "id": f"msg_{uuid.uuid4()}",
            "status": "in_progress",
            "role": "assistant",
            "content": []
        }
    })
    
    # 4. response.content_part.added 
    yield make_response_event("response.content_part.added", {
        "type": "response.content_part.added",
        "item_id": f"msg_{uuid.uuid4()}",
        "output_index": output_index,
        "content_index": content_index,
        "part": {
            "type": "output_text",
            "text": "",
            "annotations": []
        }
    })
    
    # Load model and generate (reuse existing logic)
    full_response = []
    
    try:
        # Check if model is already loaded (same pattern as chat_completions)
        # We only need to load if:
        # 1. No model is loaded at all (no pipeline/mlx_generator)
        # 2. OR the loaded model doesn't match the requested model
        is_already_loaded = False
        try:
            # Check if generator has a model loaded and it matches
            if getattr(generator, 'use_mlx', False) and getattr(generator, 'mlx_generator', None):
                is_already_loaded = True
            elif generator.pipeline:
                is_already_loaded = True
        except:
            pass
        
        needs_loading = not is_already_loaded
        
        if needs_loading:

            # Load model in background thread while sending heartbeat events
            import threading
            import queue as queue_module
            
            load_error = [None]
            load_done = threading.Event()
            
            def load_model_thread():
                try:
                    generator._load_model()
                except Exception as e:
                    load_error[0] = e
                finally:
                    load_done.set()
            
            loader_thread = threading.Thread(target=load_model_thread)
            loader_thread.start()
            
            # Send heartbeat events during model loading
            heartbeat_count = 0
            loading_started = False
            while not load_done.wait(timeout=2.0):
                heartbeat_count += 1
                # Send loading status as think content to show in Continue's "Thinking" section
                if not loading_started:
                    # First heartbeat - open think tag and show loading message
                    loading_msg = f"<think>\nLoading model {model_name}...\n"
                    loading_started = True
                else:
                    # Subsequent heartbeats - show progress dots
                    loading_msg = "." if heartbeat_count % 3 != 0 else ".\n"
                
                yield make_response_event("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "item_id": f"msg_{response_id}",
                    "output_index": output_index,
                    "content_index": content_index,
                    "delta": loading_msg
                })
                await asyncio.sleep(0)
            
            # Close the think tag if we opened it
            if loading_started:
                yield make_response_event("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "item_id": f"msg_{response_id}",
                    "output_index": output_index,
                    "content_index": content_index,
                    "delta": "\nModel loaded.\n</think>\n"
                })
                await asyncio.sleep(0)

            
            loader_thread.join()
            
            if load_error[0]:
                raise load_error[0]
        
        # Get streamer
        if getattr(generator, 'use_mlx', False) and getattr(generator, 'mlx_generator', None):
            streamer = generator.mlx_generator.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True
            )

        else:
            # PyTorch path
            conversation = messages
            prompt = generator.tokenizer.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=True
            )
            inputs = generator.tokenizer(prompt, return_tensors="pt").to(generator.device)
            
            from transformers import TextIteratorStreamer
            streamer_obj = TextIteratorStreamer(generator.tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            import threading
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p,
                streamer=streamer_obj
            )
            thread = threading.Thread(target=generator.pipeline.model.generate, kwargs=gen_kwargs)
            thread.start()
            streamer = streamer_obj
        
        # 5. Stream response.output_text.delta events for each token
        for new_text in streamer:
            if not new_text:
                continue
            
            full_response.append(new_text)
            
            yield make_response_event("response.output_text.delta", {
                "type": "response.output_text.delta",
                "item_id": f"msg_{response_id}",
                "output_index": output_index,
                "content_index": content_index,
                "delta": new_text
            })
            await asyncio.sleep(0)
        
        final_text = "".join(full_response)
        
        # 6. response.output_text.done
        yield make_response_event("response.output_text.done", {
            "type": "response.output_text.done",
            "item_id": f"msg_{response_id}",
            "output_index": output_index,
            "content_index": content_index,
            "text": final_text
        })
        
        # 7. response.output_item.done
        yield make_response_event("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": {
                "type": "message",
                "id": f"msg_{response_id}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": final_text}]
            }
        })
        
        # 8. response.completed
        yield make_response_event("response.completed", {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created_time,
                "status": "completed",
                "model": model_name,
                "output": [{
                    "type": "message",
                    "id": f"msg_{response_id}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": final_text}]
                }],
                "usage": {
                    "input_tokens": -1,
                    "output_tokens": len(full_response),
                    "total_tokens": -1
                }
            }
        })
        
        # Log response
        log_response_with_reasoning(model_name, final_text)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield make_response_event("response.failed", {
            "type": "response.failed",
            "response": {
                "id": response_id,
                "status": "failed",
                "error": {"message": str(e)}
            }
        })
    finally:
        generator._cleanup_memory()


@router.post("/responses")
async def responses_api(raw_request: Request, body: ResponsesRequest):
    """
    OpenAI Responses API endpoint (used by Continue in agent mode).
    
    Implements proper Responses API streaming format with event: prefixes.
    """
    from ai_media.models import TEXT_MODELS, get_model_id
    from ai_media.generators.text import ArticleGenerator
    
    if CONFIG["server"].get("verbose_inference"):
        print(f"\n📥 Responses API Request ({body.model})")
    
    # Convert 'input' to messages format
    if isinstance(body.input, str):
        messages = [{"role": "user", "content": body.input}]
    else:
        messages = [{"role": m.role, "content": m.content} for m in body.input]
    
    # If instructions provided, prepend as system message
    if body.instructions:
        messages.insert(0, {"role": "system", "content": body.instructions})
    
    # Get or create generator
    model_name = body.model
    base_model_name = model_name.split("-mlx")[0].split("-int")[0].split("-fp")[0]
    cache_key = base_model_name
    
    generator = model_cache.get("text", cache_key)
    if generator is None:
        generator = ArticleGenerator(
            model_name=base_model_name,
            bypass_warning=True,
            precision_force=None,
            framework_force=None
        )
        model_cache.set("text", cache_key, generator)
    
    # Generate response ID
    response_id = f"resp_{uuid.uuid4()}"
    max_tokens = body.max_output_tokens or 2048
    
    # Log first message
    first_user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "")
    if CONFIG["server"].get("verbose_inference"):
        print(f"\n📥 Inference Request ({model_name}): {first_user_msg[:50]}...")
    
    # Always stream for Responses API
    return StreamingResponse(
        generate_responses_stream(
            generator,
            messages,
            model_name,
            response_id,
            max_tokens,
            body.temperature,
            body.top_p
        ),
        media_type="text/event-stream"
    )
