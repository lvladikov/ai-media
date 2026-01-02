"""Chat WebSocket handler."""

import uuid
import concurrent.futures
from typing import List, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..state import chat_sessions, chat_manager
from ..cache import model_cache

router = APIRouter()


def generate_chat_response(message: str, history: List[Dict], model: str, is_model_cached: bool = False) -> str:
    """Generate a chat response using the LLM.
    
    Args:
        message: User's message
        history: Conversation history
        model: Model name
        is_model_cached: If True, model is already loaded (for status reporting)
    
    Returns:
        str: Response from the LLM or error message
    """
    try:
        from ai_media.generators.text import ArticleGenerator
        
        # Use cached model if same, otherwise load new
        generator = model_cache.get("text", model)
        if generator is None:
            generator = ArticleGenerator(model_name=model)
            model_cache.set("text", model, generator)
        
        response = generator.chat_single(message, history[:-1])  # Exclude current message from history
        # Note: Model stays cached for reuse
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error generating response: {str(e)}"


@router.websocket("/ws/chat")
@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat."""
    session_id = str(uuid.uuid4())
    await chat_manager.connect(session_id, websocket)
    chat_sessions[session_id] = []
    
    # Capture event loop for thread-safe callbacks
    import asyncio
    loop = asyncio.get_running_loop()
    
    try:
        # Send session ID to client
        await websocket.send_json({"type": "session", "session_id": session_id})
        
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "load":
                # Explicitly load model on connect
                model = data.get("model", "default")
                
                # Check cache first
                is_model_cached = model_cache.get("text", model) is not None
                if is_model_cached:
                    await websocket.send_json({"type": "status", "status": "ready", "message": "Model ready."})
                    await websocket.send_json({"type": "status_clear"}) # Clear loading indicator
                else:
                    await websocket.send_json({"type": "status", "status": "loading", "message": "Loading model... (this may take a moment)"})
                    
                    # Pre-load in thread
                    def preload():
                        from ai_media.generators.text import ArticleGenerator
                        
                        # Define callback to stream logs to client
                        def progress_callback(status, progress, message):
                            # Run async send_json on main loop
                            if status == "error":
                                asyncio.run_coroutine_threadsafe(
                                    websocket.send_json({"type": "status", "status": "error", "message": message}),
                                    loop
                                )
                            else:
                                asyncio.run_coroutine_threadsafe(
                                    websocket.send_json({"type": "log", "message": message}),
                                    loop
                                )
                        
                        generator = ArticleGenerator(model_name=model, progress_callback=progress_callback)
                        # Explicitly trigger heavy load
                        success = generator._load_model()
                        if success:
                            model_cache.set("text", model, generator)
                        return success
                        
                    # internal function `preload` runs in a separate thread to avoid blocking the loop
                    success = await loop.run_in_executor(None, preload)
                        
                    if success:
                        await websocket.send_json({"type": "status", "status": "ready", "message": "Model loaded."})
                        await websocket.send_json({"type": "status_clear"})
                    # Error status is already sent via progress_callback if it failed

            elif data.get("type") == "message":
                user_message = data.get("content", "")
                model = data.get("model", "default")
                
                # Check for slash commands
                if user_message.startswith("/"):
                    from ai_media.generators.text import ArticleGenerator
                    
                    # Ensure generator is loaded for commands
                    generator = model_cache.get("text", model)
                    if generator is None:
                        generator = ArticleGenerator(model_name=model)
                        model_cache.set("text", model, generator)
                    
                    cmd_result = generator.process_command(user_message, chat_sessions[session_id])
                    
                    if cmd_result["handled"]:
                        # Send error or success message
                        if cmd_result["error"]:
                            response_text = f"❌ {cmd_result['error']}"
                        else:
                            response_text = cmd_result["message"]
                            if cmd_result["context"]:
                                # Add context to history as system message
                                chat_sessions[session_id].append({"role": "system", "content": cmd_result["context"]})
                        
                        # Send response immediately
                        await websocket.send_json({
                            "type": "command_response",
                            "content": response_text,
                            "session_id": session_id,
                        })
                        continue

                # Add to history
                chat_sessions[session_id].append({"role": "user", "content": user_message})
                
                # Send acknowledgment
                await websocket.send_json({"type": "status", "status": "processing", "message": "Thinking..."})
                
                # Generate response (in thread pool to not block)
                # Generate response (in thread pool to not block)
                response = await loop.run_in_executor(
                    None, 
                    generate_chat_response, 
                    user_message,
                    chat_sessions[session_id],
                    model,
                    True # Model should be loaded now
                )

                # Parse reasoning FIRST
                from ai_media.generators.text import ArticleGenerator
                parsed = ArticleGenerator.extract_reasoning(response)
                
                # Add to history (store CLEAN content without reasoning to save context tokens)
                chat_sessions[session_id].append({"role": "assistant", "content": parsed["content"]})
                
                await websocket.send_json({
                    "type": "response",
                    "content": parsed["content"],
                    "reasoning": parsed["reasoning"],
                    "session_id": session_id,
                })
                
            elif data.get("type") == "clear":
                chat_sessions[session_id] = []
                await websocket.send_json({"type": "cleared"})
    
    except WebSocketDisconnect:
        print(f"🔌 Client disconnected (Session: {session_id})")
        chat_manager.disconnect(session_id)
        if session_id in chat_sessions:
            del chat_sessions[session_id]
            
        # Unload model to free resources
        if model:
           print(f"🧹 Unloading model {model} on disconnect...")
           model_cache.unload("text")
           
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
