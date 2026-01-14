"""Chat WebSocket handler."""

import uuid
import concurrent.futures
from typing import List, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..state import chat_sessions, chat_manager
from ..cache import model_cache

router = APIRouter()


def generate_chat_response(message: str, history: List[Dict], model: str, precision: str = None, framework: str = None, is_model_cached: bool = False) -> Dict:
    """Generate a chat response using the LLM.
    
    Args:
        message: User's message
        history: Conversation history
        model: Model name
        precision: Precision override
        framework: Framework override
        is_model_cached: If True, model is already loaded (for status reporting)
    
    Returns:
        dict: {"content": response, "reasoning": reasoning} or error message
    """
    try:
        from ai_media.generators.text import ArticleGenerator
        from ai_media.utils.precision import parse_model_precision_framework
        
        # Parse potential model:precision:framework string
        base_model, prec_suffix, fw_suffix = parse_model_precision_framework(model)

        # Prefer explicit args, fallback to suffix
        final_prec = precision or prec_suffix
        final_fw = framework or fw_suffix

        cache_key = f"{base_model}:{final_prec or ''}:{final_fw or ''}".rstrip(":")
        
        # Use cached model if same, otherwise load new
        generator = model_cache.get("text", cache_key)
        if generator is None:
            generator = ArticleGenerator(
                model_name=base_model, 
                precision_force=final_prec,
                framework_force=final_fw,
                bypass_warning=True
            )
            model_cache.set("text", cache_key, generator)
        
        response = generator.chat_single(message, history[:-1])  # Exclude current message from history
        
        # Retrieve extracted reasoning (stored by chat_single)
        reasoning = getattr(generator, 'last_reasoning', None)
        
        return {"content": response, "reasoning": reasoning}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"content": f"Error generating response: {str(e)}", "reasoning": None}


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
    
    # Helper to safe send (ignores disconnects logs but returns status)
    async def safe_send_json(data):
        try:
            await websocket.send_json(data)
            return True
        except RuntimeError:
            return False

    try:
        # Send session ID to client
        await safe_send_json({"type": "session", "session_id": session_id})
        
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "load":
                # Explicitly load model on connect
                model = data.get("model", "default")
                precision = data.get("precision")
                framework = data.get("framework")
                
                from ai_media.utils.precision import parse_model_precision_framework
                base_model, prec_suffix, fw_suffix = parse_model_precision_framework(model)
                
                # Combine overrides
                final_prec = precision or prec_suffix
                final_fw = framework or fw_suffix
                cache_key = f"{base_model}:{final_prec or ''}:{final_fw or ''}".rstrip(":")
                
                # Check cache first
                is_model_cached = model_cache.get("text", cache_key) is not None
                if is_model_cached:
                    await safe_send_json({"type": "status", "status": "ready", "message": "Model ready."})
                    await safe_send_json({"type": "status_clear"}) # Clear loading indicator
                else:
                    await safe_send_json({"type": "status", "status": "loading", "message": f"Loading {base_model}{' ('+final_prec+')' if final_prec else ''}..."})
                    
                    # Pre-load in thread
                    def preload():
                        from ai_media.generators.text import ArticleGenerator
                        
                        # Define callback to stream logs to client
                        def progress_callback(status, progress, message):
                            # Run async send_json on main loop (fire and forget)
                            asyncio.run_coroutine_threadsafe(
                                safe_send_json({"type": "log", "message": message}),
                                loop
                            )
                        
                        generator = ArticleGenerator(
                            model_name=base_model, 
                            precision_force=final_prec,
                            framework_force=final_fw,
                            progress_callback=progress_callback, 
                            bypass_warning=True
                        )
                        # Explicitly trigger heavy load
                        success = generator._load_model()
                        if success:
                            model_cache.set("text", cache_key, generator)
                        return success
                        
                    # internal function `preload` runs in a separate thread to avoid blocking the loop
                    success = await loop.run_in_executor(None, preload)
                        
                    if success:
                        await safe_send_json({"type": "status", "status": "ready", "message": "Model loaded."})
                        await safe_send_json({"type": "status_clear"})
                    # Error status is already sent via progress_callback if it failed

            elif data.get("type") == "message":
                user_message = data.get("content", "")
                model = data.get("model", "default")
                translate_input = data.get("translate_input", False)
                translate_output = data.get("translate_output", False)
                target_language = data.get("target_language", "eng_Latn")
                input_source_language = data.get("input_source_language", "auto")  # Source language for input translation
                input_translation_model = data.get("input_translation_model", "nllb-200-3.3b")  # Model for input translation
                translation_model = data.get("translation_model", "nllb-200-3.3b")  # Model for output translation
                
                # Check for slash commands
                if user_message.startswith("/"):
                    from ai_media.generators.text import ArticleGenerator
                    
                    # Ensure generator is loaded for commands
                    generator = model_cache.get("text", model)
                    if generator is None:
                        generator = ArticleGenerator(model_name=model, bypass_warning=True)
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
                        await safe_send_json({
                            "type": "command_response",
                            "content": response_text,
                            "session_id": session_id,
                        })
                        continue

                # Auto-Translate Input
                original_user_message = user_message
                if translate_input:
                    try:
                        from ai_media.generators.text import ArticleGenerator
                         # Ensure generator is loaded (or reusable static method?)
                         # We need an instance to access the pipeline logic.
                        generator = model_cache.get("text", model)
                        if generator is None:
                            generator = ArticleGenerator(model_name=model, bypass_warning=True)
                            model_cache.set("text", model, generator)
                            
                        # Translate to English using specified source language (or auto-detect for NLLB models)
                        # Keep model loaded for potential follow-up messages in chat
                        source_lang_for_translation = input_source_language if input_source_language != "auto" else "auto"
                        
                        # Notify client
                        await safe_send_json({"type": "status", "status": "processing", "message": f"Translating input ({source_lang_for_translation} -> en)..."})
                        
                        translated_input = generator.translate_text(
                            user_message, 
                            "en", 
                            source_lang=source_lang_for_translation, 
                            model_id=input_translation_model,
                            keep_loaded=True  # Chat: keep loaded for follow-up messages
                        )
                        if translated_input:
                             user_message = translated_input
                             await safe_send_json({"type": "log", "message": f"🌍 Input translated: {original_user_message} -> {user_message}"})
                    except Exception as e:
                        print(f"Server translation error: {e}")

                # Add to history (store English version for LLM context)
                chat_sessions[session_id].append({"role": "user", "content": user_message})
                
                # Send acknowledgment
                # Send acknowledgment with specific status
                from ai_media.utils.precision import parse_model_precision_framework
                base_model, prec_suffix, fw_suffix = parse_model_precision_framework(model)
                final_prec = data.get("precision") or prec_suffix
                final_fw = data.get("framework") or fw_suffix
                cache_key = f"{base_model}:{final_prec or ''}:{final_fw or ''}".rstrip(":")
                
                # Check if model is cached (for status message)
                is_cached = model_cache.get("text", cache_key) is not None
                status_msg = "Thinking..." if is_cached else f"Loading {base_model}..."
                await safe_send_json({"type": "status", "status": "loading" if not is_cached else "processing", "message": status_msg})
                
                # Generate response (in thread pool to not block)
                response = await loop.run_in_executor(
                    None, 
                    generate_chat_response, 
                    user_message,
                    chat_sessions[session_id],
                    base_model, # Pass base model, let args handle overrides
                    final_prec,
                    final_fw,
                    True # Model should be loaded now
                )

                # Response is now a dict {"content": ..., "reasoning": ...}
                parsed = {
                    "content": response["content"],
                    "reasoning": response["reasoning"]
                }
                
                # Add to history (store CLEAN content without reasoning to save context tokens)
                chat_sessions[session_id].append({"role": "assistant", "content": parsed["content"]})
                
                # Translate Output if requested
                final_content = parsed["content"]
                original_content = None # Only set if we translated
                
                if translate_output and target_language:
                     await safe_send_json({"type": "status", "status": "processing", "message": f"Translating output (en -> {target_language})..."})
                     try:
                        from ai_media.generators.text import ArticleGenerator
                        generator = model_cache.get("text", model)
                        if generator:
                             # Translate English Output -> Target
                             translated_output = generator.translate_text(
                                 parsed["content"], 
                                 target_language, 
                                 source_lang="en", 
                                 model_id=translation_model, 
                                 keep_loaded=True,
                                 is_chat=True
                             )
                             
                             if translated_output:
                                 # Extract reasoning from translation if present
                                 trans_parsed = ArticleGenerator.extract_reasoning(translated_output)
                                 
                                 if trans_parsed["content"] != parsed["content"]:
                                     original_content = parsed["content"]
                                     final_content = trans_parsed["content"]
                                     
                                     # Merge translation reasoning into main reasoning field
                                     if trans_parsed["reasoning"]:
                                          if parsed["reasoning"]:
                                              parsed["reasoning"] += f"\n\n---\n**Translation Reasoning ({translation_model}):**\n{trans_parsed['reasoning']}"
                                          else:
                                              parsed["reasoning"] = f"**Translation Reasoning ({translation_model}):**\n{trans_parsed['reasoning']}"
                     except Exception as e:
                          print(f"Output translation error: {e}")

                await safe_send_json({
                    "type": "response",
                    "content": final_content,
                    "original_content": original_content,
                    "reasoning": parsed["reasoning"],
                    "session_id": session_id,
                    # Include translated input if input was auto-translated to English
                    "translated_input": user_message if (translate_input and user_message != original_user_message) else None,
                })
                
            elif data.get("type") == "clear":
                chat_sessions[session_id] = []
                await safe_send_json({"type": "cleared"})
    
                await safe_send_json({"type": "cleared"})
    
    except (WebSocketDisconnect, RuntimeError) as e:
        # Treat RuntimeErrors (like 'WebSocket is not connected') as disconnects to ensure cleanup
        error_msg = str(e)
        if hasattr(e, 'code'): # WebSocketDisconnect has a code
             print(f"🔌 Client disconnected (Session: {session_id})")
        else:
             print(f"🔌 Client disconnected with error: {error_msg} (Session: {session_id})")

        chat_manager.disconnect(session_id)
        if session_id in chat_sessions:
            del chat_sessions[session_id]
            
        # Unload model to free resources
        if model and model_cache.is_loaded("text"):
           print(f"🧹 Unloading model {model} on disconnect...")
           model_cache.unload("text")
           
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

