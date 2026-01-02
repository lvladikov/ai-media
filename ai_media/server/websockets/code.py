"""Code generation WebSocket handler for cleanup on disconnect."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..cache import model_cache

router = APIRouter()


@router.websocket("/ws/code")
async def websocket_code(websocket: WebSocket):
    """WebSocket endpoint for code generation cleanup.
    
    When client disconnects (navigates away), we unload the text model.
    """
    await websocket.accept()
    
    try:
        # Keep connection alive - just wait for disconnect
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if model_cache.is_loaded("text"):
            print("🔌 Code Generator disconnected - unloading text model...")
            model_cache.unload("text")
        
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
