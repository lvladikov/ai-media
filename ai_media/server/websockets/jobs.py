"""Job updates WebSocket handler."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..state import jobs, job_manager

router = APIRouter()


@router.websocket("/ws/jobs")
async def websocket_jobs(websocket: WebSocket):
    """WebSocket endpoint for real-time job updates."""
    await job_manager.connect(websocket)
    try:
        # Send initial list of jobs on connect
        try:
            await websocket.send_json({
                "type": "job_list",
                "jobs": list(jobs.values())
            })
        except RuntimeError:
            # Client disconnected immediately
            pass
        
        while True:
            # Keep connection open, wait for messages (ping/pong)
            # We don't expect much input from client, mostly pushing updates
            await websocket.receive_text()
            
    except (WebSocketDisconnect, RuntimeError):
        print("🔌 Jobs connection closed - initiating cleanup...")
        job_manager.disconnect(websocket)
        
        # When the main jobs socket disconnects (Client Quit/Refresh),
        # we act as a "Global Stop" and kill all running jobs and unload models.
        # This prevents orphaned processes when Electron quits.
        from ..process_manager import terminate_all_processes
        terminate_all_processes()
        
        from ..cache import model_cache
        model_cache.unload_all() # Free up RAM too
        
        # Force GC
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
