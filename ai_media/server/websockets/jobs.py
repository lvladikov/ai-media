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
        await websocket.send_json({
            "type": "job_list",
            "jobs": list(jobs.values())
        })
        
        while True:
            # Keep connection open, wait for messages (ping/pong)
            # We don't expect much input from client, mostly pushing updates
            await websocket.receive_text()
    except WebSocketDisconnect:
        job_manager.disconnect(websocket)
