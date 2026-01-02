"""Global state for the server."""

import asyncio
from typing import Dict, Any, List
from fastapi import WebSocket


# --- Job State ---
jobs: Dict[str, Dict[str, Any]] = {}


# --- Chat State ---
chat_sessions: Dict[str, List[Dict[str, str]]] = {}  # session_id -> conversation history


# --- Event Loop Reference ---
MAIN_LOOP: asyncio.AbstractEventLoop = None


# --- Connection Managers ---

class JobConnectionManager:
    """Manage WebSocket connections for job updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


class ChatConnectionManager:
    """Manage WebSocket connections for chat."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
    
    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)


# Global manager instances
job_manager = JobConnectionManager()
chat_manager = ChatConnectionManager()
