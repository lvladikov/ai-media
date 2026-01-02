"""
AI-Media Server Package.

FastAPI server that wraps the existing ai_media modules for web/desktop access.
Provides REST API, WebSocket for chat, and SSE for real-time monitoring.
"""

from .app import create_app, main

__all__ = ["create_app", "main"]
