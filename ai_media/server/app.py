"""FastAPI application setup."""

import asyncio
import signal
import gc
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import CONFIG
from . import state


def force_cleanup():
    """Force cleanup of all processes and memory."""
    # Clear GPU memory
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    
    # Force garbage collection
    gc.collect()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    state.MAIN_LOOP = asyncio.get_running_loop()
    print("✅ AI-Media Server Ready")
    
    yield
    
    # Shutdown
    print("🛑 AI-Media Server shutting down...")
    
    # Terminate all child processes first
    from .process_manager import terminate_all_processes
    terminate_all_processes()
    
    force_cleanup()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AI-Media API",
        description="API for AI-powered media generation, transformation, and conversion.",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # CORS for React dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register routes
    from .routes import system, files, generate, text, transform, convert, upscale
    from .websockets import jobs as jobs_ws, chat as chat_ws, code as code_ws
    from . import sse
    from . import jobs as jobs_api
    
    app.include_router(system.router)
    app.include_router(files.router)
    app.include_router(generate.router)
    app.include_router(text.router)
    app.include_router(transform.router)
    app.include_router(convert.router)
    app.include_router(upscale.router)
    app.include_router(sse.router)
    app.include_router(jobs_api.router)
    app.include_router(jobs_ws.router)
    app.include_router(chat_ws.router)
    app.include_router(code_ws.router)
    
    return app


def main(host: str = None, port: int = None, reload: bool = None, reload_excludes: list = None, reload_dirs: list = None):
    """Run the server."""
    import uvicorn
    import sys
    
    # If no arguments provided, use argparse
    if host is None or port is None or reload is None:
        import argparse
        parser = argparse.ArgumentParser(description="AI-Media Web Server")
        parser.add_argument("--host", default=CONFIG["server"]["host"], help="Host to bind to")
        parser.add_argument("--port", type=int, default=CONFIG["server"]["port"], help="Port to bind to")
        parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
        args, unknown = parser.parse_known_args()
        
        host = host if host is not None else args.host
        port = port if port is not None else args.port
        reload = reload if reload is not None else args.reload
    
    def handle_shutdown(signum, frame):
        """Handle shutdown signal."""
        print("\n🛑 Shutdown signal received...")
        
        # Terminate all child processes first
        from .process_manager import terminate_all_processes
        terminate_all_processes()
        
        force_cleanup()
        
        # Force exit after cleanup
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Configure reload parameters
    reload_kwargs = {}
    if reload:
        reload_kwargs["reload"] = True
        if reload_excludes:
            reload_kwargs["reload_excludes"] = reload_excludes
            print(f"🔍 Server reload exclusions active: {len(reload_excludes)} patterns configured")
            if len(reload_excludes) > 0:
                # Show first 3 patterns as confirmation
                print(f"   (Excludes sample: {', '.join(reload_excludes[:3])}...)")
        if reload_dirs:
            reload_kwargs["reload_dirs"] = reload_dirs
            print(f"📡 Code changes watching: {', '.join(reload_dirs)}")
    
    print(f"🌐 Starting AI-Media Server on http://{host}:{port}")
    print(f"📚 API docs: http://{host}:{port}/docs")
    sys.stdout.flush()
    
    # Create custom log config to suppress health checks
    log_config = uvicorn.config.LOGGING_CONFIG.copy()
    log_config["loggers"]["uvicorn.access"] = {
        "handlers": ["access"],
        "level": "INFO",
        "propagate": False
    }
    
    # Set all uvicorn status/access logs to WARNING to keep the console clean
    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi"]:
        if logger_name in log_config["loggers"]:
            log_config["loggers"][logger_name]["level"] = "WARNING"
        else:
            # Create if missing (e.g. uvicorn root might not be in default config sometimes)
            log_config["loggers"][logger_name] = {"level": "WARNING"}

    uvicorn.run(
        "ai_media.server.app:create_app",
        host=host,
        port=port,
        factory=True,
        log_config=log_config,
        **reload_kwargs
    )


# Create app instance for uvicorn when running with factory=True
app = create_app()
