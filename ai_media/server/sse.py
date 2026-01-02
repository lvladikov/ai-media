"""SSE resource monitoring."""

import asyncio
import platform
import os
from datetime import datetime

import psutil
from fastapi import APIRouter
from starlette.responses import StreamingResponse

router = APIRouter(tags=["Monitoring"])


@router.get("/sse/resources")
async def resource_stream():
    """Server-Sent Events stream for real-time resource monitoring."""
    
    # Track the exact current process (the worker serving this request)
    current_pid = os.getpid()
    try:
        process = psutil.Process(current_pid)
        # Initialize CPU counter
        process.cpu_percent(interval=None)
    except Exception:
        process = None

    async def generate():
        while True:
            try:
                # --- Global Stats ---
                # CPU
                cpu_percent = psutil.cpu_percent(interval=None) # Non-blocking for responsiveness
                
                # RAM (Using 1024**3 to match binary GiB often used on macOS/Linux)
                gb_divisor = 1024**3
                
                ram = psutil.virtual_memory()
                ram_used_gb = round((ram.total - ram.available) / gb_divisor, 2)
                ram_total_gb = round(ram.total / gb_divisor, 2)
                
                # Swap monitoring (Common discrepancy on macOS when RAM is full)
                swap = psutil.swap_memory()
                swap_used_gb = round(swap.used / gb_divisor, 2)
                swap_total_gb = round(swap.total / gb_divisor, 2)
                
                # GPU/VRAM
                vram_used_gb = 0.0
                vram_total_gb = 0.0
                gpu_percent = 0.0
                
                try:
                    import torch
                    if torch.cuda.is_available():
                        vram_used_gb = round(torch.cuda.memory_allocated() / gb_divisor, 2)
                        vram_total_gb = round(torch.cuda.get_device_properties(0).total_memory / gb_divisor, 2)
                        # GPU utilization via nvidia-smi
                        try:
                            import subprocess
                            result = subprocess.run(
                                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=1
                            )
                            if result.returncode == 0:
                                gpu_percent = float(result.stdout.strip().split("\n")[0])
                        except Exception:
                            pass
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        # MPS doesn't have direct memory query, estimate from system
                        vram_total_gb = ram_total_gb * 0.75  # Unified memory estimate
                        try:
                            vram_used_gb = round(torch.mps.current_allocated_memory() / gb_divisor, 2)
                        except Exception:
                            pass
                except ImportError:
                    pass
                
                # --- Process Stats ---
                # --- Process Stats ---
                proc_cpu = 0.0
                proc_ram_gb = 0.0
                proc_vram_gb = 0.0
                proc_pid = current_pid
                
                if process:
                    try:
                        proc_cpu = process.cpu_percent(interval=None)
                        proc_mem = process.memory_info()
                        proc_rss = proc_mem.rss
                        
                        try:
                            import torch
                            # Mac/MPS: Unified Memory - Add Metal usage to System RAM
                            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                                # MPS allocations are often not in RSS, so we sum them for a realistic app footprint.
                                # This helps match Activity Monitor's "Memory" column.
                                proc_rss += torch.mps.current_allocated_memory()
                            
                            # Windows/Linux: Discrete VRAM - Track separately
                            elif torch.cuda.is_available():
                                proc_vram_gb = round(torch.cuda.memory_allocated() / gb_divisor, 2)
                        except Exception:
                            pass

                        proc_ram_gb = round(proc_rss / gb_divisor, 2)
                    except Exception:
                        pass

                data = {
                    "global": {
                        "cpu_percent": cpu_percent,
                        "ram_used_gb": ram_used_gb,
                        "ram_total_gb": ram_total_gb,
                        "swap_used_gb": swap_used_gb,
                        "swap_total_gb": swap_total_gb,
                        "vram_used_gb": vram_used_gb,
                        "vram_total_gb": vram_total_gb,
                        "gpu_percent": gpu_percent,
                    },
                    "process": {
                        "pid": proc_pid,
                        "cpu_percent": proc_cpu,
                        "ram_used_gb": proc_ram_gb,
                        "vram_used_gb": proc_vram_gb
                    },
                    "timestamp": datetime.now().isoformat(),
                }
                
                import json
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(1)
                
            except Exception as e:
                yield f'data: {{"error": "{str(e)}"}}\n\n'
                await asyncio.sleep(5)

    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
