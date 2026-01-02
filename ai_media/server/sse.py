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

    # Cache for the currently monitored process to ensure accurate CPU stats
    # cpu_percent requires the same object instance to calculate delta
    monitor_cache = {
        "pid": current_pid,
        "process": process
    }

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
                proc_cpu = 0.0
                proc_ram_gb = 0.0
                proc_vram_gb = 0.0
                
                # Dynamic Process Selection: Monitor active job if any, else server
                from .process_manager import job_processes
                
                target_pid = current_pid
                
                if job_processes:
                    # Monitor the most recently added active job
                    try:
                        latest_job_id = list(job_processes.keys())[-1]
                        job_proc = job_processes[latest_job_id]
                        if job_proc.is_alive():
                            target_pid = job_proc.pid
                    except Exception:
                        pass
                
                # Update cache if target changed
                if target_pid != monitor_cache["pid"]:
                    try:
                        new_proc = psutil.Process(target_pid)
                        new_proc.cpu_percent(interval=None) # Reset counter
                        monitor_cache["pid"] = target_pid
                        monitor_cache["process"] = new_proc
                    except Exception:
                        # Fallback to server if target invalid
                        monitor_cache["pid"] = current_pid
                        monitor_cache["process"] = process
                
                # Get stats from cached process
                proc_obj = monitor_cache["process"]
                proc_pid = monitor_cache["pid"]
                
                if proc_obj:
                    try:
                        # Handle case where process died since cache update
                        if proc_pid != current_pid and not proc_obj.is_running():
                            raise psutil.NoSuchProcess(proc_pid)
                            
                        proc_cpu = proc_obj.cpu_percent(interval=None)
                        proc_mem = proc_obj.memory_info()
                        proc_rss = proc_mem.rss
                        
                        try:
                            import torch
                            # Mac/MPS: Unified Memory - Add Metal usage to System RAM if monitoring self (server)
                            # If monitoring child, we can't easily see its generic Metal usage via torch calls here
                            # unless we rely on RSS. Child process usage is mostly RSS anyway.
                            # But for consistency, if we are monitoring THIS process (server):
                            if proc_pid == current_pid and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                                proc_rss += torch.mps.current_allocated_memory()
                        except Exception:
                            pass

                        proc_ram_gb = round(proc_rss / gb_divisor, 2)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        # Process died or lost access, revert to server for next tick
                        monitor_cache["pid"] = current_pid
                        monitor_cache["process"] = process
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
