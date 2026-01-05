"""SSE resource monitoring."""

import asyncio
import platform
import os
import json
import time
import subprocess
from datetime import datetime

import psutil
from fastapi import APIRouter
from starlette.responses import StreamingResponse

router = APIRouter(tags=["Monitoring"])

# Global cache for Windows GPU metrics to avoid overhead
win_gpu_cache = {
    "data": None,
    "last_updated": 0
}

win_vram_cache = {
    "data": None,
    "last_updated": 0
}

def get_windows_vram_metrics():
    global win_vram_cache
    now = time.time()
    # Cache for 2.0 seconds (VRAM changes less rapidly than load)
    if win_vram_cache["data"] is not None and (now - win_vram_cache["last_updated"] < 2.0):
        return win_vram_cache["data"]
    
    try:
        # Query Dedicated GPU Memory usage per process
        script = "Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty CounterSamples | Select-Object InstanceName, CookedValue | ConvertTo-Json"
        res = subprocess.run(["powershell", "-Command", script], capture_output=True, text=True, timeout=3.0)
        
        vram_data = []
        if res.stdout.strip():
            try:
                vram_data = json.loads(res.stdout)
                if not isinstance(vram_data, list):
                    vram_data = [vram_data]
            except Exception:
                pass
        
        win_vram_cache["data"] = vram_data
        win_vram_cache["last_updated"] = now
        return vram_data
    except Exception:
        return []


def get_windows_gpu_metrics():
    global win_gpu_cache
    now = time.time()
    # Cache for 1.5 seconds
    if win_gpu_cache["data"] is not None and (now - win_gpu_cache["last_updated"] < 1.5):
        return win_gpu_cache["data"]
    
    try:
        # Query 3D Engine utilization (matches Task Manager GPU column)
        # This is the most accurate representation of AI/Rendering activity on Windows
        script = "Get-Counter '\\GPU Engine(*3d*)\\Utilization Percentage' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty CounterSamples | Select-Object InstanceName, CookedValue | ConvertTo-Json"
        res = subprocess.run(["powershell", "-Command", script], capture_output=True, text=True, timeout=3.0)
        
        gpu_data = []
        if res.stdout.strip():
            try:
                gpu_data = json.loads(res.stdout)
                if not isinstance(gpu_data, list):
                    gpu_data = [gpu_data]
            except Exception:
                pass
        
        win_gpu_cache["data"] = gpu_data
        win_gpu_cache["last_updated"] = now
        return gpu_data
    except Exception:
        return []



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
                # GPU/VRAM
                vram_used_gb = 0.0
                vram_total_gb = 0.0
                gpu_percent = 0.0
                
                try:
                    import torch
                    if torch.cuda.is_available():
                        # Global VRAM via nvidia-smi is more representative of system state
                        try:
                            result = subprocess.run(
                                ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=1
                            )
                            if result.returncode == 0:
                                parts = result.stdout.strip().split("\n")[0].split(",")
                                if len(parts) >= 4:
                                    vram_used_gb = round(float(parts[0]) / 1024, 2)
                                    vram_total_gb = round(float(parts[1]) / 1024, 2)
                                    # Fallback/Primary choice for Windows: PowerShell 3D Engine utilization
                                    if platform.system() == "Windows":
                                        win_metrics = get_windows_gpu_metrics()
                                        if win_metrics:
                                            # Sum across all engines/PIDs to get total system load
                                            # (Win counters return per instance, we want the highest single value or sum depending on context)
                                            # To match Task Manager's global total, we use the sum of 3D engine usage across engines
                                            # but capped at 100% per GPU.
                                            total_load = sum(float(m.get('CookedValue', 0)) for m in win_metrics if 'phys_0' in m.get('InstanceName', ''))
                                            gpu_percent = min(100.0, total_load)
                                        else:
                                            # nvidia-smi fallback
                                            gpu_percent = max(float(parts[2]), float(parts[3]))
                                    else:
                                        # Linux/typical: Use max of GPU core and memory utilization
                                        gpu_percent = max(float(parts[2]), float(parts[3]))
                        except Exception:
                             # Fallback to torch for if nvidia-smi fails
                             vram_used_gb = round(torch.cuda.memory_allocated() / gb_divisor, 2)
                             vram_total_gb = round(torch.cuda.get_device_properties(0).total_memory / gb_divisor, 2)
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        # MPS doesn't have direct total VRAM query, estimate from system
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
                proc_gpu_percent = 0.0
                
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
                            if proc_pid == current_pid and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                                proc_rss += torch.mps.current_allocated_memory()
                            
                            # Windows/CUDA: Process-specific VRAM
                            if torch.cuda.is_available():
                                try:
                                    import subprocess
                                    # Collect PIDs of the target process and its children
                                    try:
                                        p_pids = {proc_pid}
                                        try:
                                            p_pids |= {c.pid for c in psutil.Process(proc_pid).children(recursive=True)}
                                        except Exception:
                                            pass
                                    except Exception:
                                        p_pids = {proc_pid}

                                    found_vram = 0.0
                                    process_on_gpu = False
                                    # Use a single nvidia-smi call for efficiency
                                    try:
                                        res = subprocess.run(
                                            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                            capture_output=True, text=True, timeout=1.0
                                        )
                                        if res.returncode == 0:
                                            for line in res.stdout.strip().split("\n"):
                                                if not line.strip() or line.startswith("pid"): continue
                                                parts = [p.strip() for p in line.split(",")]
                                                if len(parts) >= 2:
                                                    try:
                                                        chk_pid = int(parts[0])
                                                        if chk_pid in p_pids:
                                                            process_on_gpu = True
                                                            # Handle cases where memory usage is [N/A] (common with quantized models)
                                                            mem_str = parts[1].replace('MiB', '').strip()
                                                            if mem_str.replace('.', '', 1).isdigit():
                                                                found_vram += float(mem_str)
                                                    except (ValueError, IndexError):
                                                        continue
                                    except Exception:
                                        pass
                                    
                                    # If not in compute-apps, check graphics-apps (rare for ML but possible)
                                    if not process_on_gpu:
                                        try:
                                            res = subprocess.run(
                                                ["nvidia-smi", "--query-graphics-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                                capture_output=True, text=True, timeout=1.0
                                            )
                                            if res.returncode == 0:
                                                for line in res.stdout.strip().split("\n"):
                                                    if not line.strip() or line.startswith("pid"): continue
                                                    parts = [p.strip() for p in line.split(",")]
                                                    if len(parts) >= 2:
                                                        try:
                                                            chk_pid = int(parts[0])
                                                            if chk_pid in p_pids:
                                                                process_on_gpu = True
                                                                mem_str = parts[1].replace('MiB', '').strip()
                                                                if mem_str.replace('.', '', 1).isdigit():
                                                                    found_vram += float(mem_str)
                                                        except (ValueError, IndexError):
                                                            continue
                                        except Exception:
                                            pass
                                    
                                    # Update stats if process was found on GPU
                                    if process_on_gpu:
                                        proc_vram_gb = round(found_vram / 1024, 2)
                                        proc_gpu_percent = gpu_percent
                                        
                                        # FALLBACK for Windows: If nvidia-smi failed to show VRAM or showed low value
                                        # but process is known to be active on GPU.
                                        if platform.system() == "Windows":
                                            win_vram_metrics = get_windows_vram_metrics()
                                            power_vram = 0.0
                                            for pid in p_pids:
                                                pid_str = f"pid_{pid}_"
                                                for m in win_vram_metrics:
                                                    if pid_str in m.get('InstanceName', ''):
                                                        power_vram += float(m.get('CookedValue', 0))
                                            
                                            # If powershell found significant VRAM, prefer it over [N/A] from smi
                                            if power_vram > (proc_vram_gb * 1024 * 1024):
                                                proc_vram_gb = round(power_vram / (1024**3), 2)

                                    # SPECIAL CASE: If we found no VRAM from nvidia-smi but the process is the server with a cached model,
                                    # use torch.cuda.memory_allocated() which accurately tracks PyTorch allocations
                                    if proc_vram_gb == 0 and proc_pid == current_pid:
                                        # This is the server process - check if we have a model allocated via torch
                                        try:
                                            # Get accurate VRAM from torch inside the server process
                                            allocated = torch.cuda.memory_allocated()
                                            reserved = torch.cuda.memory_reserved()
                                            
                                            # Use reserved (actual VRAM footprint) as it includes caching overhead
                                            # This matches Task Manager's "Dedicated GPU memory"
                                            proc_vram_gb = round(max(allocated, reserved) / gb_divisor, 2)
                                            
                                            # On Windows, try to get more specific process load from counters
                                            if platform.system() == "Windows" and proc_vram_gb > 0.1:
                                                win_metrics = get_windows_gpu_metrics()
                                                proc_load = 0.0
                                                # Look for our specific PID in the counters
                                                pid_str = f"pid_{proc_pid}_"
                                                for m in win_metrics:
                                                    if pid_str in m.get('InstanceName', ''):
                                                        proc_load += float(m.get('CookedValue', 0))
                                                
                                                if proc_load > 0:
                                                    proc_gpu_percent = min(100.0, proc_load)
                                                else:
                                                    # Fallback to global load if we have VRAM but no specific engine load found
                                                    proc_gpu_percent = gpu_percent
                                            elif proc_vram_gb > 0.1:
                                                # Fallback for Linux/other
                                                proc_gpu_percent = gpu_percent
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
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
                        "vram_used_gb": proc_vram_gb,
                        "gpu_percent": proc_gpu_percent
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
