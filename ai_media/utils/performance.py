"""
Performance tracking and resource monitoring for AI-Media.

PerformanceTracker: Tracks and estimates generation times.
ResourceMonitor: Monitors CPU, RAM, and GPU usage during generation.
"""

import os
import json
import subprocess
import threading
import time
from .interaction import emoji
from .parsers import format_time


def write_report_json(path, stats):
    """Write generation stats to a JSON file."""
    try:
        with open(path, 'w') as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to write report JSON: {e}")


class PerformanceTracker:
    """Tracks and estimates generation times based on historical data."""
    
    def __init__(self, filepath="performance.json"):
        self.filepath = filepath
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent=2)

    def record_image(self, model, width, height, device, time_taken, cpu=0, ram=0, vram=0, gpu=0, dtype=None):
        dev_str = device.type if hasattr(device, 'type') else str(device)
        if dtype:
            key = f"{model}|{dev_str}|{dtype}|{width}x{height}"
        else:
            key = f"{model}|{dev_str}|{width}x{height}"

        if "image" not in self.data:
            self.data["image"] = {}
        
        # Re-average strategy: (last_average + new_time) / 2
        entry = self.data["image"].get(key, {})
        
        if "average_time" in entry:
            new_avg = (entry["average_time"] + time_taken) / 2.0
            entry = {
                "average_time": new_avg,
                "average_ram": (entry.get("average_ram", ram) + ram) / 2.0,
                "average_vram": (entry.get("average_vram", vram) + vram) / 2.0,
                "average_cpu": (entry.get("average_cpu", cpu) + cpu) / 2.0,
                "average_gpu": (entry.get("average_gpu", gpu) + gpu) / 2.0
            }
        else:
            entry = {
                "average_time": time_taken,
                "average_ram": ram,
                "average_vram": vram,
                "average_cpu": cpu,
                "average_gpu": gpu
            }
        
        self.data["image"][key] = entry
        self._save()

    def record_linear(self, category, model, device, duration, time_taken, width=None, height=None, cpu=0, ram=0, vram=0, gpu=0, dtype=None):
        """Record Audio/Video generation using rolling average rate (seconds to gen / seconds of content)."""
        dev_str = device.type if hasattr(device, 'type') else str(device)
        # For video, resolution also matters, so we include it in key
        if category == "video":
            if dtype:
                key = f"{model}|{dev_str}|{dtype}|{width}x{height}"
            else:
                key = f"{model}|{dev_str}|{width}x{height}"
        else:
            if dtype:
                key = f"{model}|{dev_str}|{dtype}"
            else:
                key = f"{model}|{dev_str}"
            
        if category not in self.data:
            self.data[category] = {}
        
        current_rate = time_taken / duration if duration > 0 else 0
        entry = self.data[category].get(key, {})
        
        if "average_rate" in entry:
            new_rate = (entry["average_rate"] + current_rate) / 2.0
            entry = {
                "average_rate": new_rate,
                "average_ram": (entry.get("average_ram", ram) + ram) / 2.0,
                "average_vram": (entry.get("average_vram", vram) + vram) / 2.0,
                "average_cpu": (entry.get("average_cpu", cpu) + cpu) / 2.0,
                "average_gpu": (entry.get("average_gpu", gpu) + gpu) / 2.0
            }
        else:
            entry = {
                "average_rate": current_rate, 
                "average_ram": ram, 
                "average_vram": vram, 
                "average_cpu": cpu, 
                "average_gpu": gpu
            }
            
        self.data[category][key] = entry
        self._save()

    def estimate_image(self, model, width, height, device, dtype=None):
        dev_str = device.type if hasattr(device, 'type') else str(device)
        if dtype:
            key = f"{model}|{dev_str}|{dtype}|{width}x{height}"
        else:
            key = f"{model}|{dev_str}|{width}x{height}"
        stats = self.data.get("image", {}).get(key)
        if stats and "average_time" in stats:
            return (
                stats["average_time"],
                stats.get("average_cpu", 0),
                stats.get("average_ram", 0),
                stats.get("average_vram", 0),
                stats.get("average_gpu", 0)
            )
        return 0, 0, 0, 0, 0

    def estimate_linear(self, category, model, device, duration, width=None, height=None, dtype=None):
        dev_str = device.type if hasattr(device, 'type') else str(device)
        if category == "video":
            if dtype:
                key = f"{model}|{dev_str}|{dtype}|{width}x{height}"
            else:
                key = f"{model}|{dev_str}|{width}x{height}"
        else:
            if dtype:
                key = f"{model}|{dev_str}|{dtype}"
            else:
                key = f"{model}|{dev_str}"
            
        stats = self.data.get(category, {}).get(key)
        if stats and "average_rate" in stats:
            return (
                stats["average_rate"] * duration,
                stats.get("average_cpu", 0),
                stats.get("average_ram", 0),
                stats.get("average_vram", 0),
                stats.get("average_gpu", 0)
            )
        return 0, 0, 0, 0, 0

    def print_estimate(self, est_time, est_cpu, est_ram, est_vram, est_gpu):
        """Print formatted estimation stats to console."""
        if est_time > 0:
            print(f"{emoji('⏱️ ', '')} Estimated Resources: Time: {format_time(est_time)} | RAM: {est_ram:.1f}GB | VRAM: {est_vram:.1f}GB | CPU: {est_cpu:.1f}% | GPU: {est_gpu:.1f}%")
        else:
            print(f"{emoji('⏱️ ', '')} Estimated Resources: (New combination - no history)")

    def print_actual(self, time_taken, cpu, ram, vram, gpu):
        """Print formatted actual stats to console."""
        print(f"{emoji('⏱️ ', '')} Actual Resources:    Time: {format_time(time_taken)} | RAM: {ram:.1f}GB | VRAM: {vram:.1f}GB | CPU: {cpu:.1f}% | GPU: {gpu:.1f}%")



class ResourceMonitor:
    """Monitors CPU, RAM, and GPU VRAM/Load usage in a background thread."""
    
    def __init__(self, interval=0.5):
        self.interval = interval
        self.running = False
        self.thread = None
        self.cpu_readings = []
        self.ram_readings = []
        self.vram_readings = []
        self.gpu_readings = []  # GPU Load %
        
        try:
            import psutil
            self.psutil = psutil
        except ImportError:
            self.psutil = None
            print("⚠️  'psutil' not found. Resource monitoring disabled.")
            
        # Check for torch to monitor VRAM
        try:
            import torch
            self.torch = torch
            self.has_cuda = torch.cuda.is_available()
            self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except ImportError:
            self.torch = None
            self.has_cuda = False
            self.has_mps = False

    def _monitor(self):
        while self.running:
            if self.psutil:
                cpu = self.psutil.cpu_percent(interval=None)
                ram = self.psutil.virtual_memory().used / (1024**3)  # GB
                self.cpu_readings.append(cpu)
                self.ram_readings.append(ram)
                
            # VRAM Monitoring
            vram = 0
            if self.torch:
                if self.has_cuda:
                    vram = self.torch.cuda.memory_allocated() / (1024**3)  # GB
                elif self.has_mps:
                    if hasattr(self.torch, 'mps') and hasattr(self.torch.mps, 'current_allocated_memory'):
                        vram = self.torch.mps.current_allocated_memory() / (1024**3)
                    elif hasattr(self.torch.mps, 'driver_allocated_memory'):
                        vram = self.torch.mps.driver_allocated_memory() / (1024**3)
            self.vram_readings.append(vram)
            
            # GPU Load Monitoring
            gpu_load = 0
            if self.has_cuda:
                try:
                    # Windows/Linux with NVIDIA drivers
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, check=False, timeout=1.0
                    )
                    if result.returncode == 0:
                        gpu_load = float(result.stdout.strip())
                except Exception:
                    pass
            elif self.has_mps:
                try:
                    # Apple Silicon: Query AGXAccelerator via ioreg for GPU utilization
                    import re
                    env = os.environ.copy()
                    env["MallocStackLogging"] = "0"
                    
                    result = subprocess.run(
                        ['ioreg', '-r', '-d', '1', '-w', '0', '-c', 'AGXAccelerator'],
                        capture_output=True, text=True, check=False,
                        env=env, timeout=1.0
                    )
                    if result.returncode == 0:
                        # Extract "Device Utilization %" from PerformanceStatistics
                        match = re.search(r'"Device Utilization %"=(\d+)', result.stdout)
                        if match:
                            gpu_load = float(match.group(1))
                except Exception:
                    pass
            self.gpu_readings.append(gpu_load)
            
            time.sleep(self.interval)

    def __enter__(self):
        if self.psutil:
            self.psutil.cpu_percent(interval=None)  # Prime CPU
            self.running = True
            self.thread = threading.Thread(target=self._monitor, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def get_averages(self):
        avg_cpu = sum(self.cpu_readings) / len(self.cpu_readings) if self.cpu_readings else 0
        avg_ram = sum(self.ram_readings) / len(self.ram_readings) if self.ram_readings else 0
        avg_vram = sum(self.vram_readings) / len(self.vram_readings) if self.vram_readings else 0
        avg_gpu = sum(self.gpu_readings) / len(self.gpu_readings) if self.gpu_readings else 0
        return avg_cpu, avg_ram, avg_vram, avg_gpu
