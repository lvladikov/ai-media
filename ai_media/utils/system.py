"""
System utilities for AI-Media.

Device detection, GPU memory management, system resources, and signal handling.
"""

import os
import sys
import signal
import gc

# Try to import psutil for resource monitoring
try:
    import psutil
except ImportError:
    psutil = None

# Import emoji helper for safe output on Windows terminals
def _safe_emoji(emoji_char, fallback=""):
    """Return emoji if terminal supports it, otherwise return fallback text."""
    try:
        emoji_char.encode(sys.stdout.encoding or 'utf-8')
        return emoji_char
    except (UnicodeEncodeError, LookupError, AttributeError):
        return fallback


def clear_gpu_memory():
    """Clear GPU memory cache to reduce fragmentation and prevent OOM errors.
    
    Call this between heavy operations to free unused memory.
    """
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            # MPS doesn't have explicit cache clearing, but gc helps
            pass
    except ImportError:
        pass


def is_bfloat16_supported():
    """Check if bfloat16 is supported on the current CUDA device.
    
    Returns True if:
    - CUDA is available
    - PyTorch's torch.cuda.is_bf16_supported() returns True
    
    BFloat16 offers same memory usage as float16 but better numerical
    stability due to larger exponent range. Supported on NVIDIA Ampere
    (RTX 30xx) and newer GPUs.
    
    Returns False for MPS, CPU, or older CUDA GPUs (pre-Ampere).
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.is_bf16_supported()
    except (ImportError, AttributeError):
        pass
    return False


def get_optimal_device_and_dtype(quiet=False, prefer_bfloat16=False):
    """
    Detect the best available hardware (CUDA, MPS, or CPU) 
    and return the device string and optimal torch dtype.
    
    Args:
        quiet: Suppress device detection messages
        prefer_bfloat16: If True, use bfloat16 on supported CUDA hardware
    """
    try:
        import torch
        if torch.cuda.is_available():
            if not quiet:
                print(f"🚀 Detected NVIDIA GPU: Using CUDA\n")
            # Use bfloat16 if requested and supported (Ampere+)
            if prefer_bfloat16 and torch.cuda.is_bf16_supported():
                return torch.device("cuda"), torch.bfloat16
            return torch.device("cuda"), torch.float16
            
        if torch.backends.mps.is_available():
            if not quiet:
                print(f"🍎 Detected Apple Silicon: Using MPS (Metal Performance Shaders)\n")
            return torch.device("mps"), torch.float16
    except ImportError:
        pass
    
    import torch  # Will raise if not installed
    if not quiet:
        print(f"💻 Using CPU (Slow): CUDA or MPS not detected (or torch missing)\n")
    return torch.device("cpu"), torch.float32


def get_system_resources():
    """Get available system RAM and VRAM."""
    ram_available = 0
    ram_total = 0
    vram_available = 0
    
    try:
        if psutil:
            mem = psutil.virtual_memory()
            ram_available = mem.available / (1024**3)  # GB
            ram_total = mem.total / (1024**3) # GB
    except Exception:
        pass
    
    try:
        import torch
        if torch.cuda.is_available():
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            vram_used = torch.cuda.memory_allocated(0) / (1024**3)
            vram_available = vram_total - vram_used
        elif torch.backends.mps.is_available():
            # MPS uses unified memory - estimate as 75% of RAM for GPU tasks
            try:
                if psutil:
                    vram_available = psutil.virtual_memory().available / (1024**3) * 0.75
            except:
                vram_available = 8  # Conservative default
    except ImportError:
        pass
    
    return ram_available, vram_available, ram_total


def check_resources_and_warn(model_id, width=None, height=None, duration=None, force=False, model_requirements=None, bypass_warning=False):
    """
    Check if system resources are sufficient for the requested task.
    Returns True to proceed, False to abort.
    
    Args:
        model_id: Hugging Face model ID
        width: Target width (optional)
        height: Target height (optional)
        duration: Target duration in seconds (optional)
        force: Skip confirmation prompts (overwrites and warnings)
        model_requirements: Dict of model requirements (from models.py)
        bypass_warning: Specifically skip resource warning prompts
    """
    if model_requirements is None:
        return True  # Can't check without requirements
        
    reqs = model_requirements.get(model_id)
    if not reqs:
        return True  # Unknown model, proceed with caution
    
    ram_available, vram_available, ram_total = get_system_resources()
    warnings = []
    
    # Check for half-precision support (bf16/fp16)
    import torch
    is_half_precision = False
    dtype_label = "float32"
    
    if torch.cuda.is_available():
        is_half_precision = True # CUDA usually runs fp16/bf16
        dtype_label = "bfloat16" if is_bfloat16_supported() else "float16"
    elif torch.backends.mps.is_available():
        is_half_precision = True # MPS uses fp16
        dtype_label = "float16"
        
    # Apply scaling factor for half-precision (approx 0.6x of fp32 requirements)
    scale_factor = 0.6 if is_half_precision else 1.0
    
    req_ram = reqs.get("ram", 0) * scale_factor
    req_vram = reqs.get("vram", 0) * scale_factor
    
    # Check RAM
    if ram_available > 0 and ram_available < req_ram:
        if ram_total > 0:
            warnings.append(f"RAM: {ram_available:.1f}GB available / {ram_total:.1f}GB Total, {req_ram:.1f}GB recommended ({dtype_label})")
        else:
            warnings.append(f"RAM: {ram_available:.1f}GB available, {req_ram:.1f}GB recommended ({dtype_label})")
    
    # Check VRAM (if available)
    if vram_available > 0 and vram_available < req_vram:
        warnings.append(f"VRAM: {vram_available:.1f}GB available, {req_vram:.1f}GB recommended ({dtype_label})")
    
    # Check resolution limits
    max_res = reqs.get("max_resolution")
    is_zeroscope = "zeroscope" in model_id.lower() and "xl" not in model_id.lower()
    
    if max_res and width and height:
        if width > max_res[0] or height > max_res[1]:
            if is_zeroscope:
                # Zeroscope has dynamic upscaling - this is informational, not a warning
                warnings.append(f"Resolution: {width}x{height} exceeds native {max_res[0]}x{max_res[1]} (Dynamic Upscaling will be used)")
            else:
                warnings.append(f"Resolution: {width}x{height} exceeds recommended max {max_res[0]}x{max_res[1]}")
    
    # Check duration limits (for audio/video)
    max_dur = reqs.get("max_duration")
    if max_dur and duration and duration > max_dur:
        warnings.append(f"Duration: {duration}s exceeds recommended max {max_dur}s")
    
    if not warnings:
        return True
    
    # Determine if we should bypass the prompt
    can_prompt = sys.stdin.isatty()
    should_bypass = force or bypass_warning or not can_prompt

    # Display warnings - use different style for zeroscope upscaling info
    if is_zeroscope and len(warnings) == 1 and "Dynamic Upscaling" in warnings[0]:
        # Zeroscope-specific informational message (not a scary warning)
        import torch
        is_mps = torch.backends.mps.is_available() and not torch.cuda.is_available()
        
        print("\n📐 Dynamic Upscaling Pipeline:\n")
        print(f"   Target:   {width}x{height}")
        print(f"   Native:   {max_res[0]}x{max_res[1]}")
        if is_mps:
            print(f"   Method:   Zeroscope 576w → Real-ESRGAN → Target (XL skipped on Mac)")
        else:
            print(f"   Method:   Zeroscope 576w → XL (1024x576) → Real-ESRGAN → Target")
        print(f"\n   Model: {model_id}")
        print(f"   ℹ️  This is the optimal workflow for high-res zeroscope output.\n")
        
        if should_bypass:
            if not can_prompt and not (force or bypass_warning):
                print("   (Non-interactive environment: Proceeding with dynamic upscaling...)\n")
            return True
        
        try:
            choice = input("   Proceed with dynamic upscaling? [Y/n]: ").lower().strip()
            if choice in ['', 'y', 'yes']:
                print("")  # Spacer
                return True
            print("\n💡 Tip: Use -s 576x320 for native resolution (fastest).\n")
            sys.exit(0)
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled.\n")
            sys.exit(0)
    
    # Standard warning display for other cases
    print("\n⚠️  Resource Warning:\n")
    for w in warnings:
        print(f"   • {w}")
    print(f"\n   Model: {model_id}")
    
    # Add dtype info
    import torch
    if torch.cuda.is_available():
        dtype_info = "bfloat16" if is_bfloat16_supported() else "float16"
    elif torch.backends.mps.is_available():
        dtype_info = "float32"
    else:
        dtype_info = "float32"
    print(f"   Dtype: {dtype_info}")
    
    # Check for VAE Tiling condition (Resolution > 1536x1536)
    if width and height:
        total_pixels = width * height
        if total_pixels > 3072 * 3072:  # ~9.4MP
            print(f"\n   ⚠️  CRITICAL WARNING: Resolution {width}x{height} is extremely high.")
            print(f"      Standard generation will likely fail with 'Invalid buffer size'.")
            print(f"      Recommended: Generate at 2K/4K and use an external upscaler.")
            print(f"      💡 Or try: -s 720p --upscale -uf 4x (to get 5K)\n")
        elif total_pixels > 1536 * 1536:
            print(f"\n   ℹ️  Note: VAE Tiling will be enabled to reduce memory usage.\n")
        
    print(f"   This job may cause slowdowns, swapping, or crashes.\n")
    
    if should_bypass:
        if force:
            print("   (Proceeding due to --force flag)\n")
        elif bypass_warning:
            print("   (Proceeding due to --bypass-warning flag)\n")
        elif not can_prompt:
            print("   (Non-interactive environment: Proceeding anyway...)\n")
        return True
    
    try:
        choice = input("   Continue anyway? [y/N]: ").lower().strip()
        if choice in ['y', 'yes']:
            print("")  # Spacer
            return True
        print("\n💡 Tip: Try a smaller resolution, shorter duration, or lighter model.\n")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled.\n")
        sys.exit(0)



# --- Signal Handling ---

# Global test state for CTRL+C handling
_test_state = {
    'active': False,
    'passed': 0,
    'failed': 0,
    'total': 0,
    'current_process': None  # Track subprocess for cleanup
}
_interrupted = False


def _force_exit():
    """Forcefully terminate the process after timeout."""
    print("\n💀 Force-killing process (GPU operations did not respond in time)...")
    os._exit(1)  # Immediate exit, bypasses cleanup


def _kill_test_subprocess():
    """Kill the current test subprocess and its children."""
    proc = _test_state.get('current_process')
    if proc:
        pid = proc.pid
        print(f"\n   ⚠️  Killing subprocess tree (PID {pid})...")
        if os.name == 'nt':
            try:
                import subprocess
                subprocess.run(['taskkill', '/T', '/F', '/PID', str(pid)], 
                               capture_output=True, timeout=5)
                print(f"   ✅ Process tree terminated.")
            except Exception as e:
                print(f"   ⚠️  taskkill failed: {e}")
                try:
                    proc.kill()
                except:
                    pass
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except:
                try:
                    proc.kill()
                except:
                    pass


def signal_handler(sig, frame):
    """Handle SIGINT/SIGTERM signals."""
    global _interrupted
    
    if _test_state['active']:
        # Kill subprocess first
        _kill_test_subprocess()
        
        completed = _test_state['passed'] + _test_state['failed']
        print(f"\n\n{'='*60}")
        print(f"❌ Test suite interrupted by user (CTRL+C)")
        print(f"{'='*60}")
        print(f"   Completed: {completed}/{_test_state['total']}")
        print(f"   Passed: {_test_state['passed']} ✅")
        if _test_state['failed'] > 0:
            print(f"   Failed: {_test_state['failed']} ❌")
        else:
            print(f"   Failed: {_test_state['failed']}")
        print(f"   Skipped: {_test_state['total'] - completed}")
        print(f"{'='*60}")
        sys.exit(130)
    else:
        _interrupted = True
        
        # Immediately exit - GPU/multiprocessing cleanup causes hangs
        print("\n\n⚠️  Interrupted!")
        
        # Restore terminal state if possible before brutal exit
        if sys.stdout.isatty():
            # ANSI: Disable Mouse Reporting (SGR + Basic), Reset Color, Show Cursor
            print("\033[?1000l\033[?1006l\033[0m\033[?25h", end="", flush=True)
            if os.name != 'nt':
                os.system('stty sane')
                
        os._exit(0)  # Immediate exit, no cleanup (cleanup hangs with model_cpu_offload)


def setup_signal_handlers():
    """Install signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def ensure_paths(output_path):
    """Create parent directories if needed."""
    from pathlib import Path
    if output_path is None:
        return
    parent = Path(output_path).parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
        print(f"   {_safe_emoji('📁 ', '')}Created directory: {parent}")
