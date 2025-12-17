#!/usr/bin/env python3
"""
AI-Media: Local Offline Media Generation CLI

Generate Images, Videos, and Audio using local AI models (Flux, MusicGen, etc.).
Wraps 'diffusers' and 'transformers' libraries.

Usage:
  python ai-media.py -i -p "Prompt" -o image.png
  python ai-media.py -v -p "Prompt" -o video.mp4 -l 5s
  python ai-media.py -a -p "Prompt" -o audio.mp3 -l 30s
"""

import argparse
import warnings

# Suppress common library warnings
# (Some libraries use different warning categories or print directly)
warnings.filterwarnings("ignore", message="User provided device_type of 'cuda'", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")  # Catch all
warnings.filterwarnings("ignore", message=".*upcast_vae.*deprecated.*", category=FutureWarning)
import ast
import json
import logging
import os
import re
import sys
import signal
import shutil
import time
try:
    import psutil  # For resource checking
except ImportError:
    psutil = None
from pathlib import Path
from datetime import datetime
import PIL.Image
import PIL.ImageOps

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress verbose logging from transformers/diffusers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)

# Set environment variable to suppress transformers warnings (must be before import)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Fix for deadlock warning

# CUDA Memory Optimization - Reduce fragmentation on Windows/NVIDIA
# This helps prevent "CUDA out of memory" errors even when GPU has free memory
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


# --- Constants ---
DEFAULT_IMAGE_SIZE = "720p"  # Maps to 1280x720
DEFAULT_AUDIO_SAMPLING = "32000" # Hz (MusicGen default is usually 32k)
DEFAULT_AUDIO_BITDEPTH = 16
DEFAULT_DURATION = "15s"

# Model Mappings (Short codes to Hugging Face Hub IDs)
IMAGE_MODELS = {
    "flux": "black-forest-labs/FLUX.1-schnell",        # State of the art, fast
    "flux-dev": "black-forest-labs/FLUX.1-dev",        # Higher quality, slower
    "sdxl": "stabilityai/sdxl-turbo",                  # Fast, good quality (DEFAULT)
    "sd-1.5": "runwayml/stable-diffusion-v1-5",        # Classic, lightweight
    "upscaler": "stabilityai/stable-diffusion-x4-upscaler", # 4x Upscaling
    "upscaler_x2": "stabilityai/sd-x2-latent-upscaler",     # 2x Latent Upscaling
    "default": "stabilityai/sdxl-turbo"
}

EDIT_MODELS = {
    "instruct-pix2pix": "timbrooks/instruct-pix2pix",
    "instruct-pix2pix-sdxl": "diffusers/sdxl-instructpix2pix-768",
    "remove-bg": "briaai/RMBG-1.4",
    "default": "timbrooks/instruct-pix2pix"
}

AUDIO_MODELS = {
    "musicgen-small": "facebook/musicgen-small",       # Fast, good for music
    "musicgen-medium": "facebook/musicgen-medium",     # Better quality music
    "musicgen-large": "facebook/musicgen-large",       # Best quality music
    "audioldm2": "cvssp/audioldm2",                    # General audio/SFX
    "stable-audio": "stabilityai/stable-audio-open-1.0", # Variable length, high quality (Gated)
    "bark": "suno/bark",                               # TTS / Audio (Transformer)
    "default": "facebook/musicgen-medium"
}

VIDEO_MODELS = {
    "ms-1.7b": "damo-vilab/text-to-video-ms-1.7b",     # Has watermark issues
    "zeroscope": "cerspense/zeroscope_v2_576w",        # 576x320 optimized (default)
    "cogvideox": "THUDM/CogVideoX-5b",                 # High quality (requires high VRAM)
    "svd": "stabilityai/stable-video-diffusion-img2vid-xt", # SVD Image-to-Video
    "default": "cerspense/zeroscope_v2_576w"
}

# Resolution Presets
RESOLUTIONS = {
    "480p": (854, 480),
    "576p": (1024, 576),
    "720p": (1280, 720),
    "900p": (1600, 900),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2k": (2048, 1080), # approx
    "3k": (3072, 1728),
    "2160p": (3840, 2160),
    "4k": (3840, 2160),
    "5k": (5120, 2880),
    "6k": (6144, 3456),
    "7k": (7168, 4032),
    "4320p": (7680, 4320),
    "8k": (7680, 4320),
    "9k": (9216, 5184),
    "10k": (10240, 5760),
    "hd": (1280, 720),
    "fhd": (1920, 1080),
    "uhd": (3840, 2160),
    "sd": (640, 480),
    "vga": (640, 480)
}

# Model resource requirements (estimated RAM/VRAM in GB)
# Based on model training specs, Hugging Face model cards, and practical testing.
# Format: { model_id: { "vram": X, "ram": Y, "max_resolution": (W, H) } }
MODEL_REQUIREMENTS = {
    # Image Models (max_resolution based on training data and VRAM constraints)
    "runwayml/stable-diffusion-v1-5": {"vram": 4, "ram": 8, "max_resolution": (1280, 1280)},  # Trained 512x512, works to ~1280
    "stabilityai/sdxl-turbo": {"vram": 8, "ram": 16, "max_resolution": (1536, 1536)},  # Trained 1024x1024
    "black-forest-labs/FLUX.1-schnell": {"vram": 16, "ram": 70, "max_resolution": (2048, 2048)}, # ~70GB on Mac (float32)
    "black-forest-labs/FLUX.1-dev": {"vram": 24, "ram": 80, "max_resolution": (2048, 2048)},
    # Audio Models (max_duration in seconds, based on model architecture limits)
    "facebook/musicgen-small": {"vram": 4, "ram": 8, "max_duration": 30},
    "facebook/musicgen-medium": {"vram": 8, "ram": 12, "max_duration": 60},
    "facebook/musicgen-large": {"vram": 16, "ram": 24, "max_duration": 120},
    "cvssp/audioldm2": {"vram": 8, "ram": 12, "max_duration": 60},
    "stabilityai/stable-audio-open-1.0": {"vram": 10, "ram": 16, "max_duration": 47}, # Max 47s training
    "suno/bark": {"vram": 4, "ram": 12, "max_duration": 30}, # Small/Large split, conservative est
    # Video Models (max_resolution based on training data)
    "damo-vilab/text-to-video-ms-1.7b": {"vram": 12, "ram": 16, "max_resolution": (1280, 720)},
    "cerspense/zeroscope_v2_576w": {"vram": 8, "ram": 12, "max_resolution": (576, 320)},
    "THUDM/CogVideoX-5b": {"vram": 32, "ram": 48, "max_resolution": (1920, 1080)}, # ~50GB on Mac (float32)
    "stabilityai/stable-video-diffusion-img2vid-xt": {"vram": 8, "ram": 12, "max_resolution": (1024, 576)},
    "stabilityai/stable-diffusion-x4-upscaler": {"vram": 8, "ram": 16, "max_resolution": (4096, 4096)},
    "stabilityai/sd-x2-latent-upscaler": {"vram": 4, "ram": 8, "max_resolution": (2048, 2048)},
    "timbrooks/instruct-pix2pix": {"vram": 8, "ram": 12, "max_resolution": (1024, 1024)},
    "diffusers/sdxl-instructpix2pix-768": {"vram": 10, "ram": 16, "max_resolution": (1024, 1024)},
    "briaai/RMBG-1.4": {"vram": 4, "ram": 8, "max_resolution": (2048, 2048)},
}


def clear_gpu_memory():
    """Clear GPU memory cache to reduce fragmentation and prevent OOM errors.
    
    Call this between heavy operations to free unused memory.
    """
    import gc
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


def get_system_resources():
    """Get available system RAM and VRAM."""
    ram_available = 0
    vram_available = 0
    vram_total = 0
    
    try:
        mem = psutil.virtual_memory()
        ram_available = mem.available / (1024**3)  # GB
    except ImportError:
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
                vram_available = psutil.virtual_memory().available / (1024**3) * 0.75
            except:
                vram_available = 8  # Conservative default
    except ImportError:
        pass
    
    return ram_available, vram_available


def check_resources_and_warn(model_id, width=None, height=None, duration=None, force=False):
    """
    Check if system resources are sufficient for the requested task.
    Returns True to proceed, False to abort.
    """
    reqs = MODEL_REQUIREMENTS.get(model_id)
    if not reqs:
        return True  # Unknown model, proceed with caution
    
    ram_available, vram_available = get_system_resources()
    warnings = []
    
    # Check RAM
    if ram_available > 0 and ram_available < reqs.get("ram", 0):
        warnings.append(f"RAM: {ram_available:.1f}GB available, {reqs['ram']}GB recommended")
    
    # Check VRAM (if available)
    if vram_available > 0 and vram_available < reqs.get("vram", 0):
        warnings.append(f"VRAM: {vram_available:.1f}GB available, {reqs['vram']}GB recommended")
    
    # Check resolution limits
    max_res = reqs.get("max_resolution")
    if max_res and width and height:
        if width > max_res[0] or height > max_res[1]:
            warnings.append(f"Resolution: {width}x{height} exceeds recommended max {max_res[0]}x{max_res[1]}")
    
    # Check duration limits (for audio/video)
    max_dur = reqs.get("max_duration")
    if max_dur and duration and duration > max_dur:
        warnings.append(f"Duration: {duration}s exceeds recommended max {max_dur}s")
    
    if not warnings:
        return True
    
    # Display warnings
    print("\n⚠️  Resource Warning:\n")
    for w in warnings:
        print(f"   • {w}")
    print(f"\n   Model: {model_id}")
    
    # Check for VAE Tiling condition (Resolution > 1536x1536)
    if width and height:
        total_pixels = width * height
        if total_pixels > 3072 * 3072: # ~9.4MP
             print(f"\n   ⚠️  CRITICAL WARNING: Resolution {width}x{height} is extremely high.")
             print(f"      Standard generation will likely fail with 'Invalid buffer size'.")
             print(f"      Recommended: Generate at 2K/4K and use an external upscaler.")
             print(f"      💡 Or try: -s 720p --upscale -uf 4x (to get 5K)\n")
        elif total_pixels > 1536 * 1536:
            print(f"\n   ℹ️  Note: VAE Tiling will be enabled to reduce memory usage.\n")
        
    print(f"   This job may cause slowdowns, swapping, or crashes.\n")
    
    if force:
        print("   (Proceeding due to --force flag)\n")
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
    'total': 0
}

def signal_handler(sig, frame):
    if _test_state['active']:
        completed = _test_state['passed'] + _test_state['failed']
        print(f"\n\n{'='*60}")
        print(f"❌ Test suite interrupted by user (CTRL+C)")
        print(f"{'='*60}")
        print(f"   Completed: {completed}/{_test_state['total']}")
        print(f"   Passed: {_test_state['passed']} ✅")
        print(f"   Failed: {_test_state['failed']} ❌")
        print(f"   Skipped: {_test_state['total'] - completed}")
        print(f"{'='*60}")
        sys.exit(130)
    else:
        print("\n\n⚠️  Interrupted! Cleaning up...")
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# --- Helper Parsing Functions ---

def get_optimal_device_and_dtype(quiet=False):
    """
    Detect the best available hardware (CUDA, MPS, or CPU) 
    and return the device string and optimal torch dtype.
    """
    try:
        import torch
        if torch.cuda.is_available():
            if not quiet: print(f"🚀 Detected NVIDIA GPU: Using CUDA\n")
            return torch.device("cuda"), torch.float16
            
        if torch.backends.mps.is_available():
            if not quiet: print(f"🍎 Detected Apple Silicon: Using MPS (Metal Performance Shaders)\n")
            return torch.device("mps"), torch.float16
    except ImportError:
        pass
        
    if not quiet: print(f"💻 Using CPU (Slow): CUDA or MPS not detected (or torch missing)\n")
    return torch.device("cpu"), torch.float32

def parse_size(value):
    """
    Parse size string or object into (width, height).
    Accepts:
      - "480p", "720p", "1080p", "1440p"
      - "1k", "2k", "3k", "4k", "5k", ... "10k"
      - "1280x720"
      - "w: 1280, h: 720" (Braces {} are optional)
    """
    if not value:
        return RESOLUTIONS[DEFAULT_IMAGE_SIZE]
        
    normalized = value.strip().lower()
    
    # Check presets
    if normalized in RESOLUTIONS:
        return RESOLUTIONS[normalized]
        
    # Check WxH format
    if 'x' in normalized:
        try:
            w, h = map(int, normalized.split('x'))
            return (w, h)
        except ValueError:
            pass
            
    # Check Object/JSON-like format
    if '{' in normalized and '}' in normalized:
        try:
            w = None
            h = None
            
            # Remove braces
            content = normalized.strip("{}")
            parts = content.split(',')
            
            for part in parts:
                if ':' not in part: continue
                k, v = part.split(':', 1)
                k = k.strip()
                v = int(re.sub(r'[^0-9]', '', v)) # extract number
                
                if k in ['w', 'width']:
                    w = v
                elif k in ['h', 'height']:
                    h = v
            
            if w and h:
                return (w, h)
        except Exception as e:
            print(f"Warning: Failed to parse object size '{value}': {e}")
    
    # Fallback default if parsing fails
    print(f"Warning: Could not parse size '{value}'. Using default 720p.")
    return RESOLUTIONS["720p"]


def parse_duration(value):
    """
    Parse duration into seconds (float).
    Accepts:
      - Numeric (seconds)
      - Strings: "15s", "1m", "1h50m", "50s"
      - Objects: "{h:1, m:25, s:10}"
    """
    if not value:
        return 15.0
        
    # If standard number strings "15", "15.5"
    try:
        return float(value)
    except ValueError:
        pass
        
    normalized = str(value).strip().lower()
    
    total_seconds = 0
    
    # Check Object format
    if '{' in normalized:
        try:
            content = normalized.strip("{}")
            parts = content.split(',')
            for part in parts:
                if ':' not in part: continue
                k, v = part.split(':', 1)
                k = k.strip()
                try:
                    val = float(re.sub(r'[^0-9\.]', '', v))
                except: continue
                
                if k in ['h', 'hours', 'hour']:
                    total_seconds += val * 3600
                elif k in ['m', 'mins', 'min', 'minutes', 'minute']:
                    total_seconds += val * 60
                elif k in ['s', 'sec', 'secs', 'seconds', 'second']:
                    total_seconds += val
            return total_seconds
        except Exception:
            pass

    # Check String format "1h50m10s"
    # Regex to find pairs of number+unit
    pattern = r'(\d+(?:\.\d+)?)\s*([hms])'
    matches = re.findall(pattern, normalized)
    
    if matches:
        for val_str, unit in matches:
            val = float(val_str)
            if unit == 'h':
                total_seconds += val * 3600
            elif unit == 'm':
                total_seconds += val * 60
            elif unit == 's':
                total_seconds += val
        return total_seconds
        
    # Fallback logic for "M:S" or "H:M:S" or just "50s"
    if ':' in normalized:
        parts = normalized.split(':')
        parts = [float(p) for p in parts]
        if len(parts) == 3: # H:M:S
            return parts[0]*3600 + parts[1]*60 + parts[2]
        elif len(parts) == 2: # M:S
            return parts[0]*60 + parts[1]
            
    return 15.0


def parse_sampling_rate(value):
    """Parse sampling rate string to integer Hz."""
    if not value:
        return 32000
    
    normalized = str(value).strip().lower()
    
    # Handle "44.1khz" -> 44100
    if 'k' in normalized:
        try:
            num = float(re.sub(r'[^0-9\.]', '', normalized))
            return int(num * 1000)
        except:
             pass # Fallthrough to default behavior
    
    try:
        return int(re.sub(r'[^0-9]', '', normalized))
    except:
        return 32000


# --- Format Helpers ---
def format_time(seconds):
    """Convert seconds to human readable string (e.g. 2w 1d 1h 2m 3.5s)."""
    if not seconds:
        return "0s"
        
    current = float(seconds)
    intervals = (
        ('w', 604800),  # 60 * 60 * 24 * 7
        ('d', 86400),   # 60 * 60 * 24
        ('h', 3600),    # 60 * 60
        ('m', 60),
    )
    
    result = []
    for name, count in intervals:
        value = int(current // count)
        if value:
            current -= value * count
            result.append(f"{value}{name}")
            
    # Remaining seconds with precision
    if current > 0 or not result:
        # If integer-ish, show int, else float
        if current % 1 == 0:
            result.append(f"{int(current)}s")
        else:
            result.append(f"{current:.1f}s")
            
    return " ".join(result)


def parse_bitrate(value):
    """Return bitrate string in standardized format or passed through if complex."""
    if not value:
        return None
    return value.strip()


# --- Generation Logic Wrappers ---

def generate_image(prompt, output_path, width, height, model_name="default", unsafe=False):
    """Generate image using Diffusers (Flux/SDXL)."""
    
    # Resolve Model ID
    model_id = IMAGE_MODELS.get(model_name.lower(), model_name) # Allow raw ID if not in map
    if model_name.lower() == "default": model_id = IMAGE_MODELS["default"]
    
    print(f"🎨 Generating Image")
    print(f"   Model:  {model_id}")
    print(f"   Prompt: '{prompt}'")
    print(f"   Size:   {width}x{height}")
    print(f"   Output: {output_path}")
    print("") # Spacer
    
    try:
        import torch
        from diffusers import FluxPipeline, AutoPipelineForText2Image
        
        # Determine device
        device, dtype = get_optimal_device_and_dtype(quiet=True)
        

        
        # Determine Pipeline Class based on model
        if "flux" in model_id.lower():
            # FLUX on MPS requires float32 to avoid dtype mismatch errors
            flux_dtype = torch.float32 if device.type == "mps" else dtype
            pipe = FluxPipeline.from_pretrained(
                model_id, 
                torch_dtype=flux_dtype
            )
            # Flux parameters
            extra_kwargs = {
                "guidance_scale": 0.0, 
                "num_inference_steps": 4, 
                "max_sequence_length": 256
            }
        elif "sdxl-turbo" in model_id.lower() or "turbo" in model_id.lower():
            # SDXL Turbo requires very few steps and specific settings
            # Use float32 on MPS to avoid black images (VAE produces NaN in float16)
            sdxl_dtype = torch.float32 if device.type == "mps" else dtype
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id, 
                torch_dtype=sdxl_dtype,
                variant="fp16" if sdxl_dtype == torch.float16 else None
            )
            # SDXL Turbo needs low steps and no guidance
            extra_kwargs = {
                "guidance_scale": 0.0,
                "num_inference_steps": 4
            }
        else:
            # Generic Text2Image (SD1.5, etc.)
            # On MPS, generic models are safer in float32 to prevent dtype mismatches
            run_dtype = torch.float32 if device.type == "mps" else dtype
            
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id, 
                torch_dtype=run_dtype,
                variant="fp16" if run_dtype == torch.float16 else None
            )
            extra_kwargs = {} # Use defaults
            
        pipe.to(device)
        
        # Disable NSFW safety checker if requested
        if unsafe:
            pipe.safety_checker = None
            pipe.requires_safety_checker = False
        
        # Memory optimization and VAE fix for MPS/Mac
        if device.type == "mps":
            pipe.enable_attention_slicing()
            # Fix black images on MPS: VAE produces NaN in float16, use float32
            if hasattr(pipe, 'vae'):
                pipe.vae = pipe.vae.to(torch.float32)
            # Fix Safety Checker dtype mismatch on MPS (Input Half vs Bias Float)
            if hasattr(pipe, 'safety_checker') and pipe.safety_checker is not None:
                pipe.safety_checker = pipe.safety_checker.to(torch.float32)

        # High-Resolution Memory Optimization (VAE Tiling)
        # 4K images (3840x2160 = ~8.3MP) cause massive VRAM spikes during decoding without tiling
        # Trigger tiling if pixels > 1536x1536 (~2.3MP)
        total_pixels = width * height
        if total_pixels > 1536 * 1536:
            print(f"ℹ️  High resolution detected ({width}x{height}).")
            print(f"   ✓ Enabling VAE Tiling (Memory Optimization)\n")
            if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_tiling'):
                pipe.vae.enable_tiling()
            else:
                pipe.enable_vae_tiling()
        
        print(f"🎨 Generating {width}x{height} image... (This may take a moment)")
        # Suppress RuntimeWarning from diffusers image_processor during NSFW filtering
        # (It throws "invalid value encountered in cast" when processing black images)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cast")
            output = pipe(
                prompt, 
                height=height, 
                width=width,
                **extra_kwargs
            )
        image = output.images[0]
        
        # Check for NSFW content interception
        if hasattr(output, "nsfw_content_detected") and output.nsfw_content_detected:
            # output.nsfw_content_detected is a list of booleans
            if output.nsfw_content_detected[0]:
                print(f"⚠️  Warning: Potential NSFW content detected.\n")
                print(f"The model's safety checker has blocked the image (returning a black frame).")
                print(f"👉 Please modify your prompt and try again.")
                print(f"💡 If your prompt is appropriate, try again with --unsafe to disable the safety checker.\n")
        
        image.save(output_path)
        print(f"✅ Image saved to {output_path}")
        return True
        
    except ImportError as e:
        print(f"❌ Error: Missing dependencies. {e}")
        return False
    except Exception as e:
        err_str = str(e)
        if "401" in err_str or "restricted" in err_str.lower():
            print(f"❌ Access Denied (401). Test model '{model_id}' is gated.")
            print("   👉 Solution 1: Run 'huggingface-cli login' with your token.")
            print("   👉 Solution 2: Use an open model like '--image-model sd-1.5'.")
        elif "divisible by 8" in err_str:
            print(f"❌ Resolution Error: {e}")
            
            # Smart Correction
            new_w = round(width / 8) * 8
            new_h = round(height / 8) * 8
            
            print(f"\n💡 Tip: Dimensions must be multiples of 8.")
            print(f"   Closest valid size: {new_w}x{new_h}")
            
            try:
                choice = input(f"   🔄 Retry with {new_w}x{new_h}? [y/N]: ").lower().strip()
                if choice in ['y', 'yes']:
                    print("") # Spacer
                    return generate_image(prompt, output_path, new_w, new_h, model_name=model_name, unsafe=unsafe)
            except KeyboardInterrupt:
                pass
            print("")
        elif "Invalid buffer size" in err_str:
            print(f"\n❌ Hardware Limitation Reached (Single Buffer Limit)")
            print(f"   Error: {e}")
            print(f"\n   Explanation:")
            print(f"   • Native {width}x{height} generation requires calculating a massive Attention Matrix.")
            print(f"   • This exceeded the maximum allowed size for a single tensor (usually ~4GB on MPS/Metal).")
            print(f"   • This is a hardware/driver limit, not a VRAM limit.")
            print(f"\n   💡 Solution: Use a lower resolution (e.g. 4k or 2k).")
            print(f"      (Native 5K generation requires 'MultiDiffusion' tiling which is not currently supported.)\n")
            
            # Auto-Upscale Fallback for 5K requests
            # If target was roughly 5K (5120x2880), offering 1280x720 (720p) -> x4 Upscale = 5120x2880
            # 1280 * 4 = 5120
            
            try:
                print(f"   ✨ Alternative: Generate at 1280x720 and Auto-Upscale x4?")
                print(f"      This produces a 5120x2880 (5K) image using the Upscaler model.")
                choice = input(f"   🔄 Try Auto-Upscale workflow? [y/N]: ").lower().strip()
                if choice in ['y', 'yes']:
                    print("\n📉 Switching to base resolution: 1280x720...")
                    # 1. Generate Base Image
                    success = generate_image(prompt, output_path, 1280, 720, model_name=model_name, unsafe=unsafe)
                    if success:
                        # 2. Upscale Result
                        print("")
                        return upscale_image_file(output_path, output_path, strength=0.0, factor=4.0) # Overwrite
            except KeyboardInterrupt:
                pass
            print("")
        else:
            print(f"❌ Generation failed: {e}")
        return False


def generate_caption(input_path, device, quiet=False, model_type="florence"):
    """
    Generate a text description for an image or video.
    Models: 
      - 'florence' (Microsoft Florence-2-Large, SOTA)
      - 'blip' (Salesforce BLIP-Large, Classic)
    """
    try:
        if not quiet: print(f"👁️  Analyzing input: {input_path}")
        
        # ----------------------------------------------------------------
        # MODEL: BLIP
        # ----------------------------------------------------------------
        if model_type == "blip":
            from transformers import BlipProcessor, BlipForConditionalGeneration
            from diffusers.utils import load_image
            import torch
            from PIL import Image
            import cv2
            import numpy as np

            caption_model_id = "Salesforce/blip-image-captioning-large"
            if not quiet: print(f"   Loading Caption Model: {caption_model_id}...")
            
            processor = BlipProcessor.from_pretrained(caption_model_id)
            model = BlipForConditionalGeneration.from_pretrained(caption_model_id).to(device)
            
            # Check if video
            ext = input_path.lower().split('.')[-1]
            is_video = ext in ['mp4', 'mov', 'avi', 'mkv', 'webm', 'flv', 'gif']
            
            if is_video:
                 # Video Logic (Same as before)
                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened():
                    return "Unknown video content"
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                num_samples = 10
                indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
                captions = []
                for i, idx in enumerate(indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret: continue
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    
                    inputs = processor(pil_image, return_tensors="pt").to(device)
                    out = model.generate(**inputs)
                    frame_caption = processor.decode(out[0], skip_special_tokens=True)
                    captions.append(frame_caption)
                cap.release()
                return ", ".join(captions)
            else:
                # Image Logic
                raw_image = load_image(input_path).convert('RGB')
                inputs = processor(raw_image, return_tensors="pt").to(device)
                out = model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)
                if not quiet: print(f"   Detected: '{caption}'")
                return caption

        # ----------------------------------------------------------------
        # MODEL: Florence-2 (Default/SOTA)
        # ----------------------------------------------------------------
        else:
            from transformers import AutoProcessor, AutoModelForCausalLM
            from diffusers.utils import load_image
            import cv2
            import numpy as np
            import torch
            from PIL import Image
            
            # Load Florence-2 (SOTA Captioning, ~1.5GB)
            # Note: Requires timm and trust_remote_code=True
            caption_model_id = "microsoft/Florence-2-large"
            
            if not quiet: print(f"   Loading Caption Model: {caption_model_id}...")
            
            # Florence-2 needs device_map or manual to(device). 
            # On MPS, manual to(device) is safer for now.
            processor = AutoProcessor.from_pretrained(caption_model_id, trust_remote_code=True)
            
            # Use eager attention to avoid SDPA crashes on MPS/Mac with recent transformers
            model = AutoModelForCausalLM.from_pretrained(
                caption_model_id, 
                trust_remote_code=True,
                attn_implementation="eager"
            ).to(device) 
            
            # Task prompt for Florence-2
            task_prompt = "<MORE_DETAILED_CAPTION>"
            
            # Check if video
            ext = input_path.lower().split('.')[-1]
            is_video = ext in ['mp4', 'mov', 'avi', 'mkv', 'webm', 'flv', 'gif']
            
            if is_video:
                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened():
                    if not quiet: print(f"⚠️  Could not open video: {input_path}")
                    return "Unknown video content"
                    
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = total_frames / fps if fps > 0 else 0
                
                # Select 10 evenly distributed frames
                num_samples = 10
                indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
                
                captions = []
                if not quiet: print(f"   Analyzing {num_samples} frames from video ({duration:.1f}s)...")
                
                for i, idx in enumerate(indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret: continue
                    
                    # Convert BGR (OpenCV) to RGB (PIL)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    
                    # Generate caption (Florence-2)
                    inputs = processor(text=task_prompt, images=pil_image, return_tensors="pt") # .to(device) moved below
                    
                    # Ensure pixel_values are correct dtype if needed, though from_pretrained handles it usually.
                    # On MPS, float32 is safest.
                    if device.type == "mps":
                        inputs["pixel_values"] = inputs["pixel_values"].to(device, torch.float32)
                        inputs["input_ids"] = inputs["input_ids"].to(device)
                    else:
                        inputs = inputs.to(device)

                    # Disable cache to avoid MPS past_key_values crash
                    generated_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=1024,
                        do_sample=False,
                        num_beams=3,
                        use_cache=False,
                    )
                    
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(pil_image.width, pil_image.height))
                    frame_caption = parsed_answer[task_prompt]
                    
                    timestamp = idx / fps if fps > 0 else 0
                    captions.append(f"[{timestamp:.1f}s]: {frame_caption}")
                    if not quiet: print(f"   Frame {i+1}/{num_samples} ({timestamp:.1f}s): {frame_caption}")
                    
                cap.release()
                
                # Consolidated description
                summary = ", ".join([c.split(": ")[1] for c in captions])
                return summary 
                
            else:
                # Image handling
                raw_image = load_image(input_path).convert('RGB')
                
                inputs = processor(text=task_prompt, images=raw_image, return_tensors="pt") # .to(device) moved below
                
                # Ensure pixel_values are correct dtype if needed, though from_pretrained handles it usually.
                # On MPS, float32 is safest.
                if device.type == "mps":
                    inputs["pixel_values"] = inputs["pixel_values"].to(device, torch.float32)
                    inputs["input_ids"] = inputs["input_ids"].to(device)
                else:
                    inputs = inputs.to(device)

                # Disable cache to avoid MPS past_key_values crash
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    do_sample=False,
                    num_beams=3,
                    use_cache=False,
                )
                
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(raw_image.width, raw_image.height))
                caption = parsed_answer[task_prompt]
                
                if not quiet: print(f"   Detected: '{caption}'")
                return caption
                
    except ImportError as e:
        print(f"❌ Error: Missing dependencies for captioning. {e}")
        return None
    except Exception as e:
        print(f"❌ Caption generation failed: {e}")
        return None


def generate_edit(input_path, prompt, output_path, model_name="default", guidance_scale=7.5, image_guidance_scale=1.5, steps=50, unsafe=False):
    """
    Edit an existing image based on instructions using InstructPix2Pix.
    """
    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionXLInstructPix2PixPipeline
    from diffusers.utils import load_image
    
    # Resolve Model ID
    model_id = EDIT_MODELS.get(model_name.lower(), model_name)
    if model_name.lower() == "default": model_id = EDIT_MODELS["default"]
    
    print(f"🎨 Editing Image")
    print(f"   Model:     {model_id}")
    print(f"   Input:     {input_path}")
    print(f"   Instruct:  '{prompt}'")
    print(f"   Output:    {output_path}")
    print("") 

    try:
        device, dtype = get_optimal_device_and_dtype(quiet=True)
        
        # CRITICAL FIX: InstructPix2Pix (SD1.5 based) often produces black images on MPS with float16.
        # We force float32 for this specific pipeline on MPS to ensure valid output.
        if device.type == "mps":
            print(f"   ℹ️  MPS Detected: Forcing float32 for InstructPix2Pix to prevent black images.")
            dtype = torch.float32

        # Load Input Image
        image = load_image(input_path)
        image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        
        # Initialize Pipeline
        if "sdxl" in model_id.lower():
             # SDXL InstructPix2Pix
            pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None
            ).to(device)
            # Default scales for SDXL are different
            if guidance_scale == 7.5: guidance_scale = 7.0 
            if image_guidance_scale == 1.5: image_guidance_scale = 1.25
            
        else:
            # Standard InstructPix2Pix (SD 1.5 based)
            kwargs = {}
            if unsafe:
                kwargs["safety_checker"] = None
                
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                model_id, 
                torch_dtype=dtype,
                **kwargs
            ).to(device)
            
        # Optimization
        if device.type == "mps":
             pipe.enable_attention_slicing()
        
        # Generate
        print(f"✨ Applying edits... (Steps: {steps}, Text Guide: {guidance_scale}, Image Guide: {image_guidance_scale})")
        with torch.inference_mode():
            output = pipe(
                prompt,
                image=image,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale
            )
            
        result = output.images[0]
        
        # Safety Check
        if hasattr(output, "nsfw_content_detected") and output.nsfw_content_detected:
             if output.nsfw_content_detected[0]:
                print(f"⚠️  Warning: Potential NSFW content detected. Black image returned.")
                print(f"    ℹ️  This is frequently a false positive on Mac/MPS/CPU.")
                print(f"    👉  Please retry this command with '--unsafe' to see the image.")
                
        result.save(output_path)
        print(f"✅ Edited image saved to {output_path}")
        return True

    except Exception as e:
        print(f"❌ Edit failed: {e}")
        return False


def remove_background(input_path, output_path, model_name="remove-bg", silhouette=False):
    """
    Remove background using RMBG-1.4 (Transformer based).
    """
    import torch
    from transformers import AutoModelForImageSegmentation
    from torchvision.transforms.functional import normalize
    import numpy as np
    
    print(f"✂️  Removing Background")
    print(f"   Input:  {input_path}")
    print(f"   Output: {output_path}")
    if silhouette: print(f"   Mode:   Silhouette Maker")
    print("")

    try:
        device, dtype = get_optimal_device_and_dtype(quiet=True)
        
        # Load Model
        model_id = EDIT_MODELS.get(model_name, "briaai/RMBG-1.4")
        model = AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
        model.to(device)
        model.eval()
        
        # Load Image
        image = PIL.Image.open(input_path).convert("RGB")
        image = PIL.ImageOps.exif_transpose(image)
        original_size = image.size
        
        # Preprocess (Model expects 1024x1024)
        model_input_size = (1024, 1024)
        image_scaled = image.resize(model_input_size, PIL.Image.BILINEAR)
        im_tensor = torch.tensor(np.array(image_scaled)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        im_tensor = normalize(im_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]).to(device)
        
        # Inference
        print(f"🧠 Processing...")
        with torch.inference_mode():
            result = model(im_tensor)
        
        # Post-process Mask
        result = torch.nn.functional.interpolate(result[0][0], size=original_size[::-1], mode='bilinear')
        ma = torch.sigmoid(result)
        ma = (ma - ma.min()) / (ma.max() - ma.min())
        
        # Create PIL Mask from Tensor
        mask = ma.cpu().data.numpy()[0, 0]
        mask_pil = PIL.Image.fromarray((mask * 255).astype(np.uint8))
        
        # Apply Mask
        if silhouette:
            # Create black foreground
            foreground = PIL.Image.new("RGBA", original_size, (0, 0, 0, 255))
            final_image = PIL.Image.new("RGBA", original_size, (255, 255, 255, 0)) # Transparent
            final_image.paste(foreground, (0, 0), mask_pil)
        else:
            final_image = image.copy()
            final_image.putalpha(mask_pil)
            
        final_image.save(output_path, "PNG")
        print(f"✅ Saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Background removal failed: {e}")
        return False

def generate_long_bark(prompt, processor, model, device, voice_preset, sample_rate=24000):
    """
    Generate long-form audio with Bark by splitting text into sentences.
    Avoids 'history' chaining to prevent hallucinations/degradation.
    Concatenates independent chunks with the same voice preset.
    """
    import numpy as np
    
    # Smart split by sentence ending punctuation
    # Splits on: . ? ! or newline, keeping the punctuation
    sentences = re.split(r'([.?!]+|\n+)', prompt)
    
    # Recombine split text (sentence + punctuation)
    chunks = []
    current_chunk = ""
    
    for s in sentences:
        s = s.strip()
        if not s: continue
        
        # If it's punctuation, attach to previous
        if re.match(r'^[.?!]+$', s) or re.match(r'^\n+$', s):
            if chunks:
                chunks[-1] += s
            else:
                current_chunk += s # edge case: starts with punctuation
        else:
            # If current chunk is getting too long (Bark limit ~14s is roughly ~20-30 words depending on speed)
            # Heuristic: ~200 chars or ~30 words is safer upper bound
            if len(current_chunk) + len(s) > 200:
                chunks.append(current_chunk)
                current_chunk = s
            else:
                # Basic accumulation
                if current_chunk:
                    # If current didn't end with proper punctuation, add space
                    chunks.append(current_chunk)
                    current_chunk = s
                else:
                    current_chunk = s
    
    if current_chunk:
        chunks.append(current_chunk)
        
    print(f"   ✂️  Splitting long text into {len(chunks)} chunks for stable generation...")
    full_audio = []
    
    for i, text_chunk in enumerate(chunks):
        if not text_chunk.strip(): continue
        print(f"   ▶️  Generating chunk {i+1}/{len(chunks)}: '{text_chunk[:30]}...'")
        
        # Independent generation (Best stability)
        inputs = processor(text_chunk, voice_preset=voice_preset).to(device)
        audio_array = model.generate(**inputs, do_sample=True)
        audio_array = audio_array.cpu().numpy().squeeze()
        full_audio.append(audio_array)
        
        # Add a short silence between sentences for natural pacing (0.25s)
        silence_len = int(sample_rate * 0.25)
        full_audio.append(np.zeros(silence_len))

    # Concatenate all
    if not full_audio: return np.array([])
    return np.concatenate(full_audio)

def generate_audio(prompt, output_path, duration, sampling_rate, model_name="default", image_input=None, caption_model="florence", voice_preset="v2/en_speaker_6"):
    """Generate audio using MusicGen or AudioLDM (supports Image-to-Audio via captioning)."""
    
    model_id = AUDIO_MODELS.get(model_name.lower(), model_name)
    if model_name.lower() == "default": model_id = AUDIO_MODELS["default"]
    
    import sys
    sys.setrecursionlimit(50000) # Fix for Stable Audio / torchsde recursion on MPS
    
    device, dtype = get_optimal_device_and_dtype(quiet=True)
    
    # --- Image-to-Audio Logic (Captioning) ---
    if image_input:
        caption = generate_caption(image_input, device, model_type=caption_model)
        if caption:
            # If no user prompt is provided, use the caption directly
            if not prompt:
                print(f"   ℹ️  No prompt provided. Using generated caption as prompt.")
                full_prompt = caption
            else:
                # Combine User Prompt + Image Caption
                # Prompt is the "Action" or "Style", Caption is the "Content"
                full_prompt = f"{prompt}, inspired by {caption}"
                
            print(f"   Full Prompt: '{full_prompt}'")
            # Update prompt for downstream models
            prompt = full_prompt
        else:
            print(f"⚠️  Image analysis failed. Proceeding with text prompt only.")

    print(f"🎵 Generating Audio")
    print(f"   Model:    {model_id}")
    print(f"   Prompt:   '{prompt}'")
    if image_input: print(f"   Input Img: {image_input}")
    
    if "bark" in model_id.lower():
        print(f"   Duration: Auto (Text-based)")
    else:
        print(f"   Duration: {duration}s")
        
    print(f"   Sampling: {sampling_rate}Hz")
    print(f"   Output:   {output_path}")
    print("") # Spacer
    
    try:
        import torch
        import scipy.io.wavfile
        from transformers import pipeline
        from diffusers import AudioLDM2Pipeline
        
        # Logic for Different Model Types
        if "musicgen" in model_id.lower():
            # Use Transformers Pipeline
            print(f"   Loading MusicGen pipeline...")
            synthesizer = pipeline("text-to-audio", model_id, device=device)
            
            # MusicGen: ~50 tokens per sec estimate, strict max_new_tokens
            max_tokens = int(duration * 50) 
            print(f"🎵 Synthesizing audio... (MusicGen)")
            music = synthesizer(prompt, forward_params={"max_new_tokens": max_tokens})
            
            rate = music["sampling_rate"]
            audio_data = music["audio"]
            
            # Save raw WAV (transpose for scipy)
            scipy.io.wavfile.write(output_path + ".tmp.wav", rate, audio_data.T)
            src_path = output_path + ".tmp.wav"
            
        elif "audioldm" in model_id.lower():
            # Use Diffusers Pipeline for AudioLDM2
            # Workaround for transformers compatibility: explicit load of language_model
            from transformers import GPT2LMHeadModel
            print(f"   Loading AudioLDM2 pipeline components...")
            language_model = GPT2LMHeadModel.from_pretrained(model_id, subfolder="language_model").to(dtype=dtype)
            
            pipe = AudioLDM2Pipeline.from_pretrained(
                model_id, 
                language_model=language_model,
                torch_dtype=dtype
            )
            pipe.to(device)
            
            print(f"🎵 Synthesizing audio... (AudioLDM2)")
            audio = pipe(prompt, audio_length_in_s=duration).audios[0]
            rate = 16000 # AudioLDM default usually
            
            scipy.io.wavfile.write(output_path + ".tmp.wav", rate, audio.T)
            src_path = output_path + ".tmp.wav"
            
        elif "stable-audio" in model_id.lower():
            # Use Diffusers Pipeline for Stable Audio
            from diffusers import StableAudioPipeline, EulerDiscreteScheduler
            
            print(f"   Loading StableAudioPipeline...")
            pipe = StableAudioPipeline.from_pretrained(model_id, torch_dtype=dtype)
            pipe.to(device)
            
            # Switch scheduler to EulerDiscrete to avoid torchsde recursion error on MPS
            print(f"   ℹ️  Swapping scheduler to EulerDiscrete (MPS optimization)")
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            
            print(f"🎵 Synthesizing audio... (Stable Audio)")
            # Stable Audio takes 'audio_end_in_s'
            audio = pipe(prompt, audio_start_in_s=0.0, audio_end_in_s=duration, num_inference_steps=50).audios[0]
            rate = 44100 # Standard for Stable Audio Open
            
            # Ensure numpy
            import torch
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().float().numpy()
            
            scipy.io.wavfile.write(output_path + ".tmp.wav", rate, audio.T)
            src_path = output_path + ".tmp.wav"
            
        elif "bark" in model_id.lower():
            # Use Transformers for Bark
            from transformers import BarkModel, AutoProcessor
            print(f"   Loading Bark models...")
            
            # Bark on MPS often fails with float16 (Unsupported data type 'float16'). Force float32.
            bark_dtype = dtype
            if device.type == "mps":
                bark_dtype = torch.float32
                
            processor = AutoProcessor.from_pretrained(model_id)
            model = BarkModel.from_pretrained(model_id, torch_dtype=bark_dtype).to(device)
            
            print(f"🎵 Synthesizing audio... (Bark)")
            if duration > 14:
                 print(f"   (Note: Bark generates max ~14s sequences per history block. Output will be shorter than {duration}s)")
            
            print(f"""   💡 Tip:
   *  Lyrics: Use '♪' for singing (e.g., `♪ Hello World ♪`).
   *  Effects: Use tags like `[laughter]`, `[cheers]`, `[music]`, `[sighs]`, `[gasps]`, `[clears throat]`, `—` (hesitation).
   *  Plain text without these tokens will usually be spoken as speech.
   *  Voice: Using preset '{voice_preset}'. Change with --voice-preset (e.g. 'v2/fr_speaker_1').
   *  Example: `python ai-media.py -a --audio-model bark -p "♪ Hello World ♪ [laughter]"`""")
            
            # Decide if Long-Form is needed
            # Heuristic: Text > 150 chars OR user requested > 15 seconds
            # Bark roughly does ~14s max.
            is_long = len(prompt) > 150 or duration > 15.0
            
            if is_long:
                print(f"   📜 Long text detected. Using chunked generation.")
                audio_array = generate_long_bark(prompt, processor, model, device, voice_preset)
            else:
                # Use user-specified voice preset
                inputs = processor(prompt, voice_preset=voice_preset).to(device)
                # Bark output shape: [1, length]
                audio_array = model.generate(**inputs) 
                audio_array = audio_array.cpu().numpy().squeeze()
            
            rate = model.generation_config.sample_rate # 24000
            
            # Scipy write
            scipy.io.wavfile.write(output_path + ".tmp.wav", rate, audio_array)
            src_path = output_path + ".tmp.wav"
            
        else:
            print(f"❌ Unknown model type for audio: {model_id}")
            return False

        # Conversion / Final Save
        # If user asked for .mp3, convert. If .wav, rename.
        must_convert = not output_path.lower().endswith(".wav")
        
        if must_convert:
            import subprocess
            cmd = ["ffmpeg", "-y", "-i", src_path, output_path, "-loglevel", "error"]
            try:
                subprocess.run(cmd, check=True)
                os.remove(src_path)
                print(f"✅ Converted and saved to {output_path}")
            except subprocess.CalledProcessError:
                print(f"⚠️  FFmpeg conversion failed. Saved as WAV: {src_path}")
        else:
            os.rename(src_path, output_path)
            print(f"✅ Audio saved to {output_path}")
            
        return True
        
    except ImportError as e:
        print(f"❌ Error: Missing dependencies or import failed. {e}")
        return False
    except Exception as e:
        print(f"❌ Audio generation failed: {e}")
        return False


def get_video_encoding_params(output_path):
    """Get FFmpeg encoding parameters based on output file extension.
    
    Returns a list of FFmpeg arguments for video codec, pixel format, and audio codec.
    Supports: mp4, mkv, mov, webm, wmv, avi
    """
    ext = os.path.splitext(output_path)[1].lower()
    
    # Default: H.264 for broad compatibility
    if ext in ['.mp4', '.m4v']:
        return ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"]
    elif ext == '.mkv':
        return ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"]
    elif ext == '.mov':
        return ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"]
    elif ext == '.webm':
        return ["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-c:a", "libopus", "-b:v", "2M"]
    elif ext == '.wmv':
        return ["-c:v", "wmv2", "-c:a", "wmav2", "-b:v", "2M"]
    elif ext == '.avi':
        return ["-c:v", "mpeg4", "-pix_fmt", "yuv420p", "-c:a", "mp3", "-b:v", "2M"]
    else:
        # Fallback to H.264
        return ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"]


def generate_video(prompt, output_path, duration, width, height, model_name="default", image_input=None, audio_prompt=None):
    """Generate video (Text-to-Video or Image-to-Video) with optional Audio."""
    
    # Resolve Model ID
    base_model = VIDEO_MODELS.get(model_name.lower(), model_name)
    if model_name.lower() == "default": base_model = VIDEO_MODELS["default"]
    
    # Handle Image-to-Video Logic
    is_i2v = True if image_input else False
    
    if is_i2v:
        # Check if we need to switch to an I2V variant
        if "cogvideox" in base_model.lower():
            model_id = "THUDM/CogVideoX-5b-I2V"
        elif "stable-video-diffusion" in base_model.lower() or model_name.lower() == "svd":
            model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
        else:
            print(f"⚠️  Warning: Model '{model_name}' ({base_model}) may not support Image-to-Video.")
            print(f"   Switching to 'svd' (Stable Video Diffusion) as fallback.")
            model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
    else:
        # Text-to-Video
        model_id = base_model

    print(f"🎬 Generating Video ({'Image-to-Video' if is_i2v else 'Text-to-Video'})")
    print(f"   Model:    {model_id}")
    print(f"   Prompt:   '{prompt}'")
    if is_i2v: print(f"   Input Img: {image_input}")
    if audio_prompt: print(f"   Audio:    '{audio_prompt}' (Will generate and mux)")
    print(f"   Duration: {duration}s")
    print("") # Spacer
    
    # Determine actual video output path (temp if mixing audio)
    video_out = output_path
    if audio_prompt:
        video_out = output_path + ".temp_video.mp4"
    
    try:
        import torch
        from diffusers import (
            DiffusionPipeline, 
            DPMSolverMultistepScheduler, 
            CogVideoXImageToVideoPipeline,
            StableVideoDiffusionPipeline,
            utils
        )
        from diffusers.utils import export_to_video, load_image
        
        device, dtype = get_optimal_device_and_dtype(quiet=True)
        
        # MPS FIX: These models need Float32/CPU on MPS
        # - Text-to-Video models: Float16 corrupts output
        # - SVD: 3D convolutions cause "Invalid buffer size" errors on MPS
        mps_incompatible_models = ["ms-1.7b", "text-to-video-ms-1.7b", "zeroscope", "stable-video-diffusion", "cogvideox"]
        is_mps_incompatible = any(m in model_id.lower() for m in mps_incompatible_models)
        
        if device.type == "mps" and is_mps_incompatible:
            # SVD needs CPU entirely (3D convolutions fail on MPS)
            if "stable-video-diffusion" in model_id.lower():
                print("⚠️  MPS Compatibility: SVD requires CPU on Apple Silicon.")
                print("   (3D convolutions cause 'Invalid buffer size' on MPS)")
                device = torch.device("cpu")
            else:
                print("⚠️  MPS Compatibility: Using Float32 for correct video output.")
            dtype = torch.float32
        
        print("⚠️  Video generation is resource intensive.")
        
        # --- Stage 1: Video Generation ---
        
        # Load Pipeline
        if "cogvideox" in model_id.lower() and is_i2v:
            pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
        elif "stable-video-diffusion" in model_id.lower():
            pipe = StableVideoDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None)
        else:
            # Generic / Text-to-Video - try fp16 variant first, fallback to default
            try:
                if dtype == torch.float16:
                    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, variant="fp16")
                else:
                    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
            except Exception:
                # Model doesn't have fp16 variant, load without it
                pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
        
        # Scheduler Optimization
        if hasattr(pipe, "scheduler"):
            try:
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            except: pass 
        
        # Device/Memory Optimization
        if device.type == "cpu":
            pipe.to(device)
        else:
            pipe.enable_model_cpu_offload()
            if device.type == "mps":
                pipe.enable_attention_slicing()
        
        # Generate Frames
        print(f"🎬 Rendering video frames... (This will be slow)")
        if is_i2v:
            init_image = load_image(image_input)
            init_image = init_image.resize((width, height))
            
            if "stable-video-diffusion" in model_id.lower():
                video_frames = pipe(init_image, decode_chunk_size=8).frames[0]
            else:
                video_frames = pipe(prompt=prompt, image=init_image, num_frames=49).frames[0]
        else:
            num_frames = int(duration * 16)
            video_frames = pipe(prompt, num_inference_steps=25, num_frames=num_frames).frames[0]
        
        # Save Video (raw export - may need re-encoding for compatibility)
        temp_raw_video = video_out + ".raw.mp4"
        export_to_video(video_frames, temp_raw_video, fps=7 if "stable-video-diffusion" in model_id.lower() else 16) 
        
        # Re-encode with FFmpeg for universal playback (format-aware)
        import subprocess
        encoding_params = get_video_encoding_params(video_out)
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", temp_raw_video,
                *encoding_params,
                video_out
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(temp_raw_video)
            print(f"✅ Video track saved to {video_out}")
        except Exception as e:
            # Fallback: use raw video if FFmpeg fails
            os.rename(temp_raw_video, video_out)
            print(f"⚠️  Video saved (may require VLC to play): {video_out}")
        
        # --- Stage 2 & 3: Audio Generation & Muxing ---
        if audio_prompt:
            print("🔊 Generating Audio track...")
            audio_out = output_path + ".temp_audio.wav"
            # Use default music model or let user specify? Use default for now or expose args if needed.
            # We reuse generate_audio function!
            # Note: We need to pass sampling rate, let's default to standard 32k or use global default
            audio_success = generate_audio(audio_prompt, audio_out, duration, 32000, model_name="default")
            
            if audio_success:
                print("🔗 Muxing Video and Audio...")
                # FFmpeg merge
                # -c:v copy (copy video stream)
                # -c:a aac (encode audio to aac for mp4 compatibility)
                # -shortest (cut to shortest stream, usually video)
                import subprocess
                cmd = [
                    "ffmpeg", "-y",
                    "-i", video_out,
                    "-i", audio_out,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-shortest",
                    output_path,
                    "-loglevel", "error"
                ]
                try:
                    subprocess.run(cmd, check=True)
                    print(f"✅ Final merged video saved to {output_path}")
                except subprocess.CalledProcessError:
                    print(f"❌ Muxing failed. Check FFmpeg.")
            else:
                print("❌ Audio generation failed. Returning silent video (renaming temp).")
                os.rename(video_out, output_path)
                
        return True
        
    except Exception as e:
        print(f"❌ Video generation failed: {e}")
        # Clean temp if exists
        if audio_prompt and os.path.exists(video_out):
            try: os.remove(video_out)
            except: pass
        return False


def ensure_paths(output_path):
    """Create parent directories if needed."""
    if output_path:
        p = Path(output_path)
        if p.parent:
            p.parent.mkdir(parents=True, exist_ok=True)


# --- Performance Tracking ---

class PerformanceTracker:
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

    def record_image(self, model, width, height, device, time_taken, cpu=0, ram=0):
        dev_str = device.type if hasattr(device, 'type') else str(device)
        key = f"{model}|{dev_str}|{width}x{height}"
        if "image" not in self.data: self.data["image"] = {}
        
        # New Logic: Keep only average_time
        # Re-average strategy: (last_average + new_time) / 2
        entry = self.data["image"].get(key, {})
        
        if "average_time" in entry:
            new_avg = (entry["average_time"] + time_taken) / 2.0
            entry["average_time"] = new_avg
            # Rolling average for resources too (optional, but good for stability)
            entry["average_cpu"] = (entry.get("average_cpu", cpu) + cpu) / 2.0
            entry["average_ram"] = (entry.get("average_ram", ram) + ram) / 2.0
        else:
            entry = {"average_time": time_taken, "average_cpu": cpu, "average_ram": ram}
            
        self.data["image"][key] = entry
        self._save()

    def record_linear(self, category, model, device, duration, time_taken, width=None, height=None, cpu=0, ram=0):
        """Record Audio/Video generation using rolling average rate (seconds to gen / seconds of content)."""
        dev_str = device.type if hasattr(device, 'type') else str(device)
        # For video, resolution also matters, so we include it in key
        if category == "video":
            key = f"{model}|{dev_str}|{width}x{height}"
        else:
            key = f"{model}|{dev_str}"
            
        if category not in self.data: self.data[category] = {}
        
        current_rate = time_taken / duration if duration > 0 else 0
        entry = self.data[category].get(key, {})
        
        if "average_rate" in entry:
            new_rate = (entry["average_rate"] + current_rate) / 2.0
            entry["average_rate"] = new_rate
            entry["average_cpu"] = (entry.get("average_cpu", cpu) + cpu) / 2.0
            entry["average_ram"] = (entry.get("average_ram", ram) + ram) / 2.0
        else:
            entry = {"average_rate": current_rate, "average_cpu": cpu, "average_ram": ram}
            
        self.data[category][key] = entry
        self._save()

    def estimate_image(self, model, width, height, device):
        dev_str = device.type if hasattr(device, 'type') else str(device)
        key = f"{model}|{dev_str}|{width}x{height}"
        stats = self.data.get("image", {}).get(key)
        if stats and "average_time" in stats:
            return stats["average_time"], stats.get("average_cpu", 0), stats.get("average_ram", 0)
        return None, 0, 0

    def estimate_linear(self, category, model, device, duration, width=None, height=None):
        dev_str = device.type if hasattr(device, 'type') else str(device)
        if category == "video":
            key = f"{model}|{dev_str}|{width}x{height}"
        else:
            key = f"{model}|{dev_str}"
            
        stats = self.data.get(category, {}).get(key)
        if stats and "average_rate" in stats:
            return stats["average_rate"] * duration, stats.get("average_cpu", 0), stats.get("average_ram", 0)
        return None, 0, 0

class ResourceMonitor:
    """Monitors CPU and RAM usage in a background thread."""
    def __init__(self, interval=0.5):
        self.interval = interval
        self.running = False
        self.thread = None
        self.cpu_readings = []
        self.ram_readings = []
        
        try:
            import psutil
            self.psutil = psutil
        except ImportError:
            self.psutil = None
            print("⚠️  'psutil' not found. Resource monitoring disabled.")

    def _monitor(self):
        import time
        while self.running:
            if self.psutil:
                # CPU percent (blocking for interval? No, we set interval=None for non-blocking since we sleep manually)
                # But psutil.cpu_percent with interval=None is instantaneous since last call.
                # Better to use interval=0.1 inside the blocking call or manage it. 
                # We will sleep manually.
                cpu = self.psutil.cpu_percent(interval=None)
                ram = self.psutil.virtual_memory().used / (1024**3) # GB
                self.cpu_readings.append(cpu)
                self.ram_readings.append(ram)
            time.sleep(self.interval)

    def __enter__(self):
        if self.psutil:
            # Prime CPU counter
            self.psutil.cpu_percent(interval=None)
            self.running = True
            import threading
            self.thread = threading.Thread(target=self._monitor, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def get_averages(self):
        if not self.cpu_readings: return 0, 0
        avg_cpu = sum(self.cpu_readings) / len(self.cpu_readings)
        avg_ram = sum(self.ram_readings) / len(self.ram_readings)
        return avg_cpu, avg_ram

# --- Upscaling Logic ---

# --- Upscaling Logic ---

def check_resources_and_confirm(w, h, f, dev):
    """
    Checks if the target upscale resolution is safe for the current system resources.
    Returns True if safe/confirmed, False if user aborts.
    """
    try:
        import psutil
    except ImportError:
        return True # Cannot check without psutil
        
    target_w = int(w * f)
    target_h = int(h * f)
    target_pixels = target_w * target_h
    megapixels = target_pixels / 1_000_000
    
    # Estimate RAM (Very rough heuristic for Float32 Latent Pipeline)
    # Empirical Rule of Thumb: 1MP output needs ~1GB RAM on CPU for safe execution.
    estimated_ram_gb = (megapixels * 0.8) if dev == "cpu" else (megapixels * 0.4) 
    
    vm = psutil.virtual_memory()
    available_gb = vm.available / (1024**3)
    
    is_huge = megapixels > 25  # > 5K/6K image
    is_tight = estimated_ram_gb > (available_gb * 0.9)
    
    if is_huge or is_tight:
        print("\n⚠️  RESOURCE WARNING: High-Resolution Upscale Detected")
        print(f"   Input:  {w}x{h}")
        print(f"   Target: {target_w}x{target_h} ({megapixels:.1f} MP)")
        print(f"   Device: {dev.upper()}")
        print(f"   Est. RAM Required: ~{estimated_ram_gb:.1f} GB")
        print(f"   Available RAM:      {available_gb:.1f} GB")
        
        if is_tight:
             print("   🔴 WARNING: This may cause massive swapping or system freeze.")
        elif is_huge:
             print("   🟠 WARNING: This resolution is extremely high (Billboard size).")
        
        if os.environ.get("AI_MEDIA_FORCE", "0") == "1":
             print("   (Proceeding due to Force flag)")
             return True
             
        confirm = input("\n   Do you want to proceed? [y/N]: ").strip().lower()
        return confirm == 'y'
    return True


def simple_upscale_image(image_path, output_path, factor=2.0):
    """Simple non-AI image upscaling using PIL Lanczos interpolation.
    
    Fast, preserves original quality without AI hallucination.
    Uses Lanczos algorithm which is considered the best for downscaling/upscaling.
    """
    from PIL import Image
    
    print(f"🔍 Simple Upscaling Image: {image_path}")
    print(f"   Method: PIL Lanczos (No AI)")
    print(f"   Factor: {factor}x")
    
    # Pre-flight check: Avoid wasting processing time if output exists
    if Path(output_path).exists():
        if os.environ.get("AI_MEDIA_FORCE", "0") != "1":
            print(f"\n⚠️  Output file already exists: {output_path}")
            confirm = input("   Overwrite? [y/N]: ").strip().lower()
            if confirm != 'y':
                print("❌ Aborted to prevent overwrite.")
                return False
        else:
            print(f"   ⚠️  Overwriting existing file (--force).")
    
    try:
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        target_w = int(orig_w * factor)
        target_h = int(orig_h * factor)
        
        print(f"   {orig_w}x{orig_h} → {target_w}x{target_h}")
        
        upscaled = image.resize((target_w, target_h), Image.LANCZOS)
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        upscaled.save(output_path)
        
        print(f"✅ Simple upscaled image saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Simple upscaling failed: {e}")
        return False


def simple_upscale_video(video_path, output_path, factor=2.0):
    """Simple non-AI video upscaling using FFmpeg scale filter.
    
    Fast, uses FFmpeg's high-quality Lanczos scaling.
    """
    import subprocess
    
    print(f"🔍 Simple Upscaling Video: {video_path}")
    print(f"   Method: FFmpeg Lanczos (No AI)")
    print(f"   Factor: {factor}x")
    
    # Pre-flight check: Avoid wasting processing time if output exists
    if Path(output_path).exists():
        if os.environ.get("AI_MEDIA_FORCE", "0") != "1":
            print(f"\n⚠️  Output file already exists: {output_path}")
            confirm = input("   Overwrite? [y/N]: ").strip().lower()
            if confirm != 'y':
                print("❌ Aborted to prevent overwrite.")
                return False
        else:
            print(f"   ⚠️  Overwriting existing file (--force).")
    
    try:
        # Get video dimensions first
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            str(video_path)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to probe video: {result.stderr}")
            return False
        
        orig_w, orig_h = map(int, result.stdout.strip().split('x'))
        target_w = int(orig_w * factor)
        target_h = int(orig_h * factor)
        
        # FFmpeg requires even dimensions
        target_w = target_w if target_w % 2 == 0 else target_w + 1
        target_h = target_h if target_h % 2 == 0 else target_h + 1
        
        print(f"   {orig_w}x{orig_h} → {target_w}x{target_h}")
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # FFmpeg upscale with Lanczos (format-aware encoding)
        encoding_params = get_video_encoding_params(str(output_path))
        # For upscaling, we want to preserve audio and use high quality
        # Filter out audio codec from encoding_params and use -c:a copy
        video_params = [p for i, p in enumerate(encoding_params) if not (encoding_params[i-1:i] == ["-c:a"] or p in ["aac", "libopus", "wmav2", "mp3"])]
        video_params = [p for p in video_params if p != "-c:a"]
        
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", f"scale={target_w}:{target_h}:flags=lanczos",
            *video_params,
            "-c:a", "copy",  # Preserve audio
            str(output_path),
            "-loglevel", "warning"
        ]
        
        print("   ⏳ Processing with FFmpeg...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ FFmpeg failed: {result.stderr}")
            return False
        
        print(f"✅ Simple upscaled video saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Simple upscaling failed: {e}")
        return False

def upscale_image_file(image_path, output_path, strength=0.0, factor=2.0):
    """Upscale an image using smart multi-stage AI upscaling.
       
       Uses optimal combination of x4, x2 AI passes + final Lanczos resize:
       - 6x  → 4x AI + 1.5x Lanczos
       - 8x  → 4x AI + 2x AI
       - 10x → 4x AI + 2x AI + 1.25x Lanczos
       
       Args:
           image_path: Path to image to upscale
           output_path: Output file path
           strength: Noise strength 0.0-1.0 (x4 upscaler only). Higher values add more
                     noise, allowing the model to generate more details/texture but
                     potentially diverging from the original. Default 0.0 keeps original.
           factor: Upscale factor (e.g., 2.0, 4.0, 6.0, 8.0)
    """
    
    
    # Select Model based on factor
    # <= 2.0x -> use x2 Latent Upscaler (Fast, Faithful)
    # > 2.0x  -> use x4 Upscaler (Detailed)
    use_x2_model = (factor <= 2.0)
    model_id = IMAGE_MODELS['upscaler_x2'] if use_x2_model else IMAGE_MODELS['upscaler']
    
    print(f"🚀 Upscaling Image: {image_path}")
    print(f"   Model: {model_id}")
    print(f"   Target Factor: {factor}x")
    
    # Noise level for x4 upscaler (maps 0.0-1.0 to 0-100)
    # Default to 0 (faithful to original) - user can increase for more detail generation
    noise_level = int(strength * 100)
    
    if use_x2_model:
        if strength > 0:
            print(f"   ⚠️  Note: x2 Latent Upscaler does not support --upscale-strength. Ignored.")
    else:
        if strength > 0:
            print(f"   Noise Level: {noise_level} (strength={strength} - more creative/details)")
        else:
            print(f"   Noise Level: {noise_level} (faithful to original)")
    
    # Pre-flight check: Avoid wasting processing time if output exists
    if Path(output_path).exists():
        if os.environ.get("AI_MEDIA_FORCE", "0") != "1":
            print(f"\n⚠️  Output file already exists: {output_path}")
            confirm = input("   Overwrite? [y/N]: ").strip().lower()
            if confirm != 'y':
                print("❌ Aborted to prevent overwrite.")
                return False
        else:
            print(f"   ⚠️  Overwriting existing file (--force).")
    
    if factor > 4.0:
        print(f"   ℹ️  Factor > 4x detected. This will require multiple AI passes.")
    
    try:
        from diffusers import StableDiffusionUpscalePipeline, StableDiffusionLatentUpscalePipeline
        from PIL import Image
        import torch

        device, dtype = get_optimal_device_and_dtype()
        
        # MPS GLOBAL FIX: Force CPU for ALL Upscaling (x2 and x4)
        # Reason 1 (x2): "View size not compatible" (Tensor Stride errors)
        # Reason 2 (x4): "MPSNDArrayMatrixMultiplication ... too large for kernel" (Driver Limit)
        # Only CPU can handle these massive tensors safely.
        if device.type == "mps":
            print("   ⚠️  MPS Compatibility: Switching to CPU for Upscaling (Avoids Kernel Crashes).")
            device = torch.device("cpu")
            dtype = torch.float32  # BFloat16 causes hangs on Apple Silicon CPU
        
        # Load Image
        try:
            image = Image.open(image_path).convert("RGB")
            orig_w, orig_h = image.size
        except Exception as e:
            print(f"❌ Error loading source image: {e}")
            return False

        # --- RESOURCE SAFETY CHECK ---
        if not check_resources_and_confirm(orig_w, orig_h, factor, device.type):
             print("❌ Upload aborted by user.")
             return False

        # ===== SMART MULTI-STAGE UPSCALING =====
        # Strategy: Use optimal combination of 4x, 2x AI passes + final Lanczos resize
        # Examples:
        #   6x  → 4x AI + 1.5x Lanczos
        #   8x  → 4x AI + 2x AI  
        #   10x → 4x AI + 2x AI + 1.25x Lanczos
        #   3x  → 2x AI + 1.5x Lanczos
        #   5x  → 4x AI + 1.25x Lanczos
        
        # Calculate optimal pass sequence
        passes = []
        remaining = factor
        
        while remaining >= 2.0:
            if remaining >= 4.0:
                passes.append(('x4', 4.0))
                remaining /= 4.0
            elif remaining >= 2.0:
                passes.append(('x2', 2.0))
                remaining /= 2.0
        
        # Remaining factor will be handled by Lanczos at the end
        final_lanczos_factor = remaining if remaining > 1.0 else None
        
        # Display the plan
        print(f"\n   📋 Upscale Plan:")
        for i, (model_type, scale) in enumerate(passes, 1):
            print(f"      Pass {i}: {model_type} AI ({scale}x)")
        if final_lanczos_factor:
            print(f"      Final: Lanczos resize ({final_lanczos_factor:.2f}x)")
        print("")
        
        # Load both pipelines if needed (lazy loading)
        pipe_x2 = None
        pipe_x4 = None
        
        def get_pipeline(model_type):
            nonlocal pipe_x2, pipe_x4
            
            if model_type == 'x2':
                if pipe_x2 is None:
                    print(f"   🔗 Loading x2 Latent Upscaler...")
                    pipe_x2 = StableDiffusionLatentUpscalePipeline.from_pretrained(
                        IMAGE_MODELS['upscaler_x2'],
                        torch_dtype=dtype,
                    )
                    if device.type != "cpu":
                        pipe_x2.enable_model_cpu_offload()
                    else:
                        pipe_x2.to(device)
                    if hasattr(pipe_x2, 'vae') and hasattr(pipe_x2.vae, 'enable_tiling'):
                        pipe_x2.vae.enable_tiling()
                return pipe_x2, 64  # alignment requirement
            else:
                if pipe_x4 is None:
                    print(f"   🔗 Loading x4 Upscaler...")
                    pipe_x4 = StableDiffusionUpscalePipeline.from_pretrained(
                        IMAGE_MODELS['upscaler'],
                        torch_dtype=dtype,
                        variant="fp16" if dtype == torch.float16 else None
                    )
                    if device.type != "cpu":
                        pipe_x4.enable_model_cpu_offload()
                    else:
                        pipe_x4.to(device)
                    if hasattr(pipe_x4, 'vae') and hasattr(pipe_x4.vae, 'enable_tiling'):
                        pipe_x4.vae.enable_tiling()
                return pipe_x4, 8  # alignment requirement
        
        # Upscaling prompts (neutral to avoid hallucination)
        upscale_prompt = "sharp, high resolution"
        negative_prompt = "blur, noise, artifacts, distortion, jpeg artifacts, oversaturated, low quality"
        
        current_image = image
        
        for pass_idx, (model_type, step_scale) in enumerate(passes, 1):
            print("")
            print("=" * 60)
            print(f"🎨 Pass {pass_idx}/{len(passes)}: {model_type} AI Upscaling ({step_scale}x)")
            print("=" * 60)
            print("")
            
            pipe, alignment = get_pipeline(model_type)
            
            # --- DIMENSION ALIGNMENT FIX ---
            img_w, img_h = current_image.size
            pad_w = (alignment - (img_w % alignment)) % alignment
            pad_h = (alignment - (img_h % alignment)) % alignment
            
            if pad_w > 0 or pad_h > 0:
                padded_w, padded_h = img_w + pad_w, img_h + pad_h
                final_w, final_h = int(img_w * step_scale), int(img_h * step_scale)
                print(f"   ℹ️  Temporarily padding {img_w}x{img_h} → {padded_w}x{padded_h} ({alignment}px alignment required)")
                print(f"       Will crop back to {final_w}x{final_h} after upscaling.")
                
                # Create padded image (reflect padding for seamless edges)
                padded_image = Image.new("RGB", (padded_w, padded_h))
                padded_image.paste(current_image, (0, 0))
                # Mirror-fill the padding area for better edge blending
                if pad_w > 0:
                    right_edge = current_image.crop((img_w - pad_w, 0, img_w, img_h))
                    padded_image.paste(right_edge.transpose(Image.FLIP_LEFT_RIGHT), (img_w, 0))
                if pad_h > 0:
                    bottom_edge = current_image.crop((0, img_h - pad_h, img_w, img_h))
                    padded_image.paste(bottom_edge.transpose(Image.FLIP_TOP_BOTTOM), (0, img_h))
            else:
                padded_image = current_image
            
            # Run upscaling with model-specific parameters
            if model_type == 'x2':
                # x2 Latent Upscaler: doesn't support noise_level or negative_prompt
                upscaled_result = pipe(
                    prompt=upscale_prompt, 
                    image=padded_image, 
                    num_inference_steps=50,
                ).images[0]
            else:
                # x4 Upscaler: supports noise_level and negative_prompt
                upscaled_result = pipe(
                    prompt=upscale_prompt, 
                    negative_prompt=negative_prompt,
                    image=padded_image, 
                    num_inference_steps=50,
                    noise_level=noise_level,
                ).images[0]
            
            # Crop back to target dimensions (remove padding effect)
            target_w_pass = int(img_w * step_scale)
            target_h_pass = int(img_h * step_scale)
            if upscaled_result.size != (target_w_pass, target_h_pass):
                current_image = upscaled_result.crop((0, 0, target_w_pass, target_h_pass))
            else:
                current_image = upscaled_result
            
            print(f"   ✓ Pass {pass_idx} complete: {current_image.size[0]}x{current_image.size[1]}")
        
        # Final Resize to exact factor using Lanczos (high-quality non-AI resize)
        target_w = int(orig_w * factor)
        target_h = int(orig_h * factor)
        
        if current_image.size != (target_w, target_h):
            actual_lanczos = target_w / current_image.size[0]
            print(f"\n   ↘️  Lanczos resize ({actual_lanczos:.2f}x) to exact target: {target_w}x{target_h}")
            current_image = current_image.resize((target_w, target_h), Image.LANCZOS)
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        current_image.save(output_path)
        print(f"\n✅ Upscaled image saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Upscaling failed: {e}")
        return False

def upscale_video_file(video_path, output_path, strength=0.0, factor=2.0):
    """Upscale video by extracting frames, upscaling them (recursively if needed), and stitching back."""
    print(f"🚀 Upscaling Video: {video_path}")
    print(f"   Factor: {factor}x")
    
    # Pre-flight check: Avoid wasting processing time if output exists
    if Path(output_path).exists():
        if os.environ.get("AI_MEDIA_FORCE", "0") != "1":
            print(f"\n⚠️  Output file already exists: {output_path}")
            confirm = input("   Overwrite? [y/N]: ").strip().lower()
            if confirm != 'y':
                print("❌ Aborted to prevent overwrite.")
                return False
        else:
            print(f"   ⚠️  Overwriting existing file (--force).")
    
    if factor > 4.0:
        print(f"   ℹ️  Factor > 4x detected. This will require multiple AI passes per frame.")
    
    try:
        import cv2
        import shutil
        import subprocess
        
        # --- RESOURCE SAFETY CHECK (Get dims from video metadata first) ---
        cap_chk = cv2.VideoCapture(str(video_path))
        if cap_chk.isOpened():
             v_w = int(cap_chk.get(cv2.CAP_PROP_FRAME_WIDTH))
             v_h = int(cap_chk.get(cv2.CAP_PROP_FRAME_HEIGHT))
             cap_chk.release()
             
             # Reuse global checker
             # Note: For video, we might want to check device type, but we haven't loaded torch/device yet.
             # We can assume 'mps' if on Mac for the check warning purposes, or just check CPU RAM first.
             # Let's peek device quick
             import torch
             d_type = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
             
             if not check_resources_and_confirm(v_w, v_h, factor, d_type):
                  print("❌ Upload aborted by user.")
                  return False
        else:
             print("⚠️  Could not read video metadata for resource check.")
             
        # 1. Create temp directory
        temp_dir = Path("temp_upscale_frames")
        if temp_dir.exists(): shutil.rmtree(temp_dir)
        temp_dir.mkdir()
        
        # 2. Extract Frames
        print("🎞️  Extracting frames...")
        cam = cv2.VideoCapture(str(video_path))
        fps = cam.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        while True:
            ret, frame = cam.read()
            if not ret: break
            
            frame_path = temp_dir / f"frame_{frame_count:05d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_count += 1
            
        cam.release()
        print(f"   Extracted {frame_count} frames.")
        
        # 3. Load Pipeline
        from diffusers import StableDiffusionUpscalePipeline, StableDiffusionLatentUpscalePipeline
        import torch
        from PIL import Image
        
        device, dtype = get_optimal_device_and_dtype()
        
        # MPS GLOBAL FIX: Force CPU for Video Upscaling too
        if device.type == "mps":
            print("   ⚠️  MPS Compatibility: Switching to CPU for Upscaling.")
            device = torch.device("cpu")
            dtype = torch.float32  # BFloat16 causes hangs on Apple Silicon CPU

        # Select Model based on factor
        use_x2_model = (factor <= 2.0)
        model_id = IMAGE_MODELS['upscaler_x2'] if use_x2_model else IMAGE_MODELS['upscaler']
        step_scale = 2.0 if use_x2_model else 4.0

        print(f"🔗 Loading Upscale Model ({'x2 Latent' if use_x2_model else 'x4 Std'})...")
        
        if use_x2_model:
            pipe = StableDiffusionLatentUpscalePipeline.from_pretrained(
                model_id, 
                torch_dtype=dtype, 
            )
        else:
            pipe = StableDiffusionUpscalePipeline.from_pretrained(
                model_id, 
                torch_dtype=dtype, 
                variant="fp16" if dtype == torch.float16 else None
            )
            
        # Memory Optimizations
        if device.type != "cpu":
             pipe.enable_model_cpu_offload() 
        else:
             pipe.to(device)
         
        # Enable tiling on CPU to survive high-res
        if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_tiling'):
            pipe.vae.enable_tiling()
        
        # 4. Process Frames
        print("🎨 Upscaling frames...")
        for i in range(frame_count):
            input_f = temp_dir / f"frame_{i:05d}.png"
            output_f = temp_dir / f"upscaled_{i:05d}.png"
            
            img = Image.open(input_f).convert("RGB")
            orig_w, orig_h = img.size
            
            # Recursive Loop per frame
            current_img = img
            current_scale = 1.0
            
            while current_scale < factor:
                # --- DIMENSION ALIGNMENT FIX ---
                # x2 latent upscaler: 64px, x4 upscaler: 8px
                alignment = 64 if use_x2_model else 8
                frame_w, frame_h = current_img.size
                pad_w = (alignment - (frame_w % alignment)) % alignment
                pad_h = (alignment - (frame_h % alignment)) % alignment
                
                if pad_w > 0 or pad_h > 0:
                    padded_w, padded_h = frame_w + pad_w, frame_h + pad_h
                    padded_img = Image.new("RGB", (padded_w, padded_h))
                    padded_img.paste(current_img, (0, 0))
                    if pad_w > 0:
                        right_edge = current_img.crop((frame_w - pad_w, 0, frame_w, frame_h))
                        padded_img.paste(right_edge.transpose(Image.FLIP_LEFT_RIGHT), (frame_w, 0))
                    if pad_h > 0:
                        bottom_edge = current_img.crop((0, frame_h - pad_h, frame_w, frame_h))
                        padded_img.paste(bottom_edge.transpose(Image.FLIP_TOP_BOTTOM), (0, frame_h))
                else:
                    padded_img = current_img
                
                upscaled_result = pipe(prompt="High quality", image=padded_img, num_inference_steps=15).images[0]
                
                # Crop back to target dimensions
                target_w_pass = int(frame_w * step_scale)
                target_h_pass = int(frame_h * step_scale)
                if upscaled_result.size != (target_w_pass, target_h_pass):
                    current_img = upscaled_result.crop((0, 0, target_w_pass, target_h_pass))
                else:
                    current_img = upscaled_result
                    
                current_scale *= step_scale
            
            # Final Resize
            target_w = int(orig_w * factor)
            target_h = int(orig_h * factor)
            if current_img.size != (target_w, target_h):
                current_img = current_img.resize((target_w, target_h), Image.LANCZOS)

            current_img.save(output_f)
            
            print(f"   Frame {i+1}/{frame_count} done.", end='\r')
            
        print(f"\n✅ All frames upscaled.")

        # 5. Stitch
        print("🔗 stitching video...")
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Get format-specific encoding params
        encoding_params = get_video_encoding_params(output_path)
        # Remove audio codec params since we're creating from images
        video_params = [p for i, p in enumerate(encoding_params) if not (encoding_params[i-1:i] == ["-c:a"] or p in ["aac", "libopus", "wmav2", "mp3"])]
        video_params = [p for p in video_params if p != "-c:a"]
        
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(temp_dir / "upscaled_%05d.png"),
            *video_params,
            output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        shutil.rmtree(temp_dir)
        print(f"✅ Upscaled video saved to {output_path}")
        return True

    except ImportError:
        print("❌ Missing dependencies (opencv-python, etc).")
        return False
    except Exception as e:
        print(f"❌ Video upscaling failed: {e}")
        return False

def parse_upscale_factor(val):
    """Parse upscale factor string (e.g., '2x', '4', '1.5') -> float."""
    if not val: return 2.0 # Default
    val = val.lower().strip().replace('x', '')
    try:
        f = float(val)
        if f <= 0: raise ValueError
        return f
    except:
        print(f"⚠️  Invalid upscale factor '{val}'. Using default 2.0x.")
        return 2.0

def convert_image_file(input_path, target):
    """Convert image format using PIL (no AI).
    
    Args:
        input_path: Source image file
        target: Output path, extension (.png), or format (png)
    """
    from PIL import Image
    from pathlib import Path
    
    # Parse target format
    target = target.strip()
    
    # Determine output path and format
    if '/' in target or '\\' in target or len(target) > 6:
        # It's a full path
        output_path = target
    elif target.startswith('.'):
        # It's an extension like ".png"
        name = Path(input_path).stem
        output_path = f"{name}{target}"
    else:
        # It's just a format like "png" or "PNG"
        name = Path(input_path).stem
        output_path = f"{name}.{target.lower()}"
    
    print(f"🔄 Converting Image: {input_path}")
    print(f"   Output: {output_path}")
    
    # Overwrite protection
    if Path(output_path).exists():
        if os.environ.get("AI_MEDIA_FORCE", "0") != "1":
            confirm = input(f"⚠️  '{output_path}' exists. Overwrite? [y/N]: ").strip().lower()
            if confirm != 'y':
                print("❌ Aborted.")
                return False
    
    try:
        img = Image.open(input_path)
        
        # Handle transparency for formats that don't support it
        output_ext = Path(output_path).suffix.lower()
        if output_ext in ['.jpg', '.jpeg'] and img.mode in ['RGBA', 'P']:
            print(f"   ℹ️  Converting RGBA → RGB (JPEG doesn't support transparency)")
            img = img.convert('RGB')
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        print(f"✅ Converted image saved to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def convert_image_ffmpeg(input_path, target):
    """Convert image format using FFmpeg."""
    from pathlib import Path
    import subprocess
    
    target = target.strip()
    if '/' in target or '\\' in target or len(target) > 6:
        output_path = target
    elif target.startswith('.'):
        output_path = f"{Path(input_path).stem}{target}"
    else:
        output_path = f"{Path(input_path).stem}.{target.lower()}"
    
    print(f"🔄 Converting Image (FFmpeg): {input_path}")
    print(f"   Output: {output_path}")
    
    # Overwrite protection
    if Path(output_path).exists():
        if os.environ.get("AI_MEDIA_FORCE", "0") != "1":
            confirm = input(f"⚠️  '{output_path}' exists. Overwrite? [y/N]: ").strip().lower()
            if confirm != 'y':
                print("❌ Aborted.")
                return False
    
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-i", input_path, output_path], 
                      check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Converted image saved to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def convert_video_file(input_path, target):
    """Convert video format using FFmpeg."""
    from pathlib import Path
    import subprocess
    
    target = target.strip()
    if '/' in target or '\\' in target or len(target) > 6:
        output_path = target
    elif target.startswith('.'):
        output_path = f"{Path(input_path).stem}{target}"
    else:
        output_path = f"{Path(input_path).stem}.{target.lower()}"
    
    print(f"🎬 Converting Video: {input_path}")
    print(f"   Output: {output_path}")
    
    # Overwrite protection
    if Path(output_path).exists():
        if os.environ.get("AI_MEDIA_FORCE", "0") != "1":
            confirm = input(f"⚠️  '{output_path}' exists. Overwrite? [y/N]: ").strip().lower()
            if confirm != 'y':
                print("❌ Aborted.")
                return False
    
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        encoding_params = get_video_encoding_params(output_path)
        subprocess.run(["ffmpeg", "-y", "-i", input_path, *encoding_params, output_path], 
                      check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Converted video saved to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def convert_audio_file(input_path, target):
    """Convert audio format using FFmpeg."""
    from pathlib import Path
    import subprocess
    
    target = target.strip()
    if '/' in target or '\\' in target or len(target) > 6:
        output_path = target
    elif target.startswith('.'):
        output_path = f"{Path(input_path).stem}{target}"
    else:
        output_path = f"{Path(input_path).stem}.{target.lower()}"
    
    print(f"🎵 Converting Audio: {input_path}")
    print(f"   Output: {output_path}")
    
    # Overwrite protection
    if Path(output_path).exists():
        if os.environ.get("AI_MEDIA_FORCE", "0") != "1":
            confirm = input(f"⚠️  '{output_path}' exists. Overwrite? [y/N]: ").strip().lower()
            if confirm != 'y':
                print("❌ Aborted.")
                return False
    
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-i", input_path, output_path], 
                      check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Converted audio saved to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

# --- Interactive Mode ---

def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def show_header(title="AI-Media"):
    """Show interactive mode header."""
    print(f"\n{'═'*60}")
    print(f"🎨 {title}")
    print(f"{'═'*60}\n")

def run_self_command(cmd_string):
    """Run ai-media.py with the given command arguments (cross-platform safe).
    
    Accepts a command string like: -gd "path/to/file" -cm model
    Properly handles quoted paths on both Windows and Unix.
    """
    import subprocess
    import shlex
    
    script_path = os.path.abspath(__file__)
    
    # Display what we're running
    print(f"\n🚀 Running: ai-media.py {cmd_string}\n")
    
    # Parse the command string properly
    # On Windows, shlex.split with posix=False preserves quotes, so we strip them manually
    if os.name == 'nt':
        # Use posix=True even on Windows to properly handle quotes
        # But escape backslashes first to prevent them being treated as escape chars
        escaped = cmd_string.replace('\\', '\\\\')
        try:
            args = shlex.split(escaped, posix=True)
            # Restore backslashes in paths
            args = [arg.replace('\\\\', '\\') for arg in args]
        except ValueError:
            # Fallback: manually parse
            args = []
            current = ""
            in_quotes = False
            for char in cmd_string:
                if char == '"' and not in_quotes:
                    in_quotes = True
                elif char == '"' and in_quotes:
                    in_quotes = False
                elif char == ' ' and not in_quotes:
                    if current:
                        args.append(current)
                        current = ""
                else:
                    current += char
            if current:
                args.append(current)
    else:
        # Unix/Mac - standard shlex parsing works fine
        args = shlex.split(cmd_string)
    
    # Run with subprocess (handles paths correctly on all platforms)
    subprocess.run([sys.executable, script_path] + args)

# --- Interactive Navigation Helpers ---

def get_key():
    """Read a single key press from stdin (cross-platform)."""
    if os.name == 'nt':  # Windows
        import msvcrt
        ch = msvcrt.getch()
        
        # Handle special keys (arrow keys, etc.)
        if ch in (b'\x00', b'\xe0'):  # Special key prefix
            ch2 = msvcrt.getch()
            if ch2 == b'H': return 'UP'
            if ch2 == b'P': return 'DOWN'
            if ch2 == b'G': return 'HOME'
            if ch2 == b'O': return 'END'
            if ch2 == b'I': return 'PAGE_UP'
            if ch2 == b'Q': return 'PAGE_DOWN'
            return ch2.decode('utf-8', errors='ignore')
        
        if ch == b'\r': return 'ENTER'
        if ch == b'\x03': raise KeyboardInterrupt  # CTRL+C
        return ch.decode('utf-8', errors='ignore')
    else:  # Unix/Mac
        import tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            if ch == '\x1b':  # Escape sequence
                ch2 = sys.stdin.read(1)
                
                # Handle [ sequences
                if ch2 == '[':
                    ch3 = sys.stdin.read(1)
                    if ch3 == 'A': return 'UP'
                    if ch3 == 'B': return 'DOWN'
                    if ch3 == 'H': return 'HOME'
                    if ch3 == 'F': return 'END'
                    # Handle [1~ (Home), [4~ (End), [5~ (PgUp), [6~ (PgDn)
                    if ch3 in ['1', '4', '5', '6']:
                        ch4 = sys.stdin.read(1)
                        if ch4 == '~':
                            if ch3 == '1': return 'HOME'
                            if ch3 == '4': return 'END'
                            if ch3 == '5': return 'PAGE_UP'
                            if ch3 == '6': return 'PAGE_DOWN'
                
                # Handle O sequences (OH = Home, OF = End)
                if ch2 == 'O':
                    ch3 = sys.stdin.read(1)
                    if ch3 == 'H': return 'HOME'
                    if ch3 == 'F': return 'END'
                    
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
        if ch == '\r' or ch == '\n': return 'ENTER'
        if ch == '\x03': raise KeyboardInterrupt  # CTRL+C
        return ch

def prompt_menu(prompt, options, allow_back=True, default_index=0):
    """
    Show interactive menu with arrow key navigation.
    Returns: value of selected option or None (if Back/Exit)
    """
    # ... (header omitted, unchanged)
    # Prepare items list
    items = list(options)
    if allow_back:
        items.append(("⬅️  Back", None))
    elif not options:
        # Fallback if empty
        return None

    current_idx = default_index if 0 <= default_index < len(items) else 0

    # Hide cursor
    print("\033[?25l", end="")
    
    if prompt:
        print(f"{prompt}")
    
    # ANSI constants
    UP = "\033[F"
    CLEAR_LINE = "\033[K"
    CYAN = "\033[96m" 
    RESET = "\033[0m"
    DIM = "\033[90m"

    # Reserve space for menu
    for _ in items:
        print()
    
    # Move cursor back up to start of menu
    print(UP * len(items), end="")

    try:
        while True:
            # Render Menu
            for i, (label, val) in enumerate(items):
                # Alignment logic
                is_selected = (i == current_idx)
                prefix = " > " if is_selected else "   "
                number = f"{i+1}." if i < len(options) else "0."
                
                # Ensure label spacing is consistent
                display_label = label
                
                # Formatting
                if is_selected:
                    line = f"{CYAN}{prefix}{number:<4}  {display_label}{RESET}"
                else:
                    line = f"{prefix}{number:<4}  {display_label}"
                
                print(f"{line}{CLEAR_LINE}")
            
            # Move cursor back up to start for next redraw
            print(UP * len(items), end="", flush=True)

            # Handle Input
            key = get_key()
            
            if key == 'UP':
                current_idx = (current_idx - 1) % len(items)
            elif key == 'DOWN':
                current_idx = (current_idx + 1) % len(items)
            elif key == 'PAGE_UP' or key == '[':
                current_idx = (current_idx - 3) % len(items)
            elif key == 'PAGE_DOWN' or key == ']':
                current_idx = (current_idx + 3) % len(items)
            elif key == 'HOME':
                current_idx = 0
            elif key == 'END':
                current_idx = len(items) - 1
            elif key == 'ENTER':
                # Confirm selection
                break
            elif key in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                # Direct numeric jump (0-9)
                num = int(key)
                if 1 <= num <= len(options):
                    current_idx = num - 1
                    continue
            elif key == '0' and allow_back:
                 current_idx = len(items) - 1 # Jump to last item (Back)
                 
    except KeyboardInterrupt:
        # Clean exit on CTRL+C
        print(RESET + "\n" * len(items)) # Move past menu
        print("\033[?25h", end="") # Show cursor
        return None
    finally:
        # Restore cursor
        print(RESET + "\n" * len(items)) # Move past menu
        print("\033[?25h", end="") # Show cursor

    # Re-print selection statically for history
    selected_label, selected_val = items[current_idx]
    return selected_val

def prompt_choice(prompt, options, allow_back=True):
    """Wrapper for prompt_menu (backward usage compatibility)."""
    return prompt_menu(prompt, options, allow_back)

def prompt_text(prompt, default=None, required=True):
    """Get text input from user."""
    default_str = f" [{default}]" if default else ""
    while True:
        try:
            value = input(f"{prompt}{default_str}: ").strip()
            if not value and default:
                return default
            if not value and required:
                print("   This field is required.")
                continue
            return value
        except KeyboardInterrupt:
            return None

def browse_files(start_dir="."):
    """Interactively browse file system and return selected file path."""
    current_dir = os.path.abspath(start_dir)
    
    while True:
        try:
            items = os.listdir(current_dir)
        except PermissionError:
            print(f"❌ Permission denied: {current_dir}")
            current_dir = os.path.dirname(current_dir)
            continue
            
        # Separate dirs and files
        dirs = []
        files = []
        for item in items:
            if item.startswith('.'): continue # Skip hidden
            full_path = os.path.join(current_dir, item)
            if os.path.isdir(full_path):
                dirs.append(item)
            else:
                files.append(item)
        
        dirs.sort()
        files.sort()
        
        menu_items = []
        # Add parent directory option if not at root
        if os.path.dirname(current_dir) != current_dir:
            menu_items.append(("📂 .. (Up Directory)", ".."))
        
        for d in dirs:
            menu_items.append((f"📁 {d}/", d))
            
        for f in files:
            menu_items.append((f"📄 {f}", f))
        
        # Display Menu
        print(f"\n📂 Location: {current_dir}")
        choice = prompt_choice(None, menu_items, allow_back=True)
        
        if choice is None: # Back/Cancel
            return None
        
        if choice == "..":
            current_dir = os.path.dirname(current_dir)
        else:
            selected_path = os.path.join(current_dir, choice)
            if os.path.isdir(selected_path):
                current_dir = selected_path
            else:
                return selected_path

def prompt_file(prompt, must_exist=True):
    """Get file path input from user with browsing support."""
    while True:
        # If looking for existing file, offer browse option
        if must_exist:
            print(f"\n{prompt}")
            options = [
                ("📂 Browse Files", "browse"),
                ("⌨️  Enter Path Manually", "manual"),
                ("🔙 Cancel", None)
            ]
            method = prompt_choice(None, options, allow_back=False) # We handle None manually
            
            if method is None:
                return None
            
            if method == "browse":
                path = browse_files()
                if path:
                    return path
                continue # If cancelled browsing, return to method choice
                
            elif method == "manual":
                pass # Fall through to manual input
        
        # Manual Input Logic
        try:
            path = input(f"Enter file path: ").strip()
            if not path:
                print("   This field is required.")
                continue
            if must_exist and not os.path.exists(path):
                print(f"   File not found: {path}")
                continue
            return path
        except KeyboardInterrupt:
            return None

def run_interactive(jump_point=None):
    """Run interactive menu mode.
    
    Args:
        jump_point: Optional jump path (e.g., 'image/sdxl', '1/2', 'audio/bark')
    """
    
    # Jump point mappings: name -> (menu_action, submenu_value)
    JUMP_POINTS = {
        # By name
        'image': ('image', None),
        'image/sdxl': ('image', 'sdxl'),
        'image/sd15': ('image', 'sd-1.5'),
        'image/flux': ('image', 'flux'),
        'image/flux-dev': ('image', 'flux-dev'),
        'video': ('video', None),
        'video/zeroscope': ('video', 'zeroscope'),
        'video/modelscope': ('video', 'ms-1.7b'),
        'video/cogvideox': ('video', 'cogvideox'),
        'video/svd': ('video', 'svd'),
        'audio': ('audio', None),
        'audio/musicgen': ('audio', 'musicgen-medium'),
        'audio/musicgen-small': ('audio', 'musicgen-small'),
        'audio/musicgen-large': ('audio', 'musicgen-large'),
        'audio/audioldm2': ('audio', 'audioldm2'),
        'audio/bark': ('audio', 'bark'),
        'transform': ('transform', None),
        'transform/edit': ('transform', 'edit'),
        'transform/rembg': ('transform', 'rembg'),
        'transform/silhouette': ('transform', 'silhouette'),
        'upscale': ('upscale', None),
        'convert': ('convert', None),
        'caption': ('caption', None),
        'sysinfo': ('sysinfo', None),
        # By number (matching menu order)
        '1': ('image', None),
        '1/1': ('image', 'sdxl'),
        '1/2': ('image', 'sd-1.5'),
        '1/3': ('image', 'flux'),
        '1/4': ('image', 'flux-dev'),
        '2': ('video', None),
        '2/1': ('video', 'zeroscope'),
        '2/2': ('video', 'ms-1.7b'),
        '2/3': ('video', 'cogvideox'),
        '2/4': ('video', 'svd'),
        '3': ('audio', None),
        '3/1': ('audio', 'musicgen-medium'),
        '3/2': ('audio', 'musicgen-small'),
        '3/3': ('audio', 'musicgen-large'),
        '3/4': ('audio', 'audioldm2'),
        '3/5': ('audio', 'bark'),
        '4': ('transform', None),
        '4/1': ('transform', 'edit'),
        '4/2': ('transform', 'rembg'),
        '4/3': ('transform', 'silhouette'),
        '5': ('upscale', None),
        '6': ('convert', None),
        '7': ('caption', None),
        '8': ('sysinfo', None),
    }
    
    # Parse jump point
    initial_action = None
    initial_model = None
    if jump_point and jump_point != 'menu':
        jp_lower = jump_point.lower()
        if jp_lower in JUMP_POINTS:
            initial_action, initial_model = JUMP_POINTS[jp_lower]
        else:
            print(f"⚠️  Unknown jump point: {jump_point}")
            print("   Run with --help or see README for valid jump points.")
    
    def system_info_menu():
        """Display system information."""
        clear_screen()
        show_header("System Information")
        
        import platform
        import psutil
        import torch
        
        # OS Info
        os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
        
        # CPU Info
        cpu_count = psutil.cpu_count(logical=True)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # RAM Info
        mem = psutil.virtual_memory()
        ram_total = f"{mem.total / (1024**3):.1f} GB"
        ram_used = f"{mem.used / (1024**3):.1f} GB"
        ram_avail = f"{mem.available / (1024**3):.1f} GB"
        ram_percent = f"{mem.percent}%"
        
        # GPU Info
        if torch.backends.mps.is_available():
            gpu_info = "MPS (Apple Silicon) ✅ Available"
        elif torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_info = f"CUDA ({gpu_name}, {vram:.1f} GB VRAM) ✅ Available"
        else:
            gpu_info = "CPU Only (No Acceleration Detected)"
            
        print(f"💻 OS:       {os_info}")
        print(f"🧠 CPU:      {cpu_count} Cores (Usage: {cpu_percent}%)")
        print(f"💾 RAM:      {ram_avail} Available / {ram_total} Total ({ram_percent} Used)")
        print(f"🎮 GPU:      {gpu_info}")
        print()
        
        input("Press Enter to return...")

    def main_menu():
        """Show main menu and return action."""
        clear_screen()
        show_header("AI-Media Interactive Mode")
        print("📋 What would you like to do?\n")
        
        options = [
            ("🖼️   Generate Image", "image"),
            ("🎬  Generate Video", "video"),
            ("🎵  Generate Audio", "audio"),
            ("✨  Transform/Edit Image", "transform"),
            ("📈  Upscale Media", "upscale"),
            ("🔄  Convert Media", "convert"),
            ("📝  Generate Caption", "caption"),
            ("ℹ️   System Information", "sysinfo"),
            ("❌  Exit", None)
        ]
        
        return prompt_choice(None, options, allow_back=False)
    
    def image_menu(preset_model=None):
        """Image generation submenu."""
        clear_screen()
        show_header("Image Generation")
        
        # Model selection (skip if preset)
        if preset_model:
            model = preset_model
            print(f"📦 Model: {model}\n")
        else:
            print("📦 Select Model:\n")
            # Build model options - hide Mac-specific notes on Windows
            if os.name == 'nt':  # Windows
                model_options = [
                    ("SDXL Turbo (Default, Fast) ~8GB", "sdxl"),
                    ("SD 1.5 (Lightweight) ~4GB", "sd-1.5"),
                    ("Flux Schnell (High Quality) ~12GB", "flux"),
                    ("Flux Dev (Professional) ~16GB", "flux-dev"),
                ]
            else:  # Mac/Linux
                model_options = [
                    ("SDXL Turbo (Default, Fast) ~8GB", "sdxl"),
                    ("SD 1.5 (Lightweight) ~4GB", "sd-1.5"),
                    ("Flux Schnell (High Quality, Slow on Mac) ~12GB", "flux"),
                    ("Flux Dev (Professional, Very Slow on Mac) ~16GB", "flux-dev"),
                ]
            model = prompt_choice("Model", model_options)
            if model is None:
                return
        
        # Prompt
        print()
        prompt = prompt_text("📝 Enter prompt")
        if prompt is None:
            return
        
        # Resolution
        print("\n📐 Select Resolution:\n")
        size_options = [
            ("720p (1280x720)", "720p"),
            ("1080p (1920x1080)", "1080p"),
            ("4K (3840x2160)", "4k"),
            ("64x64 (Quick Test)", "64x64"),
            ("Custom Resolution", "custom"),
        ]
        size = prompt_choice("Size", size_options)
        if size is None:
            return
            
        if size == "custom":
            print()
            size = prompt_text("Enter resolution (e.g. 512x512, 1024x768)")
            if not size:
                return
        
        # Orientation
        print("\n🔄 Select Orientation:\n")
        orient_options = [
            ("Landscape (Default)", "landscape"),
            ("Portrait", "portrait"),
            ("Square", "square"),
        ]
        orientation = prompt_choice("Orientation", orient_options)
        if orientation is None:
            return
        
        # Output
        print()
        output = prompt_text("💾 Output filename (or press Enter for auto)", required=False)
        
        # Build and run command
        cmd = f"-i -p \"{prompt}\" -s {size} -otn {orientation} --image-model {model}"
        if output:
            cmd += f" -o \"{output}\""
        
        run_self_command(cmd)
        input("\nPress Enter to continue...")
    
    def video_menu(preset_model=None):
        """Video generation submenu."""
        clear_screen()
        show_header("Video Generation")
        
        # Model selection (skip if preset)
        if preset_model:
            model = preset_model
            print(f"📦 Model: {model}\n")
        else:
            print("📦 Select Model:\n")
            model_options = [
                ("Zeroscope (Default, No Watermarks)", "zeroscope"),
                ("ModelScope (General Purpose, Has Watermarks)", "ms-1.7b"),
                ("CogVideoX (State of the Art, Slow) ~15GB", "cogvideox"),
                ("Stable Video Diffusion (Image-to-Video only)", "svd"),
            ]
            model = prompt_choice("Model", model_options)
            if model is None:
                return
        
        # Prompt or Input Image (for SVD)
        prompt = None
        input_image = None
        
        if model == 'svd':
             print("\n🖼️ Select Input Image for SVD:")
             input_image = prompt_file("Input Image")
             if not input_image:
                 return
        else:
            print()
            prompt = prompt_text("📝 Enter prompt")
            if prompt is None:
                return
        
        # Duration
        print("\n⏱️ Select Duration:\n")
        length_options = [
            ("2 seconds (Quick)", "2s"),
            ("5 seconds", "5s"),
            ("10 seconds", "10s"),
            ("Custom Duration", "custom"),
        ]
        
        # Build and run command (partial logic needed inside/after check)
        # We need duration first.
        
        length = prompt_choice("Duration", length_options)
        if length is None:
            return
            
        if length == "custom":
            print()
            length = prompt_text("Enter duration (e.g. 3s, 15s)")
            if not length:
                return

        # Output
        print()
        output = prompt_text("💾 Output filename (or press Enter for auto)", required=False)
        
        # Build Command
        cmd = f"-v -s {length} --video-model {model}"
        if prompt:
            cmd += f" -p \"{prompt}\""
        if input_image:
            cmd += f" -ii \"{input_image}\""
        if output:
             cmd += f" -o \"{output}\""
             
        run_self_command(cmd)
        input("\nPress Enter to continue...")

    
    def audio_menu(preset_model=None):
        """Audio generation submenu."""
        clear_screen()
        show_header("Audio Generation")
        
        # Model selection (skip if preset)
        if preset_model:
            model = preset_model
            print(f"📦 Model: {model}\n")
        else:
            print("📦 Select Model:\n")
            model_options = [
                ("MusicGen Medium (Default)", "musicgen-medium"),
                ("MusicGen Small (Fast)", "musicgen-small"),
                ("MusicGen Large (High Quality)", "musicgen-large"),
                ("AudioLDM2 (Sound Effects)", "audioldm2"),
                ("Bark (Speech/TTS)", "bark"),
            ]
            model = prompt_choice("Model", model_options)
            if model is None:
                return
        
        # Prompt
        print()
        if model == "bark":
            prompt = prompt_text("📝 Enter text to speak")
        else:
            prompt = prompt_text("📝 Enter audio description")
        if prompt is None:
            return
        
        # Duration (not for Bark)
        length = "10s"
        if model != "bark":
            print("\n⏱️ Select Duration:\n")
            length_options = [
                ("5 seconds", "5s"),
                ("10 seconds", "10s"),
                ("30 seconds", "30s"),
                ("Custom Duration", "custom"),
            ]
            length = prompt_choice("Duration", length_options)
            if length is None:
                return
            if length == "custom":
                print()
                length = prompt_text("Enter duration (e.g. 8s, 1m)")
                if not length:
                    return
        
        # Output
        print()
        output = prompt_text("💾 Output filename (or press Enter for auto)", required=False)
        
        # Build and run command
        cmd = f"-a -p \"{prompt}\" -l {length} --audio-model {model}"
        if output:
            cmd += f" -o \"{output}\""
        
        run_self_command(cmd)
        input("\nPress Enter to continue...")
    
    def transform_menu(preset_operation=None):
        """Image transformation submenu."""
        clear_screen()
        show_header("Transform/Edit Image")
        
        # Input file
        print("📂 Select input image:\n")
        input_file = prompt_file("Enter image path")
        if input_file is None:
            return
        
        # Operation (skip if preset)
        if preset_operation:
            operation = preset_operation
            print(f"✨ Operation: {operation}\n")
        else:
            print("\n✨ Select Operation:\n")
            op_options = [
                ("Creative Edit (AI Instruction)", "edit"),
                ("Remove Background", "rembg"),
                ("Create Silhouette", "silhouette"),
            ]
            operation = prompt_choice("Operation", op_options)
            if operation is None:
                return
        
        # Build command based on operation
        if operation == "edit":
            print()
            instruction = prompt_text("📝 Enter edit instruction (e.g., 'Make it anime')")
            if instruction is None:
                return
            cmd = f"-ti \"{input_file}\" -tp \"{instruction}\""
        elif operation == "rembg":
            cmd = f"-ti \"{input_file}\" --remove-background"
        elif operation == "silhouette":
            cmd = f"-ti \"{input_file}\" --remove-background --silhouette"
        
        # Output
        print()
        output = prompt_text("💾 Output filename (or press Enter for auto)", required=False)
        if output:
            cmd += f" -o \"{output}\""
        
        run_self_command(cmd)
        input("\nPress Enter to continue...")
    
    def upscale_menu():
        """Upscale media submenu."""
        clear_screen()
        show_header("Upscale Media")
        
        # Media type
        print("📁 Select Media Type:\n")
        type_options = [
            ("Image", "image"),
            ("Video", "video"),
        ]
        media_type = prompt_choice("Type", type_options)
        if media_type is None:
            return
        
        # Input file
        print("\n📂 Select input file:\n")
        input_file = prompt_file("Enter file path")
        if input_file is None:
            return
        
        # Upscale factor
        print("\n📈 Select Upscale Factor:\n")
        factor_options = [
            ("2x (Default)", "2x"),
            ("4x", "4x"),
            ("Custom Factor", "custom"),
        ]
        factor = prompt_choice("Factor", factor_options)
        if factor == "custom":
            print()
            factor = prompt_text("Enter factor (e.g. 3x, 8x)")
            if not factor:
                return
        if factor is None:
            return
        
        # Method
        print("\n⚙️ Select Method:\n")
        method_options = [
            ("AI Upscale (High Quality, Slow)", "ai"),
            ("Simple Upscale (Fast, Lanczos)", "simple"),
        ]
        method = prompt_choice("Method", method_options)
        if method is None:
            return
        
        # Build command
        if media_type == "image":
            cmd = f"-ui \"{input_file}\" -uf {factor}"
        else:
            cmd = f"-uv \"{input_file}\" -uf {factor}"
        
        if method == "simple":
            cmd += " -su"
        
        run_self_command(cmd)
        input("\nPress Enter to continue...")
    
    def convert_menu():
        """Convert media submenu."""
        clear_screen()
        show_header("Convert Media")
        
        # Media type
        print("📁 Select Media Type:\n")
        type_options = [
            ("Image", "image"),
            ("Video", "video"),
            ("Audio", "audio"),
        ]
        media_type = prompt_choice("Type", type_options)
        if media_type is None:
            return
        
        # Input file
        print("\n📂 Select input file:\n")
        input_file = prompt_file("Enter file path")
        if input_file is None:
            return
        
        # Target format
        print("\n🎯 Select Target Format:\n")
        if media_type == "image":
            format_options = [
                ("PNG", "png"),
                ("JPG", "jpg"),
                ("WebP", "webp"),
            ]
        elif media_type == "video":
            format_options = [
                ("MP4", "mp4"),
                ("WebM", "webm"),
                ("AVI", "avi"),
            ]
        else:
            format_options = [
                ("MP3", "mp3"),
                ("WAV", "wav"),
                ("FLAC", "flac"),
            ]
        target_format = prompt_choice("Format", format_options)
        if target_format is None:
            return
        
        # Build command
        if media_type == "image":
            cmd = f"-ci \"{input_file}\" -cit {target_format}"
        elif media_type == "video":
            cmd = f"-cv \"{input_file}\" -cvt {target_format}"
        else:
            cmd = f"-ca \"{input_file}\" -cat {target_format}"
        
        run_self_command(cmd)
        input("\nPress Enter to continue...")
    
    def caption_menu(preset_model=None):
        """Generate caption submenu."""
        clear_screen()
        show_header("Generate Caption")
        
        # Input file
        print("📂 Select input image or video:\n")
        input_file = prompt_file("Enter file path")
        if input_file is None:
            return
        
        # Model (skip if preset)
        if preset_model:
            model = preset_model
            print(f"📦 Model: {model}\n")
        else:
            print("\n📦 Select Caption Model:\n")
            model_options = [
                ("Florence-2 (Default, SOTA)", "florence"),
                ("BLIP", "blip"),
            ]
            model = prompt_choice("Model", model_options)
            if model is None:
                return
        
        # Build command
        cmd = f"-gd \"{input_file}\" -cm {model}"
        
        run_self_command(cmd)
        input("\nPress Enter to continue...")
    
    # Main loop
    first_run = True
    while True:
        # Use jump point on first run if provided
        if first_run and initial_action:
            action = initial_action
            first_run = False
        else:
            action = main_menu()
        
        if action is None:
            print("\n👋 Goodbye!")
            break
        elif action == "image":
            image_menu(initial_model if first_run or initial_action == 'image' else None)
            initial_model = None  # Clear after first use
        elif action == "video":
            video_menu(initial_model if first_run or initial_action == 'video' else None)
            initial_model = None
        elif action == "audio":
            audio_menu(initial_model if first_run or initial_action == 'audio' else None)
            initial_model = None
        elif action == "transform":
            transform_menu(initial_model if first_run or initial_action == 'transform' else None)
            initial_model = None
        elif action == "upscale":
            upscale_menu()
        elif action == "convert":
            convert_menu()
        elif action == "caption":
            caption_menu(initial_model if first_run or initial_action == 'caption' else None)
            initial_model = None
        elif action == "sysinfo":
            system_info_menu()

# --- Test Runner ---


def run_tests(verbose=False):
    """Run test suite from testing.json."""
    import shlex
    import subprocess
    
    # Use global test state for CTRL+C handling
    global _test_state
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(script_dir, "testing.json")
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        sys.exit(1)
    
    with open(test_file, "r") as f:
        data = json.load(f)
    
    tests = data.get("tests", [])
    if not tests:
        print("❌ No tests found in testing.json")
        sys.exit(1)
    
    # Warning prompt
    print(f"\n{'='*60}")
    print(f"⚠️  WARNING: Test Suite")
    print(f"{'='*60}")
    print(f"   • Found {len(tests)} test(s) to run")
    print(f"   • This may take a LONG time (30+ minutes)")
    print(f"   • Uses significant system resources (CPU, RAM, GPU)")
    print(f"   • Will download ALL models if not already cached")
    print(f"   • Models can be 2-30GB each")
    print(f"   • Press CTRL+C at any time to interrupt")
    print(f"{'='*60}")
    
    try:
        choice = input(f"\n   Continue? [Y/n]: ").lower().strip()
        if choice in ['n', 'no']:
            print("❌ Test cancelled.")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\n❌ Test cancelled.")
        sys.exit(0)
    
    print(f"\n{'='*60}")
    print(f"🧪 Running {len(tests)} test(s)")
    print(f"{'='*60}\n")
    
    passed = 0
    failed = 0
    results = []
    
    # Set global test state for CTRL+C handler
    _test_state['active'] = True
    _test_state['total'] = len(tests)
    _test_state['passed'] = 0
    _test_state['failed'] = 0
    
    for i, test in enumerate(tests):
        test_name = test.get("name", f"Test {i+1}")
        command = test.get("command", "")
        expected_inputs = test.get("expectedInputItems", [])
        expected_outputs = test.get("expectedOutputItems", [])
        
        print(f"\n{'-'*50}")
        print(f"📋 Test {i+1}/{len(tests)}: {test_name}")
        print(f"{'-'*50}")
        
        test_passed = True
        failure_reason = None
        
        # 1. Check expected input items exist
        for input_item in expected_inputs:
            input_path = os.path.join(script_dir, input_item)
            if not os.path.exists(input_path):
                print(f"❌ Missing input: {input_item}")
                test_passed = False
                failure_reason = f"Missing input: {input_item}"
                break
        
        if not test_passed:
            print(f"⏭️  Skipping due to missing inputs")
            failed += 1
            results.append((test_name, False, failure_reason))
            continue
        
        # 2. Delete expected outputs before run (clean slate)
        for output_item in expected_outputs:
            output_path = os.path.join(script_dir, output_item)
            if os.path.exists(output_path):
                os.remove(output_path)
                print(f"🗑️  Deleted: {output_item}")
        
        # 3. Run the command
        full_command = [sys.executable, os.path.join(script_dir, 'ai-media.py')] + shlex.split(command)
        print(f"🚀 Running: python ai-media.py {command}")
        
        is_interactive = test.get("interactive", False)
        interactive_wait = test.get("interactiveWait", 2.0)
        expected_stdout_items = test.get("expectedStdoutItems", [])
        
        start_time = time.time()
        current_process = None
        try:
            # Set UTF-8 for subprocess to handle emoji on Windows
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            
            # Use Popen for better control over the subprocess
            current_process = subprocess.Popen(
                full_command,
                cwd=script_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=env
            )
            
            try:
                if is_interactive:
                    if verbose: print(f"⏳ Waiting {interactive_wait}s for interactive output...")
                    time.sleep(interactive_wait)
                    # Terminate interactive process (Windows-compatible)
                    if os.name == 'nt':
                        current_process.terminate()  # Windows: terminate gracefully
                    else:
                        current_process.send_signal(signal.SIGINT)  # Unix: send CTRL+C
                
                stdout, stderr = current_process.communicate(timeout=600 if not is_interactive else 5)
                elapsed = time.time() - start_time
                
                # Show verbose output if requested
                if verbose:
                    if stdout:
                        print(f"\n--- STDOUT ---")
                        print(stdout)
                    if stderr:
                        print(f"\n--- STDERR ---")
                        print(stderr)
                    print(f"--- END ---\n")
                
                if current_process.returncode != 0 and not is_interactive:
                    # Interactive tests expect SIGINT exit code (usually 130 or 1 or 0 handling)
                    # If caught cleanly it might be 0.
                    # We only fail non-interactive tests on non-zero exit code here unless specific check later.
                    print(f"❌ Command failed with exit code {current_process.returncode}")
                    if stderr and not verbose:
                        print(f"   Error: {stderr[:200]}")
                    test_passed = False
                    failure_reason = f"Exit code {current_process.returncode}"
                
                # Check STDOUT items
                if test_passed and expected_stdout_items:
                    for item in expected_stdout_items:
                        if item not in stdout:
                            print(f"❌ Missing stdout item: '{item}'")
                            test_passed = False
                            failure_reason = f"Missing stdout: '{item}'"
                            break
                        else:
                            if verbose: print(f"✓ Found stdout item: '{item}'")
                            
            except subprocess.TimeoutExpired:
                current_process.kill()
                current_process.wait()
                elapsed = time.time() - start_time
                print(f"❌ Command timed out after 10 minutes")
                test_passed = False
                failure_reason = "Timeout"
                
        except KeyboardInterrupt:
            # Terminate subprocess and re-raise to be caught by outer handler
            if current_process:
                current_process.terminate()
                try:
                    current_process.wait(timeout=2)
                except:
                    current_process.kill()
            raise
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Command exception: {e}")
            test_passed = False
            failure_reason = str(e)
        
        # 4. Check expected output items exist
        if test_passed:
            for output_item in expected_outputs:
                output_path = os.path.join(script_dir, output_item)
                if not os.path.exists(output_path):
                    print(f"❌ Missing output: {output_item}")
                    test_passed = False
                    failure_reason = f"Missing output: {output_item}"
                    break
                else:
                    print(f"✓ Output exists: {output_item}")
        
        # 5. Update test results
        if test_passed:
            print(f"✅ PASSED ({elapsed:.1f}s)")
            passed += 1
            _test_state['passed'] = passed
            results.append((test_name, True, f"{elapsed:.1f}s"))
        else:
            print(f"❌ FAILED ({elapsed:.1f}s)")
            failed += 1
            _test_state['failed'] = failed
            results.append((test_name, False, failure_reason))
    
    # Mark test as no longer active
    _test_state['active'] = False
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"   Total:  {len(tests)}")
    print(f"   Passed: {passed} ✅")
    print(f"   Failed: {failed} ❌")
    print(f"{'='*60}")
    
    if failed > 0:
        print(f"\n❌ Failed Tests:")
        for name, success, reason in results:
            if not success:
                print(f"   - {name}: {reason}")
    
    print(f"\n📁 Results saved to: testing.json")
    
    sys.exit(0 if failed == 0 else 1)

# --- Main Logic ---

class CleanHelpFormatter(argparse.RawTextHelpFormatter):
    """Custom formatter that hides metavar and uses wider columns."""
    def __init__(self, prog):
        super().__init__(prog, max_help_position=40, width=120)
    
    def _format_action_invocation(self, action):
        if not action.option_strings:
            return super()._format_action_invocation(action)
        # For options with nargs or that take values, hide the metavar
        if action.nargs != 0 and action.option_strings:
            # Return just the option strings without the metavar
            return ', '.join(action.option_strings)
        return super()._format_action_invocation(action)

def main():
    parser = argparse.ArgumentParser(
        description="Generate, describe, upscale, and convert media using AI and FFmpeg.",
        formatter_class=CleanHelpFormatter,
        epilog="""
Examples:
  -- Media Conversion --
  python ai-media.py -ci photo.gif -cit png
  python ai-media.py -cv clip.mov -cvt mp4
  python ai-media.py -ca song.wav -cat mp3
  
  -- Transforming & Editing --
  python ai-media.py -ti "photo.jpg" -p "Make it look like an anime drawing"
  python ai-media.py -ti "photo.jpg" -p "Make it anime" -o "edits/anime_version.png"
  python ai-media.py -ti "photo.jpg" --remove-background
  python ai-media.py -ti "photo.jpg" --remove-background -o "no_bg/photo_clean.png"

  -- Image Generation --
  python ai-media.py -i -p "Cyberpunk city" -o city.png -s 720p
  python ai-media.py -i -p "Forest" -o forest.jpg -s 4k
  
  -- AI Upscaling --
  python ai-media.py -ui input.jpg -uf 2x
  python ai-media.py -ui input.jpg -uf 4x
  python ai-media.py -ui input.jpg -uf 4x -su (Simple Upscale)
  
  -- Video Generation --
  python ai-media.py -v -p "Robot dancing" -o robot.mp4 -l 5s
  python ai-media.py -v -p "Camera pans left" -ii ./start.png -o output.mp4 (Image-to-Video)
  python ai-media.py -v -p "Dancer" -ap "Techno beat" -o party.mp4 (Video+Audio Mux)
  
  -- Audio Generation --
  python ai-media.py -a -p "Jazz saxophone" -o jazz.mp3 -l 30s
  python ai-media.py -a -p "Rainforest" -o rain.wav --audio-model audioldm2
  python ai-media.py -a -p "Spooky" -ii ./haunted.jpg -o spooky.mp3 (Image-to-Audio)
  python ai-media.py -a -ii ./image.jpg -cm blip (Image-to-Audio w/ BLIP)
  python ai-media.py -a -ii ./image.jpg (Auto-caption + Audio)
  python ai-media.py -a -ii ./video.mp4 (Auto-caption Video + Audio)
  python ai-media.py -a -p "♪ In the jungle ♪ [laughter]" --audio-model bark (Bark Creative)

  -- Generate Description --
  python ai-media.py -gd -ii video.mp4
  python ai-media.py -gd -ii image.jpg -cm blip (Use simpler model)


Supported Models:
  Images:
    - sdxl (default)    : ~8GB  | stabilityai/sdxl-turbo (Open)
    - sd-1.5            : ~4GB  | runwayml/stable-diffusion-v1-5 (Open, No Login)
    - flux              : ~24GB | black-forest-labs/FLUX.1-schnell (🔒 Gated - Free Login Required)
    - flux-dev          : ~24GB | black-forest-labs/FLUX.1-dev (🔒 Gated - Free Login Required)
  
  Audio:
    - musicgen-small           : ~2GB  | Fast, good for music sketches
    - musicgen-medium (default): ~6GB  | Better composition & fidelity
    - musicgen-large           : ~10GB | Highest quality music generation
    - audioldm2                : ~4GB  | Sound effects (SFX), foley, environmental
    - stable-audio             : ~10GB | 🔒 Gated. Best for Sound Effects (SFX), Drums, Ambient.
    - bark                     : ~4GB  | Speech (TTS) & creative audio. Transformer-based
    
  Video:
    - zeroscope (default): ~4GB  | cerspense/zeroscope_v2_576w (Open, 576x320)
    - ms-1.7b            : ~10GB | damo-vilab/text-to-video-ms-1.7b (Has watermarks)
    - cogvideox          : ~15GB | THUDM/CogVideoX-5b (Open)
    - svd                : ~4GB  | stabilityai/stable-video-diffusion-img2vid-xt (Open, I2V Only)
    
  Upscaling:
    - x2 (≤2x factor)   : ~4GB  | stabilityai/sd-x2-latent-upscaler (64px alignment)
    - x4 (>2x factor)   : ~8GB  | stabilityai/stable-diffusion-x4-upscaler (8px alignment)
    
  Captioning:
    - florence (default) : ~1.5GB | microsoft/Florence-2-large (SOTA Details)
    - blip               : ~1GB   | Salesforce/blip-image-captioning-large (Simple)

  Creative Transforming & Editing:
    - instruct-pix2pix     : ~4GB  | timbrooks/instruct-pix2pix (Edit via prompts)
    - instruct-pix2pix-sdxl: ~8GB  | diffusers/sdxl-instructpix2pix-768 (High Quality)
    - remove-bg            : ~1GB  | briaai/RMBG-1.4 (State of the art BG Removal)
        """
    )
    
    # Media Conversion (Standalone - No AI)
    convert_group = parser.add_argument_group("Media Conversion")
    convert_group.add_argument("-ci", "--convert-image", metavar="FILE", help="Convert image format (e.g., gif→png)")
    convert_group.add_argument("-cit", "--convert-image-to", metavar="FMT", help="Output format (png, .webp, out.jpg)")
    convert_group.add_argument("-cv", "--convert-video", metavar="FILE", help="Convert video (mov→mp4)")
    convert_group.add_argument("-cvt", "--convert-video-to", metavar="FMT", help="Output format (mp4, .webm, out.avi)")
    convert_group.add_argument("-ca", "--convert-audio", metavar="FILE", help="Convert audio (wav→mp3)")
    convert_group.add_argument("-cat", "--convert-audio-to", metavar="FMT", help="Output format (mp3, .flac, out.ogg)")
    convert_group.add_argument("--convert-image-engine", choices=["pil", "ffmpeg"], default="pil", help="pil (default) or ffmpeg")
    
    # AI Upscaling (Standalone Mode)
    upscale_mode_group = parser.add_argument_group("AI Upscaling")
    upscale_mode_group.add_argument("-ui", "--upscale-image", metavar="FILE", help="Upscale an existing image")
    upscale_mode_group.add_argument("-uv", "--upscale-video", metavar="FILE", help="Upscale an existing video")
    
    # Modes
    mode_group = parser.add_argument_group("Generation Mode")
    mode_group.add_argument("-i", "--generate-image", action="store_true", help="Generate Image")
    mode_group.add_argument("-v", "--generate-video", action="store_true", help="Generate Video")
    mode_group.add_argument("-a", "--generate-audio", action="store_true", help="Generate Audio")
    mode_group.add_argument("-gd", "--generate-description", nargs="?", const="USE_INPUT_IMAGE", help="Generate Description (Caption) for Image or Video.")
    mode_group.add_argument("-ti", "--transform-image", nargs="?", const="USE_GENERATED", metavar="FILE", help="Transform an image. Omit FILE to auto-use generated output from -i.")
    
    # Common
    common_group = parser.add_argument_group("Common Parameters")
    common_group.add_argument("-p", "--prompt", required=False, help="Text prompt description (Required for generation modes)")
    common_group.add_argument("-ap", "--audio-prompt", help="Audio prompt for 'Video with Audio' generation (merged via FFmpeg).")
    common_group.add_argument("-o", "--output", help="Output file path. Auto-generated from prompt if omitted.")
    common_group.add_argument("--force", action="store_true", help="Skip all confirmation prompts (overwrites files, ignores resource warnings).")
    common_group.add_argument("-f", "--format", help="File format. Image: jpg/png (default: jpg). Video: mp4. Audio: mp3/wav (default: mp3).")
    
    # Shared -s
    common_group.add_argument("-s", "--size",
                              help="Resolution for Image/Video: '720p', '1080p', '4k', '8k', 'HD', '1280x720'. Default: 720p")
    
    # Orientation (swaps w/h for portrait mode)
    common_group.add_argument("-otn", "--orientation", choices=["landscape", "portrait", "square"], default="landscape",
                              help="Orientation for SDXL/Flux generation. 'portrait' swaps width/height.")
    
    # Specific options
    image_group = parser.add_argument_group("Image Options")
    image_group.add_argument("--image-model", default="default", help=f"Model: {', '.join(IMAGE_MODELS.keys())}")
    image_group.add_argument("--unsafe", action="store_true", help="Disable NSFW safety checker (Use with caution).")
    
    video_group = parser.add_argument_group("Video Options")
    video_group.add_argument("--video-model", default="default", help=f"Model: {', '.join(VIDEO_MODELS.keys())}")
    video_group.add_argument("-l", "--length", default="2s", help="Duration (e.g. '2s', '5s', '1m', '{m:1, s:30}'). Default: 2s")
    video_group.add_argument("-ii", "--input-image", help="Input image for Image-to-Video generation.")
    
    audio_group = parser.add_argument_group("Audio Options")
    audio_group.add_argument("-am", "--audio-model", default="default", help=f"Model: {', '.join(AUDIO_MODELS.keys())}")
    audio_group.add_argument("--voice-preset", default="v2/en_speaker_6", help="Bark Voice Preset (e.g. 'v2/en_speaker_6', 'v2/fr_speaker_1'). Default: v2/en_speaker_6")
    audio_group.add_argument("-m", "--sampling-rate", type=str, default="32000", help="Sampling rate (e.g. 32000, 44.1k, 48k). Default: 32000.")
    audio_group.add_argument("-b", "--bit-depth", type=int, choices=[16, 24, 32], default=16, help="Bit depth for audio conversion.")
    audio_group.add_argument("-r", "--bit-rate", help="Bit rate (e.g. 192k) for audio conversion.")
    
    # Captioning Options
    caption_group = parser.add_argument_group("Captioning Options")
    caption_group.add_argument("-cm", "--caption-model", default="florence", choices=["florence", "blip"], help="Model for description generation: 'florence' (default, SOTA) or 'blip'.")
    
    # Performance Tracking
    common_group.add_argument("-npt", "--no-performance-tracking", action="store_true", help="Disable performance tracking (performance.json).")
    
    # Safety Checker
    # common_group.add_argument("--unsafe", action="store_true",
    #                           help="Disable NSFW safety checker (reduces false positives but allows adult content).")

    transform_group = parser.add_argument_group("Transformation Options (-ti)")
    transform_group.add_argument("-tp", "--transform-prompt", help="Edit instruction for InstructPix2Pix (e.g., 'Make it anime'). Used with -ti.")
    transform_group.add_argument("--remove-background", "-rb", action="store_true", help="Remove background (Transparent PNG).")
    transform_group.add_argument("--silhouette", action="store_true", help="Create a black silhouette (requires -rb).")
    transform_group.add_argument("--image-guidance", type=float, default=1.5, help="Image guidance scale (default: 1.5). Higher = closer to original.")
    # transform_group.add_argument("--vignette", action="store_true", help="Add a vignette effect.")
    # transform_group.add_argument("--add-noise", type=float, help="Add noise strength (0.0-1.0).")

    # Time/Length
    # common_group.add_argument("-l", "--length", default=DEFAULT_DURATION,
    #                           help="Duration: '15s', '1h', '{m:2, s:30}'. Default: 15s")
                              
    # Model Selection
    # model_group = parser.add_argument_group("Model Selection")
    # model_group.add_argument("--image-model", default="default", help="Model code or ID for image generation. Default: sdxl")
    # model_group.add_argument("--audio-model", default="default", help="Model code or ID for audio generation. Default: musicgen-small")
    # model_group.add_argument("--video-model", default="default", help="Model code or ID for video generation. Default: zeroscope")
    
    # Audio Specific



    # Upscaling Options (applies to both standalone and chained upscaling)
    upscale_group = parser.add_argument_group("Upscaling Options")
    upscale_group.add_argument("-uf", "--upscale-factor", help="Upscale factor (e.g. '2x', '4'). Default: 2x")
    upscale_group.add_argument("--upscale", action="store_true", help="Enable AI Upscaling after generation (chained mode).")
    upscale_group.add_argument("-uof", "--upscaled-output-file", help="Custom filename for the upscaled output (e.g. 'highres.png').")
    upscale_group.add_argument("-us", "--upscale-strength", type=float, default=0.0, help="Upscale creativity/strength (0.0-1.0). Default: 0.0")
    upscale_group.add_argument("-su", "--simple-upscale", action="store_true", help="Use simple non-AI upscaling (PIL Lanczos for images, FFmpeg for videos). Very fast.")
    
    # Testing
    test_group = parser.add_argument_group("Testing")
    test_group.add_argument("--test", action="store_true", help="Run test suite from testing.json (quiet mode).")
    test_group.add_argument("--test-verbose", action="store_true", help="Run test suite with full output (errors, warnings, details).")
    
    # Interactive Mode
    parser.add_argument("--interactive", "-I", nargs="?", const="menu", metavar="JUMP",
                        help="Run in interactive mode. Optional: Jump point (e.g., 'image/sdxl', 'audio/bark').")
    
    args = parser.parse_args()
    
    # Run test suite if --test or --test-verbose is provided
    if args.test or args.test_verbose:
        run_tests(verbose=args.test_verbose)
        return  # run_tests calls sys.exit
    
    # Run interactive mode if --interactive is provided OR no arguments given
    # Check if any meaningful argument was provided
    has_action = (args.generate_image or args.generate_video or args.generate_audio or
                  args.generate_description or args.transform_image or
                  args.upscale_image or args.upscale_video or
                  args.convert_image or args.convert_video or args.convert_audio or
                  args.prompt)
    
    if args.interactive or not has_action:
        run_interactive(args.interactive)
        return
    
    # Prompt Validation (Required unless upscaling/converting/captioning/transforming OR using Image Input for Generation)
    standalone_modes = (args.upscale_image or args.upscale_video or 
                       args.convert_image or args.convert_video or args.convert_audio or
                       args.generate_description or args.transform_image)
    
    # Check if we are generating with an image input (valid for Audio and Video)
    is_image_based_generation = (args.generate_audio or args.generate_video) and args.input_image
    
    if not standalone_modes and not args.prompt and not is_image_based_generation:
        parser.error(" The -p/--prompt argument is required unless running in Upscale/Convert/Transform Mode or providing an Input Image.\n                            (e.g. python ai-media.py -i -p \"cat\")")
    
    # --- Logic Routing ---
    
    uf = parse_upscale_factor(args.upscale_factor)
    
    # Propagate Force Flag globally
    if args.force:
        os.environ["AI_MEDIA_FORCE"] = "1"
        


    # 0. Generate Description (Captioning)
    if args.generate_description:
        # Determine input file
        target_file = None
        if args.generate_description != "USE_INPUT_IMAGE":
            target_file = args.generate_description
        elif args.input_image:
            target_file = args.input_image
            
        if not target_file:
            print("❌ Error: --generate-description requires a file path.\n   Usage: -gd [file]  OR  -gd -ii [file]")
            sys.exit(1)
        
        device, _ = get_optimal_device_and_dtype(quiet=False)
        caption = generate_caption(target_file, device, model_type=args.caption_model)
        
        if caption:
            print(f"\n📝 Generated Description:\n{caption}\n")
            
            output_path = args.output
            if not output_path:
                 # Auto-generate filename: input_basename.txt
                 base_name = os.path.splitext(target_file)[0]
                 output_path = f"{base_name}.txt"
            
            # Ensure folder exists
            ensure_paths(output_path)
            
            # check overwrite
            if os.path.exists(output_path) and not args.force:
                print(f"⚠️  File '{output_path}' already exists.")
                try:
                    choice = input(f"   Overwrite? [y/N]: ").lower().strip()
                    if choice not in ['y', 'yes']:
                        print("❌ Operation cancelled.")
                        sys.exit(0)
                except KeyboardInterrupt:
                    print("\n❌ Operation cancelled.")
                    sys.exit(0)

            with open(output_path, "w") as f:
                f.write(caption)
            print(f"✅ Saved description to: {output_path}")
        else:
            print("❌ Failed to generate description.")
        sys.exit(0)

    # 1. Media Conversion (Highest Priority - No AI)
    if args.convert_image:
        if not args.convert_image_to:
            parser.error("Image conversion requires -cit/--convert-image-to (e.g., -ci input.gif -cit png)")
        if args.convert_image_engine == "ffmpeg":
            convert_image_ffmpeg(args.convert_image, args.convert_image_to)
        else:
            convert_image_file(args.convert_image, args.convert_image_to)
        return
    
    if args.convert_video:
        if not args.convert_video_to:
            parser.error("Video conversion requires -cvt/--convert-video-to (e.g., -cv input.mov -cvt mp4)")
        convert_video_file(args.convert_video, args.convert_video_to)
        return
    
    if args.convert_audio:
        if not args.convert_audio_to:
            parser.error("Audio conversion requires -cat/--convert-audio-to (e.g., -ca input.wav -cat mp3)")
        convert_audio_file(args.convert_audio, args.convert_audio_to)
        return
        
    # Handle -uof alias for standalone mode (treat as output)
    if args.upscaled_output_file and not args.output:
        args.output = args.upscaled_output_file
    
    # 1. Upscaling (High Priority Standalone)
    if args.upscale_image:
        if not args.output:
             name, ext = os.path.splitext(args.upscale_image)
             suffix = "simple" if args.simple_upscale else "upscaled"
             args.output = f"{name}_{suffix}_{uf}x.png"
        
        if args.simple_upscale:
            simple_upscale_image(args.upscale_image, args.output, factor=uf)
        else:
            upscale_image_file(args.upscale_image, args.output, args.upscale_strength, factor=uf)
        return

    if args.upscale_video:
        if not args.output:
             name, ext = os.path.splitext(args.upscale_video)
             suffix = "simple" if args.simple_upscale else "upscaled"
             args.output = f"{name}_{suffix}_{uf}x.mp4"
        
        if args.simple_upscale:
            simple_upscale_video(args.upscale_video, args.output, factor=uf)
        else:
            upscale_video_file(args.upscale_video, args.output, args.upscale_strength, factor=uf)
        return

    # 2. Validation for Generation
    if not any([args.generate_image, args.generate_video, args.generate_audio, args.generate_description, args.transform_image, args.upscale_image, args.upscale_video]):
        parser.error("You must specify a generation mode: -i, -v, -a, -gd, -ti (or --upscale-image/--upscale-video)")
    
    # Auto-generate output filename from prompt if not provided
    if not args.output:
        # Sanitize prompt to create safe filename (first 2 words, alphanumeric only)
        if args.prompt:
            words = re.findall(r'[a-zA-Z0-9]+', args.prompt.lower())[:2]
            if words:
                args.output = "-".join(words)
                print(f"ℹ️  No output specified. Using: {args.output}")
            else:
                parser.error("Cannot auto-generate filename: prompt contains no valid words. Please specify -o.")
        elif args.input_image:
             # Use input filename as base
             base = os.path.splitext(os.path.basename(args.input_image))[0]
             args.output = f"audio_{base}" if args.generate_audio else f"video_{base}"
             print(f"ℹ️  No output specified. Using input basename: {args.output}")

        elif args.transform_image:
             # Use transform input filename as base
             base = os.path.splitext(os.path.basename(args.transform_image))[0]
             args.output = f"{base}_transformed"
             print(f"ℹ️  No output specified. Using transform basename: {args.output}")
        else:
             parser.error("Cannot auto-generate filename (no prompt or input image). Please specify -o.")

    # Smart Extension Handling
    # If format is specified (-f png) but output doesn't have it, append it.
    if args.format:
        ext = f".{args.format.lower().lstrip('.')}"
        if not args.output.lower().endswith(ext):
            args.output += ext
            print(f"ℹ️  Appended extension '{ext}' to output path.")
    else:
        # No format specified - check if output has an extension
        _, existing_ext = os.path.splitext(args.output)
        if not existing_ext:
            # Auto-append default extension based on mode
            if args.generate_image:
                args.output += ".jpg"
                print(f"ℹ️  No extension specified. Using default: .jpg\n")
            elif args.generate_video:
                args.output += ".mp4"
                print(f"ℹ️  No extension specified. Using default: .mp4\n")
            elif args.generate_audio:
                args.output += ".mp3"
                print(f"ℹ️  No extension specified. Using default: .mp3\n")
            elif args.transform_image:
                args.output += ".png"
                print(f"ℹ️  No extension specified. Using default: .png\n")

    ensure_paths(args.output)
    
    # Check for existing file
    if os.path.exists(args.output) and not args.force:
        print(f"⚠️  File '{args.output}' already exists.")
        try:
            choice = input(f"   Overwrite? [y/N]: ").lower().strip()
            if choice not in ['y', 'yes']:
                print("❌ Operation cancelled.")
                sys.exit(0)
            print("") # Spacer
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled.")
            sys.exit(0)
    
    # Argument Resolution
    final_size = args.size

    # Initialize Performance Tracker (unless disabled)
    tracker = None
    if not args.no_performance_tracking:
        tracker = PerformanceTracker()
    
    device, _ = get_optimal_device_and_dtype()
    
    # Execution
    import time
    duration_sec = parse_duration(args.length)
    
    start_time = time.time()
    
    try:
        success = False
        if args.generate_image:
            # Resolve model ID for consistent tracking (in case defaults change)
            model_key = IMAGE_MODELS.get(args.image_model.lower(), args.image_model)
            if args.image_model.lower() == "default": model_key = IMAGE_MODELS["default"]
            w, h = parse_size(final_size)
            
            # Apply orientation swap if portrait mode
            # Apply orientation swap if portrait mode
            if args.orientation == "portrait":
                w, h = h, w
                print(f"📐 Portrait orientation: swapped to {w}x{h}")
            elif args.orientation == "square":
                # User request: use the smaller size and repeat it
                side = min(w, h)
                w, h = side, side
                print(f"📐 Square orientation: adjusted to {w}x{h}")
            
            # --- Proactive Optimization for High-Res (4K+) ---
            # Trigger if total pixels > 6MP (approx 3K territory) AND not already upscaling
            total_pixels = w * h
            if total_pixels > 6_000_000 and not args.upscale:
                # Calculate Safe 3K Base
                long_edge = max(w, h)
                scale = 3072 / long_edge
                safe_w = int(w * scale)
                safe_h = int(h * scale)
                
                # Make sure safe dimensions are divisible by 8
                safe_w = (safe_w // 8) * 8
                safe_h = (safe_h // 8) * 8
                
                calc_factor = 1 / scale
                
                print(f"\n⚠️  High Resolution Detected ({w}x{h}). Native 4K+ generation can be very slow.\n")
                print(f"💡 Recommendation: Generate at optimized 3K ({safe_w}x{safe_h}) + Auto-Upscale {calc_factor:.2f}x.\n")
                try:
                    choice = input(f"   🔄 Switch to optimized workflow? [Y/n]: ").lower().strip()
                    if choice not in ['n', 'no']:
                        print(f"\n   ✅ Switched to: Base {safe_w}x{safe_h} -> Upscale {calc_factor:.2f}x\n")
                        w = safe_w
                        h = safe_h
                        args.upscale = True
                        uf = calc_factor # Setup factor for Stage 2
                        args.force = True # Prevent double-confirmation (we just asked)
                except KeyboardInterrupt:
                    pass

            # Resource check before starting
            check_resources_and_warn(model_key, width=w, height=h, force=args.force)
            
            # Estimate
            if tracker:
                est_time, est_cpu, est_ram = tracker.estimate_image(model_key, w, h, device)
                if est_time:
                    print(f"⏱️  Estimated Resources: Time: {format_time(est_time)} | RAM: {est_ram:.1f}GB | CPU: {est_cpu:.1f}%")
                else:
                    print(f"⏱️  Estimated Resources: (Calibrating... first run for {w}x{h})")
                print("") # Spacer
            
            # Monitor and Generate
            mon_ctx = ResourceMonitor() if tracker else None
            
            if mon_ctx: mon_ctx.__enter__()
            
            if args.upscale:
                 print("")
                 print("="*60)
                 print(f"👉 Step 1 - Generate at {w}x{h}")
                 print("="*60)
                 print("")
                 
            success = generate_image(args.prompt, args.output, w, h, model_name=args.image_model, unsafe=args.unsafe)
            if mon_ctx: mon_ctx.__exit__(None, None, None)
            
            if success and tracker:
                elapsed = time.time() - start_time
                avg_cpu, avg_ram = mon_ctx.get_averages()
                print("")  # Spacer
                print(f"⏱️  Actual Resources:    Time: {format_time(elapsed)} | RAM: {avg_ram:.1f}GB | CPU: {avg_cpu:.1f}%")
                tracker.record_image(model_key, w, h, device, elapsed, cpu=avg_cpu, ram=avg_ram)
        
        elif args.generate_video:
            # Resolve model ID for consistent tracking
            model_key = VIDEO_MODELS.get(args.video_model.lower(), args.video_model)
            if args.video_model.lower() == "default": model_key = VIDEO_MODELS["default"]
            w, h = parse_size(final_size)
            
            # --- Proactive Optimization for High-Res (4K+) ---
            # User Feedback: 3K is fine, 4K is slow.
            # Trigger if total pixels > 6MP (approx 3K territory) AND not already upscaling
            total_pixels = w * h
            if total_pixels > 6_000_000 and not args.upscale:
                # Calculate Safe 3K Base
                # Target max dimension 3072 covers "3K" nicely.
                long_edge = max(w, h)
                scale = 3072 / long_edge
                safe_w = int(w * scale)
                safe_h = int(h * scale)
                
                # Make sure safe dimensions are divisible by 8 (Architecture requirement)
                safe_w = (safe_w // 8) * 8
                safe_h = (safe_h // 8) * 8
                
                calc_factor = 1 / scale
                
                print(f"\n⚠️  High Resolution Detected ({w}x{h}). Native 4K+ generation can be very slow.\n")
                print(f"💡 Recommendation: Generate at optimized 3K ({safe_w}x{safe_h}) + Auto-Upscale {calc_factor:.2f}x.\n")
                try:
                    choice = input(f"   🔄 Switch to optimized workflow? [Y/n]: ").lower().strip()
                    if choice not in ['n', 'no']:
                        print(f"\n   ✅ Switched to: Base {safe_w}x{safe_h} -> Upscale {calc_factor:.2f}x\n")
                        w = safe_w
                        h = safe_h
                        args.upscale = True
                        uf = calc_factor # Setup factor for Stage 2
                        args.force = True # Prevent double-confirmation
                except KeyboardInterrupt:
                    pass

            # Resource check before starting
            check_resources_and_warn(model_key, width=w, height=h, duration=duration_sec, force=args.force)
                
            if tracker:
                est_time, est_cpu, est_ram = tracker.estimate_linear("video", model_key, device, duration_sec, w, h)
                if est_time:
                    print(f"⏱️  Estimated Resources: Time: {format_time(est_time)} | RAM: {est_ram:.1f}GB | CPU: {est_cpu:.1f}%")
                else:
                    print(f"⏱️  Estimated Resources: (Calibrating... first run)")
                print("") # Spacer
            
            mon_ctx = ResourceMonitor() if tracker else None
            if mon_ctx: mon_ctx.__enter__()
            
            if args.upscale:
                 print("")
                 print("="*60)
                 print(f"👉 Step 1 - Generate at {w}x{h}")
                 print("="*60)
                 print("")
                 
            success = generate_video(
                args.prompt, 
                args.output, 
                duration_sec, 
                w, 
                h, 
                model_name=args.video_model,
                image_input=args.input_image,
                audio_prompt=args.audio_prompt
            )
            if mon_ctx: mon_ctx.__exit__(None, None, None)

            if success and tracker:
                elapsed = time.time() - start_time
                avg_cpu, avg_ram = mon_ctx.get_averages()
                print("")  # Spacer
                print(f"⏱️  Actual Resources:    Time: {format_time(elapsed)} | RAM: {avg_ram:.1f}GB | CPU: {avg_cpu:.1f}%")
                tracker.record_linear("video", model_key, device, duration_sec, elapsed, width=w, height=h, cpu=avg_cpu, ram=avg_ram)
        
        elif args.generate_audio:
            # Resolve model ID for consistent tracking
            model_key = AUDIO_MODELS.get(args.audio_model.lower(), args.audio_model)
            if args.audio_model.lower() == "default": model_key = AUDIO_MODELS["default"]
            hz = parse_sampling_rate(args.sampling_rate)
            
            # Resource check before starting
            check_resources_and_warn(model_key, duration=duration_sec, force=args.force)
            
            if tracker:
                est_time, est_cpu, est_ram = tracker.estimate_linear("audio", model_key, device, duration_sec)
                if est_time:
                    print(f"⏱️  Estimated Resources: Time: {format_time(est_time)} | RAM: {est_ram:.1f}GB | CPU: {est_cpu:.1f}%")
                    print("") # Spacer
                else:
                    print(f"⏱️  Estimated Resources: (Calibrating... first run)")
                    print("") # Spacer
            
            mon_ctx = ResourceMonitor() if tracker else None
            if mon_ctx: mon_ctx.__enter__()
            success = generate_audio(
                args.prompt, args.output, duration_sec, 
                args.sampling_rate, model_name=args.audio_model,
                image_input=args.input_image,
                caption_model=args.caption_model,
                voice_preset=args.voice_preset
            )
            if mon_ctx: mon_ctx.__exit__(None, None, None)
            
            if success and tracker:
                elapsed = time.time() - start_time
                avg_cpu, avg_ram = mon_ctx.get_averages()
                print("")  # Spacer
                print(f"⏱️  Actual Resources:    Time: {format_time(elapsed)} | RAM: {avg_ram:.1f}GB | CPU: {avg_cpu:.1f}%")
                tracker.record_linear("audio", model_key, device, duration_sec, elapsed, cpu=avg_cpu, ram=avg_ram)

        # X. Transform Image (-ti) - Chained or Standalone
        if args.transform_image:
            # If generation just happened (success=True), we might need to rely on the Output of that generation
            # as the input for this transformation, IF the input was specified as the output name.
            
            # Handle anonymous chain: -ti without filename uses generated output
            if args.transform_image == "USE_GENERATED":
                if success and args.output:
                    input_file = args.output
                    print(f"🔗 Anonymous chain: Using generated output '{input_file}' as transformation input")
                else:
                    print("❌ Error: -ti used without a file, but no image was generated.")
                    print("   Either specify a file: -ti photo.jpg")
                    print("   Or chain with generation: -i -p \"...\" -ti -tp \"...\"")
                    sys.exit(1)
            else:
                input_file = args.transform_image
            
            # Check if input file exists
            if not os.path.exists(input_file):
                print(f"❌ Error: Input file for transformation not found: {input_file}")
                sys.exit(1)

            # Determine output
            # If args.output was set for generation, it is currently holding that value.
            # If we want to overwrite, that's fine.
            # If args.output wasn't set, it was auto-generated.
            
            # NOTE: If we are chaining, we usually want to operate in-place OR allow a new output.
            # But argparse only allows one -o. 
            # So usually Chained = In-Place (User provided same filename).
            
            # If Standalone -ti, args.output might be empty.
            if not args.output or (args.output == input_file and not args.force and not success):
                 # Auto-generate output name if not provided OR if matches input (safe default without force)
                 # But if success=True (Generation happened), we created the file, so checking existence/overwrite again is annoying?
                 # Actually, if we generated 'file.png' and now want to transform 'file.png' -> 'file.png',
                 # we shouldn't prompt for overwrite again if we just made it? 
                 # But the transformation functions might prompt.
                 
                 if not args.output:
                     name = os.path.splitext(input_file)[0]
                     suffix = "transformed"
                     if args.remove_background: suffix = "transformed-nobg"
                     if args.transform_prompt or args.prompt: suffix = "transformed-edit"
                     args.output = f"{name}_{suffix}.png"
                 
            # Add overwrite protection and path creation (Only if we didn't just generate it?)
            # If success=True, we own the file, probably safe to overwrite?
            ensure_paths(args.output)
            
            if os.path.exists(args.output) and not args.force and not success:
                print(f"⚠️  File '{args.output}' already exists.")
                try:
                    choice = input(f"   Overwrite? [y/N]: ").lower().strip()
                    if choice not in ['y', 'yes']:
                        print("❌ Operation cancelled.")
                        sys.exit(0)
                except KeyboardInterrupt:
                    print("\n❌ Operation cancelled.")
                    sys.exit(0)

            current_input = input_file
            intermediate_file = None
            
            print("")
            print(f"🎨 Starting Transformation on: {current_input}")

            # 1. Instructional Editing (Step 1)
            # Use 'success' to chain through this block locally
            transform_success = True
            
            # Use transform_prompt if provided, otherwise fall back to prompt (for standalone -ti)
            edit_prompt = args.transform_prompt or args.prompt
            
            if edit_prompt:
                steps = 50
                model_to_use = "default" 
                if args.image_model and "sdxl" in args.image_model.lower():
                    model_to_use = "instruct-pix2pix-sdxl"
                
                # If chaining internal steps (Edit + RemoveBG), execute to a temporary intermediate file
                target_out = args.output
                if args.remove_background:
                    name_part = os.path.splitext(args.output)[0]
                    intermediate_file = f"{name_part}_temp_edit.png"
                    target_out = intermediate_file
                    
                    print("")
                    header = "🔗 Chaining detected: 2 Steps"
                    step1 = "   Step 1 - Creative Edit (InstructPix2Pix)"
                    step2 = "   Step 2 - Remove Background (RMBG-1.4)"
                    
                    max_len = max(len(header), len(step1), len(step2))
                    separator = "=" * (max_len + 4) # Add some padding

                    print(separator)
                    print(f"  {header}")
                    print(f"  {step1}")
                    print(f"  {step2}")
                    print(separator)

                    print(f"\n{separator}")
                    print(f"Step 1/2: Creative Edit")
                    print(separator)
                    print(f"🔗 Intermediate input: '{os.path.basename(intermediate_file)}'")

                transform_success = generate_edit(
                    current_input, 
                    edit_prompt, 
                    target_out, 
                    model_name=model_to_use, 
                    guidance_scale=7.5, 
                    image_guidance_scale=args.image_guidance,
                    unsafe=args.unsafe
                )
                
                if not transform_success:
                    if intermediate_file and os.path.exists(intermediate_file):
                        os.remove(intermediate_file)
                    # Don't exit, just mark fail?
                    success = False

                # Update input for next step
                if transform_success and args.remove_background:
                    current_input = intermediate_file

            # 2. Background Removal / Silhouette (Step 2 or Only Step)
            if transform_success and args.remove_background:
                if edit_prompt:
                     # Use the same separator length if available, otherwise default
                     sep = separator if 'separator' in locals() else "============================"
                     print(f"\n\n{sep}")
                     print(f"Step 2/2: Remove Background")
                     print(sep)

                transform_success = remove_background(current_input, args.output, silhouette=args.silhouette)
                
                # Cleanup intermediate
                if intermediate_file and os.path.exists(intermediate_file):
                    os.remove(intermediate_file)
                
                if not transform_success: success = False
                
            # 3. Warn if no action
            if not edit_prompt and not args.remove_background:
                 print("⚠️  Transform mode requires either a prompt (-tp or -p) or --remove-background.")
                 success = False
                 
            # Update global success for next stages (Upscale)
            if transform_success:
                success = True # Ensure we can continue to upscale if requested
            else:
                success = False

        # --- Stage 2: Upscaling (Chained) ---
        if success and args.upscale:
            print("")
            print("="*60)
            print(f"👉 Step 2 - Upscale {uf}x")
            print("="*60)
            print("")
            
            # Auto-detect mode. Overwrites the file with high-res version?
            # Or maybe appends _upscaled? 
            # User requested "taking the upscale factor as stage 2". Usually implies result replacement or refinement.
            # Let's overwrite for seamlessness, OR assume the user wants the high-res file.
            # Actually, let's create a NEW file to be safe: _upscaled.
            
            name, ext = os.path.splitext(args.output)
            suffix = "simple_upscaled" if args.simple_upscale else "upscaled"
            upscale_out = args.upscaled_output_file or f"{name}_{suffix}_{uf}x{ext}"
            
            if args.generate_image:
                if args.simple_upscale:
                    simple_upscale_image(args.output, upscale_out, factor=uf)
                else:
                    upscale_image_file(args.output, upscale_out, args.upscale_strength, factor=uf)
            elif args.generate_video:
                if args.simple_upscale:
                    simple_upscale_video(args.output, upscale_out, factor=uf)
                else:
                    upscale_video_file(args.output, upscale_out, args.upscale_strength, factor=uf)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
