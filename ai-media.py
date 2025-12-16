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
import psutil  # For resource checking
from pathlib import Path
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress verbose logging from transformers/diffusers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)

# Set environment variable to suppress transformers warnings (must be before import)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"


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

AUDIO_MODELS = {
    "musicgen-small": "facebook/musicgen-small",       # Fast, good for music
    "musicgen-medium": "facebook/musicgen-medium",     # Better quality music
    "musicgen-large": "facebook/musicgen-large",       # Best quality music
    "audioldm2": "cvssp/audioldm2",                    # General audio/SFX
    "default": "facebook/musicgen-small"
}

VIDEO_MODELS = {
    "ms-1.7b": "damo-vilab/text-to-video-ms-1.7b",     # Standard open research model
    "zeroscope": "cerspense/zeroscope_v2_576w",        # 576x320 optimized
    "cogvideox": "THUDM/CogVideoX-5b",                 # High quality (requires high VRAM)
    "svd": "stabilityai/stable-video-diffusion-img2vid-xt", # SVD Image-to-Video
    "default": "damo-vilab/text-to-video-ms-1.7b"
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
    "black-forest-labs/FLUX.1-schnell": {"vram": 12, "ram": 24, "max_resolution": (2048, 2048)},
    "black-forest-labs/FLUX.1-dev": {"vram": 16, "ram": 32, "max_resolution": (2048, 2048)},
    # Audio Models (max_duration in seconds, based on model architecture limits)
    "facebook/musicgen-small": {"vram": 4, "ram": 8, "max_duration": 30},
    "facebook/musicgen-medium": {"vram": 8, "ram": 12, "max_duration": 60},
    "facebook/musicgen-large": {"vram": 16, "ram": 24, "max_duration": 120},
    "cvssp/audioldm2": {"vram": 8, "ram": 12, "max_duration": 60},
    # Video Models (max_resolution based on training data)
    "damo-vilab/text-to-video-ms-1.7b": {"vram": 12, "ram": 16, "max_resolution": (1280, 720)},
    "cerspense/zeroscope_v2_576w": {"vram": 8, "ram": 12, "max_resolution": (576, 320)},
    "THUDM/CogVideoX-5b": {"vram": 24, "ram": 32, "max_resolution": (1920, 1080)},
    "stabilityai/stable-video-diffusion-img2vid-xt": {"vram": 8, "ram": 12, "max_resolution": (1024, 576)},
    "stabilityai/stable-diffusion-x4-upscaler": {"vram": 8, "ram": 16, "max_resolution": (4096, 4096)},
    "stabilityai/sd-x2-latent-upscaler": {"vram": 4, "ram": 8, "max_resolution": (2048, 2048)},
}


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
def signal_handler(sig, frame):
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
        num = float(re.sub(r'[^0-9\.]', '', normalized))
        return int(num * 1000)
    
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
            pipe = FluxPipeline.from_pretrained(
                model_id, 
                torch_dtype=dtype if device.type != "cpu" else torch.float32
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
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id, 
                torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None
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


def generate_audio(prompt, output_path, duration, sampling_rate, model_name="default", image_input=None):
    """Generate audio using MusicGen or AudioLDM (supports Image-to-Audio via captioning)."""
    
    model_id = AUDIO_MODELS.get(model_name.lower(), model_name)
    if model_name.lower() == "default": model_id = AUDIO_MODELS["default"]
    
    device, dtype = get_optimal_device_and_dtype()
    
    # --- Image-to-Audio Logic (Captioning) ---
    if image_input:
        print(f"👁️  Analyzing input image: {image_input}")
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            from diffusers.utils import load_image
            
            # Load BLIP (lightweight, ~500MB)
            caption_model_id = "Salesforce/blip-image-captioning-base"
            processor = BlipProcessor.from_pretrained(caption_model_id)
            model = BlipForConditionalGeneration.from_pretrained(caption_model_id).to(device)
            
            raw_image = load_image(image_input).convert('RGB')
            inputs = processor(raw_image, return_tensors="pt").to(device)
            
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            print(f"   Detected: '{caption}'")
            # Combine User Prompt + Image Caption
            # Prompt is the "Action" or "Style", Caption is the "Content"
            full_prompt = f"{prompt}, inspired by {caption}"
            print(f"   Full Prompt: '{full_prompt}'")
            
            # Update prompt for downstream models
            prompt = full_prompt
            
        except Exception as e:
            print(f"⚠️  Image analysis failed: {e}. Proceeding with text prompt only.")

    print(f"🎵 Generating Audio")
    print(f"   Model:    {model_id}")
    print(f"   Prompt:   '{prompt}'")
    if image_input: print(f"   Input Img: {image_input}")
    print(f"   Duration: {duration}s")
    print(f"   Sampling: {sampling_rate}Hz")
    print(f"   Output:   {output_path}")
    print("") # Spacer
    
    try:
        import torch
        import scipy.io.wavfile
        from transformers import pipeline
        from diffusers import AudioLDMPipeline 
        
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
            # Use Diffusers Pipeline for AudioLDM
            print(f"   Loading AudioLDM pipeline...")
            pipe = AudioLDMPipeline.from_pretrained(model_id, torch_dtype=dtype)
            pipe.to(device)
            
            print(f"🎵 Synthesizing audio... (AudioLDM)")
            audio = pipe(prompt, audio_length_in_s=duration).audios[0]
            rate = 16000 # AudioLDM default usually
            
            scipy.io.wavfile.write(output_path + ".tmp.wav", rate, audio.T)
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
        
    except ImportError:
        print("❌ Error: Missing transformers/scipy/torch/diffusers.")
        return False
    except Exception as e:
        print(f"❌ Audio generation failed: {e}")
        return False


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
        print("⚠️  Video generation is resource intensive.")
        
        # --- Stage 1: Video Generation ---
        
        # Load Pipeline
        if "cogvideox" in model_id.lower() and is_i2v:
            pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
        elif "stable-video-diffusion" in model_id.lower():
            pipe = StableVideoDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None)
        else:
            # Generic / Text-to-Video
            pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None)
        
        # Scheduler Optimization
        if hasattr(pipe, "scheduler"):
            try:
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            except: pass 
            
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
        
        # Save Video
        export_to_video(video_frames, video_out, fps=7 if "stable-video-diffusion" in model_id.lower() else 16) 
        print(f"✅ Video track saved to {video_out}")
        
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

def upscale_image_file(image_path, output_path, strength=0.0, factor=2.0):
    """Upscale an image using Stable Diffusion x4 Upscaler.
       Supports multi-pass for factors > 4x (e.g. 8x = 4x -> 4x -> resize)."""
    
    
    # Select Model based on factor
    # <= 2.0x -> use x2 Latent Upscaler (Fast, Faithful)
    # > 2.0x  -> use x4 Upscaler (Detailed)
    use_x2_model = (factor <= 2.0)
    model_id = IMAGE_MODELS['upscaler_x2'] if use_x2_model else IMAGE_MODELS['upscaler']
    
    print(f"🚀 Upscaling Image: {image_path}")
    print(f"   Model: {model_id}")
    print(f"   Target Factor: {factor}x")
    
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

        # Load Pipeline
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
        # 1. Use CPU Offload (or just move to device if already CPU)
        if device.type != "cpu":
             pipe.enable_model_cpu_offload() 
        else:
             pipe.to(device)
        
        # Note: No MPS specific tiling/slicing needed here because we forced CPU above.
        # But we still enable VAE Tiling if available to save System RAM.
        if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_tiling'):
             print("   ✓ Enabling VAE Tiling (Memory Optimization)")
             pipe.vae.enable_tiling()
        
        # Recursive Upscaling Loop
        current_image = image
        current_scale = 1.0
        pass_idx = 1
        
        # Decide base step scale
        step_scale = 2.0 if use_x2_model else 4.0
        
        while current_scale < factor:
            print("")
            print("="*60)
            print(f"🎨 Pass {pass_idx}: Upscaling {step_scale}x (Internal Base)...")
            print("="*60)
            print("")
            
            # --- DIMENSION ALIGNMENT FIX ---
            # x2 latent upscaler requires dimensions divisible by 64 (latent space)
            # x4 upscaler requires dimensions divisible by 8 (standard SD requirement)
            alignment = 64 if use_x2_model else 8
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
            
            # x2 model works better with fewer steps usually? default is fine.
            upscaled_result = pipe(
                prompt="High quality, detailed, sharp, 8k", 
                image=padded_image, 
                num_inference_steps=20,
            ).images[0]
            
            # Crop back to target dimensions (remove padding effect)
            target_w_pass = int(img_w * step_scale)
            target_h_pass = int(img_h * step_scale)
            if upscaled_result.size != (target_w_pass, target_h_pass):
                current_image = upscaled_result.crop((0, 0, target_w_pass, target_h_pass))
            else:
                current_image = upscaled_result
            
            current_scale *= step_scale
            pass_idx += 1
        
        # Final Resize to exact factor
        target_w = int(orig_w * factor)
        target_h = int(orig_h * factor)
        
        if current_image.size != (target_w, target_h):
            print(f"   ↘️  Resizing final result to exact {factor}x ({target_w}x{target_h})...")
            current_image = current_image.resize((target_w, target_h), Image.LANCZOS)
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        current_image.save(output_path)
        print(f"✅ Upscaled image saved to {output_path}")
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
        
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(temp_dir / "upscaled_%05d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
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

# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(
        description="Generate Image, Video, or Audio from text prompts. Upscale existing images and videos with AI.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  -- Image Generation --
  python ai-media.py -i -p "Cyberpunk city" -o city.png -s 720p
  python ai-media.py -i -p "Forest" -o forest.jpg -s 4k
  
  -- Upscaling --
  python ai-media.py --upscale-image input.jpg -o upscaled.png
  python ai-media.py --upscale-image input.jpg -uf 4x (Native 4x gen)
  
  -- Video Generation --
  python ai-media.py -v -p "Robot dancing" -o robot.mp4 -l 5s
  python ai-media.py -v -p "Camera pans left" -ii ./start.png -o output.mp4 (Image-to-Video)
  python ai-media.py -v -p "Dancer" -ap "Techno beat" -o party.mp4 (Video+Audio Mux)
  
  -- Audio Generation --
  python ai-media.py -a -p "Jazz saxophone" -o jazz.mp3 -l 30s
  python ai-media.py -a -p "Rainforest" -o rain.wav --audio-model audioldm2
  python ai-media.py -a -p "Spooky" -ii ./haunted.jpg -o spooky.mp3 (Image-to-Audio)

Supported Models:
  Images:
    - sdxl (default)    : ~8GB  | stabilityai/sdxl-turbo (Open)
    - sd-1.5            : ~4GB  | runwayml/stable-diffusion-v1-5 (Open, No Login)
    - flux              : ~24GB | black-forest-labs/FLUX.1-schnell (🔒 Gated - Free Login Required)
    - flux-dev          : ~24GB | black-forest-labs/FLUX.1-dev (🔒 Gated - Free Login Required)
  
  Audio:
    - musicgen-small (default) : ~2GB
    - musicgen-medium          : ~6GB
    - musicgen-large           : ~10GB
    - audioldm2                : ~4GB
    
  Video:
    - ms-1.7b (default) : ~10GB | damo-vilab/text-to-video-ms-1.7b (Open)
    - zeroscope         : ~4GB  | cerspense/zeroscope_v2_576w (Open)
    - cogvideox         : ~15GB | THUDM/CogVideoX-5b (🔒 Gated - Free Login Required)
    - svd               : ~4GB  | stabilityai/stable-video-diffusion-img2vid-xt (Open, I2V Only)
        """
    )
    
    # Modes
    mode_group = parser.add_argument_group("Generation Mode")
    mode_group.add_argument("-i", "--generate-image", action="store_true", help="Generate Image")
    mode_group.add_argument("-v", "--generate-video", action="store_true", help="Generate Video")
    mode_group.add_argument("-a", "--generate-audio", action="store_true", help="Generate Audio")
    
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
    
    # Image Input (for Image-to-Video and Image-to-Audio)
    common_group.add_argument("-ii", "--image-input", help="Input image path for Image-to-Video or Image-to-Audio generation.")
    
    # Performance Tracking
    common_group.add_argument("--npt", "--no-performance-tracking", action="store_true", dest="no_performance_tracking",
                              help="Disable performance tracking (prevents updating 'average_time' in performance.json).")
    
    # Safety Checker
    common_group.add_argument("--unsafe", action="store_true",
                              help="Disable NSFW safety checker (reduces false positives but allows adult content).")

    # Time/Length
    common_group.add_argument("-l", "--length", default=DEFAULT_DURATION,
                              help="Duration: '15s', '1h', '{m:2, s:30}'. Default: 15s")
                              
    # Model Selection
    model_group = parser.add_argument_group("Model Selection")
    model_group.add_argument("--image-model", default="default", help="Model code or ID for image generation. Default: sdxl")
    model_group.add_argument("--audio-model", default="default", help="Model code or ID for audio generation. Default: musicgen-small")
    model_group.add_argument("--video-model", default="default", help="Model code or ID for video generation. Default: ms-1.7b")
    
    # Audio Specific
    audio_group = parser.add_argument_group("Audio Parameters")
    audio_group.add_argument("-m", "--sampling", help="Sampling rate (e.g. '44100', '48k'). Default: 32000")
    audio_group.add_argument("-b", "--bit-depth", type=int, default=DEFAULT_AUDIO_BITDEPTH, help="Bit depth (16, 24). Default: 16")

    audio_group.add_argument("-r", "--bit-rate", help="Bitrate (e.g. '320k').")

    # Upscaling Parameters
    upscale_group = parser.add_argument_group("Upscaling Parameters")
    upscale_group.add_argument("--upscale-image", help="Path to input image for 4x AI Upscaling")
    upscale_group.add_argument("--upscale-video", help="Path to input video for 4x AI Upscaling")
    upscale_group.add_argument("-us", "--upscale-strength", type=float, default=0.0, help="Upscale creativity/strength (0.0-1.0). Default: 0.0")
    upscale_group.add_argument("-uf", "--upscale-factor", help="Upscale factor (e.g. '2x', '4'). Default: 2x")
    upscale_group.add_argument("--upscale", action="store_true", help="Enable 4x AI Upscaling for the generated content (Stage 2).")
    upscale_group.add_argument("-uof", "--upscaled-output-file", help="Custom filename for the upscaled output (e.g. 'highres.png').")
    
    args = parser.parse_args()
    
    # Prompt Validation (Required unless upscaling)
    if not (args.upscale_image or args.upscale_video) and not args.prompt:
        parser.error(" The -p/--prompt argument is required unless running in Upscale Mode.\n                            (e.g. python ai-media.py -i -p \"cat\")")
    
    # --- Logic Routing ---
    
    uf = parse_upscale_factor(args.upscale_factor)
    
    # Propagate Force Flag globally
    if args.force:
        os.environ["AI_MEDIA_FORCE"] = "1"
        
    # Handle -uof alias for standalone mode (treat as output)
    if args.upscaled_output_file and not args.output:
        args.output = args.upscaled_output_file
    
    # 1. Upscaling (High Priority Standalone)
    if args.upscale_image:
        if not args.output:
             name, ext = os.path.splitext(args.upscale_image)
             args.output = f"{name}_upscaled_{uf}x.png"
        upscale_image_file(args.upscale_image, args.output, args.upscale_strength, factor=uf)
        return

    if args.upscale_video:
        if not args.output:
             name, ext = os.path.splitext(args.upscale_video)
             args.output = f"{name}_upscaled_{uf}x.mp4"
        upscale_video_file(args.upscale_video, args.output, args.upscale_strength, factor=uf)
        return

    # 2. Validation for Generation
    if not any([args.generate_image, args.generate_video, args.generate_audio]):
        parser.error("You must specify a generation mode: -i, -v, -a (or --upscale-image/--upscale-video)")
    
    # Auto-generate output filename from prompt if not provided
    if not args.output:
        # Sanitize prompt to create safe filename (first 2 words, alphanumeric only)
        words = re.findall(r'[a-zA-Z0-9]+', args.prompt.lower())[:2]
        if words:
            args.output = "-".join(words)
            print(f"ℹ️  No output specified. Using: {args.output}")
        else:
            parser.error("Cannot auto-generate filename: prompt contains no valid words. Please specify -o.")

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
                image_input=args.image_input,
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
            hz = parse_sampling_rate(args.sampling)
            
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
            success = generate_audio(args.prompt, args.output, duration_sec, hz, model_name=args.audio_model, image_input=args.image_input)
            if mon_ctx: mon_ctx.__exit__(None, None, None)
            
            if success and tracker:
                elapsed = time.time() - start_time
                avg_cpu, avg_ram = mon_ctx.get_averages()
                print("")  # Spacer
                print(f"⏱️  Actual Resources:    Time: {format_time(elapsed)} | RAM: {avg_ram:.1f}GB | CPU: {avg_cpu:.1f}%")
                tracker.record_linear("audio", model_key, device, duration_sec, elapsed, cpu=avg_cpu, ram=avg_ram)

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
            upscale_out = args.upscaled_output_file or f"{name}_upscaled_{uf}x{ext}"
            
            if args.generate_image:
                upscale_image_file(args.output, upscale_out, args.upscale_strength, factor=uf)
            elif args.generate_video:
                upscale_video_file(args.output, upscale_out, args.upscale_strength, factor=uf)
    
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
