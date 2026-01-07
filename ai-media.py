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
import signal
import sys
import os
import time

# --- Loading Message Timer (must start BEFORE heavy imports) ---
# This shows "Loading..." message after 1 second if still loading modules.
# We use a module-level variable to allow cancellation from main().
import threading
_loading_timer = None
_loading_shown = False

def _show_loading_message():
    global _loading_shown
    _loading_shown = True
    print("⏳ Loading... (May take a moment while modules initialize and cache)", flush=True)

# Only start timer if likely interactive mode (no args or just --interactive)
# We do this early so it can run while heavy modules load
if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ('--interactive', '-I')):
    _loading_timer = threading.Timer(1.0, _show_loading_message)
    _loading_timer.daemon = True
    _loading_timer.start()

# Suppress common library warnings
# (Some libraries use different warning categories or print directly)
warnings.filterwarnings("ignore", message="User provided device_type of 'cuda'", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")  # Catch all
warnings.filterwarnings("ignore", message=".*upcast_vae.*deprecated.*", category=FutureWarning)

# Note: transformers/diffusers logging suppression is deferred to load_ai_modules()
# to avoid loading heavy ML libraries for lightweight operations (e.g., interactive menu).



# --- AI-Media Package Imports ---
# 1. Eagerly load lightweight configuration (constants, models)
# This allows argparse to validate choices quickly without loading heavy ML libs.
try:
    from ai_media import constants as pkg_constants
    from ai_media import models as pkg_models
    from ai_media.utils import interaction as pkg_interaction
    from ai_media.utils.system import setup_signal_handlers, _test_state
    # Flatten for direct usage
    from ai_media.constants import *
    from ai_media.models import *
    
    # Install signal handlers early for clean CTRL+C behavior
    # BUT skip if in interactive mode (parent process), let interactive module handle signals
    if not (len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ('--interactive', '-I'))):
        setup_signal_handlers()
    
    HAS_AI_MEDIA_PKG = True
except ImportError:
    HAS_AI_MEDIA_PKG = False
    _test_state = None
    # Fallback to empty if package missing (script will likely fail later)
    DEFAULT_IMAGE_SIZE = "1024x1024"
    DEFAULT_AUDIO_SAMPLING = 16000
    DEFAULT_AUDIO_BITDEPTH = "float32"
    DEFAULT_DURATION = "5.0"
    RESOLUTIONS = {}
    IMAGE_MODELS = {}
    EDIT_MODELS = {}
    AUDIO_MODELS = {}
    VIDEO_MODELS = {}
    TEXT_MODELS = {}
    MODEL_REQUIREMENTS = {}

# 2. Lazy load heavy modules (generators, utils with torch, etc.)
# Global package placeholders for lazy modules
pkg_parsers = None
pkg_system = None
pkg_performance = None
pkg_ffmpeg = None
pkg_generate_image = None
pkg_generate_audio = None
pkg_generate_video = None
pkg_generate_caption = None
pkg_generate_edit = None
pkg_remove_background = None
pkg_generate_translation = None
PkgArticleGenerator = None
PkgSubtitlesGenerator = None
PkgTranscriptionGenerator = None
PkgPerformanceTracker = None
PkgResourceMonitor = None
pkg_convert_image = None
pkg_convert_video = None
pkg_convert_audio = None
pkg_convert_document = None
pkg_simple_upscale_image = None
pkg_simple_upscale_video = None
pkg_upscale_image_fast = None
pkg_upscale_video_fast = None
pkg_upscale_image_file = None
pkg_upscale_video_file = None
pkg_prompt_menu = None
pkg_prompt_choice = None
pkg_prompt_text = None
pkg_prompt_file = None
pkg_browse_files = None
pkg_check_overwrite = None
pkg_clear_screen = None
pkg_show_header = None
pkg_get_key = None
PKG_JUMP_POINTS = None
pkg_run_interactive = None
pkg_run_tests = None
pkg_run_unit_tests = None

def load_ai_modules():
    """Load heavy AI modules on demand."""
    global pkg_parsers, pkg_system, pkg_performance, pkg_ffmpeg
    global pkg_generate_image, pkg_generate_audio, pkg_generate_video, pkg_generate_caption
    global pkg_generate_edit, pkg_remove_background, pkg_generate_translation, PkgArticleGenerator, PkgSubtitlesGenerator, PkgTranscriptionGenerator
    global PkgPerformanceTracker, PkgResourceMonitor
    global pkg_convert_image, pkg_convert_video, pkg_convert_audio, pkg_convert_document
    global pkg_simple_upscale_image, pkg_simple_upscale_video, pkg_upscale_image_fast
    global pkg_upscale_video_fast, pkg_upscale_image_file, pkg_upscale_video_file
    global pkg_prompt_menu, pkg_prompt_choice, pkg_prompt_text, pkg_prompt_file, pkg_browse_files
    global pkg_check_overwrite, pkg_clear_screen, pkg_show_header, pkg_get_key, PKG_JUMP_POINTS, pkg_run_interactive
    global pkg_run_tests, pkg_run_unit_tests, HAS_AI_MEDIA_PKG
    
    if not HAS_AI_MEDIA_PKG:
         print("❌ Error: ai_media package not found.")
         sys.exit(1)

    # Check if already loaded
    if pkg_generate_image is not None:
        return

    try:
        # Suppress noisy logging from transformers/diffusers (deferred from startup)
        try:
            import transformers.utils.logging as tf_logging
            tf_logging.set_verbosity_warning()
        except ImportError:
            pass
        try:
            import diffusers.utils.logging as df_logging 
            df_logging.set_verbosity_warning()
        except ImportError:
            pass
        
        import ai_media.utils.parsers as pkg_parsers
        import ai_media.utils.system as pkg_system
        import ai_media.utils.performance as pkg_performance
        import ai_media.utils.ffmpeg as pkg_ffmpeg
        
        from ai_media.generators import (
            generate_image,
            generate_audio,
            generate_video,
            generate_caption,
            generate_edit,
            remove_background,
            ArticleGenerator,
        )
        pkg_generate_image = generate_image
        pkg_generate_audio = generate_audio
        pkg_generate_video = generate_video
        pkg_generate_caption = generate_caption
        pkg_generate_edit = generate_edit
        pkg_remove_background = remove_background
        PkgArticleGenerator = ArticleGenerator

        from ai_media.generators.subtitles import SubtitlesGenerator
        PkgSubtitlesGenerator = SubtitlesGenerator
        
        from ai_media.generators.translation import TranslationGenerator
        pkg_generate_translation = TranslationGenerator().run
        
        from ai_media.generators.transcription import TranscriptionGenerator
        PkgTranscriptionGenerator = TranscriptionGenerator
        
        from ai_media.utils.performance import (
            PerformanceTracker,
            ResourceMonitor,
        )
        PkgPerformanceTracker = PerformanceTracker
        PkgResourceMonitor = ResourceMonitor
        
        from ai_media.conversion import (
            convert_image,
            convert_video,
            convert_audio,
            convert_document,
        )
        pkg_convert_image = convert_image
        pkg_convert_video = convert_video
        pkg_convert_audio = convert_audio
        pkg_convert_document = convert_document
        
        from ai_media.upscaling import (
            simple_upscale_image,
            simple_upscale_video,
            upscale_image_fast,
            upscale_video_fast,
            upscale_image_file,
            upscale_video_file,
        )
        pkg_simple_upscale_image = simple_upscale_image
        pkg_simple_upscale_video = simple_upscale_video
        pkg_upscale_image_fast = upscale_image_fast
        pkg_upscale_video_fast = upscale_video_fast
        pkg_upscale_image_file = upscale_image_file
        pkg_upscale_video_file = upscale_video_file
        
        from ai_media.interactive import (
            prompt_menu,
            prompt_choice,
            prompt_text,
            prompt_file,
            browse_files,
            check_overwrite,
            clear_screen,
            show_header,
            get_key,
            JUMP_POINTS,
            run_interactive,
        )
        pkg_prompt_menu = prompt_menu
        pkg_prompt_choice = prompt_choice
        pkg_prompt_text = prompt_text
        pkg_prompt_file = prompt_file
        pkg_browse_files = browse_files
        pkg_check_overwrite = check_overwrite
        pkg_clear_screen = clear_screen
        pkg_show_header = show_header
        pkg_get_key = get_key
        PKG_JUMP_POINTS = JUMP_POINTS
        pkg_run_interactive = run_interactive
        
        from ai_media.testing.integration_tests import (
            run_tests,
            run_unit_tests,
        )
        pkg_run_tests = run_tests
        pkg_run_unit_tests = run_unit_tests

        HAS_AI_MEDIA_PKG = True # Should be true already
    except ImportError as e:
        print(f"❌ Error: Failed to load ai_media lazy modules: {e}")
        sys.exit(1)
# ---------------------------------------------------


import ast

import json
import logging
import os
import signal
import sys
import re
import argparse
import time

from datetime import datetime
from datetime import datetime
import shutil
import subprocess
try:
    from ai_media.server.config import CONFIG
except ImportError:
    CONFIG = {"paths": {"media_output": "output"}}  # Fallback

try:
    import psutil  # For resource checking
except ImportError:
    psutil = None
from pathlib import Path
from datetime import datetime



def _check_ffmpeg_encoder(encoder_name, w=256, h=256):
    """
    Check if FFmpeg can actually initialize the given encoder at target resolution.
    Used for probing hardware limits (e.g. NVENC max resolution).
    """
    try:
        # Run a tiny 1-frame test encoding to null at target resolution
        cmd = [
            'ffmpeg', '-y', '-f', 'lavfi', '-i', f'nullsrc=s={w}x{h}', 
            '-c:v', encoder_name, '-t', '0.1', '-f', 'null', '-'
        ]
        
        # Suppress output unless verbose debugging is needed
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=5)
        return True
    except:
        return False

# Suppress resource_tracker warnings in subprocesses (caused by os._exit)
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:multiprocessing.resource_tracker"
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Fix for deadlock warning

# CUDA Memory Optimization - Reduce fragmentation on Windows/NVIDIA
# This helps prevent "CUDA out of memory" errors even when GPU has free memory
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


# Global arguments holder for JSON reporting
args = None

try:
    import ftfy
except ImportError:
    ftfy = None




# --- Interactive Mode ---
# -------------------------------------------------------------------
def run_interactive(jump_point=None):
    """Run interactive mode. Uses ai_media.interactive."""
    if HAS_AI_MEDIA_PKG:
        return pkg_run_interactive(jump_point)
    print("❌ ai_media package not available.")
    return


def run_tests(**kwargs):
    """Run integration tests via package."""
    if HAS_AI_MEDIA_PKG:
        # Caller (main) passes verbose and test_filter.
        # integration_tests.run_tests expects test_type as first arg or kwarg.
        # We default to "integration" since --test flag implies it.
        if 'test_type' not in kwargs:
            kwargs['test_type'] = 'integration'
        return pkg_run_tests(**kwargs)
    print("❌ ai_media package not available.")


def run_unit_tests(**kwargs):
    """Run unit tests via package."""
    if HAS_AI_MEDIA_PKG:
        return pkg_run_unit_tests(**kwargs)
    print("❌ ai_media package not available.")


class CleanHelpFormatter(argparse.RawTextHelpFormatter):

    """Custom formatter that hides metavar and uses wider columns."""
    def __init__(self, prog):
        super().__init__(prog, max_help_position=40, width=120)
    
    def _format_action_invocation(self, action):
        if not action.option_strings:
            return super()._format_action_invocation(action)
        if action.nargs != 0 and action.option_strings:
            return ', '.join(action.option_strings)
        return super()._format_action_invocation(action)


def main():
    global _loading_timer

    # Cancel loading message timer (imports complete)
    if _loading_timer:
        _loading_timer.cancel()
    
    parser = argparse.ArgumentParser(
        description="Generate, describe, upscale, and convert media using AI and FFmpeg.",
        formatter_class=CleanHelpFormatter,
        epilog="""
Examples:
  -- Image Generation --
  python ai-media.py -i -p "Cyberpunk city" -o city.png -s 720p
  python ai-media.py -i -p "Forest" -o forest.jpg -s 4k
  python ai-media.py -i -p "Capybara holding a sign" -im sd3.5-medium (SD 3.5 - Consumer-Friendly)
  python ai-media.py -i -p "Astronaut portrait" -im sd3.5-turbo (SD 3.5 - Fast, 4 steps)
  python ai-media.py -i -p "Cyberpunk city" --negative-prompt "blurry, dark, low quality" (With Negative Prompt)
  
  -- Video Generation --
  python ai-media.py -v -p "Robot dancing" -o robot.mp4 -l 5s
  python ai-media.py -v -p "Camera pans left" -ii ./start.png -o output.mp4 (Image-to-Video)
  python ai-media.py -v -p "Dancer" -ap "Techno beat" -o party.mp4 (Video+Audio Mux)
  python ai-media.py -v -p "Future city" -o city.mp4 --video-model wan2.2 (SOTA Quality)
  python ai-media.py -v -p "Driving car" -o car.mp4 --video-model ltx-video (Fast/High-Res)
  python ai-media.py -v -p "Liquid simulation" --video-model mochi-1 (High Motion)
  python ai-media.py -v -p "Cinematic panda" --video-model hunyuan (13B Model)
  
  -- Audio Generation --
  python ai-media.py -a -p "Jazz saxophone" -o jazz.mp3 -l 30s
  python ai-media.py -a -p "Rainforest" -o rain.wav --audio-model audioldm2
  python ai-media.py -a -p "Spooky" -ii ./haunted.jpg -o spooky.mp3 (Image-to-Audio)
  python ai-media.py -a -ii ./image.jpg -cm blip (Image-to-Audio w/ BLIP)
  python ai-media.py -a -ii ./image.jpg (Auto-caption + Audio)
  python ai-media.py -a -ii ./video.mp4 (Auto-caption Video + Audio)
  python ai-media.py -a -p "♪ In the jungle ♪ [laughter]" --audio-model bark (Bark Creative)

  -- Text Generation (Articles, Research, Chat, Code) --
  python ai-media.py -ga -p "Future of AI" -o article.md (Offline)
  python ai-media.py -gr -p "Latest Quantum Computing News" -o research.pdf (Online Deep Research)
  python ai-media.py -c --chat-model llama-3.1-8b (Interactive Chat)
  python ai-media.py -gc 'Write a Snake game in Python' (Code Gen)
  python ai-media.py -ga -p "Story about a cat" -o story.docx -atm mistral-nemo-12b

  -- Analysis --
  python ai-media.py -gd -ii video.mp4
  python ai-media.py -gd -ii image.jpg -cm blip (Use simpler model)

  -- Creative Image Transformation --
  python ai-media.py -ti "photo.jpg" -p "Make it look like an anime drawing"
  python ai-media.py -ti "photo.jpg" -p "Make it anime" -o "edits/anime_version.png"
  python ai-media.py -ti "photo.jpg" -p "Change text to 'Hello World'" -em qwen-image-edit (Precise Text Editing)
  python ai-media.py -ti "photo.jpg" --remove-background
  python ai-media.py -ti "photo.jpg" --remove-background -o "no_bg/photo_clean.png"

  -- Media Conversion --
  python ai-media.py -ci photo.gif -cit png
  python ai-media.py -cv clip.mov -cvt mp4
  python ai-media.py -ca song.wav -cat mp3
  python ai-media.py -cd image.jpg -cdt txt (Extract Text / OCR)
  
  -- AI Upscaling --
  python ai-media.py -ui input.jpg -uf 2x
  python ai-media.py -ui input.jpg -uf 4x
  python ai-media.py -ui input.jpg -uf 4x -su (Simple Upscale)
  python ai-media.py -uv input.mp4 -vu realesrgan (Fast AI - Recommended)
  python ai-media.py -uv input.mp4 -uf 2x -vu sd (High Detail AI)

  -- Web Server --
  python ai-media.py --serve (Loads Server + both Web & Electron)
  python ai-media.py --serve-web-only-client
  python ai-media.py --serve-no-client (Server only)
  python ai-media.py --serve --reload


Supported Models (Code : Download Size | Description):
  Images:
    - sd3.5-turbo (default)      : ~19GB | SD 3.5 Turbo. Fast (4 steps). (🔒 Gated - Free Login Required)
    - sdxl                       : ~8GB  | Fast, high quality.
    - sd-1.5                     : ~4GB  | Lightweight, lower VRAM.
    - sd3.5-medium               : ~10GB | SD 3.5. Consumer-friendly. (🔒 Gated - Free Login Required)
    - sd3.5-large                : ~19GB | SD 3.5. Best quality. (🔒 Gated - Free Login Required)
    - qwen-image-auto            : ~40GB | Best text rendering. (Auto-selects 4-bit CUDA / Full MPS)
    - qwen-image-lightning       : ~40GB | Lightning Fast (8-step). (MPS/CUDA)
    - qwen-image-4bit            : ~20GB | 4-bit Lite Qwen. (CUDA Only)
    - qwen-image-2512            : ~40GB | Qwen-Image 2512 (Latest). (MPS/Full, float32)
    - flux                       : ~24GB | High quality (🔒 Gated - Free Login Required)
    - flux-dev                   : ~24GB | Professional creative work (🔒 Gated - Free Login Required)
    - flux2                      : ~18GB | FLUX.2 4-bit. SOTA (2025). (🔒 Gated - NVIDIA RTX 3090+ only)
    - flux2-full                 : ~65GB | FLUX.2 full. (🔒 Gated - Mac: 128GB+ RAM required)
  
  Video:
    - zeroscope (default)        : ~4GB  | Fast, no watermarks. Auto-upscales with XL.
    - ms-1.7b                    : ~10GB | General purpose (has watermark issues).
    - cogvideox                  : ~15GB | High fidelity.
    - svd                        : ~4GB  | I2V Only.
    - wan2.2                     : ~30GB | SOTA (2025). Excellent quality.
    - ltx-video                  : ~12GB | Balanced speed/quality. Good motion.
    - mochi-1                    : ~19GB | High motion fidelity.
    - hunyuan                    : ~25GB | Massive scale.
    
  Audio:
    - musicgen-small             : ~2GB  | Fast, lightweight. Good for quick sketches.
    - musicgen-medium (default)  : ~6GB  | Balanced quality/speed.
    - musicgen-large             : ~10GB | High fidelity. Slower.
    - audioldm2                  : ~4GB  | Specialized in Sound Effects (SFX), foley, environmental.
    - stable-audio               : ~10GB | Variable-length, high-quality music/SFX (🔒 Gated - Free Login Required)
    - bark                       : ~4GB  | Realistic speech, music, and sound effects.
    
  Text (Articles, Research, Chat, Code):
    - deepseek-r1-qwen-7b        : ~7GB  | R1 distilled to Qwen-7B. Step-by-step reasoning.
    - deepseek-r1-qwen-14b       : ~14GB | R1 distilled to Qwen-14B. Better reasoning.
    - deepseek-r1-qwen-32b       : ~24GB | ⚠️ HIGH RAM! R1 distilled to Qwen-32B.
    - deepseek-r1-llama-8b       : ~8GB  | R1 distilled to Llama-8B. Reasoning-focused.
    - deepseek-r1-llama-70b      : ~40GB | ⚠️ HIGH RAM! R1 distilled to Llama-70B.
    - llama-3.1-8b (default)     : ~16GB | Writing, chat, and reasoning (🔒 Gated - Free Login Required)
    - mistral-nemo-12b           : ~24GB | Powerful 12B model. Large context and reasoning.
    - qwen3-8b                   : ~16GB | Qwen 3 8B (Reasoning). Strong instruction-following.
    - qwen3-14b                  : ~28GB | Qwen 3 14B (Reasoning). Great at detailed formatting.
    - qwen-coder-32b             : ~24GB | Qwen 2.5 SOTA Code Gen. (⚠️ 120GB+ RAM!)
    - qwen-coder-14b             : ~12GB | Qwen 2.5 Fast & Capable Code Gen.
    - qwen-coder-7b              : ~6GB  | Qwen 2.5 Lightweight Code Gen.
    - qwen3-coder-30b            : ~10GB | MoE (3.3B active). Efficient SOTA.

  Analysis:
    - florence (default)         : ~1.5GB | SOTA details, rich descriptions, "seeing" the scene. Fast.
    - qwen-vl                    : ~30GB  | Qwen3-VL 8B. High precision OCR & captioning.
    - blip                       : ~1GB   | Simple, concise captions. Faster but less detailed. (Not for OCR)
    - qwen3-vl-8b                : ~16GB  | Qwen3-VL 8B (explicit).
    - qwen3-vl-4b                : ~8GB   | Qwen3-VL 4B. Balanced.
    - qwen3-vl-2b                : ~4GB   | Qwen3-VL 2B. Lightweight.

  Creative Image Transformation:
    - instruct-pix2pix           : ~4GB  | Instructional image editing (e.g., "Make it anime").
    - instruct-pix2pix-sdxl      : ~8GB  | High quality, slow.
    - qwen-image-edit            : ~20GB | Best for text editing, precision. (Official Base 2511)
    - qwen-image-edit-lightning  : ~16GB | 4-step LoRA. Fast on CUDA. (⚠️ Slow on MPS!)
    - remove-bg                  : ~1GB  | Background removal and silhouette creation.

  Upscaling:
    - x2 (≤2x factor)            : ~4GB  | Fast, preserves original style.
    - x4 (>2x factor)            : ~8GB  | High detail, sharpens textures.
    - Real-ESRGAN x4plus         : ~0.3GB| Fast, faithful upscaling, better temporal consistency.
        """
    )
    
    # Generation Mode (First - what do you want to create?)
    mode_group = parser.add_argument_group("Generation Mode")
    mode_group.add_argument("-i", "--generate-image", action="store_true", help="Generate Image")
    mode_group.add_argument("-v", "--generate-video", action="store_true", help="Generate Video")
    mode_group.add_argument("-a", "--generate-audio", action="store_true", help="Generate Audio")
    
    # NEW: Article Modes
    mode_group.add_argument("-ga", "--generate-article", action="store_true", help="Generate Article (Offline)")
    mode_group.add_argument("-gr", "--generate-research", action="store_true", help="Generate Article + Research (Online)")
    mode_group.add_argument("-c", "--chat", action="store_true", help="Interactive Chat Mode")
    mode_group.add_argument("-gc", "--generate-code", nargs="?", const=True, default=False, 
                            help="Generate code. Supports multi-file projects & auto-naming. E.g.: -gc 'Write a REST API'")
    mode_group.add_argument("-gs", "--generate-subtitles", action="store_true", help="Generate Subtitles (using faster-whisper + NLLB).")
    mode_group.add_argument("-trans", "--transcribe", action="store_true", help="Transcribe Audio/Video to text/markdown.")
    
    mode_group.add_argument("-gd", "--generate-description", nargs="?", const="USE_INPUT_IMAGE", help="Analysis - Generate Description for Image or Video.")
    mode_group.add_argument("-ti", "--transform-image", nargs="?", const="USE_GENERATED", metavar="FILE", help="Transform an image. Omit FILE to auto-use generated output from -i.")
    
    # Common Parameters (applies to most modes)
    common_group = parser.add_argument_group("Common Parameters")
    common_group.add_argument("-p", "--prompt", required=False, help="Text prompt description (Required for generation modes)")
    common_group.add_argument("-o", "--output", help="Output file path. Auto-generated from prompt if omitted.")
    common_group.add_argument("--force", action="store_true", help="Skip all confirmation prompts (overwrites files and ignores resource warnings).")
    common_group.add_argument("--bypass-warning", action="store_true", help="Specifically skip resource warning prompts (safe for web client).")
    common_group.add_argument("-f", "--format", help="File format. Image: jpg/png (default: jpg). Video: mp4. Audio: mp3/wav (default: mp3). Article: md/pdf/doc/html.")
    common_group.add_argument("-s", "--size", help="Resolution: '720p', '1080p', '4k', '1280x720', '1536' (square). For zeroscope: triggers dynamic upscaling (XL + Real-ESRGAN) for targets > 576x320. Default: 720p")
    common_group.add_argument("-npt", "--no-performance-tracking", action="store_true", help="Disable performance tracking (performance.json).")
    
    
    # Specific options
    image_group = parser.add_argument_group("Image Options")
    image_models_help = [k + " (Gated)" if k in ["flux", "flux-dev", "flux2", "flux2-full", "qwen-image-lightning"] else k for k in IMAGE_MODELS.keys()]
    image_group.add_argument("-im", "--image-model", default="default", help=f"Model: {', '.join(image_models_help)}")
    image_group.add_argument("-otn", "--orientation", choices=["landscape", "portrait", "square"], default="landscape",
                              help="Orientation for SDXL/Flux generation. 'portrait' swaps width/height.")
    image_group.add_argument("--negative-prompt", help="Negative prompt (what to avoid). Only supported by standard models (SD 1.5, SD 3.5, Qwen). Ignored by Turbo/Flux.")
    image_group.add_argument("--unsafe", action="store_true", help="Disable NSFW safety checker (Use with caution).")
    
    video_group = parser.add_argument_group("Video Options")
    video_group.add_argument("-vm", "--video-model", default="default", help=f"Model: {', '.join(VIDEO_MODELS.keys())} (default: zeroscope)")
    video_group.add_argument("-l", "--length", default="2s", help="Duration (e.g. '2s', '5s', '1m', '{m:1, s:30}'). Default: 2s")
    video_group.add_argument("-ii", "--input-image", help="Input image for Image-to-Video generation.")
    video_group.add_argument("-ap", "--audio-prompt", help="Audio prompt for 'Video with Audio' generation (merged via FFmpeg).")
    video_group.add_argument("-vc", "--video-codec", choices=['auto', 'h264', 'hevc', 'av1'], default='auto',
                             help="Video Codec: h264, hevc, av1, or auto. AV1 uses hardware encoder (av1_nvenc) when available. Default: auto")
    
    audio_group = parser.add_argument_group("Audio Options")
    audio_models_help = [k + " (Gated)" if k in ["stable-audio"] else k for k in AUDIO_MODELS.keys()]
    audio_group.add_argument("-am", "--audio-model", default="default", help=f"Model: {', '.join(audio_models_help)}")
    audio_group.add_argument("--voice-preset", default="v2/en_speaker_6", help="Bark Voice Preset (e.g. 'v2/en_speaker_6'). Default: v2/en_speaker_6")
    audio_group.add_argument("-m", "--sampling-rate", type=str, default="32000", help="Sampling rate (e.g. 32000, 44.1k). Default: 32000.")
    audio_group.add_argument("-b", "--bit-depth", type=int, choices=[16, 24, 32], default=16, help="Bit depth for audio conversion.")
    audio_group.add_argument("-r", "--bit-rate", help="Bit rate (e.g. 192k) for audio conversion.")
    
    # NEW: Text Options
    text_group = parser.add_argument_group("Text/Article Options (Articles, Research, Chat, Code)")
    text_models_help = [k + " (Gated)" if k in ["llama-3.1-8b"] else k for k in TEXT_MODELS.keys()]
    text_group.add_argument("-atm", "--article-model", default="default", 
                            help=f"Model for articles/research (-ga/-gr). Options: {', '.join(text_models_help)}")
    text_group.add_argument("-chm", "--chat-model", default="default", 
                            help=f"Model for chat (-c). Options: {', '.join(text_models_help)}")
    text_group.add_argument("-cdm", "--code-model", default="default", 
                            help=f"Model for code gen (-gc). Options: {', '.join(text_models_help)}")
    text_group.add_argument("--output-format", choices=["md", "pdf", "docx", "rtf", "html", "xhtml", "json", "txt"], default="md", 
                            help="Article/research output format. Default: md")
    text_group.add_argument("-ri", "--research-iter", type=int, default=3, 
                            help="Deep research: number of sources to read. Default: 3")
    text_group.add_argument("-al", "--article-length", choices=["quick", "standard", "detailed"], default="quick",
                            help="Article length: quick (fast, ~500 words, default), standard (~1500 words), detailed (comprehensive, ~3000 words).")

    # Translation Options
    trans_group = parser.add_argument_group("Translation Options")
    trans_group.add_argument("-tr", "--translate", action="store_true", help="Translate text or matching media file.")
    trans_group.add_argument("-tl", "--target-language", help="Target language code (e.g. 'fra', 'deu', 'spa'). Required for translation.")
    trans_group.add_argument("-sl", "--source-language", help="Source language code (e.g. 'eng'). Optional/Auto.")
    trans_group.add_argument("-tm", "--translation-model", default="nllb-200-3.3b", 
                             choices=list(TRANSLATION_MODELS.keys()), 
                             help=f"Translation model. Options: {', '.join(k for k in TRANSLATION_MODELS.keys() if not k.startswith('default'))}. Default: nllb-200-3.3b")

    # Analysis Options
    caption_group = parser.add_argument_group("Analysis Options")
    caption_group.add_argument("-cm", "--caption-model", default="florence", choices=["florence", "blip", "qwen-vl", "qwen3-vl-8b", "qwen3-vl-4b", "qwen3-vl-2b"], help="Analysis models: 'florence' (default), 'blip', 'qwen-vl' (Qwen3-VL-8B), or 'qwen3-vl-2b/4b/8b'.")
    
    # Subtitles Options
    subtitles_group = parser.add_argument_group("Subtitles Options")
    subtitles_group.add_argument("--subtitle-format", default="srt", 
                                 choices=["srt", "vtt", "ass", "sub", "txt", "json"],
                                 help="Output format: srt (default), vtt, ass, sub, txt, json")
    subtitles_group.add_argument("--subtitle-fps", type=float, default=25.0,
                                 help="FPS for SUB format (frame-based). Default: 25.0")
    subtitles_group.add_argument("--subtitle-translate-to", 
                                 help="Translate subtitles to target language(s). Comma-separated (e.g. 'fr,es,ja').")
    subtitles_group.add_argument("--subtitle-source-lang", 
                                 help="Source language code (e.g. 'en'). Auto-detected if omitted.")
    subtitles_group.add_argument("--whisper-model", default="small", 
                                 choices=["tiny", "base", "small", "medium", "large-v3"], 
                                 help="Whisper model size. Default: small")
    subtitles_group.add_argument("--subtitle-vad-preset", default="normal",
                                 choices=["normal", "noisy", "sensitive"],
                                 help="VAD preset: normal (default), noisy (strict, for noisy recordings), sensitive (for quiet speech)")
    subtitles_group.add_argument("--subtitle-vad-min-silence", type=int,
                                 help="VAD: Min silence duration (ms) to split segments. Default: 2000")
    subtitles_group.add_argument("--subtitle-vad-threshold", type=float,
                                 help="VAD: Speech probability threshold (0.0-1.0). Default: 0.5")
    subtitles_group.add_argument("--subtitle-no-context", action="store_true",
                                 help="Disable conditioning on previous text (prevents hallucination loops).")
    # Legacy compatibility aliases
    subtitles_group.add_argument("--translate-to", dest="subtitle_translate_to_legacy",
                                 help=argparse.SUPPRESS)  # Hidden, for backward compat
    subtitles_group.add_argument("--source-lang", dest="subtitle_source_lang_legacy",
                                 help=argparse.SUPPRESS)  # Hidden, for backward compat

    
    # Creative Image Transformation
    transform_group = parser.add_argument_group("Creative Image Transformation Options")
    transform_group.add_argument("-tp", "--transform-prompt", help="Edit instruction for InstructPix2Pix (e.g., 'Make it anime'). Used with -ti.")
    edit_models_help = [k + " (Gated)" if k in ["qwen-image-edit-lightning"] else k for k in EDIT_MODELS.keys()]
    transform_group.add_argument("-em", "--edit-model", default="default", help=f"Model for image editing. Options: {', '.join(edit_models_help)}")
    transform_group.add_argument("-rb", "--remove-background", action="store_true", help="Remove background (Transparent PNG).")
    transform_group.add_argument("--silhouette", action="store_true", help="Create a black silhouette (requires -rb).")
    transform_group.add_argument("--image-guidance", type=float, default=1.5, help="Image guidance scale (default: 1.5). Higher = closer to original.")

    # Media Conversion (Standalone - No AI)
    convert_group = parser.add_argument_group("Media Conversion Options")
    convert_group.add_argument("-ci", "--convert-image", metavar="FILE", help="Convert image format (e.g., gif->png)")
    convert_group.add_argument("-cit", "--convert-image-to", metavar="FMT", help="Output format (png, .webp, out.jpg)")
    convert_group.add_argument("-cv", "--convert-video", metavar="FILE", help="Convert video (mov->mp4)")
    convert_group.add_argument("-cvt", "--convert-video-to", metavar="FMT", help="Output format (mp4, .webm, out.avi)")
    convert_group.add_argument("-ca", "--convert-audio", metavar="FILE", help="Convert audio (wav->mp3)")
    convert_group.add_argument("-cat", "--convert-audio-to", metavar="FMT", help="Output format (mp3, .flac, out.ogg)")
    convert_group.add_argument("--convert-image-engine", choices=["pil", "ffmpeg"], default="pil", help="pil (default) or ffmpeg")
    
    # Document Conversion
    doc_conv_group = parser.add_argument_group("Document Conversion Options")
    doc_conv_group.add_argument("-cd", "--convert-document", metavar="FILE", help="Convert document (e.g., docx->pdf) or extract text from image (ocr).")
    doc_conv_group.add_argument("-cdt", "--convert-document-to", metavar="FMT", help="Output format: md, html, pdf, docx, rtf, txt, json. Use 'txt' or 'md' for images to trigger OCR.")
    doc_conv_group.add_argument("-om", "--ocr-model", default="florence", choices=["florence", "qwen-vl"],
                               help="OCR Model: 'florence' (default, fast, ~1.5GB RAM) or 'qwen-vl' (high precision).")
    
    # AI Upscaling (Standalone Mode)
    upscale_mode_group = parser.add_argument_group("AI Upscaling Options")
    upscale_mode_group.add_argument("-ui", "--upscale-image", metavar="FILE", help="Upscale an existing image")
    upscale_mode_group.add_argument("-uv", "--upscale-video", metavar="FILE", help="Upscale an existing video")
    upscale_mode_group.add_argument("-iu", "--image-upscaler", choices=["sd", "realesrgan"], default="realesrgan", help="Model for image upscaling: 'realesrgan' (Fast, faithful, default) or 'sd' (Stable Diffusion, slower, creative)")
    upscale_mode_group.add_argument("-vu", "--video-upscaler", choices=["sd", "realesrgan"], default="realesrgan", help="Model for video upscaling: 'realesrgan' (Fast, faithful, default) or 'sd' (Stable Diffusion, slow, detailed)")

    # Upscaling Options (applies to both standalone and chained upscaling)
    upscale_group = parser.add_argument_group("Upscaling Options")
    upscale_group.add_argument("-uf", "--upscale-factor", help="Upscale factor (e.g. '2x', '4'). Default: 2x")
    upscale_group.add_argument("--upscale", action="store_true", help="Enable AI Upscaling after generation (chained mode).")
    upscale_group.add_argument("-uof", "--upscaled-output-file", help="Custom filename for the upscaled output (e.g. 'highres.png').")
    upscale_group.add_argument("-us", "--upscale-strength", type=float, default=0.0, help="Upscale creativity/strength (0.0-1.0). Default: 0.0")
    upscale_group.add_argument("-su", "--simple-upscale", action="store_true", help="Use simple non-AI upscaling (PIL Lanczos for images, FFmpeg for videos). Very fast.")
    
    # Testing
    test_group = parser.add_argument_group("Testing")
    test_group.add_argument("--test", nargs="*", help="Run integration tests from tests/integration-tests.json (quiet mode). Optional: Space-separated list of Test Names.")
    test_group.add_argument("--test-verbose", nargs="*", help="Run integration tests with full output. Optional: Space-separated list of Test Names.")
    test_group.add_argument("--unittests", nargs="?", const="ai_media.testing.unit_tests", metavar="MODULE",
                            help="Run Python unit tests (Quiet/Summary mode). Default: all tests. Examples: ai_media.testing.unit_tests.TestParseSize")
    test_group.add_argument("--unittests-verbose", nargs="?", const="ai_media.testing.unit_tests", metavar="MODULE",
                            help="Run Python unit tests (Verbose mode). Default: all tests.")
    
    # Interactive Mode
    parser.add_argument("-I", "--interactive", nargs="?", const="menu", metavar="JUMP",
                        help="Run in interactive mode. Optional: Jump point (e.g., 'image/sdxl', 'audio/bark').")
    
    # Server Mode
    server_group = parser.add_argument_group("Web Server")
    server_group.add_argument("--serve", action="store_true", help="Start the web server and launch both Web and Electron clients (Loads Server + both Web & Electron)")
    server_group.add_argument("--serve-web-only-client", action="store_true", help="Start the web server and launch only the Web client")
    server_group.add_argument("--serve-no-client", action="store_true", help="Start the backend server only (no clients)")
    server_group.add_argument("--reload", action="store_true", help="Enable auto-reload for development (On by default for Client modes)")
    server_group.add_argument("--host", default=None, help="Host for the web server (Overrides config)")
    server_group.add_argument("--port", type=int, default=None, help="Port for the backend server (Overrides config)")
    
    parser.add_argument("--report-json", help="Path to write a JSON report of the generation stats")
    parser.add_argument("--list-models", action="store_true", help="List all available models and exit.")

    # --- Parse Arguments ---
    args = parser.parse_args()
    
    # List Models
    if args.list_models:
        print("🤖 Available AI Models:")
        print("="*60)
        
        def print_category(name, models_dict):
            print(f"\n[{name}]")
            # Filter aliases to avoid clutter
            hidden_aliases = [
                "qwen-image-edit-mps",
                "wan2.2", # Alias for wan-2.2
            ]
            for short_code, hf_id in models_dict.items():
                if short_code in hidden_aliases:
                    continue
                desc = " (Default)" if short_code == "default" else ""
                print(f"  • {short_code:<25}: {hf_id}{desc}")
                
        print_category("Image Generation", IMAGE_MODELS)
        print_category("Video Generation", VIDEO_MODELS)
        print_category("Audio Generation", AUDIO_MODELS)
        print_category("Text / Reasoning", TEXT_MODELS)
        print_category("Captioning", CAPTION_MODELS)
        print_category("Image Editing", EDIT_MODELS)
        sys.exit(0)
    
    # Interactive Mode Trigger - handled BEFORE heavy module loading for instant startup
    if args.interactive or len(sys.argv) == 1:
        # Only import lightweight interactive module, skip heavy AI generators
        from ai_media.interactive import run_interactive
        run_interactive(jump_point=args.interactive if args.interactive != "menu" else None)
        sys.exit(0)
    
    # Server Mode - start web server
    serve_any = args.serve or args.serve_web_only_client or args.serve_no_client
    if serve_any:
        # Load config for host/ports (allowing CLI overrides)
        host = args.host if args.host else CONFIG["server"]["host"]
        server_port = args.port if args.port else CONFIG["server"]["port"]
        web_port = CONFIG["client"]["port"]
        
        # Determine if reload should be on (default True for clients, False for server-only)
        reload_enabled = args.reload
        if not reload_enabled and (args.serve or args.serve_web_only_client):
            reload_enabled = True
        
        # Determine which clients to start
        start_web = args.serve or args.serve_web_only_client
        start_electron = args.serve
        
        procs = []
        try:
            # Helper to launch clients
            import subprocess
            def start_client(cmd, name, env=None):
                print(f"🚀 Launching {name} client...")
                web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_media", "web")
                return subprocess.Popen(cmd, shell=True, cwd=web_dir, env=env)

            # Build environment for clients to know about ports
            env = os.environ.copy()
            env["VITE_API_PORT"] = str(server_port)
            env["VITE_WEB_PORT"] = str(web_port)
            
            if start_web or start_electron:
                def delayed_launch():
                    # Wait for server to be ready (smart polling)
                    import socket
                    print("⏳ Waiting for server to start...", end="", flush=True)
                    
                    retries = 30 # Wait up to 30 seconds
                    server_ready = False
                    
                    while retries > 0:
                        try:
                            with socket.create_connection((host, server_port), timeout=1):
                                server_ready = True
                                break
                        except (OSError, ConnectionRefusedError):
                            time.sleep(1)
                            print(".", end="", flush=True)
                            retries -= 1
                    
                    print(f"\n{'✅ Server ready!' if server_ready else '⚠️ Server wait timed out (proceeding anyway)'}")
                    
                    # Helper nested here to capture procs list
                    def start_client(cmd, name, env=None):
                        print(f"🚀 Launching {name} client...")
                        web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_media", "web")
                        return subprocess.Popen(cmd, shell=True, cwd=web_dir, env=env)

                    if start_web:
                        procs.append(start_client("npm run dev:client", "Web", env=env))
                    if start_electron:
                        # Small delay for Electron just to ensure Web/Vite bundler starts its own output first
                        # (Purely cosmetic to keep logs clean)
                        time.sleep(1)
                        procs.append(start_client("npm run electron", "Electron", env=env))

                # Start the launch thread
                import threading
                launch_thread = threading.Thread(target=delayed_launch)
                launch_thread.daemon = True
                launch_thread.start()
            
            from ai_media.server import main as server_main
            
            # Restrict watcher to source code only to avoid loops with generated content in media-output/
            reload_dirs = ["ai_media", "ai-media.py"] if reload_enabled else None
            
            # Print newline before server starts to separate it from any background thread output
            print() 
            server_main(host=host, port=server_port, reload=reload_enabled, reload_dirs=reload_dirs)
        finally:
            # Cleanup background processes
            for p in procs:
                try:
                    print(f"🛑 Stopping client process {p.pid}...")
                    p.terminate()
                except Exception:
                    pass
        sys.exit(0)
    
    # LAZY LOADING: Load heavy AI modules ONLY for non-interactive modes
    load_ai_modules()
    
    # Parse upscale factor (convert "2x" string to 2.0 float)
    if args.upscale_factor:
        args.upscale_factor = pkg_parsers.parse_upscale_factor(args.upscale_factor)
    else:
        args.upscale_factor = 2.0  # Default
    
    # Set AI_MEDIA_FORCE env var so all internal functions respect --force flag
    if args.force:
        os.environ["AI_MEDIA_FORCE"] = "1"
    if args.bypass_warning:
        os.environ["AI_MEDIA_BYPASS_WARNING"] = "1"

    # Test Triggers (Priority over generation)
    if args.test is not None:
        # Caller (main) passes verbose and test_filter.
        # we default to "integration" since --test flag implies it.
        params = {'verbose': False, 'test_filter': args.test if len(args.test) > 0 else None, 'test_type': 'integration'}
        pkg_run_tests(**params)
    if args.test_verbose is not None:
        params = {'verbose': True, 'test_filter': args.test_verbose if len(args.test_verbose) > 0 else None, 'test_type': 'integration'}
        pkg_run_tests(**params)
    if args.unittests is not None:
        pkg_run_unit_tests(test_name=args.unittests, verbose=False) 
        sys.exit(0)
    elif args.unittests_verbose is not None:
        pkg_run_unit_tests(test_name=args.unittests_verbose, verbose=True)
        sys.exit(0)

    # Prompt Check (Required for non-chat modes, but not for I2V with input_image)
    if not args.chat and not args.prompt and not args.input_image and not any([args.generate_description, args.transform_image, args.convert_image, args.convert_video, args.convert_audio, args.convert_document, args.upscale_image, args.upscale_video, args.generate_code]):
        from ai_media.utils.interaction import emoji
        print(f"{emoji('❌', '[X]')} Error: Prompt is required (use -p 'Your prompt')")
        sys.exit(1)
        
    # --- Dispatch ---
    
    # NEW: Article Generation Dispatch
    if args.chat:
        gen = PkgArticleGenerator(model_name=args.chat_model)
        gen.chat_session()
        sys.exit(0)
        
        sys.exit(0)

    elif args.generate_code:
        gen = PkgArticleGenerator(model_name=args.code_model)
        
        # Determine prompt: use -gc value if string, else use main prompt
        prompt = args.generate_code if isinstance(args.generate_code, str) else args.prompt
        
        if not prompt:
            print("❌ Error: Code generation code requires a prompt (e.g. -gc 'Write a script...' or -gc -p '...')")
            sys.exit(1)
            
        # Default to media output dir if no output specified
        code_output = args.output
        if not code_output:
            code_output = CONFIG["paths"]["media_output"]
            
        gen.generate_code(prompt, output_file=code_output)
        sys.exit(0)

    elif args.generate_article or args.generate_research:
        online = args.generate_research
        gen = PkgArticleGenerator(model_name=args.article_model)
        
        # Determine output format - prefer extension from -o, then --output-format
        output_format = args.output_format  # Default from --output-format flag
        
        if args.output:
            outfile = args.output
            # Infer format from filename extension if present
            _, ext = os.path.splitext(args.output)
            if ext:
                ext_lower = ext.lower().lstrip('.')
                valid_formats = ["md", "pdf", "docx", "rtf", "html", "xhtml", "json", "txt"]
                if ext_lower in valid_formats:
                    output_format = ext_lower
        else:
            # Slugify the prompt: lowercase, replace spaces with underscores, remove special chars
            import re
            slug = re.sub(r'[^\w\s-]', '', args.prompt.lower())
            slug = re.sub(r'[-\s]+', '_', slug).strip('_')[:50]  # Limit to 50 chars
            outfile = f"{slug}.{output_format}" if slug else f"article_{int(time.time())}.{output_format}"
            
            # Use configured output directory
            if not os.path.dirname(outfile):
                 outfile = os.path.join(CONFIG["paths"]["media_output"], outfile)
        
        # Ensure extension matches format
        base, _ = os.path.splitext(outfile)
        if not outfile.lower().endswith(f".{output_format}"):
            outfile = f"{base}.{output_format}"
        
        # Pre-generation overwrite check (unless --force)
        if not args.force:
            should_write, outfile, _, _ = pkg_check_overwrite(outfile)
            if not should_write:
                print("⏭️  Skipped.")
                sys.exit(0)
        
        gen.generate_article(
            topic=args.prompt, 
            output_file=outfile, 
            format=output_format,
            online=online,
            research_iter=args.research_iter,
            length=args.article_length
        )
        sys.exit(0)

    # -------------------------------------------------------------------
    # Auto-generate output filename from prompt if not provided
    # This centralized logic handles all generation modes
    # -------------------------------------------------------------------
    if any([args.generate_image, args.generate_video, args.generate_audio, args.transform_image, args.transcribe]):
        import re
        if not args.output:
            # Sanitize prompt to create safe filename (first 2 words, alphanumeric only)
            if args.prompt:
                words = re.findall(r'[a-zA-Z0-9]+', args.prompt.lower())[:2]
                if words:
                    args.output = "-".join(words)
                    print(f"ℹ️  No output specified. Using: {args.output}")
                else:
                    print("⚠️  Cannot auto-generate filename: prompt contains no valid words.")
                    args.output = f"output_{int(time.time())}"
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
                args.output = f"output_{int(time.time())}"
        
        # Prepend configured output directory
        if not os.path.dirname(args.output):
             args.output = os.path.join(CONFIG["paths"]["media_output"], args.output)

        # Smart Extension Handling
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
                elif args.generate_video:
                    args.output += ".mp4"
                elif args.generate_audio:
                    args.output += ".mp3"
                elif args.transform_image:
                    args.output += ".png"

        pkg_system.ensure_paths(args.output)
        
        # Check for existing file
        if not args.force:
            should_write, args.output, _, _ = pkg_check_overwrite(args.output)
            if not should_write:
                sys.exit(0)

    if args.generate_image:
        w, h = pkg_parsers.parse_size(args.size)
        if args.orientation == "portrait": w, h = h, w
        
        outfile = args.output  # Already set by centralized auto-filename logic
        
        # Resolve Model ID
        model_id = get_model_id(args.image_model, IMAGE_MODELS)
            
        success = pkg_generate_image(args.prompt, outfile, w, h, model_id, negative_prompt=args.negative_prompt, unsafe=args.unsafe, force=args.force, report_json=args.report_json)
        if not success:
            sys.exit(1)
        
        if args.upscale or args.simple_upscale:
            base, ext = os.path.splitext(outfile)
            suffix = "simple_upscaled" if args.simple_upscale else "upscaled"
            upscale_out = args.upscaled_output_file or f"{base}_{suffix}_{args.upscale_factor}{ext}"
            
            if args.simple_upscale:
                pkg_simple_upscale_image(outfile, upscale_out, factor=args.upscale_factor, force=args.force)
            else:
                # If we have an upscaler model selected (default is 'realesrgan')
                upscaler_model = args.image_upscaler or 'realesrgan'
                if upscaler_model == 'realesrgan':
                    pkg_upscale_image_fast(outfile, upscale_out, factor=args.upscale_factor)
                else:
                    pkg_upscale_image_file(outfile, upscale_out, strength=args.upscale_strength, factor=args.upscale_factor)

    elif args.generate_audio:
        dur = pkg_parsers.parse_duration(args.length)
        sr = pkg_parsers.parse_sampling_rate(args.sampling_rate)
        outfile = args.output  # Already set by centralized auto-filename logic
             
        if args.input_image:
             if not os.path.exists(args.input_image):
                 print(f"❌ Error: Input file not found: {args.input_image}")
                 sys.exit(1)
             
             print(f"👁️  Analyzing image: {args.input_image}...")
             device, _ = pkg_system.get_optimal_device_and_dtype(quiet=True, prefer_bfloat16=True)
             caption = pkg_generate_caption(args.input_image, device, model_type=args.caption_model)
             if not caption:
                 print("   Failed to generate description.")
                 sys.exit(1)
             print(f"   📝 Caption: '{caption}'")
             
             full_prompt = caption
             if args.prompt:
                 full_prompt = f"{args.prompt}. {caption}"
             
             print(f"   🎶 Generating audio for: '{full_prompt}'")
             success = pkg_generate_audio(full_prompt, outfile, dur, sr, model_name=args.audio_model, report_json=args.report_json)
             if not success:
                 sys.exit(1)
             
        else:
             success = pkg_generate_audio(args.prompt, outfile, dur, sr, model_name=args.audio_model, report_json=args.report_json)
             if not success:
                 sys.exit(1)

    elif args.generate_video:
        w, h = pkg_parsers.parse_size(args.size)
        dur = pkg_parsers.parse_duration(args.length)
        outfile = args.output  # Already set by centralized auto-filename logic
        
        # Determine num_frames and fps from duration
        # Default to 16 frames for 2 seconds, 8 fps
        num_frames = int(dur * 8) if dur else 16
        fps = 8
        if dur and dur > 2:
            fps = 16 # Increase FPS for longer videos
            num_frames = int(dur * fps)
        
        # Check resources
        if not pkg_system.check_resources_and_warn(VIDEO_MODELS[args.video_model], w, h, dur, args.force, MODEL_REQUIREMENTS):
           sys.exit(0)

        success = pkg_generate_video(
            prompt=args.prompt, 
            output_path=outfile, 
            duration=dur,
            model_name=VIDEO_MODELS[args.video_model], 
            width=w, 
            height=h,
            audio_prompt=args.audio_prompt,
            image_input=args.input_image
        )
        if not success:
            sys.exit(1)

        if args.upscale or args.simple_upscale:
            base, ext = os.path.splitext(outfile)
            suffix = "simple_upscaled" if args.simple_upscale else "upscaled"
            upscale_out = args.upscaled_output_file or f"{base}_{suffix}_{args.upscale_factor}{ext}"
            
            if args.simple_upscale:
                pkg_simple_upscale_video(outfile, upscale_out, factor=args.upscale_factor, force=args.force)
            else:
                upscaler_model = args.upscaler_model_video or 'realesrgan'
                if upscaler_model == 'realesrgan':
                    pkg_upscale_video_fast(outfile, upscale_out, factor=args.upscale_factor)
                else:
                   # Standard diffusion based (slower)
                   print("⚠️  Warning: Full diffusion video upscaling is very slow. Consider using --simple-upscale or realsrgan.")
                   # There is no direct video file upscaler in upscaling.py yet besides fast, maybe add todo
                   pkg_upscale_video_fast(outfile, upscale_out, factor=args.upscale_factor)

    # --- Standalone Modes ---
    # Translate Mode
    elif args.translate:
        if not args.target_language:
            print("❌ Error: --target-language is required for translation.")
            sys.exit(1)
        
        load_ai_modules()
        
        input_data = args.prompt if args.prompt else args.input_image # repurpose input_image for file? or create input_file arg?
        # Argument parser has -ii for input-image.
        # But my new group didn't add --input-file.
        # I can use args.input_image (reusing -ii) as generic file input, or check if prompt is a file path?
        # Let's use args.input_image as the file input (since -ii is common param).
        
        if not args.prompt and not args.input_image:
            print("❌ Error: --prompt (text) or -ii (file) required.")
            sys.exit(1)
            
        data = args.prompt if args.prompt else args.input_image
        is_file = bool(args.input_image)

        task = "t2tt"
        if is_file:
             ext = os.path.splitext(data)[1].lower()
             if ext in ['.wav', '.mp3', '.m4a', '.flac']:
                 task = "s2st" if args.format in ['wav', 'mp3'] else "s2tt"
                 print(f"🎤 Translating Audio ({task}): {data} -> {args.target_language}")
             else:
                 # Assume text file?
                 # TranslationGenerator t2tt supports text string. If file, read it?
                 if os.path.exists(data):
                     try:
                        with open(data, 'r', encoding='utf-8') as f:
                            data = f.read()
                        print(f"📄 Translating Text File: {args.input_image} -> {args.target_language}")
                     except:
                        print("❌ Error: Could not read input file (binary?). Translation currently supports text/audio.")
                        sys.exit(1)
                 else:
                     print(f"❌ Error: File not found: {data}")
                     sys.exit(1)
        else:
             print(f"📝 Translating Text: \"{data[:50]}...\" -> {args.target_language}")

        try:
             model = args.translation_model
             
             # Use Seamless only for speech translation, otherwise use text translator
             if task in ["s2st", "s2tt"]:
                 # Speech translation - use Seamless (TranslationGenerator)
                 res = pkg_generate_translation(data, args.target_language, task=task, output_path=args.output)
             else:
                 # Text translation - use NLLB/LLM via translate_text
                 # Text translation - use NLLB/LLM via ArticleGenerator
                 from ai_media.generators.text import ArticleGenerator
                 # Instantiate generator (bypass warnings for cleaner CLI output if needed)
                 gen = ArticleGenerator(model_name=model, bypass_warning=args.bypass_warning, args=args)
                 res = gen.translate_text(
                     content=data,
                     target_lang=args.target_language,
                     source_lang=args.source_language or "auto",
                     model_id=model
                 )
             
             if res:
                  print(f"✅ Translation Complete: {res if args.output else 'Output to console'}")
                  if not args.output and task == "t2tt":
                      print(f"\n{res}\n")
        except Exception as e:
             print(f"❌ Translation Failed: {e}")
             sys.exit(1)

    # Subtitle Generation
    elif args.generate_subtitles:
        input_file = args.input_image
        if not input_file:
             # Try prompt as input file if it looks like a file? No, standard is -ii.
             print("❌ Error: Please provide input video/audio with -ii / --input-image")
             sys.exit(1)
            
        if not os.path.exists(input_file):
             print(f"❌ Error: Input file not found: {input_file}")
             sys.exit(1)

        # Resolve VAD preset to params
        vad_params = {}
        if args.subtitle_vad_preset == "noisy":
            vad_params = {
                "vad_min_silence_duration_ms": 500,
                "vad_threshold": 0.7,
                "condition_on_previous_text": False,
                "no_speech_threshold": 0.4,
            }
        elif args.subtitle_vad_preset == "sensitive":
            vad_params = {
                "vad_min_silence_duration_ms": 1000,
                "vad_threshold": 0.35,
                "condition_on_previous_text": True,
                "no_speech_threshold": 0.6,
            }
        
        # Override with explicit VAD params if provided
        if args.subtitle_vad_min_silence:
            vad_params["vad_min_silence_duration_ms"] = args.subtitle_vad_min_silence
        if args.subtitle_vad_threshold:
            vad_params["vad_threshold"] = args.subtitle_vad_threshold
        if args.subtitle_no_context:
            vad_params["condition_on_previous_text"] = False

        generator = PkgSubtitlesGenerator()
        
        # Handle legacy + new translate-to flags
        translate_to = args.subtitle_translate_to or getattr(args, 'subtitle_translate_to_legacy', None)
        targets = translate_to.split(",") if translate_to else None
        
        # Handle legacy + new source-lang flags  
        source_lang = args.subtitle_source_lang or getattr(args, 'subtitle_source_lang_legacy', None)
        
        generator.run(
            input_path=input_file,
            model_size=args.whisper_model,
            source_lang=source_lang,
            target_langs=targets,
            output_format=args.subtitle_format,
            fps=args.subtitle_fps,
            **vad_params
        )
        sys.exit(0)

    elif args.transcribe:
        input_file = args.input_image
        if not input_file:
             print("❌ Error: Please provide input file with -ii / --input-image")
             sys.exit(1)
             
        if not os.path.exists(input_file):
             print(f"❌ Error: Input file not found: {input_file}")
             sys.exit(1)
             
        # Use output format from text options if standard, or infer from extension
        fmt = "markdown"
        if args.output:
            if args.output.endswith(".json"):
                fmt = "json"
        elif args.output_format == "json":
            fmt = "json"
            
        generator = PkgTranscriptionGenerator()
        result = generator.run(input_file, output_format=fmt)
        
        # Output to file or stdout
        if args.output:
            outfile = args.output
            # Ensure path
            if not os.path.dirname(outfile):
                 outfile = os.path.join(CONFIG["paths"]["media_output"], outfile)
            pkg_system.ensure_paths(outfile)
            
            with open(outfile, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"✅ Transcription saved to {outfile}")
        else:
            print(result)
        sys.exit(0)

    elif args.generate_description:
        if args.generate_description != "USE_INPUT_IMAGE":
            input_file = args.generate_description
        else:
            input_file = args.input_image
             
        if not input_file:
             print("❌ Error: Input image/video required (-gd FILE or -gd -ii FILE)")
             sys.exit(1)
             
        print(f"👁️  Analyzing image: {input_file}...")
        device, _ = pkg_system.get_optimal_device_and_dtype(quiet=True, prefer_bfloat16=True)
        desc = pkg_generate_caption(input_file, device, model_type=args.caption_model)
        if desc:
            print(f"📝 Description: {desc}")
            # Save to output file if specified
            if args.output:
                outfile = args.output
                if not os.path.dirname(outfile):
                    outfile = os.path.join(CONFIG["paths"]["media_output"], outfile)
                
                pkg_system.ensure_paths(outfile)
                
                with open(outfile, 'w') as f:
                    f.write(desc)
                print(f"✅ Caption saved to {outfile}")
            
    elif args.upscale_image:
        outfile = args.upscaled_output_file or args.output
        if not outfile:
             base, ext = os.path.splitext(args.upscale_image)
             outfile = f"{base}_upscaled{ext}"
        
        # Enforce relative path to config dir
        if not os.path.dirname(outfile):
             outfile = os.path.join(CONFIG["paths"]["media_output"], outfile)
             
        if args.simple_upscale:
             pkg_simple_upscale_image(args.upscale_image, outfile, factor=args.upscale_factor, force=args.force)
        else:
                 # If we have an upscaler model selected (default is 'realesrgan')
                upscaler_model = args.image_upscaler or 'realesrgan'
                if upscaler_model == 'realesrgan':
                    pkg_upscale_image_fast(args.upscale_image, outfile, factor=args.upscale_factor)
                else:
                    pkg_upscale_image_file(args.upscale_image, outfile, strength=args.upscale_strength, factor=args.upscale_factor)

    elif args.upscale_video:
         outfile = args.output
         if not outfile:
             base, ext = os.path.splitext(args.upscale_video)
             outfile = f"{base}_upscaled{ext}"
             
         # Enforce relative path to config dir
         if not os.path.dirname(outfile):
             outfile = os.path.join(CONFIG["paths"]["media_output"], outfile)

         if args.simple_upscale:
             pkg_simple_upscale_video(args.upscale_video, outfile, factor=args.upscale_factor, force=args.force)
         else:
             # Respect chosen upscaler
             upscaler = args.video_upscaler or 'realesrgan'
             if upscaler == 'sd':
                 pkg_upscale_video_file(args.upscale_video, outfile, factor=args.upscale_factor)
             else:
                 pkg_upscale_video_fast(args.upscale_video, outfile, factor=args.upscale_factor)

    elif args.transform_image:
        output_file = args.output
        # Enforce relative path to config dir if not set by auto-logic (which already does it) or if explicit relative
        if not os.path.dirname(output_file):
             output_file = os.path.join(CONFIG["paths"]["media_output"], output_file)

        if args.remove_background:
             pkg_remove_background(args.transform_image, output_file, silhouette=args.silhouette)
        else:
             success = pkg_generate_edit(args.transform_image, args.prompt if args.prompt else args.transform_prompt, output_file, 
                           model_name=args.edit_model,
                           guidance_scale=args.image_guidance if args.image_guidance else 7.5,
                           image_guidance_scale=args.image_guidance if args.image_guidance else 1.5,
                           unsafe=args.unsafe)
             if not success:
                 sys.exit(1)

    elif args.convert_image:
        pkg_convert_image(args.convert_image, args.convert_image_to)

    elif args.convert_video:
        pkg_convert_video(args.convert_video, args.convert_video_to)

    elif args.convert_audio:
        pkg_convert_audio(args.convert_audio, args.convert_audio_to)
        
    elif args.convert_document:
        pkg_convert_document(args.convert_document, args.convert_document_to)
if __name__ == "__main__":
    main()
