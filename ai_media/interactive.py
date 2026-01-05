"""
Interactive menu system for AI-Media.

Provides a terminal-based interactive menu for all AI-Media features.
Supports arrow-key navigation, mouse clicks (on supported terminals), and pagination.
"""

import os
import sys
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()


from .utils.interaction import (
    emoji, clear_screen, show_header, get_key, prompt_menu, prompt_choice, 
    prompt_text, browse_files, prompt_file, check_overwrite, wait_for_back,
    CLEAR_LINE, CYAN, RESET, DIM, HIDE_CURSOR, SHOW_CURSOR, UP
)



# Jump point mappings: name -> (menu_action, submenu_value)
JUMP_POINTS = {
    'image': ('image', None),
    'image/sdxl': ('image', 'sdxl'),
    'image/sd15': ('image', 'sd-1.5'),
    'image/flux': ('image', 'flux'),
    'video': ('video', None),
    'video/zeroscope': ('video', 'zeroscope'),
    'video/cogvideox': ('video', 'cogvideox'),
    'video/svd': ('video', 'svd'),
    'audio': ('audio', None),
    'audio/musicgen': ('audio', 'musicgen-medium'),
    'audio/bark': ('audio', 'bark'),
    'caption': ('caption', None),
    'article': ('article', None),
    'article/offline': ('article', 'offline'),
    'article/online': ('article', 'online'),
    'code': ('code', None),
    'chat': ('chat', None),
    'research': ('article', 'online'),
    'transform': ('transform', None),
    'upscale': ('upscale', None),
    'convert': ('convert', None),
    'test': ('test', None),
    'web': ('web', None),
}


def run_self_command(cmd_string):
    """Run ai-media.py with the given command arguments. Returns exit code."""
    import subprocess
    import shlex
    
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ai-media.py'))
    
    print(f"\n🚀 Running: ai-media.py {cmd_string}\n")
    
    if os.name == 'nt':
        escaped = cmd_string.replace('\\', '\\\\')
        try:
            args = shlex.split(escaped, posix=True)
            args = [arg.replace('\\\\', '\\') for arg in args]
        except ValueError:
            args = cmd_string.split()
    else:
        args = shlex.split(cmd_string)
    
    p = subprocess.Popen([sys.executable, script_path] + args)
    
    try:
        return p.wait()
    except KeyboardInterrupt:
        # On Ctrl+C, the child also receives the signal.
        # Wait for it to shutdown gracefully.
        try:
            return p.wait()
        except KeyboardInterrupt:
            # Force kill if mashed
            p.kill()
            return -999

def run_shell_command(cmd_string, cwd=None):
    """Run a generic shell command. Returns exit code."""
    import subprocess
    import shlex
    
    print(f"\n🚀 Running: {cmd_string}\n")
    
    try:
        if os.name == 'nt':
            p = subprocess.Popen(cmd_string, shell=True, cwd=cwd)
        else:
            args = shlex.split(cmd_string)
            p = subprocess.Popen(args, cwd=cwd)
            
        try:
            return p.wait()
        except KeyboardInterrupt:
            # Wait for child to shutdown gracefully
            try:
                return p.wait()
            except KeyboardInterrupt:
                p.kill()
                return -999
                
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return 1
def run_interactive(jump_point=None):
    """Run interactive mode.
    
    Args:
        jump_point: Optional jump path (e.g., 'image/sdxl', 'audio/bark')
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
        'caption': ('caption', None),
        'article': ('article', None),
        'article/offline': ('article', 'offline'),
        'article/online': ('article', 'online'),
        'code': ('code', None),
        'chat': ('chat', None),
        'research': ('article', 'online'),  # Alias for deep research/online article
        'transform': ('transform', None),
        'transform/edit': ('transform', 'edit'),
        'transform/rembg': ('transform', 'rembg'),
        'transform/silhouette': ('transform', 'silhouette'),
        'upscale': ('upscale', None),
        'convert': ('convert', None),
        'test': ('test', None),
        'test/unit': ('test', 'unit'),
        'test/integration': ('test', 'integration'),
        'test/codec': ('test', 'codec'),
        'sysinfo': ('sysinfo', None),
        'web': ('web', None),
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
        '4': ('caption', None),
        '5': ('article', None),
        '5/1': ('article', 'offline'),
        '5/2': ('article', 'online'),
        '6': ('code', None),
        '7': ('chat', None),
        '8': ('transform', None),
        '8/1': ('transform', 'edit'),
        '8/2': ('transform', 'rembg'),
        '8/3': ('transform', 'silhouette'),
        '9': ('convert', None),
        '10': ('upscale', None),
        '11': ('test', None),
        '11/1': ('test', 'unit'),
        '11/2': ('test', 'integration'),
        '11/3': ('test', 'codec'),
        '12': ('sysinfo', None),
        '13': ('web', None),
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
        
        # Display Loading Indicator
        print("\n⏳ Loading system information...", end="", flush=True)
        
        import platform
        import psutil
        import torch
        
        import subprocess
        
        # OS Info
        os_name = platform.system()
        os_release = platform.release()
        os_ver = platform.version()
        if os_name == "Darwin":
            mac_ver = platform.mac_ver()[0]
            os_info = f"macOS {mac_ver} ({platform.machine()})"
        elif os_name == "Windows":
             # platform.platform() gives "Windows-10-...", platform.release() gives "10" or "11"
            os_info = f"{platform.system()} {platform.release()} (Build {platform.version()})"
        else:
            os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
        
        # CPU Info
        cpu_count = psutil.cpu_count(logical=True)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # CPU Model Name
        cpu_model = platform.processor()
        try:
            if os_name == "Windows":
                # Windows CPU Name (Registry)
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                cpu_model = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            elif os_name == "Darwin":
                # Mac CPU Name (sysctl)
                result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
                if result.returncode == 0:
                    cpu_model = result.stdout.strip()
            elif os_name == "Linux":
                # Linux CPU Name (/proc/cpuinfo)
                if os.path.exists("/proc/cpuinfo"):
                    with open("/proc/cpuinfo", "r") as f:
                        for line in f:
                            if "model name" in line:
                                cpu_model = line.split(":", 1)[1].strip()
                                break
        except:
            pass
        
        # RAM Info
        mem = psutil.virtual_memory()
        ram_total = f"{mem.total / (1024**3):.1f} GB"
        ram_used = f"{mem.used / (1024**3):.1f} GB"
        ram_avail = f"{mem.available / (1024**3):.1f} GB"
        ram_percent = f"{mem.percent}%"
        
        # GPU Info
        if torch.backends.mps.is_available():
            gpu_info = "MPS (Apple Silicon) ✅ Available"
            try:
                # [NEW] Mac MPS Stats
                mem_curr = torch.mps.current_allocated_memory() / (1024**3)
                mem_driver = torch.mps.driver_allocated_memory() / (1024**3)
                
                indent = " " * 13
                if mem_curr > 0.05 or mem_driver > 0.05:
                    gpu_info += f"\n{indent}Allocs: {mem_curr:.2f} GB Current | {mem_driver:.2f} GB Driver"
            except:
                pass
        elif torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Fetch Nvidia Driver Info
            driver_info = ""
            try:
                # Run nvidia-smi to get system-level driver info
                # Output example: | NVIDIA-SMI 531.14  Driver Version: 531.14  CUDA Version: 12.1 |
                res = subprocess.run("nvidia-smi", capture_output=True, text=True)
                if res.returncode == 0:
                    import re
                    driver_match = re.search(r"Driver Version:\s*([\d\.]+)", res.stdout)
                    cuda_match = re.search(r"CUDA Version:\s*([\d\.]+)", res.stdout)
                    
                    parts = []
                    if driver_match: parts.append(f"NVIDIA Driver {driver_match.group(1)}")
                    if cuda_match: parts.append(f"CUDA {cuda_match.group(1)}")
                    
                    if parts:
                        # Format on next line with indentation (13 spaces for "🎮 GPU:      ")
                        indent = " " * 13
                        driver_info = f"\n{indent}{', '.join(parts)}"
            
            except:
                pass
            
            # Fetch Real-time GPU Stats (VRAM & Load)
            gpu_stats = ""
            try:
                # Output: 2569, 24576, 4
                stat_cmd = "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
                stat_res = subprocess.run(stat_cmd, capture_output=True, text=True)
                if stat_res.returncode == 0:
                    parts = stat_res.stdout.strip().split(',')
                    if len(parts) >= 3:
                        used_mb = int(parts[0])
                        total_mb = int(parts[1])
                        load_pct = int(parts[2])
                        
                        avail_gb = (total_mb - used_mb) / 1024
                        total_gb = total_mb / 1024
                        used_pct = (used_mb / total_mb) * 100
                        
                        indent = " " * 13
                        gpu_stats = f"\n{indent}Memory: {avail_gb:.1f} GB Available / {total_gb:.1f} GB Total ({used_pct:.1f}% Used) | Load: {load_pct}%"
            except:
                pass
            
            gpu_info = f"CUDA ({gpu_name}, {vram:.1f} GB VRAM) ✅ Available{driver_info}{gpu_stats}"
        else:
            gpu_info = "CPU Only (No Acceleration Detected)"
            
        # Dtype Info (uses centralized detection)
        try:
            from .utils.system import is_bfloat16_supported
            if torch.cuda.is_available():
                dtype_info = "bfloat16" if is_bfloat16_supported() else "float16"
            elif torch.backends.mps.is_available():
                dtype_info = "float32"
            else:
                dtype_info = "float32"
        except:
            dtype_info = "float32"
            
        # Clear Loading Indicator (Overwrite line)
        print("\r" + " " * 50 + "\r", end="", flush=True)
            
        print(f"💻 OS:       {os_info}")
        print(f"🧠 CPU:      {cpu_model} | {cpu_count} Cores (Usage: {cpu_percent}%)")
        print(f"💾 RAM:      {ram_avail} Available / {ram_total} Total ({ram_percent} Used)")
        print(f"🎮 GPU:      {gpu_info}")
        print(f"⚡ DTYPE:    {dtype_info}")
        print()
        
        prompt_menu(None, [], allow_back=True)

    def main_menu():
        """Show main menu and return action."""
        clear_screen()
        show_header("AI-Media Interactive Mode")
        print("📋 What would you like to do?\n")
        
        options = [
            ("🖼️   Generate Image", "image"),
            ("🎬  Generate Video", "video"),
            ("🎵  Generate Audio", "audio"),
            ("📝  Vision", "caption"),
            ("📰  Generate Article", "article"),
            ("💻  Generate Code", "code"),
            ("💬  Chat", "chat"),
            ("✨  Transform/Edit Image", "transform"),
            ("🔄  Convert Media", "convert"),
            ("📄  Convert Document", "doc_convert"),
            ("📈  Upscale Media", "upscale"),
            ("🧪  Run Tests", "test"),
            ("ℹ️   System Information", "sysinfo"),
            ("🌐  Web Server Mode", "web"),
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
            print(f"{emoji('⏳', '')} Loading Models...", end="", flush=True)
            # Build model options with platform-specific notes
            try:
                import torch
                is_cuda = torch.cuda.is_available()
                is_mps = torch.backends.mps.is_available() and not is_cuda
            except ImportError:
                is_cuda = False
                is_mps = False
            is_mac = sys.platform == 'darwin'
            
            model_options = [
                ("SD 3.5 Turbo (Default, Fast 4 Steps, 🔒 Gated) ~19GB", "sd3.5-turbo"),
                ("SDXL Turbo (Fast, no login) ~8GB", "sdxl"),
                ("SD 1.5 (Lightweight) ~4GB", "sd-1.5"),
            ]
            
            # SD 3.5 models (gated, work on all platforms)
            model_options.extend([
                ("SD 3.5 Medium (High Quality, 🔒 Gated) ~10GB", "sd3.5-medium"),
                ("SD 3.5 Large (Best Quality, 🔒 Gated) ~19GB", "sd3.5-large"),
            ])
            
            # Qwen-Image models
            # Auto: Smart select (Full on Mac, 4-bit on CUDA)
            model_options.append(("Qwen 2.5 Image (Auto: Best Quality) ~20-40GB", "qwen-image-auto"))
            # Lightning: Fast 8-step
            model_options.append(("Qwen 2.5 Image (Lightning: Fast 8-step) ~40GB", "qwen-image-lightning"))
            
            # Flux base models with Mac-specific notes
            if is_mac:
                model_options.extend([
                    ("Flux Schnell (High Quality, Slow on Mac) ~12GB", "flux"),
                    ("Flux Dev (Professional, Very Slow on Mac) ~16GB", "flux-dev"),
                ])
            else:
                model_options.extend([
                    ("Flux Schnell (High Quality) ~12GB", "flux"),
                    ("Flux Dev (Professional) ~16GB", "flux-dev"),
                ])
            
            # FLUX.2 models with platform-specific notes
            if is_cuda:
                # CUDA: Show 4-bit quantized option
                model_options.append(("FLUX.2 (4-bit SOTA 2025, CUDA) ~18GB", "flux2"))
            elif is_mac:
                # Mac: Show full model with RAM warning
                model_options.append(("FLUX.2 Full (SOTA 2025, ⚠️ 128GB+ RAM!) ~65GB", "flux2-full"))
            # Linux without CUDA - don't show flux2 as it won't work
            
            # Clear loading indicator
            print("\r" + " " * 50 + "\r", end="", flush=True)
            model = prompt_choice("Model", model_options)
            if model is None:
                return
        
        # Prompt
        print()
        prompt = prompt_text("📝 Enter prompt")
        if prompt is None:
            return

        # Negative Prompt (Optional)
        print("   (Tip: List content to exclude, e.g. 'blur, text'. Do NOT use 'no' or 'without'.)")
        print("   (⚠️  Note: Negative prompts have NO effect on Lightning models - they ignore CFG. Leave empty.)")
        neg_prompt = prompt_text("🚫 Enter Negative Prompt (Optional)", required=False)
        
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
        if neg_prompt:
            cmd += f" --negative-prompt \"{neg_prompt}\""
        
        run_self_command(cmd)
        wait_for_back()
    
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
                ("Wan 2.2 (Alibaba, 14B, High Quality) ~24GB", "wan-2.2"),
                ("LTX-Video (Lightricks, Fast DiT) ~16GB", "ltx-video"),
                ("Mochi 1 (Genmo, Motion SOTA) ~19GB", "mochi-1"),
                ("HunyuanVideo (Tencent, Cinematic) ~24GB", "hunyuan"),
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

        # Resolution (for Zeroscope dynamic upscaling)
        size = None
        if model == 'zeroscope':
            print("\n📐 Select Output Resolution:\n")
            
            # Check if MPS (XL is skipped on Apple Silicon)
            import torch
            is_mps = torch.backends.mps.is_available() and not torch.cuda.is_available()
            
            if is_mps:
                print("   💡 Zeroscope uses Real-ESRGAN upscaling (XL skipped on Mac)")
                size_options = [
                    ("576x320 (Native, Fast)", "576x320"),
                    ("1024x576 (ESRGAN)", "1024x576"),
                    ("720p (ESRGAN)", "720p"),
                    ("1080p (ESRGAN)", "1080p"),
                    ("Custom Resolution", "custom"),
                ]
            else:
                print("   💡 Zeroscope uses smart upscaling for higher resolutions")
                size_options = [
                    ("576x320 (Native, Fast)", "576x320"),
                    ("1024x576 (XL Upscale)", "1024x576"),
                    ("720p (XL + ESRGAN)", "720p"),
                    ("1080p (XL + ESRGAN)", "1080p"),
                    ("Custom Resolution", "custom"),
                ]
            size = prompt_choice("Size", size_options)
            if size is None:
                return
            if size == "custom":
                print()
                size = prompt_text("Enter resolution (e.g. 1280x720)")
                if not size:
                    return

        # Input Image (Optional - for non-SVD models to enable Image-to-Video)
        if model != 'svd':
            print()
            add_image = input("📂 Add input image for Image-to-Video? [y/N]: ").strip().lower()
            if add_image in ['y', 'yes']:
                input_image = prompt_file("Input Image")

        # Audio Prompt (Optional - for Video-with-Audio, available for ALL models)
        print()
        audio_prompt = prompt_text("🎵 Audio prompt (for background music, Optional)", required=False)
        audio_model = None
        if audio_prompt:
             print("\n📦 Select Audio Model for Background:\n")
             audio_model_options = [
                ("MusicGen Medium (Default)", "musicgen-medium"),
                ("MusicGen Small (Fast)", "musicgen-small"),
                ("AudioLDM2 (Sound Effects)", "audioldm2"),
             ]
             audio_model = prompt_choice("Audio Model", audio_model_options)

        # Output
        print()
        output = prompt_text("💾 Output filename (or press Enter for auto)", required=False)
        
        # Build Command
        cmd = f"-v -l {length} --video-model {model}"
        if size:
            cmd += f" -s {size}"
        if prompt:
            cmd += f" -p \"{prompt}\""
        if input_image:
            cmd += f" -ii \"{input_image}\""
        if audio_prompt:
            cmd += f" -ap \"{audio_prompt}\""
            if audio_model:
                cmd += f" -am {audio_model}"
        if output:
             cmd += f" -o \"{output}\""
             
        run_self_command(cmd)
        wait_for_back()

    
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
        wait_for_back()
    
    def transform_menu(preset_operation=None):
        """Image transformation submenu."""
        clear_screen()
        show_header("Transform/Edit Image")
        
        # Input file
        print("📂 Select input image:\n")
        input_file = prompt_file("Input Image")
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
            
            # Model selection for edit
            print("\n📦 Select Edit Model:\n")
            try:
                import torch
                is_cuda = torch.cuda.is_available()
                is_mps = torch.backends.mps.is_available() and not is_cuda
            except ImportError:
                is_cuda = False
                is_mps = False
            
            edit_model_options = [
                ("InstructPix2Pix (Default, Fast) ~4GB", "instruct-pix2pix"),
            ]
            
            
            # Add Qwen-Image-Edit
            edit_model_options.append(("Qwen-Image-Edit (Base 2511, Precise) ~20GB", "qwen-image-edit"))
            edit_model_options.append(("Qwen-Edit-Lightning (Fast 2512) ~16GB", "qwen-image-edit-lightning"))
            
            edit_model = prompt_choice("Edit Model", edit_model_options)
            if edit_model is None:
                return
            
            cmd = f"-ti \"{input_file}\" -tp \"{instruction}\" --edit-model {edit_model}"
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
        wait_for_back()
    
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
            ("AI Upscale (High Quality)", "ai"),
            ("Simple Upscale (Fast, Lanczos)", "simple"),
        ]
        method = prompt_choice("Method", method_options)
        if method is None:
            return
        
        ai_model = "realesrgan"
        video_codec = "auto"
        
        if method == "ai":
            # Select Model
            print("\n🤖 Select AI Model:\n")
            model_options = [
                ("Real-ESRGAN (Fast, Faithful)", "realesrgan"),
                ("Stable Diffusion (Slow, Creative)", "sd"),
            ]
            ai_model = prompt_choice("Model", model_options)
            if ai_model is None:
                return

            if media_type == "video":
                # Select Codec for Video
                print("\n🎥 Select Video Codec:\n")
                codec_options = [
                    ("Auto (Default)", "auto"),
                    ("H.264", "h264"),
                    ("HEVC (H.265)", "hevc"),
                    ("AV1", "av1"),
                ]
                video_codec = prompt_choice("Codec", codec_options)
                if video_codec is None:
                    return

        # Build command
        if media_type == "image":
            cmd = f"-ui \"{input_file}\" -uf {factor}"
            if method == "ai":
                cmd += f" -iu {ai_model}"
            else:
                cmd += " -su"
        else:
            cmd = f"-uv \"{input_file}\" -uf {factor}"
            if method == "ai":
                cmd += f" -vu {ai_model}"
                if video_codec != "auto":
                    cmd += f" -vc {video_codec}"
            else:
                cmd += " -su"
        
        run_self_command(cmd)
        wait_for_back()
    
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
                ("GIF", "gif"),
                ("TIFF", "tiff"),
                ("BMP", "bmp"),
                ("TXT (OCR)", "txt"),
                ("MD (OCR)", "md"),
                ("PDF (OCR)", "pdf"),
                ("DOCX (OCR)", "docx"),
                ("HTML (OCR)", "html"),
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
        doc_formats = ['txt', 'md', 'pdf', 'docx', 'html']
        if media_type == "image" and target_format in doc_formats:
            # Image to document (OCR) - prompt for OCR model
            print("\n📦 Select OCR Model:\n")
            ocr_options = [
                ("Qwen-VL (High Precision, ~30GB RAM) [Default]", "qwen-vl"),
                ("Florence-2 (Fast, Lightweight)", "florence")
            ]
            ocr_model = prompt_choice("OCR Model", ocr_options)
            if ocr_model is None:
                return
            cmd = f"-cd \"{input_file}\" -cdt {target_format} -om {ocr_model}"
        elif media_type == "image":
            cmd = f"-ci \"{input_file}\" -cit {target_format}"
        elif media_type == "video":
            cmd = f"-cv \"{input_file}\" -cvt {target_format}"
        else:
            cmd = f"-ca \"{input_file}\" -cat {target_format}"
        
        run_self_command(cmd)
        wait_for_back()
    
    def document_convert_menu():
        """Convert document format submenu."""
        clear_screen()
        show_header("Convert Document")
        
        # Info Block
        console.print(Panel(
            "[bold cyan]💡 Pro Tip:[/bold cyan] You can also convert [bold yellow]Images[/bold yellow] or [bold yellow]Scanned PDFs[/bold yellow] "
            "to extract text using [bold green]High-Precision OCR[/bold green] (Qwen-VL).",
            border_style="blue",
            padding=(0, 2)
        ))
        print()
        
        # Input file
        print("📂 Select input document:\n")
        input_file = prompt_file("Enter file path")
        if input_file is None:
            return
        
        # Target format
        print("\n🎯 Select Target Format:\n")
        format_options = [
            ("PDF - Portable Document", "pdf"),
            ("DOCX - Microsoft Word", "docx"),
            ("HTML - Web Page", "html"),
            ("XHTML - Extensible HTML", "xhtml"),
            ("MD - Markdown", "md"),
            ("RTF - Rich Text Format", "rtf"),
            ("TXT - Plain Text", "txt"),
            ("JSON - Structured Data", "json"),
        ]
        target_format = prompt_choice("Format", format_options)
        if target_format is None:
            return
        
        # Build command
        cmd = f"-cd \"{input_file}\" -cdt {target_format}"
        
        # Determine if OCR model selection is needed
        input_ext = Path(input_file).suffix.lower().lstrip('.')
        image_exts = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff']
        
        # Scanned PDF or Image input triggers OCR model choice
        if input_ext in image_exts or (input_ext == 'pdf' and target_format in ['txt', 'md', 'docx']):
            print("\n📦 Select OCR Model:\n")
            ocr_options = [
                ("Qwen-VL (High Precision, ~30GB RAM) [Default]", "qwen-vl"),
                ("Florence-2 (Fast, Lightweight)", "florence")
            ]
            ocr_model = prompt_choice("OCR Model", ocr_options)
            if ocr_model is None:
                return
            cmd += f" -om {ocr_model}"
        
        run_self_command(cmd)
        wait_for_back()
    
    def caption_menu(preset_model=None):
        """Generate caption submenu."""
        clear_screen()
        show_header("Vision")
        
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
                ("Qwen3-VL 8B (Vision-Language)", "qwen-vl"),
                ("Qwen3-VL 4B (Balanced)", "qwen3-vl-4b"),
                ("Qwen3-VL 2B (Lightweight)", "qwen3-vl-2b"),
                ("BLIP", "blip"),
            ]
            model = prompt_choice("Model", model_options)
            if model is None:
                return
        
        # Build command
        cmd = f"-gd \"{input_file}\" -cm {model}"
        
        run_self_command(cmd)
        wait_for_back()
    
    def article_menu(preset_mode=None):
        """Generate article/research submenu."""
        while True:
            clear_screen()
            show_header("Generate Article")
            
            # Mode selection (Offline/Online) - skip if preset
            if preset_mode:
                mode = preset_mode
            else:
                print("📝 Select article mode:\n")
                mode_options = [
                    ("Offline (Model's internal knowledge)", "offline"),
                    ("Online Research (Live web search)", "online"),
                ]
                mode = prompt_choice("Mode", mode_options)
                if mode is None:
                    return
            
            online = mode == "online"
            mode_label = "Deep Research (Online)" if online else "Article (Offline)"
            
            clear_screen()
            show_header(mode_label)
            
            # Topic/Prompt
            print("✏️  Enter your topic or prompt:\n")
            topic = prompt_text("Topic")
            if not topic:
                return
            
            # Model selection
            print("\n📦 Select Model:\n")
            model_options = [
                ("Llama 3.1-8B (Default)", "llama-3.1-8b"),
                ("DeepSeek R1-Qwen-7B (~7GB)", "deepseek-r1-qwen-7b"),
                ("DeepSeek R1-Qwen-14B (~14GB)", "deepseek-r1-qwen-14b"),
                ("DeepSeek R1-Qwen-32B (⚠️ ~24GB RAM!)", "deepseek-r1-qwen-32b"),
                ("DeepSeek R1-Llama-8B (~8GB)", "deepseek-r1-llama-8b"),
                ("DeepSeek R1-Llama-70B (⚠️ ~40GB RAM!)", "deepseek-r1-llama-70b"),
                ("Qwen3-8B (Newer knowledge)", "qwen3-8b"),
                ("Qwen 2.5-14B (Larger)", "qwen-2.5-14b"),
                ("Qwen3-Coder-30B (MoE, 3.3B active)", "qwen3-coder-30b"),
                ("Qwen 2.5 Coder 32B (⚠️ 120GB+ RAM!)", "qwen-coder-32b"),
                ("Mistral Nemo-12B", "mistral-nemo-12b"),
            ]
            model = prompt_choice("Model", model_options)
            if model is None:
                return
            
            # Output format
            print("\n📄 Select output format:\n")
            format_options = [
                ("Markdown (.md)", "md"),
                ("PDF (.pdf)", "pdf"),
                ("Word Document (.docx)", "docx"),
                ("HTML (.html)", "html"),
                ("XHTML (.xhtml)", "xhtml"),
                ("Rich Text Format (.rtf)", "rtf"),
                ("Plain Text (.txt)", "txt"),
                ("JSON (.json)", "json"),
            ]
            output_format = prompt_choice("Format", format_options)
            if output_format is None:
                return
            
            # Research iterations (online only)
            research_iter = 3
            if online:
                print("\n🔄 Research iterations (sources to read):\n")
                iter_options = [
                    ("3 sources (Default)", "3"),
                    ("5 sources", "5"),
                    ("10 sources", "10"),
                    ("Custom", "custom"),
                ]
                iter_choice = prompt_choice("Iterations", iter_options)
                if iter_choice is None:
                    return
                if iter_choice == "custom":
                    custom_iter = prompt_text("Number of sources")
                    if custom_iter and custom_iter.isdigit():
                        research_iter = int(custom_iter)
                else:
                    research_iter = int(iter_choice)
            
            # Article length
            print("\n📏 Article Length:\n")
            length_options = [
                ("Quick (~500 words, fast) (Default)", "quick"),
                ("Standard (~1500 words)", "standard"),
                ("Detailed (~3000 words, comprehensive)", "detailed"),
            ]
            length = prompt_choice("Length", length_options)
            if length is None:
                return
            
            # Output file path (optional)
            print("\n📁 Output file path (leave empty for auto-name):\n")
            output_path = prompt_text("File path", required=False)
            
            # Build command
            flag = "-gr" if online else "-ga"
            cmd = f'{flag} -p "{topic}" -atm {model} --output-format {output_format} -al {length}'
            if online:
                cmd += f" -ri {research_iter}"
            if output_path:
                cmd += f' -o "{output_path}"'
            
            run_self_command(cmd)
            
            # Interactive result wait
            prompt_menu(None, [], allow_back=True)
    
    def code_menu(preset_model=None):
        """Generate code submenu."""
        while True:
            clear_screen()
            show_header("Generate Code")
            
            # Code description/prompt
            print("✏️  Describe what code you want to generate:")
            print("   (be more specific for better results)\n")
            print("💡 Tip: Include folder name for multi-file projects, e.g.:")
            print('   "Create React example in folder react-example"\n')
            print("(Leave empty to go back)\n")
            description = prompt_text("Description", required=False)
            if not description:
                return
            
            # Model selection
            print("\n📦 Select Code Model:\n")
            model_options = [
                ("Llama 3.1-8B (Default)", "llama-3.1-8b"),
                ("Qwen3-Coder-30B (MoE, 3.3B active)", "qwen3-coder-30b"),
                ("Qwen 2.5 Coder 32B (⚠️ 120GB+ RAM!)", "qwen-coder-32b"),
                ("Qwen 2.5 Coder 14B", "qwen-coder-14b"),
                ("Qwen 2.5 Coder 7B", "qwen-coder-7b"),
                ("DeepSeek R1-Qwen-7B (~7GB)", "deepseek-r1-qwen-7b"),
                ("DeepSeek R1-Qwen-14B (~14GB)", "deepseek-r1-qwen-14b"),
                ("DeepSeek R1-Qwen-32B (⚠️ ~24GB RAM!)", "deepseek-r1-qwen-32b"),
                ("DeepSeek R1-Llama-8B (~8GB)", "deepseek-r1-llama-8b"),
                ("DeepSeek R1-Llama-70B (⚠️ ~40GB RAM!)", "deepseek-r1-llama-70b"),
                ("Qwen3-8B (Newer knowledge)", "qwen3-8b"),
                ("Qwen 2.5-14B (Larger)", "qwen-2.5-14b"),
            ]
            model = prompt_choice("Model", model_options)
            if model is None:
                return
            
            # Output path (optional)
            print("\n📁 Output path (optional):")
            print("   (Leave empty: uses paths/filenames from your description)")
            print("   (Existing folder: saves all generated files inside it)")
            print("   (Filename: override output name if single file)\n")
            output_path = prompt_text("Output path", required=False)
            
            # Build command
            cmd = f'-gc -p "{description}" -cdm {model}'
            if output_path:
                cmd += f' -o "{output_path}"'
            
            run_self_command(cmd)
            wait_for_back()
    
    def chat_menu(preset_model=None):
        """Interactive chat submenu."""
        clear_screen()
        show_header("Chat")
        
        # Model selection (skip if preset)
        if preset_model:
            model = preset_model
            print(f"📦 Model: {model}\n")
        else:
            print("📦 Select Chat Model:\n")
            model_options = [
                ("Llama 3.1-8B (Default)", "llama-3.1-8b"),
                ("DeepSeek R1-Qwen-7B (~7GB)", "deepseek-r1-qwen-7b"),
                ("DeepSeek R1-Qwen-14B (~14GB)", "deepseek-r1-qwen-14b"),
                ("DeepSeek R1-Qwen-32B (⚠️ ~24GB RAM!)", "deepseek-r1-qwen-32b"),
                ("DeepSeek R1-Llama-8B (~8GB)", "deepseek-r1-llama-8b"),
                ("DeepSeek R1-Llama-70B (⚠️ ~40GB RAM!)", "deepseek-r1-llama-70b"),
                ("Qwen3-8B (Newer knowledge)", "qwen3-8b"),
                ("Qwen 2.5-14B (Larger)", "qwen-2.5-14b"),
                ("Qwen3-Coder-30B (MoE, 3.3B active)", "qwen3-coder-30b"),
                ("Qwen 2.5 Coder 32B (⚠️ 120GB+ RAM!)", "qwen-coder-32b"),
                ("Qwen3-VL 8B (Vision-Language)", "qwen-vl"),
                ("Mistral Nemo-12B", "mistral-nemo-12b"),
            ]
            model = prompt_choice("Model", model_options)
            if model is None:
                return
        
        # Build command and run
        cmd = f"-c --chat-model {model}"
        
        run_self_command(cmd)
        # No "Press Enter" needed - chat exits naturally
    
    def test_menu(preset_submenu=None):
        """Test selection submenu - choose between Unit Tests and Integration Tests."""
        # If preset submenu provided, go directly to that menu
        if preset_submenu == 'unit':
            unit_test_menu()
            return
        elif preset_submenu == 'integration':
            integration_test_menu()
            return
        elif preset_submenu == 'codec':
            # Run codec test directly then return
            script_dir = os.path.dirname(os.path.abspath(__file__))
            codec_test = os.path.join(script_dir, "testing", "codec_limits_tests.py")
            if os.path.exists(codec_test):
                clear_screen()
                show_header("Codec Limits Test")
                print("Running codec limits stress test...")
                print("This tests H.264, HEVC, and AV1 encoder limits up to 20K.\n")
                import subprocess
                subprocess.run([sys.executable, codec_test])
                
                # Forcefully reset terminal to sane state (raw mode from child process leaks)
                try:
                    os.system('stty sane')  # Reset terminal on Unix
                except:
                    pass
                
                wait_for_back("Press Enter to return to menu...")
            else:
                clear_screen()
                show_header("Codec Limits Test")
                print("❌ ai_media/testing/codec_limits_tests.py not found.")
                wait_for_back()
            # Fall through to test menu loop (don't return to main menu)
        
        while True:
            clear_screen()
            show_header("Run Tests")
            
            options = [
                ("🧪  Unit Tests (Python unittest)", "UNIT"),
                ("🚀  Integration Tests (ai_media/testing/integration-tests.json)", "INTEGRATION"),
                ("📊  Codec Limits Test (ai_media/testing/codec_limits_tests.py)", "CODEC"),
            ]
            
            choice = prompt_menu("Select test type:", options)
            
            if choice is None: return
            
            if choice == "UNIT":
                unit_test_menu()
            elif choice == "INTEGRATION":
                integration_test_menu()
            elif choice == "CODEC":
                # Run codec limits test directly
                script_dir = os.path.dirname(os.path.abspath(__file__))
                codec_test = os.path.join(script_dir, "testing", "codec_limits_tests.py")
                if os.path.exists(codec_test):
                    clear_screen()
                    show_header("Codec Limits Test")
                    print("Running codec limits stress test...")
                    print("This tests H.264, HEVC, and AV1 encoder limits up to 20K.\n")
                    import subprocess
                    subprocess.run([sys.executable, codec_test])
                    
                    # Forcefully reset terminal to sane state (raw mode from child process leaks)
                    try:
                        os.system('stty sane')
                    except:
                        pass
                    
                    wait_for_back("Press Enter to return to menu...")
                else:
                    clear_screen()
                    show_header("Codec Limits Test")
                    print("❌ ai_media/testing/codec_limits_tests.py not found.")
                    print(f"   Expected location: {codec_test}")
                    wait_for_back()
    

    def unit_test_menu():
        """Unit Tests submenu - dynamically loads test classes from ai_media/testing/unit_tests.py."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_module = os.path.join(script_dir, "testing", "unit_tests.py")
        
        # Check if test file exists
        if not os.path.exists(test_module):
            clear_screen()
            show_header("Unit Tests")
            print("❌ ai_media/testing/unit_tests.py not found.")
            print(f"   Expected location: {test_module}")
            wait_for_back()
            return
        
        # Extract test classes and count tests per class from tests/ai-media_test.py
        test_classes = {}  # {class_name: test_count}
        test_method_count = 0
        try:
            with open(test_module, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            import re
            current_class = None
            class_pattern = r'^class\s+(Test\w+)\s*\(\s*(?:unittest\.)?TestCase\s*\)\s*:'
            method_pattern = r'^\s+def\s+(test_\w+)\s*\('
            
            for line in lines:
                # Check for class definition
                class_match = re.match(class_pattern, line)
                if class_match:
                    current_class = class_match.group(1)
                    test_classes[current_class] = 0
                # Check for test method
                elif current_class and re.match(method_pattern, line):
                    test_classes[current_class] += 1
                    test_method_count += 1
        except Exception as e:
            clear_screen()
            show_header("Unit Tests")
            print(f"❌ Error parsing test file: {e}")
            wait_for_back()
            return
        
        if not test_classes:
            clear_screen()
            show_header("Unit Tests")
            print("❌ No test classes found in ai_media/testing/unit_tests.py")
            wait_for_back()
            return
        
        # Build options
        options = []
        class_count = len(test_classes)
        options.append((f"🚀  Run All ({class_count} classes, {test_method_count} tests) [Summary]", "ALL_QUIET"))
        options.append((f"📜  Run All ({class_count} classes, {test_method_count} tests) [Verbose]", "ALL_VERBOSE"))
        
        for cls in sorted(test_classes.keys()):
            # Get method count for this class
            try:
                # We can inspect the class directly since we imported the module
                cls_obj = getattr(module, cls)
                method_count = len([m for m in dir(cls_obj) if m.startswith('test_')])
                options.append((f"{cls} ({method_count} tests)", cls))
            except:
                 options.append((f"{cls}", cls))

        while True:
            clear_screen()
            show_header("Unit Tests")
            prompt_text = (
                 f"Select a test class to run:\n\n"
                 f"ℹ️  Individual tests are always run in VERBOSE mode\n\n"
                 f"ℹ️  {class_count} test classes ({test_method_count} tests) found in ai_media/testing/unit_tests.py\n"
            )
            choice = prompt_choice(prompt_text, options, allow_back=True)

            if choice is None: return

            if choice == "ALL_QUIET":
                # Run all unit tests (Quiet)
                print("\n🧪 Running all unit tests (Summary)...\n")
                print("=" * 60)
                os.system(f'"{sys.executable}" -m unittest ai_media.testing.unit_tests')
            elif choice == "ALL_VERBOSE":
                # Run all unit tests (Verbose)
                print("\n🧪 Running all unit tests (Verbose)...\n")
                print("=" * 60)
                os.system(f'"{sys.executable}" -m unittest ai_media.testing.unit_tests -v')
            else:
                # Run specific test class (Always Verbose)
                print(f"\n🧪 Running {choice}...\n")
                print("=" * 60)
                os.system(f'"{sys.executable}" -m unittest ai_media.testing.unit_tests.{choice} -v')

            prompt_menu("Press Enter to continue...", [], allow_back=True)

    def integration_test_menu():
        """Integration Tests submenu - tests from ai_media/testing/integration-tests.json."""
        # Load tests
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_file = os.path.join(script_dir, "testing", "integration-tests.json")
        try:
            with open(test_file, "r") as f:
                data = json.load(f)
            tests = data.get("tests", [])
        except Exception as e:
            clear_screen()
            show_header("App Run Tests")
            print(f"❌ Error loading tests: {e}")
            prompt_menu("Press Enter...", [], allow_back=True)
            return

        if not tests:
            clear_screen()
            show_header("App Run Tests")
            print("❌ No tests found.")
            wait_for_back("Press Enter...")
            return

        # Build options
        options = []
        max_desc_len = 40  # Truncate descriptions to this length
        for t in tests:
            name = t.get("name", "Unnamed Test")
            desc = t.get("description", "")
            if desc:
                if len(desc) > max_desc_len:
                    desc = desc[:max_desc_len-3] + "..."
                display = f"{name} ({desc})"
            else:
                display = name
            options.append((display, name))
        
        # Prepend 'Run All' options and custom pattern option
        count = len(tests)
        options.insert(0, (f"🔍  Run Tests by Pattern (e.g. Interactive*)", "PATTERN"))
        options.insert(0, (f"📜  Run All {count} Tests (Verbose)", "ALL_VERBOSE"))
        options.insert(0, (f"🚀  Run All {count} Tests (Summary)", "ALL_QUIET"))

        while True:
            clear_screen()
            show_header("App Run Tests")
            choice = prompt_menu("Select a test to run:\n\nℹ️  Individual tests are always run in VERBOSE mode\n", options)
            
            if choice is None: return
            
            if choice == "ALL_QUIET":
                # Run all tests (quiet/summary mode)
                run_self_command("--test")
            elif choice == "ALL_VERBOSE":
                # Run all tests (verbose mode)
                run_self_command("--test-verbose")
            elif choice == "PATTERN":
                # Custom glob pattern
                clear_screen()
                show_header("App Run Tests - Custom Pattern")
                print("Enter a glob pattern to match test names.\n")
                print("Supported patterns:")
                print("  • *     matches everything")
                print("  • ?     matches single character")
                print("  • [seq] matches characters in seq")
                print("\nExamples:")
                print("  • Interactive*    - all interactive tests")
                print("  • Image*          - all image tests")
                print("  • *SDXL*          - tests containing SDXL")
                print("  • Video - Jump ?  - Jump 1 through Jump 9\n")
                
                pattern = prompt_text("Pattern", required=True)
                if pattern:
                    run_self_command(f"--test-verbose \"{pattern}\"")
                    prompt_menu("Press Enter to continue...", [], allow_back=True)
                continue
            else:
                # Run specific test
                # Always use verbose for single test as requested
                run_self_command(f"--test-verbose \"{choice}\"")
                
            wait_for_back()


    def web_server_menu():
        """Web Server Mode submenu."""
        while True:
            clear_screen()
            show_header("Web Server Mode")
            
            options = [
                ("🚀  Start Server (No Client)", "SERVER_ONLY"),
                ("🌐  Start Client (Web)", "WEB_CLIENT"),
                ("🔥  Start Both Server and Web Client", "BOTH_WEB"),
                ("⚡  Start Both Server and Web + Electron Dev Client", "BOTH_FULL"),
                ("🛠️   Electron Build Options", "BUILD_OPTS"),
                ("📦  Versioning Scripts", "VERSION_OPTS"),
            ]
            
            choice = prompt_choice("Select an option:", options, allow_back=True, default_index=3)
            
            if choice is None: return
            
            web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
            
            code = 0
            if choice == "SERVER_ONLY":
                code = run_self_command("--serve-no-client")
            elif choice == "WEB_CLIENT":
                code = run_shell_command("npm run dev:client", cwd=web_dir)
            elif choice == "BOTH_WEB":
                code = run_self_command("--serve-web-only-client")
            elif choice == "BOTH_FULL":
                code = run_self_command("--serve")
            elif choice == "BUILD_OPTS":
                electron_build_menu()
            elif choice == "VERSION_OPTS":
                version_menu()
                
            if choice != "BUILD_OPTS" and choice != "VERSION_OPTS":
                # If code is non-zero (likely interrupted by Ctrl+C or error), 
                # return to menu immediately instead of waiting
                if code is not None and code != 0:
                    continue
                wait_for_back()

    def version_menu():
        """Versioning scripts submenu."""
        # Get path to web/package.json
        web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
        pkg_path = os.path.join(web_dir, "package.json")
        
        while True:
            # Read current version
            current_version = "Unknown"
            try:
                if os.path.exists(pkg_path):
                    with open(pkg_path, "r") as f:
                        data = json.load(f)
                        current_version = data.get("version", "Unknown")
            except:
                pass

            clear_screen()
            show_header("Versioning Scripts")
            
            # Info Block
            console.print(Panel(
                f"[bold cyan]ℹ️  Current Version:[/bold cyan] [bold green]{current_version}[/bold green]",
                border_style="blue",
                padding=(0, 2),
                width=60
            ))
            print()
            
            options = [
                ("🩹  Patch (1.0.X)", "version:patch"),
                ("🔹  Minor (1.X.0)", "version:minor"),
                ("🔸  Major (X.0.0)", "version:major"),
            ]
            
            choice = prompt_choice("Select update type:", options, allow_back=True)
            if choice is None: return
            
            # Run npm version command
            run_shell_command(f"npm run {choice}", cwd=web_dir)
            wait_for_back("Press Enter to continue...")

    def electron_build_menu():
        """Electron build options submenu."""
        clear_screen()
        show_header("Electron Build Options")
        
        options = [
            ("🍎  Mac: ARM64 (Apple Silicon)", "electron:build:mac:arm64"),
            ("🍎  Mac: x64 (Intel)", "electron:build:mac:x64"),
            ("🍎  Mac: Universal", "electron:build:mac:universal"),
            ("🍎  Mac: All", "electron:build:mac:all"),
            ("🪟  Windows: x64", "electron:build:win:x64"),
            ("🪟  Windows: ARM64", "electron:build:win:arm64"),
            ("🪟  Windows: All", "electron:build:win:all"),
            ("🐧  Linux: x64", "electron:build:linux:x64"),
            ("🐧  Linux: ARM64", "electron:build:linux:arm64"),
            ("🐧  Linux: All", "electron:build:linux:all"),
            ("🌎  Build All Platforms", "electron:build:all"),
        ]
        
        choice = prompt_choice("Select build target:", options, allow_back=True)
        if choice is None: return
        
        web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
        run_shell_command(f"npm run {choice}", cwd=web_dir)
        wait_for_back()


    
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
        elif action == "doc_convert":
            document_convert_menu()
        elif action == "caption":
            caption_menu(initial_model if first_run or initial_action == 'caption' else None)
            initial_model = None
        elif action == "article":
            article_menu(initial_model if first_run or initial_action == 'article' else None)
            initial_model = None
        elif action == "code":
            code_menu(initial_model if first_run or initial_action == 'code' else None)
            initial_model = None
        elif action == "chat":
            chat_menu(initial_model if first_run or initial_action == 'chat' else None)
            initial_model = None
        elif action == "test":
            test_menu(initial_model if first_run or initial_action == 'test' else None)
            initial_model = None
        elif action == "sysinfo":
            system_info_menu()
        elif action == "web":
            web_server_menu()

# --- Test Runner ---


def run_tests(verbose=False, test_filter=None, exit_on_finish=True):
    """Run test suite from ai_media/testing/integration-tests.json."""
    import shlex
    import subprocess
    
    # Use global test state for CTRL+C handling
    global _test_state
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(script_dir, "testing", "integration-tests.json")
    
    if not os.path.exists(test_file):
        print(f"{emoji('❌ ', 'Error: ')}Test file not found: {test_file}")
        if exit_on_finish: sys.exit(1)
        return False
    
    with open(test_file, "r") as f:
        data = json.load(f)
    
    tests = data.get("tests", [])
    if not tests:
        print(f"{emoji('❌ ', 'Error: ')}No tests found in ai_media/testing/integration-tests.json")
        if exit_on_finish: sys.exit(1)
        return False
    
    # Filter tests if requested
    if test_filter:
        if isinstance(test_filter, list):
            print(f"{emoji('🔎 ', '')}Test Filter Mode: Searching for {len(test_filter)} specific tests...")
            tests = [t for t in tests if t.get("name") in test_filter]
        else:
             print(f"{emoji('🔎 ', '')}Single Test Mode: Searching for '{test_filter}'...")
             tests = [t for t in tests if t.get("name") == test_filter]
        
        if not tests:
            print(f"{emoji('❌ ', 'Error: ')}No tests found matching filter: {test_filter}")
            print("   Available tests are listed in the ID list, or:")
            print(f"\n{emoji('👉 ', '')}Redirecting to Interactive Test Menu in ", end="", flush=True)
            for i in range(3, 0, -1):
                print(f"{i}...", end="", flush=True)
                time.sleep(1)
            print()
            # Call interactive mode jumping to 'test' menu
            # We need to access run_interactive. It is defined in the same global scope.
            run_interactive(jump_point="test")
            return False
        
        print(f"{emoji('✅ ', '')}Found {len(tests)} test(s) matching filter.\n")

    # Warning prompt
    print(f"\n{'='*60}")
    print(f"{emoji('⚠️  ', '   ')}WARNING: Test Suite")
    print(f"{'='*60}")
    print(f"   • Integration tests can take a long time")
    print(f"   • Models will be downloaded if not present (2-30GB each)")
    print(f"   • High system resource consumption")
    print(f"   • Press CTRL+C at any time to interrupt")
    print(f"{'='*60}")
    
    if os.environ.get("AI_MEDIA_FORCE") != "1":
        try:
            choice = input(f"\n   Continue? [Y/n]: ").lower().strip()
            if choice in ['n', 'no']:
                print("❌ Test cancelled.")
                if exit_on_finish: sys.exit(0)
                return False
        except KeyboardInterrupt:
            print("\n❌ Test cancelled.")
            if exit_on_finish: sys.exit(0)
            return False
    else:
        print(f"\n   (Skipping confirmation due to --force)\n")
    
    print(f"\n{'='*60}")
    print(f"🧪 Running {len(tests)} test(s)")
    print(f"{'='*60}\n")
    
    passed = 0
    failed = 0
    skipped = 0
    results = []
    ran_count = 0
    
    # Resource aggregation variables
    total_ram = 0.0
    total_vram = 0.0
    total_cpu = 0.0
    total_gpu = 0.0
    resource_count = 0
    
    # Set global test state for CTRL+C handler
    _test_state['active'] = True
    _test_state['total'] = len(tests)
    _test_state['passed'] = 0
    _test_state['failed'] = 0

    suite_start_time = time.time()
    suite_start_time = time.time()
    # timestamp with milliseconds
    start_dt = datetime.fromtimestamp(suite_start_time)
    suite_timestamp = start_dt.strftime("%Y%m%d-%H%M%S-%f")[:-3]
    
    for i, test in enumerate(tests):
        test_name = test.get("name", f"Test {i+1}")
        command = test.get("command", "")
        expected_inputs = test.get("expectedInputItems", [])
        expected_outputs = test.get("expectedOutputItems", [])
        
        # Check for skip flag
        if test.get("skip") is True:
            print(f"\n{emoji('⏭️  ', '(-)')}Skipping test: {test_name} (skip: true)")
            skipped += 1
            continue
        
        # Formatting
        description = test.get("description", "")
        header = f"{emoji('📋 ', '')}Test {i+1}/{len(tests)}: {test_name}"
        desc = f"   {description}" if description else ""
        start_t_str = datetime.now().strftime("%H:%M:%S")
        time_line = f"   Start at: {start_t_str}"
        
        # Calculate dynamic width (min 50)
        lines = [header, desc, time_line]
        max_len = max(50, *[len(l) for l in lines if l])
        sep = "-" * max_len
        
        print(f"\n{sep}")
        print(header)
        if description:
            print(desc)
        print("")
        print(time_line)
        print(f"{sep}")
        
        test_passed = True
        failure_reason = None
        
        # 1. Check expected input items exist
        for input_item in expected_inputs:
            input_path = os.path.join(script_dir, input_item)
            if not os.path.exists(input_path):
                print(f"{emoji('❌ ', '[X] ')}Missing input: {input_item}")
                test_passed = False
                failure_reason = f"Missing input: {input_item}"
                break
        
        if not test_passed:
            print(f"{emoji('⏭️  ', '')}Skipping due to missing inputs")
            failed += 1
            results.append((test_name, False, failure_reason))
            continue
        
        # 2. Delete expected outputs before run (clean slate)
        # 2. Delete expected outputs before run (clean slate)
        for output_item in expected_outputs:
            output_path = os.path.join(script_dir, output_item)
            if os.path.exists(output_path):
                os.remove(output_path)
                print(f"{emoji('🗑️  ', '(-) ')}Deleted: {output_item}")
        
        # Prepare JSON Report Path
        import tempfile
        # Use suite-wide timestamp for better file grouping
        json_report_path = os.path.join(script_dir, f"{suite_timestamp}-{i+1:03d}-temp-performance.json")
        
        # Add --report-json arg if command supports it (assuming all ai-media commands do)
        full_command = [sys.executable, "-u", os.path.join(script_dir, 'ai-media.py')] + shlex.split(command) + ["--report-json", json_report_path]
        
        # For logging, show command without the hidden report arg
        cmd_display = f"python ai-media.py {command}"
        print(f"{emoji('🚀 ', '(>) ')}Running: {cmd_display}")
            
            # ... rest of subprocess creation ...
        
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
            # On Windows, create subprocess in new process group so it doesn't receive console Ctrl+C
            # This allows parent to catch KeyboardInterrupt and kill subprocess properly
            creation_flags = 0
            if os.name == 'nt':
                creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
            
            current_process = subprocess.Popen(
                full_command,
                cwd=script_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Merge stderr into stdout for simple streaming
                stdin=subprocess.DEVNULL if is_interactive else None,  # Isolate interactive tests from parent tty
                text=True,
                encoding='utf-8',
                errors='replace',
                env=env,
                bufsize=1, # Line buffered
                universal_newlines=True,
                creationflags=creation_flags
            )
            
            # Track subprocess for signal handler cleanup
            _test_state['current_process'] = current_process
            
            stdout_lines = []
            
            try:
                if is_interactive:
                    if verbose: print(f"{emoji('⏳ ', 'Wait: ')}Waiting {interactive_wait}s for interactive output...")
                    time.sleep(interactive_wait)
                    
                    # Terminate interactive process (Windows-compatible)
                    if os.name == 'nt':
                        current_process.terminate()  # Windows: terminate gracefully
                    else:
                        current_process.send_signal(signal.SIGINT)  # Unix: send CTRL+C
                        
                    # For interactive tests, we don't need real-time streaming, just capture final output
                    # and ensure it doesn't hang.
                    timeout_limit = test.get("timeout", 600)
                    stdout, stderr = current_process.communicate(timeout=5) # Short timeout after term
                    stdout_lines = [stdout] if stdout else []
                    
                    # Note: stderr is merged to stdout in Popen, so stderr var will be None/Empty
                    
                else:
                    # Non-interactive (Batch) mode: Stream output real-time
                    timeout_limit = test.get("timeout", 600)
                    start_read_time = time.time()
                    
                    while True:
                        # Check for timeout
                        if time.time() - start_read_time > timeout_limit:
                            raise subprocess.TimeoutExpired(full_command, timeout_limit)
                        
                        # Read line
                        line = current_process.stdout.readline()
                        if not line and current_process.poll() is not None:
                            break
                        
                        if line:
                            # Stream to user if verbose mode is on (but not for interactive tests)
                            if verbose:
                                print(line, end='', flush=True) 
                            stdout_lines.append(line)
                            
                    current_process.wait() # Ensure clean exit

                elapsed = time.time() - start_time
                stdout = "".join(stdout_lines) 
                # stderr is merged into stdout via Popen(stderr=subprocess.STDOUT)
                
                # Show verbose output marking
                if verbose and not is_interactive:
                    print(f"\n--- END ---\n")
                
                if current_process.returncode != 0 and not is_interactive:
                    # Check if it was a Ctrl+C interrupt
                    # Unix: 130 (128 + SIGINT=2), Windows: -2 or 3221225786 (STATUS_CONTROL_C_EXIT)
                    is_ctrl_c = current_process.returncode in [130, -2, 3221225786, -1073741510]
                    
                    if is_ctrl_c:
                        print(f"\n\n⚠️  Interrupted! Cleaning up...")
                        # Re-raise to trigger the outer handler
                        raise KeyboardInterrupt()
                    else:
                        # Interactive tests expect SIGINT exit code (usually 130 or 1 or 0 handling)
                        # If caught cleanly it might be 0.
                        # We only fail non-interactive tests on non-zero exit code here unless specific check later.
                        print(f"{emoji('❌ ', 'Error: ')}Command failed with exit code {current_process.returncode}")
                        # Since stderr is merged, we can't print it separately, but it's already on screen.
                        test_passed = False
                        failure_reason = f"Exit code {current_process.returncode}"
                
                # Check STDOUT items
                if test_passed and expected_stdout_items:
                    for item in expected_stdout_items:
                        if item not in stdout:
                            print(f"{emoji('❌ ', 'Error: ')}Missing stdout item: '{item}'")
                            test_passed = False
                            failure_reason = f"Missing stdout: '{item}'"
                            break
                        else:
                            if verbose: print(f"✓ Found stdout item: '{item}'")
                            
            except subprocess.TimeoutExpired:
                current_process.kill()
                current_process.wait()
                elapsed = time.time() - start_time
                print(f"{emoji('❌ ', 'Error: ')}Command timed out after {timeout_limit} seconds")
                test_passed = False
                failure_reason = "Timeout"
                
        except KeyboardInterrupt:
            # Terminate subprocess and re-raise to be caught by outer handler
            print(f"\n\n⚠️  Interrupted! Cleaning up subprocess...")
            if current_process:
                pid = current_process.pid
                
                # On Windows, use taskkill to kill the entire process tree
                if os.name == 'nt':
                    try:
                        # /T = kill child processes, /F = force
                        subprocess.run(['taskkill', '/T', '/F', '/PID', str(pid)], 
                                       capture_output=True, timeout=5)
                        print(f"   ✅ Process tree (PID {pid}) terminated.")
                    except Exception as e:
                        print(f"   ⚠️  taskkill failed: {e}, trying fallback...")
                        current_process.kill()
                else:
                    # Unix: kill process group
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                    except:
                        current_process.kill()
                
                try:
                    current_process.wait(timeout=2)
                except:
                    pass
            raise
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"{emoji('❌ ', 'Error: ')}Command exception: {e}")
            test_passed = False
            failure_reason = str(e)
        
        # 4. Check expected output items exist
        if test_passed:
            # Parse Resource Usage from stdout -> DEPRECATED/REMOVED in favor of JSON IPC
            pass

            for output_item in expected_outputs:
                output_path = os.path.join(script_dir, output_item)
                if not os.path.exists(output_path):
                    print(f"{emoji('❌ ', 'Error: ')}Missing output: {output_item}")
                    test_passed = False
                    failure_reason = f"Missing output: {output_item}"
                    break
                else:
                    print(f"✓ Output exists: {output_item}")
        
        if test_passed:
            # 5. Read JSON Report if available
            if json_report_path and os.path.exists(json_report_path):
                try:
                    with open(json_report_path, 'r') as f:
                        stats = json.load(f)
                        
                    # Extract Data
                    r_time = stats.get("time", 0)
                    r_ram = stats.get("ram", 0)
                    r_vram = stats.get("vram", 0)
                    r_cpu = stats.get("cpu", 0)
                    r_gpu = stats.get("gpu", 0)
                    
                    total_ram += r_ram
                    total_vram += r_vram
                    total_cpu += r_cpu
                    total_gpu += r_gpu
                    resource_count += 1
                    
                    # Store exact generation time in results if available
                    results.append((test_name, True, f"{r_time:.1f}s"))
                        
                except Exception as e:
                    print(f"{emoji('⚠️ ', 'Warning: ')}Failed to read stats JSON: {e}")
                    results.append((test_name, True, f"{elapsed:.1f}s"))
            else:
                # Fallback to elapsed time if no JSON report
                results.append((test_name, True, f"{elapsed:.1f}s"))
        
        # Always cleanup report file if it exists (even if test failed)
        if json_report_path and os.path.exists(json_report_path):
            try:
                os.remove(json_report_path)
            except:
                pass

        if test_passed:
            print(f"{emoji('✅ ', '')}PASSED ({elapsed:.1f}s)")
            passed += 1
            _test_state['passed'] = passed
        else:
            print(f"{emoji('❌ ', '')}FAILED ({elapsed:.1f}s)")
            failed += 1
            _test_state['failed'] = failed
            results.append((test_name, False, failure_reason))
    
        ran_count += 1
    
    # Mark test as no longer active
    _test_state['active'] = False
    
    total_duration = time.time() - suite_start_time

    # Print summary
    print(f"\n{'='*60}")
    print(f"{emoji('📊 ', '')}TEST SUMMARY")
    print(f"{'='*60}")
    print(f"   Total:  {len(tests)}")
    print(f"   Passed: {passed} {emoji('✅', '')}")
    

    
    if failed > 0:
        print(f"   Failed: {failed} {emoji('❌', '')}")
    else:
        print(f"   Failed: {failed}")
        
    if skipped > 0:
        print(f"   Skipped: {skipped} {emoji('⏭️', '')}")
        
    print(f"   Duration: {format_time(total_duration)}")
    
    if resource_count > 0:
        avg_ram = total_ram / resource_count
        avg_vram = total_vram / resource_count
        avg_cpu = total_cpu / resource_count
        avg_gpu = total_gpu / resource_count
        
        if len(tests) >= 2:
             print(f"\n   {emoji('⚖️  ', '')}Averages:\n")
        
        print(f"   RAM: {avg_ram:.1f} GB")
        print(f"   VRAM: {avg_vram:.1f} GB")
        print(f"   CPU: {avg_cpu:.1f} %")
        print(f"   GPU: {avg_gpu:.1f} %")
        
    print(f"{'='*60}")
    
    if failed > 0:
        print(f"\n❌ Failed Tests:")
        for name, success, reason in results:
            if not success:
                print(f"   - {name}: {reason}")
    
    print(f"\n✅ Test Run Complete")
    
    sys.exit(0 if failed == 0 else 1)
