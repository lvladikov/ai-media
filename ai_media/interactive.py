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


def run_self_command(cmd):
    """Run ai-media.py with the given command arguments (string or list)."""
    import subprocess
    import shlex
    
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ai-media.py'))
    
    if isinstance(cmd, list):
        # The robust way: passing a list directly to Popen
        args = cmd
        
        # OS-specific safe string representation for logging
        if os.name == 'nt':
            import subprocess as sp
            cmd_log = sp.list2cmdline(args)
        else:
            # POSIX safe join
            import shlex
            cmd_log = shlex.join(args)
            
        print(f"\n🚀 Running: ai-media.py {cmd_log}\n")
    else:
        # Legacy/String way (deprecated, use lists for prompt safety)
        cmd_string = cmd
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
    
    # Passing the list directly is handled natively by the OS (CreateProcess on Win, exec on POSIX)
    p = subprocess.Popen([sys.executable, script_path] + args)
    
    try:
        return p.wait()
    except KeyboardInterrupt:
        try:
            return p.wait()
        except KeyboardInterrupt:
            p.kill()
            return -999


def run_shell_command(cmd, cwd=None):
    """Run a generic shell command. Returns exit code."""
    import subprocess
    import shlex
    
    p = None
    try:
        if isinstance(cmd, list):
            args = cmd
            if os.name == 'nt':
                import subprocess as sp
                cmd_log = sp.list2cmdline(args)
            else:
                cmd_log = shlex.join(args)
            print(f"\n🚀 Running: {cmd_log}\n")
            p = subprocess.Popen(args, cwd=cwd)
        else:
            cmd_string = cmd
            print(f"\n🚀 Running: {cmd_string}\n")
            if os.name == 'nt':
                p = subprocess.Popen(cmd_string, shell=True, cwd=cwd)
            else:
                args = shlex.split(cmd_string)
                p = subprocess.Popen(args, cwd=cwd)
        
        return p.wait()
    except KeyboardInterrupt:
        if p:
            try:
                return p.wait()
            except KeyboardInterrupt:
                p.kill()
                return -999
        return -999
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return 1


def run_interactive(jump_point=None, ml_framework=None, precision_force=None):
    """Run interactive mode.
    
    Args:
        jump_point: Optional jump path (e.g., 'image/sdxl', 'audio/bark')
        ml_framework: Optional framework force ('torch', 'mlx')
        precision_force: Optional precision force ('int4', 'float16', etc.)
    """
    
    # Jump point mappings: name -> (menu_action, submenu_value)
    JUMP_POINTS = {
        # By name
        'image': ('image', None),
        'image/z-image': ('image', 'z-image'),
        'image/sd35-turbo': ('image', 'sd3.5-turbo'),
        'image/sdxl': ('image', 'sdxl'),
        'image/sd15': ('image', 'sd-1.5'),
        'image/sd35-medium': ('image', 'sd3.5-medium'),
        'image/sd35-large': ('image', 'sd3.5-large'),
        'image/qwen': ('image', 'qwen-image-auto'),
        'image/qwen-lightning': ('image', 'qwen-image-lightning'),
        'image/flux': ('image', 'flux'),
        'image/flux-dev': ('image', 'flux-dev'),
        'image/flux2': ('image', 'flux2'),
        'image/flux2-full': ('image', 'flux2-full'),
        'video': ('video', None),
        'video/zeroscope': ('video', 'zeroscope'),
        'video/modelscope': ('video', 'ms-1.7b'),
        'video/cogvideox': ('video', 'cogvideox'),
        'video/svd': ('video', 'svd'),
        'audio': ('audio', None),
        'audio/music': ('audio', 'music'),
        'audio/tts': ('audio', 'tts'),
        'audio/musicgen': ('audio', 'musicgen-medium'),
        'audio/musicgen-small': ('audio', 'musicgen-small'),
        'audio/musicgen-large': ('audio', 'musicgen-large'),
        'audio/audioldm2': ('audio', 'audioldm2'),
        'audio/bark': ('audio', 'bark'),
        'audio/speecht5': ('audio', 'microsoft/speecht5_tts'),
        'analysis': ('analysis', None),
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
        'doc_convert': ('doc_convert', None),
        'translate': ('translate', None),
        'test': ('test', None),
        'test/unit': ('test', 'unit'),
        'test/integration': ('test', 'integration'),
        'test/codec': ('test', 'codec'),
        'sysinfo': ('sysinfo', None),
        'web': ('web', None),
        'cleanup': ('cleanup', None),
        # By number (matching menu order approx)
        '1': ('image', None),
        '1/1': ('image', 'z-image'),
        '1/2': ('image', 'sd3.5-turbo'),
        '1/3': ('image', 'sdxl'),
        '1/4': ('image', 'sd-1.5'),
        '1/5': ('image', 'sd3.5-medium'),
        '1/6': ('image', 'sd3.5-large'),
        '1/7': ('image', 'qwen-image-auto'),
        '1/8': ('image', 'qwen-image-lightning'),
        '2': ('video', None),
        '2/1': ('video', 'zeroscope'),
        '2/2': ('video', 'ms-1.7b'),
        '2/3': ('video', 'cogvideox'),
        '2/4': ('video', 'svd'),
        '3': ('audio', None),
        '3/1': ('audio', 'music'),
        '3/2': ('audio', 'tts'),
        '3/1/1': ('audio', 'musicgen-medium'),
        '3/1/2': ('audio', 'musicgen-small'),
        '3/1/3': ('audio', 'musicgen-large'),
        '3/1/4': ('audio', 'audioldm2'),
        '3/2/1': ('audio', 'bark'),
        '3/2/2': ('audio', 'microsoft/speecht5_tts'),
        '4': ('analysis', None),
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
        '10': ('doc_convert', None),
        '11': ('translate', None),
        '12': ('upscale', None),
        '13': ('test', None),
        '13/1': ('test', 'unit'),
        '13/2': ('test', 'integration'),
        '13/3': ('test', 'codec'),
        '14': ('sysinfo', None),
        '15': ('web', None),
        '16': ('cleanup', None),
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
            gpu_info = "Apple Silicon GPU ✅ Available"
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
            
        # Framework & Dtype Info (Centralized Detection)
        from .utils.system import get_optimal_device_and_dtype
        opt_device, opt_dtype = get_optimal_device_and_dtype(
            quiet=True, 
            framework_force=ml_framework,
            precision_force=precision_force
        )
        
        is_mlx = opt_device is None
        framework_info = "MLX (Native Apple Silicon)" if is_mlx else "PyTorch"
        if not is_mlx and torch.cuda.is_available():
             framework_info = "PyTorch (CUDA)"
        elif not is_mlx and torch.backends.mps.is_available():
             framework_info = "PyTorch (MPS)"
             
        if is_mlx:
            from .server.config import CONFIG
            dtype_info = CONFIG.get("precision_force") or "int4"
        else:
            dtype_info = str(opt_dtype).replace("torch.", "")
            
        # Clear Loading Indicator (Overwrite line)
        print("\r" + " " * 50 + "\r", end="", flush=True)
            
        print(f"💻 OS:       {os_info}")
        print(f"🧠 CPU:      {cpu_model} | {cpu_count} Cores (Usage: {cpu_percent}%)")
        print(f"💾 RAM:       {ram_avail} Available / {ram_total} Total ({ram_percent} Used)")
        print(f"🎮 GPU:       {gpu_info}")
        print(f"🏛️  FRAMEWORK: {framework_info}")
        print(f"⚡ DTYPE:     {dtype_info}")
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
            ("📝  Analysis", "analysis"),
            ("📰  Generate Article", "article"),
            ("💻  Generate Code", "code"),
            ("💬  Chat", "chat"),
            ("✨  Transform/Edit Image", "transform"),
            ("🔄  Convert Media", "convert"),
            ("📄  Convert Document", "doc_convert"),
            ("🌐  Translate", "translate"),
            ("📈  Upscale Media", "upscale"),
            ("🧪  Run Tests", "test"),
            ("ℹ️   System Information", "sysinfo"),
            ("🌐  Web Server Mode", "web"),
            ("🧹  Cleanup", "cleanup"),
            ("❌  Exit", None)
        ]
        
        return prompt_choice(None, options, allow_back=False)
    
    def image_menu(preset_model=None):
        """Image generation submenu."""
        clear_screen()
        show_header("Image Generation")
        
        # Model selection logic
        framework = "auto"
        precision = "auto"
        model = None

        if preset_model:
            model = preset_model
            print(f"📦 Model: {model}\n")
        else:
            # Framework selection FIRST
            framework = "auto" # default
            
            # Check Platform
            import platform
            try:
                import torch
            except ImportError:
                torch = None

            is_mac = platform.system() == "Darwin"
            is_cuda = False
            if torch and torch.cuda.is_available():
                is_cuda = True

            # Framework Prompt (Mac only)
            if is_mac:
                from ai_media.utils.precision import is_mlx_available
                mlx_ok = is_mlx_available()
                
                if mlx_ok:
                    if ml_framework:
                         framework = ml_framework
                         print(f"\n🏗️  Framework: {framework.upper()} (Pre-selected via CLI)")
                    else:
                        print("\n🏗️  Select Framework:\n")
                        fw_options = [
                            ("MLX (Native Apple Silicon) [Default]", "mlx"),
                            ("PyTorch (MPS/CPU)", "torch")
                        ]
                        # Note: We don't filter frameworks by precision here because precision is next
                        sel_fw = prompt_choice("Framework", fw_options)
                        if sel_fw is None: return
                        framework = sel_fw

            # Precision selection
            if precision_force:
                 precision = precision_force
                 print(f"\n⚙️  Precision: {precision} (Pre-selected via CLI)")
            else:
                print("\n⚙️  Select Precision:\n")
            
            # Determine auto label based on framework/platform
            auto_label = "Auto (Platform Default)"
            if framework == "mlx":
                auto_label = "Auto (int4 - MLX Default)"
            elif framework == "torch" and is_mac:
                auto_label = "Auto (bfloat16 - Default)"
            elif is_cuda:
                auto_label = "Auto (float16 - CUDA Default)"
            else:
                auto_label = "Auto (float32 - CPU Default)"

            # Get supported precisions for this framework
            from ai_media.utils.precision import get_supported_precisions
            device_type = "cpu"
            if is_cuda: device_type = "cuda"
            elif is_mac: device_type = "mps"
            
            supported_precs = get_supported_precisions(device_type, framework)
            
            # Build precision options
            all_options = [
                ("int4 (4-bit, Fastest)", "int4"),
                ("int6 (6-bit, Balanced Speed)", "int6"),
                ("int8 (8-bit, Balanced Quality)", "int8"),
                ("bfloat16 (Brain Float)", "bfloat16"),
                ("float16 (Half)", "float16"),
                ("float32 (Full)", "float32"),
            ]
            
            precision_options = [(auto_label, "auto")]
            for label, val in all_options:
                if val in supported_precs:
                    precision_options.append((label, val))
            
            precision = prompt_choice("Precision", precision_options)
            if precision is None:
                return
            
            # Model selection with dynamic RAM
            print("\n📦 Select Model:\n")
            # print(f"{emoji('⏳', '')} Loading Models...", end="", flush=True) # Usually fast enough now
            
            from ai_media.utils.model_resources import get_image_model_options
            from ai_media.utils.system import get_system_resources
            sys_resources = get_system_resources()
            sys_ram = sys_resources.get("ram_total", 0)
            
            model_options = get_image_model_options(precision, system_ram_gb=sys_ram, is_mac=is_mac, is_cuda=is_cuda)
            
            model = prompt_choice("Model", model_options)
            if model is None:
                return
        
        # Prompt
        print()
        from ai_media.utils.prompts import RANDOM_PROMPT_TRIGGERS
        triggers_str = ", ".join([f"'{t}'" for t in RANDOM_PROMPT_TRIGGERS])
        print(f"🎲 Tip: Enter {triggers_str} for a surprise Image prompt!\n")
        print(f"   (Leave empty for random prompt)")
        
        prompt = prompt_text("📝 Enter prompt", required=False)
        if prompt is None: # Cancelled with Back/Exit command
             # Logic is tricky here. prompt_text returns None on "0" or "back".
             # If prompt is "", it returns "" if required=False.
             # We need to distinguish between empty string (random) and None (back).
             # prompt_text generally handles '0' internal checks but returns None on actual back.
             # Let's assume None = abort, "" = empty input.
             return

        # Handle random prompt trigger (keyword or empty)
        from ai_media.utils.prompts import maybe_replace_with_random, get_random_prompt
        was_random = False
        
        if not prompt:
            # Empty input -> Force random
            prompt = get_random_prompt("image")
            was_random = True
        else:
            prompt, was_random = maybe_replace_with_random(prompt, "image")

        if was_random:
            print(f"🎲 Using random prompt: {prompt}")

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
        
        # Output Format
        print("\n📄 Select Output Format:\n")
        format_options = [
            ("JPG (Smaller, Lossy)", "jpg"),
            ("PNG (Lossless)", "png"),
            ("WebP (Modern, Smaller)", "webp"),
            ("TIFF (Lossless, Large)", "tiff"),
            ("BMP (Uncompressed)", "bmp"),
            ("GIF (Limited Colors)", "gif"),
        ]
        fmt = prompt_choice("Format", format_options)
        if fmt is None:
            return
        
        # Output
        print()
        output = prompt_text("💾 Output filename (or press Enter for auto)", required=False)
        
        # Build command list
        cmd = ['-i', '-p', prompt]
        if model:
             cmd += ['--image-model', model]
        if neg_prompt:
             cmd += ['--negative-prompt', neg_prompt]
        if size:
             cmd += ['-s', size]
        if orientation and orientation != "landscape":
             cmd += ['-otn', orientation]
        if fmt:
             cmd += ['-f', fmt]
        if output:
             cmd += ['-o', output]
        
        # Add Framework/Precision params
        if precision and precision != "auto":
             cmd += ['-pf', precision]
        if framework and framework != "auto":
             cmd += ['--ml-framework', framework]
             
        # Run
        run_self_command(cmd)
        wait_for_back()

    
    def video_menu(preset_model=None):
        """Video generation submenu."""
        clear_screen()
        show_header("Video Generation")
        
        
        # Framework selection FIRST
        import platform
        is_mac = platform.system() == "Darwin"
        framework = "auto"
        precision = "auto"
        
        # Check Platform & CLI overrides
        is_cuda = False
        try:
            import torch
            if torch.cuda.is_available(): is_cuda = True
        except: pass

        # Framework Prompt (Mac only)
        if not preset_model and is_mac:
            from ai_media.utils.precision import is_mlx_available
            if is_mlx_available():
                if ml_framework:
                     framework = ml_framework
                     print(f"🏗️  Framework: {framework.upper()} (Pre-selected via CLI)")
                else:
                    print("🏗️  Select Framework:\n")
                    fw_options = [
                        ("MLX (Native Apple Silicon) [Default]", "mlx"),
                        ("PyTorch (MPS/CPU)", "torch")
                    ]
                    sel_fw = prompt_choice("Framework", fw_options)
                    if sel_fw is None: return
                    framework = sel_fw

        # Precision selection
        if not preset_model:
            if precision_force:
                 precision = precision_force
                 print(f"\n⚙️  Precision: {precision} (Pre-selected via CLI)")
            else:
                print("\n⚙️  Select Precision:\n")
            
                from ai_media.utils.precision import get_supported_precisions
                device_type = "cpu"
                if is_cuda: device_type = "cuda"
                elif is_mac: device_type = "mps"
                
                supported_precs = get_supported_precisions(device_type, framework)
                
                # Determine auto label
                auto_label = "Auto (Platform Default)"
                if framework == "mlx": auto_label = "Auto (int4/float16)"
                elif is_cuda: auto_label = "Auto (float16)"
                elif is_mac: auto_label = "Auto (float16)"
                
                all_options = [
                    ("int4 (4-bit, Fastest)", "int4"),
                    ("int6 (6-bit, Balanced Speed)", "int6"),
                    ("int8 (8-bit, Balanced Quality)", "int8"),
                    ("bfloat16 (Brain Float)", "bfloat16"),
                    ("float16 (Half)", "float16"),
                    ("float32 (Full)", "float32"),
                ]
                
                precision_options = [(auto_label, "auto")]
                for label, val in all_options:
                    if val in supported_precs:
                        precision_options.append((label, val))
                
                precision = prompt_choice("Precision", precision_options)
                if precision is None: return

        # Model selection (skip if preset)
        if preset_model:
            model = preset_model
            print(f"📦 Model: {model}\n")
        else:
            # Model selection with dynamic RAM
            print("📦 Select Model:\n")

            from ai_media.utils.model_resources import get_video_model_options
            from ai_media.utils.system import get_system_resources
            sys_resources = get_system_resources()
            sys_ram = sys_resources.get("ram_total", 0)

            model_options = get_video_model_options(precision, system_ram_gb=sys_ram, is_mac=is_mac, is_cuda=is_cuda)

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
            print()
            from ai_media.utils.prompts import RANDOM_PROMPT_TRIGGERS
            triggers_str = ", ".join([f"'{t}'" for t in RANDOM_PROMPT_TRIGGERS])
            print(f"🎲 Tip: Enter {triggers_str} for a surprise Video prompt!\n")
            print(f"   (Leave empty for random prompt)")

            prompt = prompt_text("📝 Enter prompt", required=False)
            if prompt is None:
                return

            # Handle random prompt trigger (keyword or empty)
            from ai_media.utils.prompts import maybe_replace_with_random, get_random_prompt
            was_random = False
            
            if not prompt:
                # Empty input -> Force random
                prompt = get_random_prompt("video")
                was_random = True
            else:
                prompt, was_random = maybe_replace_with_random(prompt, "video")

            if was_random:
                print(f"🎲 Using random prompt: {prompt}")
        
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

        # Output Format
        print("\n📄 Select Output Format:\n")
        format_options = [
            ("MP4 (H.264, Universal)", "mp4"),
            ("WebM (VP9, Web)", "webm"),
            ("MOV (QuickTime)", "mov"),
            ("MKV (Matroska)", "mkv"),
            ("AVI (Legacy)", "avi"),
            ("FLV (Flash)", "flv"),
            ("TS (MPEG-TS)", "ts"),
            ("GIF (Animated, No Audio)", "gif"),
        ]
        fmt = prompt_choice("Format", format_options)
        if fmt is None:
            return

        # Build Command List
        cmd = ['-v', '-l', str(length), '--video-model', model]
        if size:
            cmd += ["-s", size]
        if prompt:
            cmd += ["-p", prompt]
        if input_image:
            cmd += ["-ii", input_image]
        if audio_prompt:
            cmd += ["-ap", audio_prompt]
            if audio_model:
                cmd += ["-am", audio_model]
        if fmt:
             cmd += ["-f", fmt]
        if output:
             cmd += ["-o", output]
        
        # Add Framework/Precision params
        if precision and precision != "auto":
             cmd += f' -pf {precision}'
        if framework and framework != "auto":
             cmd += f' --ml-framework {framework}'
             
        run_self_command(cmd)
        wait_for_back()

    
    def audio_menu(preset_category=None, preset_model=None):
        """Audio generation submenu (Music & TTS)."""
        
        while True:
            # If preset_model is provided, determine category and jump straight in
            if preset_model:
                # Simple heuristic mapping
                if any(x in preset_model for x in ["musicgen", "audioldm", "stable"]):
                    category = "music"
                else:
                    category = "tts"
            elif preset_category:
                category = preset_category
            else:
                clear_screen()
                show_header("Audio Generation")
                print("🎵 Select Category:\n")
                cat_options = [
                    ("Audio / Music (MusicGen, AudioLDM)", "music"),
                    ("Text-to-Speech (Bark, SpeechT5)", "tts"),
                ]
                category = prompt_choice("Category", cat_options)
                if category is None:
                    return

            # --- Audio / Music Submenu ---
            if category == "music":
                clear_screen()
                show_header("Audio / Music Generator")
                
                # Model Selection
                if preset_model:
                    model = preset_model
                    print(f"📦 Model: {model}\n")
                else:
                    print("📦 Select Music/Audio Model:\n")
                    model_options = [
                        ("MusicGen Medium (Default)", "musicgen-medium"),
                        ("MusicGen Small (Fast)", "musicgen-small"),
                        ("MusicGen Large (High Quality)", "musicgen-large"),
                        ("AudioLDM2 (Sound Effects)", "audioldm2"),
                    ]
                    model = prompt_choice("Model", model_options)
                    if model is None: 
                        if preset_category: return # Go back if jumped here
                        continue # Go back to category select commonly

                # Prompt
                print()
                from ai_media.utils.prompts import RANDOM_PROMPT_TRIGGERS
                triggers_str = ", ".join([f"'{t}'" for t in RANDOM_PROMPT_TRIGGERS])
                print(f"🎲 Tip: Enter {triggers_str} for a surprise Music prompt!\n")
                
                prompt = prompt_text("📝 Enter audio description (e.g. 'Lo-fi hip hop beat')", required=False)
                if prompt is None: return

                # Handle random prompt
                from ai_media.utils.prompts import maybe_replace_with_random, get_random_prompt
                
                was_random = False
                if not prompt:
                    prompt = get_random_prompt("audio")
                    was_random = True
                else:
                    prompt, was_random = maybe_replace_with_random(prompt, "audio")
                
                if was_random:
                    print(f"🎲 Using random prompt: {prompt}")

                # Duration
                print("\n⏱️ Select Duration:\n")
                length_options = [
                    ("5 seconds", "5s"),
                    ("10 seconds (Default)", "10s"),
                    ("30 seconds", "30s"),
                    ("Custom Duration", "custom"),
                ]
                length = prompt_choice("Duration", length_options)
                if length is None: return
                
                if length == "custom":
                    print()
                    length = prompt_text("Enter duration (e.g. 8s, 1m)")
                    if not length: return

                # Sampling Rate
                print("\n🔊 Select Sampling Rate:\n")
                sampling_options = [
                    ("16000 Hz (Standard TTS)", "16000"),
                    ("24000 Hz (Bark Standard)", "24000"),
                    ("32000 Hz (Default)", "32000"),
                    ("44100 Hz (High Quality)", "44100"),
                    ("48000 Hz (Professional)", "48000"),
                ]
                sampling = prompt_choice("Sampling", sampling_options, default_index=2)
                if sampling is None: return
                
                # Output Format
                print("\n📄 Select Output Format:\n")
                format_options = [
                    ("MP3 (Compressed, Universal)", "mp3"),
                    ("WAV (Lossless)", "wav"),
                    ("FLAC (Lossless, Compressed)", "flac"),
                    ("OGG (Open, Lossy)", "ogg"),
                    ("M4A/AAC (Apple, Lossy)", "m4a"),
                    ("OPUS (Modern, Efficient)", "opus"),
                    ("WMA (Windows)", "wma"),
                    ("AIFF (Apple Lossless)", "aiff"),
                ]
                fmt = prompt_choice("Format", format_options)
                if fmt is None: return
                
                # Output
                print()
                output = prompt_text("💾 Output filename (or press Enter for auto)", required=False)
                
                # Build Command
                cmd = ['-a', '-p', prompt, '-l', str(length), '--audio-model', model]
                if sampling: cmd += ["-m", sampling]
                if fmt: cmd += ["-f", fmt]
                if output: cmd += ["-o", output]
                
                run_self_command(cmd)
                wait_for_back()
                if preset_model or preset_category: return

            # --- Text-to-Speech Submenu ---
            elif category == "tts":
                clear_screen()
                show_header("Text-to-Speech (TTS)")
                
                # Model Selection
                if preset_model:
                    model = preset_model
                    print(f"📦 Model: {model}\n")
                else:
                    print("📦 Select TTS Model:\n")
                    model_options = [
                        ("Bark (Expressive, Multi-speaker)", "bark"),
                        ("SpeechT5 (Fast, Efficient)", "microsoft/speecht5_tts"),
                    ]
                    model = prompt_choice("Model", model_options)
                    if model is None:
                        if preset_category: return
                        continue

                # Prompt (Text content)
                print()
                from ai_media.utils.prompts import RANDOM_PROMPT_TRIGGERS
                triggers_str = ", ".join([f"'{t}'" for t in RANDOM_PROMPT_TRIGGERS])
                print(f"🎲 Tip: Enter {triggers_str} for a surprise Speech prompt!\n")
                
                prompt = prompt_text("📝 Enter text to speak", required=False)
                if prompt is None: return
                
                # Handle random/empty prompt
                from ai_media.utils.prompts import maybe_replace_with_random, get_random_prompt
                is_bark = model == "bark"
                
                was_random = False
                if not prompt:
                    prompt = get_random_prompt("tts", strip_tokens=not is_bark)
                    was_random = True
                else:
                    prompt, was_random = maybe_replace_with_random(prompt, "tts", strip_tokens=not is_bark)
                
                if was_random:
                    print(f"🎲 Using random conversation: {prompt}")

                # Output Format
                print("\n📄 Select Output Format:\n")
                format_options = [
                    ("MP3 (Compressed, Universal)", "mp3"),
                    ("WAV (Lossless)", "wav"),
                    ("FLAC (Lossless, Compressed)", "flac"),
                    ("OGG (Open, Lossy)", "ogg"),
                    ("M4A/AAC (Apple, Lossy)", "m4a"),
                    ("OPUS (Modern, Efficient)", "opus"),
                    ("WMA (Windows)", "wma"),
                    ("AIFF (Apple Lossless)", "aiff"),
                ]
                fmt = prompt_choice("Format", format_options)
                if fmt is None: return

                # Sampling Rate
                print("\n🔊 Select Sampling Rate:\n")
                sampling_options = [
                    ("16000 Hz (Standard TTS)", "16000"),
                    ("24000 Hz (Bark Standard)", "24000"),
                    ("32000 Hz (Default)", "32000"),
                    ("44100 Hz (High Quality)", "44100"),
                    ("48000 Hz (Professional)", "48000"),
                ]
                sampling = prompt_choice("Sampling", sampling_options, default_index=2)
                if sampling is None: return

                # Output
                print()
                output = prompt_text("💾 Output filename (or press Enter for auto)", required=False)
                
                # Build Command List
                cmd = ['-a', '-p', prompt, '--audio-model', model]
                if sampling: cmd += ["-m", sampling]
                if fmt: cmd += ["-f", fmt]
                if output: cmd += ["-o", output]
                
                run_self_command(cmd)
                wait_for_back()
                if preset_model or preset_category: return
    
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
            # Handle random prompt on empty input
            instruction = prompt_text("📝 Enter edit instruction (e.g., 'Make it anime')", required=False)
            
            from ai_media.utils.prompts import maybe_replace_with_random, get_random_prompt
            was_random = False
            
            if not instruction:
                # Empty input -> Force random
                instruction = get_random_prompt("image") # Use image prompts for edits too for now
                was_random = True
            else:
                instruction, was_random = maybe_replace_with_random(instruction, "image")
                
            if was_random:
                print(f"🎲 Using random instruction: {instruction}")
            
            if not instruction: # Should technically not happen if get_random_prompt works
                 return
            
            # Framework/Precision selection for Edit
            framework = "auto"
            precision = "auto"
            
            # Check Platform
            import platform
            try:
                import torch
            except ImportError:
                torch = None

            is_mac = platform.system() == "Darwin"
            is_cuda = False
            if torch and torch.cuda.is_available():
                is_cuda = True

            # Framework Prompt (Mac only)
            if is_mac:
                from ai_media.utils.precision import is_mlx_available
                mlx_ok = is_mlx_available()
                
                if mlx_ok:
                    if ml_framework:
                         framework = ml_framework
                         print(f"\n🏗️  Framework: {framework.upper()} (Pre-selected via CLI)")
                    else:
                        print("\n🏗️  Select Framework:\n")
                        fw_options = [
                            ("MLX (Native Apple Silicon) [Default]", "mlx"),
                            ("PyTorch (MPS/CPU)", "torch")
                        ]
                        sel_fw = prompt_choice("Framework", fw_options)
                        if sel_fw is None: return
                        framework = sel_fw

            # Precision selection
            if precision_force:
                 precision = precision_force
                 print(f"\n⚙️  Precision: {precision} (Pre-selected via CLI)")
            else:
                print("\n⚙️  Select Precision:\n")
            
            # Determine auto label based on framework/platform
            auto_label = "Auto (Platform Default)"
            if framework == "mlx":
                auto_label = "Auto (int4 - MLX Default)"
            elif framework == "torch" and is_mac:
                auto_label = "Auto (bfloat16 - Default)"
            elif is_cuda:
                auto_label = "Auto (float16 - CUDA Default)"
            else:
                auto_label = "Auto (float32 - CPU Default)"

            # Get supported precisions for this framework
            from ai_media.utils.precision import get_supported_precisions
            device_type = "cpu"
            if is_cuda: device_type = "cuda"
            elif is_mac: device_type = "mps"
            
            supported_precs = get_supported_precisions(device_type, framework)
            
            # Build precision options
            all_options = [
                ("int4 (4-bit, Fastest)", "int4"),
                ("int6 (6-bit, Balanced Speed)", "int6"),
                ("int8 (8-bit, Balanced Quality)", "int8"),
                ("bfloat16 (Brain Float)", "bfloat16"),
                ("float16 (Half)", "float16"),
                ("float32 (Full)", "float32"),
            ]
            
            precision_options = [(auto_label, "auto")]
            for label, val in all_options:
                if val in supported_precs:
                    precision_options.append((label, val))
            
            precision = prompt_choice("Precision", precision_options)
            if precision is None:
                return
            
            # Model selection for edit with dynamic RAM
            print("\n📦 Select Edit Model:\n")
            
            from ai_media.utils.model_resources import get_transform_model_options
            from ai_media.utils.system import get_system_resources
            sys_resources = get_system_resources()
            sys_ram = sys_resources.get("ram_total", 0)
            
            edit_model_options = get_transform_model_options(precision, system_ram_gb=sys_ram, is_mac=is_mac, is_cuda=is_cuda)
            
            cmd = ['-ti', input_file, '-tp', instruction, '--edit-model', edit_model]
            
            # Add Framework/Precision params
            if precision and precision != "auto":
                 cmd += ['-pf', precision]
            if framework and framework != "auto":
                 cmd += ['--ml-framework', framework]
                 
        elif operation == "rembg":
            cmd = ['-ti', input_file, '--remove-background']
        elif operation == "silhouette":
            cmd = ['-ti', input_file, '--remove-background', '--silhouette']
        
        # Output
        print()
        output = prompt_text("💾 Output filename (or press Enter for auto)", required=False)
        if output:
            cmd += ['-o', output]
        
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

        # Build command list
        if media_type == "image":
            cmd = ["-ui", input_file, "-uf", factor]
            if method == "ai":
                cmd += ["-iu", ai_model]
            else:
                cmd += ["-su"]
        else:
            cmd = ["-uv", input_file, "-uf", factor]
            if method == "ai":
                cmd += ["-vu", ai_model]
                if video_codec != "auto":
                    cmd += ["-vc", video_codec]
            else:
                cmd += ["-su"]
        
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
        
        # Build command list
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
            cmd = ["-cd", input_file, "-cdt", target_format, "-om", ocr_model]
        elif media_type == "image":
            cmd = ["-ci", input_file, "-cit", target_format]
        elif media_type == "video":
            cmd = ["-cv", input_file, "-cvt", target_format]
        else:
            cmd = ["-ca", input_file, "-cat", target_format]
        
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
        
        # Build command list
        cmd = ["-cd", input_file, "-cdt", target_format]
        
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
            cmd += ["-om", ocr_model]
        
        run_self_command(cmd)
        wait_for_back()
    
    def translate_menu():
        """Translate submenu."""
        clear_screen()
        show_header("Translate")
        
        # Define supported file types
        IMAGE_EXTS = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff']
        AUDIO_EXTS = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']
        DOC_EXTS = ['.pdf', '.docx', '.doc', '.rtf', '.txt', '.md', '.html']
        
        # 1. Select Mode
        print("🌐 Select Translation Mode:\n")
        mode_options = [
            ("Text (Enter text directly)", "text"),
            (f"File (Images: {', '.join(IMAGE_EXTS[:4])}... | Audio: {', '.join(AUDIO_EXTS[:3])}... | Docs: {', '.join(DOC_EXTS[:3])}...)", "file"),
        ]
        mode = prompt_choice("Mode", mode_options)
        if mode is None: return

        # 2. Input Data
        input_data = None
        ocr_model = None
        detected_type = None  # "image", "audio", "document", "text"
        
        if mode == "text":
            print()
            input_data = prompt_text("📝 Enter text to translate")
            if not input_data: return
            detected_type = "text"
        else:
            print()
            input_data = prompt_file("Input File (Image, Audio, or Document)")
            if not input_data: return
            
            # Detect file type by extension
            ext = os.path.splitext(input_data)[1].lower()
            
            if ext in IMAGE_EXTS:
                detected_type = "image"
                print(f"\n🖼️  Detected: Image file")
                
                # OCR Model selection for images
                print("\n📦 Select OCR Model:\n")
                ocr_options = [
                    ("Florence-2 (Fast, Lightweight, Default)", "florence"),
                    ("Qwen-VL (High Precision, ~30GB)", "qwen-vl"),
                ]
                ocr_model = prompt_choice("OCR Model", ocr_options)
                if ocr_model is None: return
                
            elif ext in AUDIO_EXTS:
                detected_type = "audio"
                print(f"\n🎤 Detected: Audio file (Speech Translation)")
                
            elif ext in DOC_EXTS:
                detected_type = "document"
                print(f"\n📄 Detected: Document file")
                
            else:
                # Unknown type - try as text file
                detected_type = "document"
                print(f"\n📄 Treating as text document")
        
        # 3. Select Translation Model (from models.py)
        print("\n📦 Select Translation Model:\n")
        from ai_media.models import TRANSLATION_MODELS
        
        # Build model options from TRANSLATION_MODELS (exclude defaults)
        model_labels = {
            "nllb-200-3.3b": "NLLB 200 - 3.3B (High Quality, Default)",
            "nllb-200-distilled": "NLLB 200 - Distilled (Fast)",
            "alma-13b": "ALMA 13B (Professional Text)",
            "qwen3-8b": "Qwen 3 8B (Natural Output)",
            "qwen3-14b": "Qwen 3 14B (Best Context)",
            "llama-3.1-8b": "Llama 3.1 8B (Idioms & Nuance)",
            "seamless-m4t-v2-large": "Seamless M4T v2 (Speech Only)",
        }
        
        # Filter models based on detected type
        if detected_type == "audio":
            # Audio: show Seamless first (best for speech), then others
            model_keys = ["seamless-m4t-v2-large"]  # Only Seamless for speech
        else:
            # Text/Image/Document: show text models, exclude Seamless
            model_keys = [k for k in TRANSLATION_MODELS.keys() 
                         if not k.startswith("default") and k != "seamless-m4t-v2-large"]
        
        model_options = [(model_labels.get(k, k), k) for k in model_keys]
        model = prompt_choice("Model", model_options)
        if model is None: return

        # 4. Target Language
        print("\n🗣️  Select Target Language:\n")
        
        # Get languages with full names from pycountry (cached)
        from ai_media.models import get_nllb_languages_with_names
        
        lang_list = list(get_nllb_languages_with_names())
        lang_list.append(("Other (Manual Entry)", "manual"))
        
        target_lang = prompt_menu("Target Language", lang_list, page_size=30)
            
        if target_lang is None: return
        
        if target_lang == "manual":
            target_lang = prompt_text("Enter NLLB language code (e.g. eng_Latn, fra_Latn)")
            if not target_lang: return

        # Build Command based on detected type
        if detected_type == "text":
            cmd = ['-tr', '--target-language', target_lang, '--translation-model', model, '-p', input_data]
        elif detected_type == "image":
            # Image translation uses document convert with OCR + translate
            cmd = ['-cd', input_data, '-cdt', 'png', '--translate', '--target-language', target_lang, '--translation-model', model, '-om', ocr_model]
        elif detected_type == "audio":
            # Audio uses speech translation
            cmd = ['-tr', '--target-language', target_lang, '--translation-model', model, '-ii', input_data]
        else:
            # Document translation
            cmd = ['-tr', '--target-language', target_lang, '--translation-model', model, '-ii', input_data]
             
        run_self_command(cmd)
        wait_for_back()

    def analysis_menu(preset_model=None):
        """Analysis submenu (Caption & Subtitles)."""
        clear_screen()
        show_header("Analysis Tools")
        
        # Select Task
        print("👁️  Select Analysis Task:\n")
        task_options = [
            ("Generate Description (Image/Video)", "analysis"),
            ("Generate Subtitles (Auto-Subtitles + Translation)", "subtitles"),
        ]
        task = prompt_choice("Task", task_options)
        
        if task is None:
            return

        if task == "analysis":
            # --- Caption Flow ---
            print("\n📂 Select input image or video:\n")
            input_file = prompt_file("Enter file path")
            if input_file is None:
                return
            
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
            
            cmd = ["-gd", input_file, "-cm", model]
            run_self_command(cmd)
            wait_for_back()

        elif task == "subtitles":
            # --- Subtitles Flow ---
            print("\n🎬 Select input video or audio file:\n")
            input_file = prompt_file("Enter file path")
            if input_file is None:
                return

            print("\n📦 Select Whisper Model (Transcription):\n")
            w_options = [
                ("Small (Balanced)", "small"),
                ("Medium (Better)", "medium"),
                ("Large-v3 (Best)", "large-v3"),
                ("Base (Fast)", "base"),
                ("Tiny (Lightweight)", "tiny"),
            ]
            w_model = prompt_choice("Whisper Model", w_options)
            if w_model is None:
                return
            
            # Subtitle Format
            print("\n📄 Select Output Format:\n")
            fmt_options = [
                ("SRT (Universal, default)", "srt"),
                ("VTT (Web browsers)", "vtt"),
                ("ASS (Styled subtitles)", "ass"),
                ("SUB (MicroDVD, frame-based)", "sub"),
                ("TXT (Plain text, no timestamps)", "txt"),
                ("JSON (Structured data)", "json"),
            ]
            sub_format = prompt_choice("Format", fmt_options)
            if sub_format is None:
                sub_format = "srt"
            
            # VAD Preset
            print("\n🔊 VAD Preset (Voice Activity Detection):\n")
            vad_options = [
                ("Normal (Default)", "normal"),
                ("Noisy (Strict, for noisy recordings)", "noisy"),
                ("Sensitive (For quiet/faint speech)", "sensitive"),
            ]
            vad_preset = prompt_choice("VAD Preset", vad_options)
            if vad_preset is None:
                vad_preset = "normal"
            
            # Translation
            print()
            translate = input("🌐 Translate subtitles? [y/N]: ").strip().lower()
            target_langs = ""
            if translate in ['y', 'yes']:
                print("\n   Enter target language codes (comma separated).")
                print("   Common: es (Spanish), fr (French), de (German), ja (Japanese), zh (Chinese)")
                target_langs = prompt_text("Target Languages (e.g. 'fr,es')")
            
            cmd = ['-gs', '-ii', input_file, '--whisper-model', w_model, '--subtitle-format', sub_format]
            if vad_preset != "normal":
                cmd += ['--subtitle-vad-preset', vad_preset]
            
            if target_langs:
                cmd += ['--subtitle-translate-to', target_langs]

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
            from ai_media.utils.prompts import RANDOM_PROMPT_TRIGGERS
            triggers_str = ", ".join([f"'{t}'" for t in RANDOM_PROMPT_TRIGGERS])
            print(f"🎲 Tip: Enter {triggers_str} for a surprise topic!\n")
            print("(Random prompt used if empty)\n")
            topic = prompt_text("Topic", required=False)
            if topic is None:
                return
            if not topic:
                topic = "rndpr" # Default to random prompt if empty
            
            # Handle random prompt trigger
            from ai_media.utils.prompts import maybe_replace_with_random
            topic, was_random = maybe_replace_with_random(topic, "article")
            if was_random:
                print(f"🎲 Using random topic: {topic}")
            
            # Framework selection FIRST
            framework = "torch"
            import platform
            try:
                import torch
            except ImportError:
                torch = None # Should handle gracefully if not available (but likely is)

            is_mac = platform.system() == "Darwin"
            
            if is_mac:
                from ai_media.utils.precision import get_supported_frameworks
                from ai_media.utils.precision import is_mlx_available
                mlx_ok = is_mlx_available()
                
                if mlx_ok:
                    if ml_framework:
                         framework = ml_framework
                         print(f"\n🏗️  Framework: {framework.upper()} (Pre-selected via CLI)")
                    else:
                        print("\n🏗️  Select Framework:\n")
                        fw_options = [
                            ("MLX (Native Apple Silicon) [Default]", "mlx"),
                            ("PyTorch (MPS/CPU)", "torch")
                        ]
                        # Note: We don't filter frameworks by precision here because precision is next
                        framework = prompt_choice("Framework", fw_options)
                        if framework is None: return

            # Precision selection (filtered by framework)
            if precision_force:
                 precision = precision_force
                 print(f"\n⚙️  Precision: {precision} (Pre-selected via CLI)")
            else:
                print("\n⚙️  Select Precision:\n")
            
            # Determine auto label based on framework/platform
            auto_label = "Auto (Platform Default)"
            if framework == "mlx":
                auto_label = "Auto (int4 - MLX Default)"
            elif framework == "torch" and is_mac:
                auto_label = "Auto (bfloat16 - Default)"
            elif torch and torch.cuda.is_available():
                auto_label = "Auto (float16 - CUDA Default)"
            else:
                auto_label = "Auto (float32 - CPU Default)"

            # Get supported precisions for this framework
            from ai_media.utils.precision import get_supported_precisions
            device_type = "cpu"
            if torch and torch.cuda.is_available(): device_type = "cuda"
            elif is_mac: device_type = "mps"
            
            supported_precs = get_supported_precisions(device_type, framework)
            
            # Build precision options
            all_options = [
                ("int4 (4-bit, Fastest)", "int4"),
                ("int6 (6-bit, Balanced Speed)", "int6"),
                ("int8 (8-bit, Balanced Quality)", "int8"),
                ("bfloat16 (Brain Float)", "bfloat16"),
                ("float16 (Half)", "float16"),
                ("float32 (Full)", "float32"),
            ]
            
            precision_options = [(auto_label, "auto")]
            for label, val in all_options:
                if val in supported_precs:
                    precision_options.append((label, val))
            
            precision = prompt_choice("Precision", precision_options)
            if precision is None:
                return
            
            # Model selection with dynamic RAM based on precision
            print("\n📦 Select Model:\n")
            from ai_media.utils.model_resources import get_text_model_options
            from ai_media.utils.system import get_system_resources
            sys_resources = get_system_resources()
            sys_ram = sys_resources.get("ram_total", 0)
            
            model_options = get_text_model_options(precision, system_ram_gb=sys_ram)
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
                ("Exhaustive (~10000 words, exhaustive)", "exhaustive"),
            ]
            length = prompt_choice("Length", length_options)
            if length is None:
                return
            
            # Output file path (optional)
            print("\n📁 Output file path (leave empty for auto-name):\n")
            output_path = prompt_text("File path", required=False)
            
            # Build command list
            flag = "-gr" if online else "-ga"
            cmd = [flag, "-p", topic, "-atm", model, "--output-format", output_format, "-al", length]
            if precision != "auto":
                cmd += ["-pf", precision]
            if framework:
                cmd += ["--ml-framework", framework]
            if online:
                cmd += ["-ri", str(research_iter)]
            if output_path:
                cmd += ["-o", output_path]
            
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
            from ai_media.utils.prompts import RANDOM_PROMPT_TRIGGERS
            triggers_str = ", ".join([f"'{t}'" for t in RANDOM_PROMPT_TRIGGERS])
            print(f"🎲 Tip: Enter {triggers_str} for a surprise Coding prompt!\n")
            print("(Random prompt used if empty)\n")
            description = prompt_text("Description", required=False)
            if description is None:
                return
            if not description:
                description = "rndpr"
            
            # Handle random prompt trigger
            from ai_media.utils.prompts import maybe_replace_with_random
            description, was_random = maybe_replace_with_random(description, "code")
            if was_random:
                print(f"🎲 Random prompt: {description}")
            
            # Framework selection FIRST
            framework = "torch"
            import platform
            try:
                import torch
            except ImportError:
                torch = None

            is_mac = platform.system() == "Darwin"
            
            if is_mac:
                from ai_media.utils.precision import get_supported_frameworks
                from ai_media.utils.precision import is_mlx_available
                mlx_ok = is_mlx_available()
                
                if mlx_ok:
                    if ml_framework:
                         framework = ml_framework
                         print(f"\n🏗️  Framework: {framework.upper()} (Pre-selected via CLI)")
                    else:
                        print("\n🏗️  Select Framework:\n")
                        fw_options = [
                            ("MLX (Native Apple Silicon) [Default]", "mlx"),
                            ("PyTorch (MPS/CPU)", "torch")
                        ]
                        # Note: We don't filter frameworks by precision here because precision is next
                        framework = prompt_choice("Framework", fw_options)
                        if framework is None: return

            # Precision selection (filtered by framework)
            if precision_force:
                 precision = precision_force
                 print(f"\n⚙️  Precision: {precision} (Pre-selected via CLI)")
            else:
                print("\n⚙️  Select Precision:\n")
            
            # Determine auto label based on framework/platform
            auto_label = "Auto (Platform Default)"
            if framework == "mlx":
                auto_label = "Auto (int4 - MLX Default)"
            elif framework == "torch" and is_mac:
                auto_label = "Auto (bfloat16 - Default)"
            elif torch and torch.cuda.is_available():
                auto_label = "Auto (float16 - CUDA Default)"
            else:
                auto_label = "Auto (float32 - CPU Default)"

            # Get supported precisions for this framework
            from ai_media.utils.precision import get_supported_precisions
            device_type = "cpu"
            if torch and torch.cuda.is_available(): device_type = "cuda"
            elif is_mac: device_type = "mps"
            
            supported_precs = get_supported_precisions(device_type, framework)
            
            # Build precision options
            all_options = [
                ("int4 (4-bit, Fastest)", "int4"),
                ("int6 (6-bit, Balanced Speed)", "int6"),
                ("int8 (8-bit, Balanced Quality)", "int8"),
                ("bfloat16 (Brain Float)", "bfloat16"),
                ("float16 (Half)", "float16"),
                ("float32 (Full)", "float32"),
            ]
            
            precision_options = [(auto_label, "auto")]
            for label, val in all_options:
                if val in supported_precs:
                    precision_options.append((label, val))
            
            precision = prompt_choice("Precision", precision_options)
            if precision is None:
                return
            
            # Model selection with dynamic RAM based on precision
            print("\n📦 Select Code Model:\n")
            from ai_media.utils.model_resources import get_code_model_options
            from ai_media.utils.system import get_system_resources
            sys_resources = get_system_resources()
            sys_ram = sys_resources.get("ram_total", 0)
            
            model_options = get_code_model_options(precision, system_ram_gb=sys_ram)
            model = prompt_choice("Model", model_options)
            if model is None:
                return
            
            # Output path (optional)
            print("\n📁 Output path (optional):")
            print("   (Leave empty: uses paths/filenames from your description)")
            print("   (Existing folder: saves all generated files inside it)")
            print("   (Filename: override output name if single file)\n")
            output_path = prompt_text("Output path", required=False)
            
            # Build command list
            cmd = ['-gc', '-p', description, '-cdm', model]
            if precision != "auto":
                cmd += ['-pf', precision]
            if framework:
                cmd += ['--ml-framework', framework]
            if output_path:
                cmd += ['-o', output_path]
            
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
            # Use default precision for preset model
            precision = "auto"
        else:
            # Framework selection FIRST
            framework = "torch"
            import platform
            try:
                import torch
            except ImportError:
                torch = None

            is_mac = platform.system() == "Darwin"
            
            if is_mac:
                from ai_media.utils.precision import get_supported_frameworks
                from ai_media.utils.precision import is_mlx_available
                mlx_ok = is_mlx_available()
                
                if mlx_ok:
                    print("\n🏗️  Select Framework:\n")
                    fw_options = [
                        ("MLX (Native Apple Silicon) [Default]", "mlx"),
                        ("PyTorch (MPS/CPU)", "torch")
                    ]
                    # Note: We don't filter frameworks by precision here because precision is next
                    framework = prompt_choice("Framework", fw_options)
                    if framework is None: return

            # Precision selection (filtered by framework)
            print("\n⚙️  Select Precision:\n")
            
            # Determine auto label based on framework/platform
            auto_label = "Auto (Platform Default)"
            if framework == "mlx":
                auto_label = "Auto (int4 - MLX Default)"
            elif framework == "torch" and is_mac:
                auto_label = "Auto (bfloat16 - Default)"
            elif torch and torch.cuda.is_available():
                auto_label = "Auto (float16 - CUDA Default)"
            else:
                auto_label = "Auto (float32 - CPU Default)"

            # Get supported precisions for this framework
            from ai_media.utils.precision import get_supported_precisions
            device_type = "cpu"
            if torch and torch.cuda.is_available(): device_type = "cuda"
            elif is_mac: device_type = "mps"
            
            supported_precs = get_supported_precisions(device_type, framework)
            
            # Build precision options
            all_options = [
                ("int4 (4-bit, Fastest)", "int4"),
                ("int6 (6-bit, Balanced Speed)", "int6"),
                ("int8 (8-bit, Balanced Quality)", "int8"),
                ("bfloat16 (Brain Float)", "bfloat16"),
                ("float16 (Half)", "float16"),
                ("float32 (Full)", "float32"),
            ]
            
            precision_options = [(auto_label, "auto")]
            for label, val in all_options:
                if val in supported_precs:
                    precision_options.append((label, val))
            
            precision = prompt_choice("Precision", precision_options)
            if precision is None:
                return
            
            # Model selection with dynamic RAM based on precision
            print("\n📦 Select Chat Model:\n")
            from ai_media.utils.model_resources import get_chat_model_options
            from ai_media.utils.system import get_system_resources
            sys_resources = get_system_resources()
            sys_ram = sys_resources.get("ram_total", 0)
            
            model_options = get_chat_model_options(precision, system_ram_gb=sys_ram)
            model = prompt_choice("Model", model_options)
            if model is None:
                return
        
        # Build command list and run
        cmd = ["-c", "--chat-model", model]
        if precision != "auto":
            cmd += ["-pf", precision]
        if framework:
            cmd += ["--ml-framework", framework]
        
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
                
                wait_for_back(None)
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
                    
                    wait_for_back(None)
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
            class_pattern = r'^class\s+(Test\w+)\s*\(\s*(?:unittest\.)?(?:TestCase|IsolatedAsyncioTestCase)\s*\)\s*:'
            method_pattern = r'^\s+(?:async\s+)?def\s+(test_\w+)\s*\('
            
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
                run_shell_command([sys.executable, "-m", "unittest", "ai_media.testing.unit_tests"])
            elif choice == "ALL_VERBOSE":
                # Run all unit tests (Verbose)
                print("\n🧪 Running all unit tests (Verbose)...\n")
                print("=" * 60)
                run_shell_command([sys.executable, "-m", "unittest", "ai_media.testing.unit_tests", "-v"])
            else:
                # Run specific test class (Always Verbose)
                print(f"\n🧪 Running {choice}...\n")
                print("=" * 60)
                run_shell_command([sys.executable, "-m", "unittest", f"ai_media.testing.unit_tests.{choice}", "-v"])

            wait_for_back(prompt=None)

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
            prompt_menu(None, [], allow_back=True)
            return

        if not tests:
            clear_screen()
            show_header("App Run Tests")
            print("❌ No tests found.")
            wait_for_back(None)
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
                # ... info prints ...
                pattern = prompt_text("Pattern", required=True)
                if pattern:
                    run_self_command(["--test-verbose", pattern])
                    prompt_menu(None, [], allow_back=True)
                continue
            else:
                # Run specific test
                # Always use verbose for single test as requested
                run_self_command(["--test-verbose", choice])
                
            wait_for_back()


    def web_server_menu():
        """Web Server Mode submenu."""
        while True:
            clear_screen()
            show_header("Web Server Mode")
            
            # Detect Mac for dynamic options
            import sys
            is_mac = sys.platform == 'darwin'
            
            options = []
            
            if is_mac:
                options.append(("🧩  Start Inference Server (OpenAI Compatible) [PyTorch]", "INFERENCE_TORCH"))
                options.append(("🧩  Start Inference Server (OpenAI Compatible, Verbose) [PyTorch]", "INFERENCE_VERBOSE_TORCH"))
                options.append(("🍎  Start Inference Server (OpenAI Compatible) [MLX]", "INFERENCE_MLX"))
                options.append(("🍎  Start Inference Server (OpenAI Compatible, Verbose) [MLX]", "INFERENCE_VERBOSE_MLX"))
            else:
                options.append(("🧩  Start Inference Server (OpenAI Compatible)", "INFERENCE"))
                options.append(("🧩  Start Inference Server (OpenAI Compatible, Verbose)", "INFERENCE_VERBOSE"))
                
            options.extend([
                ("🚀  Start Server (No Client)", "SERVER_ONLY"),
                ("🌐  Start Client (Web)", "WEB_CLIENT"),
                ("🔥  Start Both Server and Web Client", "BOTH_WEB"),
                ("⚡  Start Both Server and Web + Electron Dev Client", "BOTH_FULL"),
                ("🛠️   Electron Build Options", "BUILD_OPTS"),
                ("📦  Versioning Scripts", "VERSION_OPTS"),
            ])
            
            # Find index of BOTH_WEB to set it as default (Correct for Mac, CUDA, or CPU)
            default_idx = next((i for i, opt in enumerate(options) if opt[1] == "BOTH_WEB"), 0)
            
            choice = prompt_choice("Select an option:", options, allow_back=True, default_index=default_idx)
            
            if choice is None: return
            
            web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
            
            code = 0
            # Common Commands
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
            
            # Inference Server Dynamic Options
            elif choice == "INFERENCE":
                code = run_self_command(["--inference-server"])
            elif choice == "INFERENCE_VERBOSE":
                code = run_self_command(["--inference-server-verbose"])
            elif choice == "INFERENCE_TORCH":
                code = run_self_command(["--inference-server", "--ml-framework", "torch"])
            elif choice == "INFERENCE_VERBOSE_TORCH":
                code = run_self_command(["--inference-server-verbose", "--ml-framework", "torch"])
            elif choice == "INFERENCE_MLX":
                code = run_self_command(["--inference-server", "--ml-framework", "mlx"])
            elif choice == "INFERENCE_VERBOSE_MLX":
                code = run_self_command(["--inference-server-verbose", "--ml-framework", "mlx"])
                
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
            run_shell_command(["npm", "run", choice], cwd=web_dir)
            wait_for_back(None)

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
        run_shell_command(["npm", "run", choice], cwd=web_dir)
        wait_for_back()


    
    def cleanup_menu():
        """Cleanup options submenu."""
        while True:
            clear_screen()
            show_header("Cleanup / Maintenance")
            
            options = [
                ("🧹  Clear testing/data/outputs", "--clear-data-output"),
                ("🗑️   Clear media_output", "--clear-media-output"),
                ("🔥  Clear All Output Data", "--clear-all-outputs"),
                ("🗃️   Clear Hub Model", "HUB_MODEL"),
            ]
            
            choice = prompt_choice("Select cleanup action:", options, allow_back=True)
            if choice is None: return
            
            if choice == "HUB_MODEL":
                hub_model_menu()
            else:
                # Execute cleanup
                run_self_command(choice)
                wait_for_back(None)
    
    def hub_model_menu():
        """Browse and delete cached HuggingFace hub models."""
        from .server.config import CONFIG
        from .utils.cleanup import format_size, get_folder_size, clear_directory
        
        while True:
            clear_screen()
            show_header("Clear Hub Model")
            
            # Get hub path from config
            hf_home = CONFIG["paths"].get("hf_home")
            if not hf_home:
                print("❌ hf_home not configured in config.json")
                wait_for_back(None)
                return
            
            hub_path = os.path.join(hf_home, "hub")
            if not os.path.exists(hub_path):
                print(f"❌ Hub folder not found: {hub_path}")
                wait_for_back(None)
                return
            
            # Show loading indicator
            print("⏳ Loading hub models information...", end="", flush=True)
            
            # Scan folders
            folders = []
            try:
                for item in os.listdir(hub_path):
                    # Skip hidden folders like .locks
                    if item.startswith("."):
                        continue
                    item_path = os.path.join(hub_path, item)
                    if os.path.isdir(item_path):
                        size = get_folder_size(item_path)
                        folders.append((item, size))
            except OSError as e:
                print(f"\r❌ Error scanning hub folder: {e}" + " " * 20)
                wait_for_back(None)
                return
            
            # Clear loading indicator
            print("\r" + " " * 50 + "\r", end="", flush=True)
            
            if not folders:
                print("📭 No models found in hub folder.")
                wait_for_back(None)
                return
            
            # Sort by size (largest first)
            folders.sort(key=lambda x: x[1], reverse=True)
            
            # Calculate total size
            total_size = sum(size for _, size in folders)
            
            # Build menu options
            options = []
            for name, size in folders:
                display = f"{name} ({format_size(size)})"
                options.append((display, name))
            
            # Show info panel
            console.print(Panel(
                f"[bold cyan]📁 Hub Path:[/bold cyan] {hub_path}\n"
                f"[bold cyan]📊 Total Size:[/bold cyan] {format_size(total_size)} ({len(folders)} models)",
                border_style="blue",
                padding=(0, 2),
                width=80
            ))
            print()
            
            choice = prompt_choice("Select a model to delete:", options, allow_back=True)
            if choice is None: return
            
            # Confirmation warning
            clear_screen()
            show_header("Clear Hub Model")
            console.print(f"⚠️  You are about to delete: [bold]{choice}[/bold]")
            print()
            print("This model's folder will be permanently removed.")
            print("You will need to redownload it again next time the model needs to be used.")
            print()
            
            try:
                confirm = input("Continue? [y/N]: ").lower().strip()
            except KeyboardInterrupt:
                print("\n❌ Cancelled.")
                continue
                
            if confirm != 'y':
                print("❌ Cancelled.")
                import time
                time.sleep(0.5)
                continue
            
            # Delete the folder
            folder_path = os.path.join(hub_path, choice)
            run_self_command(["--clear-hub-model", choice])
            wait_for_back(None)

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
            # Determine if initial_model is a category (music/tts) or a specific model
            if first_run and initial_action == 'audio':
                _model = initial_model
                if _model == 'music':
                    audio_menu(preset_category='music')
                elif _model == 'tts':
                    audio_menu(preset_category='tts')
                elif _model:
                    audio_menu(preset_model=_model)
                else:
                    audio_menu()
            else:
                audio_menu()
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
        elif action == "translate":
            translate_menu()
        elif action == "analysis":
            analysis_menu(initial_model if first_run or initial_action == 'analysis' else None)
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
        elif action == "cleanup":
            cleanup_menu()

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
