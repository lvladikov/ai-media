"""
Upscaling module for AI-Media.

Supports:
- Simple upscaling (PIL Lanczos for images, FFmpeg for video)
- AI upscaling with Stable Diffusion (x2/x4)
- Fast upscaling with Real-ESRGAN
"""

import os
import time
from pathlib import Path

from .models import IMAGE_MODELS
from .utils.system import get_optimal_device_and_dtype, clear_gpu_memory
from .utils.parsers import format_time
from .utils.performance import ResourceMonitor
from .utils.ffmpeg import get_video_encoding_params, _check_ffmpeg_encoder
from .utils.interaction import check_overwrite


# Check for Real-ESRGAN availability without eager importing
import importlib.util
HAS_REALESRGAN = importlib.util.find_spec("realesrgan") is not None

# Fix for basicsr compatibility with newer torchvision versions
# torchvision.transforms.functional_tensor was removed in newer versions
if HAS_REALESRGAN:
    import sys
    try:
        # Try to import the problematic module
        import torchvision.transforms.functional_tensor
    except ModuleNotFoundError:
        # Create a compatibility shim
        import torchvision.transforms.functional as _functional
        import types
        _functional_tensor = types.ModuleType('torchvision.transforms.functional_tensor')
        _functional_tensor.rgb_to_grayscale = _functional.rgb_to_grayscale
        sys.modules['torchvision.transforms.functional_tensor'] = _functional_tensor


def get_weights_cache_dir():
    """Get the cache directory for model weights (~/.cache/ai-media/weights/)."""
    cache_dir = Path.home() / ".cache" / "ai-media" / "weights"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_realesrgan_weights(model_name):
    """Download and cache Real-ESRGAN model weights. Returns path to local weights file.
    
    Args:
        model_name: Name of the model (e.g., 'RealESRGAN_x4plus')
        
    Returns:
        Path to the cached weights file
    """
    import urllib.request
    
    cache_dir = get_weights_cache_dir()
    model_path = cache_dir / f"{model_name}.pth"
    
    if not model_path.exists():
        print(f"   ⬇️  Downloading model: {model_name}...")
        url = f'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model_name}.pth'
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"   ✅ Cached to: {model_path}")
        except Exception as e:
            print(f"   ⚠️  Download failed: {e}. Using URL directly.")
            return url
    else:
        print(f"   📦 Using cached model: {model_path}")
    
    return str(model_path)


def check_resources_and_confirm(w, h, f, dev):
    """Check if target upscale resolution is safe for system resources."""
    try:
        import psutil
    except ImportError:
        return True
        
    target_w = int(w * f)
    target_h = int(h * f)
    target_pixels = target_w * target_h
    megapixels = target_pixels / 1_000_000
    
    estimated_ram_gb = (megapixels * 0.8) if dev == "cpu" else (megapixels * 0.4)
    
    vm = psutil.virtual_memory()
    available_gb = vm.available / (1024**3)
    
    is_huge = megapixels > 25
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
            print("   (Proceeding due to --force flag)")
            return True
             
        confirm = input("\n   Do you want to proceed? [y/N]: ").strip().lower()
        return confirm == 'y'
    return True


def simple_upscale_image(image_path, output_path, factor=2.0, force=False):
    """Simple non-AI image upscaling using PIL Lanczos interpolation."""
    from PIL import Image
    
    print(f"🔍 Simple Upscaling Image: {image_path}")
    print(f"   Method: PIL Lanczos (No AI)")
    print(f"   Factor: {factor}x")
    
    should_write, output_path, _, _ = check_overwrite(output_path, always_overwrite=force)
    if not should_write:
        return False
    
    try:
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        target_w = int(orig_w * factor)
        target_h = int(orig_h * factor)
        
        print(f"   {orig_w}x{orig_h} → {target_w}x{target_h}")
        
        upscaled = image.resize((target_w, target_h), Image.LANCZOS)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        upscaled.save(output_path)
        
        print(f"✅ Simple upscaled image saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Simple upscaling failed: {e}")
        return False


def simple_upscale_video(video_path, output_path, factor=2.0, force=False):
    """Simple non-AI video upscaling using FFmpeg scale filter."""
    import subprocess
    
    print(f"🔍 Simple Upscaling Video: {video_path}")
    print(f"   Method: FFmpeg Lanczos (No AI)")
    print(f"   Factor: {factor}x")
    
    should_write, output_path, _, _ = check_overwrite(output_path, always_overwrite=force)
    if not should_write:
        return False
    
    try:
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
        
        target_w = target_w if target_w % 2 == 0 else target_w + 1
        target_h = target_h if target_h % 2 == 0 else target_h + 1
        
        print(f"   {orig_w}x{orig_h} → {target_w}x{target_h}")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        encoding_params = get_video_encoding_params(str(output_path))
        video_params = [p for i, p in enumerate(encoding_params) 
                       if not (encoding_params[i-1:i] == ["-c:a"] or 
                              p in ["aac", "libopus", "wmav2", "mp3"])]
        video_params = [p for p in video_params if p != "-c:a"]
        
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", f"scale={target_w}:{target_h}:flags=lanczos",
            *video_params,
            "-c:a", "copy",
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


def upscale_image_fast(input_path, output_path, factor=4.0):
    """Upscale image using Real-ESRGAN (Fast, single pass)."""
    if not HAS_REALESRGAN:
        print("❌ Real-ESRGAN not installed. Cannot run fast upscale.")
        print("   Please install: pip install realesrgan")
        return False

    import cv2
    import numpy as np
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    
    print(f"🚀 Upscaling Image (Fast Mode): {input_path}")
    print(f"   Factor: {factor}x")
    
    try:
        start_time = time.time()
        
        if not os.path.exists(input_path):
            print(f"❌ Input file not found: {input_path}")
            return False

        should_write, output_path, _, _ = check_overwrite(output_path, always_overwrite=os.environ.get("AI_MEDIA_FORCE") == "1")
        if not should_write:
            return False

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        device, _ = get_optimal_device_and_dtype(quiet=True)
        print(f"   Device: {device}")

        model_name = 'RealESRGAN_x4plus'
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
        # Download and cache weights
        model_path_str = get_realesrgan_weights(model_name)

        upsampler = RealESRGANer(
            scale=4,
            model_path=model_path_str,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True if device.type != 'cpu' else False,
            device=device,
        )

        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"❌ Failed to load image: {input_path}")
            return False

        orig_h, orig_w = img.shape[:2]
        print(f"   Input: {orig_w}x{orig_h}")

        with ResourceMonitor() as monitor:
            print(f"   🎨 Enhancing details with Real-ESRGAN...")
            output, _ = upsampler.enhance(img, outscale=factor)
            
            out_h, out_w = output.shape[:2]
            print(f"   ✅ Enhancement complete. Final size: {out_w}x{out_h}")
            
            cv2.imwrite(output_path, output)
            print(f"\n✅ Fast upscaled image saved to {output_path}")

        duration = time.time() - start_time
        cpu_p, ram_gb, vram_gb, gpu_p = monitor.get_averages()
        print(f"   ✓ Processed in {duration:.1f}s (RAM: {ram_gb:.1f}GB | VRAM: {vram_gb:.1f}GB)")

        return True

    except Exception as e:
        print(f"❌ Fast upscaling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def upscale_video_fast(video_path, output_path, factor=4.0, codec=None):
    """Upscale video using Real-ESRGAN (Fast, single pass per frame)."""
    if not HAS_REALESRGAN:
        print("❌ Real-ESRGAN not installed. Cannot run fast upscale.")
        print("   Please install: pip install realesrgan")
        return False

    import cv2
    import subprocess
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    
    print(f"🚀 Upscaling Video (Fast Mode): {video_path}")
    print(f"   Factor: {factor}x")
    
    try:
        if not os.path.exists(video_path):
            print(f"❌ Input file not found: {video_path}")
            return False

        should_write, output_path, _, _ = check_overwrite(str(output_path), always_overwrite=os.environ.get("AI_MEDIA_FORCE") == "1")
        if not should_write:
            return False

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        device, _ = get_optimal_device_and_dtype(quiet=True)
        print(f"   Device: {device}")

        model_name = 'RealESRGAN_x4plus'
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
        # Download and cache weights
        model_path_str = get_realesrgan_weights(model_name)

        upsampler = RealESRGANer(
            scale=4,
            model_path=model_path_str,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True if device.type != 'cpu' else False,
            device=device,
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Failed to open input video.")
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"   Input: {width}x{height} @ {fps}fps ({total_frames} frames)")
        
        target_w = int(width * factor)
        target_h = int(height * factor)
        
        # Stability Check
        STABLE_LIMIT = 15360
        if target_w > STABLE_LIMIT or target_h > STABLE_LIMIT:
            print(f"   ❌ Target Resolution {target_w}x{target_h} exceeds the stable 15K limit (15360px).")
            print("      Stability cannot be guaranteed. Aborting upscale.")
            return False

        if target_w % 2 != 0:
            target_w += 1
        if target_h % 2 != 0:
            target_h += 1

        print(f"   Output: {target_w}x{target_h}")
        
        # 1. Determine Encoder and Codec
        vcodec = "libx264"
        if codec == 'av1':
            vcodec = 'av1_nvenc' if device.type == 'cuda' else 'libsvtav1'
            if not _check_ffmpeg_encoder(vcodec, target_w, target_h):
                print(f"   ⚠️  Hardware AV1 not supported. Falling back to HEVC.")
                codec = 'hevc'
        
        if codec == 'hevc':
            vcodec = 'hevc_nvenc' if device.type == 'cuda' else 'libx265'
            if not _check_ffmpeg_encoder(vcodec, target_w, target_h):
                print(f"   ⚠️  Hardware HEVC not supported. Falling back to H.264.")
                vcodec = "libx264"
                print(f"   🎞️  Using H.264 (CPU)")
            else:
                print(f"   🎞️  Using Hardware HEVC")
        elif not codec:
            # Standard auto-detection
            params = get_video_encoding_params(output_path)
            for i, p in enumerate(params):
                if p == "-c:v":
                    vcodec = params[i+1]
        
        # 2. Setup Video Writer or FFmpeg Pipe
        temp_video_out = output_path + ".temp.mp4"
        
        # If we need a specific codec or hardware acceleration, FFmpeg pipe is better
        use_ffmpeg_pipe = (vcodec != "libx264" and vcodec != "mp4v")
        
        if not use_ffmpeg_pipe:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_out, fourcc, fps, (target_w, target_h))
            if not out.isOpened():
                use_ffmpeg_pipe = True
        
        if use_ffmpeg_pipe:
            print(f"   🎞️  Encoding with FFmpeg: {vcodec}")
            cmd = [
                'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{target_w}x{target_h}', '-pix_fmt', 'bgr24', '-r', str(fps),
                '-i', '-', '-c:v', vcodec, '-pix_fmt', 'yuv420p',
                '-loglevel', 'warning', temp_video_out
            ]
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            out = None
        else:
            process = None

        start_time = time.time()
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            if frame_idx % 10 == 0 or frame_idx == 1:
                print(f"   🎨 Processing frame {frame_idx}/{total_frames}...", end='\r')
            
            upscaled_frame, _ = upsampler.enhance(frame, outscale=factor)
            
            if upscaled_frame.shape[:2] != (target_h, target_w):
                upscaled_frame = cv2.resize(upscaled_frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            
            if out:
                out.write(upscaled_frame)
            else:
                process.stdin.write(upscaled_frame.tobytes())
        
        print()
        cap.release()
        if out:
            out.release()
        if process:
            process.stdin.close()
            process.wait()
        
        # Add audio back
        from .utils.ffmpeg import has_audio_track
        
        if has_audio_track(video_path):
            print("   🔊 Muxing audio from source...", flush=True)
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_video_out,    # Input 0: Upscaled video (silent)
                "-i", video_path,        # Input 1: Original video (audio source)
                "-map", "0:v",           # Use video from input 0
                "-map", "1:a",           # Use audio from input 1
                "-c:v", "copy",          # COPY video stream (preserves codec)
                "-c:a", "aac",           # Encode audio to AAC for compatibility
                "-shortest",             # Match shortest stream duration
                output_path,
                "-loglevel", "warning"
            ]
            try:
                subprocess.run(cmd, check=True)
                os.remove(temp_video_out)
            except subprocess.CalledProcessError:
                print("   ⚠️ Audio muxing failed. Saving silent video.")
                if os.path.exists(temp_video_out):
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    os.rename(temp_video_out, output_path)
        else:
            print("   ℹ️ No audio track in source, skipping mux.")
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_video_out, output_path)
        
        duration = time.time() - start_time
        print(f"\n✅ Fast upscaled video saved to {output_path}")
        print(f"   ✓ Processed in {format_time(duration)}")
        
        return True

    except Exception as e:
        print(f"❌ Fast upscaling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def upscale_image_file(image_path, output_path, strength=0.0, factor=2.0):
    """Upscale an image using smart multi-stage AI upscaling.
       
    Uses optimal combination of x4, x2 AI passes + final Lanczos resize.
    """
    import torch
    from diffusers import StableDiffusionUpscalePipeline, StableDiffusionLatentUpscalePipeline
    from PIL import Image
    
    use_x2_model = (factor <= 2.0)
    model_id = IMAGE_MODELS.get('upscaler_x2' if use_x2_model else 'upscaler')
    if not model_id:
        model_id = "stabilityai/stable-diffusion-x4-upscaler"
    
    print(f"🚀 Upscaling Image: {image_path}")
    print(f"   Model: {model_id}")
    print(f"   Target Factor: {factor}x")
    
    # Default to 0 (faithful to original) - user can increase for more detail generation
    noise_level = int(strength * 100)
    
    should_write, output_path, _, _ = check_overwrite(output_path, always_overwrite=os.environ.get("AI_MEDIA_FORCE") == "1")
    if not should_write:
        return False
    
    try:
        start_time = time.time()
        device, dtype = get_optimal_device_and_dtype()
        
        # Force CPU for MPS (upscaling has tensor size limits)
        if device.type == "mps":
            print("   ⚠️  MPS Compatibility: Switching to CPU for Upscaling.")
            device = torch.device("cpu")
            dtype = torch.float32
        
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        
        if not check_resources_and_confirm(orig_w, orig_h, factor, device.type):
            print("❌ Aborted by user.")
            return False
        
        # Calculate pass sequence
        passes = []
        remaining = factor
        
        while remaining >= 2.0:
            if remaining >= 4.0:
                passes.append(('x4', 4.0))
                remaining /= 4.0
            elif remaining >= 2.0:
                passes.append(('x2', 2.0))
                remaining /= 2.0
        
        print(f"\n   📋 Upscale Plan:")
        for i, (model_type, scale) in enumerate(passes, 1):
            print(f"      Pass {i}: {model_type} AI ({scale}x)")
        if remaining > 1.0:
            print(f"      Final: Lanczos resize ({remaining:.2f}x)")
        
        # Lazy pipeline loaders
        pipe_x2 = None
        pipe_x4 = None
        
        def get_pipeline(model_type):
            nonlocal pipe_x2, pipe_x4
            
            if model_type == 'x2':
                if pipe_x2 is None:
                    print(f"   🔗 Loading x2 Latent Upscaler...")
                    pipe_x2 = StableDiffusionLatentUpscalePipeline.from_pretrained(
                        IMAGE_MODELS.get('upscaler_x2', 'stabilityai/sd-x2-latent-upscaler'),
                        dtype=dtype,
                    )
                    pipe_x2.to(device)
                return pipe_x2
            else:
                if pipe_x4 is None:
                    print(f"   🔗 Loading x4 Upscaler...")
                    pipe_x4 = StableDiffusionUpscalePipeline.from_pretrained(
                        IMAGE_MODELS.get('upscaler', 'stabilityai/stable-diffusion-x4-upscaler'),
                        dtype=dtype,
                    )
                    pipe_x4.to(device)
                return pipe_x4
        
        upscale_prompt = "sharp, high resolution"
        negative_prompt = "blur, noise, artifacts, distortion"
        
        current_image = image
        
        with ResourceMonitor() as monitor:
            for pass_idx, (model_type, step_scale) in enumerate(passes, 1):
                print(f"\n🎨 Pass {pass_idx}/{len(passes)}: {model_type} AI ({step_scale}x)")
                
                pipe = get_pipeline(model_type)
                
                print(f"   ⏳ Rendering...")
                if model_type == 'x2':
                    result = pipe(prompt=upscale_prompt, image=current_image, num_inference_steps=50).images[0]
                else:
                    result = pipe(
                        prompt=upscale_prompt, 
                        image=current_image,
                        negative_prompt=negative_prompt,
                        noise_level=noise_level,
                        num_inference_steps=75
                    ).images[0]
                
                current_image = result
                print(f"   ✓ Pass complete: {current_image.size[0]}x{current_image.size[1]}")
        
        # Final Lanczos resize if needed
        target_w = int(orig_w * factor)
        target_h = int(orig_h * factor)
        
        if current_image.size != (target_w, target_h):
            print(f"\n   ↘️  Lanczos resize to exact target: {target_w}x{target_h}")
            current_image = current_image.resize((target_w, target_h), Image.LANCZOS)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        current_image.save(output_path)
        print(f"\n✅ Upscaled image saved to {output_path}")
        
        duration = time.time() - start_time
        cpu_p, ram_gb, vram_gb, gpu_p = monitor.get_averages()
        print(f"   ✓ Processed in {duration:.1f}s (RAM: {ram_gb:.1f}GB | VRAM: {vram_gb:.1f}GB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Upscaling failed: {e}")
        return False


def upscale_video_file(video_path, output_path, strength=0.0, factor=2.0):
    """Upscale video by extracting frames, upscaling them (recursively if needed), and stitching back."""
    import cv2
    import shutil
    import subprocess
    import torch
    from PIL import Image
    from diffusers import StableDiffusionUpscalePipeline, StableDiffusionLatentUpscalePipeline

    print(f"🚀 Upscaling Video: {video_path}")
    print(f"   Factor: {factor}x")
    
    should_write, output_path, _, _ = check_overwrite(output_path, always_overwrite=os.environ.get("AI_MEDIA_FORCE") == "1")
    if not should_write:
        return False
    
    try:
        cap_chk = cv2.VideoCapture(str(video_path))
        if cap_chk.isOpened():
            v_w = int(cap_chk.get(cv2.CAP_PROP_FRAME_WIDTH))
            v_h = int(cap_chk.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_chk.release()
             
            device, _ = get_optimal_device_and_dtype()
            if not check_resources_and_confirm(v_w, v_h, factor, device.type):
                return False
        
        temp_dir = Path("temp_upscale_frames")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()
        
        print("🎞️  Extracting frames...")
        cam = cv2.VideoCapture(str(video_path))
        fps = cam.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            frame_path = temp_dir / f"frame_{frame_count:05d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_count += 1
        cam.release()
        
        device, dtype = get_optimal_device_and_dtype()
        if device.type == "mps":
            device = torch.device("cpu")
            dtype = torch.float32

        use_x2_model = (factor <= 2.0)
        model_id = IMAGE_MODELS.get('upscaler_x2' if use_x2_model else 'upscaler')
        step_scale = 2.0 if use_x2_model else 4.0

        if use_x2_model:
            pipe = StableDiffusionLatentUpscalePipeline.from_pretrained(model_id, torch_dtype=dtype)
        else:
            pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=dtype)
            
        pipe.to(device)
        if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_tiling'):
            pipe.vae.enable_tiling()
        
        print("🎨 Upscaling frames...")
        for i in range(frame_count):
            input_f = temp_dir / f"frame_{i:05d}.png"
            output_f = temp_dir / f"upscaled_{i:05d}.png"
            img = Image.open(input_f).convert("RGB")
            
            current_img = img
            current_scale = 1.0
            while current_scale < factor:
                current_img = pipe(prompt="High quality", image=current_img, num_inference_steps=15).images[0]
                current_scale *= step_scale
            
            target_w = int(img.size[0] * factor)
            target_h = int(img.size[1] * factor)
            if current_img.size != (target_w, target_h):
                current_img = current_img.resize((target_w, target_h), Image.LANCZOS)
            current_img.save(output_f)
            print(f"   Frame {i+1}/{frame_count} done.", end='\r')
            
        print("\n🔗 stitching video...")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", str(temp_dir / "upscaled_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.rmtree(temp_dir)
        
        # Mux audio from source if present
        from .utils.ffmpeg import has_audio_track
        
        if has_audio_track(video_path):
            print(f"   🔗 Muxing audio from source...")
            temp_output = output_path + ".temp.mp4"
            os.rename(output_path, temp_output)
            
            mux_cmd = [
                "ffmpeg", "-y",
                "-i", temp_output,
                "-i", video_path,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-shortest",
                output_path, "-loglevel", "warning"
            ]
            try:
                subprocess.run(mux_cmd, check=True)
                os.remove(temp_output)
            except:
                print("   ⚠️ Audio muxing failed, keeping silent video.")
                os.rename(temp_output, output_path)
        else:
            print(f"   ℹ️ No audio track in source, skipping mux.")
        
        print(f"✅ Upscaled video saved to {output_path}")
        return True

    except Exception as e:
        print(f"❌ Video upscaling failed: {e}")
        return False
