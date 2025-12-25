"""
Video generation module for AI-Media.

Supports: Zeroscope, CogVideoX, LTX-Video, Mochi, Wan 2.2, HunyuanVideo, and SVD.
Includes dynamic upscaling pipeline for high-resolution output.
"""

import os
import time

from ..models import VIDEO_MODELS, AUDIO_MODELS, get_model_id
from ..utils.system import get_optimal_device_and_dtype, clear_gpu_memory
from ..utils.parsers import format_time
from ..utils.performance import PerformanceTracker, ResourceMonitor, write_report_json
from ..utils.ffmpeg import get_video_encoding_params, ffmpeg_resize_video


def upscale_video_zeroscope_xl(video_frames, prompt, device=None, dtype=None, strength=0.6):
    """Upscale video frames using zeroscope_v2_XL (Video-to-Video diffusion).
    
    This function takes video frames from zeroscope_v2_576w (576x320) and upscales
    them to 1024x576 using the XL model's video-to-video pipeline with temporal
    consistency.
    
    Args:
        video_frames: List of PIL Images or numpy arrays from 576w generation
        prompt: The original text prompt (used for guidance during upscaling)
        device: torch device (auto-detected if None)
        dtype: torch dtype (auto-detected if None)
        strength: Denoise strength 0.0-1.0. Higher = more creative, lower = faithful.
                  Default 0.6 as recommended by model docs.
                  
    Returns:
        List of upscaled PIL Images at 1024x576, or None on failure
    """
    print(f"\n🔄 Upscaling with Zeroscope XL (Video-to-Video)...")
    print(f"   Input:    576x320 ({len(video_frames)} frames)")
    print(f"   Output:   1024x576")
    print(f"   Strength: {strength}")
    
    try:
        import torch
        import numpy as np
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
        from PIL import Image
        
        # Auto-detect device/dtype if not provided
        if device is None or dtype is None:
            device, dtype = get_optimal_device_and_dtype(quiet=True)
        
        # MPS limitation: Force CPU for XL V2V upscale
        original_device = device
        if device.type == "mps":
            print("   ⚠️  MPS Limitation: Forcing CPU for XL V2V upscale (MPS tensor limit).")
            device = torch.device("cpu")
            dtype = torch.float32
        
        # Load XL model
        xl_model_id = VIDEO_MODELS.get("zeroscope-xl", "cerspense/zeroscope_v2_XL")
        print(f"   Model:    {xl_model_id}")
        print(f"   Device:   {device}")
        
        pipe = DiffusionPipeline.from_pretrained(xl_model_id, torch_dtype=dtype)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        # Memory optimization
        pipe.to(device)
        if device.type == "cpu":
            pipe.enable_attention_slicing()
        
        pipe.enable_vae_slicing()
        
        # Prepare input frames - resize to 1024x576 for XL input
        print(f"   Preparing frames...")
        resized_frames = []
        for frame in video_frames:
            if isinstance(frame, np.ndarray):
                if frame.dtype in [np.float32, np.float64]:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                frame = Image.fromarray(frame)
            resized_frame = frame.resize((1024, 576), Image.Resampling.LANCZOS)
            resized_frames.append(resized_frame)
        
        # Run XL upscaling (Video-to-Video)
        print(f"   Running diffusion upscale... (This may take a while)")
        start_time = time.time()
        
        with ResourceMonitor() as monitor:
            output = pipe(
                prompt=prompt,
                video=resized_frames,
                strength=strength,
                num_inference_steps=25
            )
        
        duration = time.time() - start_time
        avg_cpu, avg_ram, avg_vram, avg_gpu = monitor.get_averages()
        print(f"   ✓ XL Upscale complete in {format_time(duration)} (RAM: {avg_ram:.1f}GB | VRAM: {avg_vram:.1f}GB)")
        
        # Cleanup XL pipeline
        del pipe
        clear_gpu_memory()
        
        # Extract output frames
        upscaled_frames = output.frames[0] if hasattr(output, 'frames') else output.images
        return upscaled_frames
        
    except Exception as e:
        print(f"   ❌ XL Upscale failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_video(prompt, output_path, duration, width, height, model_name="default", 
                   image_input=None, audio_prompt=None, audio_model="default", report_json=None):
    """Generate video (Text-to-Video or Image-to-Video) with optional Audio.
    
    For Zeroscope: Implements dynamic upscaling pipeline when target resolution
    exceeds native 576x320. Uses zeroscope_v2_XL for V2V upscaling to 1024x576,
    then Real-ESRGAN for higher resolutions.
    
    Args:
        prompt: Text description for video generation
        output_path: Path to save video file
        duration: Target duration in seconds
        width: Target width in pixels
        height: Target height in pixels
        model_name: Model short code or HF ID
        image_input: Optional image path for Image-to-Video
        audio_prompt: Optional text for audio generation to mux with video
        audio_model: Audio model for audio_prompt
        report_json: Path to write performance stats JSON
        
    Returns:
        True on success, False on failure
    """
    # Resolve Model ID
    base_model = get_model_id(model_name, VIDEO_MODELS)
    
    # --- Zeroscope Dynamic Upscaling Detection ---
    is_zeroscope = "zeroscope" in base_model.lower() and "xl" not in base_model.lower()
    zeroscope_native_w, zeroscope_native_h = 576, 320
    zeroscope_xl_w, zeroscope_xl_h = 1024, 576
    
    needs_xl_upscale = False
    needs_esrgan_upscale = False
    target_width, target_height = width, height
    gen_width, gen_height = width, height
    
    import torch
    
    if is_zeroscope and not image_input:
        if width > zeroscope_native_w or height > zeroscope_native_h:
            gen_width, gen_height = zeroscope_native_w, zeroscope_native_h
            
            is_mps = torch.backends.mps.is_available() and not torch.cuda.is_available()
            
            if is_mps:
                needs_xl_upscale = False
                needs_esrgan_upscale = True
                print(f"📐 Dynamic Upscaling Pipeline (MPS Optimized):")
                print(f"   ⚠️  Skipping XL V2V (CPU diffusion too slow on Apple Silicon)")
                print(f"   Target:  {target_width}x{target_height}")
                print(f"   Step 1:  Generate at {gen_width}x{gen_height} (Zeroscope native)")
                print(f"   Step 2:  Real-ESRGAN to ~{target_width}x{target_height}")
                print(f"   Step 3:  FFmpeg resize to exact {target_width}x{target_height}")
            else:
                needs_xl_upscale = True
                if width > zeroscope_xl_w or height > zeroscope_xl_h:
                    needs_esrgan_upscale = True
                    
                print(f"📐 Dynamic Upscaling Pipeline Activated:")
                print(f"   Target:  {target_width}x{target_height}")
                print(f"   Step 1:  Generate at {gen_width}x{gen_height} (Zeroscope native)")
                if needs_xl_upscale:
                    print(f"   Step 2:  XL Upscale to {zeroscope_xl_w}x{zeroscope_xl_h} (V2V diffusion)")
                if needs_esrgan_upscale:
                    print(f"   Step 3:  Real-ESRGAN to ~{target_width}x{target_height}")
                    print(f"   Step 4:  FFmpeg resize to exact {target_width}x{target_height}")
            print("")
    
    # Handle Image-to-Video Logic
    is_i2v = True if image_input else False
    
    if is_i2v:
        if "cogvideox" in base_model.lower():
            model_id = "THUDM/CogVideoX-5b-I2V"
        elif "wan2.2" in base_model.lower() or "wan2.2" in model_name.lower():
            model_id = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
        elif "hunyuan" in base_model.lower() or "hunyuan" in model_name.lower():
            model_id = "hunyuanvideo-community/HunyuanVideo-I2V"
        elif "stable-video-diffusion" in base_model.lower() or model_name.lower() == "svd":
            model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
        else:
            print(f"⚠️  Warning: Model '{model_name}' ({base_model}) may not support Image-to-Video.")
            print(f"   Switching to 'svd' (Stable Video Diffusion) as fallback.")
            model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
    else:
        model_id = base_model

    print(f"{'='*60}")
    if needs_xl_upscale:
        print(f"📐 Step 1: Generate at {gen_width}x{gen_height} (Zeroscope native)")
    else:
        print(f"🎬 Generating Video ({'Image-to-Video' if is_i2v else 'Text-to-Video'})")
    print(f"{'='*60}")
    print(f"   Model:    {model_id}")
    print(f"   Prompt:   '{prompt}'")
    if is_i2v:
        print(f"   Input Img: {image_input}")
    if audio_prompt:
        print(f"   Audio:    '{audio_prompt}' (Will generate and mux)")
    print(f"   Duration: {duration}s")
    print("")
    
    # Determine actual video output path (temp if mixing audio)
    video_out = output_path
    if audio_prompt:
        video_out = output_path + ".temp_video.mp4"
        audio_out = output_path + ".temp_audio.wav"
        for temp_file in [video_out, audio_out, audio_out + ".tmp.wav"]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
    
    clear_gpu_memory()
    pipe = None
    cuda_was_enabled = None
    
    try:
        from diffusers import (
            DiffusionPipeline, 
            DPMSolverMultistepScheduler, 
            CogVideoXImageToVideoPipeline,
            StableVideoDiffusionPipeline,
            WanPipeline,
            WanImageToVideoPipeline,
            HunyuanVideoPipeline,
            HunyuanVideoImageToVideoPipeline,
        )
        from diffusers.utils import export_to_video, load_image
        
        device, dtype = get_optimal_device_and_dtype(quiet=True)
        
        # MPS FIX: These models need Float32/CPU on MPS
        mps_incompatible_models = ["ms-1.7b", "text-to-video-ms-1.7b", "zeroscope", 
                                   "stable-video-diffusion", "cogvideox"]
        is_mps_incompatible = any(m in model_id.lower() for m in mps_incompatible_models)
        
        if device.type == "mps" and is_mps_incompatible:
            if "stable-video-diffusion" in model_id.lower():
                print("⚠️  MPS Compatibility: SVD requires CPU on Apple Silicon.")
                print("   (3D convolutions cause 'Invalid buffer size' on MPS)")
                device = torch.device("cpu")
            else:
                print("⚠️  MPS Compatibility: Using Float32 for correct video output.")
            dtype = torch.float32
        
        print("⚠️  Video generation is resource intensive.")
        
        # cuDNN workaround
        if torch.cuda.is_available():
            cuda_was_enabled = torch.backends.cudnn.enabled
            torch.backends.cudnn.enabled = False
        
        # --- Stage 1: Video Generation ---
        
        # Load Pipeline based on model
        if "cogvideox" in model_id.lower() and is_i2v:
            pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
            print(f"   ℹ️  Applying Memory Optimizations for CogVideoX...")
            pipe.enable_sequential_cpu_offload() 
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()

        elif "wan2.2" in model_id.lower():
            if is_i2v:
                print(f"   ℹ️  Loading Wan 2.2 Image-to-Video Pipeline...")
                pipe = WanImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
            else:
                print(f"   ℹ️  Loading Wan 2.2 Text-to-Video Pipeline...")
                pipe = WanPipeline.from_pretrained(model_id, torch_dtype=dtype)
            
            if device.type == "mps":
                print("   ℹ️  MPS: Enabling Sequential CPU Offload for Wan 2.2 (memory-safe)...")
                pipe.enable_sequential_cpu_offload()
            else:
                print("   ℹ️  Enabling Model CPU Offload for Wan 2.2...")
                pipe.enable_model_cpu_offload()
            pipe.vae.enable_tiling()
            
        elif "ltx-video" in model_id.lower():
            from diffusers import LTXPipeline
            print(f"   ℹ️  Loading LTX-Video Pipeline...")
            pipe = LTXPipeline.from_pretrained(model_id, torch_dtype=dtype)
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_tiling()

        elif "mochi-1" in model_id.lower():
            from diffusers import MochiPipeline
            print(f"   ℹ️  Loading Mochi 1 Pipeline...")
            pipe = MochiPipeline.from_pretrained(model_id, torch_dtype=dtype)
            
            if device.type == "mps":
                print("   ℹ️  MPS: Enabling Sequential CPU Offload for Mochi 1 (memory-safe)...")
                pipe.enable_sequential_cpu_offload()
            else:
                print("   ℹ️  Enabling Model CPU Offload for Mochi 1...")
                pipe.enable_model_cpu_offload()
            pipe.vae.enable_tiling()

        elif "hunyuan" in model_id.lower():
            if is_i2v:
                print(f"   ℹ️  Loading HunyuanVideo Image-to-Video Pipeline...")
                pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
            else:
                print(f"   ℹ️  Loading HunyuanVideo Text-to-Video Pipeline...")
                pipe = HunyuanVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
            
            if device.type == "mps":
                print("   ℹ️  MPS: Enabling Sequential CPU Offload for HunyuanVideo (memory-safe)...")
                pipe.enable_sequential_cpu_offload()
                try:
                    pipe.vae.enable_tiling()
                    pipe.vae.enable_slicing()
                    print("   ℹ️  Enabled VAE Tiling & Slicing for HunyuanVideo")
                except Exception as e:
                    print(f"   ⚠️  Could not enable VAE optimizations: {e}")
            else:
                print("   ℹ️  Enabling Model CPU Offload for HunyuanVideo...")
                pipe.enable_model_cpu_offload()
                pipe.vae.enable_tiling()

        elif "stable-video-diffusion" in model_id.lower():
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=dtype, 
                variant="fp16" if dtype == torch.float16 else None
            )
        else:
            # Generic / Text-to-Video
            try:
                if dtype == torch.float16:
                    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, variant="fp16")
                else:
                    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
            except Exception:
                pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
        
        # Scheduler Optimization
        if hasattr(pipe, "scheduler"):
            is_sensitive_scheduler = any(x in model_id.lower() for x in 
                                        ["stable-video-diffusion", "mochi", "ltx", "wan", "hunyuan"])
            if not is_sensitive_scheduler:
                try:
                    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                except:
                    pass 
        
        # Device/Memory Optimization
        special_handling = ["cogvideox", "wan", "ltx", "mochi", "hunyuan"]
        if not any(x in model_id.lower() for x in special_handling):
            if device.type == "cpu":
                pipe.to(device)
            else:
                pipe.enable_model_cpu_offload()
                if device.type == "mps":
                    pipe.enable_attention_slicing()
        
        # Generate Frames
        tracker = PerformanceTracker()
        
        render_width, render_height = gen_width, gen_height
        
        # Fix dimensions for specific models
        if "ltx-video" in model_id.lower():
            render_width = (render_width // 32) * 32
            render_height = (render_height // 32) * 32
        elif "mochi" in model_id.lower():
            render_width = (render_width // 16) * 16
            render_height = (render_height // 16) * 16
        
        print(f"🎬 Rendering video frames at {render_width}x{render_height}... (This might be slow)")
        
        start_time = time.time()
        with ResourceMonitor() as monitor:
            if is_i2v:
                init_image = load_image(image_input)
                init_image = init_image.resize((gen_width, gen_height))
                
                if "stable-video-diffusion" in model_id.lower():
                    video_frames = pipe(init_image).frames[0]
                elif "wan2.2" in model_id.lower():
                    video_frames = pipe(prompt=prompt, image=init_image, 
                                       num_frames=81, num_inference_steps=50).frames[0]
                elif "hunyuan" in model_id.lower():
                    video_frames = pipe(prompt=prompt, image=init_image, 
                                       num_frames=61, num_inference_steps=50).frames[0]
                else:
                    video_frames = pipe(prompt=prompt, image=init_image, 
                                       num_frames=49, guidance_scale=6.0, 
                                       num_inference_steps=50).frames[0]
            else:
                num_frames = int(duration * 16)
                video_frames = pipe(prompt, num_inference_steps=25, num_frames=num_frames).frames[0]
        
        gen_duration = time.time() - start_time
        avg_cpu, avg_ram, avg_vram, avg_gpu = monitor.get_averages()
        tracker.record_linear("video", model_id, device, duration, gen_duration, 
                             gen_width, gen_height, cpu=avg_cpu, ram=avg_ram, 
                             vram=avg_vram, gpu=avg_gpu)
        print(f"   ✓ Rendered in {format_time(gen_duration)} (RAM: {avg_ram:.1f}GB | "
              f"VRAM: {avg_vram:.1f}GB | CPU: {avg_cpu:.1f}% | GPU: {avg_gpu:.1f}%)")
        
        # --- Zeroscope Dynamic Upscaling Pipeline ---
        if needs_xl_upscale:
            del pipe
            pipe = None
            clear_gpu_memory()
            
            print(f"\n{'='*60}")
            print(f"📐 Step 2: XL Upscale to {zeroscope_xl_w}x{zeroscope_xl_h} (V2V diffusion)")
            print(f"{'='*60}")
            
            upscaled_frames = upscale_video_zeroscope_xl(
                video_frames, 
                prompt, 
                device=device, 
                dtype=dtype,
                strength=0.6
            )
            
            if upscaled_frames is not None:
                video_frames = upscaled_frames
                gen_width, gen_height = zeroscope_xl_w, zeroscope_xl_h
            else:
                print("   ⚠️  XL upscale failed, continuing with base resolution")
                needs_esrgan_upscale = False
        
        if report_json:
            stats = {
                "time": gen_duration,
                "ram": avg_ram,
                "vram": avg_vram,
                "cpu": avg_cpu,
                "gpu": avg_gpu,
                "width": target_width,
                "height": target_height
            }
            write_report_json(report_json, stats)
        
        # Save Video
        temp_raw_video = video_out + ".raw.mp4"
        export_to_video(video_frames, temp_raw_video, 
                       fps=7 if "stable-video-diffusion" in model_id.lower() else 8)
        
        # Re-encode with FFmpeg for universal playback
        import subprocess
        encoding_params = get_video_encoding_params(video_out)
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", temp_raw_video,
                *encoding_params,
                video_out
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(temp_raw_video)
            print(f"✅ Video saved to {video_out} ({gen_width}x{gen_height})")
        except Exception:
            os.rename(temp_raw_video, video_out)
            print(f"⚠️  Video saved (may require VLC to play): {video_out}")
        
        # Real-ESRGAN upscale for target > 1024x576
        if needs_esrgan_upscale:
            print(f"\n{'='*60}")
            print(f"📐 Step 3: Real-ESRGAN Upscale to ~{target_width}x{target_height}")
            print(f"{'='*60}")
            
            # Import here to avoid circular import
            from ..upscaling import upscale_video_fast
            
            esrgan_factor = max(target_width / zeroscope_xl_w, target_height / zeroscope_xl_h)
            esrgan_factor = min(esrgan_factor, 4.0)
            
            temp_esrgan_input = video_out
            temp_esrgan_output = video_out + ".esrgan.mp4"
            
            esrgan_success = upscale_video_fast(
                temp_esrgan_input, 
                temp_esrgan_output, 
                factor=esrgan_factor
            )
            
            if esrgan_success and os.path.exists(temp_esrgan_output):
                import cv2
                cap = cv2.VideoCapture(temp_esrgan_output)
                esrgan_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                esrgan_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                if esrgan_w != target_width or esrgan_h != target_height:
                    print(f"\n{'='*60}")
                    print(f"📐 Step 4: FFmpeg resize to exact {target_width}x{target_height}")
                    print(f"{'='*60}")
                    
                    temp_final = video_out + ".final.mp4"
                    if ffmpeg_resize_video(temp_esrgan_output, temp_final, target_width, target_height):
                        os.remove(temp_esrgan_output)
                        os.remove(video_out)
                        os.rename(temp_final, video_out)
                        print(f"✅ Final video: {video_out} ({target_width}x{target_height})")
                    else:
                        os.remove(video_out)
                        os.rename(temp_esrgan_output, video_out)
                        print(f"✅ Video saved: {video_out} ({esrgan_w}x{esrgan_h})")
                else:
                    os.remove(video_out)
                    os.rename(temp_esrgan_output, video_out)
                    print(f"✅ Final video: {video_out} ({target_width}x{target_height})")
            else:
                print(f"   ⚠️  Real-ESRGAN failed, keeping XL output at {gen_width}x{gen_height}")
        
        # --- Stage 2 & 3: Audio Generation & Muxing ---
        del pipe
        pipe = None
        clear_gpu_memory()
        
        if audio_prompt:
            print("🔊 Generating Audio track...")
            audio_out = output_path + ".temp_audio.wav"
            
            # Import audio generator
            from .audio import generate_audio as gen_audio
            audio_success = gen_audio(audio_prompt, audio_out, duration, 32000, model_name=audio_model)
            
            if audio_success:
                print("🔗 Muxing Video and Audio...")
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
                    for temp_file in [video_out, audio_out]:
                        if os.path.exists(temp_file):
                            try:
                                os.remove(temp_file)
                            except:
                                pass
                except subprocess.CalledProcessError:
                    print(f"❌ Muxing failed. Check FFmpeg.")
            else:
                print("❌ Audio generation failed. Returning silent video (renaming temp).")
                os.rename(video_out, output_path)
                if os.path.exists(audio_out):
                    try:
                        os.remove(audio_out)
                    except:
                        pass
                
        return True
        
    except Exception as e:
        print(f"❌ Video generation failed: {e}")
        if audio_prompt and os.path.exists(video_out):
            try:
                os.remove(video_out)
            except:
                pass
        return False
        
    finally:
        if pipe is not None:
            del pipe
        
        if cuda_was_enabled is not None:
            torch.backends.cudnn.enabled = cuda_was_enabled
             
        clear_gpu_memory()
