"""
Image generation module for AI-Media.

Supports: Flux, SDXL Turbo, Stable Diffusion 1.5, and other text-to-image models.
"""

import os
import sys
import time
import warnings

from ..models import IMAGE_MODELS, MODEL_REQUIREMENTS, get_model_id
from ..utils.system import get_optimal_device_and_dtype, clear_gpu_memory, check_resources_and_warn
from ..utils.parsers import format_time
from ..utils.performance import PerformanceTracker, ResourceMonitor, write_report_json


def generate_image(prompt, output_file, width, height, model_name="default", steps=30, 
                   guidance_scale=7.5, unsafe=False, report_json=None, force=False):
    """Generate image using Diffusers (Flux/SDXL).
    
    Args:
        prompt: Text description of desired image
        output_file: Path to save generated image
        width: Image width in pixels
        height: Image height in pixels
        model_name: Model short code or HF ID (default: 'sdxl')
        steps: Number of inference steps
        guidance_scale: Classifier-free guidance scale
        unsafe: Disable NSFW safety checker
        report_json: Path to write performance stats JSON
        force: Skip resource warnings
        
    Returns:
        True on success, False on failure
    """
    # Resolve Model ID
    model_id = get_model_id(model_name, IMAGE_MODELS)
    
    # Pre-calculate Device and Estimate Resources
    try:
        import torch
        from diffusers import FluxPipeline, AutoPipelineForText2Image
        
        # Determine device and dtype
        device, dtype = get_optimal_device_and_dtype(quiet=True, prefer_bfloat16=True)
        dtype_name = str(dtype).replace("torch.", "")
        
        # Estimate Performance
        tracker = PerformanceTracker()
        est_values = tracker.estimate_image(model_id, width, height, device, dtype=dtype_name)
        
        # Display Info Header
        print(f"Platform: {device.type.upper()} | Dtype: {dtype_name}")
        tracker.print_estimate(*est_values)
        
    except ImportError:
        print("❌ Failed to import torch/diffusers. Please check installation.")
        return False

    print(f"🎨 Generating Image")
    print(f"   Model:  {model_id}")
    print(f"   Prompt: '{prompt}'")
    print(f"   Size:   {width}x{height}")
    print(f"   Output: {output_file}")
    print("")  # Spacer
    
    # Check resources
    if not check_resources_and_warn(model_id, width=width, height=height, force=force, 
                                     model_requirements=MODEL_REQUIREMENTS):
        return False
    
    try:
        # Determine Pipeline Class based on model
        use_offload = False
        if "flux.2" in model_id.lower() or "flux2" in model_id.lower():
            # FLUX.2 uses Flux2Pipeline (different architecture from FLUX 1)
            from diffusers import Flux2Pipeline
            
            # Check if using 4-bit quantized model (requires bitsandbytes, CUDA only)
            is_quantized = "bnb" in model_id.lower() or "4bit" in model_id.lower()
            
            if is_quantized and device.type != "cuda":
                # 4-bit quantized models require bitsandbytes which only works on CUDA
                print(f"   ⚠️  4-bit quantized FLUX.2 requires CUDA and bitsandbytes.")
                print(f"   ⚠️  On Mac/MPS, use 'flux2-full' with 64GB+ RAM for CPU offloading.")
                print(f"   ⚠️  Attempting to load with CPU offloading (this will be slow)...")
                
                # Fall back to full model with offloading
                model_id = "black-forest-labs/FLUX.2-dev"
                flux2_dtype = torch.float32
                use_offload = False
            else:
                # CUDA: use bfloat16 for quantized models
                # Enable offloading even on CUDA because Flux 2 (32B) + T5 (4B+) > 24GB VRAM
                if device.type == "cuda":
                    flux2_dtype = torch.bfloat16
                    use_offload = True
                else:
                    flux2_dtype = torch.float32 
                    use_offload = False
            
            if is_quantized and device.type == "cuda":
                print(f"   ℹ️  Loading FLUX.2 Pipeline (4-bit quantized for consumer GPUs)...")
            else:
                print(f"   ℹ️  Loading FLUX.2 Pipeline with CPU offloading (requires ~64GB+ RAM)...")
            
            pipe = Flux2Pipeline.from_pretrained(
                model_id, 
                torch_dtype=flux2_dtype
            )
            
            # FLUX.2 parameters (higher quality, more steps than FLUX 1)
            extra_kwargs = {
                "guidance_scale": 4.0, 
                "num_inference_steps": 50,
            }
        elif "stable-diffusion-3.5" in model_id.lower() or "sd3.5" in model_name.lower():
            # SD 3.5 (Medium, Large, Large Turbo) uses StableDiffusion3Pipeline
            from diffusers import StableDiffusion3Pipeline
            
            # SD 3.5 works best with bfloat16 on CUDA, float32 on MPS
            sd35_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            
            print(f"   ℹ️  Loading Stable Diffusion 3.5 Pipeline...")
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                torch_dtype=sd35_dtype
            )
            
            # Enable CPU offload on CUDA to fit on 24GB cards
            if device.type == "cuda":
                use_offload = True
            
            # Turbo uses only 4 steps with zero guidance, Medium/Large use 40 steps with guidance
            is_turbo = "turbo" in model_id.lower()
            extra_kwargs = {
                "guidance_scale": 0.0 if is_turbo else 4.5,
                "num_inference_steps": 4 if is_turbo else 40,
                "max_sequence_length": 512
            }
        elif "qwen-image" in model_name.lower() and "edit" not in model_name.lower():
            # Qwen-Image Text-to-Image generation
            from diffusers import DiffusionPipeline
            
            # Auto-switch: CUDA model on MPS → switch to MPS model, and vice versa
            original_model_name = model_name
            if device.type == "mps" and "-mps" not in model_name.lower():
                print(f"   ℹ️  Switching to qwen-image-mps (4-bit quantization not supported on MPS)")
                model_id = IMAGE_MODELS["qwen-image-mps"]
                model_name = "qwen-image-mps"
            elif device.type == "cuda" and "-mps" in model_name.lower():
                print(f"   ℹ️  Switching to qwen-image (using optimized CUDA 4-bit variant)")
                model_id = IMAGE_MODELS["qwen-image"]
                model_name = "qwen-image"
            
            # CUDA: bfloat16, MPS: float32
            qwen_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            
            print(f"   ℹ️  Loading Qwen-Image Pipeline...")
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=qwen_dtype
            )
            
            # Enable CPU offload on CUDA for memory efficiency
            if device.type == "cuda":
                use_offload = True
            
            # Qwen-Image parameters: Distill uses 15 steps, 4-bit uses ~8 steps
            is_distill = "distill" in model_id.lower()
            extra_kwargs = {
                "true_cfg_scale": 4.0,
                "num_inference_steps": 15 if is_distill else 8,
            }
        elif "flux" in model_id.lower():
            # FLUX 1 (Schnell/Dev) on MPS requires float32 to avoid dtype mismatch errors
            flux_dtype = torch.float32 if device.type == "mps" else dtype
            pipe = FluxPipeline.from_pretrained(
                model_id, 
                torch_dtype=flux_dtype
            )
            
            if device.type == "cuda":
                use_offload = True
                
            # Flux 1 parameters
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
            extra_kwargs = {}  # Use defaults
            
        # Apply memory optimizations if requested
        if use_offload:
            print(f"   ℹ️  Enabling CPU offloading for memory efficiency...")
            if hasattr(pipe, 'enable_model_cpu_offload'):
                pipe.enable_model_cpu_offload()
            elif hasattr(pipe, 'enable_sequential_cpu_offload'):
                pipe.enable_sequential_cpu_offload()

        if not use_offload:
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
        start_time = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, 
                                    message="invalid value encountered in cast")
            
            # Start Resource Monitoring
            with ResourceMonitor() as monitor:
                output = pipe(
                    prompt=prompt, 
                    height=height, 
                    width=width,
                    **extra_kwargs
                )
            
            # Collect metrics
            duration = time.time() - start_time
            avg_cpu, avg_ram, avg_vram, avg_gpu = monitor.get_averages()
            
            # Record Performance
            tracker.record_image(model_id, width, height, device, duration, 
                                cpu=avg_cpu, ram=avg_ram, vram=avg_vram, gpu=avg_gpu, dtype=dtype_name)
            print(f"   ✓ Generated in {format_time(duration)} (RAM: {avg_ram:.1f}GB | "
                  f"VRAM: {avg_vram:.1f}GB | CPU: {avg_cpu:.1f}% | GPU: {avg_gpu:.1f}%)")
            
            # Write JSON report if requested
            if report_json:
                stats = {
                    "time": duration,
                    "ram": avg_ram,
                    "vram": avg_vram,
                    "cpu": avg_cpu,
                    "gpu": avg_gpu,
                    "width": width,
                    "height": height
                }
                write_report_json(report_json, stats)
            
        image = output.images[0]
        
        # Check for NSFW content interception
        if hasattr(output, "nsfw_content_detected") and output.nsfw_content_detected:
            if output.nsfw_content_detected[0]:
                print(f"⚠️  Warning: Potential NSFW content detected.\n")
                print(f"The model's safety checker has blocked the image (returning a black frame).")
                print(f"👉 Please modify your prompt and try again.")
                print(f"💡 If your prompt is appropriate, try again with --unsafe to disable the safety checker.\n")
        
        image.save(output_file)
        print(f"✅ Image saved to {output_file}")
        
        tracker.print_actual(duration, avg_cpu, avg_ram, avg_vram, avg_gpu)
        print("")  # Spacer
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
                    print("")  # Spacer
                    return generate_image(prompt, output_file, new_w, new_h, 
                                         model_name=model_name, unsafe=unsafe)
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
            
            # Auto-Upscale Fallback
            try:
                print(f"   ✨ Alternative: Generate at 1280x720 and Auto-Upscale x4?")
                print(f"      This produces a 5120x2880 (5K) image using the Upscaler model.")
                choice = input(f"   🔄 Try Auto-Upscale workflow? [y/N]: ").lower().strip()
                if choice in ['y', 'yes']:
                    print("\n📉 Switching to base resolution: 1280x720...")
                    # Import upscaler here to avoid circular import
                    from ..upscaling import upscale_image_file
                    # 1. Generate Base Image
                    success = generate_image(prompt, output_file, 1280, 720, 
                                            model_name=model_name, unsafe=unsafe)
                    if success:
                        # 2. Upscale Result
                        print("")
                        return upscale_image_file(output_file, output_file, strength=0.0, factor=4.0)
            except KeyboardInterrupt:
                pass
            print("")
        else:
            print(f"❌ Generation failed: {e}")
        return False
