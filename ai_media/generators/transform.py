"""
Image transformation module for AI-Media.

Supports: InstructPix2Pix editing and RMBG background removal.
"""

import PIL.Image
import PIL.ImageOps

from ..models import EDIT_MODELS, get_model_id
from ..utils.system import get_optimal_device_and_dtype, check_resources_and_warn


def generate_edit(input_path, prompt, output_path, model_name="default", 
                  guidance_scale=7.5, image_guidance_scale=1.5, steps=50, unsafe=False, 
                  force=False, bypass_warning=False, progress_callback=None,
                  use_mlx=None, precision=None):
    """
    Edit an existing image based on instructions.
    Args:
        input_path: Path to source image
        prompt: Edit instruction (e.g., "make it a watercolor painting")
        output_path: Path to save edited image
        model_name: 'instruct-pix2pix', 'qwen-image-edit', or 'z-image-edit'
        guidance_scale: Text guidance strength
        image_guidance_scale: Image guidance strength
        steps: Number of inference steps
        unsafe: Disable NSFW safety checker
        force: Skip confirmation prompts (overwrites and warnings)
        bypass_warning: Specifically skip resource warning prompts
        use_mlx: Force usage of MLX backend (macOS only)
        precision: Force specific precision/quantization (e.g., 'int4', 'float16')
        
    Returns:
        True on success, False on failure
    """
    import os
    import sys
    import time
    import numpy as np
    from PIL import Image

    # Determine framework
    is_mac = sys.platform == "darwin"
    if use_mlx is None:
        use_mlx = is_mac  # Default to MLX on Mac for Z-Image/Flux if supported
    
    # Resolve Model ID
    model_id = get_model_id(model_name, EDIT_MODELS)
    
    # MLX Branch
    if use_mlx and is_mac and ("z-image" in model_name.lower() or "zimage" in model_name.lower()):
        try:
            from mflux.models.z_image.variants.turbo.z_image_turbo import ZImageTurbo
            # from mflux.utils.optimal_device import get_optimal_device # Not needed/doesn't exist
            from ..models import get_mlx_model_id
            
            print(f"🍎 Using MLX Backend for Z-Image Edit")
            
            # Setup precision
            if precision is None:
                precision = "int4"  # Default for MLX
                
            quantize = None
            if precision == "int4": quantize = 4
            elif precision == "int8": quantize = 8
            
            # Resolve ID (e.g. use filipstrand repo for 4-bit)
            mlx_model_id = get_mlx_model_id(model_id, precision)
            
            # Load Model
            print(f"⏳ Loading Z-Image Turbo (MLX)...")
            mlx_model = ZImageTurbo(model_path=mlx_model_id, quantize=quantize)
            
            # Load and Resize Image
            img = Image.open(input_path).convert("RGB")
            # MLX models often prefer multiples of 16
            w, h = img.size
            new_w, new_h = (w // 16) * 16, (h // 16) * 16
            if (new_w, new_h) != (w, h):
                print(f"   ℹ️  Resizing input to {new_w}x{new_h} (multiple of 16)")
                img = img.resize((new_w, new_h), Image.LANCZOS)

            # Patch Tqdm for progress
            from tqdm import tqdm as real_tqdm
            outer_progress_callback = progress_callback
            
            # Re-use the TqdmWrapper logic from image.py if possible, but here we define it inline for simplicity
            class TqdmWrapper:
                def __init__(self, iterable=None, desc=None, total=None, *args, **kwargs):
                    self.iterable = iterable
                    self.n = 0
                    self.total = total or (len(iterable) if iterable else None)
                    self._start_time = time.time()
                    self._tqdm = real_tqdm(iterable, desc=desc, total=total, *args, **kwargs)
                def _report(self):
                    if outer_progress_callback and self.total:
                        pct = min(100, int((self.n / self.total) * 100))
                        elapsed = time.time() - self._start_time
                        eta_str = ""
                        if self.n > 0:
                            eta_secs = int((self.total - self.n) * (elapsed / self.n))
                            eta_str = f", ETA: {eta_secs//60:02d}:{eta_secs%60:02d}"
                        outer_progress_callback(pct, f"Editing: {pct}%{eta_str}")
                def update(self, n=1):
                    self.n += n
                    self._tqdm.update(n)
                    self._report()
                def __iter__(self):
                    for item in self._tqdm:
                        yield item
                        self.n += 1
                        self._report()
                def __enter__(self): return self
                def __exit__(self, *args): self._tqdm.close()
                def close(self): self._tqdm.close()
                def __getattr__(self, name): return getattr(self._tqdm, name)

            # Apply patch
            import mflux.models.z_image.variants.turbo.z_image_turbo
            mflux.models.z_image.variants.turbo.z_image_turbo.tqdm = TqdmWrapper

            # Generate (Image-to-Image / Editing)
            # mflux ZImageTurbo uses img2img if image_path is provided
            print(f"✨ Applying edits with Z-Image (MLX)...")
            zimage_steps = steps if steps != 50 else 9
            
            output_image = mlx_model.generate_image(
                prompt=prompt,
                image_path=input_path, # Pass path directly
                num_inference_steps=zimage_steps,
                seed=int(time.time() % 1000000)
            )
            
            output_image.image.save(output_path)
            print(f"✅ Edited image saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ MLX Z-Image Edit failed: {e}")
            print(f"   Falling back to PyTorch...")
            use_mlx = False
    elif use_mlx and is_mac:
        print(f"⚠️  Model '{model_name}' not supported on MLX (or not implemented). Falling back to PyTorch.")

    # PyTorch Branch
    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionXLInstructPix2PixPipeline
    from diffusers.utils import load_image
    
    # Helper for progress tracking
    class GlobalProgressTracker:
        def __init__(self, total_steps, start_time=None):
            self.total_steps = total_steps
            self.current_step = 0
            self.start_time = start_time or time.time()
            
        def update(self, n=1, model_desc="Processing"):
            self.current_step += n
            percent = min(100, int((self.current_step / self.total_steps) * 100))
            
            # Simple ETA
            elapsed = time.time() - self.start_time
            if self.current_step > 0:
                avg_time = elapsed / self.current_step
                remaining = max(0, self.total_steps - self.current_step)
                eta_secs = int(remaining * avg_time)
                mins, secs = divmod(eta_secs, 60)
                msg = f"{model_desc}: {percent}%, ETA: {mins:02d}:{secs:02d}"
                return percent, msg
            return percent, f"{model_desc}: {percent}%"
    
    print(f"🎨 Editing Image")
    print(f"   Model:     {model_id}")
    print(f"   Input:     {input_path}")
    print(f"   Instruct:  '{prompt}'")
    print(f"   Output:    {output_path}")
    print("") 
    
    # Check resources
    from ..models import MODEL_REQUIREMENTS
    if not check_resources_and_warn(model_id, force=force, bypass_warning=bypass_warning, 
                                     model_requirements=MODEL_REQUIREMENTS):
        return False

    try:
        device, dtype = get_optimal_device_and_dtype(quiet=True, prefer_bfloat16=True)
        
        # CRITICAL FIX: InstructPix2Pix (SD1.5 based) often produces black images on MPS with float16.
        # We force float32 for this specific pipeline on MPS to ensure valid output.
        if device.type == "mps":
            print(f"   ℹ️  MPS Detected: Forcing float32 for InstructPix2Pix to prevent black images.")
            dtype = torch.float32
        
        # Display platform and dtype info
        dtype_name = str(dtype).replace("torch.", "")
        print(f"   Platform: {device.type.upper()} | Dtype: {dtype_name}")

        # Load Input Image
        image = load_image(input_path)
        image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        
        # Initialize progress tracker
        global_tracker = GlobalProgressTracker(steps, start_time=time.time())

        # Define callback for Diffusers
        def diffusers_callback(step: int, timestep: int, latents: torch.FloatTensor):
            progress_data = global_tracker.update(1, model_desc="Applying edits")
            if progress_data and progress_callback:
                pct, msg = progress_data
                progress_callback(pct, msg)

        # Initialize Pipeline
        if "z-image-edit" in model_name.lower():
            # ----------------------------------------------------------------
            # Z-Image Turbo (PyTorch) - Dedicated Img2Img Pipeline
            # ----------------------------------------------------------------
            from diffusers import ZImageImg2ImgPipeline
            zimage_dtype = torch.bfloat16 if device.type == "cuda" else torch.float16
            
            print(f"   ℹ️  Loading Z-Image Turbo Img2Img Pipeline (PyTorch)...")
            if progress_callback: progress_callback(0, "Loading Z-Image Img2Img Pipeline...")
            
            pipe = ZImageImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=zimage_dtype
            )
            
            if device.type == "cuda" or device.type == "mps":
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to(device)
            
            # Z-Image optimal steps
            zimage_steps = steps if steps != 50 else 9
            global_tracker = GlobalProgressTracker(zimage_steps, start_time=time.time())
            
            start_msg = f"✨ Applying edits with Z-Image... (Steps: {zimage_steps})"
            print(start_msg)
            if progress_callback: progress_callback(0, start_msg)
            
            with torch.inference_mode():
                output = pipe(
                    prompt=prompt,
                    image=image,
                    num_inference_steps=zimage_steps,
                    guidance_scale=guidance_scale if guidance_scale != 7.5 else 3.5,
                    callback=diffusers_callback,
                    callback_steps=1
                )
            
            result = output.images[0]
            result.save(output_path)
            print(f"✅ Edited image saved to {output_path}")
            return True

        elif "qwen-image-edit-lightning" in model_name.lower():
            # ----------------------------------------------------------------
            # Qwen-Image-Edit-2512-Lightning (LoRA-based 4-step model)
            # This is a distilled LoRA model that loads on top of base 2511
            # Requires: pip install peft
            # ----------------------------------------------------------------
            from diffusers import DiffusionPipeline
            
            # Base model is the official 2511
            base_model_id = "Qwen/Qwen-Image-Edit-2511"
            lora_repo_id = model_id  # lightx2v/Qwen-Image-Edit-2512-Lightning
            
            qwen_dtype = torch.bfloat16 if device.type == "cuda" else torch.float16
            
            print(f"   ℹ️  Loading Qwen-Image-Edit Lightning (4-step LoRA)...")
            if progress_callback: progress_callback(0, "Loading Qwen-Image-Edit Lightning...")
            
            # Load base model first
            pipe = DiffusionPipeline.from_pretrained(
                base_model_id,
                torch_dtype=qwen_dtype
            )
            
            # Load LoRA weights from the Lightning repo
            lora_loaded = False
            try:
                pipe.load_lora_weights(lora_repo_id)
                print(f"   ✅ Loaded LoRA weights from {lora_repo_id}")
                lora_loaded = True
            except ImportError:
                print(f"   ⚠️  PEFT not installed. Run: pip install peft")
                print(f"   ℹ️  Falling back to base Qwen-Image-Edit model (20 steps)")
            except Exception as e:
                print(f"   ⚠️  Could not load LoRA weights: {e}")
                print(f"   ℹ️  Falling back to base Qwen-Image-Edit model (20 steps)")
            
            # Enable CPU offload
            if device.type == "cuda" or device.type == "mps":
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to(device)
            
            # Lightning model is optimized for 4 steps, base uses 20
            lightning_steps = 4 if lora_loaded else 20
            global_tracker = GlobalProgressTracker(lightning_steps, start_time=time.time())
            
            model_label = "Qwen-Edit-Lightning" if lora_loaded else "Qwen-Image-Edit (fallback)"
            start_msg = f"✨ Applying edits with {model_label}... (Steps: {lightning_steps})"
            print(start_msg)
            if progress_callback: progress_callback(0, start_msg)
            
            # Note: QwenImageEditPlusPipeline doesn't support callback parameter
            with torch.inference_mode():
                output = pipe(
                    prompt=prompt,
                    image=image,
                    num_inference_steps=lightning_steps,
                    guidance_scale=guidance_scale if guidance_scale != 7.5 else (2.0 if lora_loaded else 4.0),
                )
            
            result = output.images[0]
            result.save(output_path)
            print(f"✅ Edited image saved to {output_path}")
            return True
            
        elif "qwen-image-edit" in model_name.lower():
            # ----------------------------------------------------------------
            # Standard Qwen-Image-Edit (2511 base model)
            # ----------------------------------------------------------------
            from diffusers import DiffusionPipeline
            
            # Auto-switch: CUDA model on MPS → switch to MPS model, and vice versa
            if device.type == "mps" and "-mps" not in model_name.lower():
                print(f"   ℹ️  Switching to qwen-image-edit-mps (4-bit quantization not supported on MPS)")
                model_id = EDIT_MODELS["qwen-image-edit-mps"]
            elif device.type == "cuda" and "-mps" in model_name.lower():
                print(f"   ℹ️  Switching to qwen-image-edit (using optimized CUDA variant)")
                model_id = EDIT_MODELS["qwen-image-edit"]
            
            # CUDA: bfloat16, MPS: try float16 to save RAM
            qwen_dtype = torch.bfloat16 if device.type == "cuda" else torch.float16
            
            print(f"   ℹ️  Loading Qwen-Image-Edit Pipeline...")
            if progress_callback: progress_callback(0, "Loading Qwen-Image-Edit pipeline...")
            
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=qwen_dtype
            )
            
            # Enable CPU offload on CUDA/MPS
            if device.type == "cuda" or device.type == "mps":
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to(device)
            
            # Generate with Qwen-Image-Edit
            qwen_steps = steps if steps != 50 else 20
            # Re-init tracker with correct steps for Qwen default override
            global_tracker = GlobalProgressTracker(qwen_steps, start_time=time.time())
            
            start_msg = f"✨ Applying edits with Qwen-Image-Edit... (Steps: {qwen_steps})"
            print(start_msg)
            if progress_callback: progress_callback(0, start_msg)
            
            with torch.inference_mode():
                output = pipe(
                    prompt=prompt,
                    image=image,
                    num_inference_steps=qwen_steps,
                    guidance_scale=guidance_scale if guidance_scale != 7.5 else 4.0,
                    callback=diffusers_callback,
                    callback_steps=1
                )
            
            result = output.images[0]
            result.save(output_path)
            print(f"✅ Edited image saved to {output_path}")
            return True
            
        elif "sdxl" in model_id.lower():
            # SDXL InstructPix2Pix
            if progress_callback: progress_callback(0, "Loading SDXL InstructPix2Pix pipeline...")
            pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype
            ).to(device)
            # Default scales for SDXL are different
            if guidance_scale == 7.5:
                guidance_scale = 7.0 
            if image_guidance_scale == 1.5:
                image_guidance_scale = 1.25
            
        else:
            # Standard InstructPix2Pix (SD 1.5 based)
            kwargs = {}
            if unsafe:
                kwargs["safety_checker"] = None
                
            if progress_callback: progress_callback(0, "Loading InstructPix2Pix pipeline...")
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                model_id, 
                torch_dtype=dtype,
                **kwargs
            ).to(device)
            
        # Optimization
        if device.type == "mps":
            pipe.enable_attention_slicing()
        
        # Generate
        start_msg = f"✨ Applying edits... (Steps: {steps}, Text Guide: {guidance_scale}, Image Guide: {image_guidance_scale})"
        print(start_msg)
        if progress_callback: progress_callback(0, start_msg)
        
        with torch.inference_mode():
            output = pipe(
                prompt,
                image=image,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                callback=diffusers_callback,
                callback_steps=1
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


def remove_background(input_path, output_path, model_name="remove-bg", silhouette=False, force=False, bypass_warning=False):
    """
    Remove background using RMBG-1.4 (Transformer based).
    
    Args:
        input_path: Path to source image
        output_path: Path to save result (PNG with transparency)
        model_name: Model short code (default: 'remove-bg')
        silhouette: If True, create black silhouette instead of transparent
        force: Skip confirmation prompts (overwrites and warnings)
        bypass_warning: Specifically skip resource warning prompts
        
    Returns:
        True on success, False on failure
    """
    import torch
    import numpy as np
    from transformers import AutoModelForImageSegmentation
    from torchvision.transforms.functional import normalize
    
    print(f"✂️  Removing Background")
    print(f"   Input:  {input_path}")
    print(f"   Output: {output_path}")
    if silhouette:
        print(f"   Mode:   Silhouette Maker")
    print("")

    # Check resources
    from ..models import MODEL_REQUIREMENTS
    if not check_resources_and_warn("remove-bg", force=force, bypass_warning=bypass_warning,
                                     model_requirements=MODEL_REQUIREMENTS):
        return False

    try:
        device, dtype = get_optimal_device_and_dtype(quiet=True, prefer_bfloat16=True)
        
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
            final_image = PIL.Image.new("RGBA", original_size, (255, 255, 255, 0))  # Transparent
            final_image.paste(foreground, (0, 0), mask_pil)
        else:
            final_image = image.copy()
            final_image.putalpha(mask_pil)
            
        final_image.save(output_path, "PNG")
        print(f"✅ Saved to: {os.path.normpath(output_path)}")
        return True
        
    except Exception as e:
        print(f"❌ Background removal failed: {e}")
        return False
