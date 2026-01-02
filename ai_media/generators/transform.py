"""
Image transformation module for AI-Media.

Supports: InstructPix2Pix editing and RMBG background removal.
"""

import PIL.Image
import PIL.ImageOps

from ..models import EDIT_MODELS, get_model_id
from ..utils.system import get_optimal_device_and_dtype


def generate_edit(input_path, prompt, output_path, model_name="default", 
                  guidance_scale=7.5, image_guidance_scale=1.5, steps=50, unsafe=False, progress_callback=None):
    """
    Edit an existing image based on instructions using InstructPix2Pix.
    Args:
        input_path: Path to source image
        prompt: Edit instruction (e.g., "make it a watercolor painting")
        output_path: Path to save edited image
        model_name: 'instruct-pix2pix' or 'instruct-pix2pix-sdxl'
        guidance_scale: Text guidance strength
        image_guidance_scale: Image guidance strength
        steps: Number of inference steps
        unsafe: Disable NSFW safety checker
        
    Returns:
        True on success, False on failure
    """
    import time

    class GlobalProgressTracker:
        def __init__(self, total_steps, start_time=None):
            self.total_steps = total_steps
            self.current_step = 0
            self.start_time = start_time or time.time()
            self.last_update_ts = 0
            
        def update(self, step_increment=1, model_desc=""):
            self.current_step += step_increment
            
            # Throttle updates to ~2 times per second to avoid flooding logs
            now = time.time()
            if now - self.last_update_ts < 0.5 and self.current_step < self.total_steps:
                return None
            self.last_update_ts = now
            
            elapsed = now - self.start_time
            if self.current_step > 0:
                avg_time_per_step = elapsed / self.current_step
                remaining_steps = self.total_steps - self.current_step
                eta_seconds = int(remaining_steps * avg_time_per_step)
                eta_str = f"{eta_seconds//60}m {eta_seconds%60}s" if eta_seconds > 60 else f"{eta_seconds}s"
            else:
                eta_str = "Calculating..."
                
            percent = min(99, int((self.current_step / self.total_steps) * 100))
            return percent, f"{model_desc}: {percent}% | Step {self.current_step}/{self.total_steps} | ETA: {eta_str}"
    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionXLInstructPix2PixPipeline
    from diffusers.utils import load_image
    
    # Resolve Model ID
    model_id = get_model_id(model_name, EDIT_MODELS)
    
    print(f"🎨 Editing Image")
    print(f"   Model:     {model_id}")
    print(f"   Input:     {input_path}")
    print(f"   Instruct:  '{prompt}'")
    print(f"   Output:    {output_path}")
    print("") 

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
        if "qwen-image-edit" in model_name.lower():
            # ... (unchanged setup code) ...
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


def remove_background(input_path, output_path, model_name="remove-bg", silhouette=False):
    """
    Remove background using RMBG-1.4 (Transformer based).
    
    Args:
        input_path: Path to source image
        output_path: Path to save result (PNG with transparency)
        model_name: Model short code (default: 'remove-bg')
        silhouette: If True, create black silhouette instead of transparent
        
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
        print(f"✅ Saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Background removal failed: {e}")
        return False
