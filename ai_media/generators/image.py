"""
Image generation module for AI-Media.

Supports: Flux, SDXL Turbo, Stable Diffusion 1.5, and other text-to-image models.
"""

import os
import sys
import time
import threading
import warnings
import re
import contextlib
from datetime import datetime

from ..models import IMAGE_MODELS, MODEL_REQUIREMENTS, get_model_id
from ..utils.system import get_optimal_device_and_dtype, clear_gpu_memory, check_resources_and_warn
from ..utils.parsers import format_time
from ..utils.performance import PerformanceTracker, ResourceMonitor, write_report_json


class TqdmCapture:
    """Class to capture tqdm output from stderr and redirect to a callback."""
    def __init__(self, callback):
        self.callback = callback
        
    def write(self, text):
        # Progress bars use \r to overwrite lines. 
        # ALWAYS write original raw text to real stderr so terminal logic (TQDM) works
        sys.__stderr__.write(text)

        # Only process for web callback if there's actual content
        if self.callback and text.strip():
            try:
                # Filter out raw TQDM bar lines as these are messy in the UI
                if "|" in text and "%" in text:
                    return

                # Strip ANSI escape codes ONLY for web logs
                clean_text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
                if clean_text.strip():
                    # Pass terminal=False because we already wrote to sys.__stderr__ above
                    self.callback("loading", 0, clean_text, terminal=False) 
            except ConnectionAbortedError:
                raise # Propagate cancel signal
            except:
                pass

    def flush(self):
        sys.__stderr__.flush()


class ImageGenerator:
    """Class for generating images using Diffusers pipelines."""
    
    def __init__(self, model_id="default"):
        """Initialize the generator.
        
        Args:
            model_id: Model short code or HF ID.
        """
        self.model_name = model_id
        # Resolve Model ID immediately
        self.model_id = get_model_id(model_id, IMAGE_MODELS)
        self.pipe = None
        self.device = None
        self.dtype = None
        self.defaults = {}
        self._cancelled = False
        self._lock = threading.Lock()
        self.progress_callback = None

    def _log_status(self, status, progress, message, terminal=True):
        """Helper to report progress to callback and/or terminal."""
        if self.progress_callback:
            try:
                # status: "loading" | "generating" | "error"
                # progress: 0-100
                
                # Strip terminal-specific instructions for web UI
                clean_ui_msg = message.replace(" (check terminal for progress)", "")
                self.progress_callback(percent=progress, message=clean_ui_msg)
            except Exception as e:
                print(f"⚠️ Progress callback error: {e}")
        
        # Also print to terminal for server logs (with different icon if needed)
        if terminal:
            # Avoid double emojis if message already has one
            if not any(emoji in message for emoji in ["📚", "⏳", "⚠️", "✅", "🛑"]):
                 print(f"⏳ {message}")
            else:
                 print(message)
    def unload(self):
        """Unload the pipeline and free memory."""
        if self.pipe:
            print(f"🧹 Unloading image model: {self.model_name}")
            del self.pipe
            self.pipe = None
            clear_gpu_memory()
            
    def stop(self):
        """Signal generation to stop and unload."""
        self._cancelled = True
        self.unload()
    def _ensure_pipeline_loaded(self):
        """Load the pipeline if not already loaded."""
        if self.pipe:
            return

        import torch
        from diffusers import FluxPipeline, AutoPipelineForText2Image, Flux2Pipeline, StableDiffusion3Pipeline, DiffusionPipeline
        
        # Determine device and dtype
        device, dtype = get_optimal_device_and_dtype(quiet=True, prefer_bfloat16=True)
        self.device = device
        self.dtype = dtype
        
        use_offload = False
        extra_kwargs = {}
        pipe = None
        
        self._log_status("loading", 10, f"Loading image model: {self.model_name}...")
        
        # Capture stderr during loading
        try:
            capture = TqdmCapture(self._log_status)
            with contextlib.redirect_stderr(capture):
                # Monkey patch TQDM to ensure we capture progress bars even if stderr redirect fails
                import tqdm
                original_tqdm = tqdm.tqdm
                
                class TqdmCallbackWrapper(original_tqdm):
                    def update(self, n=1):
                        super().update(n)
                        if hasattr(self, 'total') and self.total:
                            try:
                                percent = int(self.n / self.total * 100)
                                desc = self.desc or "Loading"
                                # Use terminal=False because original TQDM (stderr) already prints the bar
                                capture.callback("loading", percent, f"{desc}: {percent}%", terminal=False)
                            except:
                                pass

                # Apply patch to source tqdm
                tqdm.tqdm = TqdmCallbackWrapper
                
                # CRITICAL: Also patch diffusers/transformers/accelerate references
                original_diffusers_tqdm = None
                try:
                    import diffusers.utils.logging as d_logging
                    if hasattr(d_logging, 'tqdm'):
                        original_diffusers_tqdm = d_logging.tqdm
                        d_logging.tqdm = TqdmCallbackWrapper
                except ImportError:
                    pass

                try:
                    # Determine Pipeline Class based on model
                    if "flux.2" in self.model_id.lower() or "flux2" in self.model_id.lower():
                         # FLUX.2
                         is_quantized = "bnb" in self.model_id.lower() or "4bit" in self.model_id.lower()
                         if is_quantized and device.type != "cuda":
                             self._log_status("loading", 10, "⚠️ 4-bit quantized FLUX.2 requires CUDA. Fallback to full model with offload.")
                             self.model_id = "black-forest-labs/FLUX.2-dev"
                             flux2_dtype = torch.float32
                             use_offload = False
                         else:
                             flux2_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
                             use_offload = True if device.type == "cuda" else False
                         
                         self._log_status("loading", 15, "Loading FLUX.2 Pipeline...")
                         pipe = Flux2Pipeline.from_pretrained(self.model_id, torch_dtype=flux2_dtype)
                         extra_kwargs = {"guidance_scale": 4.0, "num_inference_steps": 50}

                    elif "stable-diffusion-3.5" in self.model_id.lower() or "sd3.5" in self.model_name.lower():
                         # SD 3.5
                         sd35_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
                         self._log_status("loading", 15, "Loading Stable Diffusion 3.5 Pipeline...")
                         pipe = StableDiffusion3Pipeline.from_pretrained(self.model_id, torch_dtype=sd35_dtype)
                         if device.type == "cuda" or device.type == "mps":
                             use_offload = True
                         
                         is_turbo = "turbo" in self.model_id.lower()
                         extra_kwargs = {
                             "guidance_scale": 0.0 if is_turbo else 4.5,
                             "num_inference_steps": 4 if is_turbo else 40,
                             "max_sequence_length": 512
                         }

                    elif "qwen-image" in self.model_name.lower() and "edit" not in self.model_name.lower():
                         # Qwen-Image
                         qwen_dtype = torch.bfloat16 if device.type == "cuda" else torch.float16
                         self._log_status("loading", 15, "Loading Qwen-Image Pipeline...")
                         
                         if "lightning" in self.model_name.lower():
                             # Load Base
                             import warnings
                             with warnings.catch_warnings():
                                 warnings.simplefilter("ignore")
                                 pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=qwen_dtype)
                             
                             # Apply Lightning
                             try:
                                 from huggingface_hub import hf_hub_download
                                 filename = "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"
                                 self._log_status("loading", 20, "Downloading Lightning weights...")
                                 checkpoint_path = hf_hub_download(repo_id=self.model_id, filename=filename)
                                 pipe.load_lora_weights(checkpoint_path)
                                 try: pipe.fuse_lora()
                                 except: pass
                                 
                                 from diffusers import FlowMatchEulerDiscreteScheduler
                                 # Simplified scheduler replacement
                                 scheduler_config = dict(pipe.scheduler.config)
                                 for key in ["mu", "sigma_min", "sigma_max", "sigma_data"]: 
                                    if key in scheduler_config: del scheduler_config[key]
                                 pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
                                 self._log_status("loading", 30, "Switched to FlowMatchEulerDiscreteScheduler")
                             except Exception as e:
                                 self._log_status("loading", 30, f"⚠️ Lightning setup failed: {e}")
                         else:
                             pipe = DiffusionPipeline.from_pretrained(self.model_id, torch_dtype=qwen_dtype)

                         if device.type == "cuda": use_offload = True
                         is_lightning = "lightning" in self.model_name.lower()
                         steps = 4 if is_lightning else 30 
                         extra_kwargs = {
                             "true_cfg_scale": 4.0 if not is_lightning else 0, 
                             "guidance_scale": 0 if is_lightning else 7.5,
                             "num_inference_steps": steps
                         }

                    elif "flux" in self.model_id.lower():
                         # Flux 1
                         flux_dtype = torch.float32 if device.type == "mps" else dtype
                         self._log_status("loading", 15, "Loading FLUX Pipeline...")
                         pipe = FluxPipeline.from_pretrained(self.model_id, torch_dtype=flux_dtype)
                         if device.type == "cuda": use_offload = True
                         extra_kwargs = {"guidance_scale": 0.0, "num_inference_steps": 4, "max_sequence_length": 256}

                    elif "sdxl-turbo" in self.model_id.lower() or "turbo" in self.model_id.lower():
                         # SDXL Turbo
                         sdxl_dtype = torch.float32 if device.type == "mps" else dtype
                         self._log_status("loading", 15, "Loading SDXL Turbo Pipeline...")
                         pipe = AutoPipelineForText2Image.from_pretrained(self.model_id, torch_dtype=sdxl_dtype, variant="fp16" if sdxl_dtype == torch.float16 else None)
                         extra_kwargs = {"guidance_scale": 0.0, "num_inference_steps": 4}

                    else:
                         # Generic
                         run_dtype = torch.float32 if device.type == "mps" else dtype
                         self._log_status("loading", 15, "Loading Pipeline...")
                         pipe = AutoPipelineForText2Image.from_pretrained(self.model_id, torch_dtype=run_dtype, variant="fp16" if run_dtype == torch.float16 else None)
                         extra_kwargs = {}
                finally:
                    # Restore TQDM
                    tqdm.tqdm = original_tqdm
                    if original_diffusers_tqdm:
                         d_logging.tqdm = original_diffusers_tqdm
        except Exception as e:
            self._log_status("error", 0, f"Error loading image pipeline: {e}")
            # Fallback (no capture)
            pipe = AutoPipelineForText2Image.from_pretrained(self.model_id, torch_dtype=dtype)
            extra_kwargs = {}

        # Apply Model Config
        if use_offload:
            print(f"   ℹ️  Enabling CPU offloading...")
            if hasattr(pipe, 'enable_model_cpu_offload'): pipe.enable_model_cpu_offload()
            elif hasattr(pipe, 'enable_sequential_cpu_offload'): pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(device)

        # MPS Fixes
        if device.type == "mps":
             pipe.enable_attention_slicing()
             if hasattr(pipe, 'vae'): pipe.vae = pipe.vae.to(torch.float32)
             if hasattr(pipe, 'safety_checker') and pipe.safety_checker: pipe.safety_checker = pipe.safety_checker.to(torch.float32)

        self.pipe = pipe
        self.defaults = extra_kwargs

    def generate(self, prompt, width=1024, height=1024, output_file=None, steps=30, 
                 guidance_scale=7.5, negative_prompt="", unsafe=False, report_json=None, 
                 force=False, bypass_warning=False, progress_callback=None):
        """Generate an image."""
        self.progress_callback = progress_callback
        
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_prompt = "".join(x for x in prompt[:30] if x.isalnum() or x in " -_").strip().replace(" ", "_")
            model_short = self.model_name.split("/")[-1].replace(":", "")
            filename = f"{model_short}_{sanitized_prompt}_{timestamp}.png"
            
            # Load config for output directory
            try:
                from ..server.config import load_config
                config = load_config()
                output_dir = config.get("paths", {}).get("media_output")
                if not output_dir:
                    # Fallback if key missing
                    output_dir = os.path.join(os.getcwd(), "media-output")
            except Exception:
                # Fallback if load fails
                output_dir = os.path.join(os.getcwd(), "media-output")
                
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, filename)

        # Pre-calculate Device and Estimate Resources
        try:
            import torch
            from diffusers import FluxPipeline, AutoPipelineForText2Image
            
            # Determine device and dtype
            device, dtype = get_optimal_device_and_dtype(quiet=True, prefer_bfloat16=True)
            dtype_name = str(dtype).replace("torch.", "")
            
            # Estimate Performance
            tracker = PerformanceTracker()
            est_values = tracker.estimate_image(self.model_id, width, height, device, dtype=dtype_name)
            
            # Display Info Header
            print(f"Platform: {device.type.upper()} | Dtype: {dtype_name}")
            tracker.print_estimate(*est_values)
            
        except ImportError:
            print("❌ Failed to import torch/diffusers. Please check installation.")
            return []

        print(f"🎨 Generating Image")
        print(f"   Model:  {self.model_id}")
        print(f"   Prompt: '{prompt}'")
        print(f"   Size:   {width}x{height}")
        print(f"   Output: {output_file}")
        print(f"   Steps:  {steps}")
        print(f"   CFG:    {guidance_scale}")
        print("")  # Spacer
        
        # Check resources
        if not check_resources_and_warn(self.model_id, width=width, height=height, force=force, bypass_warning=bypass_warning,
                                         model_requirements=MODEL_REQUIREMENTS):
            return []
        
        try:
            # Check for SD 3.5 requirement (Divisible by 16)
            if "stable-diffusion-3.5" in self.model_id.lower() or "sd3.5" in self.model_name.lower():
                if width % 16 != 0 or height % 16 != 0:
                    new_w = round(width / 16) * 16
                    new_h = round(height / 16) * 16
                    print(f"   ℹ️  Adjusting {width}x{height} → {new_w}x{new_h} (SD 3.5 requirement)")
                    width, height = new_w, new_h

            # Ensure Pipeline is Loaded
            self._ensure_pipeline_loaded()
            pipe = self.pipe
            
            # Use defaults from load
            extra_kwargs = self.defaults.copy()
            if "negative_prompt" not in extra_kwargs and negative_prompt:
                extra_kwargs["negative_prompt"] = negative_prompt
            
            # Sync steps variable with actual inference configuration
            if "num_inference_steps" in extra_kwargs:
                steps = extra_kwargs["num_inference_steps"]
            else:
                extra_kwargs["num_inference_steps"] = steps

            # Disable NSFW safety checker if requested
            if unsafe and getattr(pipe, 'safety_checker', None) is not None:
                pipe.safety_checker = None
                pipe.requires_safety_checker = False
            
            # High-Resolution Memory Optimization (VAE Tiling)
            # Re-check based on current resolution
            total_pixels = width * height
            if total_pixels > 1536 * 1536:
                if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_tiling'):
                    pipe.vae.enable_tiling()
                elif hasattr(pipe, 'enable_vae_tiling'):
                    pipe.enable_vae_tiling()
            
            print(f"🎨 Generating {width}x{height} image... (This may take a moment)")
            
            # Suppress RuntimeWarning from diffusers image_processor during NSFW filtering
            start_time = time.time()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, 
                                        message="invalid value encountered in cast")
                
                # Define callback for Diffusers
                def callback_on_step_end(pipe, step, timestep, callback_kwargs):
                     if progress_callback:
                        # Calculate progress
                        # steps is the total inference steps
                        # step is the current step index (0-based)
                        current_step = step
                        percent = round((current_step / steps) * 100)
                        progress_callback(percent, f"Generating: {percent}%")
                    
                     return callback_kwargs

                # Start Resource Monitoring
                with ResourceMonitor() as monitor:
                    output = pipe(
                        prompt=prompt, 
                        height=height, 
                        width=width,
                        callback_on_step_end=callback_on_step_end,
                        **extra_kwargs
                    )
                
                # Collect metrics
                duration = time.time() - start_time
                avg_cpu, avg_ram, avg_vram, avg_gpu = monitor.get_averages()
                
                # Record Performance
                tracker.record_image(self.model_id, width, height, device, duration, 
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
            return [output_file]
            
        except ImportError as e:
            print(f"❌ Error: Missing dependencies. {e}")
            return []
        except Exception as e:
            err_str = str(e).lower()
            # Detect gated model access errors (various patterns from HuggingFace Hub)
            is_gated_error = any(pattern in err_str for pattern in [
                "401", "403", "restricted", "gated", "access to model", 
                "you need to agree", "accept the license", "repository is gated"
            ])
            
            if is_gated_error:
                # Check if this is the default model (SD 3.5 Turbo)
                is_default = self.model_id == IMAGE_MODELS.get("default", "") or "stable-diffusion-3.5" in self.model_id.lower()
                
                print(f"\n❌ Access Denied / Authentication Error")
                print(f"   Error Details: {e}")
                print(f"")
                print(f"   Possible causes:")
                print(f"   1. The model '{self.model_id}' is Gated and requires license acceptance.")
                print(f"   2. Your Hugging Face token is invalid or expired (triggers 401/403).")
                print(f"")
                print(f"   🔧 Troubleshooting:")
                print(f"      1. Visit: https://huggingface.co/{self.model_id}")
                print(f"         (If it asks to 'Agree', accept it. If not, it's open.)")
                print(f"      2. Run: huggingface-cli login (to refresh your token)")
                print(f"")
                
                if is_default:
                    print(f"   💡 Quick Alternative: Use an ungated model (no login required):")
                    print(f"      python ai-media.py -i -p \"your prompt\" --image-model sdxl")
                    print(f"")
                    print(f"   📖 See README.md > Gated Models for full setup instructions.")
                else:
                    print(f"   💡 Alternative: Use '--image-model sdxl' (ungated, no login required).")
            elif "divisible by 8" in err_str or "divisible by 16" in err_str:
                # Extract the actual divisor from the error message
                import re
                divisor_match = re.search(r'divisible by (\d+)', err_str)
                divisor = int(divisor_match.group(1)) if divisor_match else 8
                
                print(f"❌ Resolution Error: {e}")
                
                # Smart Correction
                new_w = round(width / divisor) * divisor
                new_h = round(height / divisor) * divisor
                
                print(f"\n💡 Tip: Dimensions must be multiples of {divisor}.")
                print(f"   Closest valid size: {new_w}x{new_h}")
                
                try:
                    choice = input(f"   🔄 Retry with {new_w}x{new_h}? [y/N]: ").lower().strip()
                    if choice in ['y', 'yes']:
                        print("")  # Spacer
                        return self.generate(prompt, width=new_w, height=new_h, output_file=output_file, 
                                             model_id=self.model_name, unsafe=unsafe)
                except KeyboardInterrupt:
                    pass
                print("")
            elif "pos_embed_max_size" in err_str:
                # SD 3.5 positional embedding limit (max latent size = 192 → hard limit ~1536x1536)
                # But quality degrades above 1296x1296, so recommend that
                max_latent = 192
                hard_limit = max_latent * 8  # 1536 (architectural limit)
                quality_max = 1296  # Recommended max before noise artifacts
                print(f"\n❌ SD 3.5 Resolution Limit Exceeded")
                print(f"   Error: {e}")
                print(f"\n   Explanation:")
                print(f"   • SD 3.5 uses fixed positional embeddings (pos_embed_max_size = {max_latent}).")
                print(f"   • Architectural hard limit: {hard_limit}x{hard_limit} pixels.")
                print(f"   • Recommended max (before noise): {quality_max}x{quality_max} pixels.")
                print(f"   • Your request ({width}x{height}) exceeds the hard limit.")
                print(f"\n   💡 Solution: Generate at ≤{quality_max}x{quality_max} and upscale, or use a different model.")
                print(f"      Example: python ai-media.py -i -p \"prompt\" -s 1024 --upscale -uf 5x\n")
                
                # Auto-Upscale Fallback
                try:
                    # Calculate base resolution that fits within limits while maintaining aspect ratio
                    max_res = quality_max
                    aspect_ratio = width / height
                    if aspect_ratio >= 1:  # Landscape or square
                        base_w = min(1024, max_res)
                        base_h = int(base_w / aspect_ratio)
                        base_h = round(base_h / 8) * 8  # Ensure divisible by 8
                    else:  # Portrait
                        base_h = min(1024, max_res)
                        base_w = int(base_h * aspect_ratio)
                        base_w = round(base_w / 8) * 8
                    
                    # Calculate upscale factor needed
                    upscale_factor = max(width / base_w, height / base_h)
                    final_w = int(base_w * upscale_factor)
                    final_h = int(base_h * upscale_factor)
                    
                    print(f"   ✨ Alternative: Generate at {base_w}x{base_h} and Auto-Upscale {upscale_factor:.1f}x?")
                    print(f"      This produces a {final_w}x{final_h} image using the Upscaler model.")
                    choice = input(f"   🔄 Try Auto-Upscale workflow? [y/N]: ").lower().strip()
                    if choice in ['y', 'yes']:
                        print(f"\n📉 Switching to base resolution: {base_w}x{base_h}...")
                        # Import upscaler here to avoid circular import
                        from ..upscaling import upscale_image_file
                        # 1. Generate Base Image (recursive call to self)
                        output = self.generate(prompt, width=base_w, height=base_h, output_file=output_file, 
                                             model_id=self.model_name, unsafe=unsafe)
                        if output:
                            # 2. Upscale Result
                            print("")
                            return upscale_image_file(output[0], output[0], strength=0.0, factor=upscale_factor)
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
                        output = self.generate(prompt, width=1280, height=720, output_file=output_file, 
                                             model_id=self.model_name, unsafe=unsafe)
                        if output:
                            # 2. Upscale Result
                            print("")
                            return upscale_image_file(output[0], output[0], strength=0.0, factor=4.0)
                except KeyboardInterrupt:
                    pass
                print("")
            else:
                print(f"❌ Generation failed: {e}")
            return []


# WRAPPER FUNCTION (Backward Compatibility for CLI/Interactive)
def generate_image(prompt, output_file, width, height, model_name="default", steps=30, 
                   guidance_scale=7.5, negative_prompt="", unsafe=False, report_json=None, force=False, bypass_warning=False, progress_callback=None):
    """Wrapper for ImageGenerator to maintain CLI compatibility.
    
    See ImageGenerator.generate for args.
    """
    generator = ImageGenerator(model_id=model_name)
    outputs = generator.generate(
        prompt=prompt, 
        width=width, 
        height=height, 
        output_file=output_file, 
        steps=steps,
        guidance_scale=guidance_scale, 
        negative_prompt=negative_prompt, 
        unsafe=unsafe, 
        report_json=report_json, 
        force=force, 
        bypass_warning=bypass_warning, 
        progress_callback=progress_callback
    )
    return len(outputs) > 0
