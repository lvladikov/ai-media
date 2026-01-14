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
import platform
from datetime import datetime

from ..models import IMAGE_MODELS, MODEL_REQUIREMENTS, get_model_id
from ..utils.system import get_optimal_device_and_dtype, clear_gpu_memory, check_resources_and_warn
from ..utils.parsers import format_time
from ..utils.performance import PerformanceTracker, ResourceMonitor, write_report_json
from ..utils.transformers_patch import ensure_patch_applied, cleanup_patch

# Suppress HuggingFace Hub warning about symlinks
warnings.filterwarnings("ignore", message="The `local_dir_use_symlinks` argument is deprecated")


from ..utils.progress import capture_tqdm_progress, TqdmCapture


class ImageGenerator:
    """Class for generating images using Diffusers pipelines."""
    
    def __init__(self, model_id="default", use_mlx=None, precision=None, **kwargs):
        """Initialize the generator.
        
        Args:
        """
        # Parse precision suffix (e.g. flux-dev:int8)
        if ":" in model_id:
            parts = model_id.split(":")
            # Allow common precisions
            if len(parts) == 2 and parts[1] in ["int4", "int6", "int8", "float16", "bfloat16", "float32"]:
                model_id = parts[0]
                if precision is None:
                    precision = parts[1]
        
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
        
        # Normalize and detect framework/device
        # Map boolean use_mlx to framework strings for central utility
        framework_force = use_mlx
        if use_mlx is True: framework_force = "mlx"
        if use_mlx is False: framework_force = "torch"
        
        self.device, self.dtype = get_optimal_device_and_dtype(
            quiet=True,
            precision_force=precision,
            framework_force=framework_force,
            prefer_mlx=True # Default to MLX on Mac
        )
        
        # Signal MLX usage (device=None)
        self.use_mlx = self.device is None
        self.precision_override = precision
        
        if self.use_mlx:
            self.device = "mlx" # String marker
            if not self.precision_override:
                 # Check config for precision if not forced
                 from ..server.config import CONFIG
                 self.precision_override = CONFIG.get("precision_force")
        
        # Override if model is specifically an MLX-only model (though we handle this in _load mostly)
        self.mlx_model = None

    def _log_status(self, status, progress, message, terminal=True):
        """Helper to report progress to callback and/or terminal."""
        if self.progress_callback:
            try:
                # status: "loading" | "generating" | "error"
                # progress: 0-100
                
                # Strip terminal-specific instructions for web UI
                clean_ui_msg = message.replace(" (check terminal for progress)", "")
                
                # Filter out bypass warnings from UI (keep only in terminal)
                if "(Proceeding due to --bypass-warning flag)" in clean_ui_msg:
                    return

                self.progress_callback(percent=progress, message=clean_ui_msg)
            except Exception as e:
                print(f"⚠️ Progress callback error: {e}")
        
        # Also print to terminal for server logs (with different icon if needed)
        # Suppress terminal print if we already sent to callback AND we are being captured
        # (detected via 'queue' attribute on sys.stdout from StreamLogger in process_manager.py)
        # to avoid double logs in the web client. 
        # HOWEVER: Always print if it's an ERROR or if callback is missing.
        is_captured = hasattr(sys.stdout, 'queue')
        should_print = terminal and (not self.progress_callback or not is_captured or status == "error")

        if should_print:
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
        
        if self.mlx_model:
            print(f"🧹 Unloading MLX image model: {self.model_name}")
            del self.mlx_model
            self.mlx_model = None
            try:
                import mlx.core as mx
                mx.clear_cache()
            except:
                pass
            
    def stop(self):
        """Signal generation to stop."""
        self._cancelled = True
        # Do NOT unload, so next request is fast.
        # Use self.unload() explicitly if full cleanup is needed.
        
    def _ensure_pipeline_loaded(self):
        """Load the pipeline if not already loaded."""
        if self.pipe or self.mlx_model:
            return

        # Check for MLX usage
        if self.use_mlx:
            return self._load_pipeline_mlx()

        import torch
        from ..utils.transformers_patch import ensure_patch_applied
        ensure_patch_applied()
        
        from diffusers import FluxPipeline, AutoPipelineForText2Image, Flux2Pipeline, StableDiffusion3Pipeline, DiffusionPipeline
        
        # --- EPHEMERAL PATCH CLEANUP: Remove file immediately after import ---
        # The file has served its purpose (bypassing the import check).
        # We delete it now to keep the venv clean during runtime.
        cleanup_patch()
        
        # Determine device and dtype (Force torch framework detection)
        framework_arg = "mlx" if self.use_mlx else "torch"
        device, dtype = get_optimal_device_and_dtype(quiet=True, prefer_bfloat16=True, prefer_mlx=False, framework_force=framework_arg, precision_force=self.precision_override)
        self.device = device
        self.dtype = dtype
        
        use_offload = False
        extra_kwargs = {}
        pipe = None
        
        # Special check for Flux + MPS: Automatically suggest MLX if available but not selected?
        # For now we stick to explicit selection to avoid confusion.
        
        self._log_status("loading", 10, f"Loading image model: {self.model_name}...")
        
        # Capture stderr during loading
        try:
            with capture_tqdm_progress(self._log_status):
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
                     # Force bfloat16 for stability and lower memory on MPS/CUDA
                     sd35_dtype = torch.bfloat16 if device.type in ["cuda", "mps"] else torch.float32
                     if sd35_dtype != dtype:
                         dtype_str = "bfloat16" if sd35_dtype == torch.bfloat16 else "float32"
                         source_str = str(dtype).replace("torch.", "")
                         msg = f"⚠️  Auto-upgrading stability precision: {source_str} -> {dtype_str}"
                         self._log_status("loading", 15, msg)
                     
                     self._log_status("loading", 15, "Loading Stable Diffusion 3.5 Pipeline...")
                     pipe = StableDiffusion3Pipeline.from_pretrained(self.model_id, torch_dtype=sd35_dtype)
                     
                     # Force CPU offloading for better RAM management on all devices
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
                         # Suppress harmless warnings about quantization and config attributes
                         import warnings
                         with warnings.catch_warnings():
                             warnings.filterwarnings("ignore", message=".*no linear modules were found.*")
                             warnings.filterwarnings("ignore", message=".*pooled_projection_dim.*")
                             warnings.filterwarnings("ignore", message=".*torch_dtype is deprecated.*")
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

                elif "z-image" in self.model_name.lower() or "zimage" in self.model_name.lower():
                     # Z-Image Turbo (Alibaba/Tongyi) - via diffusers
                     from diffusers import ZImagePipeline
                     # Force bfloat16 for stability and lower memory on MPS/CUDA
                     zimage_dtype = torch.bfloat16 if device.type in ["cuda", "mps"] else torch.float32
                     if zimage_dtype != dtype:
                         dtype_str = "bfloat16" if zimage_dtype == torch.bfloat16 else "float32"
                         source_str = str(dtype).replace("torch.", "")
                         msg = f"⚠️  Auto-upgrading stability precision: {source_str} -> {dtype_str}"
                         self._log_status("loading", 15, msg)
                         
                     self._log_status("loading", 15, "Loading Z-Image Turbo Pipeline (PyTorch)...")
                     pipe = ZImagePipeline.from_pretrained(self.model_id, torch_dtype=zimage_dtype)
                     
                     # Force CPU offloading for better RAM management
                     use_offload = True
                     extra_kwargs = {"guidance_scale": 3.5, "num_inference_steps": 9}

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

    def _load_pipeline_mlx(self):
        """Load MLX pipeline using mflux."""
        try:
            from mlx.utils import tree_flatten
            import mlx.core as mx
        except ImportError:
            self._log_status("error", 0, "❌ mflux not installed. Cannot use MLX for image generation.")
            raise

        self._log_status("loading", 10, f"Loading MLX model: {self.model_name}...")
        
        # Resolve MLX model ID using mappings
        from ..models import MLX_MODEL_MAPPINGS, get_mlx_model_id
        
        # Determine precision 
        # Default to "int4" (mflux convention) unless overridden
        target_precision = "int4"
        if self.precision_override:
            # Normalize precision string
            p = self.precision_override.lower()
            if "bf16" in p or "bfloat16" in p: target_precision = "bf16" 
            elif "float16" in p or "fp16" in p or "16" in p: target_precision = "float16"
            elif "int8" in p or "8" in p: target_precision = "int8"
            elif "int6" in p or "6" in p: target_precision = "int6"
            elif "int4" in p or "4" in p: target_precision = "int4"
            
        mlx_model_id = get_mlx_model_id(self.model_id, target_precision)
        
        # Verify quantize integer for mflux instantiation
        quantize_val = 4 # Default
        if target_precision == "int8": quantize_val = 8
        elif target_precision == "int6": quantize_val = 6
        elif target_precision in ["float16", "bf16"]: quantize_val = None # None means 16-bit usually in mflux if loaded that way?
        # Actually mflux Flux1.from_name takes `quantize` arg which is int (4 or 8) or None.
        
        self._log_status("loading", 15, f"Target Precision: {target_precision} (Quantize: {quantize_val})")
        
        # Update override so resource checker knows what we loaded
        if not self.precision_override:
            self.precision_override = target_precision
        
        # Mapping checks
        short_name = self.model_name.lower()
        
        try:
            if "qwen" in short_name:
                from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
                self._log_status("loading", 20, f"Loading Qwen Image (MLX)...")
                
                # Check if we have a specific repo ID or local path
                # mflux uses `model_path` argument.
                # If mapped ID is a repo (contains /), use that.
                
                # Qwen models (especially 4bit variants) don't play nice with additional runtime quantization attempts
                # due to shape mismatches (group size 64 vs 1). 
                # They are likely already optimized/quantized or require no further quantization relative to group size.
                # Qwen models (especially 4bit variants) require explicit quantization config on load
                # to correctly interpret the packed weights.
                # If we pass None, it treats them as raw, leading to shape mismatches.
                qwen_quant_val = quantize_val if quantize_val is not None else 4
                
                self.mlx_model = QwenImage(
                    model_path=mlx_model_id,
                    quantize=qwen_quant_val 
                )
                
            elif "z-image" in short_name or "z_image" in short_name or "zimage" in short_name:
                # Z-Image (Alibaba/Tongyi) - NOT the same as SDXL Turbo!
                from mflux.models.z_image.variants.turbo.z_image_turbo import ZImageTurbo
                self._log_status("loading", 20, f"Loading Z-Image Turbo (MLX)...")
                
                self.mlx_model = ZImageTurbo(
                    model_path=mlx_model_id,
                    quantize=quantize_val
                )
                


                
            elif "flux" in short_name or "default" in short_name:
                # Flux (Default)
                from mflux.models.flux.variants.txt2img.flux import Flux1
                
                # Map model name to mflux aliases if possible
                mflux_alias = "schnell" # Default
                if "dev" in short_name: mflux_alias = "dev"
                elif "schnell" in mflux_alias: mflux_alias = "schnell"
                
                self._log_status("loading", 20, f"Loading Flux (MLX): {mflux_alias}...")
                self.mlx_model = Flux1.from_name(mflux_alias, quantize=quantize_val)
                
            else:
                # Model not supported on MLX - offer fallback to PyTorch/MPS
                from rich.console import Console
                console = Console()
                console.print(f"\n[bold yellow]⚠️  MLX Not Supported:[/bold yellow] Model '{self.model_name}' is not available on MLX.")
                console.print(f"[dim]   Supported MLX models: Flux, Qwen-Image, Z-Image.[/dim]")
                console.print(f"[dim]   💡 Tip: For MLX-native fast generation, try 'z-image' (similar to SDXL Turbo).[/dim]\n")
                
                # Check if we're in server mode (bypass_warning or non-interactive) or CLI mode
                # In server/web mode: auto-switch. In CLI mode: prompt user.
                import sys
                is_auto_mode = getattr(self, '_bypass_warning', False) or not sys.stdin.isatty()
                
                if is_auto_mode:
                    # Server mode: auto-switch
                    console.print(f"[yellow]   ➡️  Auto-switching to PyTorch/MPS framework...[/yellow]\n")
                    self._log_status("loading", 15, f"⚠️ MLX not supported for {self.model_name}, switching to PyTorch/MPS...")
                else:
                    # CLI mode: prompt user
                    console.print(f"[bold]   Switch to PyTorch/MPS to run this model? [/bold]", end="")
                    try:
                        response = input("[Y/n]: ").strip().lower()
                        if response in ('n', 'no'):
                            console.print(f"[red]   Generation cancelled.[/red]")
                            raise ValueError(f"Model '{self.model_name}' not supported on MLX. User declined PyTorch fallback.")
                    except EOFError:
                        pass  # Non-interactive, proceed with fallback
                    
                    console.print(f"[green]   ✓ Switching to PyTorch/MPS framework...[/green]\n")
                    self._log_status("loading", 15, f"Switching to PyTorch/MPS for {self.model_name}...")
                
                # Switch to PyTorch
                self.use_mlx = False
                self.mlx_model = None
                
                # Load via PyTorch path instead
                return self._ensure_pipeline_loaded()

            self._log_status("loading", 100, f"✅ MLX Model loaded")
            
        except Exception as e:
            self._log_status("error", 0, f"❌ MLX Load Error: {e}")
            raise e

    def _generate_mlx(self, prompt, width, height, output_file, steps, guidance_scale, user_specified_steps=False, user_specified_guidance=False):
        """Handle generation for MLX models using mflux."""
        try:
            from ..utils.performance import PerformanceTracker # Import here to avoid circular
            
            import time
            start_time = time.time()
            
            # Seed logic
            import random
            seed = random.randint(0, 2**32 - 1)
            
            # Execute Generation
            # Use inspection to determine supported arguments
            import inspect
            sig = inspect.signature(self.mlx_model.generate_image)
            supported_args = sig.parameters.keys()
            
            gen_kwargs = {
                "seed": seed,
                "prompt": prompt,
                "height": height,
                "width": width
            }
            
            # Conditionally add optional args if supported by the model wrapper
            requests_steps = steps
            if "steps" in supported_args:
                gen_kwargs["steps"] = steps
            elif "num_inference_steps" in supported_args:
                gen_kwargs["num_inference_steps"] = steps
            else:
                requests_steps = "Auto"
                
            if "guidance" in supported_args:
                gen_kwargs["guidance"] = guidance_scale
                
            # Tqdm Patch for MLX Progress
            import sys
            from tqdm import tqdm as real_tqdm
            outer_progress_callback = self.progress_callback
            original_tqdms = {}
            
            class TqdmWrapper:
                def __init__(self, iterable=None, desc=None, total=None, *args, **kwargs):
                    self.iterable = iterable
                    self.desc = desc or "Generating"
                    self.total = total or (len(iterable) if iterable else None)
                    self.n = 0
                    self._start_time = time.time()
                    self._tqdm = real_tqdm(iterable, desc=desc, total=total, *args, **kwargs)

                def _report_progress(self):
                    if outer_progress_callback and self.total and self.total > 0:
                        percent = min(100, int((self.n / self.total) * 100))
                        # Calculate ETA
                        elapsed = time.time() - self._start_time
                        if self.n > 0:
                            avg_time = elapsed / self.n
                            remaining = max(0, self.total - self.n)
                            eta_secs = int(remaining * avg_time)
                            mins, secs = divmod(eta_secs, 60)
                            outer_progress_callback(percent, f"Generating: {percent}%, Remaining Time: {mins:02d}:{secs:02d}")
                        else:
                            outer_progress_callback(percent, f"Generating: {percent}%")

                def update(self, n=1):
                    self.n += n
                    if self._tqdm: self._tqdm.update(n)
                    self._report_progress()
                
                def __iter__(self):
                    if self.iterable:
                        for item in self._tqdm:
                            yield item
                            self.n += 1
                            self._report_progress()

                def __enter__(self): return self
                def __exit__(self, *args): 
                    if self._tqdm: self._tqdm.close()
                
                def close(self):
                    if self._tqdm: self._tqdm.close()
                
                @property
                def format_dict(self):
                    if self._tqdm: return self._tqdm.format_dict
                    return {"n": self.n, "total": self.total}
                
                def __getattr__(self, name):
                    if self._tqdm:
                        return getattr(self._tqdm, name)
                    raise AttributeError(f"'TqdmWrapper' has no attribute '{name}'")

            # Patch mflux modules
            for name, module in sys.modules.items():
                if name.startswith("mflux") and hasattr(module, "tqdm"):
                    if module.tqdm != TqdmWrapper:
                        original_tqdms[name] = module.tqdm
                        module.tqdm = TqdmWrapper

            try:
                self._log_status("loading", 50, f"Starting MLX Generation ({width}x{height}, steps: {requests_steps})...")
                
                # Execute generation
                image = self.mlx_model.generate_image(**gen_kwargs)
            finally:
                # Restore original TQDMs
                for name, original in original_tqdms.items():
                    if name in sys.modules:
                        sys.modules[name].tqdm = original
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Save
            # mflux GeneratedImage.save defaults to overwrite=False (rename), 
            # but we handle conflict resolution upstream.
            import inspect
            if hasattr(image, "save"):
                try:
                    sig = inspect.signature(image.save)
                    if "overwrite" in sig.parameters:
                        image.save(output_file, overwrite=True)
                    else:
                        image.save(output_file)
                except ValueError: # built-in method like PIL might fail signature check
                     image.save(output_file)
            else:
                 # Should not happen typically
                 image.save(output_file)
                 
            print(f"✅ Image saved to {output_file}")
            
            # Log metrics (approximate for MLX)
            avg_ram = 0 
            avg_cpu = 0
            avg_gpu = 0
            
            tracker = PerformanceTracker()
            tracker.print_actual(duration, avg_cpu, avg_ram, 0, avg_gpu)
            
            # Update stats for reporting
            self._log_status("complete", 100, "Generation Complete") 
            
            return [output_file]
            
        except Exception as e:
            self._log_status("error", 0, f"❌ MLX Generation Error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def generate(self, prompt, width=1024, height=1024, output_file=None, steps=None, 
                 guidance_scale=None, negative_prompt="", unsafe=False, report_json=None, 
                 force=False, bypass_warning=False, progress_callback=None, allow_header=True, **kwargs):
        """Generate an image."""
        self._cancelled = False # Reset cancellation flag
        self._bypass_warning = bypass_warning  # Store for _load_pipeline_mlx auto-switch
        self.progress_callback = progress_callback
        
        # Track if specific steps were requested by user
        user_specified_steps = steps is not None
        if steps is None:
            # Smart defaults based on model type
            lower_name = self.model_name.lower()
            if "z-image" in lower_name:
                steps = 9
            elif "turbo" in lower_name or "schnell" in lower_name:
                steps = 4 
            elif "lightning" in lower_name:
                steps = 8
            else:
                steps = 30  # Default fallback for others

        # Track if specific guidance was requested
        user_specified_guidance = guidance_scale is not None
        if guidance_scale is None:
             guidance_scale = 7.5 # Fallback default
        
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_prompt = "".join(x for x in prompt[:30] if x.isalnum() or x in " -_").strip().replace(" ", "_")
            model_short = self.model_name.split("/")[-1].replace(":", "")
            filename = f"{model_short}_{sanitized_prompt}_{timestamp}.png"
            
            # Load config for output directory
            try:
                from ..server.config import CONFIG
                output_dir = CONFIG.get("paths", {}).get("media_output")
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
            # Apply Transformers v5 patch BEFORE importing diffusers
            from ..utils.transformers_patch import ensure_patch_applied
            ensure_patch_applied()
            from diffusers import FluxPipeline, AutoPipelineForText2Image
            
            # Determine device and dtype
            if self.use_mlx and not self.precision_override:
                 self.precision_override = "int4" # Default for MLX/mflux

            framework_arg = "mlx" if self.use_mlx else "torch"
            device, dtype = get_optimal_device_and_dtype(quiet=True, prefer_bfloat16=True, framework_force=framework_arg, precision_force=self.precision_override)
            
            if device is None and self.use_mlx:
                # MLX Case
                platform_name = "MLX"
                dtype_name = f"{self.precision_override}"
            else:
                platform_name = device.type.upper() if device else "CPU"
                dtype_name = str(dtype).replace("torch.", "")

            
            # Estimate Performance
            tracker = PerformanceTracker()
            # est_values = tracker.estimate_image(self.model_id, width, height, device, dtype=dtype_name)
            
            # Display Info Header
            if allow_header:
                self._log_status("loading", 0, f"Platform: {platform_name} | Dtype: {dtype_name}")
                # est_msg = tracker.get_estimate_msg(*est_values)
                # if est_msg:
                #     self._log_status("loading", 0, est_msg)

            
        except ImportError:
            self._log_status("error", 0, "❌ Failed to import torch/diffusers. Please check installation.")
            return []

        # Reverted: User found full header dump too noisy
        # self._log_status("loading", 0, f"🎨 Generating Image")
        # self._log_status("loading", 0, f"   Model:  {self.model_id}") ...
        print(f"🎨 Generating Image")
        print(f"   Model:  {self.model_id}")
        print(f"   Prompt: '{prompt}'")
        print(f"   Size:   {width}x{height}")
        print(f"   Output: {output_file}")
        print(f"   Steps:  {steps if user_specified_steps else 'Auto (Model Default)'}")
        print(f"   CFG:    {guidance_scale if user_specified_guidance else 'Auto (Model Default)'}")

        print(f"   Framework: {'MLX' if self.use_mlx else 'PyTorch'}")
        print("")  # Spacer
        
        # Helper to pipe messages to client if callback available
        def log_warn(msg):
            self._log_status("loading", 0, msg)
            
        # Check resources
        check_prec = self.precision_override
        if self.use_mlx and not check_prec:
            check_prec = "int4"

        if not check_resources_and_warn(self.model_id, width=width, height=height, force=force, bypass_warning=bypass_warning,
                                         model_requirements=MODEL_REQUIREMENTS, callback=self.progress_callback, 
                                         is_mlx=self.use_mlx, precision=check_prec):
            return []
        
        try:
            # Check for SD 3.5 requirement (Divisible by 16)
            if "stable-diffusion-3.5" in self.model_id.lower() or "sd3.5" in self.model_name.lower():
                if width % 16 != 0 or height % 16 != 0:
                    new_w = round(width / 16) * 16
                    new_h = round(height / 16) * 16
                    log_warn(f"   ℹ️  Adjusting {width}x{height} → {new_w}x{new_h} (SD 3.5 requirement)")
                    width, height = new_w, new_h

            # Ensure Pipeline is Loaded
            # We capture stray stderr warnings from diffusers/torch during the whole generation
            capture = TqdmCapture(self._log_status)
            with contextlib.redirect_stderr(capture):
                self._ensure_pipeline_loaded()
                
            # MLX Branch - Handle separately
            if self.use_mlx:
                 return self._generate_mlx(prompt, width, height, output_file, steps, guidance_scale, user_specified_steps, user_specified_guidance)
            
            pipe = self.pipe
            
            # Use defaults from load
            extra_kwargs = self.defaults.copy()
            
            # Sync negative prompt - pass explicit "" if user didn't specify, to ensure CFG works if model expects it
            if "negative_prompt" not in extra_kwargs and negative_prompt is not None:
                extra_kwargs["negative_prompt"] = negative_prompt
            
            # Sync guidance scale
            # Priority: User Input > Model Default > General Default
            if "guidance_scale" in extra_kwargs and not user_specified_guidance:
                 guidance_scale = extra_kwargs["guidance_scale"]
            else:
                 extra_kwargs["guidance_scale"] = guidance_scale
            
            # Sync steps variable with actual inference configuration
            # Priority: User Input > Model Default > General Default (30)
            if "num_inference_steps" in extra_kwargs and not user_specified_steps:
                 steps = extra_kwargs["num_inference_steps"] # Use model default
            else:
                 extra_kwargs["num_inference_steps"] = steps # Override model default with user choice

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
                
                # Track pure inference time (ignoring setup/step 0 overhead)
                inference_baseline_time = None

                # Define callback for Diffusers
                def callback_on_step_end(pipe, step, timestep, callback_kwargs):
                     nonlocal inference_baseline_time
                     
                     # Check for cancellation
                     if self._cancelled:
                         raise KeyboardInterrupt("Generation cancelled by user.")

                     if progress_callback:
                        # Calculate progress
                        # steps is the total inference steps
                        # step is the current step index (0-based)
                        current_step = step
                        percent = min(100, int(((current_step + 1) / steps) * 100))
                        
                        # Calculate ETA
                        # We skip step 0 for speed estimation because it includes significant 
                        # model offloading/setup time (often 10s+) which skews the average.
                        eta_str = ""
                        
                        if current_step == 0:
                            # Mark the time when step 0 FINISHED. 
                            # Future steps will be measured relative to this timestamp.
                            inference_baseline_time = time.time()
                            eta_str = "Calculating..."
                        elif inference_baseline_time is not None:
                             # Steps completed SINCE baseline (step 0): current_step
                             # e.g. at step 1, 1 step has passed. at step 2, 2 steps have passed.
                             duration = time.time() - inference_baseline_time
                             measure_steps = current_step
                             
                             if measure_steps > 0:
                                 avg_time_per_step = duration / measure_steps
                                 remaining_steps = max(0, steps - (current_step + 1))
                                 eta_seconds = int(remaining_steps * avg_time_per_step)
                                 mins, secs = divmod(eta_seconds, 60)
                                 eta_str = f"{mins:02d}:{secs:02d}"
                        
                        if eta_str:
                             progress_callback(percent, f"Generating: {percent}%, Remaining Time: {eta_str}")
                        else:
                             progress_callback(percent, f"Generating: {percent}%")
                    
                     return callback_kwargs

                # Start Resource Monitoring
                with ResourceMonitor() as monitor:
                    # PyTorch / Diffusers Generation
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
                    log_warn(f"⚠️  Warning: Potential NSFW content detected.\n")
                    log_warn(f"The model's safety checker has blocked the image (returning a black frame).")
                    log_warn(f"👉 Please modify your prompt and try again.")
                    log_warn(f"💡 If your prompt is appropriate, try again with --unsafe to disable the safety checker.\n")
            
            image.save(output_file)
            print(f"✅ Image saved to {output_file}")
            
            tracker.print_actual(duration, avg_cpu, avg_ram, avg_vram, avg_gpu)
            print("")  # Spacer
            return [output_file]
            
        except KeyboardInterrupt:
            print("\n🛑 Generation Cancelled.")
            return []
        except ImportError as e:
            self._log_status("error", 0, f"❌ Error: Missing dependencies. {e}")
            return []
        except Exception as e:
            err_str = str(e).lower()
            # Detect gated model access errors (various patterns from HuggingFace Hub)
            is_gated_error = any(pattern in err_str for pattern in [
                "401", "403", "restricted", "gated", "access to model", 
                "you need to agree", "accept the license", "repository is gated"
            ])
            
            if is_gated_error:
                # Identify if the current model is one of the known gated models
                # (Z-Image is NOT gated, while SD 3.5 and Flux are)
                known_gated_patterns = ["stable-diffusion-3.5", "flux", "stabilityai", "black-forest-labs"]
                is_known_gated = any(p in self.model_id.lower() for p in known_gated_patterns)
                
                log_warn(f"\n❌ Access Denied / Authentication Error")
                log_warn(f"   Error Details: {e}")
                log_warn(f"")
                log_warn(f"   Possible causes:")
                if is_known_gated:
                    log_warn(f"   1. The model '{self.model_id}' is Gated and requires license acceptance.")
                    log_warn(f"   2. Your Hugging Face token is invalid or expired.")
                else:
                    log_warn(f"   1. Your Hugging Face token is invalid, expired, or missing.")
                
                log_warn(f"")
                log_warn(f"   🔧 Troubleshooting:")
                if is_known_gated:
                    log_warn(f"      1. Visit: https://huggingface.co/{self.model_id}")
                    log_warn(f"         (You must Accept the License at HuggingFace to use this model.)")
                    log_warn(f"      2. Run: huggingface-cli login (to refresh your token)")
                else:
                    log_warn(f"      1. Run: huggingface-cli login (to verify your authentication)")
                log_warn(f"")
                
                if is_known_gated:
                    log_warn(f"   💡 Quick Alternative: Use the default ungated model (no login required):")
                    log_warn(f"      python ai-media.py -i -p \"your prompt\" --image-model z-image")
                    log_warn(f"")
                    log_warn(f"   📖 See README.md > Gated Models for full setup instructions.")
                else:
                    log_warn(f"   💡 Alternative: Try '--image-model sdxl' (widely compatible/ungated).")
            elif "divisible by 8" in err_str or "divisible by 16" in err_str:
                # Extract the actual divisor from the error message
                import re
                divisor_match = re.search(r'divisible by (\d+)', err_str)
                divisor = int(divisor_match.group(1)) if divisor_match else 8
                
                log_warn(f"❌ Resolution Error: {e}")
                
                # Smart Correction
                new_w = round(width / divisor) * divisor
                new_h = round(height / divisor) * divisor
                
                log_warn(f"\n💡 Tip: Dimensions must be multiples of {divisor}.")
                log_warn(f"   Closest valid size: {new_w}x{new_h}")
                
                try:
                    if bypass_warning or force:
                         log_warn(f"   (Proceeding due to --bypass-warning flag)")
                         choice = 'y'
                    else:
                         choice = input(f"   🔄 Retry with {new_w}x{new_h}? [Y/n]: ").lower().strip()
                         
                    if choice in ['', 'y', 'yes']:
                        print("")  # Spacer
                        return self.generate(prompt, width=new_w, height=new_h, output_file=output_file, 
                                             model_id=self.model_name, unsafe=unsafe,
                                             force=force, bypass_warning=bypass_warning,
                                             progress_callback=progress_callback,
                                             allow_header=False)
                except KeyboardInterrupt:
                    pass
                print("")
            elif "pos_embed_max_size" in err_str:
                # SD 3.5 positional embedding limit (max latent size = 192 → hard limit ~1536x1536)
                # But quality degrades above 1296x1296, so recommend that
                max_latent = 192
                hard_limit = max_latent * 8  # 1536 (architectural limit)
                quality_max = 1296  # Recommended max before noise artifacts
                log_warn(f"\n❌ SD 3.5 Resolution Limit Exceeded")
                log_warn(f"   Error: {e}")
                log_warn(f"\n   Explanation:")
                log_warn(f"   • SD 3.5 uses fixed positional embeddings (pos_embed_max_size = {max_latent}).")
                log_warn(f"   • Architectural hard limit: {hard_limit}x{hard_limit} pixels.")
                log_warn(f"   • Recommended max (before noise): {quality_max}x{quality_max} pixels.")
                log_warn(f"   • Your request ({width}x{height}) exceeds the hard limit.")
                log_warn(f"\n   💡 Solution: Generate at ≤{quality_max}x{quality_max} and upscale, or use a different model.")
                log_warn(f"      Example: python ai-media.py -i -p \"prompt\" -s 1024 --upscale -uf 5x\n")
                
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
                    
                    log_warn(f"   ✨ Alternative: Generate at {base_w}x{base_h} and Auto-Upscale {upscale_factor:.1f}x?")
                    log_warn(f"      This produces a {final_w}x{final_h} image using the Upscaler model.")
                    if bypass_warning or force:
                         log_warn(f"   (Proceeding due to --bypass-warning flag)")
                         choice = 'y'
                    else:
                         choice = input(f"   🔄 Try Auto-Upscale workflow? [Y/n]: ").lower().strip()
                    if choice in ['', 'y', 'yes']:
                        log_warn(f"\n📉 Switching to base resolution: {base_w}x{base_h}...")
                        # Import upscaler here to avoid circular import
                        from ..upscaling import upscale_image_file
                        # 1. Generate Base Image (recursive call to self)
                        output = self.generate(prompt, width=base_w, height=base_h, output_file=output_file, 
                                             model_id=self.model_name, unsafe=unsafe,
                                             force=force, bypass_warning=bypass_warning,
                                             progress_callback=progress_callback,
                                             allow_header=False)
                        if output:
                            # 2. Upscale Result
                            print("")

                            return upscale_image_file(output[0], output[0], strength=0.0, factor=upscale_factor,
                                                    progress_callback=self.progress_callback,
                                                    check_cancelled=lambda: self._cancelled)
                except KeyboardInterrupt:
                    pass
                print("")
            elif "Invalid buffer size" in err_str:
                log_warn(f"\n❌ Hardware Limitation Reached (Single Buffer Limit)")
                log_warn(f"   Error: {e}")
                log_warn(f"\n   Explanation:")
                log_warn(f"   • Native {width}x{height} generation requires calculating a massive Attention Matrix.")
                log_warn(f"   • This exceeded the maximum allowed size for a single tensor (usually ~4GB on MPS/Metal).")
                log_warn(f"   • This is a hardware/driver limit, not a VRAM limit.")
                log_warn(f"\n   💡 Solution: Use a lower resolution (e.g. 4k or 2k).")
                log_warn(f"      (Native 5K generation requires 'MultiDiffusion' tiling which is not currently supported.)\n")
                
                # Auto-Upscale Fallback
                try:
                    log_warn(f"   ✨ Alternative: Generate at 1280x720 and Auto-Upscale x4?")
                    log_warn(f"      This produces a 5120x2880 (5K) image using the Upscaler model.")
                    if bypass_warning or force:
                         log_warn(f"   (Proceeding due to --bypass-warning flag)")
                         choice = 'y'
                    else:
                         choice = input(f"   🔄 Try Auto-Upscale workflow? [Y/n]: ").lower().strip()
                    if choice in ['', 'y', 'yes']:
                        log_warn("\n📉 Switching to base resolution: 1280x720...")
                        # Import upscaler here to avoid circular import
                        from ..upscaling import upscale_image_file
                        # 1. Generate Base Image
                        output = self.generate(prompt, width=1280, height=720, output_file=output_file, 
                                             model_id=self.model_name, unsafe=unsafe,
                                             force=force, bypass_warning=bypass_warning,
                                             progress_callback=progress_callback,
                                             allow_header=False)
                        if output:
                            # 2. Upscale Result
                            print("")

                            return upscale_image_file(output[0], output[0], strength=0.0, factor=4.0,
                                                    progress_callback=self.progress_callback,
                                                    check_cancelled=lambda: self._cancelled)
                except KeyboardInterrupt:
                    pass
                print("")
            else:
                self._log_status("error", 0, f"❌ Generation failed: {e}")
            return []


# WRAPPER FUNCTION (Backward Compatibility for CLI/Interactive)
def generate_image(prompt, output_file, width, height, model_name="default", steps=None, 
                   guidance_scale=None, negative_prompt="", unsafe=False, report_json=None, force=False, bypass_warning=False, progress_callback=None, use_mlx=None, precision=None):
    """Wrapper for ImageGenerator to maintain CLI compatibility.
    
    See ImageGenerator.generate for args.
    """
    generator = ImageGenerator(model_id=model_name, use_mlx=use_mlx, precision=precision)
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
