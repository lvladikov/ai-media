"""
MLX-native text generation for Apple Silicon.

Uses mlx-lm for fast, efficient text generation with 4-bit/8-bit quantization.
This module is only loaded on macOS when MLX is available.
"""

import sys

# Guard: Only import on macOS
if sys.platform != "darwin":
    raise ImportError("MLX is only available on macOS")


class MLXTextGenerator:
    """MLX-native text generator for Apple Silicon.
    
    Provides fast text generation using mlx-lm with support for:
    - 6-bit quantization (balanced speed, ~97% quality)
    - 8-bit quantization (balanced, ~98% quality)
    - bfloat16/float16/float32 (full precision)
    """
    
    def __init__(self, model_name: str, precision: str = "int4", progress_callback=None):
        """Initialize MLX text generator.
        
        Args:
            model_name: HuggingFace model ID or path
            precision: "int4", "int6", "int8", "float16", "bfloat16", "float32"
            progress_callback: Optional callback for progress updates
        """
        self.model_name = model_name
        self.precision = precision
        self.progress_callback = progress_callback
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
    def load(self):
        """Load the model with the specified precision."""
        if self._loaded:
            return
            
        try:
            from mlx_lm import load
            import mlx.core as mx
            import mlx.nn as nn
            from mlx.utils import tree_map
        except ImportError:
            raise ImportError("mlx-lm is required for MLX text generation. Install with: pip install mlx-lm")
        
        if self.progress_callback:
            self.progress_callback("loading", 10, f"Loading {self.model_name} ({self.precision})...")
        
        # Resolve best MLX model ID (e.g. switch to mlx-community version)
        from ai_media.models import get_mlx_model_id
        optimized_id = get_mlx_model_id(self.model_name, self.precision)
        
        if optimized_id != self.model_name:
            print(f"⚡ Switching to optimized MLX model: {optimized_id}")
            self.model_name = optimized_id
            
        print(f"🍎 Loading MLX model: {self.model_name} ({self.precision})")
        
        try:
            # 1. LOAD LAZY
            # We load lazily to avoid allocating full precision weights immediately.
            # This gives us a chance to quantize or cast BEFORE materialization.
            self.model, self.tokenizer = load(self.model_name, lazy=True)

            # 2. APPLY PRECISION (Quantize or Cast)
            if self.precision in ["int4", "int6", "int8"]:
                # Check if model is already quantized (by name)
                if "-4bit" in self.model_name or "-6bit" in self.model_name or "-8bit" in self.model_name or "-Q" in self.model_name:
                    print("   ℹ️  Model appears pre-quantized, skipping explicit quantization.")
                else:
                    # Apply on-the-fly quantization to the lazy model structure
                    bits = {"int4": 4, "int6": 6, "int8": 8}[self.precision]
                    print(f"   🔨 Quantizing model to {bits}-bit (Group Size: 64)...")
                    nn.quantize(self.model, bits=bits, group_size=64)
                    
            else:
                target_dtype = None
                if self.precision == "float32":
                     target_dtype = mx.float32
                     print("   🔼 Upcasting to float32...")
                elif self.precision == "float16":
                     target_dtype = mx.float16
                elif self.precision == "bfloat16":
                     target_dtype = mx.bfloat16
                
                if target_dtype:
                    # Standard MLX cast: map astype over parameters and update model
                    self.model.update(tree_map(lambda p: p.astype(target_dtype), self.model.parameters()))

            # 3. MATERIALIZE (Trigger Load/Compute)
            if self.progress_callback:
                 self.progress_callback("loading", 50, "Materializing weights...")
            
            mx.eval(self.model.parameters())
            
            self._loaded = True
            
            if self.progress_callback:
                self.progress_callback("ready", 100, "Model loaded")
            
            print(f"   ✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading MLX model: {e}")
            raise e
        
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7,
                 top_p: float = 1.0, stream: bool = False, stop_sequences: list = None):
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            top_p: Top-p sampling parameter
            stream: Whether to stream tokens
            stop_sequences: List of sequences that stop generation
            
        Yields:
            If stream=True: Individual tokens as they are generated
            If stream=False: Returns complete generated text
        """
        if not self._loaded:
            self.load()
            
        try:
            from mlx_lm import generate, stream_generate
        except ImportError:
            raise ImportError("mlx-lm is required")
        
        # Build generation kwargs
        gen_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
        
        # Add sampling parameters
        try:
            from mlx_lm.sample_utils import make_sampler
            # Create a sampler if temperature or top_p is provided
            # mlx-lm now prefers a sampler object over direct temp/top_p kwargs
            sampler = make_sampler(
                temp=temperature if temperature is not None else 0.0,
                top_p=top_p if top_p is not None else 1.0
            )
            gen_kwargs["sampler"] = sampler
        except ImportError:
            # Fallback for older mlx-lm versions where it's passed directly
            if temperature is not None:
                gen_kwargs["temp"] = temperature
            if top_p is not None and top_p < 1.0:
                gen_kwargs["top_p"] = top_p
            
        if stream:
            return self._generate_stream(gen_kwargs)
        else:
            # Non-streaming: Return full text string
            full_text = ""
            for token in self._generate_stream(gen_kwargs):
                full_text += token
            return full_text
            
    def _generate_stream(self, gen_kwargs):
        """Helper to yield tokens from mlx-lm stream_generate."""
        try:
            from mlx_lm import stream_generate
        except ImportError:
             raise ImportError("mlx-lm is required")

        # Streaming generation
        for response in stream_generate(**gen_kwargs):
            # response is a GenerationResponse object in new mlx-lm
            if hasattr(response, 'text'):
                yield response.text
            else:
                # Fallback for older versions where it might be a string
                yield response
    
    def chat(self, messages: list, max_tokens: int = 512, temperature: float = 0.7,
             top_p: float = 1.0, stream: bool = False):
        """Chat completion using message format.
        
        Args:
            messages: List of {"role": str, "content": str} dicts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stream: Whether to stream tokens
            
        Returns/Yields:
            Generated response text (or tokens if streaming)
        """
        if not self._loaded:
            self.load()
            
        # Build chat prompt using tokenizer's chat template
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception:
            # Fallback to simple format if no chat template
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            prompt += "Assistant: "
        
        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream
        )
    
    def unload(self):
        """Unload the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print(f"🧹 MLX model unloaded")
