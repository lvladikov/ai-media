"""
Text/Article/Code generation module for AI-Media.

Supports: Article generation, code generation, chat sessions, and deep research.
Uses LLMs like Llama, Qwen, DeepSeek, and Mistral.
"""

import os
import re
import time
import threading
from datetime import datetime
from transformers import StoppingCriteria, StoppingCriteriaList, pipeline
import torch
import gc

from ..models import TEXT_MODELS, NLLB_LANGUAGE_CODES, get_model_id
from ..utils.system import get_optimal_device_and_dtype


from ..utils.interaction import check_overwrite, prompt_choice

# Comprehensive color reference for ANSI TrueColor output
# Format: "ColorName=R,G,B" - used in system prompts to teach the model correct RGB values
COLOR_REFERENCE = (
    # Basic Colors
    "Red=255,0,0 | Green=0,128,0 | Blue=0,0,255 | Yellow=255,255,0 | "
    "Cyan=0,255,255 | Magenta=255,0,255 | White=255,255,255 | Black=0,0,0 | "
    # Grays
    "Gray=128,128,128 | DarkGray=64,64,64 | LightGray=192,192,192 | Silver=192,192,192 | "
    # Reds/Pinks
    "Pink=255,192,203 | HotPink=255,105,180 | DeepPink=255,20,147 | Crimson=220,20,60 | "
    "DarkRed=139,0,0 | Maroon=128,0,0 | Salmon=250,128,114 | Coral=255,127,80 | "
    # Oranges
    "Orange=255,165,0 | DarkOrange=255,140,0 | OrangeRed=255,69,0 | Tomato=255,99,71 | "
    # Browns
    "Brown=165,42,42 | Chocolate=210,105,30 | SaddleBrown=139,69,19 | Sienna=160,82,45 | Tan=210,180,140 | "
    # Yellows/Golds
    "Gold=255,215,0 | Khaki=240,230,140 | LemonChiffon=255,250,205 | LightYellow=255,255,224 | "
    # Greens
    "Lime=0,255,0 | LimeGreen=50,205,50 | LightGreen=144,238,144 | DarkGreen=0,100,0 | "
    "ForestGreen=34,139,34 | SeaGreen=46,139,87 | SpringGreen=0,255,127 | Olive=128,128,0 | "
    "Teal=0,128,128 | Aquamarine=127,255,212 | MediumSeaGreen=60,179,113 | "
    # Blues
    "Navy=0,0,128 | DarkBlue=0,0,139 | MediumBlue=0,0,205 | RoyalBlue=65,105,225 | "
    "SkyBlue=135,206,235 | LightBlue=173,216,230 | DeepSkyBlue=0,191,255 | DodgerBlue=30,144,255 | "
    "CornflowerBlue=100,149,237 | SteelBlue=70,130,180 | CadetBlue=95,158,160 | "
    # Purples/Violets
    "Purple=128,0,128 | Violet=238,130,238 | Indigo=75,0,130 | DarkViolet=148,0,211 | "
    "Orchid=218,112,214 | Plum=221,160,221 | Lavender=230,230,250 | MediumPurple=147,112,219 | "
    "SlateBlue=106,90,205 | DarkSlateBlue=72,61,139 | MediumSlateBlue=123,104,238 | "
    # Others
    "Turquoise=64,224,208 | MediumTurquoise=72,209,204 | DarkTurquoise=0,206,209 | "
    "Aqua=0,255,255 | Beige=245,245,220 | Ivory=255,255,240 | Azure=240,255,255 | "
    "MistyRose=255,228,225 | Snow=255,250,250 | Honeydew=240,255,240 | AliceBlue=240,248,255"
)




class CancelStopCriteria(StoppingCriteria):
    """Criteria to stop HuggingFace generation when is_cancelled is True."""
    def __init__(self, generator):
        self.generator = generator
        
    def __call__(self, input_ids, scores, **kwargs):
        if self.generator.is_cancelled:
            return True
        return False

class ArticleGenerator:
    """Text generation for articles, code, research, and chat using LLMs."""
    
    def __init__(self, model_name="llama-3.1-8b", device=None, args=None, 
                 force=False, bypass_warning=False, progress_callback=None):
        """Initialize the article generator.
        
        Args:
            model_name: Model short code or HF ID
            device: Torch device (auto-detected if None)
            args: Optional argparse namespace for flags like --force
            progress_callback: Optional async function(status, progress, message) to report progress
        """
        import torch
        self.torch = torch
        
        self.model_name = get_model_id(model_name, TEXT_MODELS)
        self.device = device or get_optimal_device_and_dtype(quiet=True, prefer_bfloat16=True)[0]
        self.pipeline = None
        self.translation_pipeline = None
        self.translation_model_id = None
        self.model = None
        self.tokenizer = None
        self.args = args
        self.force = force
        self.bypass_warning = bypass_warning
        self.progress_callback = progress_callback
        
        # Import DDGS for web search
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            self.ddgs = DDGS()
        except ImportError:
            self.ddgs = None
            print("⚠️  duckduckgo_search/ddgs not installed. Online search unavailable.")
            
        self.location_cache = None
        self._last_location = None  # Track location for change detection
        self._first_message_sent = False  # Track if full system prompt was sent
        self._lock = threading.Lock()
        self.last_reasoning = None  # Store reasoning from last generation
        self.last_error = None      # Store last critical error message
        self.is_cancelled = False

    def stop(self):
        """Signal the generator to stop current operation."""
        if self.is_cancelled:
            return
        self.is_cancelled = True
        print(f"🛑 Interruption requested for {self.model_name}")

    def translate_text(self, content: str, target_lang: str, source_lang: str = "eng_Latn", model_id: str = "nllb-200-3.3b", keep_loaded: bool = False, is_chat: bool = False):
        """Translate content using selected translation model with intelligent memory management.
        
        Args:
            content: Text to translate
            target_lang: Target language code (e.g., "en", "es", "spa_Latn")
            source_lang: Source language code or "auto" for auto-detection (NLLB only)
            model_id: Translation model ID (nllb-200-3.3b, alma-13b, qwen3-8b, etc.)
            keep_loaded: If True, keep model in memory after translation (for chat). Default: False
            is_chat: If True, preserves <think> tags in output for chat UI. Default: False (strips reasoning)
            
        Returns:
            Translated text or None on failure
            
        Notes:
            - NLLB models: Fast specialized pipeline, supports auto-detect
            - LLM models: Slower prompt-based translation, requires explicit source language
            - If keep_loaded=False, model unloads immediately after translation
        """
        if not content.strip():
            return ""

        from ai_media.models import TRANSLATION_MODELS
        
        # Get HuggingFace model ID
        hf_model_id = TRANSLATION_MODELS.get(model_id, TRANSLATION_MODELS.get("default_text"))
        is_nllb = "nllb" in model_id.lower()
        
        print(f"🌍 Translating with {model_id} ({'keeping loaded' if keep_loaded else 'will unload'})")
        
        # Auto-detect source language if requested (Global for both NLLB and LLMs)
        if source_lang == "auto":
            try:
                from langdetect import detect
                detected = detect(content)
                # Map 2-letter code to NLLB/Standard format which both paths understand
                # NLLB needs 'eng_Latn', LLM mapper understands 'eng_Latn' or 'en'
                # We normalize to NLLB code if possible, or keep detected 2-letter
                source_lang = NLLB_LANGUAGE_CODES.get(detected, detected)
                print(f"   🔍 Detected language: {detected} -> {source_lang}")
            except Exception as e:
                print(f"   ⚠️ Auto-detect failed: {e}, defaulting to English")
                source_lang = "eng_Latn"

        # === NLLB MODELS: Specialized Translation Pipeline ===
        if is_nllb:
            # Map language codes to NLLB format
            tgt_nllb = NLLB_LANGUAGE_CODES.get(target_lang, target_lang)
            src_nllb = NLLB_LANGUAGE_CODES.get(source_lang, source_lang) # Already normalized above ideally
            
            if not tgt_nllb:
                print(f"   ❌ Unknown target language: {target_lang}")
                return None

            # Bypass if source and target are the same language
            if src_nllb == tgt_nllb:
                print(f"   ⏭️ Source and target language match ({src_nllb}), skipping translation.")
                return content
                
            print(f"   📝 {src_nllb} -> {tgt_nllb}")
            if self.progress_callback:
                self.progress_callback("generating", 95, f"Translating to {target_lang}...")

            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
                
                # Check for cached translation model
                if self.translation_pipeline and self.translation_model_id == model_id:
                     print(f"   ⚡ Using cached translation model: {model_id}")
                     translator = self.translation_pipeline
                else:
                    # Unload previous if exists
                    if self.translation_pipeline:
                         print(f"   ♻️ Switching translation model: {self.translation_model_id} -> {model_id}")
                         del self.translation_pipeline
                         self.translation_pipeline = None
                         self.translation_model_id = None
                         gc.collect()
                         if self.device.type == "cuda":
                             self.torch.cuda.empty_cache()
                         elif self.device.type == "mps":
                             self.torch.mps.empty_cache()

                    device = self.device
                    dtype = self.torch.float32
                    if device.type == "cuda":
                        from ..utils.system import is_bfloat16_supported
                        dtype = self.torch.bfloat16 if is_bfloat16_supported() else self.torch.float16
                    
                    # Load NLLB model
                    print(f"   ⏳ Loading {model_id}...")
                    dtype_name = str(dtype).replace("torch.", "")
                    print(f"   Platform: {device.type.upper()} | Dtype: {dtype_name}")
                    
                    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
                    model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_id, torch_dtype=dtype).to(device)
                    
                    translator = pipeline(
                        "translation",
                        model=model,
                        tokenizer=tokenizer,
                        # Don't bake in languages here, allow dynamic override
                        max_length=512,
                        device=0 if device.type == "cuda" else -1
                    )
                    
                    # Cache the new pipeline
                    self.translation_pipeline = translator
                    self.translation_model_id = model_id
                    print(f"   ✅ Model loaded")
                
                # Translate line-by-line to preserve formatting (headers, lists, etc)
                lines = content.split('\n')
                translated_lines = []
                
                def translate_chunk(text):
                    if not text.strip(): 
                        return ""
                    try:
                        # Pass dynamic language args
                        res = translator(text, src_lang=src_nllb, tgt_lang=tgt_nllb)
                        return res[0]['translation_text']
                    except Exception as e:
                        print(f"   ⚠️ Chunk translation failed: {e}")
                        return text
                
                # Iterate each line to prevent NLLB from stripping newlines in batches
                for line in lines:
                    if not line.strip():
                        translated_lines.append("")
                    else:
                        translated_lines.append(translate_chunk(line))
                
                result = "\n".join(translated_lines)
                print(f"   ✅ Translation complete ({len(result)} chars)")
                return result

            except Exception as e:
                print(f"   ❌ NLLB translation failed: {e}")
                import traceback
                traceback.print_exc()
                return None
            finally:
                # Clean up ONLY if explicit unload requested (keep_loaded=False)
                # But since we cache in self.translation_pipeline, we don't del local vars
                # If keep_loaded=False, we unload from self.
                if not keep_loaded:
                    print(f"   🧹 Unloading {model_id}...")
                    if self.translation_pipeline:
                        del self.translation_pipeline
                        self.translation_pipeline = None
                        self.translation_model_id = None
                    
                    # Also clear separate vars if they were created locally (not really needed if pipeline wraps them)
                    if 'model' in locals() and model: del model
                    if 'tokenizer' in locals() and tokenizer: del tokenizer
                    
                    gc.collect()
                    if hasattr(self, 'device'): # self.device might be used
                         if self.device.type == "cuda":
                             self.torch.cuda.empty_cache()
                         elif self.device.type == "mps":
                             self.torch.mps.empty_cache()
                    print(f"   ✅ Model unloaded")
        
        # === LLM MODELS: Prompt-Based Translation ===
        else:
            # Map language codes to readable names for prompts
            lang_names = {
                "eng_Latn": "English", "en": "English",
                "spa_Latn": "Spanish", "es": "Spanish",
                "fra_Latn": "French", "fr": "French",
                "deu_Latn": "German", "de": "German",
                "ita_Latn": "Italian", "it": "Italian",
                "por_Latn": "Portuguese", "pt": "Portuguese",
                "zho_Hans": "Chinese (Simplified)", "zh": "Chinese",
                "jpn_Jpan": "Japanese", "ja": "Japanese",
                "kor_Hang": "Korean", "ko": "Korean",
                "rus_Cyrl": "Russian", "ru": "Russian",
                "arb_Arab": "Arabic", "ar": "Arabic",
                "hin_Deva": "Hindi", "hi": "Hindi",
            }
            
            # Try to resolve readable language name using pycountry or smart fallback
            def get_lang_name(code):
                # Check hardcoded map first
                if code in lang_names: return lang_names[code]
                
                # Try pycountry if installed
                try:
                    import pycountry
                    # Try 3-letter ISO code (e.g., 'bul' from 'bul_Cyrl')
                    iso_code = code.split('_')[0]
                    lang = pycountry.languages.get(alpha_3=iso_code)
                    if lang: return lang.name
                except ImportError:
                    pass
                
                # Fallback: Use code but try to make it readable (bul_Cyrl -> Bul)
                # Just return code if all else fails, user can fix prompt manually if really needed
                return code

            src_name = get_lang_name(source_lang)
            tgt_name = get_lang_name(target_lang)
            
            print(f"   📝 {src_name} -> {tgt_name} (LLM translation)")
            
            # Check if we need to load the LLM or switch models
            current_model_loaded = (self.model is not None)
            is_same_model = (self.model_name == hf_model_id) if current_model_loaded else False
            
            if current_model_loaded and not is_same_model:
                print(f"   ♻️ Switching model for translation: {self.model_name} -> {hf_model_id}")
                self._unload_model()
                current_model_loaded = False
            
            need_to_load = not current_model_loaded
            
            try:
                # Load LLM if not already loaded
                if need_to_load:
                    print(f"   ⏳ Loading {model_id}...")
                    # Temporarily change model_name to load translation LLM
                    original_model_name = self.model_name
                    self.model_name = hf_model_id
                    success = self._load_model()
                    if not success:
                        msg = f"Failed to load {model_id}"
                        if self.last_error:
                            msg += f": {self.last_error}"
                        raise Exception(msg)
                else:
                    print(f"   ✅ Using already-loaded model")
                
                # Create translation prompt
                prompt = f"""Translate the following {src_name} text to {tgt_name}.
Provide ONLY the translation, no explanations or additional text.

{src_name} text:
{content}

{tgt_name} translation:"""
                
                # Check if model has a chat template or is ALMA
                if "alma" in model_id.lower():
                    # ALMA models require specific prompt format:
                    # Translate this from {src} to {tgt}:
                    # {src}: {content}
                    # {tgt}:
                    prompt = f"Translate this from {src_name} to {tgt_name}:\n{src_name}: {content}\n{tgt_name}:"
                    full_prompt = prompt
                elif self.pipeline.tokenizer.chat_template:
                    # Create chat structure for instruction-tuned models
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                    # Apply chat template
                    full_prompt = self.pipeline.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    # Fallback for base models or those without template
                    # Just use the raw prompt which is already formatted as a completion task
                    full_prompt = prompt
                
                # Generate translation
                # Use a lock to prevent concurrent access issues
                with self._lock:
                    outputs = self.pipeline(
                        full_prompt,
                        max_new_tokens=2048,
                        do_sample=True,
                        temperature=0.3, # Low temp for translation accuracy
                        return_full_text=False
                    )
                
                result = outputs[0]['generated_text'].strip()
                
                # Handle reasoning blocks (<think>)
                if result:
                    if not is_chat:
                        # Strip reasoning for non-chat contexts (CLI, etc.)
                        import re
                        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
                    # If is_chat is True, we preserve <think> tags for the UI to render
                
                if result:
                    print(f"   ✅ Translation complete ({len(result)} chars)")
                    return result
                else:
                    print(f"   ❌ LLM returned empty translation")
                    return None
                    
            except Exception as e:
                print(f"   ❌ LLM translation failed: {e}")
                import traceback
                traceback.print_exc()
                return None
            finally:
                # Unload LLM if we loaded it AND keep_loaded=False
                if need_to_load and not keep_loaded:
                    print(f"   🧹 Unloading {model_id}...")
                    self._unload_model()
                    print(f"   ✅ Model unloaded")
                elif not need_to_load and not keep_loaded:
                    # Model was already loaded (chat context), but we DON'T unload
                    # because chat needs to keep it for follow-up messages
                    pass


    def _translate_content(self, content: str, target_lang: str, model_id: str = "nllb-200-3.3b"):
        """Legacy alias for internal article gen use."""
        return self.translate_text(content, target_lang, source_lang="eng_Latn", model_id=model_id)

        
    @staticmethod
    def extract_reasoning(content: str) -> dict:
        """Extract reasoning and final answer from deepseek content.
        
        Args:
            content: Raw model output.
            
        Returns:
            dict: {
                "reasoning": str or None,
                "content": str
            }
        """
        if '</think>' in content:
            # Split by closing tag to be robust against missing opening tag
            parts = content.split('</think>', 1)
            reasoning = parts[0].replace('<think>', '').strip()
            final_answer = parts[1].strip() if len(parts) > 1 else ""
            
            return {
                "reasoning": reasoning,
                "content": final_answer
            }
        
        return {
            "reasoning": None,
            "content": content
        }
        

        
        # Import DDGS for web search
        # Try importing 'ddgs' first (new package name), then 'duckduckgo_search' (legacy)
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            self.ddgs = DDGS()
        except ImportError:
            self.ddgs = None
            print("⚠️  duckduckgo_search/ddgs not installed. Online search unavailable.")
        
        self.location_cache = None
        
    def _cleanup_memory(self):
        """Perform light memory cleanup (GC and Cache) without unloading models."""
        import gc
        gc.collect()
        if self.torch.backends.mps.is_available():
            self.torch.mps.empty_cache()
        elif self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _unload_model(self):
        """Fully unload the model and tokenizer to free all possible RAM/VRAM."""
        import gc
        
        # 1. Break down pipeline
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            # Manually break internal references if possible
            if hasattr(self.pipeline, 'model'):
                del self.pipeline.model
            if hasattr(self.pipeline, 'tokenizer'):
                del self.pipeline.tokenizer
            del self.pipeline
            self.pipeline = None

        # 2. Delete direct model references
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        # 3. Clear any stopping criteria that might hold references
        if hasattr(self, 'stopping_criteria'):
            del self.stopping_criteria

        # 4. Aggressive GC
        # Run multiple times to clear reference cycles
        for _ in range(3):
            gc.collect()
        
        # 5. Native Memory Release
        if self.torch.backends.mps.is_available():
            self.torch.mps.empty_cache()
            # Try to force sync to ensure memory is actually freed
            try:
                self.torch.mps.synchronize() 
            except:
                pass
        elif self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            self.torch.cuda.ipc_collect()

    def _log_status(self, status, progress, message, terminal=True):
        """Helper to report progress to callback and/or terminal."""
        if self.progress_callback:
            try:
                # status: "loading" | "generating" | "error"
                # progress: 0-100
                
                # Strip terminal-specific instructions for web UI
                clean_ui_msg = message.replace(" (check terminal for progress)", "")
                self.progress_callback(status, progress, clean_ui_msg)
            except Exception as e:
                print(f"⚠️ Progress callback error: {e}")
        
        # Also print to terminal for server logs (with different icon if needed)
        if terminal:
            # Avoid double emojis if message already has one
            if not any(emoji in message for emoji in ["📚", "⏳", "⚠️", "✅", "🛑"]):
                 print(f"⏳ {message}")
            else:
                 print(message)

    def _load_model(self):
        """Load the LLM pipeline."""
        if self.pipeline:
            return True
            
        if self.is_cancelled:
            self._log_status("error", 0, f"Model loading skipped for {self.model_name} (Cancelled)")
            return
        
        from transformers import AutoTokenizer, pipeline
        
        self._log_status("loading", 10, f"Loading Text Model: {self.model_name}...")
        
        # Check resources before loading
        from ..utils.system import check_resources_and_warn
        from ..models import MODEL_REQUIREMENTS
        
        # Determine effective flags (either explicit or from args)
        eff_force = self.force or (self.args.force if self.args and hasattr(self.args, 'force') else False)
        eff_bypass = self.bypass_warning or (self.args.bypass_warning if self.args and hasattr(self.args, 'bypass_warning') else False)
        
        if not check_resources_and_warn(self.model_name, force=eff_force, bypass_warning=eff_bypass, 
                                         model_requirements=MODEL_REQUIREMENTS):
            self._log_status("error", 0, "Aborted: System resource check failed.")
            return False

        try:
            # Use bfloat16 on CUDA if supported (Ampere+), otherwise float16
            from ..utils.system import is_bfloat16_supported
            if self.device.type == "cuda":
                dtype = self.torch.bfloat16 if is_bfloat16_supported() else self.torch.float16
            elif self.device.type == "mps":
                # Use float16 for large models (>14B) to fit in RAM, even if less stable
                if any(size in self.model_name.lower() for size in ["30b", "32b", "70b"]):
                    dtype = self.torch.float16
                else:
                    dtype = self.torch.float32  # MPS uses float32 for stability on smaller models
            else:
                dtype = self.torch.float32
            
            # Workaround: Qwen3/Llama have numerical instability on MPS float16 (but we must use it for huge models)
            if self.device.type == "mps" and any(m in self.model_name.lower() for m in ["qwen3", "llama"]):
                if dtype == self.torch.float32:
                    self._log_status("loading", 12, f"{self.model_name} detected on MPS - using fp32 for stability...")
                    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                else:
                    self._log_status("loading", 12, f"{self.model_name} detected on MPS - using fp16 to conserve memory (may be unstable)...")
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Memory optimization: 4-bit on CUDA
            quantization_config = None
            if self.device.type == "cuda":
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=dtype,
                        llm_int8_enable_fp32_cpu_offload=True
                    )
                    # Suppress "clean_up_tokenization_spaces" warning
                    tokenizer.clean_up_tokenization_spaces = True
                except ImportError:
                    pass
            
            model_kwargs = {"dtype": dtype}
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            elif self.device.type == 'cuda':
                # Use 'auto' to enable offloading to CPU/RAM if the model is too large for VRAM
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = self.device
                
            # Load model manually to prevent 'quantization_config' leakage into generate()
            from transformers import AutoModelForCausalLM
            import sys
            import contextlib
            
            # Simple wrapper to capture tqdm output (which goes to stderr)
            class TqdmCapture:
                def __init__(self, callback):
                    self.callback = callback
                    self.buffer = ""
                    
                def write(self, text):
                    # Progress bars use \r to overwrite lines. 
                    # We pass raw text to frontend which handles the display logic or
                    # we can strip it here. For now, pass raw to let frontend handle animation
                    # ALWAYS write original raw text to real stderr so terminal logic (TQDM) works
                    sys.__stderr__.write(text)

                    # Only process for web callback if there's actual content
                    if self.callback and text.strip():
                        try:
                            # Filter out raw TQDM bar lines as these are messy in the UI
                            # Our TqdmCallbackWrapper (monkey patch) handles these elegantly
                            if "|" in text and "%" in text:
                                return

                            # Strip ANSI escape codes (like [A) ONLY for web logs
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

            # Ensure model_kwargs are clean for from_pretrained
            device_map_info = model_kwargs.get('device_map', 'Default')
            self._log_status("loading", 15, f"Using Device Map: {device_map_info}")
            self._log_status("loading", 20, "Downloading/Loading model weights... (check terminal for progress)")
            
            # Capture stderr during loading
            try:
                # We pass our helper directly to capture progress
                capture = TqdmCapture(self._log_status)
                with contextlib.redirect_stderr(capture):
                    # Monkey patch TQDM to ensure we capture progress bars even if stderr redirect fails
                    # transformers often uses tqdm.auto, so we try to patch standard tqdm which others inherit/alias
                    import tqdm
                    original_tqdm = tqdm.tqdm
                    
                    class TqdmCallbackWrapper(original_tqdm):
                        def update(self, n=1):
                            super().update(n)
                            if hasattr(self, 'total') and self.total:
                                try:
                                    percent = int(self.n / self.total * 100)
                                    desc = self.desc or "Loading"
                                    # Use the capture.callback which is self._log_status
                                    # Pass terminal=False because the original TQDM (stderr) already prints the bar
                                    capture.callback("loading", percent, f"{desc}: {percent}%", terminal=False)
                                except:
                                    pass

                    # Apply patch to source tqdm
                    tqdm.tqdm = TqdmCallbackWrapper
                    
                    # CRITICAL: Also patch transformers/accelerate references which imported it early
                    # 'Loading checkpoint shards' usually comes from transformers.modeling_utils which uses logging.tqdm
                    original_transformers_tqdm = None
                    try:
                        from transformers.utils import logging as hf_logging
                        if hasattr(hf_logging, 'tqdm'):
                            original_transformers_tqdm = hf_logging.tqdm
                            hf_logging.tqdm = TqdmCallbackWrapper
                    except ImportError:
                        pass
                        
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            **model_kwargs
                        )
                    finally:
                        # Restore TQDM source
                        tqdm.tqdm = original_tqdm
                        # Restore transformers reference
                        if original_transformers_tqdm:
                            hf_logging.tqdm = original_transformers_tqdm
            except Exception as e:
                self._log_status("error", 0, f"Error loading model with capture: {e}")
                # Fallback
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            
            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
            
            # Update internal state references
            self.model = model
            self.tokenizer = tokenizer
            
            dtype_name = str(dtype).replace("torch.", "")
            self._log_status("loading", 20, f"Model loaded. Platform: {self.device.type.upper()} | Dtype: {dtype_name}")
            return True
            
        except RuntimeError as e:
            error_msg = str(e)
            if "Invalid buffer size" in error_msg or "out of memory" in error_msg.lower():
                size_match = re.search(r'(\d+\.?\d*)\s*(GiB|GB|MiB|MB)', error_msg)
                full_error = f"❌ Model too large for this system.\nAllocation failed when trying to reserve {size_match.group(0) if size_match else 'memory'}.\nThe model '{self.model_name}' cannot fit in available memory."
                self.last_error = full_error
                print(full_error)
                if self.progress_callback:
                    self.progress_callback("error", 0, full_error)
                return False
            else:
                err = f"❌ Failed to load model: {e}"
                self.last_error = err
                print(err)
                if self.progress_callback:
                    self.progress_callback("error", 0, err)
                return False
                
        except OSError as e:
            error_msg = str(e)
            if "not a valid model identifier" in error_msg or "Repository Not Found" in error_msg:
                full_error = f"❌ Model not found: '{self.model_name}'\nThis model doesn't exist on HuggingFace."
                self.last_error = full_error
                print(full_error)
                if self.progress_callback:
                    self.progress_callback("error", 0, full_error)
                return False
            else:
                err = f"❌ Failed to load model: {e}"
                print(err)
                if self.progress_callback:
                    self.progress_callback("error", 0, err)
                return False

        except ConnectionAbortedError:
            print(f"🛑 Model loading cancelled (Client disconnected).")
            return False
                
        except (ValueError, Exception) as e:
            error_msg = str(e)
            if "Invalid buffer size" in error_msg or "out of memory" in error_msg.lower():
                size_match = re.search(r'(\d+\.?\d*)\s*(GiB|GB|MiB|MB)', error_msg)
                full_error = f"❌ Model too large for this system.\nAllocation failed when trying to reserve {size_match.group(0) if size_match else 'memory'}.\nThe model '{self.model_name}' cannot fit in available memory."
                print(full_error)
                if self.progress_callback:
                    self.progress_callback("error", 0, full_error)
                return False
            else:
                err = f"❌ Failed to load model: {e}"
                self.last_error = err
                print(err)
                if self.progress_callback:
                    self.progress_callback("error", 0, err)
                return False

    def deep_research(self, query, iterations=3, max_images=5):
        """Perform recursive web search and summarization."""
        if not self.ddgs:
            print("❌ Online search unavailable (duckduckgo_search not installed)")
            return ""
            
        # Refine query for "latest" or "news" searches
        current_year = datetime.now().year
        refined_query = query
        
        # Keywords that suggest a need for a temporal anchor
        is_news_query = any(w in query.lower() for w in ["news", "latest", "update", "breaking"])
        # Keywords that already provide temporal context (don't add year if these are present)
        has_temporal = any(w in query.lower() for w in ["today", "yesterday", "week", "month", "year", "now", "current"])
        
        if is_news_query and not has_temporal and str(current_year) not in query:
            refined_query = f"{query} {current_year}"
            print(f"   💡 Refined news query: '{refined_query}'")

        if self.progress_callback:
            self.progress_callback("generating", 30, f"Deep Researching: '{refined_query}' ({iterations} iterations)...")
        else:
            print(f"\n🔎 Deep Researching: '{refined_query}' ({iterations} iterations)...")
        results = []
        
        # 1. Initial Broad Search
        try:
            search_results = list(self.ddgs.text(refined_query, max_results=iterations))
            pad_width = len(str(iterations))
            for i, res in enumerate(search_results, 1):
                num_str = str(i).zfill(pad_width)
                msg = f"Reading [{num_str}]: {res['title']}..."
                print(f"   {msg}")
                if self.progress_callback:
                   self.progress_callback("generating", 30 + int((i/iterations)*10), msg)
                
                content = res.get('body', '') or res.get('snippet', '')
                
                # Attempt deep scraping for better context
                try:
                    import requests
                    from bs4 import BeautifulSoup
                    
                    page = requests.get(res['href'], timeout=5, headers={"User-Agent": "Mozilla/5.0"})
                    if page.status_code == 200:
                        soup = BeautifulSoup(page.text, 'html.parser')
                        paragraphs = [p.get_text().strip() for p in soup.find_all('p')]
                        full_text = ' '.join(p for p in paragraphs if p)
                        
                        if len(full_text) > 200:
                            content = full_text[:4000] + "..."
                except Exception:
                    pass
                
                results.append(f"Source: {res['title']}\nURL: {res['href']}\nContent: {content}\n")
                time.sleep(1.0)
        except Exception as e:
            print(f"⚠️ Search error: {e}")
        
        # 2. Image Search
        image_results = []
        # Support both explicit include_images boolean (old API) or int count (new API)
        should_search_images = max_images > 0
        
        if should_search_images:
            try:
                image_query = f"{query} photos"
                if self.progress_callback:
                    self.progress_callback("generating", 45, "Searching for images...")
                else:
                    print(f"   🖼️  Searching for images...")

                image_search = list(self.ddgs.images(image_query, max_results=max_images))
                
                for img in image_search:
                    img_url = img.get('image', '')
                    img_title = img.get('title', 'Image')
                    if img_url and img_url.startswith('http'):
                        image_results.append(f"![{img_title}]({img_url})")
                
                if image_results:
                    if self.progress_callback:
                        self.progress_callback("generating", 48, f"Found {len(image_results)} images")
                    else:
                        print(f"   ✅ Found {len(image_results)} images")
            except Exception as e:
                print(f"   ⚠️  Image search failed: {e}")
            
        research_context = f"CURRENT DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        research_context += "The following information was retrieved from the web. Prioritize RECENT data over historical matches.\n\n"
        research_context += "\n\n".join(results)
        
        if image_results:
            research_context += "\n\n## Available Images (found during research)\n"
            research_context += "\n".join(image_results)
        
        return research_context

    def generate_article(self, topic, output_file, format="md", online=False, 
                        research_iter=3, max_images=5, length="quick", translate=False, target_language=None, translation_model="nllb-200-3.3b"):
        """Generate full article with optional research.
        
        Args:
            topic: Article topic/title
            output_file: Path to save output
            format: Output format (md, html, pdf, docx, rtf, txt, json)
            online: Enable deep research mode
            research_iter: Number of search iterations
            max_images: Maximum number of images to fetch
            length: 'quick' (512 tokens), 'standard' (2048), 'detailed' (4096)
        """
        from rich.console import Console
        console = Console()
        
        length_config = {
            "quick": {"tokens": 512, "desc": "concise"},
            "standard": {"tokens": 2048, "desc": "balanced"},
            "detailed": {"tokens": 4096, "desc": "comprehensive"},
        }
        config = length_config.get(length, length_config["detailed"])
        max_tokens = config["tokens"]
        style = config["desc"]
        
        research_data = ""
        if online:
            with console.status(f"[bold green]Thinking... (Deep Research Iterations {research_iter})[/bold green]", spinner="dots"):
                research_data = self.deep_research(topic, iterations=research_iter, max_images=max_images)
        
        self._load_model()
        if not self.pipeline:
            return
        
        if self.progress_callback:
            self.progress_callback("generating", 40, f"Writing {style} article on '{topic}'...")
        else:
            print(f"✍️  Writing {style} article on '{topic}'...")
        
        # Prompt Engineering
        if research_data:
            system_prompt = (
                f"You are an expert investigative journalist. Write a {style}, well-structured "
                "article based on the following research context. Use Markdown formatting. "
                "Cite sources where appropriate. "
                "CRITICAL: The context contains a list of 'Available Images'. If these images are "
                "relevant to the content, please embed them using standard Markdown "
                "syntax `![Alt Text](URL)` where they fit the narrative. Do not force the inclusion "
                "of irrelevant images."
            )
            user_prompt = f"Topic: {topic}\n\nResearch Context:\n{research_data}\n\nArticle:"
        else:
            system_prompt = (
                f"You are a creative writer and expert knowledge base. Write a {style}, "
                "well-structured article on the following topic. Use Markdown formatting."
            )
            user_prompt = f"Topic: {topic}\n\nArticle:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        full_prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        with console.status("[bold green]Thinking... (Writing Article)[/bold green]", spinner="dots"):
            # Prepare stopping criteria
            stop_criteria = StoppingCriteriaList([CancelStopCriteria(self)])
            self.is_cancelled = False # Reset before generation
            
            with self._lock:  # Ensure thread-safe model access
                outputs = self.pipeline(
                    full_prompt, 
                    max_new_tokens=max_tokens, 
                    do_sample=True, 
                    temperature=0.7,
                    return_full_text=False,
                    stopping_criteria=stop_criteria
                )
        
        raw_text = outputs[0]['generated_text'].strip()
        
        # Extract reasoning
        extracted = self.extract_reasoning(raw_text)
        self.last_reasoning = extracted["reasoning"]
        final_md = extracted["content"]
        
        # Extract and save <think> blocks separately (Legacy/Additional safeguard)
        think_matches = re.findall(r'<think>(.*?)</think>', raw_text, re.DOTALL)
        if think_matches:
            final_md = re.sub(r'<think>.*?</think>\s*', '', final_md, flags=re.DOTALL).strip()
            base, ext = os.path.splitext(output_file)
            think_file = f"{base}-think.md"
            try:
                with open(think_file, "w", encoding="utf-8") as f:
                    f.write("# Reasoning Process\n\n")
                    for i, block in enumerate(think_matches, 1):
                        if len(think_matches) > 1:
                            f.write(f"## Block {i}\n\n")
                        f.write(block.strip() + "\n\n")
                print(f"💭 Reasoning saved to: {think_file}")
            except Exception as e:
                print(f"⚠️  Could not save reasoning: {e}")
        
        # Save in requested format
        failed_images = self._save_formatted(final_md, output_file, format, online=online)
        
        # Offer retry if offline and images failed
        if not online and failed_images > 0:
            print(f"\n⚠️  {failed_images} image(s) could not be fetched (hallucinated URLs).")
            print("💡 Tip: Offline models (-ga) cannot provide real image URLs.")
            print("   Options:")
            print("   • Use Deep Research (-gr) to find real images from the web")
            print("   • Remove 'images' from your prompt for text-only articles")
            
            retry = prompt_choice("What would you like to do?", [
                ("Retry with Deep Research (online)", "y"),
                ("Keep current output (no images)", "n")
            ])
            
            if retry == "y":
                print("\n🔄 Retrying with Deep Research...")
                self.generate_article(
                    topic=topic,
                    output_file=output_file,
                    format=format,
                    online=True,
                    research_iter=research_iter,
                    length=length,
                    max_images=max_images,
                    translate=translate,
                    target_language=target_language
                )
        
        final_output_path = output_file

        # Validation / Translation Step
        if translate and target_language:
            translated_content = self._translate_content(final_md, target_language, model_id=translation_model)
            if translated_content:
                # Save translated file
                base, ext = os.path.splitext(output_file)
                trans_file = f"{base}.{target_language}{ext}"
                try:
                    self._save_formatted(translated_content, trans_file, format, online=online)
                    trans_file = os.path.normpath(trans_file)
                    print(f"✅ Translated article saved to: {trans_file}")
                    final_output_path = trans_file
                except Exception as e:
                    print(f"⚠️ Failed to save translation: {e}")
        
        return final_output_path

    def generate_code(self, prompt, output_file=None):
        """Generate Code from Prompt (supports multi-file output)."""
        self._load_model()
        if not self.pipeline:
            return False
        
        from rich.console import Console
        console = Console()
        console.print(f"💻 Generating Code for: '{prompt}'...")
        
        if self.progress_callback:
            self.progress_callback("generating", 10, "Preparing prompt...")

        system_prompt = (
            "You are an expert coding assistant. Write clean, efficient, and well-commented code "
            "based on the user's request. Return ONLY the code blocks. "
            "IMPORTANT: Before EACH code file, include a comment line with the filename, "
            "e.g., '# filename: my_script.py' or '// filename: src/utils.js'. "
            "You can use folder paths like 'src/module/file.py'. "
            "If multiple files are needed, separate them with filename comments. "
            "Do NOT include markdown backticks. "
            "CRITICAL: Do NOT write any conversational text, introductions, or conclusions outside of code comments. "
            "Any explanation MUST be inside a comment block valid for the detected language "
            "(e.g. // for Rust/C/JS, # for Python). Never output invalid syntax."
        )
        user_prompt = f"Request: {prompt}\n\nCode:"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        full_prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        if self.progress_callback:
            self.progress_callback("generating", 20, "Generating code... (this may take a while)")

        console.print()
        with console.status("[yellow] Thinking...[/yellow]", spinner="dots"):
            # Prepare stopping criteria
            stop_criteria = StoppingCriteriaList([CancelStopCriteria(self)])
            self.is_cancelled = False # Reset before generation
            
            with self._lock:  # Ensure thread-safe model access
                outputs = self.pipeline(
                    full_prompt, 
                    max_new_tokens=4096, 
                    do_sample=True, 
                    temperature=0.2,
                    top_p=0.9,
                    return_full_text=False,
                    stopping_criteria=stop_criteria
                )
        
        response_raw = outputs[0]['generated_text'].strip()
        
        # Extract reasoning if present
        extracted = self.extract_reasoning(response_raw)
        self.last_reasoning = extracted["reasoning"]
        response = extracted["content"]
        
        # Try to detect language from markdown fence
        detected_lang = None
        match = re.search(r"```(\w+)\s*\n", response)
        if match:
            detected_lang = match.group(1).lower()

        # STRICT PARSING: Use variable-length fence matching to handle nested code blocks correctly.
        # Matches: (3+ backticks) [optional lang] [newline?] (content) (SAME backticks)
        code_blocks = re.findall(r"(`{3,})(?:\w+)?\n?(.*?)\1", response, re.DOTALL)
        
        if code_blocks:
            # code_blocks is list of tuples [('```', 'content'), ...] -> extract second query
            extracted_content = [block[1].strip() for block in code_blocks]
            
            print(f"   ✂️  Extracted {len(extracted_content)} code block(s) and discarded conversational text.")
            if self.progress_callback:
                self.progress_callback("generating", 85, "Cleaning conversational text...")
            
            # Join blocks with newlines
            response = "\n\n".join(extracted_content)
        else:
            # Fallback: No fences found (raw output)
            pass
        
        if self.progress_callback:
            self.progress_callback("generating", 80, "Parsing output...")

        # Parse multiple files from response
        file_pattern = re.compile(
            r"^(?:#|//)\s*(?:filename:\s*)?([^\s]+\.(?:py|js|ts|jsx|tsx|html|css|java|cpp|c|h|go|rs|rb|php|sh|sql|json|yaml|yml|md|txt))\s*$",
            re.IGNORECASE | re.MULTILINE
        )
        
        parts = file_pattern.split(response)
        files_to_write = []
        
        if len(parts) > 1:
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    filename = parts[i].strip()
                    content = parts[i + 1].strip()
                    if filename and content:
                        files_to_write.append((filename, content))
        else:
            content = response.strip()
            if output_file:
                if "." in os.path.basename(output_file):
                    files_to_write.append((output_file, content))
                else:
                    ext = self._infer_extension(content, prompt=prompt, language=detected_lang)
                    files_to_write.append((f"{output_file}{ext}", content))
            else:
                ext = self._infer_extension(content, prompt=prompt, language=detected_lang)
                files_to_write.append((f"generated_code_{int(time.time())}{ext}", content))
        
        if self.progress_callback:
            self.progress_callback("generating", 90, f"Saving {len(files_to_write)} file(s)...")

        # Write all files
        output_is_dir = output_file and os.path.isdir(output_file)
        always_overwrite = self.args.force if self.args and hasattr(self.args, 'force') else False
        never_overwrite = False
        
        for filepath, content in files_to_write:
            try:
                final_path = filepath
                
                if output_file:
                    if output_is_dir:
                        final_path = os.path.join(output_file, filepath)
                    elif len(files_to_write) == 1:
                        final_path = output_file

                should_write, final_path, always_overwrite, never_overwrite = check_overwrite(
                    final_path, always_overwrite, never_overwrite
                )
                
                if final_path is None:
                    print("🛑 Code generation cancelled.")
                    break
                
                if not should_write:
                    continue
                dir_path = os.path.dirname(final_path)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
                
                with open(final_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Code saved to: {os.path.normpath(final_path)}")
            except Exception as e:
                print(f"❌ Error saving {filepath}: {e}")
                
        return True
                
    def _infer_extension(self, code_content, prompt=None, language=None):
        """Infer file extension from code content, detected language, and optional prompt."""
        # Language tag from markdown fence is strongest signal
        if language:
            lang = language.lower()
            if lang in ['python', 'py']: return '.py'
            if lang in ['javascript', 'js', 'node']: return '.js'
            if lang in ['typescript', 'ts']: return '.ts'
            if lang in ['go', 'golang']: return '.go'
            if lang in ['rust', 'rs']: return '.rs'
            if lang in ['cpp', 'c++']: return '.cpp'
            if lang in ['java']: return '.java'
            if lang in ['html']: return '.html'
            if lang in ['css']: return '.css'
            if lang in ['sql']: return '.sql'
            if lang in ['bash', 'sh', 'shell']: return '.sh'
            if lang in ['json']: return '.json'
            if lang in ['xml']: return '.xml'
            if lang in ['yaml', 'yml']: return '.yaml'
        
        # Check for shebangs first
        if code_content.startswith("#!"):
            if "python" in code_content.split("\n", 1)[0]:
                return ".py"
            if "bash" in code_content.split("\n", 1)[0] or "sh" in code_content.split("\n", 1)[0]:
                return ".sh"
            if "node" in code_content.split("\n", 1)[0]:
                return ".js"

        # Content based checks
        if "package main" in code_content or ("func main()" in code_content and "{" in code_content):
            return ".go"
        elif "fn main" in code_content or "use std::" in code_content:
            return ".rs"
        elif "#include" in code_content:
            if "<iostream>" in code_content or "std::" in code_content:
                return ".cpp"
            return ".c"
        elif "public class" in code_content or "System.out.println" in code_content:
            return ".java"
        elif "<html" in code_content or "<!DOCTYPE html" in code_content:
            return ".html"
        elif "import " in code_content or "def " in code_content or "print(" in code_content:
            # Python is common, so check it after specific compiled languages but before generic JS
            return ".py"
        elif "function " in code_content or "const " in code_content or "console.log" in code_content:
            return ".js"
        # Prompt-based fallback
        if prompt:
            prompt_lower = prompt.lower()
            if "python" in prompt_lower: return ".py"
            if "javascript" in prompt_lower or "node" in prompt_lower or " js " in prompt_lower: return ".js"
            if "typescript" in prompt_lower or " ts " in prompt_lower: return ".ts"
            if "golang" in prompt_lower or " go " in prompt_lower: return ".go"
            if "rust" in prompt_lower: return ".rs"
            if "cpp" in prompt_lower or "c++" in prompt_lower: return ".cpp"
            if "java" in prompt_lower: return ".java"
            if "html" in prompt_lower: return ".html"
            if "css" in prompt_lower: return ".css"
            if "sql" in prompt_lower: return ".sql"
            if "bash" in prompt_lower or "shell" in prompt_lower: return ".sh"
            if "json" in prompt_lower: return ".json"

        return ".txt"

    def _save_formatted(self, markdown_text, filename, fmt, online=False):
        """Convert and save to specific format.
        
        Returns:
            int: Number of failed image fetches (legacy return, now 0)
        """
        from ai_media.utils.text_conversion import convert_text
        
        failed_image_count = 0
        base, _ = os.path.splitext(filename)
        
        if not filename.lower().endswith(f".{fmt}"):
            filename = f"{base}.{fmt}"
            
        print(f"💾 Saving as {fmt.upper()}...")
        
        try:
            content_bytes = convert_text(markdown_text, fmt, filename)
            
            mode = "wb" if fmt in ["pdf", "docx"] else "w"
            encoding = None if fmt in ["pdf", "docx"] else "utf-8"
            
            # Ensure directory exists
            output_dir = os.path.dirname(filename)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            with open(filename, mode, encoding=encoding) as f:
                if mode == "wb":
                    f.write(content_bytes)
                else:
                    f.write(content_bytes.decode("utf-8"))
                    
            print(f"✅ Saved to {os.path.normpath(filename)}")
        except Exception as e:
            print(f"❌ Error saving {filename}: {e}")
            
        return failed_image_count

    def _detect_location(self):
        """Detect approximate location based on IP."""
        if self.location_cache:
            return self.location_cache
            
        try:
            import json
            from urllib.request import urlopen
            # Short timeout to avoid hanging
            with urlopen("http://ip-api.com/json/", timeout=1.5) as response:
                ip_data = json.loads(response.read().decode())
                if ip_data.get('status') == 'success':
                    self.location_cache = f"{ip_data.get('city')}, {ip_data.get('regionName')}, {ip_data.get('country')}"
                    return self.location_cache
        except Exception as e:
            pass
            
        return "Unknown"

    def chat_single(self, message: str, history: list = None, stream_callback=None) -> str:
        """Generate a single chat response.
        
        Args:
            message: The user's message
            history: List of previous messages as [{"role": "user"|"assistant", "content": "..."}]
            stream_callback: Optional callback for streaming progress/status updates
            
        Returns:
            The assistant's response string
        """
        self._load_model()
        if not self.pipeline:
            return "Error: Model failed to load"
        
        history = history or []
        
        # Build conversation with system prompt
        # Optimization: Full prompt only on first message, minimal updates after
        # Use explicit, clear format to prevent model confusion/hallucination
        # The dynamic %c format (Thu Jan 1 ...) caused the model to over-reason about the date validity
        current_time = datetime.now().strftime("%A, %B %d, %Y %I:%M:%S %p")
        current_location = self._detect_location()
        
        is_first_message = not self._first_message_sent
        location_changed = self._last_location is not None and self._last_location != current_location
        
        # Always send the full system prompt with the current time.
        # Previously, we reduced this on subsequent turns, but that removed the "Source of truth" instruction
        # and "Helpful assistant" identity, causing the model to degrade and refuse to answer time questions.
        # Refined System Prompt: More concise to prevent over-reasoning/hallucination
        # DeepSeek R1 Distill is sensitive to prompt length and can over-analyze simple greetings.
        system_prompt = (
            "You are a helpful AI assistant. Provide direct, accurate, and concise answers."
            f"Current date and time: {current_time}. Location: {current_location}. "
            "Use standard Markdown for all formatting (tables, lists, headers). "
            "For color requests, use ANSI escape codes (e.g., \\033[31m for red) - our terminal interface supports them. "
            "Avoid raw HTML unless specifically asked for a website design context. "
            "If the user says a simple greeting (like 'hi' or 'hello'), just greet them back warmly and briefly."
        )
        
        if is_first_message:
            self._first_message_sent = True
            self._last_location = current_location
        elif location_changed:
            self._last_location = current_location
        
        # Construct message history
        # Optimization: Merge system messages into one if possible, or convert late ones to user messages
        # to ensure the model pays attention to them.
        processed_history = []
        for msg in history:
            if msg["role"] == "system":
                # Convert following system messages to user messages to ensure LLM visibility
                processed_history.append({"role": "user", "content": f"[SYSTEM CONTEXT UPDATE]\n{msg['content']}"})
            else:
                processed_history.append(msg)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(processed_history)
        messages.append({"role": "user", "content": message})
        
        # Apply chat template
        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        
        # Moderate max tokens for chat to prevent runaway generation/RAM spikes
        # Articles/Research still use the dedicated high-limit methods
        chat_max_tokens = 2048 
        
        # Generate response with LOCK to prevent concurrent model use
        with self._lock:
            outputs = self.pipeline(
                prompt,
                max_new_tokens=chat_max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                return_full_text=False, # Optimization: Only return new tokens
            )
        
        response = outputs[0]['generated_text'].strip()
        
        # Cleanup memory after generation
        self._cleanup_memory()
        
        return response


    def process_command(self, user_input, history):
        """Process slash commands shared between CLI and Web."""
        response = {"handled": False, "message": "", "context": "", "error": ""}
        
        # /read <path>
        if user_input.startswith("/read "):
            response["handled"] = True
            file_path = user_input[6:].strip()
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    response["context"] = f"\n\n[File Context: {file_path}]\n{content}\n"
                    response["message"] = f"📄 Added file context: {file_path}"
                except Exception as e:
                    response["error"] = f"Error reading file: {e}"
            else:
                response["error"] = f"File not found: {file_path}"
            return response

        # /reset (Manual RAM cleanup)
        if user_input.strip() == "/reset":
            response["handled"] = True
            self._unload_model()
            response["message"] = "🧹 Model fully unloaded from RAM/VRAM. Next message will trigger reload."
            return response

        # /search <query>
        is_search = False
        for s_cmd in ["/search", "/online-search"]:
            if user_input.startswith(s_cmd + " ") or user_input.startswith(s_cmd + "|"):
                is_search = True
                break
        
        if is_search:
            response["handled"] = True
            parts = user_input.split(' ', 1)
            cmd_part = parts[0]
            query = parts[1].strip() if len(parts) > 1 else ""
            
            iterations = 3
            if '|' in cmd_part:
                try:
                    iter_str = cmd_part.split('|', 1)[1]
                    if iter_str.isdigit():
                        iterations = int(iter_str)
                except:
                    pass
            
            if query:
                try:
                    # Chat search skips images
                    search_results = self.deep_research(query, iterations=iterations, max_images=0)
                    if search_results:
                        response["context"] = f"\n\n[Online Search Context: '{query}']\n{search_results}\n"
                        response["message"] = f"🌍 Added search results for: '{query}'. You can ask questions about it or ask for summary."
                    else:
                        response["message"] = f"⚠️ No results found for: '{query}'"
                except Exception as e:
                    response["error"] = f"Search error: {e}"
            else:
                response["error"] = "Please provide a search query."
            return response

        # /save <path>
        if user_input.startswith("/save"):
            response["handled"] = True
            # Handle /save|all syntax or regular /save
            parts = user_input.split(' ', 1)
            cmd_part = parts[0]
            file_path = parts[1].strip() if len(parts) > 1 else ""
            
            save_all = "|all" in cmd_part.lower()
            content_to_save = None
            label = "response"
            ext_suggestion = ".md"

            if save_all:
                # Format full history as markdown
                history_content = "# Chat Conversation History\n\n"
                for msg in history:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    history_content += f"## {role}\n{msg['content']}\n\n"
                content_to_save = history_content
                label = "full conversation history"
                ext_suggestion = ".md"
            else:
                # 1. Try to find last code block
                for msg in reversed(history):
                    if msg["role"] == "assistant":
                        content = msg["content"]
                        # Attempt to find code blocks and language
                        matches = re.findall(r"```(.*?)\n(.*?)```", content, re.DOTALL)
                        if matches:
                            lang, code = matches[-1]
                            content_to_save = code
                            label = "code block"
                            # Suggest extension based on language
                            lang_map = {"python": ".py", "bash": ".sh", "javascript": ".js", "html": ".html", "css": ".css", "markdown": ".md"}
                            ext_suggestion = lang_map.get(lang.strip().lower(), ".txt")
                            
                            # Heuristic: look for filename in the text before the block
                            if not file_path:
                                fn_match = re.search(r"(\w+[\.\w]+)", content[:content.find("```")].split("\n")[-1])
                                if fn_match and "." in fn_match.group(1):
                                    suggested_fn = fn_match.group(1)
                                    if os.path.splitext(suggested_fn)[1] in lang_map.values():
                                        file_path = suggested_fn
                            break
                        else:
                            # 2. Fallback to full last response
                            content_to_save = content
                            label = "last response"
                            ext_suggestion = ".md"
                            break

            if not file_path:
                # Try to get a descriptive name from context
                context_str = ""
                if save_all and history:
                    for msg in history:
                        if msg["role"] == "user" and not msg["content"].strip().startswith("/"):
                            context_str = msg["content"]
                            break
                elif history:
                    for msg in reversed(history):
                        if msg["role"] == "user" and not msg["content"].strip().startswith("/"):
                            context_str = msg["content"]
                            break
                
                # Slugify: lowercase, alphanumeric and underscores only
                clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', context_str).strip()
                slug = re.sub(r'\s+', '_', clean_text[:25]).lower()
                
                ts = int(time.time())
                if save_all:
                    prefix = f"chat_{slug}" if slug else "chat_all"
                else:
                    type_prefix = "code" if label == "code block" else "resp"
                    prefix = f"{type_prefix}_{slug}" if slug else type_prefix
                    
                file_path = f"{prefix}_{ts}{ext_suggestion}"
            elif "." not in os.path.basename(file_path):
                # Add suggested extension if missing
                file_path += ext_suggestion

            if content_to_save:
                # Check overwrite before saving. 
                # NOTE: In Web context, we might want to just force or return content? 
                # For now keeping CLI logic but adapting for return.
                always_overwrite = self.args.force if self.args and hasattr(self.args, 'force') else False
                should_write, final_path, _, _ = check_overwrite(file_path, always_overwrite=always_overwrite)
                if should_write:
                    try:
                        with open(final_path, "w", encoding="utf-8") as f:
                            f.write(content_to_save)
                        response["message"] = f"💾 Exported {label} to: {final_path}"
                    except Exception as e:
                        response["error"] = f"Error saving file: {e}"
                else:
                    response["message"] = "Save cancelled (skipped)."
            else:
                 response["error"] = "No conversation content found to save."
            
            return response
        
        return response

    def chat_session(self):
        """Interactive Chat Loop."""
        from rich.console import Console
        from rich.markdown import Markdown
        from prompt_toolkit import PromptSession, HTML
        from prompt_toolkit.history import InMemoryHistory
        from prompt_toolkit.lexers import Lexer
        from prompt_toolkit.styles import Style
        from prompt_toolkit.completion import NestedCompleter, PathCompleter, FuzzyCompleter
        
        console = Console()
        
        self._load_model()
        if not self.pipeline:
            return
            
        history = []
        pending_context = ""
        
        # Detect location once at session start
        user_location = "Unknown"
        try:
            import json
            from urllib.request import urlopen
            # Short timeout to avoid hanging
            with urlopen("http://ip-api.com/json/", timeout=1.5) as response:
                ip_data = json.loads(response.read().decode())
                if ip_data.get('status') == 'success':
                    user_location = f"{ip_data.get('city')}, {ip_data.get('regionName')}, {ip_data.get('country')}"
        except:
            pass
        
        class ChatLexer(Lexer):
            def lex_document(self, document):
                def get_line_tokens(line_number):
                    line = document.lines[line_number]
                    for cmd in ['/read', '/save', '/search', '/online-search']:
                        if line.startswith(cmd):
                            base_len = len(cmd)
                            if len(line) > base_len and line[base_len] == '|':
                                end_pos = line.find(' ', base_len)
                                if end_pos == -1:
                                    end_pos = len(line)
                                return [
                                    ('class:command', line[:end_pos]),
                                    ('', line[end_pos:])
                                ]
                            return [
                                ('class:command', cmd),
                                ('', line[base_len:])
                            ]
                    return [('', line)]
                return get_line_tokens

        chat_style = Style.from_dict({
            'command': '#ff00ff bold',
        })

        session = PromptSession(
            history=InMemoryHistory(),
            lexer=ChatLexer(),
            style=chat_style
        )
        
        path_completer = FuzzyCompleter(PathCompleter(expanduser=True))
        completer = NestedCompleter.from_nested_dict({
            '/read': path_completer,
            '/save': path_completer,
            '/search': None,
            '/online-search': None,
            'exit': None,
            'quit': None,
        })
        
        console.print(f"\n💬 [bold]Chat Session Started[/bold] (Model: [bold cyan]{self.model_name}[/bold cyan])")
        console.print("   Type '[bold]exit[/bold]' or '[bold]quit[/bold]' to end.")
        console.print("   Commands: [bold]/read <path>[/bold], [bold]/save[/bold][bold]|all[/bold] [bold]<path>[/bold], [bold]/search[/bold][bold]|N[/bold] [bold]<query>[/bold]")
        console.print("   [dim]💡 Tip: /save saves last code or full response. Use |all for full history.[/dim]")
        console.print("   [dim]💡 Tip: Use /search query or /search|5 query for deeper results.[/dim]\n")
        
        while True:
            try:
                user_input = session.prompt(HTML('<b fg="blue">You:</b> '), completer=completer, complete_while_typing=True)
                if user_input.strip().lower() in ['exit', 'quit']:
                    break
                
                # Check for slash commands first using shared logic
                cmd_result = self.process_command(user_input, history)
                if cmd_result["handled"]:
                    if cmd_result["error"]:
                        console.print(f"[bold red]❌ {cmd_result['error']}[/bold red]")
                    else:
                        if cmd_result["context"]:
                            pending_context += cmd_result["context"]
                        if cmd_result["message"]:
                            # Style message based on type (search vs file)
                            if "search" in cmd_result["message"].lower():
                                console.print(f"[bold green]{cmd_result['message']}[/bold green]")
                            elif "file" in cmd_result["message"].lower():
                                console.print(f"[bold green]{cmd_result['message']}[/bold green]")
                            else:
                                console.print(f"[bold green]{cmd_result['message']}[/bold green]")
                    continue

                # Construct prompt
                final_content = user_input
                if pending_context:
                    final_content = pending_context + "\n" + user_input
                    pending_context = ""
                
                history.append({"role": "user", "content": final_content})
                
                if len(history) > 20:
                    history = history[-10:]
                
                # Build prompt with dynamic system context (Time/Location)
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Use refined system prompt to prevent over-reasoning/hallucination
                system_prompt = (
                    "You are a helpful AI assistant. Provide direct, accurate, and concise answers. "
                    f"Current date and time: {current_time}. User location: {user_location}. "
                    "Use standard Markdown for all formatting (tables, lists, headers). "
                    "For color requests, use ANSI escape codes (e.g., \\033[31m for red) - our terminal interface supports them. "
                    "Avoid raw HTML unless specifically asked for a website design context. "
                    "If the user says a simple greeting (like 'hi' or 'hello'), just greet them back warmly and briefly."
                )
                
                prompt_messages = [{"role": "system", "content": system_prompt}] + history
                
                prompt = self.pipeline.tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
                
                with console.status("[yellow]Thinking...[/yellow]", spinner="dots"):
                    with self._lock:  # Ensure thread-safe model access
                        outputs = self.pipeline(
                            prompt, 
                            max_new_tokens=2048, # Reduced to prevent runaway generation
                            do_sample=True, 
                            temperature=0.7,
                            top_p=0.9,
                            return_full_text=False,
                        )
                
                console.print("[bold green]Bot:[/bold green]")
                
                response_text = outputs[0]['generated_text'].strip()
                
                # Handle DeepSeek R1 reasoning using shared logic
                parsed = self.extract_reasoning(response_text)
                
                if parsed["reasoning"]:
                    console.print("[dim italic]💭 Reasoning:[/dim italic]")
                    console.print(f"[dim italic]{parsed['reasoning']}[/dim italic]")
                    console.print("")  # Spacer
                    if parsed["content"]:
                        console.print("[bold]Answer:[/bold]")
                        console.print(Markdown(parsed["content"]))
                else:
                    console.print(Markdown(parsed["content"]))
                console.print("")
                
                # Keep original response in history to maintain reasoning context for model
                # But display has been handled
                history.append({"role": "assistant", "content": response_text})
                
            except KeyboardInterrupt:
                console.print("\n")
                break
            except Exception as e:
                console.print(f"[bold red]❌ Error:[/bold red] {e}")

