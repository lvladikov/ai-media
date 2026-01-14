
import os
import sys
import transformers
import atexit

# --- Rich Console Initialization ---
try:
    from rich.console import Console
    from rich.theme import Theme
    custom_theme = Theme({
        "info": "dim cyan",
        "warning": "magenta",
        "danger": "bold red"
    })
    console = Console(theme=custom_theme)
except ImportError:
    # Fallback for environments without rich
    class MockConsole:
        def print(self, *args, **kwargs):
            if "style" in kwargs: del kwargs["style"]
            print(*args, **kwargs)
    console = MockConsole()

# --- TRANSFORMERS V5 SELF-HEALING PATCH (MT5 FIX) ---
# Transformers v5.0.0+ removed MT5Tokenizer, which is hard-required by Diffusers for SDXL.
# This utility manages the ephemeral "shim" file to provide compatibility.

PATCH_SIG = "# @AI-MEDIA-AUTO-GENERATED-PATCH: MT5-FIX-V1"
# console.print(f"\n[italic green dim][System] Runtime injection success: transformers.MT5Tokenizer is now available.[/italic green dim]")
_TARGET_FILE_SHIM = None
_FILE_SETUP_DONE = False

def _get_target_path():
    global _TARGET_FILE_SHIM
    if _TARGET_FILE_SHIM: return _TARGET_FILE_SHIM
    try:
        tf_path = os.path.dirname(transformers.__file__)
        mt5_dir = os.path.join(tf_path, "models", "mt5")
        _TARGET_FILE_SHIM = os.path.join(mt5_dir, "tokenization_mt5.py")
        return _TARGET_FILE_SHIM
    except:
        return None

def cleanup_patch():
    """Delete the patch file on exit ONLY if it has our signature."""
    path = _get_target_path()
    if not path: return

    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                first_line = f.readline().strip()
            if first_line == PATCH_SIG:
                os.remove(path)
                # print(f"[System] Cleaned up ephemeral patch: {path}")
    except Exception:
        pass

def _create_shim_file(target_file):
    """Create the shim file physically."""
    try:
        mt5_dir = os.path.dirname(target_file)
        if os.path.isdir(mt5_dir) and not os.path.exists(target_file):
            # print(f"[System] Transformers v5 detected. Creating persistent shim...")
            with open(target_file, "w") as f:
                f.write(f"{PATCH_SIG}\n")
                f.write("from transformers.utils import logging\n")
                f.write("try:\n")
                f.write("    from transformers.models.t5.tokenization_t5 import T5Tokenizer\n")
                f.write("except ImportError:\n")
                f.write("    from transformers import T5Tokenizer\n\n")
                f.write("try:\n")
                f.write("    from transformers.models.t5.tokenization_t5 import T5TokenizerFast\n")
                f.write("except ImportError:\n")
                f.write("    try: from transformers import T5TokenizerFast\n")
                f.write("    except: T5TokenizerFast = None\n\n")
                f.write("logger = logging.get_logger(__name__)\n")
                f.write("MT5Tokenizer = T5Tokenizer\n")
                f.write("MT5TokenizerFast = T5TokenizerFast\n")
                f.write("logger.warning('\\nAI-Media: Using T5Tokenizer as MT5Tokenizer fallback (Transformers v5 compatibility).')\n")
            
            # Register atexit backup
            atexit.register(cleanup_patch)
            
            # Register in Lazy Import (helper)
            if hasattr(transformers, "_import_structure"):
                import_struct = transformers._import_structure
                mt5_key = "models.mt5.tokenization_mt5"
                if mt5_key not in import_struct: 
                    import_struct[mt5_key] = []
                
                if "MT5Tokenizer" not in import_struct[mt5_key]:
                    import_struct[mt5_key].append("MT5Tokenizer")
            
            return True
    except Exception as e:
        console.print(f"[Patch] Failed to create shim file: {e}", style="danger")
        return False

def _inject_runtime(target_file):
    """Force inject MT5Tokenizer into transformers module."""
    try:
         import importlib.util
         spec = importlib.util.spec_from_file_location("tokenization_mt5", target_file)
         module = importlib.util.module_from_spec(spec)
         sys.modules["transformers.models.mt5.tokenization_mt5"] = module
         spec.loader.exec_module(module)
         
         # 1. Force injection into main module instance
         transformers.MT5Tokenizer = module.T5Tokenizer
         transformers.MT5TokenizerFast = module.T5TokenizerFast
         
         # 2. Force injection into sys.modules dictionary (crucial for from-imports)
         if "transformers" in sys.modules:
             sys.modules["transformers"].MT5Tokenizer = module.T5Tokenizer
             sys.modules["transformers"].MT5TokenizerFast = module.T5TokenizerFast
             # Also register as submodule (just in case)
             sys.modules["transformers.MT5Tokenizer"] = module
             
         # 3. Patch __all__ to satisfy checks
         if hasattr(transformers, "__all__"):
             if "MT5Tokenizer" not in transformers.__all__:
                 transformers.__all__.append("MT5Tokenizer")
             if "MT5TokenizerFast" not in transformers.__all__:
                 transformers.__all__.append("MT5TokenizerFast")
                 
         # 4. Double-check _import_structure logic (in case it wasn't done or was reset)
         if hasattr(transformers, "_import_structure"):
             if "models.mt5" in transformers._import_structure:
                 # It might be a dict or list
                 mt5_val = transformers._import_structure["models.mt5"]
                 if isinstance(mt5_val, list) and "tokenization_mt5" not in mt5_val:
                     mt5_val.append("tokenization_mt5")
                 elif isinstance(mt5_val, dict):
                    # Should be list of classes? No, import structure is {module: [classes]} or {module: {sub: [classes]}}
                    # Actually for lazy module it is {submodule: [classes]} usually?
                    pass 
             
             # Also explicit key
             if "models.mt5.tokenization_mt5" not in transformers._import_structure:
                  transformers._import_structure["models.mt5.tokenization_mt5"] = ["MT5Tokenizer", "MT5TokenizerFast"]
             else:
                  val = transformers._import_structure["models.mt5.tokenization_mt5"]
                  if "MT5Tokenizer" not in val: val.append("MT5Tokenizer")

         console.print(f"[italic green dim][System] Runtime injection success: transformers.MT5Tokenizer is now available.[/italic green dim]")
         return True
    except Exception as e:
         console.print(f"[bold red][Patch] Runtime injection failed: {e}[/bold red]")
         import traceback
         traceback.print_exc()
         return False

def ensure_patch_applied():
    """
    Called at runtime (e.g. inside generate()) to ensure patch is ACTIVE.
    If attribute is missing, it will re-create file and re-inject.
    This function is idempotent and fast if already applied.
    """
    # Fast path: already patched
    if hasattr(transformers, "MT5Tokenizer"):
        return
    
    target_file = _get_target_path()
    if not target_file: 
        return
        
    # Ensure shim file exists and is valid
    # Always recreate if missing OR if existing file doesn't have our signature
    need_create = True
    if os.path.exists(target_file):
        try:
            with open(target_file, "r") as f:
                first_line = f.readline().strip()
            if first_line == PATCH_SIG:
                need_create = False  # File exists and is valid
        except Exception:
            pass  # Corrupted, will recreate
    
    if need_create:
        _create_shim_file(target_file)
        
    # Force Runtime Injection
    _inject_runtime(target_file)

def apply_patch():
    """
    Called at STARTUP (module level).
    Handles cleanup of STALE files from previous runs.
    Then ensures patch applied.
    """
    global _FILE_SETUP_DONE
    if _FILE_SETUP_DONE: 
        ensure_patch_applied()
        return
        
    target_file = _get_target_path()
    if not target_file: return
    
    # 1. Startup Cleanup (Run once per process)
    # If file exists at startup, it's from a dead process. Clean it.
    if os.path.exists(target_file):
        cleanup_patch()
    
    _FILE_SETUP_DONE = True
    
    # 2. Apply
    ensure_patch_applied()

