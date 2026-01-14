"""
Progress tracking utilities for AI-Media.
Captures TQDM output and redirects to callbacks with formatted percentage and ETA.
"""

import os
import sys
import time
import re
import contextlib
from typing import Callable, Optional


def _safe_call_callback(callback: Callable, status: str, progress: int, message: str, terminal: bool = False):
    """Call a progress callback safely, adapting to its signature."""
    if not callback:
        return
        
    try:
        import inspect
        sig = inspect.signature(callback)
        params = list(sig.parameters.values())
        
        # Determine how many positional arguments to pass
        if len(params) >= 4:
            callback(status, progress, message, terminal)
        elif len(params) == 3:
            callback(status, progress, message)
        else:
            # Fallback for very simple callbacks
            callback(progress, message)
    except Exception:
        # Final fallback: just try to call with 3
        try:
            callback(status, progress, message)
        except:
            pass


class TqdmCapture:
    """Class to capture tqdm output from stderr and redirect to a callback.
    
    Accumulates ANSI-stripped text and searches for complete tqdm patterns.
    """
    def __init__(self, callback: Callable):
        self.callback = callback
        self._buffer = ""
        self._last_percent = -1
        
    def write(self, text: str):
        # Always write original raw text to real stderr so terminal logic (TQDM) works
        sys.__stderr__.write(text)

        if not self.callback:
            return
            
        try:
            # Strip ANSI codes BEFORE accumulating to avoid index misalignment
            clean_text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
            self._buffer += clean_text
            
            # Limit buffer size to prevent memory issues
            if len(self._buffer) > 5000:
                self._buffer = self._buffer[-2500:]
            
            # Look for complete tqdm pattern anywhere in buffer
            # Pattern: "XX%|...| N/M [elapsed<remaining, speed]"
            tqdm_match = re.search(r'(\d+)%\|[^|]*\|\s*\d+/\d+\s*\[[^\]]*<([\d:]+),\s*[\d.]+[a-z/]+\]', self._buffer)
            
            if tqdm_match:
                # Just clear the matched portion from buffer
                # Progress updates are now handled by StreamLogger in process_manager.py
                self._buffer = self._buffer[tqdm_match.end():]
                return

            
            # Process non-tqdm content when we see newlines
            if '\n' in self._buffer:
                lines = self._buffer.split('\n')
                # Keep last partial line in buffer
                self._buffer = lines[-1] if lines else ""
                
                for line in lines[:-1]:
                    line = line.strip()
                    if not line:
                        continue
                    # Skip partial bar lines
                    if "|" in line and "%" in line:
                        continue
                    if line.startswith('[') and '<' in line:
                        continue
                    # Log other messages
                    _safe_call_callback(self.callback, "loading", 0, line, terminal=False)
                        
        except Exception:
            pass

    def flush(self):
        sys.__stderr__.flush()

    def isatty(self):
        return sys.__stderr__.isatty()
    
    @property
    def encoding(self):
        return sys.__stderr__.encoding


@contextlib.contextmanager
def capture_tqdm_progress(callback: Callable):
    """Context manager to capture TQDM output and redirect to callback.
    
    Patches tqdm.tqdm and tqdm.auto.tqdm with a wrapper class to intercept updates.
    """
    try:
        import tqdm
    except ImportError:
        yield
        return

    # 1. Save original classes
    original_tqdm = tqdm.tqdm
    original_auto_tqdm = getattr(tqdm.auto, 'tqdm', None) if hasattr(tqdm, 'auto') else None
    
    # 2. Define Wrapper
    class TqdmCallbackWrapper(original_tqdm):
        _static_callback = None
        
        def __init__(self, *args, **kwargs):
            # Force restricted inputs to prevent wrapping/spam
            if "ncols" not in kwargs:
                try:
                    width = os.get_terminal_size().columns
                    kwargs["ncols"] = min(width - 5, 120) 
                except:
                    kwargs["ncols"] = 100
            
            # Pass to super
            super().__init__(*args, **kwargs)
            self._last_callback_time = 0

        def set_description(self, desc=None, refresh=True):
            # Intercept and truncate super long descriptions
            if desc and len(desc) > 40:
                if "Materializing param" in desc:
                    desc = "Materializing params..."
                elif len(desc) > 50:
                    desc = desc[:47] + "..."
            super().set_description(desc, refresh)

        def update(self, n=1):
            super().update(n)
            
            # Callback Logic
            if hasattr(self, 'total') and self.total and TqdmCallbackWrapper._static_callback:
                try:
                    percent = min(100, int(self.n / self.total * 100))
                    current_time = time.time()
                    
                    # Throttle: Only update every 0.2s or if complete (100%)
                    if percent >= 100 or (current_time - self._last_callback_time) > 0.2:
                        self._last_callback_time = current_time
                        desc = self.desc or "Processing"
                        # Clean up desc
                        if ":" in desc: desc = desc.split(":")[0]
                        
                        clean_msg = f"{desc}: {percent}%"
                        
                        # Estimate remaining time
                        remaining_str = ""
                        if hasattr(self, 'format_dict'):
                            d = self.format_dict
                            rem = d.get('remaining')
                            if rem and isinstance(rem, (int, float)):
                                m, s = divmod(int(rem), 60)
                                remaining_str = f", Remaining Time: {m:02d}:{s:02d}"
                        
                        _safe_call_callback(TqdmCallbackWrapper._static_callback, "loading", percent, f"{clean_msg}{remaining_str}", terminal=False)
                except Exception:
                    pass

    # 3. Inject callback
    TqdmCallbackWrapper._static_callback = callback
    
    # 4. Capture stderr
    capture = TqdmCapture(callback)
    
    original_external_tqdms = {}
    
    try:
        with contextlib.redirect_stderr(capture):
            # 5. Apply Patches
            tqdm.tqdm = TqdmCallbackWrapper
            if hasattr(tqdm, 'auto'):
                tqdm.auto.tqdm = TqdmCallbackWrapper
            
            # 6. Aggressively patch known libraries that might have pre-imported tqdm
            target_modules = [
                "diffusers.utils.logging", 
                "transformers.utils.logging",
                "accelerate.utils",
                "mflux.utils"
            ]
            for mod_name in target_modules:
                if mod_name in sys.modules:
                    mod = sys.modules[mod_name]
                    if hasattr(mod, 'tqdm'):
                        original_external_tqdms[mod_name] = mod.tqdm
                        mod.tqdm = TqdmCallbackWrapper

            yield
            
    finally:
        # 7. Restore Everything
        tqdm.tqdm = original_tqdm
        if original_auto_tqdm and hasattr(tqdm, 'auto'):
            tqdm.auto.tqdm = original_auto_tqdm
            
        for mod_name, orig in original_external_tqdms.items():
            if mod_name in sys.modules:
                sys.modules[mod_name].tqdm = orig
        
        TqdmCallbackWrapper._static_callback = None
