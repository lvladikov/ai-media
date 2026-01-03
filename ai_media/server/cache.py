import os
import sys
import threading
from typing import Any, Dict, Optional

# Helper for safe emoji printing on Windows
def _safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        try:
            # Fallback to ascii/replacement if terminal can't handle emoji
            print(msg.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
        except:
            pass

class ModelCache:
    """Intelligent model caching to avoid unnecessary load/unload cycles.
    
    Models stay loaded as long as the next request uses the same model.
    When a different model is requested, the old one is unloaded first.
    """
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def get(self, category: str, model_name: str) -> Optional[Any]:
        """
        Get a cached model if it matches, otherwise return None.
        
        Args:
            category: 'text', 'image', 'audio', 'video', 'transform'
            model_name: The model identifier
        
        Returns:
            Cached model instance if same model, None if different or not cached
        """
        with self._lock:
            if category not in self._cache:
                return None
            
            cached = self._cache[category]
            if cached.get("model_name") == model_name:
                return cached.get("instance")
            
            # Different model requested, unload old one (needs to be within lock)
            self._unload_internal(category)
            return None
    
    def set(self, category: str, model_name: str, instance: Any):
        """
        Cache a model instance.
        """
        with self._lock:
            self._cache[category] = {"model_name": model_name, "instance": instance}
    
    def is_loaded(self, category: str) -> bool:
        """
        Check if a model category is currently loaded in cache.
        """
        with self._lock:
            return category in self._cache
    
    def unload(self, category: str):
        """
        Unload a specific model category from cache.
        """
        with self._lock:
            self._unload_internal(category)
    
    def _unload_internal(self, category: str):
        """Internal unload without lock for call-within-lock usage."""
        if category in self._cache:
            try:
                instance = self._cache[category].get("instance")
                # Ensure we stop any running generation
                if instance and hasattr(instance, "stop"):
                    _safe_print(f"🧹 Stopping {category} generator before unload...")
                    instance.stop()
                elif instance and hasattr(instance, "is_cancelled"):
                     # Fallback for simple flag setting
                     instance.is_cancelled = True
            except Exception as e:
                _safe_print(f"⚠️ Error stopping {category}: {e}")
                
            del self._cache[category]
            self._clear_memory()
    
    def unload_all(self):
        """
        Unload all cached models.
        """
        with self._lock:
            self._cache.clear()
            self._clear_memory()
    
    def _clear_memory(self):
        """Clear GPU memory after unloading."""
        try:
            from ai_media.utils.system import clear_gpu_memory
            clear_gpu_memory()
        except Exception:
            pass
    
    def get_status(self) -> Dict[str, str]:
        """Get current cache status."""
        return {cat: info.get("model_name", "unknown") for cat, info in self._cache.items()}


# Global cache instance
model_cache = ModelCache()
