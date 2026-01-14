"""System, health, and model routes."""

import json
import platform
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import CONFIG
from ..cache import model_cache

router = APIRouter(tags=["System"])


@router.get("/api/config")
async def get_config():
    """Get public configuration (safe values only)."""
    return {
        "server": CONFIG.get("server", {}),
        "client": CONFIG.get("client", {}),
        "preferences": CONFIG.get("preferences", {}),
    }


@router.put("/api/config")
async def update_config(request: dict):
    """Update configuration preferences."""
    from ..models import ConfigUpdateRequest
    
    config_path = Path(__file__).parent.parent.parent.parent / "config.json"
    
    try:
        # Load existing config
        existing = {}
        if config_path.exists():
            with open(config_path) as f:
                existing = json.load(f)
        
        # Update preferences
        if "preferences" not in existing:
            existing["preferences"] = {}
        
        if request.get("theme"):
            existing["preferences"]["theme"] = request["theme"]
        
        # Save back
        with open(config_path, "w") as f:
            json.dump(existing, f, indent=2)
        
        return {"message": "Configuration updated", "preferences": existing.get("preferences", {})}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "AI-Media server is running"}


@router.get("/api/system")
async def get_system_info():
    """Get system information including device, VRAM, and RAM."""
    import torch
    from ai_media.utils.system import get_optimal_device_and_dtype
    
    # Use centralized detection
    opt_device, opt_dtype = get_optimal_device_and_dtype(quiet=True)
    is_mlx = opt_device is None
    
    # Detect device (physical hardware info)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        gpu_name = "Apple Silicon GPU"
        vram_total = None  # Unified memory
    else:
        gpu_name = None
        vram_total = None
    
    # Resolve Dtype for display
    if is_mlx:
         # Default MLX to int4 unless config says otherwise
         dtype_str = CONFIG.get("precision_force") or "int4"
    else:
         dtype_str = str(opt_dtype).replace("torch.", "")
    
    # RAM
    import psutil
    ram_total = psutil.virtual_memory().total / (1024**3)
    
    return {
        "device": str(opt_device) if opt_device else "mlx",
        "framework": "mlx" if is_mlx else "torch",
        "dtype": dtype_str,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        "mlx_available": is_mlx, # If we resolved to it, it's available
        "gpu_name": gpu_name,
        "vram_total_gb": round(vram_total, 2) if vram_total else None,
        "ram_total_gb": round(ram_total, 2),
        "platform": platform.system(),
        "python_version": sys.version.split()[0],
    }


@router.get("/api/constants")
async def get_constants():
    """Get application constants like resolution presets."""
    from ai_media.constants import get_resolutions
    return {"resolutions": get_resolutions()}


@router.get("/api/system/languages")
async def get_languages():
    """
    Get all NLLB languages with human-readable names.
    Uses pycountry for name resolution, cached for performance.
    """
    from ai_media.models import get_nllb_languages_with_names, ALL_NLLB_CODES
    
    languages = get_nllb_languages_with_names()
    
    return {
        "languages": [
            {"label": name, "value": code}
            for name, code in languages
        ],
        "total": len(ALL_NLLB_CODES)
    }


@router.get("/api/system/translation-models")
async def get_translation_models():
    """Get available translation models."""
    from ai_media.models import TRANSLATION_MODELS, MODEL_REQUIREMENTS
    
    models = []
    for name, model_id in TRANSLATION_MODELS.items():
        if name.startswith("default"):
            continue
        
        reqs = MODEL_REQUIREMENTS.get(model_id, {})
        models.append({
            "value": name,
            "label": name.replace("-", " ").replace(".", " ").title(),
            "model_id": model_id,
            "vram_required": reqs.get("vram"),
            "ram_required": reqs.get("ram"),
        })
    
    return {"models": models}


@router.get("/api/models")
async def get_all_models():
    """Get all available models grouped by category."""
    from ai_media.models import IMAGE_MODELS, VIDEO_MODELS, AUDIO_MODELS, TEXT_MODELS
    
    def format_models(model_dict: dict, category: str):
        from ai_media.models import MODEL_REQUIREMENTS
        result = []
        for name, model_id in model_dict.items():
            if name == "default":
                continue
                
            # Enrich with requirements if available
            reqs = MODEL_REQUIREMENTS.get(model_id, {})
            model_info = {
                "name": name,
                "model_id": model_id,
                "category": category,
                "is_default": model_dict.get("default") == model_id,
            }
            # Add optional fields if present in requirements
            if "vram" in reqs:
                model_info["vram_required"] = reqs["vram"]
            if "ram" in reqs:
                model_info["ram_required"] = reqs["ram"]
            if "max_resolution" in reqs:
                model_info["max_resolution"] = reqs["max_resolution"]
            if "max_duration" in reqs:
                model_info["max_duration"] = reqs["max_duration"]
            result.append(model_info)
        return result
    
    return {
        "image": format_models(IMAGE_MODELS, "image"),
        "video": format_models(VIDEO_MODELS, "video"),
        "audio": format_models(AUDIO_MODELS, "audio"),
        "text": format_models(TEXT_MODELS, "text"),
    }


@router.get("/api/models/{category}")
async def get_models_by_category(category: str):
    """Get models for a specific category."""
    from ai_media.models import IMAGE_MODELS, VIDEO_MODELS, AUDIO_MODELS, TEXT_MODELS
    
    category_map = {
        "image": IMAGE_MODELS,
        "video": VIDEO_MODELS,
        "audio": AUDIO_MODELS,
        "text": TEXT_MODELS,
    }
    
    if category not in category_map:
        raise HTTPException(status_code=404, detail=f"Unknown category: {category}")
    
    models = category_map[category]
    from ai_media.models import MODEL_REQUIREMENTS
    return [
        {
            "name": name,
            "model_id": model_id,
            "vram_required": MODEL_REQUIREMENTS.get(model_id, {}).get("vram"),
            "ram_required": MODEL_REQUIREMENTS.get(model_id, {}).get("ram"),
            "max_resolution": MODEL_REQUIREMENTS.get(model_id, {}).get("max_resolution"),
        }
        for name, model_id in models.items() if name != "default"
    ]


# --- Cache Management ---

@router.get("/api/cache")
async def get_cache_status():
    """Get current model cache status."""
    return {
        "cached_models": model_cache.get_status(),
        "message": "Models will be reused if same model is requested again"
    }


@router.delete("/api/cache")
async def clear_cache():
    """Unload all cached models to free memory."""
    model_cache.unload_all()
    return {"message": "All cached models unloaded"}


@router.delete("/api/cache/{category}")
async def clear_cache_category(category: str):
    """Unload a specific model category (text, image, audio, video)."""
    print(f"🧹 Unloading {category} model via API...")
    model_cache.unload(category)
    return {"message": f"Unloaded {category} model"}
