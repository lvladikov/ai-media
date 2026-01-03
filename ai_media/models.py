"""
Model mappings and resource requirements for AI-Media.

Short codes are mapped to Hugging Face Hub IDs.
These are lightweight dictionaries that can be imported without loading heavy ML libraries.
"""

# --- Image Models ---
IMAGE_MODELS = {
    "flux": "black-forest-labs/FLUX.1-schnell",        # State of the art, fast
    "flux-dev": "black-forest-labs/FLUX.1-dev",        # Higher quality, slower
    "flux2": "diffusers/FLUX.2-dev-bnb-4bit",          # FLUX 2 (4-bit quantized for consumer GPUs)
    "flux2-full": "black-forest-labs/FLUX.2-dev",      # FLUX 2 (Full model, 90GB+ VRAM required)
    "sdxl": "stabilityai/sdxl-turbo",                  # Fast, good quality
    "sd-1.5": "runwayml/stable-diffusion-v1-5",        # Classic, lightweight
    "sd3.5-medium": "stabilityai/stable-diffusion-3.5-medium",    # SD 3.5 Medium (consumer-friendly)
    "sd3.5-large": "stabilityai/stable-diffusion-3.5-large",       # SD 3.5 Large (best quality)
    "sd3.5-turbo": "stabilityai/stable-diffusion-3.5-large-turbo", # SD 3.5 Turbo (fast, 4 steps) (DEFAULT)
    "qwen-image": "ovedrive/qwen-image-4bit",                      # Qwen-Image 4-bit (CUDA, 20GB)
    "qwen-image-auto": "ovedrive/qwen-image-4bit",                 # Auto alias (defaults to CUDA 4-bit ID, logic switches on MPS)
    "qwen-image-4bit": "ovedrive/qwen-image-4bit",                 # Explicit 4-bit alias
    "qwen-image-lightning": "lightx2v/Qwen-Image-2512-Lightning",  # Qwen-Image Lightning (Fast 8-step, Works on MPS)
    "qwen-image-2512": "Qwen/Qwen-Image",                          # Qwen-Image 2512 (Latest) (MPS, ~40GB RAM)
    "upscaler": "stabilityai/stable-diffusion-x4-upscaler",  # 4x Upscaling
    "upscaler_x2": "stabilityai/sd-x2-latent-upscaler",      # 2x Latent Upscaling
    "default": "stabilityai/stable-diffusion-3.5-large-turbo"
}

# --- Edit/Transform Models ---
EDIT_MODELS = {
    "instruct-pix2pix": "timbrooks/instruct-pix2pix",
    "instruct-pix2pix-sdxl": "diffusers/sdxl-instructpix2pix-768",
    "qwen-image-edit": "Qwen/Qwen-Image-Edit-2511",         # Qwen-Image-Edit (CUDA, 4-bit)
    "qwen-image-edit-mps": "Qwen/Qwen-Image-Edit-2511",     # Qwen-Image-Edit (MPS, float32)
    # The 'Lightning' 2512 model is hosted by lightx2v and uses LoRA/distillation
    "qwen-image-edit-lightning": "lightx2v/Qwen-Image-Edit-2512-Lightning",
    "remove-bg": "briaai/RMBG-1.4",
    "default": "timbrooks/instruct-pix2pix"
}

# --- Audio Models ---
AUDIO_MODELS = {
    "musicgen-small": "facebook/musicgen-small",       # Fast, good for music
    "musicgen-medium": "facebook/musicgen-medium",     # Better quality music
    "musicgen-large": "facebook/musicgen-large",       # Best quality music
    "audioldm2": "cvssp/audioldm2",                    # General audio/SFX
    "stable-audio": "stabilityai/stable-audio-open-1.0",  # Variable length, high quality (Gated)
    "bark": "suno/bark",                               # TTS / Audio (Transformer)
    "default": "facebook/musicgen-medium"
}

# --- Video Models ---
VIDEO_MODELS = {
    "ms-1.7b": "damo-vilab/text-to-video-ms-1.7b",     # Has watermark issues
    "zeroscope": "cerspense/zeroscope_v2_576w",        # 576x320 optimized (default)
    "zeroscope-xl": "cerspense/zeroscope_v2_XL",       # 1024x576 V2V upscaler (internal use)
    "cogvideox": "THUDM/CogVideoX-5b",                 # High quality (requires high VRAM)
    "wan-2.2": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",     # Alibaba Wan 2.2 (14B)
    "wan2.2": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",      # (Alias)
    "ltx-video": "Lightricks/LTX-Video",               # Lightricks LTX-Video (Fast, High Res)
    "mochi-1": "genmo/mochi-1-preview",                # Mochi 1 (Physics/Motion SOTA)
    "hunyuan": "hunyuanvideo-community/HunyuanVideo",  # HunyuanVideo (13B, Cinematic)
    "svd": "stabilityai/stable-video-diffusion-img2vid-xt",  # SVD Image-to-Video
    "default": "cerspense/zeroscope_v2_576w"
}

# --- Text Models ---
TEXT_MODELS = {
    # Reasoning-focused (Chain-of-Thought) - DeepSeek R1 Distilled
    # These show step-by-step reasoning before the final answer
    "deepseek-r1-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",      # ~7GB VRAM
    "deepseek-r1-qwen-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",    # ~14GB VRAM
    "deepseek-r1-qwen-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",    # ~24GB VRAM
    "deepseek-r1-llama-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",    # ~8GB VRAM
    "deepseek-r1-llama-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",  # ~40GB VRAM (requires high-end GPU)
    # General-purpose (Newer knowledge cutoffs)
    "qwen3-8b": "Qwen/Qwen3-8B",  # Note: May have MPS issues on Apple Silicon
    "qwen-2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    # Coding-specific models
    "qwen-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",    # Qwen 2.5 SOTA Code Gen (~24GB VRAM, 120GB RAM)
    "qwen-coder-14b": "Qwen/Qwen2.5-Coder-14B-Instruct",    # Qwen 2.5 Fast & Capable (~12GB VRAM)
    "qwen-coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",      # Qwen 2.5 Lightweight (~6GB VRAM)
    "qwen3-coder-30b": "Qwen/Qwen3-Coder-30B-A3B-Instruct", # MoE (3.3B active, ~10GB VRAM)
    # Established models
    "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistral-nemo-12b": "mistralai/Mistral-Nemo-Instruct-2407",
    # Default: Llama 3.1 (stable on all platforms)
    "default": "meta-llama/Meta-Llama-3.1-8B-Instruct"
}

# --- Caption/Description Models ---
CAPTION_MODELS = {
    "florence": "microsoft/Florence-2-large",
    # Qwen3-VL series (latest, better compatibility)
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",     # 8B, best quality
    "qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Instruct",     # 4B, balanced
    "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",     # 2B, lightweight
    "qwen-vl": "Qwen/Qwen3-VL-8B-Instruct",         # Alias (now points to Qwen3)
    "blip": "Salesforce/blip-image-captioning-large",
    "default": "microsoft/Florence-2-large"
}

# --- Model Resource Requirements ---
# Estimated RAM/VRAM in GB
# Based on model training specs, Hugging Face model cards, and practical testing.
# Format: { model_id: { "vram": X, "ram": Y, "max_resolution": (W, H) or None } }
MODEL_REQUIREMENTS = {
    # Image Models (max_resolution based on training data and VRAM constraints)
    "runwayml/stable-diffusion-v1-5": {"vram": 4, "ram": 8, "max_resolution": (1280, 1280)},
    "stabilityai/sdxl-turbo": {"vram": 8, "ram": 16, "max_resolution": (1536, 1536)},
    "black-forest-labs/FLUX.1-schnell": {"vram": 16, "ram": 70, "max_resolution": (2048, 2048)},
    "black-forest-labs/FLUX.1-dev": {"vram": 24, "ram": 80, "max_resolution": (2048, 2048)},
    "diffusers/FLUX.2-dev-bnb-4bit": {"vram": 20, "ram": 32, "max_resolution": (4096, 4096)},
    "black-forest-labs/FLUX.2-dev": {"vram": 90, "ram": 120, "max_resolution": (4096, 4096)},
    "stabilityai/stable-diffusion-3.5-medium": {"vram": 10, "ram": 24, "max_resolution": (1296, 1296)},
    "stabilityai/stable-diffusion-3.5-large": {"vram": 19, "ram": 40, "max_resolution": (1296, 1296)},
    "stabilityai/stable-diffusion-3.5-large-turbo": {"vram": 19, "ram": 40, "max_resolution": (1296, 1296)},
    "ovedrive/qwen-image-4bit": {"vram": 20, "ram": 32, "max_resolution": (1664, 1664)},
    "lightx2v/Qwen-Image-2512-Lightning": {"vram": 40, "ram": 80, "max_resolution": (1664, 1664)}, # Base model size
    "Qwen/Qwen-Image": {"vram": 40, "ram": 80, "max_resolution": (1664, 1664)}, # Covers qwen-image-2512 too
    
    # Audio Models (max_duration in seconds, based on model architecture limits)
    "facebook/musicgen-small": {"vram": 4, "ram": 8, "max_duration": 30},
    "facebook/musicgen-medium": {"vram": 8, "ram": 12, "max_duration": 60},
    "facebook/musicgen-large": {"vram": 16, "ram": 24, "max_duration": 120},
    "cvssp/audioldm2": {"vram": 8, "ram": 12, "max_duration": 60},
    "stabilityai/stable-audio-open-1.0": {"vram": 10, "ram": 16, "max_duration": 47},
    "suno/bark": {"vram": 4, "ram": 12, "max_duration": 30},
    
    # Video Models (max_resolution based on training data)
    "damo-vilab/text-to-video-ms-1.7b": {"vram": 12, "ram": 16, "max_resolution": (1280, 720)},
    "cerspense/zeroscope_v2_576w": {"vram": 8, "ram": 12, "max_resolution": (576, 320)},
    "cerspense/zeroscope_v2_XL": {"vram": 10, "ram": 16, "max_resolution": (1024, 576)},
    "THUDM/CogVideoX-5b": {"vram": 32, "ram": 48, "max_resolution": (1920, 1080)},
    "stabilityai/stable-video-diffusion-img2vid-xt": {"vram": 8, "ram": 12, "max_resolution": (1024, 576)},
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": {"vram": 24, "ram": 64, "max_resolution": (1280, 720)},
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers": {"vram": 24, "ram": 64, "max_resolution": (1280, 720)},
    "Lightricks/LTX-Video": {"vram": 16, "ram": 32, "max_resolution": (1216, 704)},
    "genmo/mochi-1-preview": {"vram": 19, "ram": 48, "max_resolution": (848, 480)},
    "hunyuanvideo-community/HunyuanVideo": {"vram": 24, "ram": 64, "max_resolution": (1280, 720)},
    "hunyuanvideo-community/HunyuanVideo-I2V": {"vram": 24, "ram": 64, "max_resolution": (1280, 720)},
    
    # Upscaling Models
    "stabilityai/stable-diffusion-x4-upscaler": {"vram": 8, "ram": 16, "max_resolution": (4096, 4096)},
    "stabilityai/sd-x2-latent-upscaler": {"vram": 4, "ram": 8, "max_resolution": (2048, 2048)},
    
    # Edit/Transform Models
    "timbrooks/instruct-pix2pix": {"vram": 8, "ram": 12, "max_resolution": (1024, 1024)},
    "diffusers/sdxl-instructpix2pix-768": {"vram": 10, "ram": 16, "max_resolution": (1024, 1024)},
    "Qwen/Qwen-Image-Edit-2511": {"vram": 20, "ram": 40, "max_resolution": (1664, 1664)},
    "lightx2v/Qwen-Image-Edit-2512-Lightning": {"vram": 16, "ram": 32, "max_resolution": (1664, 1664)},
    "briaai/RMBG-1.4": {"vram": 4, "ram": 8, "max_resolution": (2048, 2048)},
    
    # Text Models
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {"vram": 16, "ram": 24, "max_resolution": None},
    "mistralai/Mistral-Nemo-Instruct-2407": {"vram": 24, "ram": 32, "max_resolution": None},
    "Qwen/Qwen2.5-14B-Instruct": {"vram": 28, "ram": 48, "max_resolution": None},
    "Qwen/Qwen2.5-Coder-32B-Instruct": {"vram": 24, "ram": 120, "max_resolution": None},  # CUDA only, needs 120GB+ RAM on MPS
    "Qwen/Qwen2.5-Coder-14B-Instruct": {"vram": 12, "ram": 30, "max_resolution": None},
    "Qwen/Qwen2.5-Coder-7B-Instruct": {"vram": 6, "ram": 16, "max_resolution": None},
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": {"vram": 10, "ram": 20, "max_resolution": None},  # MoE, 3.3B active
    # Vision-Language Models
    "Qwen/Qwen3-VL-8B-Instruct": {"vram": 16, "ram": 28, "max_resolution": None},
    "Qwen/Qwen3-VL-4B-Instruct": {"vram": 8, "ram": 16, "max_resolution": None},
    "Qwen/Qwen3-VL-2B-Instruct": {"vram": 4, "ram": 8, "max_resolution": None},
}


def get_model_id(model_name, model_dict):
    """
    Get the full Hugging Face model ID from a short code.
    
    Args:
        model_name: Short code (e.g., 'sdxl') or full HF ID
        model_dict: One of IMAGE_MODELS, VIDEO_MODELS, etc.
    
    Returns:
        Full Hugging Face model ID
    """
    model_id = model_dict.get(model_name.lower(), model_name)
    if model_name.lower() == "default":
        model_id = model_dict["default"]
    return model_id
