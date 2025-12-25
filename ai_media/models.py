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
    "sdxl": "stabilityai/sdxl-turbo",                  # Fast, good quality (DEFAULT)
    "sd-1.5": "runwayml/stable-diffusion-v1-5",        # Classic, lightweight
    "upscaler": "stabilityai/stable-diffusion-x4-upscaler",  # 4x Upscaling
    "upscaler_x2": "stabilityai/sd-x2-latent-upscaler",      # 2x Latent Upscaling
    "default": "stabilityai/sdxl-turbo"
}

# --- Edit/Transform Models ---
EDIT_MODELS = {
    "instruct-pix2pix": "timbrooks/instruct-pix2pix",
    "instruct-pix2pix-sdxl": "diffusers/sdxl-instructpix2pix-768",
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
    "wan2.2": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",      # Alibaba Wan 2.2 (14B)
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
    # Established models
    "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistral-nemo-12b": "mistralai/Mistral-Nemo-Instruct-2407",
    # Default: Llama 3.1 (stable on all platforms)
    "default": "meta-llama/Meta-Llama-3.1-8B-Instruct"
}

# --- Caption/Description Models ---
CAPTION_MODELS = {
    "florence": "microsoft/Florence-2-large",
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
    "briaai/RMBG-1.4": {"vram": 4, "ram": 8, "max_resolution": (2048, 2048)},
    
    # Text Models
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {"vram": 16, "ram": 24, "max_resolution": None},
    "mistralai/Mistral-Nemo-Instruct-2407": {"vram": 24, "ram": 32, "max_resolution": None},
    "Qwen/Qwen2.5-14B-Instruct": {"vram": 28, "ram": 48, "max_resolution": None},
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
