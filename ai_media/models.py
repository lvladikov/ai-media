"""
Model mappings and resource requirements for AI-Media.

Short codes are mapped to Hugging Face Hub IDs.
These are lightweight dictionaries that can be imported without loading heavy ML libraries.
"""

# Complete NLLB/FLORES-200 language codes (200+ languages)
# Source: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
ALL_NLLB_CODES = [
    "ace_Arab", "ace_Latn", "acm_Arab", "acq_Arab", "aeb_Arab", "afr_Latn", "ajp_Arab",
    "aka_Latn", "amh_Ethi", "apc_Arab", "arb_Arab", "arb_Latn", "ars_Arab", "ary_Arab",
    "arz_Arab", "asm_Beng", "ast_Latn", "awa_Deva", "ayr_Latn", "azb_Arab", "azj_Latn",
    "bak_Cyrl", "bam_Latn", "ban_Latn", "bel_Cyrl", "bem_Latn", "ben_Beng", "bho_Deva",
    "bjn_Arab", "bjn_Latn", "bod_Tibt", "bos_Latn", "bug_Latn", "bul_Cyrl", "cat_Latn",
    "ceb_Latn", "ces_Latn", "cjk_Latn", "ckb_Arab", "crh_Latn", "cym_Latn", "dan_Latn",
    "deu_Latn", "dik_Latn", "dyu_Latn", "dzo_Tibt", "ell_Grek", "eng_Latn", "epo_Latn",
    "est_Latn", "eus_Latn", "ewe_Latn", "fao_Latn", "fij_Latn", "fin_Latn", "fon_Latn",
    "fra_Latn", "fur_Latn", "fuv_Latn", "gaz_Latn", "gla_Latn", "gle_Latn", "glg_Latn",
    "grn_Latn", "guj_Gujr", "hat_Latn", "hau_Latn", "heb_Hebr", "hin_Deva", "hne_Deva",
    "hrv_Latn", "hun_Latn", "hye_Armn", "ibo_Latn", "ilo_Latn", "ind_Latn", "isl_Latn",
    "ita_Latn", "jav_Latn", "jpn_Jpan", "kab_Latn", "kac_Latn", "kam_Latn", "kan_Knda",
    "kas_Arab", "kas_Deva", "kat_Geor", "kaz_Cyrl", "kbp_Latn", "kea_Latn", "khk_Cyrl",
    "khm_Khmr", "kik_Latn", "kin_Latn", "kir_Cyrl", "kmb_Latn", "kmr_Latn", "knc_Arab",
    "knc_Latn", "kon_Latn", "kor_Hang", "lao_Laoo", "lij_Latn", "lim_Latn", "lin_Latn",
    "lit_Latn", "lmo_Latn", "ltg_Latn", "ltz_Latn", "lua_Latn", "lug_Latn", "luo_Latn",
    "lus_Latn", "lvs_Latn", "mag_Deva", "mai_Deva", "mal_Mlym", "mar_Deva", "min_Arab",
    "min_Latn", "mkd_Cyrl", "plt_Latn", "mlt_Latn", "mni_Beng", "mos_Latn", "mri_Latn",
    "mya_Mymr", "nld_Latn", "nno_Latn", "nob_Latn", "npi_Deva", "nso_Latn", "nus_Latn",
    "nya_Latn", "oci_Latn", "ory_Orya", "pag_Latn", "pan_Guru", "pap_Latn", "pbt_Arab",
    "pes_Arab", "pol_Latn", "por_Latn", "prs_Arab", "quy_Latn", "ron_Latn", "run_Latn",
    "rus_Cyrl", "sag_Latn", "san_Deva", "sat_Olck", "scn_Latn", "shn_Mymr", "sin_Sinh",
    "slk_Latn", "slv_Latn", "smo_Latn", "sna_Latn", "snd_Arab", "som_Latn", "sot_Latn",
    "spa_Latn", "als_Latn", "srd_Latn", "srp_Cyrl", "ssw_Latn", "sun_Latn", "swe_Latn",
    "swh_Latn", "szl_Latn", "tam_Taml", "taq_Latn", "taq_Tfng", "tat_Cyrl", "tel_Telu",
    "tgk_Cyrl", "tgl_Latn", "tha_Thai", "tir_Ethi", "tpi_Latn", "tsn_Latn", "tso_Latn",
    "tuk_Latn", "tum_Latn", "tur_Latn", "twi_Latn", "tzm_Tfng", "uig_Arab", "ukr_Cyrl",
    "umb_Latn", "urd_Arab", "uzn_Latn", "vec_Latn", "vie_Latn", "war_Latn", "wol_Latn",
    "xho_Latn", "ydd_Hebr", "yor_Latn", "yue_Hant", "zho_Hans", "zho_Hant", "zsm_Latn",
    "zul_Latn",
]

def get_nllb_language_name(nllb_code: str) -> str:
    """
    Get human-readable language name from NLLB code using pycountry.
    
    NLLB codes follow format: {iso639_3}_{script} (e.g., 'eng_Latn', 'fra_Latn')
    Falls back to code itself if language not found.
    Uses caching for fast repeated lookups.
    """
    # Check cache first
    if nllb_code in _language_name_cache:
        return _language_name_cache[nllb_code]
    
    try:
        import pycountry
        # Extract the ISO 639-3 code (first 3 chars)
        iso_code = nllb_code.split("_")[0]
        
        # Try ISO 639-3 first
        lang = pycountry.languages.get(alpha_3=iso_code)
        if lang:
            name = lang.name
        else:
            # Fallback: try ISO 639-2 bibliographic code
            lang = pycountry.languages.get(bibliographic=iso_code)
            if lang:
                name = lang.name
            else:
                # Fallback: return the code capitalized
                name = iso_code.capitalize()
    except ImportError:
        # pycountry not installed, return code
        name = nllb_code.split("_")[0].upper()
    except Exception:
        name = nllb_code.split("_")[0].upper()
    
    # Cache and return
    _language_name_cache[nllb_code] = name
    return name

# Cache for language names
_language_name_cache = {}
_cached_languages_with_names = None


def get_nllb_languages_with_names():
    """
    Get list of all NLLB languages with human-readable names.
    Returns: List of (display_name, nllb_code) tuples, sorted alphabetically.
    Uses caching for fast repeated calls.
    """
    global _cached_languages_with_names
    
    if _cached_languages_with_names is not None:
        return _cached_languages_with_names
    
    languages = []
    for code in ALL_NLLB_CODES:
        name = get_nllb_language_name(code)
        # Add script info for codes with same language but different scripts
        script = code.split("_")[1] if "_" in code else ""
        if script and script not in ["Latn", "Arab", "Cyrl"]:
            # Add script for non-Latin/Arabic/Cyrillic scripts
            name = f"{name} ({script})"
        languages.append((name, code))
    
    # Sort by name and cache
    _cached_languages_with_names = sorted(languages, key=lambda x: x[0])
    return _cached_languages_with_names


# Short code mapping (for backward compatibility with existing code)
NLLB_LANGUAGE_CODES = {
    "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "ru": "rus_Cyrl", "zh": "zho_Hans",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "ar": "arb_Arab", "hi": "hin_Deva",
    "nl": "nld_Latn", "pl": "pol_Latn", "tr": "tur_Latn", "sv": "swe_Latn",
    "no": "nob_Latn", "da": "dan_Latn", "fi": "fin_Latn", "el": "ell_Grek",
    "he": "heb_Hebr", "th": "tha_Thai", "vi": "vie_Latn", "id": "ind_Latn",
    "bg": "bul_Cyrl", "uk": "ukr_Cyrl", "cs": "ces_Latn", "ro": "ron_Latn",
    "hu": "hun_Latn", "sk": "slk_Latn", "hr": "hrv_Latn", "sr": "srp_Cyrl",
    "sl": "slv_Latn", "et": "est_Latn", "lv": "lvs_Latn", "lt": "lit_Latn",
    "mk": "mkd_Cyrl", "sq": "als_Latn", "bs": "bos_Latn", "mt": "mlt_Latn",
    "is": "isl_Latn", "ga": "gle_Latn", "cy": "cym_Latn", "af": "afr_Latn",
    "sw": "swh_Latn", "bn": "ben_Beng", "ta": "tam_Taml", "te": "tel_Telu",
    "ml": "mal_Mlym", "kn": "kan_Knda", "mr": "mar_Deva", "gu": "guj_Gujr",
    "pa": "pan_Guru", "ur": "urd_Arab", "fa": "pes_Arab", "ms": "zsm_Latn",
    "tl": "tgl_Latn", "my": "mya_Mymr", "km": "khm_Khmr", "lo": "lao_Laoo",
    "ne": "npi_Deva", "si": "sin_Sinh", "ka": "kat_Geor", "hy": "hye_Armn",
    "az": "azj_Latn", "kk": "kaz_Cyrl", "uz": "uzn_Latn", "mn": "khk_Cyrl",
}

# --- Image Models ---

IMAGE_MODELS = {
    "flux": "black-forest-labs/FLUX.1-schnell",        # State of the art, fast
    "flux-dev": "black-forest-labs/FLUX.1-dev",        # Higher quality, slower
    "flux2": "black-forest-labs/FLUX.2-dev",
    "flux2-full": "black-forest-labs/FLUX.2-pro",
    "z-image": "Tongyi-MAI/Z-Image-Turbo",             # 6B params, fast (9 steps)
    "sdxl": "stabilityai/sdxl-turbo",                  # Fast, good quality
    "sd-1.5": "stable-diffusion-v1-5/stable-diffusion-v1-5",        # Classic, lightweight (Mirror)
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
    "z-image-edit": "Tongyi-MAI/Z-Image-Turbo",              # Z-Image Turbo for high-speed editing
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
    "qwen3-8b": "Qwen/Qwen3-8B",  # Qwen 3 (Base/Instruct hybrid)
    "qwen3-14b": "Qwen/Qwen3-14B", # Qwen 3 (14B)
    # Coding-specific models
    "qwen-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",    # Qwen 2.5 SOTA Code Gen (~24GB VRAM, 120GB RAM)
    "qwen-coder-14b": "Qwen/Qwen2.5-Coder-14B-Instruct",    # Qwen 2.5 Fast & Capable (~12GB VRAM)
    "qwen-coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",      # Qwen 2.5 Lightweight (~6GB VRAM)
    "qwen3-coder-30b": "Qwen/Qwen3-Coder-30B-A3B-Instruct",   # Qwen 3 MoE Coder (3.3B active, ~2GB)
    # Established models
    "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistral-nemo-12b": "mistralai/Mistral-Nemo-Instruct-2407",
    # Qwen 3 (Opus 4.5 Distill)
    "qwen3-opus-4.5-8b": "TeichAI/Qwen3-8B-Claude-4.5-Opus-High-Reasoning-Distill",
    "qwen3-opus-4.5-14b": "TeichAI/Qwen3-14B-Claude-4.5-Opus-High-Reasoning-Distill",
    # Qwen 3 (GPT-5.2 Distill)
    "qwen3-gpt-5.2-8b": "TeichAI/Qwen3-8B-GPT-5.2-High-Reasoning-Distill",
    "qwen3-gpt-5.2-14b": "TeichAI/Qwen3-14B-GPT-5.2-High-Reasoning-Distill",
    # Default: Llama 3.1 (stable on all platforms)
    "default": "meta-llama/Meta-Llama-3.1-8B-Instruct"
}

# --- Caption/Description Models ---
CAPTION_MODELS = {
    "florence": "microsoft/Florence-2-large",
    # Qwen2-VL series (latest, better compatibility)
    "qwen3-vl-8b": "Qwen/Qwen2-VL-7B-Instruct",     # 7B, best quality
    "qwen3-vl-4b": "Qwen/Qwen2-VL-7B-Instruct",     # Fallback to 7B (no 4B in 2.5) or 2B
    "qwen3-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",     # 2B, lightweight
    "qwen-vl": "Qwen/Qwen2-VL-7B-Instruct",         # Alias
    "blip": "Salesforce/blip-image-captioning-large",
    "default": "microsoft/Florence-2-large"
}

# --- Translation Models ---
TRANSLATION_MODELS = {
    # Specialized Translation
    "seamless-m4t-v2-large": "facebook/seamless-m4t-v2-large", # SOTA Speech/Text Translation
    "nllb-200-3.3b": "facebook/nllb-200-3.3B",                   # High quality text translation
    "nllb-200-distilled": "facebook/nllb-200-distilled-600M",    # Fast text translation
    
    "qwen3-8b": "Qwen/Qwen3-8B", # Qwen 3 (Base/Instruct hybrid)
    "qwen3-14b": "Qwen/Qwen3-14B", # Qwen 3 (14B)
    "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "alma-13b": "haoranxu/ALMA-13B-R",  # Specialized translation model
    
    "default_audio": "facebook/seamless-m4t-v2-large",
    "default_text": "facebook/nllb-200-3.3B"
}

# --- Model Resource Requirements ---
# Estimated RAM/VRAM in GB
# Based on model training specs, Hugging Face model cards, and practical testing.
# Format: { model_id: { "vram": X, "ram": Y, "max_resolution": (W, H) or None } }
MODEL_REQUIREMENTS = {
    # Image Models (max_resolution based on training data and VRAM constraints)
    "stable-diffusion-v1-5/stable-diffusion-v1-5": {"vram": 4, "ram": 8, "max_resolution": (1280, 1280)},
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
    "Tongyi-MAI/Z-Image-Turbo": {"vram": 16, "ram": 48, "max_resolution": (2048, 2048)}, # Alibaba Z-Image, 6B params
    
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
    # ZImage has already been listed under Image Models and it;s the same name for Transforms
    
    # Text Models
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {"vram": 16, "ram": 24, "max_resolution": None},
    "mistralai/Mistral-Nemo-Instruct-2407": {"vram": 24, "ram": 32, "max_resolution": None},
    "Qwen/Qwen2.5-14B-Instruct": {"vram": 28, "ram": 48, "max_resolution": None},
    "Qwen/Qwen2.5-Coder-32B-Instruct": {"vram": 24, "ram": 120, "max_resolution": None},  # CUDA only, needs 120GB+ RAM on MPS
    "Qwen/Qwen2.5-Coder-14B-Instruct": {"vram": 12, "ram": 30, "max_resolution": None},
    "Qwen/Qwen2.5-Coder-7B-Instruct": {"vram": 6, "ram": 16, "max_resolution": None},
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": {"vram": 10, "ram": 20, "max_resolution": None},  # MoE, 3.3B active
    # Vision-Language Models
    "Qwen/Qwen2-VL-7B-Instruct": {"vram": 16, "ram": 28, "max_resolution": None},
    "Qwen/Qwen2-VL-2B-Instruct": {"vram": 4, "ram": 8, "max_resolution": None},

    # Translation Models
    "facebook/seamless-m4t-v2-large": {"vram": 10, "ram": 16, "max_duration": 60}, # ~4.6B params
    "facebook/nllb-200-3.3B": {"vram": 8, "ram": 16, "max_resolution": None},
    "facebook/nllb-200-distilled-600M": {"vram": 4, "ram": 8, "max_resolution": None},
    "Qwen/Qwen2.5-7B-Instruct": {"vram": 16, "ram": 24, "max_resolution": None},
    "Qwen/Qwen3-8B": {"vram": 16, "ram": 24, "max_resolution": None},
    "Qwen/Qwen3-14B": {"vram": 28, "ram": 48, "max_resolution": None},
    "haoranxu/ALMA-13B-R": {"vram": 26, "ram": 48, "max_resolution": None}, # Based on LLaMA-2-13b base

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


# --- MLX Community Model Mappings ---
# Maps standard Hugging Face model IDs to optimized/quantized MLX Community versions.
# This prevents downloading full weights and quantizing on-the-fly.
MLX_MODEL_MAPPINGS = {
    # DeepSeek R1
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": {
        "int4": "mlx-community/DeepSeek-R1-Distill-Llama-70B-4bit",
        "int6": "mlx-community/DeepSeek-R1-Distill-Llama-70B-6bit",
        "int8": "mlx-community/DeepSeek-R1-Distill-Llama-70B-8bit",
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "int4": "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit",
        "int6": "mlx-community/DeepSeek-R1-Distill-Llama-8B-6bit",
        "int8": "mlx-community/DeepSeek-R1-Distill-Llama-8B-8bit",
        "bfloat16": "mlx-community/DeepSeek-R1-Distill-Llama-8B-bf16",
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {
        "int4": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",
        "int6": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-6bit",
        "int8": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-8bit",
        "bfloat16": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-bf16",
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
        "int4": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",
        "int6": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-6bit",
        "int8": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-8bit",
        "bfloat16": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-bf16",
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "int4": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
        "int6": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-6bit",
        "int8": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-8bit",
        "bfloat16": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-bf16",
    },
    
    # Image Models (Flux Mappings for mflux)
    # mflux.Flux1.from_name expects aliases "schnell", "dev" or HF repo IDs
    "black-forest-labs/FLUX.1-schnell": {
        "default": "schnell",
        "int4": "schnell", # mflux handles quant flag separately, but we map here for consistency
        "bf16": "schnell",
    },
    "black-forest-labs/FLUX.1-dev": {
        "default": "dev", 
        "int4": "dev",
        "bf16": "dev",
    },
    
    # Llama 3.1
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "int4": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "int6": "mlx-community/Meta-Llama-3.1-8B-Instruct-6bit",
        "int8": "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
        "bfloat16": "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16",
    },
    
    # Qwen 2.5 Coding
    "Qwen/Qwen2.5-Coder-32B-Instruct": {
        "int4": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
        "int6": "mlx-community/Qwen2.5-Coder-32B-Instruct-6bit",
        "int8": "mlx-community/Qwen2.5-Coder-32B-Instruct-8bit",
        "bfloat16": "mlx-community/Qwen2.5-Coder-32B-Instruct-bf16",
    },
    
    # Qwen Image
    "Qwen/Qwen-Image": {
        "default": "Qwen/Qwen-Image", # mflux supports HF repo directly
        "int4": "Qwen/Qwen-Image",    # quantization applied on load by mflux
    },
    "ovedrive/qwen-image-4bit": {
        "default": "Qwen/Qwen-Image", # Map to base model for proper mflux loading
        "int4": "Qwen/Qwen-Image",
        "int8": "Qwen/Qwen-Image",
    },
    
    # Z-Image (Alibaba/Tongyi)
    # mflux's ZImageTurbo class is for Z-Image
    "Tongyi-MAI/Z-Image-Turbo": {
        "default": "filipstrand/Z-Image-Turbo-mflux-4bit",
        "int4": "filipstrand/Z-Image-Turbo-mflux-4bit",
    },

    "Qwen/Qwen2.5-Coder-14B-Instruct": {
        "int4": "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
        "int6": "mlx-community/Qwen2.5-Coder-14B-Instruct-6bit",
        "int8": "mlx-community/Qwen2.5-Coder-14B-Instruct-8bit",
    },
    "Qwen/Qwen2.5-Coder-7B-Instruct": {
        "int4": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        "int6": "mlx-community/Qwen2.5-Coder-7B-Instruct-6bit",
        "int8": "mlx-community/Qwen2.5-Coder-7B-Instruct-8bit",
    },
    
    # Qwen 3 (Base/Instruct)
    "Qwen/Qwen3-8B": {
        "int4": "mlx-community/Qwen3-8B-4bit",
        "int6": "mlx-community/Qwen3-8B-6bit",
        "int8": "mlx-community/Qwen3-8B-8bit",
        "bfloat16": "mlx-community/Qwen3-8B-bf16",
    },
    "Qwen/Qwen3-14B": {
        "int4": "mlx-community/Qwen3-14B-4bit",
        "int6": "mlx-community/Qwen3-14B-6bit",
        "int8": "mlx-community/Qwen3-14B-8bit",
        "bfloat16": "mlx-community/Qwen3-14B-bf16",
    },
    
    # Qwen 3 Coder
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": {
        "int4": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
        "int6": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-6bit",
        "int8": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit",
    },
    
    # Mistral
    "mistralai/Mistral-Nemo-Instruct-2407": {
        "int4": "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
        "int8": "mlx-community/Mistral-Nemo-Instruct-2407-8bit",
        "bfloat16": "mlx-community/Mistral-Nemo-Instruct-2407-bf16",
    },
    
    # Qwen 3 Opus 4.5 Distill
    "TeichAI/Qwen3-8B-Claude-4.5-Opus-High-Reasoning-Distill": {
        "int4": "leonsarmiento/Qwen3-8B-Claude-4.5-Opus-High-Reasoning-Distill-8bit-mlx", # 4-bit not avail, fallback to 8-bit
        "int8": "leonsarmiento/Qwen3-8B-Claude-4.5-Opus-High-Reasoning-Distill-8bit-mlx",
    },

    "TeichAI/Qwen3-14B-Claude-4.5-Opus-High-Reasoning-Distill": {
         "int4": "leonsarmiento/Qwen3-14B-Claude-4.5-Opus-High-Reasoning-Distill-8bit-mlx", # Fallback to 8-bit
         "int8": "leonsarmiento/Qwen3-14B-Claude-4.5-Opus-High-Reasoning-Distill-8bit-mlx",
    },

    # Qwen 3 GPT-5.2 Distill
    "TeichAI/Qwen3-8B-GPT-5.2-High-Reasoning-Distill": {
        "int4": "nightmedia/Qwen3-8B-GPT-5.2-High-Reasoning-Distill-qx86-hi-mlx", # Mixed precision (6/8-bit usually)
        "int8": "nightmedia/Qwen3-8B-GPT-5.2-High-Reasoning-Distill-qx86-hi-mlx",
    },
    "TeichAI/Qwen3-14B-GPT-5.2-High-Reasoning-Distill": {
        "int4": "introvoyz041/Qwen3-14B-GPT-5.2-High-Reasoning-Distill-mlx-4Bit",
        "int8": "introvoyz041/Qwen3-14B-GPT-5.2-High-Reasoning-Distill-mlx-4Bit", # 4-bit only avail
    },
    
    # --- Expanded Image/Transform Mappings ---
    
    # Flux (mflux handles quantization via flags, but we map inputs for consistency)
    "black-forest-labs/FLUX.1-schnell": {
        "default": "schnell",
        "int4": "schnell", 
        "int8": "schnell",
        "float16": "schnell",
        "bfloat16": "schnell",
        "float32": "schnell",
    },
    "black-forest-labs/FLUX.1-dev": {
        "default": "dev", 
        "int4": "dev",
        "int8": "dev",
        "float16": "dev",
        "bfloat16": "dev",
        "float32": "dev",
    },
    "black-forest-labs/FLUX.2-pro": {
        "default": "pro", 
        "int4": "pro",
        "int6": "pro",
        "int8": "pro",
        "float16": "pro",
        "bfloat16": "pro",
        "float32": "pro",
    },
    
    # Z-Image (mflux)
    "Tongyi-MAI/Z-Image-Turbo": {
        "default": "filipstrand/Z-Image-Turbo-mflux-4bit",
        "int4": "filipstrand/Z-Image-Turbo-mflux-4bit",
        "int8": "Tongyi-MAI/Z-Image-Turbo",    # Fallback to base for higher precisions (mflux loads it)
        "float16": "Tongyi-MAI/Z-Image-Turbo",
        "bfloat16": "Tongyi-MAI/Z-Image-Turbo",
        "float32": "Tongyi-MAI/Z-Image-Turbo",
    },
    
    # Qwen Image (VL based)
    "Qwen/Qwen-Image": {
        "default": "Qwen/Qwen-Image", 
        "int4": "Qwen/Qwen-Image", 
        "int8": "Qwen/Qwen-Image",
        "float16": "Qwen/Qwen-Image",
        "bfloat16": "Qwen/Qwen-Image",
    },
    
    # Qwen Image Edit
    "Qwen/Qwen-Image-Edit-2511": {
        "default": "Qwen/Qwen-Image-Edit-2511", 
        "int4": "Qwen/Qwen-Image-Edit-2511",
        "int8": "Qwen/Qwen-Image-Edit-2511",
        "float16": "Qwen/Qwen-Image-Edit-2511",
        "bfloat16": "Qwen/Qwen-Image-Edit-2511",
    },
    # Qwen Image Edit Lightning
    "lightx2v/Qwen-Image-Edit-2512-Lightning": {
        "default": "lightx2v/Qwen-Image-Edit-2512-Lightning", 
        "int4": "lightx2v/Qwen-Image-Edit-2512-Lightning",
        "int8": "lightx2v/Qwen-Image-Edit-2512-Lightning",
        "float16": "lightx2v/Qwen-Image-Edit-2512-Lightning",
        "bfloat16": "lightx2v/Qwen-Image-Edit-2512-Lightning",
    }
}


def get_mlx_model_id(base_model_id: str, precision: str = "int4") -> str:
    """
    Resolve the best MLX-optimized model ID for the given base model and precision.
    
    Args:
        base_model_id: Hugging Face model ID
        precision: Targeted precision
        
    Returns:
        The mapped MLX Community ID if found, otherwise the original base_model_id.
    """
    if base_model_id in MLX_MODEL_MAPPINGS:
        variants = MLX_MODEL_MAPPINGS[base_model_id]
        if precision in variants:
            return variants[precision]
            
    return base_model_id
