"""
Model resource estimation utilities.

Provides functions to calculate RAM/VRAM requirements for AI models
based on their parameter count and selected precision/dtype.
"""

# Model parameter counts (in billions)
TEXT_MODEL_PARAMS = {
    # DeepSeek R1 Distilled
    "deepseek-r1-qwen-7b": 7,
    "deepseek-r1-qwen-14b": 14,
    "deepseek-r1-qwen-32b": 32,
    "deepseek-r1-llama-8b": 8,
    "deepseek-r1-llama-70b": 70,
    # Qwen 3
    "qwen3-8b": 8,
    "qwen3-14b": 14,
    "qwen3-opus-4.5-8b": 8,
    "qwen3-opus-4.5-14b": 14,
    "qwen3-gpt-5.2-8b": 8,
    "qwen3-gpt-5.2-14b": 14,
    # Qwen Coders
    "qwen-coder-32b": 32,
    "qwen-coder-14b": 14,
    "qwen-coder-7b": 7,
    "qwen3-coder-30b": 3.3,  # MoE architecture, only 3.3B active params
    # General
    "llama-3.1-8b": 8,
    "mistral-nemo-12b": 12,
    # Vision-Language
    "qwen-vl": 8,
    "qwen3-vl-8b": 8,
    "qwen3-vl-4b": 4,
    "qwen3-vl-2b": 2,
}

# Image parameter counts (in billions)
IMAGE_MODEL_PARAMS = {
    'sd3.5-turbo': 8,      # ~19GB total
    'sd3.5-medium': 4,     # ~10GB
    'sd3.5-large': 8,      # ~19GB
    'sdxl': 7,             # ~8-9GB
    'sd-1.5': 1.5,         # ~4GB
    'z-image': 6,          # ~8GB
    'qwen-image': 7,       # ~20-40GB
    'qwen-image-auto': 7,
    'qwen-image-4bit': 7,
    'qwen-image-lightning': 14, 
    'qwen-image-2512': 14,
    'flux': 12,            # Schnell ~12GB
    'flux-dev': 12,        # Dev ~16GB
    'flux2': 12,           # 4-bit quantized default
    'flux2-full': 30,      # Massive >65GB
    'instruct-pix2pix': 3,
    'qwen-image-edit': 7,
    'qwen-image-edit-lightning': 14,
}

# Video parameter counts (in billions)
VIDEO_MODEL_PARAMS = {
    'wan-2.2': 14,
    'wan-2.2-5b': 5,       # 5B variant
    'wan2.2': 14,
    'wan2.2-5b': 5,        # 5B variant
    'cogvideox': 5,        # 5B variant
    'hunyuan': 13,         # ~13B
    'ltx-video': 2,        # ~2B (DiT)
    'mochi-1': 10,         # ~10B
    'svd': 2,              # ~1.5-3B UNet
    'zeroscope': 1.7,      # ~1.7B
    'zeroscope-xl': 1.7,
    'ms-1.7b': 1.7,
}

# Display names for models (without RAM info)
TEXT_MODEL_NAMES = {
    "deepseek-r1-qwen-7b": "DeepSeek R1-Qwen-7B",
    "deepseek-r1-qwen-14b": "DeepSeek R1-Qwen-14B",
    "deepseek-r1-qwen-32b": "DeepSeek R1-Qwen-32B",
    "deepseek-r1-llama-8b": "DeepSeek R1-Llama-8B",
    "deepseek-r1-llama-70b": "DeepSeek R1-Llama-70B",
    "qwen3-8b": "Qwen 3 8B (Reasoning)",
    "qwen3-14b": "Qwen 3 14B (Reasoning)",
    "qwen3-opus-4.5-8b": "Qwen 3 Opus 4.5 Distill (8B)",
    "qwen3-opus-4.5-14b": "Qwen 3 Opus 4.5 Distill (14B)",
    "qwen3-gpt-5.2-8b": "Qwen 3 GPT-5.2 Distill (8B)",
    "qwen3-gpt-5.2-14b": "Qwen 3 GPT-5.2 Distill (14B)",
    "qwen-coder-32b": "Qwen 2.5 Coder 32B",
    "qwen-coder-14b": "Qwen 2.5 Coder 14B",
    "qwen-coder-7b": "Qwen 2.5 Coder 7B",
    "qwen3-coder-30b": "Qwen 3 Coder 30B (MoE, 3.3B active)",
    "llama-3.1-8b": "Llama 3.1-8B",
    "mistral-nemo-12b": "Mistral Nemo-12B",
    "qwen-vl": "Qwen3-VL 8B",
}

# Bytes per weight for each precision
BYTES_PER_WEIGHT = {
    "float32": 4.0,
    "bfloat16": 2.0,
    "float16": 2.0,
    "int8": 1.0,
    "int6": 0.75,
    "int4": 0.5,
    "auto": 2.0,  # Assume bfloat16 for auto
}

# Overhead multiplier for KV cache, activations, etc.
OVERHEAD_MULTIPLIER = 1.2


def calculate_model_ram(model_id: str, precision: str = "auto") -> float:
    """
    Calculate estimated RAM in GB for a model at given precision.
    
    Args:
        model_id: Model identifier (e.g., 'llama-3.1-8b')
        precision: Precision string ('int4', 'int6', 'int8', 'float16', 'bfloat16', 'float32', 'auto')
    
    Returns:
        Estimated RAM in GB
    """
    # Resolve "auto" to actual precision for calculation
    # Identify model type and params first
    model_type = "text"
    params_b = TEXT_MODEL_PARAMS.get(model_id)
    
    if params_b is None:
        params_b = IMAGE_MODEL_PARAMS.get(model_id)
        if params_b is not None:
            model_type = "image"
    
    if params_b is None:
        params_b = VIDEO_MODEL_PARAMS.get(model_id)
        if params_b is not None:
            model_type = "video"

    # Partial matches for video (e.g. wan-2.2 aliases)
    if params_b is None:
        for k, v in VIDEO_MODEL_PARAMS.items():
            if k in model_id.lower():
                params_b = v
                model_type = "video"
                break
    
    if params_b is None:
        params_b = 8  # Default to 8B if unknown

    # Resolve "auto" to actual precision for calculation
    resolved_precision = precision
    if precision == "auto":
        from .precision import resolve_precision
        import platform
        
        # Estimate based on platform default
        device_type = "cpu"
        framework = "torch"
        
        if platform.system() == "Darwin":
             device_type = "mps" # Or mlx, doesn't matter for resolve logic unless specified
             framework = "mlx"   # Assume MLX default on Mac for estimation
        else:
             import torch
             if torch.cuda.is_available():
                 device_type = "cuda"
        
        resolved_precision = resolve_precision(
             device_type=device_type, 
             framework=framework,
             model_type=model_type
        )

    if params_b is None:
        params_b = 8  # Default to 8B if unknown
        
    bytes_per = BYTES_PER_WEIGHT.get(resolved_precision, 2.0)
    base_ram = params_b * bytes_per
    return base_ram * OVERHEAD_MULTIPLIER


def format_ram_estimate(model_id: str, precision: str = "auto") -> str:
    """
    Format RAM estimate as a string (e.g., '~7GB').
    
    Args:
        model_id: Model identifier
        precision: Precision string
    
    Returns:
        Formatted string like '~7GB' or '~24GB'
    """
    ram_gb = calculate_model_ram(model_id, precision)
    return f"~{ram_gb:.0f}GB"


def format_ram_warning(model_id: str, precision: str = "auto", system_ram_gb: float = 0) -> str:
    """
    Format RAM estimate with optional warning emoji.
    
    Args:
        model_id: Model identifier
        precision: Precision string
        system_ram_gb: System RAM in GB (0 to skip warning check)
    
    Returns:
        Formatted string like '~7GB' or '⚠️ ~120GB'
    """
    ram_gb = calculate_model_ram(model_id, precision)
    warning = ""
    # Warn if model requires >80% of system RAM, or >32GB absolute
    if system_ram_gb > 0 and ram_gb > system_ram_gb * 0.8:
        warning = "⚠️ "
    elif ram_gb > 32:
        warning = "⚠️ "
    return f"{warning}~{ram_gb:.0f}GB"


def get_model_display_with_ram(model_id: str, precision: str = "auto", system_ram_gb: float = 0) -> str:
    """
    Get display name with dynamic RAM estimate.
    
    Args:
        model_id: Model identifier
        precision: Precision string
        system_ram_gb: System RAM in GB (0 to skip warning check)
    
    Returns:
        Display string like 'Llama 3.1-8B (~10GB)' or 'DeepSeek R1-70B (⚠️ ~84GB)'
    """
    name = TEXT_MODEL_NAMES.get(model_id, model_id)
    ram_str = format_ram_warning(model_id, precision, system_ram_gb)
    return f"{name} ({ram_str})"


def get_text_model_options(precision: str = "auto", system_ram_gb: float = 0) -> list:
    """
    Generate text model options list for interactive menus.
    
    Args:
        precision: Selected precision for RAM calculation
        system_ram_gb: System RAM in GB for warning threshold
    
    Returns:
        List of tuples: (display_name, model_id)
    """
    # Ordered list of models for menus
    model_order = [
        "llama-3.1-8b",
        "qwen3-coder-30b",
        "qwen-coder-32b",
        "qwen-coder-14b",
        "qwen-coder-7b",
        "deepseek-r1-qwen-7b",
        "deepseek-r1-qwen-14b",
        "deepseek-r1-qwen-32b",
        "deepseek-r1-llama-8b",
        "deepseek-r1-llama-70b",
        "qwen3-8b",
        "qwen3-14b",
        "qwen3-opus-4.5-8b",
        "qwen3-opus-4.5-14b",
        "qwen3-gpt-5.2-8b",
        "qwen3-gpt-5.2-14b",
    ]
    
    options = []
    for model_id in model_order:
        display = get_model_display_with_ram(model_id, precision, system_ram_gb)
        # Mark default
        if model_id == "llama-3.1-8b":
            display += " (Default)"
        options.append((display, model_id))
    
    return options


def get_chat_model_options(precision: str = "auto", system_ram_gb: float = 0) -> list:
    """
    Generate chat model options list for interactive menus.
    
    Args:
        precision: Selected precision for RAM calculation
        system_ram_gb: System RAM in GB for warning threshold
    
    Returns:
        List of tuples: (display_name, model_id)
    """
    model_order = [
        "llama-3.1-8b",
        "deepseek-r1-qwen-7b",
        "deepseek-r1-qwen-14b",
        "deepseek-r1-qwen-32b",
        "deepseek-r1-llama-8b",
        "deepseek-r1-llama-70b",
        "qwen3-8b",
        "qwen3-14b",
        "qwen3-opus-4.5-8b",
        "qwen3-opus-4.5-14b",
        "qwen3-gpt-5.2-8b",
        "qwen3-gpt-5.2-14b",
        "qwen3-coder-30b",
        "qwen-coder-32b",
        "mistral-nemo-12b",
    ]
    
    options = []
    for model_id in model_order:
        display = get_model_display_with_ram(model_id, precision, system_ram_gb)
        if model_id == "llama-3.1-8b":
            display += " (Default)"
        options.append((display, model_id))
    
    return options


def get_code_model_options(precision: str = "auto", system_ram_gb: float = 0) -> list:
    """
    Generate code model options list for interactive menus.
    
    Args:
        precision: Selected precision for RAM calculation
        system_ram_gb: System RAM in GB for warning threshold
    
    Returns:
        List of tuples: (display_name, model_id)
    """
    model_order = [
        "llama-3.1-8b",
        "qwen3-coder-30b",
        "qwen-coder-32b",
        "qwen-coder-14b",
        "qwen-coder-7b",
        "deepseek-r1-qwen-7b",
        "deepseek-r1-qwen-14b",
        "deepseek-r1-qwen-32b",
        "deepseek-r1-llama-8b",
        "deepseek-r1-llama-70b",
        "qwen3-8b",
        "qwen3-14b",
        "qwen3-opus-4.5-8b",
        "qwen3-opus-4.5-14b",
        "qwen3-gpt-5.2-8b",
        "qwen3-gpt-5.2-14b",
    ]
    
    options = []
    for model_id in model_order:
        display = get_model_display_with_ram(model_id, precision, system_ram_gb)
        if model_id == "llama-3.1-8b":
            display += " (Default)"
        options.append((display, model_id))
    
    return options


def get_image_model_options(precision: str = "auto", system_ram_gb: float = 0, is_mac: bool = False, is_cuda: bool = False) -> list:
    """
    Generate image model options list for interactive menus.
    
    Args:
        precision: Selected precision for RAM calculation
        system_ram_gb: System RAM in GB for warning threshold
        is_mac: Whether running on macOS
        is_cuda: Whether running with CUDA
    
    Returns:
        List of tuples: (display_name, model_id)
    """
    options = []
    
    # helper to format
    def add_opt(mid, base_label):
        ram_str = format_ram_warning(mid, precision, system_ram_gb)
        options.append((f"{base_label} {ram_str}", mid))

    add_opt("z-image", "Z-Image Turbo (Default, Alibaba, Fast 9 Steps)")
    add_opt("sd3.5-turbo", "SD 3.5 Turbo (Fast 4 Steps, 🔒 Gated)")
    add_opt("sdxl", "SDXL Turbo (Fast, no login)")
    add_opt("sd-1.5", "SD 1.5 (Lightweight)")
    
    # SD 3.5 Models
    add_opt("sd3.5-medium", "SD 3.5 Medium (High Quality, 🔒 Gated)")
    add_opt("sd3.5-large", "SD 3.5 Large (Best Quality, 🔒 Gated)")
    
    # Qwen
    add_opt("qwen-image-auto", "Qwen 2.5 Image (Auto: Best Quality)")
    add_opt("qwen-image-lightning", "Qwen 2.5 Image (Lightning: Fast 8-step)")
    
    # Flux
    flux_note = "High Quality, Slow on Mac" if is_mac else "High Quality"
    flux_dev_note = "Professional, Very Slow on Mac" if is_mac else "Professional"
    
    add_opt("flux", f"Flux Schnell ({flux_note})")
    add_opt("flux-dev", f"Flux Dev ({flux_dev_note})")
    
    # FLUX.2
    if is_cuda:
        add_opt("flux2", "FLUX.2 (4-bit SOTA 2025, CUDA)")
    elif is_mac:
        add_opt("flux2-full", "FLUX.2 Full (SOTA 2025, ⚠️ 128GB+ RAM!)")
        
    return options


def get_transform_model_options(precision: str = "auto", system_ram_gb: float = 0, is_mac: bool = False, is_cuda: bool = False) -> list:
    """
    Generate transform (edit) model options list.
    
    Args:
        precision: Selected precision for RAM calculation
        system_ram_gb: System RAM in GB
        is_mac: Whether running on macOS
        is_cuda: Whether running with CUDA
    
    Returns:
        List of tuples: (display_name, model_id)
    """
    options = []
    
    def add_opt(mid, base_label):
        # Map edit model aliases to param keys if needed
        # e.g. z-image-edit -> z-image
        param_key = mid
        if mid == 'z-image-edit': param_key = 'z-image'
        
        ram_str = format_ram_warning(param_key, precision, system_ram_gb)
        options.append((f"{base_label} {ram_str}", mid))

    add_opt("instruct-pix2pix", "InstructPix2Pix (Default, Fast)")
    
    # Qwen Edit Models
    add_opt("qwen-image-edit", "Qwen-Image-Edit (Base 2511, Precise)")
    add_opt("qwen-image-edit-lightning", "Qwen-Edit-Lightning (Fast 2512)")
    
    # Z-Image
    add_opt("z-image-edit", "Z-Image Turbo (Alibaba, Fast)")

    return options


def get_video_model_options(precision: str = "auto", system_ram_gb: float = 0, is_mac: bool = False, is_cuda: bool = False) -> list:
    """
    Generate video model options list for interactive menus.
    
    Args:
        precision: Selected precision for RAM calculation
        system_ram_gb: System RAM in GB
        is_mac: Whether running on macOS
        is_cuda: Whether running with CUDA
    
    Returns:
        List of tuples: (display_name, model_id)
    """
    options = []
    
    def add_opt(mid, base_label):
        ram_str = format_ram_warning(mid, precision, system_ram_gb)
        options.append((f"{base_label} {ram_str}", mid))

    add_opt("zeroscope", "Zeroscope (Default, No Watermarks)")
    add_opt("ms-1.7b", "ModelScope (General Purpose, Has Watermarks)")
    add_opt("cogvideox", "CogVideoX (State of the Art, Slow)")
    add_opt("wan-2.2", "Wan 2.2 (Alibaba, 14B, High Quality)")
    add_opt("wan-2.2-5b", "Wan 2.2 (Alibaba, 5B, Fast)")
    add_opt("ltx-video", "LTX-Video (Lightricks, Fast DiT)")
    add_opt("mochi-1", "Mochi 1 (Genmo, Motion SOTA)")
    add_opt("hunyuan", "HunyuanVideo (Tencent, Cinematic)")
    add_opt("svd", "Stable Video Diffusion (Image-to-Video only)")
    
    return options
