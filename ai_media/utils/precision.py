"""
Precision and framework resolution utilities for AI-Media.

Handles selection of:
- Data types (int4, int6, int8, float16, bfloat16, float32)
- ML frameworks (torch, mlx)

Based on user preferences, platform capabilities, and model requirements.
"""

import sys

# Valid precision values
PRECISIONS = ["int4", "int6", "int8", "float16", "bfloat16", "float32"]

# Valid frameworks
FRAMEWORKS = ["torch", "mlx"]

# Platform-specific precision support
PRECISION_SUPPORT = {
    "cuda": ["int4", "int8", "float16", "bfloat16", "float32"],
    "mps": ["float16", "bfloat16", "float32"],  # int6/int8 experimental, int4 via MLX only
    "mlx": ["int4", "int6", "int8", "float16", "bfloat16", "float32"],
    "cpu": ["float32"],
}


def is_mlx_available():
    """Check if MLX is available on this system (macOS only)."""
    if sys.platform != "darwin":
        return False
    try:
        import mlx.core
        return True
    except ImportError:
        return False


def get_supported_frameworks(precision: str = "auto") -> list:
    """
    Get list of supported frameworks for this platform and precision.
    
    Args:
        precision: Selected precision (int4, int6, int8, float16, etc.)
        
    Returns:
        List of framework strings ("torch", "mlx")
    """
    frameworks = ["torch"]
    
    if is_mlx_available():
        frameworks.append("mlx")
    
    # On Mac, some precisions are ONLY supported by MLX or ONLY by Torch
    if sys.platform == "darwin":
        if precision in ["int4", "int6", "int8"]:
            return ["mlx"] # Torch/MPS doesn't support these well yet
    
    return frameworks


def get_supported_precisions(device_type: str, framework: str = "torch") -> list:
    """
    Get list of supported precisions for a device/framework combination.
    
    Args:
        device_type: "cuda", "mps", or "cpu"
        framework: "torch" or "mlx"
    
    Returns:
        List of supported precision strings
    """
    if framework == "mlx":
        return PRECISION_SUPPORT.get("mlx", ["float32"])
    return PRECISION_SUPPORT.get(device_type, ["float32"])


def resolve_framework(forced_framework: str = None, device_type: str = None) -> str:
    """
    Resolve which ML framework to use.
    
    Args:
        forced_framework: User-specified framework ("torch" or "mlx")
        device_type: Detected device type ("cuda", "mps", "cpu")
    
    Returns:
        "torch" or "mlx"
    """
    # If user specified a framework, validate and return it
    if forced_framework:
        if forced_framework == "mlx":
            if not is_mlx_available():
                print("⚠️  MLX not available on this system. Falling back to PyTorch.")
                return "torch"
            return "mlx"
        return "torch"
    
    # Auto-select: MLX only used when explicitly requested (for now)
    # Future: Could auto-prefer MLX for text models on Mac
    return "torch"


def resolve_precision(
    forced_precision: str = None,
    device_type: str = "cpu",
    framework: str = "torch",
    model_type: str = "text",
    prefer_bfloat16: bool = True
) -> str:
    """
    Resolve the final precision to use.
    
    Args:
        forced_precision: User-specified precision (int4, int6, int8, float16, bfloat16, float32)
        device_type: Device type ("cuda", "mps", "cpu")
        framework: ML framework ("torch", "mlx")
        model_type: Type of model ("text", "image", "audio", "video")
        prefer_bfloat16: Whether to prefer bfloat16 over float16 when available
    
    Returns:
        Precision string (int4, int6, int8, float16, bfloat16, float32)
    """
    supported = get_supported_precisions(device_type, framework)
    
    # If user forced a precision, validate it's supported
    if forced_precision:
        if forced_precision in supported:
            return forced_precision
        else:
            print(f"⚠️  Precision '{forced_precision}' not supported on {device_type}/{framework}. "
                  f"Available: {', '.join(supported)}")
            # Fall through to auto-selection
    
    # Auto-select based on device and model type
    if framework == "mlx":
        # MLX: Default to 4-bit for text (fastest), bfloat16 for images
        if model_type == "text":
            return "int4"
        return "bfloat16"
    
    # PyTorch/CUDA
    if device_type == "cuda":
        if prefer_bfloat16 and "bfloat16" in supported:
            return "bfloat16"
        return "float16"
    
    # PyTorch/MPS
    if device_type == "mps":
        # MPS now supports float16/bfloat16 well
        if prefer_bfloat16 and "bfloat16" in supported:
            return "bfloat16"
        return "float16"
    
    # CPU fallback
    return "float32"


def precision_to_torch_dtype(precision: str):
    """
    Convert precision string to torch dtype.
    
    Args:
        precision: Precision string (int4, int6, int8, float16, bfloat16, float32)
    
    Returns:
        torch.dtype or None for quantized types
    """
    import torch
    
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        # int4/int6/int8 don't map directly to torch dtypes
        # They require special loading (bitsandbytes, etc.)
        "int8": None,
        "int6": None,
        "int4": None,
    }
    return mapping.get(precision, torch.float32)


def get_quantization_config(precision: str):
    """
    Get bitsandbytes quantization config for int4/int8.
    
    Args:
        precision: "int4" or "int8"
    
    Returns:
        BitsAndBytesConfig or None
    """
    if precision not in ["int4", "int8"]:
        return None
    
    try:
        from transformers import BitsAndBytesConfig
        
        if precision == "int4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="bfloat16",
                bnb_4bit_quant_type="nf4",
            )
        elif precision == "int8":
            return BitsAndBytesConfig(load_in_8bit=True)
    except ImportError:
        print("⚠️  bitsandbytes not available for quantization. Using float16.")
        return None
    
    return None


def parse_model_precision_framework(model_string: str) -> tuple:
    """
    Parse model string with optional precision and framework segments.
    Syntax: 
    - model:precision:framework (e.g., 'llama-3.1-8b:int4:mlx')
    - model:precision (e.g., 'llama-3.1-8b:int4')
    - model::framework (e.g., 'llama-3.1-8b::mlx')
    
    Args:
        model_string: The model string to parse
    
    Returns:
        (base_model_name, precision_or_none, framework_or_none)
    """
    if not ":" in model_string:
        return model_string, None, None
    
    parts = model_string.split(":")
    base_name = parts[0]
    precision = None
    framework = None
    
    if len(parts) == 2:
        # model:extra
        extra = parts[1].lower()
        if extra in PRECISIONS:
            precision = extra
        elif extra in FRAMEWORKS:
            framework = extra
    elif len(parts) == 3:
        # model:precision:framework or model::framework
        if parts[1].lower() in PRECISIONS:
            precision = parts[1].lower()
        
        if parts[2].lower() in FRAMEWORKS:
            framework = parts[2].lower()
            
    return base_name, precision, framework


def parse_model_precision_suffix(model_name: str) -> tuple:
    """Legacy wrapper for backward compatibility."""
    base, prec, fw = parse_model_precision_framework(model_name)
    return base, prec
