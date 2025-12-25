"""
AI-Media Package

Local offline media generation CLI - modular package structure.
Each submodule is lazily loaded for fast startup time.
"""

__version__ = "1.0.0"

# Public API 
# Note: generators and utils are NOT imported here to allow lazy loading.
# They must be imported explicitly (e.g. from ai_media.generators import ...)
from .constants import RESOLUTIONS, DEFAULT_IMAGE_SIZE, DEFAULT_DURATION
from .models import IMAGE_MODELS, VIDEO_MODELS, AUDIO_MODELS, TEXT_MODELS, EDIT_MODELS, get_model_id, MODEL_REQUIREMENTS
