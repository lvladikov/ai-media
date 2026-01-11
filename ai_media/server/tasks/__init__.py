"""Background tasks package init."""

import os


def get_relative_path(full_path: str) -> str:
    """Convert a full absolute path to a path relative to media_output.
    
    This ensures URLs are clean like /api/files/image.png instead of
    /api/files/Volumes/.../media-output/image.png
    """
    from ..config import CONFIG
    media_output = CONFIG["paths"]["media_output"]
    
    try:
        # Compute relative path from media_output directory
        rel_path = os.path.relpath(full_path, media_output)
        # If relpath starts with "..", the file is outside media_output - use basename
        if rel_path.startswith(".."):
            return os.path.basename(full_path)
        return rel_path
    except ValueError:
        # Different drives on Windows - just use filename
        return os.path.basename(full_path)


from . import image
from . import video
from . import audio
from . import text
from . import transform
from . import convert
from . import upscale

__all__ = ["image", "video", "audio", "text", "transform", "convert", "upscale", "get_relative_path"]
