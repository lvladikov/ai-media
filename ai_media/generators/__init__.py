"""
AI Generation modules for AI-Media.

Submodules:
- image: Image generation (Flux, SDXL, SD 1.5)
- video: Video generation (Zeroscope, CogVideoX, LTX, etc.)
- audio: Audio generation (MusicGen, AudioLDM2, Bark)
- description: Image/Video captioning (Florence, BLIP)
- transform: Image editing (InstructPix2Pix, background removal)
- text: Article, research, chat, and code generation
"""

from .image import generate_image
from .audio import generate_audio
from .video import generate_video
from .description import generate_caption
from .transform import generate_edit, remove_background
from .text import ArticleGenerator

__all__ = [
    'generate_image',
    'generate_audio',
    'generate_video',
    'generate_caption',
    'generate_edit',
    'remove_background',
    'ArticleGenerator',
]
