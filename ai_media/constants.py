"""
Constants and default values for AI-Media.

These are lightweight values that can be imported without loading heavy ML libraries.
"""

# --- Default Values ---
DEFAULT_IMAGE_SIZE = "720p"  # Maps to 1280x720
DEFAULT_AUDIO_SAMPLING = "32000"  # Hz (MusicGen default is usually 32k)
DEFAULT_AUDIO_BITDEPTH = 16
DEFAULT_DURATION = "15s"

# --- Resolution Presets ---
RESOLUTIONS = {
    "480p": (854, 480),
    "576p": (1024, 576),
    "720p": (1280, 720),
    "900p": (1600, 900),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2k": (2048, 1080),  # approx
    "3k": (3072, 1728),
    "2160p": (3840, 2160),
    "4k": (3840, 2160),
    "5k": (5120, 2880),
    "6k": (6144, 3456),
    "7k": (7168, 4032),
    "4320p": (7680, 4320),
    "8k": (7680, 4320),
    "9k": (9216, 5184),
    "10k": (10240, 5760),
    "hd": (1280, 720),
    "fhd": (1920, 1080),
    "uhd": (3840, 2160),
    "sd": (640, 480),
    "vga": (640, 480)
}

def get_resolutions():
    """Get all resolution presets."""
    return RESOLUTIONS
