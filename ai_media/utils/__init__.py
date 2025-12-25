"""
Utility modules for AI-Media.

Submodules:
- system: Device detection, GPU memory, signal handling
- parsers: Size, duration, bitrate parsing
- performance: PerformanceTracker, ResourceMonitor
- ffmpeg: FFmpeg encoding helpers
"""

from .parsers import parse_size, parse_duration, parse_sampling_rate, parse_bitrate, format_time
from .system import get_optimal_device_and_dtype, clear_gpu_memory, get_system_resources
from .system import check_resources_and_warn, setup_signal_handlers, ensure_paths
from .performance import PerformanceTracker, ResourceMonitor
from .ffmpeg import get_video_encoding_params, ffmpeg_resize_video, get_video_info, has_audio_track

__all__ = [
    'parse_size', 'parse_duration', 'parse_sampling_rate', 'parse_bitrate', 'format_time',
    'get_optimal_device_and_dtype', 'clear_gpu_memory', 'get_system_resources',
    'check_resources_and_warn', 'setup_signal_handlers', 'ensure_paths',
    'PerformanceTracker', 'ResourceMonitor',
    'get_video_encoding_params', 'ffmpeg_resize_video', 'get_video_info', 'has_audio_track',
]
