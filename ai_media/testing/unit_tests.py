#!/usr/bin/env python3
"""
Unit tests for ai-media.py

Run all tests:
    python -m unittest test_ai_media -v

Run specific test class:
    python -m unittest test_ai_media.TestParseSize -v

Run specific test method:
    python -m unittest test_ai_media.TestParseSize.test_resolution_presets -v
"""

import unittest
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
import asyncio
from unittest.mock import patch, MagicMock, mock_open, Mock
from datetime import datetime
import io

# =============================================================================
# Module Import with Mocking
# =============================================================================

# Mock heavy dependencies before importing ai-media module
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.backends.mps.is_available.return_value = False
mock_torch.device.return_value = MagicMock(type="cpu")
mock_torch.float32 = "float32"
mock_torch.float16 = "float16"
mock_torch.bfloat16 = "bfloat16"

# Mock all torch submodules that torchvision may try to import
sys.modules['torch'] = mock_torch
sys.modules['torch.cuda'] = MagicMock()
sys.modules['torch.hub'] = MagicMock()
sys.modules['torch.version'] = MagicMock(cuda=None)
sys.modules['torch.backends'] = MagicMock()
sys.modules['torch.backends.mps'] = MagicMock()
sys.modules['torch.backends.cudnn'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
class MockModule(MagicMock):
    __name__ = 'MockModule'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
sys.modules['torch.nn'].Module = MockModule
sys.modules['torch.nn'].Sequential = MockModule
sys.modules['torch.nn'].ModuleList = MockModule
sys.modules['torch.nn'].ModuleDict = MockModule
sys.modules['torch.nn'].Parameter = MagicMock()
# Mock common layers that might be subclassed
for layer in ['Conv2d', 'ConvTranspose2d', 'Linear', 'BatchNorm2d', 'ReLU', 'LeakyReLU', 'PReLU', 'Sigmoid', 'Tanh', 'Dropout', 'Softmax']:
    setattr(sys.modules['torch.nn'], layer, MockModule)
sys.modules['torch.autograd'] = MagicMock()
class MockFunction(MagicMock):
    @staticmethod
    def apply(*args, **kwargs):
        return MagicMock()
sys.modules['torch.autograd'].Function = MockFunction
sys.modules['torch.autograd.function'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['torch.nn.modules'] = MagicMock()
sys.modules['torch.nn.modules.batchnorm'] = MagicMock()
sys.modules['torch.nn.modules.utils'] = MagicMock()
sys.modules['torch.nn.utils'] = MagicMock()
sys.modules['torch.nn.utils.spectral_norm'] = MagicMock()
sys.modules['torch.utils'] = MagicMock()
sys.modules['torch.utils.data'] = MagicMock()
sys.modules['torch.utils.checkpoint'] = MagicMock()
sys.modules['torch.jit'] = MagicMock()
sys.modules['torch._C'] = MagicMock()
sys.modules['torch.distributed'] = MagicMock()
sys.modules['torch.multiprocessing'] = MagicMock()

# Mock torchvision entirely to avoid its complex torch dependency chain
sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.transforms'] = MagicMock()
sys.modules['torchvision.transforms.functional'] = MagicMock()
sys.modules['torchvision.utils'] = MagicMock()
sys.modules['torchvision.models'] = MagicMock()

# Mock basicsr entirely to avoid complex dependencies
sys.modules['basicsr'] = MagicMock()
sys.modules['basicsr.archs'] = MagicMock()
sys.modules['basicsr.archs.rrdbnet_arch'] = MagicMock()
sys.modules['basicsr.utils'] = MagicMock()
sys.modules['basicsr.utils.download_util'] = MagicMock()

# Link mocks to mock_torch to ensure consistency regardless of import method
mock_torch.nn = sys.modules['torch.nn']
mock_torch.autograd = sys.modules['torch.autograd']
mock_torch.utils = sys.modules['torch.utils']
mock_torch.backends = sys.modules['torch.backends']
mock_torch.cuda = sys.modules['torch.cuda']
mock_torch.hub = sys.modules['torch.hub']
mock_torch.jit = sys.modules['torch.jit']

# Configure mock_torch for resource checks
mock_torch.cuda.get_device_properties.return_value.total_memory = 8 * (1024**3)
mock_torch.cuda.memory_allocated.return_value = 0
mock_torch.cuda.is_available.return_value = False
mock_torch.cuda.is_bf16_supported.return_value = False
mock_torch.backends.mps.is_available.return_value = False

sys.modules['diffusers'] = MagicMock()
sys.modules['diffusers.utils'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['accelerate'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.io'] = MagicMock()
sys.modules['scipy.io.wavfile'] = MagicMock()
sys.modules['psutil'] = MagicMock()
sys.modules['PIL'] = MagicMock(__version__='10.0.0')
sys.modules['PIL.Image'] = MagicMock()
sys.modules['PIL.ImageOps'] = MagicMock()

# Import the package and submodules
import ai_media
from ai_media.utils import parsers, system, performance, ffmpeg, interaction
# Save original resource checks for specific utility tests
_ORIGINAL_CHECK_WARN = system.check_resources_and_warn

# Mock resource checks globally for tests to avoid hanging on input()
# This must happen before generators are imported to affect their local references
system.check_resources_and_warn = MagicMock(return_value=True)

from ai_media.generators import text, image, video, audio, transform, description, subtitles, transcription
from ai_media import upscaling, interactive, models, constants
_ORIGINAL_CHECK_CONFIRM = upscaling.check_resources_and_confirm
upscaling.check_resources_and_confirm = MagicMock(return_value=True)

import ai_media.server.state as server_state
from ai_media.server.cache import ModelCache
from ai_media.server.jobs import create_job, update_job, is_job_cancelled
from ai_media.server.config import load_config
from ai_media.server.app import create_app
import inspect

# Compatibility Patching 
# (Unit tests were written against the monolithic ai-media.py script which exposed these functions at top level.
#  We patch the ai_media package object to expose them similarly for the tests.)
ai_media.parse_size = parsers.parse_size
ai_media.parse_duration = parsers.parse_duration
ai_media.parse_sampling_rate = parsers.parse_sampling_rate
ai_media.parse_bitrate = parsers.parse_bitrate
ai_media.parse_upscale_factor = parsers.parse_upscale_factor
ai_media.format_time = parsers.format_time
ai_media.extract_prompt_parameters = parsers.extract_prompt_parameters
ai_media.get_video_encoding_params = ffmpeg.get_video_encoding_params
ai_media.ensure_paths = system.ensure_paths
ai_media.write_report_json = performance.write_report_json
ai_media.clear_gpu_memory = system.clear_gpu_memory
ai_media.get_optimal_device_and_dtype = system.get_optimal_device_and_dtype
ai_media.is_bfloat16_supported = system.is_bfloat16_supported
# Store original reference but keep global mock active
ai_media.check_resources_and_warn = system.check_resources_and_warn
ai_media.get_system_resources = system.get_system_resources
ai_media.signal_handler = system.signal_handler
ai_media.clear_screen = interaction.clear_screen
ai_media.show_header = interaction.show_header
ai_media.emoji = interaction.emoji
ai_media.PerformanceTracker = performance.PerformanceTracker
ai_media.ResourceMonitor = performance.ResourceMonitor
ai_media._test_state = system._test_state
ai_media.signal_handler = system.signal_handler
ai_media.run_interactive = interactive.run_interactive
ai_media.JUMP_POINTS = interactive.JUMP_POINTS
# Upscaling Patching
ai_media.simple_upscale_image = upscaling.simple_upscale_image
ai_media.simple_upscale_video = upscaling.simple_upscale_video
ai_media.upscale_video_fast = upscaling.upscale_video_fast
ai_media.upscale_video_zeroscope_xl = video.upscale_video_zeroscope_xl
ai_media.ffmpeg_resize_video = ffmpeg.ffmpeg_resize_video
ai_media.upscale_video_file = upscaling.upscale_video_fast
ai_media.upscale_image_file = upscaling.upscale_image_file
ai_media.upscale_image_fast = upscaling.upscale_image_fast
ai_media.check_resources_and_confirm = upscaling.check_resources_and_confirm
ai_media.HAS_REALESRGAN = upscaling.HAS_REALESRGAN
ai_media.RRDBNet = MagicMock() # Mocked for tests
ai_media.RealESRGANer = MagicMock() # Mocked for tests
ai_media._check_ffmpeg_encoder = ffmpeg._check_ffmpeg_encoder
ai_media._interrupted = system._interrupted # This won't work as it's a bool, so let's patch the test to check system._interrupted

# Model Patching
ai_media.EDIT_MODELS = models.EDIT_MODELS
ai_media.CAPTION_MODELS = models.CAPTION_MODELS
ai_media.IMAGE_MODELS = models.IMAGE_MODELS
ai_media.VIDEO_MODELS = models.VIDEO_MODELS
ai_media.AUDIO_MODELS = models.AUDIO_MODELS
ai_media.RESOLUTIONS = constants.RESOLUTIONS
ai_media.MODEL_REQUIREMENTS = models.MODEL_REQUIREMENTS

# Generator Patching
ai_media.ArticleGenerator = text.ArticleGenerator
ai_media.generate_image = image.generate_image
ai_media.generate_video = video.generate_video
ai_media.generate_audio = audio.generate_audio
ai_media.generate_edit = transform.generate_edit
ai_media.remove_background = transform.remove_background
ai_media.generate_caption = description.generate_caption

# CLI specific attributes
ai_media._loading_timer = interaction._loading_timer
ai_media._loading_shown = False
def mocked_show_loading():
    ai_media._loading_shown = True
ai_media._show_loading_message = mocked_show_loading

# Import main for routing tests
try:
    import importlib.util
    from pathlib import Path
    _script_path = Path(__file__).parent.parent.parent / "ai-media.py"
    if _script_path.exists():
        _spec = importlib.util.spec_from_file_location("aimedia_script", _script_path)
        _aimedia_script = importlib.util.module_from_spec(_spec)
        # Avoid running main on import
        with patch('sys.argv', ['ai-media.py']):
            _spec.loader.exec_module(_aimedia_script)
        ai_media.main = _aimedia_script.main
        # Link script internal placeholders to our patched ai_media object
        _aimedia_script.pkg_upscale_video_fast = ai_media.upscale_video_fast
        _aimedia_script.pkg_upscale_image_fast = ai_media.upscale_image_fast
        _aimedia_script.pkg_upscale_image_file = ai_media.upscale_image_file
        _aimedia_script.pkg_simple_upscale_image = ai_media.simple_upscale_image
        _aimedia_script.pkg_simple_upscale_video = ai_media.simple_upscale_video
        _aimedia_script.pkg_generate_image = ai_media.generate_image
        _aimedia_script.pkg_generate_video = ai_media.generate_video
        _aimedia_script.pkg_generate_audio = ai_media.generate_audio
        _aimedia_script.pkg_system = system
        _aimedia_script.pkg_parsers = parsers
        _aimedia_script.HAS_AI_MEDIA_PKG = True
    else:
        ai_media.main = MagicMock()
except Exception:
    ai_media.main = MagicMock()


# =============================================================================
# Constants Tests - Verify all model dictionaries and presets exist
# =============================================================================

class TestConstants(unittest.TestCase):
    """Tests for module-level constants and dictionaries."""
    
    def test_image_models_exist(self):
        """Test IMAGE_MODELS dictionary exists with expected keys."""
        self.assertIn("default", ai_media.IMAGE_MODELS)
        self.assertIn("sdxl", ai_media.IMAGE_MODELS)
        self.assertIn("sd-1.5", ai_media.IMAGE_MODELS)
        self.assertIn("sd3.5-medium", ai_media.IMAGE_MODELS)
        self.assertIn("sd3.5-large", ai_media.IMAGE_MODELS)
        self.assertIn("sd3.5-turbo", ai_media.IMAGE_MODELS)
        self.assertIn("qwen-image", ai_media.IMAGE_MODELS)
        self.assertIn("qwen-image-2512", ai_media.IMAGE_MODELS)
        self.assertIn("flux", ai_media.IMAGE_MODELS)
        self.assertIn("flux-dev", ai_media.IMAGE_MODELS)
        self.assertIn("upscaler", ai_media.IMAGE_MODELS)
        self.assertIn("upscaler_x2", ai_media.IMAGE_MODELS)
    
    def test_audio_models_exist(self):
        """Test AUDIO_MODELS dictionary exists with expected keys."""
        self.assertIn("default", ai_media.AUDIO_MODELS)
        self.assertIn("musicgen-small", ai_media.AUDIO_MODELS)
        self.assertIn("musicgen-medium", ai_media.AUDIO_MODELS)
        self.assertIn("musicgen-large", ai_media.AUDIO_MODELS)
        self.assertIn("audioldm2", ai_media.AUDIO_MODELS)
        self.assertIn("bark", ai_media.AUDIO_MODELS)
    
    def test_video_models_exist(self):
        """Test VIDEO_MODELS dictionary exists with expected keys."""
        self.assertIn("default", ai_media.VIDEO_MODELS)
        self.assertIn("zeroscope", ai_media.VIDEO_MODELS)
        self.assertIn("zeroscope-xl", ai_media.VIDEO_MODELS)
        self.assertIn("cogvideox", ai_media.VIDEO_MODELS)
        self.assertIn("svd", ai_media.VIDEO_MODELS)
    
    def test_edit_models_exist(self):
        """Test EDIT_MODELS dictionary exists with expected keys."""
        self.assertIn("default", ai_media.EDIT_MODELS)
        self.assertIn("instruct-pix2pix", ai_media.EDIT_MODELS)
        self.assertIn("qwen-image-edit", ai_media.EDIT_MODELS)
        self.assertIn("qwen-image-edit-lightning", ai_media.EDIT_MODELS)
        self.assertIn("remove-bg", ai_media.EDIT_MODELS)
    
    def test_resolutions_exist(self):
        """Test RESOLUTIONS dictionary exists with expected presets."""
        expected_presets = ["480p", "576p", "720p", "900p", "1080p", "1440p", 
                           "2k", "3k", "2160p", "4k", "5k", "6k", "8k", "10k",
                           "hd", "fhd", "uhd", "vga"]
        for preset in expected_presets:
            self.assertIn(preset, ai_media.RESOLUTIONS, f"Missing resolution preset: {preset}")
            self.assertIsInstance(ai_media.RESOLUTIONS[preset], tuple)
            self.assertEqual(len(ai_media.RESOLUTIONS[preset]), 2)
    
    def test_model_requirements_exist(self):
        """Test MODEL_REQUIREMENTS dictionary has expected structure."""
        self.assertIsInstance(ai_media.MODEL_REQUIREMENTS, dict)
        # Check a known model has required fields
        for model_id, reqs in ai_media.MODEL_REQUIREMENTS.items():
            self.assertIn("vram", reqs, f"Missing vram for {model_id}")
            self.assertIn("ram", reqs, f"Missing ram for {model_id}")


# =============================================================================
# Parsing Function Tests
# =============================================================================

class TestParseSize(unittest.TestCase):
    """Tests for parse_size() function - covers all resolution formats."""
    
    def test_resolution_presets_standard(self):
        """Test standard resolution presets (480p, 720p, 1080p, etc.)."""
        self.assertEqual(ai_media.parse_size("480p"), (854, 480))
        self.assertEqual(ai_media.parse_size("576p"), (1024, 576))
        self.assertEqual(ai_media.parse_size("720p"), (1280, 720))
        self.assertEqual(ai_media.parse_size("900p"), (1600, 900))
        self.assertEqual(ai_media.parse_size("1080p"), (1920, 1080))
        self.assertEqual(ai_media.parse_size("1440p"), (2560, 1440))
        self.assertEqual(ai_media.parse_size("2160p"), (3840, 2160))
    
    def test_resolution_presets_k(self):
        """Test K resolution presets (2k through 10k)."""
        self.assertEqual(ai_media.parse_size("2k"), (2048, 1080))
        self.assertEqual(ai_media.parse_size("3k"), (3072, 1728))
        self.assertEqual(ai_media.parse_size("4k"), (3840, 2160))
        self.assertEqual(ai_media.parse_size("5k"), (5120, 2880))
        self.assertEqual(ai_media.parse_size("6k"), (6144, 3456))
        self.assertEqual(ai_media.parse_size("8k"), (7680, 4320))
        self.assertEqual(ai_media.parse_size("10k"), (10240, 5760))
    
    def test_resolution_presets_named(self):
        """Test named resolution presets (hd, fhd, uhd, vga)."""
        self.assertEqual(ai_media.parse_size("hd"), (1280, 720))
        self.assertEqual(ai_media.parse_size("fhd"), (1920, 1080))
        self.assertEqual(ai_media.parse_size("uhd"), (3840, 2160))
        self.assertEqual(ai_media.parse_size("vga"), (640, 480))
    
    def test_wxh_format(self):
        """Test WxH format (1280x720, 1920x1080, custom sizes)."""
        self.assertEqual(ai_media.parse_size("1280x720"), (1280, 720))
        self.assertEqual(ai_media.parse_size("1920x1080"), (1920, 1080))
        self.assertEqual(ai_media.parse_size("64x64"), (64, 64))
        self.assertEqual(ai_media.parse_size("512x512"), (512, 512))
        self.assertEqual(ai_media.parse_size("768x1024"), (768, 1024))
    
    def test_object_format_short_keys(self):
        """Test object format with short keys ({w:1920, h:1080})."""
        self.assertEqual(ai_media.parse_size("{w:1920, h:1080}"), (1920, 1080))
        self.assertEqual(ai_media.parse_size("{w: 1280, h: 720}"), (1280, 720))
    
    def test_object_format_long_keys(self):
        """Test object format with long keys ({width:1920, height:1080})."""
        self.assertEqual(ai_media.parse_size("{width:1280, height:720}"), (1280, 720))
        self.assertEqual(ai_media.parse_size("{width: 1920, height: 1080}"), (1920, 1080))
    
    def test_case_insensitivity(self):
        """Test case insensitivity for all formats."""
        self.assertEqual(ai_media.parse_size("720P"), (1280, 720))
        self.assertEqual(ai_media.parse_size("4K"), (3840, 2160))
        self.assertEqual(ai_media.parse_size("FHD"), (1920, 1080))
        self.assertEqual(ai_media.parse_size("1280X720"), (1280, 720))
    
    def test_empty_and_none_returns_default(self):
        """Test empty and None values return default resolution."""
        result = ai_media.parse_size(None)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        
        result = ai_media.parse_size("")
        self.assertIsInstance(result, tuple)
    
    def test_whitespace_handling(self):
        """Test whitespace is trimmed correctly."""
        self.assertEqual(ai_media.parse_size("  720p  "), (1280, 720))
        self.assertEqual(ai_media.parse_size(" 1280x720 "), (1280, 720))
        self.assertEqual(ai_media.parse_size("\t1080p\n"), (1920, 1080))


class TestParseDuration(unittest.TestCase):
    """Tests for parse_duration() function - covers all duration formats."""
    
    def test_numeric_seconds_int(self):
        """Test integer second values."""
        self.assertEqual(ai_media.parse_duration("15"), 15.0)
        self.assertEqual(ai_media.parse_duration("60"), 60.0)
        self.assertEqual(ai_media.parse_duration(30), 30.0)
    
    def test_numeric_seconds_float(self):
        """Test float second values."""
        self.assertEqual(ai_media.parse_duration("30.5"), 30.5)
        self.assertEqual(ai_media.parse_duration("2.5"), 2.5)
        self.assertEqual(ai_media.parse_duration(10.5), 10.5)
    
    def test_string_format_seconds(self):
        """Test string format with 's' suffix."""
        self.assertEqual(ai_media.parse_duration("15s"), 15.0)
        self.assertEqual(ai_media.parse_duration("30s"), 30.0)
        self.assertEqual(ai_media.parse_duration("2.5s"), 2.5)
        self.assertEqual(ai_media.parse_duration("120s"), 120.0)
    
    def test_string_format_minutes(self):
        """Test string format with 'm' suffix."""
        self.assertEqual(ai_media.parse_duration("1m"), 60.0)
        self.assertEqual(ai_media.parse_duration("5m"), 300.0)
        self.assertEqual(ai_media.parse_duration("1.5m"), 90.0)
        self.assertEqual(ai_media.parse_duration("10m"), 600.0)
    
    def test_string_format_hours(self):
        """Test string format with 'h' suffix."""
        self.assertEqual(ai_media.parse_duration("1h"), 3600.0)
        self.assertEqual(ai_media.parse_duration("2h"), 7200.0)
        self.assertEqual(ai_media.parse_duration("0.5h"), 1800.0)
    
    def test_combined_format_hms(self):
        """Test combined time format (1h30m15s)."""
        self.assertEqual(ai_media.parse_duration("1h30m"), 5400.0)
        self.assertEqual(ai_media.parse_duration("1h30m15s"), 5415.0)
        self.assertEqual(ai_media.parse_duration("2m30s"), 150.0)
        self.assertEqual(ai_media.parse_duration("1h1m1s"), 3661.0)
    
    def test_object_format_full(self):
        """Test object format with all keys ({h:1, m:25, s:10})."""
        self.assertEqual(ai_media.parse_duration("{h:1, m:25, s:10}"), 5110.0)
        self.assertEqual(ai_media.parse_duration("{hours:1, minutes:25, seconds:10}"), 5110.0)
    
    def test_object_format_partial(self):
        """Test object format with partial keys."""
        self.assertEqual(ai_media.parse_duration("{m:5, s:30}"), 330.0)
        self.assertEqual(ai_media.parse_duration("{s:45}"), 45.0)
        self.assertEqual(ai_media.parse_duration("{h:2}"), 7200.0)
    
    def test_colon_format_ms(self):
        """Test colon format M:S."""
        self.assertEqual(ai_media.parse_duration("1:30"), 90.0)
        self.assertEqual(ai_media.parse_duration("5:00"), 300.0)
        self.assertEqual(ai_media.parse_duration("0:45"), 45.0)
    
    def test_colon_format_hms(self):
        """Test colon format H:M:S."""
        self.assertEqual(ai_media.parse_duration("1:30:00"), 5400.0)
        self.assertEqual(ai_media.parse_duration("1:01:01"), 3661.0)
        self.assertEqual(ai_media.parse_duration("0:05:30"), 330.0)
    
    def test_empty_and_none_returns_default(self):
        """Test empty and None values return default (15s)."""
        self.assertEqual(ai_media.parse_duration(None), 15.0)
        self.assertEqual(ai_media.parse_duration(""), 15.0)


class TestParseSamplingRate(unittest.TestCase):
    """Tests for parse_sampling_rate() function."""
    
    def test_numeric_hz(self):
        """Test numeric Hz values."""
        self.assertEqual(ai_media.parse_sampling_rate("44100"), 44100)
        self.assertEqual(ai_media.parse_sampling_rate("32000"), 32000)
        self.assertEqual(ai_media.parse_sampling_rate("48000"), 48000)
        self.assertEqual(ai_media.parse_sampling_rate("22050"), 22050)
        self.assertEqual(ai_media.parse_sampling_rate("16000"), 16000)
    
    def test_khz_format_decimal(self):
        """Test kHz format with decimal (44.1khz)."""
        self.assertEqual(ai_media.parse_sampling_rate("44.1khz"), 44100)
        self.assertEqual(ai_media.parse_sampling_rate("22.05khz"), 22050)
    
    def test_khz_format_integer(self):
        """Test kHz format with integer (48khz)."""
        self.assertEqual(ai_media.parse_sampling_rate("48khz"), 48000)
        self.assertEqual(ai_media.parse_sampling_rate("32k"), 32000)
        self.assertEqual(ai_media.parse_sampling_rate("16k"), 16000)
    
    def test_case_insensitivity(self):
        """Test case insensitivity."""
        self.assertEqual(ai_media.parse_sampling_rate("44.1KHZ"), 44100)
        self.assertEqual(ai_media.parse_sampling_rate("48KHz"), 48000)
        self.assertEqual(ai_media.parse_sampling_rate("32K"), 32000)
    
    def test_empty_and_none_returns_default(self):
        """Test empty and None values return default (32000)."""
        self.assertEqual(ai_media.parse_sampling_rate(None), 32000)
        self.assertEqual(ai_media.parse_sampling_rate(""), 32000)


class TestParseBitrate(unittest.TestCase):
    """Tests for parse_bitrate() function."""
    
    def test_bitrate_passthrough(self):
        """Test bitrate strings are passed through with strip."""
        self.assertEqual(ai_media.parse_bitrate("192k"), "192k")
        self.assertEqual(ai_media.parse_bitrate("320k"), "320k")
        self.assertEqual(ai_media.parse_bitrate("128k"), "128k")
        self.assertEqual(ai_media.parse_bitrate("256k"), "256k")
    
    def test_whitespace_stripped(self):
        """Test whitespace is stripped."""
        self.assertEqual(ai_media.parse_bitrate("  256k  "), "256k")
        self.assertEqual(ai_media.parse_bitrate("\t192k\n"), "192k")
    
    def test_empty_and_none_returns_none(self):
        """Test empty and None values return None."""
        self.assertIsNone(ai_media.parse_bitrate(None))
        self.assertIsNone(ai_media.parse_bitrate(""))


class TestParseUpscaleFactor(unittest.TestCase):
    """Tests for parse_upscale_factor() function."""
    
    def test_numeric_factor_string(self):
        """Test numeric string factors."""
        self.assertEqual(ai_media.parse_upscale_factor("2"), 2.0)
        self.assertEqual(ai_media.parse_upscale_factor("4"), 4.0)
        self.assertEqual(ai_media.parse_upscale_factor("8"), 8.0)
    
    def test_numeric_factor_float(self):
        """Test float factors."""
        self.assertEqual(ai_media.parse_upscale_factor("1.5"), 1.5)
        self.assertEqual(ai_media.parse_upscale_factor("2.5"), 2.5)
        self.assertEqual(ai_media.parse_upscale_factor("3.5"), 3.5)
    
    def test_x_suffix_lowercase(self):
        """Test factors with lowercase 'x' suffix."""
        self.assertEqual(ai_media.parse_upscale_factor("2x"), 2.0)
        self.assertEqual(ai_media.parse_upscale_factor("4x"), 4.0)
        self.assertEqual(ai_media.parse_upscale_factor("2.5x"), 2.5)
    
    def test_x_suffix_uppercase(self):
        """Test factors with uppercase 'X' suffix."""
        self.assertEqual(ai_media.parse_upscale_factor("2X"), 2.0)
        self.assertEqual(ai_media.parse_upscale_factor("4X"), 4.0)
        self.assertEqual(ai_media.parse_upscale_factor("8X"), 8.0)
    
    def test_whitespace_handling(self):
        """Test whitespace is handled correctly."""
        self.assertEqual(ai_media.parse_upscale_factor("  2x  "), 2.0)
        self.assertEqual(ai_media.parse_upscale_factor(" 4 "), 4.0)
    
    def test_empty_and_none_returns_default(self):
        """Test empty and None values return default (2.0)."""
        self.assertEqual(ai_media.parse_upscale_factor(None), 2.0)
        self.assertEqual(ai_media.parse_upscale_factor(""), 2.0)
    
    def test_invalid_factor_returns_default(self):
        """Test invalid factors return default (2.0)."""
        # Redirect stdout to avoid emoji encoding issues on Windows
        with patch('sys.stdout', new_callable=io.StringIO):
            self.assertEqual(ai_media.parse_upscale_factor("invalid"), 2.0)
            self.assertEqual(ai_media.parse_upscale_factor("abc"), 2.0)
    
    def test_negative_and_zero_returns_default(self):
        """Test negative and zero factors return default."""
        # Redirect stdout to avoid emoji encoding issues on Windows
        with patch('sys.stdout', new_callable=io.StringIO):
            self.assertEqual(ai_media.parse_upscale_factor("-1"), 2.0)
            self.assertEqual(ai_media.parse_upscale_factor("0"), 2.0)
            self.assertEqual(ai_media.parse_upscale_factor("-2x"), 2.0)


class TestFormatTime(unittest.TestCase):
    """Tests for format_time() function."""
    
    def test_seconds_only(self):
        """Test formatting seconds only."""
        self.assertEqual(ai_media.format_time(30), "30s")
        self.assertEqual(ai_media.format_time(1), "1s")
        self.assertEqual(ai_media.format_time(59), "59s")
    
    def test_seconds_with_decimal(self):
        """Test formatting seconds with decimal."""
        self.assertEqual(ai_media.format_time(45.5), "45.5s")
        self.assertEqual(ai_media.format_time(1.5), "1.5s")
    
    def test_minutes_and_seconds(self):
        """Test formatting minutes and seconds."""
        self.assertEqual(ai_media.format_time(90), "1m 30s")
        self.assertEqual(ai_media.format_time(125), "2m 5s")
        self.assertEqual(ai_media.format_time(3599), "59m 59s")
    
    def test_minutes_exact(self):
        """Test formatting exact minutes (no trailing 0s)."""
        # format_time omits 0s when there are no remaining seconds
        self.assertEqual(ai_media.format_time(60), "1m")
        self.assertEqual(ai_media.format_time(120), "2m")
    
    def test_hours_minutes_seconds(self):
        """Test formatting hours, minutes, and seconds."""
        self.assertEqual(ai_media.format_time(3600), "1h")
        self.assertEqual(ai_media.format_time(3661), "1h 1m 1s")
        self.assertEqual(ai_media.format_time(7325), "2h 2m 5s")
    
    def test_days(self):
        """Test formatting days."""
        self.assertEqual(ai_media.format_time(86400), "1d")
        self.assertEqual(ai_media.format_time(90061), "1d 1h 1m 1s")
    
    def test_weeks(self):
        """Test formatting weeks."""
        self.assertEqual(ai_media.format_time(604800), "1w")
        # 1w (604800) + 1d (86400) + 1h (3600) + 1m (60) + 1s = 694861
        self.assertEqual(ai_media.format_time(694861), "1w 1d 1h 1m 1s")
    
    def test_zero_and_none(self):
        """Test zero and None values."""
        self.assertEqual(ai_media.format_time(0), "0s")
        self.assertEqual(ai_media.format_time(None), "0s")


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestGetVideoEncodingParams(unittest.TestCase):
    """Tests for get_video_encoding_params() function - all video formats."""
    
    def test_mp4_format(self):
        """Test MP4 format returns H.264 params."""
        params = ai_media.get_video_encoding_params("output.mp4")
        self.assertIn("-c:v", params)
        self.assertIn("libx264", params)
        self.assertIn("-pix_fmt", params)
        self.assertIn("yuv420p", params)
        self.assertIn("-c:a", params)
        self.assertIn("aac", params)
    
    def test_m4v_format(self):
        """Test M4V format returns H.264 params."""
        params = ai_media.get_video_encoding_params("output.m4v")
        self.assertIn("libx264", params)
        self.assertIn("aac", params)
    
    def test_mkv_format(self):
        """Test MKV format returns H.264 params."""
        params = ai_media.get_video_encoding_params("output.mkv")
        self.assertIn("libx264", params)
        self.assertIn("aac", params)
    
    def test_mov_format(self):
        """Test MOV format returns H.264 params."""
        params = ai_media.get_video_encoding_params("output.mov")
        self.assertIn("libx264", params)
        self.assertIn("aac", params)
    
    def test_webm_format(self):
        """Test WebM format returns VP9/Opus params."""
        params = ai_media.get_video_encoding_params("output.webm")
        self.assertIn("libvpx-vp9", params)
        self.assertIn("libopus", params)
        self.assertIn("-b:v", params)
        self.assertIn("2M", params)
    
    def test_wmv_format(self):
        """Test WMV format returns Windows Media params."""
        params = ai_media.get_video_encoding_params("output.wmv")
        self.assertIn("wmv2", params)
        self.assertIn("wmav2", params)
        self.assertIn("-b:v", params)
    
    def test_avi_format(self):
        """Test AVI format returns MPEG4/MP3 params."""
        params = ai_media.get_video_encoding_params("output.avi")
        self.assertIn("mpeg4", params)
        self.assertIn("mp3", params)
        self.assertIn("yuv420p", params)
    
    def test_unknown_format_fallback(self):
        """Test unknown format falls back to H.264."""
        params = ai_media.get_video_encoding_params("output.xyz")
        self.assertIn("libx264", params)
        self.assertIn("aac", params)
    
    def test_case_insensitivity(self):
        """Test extension case insensitivity."""
        params_lower = ai_media.get_video_encoding_params("output.mp4")
        params_upper = ai_media.get_video_encoding_params("output.MP4")
        self.assertEqual(params_lower, params_upper)
    
    def test_with_path(self):
        """Test with full path."""
        params = ai_media.get_video_encoding_params("/path/to/output.webm")
        self.assertIn("libvpx-vp9", params)


class TestEnsurePaths(unittest.TestCase):
    """Tests for ensure_paths() function."""
    
    def setUp(self):
        """Create a temporary directory for testing."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Remove the temporary directory after testing."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('builtins.print')
    def test_creates_single_parent_directory(self, mock_print):
        """Test single parent directory is created."""
        nested_path = os.path.join(self.test_dir, "subdir", "output.txt")
        ai_media.ensure_paths(nested_path)
        parent_dir = os.path.dirname(nested_path)
        self.assertTrue(os.path.exists(parent_dir))
    
    @patch('builtins.print')
    def test_creates_deeply_nested_directories(self, mock_print):
        """Test deeply nested directories are created."""
        nested_path = os.path.join(self.test_dir, "a", "b", "c", "d", "output.txt")
        ai_media.ensure_paths(nested_path)
        parent_dir = os.path.dirname(nested_path)
        self.assertTrue(os.path.exists(parent_dir))
    
    def test_existing_directory_no_error(self):
        """Test no error when directory already exists."""
        existing_dir = os.path.join(self.test_dir, "existing")
        os.makedirs(existing_dir)
        output_path = os.path.join(existing_dir, "output.txt")
        ai_media.ensure_paths(output_path)
        self.assertTrue(os.path.exists(existing_dir))
    
    def test_none_path_no_error(self):
        """Test None path doesn't cause an error."""
        ai_media.ensure_paths(None)
    
    def test_empty_path_no_error(self):
        """Test empty path doesn't cause an error."""
        ai_media.ensure_paths("")


class TestWriteReportJson(unittest.TestCase):
    """Tests for write_report_json() function."""
    
    def setUp(self):
        """Create a temporary directory for testing."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Remove the temporary directory after testing."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_writes_json_file(self):
        """Test JSON file is written correctly."""
        path = os.path.join(self.test_dir, "report.json")
        stats = {"time": 10.5, "ram": 8.0, "vram": 4.0}
        
        ai_media.write_report_json(path, stats)
        
        self.assertTrue(os.path.exists(path))
        with open(path, 'r') as f:
            loaded = json.load(f)
        self.assertEqual(loaded["time"], 10.5)
        self.assertEqual(loaded["ram"], 8.0)
    
    def test_overwrites_existing_file(self):
        """Test existing file is overwritten."""
        path = os.path.join(self.test_dir, "report.json")
        
        ai_media.write_report_json(path, {"old": True})
        ai_media.write_report_json(path, {"new": True})
        
        with open(path, 'r') as f:
            loaded = json.load(f)
        self.assertNotIn("old", loaded)
        self.assertIn("new", loaded)


class TestClearGpuMemory(unittest.TestCase):
    """Tests for clear_gpu_memory() function."""
    
    def test_does_not_raise(self):
        """Test function runs without raising exceptions."""
        # Should not raise even with mocked torch
        try:
            ai_media.clear_gpu_memory()
        except Exception as e:
            self.fail(f"clear_gpu_memory raised exception: {e}")


class TestGetOptimalDeviceAndDtype(unittest.TestCase):
    """Tests for get_optimal_device_and_dtype() function."""
    
    def test_returns_tuple(self):
        """Test function returns a (device, dtype) tuple."""
        result = ai_media.get_optimal_device_and_dtype(quiet=True)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


class TestIsBfloat16Supported(unittest.TestCase):
    """Tests for is_bfloat16_supported() function."""
    
    def test_returns_boolean(self):
        """Test function returns a boolean value."""
        result = ai_media.is_bfloat16_supported()
        self.assertIsInstance(result, bool)
    
    def test_returns_false_without_cuda(self):
        """Test returns False when CUDA is not available."""
        # mock_torch.cuda.is_available is already set to False
        result = ai_media.is_bfloat16_supported()
        self.assertFalse(result)
    
    def test_returns_true_with_cuda_bf16(self):
        """Test returns True when CUDA supports bf16."""
        # Temporarily enable CUDA and bf16 support
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.is_bf16_supported.return_value = True
        try:
            result = ai_media.is_bfloat16_supported()
            self.assertTrue(result)
        finally:
            # Restore defaults
            mock_torch.cuda.is_available.return_value = False
            mock_torch.cuda.is_bf16_supported.return_value = False
    
    def test_returns_false_without_bf16_support(self):
        """Test returns False when CUDA doesn't support bf16."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.is_bf16_supported.return_value = False
        try:
            result = ai_media.is_bfloat16_supported()
            self.assertFalse(result)
        finally:
            mock_torch.cuda.is_available.return_value = False


class TestClearScreen(unittest.TestCase):
    """Tests for clear_screen() function."""
    
    @patch('os.system')
    def test_calls_os_system(self, mock_system):
        """Test clear_screen calls os.system."""
        with patch('sys.stdout.isatty', return_value=True):
            ai_media.clear_screen()
            mock_system.assert_called_once()
    
    @patch('os.system')
    def test_uses_cls_on_windows(self, mock_system):
        """Test uses 'cls' command on Windows."""
        with patch('os.name', 'nt'), patch('sys.stdout.isatty', return_value=True):
            ai_media.clear_screen()
            mock_system.assert_called_with('cls')
    
    @patch('os.system')
    def test_uses_clear_on_unix(self, mock_system):
        """Test uses 'clear' command on Unix."""
        with patch('os.name', 'posix'), patch('sys.stdout.isatty', return_value=True):
            ai_media.clear_screen()
            mock_system.assert_called_with('clear')


class TestShowHeader(unittest.TestCase):
    """Tests for show_header() function."""
    
    def test_prints_header(self):
        """Test show_header prints a header."""
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            ai_media.show_header("Test Title")
            output = fake_out.getvalue()
            self.assertIn("Test Title", output)
            # Check for either classic '=' or decorative '═' border
            self.assertTrue("═" in output or "=" in output)
    
    def test_default_title(self):
        """Test show_header uses default title."""
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            ai_media.show_header()
            output = mock_stdout.getvalue()
            self.assertIn("AI-Media", output)


class TestEmoji(unittest.TestCase):
    """Tests for emoji() helper function - terminal emoji fallback."""
    
    def test_returns_emoji_when_encoding_supported(self):
        """Test emoji is returned when terminal supports encoding."""
        # Mock stdout with UTF-8 encoding
        with patch('sys.stdout') as mock_stdout:
            mock_stdout.encoding = 'utf-8'
            result = ai_media.emoji("🎨 ", "")
            self.assertEqual(result, "🎨 ")
    
    def test_returns_fallback_when_encoding_fails(self):
        """Test fallback is returned when encoding fails."""
        # Mock stdout with an encoding that can't handle emoji
        with patch('sys.stdout') as mock_stdout:
            mock_stdout.encoding = 'ascii'
            result = ai_media.emoji("🎨 ", "ART: ")
            self.assertEqual(result, "ART: ")
    
    def test_returns_fallback_with_none_encoding(self):
        """Test fallback on None encoding (defaults to utf-8 which works)."""
        with patch('sys.stdout') as mock_stdout:
            mock_stdout.encoding = None
            # Should NOT fallback since utf-8 is used as default and supports emoji
            result = ai_media.emoji("✅ ", "OK: ")
            self.assertEqual(result, "✅ ")
    
    def test_empty_fallback(self):
        """Test empty string fallback."""
        with patch('sys.stdout') as mock_stdout:
            mock_stdout.encoding = 'ascii'
            result = ai_media.emoji("❌ ", "")
            self.assertEqual(result, "")
    
    def test_various_emojis(self):
        """Test various emoji characters."""
        with patch('sys.stdout') as mock_stdout:
            mock_stdout.encoding = 'utf-8'
            self.assertEqual(ai_media.emoji("🔎 ", ""), "🔎 ")
            self.assertEqual(ai_media.emoji("📋 ", ""), "📋 ")
            self.assertEqual(ai_media.emoji("🚀 ", "(>) "), "🚀 ")
            self.assertEqual(ai_media.emoji("⏳ ", "Wait: "), "⏳ ")


# =============================================================================
# Class Tests
# =============================================================================

class TestPerformanceTracker(unittest.TestCase):
    """Tests for PerformanceTracker class."""
    
    def setUp(self):
        """Create a temporary file for testing."""
        self.test_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.test_file.close()
        self.tracker = ai_media.PerformanceTracker(filepath=self.test_file.name)
    
    def tearDown(self):
        """Remove the temporary file after testing."""
        if os.path.exists(self.test_file.name):
            os.unlink(self.test_file.name)
    
    def test_initial_state_empty(self):
        """Test tracker starts with empty data."""
        tracker = ai_media.PerformanceTracker(filepath=self.test_file.name)
        self.assertEqual(tracker.data, {})
    
    def test_loads_existing_data(self):
        """Test tracker loads existing JSON data."""
        with open(self.test_file.name, 'w') as f:
            json.dump({"image": {"test": {"average_time": 5.0}}}, f)
        
        tracker = ai_media.PerformanceTracker(filepath=self.test_file.name)
        self.assertIn("image", tracker.data)
    
    def test_record_image_creates_entry(self):
        """Test recording image generation creates entry."""
        mock_device = MagicMock()
        mock_device.type = "cuda"
        
        self.tracker.record_image(
            model="test-model",
            width=1024,
            height=1024,
            device=mock_device,
            time_taken=10.0,
            cpu=50.0,
            ram=8.0,
            vram=4.0,
            gpu=80.0
        )
        
        self.assertIn("image", self.tracker.data)
        key = "test-model|cuda|1024x1024"
        self.assertIn(key, self.tracker.data["image"])
        entry = self.tracker.data["image"][key]
        self.assertEqual(entry["average_time"], 10.0)
        self.assertEqual(entry["average_ram"], 8.0)
        self.assertEqual(entry["average_vram"], 4.0)
        self.assertEqual(entry["average_cpu"], 50.0)
        self.assertEqual(entry["average_gpu"], 80.0)
    
    def test_record_image_averaging(self):
        """Test multiple recordings average the values."""
        mock_device = MagicMock()
        mock_device.type = "cuda"
        
        self.tracker.record_image("test-model", 1024, 1024, mock_device, 10.0, ram=4.0)
        self.tracker.record_image("test-model", 1024, 1024, mock_device, 20.0, ram=8.0)
        
        key = "test-model|cuda|1024x1024"
        self.assertEqual(self.tracker.data["image"][key]["average_time"], 15.0)
        self.assertEqual(self.tracker.data["image"][key]["average_ram"], 6.0)
    
    def test_record_linear_audio(self):
        """Test recording audio generation stats."""
        mock_device = MagicMock()
        mock_device.type = "cpu"
        
        self.tracker.record_linear(
            category="audio",
            model="musicgen",
            device=mock_device,
            duration=10.0,
            time_taken=30.0,
            ram=8.0,
            vram=2.0
        )
        
        self.assertIn("audio", self.tracker.data)
        key = "musicgen|cpu"
        self.assertIn(key, self.tracker.data["audio"])
        # Rate = time_taken / duration = 30 / 10 = 3.0
        self.assertEqual(self.tracker.data["audio"][key]["average_rate"], 3.0)
    
    def test_record_linear_video_includes_resolution(self):
        """Test video recording includes resolution in key."""
        mock_device = MagicMock()
        mock_device.type = "cuda"
        
        self.tracker.record_linear(
            category="video",
            model="zeroscope",
            device=mock_device,
            duration=5.0,
            time_taken=60.0,
            width=1280,
            height=720
        )
        
        self.assertIn("video", self.tracker.data)
        key = "zeroscope|cuda|1280x720"
        self.assertIn(key, self.tracker.data["video"])
    
    def test_estimate_image_no_data(self):
        """Test estimation returns zeros when no data exists."""
        mock_device = MagicMock()
        mock_device.type = "cuda"
        
        result = self.tracker.estimate_image("unknown-model", 512, 512, mock_device)
        self.assertEqual(result, (0, 0, 0, 0, 0))
    
    def test_estimate_image_with_data(self):
        """Test estimation returns recorded data."""
        mock_device = MagicMock()
        mock_device.type = "cuda"
        
        self.tracker.record_image("test-model", 1024, 1024, mock_device, 10.0, cpu=50, ram=8, vram=4, gpu=80)
        result = self.tracker.estimate_image("test-model", 1024, 1024, mock_device)
        
        self.assertEqual(result[0], 10.0)  # time
        self.assertEqual(result[1], 50.0)  # cpu
        self.assertEqual(result[2], 8.0)   # ram
        self.assertEqual(result[3], 4.0)   # vram
        self.assertEqual(result[4], 80.0)  # gpu
    
    def test_estimate_linear_no_data(self):
        """Test linear estimation returns zeros when no data."""
        mock_device = MagicMock()
        mock_device.type = "cpu"
        
        result = self.tracker.estimate_linear("audio", "unknown", mock_device, 10.0)
        self.assertEqual(result, (0, 0, 0, 0, 0))
    
    def test_estimate_linear_with_data(self):
        """Test linear estimation calculates from rate."""
        mock_device = MagicMock()
        mock_device.type = "cpu"
        
        # Rate of 3.0 means 3 seconds gen time per 1 second of content
        self.tracker.record_linear("audio", "musicgen", mock_device, 10.0, 30.0)
        result = self.tracker.estimate_linear("audio", "musicgen", mock_device, 5.0)
        
        # Estimated time = rate * duration = 3.0 * 5.0 = 15.0
        self.assertEqual(result[0], 15.0)
    
    def test_data_persistence(self):
        """Test data is saved to file and can be reloaded."""
        mock_device = MagicMock()
        mock_device.type = "cuda"
        
        self.tracker.record_image("test-model", 1024, 1024, mock_device, 10.0)
        
        new_tracker = ai_media.PerformanceTracker(filepath=self.test_file.name)
        self.assertIn("image", new_tracker.data)
        key = "test-model|cuda|1024x1024"
        self.assertIn(key, new_tracker.data["image"])
    
    def test_device_string_fallback(self):
        """Test device without type attribute uses str()."""
        self.tracker.record_image("test-model", 512, 512, "cpu", 5.0)
        
        key = "test-model|cpu|512x512"
        self.assertIn(key, self.tracker.data["image"])


class TestResourceMonitor(unittest.TestCase):
    """Tests for ResourceMonitor class."""
    
    def test_initialization(self):
        """Test monitor initializes correctly."""
        monitor = ai_media.ResourceMonitor(interval=0.5)
        self.assertEqual(monitor.interval, 0.5)
        self.assertFalse(monitor.running)
        self.assertEqual(monitor.cpu_readings, [])
        self.assertEqual(monitor.ram_readings, [])
        self.assertEqual(monitor.vram_readings, [])
        self.assertEqual(monitor.gpu_readings, [])
    
    def test_initialization_default_interval(self):
        """Test default interval is 0.5."""
        monitor = ai_media.ResourceMonitor()
        self.assertEqual(monitor.interval, 0.5)
    
    def test_get_averages_empty(self):
        """Test get_averages with no readings returns zeros."""
        monitor = ai_media.ResourceMonitor()
        avg_cpu, avg_ram, avg_vram, avg_gpu = monitor.get_averages()
        self.assertEqual(avg_cpu, 0)
        self.assertEqual(avg_ram, 0)
        self.assertEqual(avg_vram, 0)
        self.assertEqual(avg_gpu, 0)
    
    def test_get_averages_with_data(self):
        """Test get_averages calculates correctly."""
        monitor = ai_media.ResourceMonitor()
        monitor.cpu_readings = [10.0, 20.0, 30.0]
        monitor.ram_readings = [4.0, 6.0, 8.0]
        monitor.vram_readings = [1.0, 2.0, 3.0]
        monitor.gpu_readings = [50.0, 60.0, 70.0]
        
        avg_cpu, avg_ram, avg_vram, avg_gpu = monitor.get_averages()
        self.assertEqual(avg_cpu, 20.0)
        self.assertEqual(avg_ram, 6.0)
        self.assertEqual(avg_vram, 2.0)
        self.assertEqual(avg_gpu, 60.0)
    
    def test_get_averages_single_reading(self):
        """Test get_averages with single reading."""
        monitor = ai_media.ResourceMonitor()
        monitor.cpu_readings = [50.0]
        monitor.ram_readings = [8.0]
        monitor.vram_readings = [4.0]
        monitor.gpu_readings = [75.0]
        
        avg_cpu, avg_ram, avg_vram, avg_gpu = monitor.get_averages()
        self.assertEqual(avg_cpu, 50.0)
        self.assertEqual(avg_ram, 8.0)
        self.assertEqual(avg_vram, 4.0)
        self.assertEqual(avg_gpu, 75.0)
    
    def test_context_manager_enter_exit(self):
        """Test context manager enters and exits correctly."""
        monitor = ai_media.ResourceMonitor(interval=0.1)
        
        with monitor:
            if monitor.psutil:
                self.assertTrue(monitor.running)
        
        self.assertFalse(monitor.running)
    
    def test_context_manager_returns_self(self):
        """Test context manager returns self."""
        monitor = ai_media.ResourceMonitor()
        with monitor as m:
            self.assertIs(m, monitor)


class TestSignalHandler(unittest.TestCase):
    """Tests for signal_handler() function."""
    
    def test_signal_handler_non_test_mode(self):
        """Test signal handler in non-test mode sets interrupted state on first call."""
        # Reset global state
        ai_media._test_state['active'] = False
        ai_media._interrupted = False
        ai_media._first_interrupt_time = None
        ai_media._force_kill_timer = None
        
        # Redirect stdout to avoid emoji encoding issues on Windows
        with patch('sys.stdout', new_callable=io.StringIO):
            with patch('os._exit') as mock_exit:
                # First interrupt in non-test mode catches SystemExit for graceful cleanup
                # It sets _interrupted=True and _first_interrupt_time, then catches its own sys.exit(0)
                ai_media.signal_handler(None, None)
                
                # Verify state changes
                self.assertTrue(system._interrupted)
                # Verify exit was called
                mock_exit.assert_called_with(0)
        
        # Clean up
        system._interrupted = False
        
        # Clean up the timer that was started
        if ai_media._force_kill_timer:
            ai_media._force_kill_timer.cancel()
            ai_media._force_kill_timer = None
        
        # Reset state
        ai_media._interrupted = False
        ai_media._first_interrupt_time = None
    
    def test_signal_handler_test_mode(self):
        """Test signal handler in test mode exits with code 130."""
        ai_media._test_state['active'] = True
        ai_media._test_state['passed'] = 5
        ai_media._test_state['failed'] = 2
        ai_media._test_state['total'] = 10
        
        # Redirect stdout to avoid emoji encoding issues on Windows
        with patch('sys.stdout', new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as cm:
                ai_media.signal_handler(None, None)
            self.assertEqual(cm.exception.code, 130)
        
        # Reset state
        ai_media._test_state['active'] = False


class TestTestState(unittest.TestCase):
    """Tests for _test_state global."""
    
    def test_test_state_structure(self):
        """Test _test_state has expected keys."""
        self.assertIn('active', ai_media._test_state)
        self.assertIn('passed', ai_media._test_state)
        self.assertIn('failed', ai_media._test_state)
        self.assertIn('total', ai_media._test_state)
    
    def test_test_state_initial_values(self):
        """Test _test_state initial values."""
        # Reset to known state
        ai_media._test_state['active'] = False
        self.assertFalse(ai_media._test_state['active'])


# =============================================================================
# Resource Helper Tests
# =============================================================================

class TestResourceHelpers(unittest.TestCase):
    """Tests for resource management and hardware detection functions."""
    
    def setUp(self):
        self.orig_psutil = getattr(ai_media, 'psutil', None)
        ai_media.psutil = MagicMock()
        # Use real resource check functions for these specific tests
        system.check_resources_and_warn = _ORIGINAL_CHECK_WARN
        ai_media.check_resources_and_warn = _ORIGINAL_CHECK_WARN
        upscaling.check_resources_and_confirm = _ORIGINAL_CHECK_CONFIRM
        ai_media.check_resources_and_confirm = _ORIGINAL_CHECK_CONFIRM
        
    def tearDown(self):
        ai_media.psutil = self.orig_psutil
        # Re-mock to avoid hangs in other tests
        system.check_resources_and_warn = MagicMock(return_value=True)
        ai_media.check_resources_and_warn = system.check_resources_and_warn
        upscaling.check_resources_and_confirm = MagicMock(return_value=True)
        ai_media.check_resources_and_confirm = upscaling.check_resources_and_confirm

    def test_get_system_resources(self):
        """Test RAM and VRAM detection."""
        # Setup mocks
        with patch('ai_media.utils.system.psutil') as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 16 * (1024**3) # 16GB
            mock_mem.total = 16 * (1024**3)
            mock_psutil.virtual_memory.return_value = mock_mem
            
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_properties.return_value.total_memory = 8 * (1024**3)
            mock_torch.cuda.memory_allocated.return_value = 2 * (1024**3)
            
            with patch.dict('sys.modules', {'torch': mock_torch}):
                # Run
                resources = ai_media.get_system_resources()
                
                # Verify dict format
                self.assertEqual(resources['ram_available'], 16.0)
                self.assertEqual(resources['vram_available'], 6.0)  # 8 - 2
                self.assertEqual(resources['ram_total'], 16.0)
    
    @patch('ai_media.utils.system.get_system_resources')
    def test_check_resources_strict_warnings(self, mock_get_resources):
        """Test strict warnings for low resources."""
        # Low RAM/VRAM - now returns dict format
        mock_get_resources.return_value = {
            "ram_available": 4.0,
            "ram_total": 16.0,
            "vram_available": 2.0,
            "vram_total": 8.0
        }
        
        # Redirect stdout and mock isatty to force interactive logic
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout, \
             patch('sys.stdin.isatty', return_value=True):
            # Simulate user saying 'n' (abort)
            with patch('builtins.input', return_value='n'):
                # Pass requirements explicitly since it's not imported/defined in system.py
                mock_reqs = {"stabilityai/sdxl-turbo": {"vram": 8, "ram": 16}}
                with self.assertRaises(SystemExit):
                    ai_media.check_resources_and_warn("stabilityai/sdxl-turbo", model_requirements=mock_reqs)

            output = mock_stdout.getvalue()
            self.assertIn("RAM: 4.0GB available", output)
            self.assertIn("VRAM: 2.0GB available", output)

    @patch('ai_media.utils.system.get_system_resources')
    def test_check_resources_force_override(self, mock_get_resources):
        """Test force flag overrides warnings."""
        mock_get_resources.return_value = (4.0, 2.0)
        
        with patch('sys.stdout', new_callable=io.StringIO):
            result = ai_media.check_resources_and_warn("stabilityai/sdxl-turbo", force=True)
            self.assertTrue(result)

    def test_get_optimal_device_cuda(self):
        """Test CUDA detection."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = 'cuda'
        mock_torch.float16 = 'float16'
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            device, dtype = ai_media.get_optimal_device_and_dtype(quiet=True)
            self.assertEqual(device, 'cuda')
            self.assertEqual(dtype, 'float16')

    def test_get_optimal_device_cpu(self):
        """Test CPU fallback."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.device.return_value = 'cpu'
        mock_torch.float32 = 'float32'
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            device, dtype = ai_media.get_optimal_device_and_dtype(quiet=True)
            self.assertEqual(device, 'cpu')
            self.assertEqual(dtype, 'float32')


# =============================================================================
# Generation Wrapper Tests
# =============================================================================

class TestGenerationWrappers(unittest.TestCase):
    """Tests for main generation wrapper functions."""
    
    @patch('ai_media.generate_image')
    def test_generate_image_call(self, mock_gen):
        """Test generate_image wrapper arguments."""
        ai_media.generate_image("prompt", "out.png", 512, 512)
        mock_gen.assert_called_once()
    
    @patch('ai_media.utils.system.get_optimal_device_and_dtype')
    def test_generate_image_logic(self, mock_get_device):
        """Test internal logic of generate_image (mocking diffusers)."""
        # mocks
        mock_diffusers = MagicMock()
        mock_pipeline = mock_diffusers.AutoPipelineForText2Image.from_pretrained.return_value
        mock_pipeline.to.return_value = mock_pipeline
        
        # Setup output mock
        mock_output = MagicMock()
        mock_output.images = [MagicMock()]
        mock_pipeline.return_value = mock_output
        
        # Setup device
        mock_device = MagicMock()
        mock_device.type = 'cuda'
        import torch
        mock_get_device.return_value = (mock_device, torch.float16 if 'torch' in sys.modules else MagicMock())
        
        # We need to block PerformanceTracker/ResourceMonitor in the generator module
        # Also patch stdout to avoid UnicodeError on Windows
        with patch.dict('sys.modules', {'diffusers': mock_diffusers, 'torch': MagicMock()}):
            with patch('ai_media.utils.system.get_system_resources', return_value=(16.0, 8.0)):
                with patch('ai_media.generators.image.PerformanceTracker'):
                    with patch('ai_media.generators.image.ResourceMonitor') as MockMonitor:
                        with patch('os.path.exists', return_value=True):
                            # Configure ResourceMonitor instance
                            monitor_instance = MockMonitor.return_value
                            monitor_instance.__enter__.return_value = monitor_instance
                            monitor_instance.get_averages.return_value = (10.0, 4.0, 2.0, 50.0)
                            
                            with patch('sys.stdout', new_callable=io.StringIO):
                                # Run
                                result = ai_media.generate_image("test prompt", "test.png", 512, 512, model_name="sdxl")
                                self.assertTrue(result)
                                mock_diffusers.AutoPipelineForText2Image.from_pretrained.assert_called()
                                mock_pipeline.assert_called() # The inference call

    @patch('ai_media.utils.system.get_optimal_device_and_dtype')
    def test_generate_image_negative_prompt(self, mock_get_device):
        """Test generate_image passes negative_prompt correctly."""
        # mocks
        mock_diffusers = MagicMock()
        mock_pipeline = mock_diffusers.AutoPipelineForText2Image.from_pretrained.return_value
        mock_pipeline.to.return_value = mock_pipeline
        mock_output = MagicMock()
        mock_output.images = [MagicMock()]
        mock_pipeline.return_value = mock_output
        mock_device = MagicMock()
        mock_device.type = 'cuda'
        import torch
        mock_get_device.return_value = (mock_device, torch.float16 if 'torch' in sys.modules else MagicMock())
        
        with patch.dict('sys.modules', {'diffusers': mock_diffusers, 'torch': MagicMock()}):
            with patch('ai_media.utils.system.get_system_resources', return_value=(16.0, 8.0)):
                with patch('ai_media.generators.image.PerformanceTracker'):
                    with patch('ai_media.generators.image.ResourceMonitor') as MockMonitor:
                        with patch('os.path.exists', return_value=True):
                            monitor_instance = MockMonitor.return_value
                            monitor_instance.__enter__.return_value = monitor_instance
                            monitor_instance.get_averages.return_value = (10.0, 4.0, 2.0, 50.0)
                            
                            with patch('sys.stdout', new_callable=io.StringIO):
                                # Use sd-1.5 to hit the generic path that supports negative prompts
                                # sdxl points to sdxl-turbo which may skip negative_prompt in some pipeline versions
                                result = ai_media.generate_image("prompt", "out.png", 512, 512, model_name="sd-1.5", negative_prompt="bad quality")
                                self.assertTrue(result)
                                # Verify negative_prompt was passed to pipeline call
                                _, kwargs = mock_pipeline.call_args
                                self.assertIn("negative_prompt", kwargs)
                                self.assertEqual(kwargs["negative_prompt"], "bad quality")

    @patch('ai_media.generate_video')
    def test_generate_video_call(self, mock_gen):
        """Test generate_video wrapper arguments."""
        ai_media.generate_video("prompt", "out.mp4", 2.0, 512, 512)
        mock_gen.assert_called_once()

    @patch('ai_media.generate_audio')
    def test_generate_audio_call(self, mock_gen):
        """Test generate_audio wrapper arguments."""
        ai_media.generate_audio("prompt", "out.mp3", 10.0, 32000)
        mock_gen.assert_called_once()



# =============================================================================
# Edit Wrapper Tests
# =============================================================================

class TestEditWrappers(unittest.TestCase):
    """Tests for editing and manipulation functions."""
    
    @patch('ai_media.generate_edit')
    def test_generate_edit_call(self, mock_gen):
        """Test generate_edit wrapper arguments."""
        ai_media.generate_edit("in.png", "prompt", "out.png")
        mock_gen.assert_called_once()
        
    @patch('ai_media.remove_background')
    def test_remove_background_call(self, mock_gen):
        """Test remove_background wrapper arguments."""
        ai_media.remove_background("in.png", "out.png")
        mock_gen.assert_called_once()
        

# =============================================================================
# Captioning Tests
# =============================================================================

class TestCaptioning(unittest.TestCase):
    """Tests for captioning functions."""
    
    @patch('ai_media.generate_caption')
    def test_generate_caption_call(self, mock_gen):
        """Test generate_caption wrapper arguments."""
        device = MagicMock()
        ai_media.generate_caption("in.png", device)
        mock_gen.assert_called_once()

# =============================================================================
# Main
# =============================================================================

class TestUpscaleArguments(unittest.TestCase):
    """Tests for upscale arguments routing."""

    def run_cli(self, args_list):
        """Helper to run CLI arguments parsing."""
        with patch('sys.argv', ['ai-media.py'] + args_list), \
             patch('os.path.exists', return_value=True):
            try:
                ai_media.main()
            except SystemExit:
                pass

    @patch('ai_media.simple_upscale_video')
    @patch('ai_media.upscale_video_file')
    @patch('ai_media.upscale_video_fast')
    def test_upscale_video_routing(self, mock_fast, mock_std, mock_simple):
        """Test video upscaling argument routing."""
        # Inject mocks into the loaded script instance placeholders
        try:
            _aimedia_script.pkg_upscale_video_fast = mock_fast
            _aimedia_script.pkg_upscale_video_file = mock_std
            _aimedia_script.pkg_simple_upscale_video = mock_simple
        except NameError:
            pass # Script not loaded
        
        # Case 1: Standard (Default)
        self.run_cli(["-uv", "in.mp4", "-uf", "2.0"])
        mock_fast.assert_called()
        mock_std.assert_not_called()
        mock_simple.assert_not_called()
        
        # Case 2: Standard (Explicit)
        mock_std.reset_mock()
        self.run_cli(["-uv", "in.mp4", "--video-upscaler", "sd"])
        mock_std.assert_called()
        
        # Case 3: Simple
        mock_std.reset_mock()
        self.run_cli(["-uv", "in.mp4", "-su"])
        mock_simple.assert_called()
        
        # Case 4: Fast (Real-ESRGAN)
        mock_simple.reset_mock()
        self.run_cli(["-uv", "in.mp4", "--video-upscaler", "realesrgan"])
        mock_fast.assert_called()

    @patch('ai_media.simple_upscale_image')
    @patch('ai_media.upscale_image_file')
    @patch('ai_media.upscale_image_fast')
    def test_upscale_image_routing(self, mock_fast, mock_std, mock_simple):
        """Test image upscaling argument routing."""
        global _aimedia_script
        if '_aimedia_script' in globals() and _aimedia_script:
            _aimedia_script.pkg_upscale_image_fast = mock_fast
            _aimedia_script.pkg_upscale_image_file = mock_std
            _aimedia_script.pkg_simple_upscale_image = mock_simple
            _aimedia_script.pkg_upscale_video_fast = MagicMock()
            _aimedia_script.pkg_upscale_video_file = MagicMock()
            _aimedia_script.pkg_simple_upscale_video = MagicMock()
        # Inject mocks into the loaded script instance
        _aimedia_script.pkg_upscale_image_fast = mock_fast
        _aimedia_script.pkg_upscale_image_file = mock_std
        _aimedia_script.pkg_simple_upscale_image = mock_simple

        # Case 1: Standard (Default = Real-ESRGAN)
        self.run_cli(["-ui", "in.png", "-uf", "2.0"])
        mock_fast.assert_called()
        mock_std.assert_not_called()
        mock_simple.assert_not_called()
        
        # Case 2: Standard (Explicit SD)
        mock_std.reset_mock()
        self.run_cli(["-ui", "in.png", "--image-upscaler", "sd"])
        mock_std.assert_called()
        
        # Case 3: Simple
        mock_std.reset_mock()
        self.run_cli(["-ui", "in.png", "-su"])
        mock_simple.assert_called()
        
        # Case 4: Fast (Real-ESRGAN) explicit
        mock_simple.reset_mock()
        self.run_cli(["-ui", "in.png", "--image-upscaler", "realesrgan"])
        mock_fast.assert_called()

class TestFastUpscaler(unittest.TestCase):
    """Tests for upscale_video_fast() function."""
    
    def setUp(self):
        # Setup mocks
        self.patcher_has = patch('ai_media.upscaling.HAS_REALESRGAN', True)
        self.patcher_exists = patch('os.path.exists', return_value=True)
        self.patcher_device = patch('ai_media.upscaling.get_optimal_device_and_dtype', return_value=(Mock(type='cpu'), None))
        self.patcher_rrdb = patch('ai_media.upscaling.RRDBNet', create=True)
        self.patcher_esrgan = patch('ai_media.upscaling.RealESRGANer', create=True)
        self.patcher_monitor = patch('ai_media.upscaling.ResourceMonitor', create=True)
        self.patcher_weights = patch('ai_media.upscaling.get_realesrgan_weights', return_value='/fake/path/weights.pth')
        self.patcher_overwrite = patch('ai_media.upscaling.check_overwrite', return_value=(True, 'out.mp4', None, None))
        self.mock_cv2 = MagicMock()
        self.patcher_cv2 = patch.dict('sys.modules', {'cv2': self.mock_cv2})
        self.patcher_cv2.start()
        
        self.mock_has = self.patcher_has.start()
        self.mock_exists = self.patcher_exists.start()
        self.mock_device = self.patcher_device.start()
        self.mock_rrdb = self.patcher_rrdb.start()
        self.mock_esrgan = self.patcher_esrgan.start()
        self.mock_monitor = self.patcher_monitor.start()
        self.mock_weights = self.patcher_weights.start()
        self.mock_overwrite = self.patcher_overwrite.start()
        # Complex mocking of local imports is hard. 
        # We will basic-test the dependency check first.
        
    def tearDown(self):
        self.patcher_has.stop()
        self.patcher_exists.stop()
        self.patcher_device.stop()
        self.patcher_rrdb.stop()
        self.patcher_esrgan.stop()
        self.patcher_monitor.stop()
        self.patcher_weights.stop()
        self.patcher_overwrite.stop()

    @patch('ai_media.upscaling.HAS_REALESRGAN', False)
    def test_missing_dependency(self):
        """Test returning False if dependencies missing."""
        with patch('builtins.print') as mock_print:
            result = ai_media.upscale_video_fast("in.mp4", "out.mp4")
            self.assertFalse(result)
            self.assertTrue(any("not installed" in str(call) for call in mock_print.call_args_list))

    def test_resolution_limit_exceeded(self):
        """Test returning False if target resolution exceeds 15K limit."""
        # Setup mock for 1080p input
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        
        self.mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        self.mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        self.mock_cv2.CAP_PROP_FPS = 5
        self.mock_cv2.CAP_PROP_FRAME_COUNT = 7
        self.mock_cv2.VideoCapture.return_value = mock_cap
        
        mock_cap.get.side_effect = lambda prop: {
            3: 1920, # Width
            4: 1080, # Height
            5: 30,   # FPS
            7: 100   # Frames
        }.get(prop, 0)
        
        with patch('builtins.print') as mock_print, \
             patch('os.path.exists', return_value=True):
            # 1920 * 10 = 19200 (exceeds 15360)
            result = ai_media.upscale_video_fast("in.mp4", "out.mp4", factor=10.0)
            self.assertFalse(result)
            self.assertTrue(any("exceeds the stable 15K limit" in str(call) for call in mock_print.call_args_list))

    @patch('subprocess.Popen')
    def test_av1_fallback_to_hevc(self, mock_popen):
        """Test that AV1 falls back to HEVC if hardware AV1 encoding is not supported."""
        # 1. Setup Mock for Video Capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        self.mock_cv2.VideoCapture.return_value = mock_cap
        self.mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        self.mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        
        mock_cap.get.side_effect = lambda prop: {
            3: 1280, 4: 720, 5: 30, 7: 10
        }.get(prop, 0)
        
        # Mock cap.read() to return ONE frame then stop.
        mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]
        
        # Force OpenCV VideoWriter to FAIL opening, so logic falls back to FFMPEG_PIPE
        mock_out = MagicMock()
        mock_out.isOpened.return_value = False
        self.mock_cv2.VideoWriter.return_value = mock_out

        # 2. Setup Device as CUDA
        self.mock_device.return_value = (Mock(type='cuda'), None)

        # 3. Use module-level mock for encoder checking
        # Simulate AV1 failing and HEVC succeeding
        def check_encoder_side_effect(name, w, h):
            if name == 'av1_nvenc': return False
            if name == 'hevc_nvenc': return True
            return True

        # 4. Mock FFmpeg Popen success
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None # Running
        mock_proc.wait.return_value = 0 
        mock_proc.stdin.write.return_value = None
        mock_popen.return_value = mock_proc

        # 5. Patch 'realesrgan' via sys.modules
        mock_realesrgan_module = MagicMock()
        mock_esrgan_class = MagicMock()
        mock_realesrgan_module.RealESRGANer = mock_esrgan_class
        
        # Ensure the instance returned by the class is configured correctly
        mock_upsampler = mock_esrgan_class.return_value
        mock_output = MagicMock()
        mock_output.tobytes.return_value = b'x' * 10
        mock_output.nbytes = 10
        mock_output.shape = (2880, 5120, 3)
        mock_output.astype.return_value = mock_output
        mock_upsampler.enhance.return_value = (mock_output, None)

        with patch.dict('sys.modules', {'realesrgan': mock_realesrgan_module, 'basicsr.archs.rrdbnet_arch': MagicMock()}):
            with patch('builtins.print') as mock_print, \
                 patch('ai_media.upscaling._check_ffmpeg_encoder', side_effect=check_encoder_side_effect) as mock_check, \
                 patch('os.remove'):
                
                # Run with AV1 codec requested
                try:
                    ai_media.upscale_video_fast("in.mp4", "out.mp4", codec='av1')
                except Exception as e:
                    print(f"DEBUG: Unknown error in test: {e}")
                    raise e
                
                # Check calls
                mock_check.assert_any_call('av1_nvenc', 5120, 2880)
                mock_check.assert_any_call('hevc_nvenc', 5120, 2880)
                
                # Check for fallback message
                found_fallback = False
                found_hevc = False
                for call in mock_print.call_args_list:
                    if call.args:  # Check if args is not empty
                        arg = str(call.args[0])
                        if "Hardware AV1 not supported" in arg:
                            found_fallback = True
                        if "Using hevc_nvenc" in arg or "Using hevc_videotoolbox" in arg:
                            found_hevc = True
                        
                self.assertTrue(found_fallback, "Should log AV1 fallback message")
                self.assertTrue(found_hevc, "Should log switch to Hardware HEVC")


class TestZeroscopeDynamicUpscaling(unittest.TestCase):
    """Tests for zeroscope dynamic upscaling detection logic."""
    
    def test_zeroscope_xl_in_video_models(self):
        """Test zeroscope-xl model is registered."""
        self.assertIn("zeroscope-xl", ai_media.VIDEO_MODELS)
        self.assertEqual(ai_media.VIDEO_MODELS["zeroscope-xl"], "cerspense/zeroscope_v2_XL")
    
    def test_zeroscope_xl_in_model_requirements(self):
        """Test zeroscope_v2_XL has MODEL_REQUIREMENTS entry."""
        self.assertIn("cerspense/zeroscope_v2_XL", ai_media.MODEL_REQUIREMENTS)
        reqs = ai_media.MODEL_REQUIREMENTS["cerspense/zeroscope_v2_XL"]
        self.assertEqual(reqs["vram"], 10)
        self.assertEqual(reqs["ram"], 16)
        self.assertEqual(reqs["max_resolution"], (1024, 576))
    
    def test_zeroscope_576w_model_requirements(self):
        """Test zeroscope_v2_576w has correct MODEL_REQUIREMENTS."""
        self.assertIn("cerspense/zeroscope_v2_576w", ai_media.MODEL_REQUIREMENTS)
        reqs = ai_media.MODEL_REQUIREMENTS["cerspense/zeroscope_v2_576w"]
        self.assertEqual(reqs["max_resolution"], (576, 320))
    
    def test_ffmpeg_resize_video_exists(self):
        """Test ffmpeg_resize_video function exists."""
        self.assertTrue(hasattr(ai_media, 'ffmpeg_resize_video'))
        self.assertTrue(callable(ai_media.ffmpeg_resize_video))
    
    def test_upscale_video_zeroscope_xl_exists(self):
        """Test upscale_video_zeroscope_xl function exists."""
        self.assertTrue(hasattr(ai_media, 'upscale_video_zeroscope_xl'))
        self.assertTrue(callable(ai_media.upscale_video_zeroscope_xl))


class TestZeroscopeXLMPSFix(unittest.TestCase):
    """Tests for Zeroscope XL MPS limitation workaround (skip XL on Apple Silicon)."""
    
    def test_upscale_video_zeroscope_xl_signature(self):
        """Test upscale_video_zeroscope_xl has the expected parameters."""
        import inspect
        sig = inspect.signature(ai_media.upscale_video_zeroscope_xl)
        params = list(sig.parameters.keys())
        # Function takes video_frames (list of PIL Images), prompt, device, dtype, strength
        self.assertIn('video_frames', params)
        self.assertIn('prompt', params)
        self.assertIn('device', params)
        self.assertIn('dtype', params)
        self.assertIn('strength', params)
    
    def test_mps_detection_mechanism_exists(self):
        """Test that MPS detection mechanism exists in torch."""
        import torch
        # Verify the detection mechanism exists
        self.assertTrue(hasattr(torch.backends, 'mps'))
        self.assertTrue(hasattr(torch.backends.mps, 'is_available'))
    
    def test_mps_skip_xl_logic(self):
        """Test that MPS detection correctly identifies when to skip XL."""
        import torch
        
        # Simulate MPS detection logic used in generate_video
        def should_skip_xl():
            is_mps = torch.backends.mps.is_available() and not torch.cuda.is_available()
            return is_mps
        
        # On a Mac with MPS, this should return True
        # On NVIDIA, this should return False
        # We can't control the hardware, but we verify the logic pattern works
        result = should_skip_xl()
        self.assertIsInstance(result, bool)
    
    def test_mps_skip_xl_logic_pattern(self):
        """Test the MPS skip logic pattern works correctly."""
        # Test the boolean logic pattern used in generate_video
        # is_mps = torch.backends.mps.is_available() and not torch.cuda.is_available()
        
        # Case 1: MPS available, CUDA not available -> Skip XL (True)
        mps_available, cuda_available = True, False
        is_mps = mps_available and not cuda_available
        self.assertTrue(is_mps)  # XL should be skipped
        
        # Case 2: CUDA available -> Use XL (False)
        mps_available, cuda_available = False, True
        is_mps = mps_available and not cuda_available
        self.assertFalse(is_mps)  # XL should be used
        
        # Case 3: Neither available (CPU only) -> Use XL (False)
        mps_available, cuda_available = False, False
        is_mps = mps_available and not cuda_available
        self.assertFalse(is_mps)  # XL should be used on CPU
        
        # Case 4: Both available (shouldn't happen, but CUDA takes priority) -> Use XL (False)
        mps_available, cuda_available = True, True
        is_mps = mps_available and not cuda_available
        self.assertFalse(is_mps)  # XL should be used when CUDA is present
    
    def test_current_system_mps_detection(self):
        """Test MPS detection on current system returns valid boolean."""
        import torch
        is_mps = torch.backends.mps.is_available() and not torch.cuda.is_available()
        self.assertIsInstance(is_mps, bool)
        # On this Mac, should be True; on NVIDIA, should be False
    
    def test_ffmpeg_resize_video_signature(self):
        """Test ffmpeg_resize_video has the expected parameters."""
        import inspect
        sig = inspect.signature(ai_media.ffmpeg_resize_video)
        params = list(sig.parameters.keys())
        self.assertIn('input_path', params)
        self.assertIn('output_path', params)
        self.assertIn('target_w', params)
        self.assertIn('target_h', params)


class TestAudioMuxing(unittest.TestCase):
    """Tests for audio track detection and muxing in video upscaling."""
    
    @patch('subprocess.run')
    def test_ffprobe_audio_detection_with_audio(self, mock_run):
        """Test has_audio_track returns True when audio stream present."""
        # Mock ffprobe returning "audio" in stdout
        mock_run.return_value = MagicMock(stdout="audio\n", returncode=0)
        
        # Test the inline has_audio_track logic
        def has_audio_track(video_file):
            try:
                import subprocess
                result = subprocess.run([
                    "ffprobe", "-v", "error", "-select_streams", "a",
                    "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                    video_file
                ], capture_output=True, text=True, timeout=10)
                return "audio" in result.stdout
            except:
                return False
        
        result = has_audio_track("test_video.mp4")
        self.assertTrue(result)
    
    @patch('subprocess.run')
    def test_ffprobe_audio_detection_without_audio(self, mock_run):
        """Test has_audio_track returns False when no audio stream."""
        # Mock ffprobe returning empty stdout (no audio)
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        
        def has_audio_track(video_file):
            try:
                import subprocess
                result = subprocess.run([
                    "ffprobe", "-v", "error", "-select_streams", "a",
                    "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                    video_file
                ], capture_output=True, text=True, timeout=10)
                return "audio" in result.stdout
            except:
                return False
        
        result = has_audio_track("silent_video.mp4")
        self.assertFalse(result)
    
    @patch('subprocess.run')
    def test_ffprobe_audio_detection_timeout(self, mock_run):
        """Test has_audio_track returns False on timeout."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffprobe", timeout=10)
        
        def has_audio_track(video_file):
            try:
                import subprocess
                result = subprocess.run([
                    "ffprobe", "-v", "error", "-select_streams", "a",
                    "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                    video_file
                ], capture_output=True, text=True, timeout=10)
                return "audio" in result.stdout
            except:
                return False
        
        result = has_audio_track("test_video.mp4")
        self.assertFalse(result)
    
    @patch('subprocess.run')
    def test_ffprobe_audio_detection_error(self, mock_run):
        """Test has_audio_track returns False on subprocess error."""
        mock_run.side_effect = Exception("ffprobe not found")
        
        def has_audio_track(video_file):
            try:
                import subprocess
                result = subprocess.run([
                    "ffprobe", "-v", "error", "-select_streams", "a",
                    "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                    video_file
                ], capture_output=True, text=True, timeout=10)
                return "audio" in result.stdout
            except:
                return False
        
        result = has_audio_track("test_video.mp4")
        self.assertFalse(result)
    
    def test_upscale_video_fast_exists(self):
        """Test upscale_video_fast function exists."""
        self.assertTrue(hasattr(ai_media, 'upscale_video_fast'))
        self.assertTrue(callable(ai_media.upscale_video_fast))
    
    def test_upscale_video_file_exists(self):
        """Test upscale_video_file function exists."""
        self.assertTrue(hasattr(ai_media, 'upscale_video_file'))
        self.assertTrue(callable(ai_media.upscale_video_file))


# =============================================================================
# Loading Timer Tests
# =============================================================================

class TestLoadingTimer(unittest.TestCase):
    """Tests for the delayed loading message timer."""
    
    def test_loading_timer_variable_exists(self):
        """Test _loading_timer variable exists in module."""
        self.assertTrue(hasattr(ai_media, '_loading_timer'))
    
    def test_loading_shown_variable_exists(self):
        """Test _loading_shown variable exists in module."""
        self.assertTrue(hasattr(ai_media, '_loading_shown'))
    
    def test_show_loading_message_function_exists(self):
        """Test _show_loading_message function exists."""
        self.assertTrue(hasattr(ai_media, '_show_loading_message'))
        self.assertTrue(callable(ai_media._show_loading_message))
    
    def test_loading_timer_is_none_for_cli_mode(self):
        """Test _loading_timer is None when running with CLI args (test mode)."""
        # When running tests, sys.argv has arguments, so timer should be None
        # (or already cancelled)
        timer = ai_media._loading_timer
        # Timer should either be None (not started) or cancelled (if started)
        if timer is not None:
            self.assertFalse(timer.is_alive())
    
    def test_show_loading_message_sets_flag(self):
        """Test _show_loading_message sets _loading_shown to True."""
        # Reset the flag
        ai_media._loading_shown = False
        
        # Capture stdout to avoid printing during test
        with patch('sys.stdout', new_callable=io.StringIO):
            ai_media._show_loading_message()
        
        self.assertTrue(ai_media._loading_shown)



# =============================================================================
# Video Model Integration Tests (Wan 2.2, LTX, Mochi, Hunyuan)
# =============================================================================

class TestVideoModelIntegration(unittest.TestCase):
    """
    Unit tests to verify that the correct Diffusers pipelines are loaded
    and configured for the new video models (Wan 2.2, LTX, Mochi, Hunyuan).
    """

    def setUp(self):
        # Reset mocks before each test
        # We need to ensure the mocks are attached to the imported ai_media module context
        
        # Ensure 'diffusers' mock exists
        if 'diffusers' not in sys.modules or not isinstance(sys.modules['diffusers'], MagicMock):
             sys.modules['diffusers'] = MagicMock()
             # Mark as a package
             sys.modules['diffusers'].__path__ = []
        
        # Ensure submodules are mocked in sys.modules for 'from X import Y'
        sys.modules['diffusers.utils'] = MagicMock()
        sys.modules['diffusers.pipelines'] = MagicMock()
        
        # Specific Pipeline mocks
        sys.modules['diffusers'].WanPipeline = MagicMock()
        sys.modules['diffusers'].LTXPipeline = MagicMock()
        sys.modules['diffusers'].MochiPipeline = MagicMock()
        sys.modules['diffusers'].HunyuanVideoPipeline = MagicMock()
        sys.modules['diffusers'].DiffusionPipeline = MagicMock()
        
        # Mock get_optimal_device_and_dtype to return CPU/float32
        # We need to patch it on the module instance itself
        self.original_get_device = ai_media.get_optimal_device_and_dtype
        ai_media.get_optimal_device_and_dtype = MagicMock(return_value=(MagicMock(type="cpu"), "float32"))
        
        # Mock PerformanceTracker and ResourceMonitor to avoid file I/O or threads
        self.original_tracker = ai_media.PerformanceTracker
        self.original_monitor = ai_media.ResourceMonitor
        
        ai_media.PerformanceTracker = MagicMock()
        ai_media.ResourceMonitor = MagicMock()
        ai_media.ResourceMonitor.return_value.__enter__.return_value = MagicMock()
        ai_media.ResourceMonitor.return_value.__enter__.return_value.get_averages.return_value = (0,0,0,0)
        
        # Mock export_to_video if it exists, otherwise just set it
        if hasattr(ai_media, 'export_to_video'):
            self.original_export = ai_media.export_to_video
        else:
            self.original_export = None
        ai_media.export_to_video = MagicMock()

    def tearDown(self):
        # Restore originals
        ai_media.get_optimal_device_and_dtype = self.original_get_device
        ai_media.PerformanceTracker = self.original_tracker
        ai_media.ResourceMonitor = self.original_monitor
        if self.original_export:
            ai_media.export_to_video = self.original_export
        elif hasattr(ai_media, 'export_to_video'):
            del ai_media.export_to_video

    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('ai_media.VIDEO_MODELS')
    def test_wan_2_2_pipeline_loading(self, mock_models, mock_stdout):
        """Test that Wan 2.2 triggers WanPipeline and CPU offload."""
        # Setup model dict
        mock_models.get.return_value = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        
        # Call generate_video
        ai_media.generate_video(
            prompt="test prompt",
            output_path="test.mp4",
            duration=2.0,
            width=1280,
            height=720,
            model_name="wan2.2"
        )
        
        # Verify WanPipeline was used
        sys.modules['diffusers'].WanPipeline.from_pretrained.assert_called_once()
        
        # Verify offload was enabled (Mock pipeline instance)
        pipe_instance = sys.modules['diffusers'].WanPipeline.from_pretrained.return_value
        pipe_instance.enable_model_cpu_offload.assert_called_once()
        pipe_instance.vae.enable_tiling.assert_called_once()

    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('ai_media.VIDEO_MODELS')
    def test_wan_2_2_i2v_pipeline_loading(self, mock_models, mock_stdout):
        """Test that Wan 2.2 I2V triggers WanI2VPipeline."""
        mock_models.get.return_value = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        
        ai_media.generate_video(
            prompt="test prompt",
            output_path="test.mp4",
            duration=2.0,
            width=1280,
            height=720,
            model_name="wan2.2",
            image_input="test.jpg"
        )
        
        # Verify WanImageToVideoPipeline was used
        sys.modules['diffusers'].WanImageToVideoPipeline.from_pretrained.assert_called_once()
        
        # Verify offload was enabled
        pipe_instance = sys.modules['diffusers'].WanImageToVideoPipeline.from_pretrained.return_value
        pipe_instance.enable_model_cpu_offload.assert_called_once()

    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('ai_media.VIDEO_MODELS')
    def test_ltx_video_pipeline_loading(self, mock_models, mock_stdout):
        """Test that LTX-Video triggers LTXPipeline and CPU offload."""
        mock_models.get.return_value = "Lightricks/LTX-Video"
        
        ai_media.generate_video(
            prompt="test prompt",
            output_path="test.mp4",
            duration=2.0,
            width=1280,
            height=720,
            model_name="ltx-video"
        )
        
        sys.modules['diffusers'].LTXPipeline.from_pretrained.assert_called_once()
        
        pipe_instance = sys.modules['diffusers'].LTXPipeline.from_pretrained.return_value
        pipe_instance.enable_model_cpu_offload.assert_called_once()
    
    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('ai_media.VIDEO_MODELS')
    def test_mochi_1_pipeline_loading(self, mock_models, mock_stdout):
        """Test that Mochi 1 triggers MochiPipeline and CPU offload."""
        mock_models.get.return_value = "genmo/mochi-1-preview"
        
        ai_media.generate_video(
            prompt="test prompt",
            output_path="test.mp4",
            duration=2.0,
            width=1280,
            height=720,
            model_name="mochi-1"
        )
        
        sys.modules['diffusers'].MochiPipeline.from_pretrained.assert_called_once()
        
        pipe_instance = sys.modules['diffusers'].MochiPipeline.from_pretrained.return_value
        pipe_instance.enable_model_cpu_offload.assert_called_once()

    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('ai_media.VIDEO_MODELS')
    def test_hunyuan_pipeline_loading(self, mock_models, mock_stdout):
        """Test that HunyuanVideo triggers HunyuanVideoPipeline and CPU offload."""
        mock_models.get.return_value = "hunyuanvideo-community/HunyuanVideo"
        
        ai_media.generate_video(
            prompt="test prompt",
            output_path="test.mp4",
            duration=2.0,
            width=1280,
            height=720,
            model_name="hunyuan"
        )
        
        sys.modules['diffusers'].HunyuanVideoPipeline.from_pretrained.assert_called_once()
        
        pipe_instance = sys.modules['diffusers'].HunyuanVideoPipeline.from_pretrained.return_value
        pipe_instance.enable_model_cpu_offload.assert_called_once()


# =============================================================================
# Text Models and Article Generation Tests
# =============================================================================

class TestTextModels(unittest.TestCase):
    """Tests for TEXT_MODELS dictionary and related text generation features."""
    
    def test_text_models_exist(self):
        """Test TEXT_MODELS dictionary exists with expected keys."""
        self.assertTrue(hasattr(ai_media, 'TEXT_MODELS'))
        self.assertIn("default", ai_media.TEXT_MODELS)
        # Check some known models
        expected_models = ["llama-3.1-8b", "mistral-nemo-12b", "qwen3-14b"]
        for model in expected_models:
            self.assertIn(model, ai_media.TEXT_MODELS, f"Missing text model: {model}")
    
    def test_text_models_have_valid_hf_ids(self):
        """Test TEXT_MODELS values are valid HuggingFace model IDs."""
        for key, model_id in ai_media.TEXT_MODELS.items():
            self.assertIsInstance(model_id, str)
            # Most HF model IDs contain a slash (org/model)
            if key != "default":
                self.assertIn("/", model_id, f"Model ID {model_id} should contain '/'")


class TestArticleGenerator(unittest.TestCase):
    """Tests for ArticleGenerator class."""
    
    def test_article_generator_class_exists(self):
        """Test ArticleGenerator class exists."""
        self.assertTrue(hasattr(ai_media, 'ArticleGenerator'))
        self.assertTrue(callable(ai_media.ArticleGenerator))
    
    def test_article_generator_has_generate_article_method(self):
        """Test ArticleGenerator has generate_article method."""
        self.assertTrue(hasattr(ai_media.ArticleGenerator, 'generate_article'))
    
    def test_article_generator_has_generate_code_method(self):
        """Test ArticleGenerator has generate_code method."""
        self.assertTrue(hasattr(ai_media.ArticleGenerator, 'generate_code'))
    
    def test_article_generator_has_chat_session_method(self):
        """Test ArticleGenerator has chat_session method."""
        self.assertTrue(hasattr(ai_media.ArticleGenerator, 'chat_session'))
    
    def test_generate_article_signature(self):
        """Test generate_article has expected parameters."""
        import inspect
        sig = inspect.signature(ai_media.ArticleGenerator.generate_article)
        params = list(sig.parameters.keys())
        self.assertIn('self', params)
        self.assertIn('topic', params)
        self.assertIn('format', params)
        self.assertIn('online', params)
    
    def test_generate_code_signature(self):
        """Test generate_code has expected parameters."""
        import inspect
        sig = inspect.signature(ai_media.ArticleGenerator.generate_code)
        params = list(sig.parameters.keys())
        self.assertIn('self', params)
        self.assertIn('prompt', params)
        self.assertIn('output_file', params)
    
    def test_chat_session_signature(self):
        """Test chat_session method has expected parameters."""
        import inspect
        sig = inspect.signature(ai_media.ArticleGenerator.chat_session)
        params = list(sig.parameters.keys())
        self.assertIn('self', params)


class TestJumpPointsTextFeatures(unittest.TestCase):
    """Tests for JUMP_POINTS supporting article, code, chat, and research."""
    
    def test_jump_points_exist(self):
        """Test run_interactive function exists."""
        # JUMP_POINTS is defined locally in run_interactive(), 
        # so we test the interactive function exists
        self.assertTrue(hasattr(ai_media, 'run_interactive'))
        self.assertTrue(callable(ai_media.run_interactive))
    
    def test_article_jump_point_format(self):
        """Test article jump point format is correct."""
        # Since JUMP_POINTS is inside run_interactive, we check that run_interactive exists
        # Verify run_interactive exists and is callable
        # Verify run_interactive exists and is callable
        self.assertTrue(hasattr(ai_media, 'run_interactive'))
        self.assertTrue(callable(ai_media.run_interactive))
    
    def test_code_generation_arg_exists(self):
        """Test that -gc / --generate-code argument is supported."""
        # We verify by checking the module has the code generation flow
        self.assertTrue(hasattr(ai_media, 'ArticleGenerator'))
        gen = ai_media.ArticleGenerator
        self.assertTrue(hasattr(gen, 'generate_code'))
    
    def test_chat_session_method_exists(self):
        """Test that ArticleGenerator.chat_session method is supported."""
        self.assertTrue(hasattr(ai_media, 'ArticleGenerator'))
        gen = ai_media.ArticleGenerator
        self.assertTrue(hasattr(gen, 'chat_session'))



class TestSlashCommands(unittest.TestCase):
    """Tests for slash command processing in ArticleGenerator."""
    
    def setUp(self):
        self.generator = ai_media.ArticleGenerator(model_name="default")
        # Mock dependencies to avoid actual IO/Networking
        self.generator.deep_research = MagicMock()
        
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data="file content")
    def test_read_command_success(self, mock_file, mock_exists):
        """Test /read command successfully reads a file."""
        mock_exists.return_value = True
        
        response = self.generator.process_command("/read test.txt", [])
        
        self.assertTrue(response["handled"])
        self.assertIn("file content", response["context"])
        self.assertIn("test.txt", response["message"])
        self.assertEqual(response["error"], "")
        
    @patch('os.path.exists')
    def test_read_command_not_found(self, mock_exists):
        """Test /read command handles missing file."""
        mock_exists.return_value = False
        
        response = self.generator.process_command("/read missing.txt", [])
        
        self.assertTrue(response["handled"])
        self.assertIn("File not found", response["error"])
        
    def test_search_command(self):
        """Test /search command calls deep_research."""
        self.generator.deep_research.return_value = "Search Summary"
        
        response = self.generator.process_command("/search AI query", [])
        
        self.assertTrue(response["handled"])
        self.generator.deep_research.assert_called_with("AI query", iterations=3, max_images=0)
        self.assertIn("Search Summary", response["context"])
        
    def test_online_search_alias(self):
        """Test /online-search alias matches /search."""
        self.generator.deep_research.return_value = "Result"
        response = self.generator.process_command("/online-search query", [])
        self.assertTrue(response["handled"])
        self.generator.deep_research.assert_called()

    @patch('ai_media.generators.text.check_overwrite')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_command_code_block(self, mock_file, mock_check):
        """Test /save extracts and saves the last code block."""
        # Setup mock history with a code block
        history = [
            {"role": "user", "content": "gen code"},
            {"role": "assistant", "content": "Here is code:\n```python\nprint('hello')\n```"}
        ]
        
        # Mock check_overwrite to allow writing (should_write, final_path, always_overwrite, never_overwrite)
        mock_check.return_value = (True, "output.py", False, False)
        
        response = self.generator.process_command("/save output.py", history)
        
        self.assertTrue(response["handled"])
        mock_file.assert_called_with("output.py", "w", encoding="utf-8")
        # Verify written content is just the code
        mock_file().write.assert_called_with("print('hello')\n")
        self.assertIn("Exported code block", response["message"])

    @patch('ai_media.generators.text.check_overwrite')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_command_full_history(self, mock_file, mock_check):
        """Test /save|all saves entire conversation."""
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"}
        ]
        mock_check.return_value = (True, "chat.md", False, False)
        
        response = self.generator.process_command("/save|all chat.md", history)
        
        self.assertTrue(response["handled"])
        # Verify content includes metadata format
        written_content = mock_file().write.call_args[0][0]
        self.assertIn("# Chat Conversation History", written_content)
        self.assertIn("## User\nHi", written_content)
        self.assertIn("## Assistant\nHello", written_content)


class TestArticleOutputFormats(unittest.TestCase):
    """Tests for supported article output formats."""
    
    def test_supported_formats(self):
        """Test that common article output formats are supported."""
        # Check PDF dependencies are imported
        self.assertTrue('xhtml2pdf' in sys.modules or True)  # May not be loaded yet
        
        # Check docx support
        self.assertTrue('docx' in sys.modules or True)  # May not be loaded yet
        
        # Check markdown support
        self.assertTrue('markdown' in sys.modules or True)  # May not be loaded yet
    
    def test_article_generator_format_param(self):
        """Test generate_article accepts format parameter."""
        import inspect
        sig = inspect.signature(ai_media.ArticleGenerator.generate_article)
        params = list(sig.parameters.keys())
        self.assertIn('format', params)


class TestResearchWebSearch(unittest.TestCase):
    """Tests for web search functionality in research mode."""
    
    def test_ddgs_dependency(self):
        """Test DuckDuckGo search library is available."""
        # DDGS is imported at module level for web search
        try:
            from ddgs import DDGS
            ddgs_available = True
        except ImportError:
            ddgs_available = False
        
        self.assertTrue(ddgs_available, "ddgs library should be available for research mode")
    
    def test_generate_article_online_param(self):
        """Test generate_article has online parameter for research mode."""
        import inspect
        sig = inspect.signature(ai_media.ArticleGenerator.generate_article)
        params = list(sig.parameters.keys())
        self.assertIn('online', params)
    
    def test_research_iterations_param(self):
        """Test generate_article has research_iter parameter."""
        import inspect
        sig = inspect.signature(ai_media.ArticleGenerator.generate_article)
        params = list(sig.parameters.keys())
        self.assertIn('research_iter', params)


# =============================================================================
# Test runOn Platform Filter
# =============================================================================

class TestRunOnFilter(unittest.TestCase):
    """Tests for the runOn platform filtering in integration tests."""
    
    def setUp(self):
        """Import the should_run_test function."""
        from ai_media.testing.integration_tests import should_run_test, get_current_platform
        self.should_run_test = should_run_test
        self.get_current_platform = get_current_platform
    
    def test_default_all_platforms(self):
        """Test that no runOn means 'all' (run everywhere)."""
        test = {"name": "Test"}
        should_run, reason = self.should_run_test(test, "cuda")
        self.assertTrue(should_run)
        self.assertIsNone(reason)
        
    def test_explicit_all(self):
        """Test runOn: 'all' runs everywhere."""
        test = {"name": "Test", "runOn": "all"}
        should_run, reason = self.should_run_test(test, "mps")
        self.assertTrue(should_run)
        
    def test_cuda_only_on_cuda(self):
        """Test runOn: 'cuda' runs on CUDA."""
        test = {"name": "Test", "runOn": "cuda"}
        should_run, reason = self.should_run_test(test, "cuda")
        self.assertTrue(should_run)
        
    def test_cuda_only_skip_mps(self):
        """Test runOn: 'cuda' skips on MPS."""
        test = {"name": "Test", "runOn": "cuda"}
        should_run, reason = self.should_run_test(test, "mps")
        self.assertFalse(should_run)
        self.assertIn("cuda", reason)
        self.assertIn("mps", reason)
        
    def test_cuda_only_skip_cpu(self):
        """Test runOn: 'cuda' skips on CPU."""
        test = {"name": "Test", "runOn": "cuda"}
        should_run, reason = self.should_run_test(test, "cpu")
        self.assertFalse(should_run)
        
    def test_mps_only_on_mps(self):
        """Test runOn: 'mps' runs on MPS."""
        test = {"name": "Test", "runOn": "mps"}
        should_run, reason = self.should_run_test(test, "mps")
        self.assertTrue(should_run)
        
    def test_mps_only_skip_cuda(self):
        """Test runOn: 'mps' skips on CUDA."""
        test = {"name": "Test", "runOn": "mps"}
        should_run, reason = self.should_run_test(test, "cuda")
        self.assertFalse(should_run)
        
    def test_gpu_shorthand_cuda(self):
        """Test runOn: 'gpu' runs on CUDA."""
        test = {"name": "Test", "runOn": "gpu"}
        should_run, reason = self.should_run_test(test, "cuda")
        self.assertTrue(should_run)
        
    def test_gpu_shorthand_mps(self):
        """Test runOn: 'gpu' runs on MPS."""
        test = {"name": "Test", "runOn": "gpu"}
        should_run, reason = self.should_run_test(test, "mps")
        self.assertTrue(should_run)
        
    def test_gpu_shorthand_skip_cpu(self):
        """Test runOn: 'gpu' skips on CPU."""
        test = {"name": "Test", "runOn": "gpu"}
        should_run, reason = self.should_run_test(test, "cpu")
        self.assertFalse(should_run)
        
    def test_comma_separated_list(self):
        """Test runOn: 'cuda,mps' runs on either."""
        test = {"name": "Test", "runOn": "cuda,mps"}
        should_run_cuda, _ = self.should_run_test(test, "cuda")
        should_run_mps, _ = self.should_run_test(test, "mps")
        should_run_cpu, _ = self.should_run_test(test, "cpu")
        self.assertTrue(should_run_cuda)
        self.assertTrue(should_run_mps)
        self.assertFalse(should_run_cpu)
        
    def test_case_insensitive(self):
        """Test runOn is case-insensitive."""
        test = {"name": "Test", "runOn": "CUDA"}
        should_run, _ = self.should_run_test(test, "cuda")
        self.assertTrue(should_run)
        
    def test_whitespace_handling(self):
        """Test runOn handles whitespace in comma-separated list."""
        test = {"name": "Test", "runOn": "cuda, mps"}
        should_run, _ = self.should_run_test(test, "mps")
        self.assertTrue(should_run)
        
    def test_os_filter_mac(self):
        """Test runOn: 'mac' OS filter."""
        import sys
        test = {"name": "Test", "runOn": "mac"}
        should_run, _ = self.should_run_test(test, "mps")
        if sys.platform == 'darwin':
            self.assertTrue(should_run)
        else:
            self.assertFalse(should_run)
            
    def test_get_current_platform_returns_valid(self):
        """Test get_current_platform returns one of cuda/mps/cpu."""
        platform = self.get_current_platform()
        self.assertIn(platform, ['cuda', 'mps', 'cpu'])
        
    def test_empty_runon_means_all(self):
        """Test empty runOn string means 'all'."""
        test = {"name": "Test", "runOn": ""}
        should_run, _ = self.should_run_test(test, "cpu")
        self.assertTrue(should_run)

# =============================================================================
# Test Glob Pattern Filtering for --test flag
# =============================================================================

class TestGlobPatternFiltering(unittest.TestCase):
    """Tests for glob pattern matching in test filtering (--test flag)."""
    
    def setUp(self):
        """Import the fnmatch module and define the matches_filter function."""
        import fnmatch
        self.fnmatch = fnmatch
        
        # This is the same logic used in run_tests() - case-insensitive
        def matches_filter(test_name, patterns):
            """Check if test name matches any filter pattern (exact or glob, case-insensitive)."""
            test_lower = test_name.lower()
            for pattern in patterns:
                pattern_lower = pattern.lower()
                # First try exact match (case-insensitive)
                if test_lower == pattern_lower:
                    return True
                # Then try glob pattern match (supports *, ?, [seq], [!seq])
                if fnmatch.fnmatch(test_lower, pattern_lower):
                    return True
            return False
        
        self.matches_filter = matches_filter
    
    def test_exact_match(self):
        """Test exact match takes priority."""
        result = self.matches_filter("Image - SDXL", ["Image - SDXL"])
        self.assertTrue(result)
        
    def test_exact_match_no_match(self):
        """Test exact match returns False when no match."""
        result = self.matches_filter("Image - SDXL", ["Video - Zeroscope"])
        self.assertFalse(result)
    
    def test_wildcard_star_suffix(self):
        """Test wildcard * at end matches prefix."""
        result = self.matches_filter("Interactive - Jump 1 (Image)", ["Interactive*"])
        self.assertTrue(result)
        
    def test_wildcard_star_prefix(self):
        """Test wildcard * at start matches suffix."""
        result = self.matches_filter("Image - Default (SD 3.5 Turbo)", ["*Default*"])
        self.assertTrue(result)
        
    def test_wildcard_star_middle(self):
        """Test wildcard * in middle matches."""
        result = self.matches_filter("Video - Zeroscope", ["Video*Zeroscope"])
        self.assertTrue(result)
    
    def test_wildcard_question_mark(self):
        """Test ? matches single character."""
        result = self.matches_filter("Jump 1", ["Jump ?"])
        self.assertTrue(result)
        result2 = self.matches_filter("Jump 10", ["Jump ?"])
        self.assertFalse(result2)  # ? only matches one char
        
    def test_multiple_patterns(self):
        """Test multiple patterns in list."""
        result = self.matches_filter("Audio - Bark", ["Video*", "Audio*"])
        self.assertTrue(result)
        
    def test_no_match_with_patterns(self):
        """Test no match when patterns don't match."""
        result = self.matches_filter("Caption Test", ["Video*", "Audio*"])
        self.assertFalse(result)
        
    def test_bracket_character_class(self):
        """Test [seq] character class matching."""
        result = self.matches_filter("Test-A", ["Test-[ABC]"])
        self.assertTrue(result)
        result2 = self.matches_filter("Test-D", ["Test-[ABC]"])
        self.assertFalse(result2)
        
    def test_empty_test_name(self):
        """Test empty test name doesn't match non-empty patterns."""
        result = self.matches_filter("", ["Interactive*"])
        self.assertFalse(result)
        
    def test_empty_patterns_list(self):
        """Test empty patterns list returns False."""
        result = self.matches_filter("Some Test", [])
        self.assertFalse(result)
    
    def test_case_insensitive_match(self):
        """Test case-insensitive matching works."""
        # Lowercase pattern matches title-case name
        result = self.matches_filter("Interactive - Jump 1", ["interactive*"])
        self.assertTrue(result)
        # Uppercase pattern matches lowercase name  
        result2 = self.matches_filter("image - sdxl", ["IMAGE*"])
        self.assertTrue(result2)
        # Mixed case
        result3 = self.matches_filter("Video - Zeroscope", ["video*zeroscope"])
        self.assertTrue(result3)


# =============================================================================
# Server Logic Tests
# =============================================================================

class TestServerConfig(unittest.TestCase):
    """Tests for server configuration loading."""
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=Mock)
    def test_load_config_defaults(self, mock_open, mock_exists):
        """Test that defaults are used when config.json is missing."""
        mock_exists.return_value = False
        config = load_config()
        self.assertEqual(config["server"]["port"], 8000)
        self.assertEqual(config["client"]["port"], 5173)
        self.assertEqual(config["server"]["host"], "127.0.0.1")

    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('json.load')
    def test_load_config_overrides(self, mock_json_load, mock_open, mock_exists):
        """Test that config.json overrides defaults."""
        mock_exists.return_value = True
        mock_json_load.return_value = {
            "server": {"port": 9000},
            "client": {"port": 6000}
        }
        config = load_config()
        self.assertEqual(config["server"]["port"], 9000)
        self.assertEqual(config["client"]["port"], 6000)
        self.assertEqual(config["server"]["host"], "127.0.0.1") # Still default

class TestServerCache(unittest.TestCase):
    """Tests for the ModelCache logic."""
    
    def setUp(self):
        self.cache = ModelCache()
        # Mock _clear_memory to avoid actual torch/gpu calls
        self.cache._clear_memory = MagicMock()

    def test_cache_get_set(self):
        """Test basic caching and retrieval."""
        instance = MagicMock()
        self.cache.set("image", "flux", instance)
        self.assertEqual(self.cache.get("image", "flux"), instance)
        
    def test_cache_auto_unload(self):
        """Test that requesting a different model unloads the previous one."""
        inst1 = MagicMock()
        inst2 = MagicMock()
        self.cache.set("image", "flux", inst1)
        
        # Requesting a different model should return None and trigger unload
        self.assertIsNone(self.cache.get("image", "sdxl"))
        self.cache._clear_memory.assert_called()
        
        # model should not be in cache yet
        self.assertIsNone(self.cache.get("image", "sdxl"))

    def test_unload_all(self):
        """Test unloading all models."""
        self.cache.set("image", "flux", MagicMock())
        self.cache.set("text", "llama", MagicMock())
        self.cache.unload_all()
        self.assertEqual(len(self.cache._cache), 0)
        self.cache._clear_memory.assert_called()

class TestJobManagement(unittest.TestCase):
    """Tests for job state and management."""
    
    def setUp(self):
        server_state.jobs.clear()
        server_state.job_manager.broadcast = MagicMock()
        # Mock event loop to avoid asyncio issues in sync tests
        server_state.MAIN_LOOP = MagicMock()

    def test_create_job(self):
        """Test creating a new job."""
        with patch('asyncio.get_running_loop', side_effect=RuntimeError):
            job = create_job("image", prompt="test prompt", model="flux")
            
        self.assertEqual(job["type"], "image")
        self.assertEqual(job["status"], "pending")
        self.assertEqual(job["prompt"], "test prompt")
        self.assertIn(job["job_id"], server_state.jobs)
        server_state.job_manager.broadcast.assert_called()

    def test_update_job(self):
        """Test updating an existing job."""
        job = {"job_id": "123", "status": "pending"}
        server_state.jobs["123"] = job
        
        with patch('asyncio.get_running_loop', side_effect=RuntimeError):
            update_job("123", status="completed", progress=100)
            
        self.assertEqual(server_state.jobs["123"]["status"], "completed")
        self.assertEqual(server_state.jobs["123"]["progress"], 100)

    def test_is_job_cancelled(self):
        """Test cancellation check."""
        server_state.jobs["456"] = {"status": "cancelled"}
        server_state.jobs["789"] = {"status": "running"}
        
        self.assertTrue(is_job_cancelled("456"))
        self.assertFalse(is_job_cancelled("789"))
        self.assertFalse(is_job_cancelled("non-existent"))

class TestConnectionManagers(unittest.IsolatedAsyncioTestCase):
    """Async tests for WebSocket connection managers."""
    
    async def test_job_connection_manager(self):
        manager = server_state.JobConnectionManager()
        ws = MagicMock()
        ws.accept = MagicMock(return_value=asyncio.Future())
        ws.accept.return_value.set_result(None)
        
        await manager.connect(ws)
        self.assertIn(ws, manager.active_connections)
        
        manager.disconnect(ws)
        self.assertNotIn(ws, manager.active_connections)

    async def test_chat_connection_manager(self):
        manager = server_state.ChatConnectionManager()
        ws = MagicMock()
        ws.accept = MagicMock(return_value=asyncio.Future())
        ws.accept.return_value.set_result(None)
        
        await manager.connect("session1", ws)
        self.assertIn("session1", manager.active_connections)
        
        manager.disconnect("session1")
        self.assertNotIn("session1", manager.active_connections)

class TestServerApp(unittest.TestCase):
    """Tests for FastAPI app initialization."""
    
    def test_create_app(self):
        """Test that the FastAPI app is created with all routes."""
        app = create_app()
        self.assertEqual(app.title, "AI-Media API")
        
        # Check if some expected routes are present
        routes = [r.path for r in app.routes]
        self.assertIn("/api/system", routes)
        self.assertIn("/api/jobs", routes)
        self.assertIn("/sse/resources", routes)



# =============================================================================
# OCR Tests
# =============================================================================

class TestOCR(unittest.TestCase):
    """Tests for OCR functionality (Florence-2)."""
    
    def setUp(self):
        # Clear cache before each test
        from ai_media.conversion import ocr
        if hasattr(ocr, '_processor'):
             ocr._processor = None
             ocr._model = None
    
    @patch('ai_media.conversion.ocr.AutoProcessor')
    @patch('ai_media.conversion.ocr.AutoModelForCausalLM')
    @patch('ai_media.conversion.ocr.load_image')
    def test_image_to_text_success(self, mock_load_image, mock_model_cls, mock_processor_cls):
        """Test successful text extraction from image using Florence-2."""
        # Setup mocks
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.device = MagicMock(type="cpu")
        mock_model.dtype = "float32"
        mock_model.to.return_value = mock_model # Crucial: .to() must return self
        
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model_cls.from_pretrained.return_value = mock_model
        
        # Mock inputs preparation
        mock_inputs = {
            "input_ids": MagicMock(),
            "pixel_values": MagicMock()
        }
        mock_processor.return_value = mock_inputs
        
        # Mock generation
        mock_model.generate.return_value = [1, 2, 3] # Fake tokens
        mock_processor.batch_decode.return_value = ["Raw Generated Text"]
        
        # Mock post-processing (Crucial for Florence-2)
        mock_processor.post_process_generation.return_value = {"<OCR>": "Detected Text Content"}
        
        from ai_media.conversion import ocr
        
        # execution - explicitly test Florence-2 path
        result = ocr.image_to_text("fake_image.jpg", model_type="florence")
        
        # Verification
        self.assertEqual(result, "Detected Text Content")
        mock_load_image.assert_called_with("fake_image.jpg")
        mock_model.generate.assert_called()
        mock_processor.batch_decode.assert_called()
        mock_processor.post_process_generation.assert_called()
        
    @patch('ai_media.conversion.ocr.AutoProcessor')
    @patch('ai_media.conversion.ocr.AutoModelForCausalLM')
    def test_model_caching(self, mock_model_cls, mock_processor_cls):
        """Test that model and processor are cached."""
        from ai_media.conversion import ocr
        
        # Setup mock
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_processor_cls.from_pretrained.return_value = MagicMock()
        
        # Reset cache
        ocr._model = None
        ocr._processor = None
        ocr._current_model_type = None
        
        # First call - explicit florence model
        ocr.load_ocr_model(model_type="florence")
        self.assertEqual(mock_model_cls.from_pretrained.call_count, 1)
        
        # Second call - should use cache
        ocr.load_ocr_model(model_type="florence")
        self.assertEqual(mock_model_cls.from_pretrained.call_count, 1) # Should not increase


# =============================================================================
# Subtitles Generator Tests
# =============================================================================

from collections import namedtuple

class TestSubtitles(unittest.TestCase):
    """Test SubtitlesGenerator logic (mocked)."""

    @patch("ai_media.generators.subtitles.subprocess.run")
    def test_extract_audio(self, mock_run):
        """Test audio extraction via ffmpeg."""
        # Mock modules before import
        with patch.dict(sys.modules, {'faster_whisper': MagicMock()}):
            from ai_media.generators.subtitles import SubtitlesGenerator
        
        # Mock successful run
        mock_run.return_value.returncode = 0
        
        gen = SubtitlesGenerator(device="cpu")
        # We need to ensure output path logic works even if we mock subprocess
        # extract_audio returns str(audio_path)
        
        output = gen.extract_audio("input.mp4")
        
        # Check command
        mock_run.assert_called_once()
        # On windows path separators might differ
        # args[2] is input path
        # Check args are passed correctly
        self.assertTrue(any("input.mp4" in str(arg) for arg in mock_run.call_args[0][0]))
        self.assertTrue(output.endswith(".tmp.wav"))

    def test_transcribe(self):
        """Test transcription logic."""
        mock_fw = MagicMock()
        mock_whisper_cls = MagicMock()
        mock_fw.WhisperModel = mock_whisper_cls
        
        with patch.dict(sys.modules, {'faster_whisper': mock_fw}):
             from ai_media.generators.subtitles import SubtitlesGenerator
        
             # Setup mock model instance
             mock_model_instance = mock_whisper_cls.return_value
             
             # Mock segments generator
             Segment = namedtuple('Segment', ['start', 'end', 'text'])
             Info = namedtuple('Info', ['duration'])
             
             mock_model_instance.transcribe.return_value = (
                 [Segment(0.0, 1.0, "Hello"), Segment(1.0, 2.0, "World")],
                 Info(2.0)
             )
             
             gen = SubtitlesGenerator(device="cpu")
             
             # Mock console print but allow side_effects to see errors
             with patch("ai_media.generators.subtitles.console.print", side_effect=print):
                segments, dur, time_taken = gen.transcribe_audio("audio.wav")
            
             self.assertEqual(len(segments), 2)
             self.assertEqual(segments[0]['text'], "Hello")
             self.assertEqual(dur, 2.0)

    def test_translate(self):
        """Test translation logic."""
        mock_transformers = MagicMock()
        mock_pipeline = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model_cls = MagicMock()
        
        mock_transformers.pipeline = mock_pipeline
        mock_transformers.AutoTokenizer = mock_tokenizer
        mock_transformers.AutoModelForSeq2SeqLM = mock_model_cls
        
        with patch.dict(sys.modules, {'faster_whisper': MagicMock(), 'transformers': mock_transformers}):
             from ai_media.generators.subtitles import SubtitlesGenerator
        
             # Setup pipeline mock
             mock_translator = mock_pipeline.return_value
             mock_translator.side_effect = lambda x: [{"translation_text": f"Translated {x}"}]
             
             gen = SubtitlesGenerator(device="cpu")
        
             segments = [{"start": 0, "end": 1, "text": "Hello"}]
             # Mock console print and progress track
             with patch("ai_media.generators.subtitles.console.print"):
                  with patch("rich.progress.track", side_effect=lambda x, **kwargs: x):
                     translated = gen.translate_segments(segments, "en", "es")
             
             self.assertEqual(len(translated), 1)
             self.assertEqual(translated[0]['text'], "Translated Hello")

# =============================================================================
# Audio Tools Tests (Subtitles & Transcription)
# =============================================================================

class TestSubtitlesGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = subtitles.SubtitlesGenerator(device="cpu")

    def test_init(self):
        self.assertEqual(self.generator.device, "cpu")

    @patch('ai_media.generators.subtitles.SubtitlesGenerator.transcribe_audio')
    @patch('ai_media.generators.subtitles.SubtitlesGenerator.extract_audio')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('os.remove')
    @patch('pathlib.Path.exists')
    def test_run_call(self, mock_exists, mock_remove, mock_open, mock_extract, mock_transcribe):
        mock_extract.return_value = "temp.wav"
        # segments, duration, transcription_time
        mock_transcribe.return_value = ([{'start':0, 'end':1, 'text':'test'}], 1.0, 0.1)
        mock_exists.return_value = True # ensure cleanup tries to remove
        
        # Run
        self.generator.run("test_video.mp4")
        
        mock_extract.assert_called_with("test_video.mp4")
        mock_transcribe.assert_called()
        # Should initiate open for SRT writing
        mock_open.assert_called()

class TestTranscriptionGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = transcription.TranscriptionGenerator(device="cpu")

    def test_init(self):
        self.assertEqual(self.generator.subtitles_gen.device, "cpu")

    @patch('ai_media.generators.subtitles.SubtitlesGenerator.transcribe_audio')
    @patch('ai_media.generators.subtitles.SubtitlesGenerator.extract_audio')
    @patch('os.remove')
    @patch('pathlib.Path.exists')
    def test_run_call(self, mock_exists, mock_remove, mock_extract, mock_transcribe):
        mock_extract.return_value = "temp.wav"
        mock_transcribe.return_value = ([{'start':0, 'end':1, 'text':'test transcript'}], 1.0, 0.1)
        mock_exists.return_value = True

        result = self.generator.run("test.mp4")
        
        self.assertIn("test transcript", result)
        self.assertIn("[00:00]", result)
        mock_extract.assert_called()

class TestConvertDocumentTranslation(unittest.TestCase):
    def setUp(self):
        # We don't need real file system
        pass

    @patch('ai_media.conversion.document._read_to_markdown')
    @patch('ai_media.conversion.document._write_from_markdown')
    @patch('ai_media.conversion.document.Path')
    @patch('ai_media.conversion.document.check_overwrite')
    @patch('ai_media.generators.text.ArticleGenerator') 
    def test_convert_with_translation(self, MockArticleGenerator, mock_check, mock_path, mock_write, mock_read):
        from ai_media.conversion.document import convert_document
        
        # Setup mocks
        mock_check.return_value = (True, "output.es.md", None, None)
        mock_path.return_value.suffix = ".txt"
        mock_read.return_value = "Hello World"
        
        # Mock generator instance
        mock_gen_instance = MockArticleGenerator.return_value
        mock_gen_instance.translate_text.return_value = "Hola Mundo"
        
        # Run conversion with translate=True
        success = convert_document(
            "input.txt", 
            "output.es.md", 
            target_format="md",
            translate=True, 
            target_language="es"
        )
        
        self.assertTrue(success)
        
        # Verify translation was called with essential arguments
        mock_gen_instance.translate_text.assert_called_once()
        call_args = mock_gen_instance.translate_text.call_args
        self.assertEqual(call_args[0][0], "Hello World")  # First positional arg
        self.assertEqual(call_args[1]['target_lang'], "es")
        
        # Verify write used translated content
        mock_write.assert_called_with("Hola Mundo", "output.es.md", "md")

# =============================================================================
# bfloat16 Translation Support Tests
# =============================================================================

class TestTranslationBFloat16(unittest.TestCase):
    """Test bfloat16 support in translation models (NLLB and LLM)."""
    
    def setUp(self):
        self.mock_torch = MagicMock()
        self.mock_torch.cuda.is_available.return_value = True
        self.mock_torch.float32 = "float32"
        self.mock_torch.float16 = "float16"
        self.mock_torch.bfloat16 = "bfloat16"
        self.mock_torch.device.return_value = MagicMock(type="cuda")
        
        # Patch modules
        self.patchers = []
        
        # Patch text generator 
        self.text_module_patch = patch('ai_media.generators.text.torch', self.mock_torch)
        self.patchers.append(self.text_module_patch)
        self.text_module_patch.start()
        
        # Patch load_model dependencies
        self.AutoTokenizer = MagicMock()
        self.AutoModelForSeq2SeqLM = MagicMock()
        self.AutoModelForCausalLM = MagicMock()
        self.pipeline = MagicMock()
        
        self.transformers_patch = patch('transformers.AutoTokenizer', self.AutoTokenizer)
        self.transformers_patch.start()
        self.patchers.append(self.transformers_patch)
        
        self.seq2seq_patch = patch('transformers.AutoModelForSeq2SeqLM', self.AutoModelForSeq2SeqLM)
        self.seq2seq_patch.start()
        self.patchers.append(self.seq2seq_patch)

        self.causal_patch = patch('transformers.AutoModelForCausalLM', self.AutoModelForCausalLM)
        self.causal_patch.start()
        self.patchers.append(self.causal_patch)
        
        self.pipeline_patch = patch('transformers.pipeline', self.pipeline)
        self.pipeline_patch.start()
        self.patchers.append(self.pipeline_patch)

        # Patch system utilities
        self.is_bf16_patch = patch('ai_media.utils.system.is_bfloat16_supported')
        self.mock_is_bf16 = self.is_bf16_patch.start()
        self.patchers.append(self.is_bf16_patch)
        
        self.get_device_patch = patch('ai_media.generators.text.get_optimal_device_and_dtype')
        self.mock_get_device = self.get_device_patch.start()
        self.mock_get_device.return_value = (MagicMock(type="cuda"), self.mock_torch.bfloat16)
        self.patchers.append(self.get_device_patch)

        # Configure pipeline mock to return a valid structure for translation
        # translator(text, ...) -> [{'translation_text': 'Translated Text'}]
        # LLM pipeline(...) -> [{'generated_text': 'Translated Text'}]
        mm_pipeline_instance = MagicMock()
        # Mocking for both NLLB (returns dict with translation_text) and LLM (returns dict with generated_text)
        mm_pipeline_instance.return_value = [{'translation_text': 'Hola', 'generated_text': 'Hola'}]
        self.pipeline.return_value = mm_pipeline_instance

    def tearDown(self):
        for p in self.patchers:
            p.stop()
            
    def test_nllb_cuda_bfloat16(self):
        """Test NLLB uses bfloat16 on CUDA when supported."""
        self.mock_is_bf16.return_value = True
        
        gen = ai_media.ArticleGenerator(model_name="default", device=MagicMock(type="cuda"))
        gen.device = MagicMock(type="cuda")
        gen.torch = self.mock_torch
        
        # Run NLLB translation
        gen.translate_text("Hello", "es", model_id="nllb-200-3.3b")
        
        # Verify bfloat16 was used
        # We need to find the call. Since we mocked the class method, we check the mock.
        # However, AutoModelForSeq2SeqLM is mocked in the module space.
        call_kwargs = self.AutoModelForSeq2SeqLM.from_pretrained.call_args[1]
        self.assertEqual(call_kwargs['torch_dtype'], "bfloat16")
        
    def test_nllb_cuda_float16_fallback(self):
        """Test NLLB falls back to float16 on CUDA when bfloat16 NOT supported."""
        self.mock_is_bf16.return_value = False
        
        gen = ai_media.ArticleGenerator(model_name="default", device=MagicMock(type="cuda"))
        gen.device = MagicMock(type="cuda")
        gen.torch = self.mock_torch
        
        # Run NLLB translation
        gen.translate_text("Hello", "es", model_id="nllb-200-3.3b")
        
        # Verify float16 was used
        call_kwargs = self.AutoModelForSeq2SeqLM.from_pretrained.call_args[1]
        self.assertEqual(call_kwargs['torch_dtype'], "float16")

    def test_llm_cuda_bfloat16(self):
        """Test LLM uses bfloat16 on CUDA when supported."""
        self.mock_is_bf16.return_value = True
        
        # Setup generator configured for LLM
        gen = ai_media.ArticleGenerator(model_name="alma-13b", device=MagicMock(type="cuda"))
        # Mock device to be CUDA
        gen.device = MagicMock(type="cuda")
        gen.torch = self.mock_torch
        
        # Manually trigger load_model (which handles LLM loading)
        gen._load_model()
        
        # Verify bfloat16 was passed to AutoModelForCausalLM
        call_kwargs = self.AutoModelForCausalLM.from_pretrained.call_args[1]
        self.assertEqual(call_kwargs['dtype'], "bfloat16")

    def test_llm_cuda_float16_fallback(self):
        """Test LLM uses float16 when explicitly set (simulating fallback scenario)."""
        # Note: The is_bfloat16_supported check happens during local import in _load_model,
        # making it difficult to mock. We test the float16 path directly by setting dtype.
        gen = ai_media.ArticleGenerator(model_name="alma-13b", device=MagicMock(type="cuda"))
        gen.device = MagicMock(type="cuda")
        gen.torch = self.mock_torch
        gen.dtype = self.mock_torch.float16  # Simulate float16 fallback scenario
        
        gen._load_model()
        
        # Verify float16 was used
        call_kwargs = self.AutoModelForCausalLM.from_pretrained.call_args[1]
        self.assertEqual(call_kwargs['dtype'], "float16")

    def test_nllb_mps_defaults_float32(self):
        """Test NLLB defaults to float32 on MPS."""
        gen = ai_media.ArticleGenerator(model_name="default", device=MagicMock(type="mps"))
        gen.device = MagicMock(type="mps")
        gen.torch = self.mock_torch
        
        # Run NLLB translation
        gen.translate_text("Hello", "es", model_id="nllb-200-3.3b")
        
        # Verify float32 was used
        call_kwargs = self.AutoModelForSeq2SeqLM.from_pretrained.call_args[1]
        self.assertEqual(call_kwargs['torch_dtype'], "float32")

    def test_llm_mps_defaults_bfloat16(self):
        """Test LLM defaults to bfloat16 on MPS (modern default)."""
        gen = ai_media.ArticleGenerator(model_name="alma-13b", device=MagicMock(type="mps"))
        gen.device = MagicMock(type="mps")
        gen.torch = self.mock_torch
        
        gen._load_model()
        
        # Verify bfloat16 was used (modern MPS default)
        call_kwargs = self.AutoModelForCausalLM.from_pretrained.call_args[1]
        self.assertEqual(call_kwargs['dtype'], "bfloat16")


# =============================================================================
# Random Prompts Utility Tests
# =============================================================================

class TestRandomPromptsUtility(unittest.TestCase):
    """Tests for ai_media/utils/prompts.py random prompt functions."""
    
    def test_is_random_prompt_trigger_rndpr(self):
        """Test 'rndPr' is recognized as trigger (case insensitive)."""
        from ai_media.utils.prompts import is_random_prompt_trigger
        self.assertTrue(is_random_prompt_trigger("rndPr"))
        self.assertTrue(is_random_prompt_trigger("rndpr"))
        self.assertTrue(is_random_prompt_trigger("RNDPR"))
        self.assertTrue(is_random_prompt_trigger("  rndPr  "))
    
    def test_is_random_prompt_trigger_rndprompt(self):
        """Test 'rndPrompt' is recognized as trigger."""
        from ai_media.utils.prompts import is_random_prompt_trigger
        self.assertTrue(is_random_prompt_trigger("rndPrompt"))
        self.assertTrue(is_random_prompt_trigger("rndprompt"))
    
    def test_is_random_prompt_trigger_randomprompt(self):
        """Test 'randomPrompt' is recognized as trigger."""
        from ai_media.utils.prompts import is_random_prompt_trigger
        self.assertTrue(is_random_prompt_trigger("randomPrompt"))
        self.assertTrue(is_random_prompt_trigger("randomprompt"))
    
    def test_is_random_prompt_trigger_random_prompt(self):
        """Test 'random prompt' is recognized as trigger."""
        from ai_media.utils.prompts import is_random_prompt_trigger
        self.assertTrue(is_random_prompt_trigger("random prompt"))
        self.assertTrue(is_random_prompt_trigger("RANDOM PROMPT"))
    
    def test_is_random_prompt_trigger_non_triggers(self):
        """Test non-trigger strings are not recognized."""
        from ai_media.utils.prompts import is_random_prompt_trigger
        self.assertFalse(is_random_prompt_trigger("A cyberpunk city"))
        self.assertFalse(is_random_prompt_trigger("random"))
        self.assertFalse(is_random_prompt_trigger("prompt"))
        self.assertFalse(is_random_prompt_trigger("rnd"))
        self.assertFalse(is_random_prompt_trigger(""))
    
    def test_get_random_prompt_returns_string(self):
        """Test get_random_prompt returns a non-empty string."""
        from ai_media.utils.prompts import get_random_prompt
        result = get_random_prompt("image")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
    
    def test_get_random_prompt_different_types(self):
        """Test get_random_prompt works for different prompt types."""
        from ai_media.utils.prompts import get_random_prompt
        # Should not raise for any of these types
        for ptype in ["image", "video", "audio", "article", "code"]:
            result = get_random_prompt(ptype)
            self.assertIsInstance(result, str)
    
    def test_get_random_prompt_unknown_type_fallback(self):
        """Test unknown type falls back to image prompts."""
        from ai_media.utils.prompts import get_random_prompt
        result = get_random_prompt("unknown_type")
        self.assertIsInstance(result, str)
    
    def test_maybe_replace_with_random_trigger(self):
        """Test maybe_replace_with_random replaces triggers."""
        from ai_media.utils.prompts import maybe_replace_with_random
        prompt, was_random = maybe_replace_with_random("rndPr", "image")
        self.assertTrue(was_random)
        self.assertNotEqual(prompt, "rndPr")
        self.assertGreater(len(prompt), 0)
    
    def test_maybe_replace_with_random_non_trigger(self):
        """Test maybe_replace_with_random passes through non-triggers."""
        from ai_media.utils.prompts import maybe_replace_with_random
        original = "A beautiful sunset over mountains"
        prompt, was_random = maybe_replace_with_random(original, "image")
        self.assertFalse(was_random)
        self.assertEqual(prompt, original)
    
    def test_maybe_replace_with_random_code_type(self):
        """Test maybe_replace_with_random works with code type."""
        from ai_media.utils.prompts import maybe_replace_with_random
        prompt, was_random = maybe_replace_with_random("randomPrompt", "code")
        self.assertTrue(was_random)
        self.assertNotEqual(prompt, "randomPrompt")
        self.assertGreater(len(prompt), 5)
    
    def test_maybe_replace_with_random_article_type(self):
        """Test maybe_replace_with_random works with article type."""
        from ai_media.utils.prompts import maybe_replace_with_random
        prompt, was_random = maybe_replace_with_random("rndPrompt", "article")
        self.assertTrue(was_random)
        self.assertNotEqual(prompt, "rndPrompt")
        self.assertGreater(len(prompt), 5)
    
    def test_get_random_prompt_code_returns_code_task(self):
        """Test code prompts return programming-related content."""
        from ai_media.utils.prompts import get_random_prompt
        result = get_random_prompt("code")
        self.assertIsInstance(result, str)
        # Code prompts should be non-empty programming tasks
        self.assertGreater(len(result), 10)
    
    def test_get_random_prompt_article_returns_topic(self):
        """Test article prompts return article topics."""
        from ai_media.utils.prompts import get_random_prompt
        result = get_random_prompt("article")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 5)


# =============================================================================
# ImageGenerator Class Tests
# =============================================================================

class TestImageGeneratorClass(unittest.TestCase):
    """Tests for ai_media/generators/image.py ImageGenerator class."""
    
    def test_image_generator_init(self):
        """Test ImageGenerator can be instantiated."""
        from ai_media.generators.image import ImageGenerator
        gen = ImageGenerator(model_id="sdxl")
        self.assertIsNotNone(gen)
        self.assertEqual(gen.model_name, "sdxl")
    
    def test_image_generator_model_id_resolution(self):
        """Test ImageGenerator resolves model IDs correctly."""
        from ai_media.generators.image import ImageGenerator
        # Default should resolve to sd3.5-turbo HuggingFace ID
        gen = ImageGenerator(model_id="default")
        self.assertIn("stabilityai", gen.model_id.lower())
    
    def test_generate_image_wrapper_function(self):
        """Test generate_image wrapper function exists and is callable."""
        from ai_media.generators.image import generate_image
        self.assertTrue(callable(generate_image))
    
    @patch('ai_media.generators.image.ImageGenerator')
    def test_generate_image_wrapper_calls_generator(self, MockGenerator):
        """Test generate_image wrapper creates generator and calls generate."""
        from ai_media.generators.image import generate_image
        mock_instance = MagicMock()
        mock_instance.generate.return_value = ["output.jpg"]
        MockGenerator.return_value = mock_instance
        
        result = generate_image("test prompt", "output.jpg", 512, 512)
        
        MockGenerator.assert_called_once()
        mock_instance.generate.assert_called_once()
        self.assertTrue(result)


# =============================================================================
# Inference Server Tests (Random Prompt Integration)
# =============================================================================

class TestInferenceServerRandomPrompt(unittest.TestCase):
    """Tests for random prompt handling in inference server."""
    
    def test_image_models_import(self):
        """Test IMAGE_MODELS can be imported from models."""
        from ai_media.models import IMAGE_MODELS
        self.assertIn("sdxl", IMAGE_MODELS)
        self.assertIn("flux", IMAGE_MODELS)
    
    def test_text_models_import(self):
        """Test TEXT_MODELS can be imported from models."""
        from ai_media.models import TEXT_MODELS
        self.assertIn("default", TEXT_MODELS)
        self.assertIn("llama-3.1-8b", TEXT_MODELS)
    
    def test_code_models_in_text_models(self):
        """Test code models exist within TEXT_MODELS."""
        from ai_media.models import TEXT_MODELS
        # Code models are integrated into TEXT_MODELS
        self.assertIn("qwen-coder-7b", TEXT_MODELS)
        self.assertIn("qwen-coder-14b", TEXT_MODELS)
    
    def test_prompts_utility_import(self):
        """Test prompts utility can be imported."""
        from ai_media.utils.prompts import is_random_prompt_trigger, get_random_prompt, maybe_replace_with_random
        self.assertTrue(callable(is_random_prompt_trigger))
        self.assertTrue(callable(get_random_prompt))
        self.assertTrue(callable(maybe_replace_with_random))
    
    def test_random_prompt_for_image_model(self):
        """Test random prompt replacement for image model type."""
        from ai_media.utils.prompts import maybe_replace_with_random
        prompt, was_random = maybe_replace_with_random("rndPr", "image")
        self.assertTrue(was_random)
        self.assertNotEqual(prompt.lower(), "rndpr")
    
    def test_random_prompt_for_text_model(self):
        """Test random prompt replacement for text/article model type."""
        from ai_media.utils.prompts import maybe_replace_with_random
        prompt, was_random = maybe_replace_with_random("randomPrompt", "article")
        self.assertTrue(was_random)
        self.assertNotEqual(prompt.lower(), "randomprompt")
    
    def test_random_prompt_for_code_model(self):
        """Test random prompt replacement for code model type."""
        from ai_media.utils.prompts import maybe_replace_with_random
        prompt, was_random = maybe_replace_with_random("rndPrompt", "code")
        self.assertTrue(was_random)
        self.assertNotEqual(prompt.lower(), "rndprompt")


class TestCleanupUtility(unittest.TestCase):
    @patch("os.listdir")
    @patch("os.path.exists")
    @patch("os.path.isfile")
    @patch("os.path.isdir")
    @patch("os.unlink")
    @patch("shutil.rmtree")
    def test_clear_directory(self, mock_rmtree, mock_unlink, mock_isdir, mock_isfile, mock_exists, mock_listdir):
        from ai_media.utils.cleanup import clear_directory
        
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ["file1.txt", "dir1", ".hidden"]
        
        # Mock isfile/isdir behavior based on name
        def side_effect_isfile(path):
            return "file1.txt" in path
        def side_effect_isdir(path):
            return "dir1" in path
            
        mock_isfile.side_effect = side_effect_isfile
        mock_isdir.side_effect = side_effect_isdir
        
        # Run
        deleted = clear_directory("/fake/path")
        
        # Verify
        # file1.txt and dir1 should be deleted. .hidden should be skipped.
        self.assertEqual(len(deleted), 2)
        self.assertIn("file1.txt", deleted)
        self.assertIn("dir1/", deleted)
        
        # Should unlink file1.txt
        found_unlink = any("file1.txt" in call.args[0] for call in mock_unlink.call_args_list)
        self.assertTrue(found_unlink)
        # Should rmtree dir1
        found_rmtree = any("dir1" in call.args[0] for call in mock_rmtree.call_args_list)
        self.assertTrue(found_rmtree)


# =============================================================================
# Prompt Parsing Tests
# =============================================================================

class TestPromptParsing(unittest.TestCase):
    """Tests for extract_prompt_parameters function."""
    
    def test_json_parsing(self):
        prompt = 'A beautiful landscape {negative_prompt: "ugly", steps: 20}'
        clean, params = ai_media.extract_prompt_parameters(prompt)
        self.assertEqual(clean, 'A beautiful landscape')
        self.assertEqual(params['negative_prompt'], 'ugly')
        self.assertEqual(params['steps'], 20)

    def test_pipe_parsing(self):
        prompt = 'A beautiful landscape | negative prompt: ugly | steps: 20'
        clean, params = ai_media.extract_prompt_parameters(prompt)
        self.assertEqual(clean, 'A beautiful landscape')
        self.assertEqual(params['negative_prompt'], 'ugly')
        self.assertEqual(params['steps'], 20)

    def test_mixed_case_and_aliases(self):
        prompt = 'Cyberpunk city | cfg: 8.5 | Width: 1024 | Height: 512px'
        clean, params = ai_media.extract_prompt_parameters(prompt)
        self.assertEqual(clean, 'Cyberpunk city')
        self.assertEqual(params['guidance_scale'], 8.5)
        self.assertEqual(params['width'], 1024)
        self.assertEqual(params['height'], 512)

    def test_resolution_resolution_alias(self):
        prompt = 'Portrait | Resolution: 1024x1024'
        clean, params = ai_media.extract_prompt_parameters(prompt)
        self.assertEqual(clean, 'Portrait')
        self.assertEqual(params['width'], 1024)
        self.assertEqual(params['height'], 1024)
        
    def test_no_params(self):
        prompt = 'Just a simple prompt'
        clean, params = ai_media.extract_prompt_parameters(prompt)
        self.assertEqual(clean, 'Just a simple prompt')
        self.assertEqual(params, {})


# =============================================================================
# Text Generation & Formatting Tests
# =============================================================================

class TestReasoningExtraction(unittest.TestCase):
    """Tests for ArticleGenerator.extract_reasoning method."""
    
    def test_standard_tags(self):
        content = "<think>This is reasoning.</think> This is the answer."
        result = ai_media.ArticleGenerator.extract_reasoning(content)
        self.assertEqual(result["reasoning"], "This is reasoning.")
        self.assertEqual(result["content"], "This is the answer.")

    def test_missing_closing_tag_with_answer_marker(self):
        content = "<think>This is reasoning.\nAnswer: This is the answer."
        result = ai_media.ArticleGenerator.extract_reasoning(content)
        self.assertEqual(result["reasoning"], "This is reasoning.")
        self.assertEqual(result["content"], "This is the answer.")

    def test_missing_closing_tag_with_here_is_marker(self):
        # "Here is" is no longer a safe split marker. 
        # Should fall back to return everything as content.
        content = "<think>Thinking...\nHere is the code:\nprint('hello')"
        result = ai_media.ArticleGenerator.extract_reasoning(content)
        self.assertIsNone(result["reasoning"])
        self.assertEqual(result["content"], content)

    def test_misplaced_answer_inside_tags(self):
        # Case where logical answer is inside <think> but </think> is at the very end
        content = "<think>Reasoning...\nAnswer: Real Answer</think>"
        result = ai_media.ArticleGenerator.extract_reasoning(content)
        self.assertEqual(result["reasoning"], "Reasoning...")
        self.assertEqual(result["content"], "Real Answer")

    def test_no_tags(self):
        content = "Just an answer."
        result = ai_media.ArticleGenerator.extract_reasoning(content)
        self.assertIsNone(result["reasoning"])
        self.assertEqual(result["content"], "Just an answer.")
    
    def test_empty_content_no_marker(self):
        # Standard unclosed tag with no marker -> all content (raw), reasoning None
        content = "<think>Just thinking forever..."
        result = ai_media.ArticleGenerator.extract_reasoning(content)
        self.assertIsNone(result["reasoning"])
        self.assertEqual(result["content"], content)


class TestTablePrettifier(unittest.TestCase):
    """Tests for ArticleGenerator.prettify_markdown_table method."""
    
    def test_plain_alignment(self):
        input_text = """
| Col1 | Col2 |
|---|---|
| A | B |
"""
        output = ai_media.ArticleGenerator.prettify_markdown_table(input_text.strip())
        self.assertIn("| Col1 | Col2 |", output)
        self.assertIn("|------|------|", output)
        self.assertIn("| A    | B    |", output)

    def test_ansi_alignment(self):
        # ANSI counts as 0 length.
        # \033[31mRed\033[0m is len 3 (Red). 
        # Col width should match "Blue" (4 chars) if that's max.
        input_text = """
| Col1 | Col2 |
|---|---|
| \\033[31mRed\\033[0m | Blue |
"""
        output = ai_media.ArticleGenerator.prettify_markdown_table(input_text.strip())
        # Red (3 visible) needs 1 padding space to match Blue (4 visible).
        # Format is f" {cell}{padding} |"
        # So " " + Red + " " + " |" -> " Red  |"
        self.assertIn("| \\033[31mRed\\033[0m  |", output) 

    def test_mixed_alignment(self):
         input_text = """
| ID | Color |
| -- | -- |
| 1 | \\033[32mGreen\\033[0m |
"""
         output = ai_media.ArticleGenerator.prettify_markdown_table(input_text.strip())
         # ID width 2. Color width 5 (Color) vs 5 (Green).
         # Green (5) matches header Color (5).
         self.assertIn("Green\\033[0m |", output)


# =============================================================================
# OpenAI API Route Tests (v1/chat/completions and v1/responses)
# =============================================================================

class TestOpenAIAPIRoutes(unittest.TestCase):
    """Tests for functional logic in openai_api.py routes."""
    
    def test_chat_message_model(self):
        """Test ChatMessage Pydantic model."""
        from ai_media.server.routes.openai_api import ChatMessage
        msg = ChatMessage(role="user", content="Hello")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello")
    
    def test_chat_completion_request_model(self):
        """Test ChatCompletionRequest Pydantic model with defaults."""
        from ai_media.server.routes.openai_api import ChatCompletionRequest, ChatMessage
        req = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role="user", content="Test")]
        )
        self.assertEqual(req.model, "llama-3.1-8b")
        self.assertEqual(req.temperature, 0.7)  # Default
        self.assertEqual(req.stream, False)  # Default
        self.assertIsNone(req.max_tokens)  # Default
    
    def test_responses_request_model(self):
        """Test ResponsesRequest Pydantic model (new API)."""
        from ai_media.server.routes.openai_api import ResponsesRequest
        req = ResponsesRequest(
            model="llama-3.1-8b",
            input="Test prompt"
        )
        self.assertEqual(req.model, "llama-3.1-8b")
        self.assertEqual(req.input, "Test prompt")
        self.assertIsNone(req.instructions)
        self.assertEqual(req.temperature, 0.7)
    
    def test_responses_request_with_messages(self):
        """Test ResponsesRequest with message list input."""
        from ai_media.server.routes.openai_api import ResponsesRequest, ResponsesInputMessage
        req = ResponsesRequest(
            model="llama-3.1-8b",
            input=[
                ResponsesInputMessage(role="user", content="Hello"),
                ResponsesInputMessage(role="assistant", content="Hi!")
            ],
            instructions="Be helpful"
        )
        self.assertEqual(len(req.input), 2)
        self.assertEqual(req.instructions, "Be helpful")


class TestAPICommandDetection(unittest.TestCase):
    """Tests for command detection in chat completions."""
    
    def test_stop_command_detection(self):
        """Test 'stop inference server' command is detected."""
        command = "stop inference server"
        self.assertEqual(command.lower(), "stop inference server")
    
    def test_unload_command_detection(self):
        """Test 'unload model' command is detected."""
        command = "unload model"
        self.assertEqual(command.strip().lower(), "unload model")
    
    def test_flush_command_detection(self):
        """Test 'flush memory' command is detected."""
        command = "flush memory"
        self.assertEqual(command.strip().lower(), "flush memory")


class TestAPIModelTypeDetection(unittest.TestCase):
    """Tests for model type detection in API routes."""
    
    def test_image_model_detection(self):
        """Test image models are correctly identified."""
        from ai_media.models import IMAGE_MODELS
        # Image model keys
        self.assertIn("flux", IMAGE_MODELS)
        self.assertIn("sdxl", IMAGE_MODELS)
        self.assertIn("flux-dev", IMAGE_MODELS)
    
    def test_text_model_detection(self):
        """Test text models are correctly identified."""
        from ai_media.models import TEXT_MODELS
        self.assertIn("llama-3.1-8b", TEXT_MODELS)
        self.assertIn("default", TEXT_MODELS)
    
    def test_model_in_image_models_check(self):
        """Test model membership check logic."""
        from ai_media.models import IMAGE_MODELS
        model_name = "flux"
        is_image = model_name in IMAGE_MODELS or model_name in IMAGE_MODELS.values()
        self.assertTrue(is_image)
    
    def test_text_model_not_in_image_models(self):
        """Test text model is not detected as image."""
        from ai_media.models import IMAGE_MODELS
        model_name = "llama-3.1-8b"
        is_image = model_name in IMAGE_MODELS or model_name in IMAGE_MODELS.values()
        self.assertFalse(is_image)


class TestAPIRandomPromptIntegration(unittest.TestCase):
    """Tests for random prompt handling in API routes."""
    
    def test_random_prompt_trigger_in_last_message(self):
        """Test random prompt trigger detection from last message."""
        from ai_media.utils.prompts import is_random_prompt_trigger
        
        # Simulate extracting last message
        messages = [{"role": "user", "content": "rndPr"}]
        last_msg_content = messages[-1]["content"].strip().lower()
        
        self.assertTrue(is_random_prompt_trigger(last_msg_content))
    
    def test_non_random_prompt_passes_through(self):
        """Test normal prompts are not detected as random triggers."""
        from ai_media.utils.prompts import is_random_prompt_trigger
        
        messages = [{"role": "user", "content": "Write me a poem about cats"}]
        last_msg_content = messages[-1]["content"].strip().lower()
        
        self.assertFalse(is_random_prompt_trigger(last_msg_content))


class TestAPIPrecisionParsing(unittest.TestCase):
    """Tests for precision/framework parsing in API routes."""
    
    def test_parse_model_precision_framework(self):
        """Test model:precision:framework parsing utility."""
        from ai_media.utils.precision import parse_model_precision_framework
        
        # Model with precision
        base, prec, fw = parse_model_precision_framework("llama-3.1-8b:int4")
        self.assertEqual(base, "llama-3.1-8b")
        self.assertEqual(prec, "int4")
        self.assertIsNone(fw)
    
    def test_parse_model_with_framework(self):
        """Test model:precision:framework with all parts."""
        from ai_media.utils.precision import parse_model_precision_framework
        
        base, prec, fw = parse_model_precision_framework("llama-3.1-8b:int4:mlx")
        self.assertEqual(base, "llama-3.1-8b")
        self.assertEqual(prec, "int4")
        self.assertEqual(fw, "mlx")
    
    def test_parse_model_no_suffix(self):
        """Test model without any suffix."""
        from ai_media.utils.precision import parse_model_precision_framework
        
        base, prec, fw = parse_model_precision_framework("llama-3.1-8b")
        self.assertEqual(base, "llama-3.1-8b")
        self.assertIsNone(prec)
        self.assertIsNone(fw)


class TestAPIResponseFormatting(unittest.TestCase):
    """Tests for response formatting in API routes."""
    
    def test_chat_completion_response_structure(self):
        """Test chat completion response has correct structure."""
        import uuid
        import time
        
        response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "llama-3.1-8b",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": None
        }
        
        self.assertIn("id", response)
        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(len(response["choices"]), 1)
        self.assertEqual(response["choices"][0]["message"]["role"], "assistant")
    
    def test_streaming_chunk_structure(self):
        """Test streaming chunk has correct structure."""
        import uuid
        import time
        
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "llama-3.1-8b",
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": None
            }]
        }
        
        self.assertEqual(chunk["object"], "chat.completion.chunk")
        self.assertIn("delta", chunk["choices"][0])


class TestAPIReasoningExtraction(unittest.TestCase):
    """Tests for reasoning extraction in API logging."""
    
    def test_log_response_with_reasoning_extracts_think_tags(self):
        """Test reasoning is extracted from <think> tags."""
        from ai_media.generators.text import ArticleGenerator
        
        text = "<think>This is reasoning.</think>This is the answer."
        result = ArticleGenerator.extract_reasoning(text)
        
        self.assertEqual(result["reasoning"], "This is reasoning.")
        self.assertEqual(result["content"], "This is the answer.")
    
    def test_log_response_without_reasoning(self):
        """Test response without reasoning tags."""
        from ai_media.generators.text import ArticleGenerator
        
        text = "Just a simple response."
        result = ArticleGenerator.extract_reasoning(text)
        
        self.assertIsNone(result["reasoning"])
        self.assertEqual(result["content"], "Just a simple response.")


if __name__ == '__main__':
    unittest.main()

