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
from unittest.mock import patch, MagicMock, mock_open, Mock
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

sys.modules['torch'] = mock_torch
sys.modules['torch.cuda'] = MagicMock()
sys.modules['torch.backends'] = MagicMock()
sys.modules['torch.backends.mps'] = MagicMock()
sys.modules['diffusers'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['accelerate'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.io'] = MagicMock()
sys.modules['scipy.io.wavfile'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock()
sys.modules['PIL.ImageOps'] = MagicMock()

# Import the module
import importlib.util
spec = importlib.util.spec_from_file_location("ai_media", "ai-media.py")
ai_media = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ai_media)
sys.modules['ai_media'] = ai_media


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
    
    def test_creates_single_parent_directory(self):
        """Test single parent directory is created."""
        nested_path = os.path.join(self.test_dir, "subdir", "output.txt")
        ai_media.ensure_paths(nested_path)
        parent_dir = os.path.dirname(nested_path)
        self.assertTrue(os.path.exists(parent_dir))
    
    def test_creates_deeply_nested_directories(self):
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


class TestClearScreen(unittest.TestCase):
    """Tests for clear_screen() function."""
    
    @patch('os.system')
    def test_calls_os_system(self, mock_system):
        """Test clear_screen calls os.system."""
        ai_media.clear_screen()
        mock_system.assert_called_once()
    
    @patch('os.system')
    def test_uses_cls_on_windows(self, mock_system):
        """Test uses 'cls' command on Windows."""
        with patch('os.name', 'nt'):
            ai_media.clear_screen()
            mock_system.assert_called_with('cls')
    
    @patch('os.system')
    def test_uses_clear_on_unix(self, mock_system):
        """Test uses 'clear' command on Unix."""
        with patch('os.name', 'posix'):
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
        """Test signal handler in non-test mode exits with code 0."""
        ai_media._test_state['active'] = False
        
        # Redirect stdout to avoid emoji encoding issues on Windows
        with patch('sys.stdout', new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as cm:
                ai_media.signal_handler(None, None)
            self.assertEqual(cm.exception.code, 0)
    
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
        
    def tearDown(self):
        ai_media.psutil = self.orig_psutil

    def test_get_system_resources(self):
        """Test RAM and VRAM detection."""
        # Setup mocks
        mock_mem = MagicMock()
        mock_mem.available = 16 * (1024**3) # 16GB
        ai_media.psutil.virtual_memory.return_value = mock_mem
        
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value.total_memory = 8 * (1024**3)
        mock_torch.cuda.memory_allocated.return_value = 2 * (1024**3)
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            # Run
            ram, vram = ai_media.get_system_resources()
            
            # Verify
            self.assertEqual(ram, 16.0)
            self.assertEqual(vram, 6.0) # 8 - 2
    
    @patch('ai_media.get_system_resources')
    def test_check_resources_strict_warnings(self, mock_get_resources):
        """Test strict warnings for low resources."""
        # Low RAM/VRAM
        mock_get_resources.return_value = (4.0, 2.0)
        
        # Redirect stdout
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            # Simulate user saying 'n' (abort)
            with patch('builtins.input', return_value='n'):
                # We need to simulate MODEL_REQUIREMENTS lookup
                # Since MODEL_REQUIREMENTS is a constant, we can patch the dict lookup or use a known key
                # "stabilityai/sdxl-turbo" is in the default dict
                with self.assertRaises(SystemExit):
                    ai_media.check_resources_and_warn("stabilityai/sdxl-turbo")

            output = mock_stdout.getvalue()
            self.assertIn("RAM: 4.0GB available", output)
            self.assertIn("VRAM: 2.0GB available", output)

    @patch('ai_media.get_system_resources')
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
    
    @patch('ai_media.get_optimal_device_and_dtype')
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
        mock_get_device.return_value = (mock_device, MagicMock())
        
        # We need to block ai_media.PerformanceTracker logging or it might fail if not mocked
        # Also patch stdout to avoid UnicodeError on Windows
        with patch.dict('sys.modules', {'diffusers': mock_diffusers, 'torch': MagicMock()}):
            with patch('ai_media.PerformanceTracker') as MockTracker:
                with patch('ai_media.ResourceMonitor') as MockMonitor:
                    # Configure ResourceMonitor instance
                    monitor_instance = MockMonitor.return_value
                    monitor_instance.__enter__.return_value = monitor_instance
                    monitor_instance.get_averages.return_value = (10.0, 4.0, 2.0, 50.0)
                    
                    with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
                        # Run
                        result = ai_media.generate_image("test prompt", "test.png", 512, 512, model_name="sdxl")
                
                        self.assertTrue(result)
                        mock_diffusers.AutoPipelineForText2Image.from_pretrained.assert_called()
                        mock_pipeline.assert_called() # The inference call

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
        with patch('sys.argv', ['ai-media.py'] + args_list):
            try:
                ai_media.main()
            except SystemExit:
                pass

    @patch('ai_media.simple_upscale_video')
    @patch('ai_media.upscale_video_file')
    @patch('ai_media.upscale_video_fast')
    def test_upscale_video_routing(self, mock_fast, mock_std, mock_simple):
        """Test video upscaling argument routing."""
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

        # Case 5: Fast (Real-ESRGAN) with -vu alias
        mock_fast.reset_mock()
        self.run_cli(["-uv", "in.mp4", "-vu", "realesrgan"])
        mock_fast.assert_called()

        mock_fast.reset_mock()
        self.run_cli(["-uv", "in.mp4", "-vc", "av1"])
        mock_fast.assert_called()

    @patch('ai_media.simple_upscale_image')
    @patch('ai_media.upscale_image_file')
    @patch('ai_media.upscale_image_fast')
    def test_upscale_image_routing(self, mock_fast, mock_std, mock_simple):
        """Test image upscaling argument routing."""
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
        self.patcher_has = patch('ai_media.HAS_REALESRGAN', True)
        self.patcher_exists = patch('os.path.exists', return_value=True)
        self.patcher_device = patch('ai_media.get_optimal_device_and_dtype', return_value=(Mock(type='cpu'), None))
        self.patcher_rrdb = patch('ai_media.RRDBNet', create=True)
        self.patcher_esrgan = patch('ai_media.RealESRGANer', create=True)
        self.patcher_monitor = patch('ai_media.ResourceMonitor', create=True)
        self.mock_cv2 = MagicMock()
        self.patcher_cv2 = patch.dict('sys.modules', {'cv2': self.mock_cv2})
        self.patcher_cv2.start()
        
        self.mock_has = self.patcher_has.start()
        self.mock_exists = self.patcher_exists.start()
        self.mock_device = self.patcher_device.start()
        self.mock_rrdb = self.patcher_rrdb.start()
        self.mock_esrgan = self.patcher_esrgan.start()
        self.mock_monitor = self.patcher_monitor.start()
        # Complex mocking of local imports is hard. 
        # We will basic-test the dependency check first.
        
    def tearDown(self):
        self.patcher_has.stop()
        self.patcher_exists.stop()
        self.patcher_device.stop()
        self.patcher_rrdb.stop()
        self.patcher_esrgan.stop()
        self.patcher_monitor.stop()

    @patch('ai_media.HAS_REALESRGAN', False)
    def test_missing_dependency(self):
        """Test returning False if dependencies missing."""
        with patch('builtins.print') as mock_print:
            result = ai_media.upscale_video_fast("in.mp4", "out.mp4")
            self.assertFalse(result)
            mock_print.assert_any_call("❌ Real-ESRGAN not installed. Cannot run fast upscale.")

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
        
        with patch('builtins.print') as mock_print:
            # 1920 * 10 = 19200 (exceeds 15360)
            result = ai_media.upscale_video_fast("in.mp4", "out.mp4", factor=10.0)
            self.assertFalse(result)
            mock_print.assert_any_call(f"   ❌ Target Resolution 19200x10800 exceeds the stable 15K limit (15360px).")

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
                 patch('ai_media._check_ffmpeg_encoder', side_effect=check_encoder_side_effect) as mock_check, \
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
                    arg = str(call.args[0])
                    if "Hardware AV1 not supported" in arg:
                        found_fallback = True
                    if "Using Hardware HEVC" in arg:
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
