"""
Media conversion module for AI-Media.

Supports: Image, video, and audio format conversion using PIL and FFmpeg.
"""

import os
import subprocess
from pathlib import Path

from ..utils.ffmpeg import get_video_encoding_params
from ..utils.interaction import check_overwrite


def _parse_target_path(input_path, target):
    """Parse target into output path based on format specification."""
    target = target.strip()
    
    if '/' in target or '\\' in target or len(target) > 6:
        # It's a full path
        return target
    elif target.startswith('.'):
        # It's an extension like ".png"
        name = Path(input_path).stem
        return f"{name}{target}"
    else:
        # It's just a format like "png" or "PNG"
        name = Path(input_path).stem
        return f"{name}.{target.lower()}"


def convert_image(input_path, target):
    """Convert image format using PIL.
    
    Args:
        input_path: Source image file
        target: Output path, extension (.png), or format (png)
    """
    from PIL import Image
    
    output_path = _parse_target_path(input_path, target)
    
    print(f"🔄 Converting Image: {input_path}")
    print(f"   Output: {output_path}")
    
    should_write, output_path, _, _ = check_overwrite(output_path, always_overwrite=os.environ.get("AI_MEDIA_FORCE") == "1")
    if not should_write:
        return False
    
    try:
        # Temporarily disable PIL's decompression bomb limit for large images
        original_max = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None
        
        try:
            img = Image.open(input_path)
            
            # Handle transparency for formats that don't support it
            output_ext = Path(output_path).suffix.lower()
            if output_ext in ['.jpg', '.jpeg'] and img.mode in ['RGBA', 'P']:
                print(f"   ℹ️  Converting RGBA → RGB (JPEG doesn't support transparency)")
                img = img.convert('RGB')
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path)
            print(f"✅ Converted image saved to {output_path}")
            return True
        finally:
            Image.MAX_IMAGE_PIXELS = original_max
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False


def convert_image_ffmpeg(input_path, target):
    """Convert image format using FFmpeg."""
    output_path = _parse_target_path(input_path, target)
    
    print(f"🔄 Converting Image (FFmpeg): {input_path}")
    print(f"   Output: {output_path}")
    
    should_write, output_path, _, _ = check_overwrite(output_path, always_overwrite=os.environ.get("AI_MEDIA_FORCE") == "1")
    if not should_write:
        return False
    
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-i", input_path, output_path], 
                      check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Converted image saved to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False


def convert_video(input_path, target):
    """Convert video format using FFmpeg."""
    output_path = _parse_target_path(input_path, target)
    
    print(f"🎬 Converting Video: {input_path}")
    print(f"   Output: {output_path}")
    
    should_write, output_path, _, _ = check_overwrite(output_path, always_overwrite=os.environ.get("AI_MEDIA_FORCE") == "1")
    if not should_write:
        return False
    
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        encoding_params = get_video_encoding_params(output_path)
        subprocess.run(["ffmpeg", "-y", "-i", input_path, *encoding_params, output_path], 
                      check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Converted video saved to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False


def convert_audio(input_path, target):
    """Convert audio format using FFmpeg."""
    output_path = _parse_target_path(input_path, target)
    
    print(f"🎵 Converting Audio: {input_path}")
    print(f"   Output: {output_path}")
    
    should_write, output_path, _, _ = check_overwrite(output_path, always_overwrite=os.environ.get("AI_MEDIA_FORCE") == "1")
    if not should_write:
        return False
    
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-i", input_path, output_path], 
                      check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Converted audio saved to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False
