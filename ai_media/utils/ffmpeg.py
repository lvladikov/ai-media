"""
FFmpeg utilities for AI-Media.

Video encoding parameters, resizing, and format handling.
"""

import os
import subprocess


def get_video_encoding_params(output_path):
    """Get FFmpeg encoding parameters based on output file extension.
    
    Returns a list of FFmpeg arguments for video codec, pixel format, and audio codec.
    Supports: mp4, mkv, mov, webm, wmv, avi.
    Utilizes hardware acceleration (NVENC/VideoToolbox) if available.
    """
    import torch
    ext = os.path.splitext(output_path)[1].lower()
    
    # Platform detection
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()
    
    # 1. Video Codec Selection (Default to H.264 for widest compatibility)
    vcodec = "libx264"
    if ext in ['.mp4', '.m4v', '.mkv', '.mov']:
        if has_cuda:
            vcodec = "h264_nvenc"
        elif has_mps:
            vcodec = "h264_videotoolbox"
    elif ext == '.webm':
        vcodec = "libvpx-vp9"
    elif ext == '.wmv':
        vcodec = "wmv2"
    elif ext == '.avi':
        vcodec = "mpeg4"
        
    # 2. Audio Codec Selection
    acodec = "aac"
    if ext == '.webm':
        acodec = "libopus"
    elif ext == '.wmv':
        acodec = "wmav2"
    elif ext == '.avi':
        acodec = "mp3"
        
    # 3. Parameters
    params = ["-c:v", vcodec, "-pix_fmt", "yuv420p", "-c:a", acodec]
    
    # Add bitrate for less efficient formats
    if ext in ['.webm', '.wmv', '.avi']:
        params.extend(["-b:v", "2M"])
        
    return params


def ffmpeg_resize_video(input_path, output_path, target_w, target_h):
    """Resize video to exact target dimensions using FFmpeg Lanczos.
    
    Used as a final step when AI upscalers produce dimensions that don't match
    the target exactly (e.g., Real-ESRGAN's fixed 4x scale).
    
    Args:
        input_path: Path to input video
        output_path: Path for output video  
        target_w: Target width (will be made even for codec compatibility)
        target_h: Target height (will be made even for codec compatibility)
        
    Returns:
        True on success, False on failure
    """
    # Ensure dimensions are even (required by most codecs)
    target_w = target_w if target_w % 2 == 0 else target_w + 1
    target_h = target_h if target_h % 2 == 0 else target_h + 1
    
    print(f"   📐 FFmpeg resize: {target_w}x{target_h}")
    
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", f"scale={target_w}:{target_h}:flags=lanczos",
            "-c:a", "copy",  # Copy audio stream unchanged
            output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"   ⚠️  FFmpeg resize failed: {e}")
        return False


def get_video_info(video_path):
    """Get video information using ffprobe.
    
    Returns:
        dict with 'width', 'height', 'fps', 'duration', 'has_audio' or None on failure
    """
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json
        data = json.loads(result.stdout)
        
        info = {
            'width': None,
            'height': None,
            'fps': None,
            'duration': None,
            'has_audio': False
        }
        
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                info['width'] = stream.get('width')
                info['height'] = stream.get('height')
                # Parse framerate (can be "30/1" or "29.97")
                fps_str = stream.get('r_frame_rate', '30/1')
                if '/' in fps_str:
                    num, den = map(float, fps_str.split('/'))
                    info['fps'] = num / den if den else 30.0
                else:
                    info['fps'] = float(fps_str)
            elif stream.get('codec_type') == 'audio':
                info['has_audio'] = True
                
        # Duration from format
        if 'format' in data:
            info['duration'] = float(data['format'].get('duration', 0))
            
        return info
    except Exception:
        return None


def has_audio_track(video_path):
    """Check if video file has an audio track using ffprobe."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-select_streams', 'a', 
             '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, check=True
        )
        return 'audio' in result.stdout.lower() or len(result.stdout.strip()) > 0
    except Exception:
        return False


def _check_ffmpeg_encoder(encoder_name, w=256, h=256):
    """
    Check if FFmpeg can actually initialize the given encoder at target resolution.
    Used for probing hardware limits (e.g. NVENC max resolution).
    """
    try:
        # Run a tiny 1-frame test encoding to null at target resolution
        cmd = [
            'ffmpeg', '-y', '-f', 'lavfi', '-i', f'nullsrc=s={w}x{h}', 
            '-c:v', encoder_name, '-t', '0.1', '-f', 'null', '-'
        ]
        
        # Suppress output unless verbose debugging is needed
        subprocess.run(cmd, stderr=subprocess.STDOUT, stdout=subprocess.DEVNULL, timeout=5, check=True)
        return True
    except:
        return False
