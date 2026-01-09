"""
Parsing utilities for AI-Media.

Functions for parsing size, duration, sampling rate, and other user inputs.
"""

import re
from ..constants import RESOLUTIONS, DEFAULT_IMAGE_SIZE


def parse_size(value):
    """
    Parse size string or object into (width, height).
    Accepts:
      - Presets: "480p", "720p", "1080p", "1440p", "2k", "3k", "4k", "5k", ..., "10k"
      - WxH format: "1280x720"
      - Single number (square): "1536" → (1536, 1536)
      - Object format: "w: 1280, h: 720" (Braces {} are optional)
    """
    if not value:
        return RESOLUTIONS[DEFAULT_IMAGE_SIZE]
        
    normalized = value.strip().lower()
    
    # Check presets first
    if normalized in RESOLUTIONS:
        return RESOLUTIONS[normalized]
        
    # Check WxH format
    if 'x' in normalized:
        try:
            w, h = map(int, normalized.split('x'))
            return (w, h)
        except ValueError:
            pass
    
    # Check single number format (square image)
    if normalized.isdigit():
        size = int(normalized)
        return (size, size)
            
    # Check Object/JSON-like format
    if '{' in normalized and '}' in normalized:
        try:
            w = None
            h = None
            
            # Remove braces
            content = normalized.strip("{}")
            parts = content.split(',')
            
            for part in parts:
                if ':' not in part:
                    continue
                k, v = part.split(':', 1)
                k = k.strip()
                v = int(re.sub(r'[^0-9]', '', v))  # extract number
                
                if k in ['w', 'width']:
                    w = v
                elif k in ['h', 'height']:
                    h = v
            
            if w and h:
                return (w, h)
        except Exception as e:
            print(f"Warning: Failed to parse object size '{value}': {e}")
    
    # Fallback default if parsing fails
    print(f"Warning: Could not parse size '{value}'. Using default 720p.")
    return RESOLUTIONS["720p"]


def parse_duration(value):
    """
    Parse duration into seconds (float).
    Accepts:
      - Numeric (seconds)
      - Strings: "15s", "1m", "1h50m", "50s"
      - Objects: "{h:1, m:25, s:10}"
    """
    if not value:
        return 15.0
        
    # If standard number strings "15", "15.5"
    try:
        return float(value)
    except ValueError:
        pass
        
    normalized = str(value).strip().lower()
    
    total_seconds = 0
    
    # Check Object format
    if '{' in normalized:
        try:
            content = normalized.strip("{}")
            parts = content.split(',')
            for part in parts:
                if ':' not in part:
                    continue
                k, v = part.split(':', 1)
                k = k.strip()
                try:
                    val = float(re.sub(r'[^0-9\.]', '', v))
                except:
                    continue
                
                if k in ['h', 'hours', 'hour']:
                    total_seconds += val * 3600
                elif k in ['m', 'mins', 'min', 'minutes', 'minute']:
                    total_seconds += val * 60
                elif k in ['s', 'sec', 'secs', 'seconds', 'second']:
                    total_seconds += val
            return total_seconds
        except Exception:
            pass

    # Check String format "1h50m10s"
    # Regex to find pairs of number+unit
    pattern = r'(\d+(?:\.\d+)?)\s*([hms])'
    matches = re.findall(pattern, normalized)
    
    if matches:
        for val_str, unit in matches:
            val = float(val_str)
            if unit == 'h':
                total_seconds += val * 3600
            elif unit == 'm':
                total_seconds += val * 60
            elif unit == 's':
                total_seconds += val
        return total_seconds
        
    # Fallback logic for "M:S" or "H:M:S" or just "50s"
    if ':' in normalized:
        parts = normalized.split(':')
        parts = [float(p) for p in parts]
        if len(parts) == 3:  # H:M:S
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:  # M:S
            return parts[0] * 60 + parts[1]
            
    return 15.0


def parse_sampling_rate(value):
    """Parse sampling rate string to integer Hz."""
    if not value:
        return 32000
    
    normalized = str(value).strip().lower()
    
    # Handle "44.1khz" -> 44100
    if 'k' in normalized:
        try:
            num = float(re.sub(r'[^0-9\.]', '', normalized))
            return int(num * 1000)
        except:
            pass  # Fallthrough to default behavior
    
    try:
        return int(re.sub(r'[^0-9]', '', normalized))
    except:
        return 32000


def parse_bitrate(value):
    """Return bitrate string in standardized format or passed through if complex."""
    if not value:
        return None
    return value.strip()


def parse_upscale_factor(val):
    """Parse upscale factor string (e.g., '2x', '4', '1.5') -> float."""
    if val is None:
        return 2.0
    s = str(val).lower().strip()
    s = s.replace('x', '')
    try:
        f = float(s)
        return f if f > 0 else 2.0
    except ValueError:
        return 2.0


def format_time(seconds):
    """Convert seconds to human readable string (e.g. 2w 1d 1h 2m 3.5s)."""
    if not seconds:
        return "0s"
        
    current = float(seconds)
    intervals = (
        ('w', 604800),  # 60 * 60 * 24 * 7
        ('d', 86400),   # 60 * 60 * 24
        ('h', 3600),    # 60 * 60
        ('m', 60),
    )
    
    result = []
    for name, count in intervals:
        value = int(current // count)
        if value:
            current -= value * count
            result.append(f"{value}{name}")
            
    # Remaining seconds with precision
    if current > 0 or not result:
        # If integer-ish, show int, else float
        if current % 1 == 0:
            result.append(f"{int(current)}s")
        else:
            result.append(f"{current:.1f}s")
            
    return " ".join(result)


def extract_prompt_parameters(prompt_text):
    """
    Extracts generation parameters from the prompt string.
    Supports:
    1. JSON style: "prompt {key: val}" (at end of string)
    2. Pipe style: "prompt | key: val | key: val"
    """
    params = {}
    clean_prompt = prompt_text.strip()
    
    # 1. JSON Extraction (Look for last {)
    last_brace = clean_prompt.rfind('{')
    if last_brace != -1:
        potential_json = clean_prompt[last_brace:]
        try:
            # Try strict JSON first
            import json
            extracted = json.loads(potential_json)
            if isinstance(extracted, dict):
                params = extracted
                clean_prompt = clean_prompt[:last_brace].strip()
        except:
            # Fallback: simple text parsing inside braces
            # Remove braces
            inner = potential_json.strip("{}").strip()
            if inner:
                # Use regex to split by comma ONLY if not inside quotes
                import re
                # Pattern matches comma not followed by an odd number of quotes
                pairs = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', inner)
                
                temp_params = {}
                valid = True
                for p in pairs:
                    if ':' not in p:
                        valid = False; break
                    k, v = p.split(':', 1)
                    k = k.strip()
                    v = v.strip()
                    # Remove surrounding quotes from value if present
                    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                        v = v[1:-1]
                    temp_params[k] = v
                if valid:
                    params = temp_params
                    clean_prompt = clean_prompt[:last_brace].strip()

    # 2. Pipe Extraction (if no JSON found or prompt still has pipes)
    if not params and '|' in clean_prompt:
        parts = clean_prompt.split('|')
        clean_prompt = parts[0].strip()
        for part in parts[1:]:
            if ':' in part:
                k, v = part.split(':', 1)
                params[k.strip()] = v.strip()
    
    # Normalize Keys and Values
    normalized = {}
    
    # Map common aliases to internal arguments
    key_map = {
        'negative prompt': 'negative_prompt', 'negative_prompt': 'negative_prompt', 
        'negative-prompt': 'negative_prompt', 'negativeprompt': 'negative_prompt',
        'negative': 'negative_prompt', 'neg': 'negative_prompt', 'not': 'negative_prompt',
        
        'steps': 'steps', 'step': 'steps', 'inference steps': 'steps', 'num_inference_steps': 'steps',
        'cfg': 'guidance_scale', 'guidance': 'guidance_scale', 'guidance_scale': 'guidance_scale',
        'text guidance': 'guidance_scale', 'textguidance': 'guidance_scale',
        
        'width': 'width', 'w': 'width',
        'height': 'height', 'h': 'height',
        'resolution': 'resolution', 'size': 'resolution', 'res': 'resolution'
    }

    for k, v in params.items():
        k_lower = k.lower().replace('_', ' ').strip() # spacing normalization
        # Try exact match or spacing variant
        matched_key = None
        if k_lower in key_map:
            matched_key = key_map[k_lower]
        else:
            # Try to match known aliases
            for alias, target in key_map.items():
                if k_lower == alias:
                    matched_key = target
                    break
        
        if matched_key:
            # Value Conversion
            try:
                if matched_key == 'steps':
                    normalized['steps'] = int(v)
                elif matched_key == 'guidance_scale':
                    normalized['guidance_scale'] = float(v)
                elif matched_key in ['width', 'height']:
                    # Remove 'px'
                    v_clean = v.lower().replace('px', '').strip()
                    normalized[matched_key] = int(v_clean)
                elif matched_key == 'resolution':
                    # Use robust parser which handles 5k, 1080p, 1024x1024, etc.
                    w_val, h_val = parse_size(v)
                    normalized['width'] = w_val
                    normalized['height'] = h_val
                else:
                    # Strings (negative prompt)
                    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                        v = v[1:-1]
                    normalized[matched_key] = v
            except ValueError:
                pass

    return clean_prompt, normalized
