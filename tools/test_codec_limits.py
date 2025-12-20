import subprocess
import time
import sys
import platform

def get_platform_info():
    """
    Detects if the system is Windows/Linux (likely CUDA) or Mac (likely MPS/Videotoolbox).
    Returns dict with platform details.
    """
    info = {
        "system": platform.system(),
        "accel_type": "cpu", # default
        "encoders": {}
    }
    
    # Check what FFmpeg actually supports to avoid testing non-existent encoders
    available_encoders = ""
    try:
        available_encoders = subprocess.check_output(['ffmpeg', '-encoders'], stderr=subprocess.STDOUT, text=True)
    except:
        pass

    def filter_supported(enc_list):
        return [e for e in enc_list if e in available_encoders]

    # Simple check for macOS
    if info["system"] == "Darwin":
        info["accel_type"] = "mps"
        # macOS Hardware Encoders
        info["encoders"] = {
            "h264": filter_supported(["h264_videotoolbox", "libx264"]),
            "hevc": filter_supported(["hevc_videotoolbox", "libx265"]),
            "av1":  filter_supported(["av1_videotoolbox", "libsvtav1", "libaom-av1"]) 
        }
    else:
        # Assume Windows/Linux with NVIDIA for now
        is_cuda = False
        try:
            subprocess.run(['nvidia-smi'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            is_cuda = True
        except:
            pass

        if is_cuda:
            info["accel_type"] = "cuda"
            info["encoders"] = {
                "h264": filter_supported(["h264_nvenc", "libx264"]),
                "hevc": filter_supported(["hevc_nvenc", "libx265"]),
                "av1":  filter_supported(["av1_nvenc", "libsvtav1"]) 
            }
        else:
             # CPU only fallback or other acceleration (e.g. Intel/AMD)
             info["encoders"] = {
                "h264": filter_supported(["h264_qsv", "h264_amf", "libx264"]),
                "hevc": filter_supported(["hevc_qsv", "hevc_amf", "libx265"]),
                "av1":  filter_supported(["av1_qsv", "av1_amf", "libsvtav1"])
            }
            
    return info

def check_encoder_resolution(encoder, width, height, timeout=60):
    """
    Tries to encode a 1-second null video at the given resolution.
    Returns (Success, Message, Duration)
    """
    print(f"   Testing {encoder} @ {width}x{height}...", end="", flush=True)
    
    cmd = [
        'ffmpeg', '-y', 
        '-f', 'lavfi', '-i', f'color=c=black:s={width}x{height}:r=30',
        '-c:v', encoder,
        '-t', '1.0',
        '-f', 'null', '-'
    ]
    
    # Optimization/Speed flags
    if "nvenc" in encoder:
        cmd.extend(['-preset', 'p1']) 
    elif "videotoolbox" in encoder:
        cmd.extend(['-realtime', 'true'])
    elif "libx264" in encoder or "libx265" in encoder:
        cmd.extend(['-preset', 'ultrafast'])
    elif "libsvtav1" in encoder:
        cmd.extend(['-preset', '12']) # max speed

    try:
        start = time.time()
        # Increased timeout for high-res software encoding
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=timeout)
        duration = time.time() - start
        
        if proc.returncode == 0:
            print(f" ✅ ({duration:.2f}s)")
            return True, "OK", duration
        else:
            print(f" ❌")
            # Extract basic error
            err_lines = proc.stderr.split('\n')
            last_err = "Unknown Error"
            for line in err_lines[-10:]:
                if "Error" in line or "failed" in line or "incorrect" in line or "Horizontal" in line:
                    last_err = line.strip()
                    break
            return False, f"Err: {last_err[:50]}...", duration
            
    except subprocess.TimeoutExpired:
        print(" ⏱️ Timeout")
        return False, "Timeout", timeout
    except Exception as e:
        print(f" 💥")
        return False, str(e), 0

def main():
    print("===========================================")
    print("   Universal Codec Resolution Limit Test")
    print("===========================================")
    
    sys_info = get_platform_info()
    print(f"System: {sys_info['system']}")
    print(f"Acceleration: {sys_info['accel_type'].upper()}")
    
    RESOLUTIONS = [
        ("4K", 3840, 2160),
        ("8K", 7680, 4320),
        ("8K+", 8192, 4320),
        ("10K", 10240, 5760),
        ("12K", 12288, 6480),
        ("15K", 15360, 8640), 
        ("16K", 16384, 8640),
        ("18K", 18432, 9720),
        ("20K", 20480, 10800)
    ]
    
    results = [] # List of dicts for summary
    
    # Iterate over codecs H264, HEVC, AV1
    for codec in ["h264", "hevc", "av1"]:
        print(f"\n🔹 Testing {codec.upper()} Codec family")
        for enc in sys_info["encoders"].get(codec, []):
            print(f"   [Encoder: {enc}]")
            for res_name, w, h in RESOLUTIONS:
                # Progressive Timeout Logic
                if w <= 7680:
                    t_out = 60
                elif w <= 12288:
                    t_out = 120
                else:
                    t_out = 300 # 5 mins for 15K+ on software
                
                # Fast fail check: if previous resolution failed with "Error" (not timeout), 
                # likelihood of next one passing is low for hardware. 
                # But software might just be slow. We'll run all for completeness unless user aborts.
                
                success, msg, dur = check_encoder_resolution(enc, w, h, timeout=t_out)
                
                results.append({
                    "codec": codec,
                    "encoder": enc,
                    "res": res_name,
                    "w": w, "h": h,
                    "pass": success,
                    "note": msg
                })

    print("\n\n===========================================")
    print("   SUMMARY RESULTS")
    print("===========================================")
    # Header - Dynamic based on resolutions
    headers = [r[0] for r in RESOLUTIONS]
    header_str = " | ".join([f"{h:<5}" for h in headers])
    print(f"{'Encoder':<20} | {header_str}")
    print("-" * (20 + 3 + len(header_str)))
    
    # pivot results
    encoders_seen = []
    # get manual order based on loop
    for entry in results:
        if entry["encoder"] not in encoders_seen:
            encoders_seen.append(entry["encoder"])
            
    for enc in encoders_seen:
        row = f"{enc:<20} | "
        for res_name, _, _ in RESOLUTIONS:
            # find match
            match = next((x for x in results if x["encoder"] == enc and x["res"] == res_name), None)
            if match:
                symbol = "✅" if match["pass"] else "❌"
                row += f"{symbol:<5} | "
            else:
                row += f"{'-':<5} | "
        print(row)
        
    print("===========================================")

if __name__ == "__main__":
    main()
