import subprocess
import time
import sys
import platform

def is_hardware_encoder(name):
    """
    Returns True if the encoder name looks like a hardware-accelerated one.
    """
    hw_keywords = ["videotoolbox", "nvenc", "qsv", "amf", "vaapi", "omx", "v4l2m2m"]
    return any(kw in name.lower() for kw in hw_keywords)

def is_hardware_decoder(name):
    """
    Returns True if the decoder name looks like a hardware-accelerated one.
    """
    hw_keywords = ["videotoolbox", "nvdec", "qsv", "amf", "vaapi", "omx", "v4l2m2m", "cuvid"]
    return any(kw in name.lower() for kw in hw_keywords)

def get_platform_info():
    """
    Detects if the system is Windows/Linux (likely CUDA) or Mac (likely MPS/Videotoolbox).
    Returns dict with platform details.
    """
    info = {
        "system": platform.system(),
        "accel_type": "cpu", # default
        "encoders": {},
        "decoders": {}
    }
    
    # Check what FFmpeg actually supports to avoid testing non-existent encoders
    available_encoders = ""
    available_decoders = ""
    try:
        available_encoders = subprocess.check_output(['ffmpeg', '-encoders'], stderr=subprocess.STDOUT, text=True)
        available_decoders = subprocess.check_output(['ffmpeg', '-decoders'], stderr=subprocess.STDOUT, text=True)
    except:
        pass

    def filter_supported_encoders(enc_list):
        return [e for e in enc_list if e in available_encoders]

    def filter_supported_decoders(dec_list):
        return [d for d in dec_list if d in available_decoders]

    # Simple check for macOS
    if info["system"] == "Darwin":
        info["accel_type"] = "mps"
        # macOS Hardware Encoders
        info["encoders"] = {
            "h264": filter_supported_encoders(["h264_videotoolbox", "libx264"]),
            "hevc": filter_supported_encoders(["hevc_videotoolbox", "libx265"]),
            "av1":  filter_supported_encoders(["av1_videotoolbox", "libsvtav1", "libaom-av1"]) 
        }
        info["decoders"] = {
            "h264": filter_supported_decoders(["h264_videotoolbox", "h264"]),
            "hevc": filter_supported_decoders(["hevc_videotoolbox", "hevc"]),
            "av1":  filter_supported_decoders(["av1_videotoolbox", "dav1d", "libdav1d", "av1"])
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
                "h264": filter_supported_encoders(["h264_nvenc", "libx264"]),
                "hevc": filter_supported_encoders(["hevc_nvenc", "libx265"]),
                "av1":  filter_supported_encoders(["av1_nvenc", "libsvtav1"]) 
            }
            info["decoders"] = {
                "h264": filter_supported_decoders(["h264_cuvid", "h264"]),
                "hevc": filter_supported_decoders(["hevc_cuvid", "hevc"]),
                "av1":  filter_supported_decoders(["av1_cuvid", "dav1d", "av1"])
            }
        else:
             # CPU only fallback or other acceleration (e.g. Intel/AMD)
             info["encoders"] = {
                "h264": filter_supported_encoders(["h264_qsv", "h264_amf", "libx264"]),
                "hevc": filter_supported_encoders(["hevc_qsv", "hevc_amf", "libx265"]),
                "av1":  filter_supported_encoders(["av1_qsv", "av1_amf", "libsvtav1"])
            }
             info["decoders"] = {
                "h264": filter_supported_decoders(["h264_qsv", "h264_amf", "h264"]),
                "hevc": filter_supported_decoders(["hevc_qsv", "hevc_amf", "hevc"]),
                "av1":  filter_supported_decoders(["av1_qsv", "av1_amf", "dav1d", "av1"])
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

def check_decoder_resolution(decoder, codec_type, width, height, timeout=300):
    """
    Tries to decode a 1-frame test video.
    Returns (Success, Message, Duration)
    """
    print(f"   Testing {decoder} @ {width}x{height}...", end="", flush=True)
    
    # We need to feed the decoder some valid bitstream. 
    # We'll generate a very short bitstream using a standard software encoder.
    src_encoder = "libx264"
    if codec_type == "hevc": src_encoder = "libx265"
    elif codec_type == "av1": src_encoder = "libsvtav1"

    # Single command with a pipe to avoid intermediate files
    # Generating 1 frame is enough to test if decoder can initialize and decode
    cmd = f'ffmpeg -y -f lavfi -i color=c=black:s={width}x{height}:r=30 -t 0.1 -c:v {src_encoder} -preset ultrafast -f nut - | ffmpeg -vcodec {decoder} -i - -f null -'
    
    try:
        start = time.time()
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=timeout)
        duration = time.time() - start
        
        if proc.returncode == 0:
            print(f" ✅ ({duration:.2f}s)")
            return True, "OK", duration
        else:
            print(f" ❌")
            err_lines = proc.stderr.split('\n')
            last_err = "Unknown Error"
            for line in err_lines[-10:]:
                if "Error" in line or "failed" in line or "incorrect" in line or "Horizontal" in line or "not supported" in line:
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
    
    results_enc = []
    results_dec = []
    
    print("\n===================================================")
    print("Encoding Tests (most important for ai-media features)")
    print("=====================================================")

    # Iterate over codecs H264, HEVC, AV1
    try:
        for codec in ["h264", "hevc", "av1"]:
            encoders = sys_info["encoders"].get(codec, [])
            if not encoders: continue
            
            print(f"\n🔹 Testing {codec.upper()} Encoders")
            for enc in encoders:
                label = "HW Encoder" if is_hardware_encoder(enc) else "SW Encoder"
                print(f"   [{label}: {enc}]")
                for res_name, w, h in RESOLUTIONS:
                    t_out = 300 # 5 mins
                    
                    success, msg, dur = check_encoder_resolution(enc, w, h, timeout=t_out)
                    
                    results_enc.append({
                        "codec": codec,
                        "encoder": enc,
                        "res": res_name,
                        "w": w, "h": h,
                        "pass": success,
                        "note": msg
                    })
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Showing partial summary...")
        print_summary(results_enc, RESOLUTIONS, "ENCODER")
        sys.exit(0)

    print("\n==============================================")
    print("Decoding Tests (just for information)")
    print("==============================================")

    try:
        for codec in ["h264", "hevc", "av1"]:
            decoders = sys_info["decoders"].get(codec, [])
            if not decoders: continue

            print(f"\n🔹 Testing {codec.upper()} Decoders")
            for dec in decoders:
                label = "HW Decoder" if is_hardware_decoder(dec) else "SW Decoder"
                print(f"   [{label}: {dec}]")
                for res_name, w, h in RESOLUTIONS:
                    t_out = 300 # 5 mins
                    
                    success, msg, dur = check_decoder_resolution(dec, codec, w, h, timeout=t_out)
                    
                    results_dec.append({
                        "codec": codec,
                        "decoder": dec,
                        "res": res_name,
                        "w": w, "h": h,
                        "pass": success,
                        "note": msg
                    })
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Showing partial summary...")

    print_summary(results_enc, RESOLUTIONS, "ENCODER")
    print_summary(results_dec, RESOLUTIONS, "DECODER")

def print_summary(results, resolutions, type_label):
    if not results:
        return

    print(f"\n\n===========================================")
    print(f"   SUMMARY RESULTS: {type_label}S")
    print("===========================================")
    
    headers = [r[0] for r in resolutions]
    header_str = " | ".join([f"{h:<5}" for h in headers])
    print(f"{type_label:<25} | {header_str}")
    print("-" * (25 + 3 + len(header_str)))
    
    key_name = "encoder" if type_label == "ENCODER" else "decoder"
    items_seen = []
    for entry in results:
        if entry[key_name] not in items_seen:
            items_seen.append(entry[key_name])
            
    for item in items_seen:
        is_hw = is_hardware_encoder(item) if type_label == "ENCODER" else is_hardware_decoder(item)
        label = "HW" if is_hw else "SW"
        display_name = f"[{label}] {item}"
        row = f"{display_name:<25} | "
        for res_name, _, _ in resolutions:
            match = next((x for x in results if x[key_name] == item and x["res"] == res_name), None)
            if match:
                symbol = "✅" if match["pass"] else "❌"
                row += f"{symbol:<5} | "
            else:
                row += f"{'-':<5} | "
        print(row)
    print("===========================================")

if __name__ == "__main__":
    main()
