import subprocess
import time
import sys
import platform
import threading
import os

# Global skip flag
skip_requested = False
skip_all_requested = False
skip_lock = threading.Lock()
input_thread = None
input_thread_stop = False

def start_input_listener():
    """Start a background thread to listen for 'S' key to skip tests."""
    global input_thread, input_thread_stop, skip_requested
    input_thread_stop = False
    
    def listener():
        global skip_requested, skip_all_requested, input_thread_stop
        if platform.system() == "Windows":
            import msvcrt
            while not input_thread_stop:
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                    if key == 's':
                        with skip_lock:
                            skip_requested = True
                    elif key == 'r':
                        with skip_lock:
                            skip_all_requested = True
                time.sleep(0.05)
        else:
            # Unix - set terminal to raw mode once
            import termios
            import tty
            try:
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setcbreak(fd)
                    while not input_thread_stop:
                        # Use select with a short timeout
                        import select
                        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                        if rlist:
                            key = sys.stdin.read(1).lower()
                            if key == 's':
                                with skip_lock:
                                    skip_requested = True
                            elif key == 'r':
                                with skip_lock:
                                    skip_all_requested = True
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass
    
    input_thread = threading.Thread(target=listener, daemon=True)
    input_thread.start()

def stop_input_listener():
    """Stop the background input listener thread."""
    global input_thread_stop
    input_thread_stop = True

def reset_skip_flag():
    """Reset the skip flag for the next test."""
    global skip_requested
    with skip_lock:
        skip_requested = False

def is_skip_requested():
    """Check if skip was requested (single test or all remaining)."""
    with skip_lock:
        return skip_requested or skip_all_requested

def is_skip_all_requested():
    """Check if skip all remaining was requested."""
    with skip_lock:
        return skip_all_requested

def run_with_skip_support(cmd, timeout, shell=False):
    """
    Runs a subprocess with support for 'S' key to skip.
    Returns (returncode, stderr, was_skipped, duration)
    """
    reset_skip_flag()
    
    start = time.time()
    proc = subprocess.Popen(
        cmd, 
        shell=shell,
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.PIPE, 
        text=True
    )
    
    # Poll until process finishes or skip is requested
    while proc.poll() is None:
        if is_skip_requested():
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except:
                proc.kill()
            duration = time.time() - start
            return -1, "", True, duration
        
        # Check timeout
        if time.time() - start > timeout:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except:
                proc.kill()
            duration = time.time() - start
            return -2, "Timeout", False, duration
        
        time.sleep(0.1)  # Small delay to avoid busy-waiting
    
    duration = time.time() - start
    stderr = proc.stderr.read() if proc.stderr else ""
    return proc.returncode, stderr, False, duration


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
    Returns (Success, Message, Duration, Skipped)
    """
    res_str = f"{width}x{height}"
    print(f"   Testing {encoder:<20} @ {res_str:<12}", end="", flush=True)
    
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
        returncode, stderr, was_skipped, duration = run_with_skip_support(cmd, timeout, shell=False)
        
        if was_skipped:
            print(" ⏩ Skipped")
            return False, "Skipped", duration, True
        elif returncode == -2:  # Timeout
            print(" ⏱️ Timeout")
            return False, "Timeout", timeout, False
        elif returncode == 0:
            print(f" ✅ ({duration:.2f}s)")
            return True, "OK", duration, False
        else:
            print(f" ❌")
            # Extract basic error
            err_lines = stderr.split('\n')
            last_err = "Unknown Error"
            for line in err_lines[-10:]:
                if "Error" in line or "failed" in line or "incorrect" in line or "Horizontal" in line:
                    last_err = line.strip()
                    break
            return False, f"Err: {last_err[:50]}...", duration, False
            
    except Exception as e:
        print(f" 💥")
        return False, str(e), 0, False

def check_decoder_resolution(decoder, codec_type, width, height, timeout=300):
    """
    Tries to decode a 1-frame test video.
    Returns (Success, Message, Duration, Skipped)
    """
    res_str = f"{width}x{height}"
    print(f"   Testing {decoder:<20} @ {res_str:<12}", end="", flush=True)
    
    # We need to feed the decoder some valid bitstream. 
    # We'll generate a very short bitstream using a standard software encoder.
    src_encoder = "libx264"
    if codec_type == "hevc": 
        src_encoder = "libx265"
    elif codec_type == "av1": 
        src_encoder = "libsvtav1"

    # Windows: Use temp files (pipes fail for AV1)
    # Mac/Linux: Use faster pipe method
    use_temp_file = (platform.system() == "Windows")
    
    if use_temp_file:
        import tempfile
        
        container = "mkv" if codec_type == "av1" else "mp4"  # MKV works better for AV1 than webm
        with tempfile.NamedTemporaryFile(suffix=f'.{container}', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            start = time.time()
            
            # Step 1: Generate test bitstream to temp file
            # Build preset based on encoder type
            if src_encoder == "libsvtav1":
                preset_args = ['-preset', '12']  # SVT-AV1 uses numeric presets
            elif src_encoder in ["libx264", "libx265"]:
                preset_args = ['-preset', 'ultrafast']
            else:
                preset_args = []
            
            encode_cmd = [
                'ffmpeg', '-y', '-f', 'lavfi', '-i', f'color=c=black:s={width}x{height}:r=30',
                '-t', '0.1', '-c:v', src_encoder, *preset_args,
                temp_path
            ]
            proc1 = subprocess.run(encode_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=120)
            if proc1.returncode != 0:
                print(f" ❌ (encode failed)")
                return False, "Encode failed", 0, False
            
            # Step 2: Decode the temp file with the target decoder (with skip support)
            decode_cmd = [
                'ffmpeg', '-y', '-c:v', decoder, '-i', temp_path, '-f', 'null', '-'
            ]
            returncode, stderr, was_skipped, dur = run_with_skip_support(decode_cmd, timeout, shell=False)
            duration = time.time() - start
            
            if was_skipped:
                print(" ⏩ Skipped")
                return False, "Skipped", duration, True
            elif returncode == -2:  # Timeout
                print(" ⏱️ Timeout")
                return False, "Timeout", timeout, False
            elif returncode == 0:
                print(f" ✅ ({duration:.2f}s)")
                return True, "OK", duration, False
            else:
                print(f" ❌")
                return False, "Decode failed", duration, False
                
        except Exception as e:
            print(f" 💥")
            return False, str(e), 0, False
        finally:
            try:
                os.remove(temp_path)
            except:
                pass
    else:
        # Mac/Linux: Use pipe (faster)
        # Build preset based on encoder type
        if src_encoder == "libsvtav1":
            preset_str = "-preset 12"
        elif src_encoder in ["libx264", "libx265"]:
            preset_str = "-preset ultrafast"
        else:
            preset_str = ""
        
        cmd = f'ffmpeg -y -f lavfi -i color=c=black:s={width}x{height}:r=30 -t 0.1 -c:v {src_encoder} {preset_str} -f nut - | ffmpeg -vcodec {decoder} -i - -f null -'
        
        try:
            returncode, stderr, was_skipped, duration = run_with_skip_support(cmd, timeout, shell=True)
            
            if was_skipped:
                print(" ⏩ Skipped")
                return False, "Skipped", duration, True
            elif returncode == -2:  # Timeout
                print(" ⏱️ Timeout")
                return False, "Timeout", timeout, False
            elif returncode == 0:
                print(f" ✅ ({duration:.2f}s)")
                return True, "OK", duration, False
            else:
                print(f" ❌")
                return False, "Decode failed", duration, False
                
        except Exception as e:
            print(f" 💥")
            return False, str(e), 0, False

def main():
    print("===========================================")
    print("🎬 Universal Codec Resolution Limit Test")
    print("===========================================")
    
    sys_info = get_platform_info()
    print(f"System: {sys_info['system']}")
    print(f"Acceleration: {sys_info['accel_type'].upper()}")
    print("\n💡 Tip: Press 'S' to skip the current test")
    print("        Press 'R' to skip all remaining tests")
    
    # Start background thread to listen for 'S' key
    start_input_listener()
    
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
    
    print("\n=====================================================")
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
                    t_out = 600 # 10 mins
                    
                    success, msg, dur, skipped = check_encoder_resolution(enc, w, h, timeout=t_out)
                    
                    results_enc.append({
                        "codec": codec,
                        "encoder": enc,
                        "res": res_name,
                        "w": w, "h": h,
                        "pass": success,
                        "note": msg,
                        "skipped": skipped
                    })
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Showing partial summary...")
        stop_input_listener()
        print_summary(results_enc, RESOLUTIONS, "ENCODER")
        sys.exit(0)

    print("\n==============================================")
    print("Decoding Tests (just for your information)")
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
                    t_out = 600 # 10 mins
                    
                    success, msg, dur, skipped = check_decoder_resolution(dec, codec, w, h, timeout=t_out)
                    
                    results_dec.append({
                        "codec": codec,
                        "decoder": dec,
                        "res": res_name,
                        "w": w, "h": h,
                        "pass": success,
                        "note": msg,
                        "skipped": skipped
                    })
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Showing partial summary...")

    # Stop background input listener
    stop_input_listener()

    print_summary(results_enc, RESOLUTIONS, "ENCODER")
    print_summary(results_dec, RESOLUTIONS, "DECODER")

def print_summary(results, resolutions, type_label):
    if not results:
        return

    # Fixed column widths
    name_width = 25
    col_width = 6  # Width for each resolution column
    
    # Build header row first to calculate total width
    headers = [r[0] for r in resolutions]
    header_row = f"{type_label:<{name_width}} |"
    for h in headers:
        header_row += f" {h:^{col_width}}|"
    
    table_width = len(header_row)
    
    print(f"\n\n{'='*table_width}")
    print(f"📊 SUMMARY RESULTS: {type_label}S")
    print(f"{'='*table_width}")
    print(header_row)
    print("-" * table_width)
    
    key_name = "encoder" if type_label == "ENCODER" else "decoder"
    items_seen = []
    for entry in results:
        if entry[key_name] not in items_seen:
            items_seen.append(entry[key_name])
            
    for item in items_seen:
        is_hw = is_hardware_encoder(item) if type_label == "ENCODER" else is_hardware_decoder(item)
        label = "HW" if is_hw else "SW"
        display_name = f"[{label}] {item}"
        row = f"{display_name:<{name_width}} |"
        for res_name, _, _ in resolutions:
            match = next((x for x in results if x[key_name] == item and x["res"] == res_name), None)
            if match:
                if match.get("skipped"):
                    symbol = "⏩"  # Skipped
                elif match["pass"]:
                    symbol = "✅"
                else:
                    symbol = "❌"
                row += f"  {symbol}   |"
            else:
                row += f"  ➖   |"  # emoji dash for not tested
        print(row)
    print(f"{'='*table_width}")

if __name__ == "__main__":
    main()
