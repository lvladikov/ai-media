"""
Audio generation module for AI-Media.

Supports: MusicGen, AudioLDM2, Stable Audio, and Bark (TTS/SFX).
"""

import os
import re
import time

from ..models import AUDIO_MODELS, get_model_id
from ..utils.system import get_optimal_device_and_dtype
from ..utils.parsers import format_time
from ..utils.performance import PerformanceTracker, ResourceMonitor, write_report_json
from .description import generate_caption


def generate_long_bark(prompt, processor, model, device, voice_preset, sample_rate=24000):
    """
    Generate long-form audio with Bark by splitting text into sentences.
    Avoids 'history' chaining to prevent hallucinations/degradation.
    Concatenates independent chunks with the same voice preset.
    """
    import numpy as np
    
    # Smart split by sentence ending punctuation
    sentences = re.split(r'([.?!]+|\n+)', prompt)
    
    # Recombine split text (sentence + punctuation)
    chunks = []
    current_chunk = ""
    
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        
        # If it's punctuation, attach to previous
        if re.match(r'^[.?!]+$', s) or re.match(r'^\n+$', s):
            if chunks:
                chunks[-1] += s
            else:
                current_chunk += s
        else:
            # If current chunk is getting too long (Bark limit ~14s is roughly ~20-30 words)
            # Heuristic: ~200 chars or ~30 words is safer upper bound
            if len(current_chunk) + len(s) > 200:
                chunks.append(current_chunk)
                current_chunk = s
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = s
                else:
                    current_chunk = s
    
    if current_chunk:
        chunks.append(current_chunk)
        
    print(f"   ✂️  Splitting long text into {len(chunks)} chunks for stable generation...")
    full_audio = []
    
    for i, text_chunk in enumerate(chunks):
        if not text_chunk.strip():
            continue
        print(f"   ▶️  Generating chunk {i+1}/{len(chunks)}: '{text_chunk[:30]}...'")
        
        # Independent generation (Best stability)
        inputs = processor(text_chunk, voice_preset=voice_preset).to(device)
        audio_array = model.generate(**inputs, do_sample=True)
        audio_array = audio_array.cpu().numpy().squeeze()
        full_audio.append(audio_array)
        
        # Add a short silence between sentences for natural pacing (0.25s)
        silence_len = int(sample_rate * 0.25)
        full_audio.append(np.zeros(silence_len))

    # Concatenate all
    if not full_audio:
        return np.array([])
    return np.concatenate(full_audio)


def generate_audio(prompt, output_path, duration, sampling_rate, model_name="default", 
                   image_input=None, caption_model="florence", voice_preset="v2/en_speaker_6",
                   report_json=None):
    """Generate audio using MusicGen, AudioLDM2, Stable Audio, or Bark.
    
    Args:
        prompt: Text description of desired audio
        output_path: Path to save audio file
        duration: Target duration in seconds
        sampling_rate: Output sampling rate (Hz)
        model_name: Model short code or HF ID
        image_input: Optional image path for image-to-audio
        caption_model: Caption model for image input ('florence' or 'blip')
        voice_preset: Voice preset for Bark TTS
        report_json: Path to write performance stats JSON
        
    Returns:
        True on success, False on failure
    """
    model_id = get_model_id(model_name, AUDIO_MODELS)
    
    import sys
    sys.setrecursionlimit(50000)  # Fix for Stable Audio / torchsde recursion on MPS
    
    device, dtype = get_optimal_device_and_dtype(quiet=True, prefer_bfloat16=True)
    
    # --- Image-to-Audio Logic (Captioning) ---
    if image_input:
        caption = generate_caption(image_input, device, model_type=caption_model)
        if caption:
            if not prompt:
                print(f"   ℹ️  No prompt provided. Using generated caption as prompt.")
                full_prompt = caption
            else:
                full_prompt = f"{prompt}, inspired by {caption}"
                
            print(f"   Full Prompt: '{full_prompt}'")
            prompt = full_prompt
        else:
            print(f"⚠️  Image analysis failed. Proceeding with text prompt only.")

    
    # Pre-calculate Estimate
    dtype_name = str(dtype).replace("torch.", "")
    tracker = PerformanceTracker()
    est_values = tracker.estimate_linear("audio", model_id, device, duration, dtype=dtype_name)
    
    # Display Info Header
    print(f"Platform: {device.type.upper()} | Dtype: {dtype_name}")
    tracker.print_estimate(*est_values)

    print(f"🎵 Generating Audio")
    print(f"   Model:    {model_id}")
    print(f"   Prompt:   '{prompt}'")
    if image_input:
        print(f"   Input Img: {image_input}")
    
    if "bark" in model_id.lower():
        print(f"   Duration: Auto (Text-based)")
    else:
        print(f"   Duration: {duration}s")
        
    print(f"   Sampling: {sampling_rate}Hz")
    print(f"   Output:   {output_path}")
    print("")
    
    try:
        import torch
        import scipy.io.wavfile
        from transformers import pipeline
        from diffusers import AudioLDM2Pipeline
        
        # Logic for Different Model Types
        if "musicgen" in model_id.lower():
            print(f"   Loading MusicGen pipeline...")
            # Use device_map="auto" for offloading if CUDA
            if device.type == "cuda":
                synthesizer = pipeline("text-to-audio", model_id, device_map="auto")
            else:
                synthesizer = pipeline("text-to-audio", model_id, device=device)
            
            max_tokens = int(duration * 50)
            print(f"🎵 Synthesizing audio... (MusicGen)")
            
            start_time = time.time()
            with ResourceMonitor() as monitor:
                music = synthesizer(prompt, forward_params={"max_new_tokens": max_tokens})
            
            gen_duration = time.time() - start_time
            avg_cpu, avg_ram, avg_vram, avg_gpu = monitor.get_averages()
            tracker.record_linear("audio", model_id, device, duration, gen_duration, 
                                 cpu=avg_cpu, ram=avg_ram, vram=avg_vram, gpu=avg_gpu, dtype=dtype_name)
            print(f"   ✓ Generated in {format_time(gen_duration)} (RAM: {avg_ram:.1f}GB | "
                  f"VRAM: {avg_vram:.1f}GB | CPU: {avg_cpu:.1f}% | GPU: {avg_gpu:.1f}%)")
            tracker.print_actual(gen_duration, avg_cpu, avg_ram, avg_vram, avg_gpu)
            
            if report_json:
                stats = {"time": gen_duration, "ram": avg_ram, "vram": avg_vram, 
                        "cpu": avg_cpu, "gpu": avg_gpu}
                write_report_json(report_json, stats)
            
            rate = music["sampling_rate"]
            audio_data = music["audio"]
            
            scipy.io.wavfile.write(output_path + ".tmp.wav", rate, audio_data.T)
            src_path = output_path + ".tmp.wav"
            
        elif "audioldm" in model_id.lower():
            from transformers import GPT2LMHeadModel
            print(f"   Loading AudioLDM2 pipeline components...")
            language_model = GPT2LMHeadModel.from_pretrained(model_id, subfolder="language_model").to(dtype=dtype)
            
            pipe = AudioLDM2Pipeline.from_pretrained(
                model_id, 
                language_model=language_model,
                torch_dtype=dtype
            )

            
            if device.type == "cuda":
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(device)
            
            print(f"🎵 Synthesizing audio... (AudioLDM2)")
            
            start_time = time.time()
            with ResourceMonitor() as monitor:
                audio = pipe(prompt, audio_length_in_s=duration).audios[0]
            
            gen_duration = time.time() - start_time
            avg_cpu, avg_ram, avg_vram, avg_gpu = monitor.get_averages()
            tracker.record_linear("audio", model_id, device, duration, gen_duration,
                                 cpu=avg_cpu, ram=avg_ram, vram=avg_vram, gpu=avg_gpu, dtype=dtype_name)
            print(f"   ✓ Generated in {format_time(gen_duration)} (RAM: {avg_ram:.1f}GB | "
                  f"VRAM: {avg_vram:.1f}GB | CPU: {avg_cpu:.1f}% | GPU: {avg_gpu:.1f}%)")
            tracker.print_actual(gen_duration, avg_cpu, avg_ram, avg_vram, avg_gpu)
            
            if report_json:
                stats = {"time": gen_duration, "ram": avg_ram, "vram": avg_vram,
                        "cpu": avg_cpu, "gpu": avg_gpu}
                write_report_json(report_json, stats)
            
            rate = 16000
            scipy.io.wavfile.write(output_path + ".tmp.wav", rate, audio.T)
            src_path = output_path + ".tmp.wav"
            
        elif "stable-audio" in model_id.lower():
            from diffusers import StableAudioPipeline, EulerDiscreteScheduler
            
            print(f"   Loading StableAudioPipeline...")
            pipe = StableAudioPipeline.from_pretrained(model_id, torch_dtype=dtype)
            
            if device.type == "cuda":
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(device)
            
            print(f"   ℹ️  Swapping scheduler to EulerDiscrete (MPS optimization)")
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            
            print(f"🎵 Synthesizing audio... (Stable Audio)")

            start_time = time.time()
            with ResourceMonitor() as monitor:
                audio = pipe(prompt, audio_start_in_s=0.0, audio_end_in_s=duration, 
                           num_inference_steps=50).audios[0]

            gen_duration = time.time() - start_time
            avg_cpu, avg_ram, avg_vram, avg_gpu = monitor.get_averages()
            tracker.record_linear("audio", model_id, device, duration, gen_duration,
                                 cpu=avg_cpu, ram=avg_ram, vram=avg_vram, gpu=avg_gpu, dtype=dtype_name)
            print(f"   ✓ Generated in {format_time(gen_duration)} (RAM: {avg_ram:.1f}GB | "
                  f"VRAM: {avg_vram:.1f}GB | CPU: {avg_cpu:.1f}% | GPU: {avg_gpu:.1f}%)")
            tracker.print_actual(gen_duration, avg_cpu, avg_ram, avg_vram, avg_gpu)
            
            rate = 44100
            
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().float().numpy()
            
            scipy.io.wavfile.write(output_path + ".tmp.wav", rate, audio.T)
            src_path = output_path + ".tmp.wav"
            
        elif "bark" in model_id.lower():
            from transformers import BarkModel, AutoProcessor
            print(f"   Loading Bark models...")
            
            # Bark requires float32 on all platforms
            bark_dtype = torch.float32
                
            processor = AutoProcessor.from_pretrained(model_id)
            model = BarkModel.from_pretrained(model_id, torch_dtype=bark_dtype).to(device)
            
            print(f"🎵 Synthesizing audio... (Bark)")
            if duration > 14:
                print(f"   (Note: Bark generates max ~14s sequences per history block. "
                      f"Output will be shorter than {duration}s)")
            
            print(f"""   💡 Tip:
   *  Lyrics: Use '♪' for singing (e.g., `♪ Hello World ♪`).
   *  Effects: Use tags like `[laughter]`, `[cheers]`, `[music]`, `[sighs]`, `[gasps]`, `[clears throat]`, `—` (hesitation).
   *  Plain text without these tokens will usually be spoken as speech.
   *  Voice: Using preset '{voice_preset}'. Change with --voice-preset (e.g. 'v2/fr_speaker_1').
   *  Example: `python ai-media.py -a --audio-model bark -p "♪ Hello World ♪ [laughter]"`""")
            
            # Decide if Long-Form is needed
            is_long = len(prompt) > 150 or duration > 15.0

            start_time = time.time()
            with ResourceMonitor() as monitor:
                if is_long:
                    print(f"   📜 Long text detected. Using chunked generation.")
                    audio_array = generate_long_bark(prompt, processor, model, device, voice_preset)
                else:
                    inputs = processor(prompt, voice_preset=voice_preset).to(device)
                    audio_array = model.generate(**inputs)
                    audio_array = audio_array.cpu().numpy().squeeze()
            
            gen_duration = time.time() - start_time
            avg_cpu, avg_ram, avg_vram, avg_gpu = monitor.get_averages()
            tracker.record_linear("audio", model_id, device, duration, gen_duration,
                                 cpu=avg_cpu, ram=avg_ram, vram=avg_vram, gpu=avg_gpu, dtype="float32")
            print(f"   ✓ Generated in {format_time(gen_duration)} (RAM: {avg_ram:.1f}GB | "
                  f"VRAM: {avg_vram:.1f}GB | CPU: {avg_cpu:.1f}% | GPU: {avg_gpu:.1f}%)")
            tracker.print_actual(gen_duration, avg_cpu, avg_ram, avg_vram, avg_gpu)
            
            if report_json:
                stats = {"time": gen_duration, "ram": avg_ram, "vram": avg_vram,
                        "cpu": avg_cpu, "gpu": avg_gpu}
                write_report_json(report_json, stats)
            
            rate = model.generation_config.sample_rate
            scipy.io.wavfile.write(output_path + ".tmp.wav", rate, audio_array)
            src_path = output_path + ".tmp.wav"
            
        else:
            print(f"❌ Unknown model type for audio: {model_id}")
            return False

        # Conversion / Final Save
        must_convert = not output_path.lower().endswith(".wav")
        
        if must_convert:
            import subprocess
            cmd = ["ffmpeg", "-y", "-i", src_path, output_path, "-loglevel", "error"]
            try:
                subprocess.run(cmd, check=True)
                os.remove(src_path)
                print(f"✅ Converted and saved to {output_path}")
            except subprocess.CalledProcessError:
                print(f"⚠️  FFmpeg conversion failed. Saved as WAV: {src_path}")
        else:
            os.rename(src_path, output_path)
            print(f"✅ Audio saved to {output_path}")
            
        return True
        
    except ImportError as e:
        print(f"❌ Error: Missing dependencies or import failed. {e}")
        return False
    except Exception as e:
        print(f"❌ Audio generation failed: {e}")
        return False
