# Video Generation

Create engaging short video clips using advanced open-source video diffusion models. This tool simplifies the complex process of video generation, offering three powerful modes:

### 1. Text-to-Video
Generate video content from scratch using descriptive text prompts.
*   **Best for**: Abstract visuals, stock footage replacement, surreal scenes, or quick motion concepts.
*   **Models**: Supports a wide range from fast/efficient models (**Zeroscope**, **LTX-Video**) to massive state-of-the-art cinematic generators (**Wan 2.2**, **HunyuanVideo**).

### 2. Image-to-Video (`-ii`)
Bring your static images to life. The model takes your image as the starting frame and animates it based on your prompt.
*   **Best for**: Animating photos ("Make the water flow"), adding movement to AI-generated art, or creating cinematographs.
*   **Models**: Stable Video Diffusion (SVD), CogVideoX, and LTX-Video support this mode.

### 3. Audio-Reactive Video (`-ap`)
Automatically synchronize video generation with audio. You provide an Audio Prompt (e.g., "Techno beat"), and the tool:
1.  Generates a unique audio track.
2.  Generates a video matching your visual prompt.
3.  Muxes them together into a final MP4 file.

← [Back to Main README](../README.md)

## Options

| Option | Description |
| :--- | :--- |
| `--video-model` | Model: `zeroscope` (default), `ms-1.7b`, `wan2.2`, `ltx-video`, `mochi-1`, `hunyuan`, `cogvideox`, `svd`. |
| `-s, --size` | Target resolution. For zeroscope: triggers **Dynamic Upscaling** (XL + Real-ESRGAN). |
| `-l, --length` | Duration. Supports "2s", "5s", "1m", `{m:1, s:30}`. Default: 2s. |
| `-ii, --input-image` | Source image path for **Image-to-Video** generation (SVD, CogVideoX I2V). |
| `-ap, --audio-prompt` | **Text prompt** for generating background audio (e.g., "Techno beat"). Automatically muxes with video. Use `-am` to select audio model. |
| `-p, --prompt` | Text description of content to generate. |
| `-o, --output` | Output filename/path. Default: mp4. The folder where files are generated is configured in `config.json` under `paths.media_output`. |

See [Video Generation Examples](#examples) and [Models](#models).

### Supported Resolutions (`-s` or `--size`)
The tool supports natural language and object-style inputs:
- **Presets**: `480p`, `576p`, `720p`, `900p`, `1080p`, `1440p`, `1k` ... `10k`, `HD`, `FHD`, `UHD`
- **Dimensions**: `1280x720`, `1024x1024`
- **Objects**: `{w: 800, h: 600}`, `{width: 1920, height: 1080}`

### Supported Durations (`-l` or `--length`)
- **Strings**: `15s`, `1m`, `1h30m5s`
- **Objects**: `{m: 1, s: 30}`, `{hours: 1, minutes: 15}`
- **Numeric**: `30` (interpreted as seconds)

## Models

| Model | Code | Resolution | Download | VRAM | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Wan 2.2** | `wan2.2` | Any | ~30GB | ~24GB | **SOTA (2025)**. Excellent quality. ⚠️ **Impractical on Mac** (Too slow). |
| **LTX-Video** | `ltx-video` | Any (x32) | ~12GB | ~12GB | Balanced speed/quality (~35s for 2s on Mac). Good motion. |
| **Mochi 1** | `mochi-1` | Any (x16) | ~19GB | ~20GB | High motion fidelity. ⚠️ **Slow on Mac** (Sequential Offload). |
| **HunyuanVideo** | `hunyuan` | Any | ~25GB | ~80GB+ | Massive scale. ❌ **Incompatible with <64GB Mac**. |
| **Zeroscope** | `zeroscope` | 576×320 (native) | ~4GB | ~6GB | **Default**. Fast, no watermarks. Auto-upscales with XL. |
| **Zeroscope XL** | `zeroscope-xl` | 1024×576 | ~6GB | ~10GB | *Internal V2V upscaler*. |
| **CogVideoX** | `cogvideox` | Any | ~22GB | ~50GB+ | High fidelity. **WARNING: Very Heavy on all Systems**. |
| **Stable Video Diffusion** | `svd` | Any | ~4GB | ~8GB | **I2V Only**. ⚠️ *Very slow on Apple Silicon (CPU only).* |
| **ModelScope** | `ms-1.7b` | Any | ~10GB | ~12GB | **Legacy**. General purpose (has watermark issues). |

> [!IMPORTANT]
> **Mac M-Series (MPS) Performance Note:**
> Massive video models require enormous unified memory and specific optimizations on Mac:
> *   **LTX-Video**: ✅ Runs great natively (~35s total for 2s video). Best choice for Mac.
> *   **Mochi 1**: ⚠️ Works but is slow (~50s/step, ~25m total) due to required **Sequential CPU Offload**.
> *   **Wan 2.2**: ⚠️ Technically runs but is **impractical** (4+ hours for 2s video).
> *   **HunyuanVideo**: ❌ **Fails** on 64GB Macs. Attempts to allocate >80GB buffer even with offloading.
> *   **XL V2V**: ⚠️ **Diffusion is skipped (CPU-only = hours per video). Goes directly: 576×320 → Real-ESRGAN → FFmpeg resize. Faster but may have slight frame-to-frame variation.
> *   **Text-to-Video** models use **Float32** on MPS (Metal). Float16 produces corrupted/black frames.

> [!NOTE]
> **Zeroscope Dynamic Upscaling:** When you request a resolution higher than 576×320 with the `zeroscope` model (e.g., `-s 1080p`), the script automatically:
> 1. Generates at native 576×320 (fast, stable)
> 2. **NVIDIA/CUDA:** Upscales to 1024×576 using `zeroscope_v2_XL` (temporal-aware V2V diffusion)
> 3. Further upscales using Real-ESRGAN if target > 1024×576
> 4. Final FFmpeg resize to exact target dimensions if needed
>
> **ℹ️ NVIDIA/CUDA:** XL V2V diffusion is confirmed to work well on NVIDIA/CUDA.

> [!WARNING]
> **Watermarks in Output:** Some models (especially `ms-1.7b`) may produce videos with Shutterstock watermarks. This is because these open-source research models were trained on datasets that included watermarked stock footage. The model learned to reproduce the watermark as part of the visual pattern. This is baked into the model weights.

> [!NOTE]
> **FFmpeg Re-encoding:** Generated videos are automatically re-encoded with FFmpeg (`libx264` + `yuv420p`) for universal playback. The raw output from `diffusers` uses a codec that macOS Finder/QuickTime cannot preview (shows green frames), but the re-encoded version works in all players and displays proper thumbnails.

## Examples

### Basic Usage (Quick Start)

```bash
# Native 576x320 video (Fastest) - Default Model (Zeroscope)
python ai-media.py -v -p "A robot dancing" -l 2s
python ai-media.py --generate-video --prompt "A robot dancing" --length 2s

# Higher Quality (Auto-Upscaled to 1080p)
# Triggers Zeroscope XL + Real-ESRGAN pipeline
python ai-media.py -v -p "Ocean waves crashing" -s 1080p
python ai-media.py -v -p "Ocean waves crashing" --size 1080p
```

### Model Selection & Performance

Choose the right tool for the job.

```bash
# LTX-Video (Balanced Choice for Mac/Consumer GPU)
# Good quality, reasonable speed (~35s for 2s on Mac M1/M2 Max).
python ai-media.py -v -p "A cinematic drone shot of a forest" -vm ltx-video
python ai-media.py -v -p "A cinematic drone shot of a forest" --video-model ltx-video

# Wan 2.2 (State-of-the-Art Quality)
# ⚠️ WARNING: Extremely slow on Mac (Hours). Best for NVIDIA 24GB+ cards.
python ai-media.py -v -p "Hollywood movie scene, explosion" --video-model wan2.2

# Mochi 1 (High Motion Fidelity)
# Great for fluid dynamics, but requires sequential offload (Slow on Mac).
python ai-media.py -v -p "Milk pouring into coffee, slow motion" -vm mochi-1
```

### Image-to-Video (Animating Images)

Bring static images to life.

```bash
# Animate an existing image
# "Camera pan" or movement prompts work best.
python ai-media.py -v -p "Camera pans right, leaves rustling" -ii "./park.jpg"
python ai-media.py -v -p "Camera pans right" --input-image "./park.jpg"

# Using Stability SVD (Image-to-Video Specialist)
# ⚠️ SVD is CPU-only on Mac (Slow), GPU on NVIDIA.
python ai-media.py -v -ii "./character.png" --video-model svd
```

### Audio-Reactive Video

Generate a video with a matching AI-generated soundtrack.

```bash
# 1. Generates Video
# 2. Generates Audio based on separate prompt
# 3. Muxes them together
python ai-media.py -v -p "Cyberpunk dancers" -ap "Heavy techno beat" -l 5s
python ai-media.py -v -p "Cyberpunk dancers" --audio-prompt "Heavy techno beat" --length 5s
```

### Advanced Usage

```bash
# Long Duration (1.5 minutes)
# Note: Generating 90s of video takes a LONG time.
python ai-media.py -v -p "Clouds passing" -l "{m:1, s:30}"

# HunyuanVideo (Massive Scale)
# ❌ Likely to crash on <64GB RAM systems.
python ai-media.py -v -p "Panda custom generation" -vm hunyuan -s 720p -o panda.mp4
```

### Zeroscope Dynamic Upscaling Pipeline

When generating video with zeroscope at resolutions above native 576x320, a multi-stage temporal upscaling pipeline is used automatically:

| Stage | Resolution | Method |
| :--- | :--- | :--- |
| 1. Generate | 576x320 | zeroscope_v2_576w (base model) |
| 2. Temporal Upscale | 1024x576 | zeroscope_v2_XL (V2V diffusion) |
| 3. AI Upscale | 2x-4x | Real-ESRGAN (frame-by-frame) |
| 4. Final Resize | Target | FFmpeg Lanczos |

Use `-s 576x320` to skip upscaling and generate at native resolution (fastest).

← [Back to Main README](../README.md)
