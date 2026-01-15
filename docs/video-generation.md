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
| `-o, --output` | Output filename/path. Auto-generated from prompt if omitted. The folder where files are generated is configured in `config.json` under `paths.media_output`. |
| `-f, --format` | Output format: `mp4` (default), `webm`, `mov`, `mkv`, `avi`, `flv`, `ts`, `gif`. See [Output Formats](#output-formats---o-or---output). |

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

### Output Formats (`-o` or `--output`)

The output format is determined by the file extension you specify. The tool generates video internally, then **automatically converts** to your desired format using ffmpeg.

**Supported Formats:**
| Format | Extension | Notes |
| :--- | :--- | :--- |
| MP4 | `.mp4` | **Default**. H.264 codec, universal compatibility. |
| WebM | `.webm` | VP9 codec. Great for web. |
| MOV | `.mov` | Apple QuickTime. Good for editing. |
| MKV | `.mkv` | Matroska. Flexible container. |
| AVI | `.avi` | Legacy format. Wide compatibility. |
| FLV | `.flv` | Flash Video. Legacy streaming format. |
| TS | `.ts` | MPEG-TS. Broadcast/streaming format. |
| GIF | `.gif` | Animated GIF. Large files, no audio. |

**Examples:**
```bash
# MP4 (default) - using -o with extension
python ai-media.py -v -p "Ocean waves" -o my_video.mp4

# MP4 - using -f flag (explicit format)
python ai-media.py -v -p "Ocean waves" -f mp4

# WebM (web-optimized)
python ai-media.py -v -p "Ocean waves" -f webm
python ai-media.py -v -p "Ocean waves" --format webm

# GIF (animated, no audio)
python ai-media.py -v -p "Ocean waves" -f gif

# MOV (QuickTime)
python ai-media.py -v -p "Ocean waves" -f mov
```

> [!TIP]
> If no output is specified, a unique filename with `.mp4` extension is generated automatically.

 
## Precision & Framework Control

AI-Media supports fine-grained control over model precision and ML framework. For a deep dive into precision types and their trade-offs, see **[Precisions Explained](precisions-explained.md)**.

### Quick Reference

| Option | Description |
|--------|-------------|
| `--precision-force`, `-pf` | Force precision: `int4`, `int6`, `int8`, `float16`, `bfloat16`, `float32` |
| `--ml-framework`, `-mf` | Force framework (Mac): `mlx` (native) or `torch` (PyTorch MPS) |
 
## Platform Defaults (Video Models)
 
When no precision is specified, AI-Media selects the best settings for your hardware:
 
| Platform | Default Precision | Default Framework | Notes |
|----------|-------------------|-------------------|-------|
| **CUDA (NVIDIA)** | `float16` | PyTorch | Standard backend. Fast on 30/40-series cards. |
| **MPS (Mac PyTorch)**| `float16` | PyTorch | Legacy default. Used when MLX port is unavailable. |
| **MLX (Mac Native)** | `int4` | MLX | Optimized for Apple Silicon. Requires `mlx-vlm`. |

---

## Models

| Model | Code | Resolution | MLX Support | VRAM (Est) | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Wan 2.2** | `wan2.2` | Any | ✅ **Native** (int4) | ~16GB (Mac) | **SOTA (2025)**. Excellent quality. Now fast on M1/M2/M3 via MLX. |
| **LTX-Video** | `ltx-video` | Any (x32) | ✅ **Native** (int4) | ~12GB | Balanced speed/quality (~35s for 2s). Good motion. |
| **HunyuanVideo** | `hunyuan` | Any | ✅ **Native** (int4) | ~24GB (Mac) | Massive scale. Requires 32GB+ Unified Memory. |
| **CogVideoX** | `cogvideox` | Any | ✅ **Native** (int4) | ~18GB (Mac) | High fidelity. Heavy on CUDA/PyTorch, manageable on MLX. |
| **Mochi 1** | `mochi-1` | Any (x16) | ❌ **No** (Runs on PyTorch)* | ~20GB | High motion fidelity. Uses standard (slower) PyTorch backend. |
| **Zeroscope** | `zeroscope` | 576×320 | ❌ No | ~6GB | **Default**. Fast, no watermarks. Auto-upscales with XL. |
| **Stable Video Diffusion** | `svd` | Any | ❌ No | ~8GB | **I2V Only**. Slower on Mac (CPU fallback). |
| **ModelScope** | `ms-1.7b` | Any | ❌ No | ~12GB | Legacy. |

> [!NOTE]
> ***Mochi 1 (PyTorch Only)**: While a "Partial" MLX port exists in the community (DiT-only), it requires a complex hybrid setup (MLX DiT + PyTorch VAE) that causes severe memory swapping and bottlenecks. For stability and reliability, we strictly use the **unified PyTorch (MPS)** pipeline for Mochi 1.

> [!IMPORTANT]
> **Mac M-Series (Apple Silicon) Performance:**
> We offer two backends for Mac users:
> 1.  **MLX Native (Recommended)**: Utilizes `mlx-community` 4-bit quantized models. fast, low memory, and optimized for M1/M2/M3 chips.
>     *   **Why `mlx-vlm`?** To these new models, a video is just a sequence of visual tokens, just like text is a sequence of word tokens. Because `mlx-vlm` is already built to pipe "Images + Text" into a Transformer, it became the natural home for these advanced video models on Apple Silicon.
>     *   **Wan 2.2**: ✅ Runs native via `mlx-vlm`. (Use `int4` precision).
>     *   **LTX-Video**: ✅ Runs native.
>     *   **HunyuanVideo**: ✅ Runs native (4-bit).
>     *   **CogVideoX**: ✅ Runs native (4-bit).
> 2.  **PyTorch (MPS Fallback)**: Uses standard Diffusers.
>     *   **Text-to-Video** models use **Float32** on MPS (Metal) to avoid artifacts, which is slower and memory-hungry.
>     *   **Legacy Models**: Zeroscope and SVD currently use this fallback as native MLX ports are not yet standard.

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
# LTX-Video (Native Mac Speed & Quality)
# Runs natively on specialized MLX backend (if chosen or default on Mac).
python ai-media.py -v -p "A cinematic drone shot of a forest" -vm ltx-video
python ai-media.py -v -p "A cinematic drone shot of a forest" --video-model ltx-video

# Wan 2.2 (SOTA Quality)
# Now runs EFFICIENTLY on Mac via Native MLX (int4).
# No longer extremely slow.
python ai-media.py -v -p "Hollywood movie scene, explosion" --video-model wan2.2 -mf mlx -pf int4

# Mochi 1 (High Motion Fidelity)
# ⚠️ Note: Still uses slower PyTorch backend on Mac (Partial support).
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
# Now accessible on 32GB+ Macs via MLX (int4).
python ai-media.py -v -p "Panda custom generation" -vm hunyuan -mf mlx -pf int4 -s 720p -o panda.mp4
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
