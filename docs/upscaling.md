# Upscaling

Increase the resolution of images and videos using both AI and traditional algorithms. This tool handles the challenge of making low-res media look sharp on high-res displays.

### 1. AI Upscaling (Image & Video)
Use deep learning to hallucinate missing details and sharpen edges.
*   **Real-ESRGAN (Default)**: Fast (0.5s/image) and faithful to the source. Removes noise and JPEG artifacts. Best for 90% of use cases.
*   **Stable Diffusion Upscale**: Slower and more "creative". It re-dreams the image at a higher resolution, adding textures (like wood grain or skin pores) that weren't there originally.

### 2. Video Upscaling
Applying upscale models to 24+ frames per second.
*   **Performance**: The script uses a highly optimized FFmpeg pipe to process frames instantly as they come out of the AI model, ensuring efficiency even for long clips.
*   **Encoding**: Automatically switches codecs (H.264 vs HEVC) based on the output resolution (e.g., forcing HEVC for 8K content).

### 3. Simple Upscaling (Non-AI)
*   **Lanczos**: A high-quality mathematical resize. Instant speed, but adds no new detail. Good for quick resizing without altering the image content.

← [Back to Main README](../README.md)

## Options

| Argument | Description | Default |
| :--- | :--- | :--- |
| `-ui`, `--upscale-image` | Path to the image file to upscale. | `None` |
| `-uv`, `--upscale-video` | Path to the video file to upscale. | `None` |
| `-uof`, `--upscaled-output-file` | Custom filename for the upscaled output. | Auto: `name_upscaled_{factor}x.ext` |
| `-uf`, `--upscale-factor` | Multiplier for resolution (e.g., `1.5`, `2.0`, `4.0`). | `2.0` |
| `-us`, `--upscale-strength` | Noise strength (`0.0`-`1.0`). Higher = closer to original structure. **x4 upscaler only** (ignored for x2 latent). | `0.0` |
| `-iu`, `--image-upscaler` | Model for image upscaling: `realesrgan` (Fast, Faithful) or `sd` (Slow, Creative). | `realesrgan` |
| `-vu`, `--video-upscaler` | Model for video upscaling: `realesrgan` (Fast, Faithful) or `sd` (Slow, Creative). | `realesrgan` |
| `-vc`, `--video-codec` | Preferred Codec: `auto` (Default), `h264`, `hevc`, `av1`. See **Codec Logic** below. |
| `-su`, `--simple-upscale` | Use simple non-AI upscaling (PIL Lanczos for images, FFmpeg for videos). **Video:** Extremely fast (skips frame extraction). **Image:** Instant. | `False` |

> [!NOTE]
> **Video Upscaling Modes:**
> *   **Default (Real-ESRGAN):** Uses `RealESRGAN_x4plus`. **Fast** & faithful.
> *   **Stable Diffusion (`-vu sd`)**: Creative AI upscaling. **Slow**.
> *   **Simple (`-su`)**: Instant scaling. No new detail.
>
> **Codec Logic & Auto-Switching:**
> The script automatically probes your hardware capabilities for the target resolution.
> *   **`h264`**: Tries hardware acceleration first (`h264_nvenc` for NVIDIA, `h264_videotoolbox` for Mac). Falls back to the universal software encoder `libx264`. (Switches to HEVC if >8K).
> *   **`hevc`**: Tries hardware acceleration first (`hevc_nvenc` for NVIDIA, `hevc_videotoolbox` for Mac). Falls back to the universal software encoder `libx265`.
> *   **`av1`**: Tries hardware acceleration first (`av1_nvenc`). Falls back to HEVC (Hardware then Software).
>
> **Check Your System Limits:**
> Run `python tests/test_codec_limits.py` to stress-test your system's encoding capabilities up to 20K.

See [Upscaling Examples](#examples) and [Models](#models).

### Example: 8K Video Upscaling with HEVC

```bash
python ai-media.py -uv input_video.mp4 -uf 8.0 -vc hevc
```

> [!NOTE]
> **Resource Safety Check:** Before starting, the script calculates the target resolution (e.g., 8K = 33MP) and estimated RAM usage. If it detects a risk of massive swapping or system freeze ("Billboard Sizing"), it will warn you and ask for confirmation. Use `--force` to bypass this.

> [!IMPORTANT]
> **MacOS/Apple Silicon - Upscaling:** Enforced to run on **CPU** (Float32). This is due to PyTorch MPS limitations:
> 1. **Kernel Size Limit:** High-resolution tensors (4K+) exceed the MPS driver's maximum dimensions, causing crashes.
> 2. **BFloat16 Incomplete:** CPU BFloat16 causes hangs due to unoptimized code paths.
>
> **Result:** CPU + Float32 uses ~80GB RAM for 12K output and is slow, but is the only stable option.

## Models

| Model | Code | Download | VRAM | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **SD x2 Latent** | `upscaler_x2` | ~4GB | ~8GB | **Factors ≤ 2x**. Fast, preserves original style. |
| **SD x4 Upscaler** | `upscaler` | ~8GB | ~16GB | **Factors > 2x**. High detail, sharpens textures. |
| **Real-ESRGAN** | `realesrgan` | ~0.3GB | ~2GB | **Fast Mode**. Fast, faithful upscaling, better temporal consistency. |

## Architecture

The script uses a **Smart Multi-Stage Approach** to balance speed and quality. This is key for reaching arbitrary high resolutions (e.g. 8K, 12K) even when the AI models typically scale by fixed integers (2x, 4x).

**How it works:**
1.  **AI Pass**: The script applies the closest AI upscaler (e.g., 4x Real-ESRGAN).
2.  **Lanczos Resize**: To hit your *exact* requested resolution (which might be 5.21x or 6x, not just 4x), a final high-quality Lanczos resize is used to adjust the AI output to current dimensions.

This combination allows for:
- **Exact Dimension Targets**: You get exactly the pixel count you asked for.
- **Maximum Quality**: AI does the heavy lifting of detailing, Lanczos handles the final fitting.

| Factor | AI Passes | Final Resize | Example |
| :--- | :--- | :--- | :--- |
| **≤ 2x** | 1x x2 Latent | None | 720p → 1080p |
| **3x** | 1x x2 Latent | 1.5x Lanczos | 720p → ~2K |
| **4x** | 1x x4 Upscaler | None | 720p → ~3K |
| **5x** | 1x x4 Upscaler | 1.25x Lanczos | 720p → ~4K |
| **6x** | 1x x4 Upscaler | 1.5x Lanczos | 720p → ~4K |
| **8x** | 1x x4 + 1x x2 | None | 720p → ~6K |
| **10x** | 1x x4 + 1x x2 | 1.25x Lanczos | 720p → ~7K |

**Model Details:**
- **SD x2 Latent** (`upscaler_x2`): Works in latent space, fast and faithful to original style.
- **SD x4 Upscaler** (`upscaler`): Generates new texture details. Supports `--upscale-strength`.
- **Lanczos**: High-quality non-AI resize for fractional remainder.

> [!TIP]
> **Reducing Upscaling Artifacts:** By default, the x4 upscaler uses `noise_level=0` which stays faithful to the original image. If your upscaled images look too "painted" or have artifacts:
> - Keep `--upscale-strength 0.0` (default) for most AI-generated images
> - Try `-us 0.2` to `-us 0.5` if you want more detail generation (real photos)
> - Higher values (`-us 0.8+`) give the model more creative freedom but may diverge from the original
>
> The upscaler also uses negative prompts internally to reduce blur, noise, and JPEG artifacts.

## Examples

## Examples

### Image Upscaling

```bash
# Real-ESRGAN (Default) - Fast & Faithful
# Best for 90% of use cases.
python ai-media.py -ui photo.jpg -uf 4.0
python ai-media.py --upscale-image photo.jpg --upscale-factor 4.0

# Stable Diffusion Upscale - Creative Detail
# Slower, adds texture/detail not present in original. Good for rough sketches or low-res art.
python ai-media.py -ui art.jpg -uf 2.0 -iu sd
python ai-media.py -ui art.jpg -uf 2.0 --image-upscaler sd

# Simple Upscale (Lanczos)
# Instant resize, no AI.
python ai-media.py -ui icon.png -uf 4 -su
python ai-media.py -ui icon.png -uf 4 --simple-upscale
```

### Video Upscaling

```bash
# Upscale 1080p -> 4K (Real-ESRGAN)
# Fastest AI method (~1-2s per frame).
python ai-media.py -uv clip.mp4 -uf 2.0
python ai-media.py --upscale-video clip.mp4 --upscale-factor 2.0

# Force HEVC Codec (Best for 8K)
python ai-media.py -uv video.mp4 -uf 4.0 -vc hevc
python ai-media.py -uv video.mp4 -uf 4.0 --video-codec hevc
```

### Chaining (Creation + Upscale)

Generate media and immediately upscale it in one command.

```bash
# Generate 720p Image -> Upscale to ~3K (4x)
python ai-media.py -i -p "Epic mountain" -s 720p --upscale -uf 4x
```

> [!NOTE]
> **Standalone Video Upscaling Modes (`-uv`):**
>
> | Mode | Speed | Quality | Method |
> | :--- | :--- | :--- | :--- |
> | **Real-ESRGAN** (default) | **Fast** (~1s/frame) | Faithful enhancement | Single forward pass, direct FFmpeg pipe |
> | **Stable Diffusion** (`-vu sd`) | Slow (~10s/frame) | Creative/detailed | 50 diffusion steps, disk extraction |
> | **Simple** (`-su`) | Instant | No AI enhancement | FFmpeg Lanczos scaling |
>
> Audio is automatically preserved from the source video if present.

> [!NOTE]
> **Video Encoding (Cross-Platform):**
> *   **macOS/Linux**: OpenCV uses native system codecs (AVFoundation/FFmpeg) for H.264 encoding.
> *   **Windows**: Falls back to a robust FFmpeg pipe for encoding. The "FFmpeg Pipe initialized" message is expected behavior.
> *   **8K+ Support**: Automatically switches to HEVC (H.265) for resolutions above 8192x4320 (H.264's maximum).

### Smart Format Handling

```bash
# Implicit Format (Inferred from filename)
python ai-media.py -i -p "Cat" -o cat.png      # Generates PNG
python ai-media.py -i -p "Cat" -o cat.jpg      # Generates JPEG

# Implicit Extension (Inferred from format flag)
python ai-media.py -i -p "Cat" -o my_image -f png   # Auto-saves as "my_image.png"
```

← [Back to Main README](../README.md)
