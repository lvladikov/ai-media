# Image Generation

Generate high-quality images from text prompts using state-of-the-art diffusion models running locally on your machine.

This tool provides a unified interface for multiple generations of image models, handling all the complex pipeline setup, memory management, and optimization for you.

### What can you do?
*   **Text-to-Image**: Turn "A cyberpunk city" into a detailed 4K wallpaper.
*   **Format Control**: Generate in Landscape (16:9), Portrait (9:16), or Square (1:1), and save as lossless PNG or compressed JPG.
*   **Model Flexibility**: Switch between **SDXL** (modern, high detail), **SD 1.5** (fast, artistic), or **Flux** (cutting-edge text and realism).
*   **Proactive Optimization**: Automatically handles "Out of Memory" risks. If you ask for a massive 8K image, it smartly generates a smaller base and uses AI upscaling to reach your target without crashing your system.

← [Back to Main README](../README.md)

## Options

| Option | Description |
| :--- | :--- |
| `--image-model` | Model: `sd3.5-turbo` (default), `sdxl`, `sd-1.5`, `sd3.5-medium`, `sd3.5-large`, `flux`, `flux-dev`, `qwen-image`, `qwen-image-mps`. See [Models](#models) below. |
| `-otn, --orientation` | `landscape` (default), `portrait`, or `square`. Portrait swaps w/h. |
| `--unsafe` | Disable NSFW safety checker (reduces false positives). |
| `-p, --prompt` | Text description of content to generate. |
| `-o, --output` | Output filename/path. **Optional**: auto-generated from first 2 words of prompt if omitted. |
| `-f, --format` | File format: jpg, png (default: jpg). |
| `-s, --size` | Resolution. Supports "720p", "1080p", "4k", "8k", "HD", "1280x720", "1536" (square), `{w:1280, h:720}`. Default: 720p. |

See [Image Generation Examples](#examples) and [Models](#models).

### Supported Resolutions (`-s` or `--size`)
The tool supports natural language and object-style inputs:
- **Presets**: `480p`, `576p`, `720p`, `900p`, `1080p`, `1440p`, `1k` ... `10k`, `HD`, `FHD`, `UHD`
- **Dimensions**: `1280x720`, `1024x1024`
- **Single Number (Square)**: `1536` → 1536×1536, `1024` → 1024×1024
- **Objects**: `{w: 800, h: 600}`, `{width: 1920, height: 1080}`

## High-Resolution Optimization (Proactive Workflow)

Generating native 4K+ or 8K images in a single pass can be extremely slow and frequently triggers "Out of Memory" (OOM) errors or driver crashes (especially on Apple Silicon).

To solve this, `ai-media` includes a proactive optimization workflow for high-res requests:

1.  **Threshold Detection**: If you request a size larger than **6 Megapixels** (approx. 3K territory), the script will detect it.
2.  **Interactive Recommendation**: You will be prompted to switch to the **Optimized Workflow**:
    -   **Step 1**: Generate at an optimized **3K base** (3072px long edge). This is fast and fits in most GPU buffers.
    -   **Step 2**: Automatically **AI Upscale** the result to your exact target resolution.
3.  **Benefits**: 
    -   **Stability**: Avoids kernel crashes and OOM errors.
    -   **Speed**: Generating at 3K + Upscaling is often faster than native 4K/8K.
    -   **Seamlessness**: The upscale happens automatically after generation finishes.

### Smart Multi-Stage Strategy (How it reaches 8K/16K)
To ensure the highest quality and exact dimensions, the script uses a **multi-stage pipeline**:
-   **AI Passes**: It first applies high-performance 4x models like **Real-ESRGAN** (default) or **Stable Diffusion** (if requested).
-   **Lanczos Resize**: Because high-res targets (like 5.21x) rarely land on clean multiples, a final high-quality **Lanczos resize** is applied to match your requested resolution exactly.
    - *Example (6x)*: 4x AI Pass + 1.5x Lanczos Resize.
    - *Example (5.21x)*: 4x AI Pass + 1.30x Lanczos Resize.

> [!TIP]
> This workflow defaults to **Real-ESRGAN** for the AI pass, ensuring a fast and faithful expansion of the 3K base image.

## Models

| Model | Code | Download | VRAM | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **SDXL Turbo** | `sdxl` | ~8GB (16GB on Mac) | ~8GB (~16GB on Mac) | Fast, reasonable quality, older model - use this as Default if you haven't yet Accepted License at HuggingFace for the Gated current Default sd3.5-turbo. Uses float32 on Apple Silicon. |
| **SD 1.5** | `sd-1.5` | ~4GB | ~4GB | Lightweight, lower VRAM. ⚠️ NSFW filter issues on non-CUDA. Older model. |
| **SD 3.5 Medium** | `sd3.5-medium` | ~10GB | ~10GB | Consumer-friendly, high quality. 🔒 **Gated**. |
| **SD 3.5 Large** | `sd3.5-large` | ~19GB | ~19GB | Best quality. 🔒 **Gated**. |
| **SD 3.5 Large Turbo** | `sd3.5-turbo` | ~19GB | ~19GB | **Default**. Fast (4 steps) and Good Quality. 🔒 **Gated**. |
| **Qwen-Image** | `qwen-image` | ~20GB | ~20GB | Best text rendering. 🔒 **CUDA only** (4-bit). |
| **Qwen-Image (MPS)** | `qwen-image-mps` | ~40GB | ~40GB | Text rendering on Mac. Float32. |
| **Flux Schnell** | `flux` | ~33GB | ~12GB+ (~70GB on Mac) | High quality. 🔒 **Gated**. **⚠️ Impractical on Mac (Slow)**. |
| **Flux Dev** | `flux-dev` | ~33GB | ~16GB+ (~80GB on Mac) | Professional creative work. 🔒 **Gated**. **⚠️ Impractical on Mac**. |
| **FLUX.2 (4-bit)** | `flux2` | ~18GB | ~20GB VRAM | State-of-the-art. 4K capable. 🔒 **Gated**. **NVIDIA RTX 3090+ recommended**. |
| **FLUX.2 (Full)** | `flux2-full` | ~65GB | ~90GB+ VRAM/RAM | Maximum quality. 🔒 **Gated**. ⚠️ Mac: 128GB+ RAM required. |

> [!NOTE]
> **Apple Silicon/MPS:** SDXL Turbo uses float32 precision on Mac to avoid black images (float16 produces NaN values in VAE). This doubles memory usage compared to NVIDIA/CUDA.
>
> **FLUX.2 on Mac:** The 4-bit quantized version (`flux2`) requires `bitsandbytes` which only works on CUDA/NVIDIA GPUs. On Mac, it falls back to the full model with CPU offloading. **⚠️ Even 64GB unified RAM is not enough** — the process will be killed by Mac OS OOM. Recommended only for **high-end Macs with 128GB+ RAM**. For most Mac users, use `flux`, `sd3.5-turbo` or `sdxl` instead.
>
> **High Resolution & SD 3.5 Limit:** SD 3.5 models have a recommended max resolution of **1296×1296** (higher causes noise artifacts). Architectural hard limit is 1536×1536. For larger sizes, use `--upscale`. VAE Tiling is auto-enabled for resolutions above 1536×1536.
>
> **Gated Models (Flux, FLUX.2, SD 3.5):** Require HuggingFace login. Accept the license at [huggingface.co/black-forest-labs](https://huggingface.co/black-forest-labs) for Flux or [huggingface.co/stabilityai](https://huggingface.co/stabilityai) for SD 3.5 and run `huggingface-cli login`.

### SD 3.5 Large vs Large Turbo

Both models share the same **8.1B parameter** architecture but differ in speed and quality:

| Aspect | SD 3.5 Large | SD 3.5 Large Turbo |
|--------|-------------|-------------------|
| **Steps** | 40 | 4 (10x faster) |
| **Speed** | ~30-60 sec | ~5-10 sec |
| **Quality** | Best details, anatomy, text | Slightly less nuanced |
| **VRAM** | ~19 GB | ~19 GB |

**Turbo** is a distilled version trained to produce high-quality results in fewer steps without guidance. Use **Large** for final production images; use **Turbo** for quick iterations and batch generation.

### Qwen-Image: Best Text Rendering

> [!IMPORTANT]
> **Platform-Specific Variants:** Qwen-Image uses 4-bit quantization (`bitsandbytes`) which only works on CUDA/NVIDIA GPUs.
> - **CUDA:** Uses `qwen-image` (4-bit, 20GB VRAM, 8 steps)
> - **MPS (Mac):** Uses `qwen-image-mps` (Full, float32, 15 steps)
>
> The script **automatically switches** to the correct variant for your hardware. If you select `qwen-image` on Mac, it will switch to `qwen-image-mps` and display an info message.

| Feature | Qwen-Image |
|---------|-----------|
| **Text in images** | ✅ Best (English + Chinese) |
| **Parameters** | 20B |
| **Speed** | ~8-15 steps |

## Examples

### Basic Usage (Quick Start)

```bash
# Generate a standard 720p image (Default model: SD 3.5 Turbo)
python ai-media.py -i -p "Cyberpunk city at night with neon lights"

# Explicit output filename
python ai-media.py --generate-image --prompt "A cute robot holding a flower" --output robot.png
```

### Model Comparison & Resources

Different models have different strengths and resource requirements.

```bash
# SDXL Turbo - Fast & High Quality
# Uses ~8GB VRAM (or ~16GB RAM on Mac). Best all-rounder without login.
python ai-media.py -i -p "Cinematic portrait of an astronaut" --image-model sdxl

# Stable Diffusion 1.5 - Low Resource / Vintage Style
# Uses only ~4GB VRAM. Good for older GPUs or specific artistic styles.
python ai-media.py -i -p "Oil painting of a cottage" -im sd-1.5

# Flux Schnell/Dev - State-of-the-Art (Gated)
# Requires HuggingFace login. VERY memory intensive (~33GB+ VRAM/RAM).
# Delivers incredible text adherence and realism.
python ai-media.py -i -p "A sign that says 'HELLO WORLD' in neon text" -im flux

# SD 3.5 Medium - Best balance of quality/speed for consumer hardware (Gated)
# ~10GB VRAM. Great prompt following, improved anatomy/typography.
python ai-media.py -i -p "A capybara holding a sign that says Hello World" -im sd3.5-medium

# SD 3.5 Large Turbo (Default) - Fast high-quality generation (Gated)
# Only 4 inference steps, ~19GB VRAM.
python ai-media.py -i -p "Photorealistic portrait of an astronaut" -im sd3.5-turbo
```

### Aspect Ratios & Formats

Control the shape and file type of your output.

```bash
# Portrait Mode (9:16) - Great for phone wallpapers or character art
# Swaps dimensions: 720x1280 instead of 1280x720
python ai-media.py -i -p "Full body fashion photo" -otn portrait
# Long form:
python ai-media.py --generate-image --prompt "Fashion photo" --orientation portrait

# Custom Resolution (Square)
python ai-media.py -i -p "A symmetric mandala" -s 1024x1024
# Long form:
python ai-media.py -i -p "A symmetric mandala" --size 1024x1024

# Quality Formats (PNG vs JPG)
# Use PNG for lossless quality, JPG for smaller files.
python ai-media.py -i -p "Detailed map interface" -f png
python ai-media.py -i -p "Detailed map interface" --format png
```

### Advanced Optimization (High-Res 4K+)

For very large images, use the "Proactive Optimization" workflow to avoid crashes.

```bash
# Generate a 16K image (Massive!)
# 1. Generates 3K base image (stable)
# 2. Upscales 5.2x to reach exactly 16000x16000
python ai-media.py -i -p "Detailed nebula space background" -s 16000x16000 --image-model sd-1.5

# Chained Generation -> Upscale
# Manually generate at 720p, then upscale 2x immediately.
python ai-media.py -i -p "Mountains" -s 720p --upscale -uf 2x
python ai-media.py -i -p "Mountains" -s 720p -u --upscale-factor 2x
```

### Safety Options

```bash
# Disable Safety Checker
# Useful if you get false positive black images (common on Mac/MPS with SD 1.5).
python ai-media.py -i -p "Classical marble statue" --unsafe
```

← [Back to Main README](../README.md)
