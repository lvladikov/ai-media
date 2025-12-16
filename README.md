# AI-Media

Generate images, videos, and audio locally using state-of-the-art open source AI models. Upscale existing images and videos with AI. This tool wraps libraries like `diffusers` and `transformers` into a simple, unified command-line interface.

## Features

- 🎨 **Image Generation** - Text-to-Image using models like generic Flux/SDXL (via `diffusers`)
- 🎬 **Video Generation** - Supports **Text-to-Video**, **Image-to-Video**, and **Video-with-Audio** (automatic Muxing with FFmpeg).
- 🎵 **Audio Generation** - Supports **Text-to-Audio** and **Image-to-Audio** (using Visual Captioning). Models: MusicGen, AudioLDM 2.
- 📈 **AI Upscaling** - Upscale images and videos using Stable Diffusion models (x2 Latent, x4 Standard). Supports custom factors and chained workflows.
- ⚙️ **Power User Controls**
    - Flexible resolution parsing (strings like "720p", "4k", "1920x1080", or objects like `{w:1920, h:1080}`)
    - Smart time parsing ("1h50m", "15s", `{m:2, s:30}`)
    - Format conversion (JPG, PNG, MP4, MP3, WAV, etc.)
- 🚀 **Hardware Accelerated** - Auto-detects and optimizes for:
    - 🍏 **Apple Silicon** (MPS / Metal)
    - 🟢 **NVIDIA GPUs** (CUDA + Float16)
    - 💻 **CPU Performance Tracking** - To improve estimation accuracy, the script creates a `performance.json` file in its directory. This file is **local only** and never uploaded. To disable, use `--no-performance-tracking` (or `--npt`). 
    - [See details](#performance-tracking).

## Prerequisites

1.  **Python 3.10 - 3.12** (3.12 is recommended for improved performance)
2.  **FFmpeg** (Recommended for advanced format conversion)
    -   macOS: `brew install ffmpeg`

### Python Dependencies (installed via requirements.txt)

- **diffusers**: State-of-the-art Image & Video generation pipelines
- **transformers**: Audio generation & text processing models
- **torch**: Core deep learning framework & hardware acceleration (CUDA/MPS)
- **accelerate**: Optimization for efficient large model loading
- **opencv-python**: Video frame processing & manipulation
- **scipy**: Audio signal processing & file handling

### 🔐 Gated Models (Optional)
Some state-of-the-art models (like `FLUX.1` or `CogVideoX`) require authentication (but are **free to use**):
1.  **The CLI will be installed as part of the installation process (requirements.txt)**
2.  Create a **Free** [Hugging Face Account](https://huggingface.co/join).
3.  Accept the model license on the model card page (Free).
4.  **Login**: Run `huggingface-cli login` in your terminal and paste your Access Token.
*Note: The default model `sd-1.5` is open and requires no login.*

## Installation

```bash
# 1. Clone or navigate to the project
cd ai-media

# 2. Create virtual environment
# TIP: Create with a specific version (e.g. 3.12) if your default 'python3' is too new (like 3.14).

   # macOS (Homebrew):
   /opt/homebrew/bin/python3.12 -m venv venv

   # or
   
   python3.12 -m venv venv

   # Linux:
   python3.12 -m venv venv

   # Windows:
   py -3.12 -m venv venv

# 3. Activate environment
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. [Optional] Login to Hugging Face (Free); See Gated Models (Optional) section above.
# Only required if you want to use Gated models like Flux or CogVideoX.
huggingface-cli login
# (Paste your Access Token when prompted. It is invisible.)
```

## Usage

The script `ai-media.py` is the main entry point.

### AI Upscaling (Standalone)
You can directly upscale existing images or videos without generating new ones.

```bash
# Upscale an image to 4K (assuming input is 1920x1080) with 2x factor
python ai-media.py --upscale-image "my-image.jpg" -uf 2.0

# Upscale a video
python ai-media.py --upscale-video "my-video.mp4" -uf 2.0
```

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--upscale-image` | Path to the image file you want to upscale. | `None` |
| `--upscale-video` | Path to the video file you want to upscale. | `None` |
| `-uof`, `--upscaled-output-file` | Custom filename for the upscaled output. | Auto: `name_upscaled_{factor}x.ext` |
| `-uf`, `--upscale-factor` | Multiplier for resolution (e.g., `1.5`, `2.0`, `4.0`). | `2.0` |
| `-us`, `--upscale-strength` | *Experimental*: Noise strength. | `0.0` |

> [!NOTE]
> **Resource Safety Check:** Before starting, the script calculates the target resolution (e.g., 8K = 33MP) and estimated RAM usage. If it detects a risk of massive swapping or system freeze ("Billboard Sizing"), it will warn you and ask for confirmation. Use `--force` to bypass this.

> [!IMPORTANT]
> **MacOS Users:** Upscaling is enforced to run on **CPU** (Float32) on Apple Silicon. This is due to two known PyTorch MPS limitations:
> 1. **Kernel Size Limit:** High-resolution tensors (4K+) exceed the MPS driver's maximum dimensions, causing crashes.
> 2. **BFloat16 Incomplete:** CPU BFloat16 (for RAM savings) causes hangs due to unoptimized code paths.
>
> **Result:** CPU + Float32 uses ~80GB RAM for 12K output and is slow, but is the only stable option.
> **Outlook:** Apple's **Metal4** (macOS 16, expected 2026) promises native tensor support that should resolve these issues.

> [!TIP]
> **Overwrite Protection:** If the output file already exists, you'll be prompted before processing starts. Use `--force` to skip this check.

---
### AI Upscaling (Chained)
You can auto-upscale immediately after generation using `--upscale` (Stage 2).
This preserves your original generation and creates a separate upscaled file.

```bash
# Generate Image -> Auto-Upscale to 4K (2x)
python ai-media.py -i -p "Cyberpunk city" -o "city.png" --upscale

# Outputs:
# 1. city.png (Original 100%)
# 2. city_upscaled_2.0x.png (High-Res 200%)

# Custom Upscale Filename (-uof)
python ai-media.py -i -p "Cyberpunk city" --upscale -uof "city_poster.png"

# Video Upscaling
python ai-media.py -v -p "Robot dance" --upscale -uf 2.0
```

---
### Generation Modes

- `-i, --generate-image`: Generate an image
- `-v, --generate-video`: Generate a video
- `-a, --generate-audio`: Generate audio/music

### Examples

**Image Generation**
```bash
# Standard 720p image
python ai-media.py -i -p "Cyberpunk city at night" -o city.png -s 720p

# Auto-filename from prompt (creates "haunted-house.jpg")
python ai-media.py -i -p "Haunted house at dusk"

# Custom resolution (1920x1080) in a specific folder (auto-created)
python ai-media.py -i -p "Forest landscape" -o "./outputs/landscapes/forest.jpg" -s 1920x1080

# Advanced JSON size & Model selection
python ai-media.py -i -p "Portrait of a wizard" -o wizard.png --size "{w: 512, h: 768}" --image-model sd-1.5
```

**Video Generation**
```bash
# Standard 5s clip
python ai-media.py -v -p "A robot dancing" -l 5s -o robot.mp4

# Auto-filename from prompt (creates "dancing-robot.mp4")
python ai-media.py -v -p "Dancing robot in neon" -l 5s

# Image-to-Video (using input image + text action)
python ai-media.py -v -p "Camera pans left" -ii "./start_frame.png" -o animated.mp4

# Video with Audio (Generate Video, then Audio, then Merge)
python ai-media.py -v -p "Cyberpunk dancers" --audio-prompt "Heavy techno beat" -l 10s -o party.mp4

# Long video (1m 30s) using Zeroscope model
python ai-media.py -v -p "Drone flight over mountains" -l "{m:1, s:30}" -o flight.mp4 --video-model zeroscope
```

**Audio Generation**
```bash
# 30s MP3 clip
python ai-media.py -a -p "Smooth jazz saxophone" -l 30s -o jazz.mp3

# Auto-filename from prompt (creates "jazz-piano.mp3")
python ai-media.py -a -p "Jazz piano solo" -l 30s

# High-Quality WAV (48kHz, 24-bit)
python ai-media.py -a -p "Rainforest ambience" -l 1m -o rain.wav -m 48000 -b 24 --audio-model audioldm2

# Image-to-Audio (Generate sound matching an image)
python ai-media.py -a -p "Mystery theme" -ii "./haunted_house.jpg" -o mystery.mp3
```

**AI Upscaling**
```bash
# Standalone Image Upscale (input.jpg -> output_upscaled.png)
python ai-media.py --upscale-image input.jpg

# Custom Upscale Factor (e.g., 2x, 8x)
# Note: Factors > 4x use multi-pass upscaling (slower but necessary)
python ai-media.py --upscale-image input.jpg -uf 8x

# Upscale Video (Frame-by-Frame, slow but high quality)
python ai-media.py --upscale-video clip.mp4

# Chained Generation (Generate -> Upscale)
# Generates a 720p image, then immediately upscales it to 5K (4x)
python ai-media.py -i -p "Epic mountain" -s 720p --upscale -uf 4x

# High-Resolution Optimization (>4K)
# Native 4K/5K generation can be slow/unstable. To get 4K result faster:
# 1. Generate at optimized 3K (e.g. 2688x1536)
# 2. Auto-Upscale to recover resolution
# The script will PROACTIVELY suggest this if you request >6MP (4K+).
python ai-media.py -i -p "Detail test" -s 4k
# Follow the interactive prompt:
# 💡 Recommendation: Generate at optimized 3K... + Auto-Upscale...
# 🔄 Switch to optimized workflow? [Y/n]: Y
```


**Smart Format Handling**
```bash
# Implicit Format (Inferred from filename)
python ai-media.py -i -p "Cat" -o cat.png      # Generates PNG
python ai-media.py -i -p "Cat" -o cat.jpg      # Generates JPEG

# Implicit Extension (Inferred from format flag)
python ai-media.py -i -p "Cat" -o my_image -f png   # Auto-saves as "my_image.png"
```

### Common Options

| Option | Description |
| :--- | :--- |
| `-p, --prompt` | Text description of content to generate. |
| `-o, --output` | Output filename/path. **Optional**: auto-generated from first 2 words of prompt if omitted. |
| `-f, --format` | Explicit file format. **Image**: jpg, png (default: jpg). **Video**: mp4 (default: mp4). **Audio**: mp3, wav (default: mp3). |
| `--force` | Skip all confirmation prompts (overwrites existing files and ignores resource warnings). |
| `-s, --size` | Resolution. Supports "720p", "1080p", "4k", "8k", "HD", "1280x720", `{w:1280, h:720}`. Default: 720p. |
| `-l, --length` | Duration. Supports "15s", "1m", "1h30m", `{m:1, s:30}`. Default: 15s. |
| `-ii, --image-input` | Source image path for **Image-to-Video** generation. |
| `--npt, --no-performance-tracking` | Disable creating/updating `performance.json` and time estimates. [Read more](#performance-tracking). |
| `--unsafe` | Disable NSFW safety checker (reduces false positives). [Read more](#safety-checker). |

### Audio Options

| Option | Description |
| :--- | :--- |
| `-m, --sampling` | Sample rate in Hz (e.g. `44100`, `48k`, `32000`). Default: 32000. |
| `-b, --bit-depth` | Bit depth (16, 24, 32). Default: 16. |
| `-r, --bit-rate` | Target bitrate (e.g. `128k`, `320kbps`). |

## Models & Options

### Image Models (`--image-model`)

| Model | Code | Download | VRAM | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **SDXL Turbo** | `sdxl` | ~8GB (16GB on Mac) | ~8GB (~16GB on Mac) | **Default**. Fast, high quality. Uses float32 on Apple Silicon. |
| **SD 1.5** | `sd-1.5` | ~4GB | ~4GB | Lightweight, lower VRAM. ⚠️ NSFW filter issues on non-CUDA. |
| **Flux Schnell** | `flux` | ~24GB | ~12GB+ | High quality. 🔒 **Gated** (Requires Hugging Face Login). |
| **Flux Dev** | `flux-dev` | ~24GB | ~16GB+ | Professional creative work. 🔒 **Gated** (Login required). |

> [!NOTE]
> **Apple Silicon/MPS:** SDXL Turbo uses float32 precision on Mac to avoid black images (float16 produces NaN values in VAE). This doubles memory usage compared to NVIDIA/CUDA.
>
> **High Resolution (4K+):** For resolutions larger than 1536x1536 (e.g., 4K), the script automatically enables **VAE Tiling**. This processes the image in chunks to prevent "Out of Memory" errors, though generation will be slightly slower.

### Audio Models (`--audio-model`)

| Model | Code | Download | VRAM | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **MusicGen Small** | `musicgen-small` | ~2GB | ~4GB | **Default**. Quick music sketches. |
| **MusicGen Medium** | `musicgen-medium` | ~6GB | ~8GB | Better composition & fidelity. |
| **MusicGen Large** | `musicgen-large` | ~10GB | ~16GB | Highest quality music generation. |
| **AudioLDM 2** | `audioldm2` | ~4GB | ~8GB | Sound effects (SFX), foley, environmental. |

### Video Models (`--video-model`)

| Model | Code | Download | VRAM | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **ModelScope** | `ms-1.7b` | ~10GB | ~12GB | **Default**. General purpose short clips. |
| **Zeroscope** | `zeroscope` | ~4GB | ~8GB | Widescreen format, lower memory. |
| **CogVideoX** | `cogvideox` | ~15GB | ~24GB | High fidelity. **Supports Image-to-Video**. 🔒 **Gated**. |
| **Stable Video Diffusion** | `svd` | ~4GB | ~8GB | **I2V Only**. Industry standard for animating images. |

### Upscaling Models (Auto-selected based on factor)

| Model | Code | Download | VRAM | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **SD x2 Latent** | `upscaler_x2` | ~4GB | ~8GB | **Factors ≤ 2x**. Fast, preserves original style. |
| **SD x4 Upscaler** | `upscaler` | ~8GB | ~16GB | **Factors > 2x**. High detail, sharpens textures. |

### AI Upscaling Architecture

The script uses a **Dual-Model Approach** to balance speed and quality:

1.  **Small Boosts (≤ 2.0x)**: Uses the **Latent Upscaler (`sd-x2-latent-upscaler`)**.
    *   **Why?** It works in "latent space" (like the generator), making it fast and excellent at preserving the original image's style and composition without hallucinating weird details.
    *   *Perfect for: Sharpening 720p to 1080p, or 3K to 4K.*

2.  **Large Boosts (> 2.0x)**: Uses the **Super-Resolution Upscaler (`stable-diffusion-x4-upscaler`)**.
    *   **Why?** It generates brand new texture details that didn't exist before.
    *   **Supersampling**: If you ask for 3x (e.g. 720p -> 4K), the model internally generates a **4x** image (Supersampling) and then carefully downscales it to your exact target. This results in cleaner edges than generating at the lower resolution directly


## Supported Resolutions & Times

### Resolution Parsing (`-s` or `--size`)
The tool supports natural language and object-style inputs:
- **Presets**: `480p`, `576p`, `720p`, `900p`, `1080p`, `1440p`, `1k` ... `10k`, `HD`, `FHD`, `UHD`
- **Dimensions**: `1280x720`, `1024x1024`
- **Objects**: `{w: 800, h: 600}`, `{width: 1920, height: 1080}`

### Duration Parsing (`-l` or `--length`)
- **Strings**: `15s`, `1m`, `1h30m5s`
- **Objects**: `{m: 1, s: 30}`, `{hours: 1, minutes: 15}`
- **Numeric**: `30` (interpreted as seconds)

## Performance Tracking

The tool includes a smart performance tracking system designed to help you plan your work.

### ❓ Why track performance?
Generative AI tasks can vary wildly in duration depending on your specific hardware (GPU, RAM, CPU). By recording the execution time of your previous runs, `ai-media` calculates personalized **Time Estimates** for future jobs.
*Example*: If your machine takes 2 minutes to generate a 5-second video, the CLI will learn this and estimate ~4 minutes when you ask for a 10-second video.

### 🔒 Privacy & Data
**No personal information is stored.**
The `performance.json` file is strictly technical and local. It **never** records:
- ❌ File paths or filenames
- ❌ Prompts or content specifics
- ❌ User identity

It **only** records anonymous metrics:
- ✅ Model ID (e.g., `flux`)
- ✅ Device Type (e.g., `cuda`, `mps`)
- ✅ Resolution/Size
- ✅ Generation Duration
- ✅ Average CPU Usage (%)
- ✅ Average RAM Usage (GB)

### 🚫 Opting Out
If you prefer not to use this feature, you can completely disable the reading and writing of this file by using the `--npt` or `--no-performance-tracking` flag.

## Resource Safety Check

Before starting any generation task, the tool automatically checks your system's available RAM and VRAM against the requirements of your selected model and parameters.

### ⚠️ When You'll See a Warning
You'll be prompted if:
- **Insufficient RAM**: Available memory is below what the model typically needs
- **Insufficient VRAM**: GPU memory may not support the model
- **Resolution too high**: Requested resolution exceeds the model's recommended maximum
- **Duration too long**: Audio/video length exceeds safe generation limits

### 📋 Example Warning
```
⚠️  Resource Warning:

   • RAM: 6.2GB available, 12GB recommended
   • Resolution: 1920x1080 exceeds recommended max 1280x720

   Model: damo-vilab/text-to-video-ms-1.7b
   This job may cause slowdowns, swapping, or crashes.

   Continue anyway? [y/N]:
```

### 🔧 Options
- **y**: Proceed despite the warning
- **N** (default): Cancel and adjust parameters
- **--force**: Skip all confirmation prompts (overwrites existing files and ignores resource warnings).

## Safety Checker

**Image generation models only.** Video and audio models do not have safety checkers.

Image models include an NSFW safety checker that blocks potentially inappropriate content. However, this safety checker (especially on SD 1.5) is known to have **false positives** - blocking completely innocent prompts.

> [!WARNING]
> **Non-NVIDIA Hardware (Apple Silicon, CPU):** The NSFW safety checker model does not work correctly on non-CUDA hardware and produces false positives on almost all prompts. If you're on **Apple Silicon/MPS**, you will likely need to use `--unsafe` with SD 1.5.

### 🚫 False Positive Example
Certain prompts may trigger the filter unexpectedly, resulting in:
```
⚠️  Warning: Potential NSFW content detected.

The model's safety checker has blocked the image (returning a black frame).
👉 Please modify your prompt and try again.
💡 If your prompt is appropriate, try again with --unsafe to disable the safety checker.
```

### 🔓 Disabling the Safety Checker
If you're confident your prompt is appropriate and still getting blocked, use the `--unsafe` flag:
```bash
python ai-media.py -i -p "Haunted house at night" -o haunted.jpg --unsafe
```

> [!CAUTION]
> Using `--unsafe` disables all content filtering. You are responsible for ensuring your prompts and outputs comply with applicable laws and ethical guidelines.

## Troubleshooting

### ❌ Python Not Found / Module Errors
**Error**: `python: command not found` or `ModuleNotFoundError: No module named 'diffusers'`
**Cause**: The virtual environment is not activated in your current terminal session.
**Solution**:
- Activate the virtual environment for **every new terminal session**:
```bash
source venv/bin/activate
```
- You should see `(venv)` at the beginning of your prompt when activated.

### ❌ Resolution Error (Divisible by 8)
**Error**: `height and width have to be divisible by 8`
**Cause**: Deep learning models (UNets) process images in blocks of 8 pixels.
**Solution**:
- The script automatically offers to retry with valid dimensions (e.g. `340x333` → `344x336`).
- Press **'y'** when prompted to auto-fix and resume.

### ❌ Hardware Limitation (Invalid Buffer Size)
**Error**: `Invalid buffer size: ... GiB`
**Cause**: Attempting to generate extremely high resolutions (e.g., native 5K/8K) in a single pass.
- **Why**: "Attention" calculations scale quadratically. A 5K image (14.7MP) creates internal tensors larger than what a single GPU buffer can hold (often limited to ~4GB on MPS, regardless of total VRAM).
**Solution**:
- **Generate Lower, Upscale Later**: Generate at 4K (3840x2160) or 2K.
- **Wait for Updates**: Future versions may support "MultiDiffusion" (Tiled Generation) to bypass this limit.

### ❓ High RAM Usage (50GB+)
**Observation**: "I enabled VAE Tiling but it still uses ~50GB RAM for 4K."
**Explanation**:
- **VAE Tiling** only optimizes the *final* step (decoding Latents to Pixels), preventing a massive spike that would otherwise crash the system (saving ~10-20GB).
- **The UNet (Main Generation)** still requires holding the entire image context in memory during the process. For 4K/float32, utilizing ~50GB is normal and unavoidable without "Tiled Diffusion".

### ❓ Why do I see `alignment required` during upscaling?
**Message**: `ℹ️ Temporarily padding 1280x720 → 1280x768 (64px alignment required)`
**Cause**: The upscaler models require specific dimension alignments:
- **x2 latent upscaler**: Divisible by **64** (operates in latent space)
- **x4 upscaler**: Divisible by **8** (standard Stable Diffusion requirement)

**What happens**:
1. Your image is temporarily **padded** to the nearest 64-pixel boundary
2. Upscaling is performed on the padded image
3. The result is **cropped back** to your exact target dimensions
**Impact**: The final output matches your requested dimensions exactly. The padding is only used internally.

## Dependencies

This project uses the following open-source libraries:

| Dependency | Purpose | GitHub |
| :--- | :--- | :--- |
| [diffusers](https://github.com/huggingface/diffusers) | Image/Video generation pipelines | [huggingface/diffusers](https://github.com/huggingface/diffusers) |
| [transformers](https://github.com/huggingface/transformers) | Audio/Text generation pipelines | [huggingface/transformers](https://github.com/huggingface/transformers) |
| [PyTorch](https://github.com/pytorch/pytorch) | Deep learning framework & Hardware detection | [pytorch/pytorch](https://github.com/pytorch/pytorch) |
| [accelerate](https://github.com/huggingface/accelerate) | Optimization & large model handling | [huggingface/accelerate](https://github.com/huggingface/accelerate) |
| [FFmpeg](https://github.com/FFmpeg/FFmpeg) | Media processing & format conversion | [FFmpeg/FFmpeg](https://github.com/FFmpeg/FFmpeg) |
| [sentencepiece](https://github.com/google/sentencepiece) | Tokenization for text processing | [google/sentencepiece](https://github.com/google/sentencepiece) |
| [safetensors](https://github.com/huggingface/safetensors) | Safe model loading format | [huggingface/safetensors](https://github.com/huggingface/safetensors) |
| [scipy](https://github.com/scipy/scipy) | Audio processing mathematics | [scipy/scipy](https://github.com/scipy/scipy) |
| [opencv-python](https://github.com/opencv/opencv-python) | Video frame processing | [opencv/opencv](https://github.com/opencv/opencv) |

**AI Models used:**
- **Flux** by Black Forest Labs - [black-forest-labs/flux](https://github.com/black-forest-labs/flux)
- **Stable Diffusion XL** by Stability AI - [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models)
- **Stable Diffusion 1.5** by RunwayML - [runwayml/stable-diffusion](https://github.com/runwayml/stable-diffusion)
- **MusicGen** by Meta AI - [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
- **AudioLDM 2** by Haohe Liu etc. - [haoheliu/AudioLDM2](https://github.com/haoheliu/AudioLDM2)
- **ModelScope** by Alibaba - [modelscope/modelscope](https://github.com/modelscope/modelscope)
- **Zeroscope** by Cerspense - [cerspense/zeroscope](https://huggingface.co/cerspense/zeroscope_v2_576w)
- **CogVideoX** by THUDM - [THUDM/CogVideo](https://github.com/THUDM/CogVideo)
- **Stable Video Diffusion** by Stability AI - [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models)
- **Stable Diffusion x2 Latent Upscaler** by Stability AI - [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)
- **Stable Diffusion x4 Upscaler** by Stability AI - [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)


## Disclaimer

This tool is provided for **personal use only**. The project owner and contributors assume no responsibility or liability for how users choose to use this script or for any content generated by it. Users are solely responsible for ensuring their use of this tool complies with all applicable laws and regulations in their jurisdiction. 

Additionally, please be aware that AI models for image, video, and audio generation are probabilistic. While often highly accurate, results are **not guaranteed to be perfect** and may contain inaccuracies, omissions, or hallucinations. Manual review and editing of generated content is strongly recommended, especially for critical applications.

Project code was written with the help of [Antigravity](https://antigravity.google/) under human (developer) initiative on all the features, guidance, and improvements over many iterations (even before it was first published on GitHub).

## License

MIT License