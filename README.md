# AI-Media

Generate images, videos, and audio locally using state-of-the-art open source AI models. Transform and edit images with natural language instructions or remove backgrounds. Describe and analyze media content. Upscale existing media with or without AI. Convert between formats instantly. This tool wraps libraries like `diffusers`, `transformers`, and `FFmpeg` into a simple, unified command-line interface.

## Features

- 🔄 **Media Conversion** - Instantly convert images, videos, and audio between formats (no AI, uses PIL/FFmpeg).
- 🎨 **Image Generation** - Text-to-Image using models like Flux/SDXL (via `diffusers`).
- 🎬 **Video Generation** - **Text-to-Video**, **Image-to-Video**, and **Video-with-Audio** (automatic muxing with FFmpeg).
- 🎵 **Audio Generation** - **Text-to-Audio** (either instructional prompt with most models, or text to speech with multi language support and human speaker voices with the Bark model) and **Image-to-Audio** / **Video-to-Audio** (using Visual Captioning). Models: MusicGen, AudioLDM 2.
- 🪄 **Creative Image Transformations** - Edit images using natural language instructions (InstructPix2Pix) or remove backgrounds (RMBG-1.4). Supports style transfer (Anime, Oil Painting), content modification (features, age), and utility tasks (Background Removal, Silhouettes).
- 📈 **Upscaling** - Upscale images and videos using AI (Stable Diffusion x2/x4) or simple non-AI (Lanczos/FFmpeg). Supports custom factors and chained workflows.
- 📝 **Description Generation** - Generate a description for an image or video using models like Florence/BLIP (via `transformers`).
- 🖥️ **Interactive Menus** - Optional guided menu system with arrow key navigation for all features, when no parameters are provided to the main script. [See details](#interactive-mode).
- ⚙️ **Power User Controls**
    - Flexible resolution parsing (strings like "720p", "4k", "1920x1080", or objects like `{w:1920, h:1080}`)
    - Smart time parsing ("1h50m", "15s", `{m:2, s:30}`)
- 🚀 **Hardware Accelerated** - Auto-detects and optimizes for:
    - 🍏 **Apple Silicon** (MPS / Metal)
    - 🟢 **NVIDIA GPUs** (CUDA + Float16)
    - 💻 **CPU Performance Tracking** - To improve estimation accuracy, the script creates a `performance.json` file in its directory. This file is **local only** and never uploaded. To disable, use `--no-performance-tracking` (or `-npt`). 
    - [See details](#performance-tracking).

## Prerequisites

1.  **Python 3.10 - 3.12** (3.12 is recommended for improved performance)
2.  **FFmpeg** (Required for video generation, conversion, and proper playback)
    -   macOS: `brew install ffmpeg`
    -   Linux: `apt install ffmpeg`
    -   Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### Python Dependencies (installed via requirements.txt)

- **diffusers**: State-of-the-art Image & Video generation pipelines
- **transformers**: Audio generation & text processing models
- **torch**: Core deep learning framework & hardware acceleration (CUDA/MPS)
- **accelerate**: Optimization for efficient large model loading
- **opencv-python**: Video frame processing & manipulation
- **scipy**: Audio signal processing & file handling

### 🔐 Gated Models (Optional)
Some state-of-the-art models (like `FLUX.1`) require authentication (but are **free to use**):

> [!CAUTION]
> **Mac Users:** `FLUX.1` is extremely resource-intensive (~70GB+ RAM/Swap) and slow on Apple Silicon. It is **not recommended** for most Mac users.

1.  **The CLI will be installed as part of the installation process (requirements.txt)**
2.  Create a **Free** [Hugging Face Account](https://huggingface.co/join).
3.  **Accept model licenses**: Visit each model page and click **"Agree and access repository"** (one-time per model):
    | Model | Accept License |
    | :--- | :--- |
    | FLUX.1-schnell (`flux`) | [Accept License](https://huggingface.co/black-forest-labs/FLUX.1-schnell) |
    | FLUX.1-dev (`flux-dev`) | [Accept License](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
    | Stable Audio Open (`stable-audio`) | [Accept License](https://huggingface.co/stabilityai/stable-audio-open-1.0) |
4.  **Create an Access Token**: Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens) and create a new token:
    - **Quick option**: Select **"Read"** token type for simple read access to all repos.
    - **Fine-grained option**: Select **"Fine-grained"** and enable **"Read access to contents of all public gated repos you can access"** under Repositories.
5.  **Login**: Run `hf auth login` in your terminal, paste your Access Token, and answer **`n`** to "Add token as git credential?" (only needed for pushing to HF repos).

*Note: The default model `sd-1.5` is open and requires no login.*

---


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
    # Windows:
    .venv\Scripts\activate.bat
    # or for PowerShell:
    # if neededSet-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    .venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. [Optional] Login to Hugging Face (Free); See Gated Models (Optional) section above.
# Only required if you want to use Gated models like Flux.
hf auth login
# (Paste your Access Token when prompted. It is invisible.)
```

## Usage

The script `ai-media.py` is the main entry point.

### AI Upscaling (Standalone)
You can directly upscale existing images or videos without generating new ones.

```bash
# Upscale an image to 4K (assuming input is 1920x1080) with 2x factor
python ai-media.py -ui "my-image.jpg" -uf 2.0

# Upscale a video
python ai-media.py -uv "my-video.mp4" -uf 2.0

# Simple upscale (no AI - uses Lanczos/FFmpeg, very fast)
python ai-media.py -ui "photo.jpg" -uf 4 -su
python ai-media.py -uv "clip.mp4" -uf 2 -su
```

| Argument | Description | Default |
| :--- | :--- | :--- |
| `-ui`, `--upscale-image` | Path to the image file to upscale. | `None` |
| `-uv`, `--upscale-video` | Path to the video file to upscale. | `None` |
| `-uof`, `--upscaled-output-file` | Custom filename for the upscaled output. | Auto: `name_upscaled_{factor}x.ext` |
| `-uf`, `--upscale-factor` | Multiplier for resolution (e.g., `1.5`, `2.0`, `4.0`). | `2.0` |
| `-us`, `--upscale-strength` | Noise strength (`0.0`-`1.0`). Higher values allow the model to generate more texture/detail but may diverge from the original. **x4 upscaler only** (ignored for x2 latent). | `0.0` |
| `-su`, `--simple-upscale` | Use simple non-AI upscaling (PIL Lanczos for images, FFmpeg for videos). Very fast, preserves original quality. | `False` |

> [!NOTE]
> **Resource Safety Check:** Before starting, the script calculates the target resolution (e.g., 8K = 33MP) and estimated RAM usage. If it detects a risk of massive swapping or system freeze ("Billboard Sizing"), it will warn you and ask for confirmation. Use `--force` to bypass this.

> [!IMPORTANT]
> **MacOS/Apple Silicon - Upscaling:** Enforced to run on **CPU** (Float32). This is due to PyTorch MPS limitations:
> 1. **Kernel Size Limit:** High-resolution tensors (4K+) exceed the MPS driver's maximum dimensions, causing crashes.
> 2. **BFloat16 Incomplete:** CPU BFloat16 causes hangs due to unoptimized code paths.
>
> **Result:** CPU + Float32 uses ~80GB RAM for 12K output and is slow, but is the only stable option.

> [!TIP]
> **Overwrite Protection:** If the output file already exists, you'll be prompted before processing starts. Use `--force` to skip this check.

---

### 🔄 Media Conversion (No AI)

Convert images, videos, and audio between formats using PIL (images) or FFmpeg (all formats).

```bash
# Image Conversion (PIL default, FFmpeg optional)
python ai-media.py -ci input.gif -cit png
python ai-media.py -ci input.png -cit output.webp --convert-image-engine ffmpeg

# Video Conversion (FFmpeg)
python ai-media.py -cv input.mov -cvt mp4
python ai-media.py -cv clip.avi -cvt output/converted.webm

# Audio Conversion (FFmpeg)
python ai-media.py -ca input.wav -cat mp3
python ai-media.py -ca song.flac -cat output/song.ogg
```

| Argument | Description |
| :--- | :--- |
| `-ci`, `--convert-image` | Source image path. |
| `-cit`, `--convert-image-to` | Target format/path (`png`, `.webp`, `out.jpg`). |
| `-cv`, `--convert-video` | Source video path. |
| `-cvt`, `--convert-video-to` | Target format/path (`mp4`, `.webm`, `out.avi`). |
| `-ca`, `--convert-audio` | Source audio path. |
| `-cat`, `--convert-audio-to` | Target format/path (`mp3`, `.flac`, `out.ogg`). |
| `--convert-image-engine` | Image engine: `pil` (default) or `ffmpeg`. |

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

**Description Generation (Captioning)**
Analyze images or videos to generate text descriptions (captions).
```bash
# Describe an image (prints to console + saves to text file)
python ai-media.py -gd input.jpg

# Use a specific model (florence or blip)
python ai-media.py -gd input.jpg -cm blip

# Describe a video (analyzes 10 frames + summary)
python ai-media.py -gd clip.mp4

# Custom output filename
python ai-media.py -gd input.jpg -o my_caption.txt
```

---
### 🎨 Creative Image Transformation / Editing
Edit existing images using AI instructions or remove backgrounds.

| Argument | Description |
| :--- | :--- |
| `-ti`, `--transform-image` | Path to the image file to transform. |
| `-p`, `--prompt` | Edit instruction (works for standalone transformations). |
| `-tp`, `--transform-prompt` | Edit instruction for chaining with generation (e.g., `-i -p "..." -ti file -tp "..."`). |
| `--remove-background`, `-rb` | Remove background (outputs transparent PNG). |
| `--silhouette` | Create a black silhouette (requires `--remove-background`). |
| `--image-guidance` | Image guidance scale (default: `1.5`). Higher = closer to original structure. |

> [!NOTE]
> **`-p` vs `-tp`**: For standalone transformations (`-ti` only), use `-p`. When chaining generation (`-i`) with transformation (`-ti`), use `-p` for the **generation prompt** and `-tp` for the **edit instruction**.

**1. Instructional Editing (InstructPix2Pix)**
Change style, remove objects, or modify content using natural language.
```bash
# Turn into Anime
python ai-media.py -ti photo.jpg -tp "Make it look like an anime drawing"

# Remove a beard
python ai-media.py -ti face.jpg -tp "Remove the beard"

# Change style to Oil Painting
python ai-media.py -ti landscape.jpg -tp "Make it look like an oil painting by Van Gogh"
```

**2. Background Removal (RMBG-1.4)**
Create transparent PNGs or silhouettes.
```bash
# Remove Background (Transparent PNG)
python ai-media.py -ti photo.jpg --remove-background

# Create a Silhouette (Black Subject, White Background)
python ai-media.py -ti dancer.jpg --remove-background --silhouette
```

**3. Transformation Recipe Book 🪄**
Here are prompt examples for common editing tasks.

| Goal | Command Pattern |
| :--- | :--- |
| **Styles** | |
| Anime / Manga | `-tp "Turn him into an anime character"` |
| Disney / Pixar | `-tp "Make it look like a 3D Pixar character"` |
| Studio Ghibli | `-tp "Make it look like a Studio Ghibli movie"` |
| Oil Painting | `-tp "Make it look like an oil painting"` |
| Watercolor | `-tp "Turn this into a watercolor painting"` |
| Pencil Sketch | `-tp "Turn this into a pencil sketch"` |
| Cartoon | `-tp "Turn this into a flat cartoon"` |
| Coloring Page | `-tp "Make it a black and white coloring page"` |
| Sticker | `-tp "Turn this into a sticker with a white outline"` |
| **Photo Manipulations** | |
| Remove Beard | `-tp "Remove the beard"` |
| Change Hairstyle | `-tp "Give him a mohawk hairstyle"` |
| Facial Expressions | `-tp "Make him smile"`, `-tp "Make her look surprised"` |
| Age / Baby | `-tp "Make him look like a baby"` |
| Caricature | `-tp "Turn this into a funny caricature"` |
| Recolor | `-tp "Change the red dress to blue"` |
| Colorize B&W | `-tp "Colorize this photo"` |
| Sketch to Image | `-tp "Turn this sketch into a photo of an apple"` |
| **Removal** | |
| Background | `--remove-background` (No prompt needed) |
| Silhouette | `--remove-background --silhouette` |
| Text / Objects | `-tp "Remove the text"`, `-tp "Remove the cup"` (Experimental) |

**4. Chaining Transformations 🔗**
You can mix commands! The tool automatically executes them in the correct order: **Edit First → Remove Background Second**. This preserves transparency.

```bash
# 1. Edit Style -> Remove Background
# Result: A transparent PNG of the anime character
python ai-media.py -ti photo.jpg -tp "Make it anime" --remove-background

# 2. Modify Subject -> Create Silhouette
# Result: A black silhouette of the modified subject (e.g. adding a hat)
python ai-media.py -ti photo.jpg -tp "Put a hat on him" --remove-background --silhouette
```

**5. Chaining Generations and Transformations 🔗**
Chain **Image Generation** with **Transformations** in a single command. Use `-ti` without a filename to automatically use the generated output.

```bash
# Simplified syntax: -ti auto-uses generated output
python ai-media.py -i -p "Portrait of a knight" -ti -tp "Make him hold a sword"

# With --remove-background (triple chain!)
python ai-media.py -i -p "Photo of a cat" -ti -tp "Make it anime" --remove-background

# Create silhouette from generated image
python ai-media.py -i -p "Dancer on stage" -ti -tp "Add dramatic lighting" --remove-background --silhouette

# Explicit filename (traditional syntax still works)
python ai-media.py -i -p "A knight" -o knight.png -ti knight.png -tp "Add sword"
```

> [!TIP]
> Use `-p` for the **generation** prompt and `-tp` for the **edit** instruction. This allows completely different prompts for each stage.

**6. Advanced Options**
```bash
# Custom Output Filename
python ai-media.py -ti photo.jpg -tp "Anime" -o "anime_version.png"

# Save to Subfolder (Dir is auto-created)
python ai-media.py -ti photo.jpg -tp "Anime" -o "edits/versions/anime_v1.png"

# Control guidance (Higher = Stick closer to original structure)
python ai-media.py -ti photo.jpg -tp "Cyborg" --image-guidance 1.8
```

---
### Generation Modes

- `-i, --generate-image`: Generate an image
- `-v, --generate-video`: Generate a video
- `-a, --generate-audio`: Generate audio/music
- `-gd, --generate-description`: Describe an image or video

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

# Different output formats (auto-detected from extension)
python ai-media.py -i -p "Logo design" -o logo.png           # PNG (lossless, transparency)
python ai-media.py -i -p "Web banner" -o banner.webp         # WebP (modern, small size)
python ai-media.py -i -p "Animation frame" -o frame.gif      # GIF
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

# Different output formats (auto-detected from extension)
python ai-media.py -v -p "Nature scene" -l 5s -o nature.mkv    # MKV (H.264)
python ai-media.py -v -p "Web clip" -l 5s -o clip.webm         # WebM (VP9)
python ai-media.py -v -p "Legacy format" -l 5s -o old.avi      # AVI (MPEG4)
python ai-media.py -v -p "Windows format" -l 5s -o win.wmv     # WMV
```

**Audio Generation**
```bash
# 30s MP3 clip
python ai-media.py -a -p "Smooth jazz saxophone" -l 30s -o jazz.mp3

# Auto-filename from prompt (creates "jazz-piano.mp3")
python ai-media.py -a -p "Jazz piano solo" -l 30s

# High-Quality WAV (48kHz, 24-bit)
python ai-media.py -a -p "Rainforest ambience" -l 1m -o rain.wav -m 48000 -b 24 --audio-model audioldm2

# Image-to-Audio plus Prompt (Generate sound matching an image)
python ai-media.py -a -p "Mystery theme" -ii "./haunted_house.jpg" -o mystery.mp3

# Image-to-Audio (Auto-Caption: No prompt needed)
python ai-media.py -a -ii "./beach.jpg" -o beach_sounds.mp3

# Video-to-Audio (Auto-Caption Video Frames + Audio Gen)
python ai-media.py -a -ii "./clip.mp4" -l 10s -o soundcheck.mp3

# Image-to-Audio plus Prompt with specific caption model (BLIP)
python ai-media.py -a -p "Mystery theme" -ii "./haunted.jpg" -cm blip

# Different output formats (auto-detected from extension)
python ai-media.py -a -p "Epic orchestra" -l 30s -o epic.flac   # FLAC (lossless)
python ai-media.py -a -p "Game music" -l 30s -o game.ogg        # OGG Vorbis
python ai-media.py -a -p "Podcast intro" -l 10s -o intro.aac    # AAC
```

**AI Upscaling**
```bash
# Standalone Image Upscale (input.jpg -> output_upscaled.png)
python ai-media.py -ui input.jpg

# Default (faithful upscaling, less artifacts)
python ai-media.py -ui original.jpg -uf 4.0

# If you want MORE detail generation (real photos, very low-res inputs)
python ai-media.py -ui photo.jpg -uf 4.0 -us 0.3

# Maximum creative freedom (may diverge from original)
python ai-media.py -ui input.jpg -uf 4.0 -us 0.8

# Custom Upscale Factor (e.g., 2x, 6x, 8x)
# Smart multi-stage: 6x = 4x AI + 1.5x Lanczos, 8x = 4x AI + 2x AI
python ai-media.py -ui input.jpg -uf 6x

# Upscale Video (Frame-by-Frame, slow but high quality)
python ai-media.py -uv clip.mp4

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
| `-otn, --orientation` | `landscape` (default), `portrait`, or `square`. Portrait swaps w/h. Square effectively crops/forces 1:1 aspect ratio using the smaller dimension. |
| `-l, --length` | Duration. Supports "15s", "1m", "1h30m", `{m:1, s:30}`. Default: 15s. |
| `-ii, --image-input` | Source image path for **Image-to-Video** generation. |
| `-npt, --no-performance-tracking` | Disable creating/updating `performance.json` and time estimates. [Read more](#performance-tracking). |
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
| **Flux Schnell** | `flux` | ~33GB | ~12GB+ (~70GB on Mac) | High quality. 🔒 **Gated**. **⚠️ Impractical on Mac (Slow)**. |
| **Flux Dev** | `flux-dev` | ~33GB | ~16GB+ (~80GB on Mac) | Professional creative work. 🔒 **Gated**. **⚠️ Impractical on Mac**. |

> [!NOTE]
> **Apple Silicon/MPS:** SDXL Turbo uses float32 precision on Mac to avoid black images (float16 produces NaN values in VAE). This doubles memory usage compared to NVIDIA/CUDA.
>
> **High Resolution (4K+):** For resolutions larger than 1536x1536 (e.g., 4K), the script automatically enables **VAE Tiling**. This processes the image in chunks to prevent "Out of Memory" errors, though generation will be slightly slower.

### Audio Models (`--audio-model`)

| Model | Code | Download | VRAM | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **MusicGen Small** | `musicgen-small` | ~2GB | ~4GB | Fast, good for music sketches. |
| **MusicGen Medium** | `musicgen-medium` | ~6GB | ~8GB | **Default**. Better composition & fidelity. |
| **MusicGen Large** | `musicgen-large` | ~10GB | ~16GB | Highest quality music generation. |
| **AudioLDM 2** | `audioldm2` | ~4GB | ~8GB | Sound effects (SFX), foley, environmental. |
| **Stable Audio** | `stable-audio` | ~10GB | ~16GB | 🔒 **Gated**. Best for Sound Effects (SFX), Drums, Ambient. |
| **Bark** | `bark` | ~4GB | ~12GB | Speech (TTS) & creative audio. Transformer-based. |


### Bark Configuration (`--audio-model bark`)

Bark is a transformer-based model that can generate highly realistic speech as well as other audio (music, background noise, etc.).

**1. Special Tokens / Sound Effects**
To generate non-speech audio, use these tags in your prompt:
*   `[laughter]`, `[cheers]`, `[music]`, `[sighs]`, `[gasps]`, `[clears throat]`
*   `—` or `...` (hesitations)
*   `♪` (wrap lyrics for singing, e.g. `♪ Hello World ♪`)

> [!TIP]
> **Token Reliability**: These sound effects are probabilistic and may not work with every voice or seed.
> *   **Try different voices**: Some speakers "laugh" better than others.
> *   **Context matters**: A prompt like *"That was funny! [laughter]"* works better than just `[laughter]`.
> *   **Singing**: Lyrics wrapping `♪` works best with short, rhythmic lines.

**2. Voice Presets (`--voice-preset`)**
You can change the speaker using the `--voice-preset` flag (default: `v2/en_speaker_6`).
*   **Format**: `v2/{lang}_speaker_{0-9}`
*   **Languages**: `en` (English), `fr` (French), `de` (German), `es` (Spanish), `it` (Italian), `ja` (Japanese), `zh` (Chinese), `pt` (Portuguese), `ru` (Russian).
*   **Reference**: [Bark Speaker Library (Audio Samples)](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc8cd1ed101043facc93a945395850fb)

> **Example**: `python ai-media.py -a -am bark -p "♪ In the jungle ♪ [laughter]" --voice-preset v2/it_speaker_2`

**3. Auto-Chunking & Unlimited Length** ♾️
By default, the Bark model can only generate ~14 seconds of audio per pass. This script includes an **automatic long-form generation** feature.
*   **Triggers**: This mode activates automatically if your text is long (>150 characters) or if you explicitly set a long duration (e.g. `--length 20s`).
*   **Audio Length**: The final audio length depends **entirely on your text**. (The `--length` flag effectively serves as a "force split" switch for Bark).
*   **Process**: The script splits your text into sentences and generates them in independent, stable chunks to ensure voice consistency.
*   **Usage**: Just provide a long text prompt.
    *   `python ai-media.py -a -am bark -p "This is a very long story..."`
    *   `python ai-media.py -a -am bark --voice-preset v2/en_speaker_6 -p "This is the first sentence. And this is the second one! Now we can go on forever without the model cutting us off. Like I am continuing here for a long long time [laughter]. Oh no [gasp], why did I do that!"`



### Video Models (`--video-model`)

| Model | Code | Resolution | Download | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Zeroscope** | `zeroscope` | 576×320 | ~4GB | **Default**. Fast, no watermarks. |
| **ModelScope** | `ms-1.7b` | Any | ~10GB | General purpose (has watermark issues). |
| **CogVideoX** | `cogvideox` | Any | ~22GB | High fidelity. **WARNING: Impractical on Mac** (~50GB+ RAM). |
| **Stable Video Diffusion** | `svd` | Any | ~4GB | **I2V Only**. ⚠️ *Very slow on Apple Silicon (CPU only).* |

> [!WARNING]
> **Watermarks in Output:** Some models (especially `ms-1.7b`) may produce videos with Shutterstock watermarks. This is because these open-source research models were trained on datasets that included watermarked stock footage. The model learned to reproduce the watermark as part of the visual pattern. This is baked into the model weights.

> [!IMPORTANT]
> **MacOS/Apple Silicon - Video Generation:** Text-to-Video models use **Float32** on MPS (Metal). Float16 produces corrupted/black frames.

> [!NOTE]
> **FFmpeg Re-encoding:** Generated videos are automatically re-encoded with FFmpeg (`libx264` + `yuv420p`) for universal playback. The raw output from `diffusers` uses a codec that macOS Finder/QuickTime cannot preview (shows green frames), but the re-encoded version works in all players and displays proper thumbnails.

### Creative Transformation / Editing Models (`-ti`)

| Model | Code | Download | VRAM | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **InstructPix2Pix** | `instruct-pix2pix` | ~4GB | ~8GB (High Precision) | Instructional image editing (e.g., "Make it anime"). |
| **RMBG-1.4** | `remove-bg` | ~0.2GB | ~2GB | Background removal and silhouette creation. |

### Upscaling Models (Auto-selected based on factor)

| Model | Code | Download | VRAM | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **SD x2 Latent** | `upscaler_x2` | ~4GB | ~8GB | **Factors ≤ 2x**. Fast, preserves original style. |
| **SD x4 Upscaler** | `upscaler` | ~8GB | ~16GB | **Factors > 2x**. High detail, sharpens textures. |

### Caption Models (`--caption-model` or `-cm`)

| Model | Code | Size | Best For |
| :--- | :--- | :--- | :--- |
| **Florence-2 Large** | `florence` | ~1.5GB | **Default**. SOTA details, rich descriptions, "seeing" the scene. |
| **BLIP Large** | `blip` | ~0.9GB | **Legacy**. Simple, concise captions. Faster but less detailed. |

### AI Upscaling Architecture

The script uses a **Smart Multi-Stage Approach** to balance speed and quality:

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

## Interactive Mode

The interactive mode offers a guided menu system for all features. It runs automatically if no arguments are provided, or explicitly via `--interactive`.

```bash
# Run interactive menu
python ai-media.py
# OR
python ai-media.py --interactive
```

### Fast Jump Points

You can jump directly to specific submenus or models using shortcut paths with `--interactive`:

| Menu # | Task | Jump Point | Description |
| :--- | :--- | :--- | :--- |
| `1` | **Image** | `image` | Image Menu |
| `1/1` | | `image/sdxl` | SDXL Turbo (Fast) |
| `1/2` | | `image/sd15` | SD 1.5 (Regular) |
| `1/3` | | `image/flux` | Flux Schnell |
| `1/4` | | `image/flux-dev` | Flux Dev |
| `2` | **Video** | `video` | Video Menu |
| `2/1` | | `video/zeroscope` | Zeroscope (No Watermark) |
| `2/2` | | `video/modelscope` | ModelScope (General) |
| `2/3` | | `video/cogvideox` | CogVideoX |
| `2/4` | | `video/svd` | Stable Video Diffusion |
| `3` | **Audio** | `audio` | Audio Menu |
| `3/1` | | `audio/musicgen` | MusicGen Medium |
| `3/2` | | `audio/musicgen-small` | MusicGen Small (Fast) |
| `3/3` | | `audio/musicgen-large` | MusicGen Large (Quality) |
| `3/4` | | `audio/audioldm2` | AudioLDM2 (SFX) |
| `3/5` | | `audio/bark` | Bark (TTS) |
| `4` | **Edit** | `transform` | Transform Menu |
| `4/1` | | `transform/edit` | Creative Edit |
| `4/2` | | `transform/rembg` | Background Removal |
| `4/3` | | `transform/silhouette` | Silhouette |
| `5` | **Other** | `upscale` | Upscale Menu |
| `6` | | `convert` | Convert Menu |
| `7` | | `caption` | Caption Menu |
| `8` | | `sysinfo` | System Information |

```bash
python ai-media.py --interactive "image/sdxl"
python ai-media.py --interactive "audio/bark"
python ai-media.py --interactive 8
python ai-media.py --interactive "4/2"
python ai-media.py --interactive 3/5
```


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
If you prefer not to use this feature, you can completely disable the reading and writing of this file by using the `-npt` or `--no-performance-tracking` flag.

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

**Image generation and editing models only.** Video and audio models do not have safety checkers.

Image generation (SDXL/Flux/SD1.5) and **InstructPix2Pix Editing** models include an NSFW safety checker. This checker is known to have **false positives**, especially on non-CUDA hardware.

> [!WARNING]
> **Non-NVIDIA Hardware (Apple Silicon, CPU):** The NSFW safety checker model frequently produces false positives (black images) on Apple Silicon/MPS. This most commonly affects **Stable Diffusion 1.5** (`-i`) and **InstructPix2Pix** (`-ti`). If you encounter black outputs, you will likely need to use `--unsafe`.

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

### ❓ Duplicate/Cloned Faces in Portraits
**Symptom**: When generating a portrait of "one person", you get two identical or near-identical faces.
**Cause**: SDXL-Turbo on wide aspect ratios (like 1280x720) tends to fill horizontal space with duplicates. The model's training data includes many multi-person shots at these ratios.
**Solution**: Use `-otn portrait` to swap to vertical orientation:
```bash
python ai-media.py -i -p "Portrait of a woman" -otn portrait
# Generates at 720x1280 instead of 1280x720
```

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

### ❌ Windows: NVIDIA GPU Not Detected (Using CPU)
**Symptom**: You have an NVIDIA GPU with CUDA support, but the script shows:
```
💻 Using CPU (Slow): CUDA or MPS not detected (or torch missing)
```
**Cause**: Mismatch between the system-installed CUDA version and the PyTorch bundled CUDA libraries. The default `pip install torch` may install a CPU-only or incompatible CUDA version.
**Solution**: Reinstall PyTorch with the correct CUDA version for your system:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```
> [!TIP]
> Replace `cu130` with your CUDA version (e.g., `cu118` for CUDA 11.8, `cu121` for CUDA 12.1). Check your CUDA version with `nvcc --version` or `nvidia-smi`.

After reinstalling, you should see:
```
🚀 Detected NVIDIA GPU: Using CUDA
```

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
| [timm](https://github.com/huggingface/pytorch-image-models) | Image models (Required for Florence-2) | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) |
| [einops](https://github.com/arogozhnikov/einops) | Tensor operations (Required for Florence-2) | [arogozhnikov/einops](https://github.com/arogozhnikov/einops) |

**AI Models used:**
- **Flux** by Black Forest Labs - [black-forest-labs/flux](https://github.com/black-forest-labs/flux)
- **Stable Diffusion XL** by Stability AI - [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models)
- **Stable Diffusion 1.5** by RunwayML - [runwayml/stable-diffusion](https://github.com/runwayml/stable-diffusion)
- **MusicGen** by Meta AI - [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
- **AudioLDM 2** by Haohe Liu etc. - [haoheliu/AudioLDM2](https://github.com/haoheliu/AudioLDM2)
- **Stable Audio Open** by Stability AI - [stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0)
- **Bark** by Suno - [suno-ai/bark](https://github.com/suno-ai/bark)
- **ModelScope** by Alibaba - [modelscope/modelscope](https://github.com/modelscope/modelscope)
- **Zeroscope** by Cerspense - [cerspense/zeroscope](https://huggingface.co/cerspense/zeroscope_v2_576w)
- **CogVideoX** by THUDM - [THUDM/CogVideo](https://github.com/THUDM/CogVideo)
- **Stable Video Diffusion** by Stability AI - [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models)
- **Stable Diffusion x2 Latent Upscaler** by Stability AI - [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)
- **Stable Diffusion x4 Upscaler** by Stability AI - [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)
- **Florence-2** by Microsoft - [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large)
- **BLIP** by Salesforce - [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large)
- **InstructPix2Pix** by Tim Brooks et al. - [timbrooks/instruct-pix2pix](https://github.com/timathy/instruct-pix2pix)
- **RMBG-1.4** by BRIA AI - [briaai/RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4)

---

## Testing (Internal)

This project includes an automated test suite for development and verification. It's primarily for internal use, but if you want to see everything in action, you're welcome to run it.

```bash
# Run tests (quiet mode)
python ai-media.py --test

# Run tests with full output
python ai-media.py --test-verbose
```

| File/Folder | Description |
| :--- | :--- |
| `testing.json` | Test configurations (commands, expected outputs) |
| `testData/inputs/` | Sample input files for tests |
| `testData/outputs/` | Generated outputs (git-ignored) |

> [!WARNING]
> - This may take a **long time** (30+ minutes)
> - Uses significant system resources (CPU, RAM, GPU)
> - Will download **all models** if not already cached (2-30GB each)
> - Press `CTRL+C` at any time to interrupt

## Disclaimer

This tool is provided for **personal use only**. The project owner and contributors assume no responsibility or liability for how users choose to use this script or for any content generated by it. Users are solely responsible for ensuring their use of this tool complies with all applicable laws and regulations in their jurisdiction. 

Additionally, please be aware that AI models for image, video, and audio generation are probabilistic. While often highly accurate, results are **not guaranteed to be perfect** and may contain inaccuracies, omissions, or hallucinations. Manual review and editing of generated content is strongly recommended, especially for critical applications.

Project code was written with the help of [Antigravity](https://antigravity.google/) under human (developer) initiative on all the features, guidance, and improvements over many iterations (even before it was first published on GitHub).

## License

MIT License