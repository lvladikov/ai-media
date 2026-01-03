# AI-Media

Generate images, videos, and audio locally using state-of-the-art open source AI models. Write articles, chat interactively, and generate code—all powered by local LLMs running entirely on your machine. Optionally enable live web search for deep research and real-time context in chat. Transform and edit images with natural language instructions or remove backgrounds. Describe and analyze media content. Upscale existing media with or without AI. Convert media and documents between formats instantly.

AI-Media provides **three ways to interact** with these powerful local models:
- **CLI** — Direct command-line execution for scripting and automation
- **Interactive Mode** — Guided Python menus with arrow key navigation (and mouse support on some terminals)
- **Web Client & Desktop App** — Full-featured React Web Client running in browser or as an Electron app, with real-time resource monitoring and job management

Under the hood, it wraps libraries like `diffusers`, `transformers`, and `FFmpeg` into a unified Python wrapper. Unit and integration tests verify all functionality.

Designed for personal use and experimentation, AI-Media demonstrates that state-of-the-art AI models can run locally on consumer hardware. While video generation is resource-intensive, text-based models like DeepSeek and Llama provide an excellent starting point for chat, coding, and article generation.

![Infographic created with NotebookLM](screenshots/infographic.png) 

## Features

- 🎨 **Image Generation** - **Text-to-Image** using models like Flux/SDXL (via `diffusers`). See [Image Options](docs/image-generation.md#options), [Examples](docs/image-generation.md#examples) and [Models](docs/image-generation.md#models).
- 🎬 **Video Generation** - **Text-to-Video**, **Image-to-Video**, and **Text/Image + Audio (prompt) to Video**. See [Video Options](docs/video-generation.md#options), [Examples](docs/video-generation.md#examples) and [Models](docs/video-generation.md#models).
- 🎵 **Audio Generation** - **Text-to-Audio** (either instructional prompt with most models, or text to speech with multi language support and human speaker voices with the Bark model) and **Image-to-Audio** / **Video-to-Audio** (using Visual Captioning + Audio Generation). Models: MusicGen, AudioLDM 2. See [Audio Options](docs/audio-generation.md#options), [Examples](docs/audio-generation.md#examples) and [Models](docs/audio-generation.md#models).
- 📝 **Description Generation** - **Generate a description** for an image or video (sample 10 evenly picked frames used) using models like Florence/BLIP (via `transformers`). See [Description Options](docs/description-generation.md#options), [Examples](docs/description-generation.md#examples) and [Models](docs/description-generation.md#models). If you are interested in producing a subtitle file based on Audio or Video using AI, see my [auto-subtitles project](https://github.com/lvladikov/auto-subtitles).
- ✍️ **Article/Research/Code Generation** - Generate comprehensive **Articles** (offline), perform **Deep Research** (online search + summary), and generate **Code** for scripts, including multi file and folder projects (offline). Includes an interactive **Chat** session that runs on **fully offline models** but can dynamically pull live web content via the `/search` command. Chat can **read, discuss, generate, and save content (code or otherwise)**. See [Article Options](docs/article-generation.md#article--text-options), [Code Options](docs/article-generation.md#code-options), [Examples](docs/article-generation.md#examples) and [Models](docs/article-generation.md#text-models).
- 🪄 **Creative Image Transformations** - **Edit images using natural language instructions** (InstructPix2Pix) or **remove backgrounds** (RMBG-1.4). Supports style transfer (Anime, Oil Painting), content modification (features, age), and utility tasks (Background Removal, Silhouettes). See [Transform Options](docs/creative-transformations.md#options), [Examples](docs/creative-transformations.md#examples) and [Models](docs/creative-transformations.md#models).
- 🔄 **Media Conversion** - **Instantly convert** images, videos, and audio between formats (no AI, uses PIL/FFmpeg). See [Media Conversion Options](docs/media-conversion.md#media-conversion-options) and [Examples](docs/media-conversion.md#examples).
- 📄 **Document Conversion** - **Convert documents** between formats (MD, HTML, PDF, DOCX, RTF, TXT, JSON). See [Document Conversion Options](docs/media-conversion.md#document-conversion-options).
- 📈 **Upscaling** - **Upscale** images and videos using AI (Real-ESRGAN for fast/faithful, Stable Diffusion for creative) or simple non-AI (Lanczos/FFmpeg). Supports any resolution (8K+ auto-encodes as HEVC). See [Upscaling Options](docs/upscaling.md#options), [Examples](docs/upscaling.md#examples) and [Models](docs/upscaling.md#models).
- 🖥️ **Interactive Mode** - Optional **guided menu system** with arrow key navigation for all features, when no parameters are provided to the main script. [See details](#interactive-mode).
- 🌐 **Web Client & Desktop App** - Browser-based interface and Electron desktop app for all features with real-time resource monitoring. Launch with `python ai-media.py --serve` (both clients), `--serve-web-only-client`, or `--serve-electron-dev-only-client`. See [Web Client](docs/web-client.md).
- 🧪 **Testing** - **Unit and integration tests** to verify the functionality of the tool. See [Testing](docs/testing.md).
- ⚙️ **Power User Controls**
    - Flexible resolution parsing (strings like "720p", "4k", "1920x1080", or objects like `{w:1920, h:1080}`)
    - Smart time parsing ("1h50m", "15s", `{m:2, s:30}`)
- 🚀 **Hardware Accelerated** - Auto-detects and optimizes for:
    - 🍏 **Apple Silicon** (MPS / Metal)
    - 🟢 **NVIDIA GPUs** (CUDA + BFloat16 on RTX 30xx+ / Float16 on older)
    - 🟡 **Codec Analysis Tool** - Verify your system's hardware and software encoding limits. See [Codec Analysis Tool](docs/testing.md#codec-analysis-tool).
    - 💻 **Performance Tracking** - To improve estimation accuracy, the script creates a `performance.json` file in its directory. This file is **local only**. See [Performance Tracking](docs/performance-tracking.md).

> [!TIP]
> **BFloat16 Precision (NVIDIA RTX 30xx+):** On supported CUDA devices (Ampere architecture and newer), this script automatically uses `bfloat16` precision for all AI operations. BFloat16 offers the same memory efficiency as float16 but with better numerical stability (larger exponent range prevents overflow/underflow). The script auto-detects bf16 support via PyTorch and falls back to float16 on older GPUs.


> [!NOTE]
> **Performance Reality (2025):** NVIDIA GPUs with CUDA currently deliver the fastest AI processing due to a mature ecosystem refined since 2006. However, **with optimizations in this script, all operations run successfully on Apple Silicon/MPS—just behind NVIDIA performance**. See Mac-specific tweaks in [Image Models](docs/image-generation.md#models), [Video Models](docs/video-generation.md#models), and [Upscaling](docs/upscaling.md#options). Currently, bfloat16 support on MPS is incomplete (causes hangs), so this script enforces float32 precision—doubling memory usage but ensuring stability. Future bfloat16 improvements in PyTorch and Apple Silicon are expected, which would mean less RAM usage while maintaining great precision. Apple's unified memory architecture already provides advantages for memory-heavy tasks and energy efficiency.


## Prerequisites

1.  **Python 3.10 - 3.12** (3.12 is recommended for improved performance)
2.  **FFmpeg** (Required for video generation, conversion, and proper playback)
    -   macOS: `brew install ffmpeg`
    -   Linux: `apt install ffmpeg`
    -   Windows: `winget install ffmpeg` or Download from [ffmpeg.org](https://ffmpeg.org/download.html)

3.  **Python Dependencies** (installed automatically via requirements.txt, see [Installation - Step 4](#installation))

    - **diffusers**: State-of-the-art Image & Video generation pipelines
    - **transformers**: Audio generation & text processing models
    - **torch**: Core deep learning framework & hardware acceleration (CUDA/MPS)
    - **accelerate**: Optimization for efficient large model loading
    - **bitsandbytes**: 4-bit quantization for reducing VRAM usage (FLUX.2, LTX)
    - **opencv-python**: Video frame processing & manipulation
    - **scipy**: Audio signal processing & file handling
    - **realesrgan**: Real-ESRGAN for faster, high-quality image/video upscaling
    - **imageio-ffmpeg**: FFmpeg bindings for video export (used by diffusers)
    - **ddgs**: Deep research (free web search)
    - **markdown**, **python-docx**, **xhtml2pdf**: Document format conversion
    - **rich**: Beautiful terminal formatting, syntax highlighting, and progress spinners
    - **prompt_toolkit**: Interactive command line features (history, arrow keys, tab autocomplete)
    - **peft**: Parameter-Efficient Fine-Tuning for LoRA model loading
    - **psutil**: System resource monitoring (RAM/CPU tracking)
    - **beautifulsoup4**: Web scraping for Deep Research
    - **Web Server**: `fastapi`, `uvicorn`, `python-multipart`, `sse-starlette`, `websockets` for the Web UI & API

4.  **Gated Models (Optional)**
    Some state-of-the-art models (like `FLUX.1`) require Hugging Face authentication (but are **free to use**):

    1.  **The Hugging Face CLI will be installed as part of the installation process (requirements.txt)**
    2.  Create a **Free** [Hugging Face Account](https://huggingface.co/join).
    3.  **Accept model licenses**: Visit each model page and click **"Agree and access repository"** (one-time per model):
        | Model | Accept License |
        | :--- | :--- |
        | FLUX.1-schnell (`flux`) | [Accept License](https://huggingface.co/black-forest-labs/FLUX.1-schnell) |
        | FLUX.1-dev (`flux-dev`) | [Accept License](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
        | FLUX.2 (`flux2`, `flux2-full`) | [Accept License](https://huggingface.co/black-forest-labs/FLUX.2-dev) |
        | SD 3.5 Medium (`sd3.5-medium`) | [Accept License](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) |
        | SD 3.5 Large (`sd3.5-large`) | [Accept License](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) |
        | SD 3.5 Large Turbo (`sd3.5-turbo`) | [Accept License](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo) |
        | Stable Audio Open (`stable-audio`) | [Accept License](https://huggingface.co/stabilityai/stable-audio-open-1.0) |
        | Llama 3.1 8B Instruct (`llama-3.1-8b`) | [Accept License](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) |
    
    > [!NOTE]
    > **Llama 3.1 Approval**: Access to Meta models often requires valid affiliation details and manual approval, which typically takes **15 minutes to 1 hour** (rarely up to 24h). You will receive an email from Hugging Face once approved.
    4.  **Create an Access Token**: Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens) and create a new token:
        - **Quick option**: Select **"Read"** token type for simple read access to all repos.
        - **Fine-grained option**: Select **"Fine-grained"** and enable **"Read access to contents of all public gated repos you can access"** under Repositories.
    5.  **Login**: Run `hf auth login` in your terminal, paste your Access Token, and answer **`n`** to "Add token as git credential?" (only needed for pushing to HF repos).

5.  **Models Storage and Cache (Important)**

    As you use more models and their variants, the cache can grow significantly—easily reaching **hundreds of gigabytes**. By default, Hugging Face stores downloaded models in your home directory, which may fill up your OS disk and cause system issues (especially if your boot drive isn't your largest):

    | Platform | Default Cache Location |
    | :--- | :--- |
    | **macOS** | `~/.cache/huggingface/` |
    | **Linux** | `~/.cache/huggingface/` |
    | **Windows** | `C:\Users\<username>\.cache\huggingface\` |

    **Recommended: Configure a Custom Location**

    Before installing dependencies and downloading models, set this environment variable to redirect the cache to a larger disk:

    - **`HF_HOME`**: Sets the root directory for all Hugging Face data (models, datasets, tokens).

    **Setup Instructions:**

    1. Create your cache directory on a disk with sufficient space (e.g., external drive, secondary partition):
       ```bash
       mkdir -p /path/to/your/huggingface
       ```

    2. Add the environment variable to your shell configuration:

       **macOS / Linux (zsh):** Add to `~/.zshrc`:
       ```bash
       export HF_HOME="/path/to/your/huggingface"
       ```

       **macOS / Linux (bash):** Add to `~/.bashrc`:
       ```bash
       export HF_HOME="/path/to/your/huggingface"
       ```

       **Windows:** Set via System Properties → Environment Variables → New User Variable:
       - Variable name: `HF_HOME`
       - Variable value: `D:\huggingface`
       - (Or run `setx HF_HOME "D:\huggingface"` in Command Prompt to set permanently)

    3. Reload your shell configuration (or restart your terminal):
       - **macOS / Linux (zsh):** `source ~/.zshrc`
       - **macOS / Linux (bash):** `source ~/.bashrc`
       - **Windows:** Close and reopen your terminal (or start a new Command Prompt/PowerShell window)
    
    4. Proceed with the [Installation](#installation) steps.

    > [!TIP]
    > **Avoid symlinks.** While symlinking the default cache folder may seem convenient, it can cause intermittent "file not found" errors in some model loaders. Using the official `HF_HOME` environment variable is the recommended approach.

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
    # if needed: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    .venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create config.json in the root directory of the project
    #5.1.  **Create the file**: Copy `config.sample.json` to `config.json` in the project root.
        - **macOS/Linux**: `cp config.sample.json config.json`
        - **Windows**: `copy config.sample.json config.json`

    #5.2.  **Configure paths**: Open `config.json` and adjust the following values:
        - `paths.hf_home`: Absolute path to your HuggingFace model cache (e.g., `~/.cache/huggingface`).
        - `paths.python_venv`: Path to your Python virtual environment (e.g., `./venv`).
        - `paths.ai_media`: Absolute path to the AI-Media project directory. (e.g. whatever `./` resolves to as absolute path.
        - `paths.ffmpeg`: Path to the `ffmpeg` executable.
        - `paths.media_output`: Directory where generated files will be saved (both from CLI and Web UI).
    #5.3 How to Find Paths
        - If you are unsure where to find these paths, use these commands in your terminal:
            - **FFmpeg Path**:
                - **macOS / Linux**: Run `which ffmpeg`.
                - **Windows**: Run `where ffmpeg` in Command Prompt.
            - **Default HuggingFace Cache**:
                - **macOS / Linux**: `~/.cache/huggingface`
    - **Windows**: `C:\Users\<username>\.cache\huggingface`

# 6. [Optional] Login to Hugging Face (Free); See Gated Models (Optional) section above.
# Only required if you want to use Gated models like Flux.
hf auth login
# (Paste your Access Token when prompted. It is invisible.)

# 7. [Optional] Install Web Server and Client + Electron standalone apps dependencies
# If you only want to use the ai-media.py CLI, you can skip this step. 
# Though the Web interface is highly recommended for ease of use.
cd ai_media/web
npm install
cd ../..
```

## Usage

The script `ai-media.py` serves as the main entry point, relying on feature modules located in the `ai_media` directory. To run the tool from a different location, ensure you move both `ai-media.py` and `requirements.txt` files, as well as the `ai_media` directory together.

**Example Command:**

![Image Generation Example](screenshots/image-gen.png)


**Example Output:**

![Image Generation Example Result](screenshots/image-gen-result.jpg)



### Generation Modes

- `-i, --generate-image`: Generate an image
- `-v, --generate-video`: Generate a video
- `-a, --generate-audio`: Generate audio/music
- `-ga, --generate-article`: Generate an article (Offline)
- `-gr, --generate-research`: Generate an article with Deep Research (Online)
- `-c, --chat`: Interactive chat mode
- `-gc, --generate-code`: Generate code (scripts, projects, etc.)
- `-gd, --generate-description`: Describe an image or video
- `-ti, --transform-image`: Creatively transform/edit an image
- `-ci/-cv/-ca`: Convert media formats (GIF, PNG, MP4, WAV, MP3).
- `-cd, --convert-document`: Convert document formats (MD, HTML, PDF, DOCX, RTF, TXT, JSON).
- `-ui/-uv`: Upscale media
- `-I, --interactive`: Launch the full Interactive Menu (with Mouse Support!) 🖱️

### Common Options

| Option | Description |
| :--- | :--- |
| `-p, --prompt` | Text description of content to generate. |
| `-o, --output` | Output filename/path. **Optional**: auto-generated if omitted (the folder where files are generated is configured in `config.json` under `paths.media_output`). |
| `-f, --format` | Explicit file format. **Image**: jpg, png (default: jpg). **Video**: mp4 (default: mp4). **Audio**: mp3, wav (default: mp3). |
| `--force` | Skip all confirmation prompts (overwrites existing files and ignores resource warnings). |
| `-s, --size` | Resolution. Supports "720p", "1080p", "4k", "8k", "HD", "1280x720", `{w:1280, h:720}`. Default: 720p. |
| `-npt, --no-performance-tracking` | Disable creating/updating `performance.json` and time estimates. [Read more](docs/performance-tracking.md). |

## Quick Examples per Feature

| Feature | Example Command |
| :--- | :--- |
| **Image** | `python ai-media.py -i -p "Cyberpunk city" -s 1080p` |
| **Video** | `python ai-media.py -v -p "Ocean waves" -l 5s` |
| **Audio** | `python ai-media.py -a -p "Lo-fi beat" -l 30s` |
| **Article** | `python ai-media.py -ga -p "Future of AI"` |
| **Research** | `python ai-media.py -gr -p "SpaceX news 2025"` |
| **Chat** | `python ai-media.py -c -chm deepseek-r1-llama-8b` |
| **Code** | `python ai-media.py -gc "Python script to resize images"` |
| **Describe** | `python ai-media.py -gd -ii photo.jpg` |
| **Edit** | `python ai-media.py -ti photo.jpg -tp "Make it anime"` |
| **Upscale** | `python ai-media.py -ui photo.jpg -uf 4.0` |
| **Convert** | `python ai-media.py -cv video.mov -cvt mp4` |
| **Web UI** | `python ai-media.py --serve` (Web & Electron) |

### Feature Documentation

For detailed options, models, and examples for each feature, see the dedicated documentation:

| Feature | Documentation |
| :--- | :--- |
| 🎨 **Image Generation** | [docs/image-generation.md](docs/image-generation.md) |
| 🎬 **Video Generation** | [docs/video-generation.md](docs/video-generation.md) |
| 🎵 **Audio Generation** | [docs/audio-generation.md](docs/audio-generation.md) |
| ✍️ **Article, Chat & Code** | [docs/article-generation.md](docs/article-generation.md) |
| 📝 **Description Generation** | [docs/description-generation.md](docs/description-generation.md) |
| 🪄 **Creative Transformations** | [docs/creative-transformations.md](docs/creative-transformations.md) |
| 🔄 **Media & Document Conversion** | [docs/media-conversion.md](docs/media-conversion.md) |
| 📈 **Upscaling** | [docs/upscaling.md](docs/upscaling.md) |
| 🖥️ **Interactive Menu** | [docs/interactive-menu.md](docs/interactive-menu.md) |
| 🌐 **Web Client & Electron** | [docs/web-client.md](docs/web-client.md) |
| 🧪 **Testing & Codec Analysis** | [docs/testing.md](docs/testing.md) |
| 🔧 **Troubleshooting** | [docs/troubleshooting.md](docs/troubleshooting.md) |
| 📊 **Performance Tracking** | [docs/performance-tracking.md](docs/performance-tracking.md) |
| 🛡️ **Safety & Resources** | [docs/safety-and-resources.md](docs/safety-and-resources.md) |





## Dependencies
 
This project uses the following open-source libraries:

| Dependency | Purpose | GitHub |
| :--- | :--- | :--- |
| [diffusers](https://github.com/huggingface/diffusers) | Image/Video generation pipelines | [huggingface/diffusers](https://github.com/huggingface/diffusers) |
| [transformers](https://github.com/huggingface/transformers) | Audio/Text generation pipelines | [huggingface/transformers](https://github.com/huggingface/transformers) |
| [PyTorch](https://github.com/pytorch/pytorch) | Deep learning framework & Hardware detection | [pytorch/pytorch](https://github.com/pytorch/pytorch) |
| [accelerate](https://github.com/huggingface/accelerate) | Optimization & large model handling | [huggingface/accelerate](https://github.com/huggingface/accelerate) |
| [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | 4-bit quantization for large models | [TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes) |
| [FFmpeg](https://github.com/FFmpeg/FFmpeg) | Media processing & format conversion | [FFmpeg/FFmpeg](https://github.com/FFmpeg/FFmpeg) |
| [sentencepiece](https://github.com/google/sentencepiece) | Tokenization for text processing | [google/sentencepiece](https://github.com/google/sentencepiece) |
| [safetensors](https://github.com/huggingface/safetensors) | Safe model loading format | [huggingface/safetensors](https://github.com/huggingface/safetensors) |
| [scipy](https://github.com/scipy/scipy) | Audio processing mathematics | [scipy/scipy](https://github.com/scipy/scipy) |
| [opencv-python](https://github.com/opencv/opencv-python) | Video frame processing | [opencv/opencv](https://github.com/opencv/opencv) |
| [timm](https://github.com/huggingface/pytorch-image-models) | Image models (Required for Florence-2) | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) |
| [einops](https://github.com/arogozhnikov/einops) | Tensor operations (Required for Florence-2) | [arogozhnikov/einops](https://github.com/arogozhnikov/einops) |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | Real-ESRGAN upscaling | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) |
| [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) | FFmpeg bindings for video export | [imageio/imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) |
| [rich](https://github.com/Textualize/rich) | Beautiful terminal formatting & syntax highlighting | [Textualize/rich](https://github.com/Textualize/rich) |
| [prompt_toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit) | Interactive CLI history and navigation | [prompt-toolkit/python-prompt-toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit) |
| [ddgs](https://github.com/deedy5/duckduckgo_search) | Internet search for Deep Research | [deedy5/duckduckgo_search](https://github.com/deedy5/duckduckgo_search) |
| [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/) | Web content parsing for research | [w3c/beautifulsoup4](https://git.launchpad.net/beautifulsoup) |
| [markdown](https://github.com/Python-Markdown/markdown) | Markdown to HTML conversion | [Python-Markdown/markdown](https://github.com/Python-Markdown/markdown) |
| [python-docx](https://github.com/python-openxml/python-docx) | DOCX document creation | [python-openxml/python-docx](https://github.com/python-openxml/python-docx) |
| [xhtml2pdf](https://github.com/xhtml2pdf/xhtml2pdf) | HTML to PDF conversion | [xhtml2pdf/xhtml2pdf](https://github.com/xhtml2pdf/xhtml2pdf) |
| [psutil](https://github.com/giampaolo/psutil) | System resource monitoring | [giampaolo/psutil](https://github.com/giampaolo/psutil) |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | HF Model downloading & authentication | [huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub) |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | HF Model downloading & authentication | [huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub) |
| [ftfy](https://github.com/rspeer/python-ftfy) | Text encoding fixes | [rspeer/python-ftfy](https://github.com/rspeer/python-ftfy) |
| [fastapi](https://github.com/tiangolo/fastapi) | High-performance web framework for API | [tiangolo/fastapi](https://github.com/tiangolo/fastapi) |
| [uvicorn](https://github.com/encode/uvicorn) | Lightning-fast ASGI server for Python | [encode/uvicorn](https://github.com/encode/uvicorn) |
| [python-multipart](https://github.com/Kludex/python-multipart) | Multipart/form-data support for file uploads | [Kludex/python-multipart](https://github.com/Kludex/python-multipart) |
| [sse-starlette](https://github.com/sysid/sse-starlette) | Server-Sent Events (SSE) for real-time monitoring | [sysid/sse-starlette](https://github.com/sysid/sse-starlette) |
| [websockets](https://github.com/python-websockets/websockets) | WebSocket protocol implementation for Chat | [python-websockets/websockets](https://github.com/python-websockets/websockets) |
| [peft](https://github.com/huggingface/peft) | Parameter-Efficient Fine-Tuning for LoRA loading | [huggingface/peft](https://github.com/huggingface/peft) |

**AI Models used:**

- **Flux** by Black Forest Labs - [black-forest-labs/flux](https://github.com/black-forest-labs/flux)
- **FLUX.2** by Black Forest Labs - [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev)
- **Stable Diffusion XL** by Stability AI - [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models)
- **Stable Diffusion 1.5** by RunwayML - [runwayml/stable-diffusion](https://github.com/runwayml/stable-diffusion)
- **Stable Diffusion 3.5 Medium** by Stability AI - [stabilityai/stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
- **Stable Diffusion 3.5 Large** by Stability AI - [stabilityai/stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
- **Stable Diffusion 3.5 Large Turbo** by Stability AI - [stabilityai/stable-diffusion-3.5-large-turbo](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo)
- **Qwen-Image** (Text-to-Image, best text rendering) - [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) (v2512)
- **Qwen-Image-Lightning** (Fast 8-step Image Gen) - [lightx2v/Qwen-Image-2512-Lightning](https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning)
- **Qwen-Image-4bit** (CUDA 4-bit Image Gen) - [ovedrive/qwen-image-4bit](https://huggingface.co/ovedrive/qwen-image-4bit)
- **Qwen-Image-Edit** (Image editing) - [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- **Qwen-Image-Edit-Lightning** (Fast 2512 Edit) - [lightx2v/Qwen-Image-Edit-2512-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2512-Lightning)
- **Qwen-Coder** (Code Generation) - [Qwen/Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)
- **Qwen3-VL** (Vision Language) - [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- **MusicGen** by Meta AI - [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
- **AudioLDM 2** by Haohe Liu etc. - [haoheliu/AudioLDM2](https://github.com/haoheliu/AudioLDM2)
- **Stable Audio Open** by Stability AI - [stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0)
- **Bark** by Suno - [suno-ai/bark](https://github.com/suno-ai/bark)
- **ModelScope** by Alibaba - [modelscope/modelscope](https://github.com/modelscope/modelscope)
- **Zeroscope** by Cerspense - [cerspense/zeroscope](https://huggingface.co/cerspense/zeroscope_v2_576w)
- **CogVideoX** by THUDM - [THUDM/CogVideo](https://github.com/THUDM/CogVideo)
- **Wan 2.2** by Alibaba PAI - [Alibaba-PAI/Wan-2.2-T2V-14B](https://huggingface.co/Alibaba-PAI/Wan-2.2-T2V-14B)
- **LTX-Video** by Lightricks - [Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video)
- **Mochi 1** by Genmo - [genmo/mochi-1-preview](https://huggingface.co/genmo/mochi-1-preview)
- **HunyuanVideo** by Tencent - [Tencent/HunyuanVideo](https://github.com/Tencent/HunyuanVideo)
- **Stable Video Diffusion** by Stability AI - [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models)
- **Stable Diffusion x2 Latent Upscaler** by Stability AI - [stabilityai/sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)
- **Stable Diffusion x4 Upscaler** by Stability AI - [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)
- **Florence-2** by Microsoft - [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large)
- **BLIP** by Salesforce - [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large)
- **InstructPix2Pix** by Tim Brooks et al. - [timbrooks/instruct-pix2pix](https://github.com/timathy/instruct-pix2pix)
- **RMBG-1.4** by BRIA AI - [briaai/RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4)

## Disclaimer

This tool is provided for **personal use only**. The project owner and contributors assume no responsibility or liability for how users choose to use this script or for any content generated by it. Users are solely responsible for ensuring their use of this tool complies with all applicable laws and regulations in their jurisdiction. 

Additionally, please be aware that AI models for image, video, and audio generation are probabilistic. While often highly accurate, results are **not guaranteed to be perfect** and may contain inaccuracies, omissions, or hallucinations. Manual review and editing of generated content is strongly recommended, especially for critical applications.

The project code was developed with the support of [Antigravity](https://antigravity.google/). As the developer, I defined the feature set, provided guidance, and iterated on improvements across multiple versions—even prior to its initial release on GitHub. In addition, I manually tested all features to ensure reliability and quality.

## License

MIT License