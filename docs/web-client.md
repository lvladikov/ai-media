# Web Client & Desktop Application

AI-Media includes a modern web-based interface and cross-platform desktop application for accessing all features through a graphical UI.

## Overview

The web interface provides:
- 🖥️ **Browser-based access** to all AI-Media features
- 📊 **Real-time resource monitoring** (CPU, RAM, VRAM, GPU)
- 📁 **File upload support** for images, documents, and media
- 💬 **Interactive chat** with streaming responses
- 📱 **Responsive sidebar navigation** for all features
- 🖼️ **Output preview** for generated images, videos, audio, PDFs, and code
- 📂 **Download or open folder** for generated media

## Architecture

```
┌─────────────────────────────────────────────────────┐
│             Electron Shell (Desktop)                │
│  ┌───────────────────────────────────────────────┐  │
│  │        React Frontend (Vite + Tailwind)       │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                        │
                HTTP / WebSockets
                        ▼
┌─────────────────────────────────────────────────────┐
│           FastAPI Server (ai_media/server)          │
│    Direct imports from existing ai_media modules    │
└─────────────────────────────────────────────────────┘
```

The application follows a client-server architecture. The **React Frontend** provides the UI and logic, which can run either in a standard web browser or wrapped within the **Electron Shell** for a native desktop experience. Both clients communicate with a **FastAPI Server** that wraps the existing Python modules. The CLI and Interactive Menu remain fully functional and share the same underlying logic.

## Running the Web Server

```bash
# Start the backend server and launch BOTH Web and Electron clients
# (Defaults: Reload ON, Ports from config.json)
python ai-media.py --serve

# Start backend server + ONLY the Web UI
python ai-media.py --serve-web-only-client

# Start backend server + ONLY the Electron Dev app
python ai-media.py --serve-electron-dev-only-client

# Start ONLY the backend server (no clients)
# (Defaults: Reload OFF)
python ai-media.py --serve-no-client

# Enable auto-reload explicitly (useful for server-only mode)
python ai-media.py --serve-no-client --reload
```

The server runs at `http://localhost:8000` by default with:
- Swagger docs at `/docs`
- API endpoints at `/api/*`
- WebSocket for chat at `/ws/chat`
- SSE for resource monitoring at `/sse/resources`

## Quick Start (Recommended)

The easiest way to run both the server and client is from the `ai_media/web/` folder:

```bash
cd ai_media/web
npm install
npm run dev:both    # Starts both Python server and React client
```

This uses `concurrently` to run:
- Python server at `http://localhost:8000`
- React dev server at `http://localhost:5173`

**Alternative commands:**
```bash
npm run dev:client   # React only (if server running elsewhere)
npm run dev:server   # Python server only
npm run dev          # Alias for dev:client
```

## Desktop Application (Electron)

The desktop app bundles the web client for native experience on:
- **macOS** (x64, arm64, universal)
- **Windows** (x64, arm64)
- **Linux** (x64, arm64)

### Running Electron Dev Mode

```bash
npm run electron     # Starts Electron (requires dev:client running)
```

### Building Desktop Apps

From the `ai_media/web/` folder:

```bash
# macOS
npm run electron:build:mac:all      # All architectures
npm run electron:build:mac:arm64    # Apple Silicon only
npm run electron:build:mac:x64      # Intel only

# Windows
npm run electron:build:win:all
npm run electron:build:win:x64
npm run electron:build:win:arm64

# Linux
npm run electron:build:linux:all
npm run electron:build:linux:x64
npm run electron:build:linux:arm64

# All platforms
npm run electron:build:all
```

Builds output to `dist/desktop/` in the project root.

## Configuration (`config.json`)

The application uses a centralized `config.json` file to manage paths, server settings, and user preferences. This file is shared between the Python backend, the Vite development server, and the Electron desktop application.

> [!IMPORTANT]
> `config.json` is ignored by git for security and local environment flexibility. Always use `config.sample.json` as a starting point.

### Setup Instructions

1.  **Create the file**: Copy `config.sample.json` to `config.json` in the project root.
    - **macOS/Linux**: `cp config.sample.json config.json`
    - **Windows**: `copy config.sample.json config.json`

2.  **Configure paths**: Open `config.json` and adjust the following values:
    - `paths.hf_home`: Absolute path to your HuggingFace model cache (e.g., `/Users/name/.cache/huggingface`).
    - `paths.python_venv`: Path to your Python virtual environment (e.g., `./venv`).
    - `paths.ai_media`: Absolute path to the AI-Media project directory.
    - `paths.ffmpeg`: Path to the `ffmpeg` executable.
    - `paths.media_output`: Directory where generated files will be saved.

3.  **Configure server/client**:
    - `server.host` / `server.port`: The backend API server address (default: `127.0.0.1:8000`).
    - `client.host` / `client.port`: The frontend development server address (default: `127.0.0.1:5175`).

### How to Find Paths

If you are unsure where to find these paths, use these commands in your terminal:

- **FFmpeg Path**:
    - **macOS / Linux**: Run `which ffmpeg`.
    - **Windows**: Run `where ffmpeg` in Command Prompt.
- **Default HuggingFace Cache**:
    - **macOS / Linux**: `~/.cache/huggingface`
    - **Windows**: `C:\Users\<username>\.cache\huggingface`


### Usage by Platform

| Environment | Usage |
| :--- | :--- |
| **Web (Dev)** | Vite reads `config.json` at startup to set the dev server port. The Python backend reads it for paths and server binding. |
| **Electron (Dev)** | Electron loads the frontend from the `client.port` defined in `config.json`. |
| **Electron (Built)** | The packaged app looks for `config.json` in its root folder to locate the Python backend and model files. |

### Electron Deployment

When building and distributing the Electron application, `config.json` is **not bundled** inside the app to keep the package small and allow for local configuration.

> [!TIP]
> **File Placement**: For the built application to function correctly, `config.json` must be placed in the **same directory** as the application executable:
> - **macOS**: Next to the **AI-Media.app** bundle (one level above the `.app` folder).
> - **Windows**: Next to **AI-Media.exe**.
> - **Linux**: Next to the AppImage or binary.

If `config.json` is missing on the first run, the application will attempt to guide you through a setup wizard to generate it.

## Static Web Build

For deployment to a web server:

```bash
cd ai_media/web
npm run build
```

Output goes to `dist/web/` for hosting anywhere.

## Features Available

All CLI features are available via the web interface:

| Feature | Endpoint | Description |
|---------|----------|-------------|
| Image Generation | `/api/generate/image` | Text-to-image |
| Video Generation | `/api/generate/video` | Text/image-to-video |
| Audio Generation | `/api/generate/audio` | Text-to-audio/music |
| Article Generation | `/api/generate/article` | Offline article writing |
| Code Generation | `/api/generate/code` | Code generation |
| Chat | `/ws/chat` | Interactive chat (WebSocket) |
| Transform | `/api/transform` | Image editing/background removal |
| Upscale | `/api/upscale` | AI and non-AI upscaling |
| Convert | `/api/convert` | Media/document conversion |

## Preview & File Actions

After generation completes, outputs are automatically displayed in a preview modal:

| File Type | Supported Formats | Preview Component |
|-----------|-------------------|-------------------|
| Images | JPG, PNG, GIF, WebP, BMP, TIFF | In-browser display |
| Videos | MP4, WebM, MOV, MKV | HTML5 video player |
| Audio | MP3, WAV, FLAC, M4A, AAC | Audio player with waveform |
| Documents | PDF | PDF viewer (react-pdf) |
| Code/Text | MD, TXT, JSON, HTML, CSS, JS, PY | Syntax highlighted (PrismJS) |

For each generated file, you can:
- **📥 Download** - Save the file to your downloads folder
- **📂 Open Folder** - Reveal the file in Finder/Explorer (Electron only)
- **🚀 Open with System App** - Open in default application (Electron only)

Unsupported file types show a fallback with the option to open in the system default application.

## Model Caching

The server intelligently caches loaded AI models to avoid unnecessary reload cycles:

| Scenario | Behavior |
|----------|----------|
| Chat → Article (same LLM) | ♻️ Model reused |
| Chat → Code gen (same LLM) | ♻️ Model reused |  
| Chat with Llama → Chat with DeepSeek | 🔄 Unload old, load new |
| Text generation → Image generation | 🧹 Unload LLM, load diffusion |

**Cache Management Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/cache` | GET | View currently cached models |
| `/api/cache` | DELETE | Unload all cached models |
| `/api/cache/{category}` | DELETE | Unload specific category (text, image, audio, video) |

This significantly speeds up workflows that use the same model across different features (e.g., switching between chat and article generation with the same LLM).

## Job Cancellation

Jobs can be cancelled at any time by clicking the **Cancel Job** button in the progress modal or the job list. The cancellation behavior varies by task type:

| Task Type | Cancellation Method | Behavior |
|-----------|---------------------|----------|
| **Image, Video, Audio** | ⚡ Process Kill | Immediate termination via SIGTERM. GPU memory freed instantly. |
| **Article, Code** | ⚡ Process Kill | Immediate termination. Model loading/generation stops mid-way. |
| **Transform, Upscale, Convert** | ⚡ Process Kill | Immediate termination. Partial output files are cleaned up. |
| **Chat** | 🛑 Graceful Stop | Uses `generator.stop()` flag. Model stays cached for the session. |

> [!NOTE]
> **Image/Video/Audio/Article/Code/Transform/Upscale/Convert** tasks run in separate child processes which are killed immediately on cancellation. This frees GPU/RAM memory but means the model must reload on the next job (it is a quick process few seconds usually)

> [!TIP]
> **Chat** sessions keep the model cached between messages. Cancelling a chat response stops generation gracefully without unloading the model, making subsequent messages faster.

### Server Shutdown

Pressing `Ctrl+C` in the terminal terminates all running child processes before shutting down the server. Child processes are designed to ignore `SIGINT` directly, so only the parent server handles the shutdown signal.

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | React 18, Vite, TypeScript |
| Styling | Tailwind CSS |
| State | Zustand |
| Code Highlighting | PrismJS |
| Backend | FastAPI, Uvicorn |
| Desktop | Electron 30.5.1 |
| Build | electron-builder |


