# OpenAI-Compatible Inference Server

AI-Media includes a built-in inference server that exposes an OpenAI-compatible API (`/v1`). This allows you to use your local AI-Media models with third-party tools like **Continue**, **LM Studio**, **Cursor**, or any application that supports the OpenAI API specification.

## Quick Start

To start the inference server, use the `--inference-server` flag:

```bash
python ai-media.py --inference-server
```

You can optionally specify a custom port (default is 8000 if not configured differently):

```bash
python ai-media.py --inference-server --port 8090
```

To see detailed logs of every request and response (including model reasoning), use the verbose flag:

```bash
python ai-media.py --inference-server-verbose
```

## Stopping the Server

You can stop the server at any time using one of these methods:

1.  **Terminal**: Press `CTRL+C` in the terminal where the server is running.
2.  **Chat Command**: Send the message `stop inference server` (case-insensitive) through your chat client (Continue, LM Studio, etc.). The server will acknowledge the request both on terminal and in Continue and shut down.
3.  **Kill Command**: Find the process ID (`ps aux | grep ai-media`) and run `kill <pid>`.

---

## Memory Management

AI-Media's inference server includes intelligent memory management to prevent "Out of Memory" (OOM) errors and keep your system responsive.

### 1. Single Model Policy (Automatic)
The server enforces a **"Single Active Model Category"** policy.
- If you are chatting with a **Text Model** (e.g., Llama/Qwen) and then request an image from an **Image Model** (e.g., Flux), the server will **automatically unload the text model** to free VRAM before loading the image model.
- Switching back to text will unload the image model.
- This ensures you never have two massive models (e.g., 20GB + 12GB) competing for VRAM simultaneously.
- Using different models within the *same* category (e.g., switching from Llama to Qwen) also unloads the previous one.

### 2. Manual Commands
You can manually control memory usage by sending these commands as chat messages:

- **`unload model`**: Immediately unloads the currently active model from RAM/VRAM.
- **`flush memory`**: Forces Python garbage collection to release any "ghost" memory references and clears the GPU cache.

Example:
> **User**: unload model
>
> **Server**: Model llama-3.1-8b unloaded and memory flushed.

### 3. Random Prompt Command
Need inspiration? Send one of these trigger phrases to get a random creative prompt:

- `rndPr`
- `rndPrompt`
- `randomPrompt`
- `random prompt`

The server returns a random prompt appropriate for your selected model:
- **Image models** (Flux, SDXL, etc.): Returns creative image generation prompts
- **Text models** (Qwen, Llama, etc.): Returns code/programming task prompts

Example:
> **User** (with Flux model selected): rndPr
>
> **Server**: 🎲 **Random Prompt**
>
> A serene Japanese garden with cherry blossoms falling into a koi pond

---

Once running, the server provides:
- **Base URL**: `http://localhost:8000/v1` (or your custom port)
- **API Key**: Any string (e.g. `local` or `sk-test`). The server accepts anything.
- **Models Endpoint**: `http://localhost:8000/v1/models` (Lists all available **Text** and **Image** models in AI-Media)
- **Chat Endpoint**: `http://localhost:8000/v1/chat/completions`

---

## Using with "Continue" (VS Code / JetBrains)

[Continue](https://continue.dev/) is an open-source AI code assistant. You can configure it to use AI-Media as its backend.

1. **Start AI-Media Server**:
   ```bash
   python ai-media.py --inference-server
   ```

2. **Edit Continue Configuration**:
   - Open your `~/.continue/config.yaml` (recommended) or `~/.continue/config.json` file (or click the settings gear icon in the Continue extension).
   
   > [!TIP]
   > **Example Configuration File**
   > A complete, ready-to-use configuration file with all supported models is available in this repository:
   > [`extras/continue-dev-example-config/config.sample.yaml`](../extras/continue-dev-example-config/config.sample.yaml)
   >
   > You can simply copy the contents of this file into your `~/.continue/config.yaml` to get started with all AI-Media models pre-configured.

   **Option A: config.yaml (Recommended/New)**
   
   ```yaml
   models:
     - name: AI Media (Qwen Coder 32B)
       provider: openai
       model: qwen-coder-32b
       apiBase: http://localhost:8000/v1
       apiKey: local

     - name: AI Media (DeepSeek R1 Distill)
       provider: openai
       model: deepseek-r1-qwen-14b
       apiBase: http://localhost:8000/v1
       apiKey: local
   ```

   **Option B: config.json (Legacy)**

   ```json
   {
     "models": [
       {
         "title": "AI Media (Qwen Coder 32B)",
         "provider": "openai",
         "model": "qwen-coder-32b",
         "apiBase": "http://localhost:8000/v1",
         "apiKey": "local" 
       },
       {
         "title": "AI Media (DeepSeek R1 Distill)",
         "provider": "openai",
         "model": "deepseek-r1-qwen-14b",
         "apiBase": "http://localhost:8000/v1",
         "apiKey": "local"
       }
     ]
   }
   ```

3. **Select the Model**: In the Continue dropdown, select "AI Media (Qwen Coder 32B)" (or whichever title you gave it).

**Notes for Continue:**
- **Tab Autocomplete**: For autocomplete, configure the `tabAutocompleteModel` section. `qwen-coder-7b` is a good fast choice for autocomplete.

  **Option A: config.yaml (Recommended/New)**
  ```yaml
  tabAutocompleteModel:
    title: Autocomplete
    provider: openai
    model: qwen-coder-7b
    apiBase: http://localhost:8000/v1
    apiKey: local
  ```

  **Option B: config.json (Legacy)**
  ```json
  "tabAutocompleteModel": {
    "title": "Autocomplete",
    "provider": "openai",
    "model": "qwen-coder-7b",
    "apiBase": "http://localhost:8000/v1",
    "apiKey": "local"
  }
  ```

---

## Adding Context (Files, Folders, Workspace)

AI-Media's inference server supports the full OpenAI Chat schema, meaning it can handle any context your client (Continue, VS Code, etc.) sends.

**How it works:**
Clients like Continue resolve references like `@File`, `@Folder`, or `@Codebase` (similar to `@workspace` in GitHub Copilot) into raw text *before* sending the request. The AI-Media server then processes this rich context.

**Common Context Providers:**
- **`@File` / `@Open Files`**: Attaches specific file contents.
- **`@Folder`**: Attaches an entire directory.
- **`@Codebase` / `@workspace`**: Uses embeddings to find relevant snippets across your project.
- **`@Docs`**: Attaches external documentation.

> [!IMPORTANT]
> **Context Must Be Resolved**
> When using context providers like `@File` or `@Folder`, ensure you **select the item from the dropdown menu** so it appears as a blue "chip" in your chat bar. 
> 
> If you type `@src/config.py` but don't select it, it sends as **raw text**. The model might get confused and try to "read" that path itself using internal tools (which will fail), rather than receiving the actual file content from Continue.

Because AI-Media models support large context windows (depending on the model, e.g., 32k or 128k tokens for Qwen), you can confidently use these features to "chat with your codebase".

> [!CAUTION]
> **Resource Warning**: While a small model (e.g., Qwen 7B) acts as the "brain," the "memory" (Context Window) consumes significant RAM/VRAM. processing 128k tokens of context can easily require **30GB+ of RAM**, even for a defined 7B model. Ensure your hardware can handle the context size you are providing.

---

## Using with LM Studio

Although LM Studio is typically a server itself, you can connect it to AI-Media or generic OpenAI clients. If you have a client that expects an OpenAI server, point it to `http://localhost:8000/v1`.

---

## Available Models

The server automatically detects and exposes all text models defined in AI-Media (`ai_media/models.py`). 

To see the exact model IDs to use in your configuration, you can:
1. Visit `http://localhost:8000/v1/models` in your browser while the server is running.
2. Or use the CLI list command: `python ai-media.py --list-models`

**Recommended Models for Coding:**
- `qwen-coder-32b`: State-of-the-art coding capabilities (Requires 24GB+ RAM/VRAM)
- `qwen-coder-14b`: Excellent balance of speed and capability
- `deepseek-r1-qwen-14b`: Strong reasoning capabilities

**Recommended Models for Chat:**
- `llama-3.1-8b`
- `mistral-nemo-12b`

**Recommended Models for Image Generation:**
- `flux`: FLUX.1 (Schnell) - Fast and high quality
- `sdxl`: SDXL Turbo - Very fast
- `sd3.5-large`: Stable Diffusion 3.5 Large

---

## Image Generation via Chat

You can generate images directly through the Chat API by selecting an image model ID and sending a text prompt. The server will return the image as a Markdown link.

**Example Request:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "flux",
    "messages": [{"role": "user", "content": "A cyberpunk city at night with neon lights"}]
  }'
```

**Response:**

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1709...,
  "model": "flux",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "![Generated Image](http://localhost:8000/api/files/uploads/flux_generated_image.png)"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": { ... }
}
```

### Using in Continue / VS Code

To generate an image in Continue:

1.  **Configure the Model**: Add an entry for the image model in your `config.yaml` or `config.json`.
    ```yaml
    - name: AI Media (Flux Image Gen)
      provider: openai
      model: flux
      apiBase: http://localhost:8000/v1
      apiKey: local
    ```
2.  **Select the Model**: In the Continue chat dropdown, switch from your coding model (e.g. Qwen) to **Flux Image Gen**.
3.  **Type Your Prompt**: Enter your image description (e.g., *"A futuristic dashboard UI design, sleek, dark mode"*).
4.  **View Result**: The server will generate the image and Continue will render it inline using the returned specific markdown link to the locally generated file.

**Note**: You must switch to the image model explicitly. Asking the text model (Qwen) to "generate an image" will not work unless that model has tool-use capabilities configured, which is separate from this direct inference feature.

---

## Learn More

For full documentation on available features and models, see:

- [Image Generation Guide](image-generation.md)
- [Video Generation Guide](video-generation.md)
- [Audio Generation Guide](audio-generation.md)
- [Article & Text Generation Guide](text-generation.md)
- [Analysis & Description Guide](analysis.md)
- [Creative Transformations Guide](creative-transformations.md)

