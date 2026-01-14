# Article, Research, Chat & Code Generation



This module is your all-in-one text intelligence hub. It uses local Large Language Models (LLMs) to perform four distinct types of tasks:

1.  **✍️ Article Generation (`-ga`)**:
    *   **What it is**: Writes comprehensive, structured articles on any topic using the model's internal training data.
    *   **Best for**: Creative writing, explaining concepts, drafting support, or generating content where current-day news isn't required.
    *   **Key Feature**: Offline capable. No internet required.

2.  **🌐 Deep Research (`-gr`)**:
    *   **What it is**: An autonomous research agent. It searches the live web (via DuckDuckGo), reads multiple sources, synthesizes the information, and writes a summary report.
    *   **Best for**: Current events, stock trends, sports scores, or finding specific information not in the model's training set (e.g., "iPhone 16 release date").
    *   **Key Feature**: Cites sources and overcomes the "knowledge cutoff" of static models.

3.  **💬 Interactive Chat (`-c`)**:
    *   **What it is**: A persistent conversational session, similar to ChatGPT but running locally.
    *   **Best for**: Brainstorming, Q&A, roleplay, or iterative problem solving.
    *   **Key Feature**: **Dynamic Context Management**.
        *   `/read [file]`: Reads a local file into the chat context, allowing you to discuss code, documents, or logs.
        *   `/search [query]`: A variant of `/read` that pulls context from **online sources** instead of local files.
        *   `/save [filename]`: Saves the current conversation or generated content to a file.
        *   *Result*: You can build a workspace of knowledge (files + web search) and have a grounded discussion about it.

4.  **💻 Code Generation (`-gc`)**:
    *   **What it is**: A specialized mode for writing software. It can generate single scripts or scaffold entire multi-file projects (creating folders and files automatically).
    *   **Best for**: Python scripts, React components, boilerplate, unit tests, or learning new languages.
    *   **Key Feature**: Recognized "project" requests and automatically builds the directory structure.

← [Back to Main README](../README.md)

## Article & Text Options

| Option | Description |
| :--- | :--- |
| `-ga, --generate-article` | Generate an article offline (using model knowledge only). |
| `-gr, --generate-research` | Generate an article with "Deep Research" (uses DuckDuckGo to search & summarize). |
| `-c, --chat` | Start an interactive chat session. |
| `-atm, --article-model` | Model for article generation. Default: `default` (Llama-3.1-8B). |
| `-chm, --chat-model` | Model for chat. Default: `default`. |
| `--output-format` | Output format: `md` (default), `pdf`, `docx`, `html`, `json`. |
| `-ri, --research-iter` | Deep Research iterations (number of sources to read). Default: `3`. |
| `-al, --article-length` | Article length: `quick` (~500 words, fast, default), `standard` (~1500), `detailed` (~3000), `exhaustive` (~10000). **Tip**: Qwen models follow length instructions most reliably. |
| `-p, --prompt` | Text description/topic for article generation. |
| `-o, --output` | Output filename/path. **Optional**: auto-generated if omitted (the folder where files are generated is configured in `config.json` under `paths.media_output`). |

See [Article & Chat Examples](#examples) and [Models](#text-models).

## Code Options

| Option | Description |
| :--- | :--- |
| `-gc, --generate-code` | Generate code based on a text prompt. |
| `-cdm, --code-model` | Model for code generation. Default: `llama-3.1-8b`. |
| `-o, --output` | **Optional** Output path. **Empty**: Uses filenames/paths from your prompt (Recommended). **Dir**: Saves all files inside this folder. **File**: Overrides filename (single-file output only). The folder where files are generated is configured in `config.json` under `paths.media_output`. |

### Code Output Folder vs File interpretation

If you want generated files saved in a subfolder, **the folder must exist before running the command**.

- **Folder exists**: All generated files are saved inside it.
- **Folder doesn't exist**: The path is interpreted as a filename (single-file output only, model decides on the extension based on context), and for multi-file projects it will be ignored (files saved to current directory with model-chosen names).

```bash
# Create the folder first, then generate
mkdir ./my-web-app
python ai-media.py -gc "Create a Flask backend with a React frontend" -o ./my-web-app
```

### Time and Location in Chat

Some advanced chat models such as DeepSeek R1 Qwen 32B and DeepSeek R1 Llama 70B can work with provided in advance time and location context. The script provides automatically the time from your system and uses IP based geolocation [using IP API](http://ip-api.com) to inform the model about the current time and location. It uses a very short timeout to get the location as fast as possible and falls back to "Unknown" if it fails.

See [Code Generation Examples](#examples) and [Models](#text-models).

## Text Models

### Reasoning-Focused (Chain-of-Thought)

> **What are "Reasoning" models?** These high-performance models excel at complex logic, math, code generation, and step-by-step analysis.
> *   **DeepSeek R1**: "Distilled" models that show their internal thinking process (Chain-of-Thought). "Distillation" transfers R1's reasoning capabilities into smaller, faster base models like Qwen or Llama. The result: you get R1's explicit reasoning style on consumer hardware, using the efficient architecture of the base model.
> *   **Qwen 3**: Latest generation models with native robust reasoning and instruction following capabilities.

| Model | Code | Download | VRAM | Context | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 3 8B (Reasoning)** | `qwen3-8b` | ~16GB | ~16GB | 128k | **(Default)** Latest Qwen. Strong instruction-following. |
| **Qwen 3 14B (Reasoning)** | `qwen3-14b` | ~28GB | ~28GB | 128k | Qwen 3. Great at detailed formatting. |
| **DeepSeek R1 Qwen 7B** | `deepseek-r1-qwen-7b` | ~4GB | ~7GB | 128k | Lightweight, fast. Good starting point. |
| **DeepSeek R1 Qwen 14B** | `deepseek-r1-qwen-14b` | ~8GB | ~14GB | 128k | Better reasoning quality. |
| **DeepSeek R1 Qwen 32B** | `deepseek-r1-qwen-32b` | ~18GB | ~24GB | 128k | High quality. |
| **DeepSeek R1 Llama 8B** | `deepseek-r1-llama-8b` | ~5GB | ~8GB | 128k | Llama architecture variant. |
| **DeepSeek R1 Llama 70B** | `deepseek-r1-llama-70b` | ~35GB | ~40GB | 128k | Best quality. |

> [!NOTE]
> All DeepSeek R1 distilled models are **fully open and ungated** (MIT license). No HuggingFace login required.

### General-Purpose

| Model | Code | Download | VRAM | Context | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama 3.1 8B** | `llama-3.1-8b` | ~5GB | ~16GB | 128k | 🔒 **Gated**. Open SOTA 8B. General writing, chat, reasoning. |
| **Mistral Nemo 12B** | `mistral-nemo-12b` | ~7GB | ~24GB | 128k | Powerful 12B. Large context window, strong reasoning. |
| **Qwen 2.5 Coder 32B** | `qwen-coder-32b` | ~20GB | ~24GB | 128k | SOTA Code Gen. |
| **Qwen 2.5 Coder 14B** | `qwen-coder-14b` | ~10GB | ~12GB | 128k | Fast & Capable Code Gen. |
| **Qwen 2.5 Coder 7B** | `qwen-coder-7b` | ~5GB | ~6GB | 128k | Lightweight Code Gen. |
| **Qwen3 Coder 30B** | `qwen3-coder-30b` | ~8GB | ~10GB | 32k | **MoE** (3.3B active). Efficient SOTA. |

- All models are quantized (4-bit) on CUDA where possible to fit in consumer GPU memory.

---

## Precision & Framework Control

AI-Media supports fine-grained control over model precision and ML framework. For a deep dive into precision types and their trade-offs, see **[Precisions Explained](precisions-explained.md)**.

### Quick Reference

| Option | Description |
|--------|-------------|
| `--precision-force`, `-pf` | Force precision: `int4`, `int6`, `int8`, `float16`, `bfloat16`, `float32` |
| `--ml-framework`, `-mf` | Force framework (Mac): `mlx` (native) or `torch` (PyTorch MPS) |

### Platform Defaults (Text Models)

When no precision is specified, AI-Media automatically selects optimal settings:

| Platform | Default Precision | Default Framework | Notes |
|----------|------------------|-------------------|-------|
| **CUDA (NVIDIA)** | `int4` (4-bit bitsandbytes) | PyTorch | Auto-quantized for consumer GPUs |
| **MPS (Mac PyTorch)** | `float16` | PyTorch | Stable, no quantization on MPS |
| **MLX (Mac Native)** | `int4` | MLX | Fastest inference on Apple Silicon |

> [!NOTE]
> **Why bfloat16 on Mac?** MPS doesn't support 4-bit quantization natively. Use MLX (`-mf mlx`) for quantized models on Apple Silicon.

### CLI Examples

```bash
# Chat with forced bfloat16 (default on MPS now)
python ai-media.py -c -chm llama-3.1-8b -pf bfloat16

# MLX with int4 quantization (Mac - fastest)
python ai-media.py -c -chm qwen3-8b -mf mlx -pf int4

# CUDA with int8 quantization
python ai-media.py -ga -p "Topic" -pf int8

# Code generation with MLX
python ai-media.py -gc "Python script" -mf mlx -pf int4
```

> [!TIP]
> **Mac Users**: For best text generation performance, use MLX with int4 (`-mf mlx -pf int4`). It's significantly faster than PyTorch/MPS. See [Inference Server](inference-server.md) for API usage with precision.

---


## Examples

### Article Generation (`-ga`)

> [!TIP]
> **Need inspiration?** Use `rndPr`, `rndPrompt`, `randomPrompt`, or `random prompt` as your topic to get a randomly selected article topic.

```bash
# Quick Article (Default settings)
python ai-media.py -ga -p "The benefits of meditation"
python ai-media.py --generate-article --prompt "The benefits of meditation"

# Generate with a random topic
python ai-media.py -ga -p rndPr

# Detailed Article with Specific Model
# Using a reasoning model (DeepSeek R1 distilled) for deeper analysis.
python ai-media.py -ga -p "Analysis of Quantum Computing" -atm deepseek-r1-qwen-7b -al detailed
python ai-media.py -ga -p "Topic" --article-model deepseek-r1-qwen-7b --article-length detailed

# Custom Output Format
python ai-media.py -ga -p "Quarterly Report" --output-format pdf -o report.pdf
```

### Deep Research (`-gr`)

Performs live web searches (DuckDuckGo) before writing.

```bash
# Latest News (Knowledge beyond training cutoff)
python ai-media.py -gr -p "SpaceX launches in 2026"
python ai-media.py --generate-research --prompt "SpaceX launches in 2026"

# Intensive Research (More Sources)
# -ri 10 reads 10 distinct sources for a broader view.
python ai-media.py -gr -p "Stock market trends" -ri 10
python ai-media.py -gr -p "Stock market trends" --research-iter 10
```

### Interactive Chat (`-c`)

```bash
# Default Session (Llama 3.1 8B)
python ai-media.py -c
python ai-media.py --chat

# Reasoning-Focused Chat
# Use DeepSeek R1 variants for math, logic, or complex troubleshooting.
# Note: These models display their internal "Thinking" process (inside <think> tags) separated from the final answer.
python ai-media.py -c -chm deepseek-r1-llama-8b
python ai-media.py -c --chat-model deepseek-r1-llama-8b
```

### Code Generation (`-gc`)

> [!TIP]
> **Need inspiration?** Use `rndPr`, `rndPrompt`, `randomPrompt`, or `random prompt` as your prompt to get a randomly selected code task.

```bash
# Simple Script (Auto-saved)
python ai-media.py -gc "Write a python script to ping a server"
python ai-media.py --generate-code "Write a python script to ping a server"

# Generate with a random prompt
python ai-media.py -gc rndPr

# Specific Filename
python ai-media.py -gc "Write a snake game in html/js" -o snake.html

# Full Project Generation (Folder Output)
# IMPORTANT: Create the folder first! Otherwise -o is treated as a filename.
mkdir ./my-web-app
python ai-media.py -gc "Create a Flask backend with a React frontend" -o ./my-web-app
# Result:
# ./my-web-app/app.py
# ./my-web-app/src/App.js
# ...

# Complex Logic (Reasoning Model)
# DeepSeek R1 is excellent for complex algorithms.
python ai-media.py -gc "Implement A* search algorithm optimization" -cdm deepseek-r1-qwen-14b
python ai-media.py -gc "Prompt" --code-model deepseek-r1-qwen-14b
```

← [Back to Main README](../README.md)
