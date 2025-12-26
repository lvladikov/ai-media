# Safety & Resource Checks

Details on how the tool manages system resources and content safety.

← [Back to Main README](../README.md)

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

   • RAM: 28.0GB available, 42.0GB recommended (bfloat16)
   • Resolution: 1920x1080 exceeds recommended max 1280x720

   Model: black-forest-labs/FLUX.1-schnell
   Dtype: bfloat16
   This job may cause slowdowns, swapping, or crashes.

   Continue anyway? [y/N]:
```

### 🔧 Options
- **y**: Proceed despite the warning
- **N** (default): Cancel and adjust parameters
- **--force**: Skip all confirmation prompts (overwrites existing files and ignores resource warnings).


## Understanding Memory & Model Loading

AI models are large files that must be loaded into memory before generation can begin. Understanding the memory hierarchy helps explain why certain models require more resources and why some operations are slow.

### Memory Optimization (BFloat16 / Float16)
The system automatically detects if your hardware supports half-precision operations (`bfloat16` on CUDA, `float16` on MPS).
- **If supported**: Memory REQUIREMENTS are automatically scaled down by ~40% (0.6x factor) before checking.
- **If unsupported**: The full 32-bit (Float32) memory requirements are used for validation.

### Memory Types & Speed

| Memory Type | Typical Speed | Latency | Use Case |
|-------------|---------------|---------|----------|
| **GPU VRAM** | 500-1000 GB/s | ~ns | Active model computation (CUDA) |
| **Unified Memory** | 200-400 GB/s | ~ns | Apple Silicon (MPS) - shared CPU/GPU |
| **System RAM** | 25-50 GB/s | ~ns | Model storage, CPU offloading |
| **NVMe SSD** | 3-7 GB/s | ~μs | Model file storage, swap |
| **SATA SSD** | 0.5-0.6 GB/s | ~μs | Slower model loading |
| **HDD** | 0.1-0.2 GB/s | ~ms | Significantly slower, not recommended |

### Why Models Need RAM/VRAM

During inference, neural networks access model weights **millions of times per second** in complex, non-sequential patterns. Even the fastest NVMe SSD is 10-50x slower than RAM, making disk-based inference impractical for real-time generation.

**"Loading checkpoint shards"** - Large models (like Flux at ~23GB) are split into multiple files called "shards" (typically 2-5GB each). When loading, you'll see progress like:
```
Loading checkpoint shards: 33%|███▎ | 1/3 [00:04<00:08, 4.46s/it]
```
This indicates the model is being read from disk into memory piece by piece.

### Platform Differences

#### NVIDIA GPUs (CUDA)
- **VRAM**: Dedicated high-bandwidth memory.
- **Dtype**: Automatically prefers `bfloat16` (Ampere+) or `float16` to reduce memory usage by ~50% vs Float32.
- **CPU Offloading**: The system now automatically uses `enable_model_cpu_offload()` for large models (**Flux 1 & 2**, **Wan 2.2**, **SVD**, **MusicGen**, **AudioLDM2**, **SD Upscaler**). This keeps only the active module (e.g., Transformer block) in VRAM and the rest in RAM, allowing massive 30GB+ models to run on 8GB-16GB GPUs.
- **Quantization**: **Flux 2** and large Text Models (**deepseek-r1-llama-70b**, **qwen-32b**) automatically use 4-bit quantization (`bitsandbytes`) to drastically reduce VRAM usage (e.g., 70B model fits in ~35GB).

#### Apple Silicon (MPS)
- **Unified Memory**: CPU and GPU share the same RAM. explicit "CPU Offloading" is **disabled** because it conflicts with macOS's native memory management.
- **Dtype**: Uses `float16` by default for performance and memory savings.
- **Optimization Strategy**: Instead of offloading, the system enables **Attention Slicing** on Mac. This splits large compute operations into smaller chunks to keep peak memory usage below the "Wired Memory" limit, preventing crashes with large models like Flux.
- **Stability Fixes**: The system automatically enforces Float32 for VAEs (Flux/SDXL/Video) on MPS to prevent NaN errors (which cause black images).

### Techniques to Reduce Memory Usage

| Technique | Memory Savings | Trade-off |
|-----------|----------------|-----------|
| **Half-Precision (BF16/FP16)** | ~50% | Negligible quality difference (Default) |
| **Quantization** (4-bit/8-bit) | 50-75% | Slight quality loss (Text/Flux 2) |
| **CPU Offloading** | Variable | Slower generation (CUDA Only) |
| **Attention Slicing** | 10-20% Peak | Slightly slower (MPS/Mac) |
| **Smaller Resolution** | Significant | Lower output quality |
| **Smaller Model Variant** | Variable | Different capabilities |

### Recommended Storage

For optimal model loading times:
- ✅ **NVMe SSD**: Recommended for model storage (~5-10s load times)
- ⚠️ **SATA SSD**: Acceptable but slower (~20-40s load times)
- ❌ **HDD**: Not recommended (~60-120s+ load times)

Models are typically cached in `~/.cache/huggingface/` after first download.


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
