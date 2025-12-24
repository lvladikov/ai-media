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
