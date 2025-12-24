# Troubleshooting

Common issues and solutions for AI Media.

← [Back to Main README](../README.md)

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
