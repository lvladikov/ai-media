# Performance Tracking

The tool includes a smart performance tracking system designed to help you plan your work.

← [Back to Main README](../README.md)

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
- ✅ Average GPU Load (%) - *NVIDIA (CUDA) only*
- ✅ Average VRAM Usage (GB) - *NVIDIA (CUDA) and Apple Silicon (MPS)*

You can safely delete `performance.json` at any time to reset estimates.

### 🚫 Opting Out
If you prefer not to use this feature, you can completely disable the reading and writing of this file by using the `-npt` or `--no-performance-tracking` flag.
