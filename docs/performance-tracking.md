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

### 📂 Data Structure & Naming Convention

The `performance.json` file uses a specific pipe-separated key format to store metrics for unique combinations of model type, hardware, precision, and resolution.

**Key Format:**
`Model Name | Device | Precision (dtype) | Resolution`

**Example:**
```json
{
  "image": {
    "runwayml/stable-diffusion-v1-5|cuda|float16|512x512": {
      "average_time": 4.25,
      "average_cpu": 15.0,
      "average_ram": 6.5,
      "average_vram": 4.1,
      "average_gpu": 98.5
    }
  }
}
```

You can safely delete `performance.json` at any time to reset estimates.

### 🚫 Opting Out
If you prefer not to use this feature, you can completely disable the reading and writing of this file by using the `-npt` or `--no-performance-tracking` flag.

> **Temporary Files:** During test execution, temporary JSON files (e.g., `*-temp-performance.json`) are created to robustly track resource usage for each test. These files are automatically deleted as each test completes. This JSON IPC approach is used because tests run in isolated subprocesses where shared memory/global variables are not accessible by the parent runner.

---

**See also:** [Understanding Memory & Model Loading](safety-and-resources.md#understanding-memory--model-loading) - Learn why models require RAM/VRAM, storage recommendations, and platform differences between CUDA and MPS.