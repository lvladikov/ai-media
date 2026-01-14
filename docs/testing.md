# Testing

🧪 The included test suite covers unit tests (Python logic) and integration tests (full generation pipelines).

← [Back to Main README](../README.md)

## Unit Tests

Run the Python unit test suite to verify internal logic.

```bash
# Run all unit tests (Quiet/Summary mode - Default)
python ai-media.py --unittests

# Run all unit tests (Verbose mode - details for every test)
python ai-media.py --unittests-verbose

# Run a specific test class
python ai-media.py --unittests ai_media.testing.unit_tests.TestParseSize

# Run a specific test class (Verbose)
python ai-media.py --unittests-verbose ai_media.testing.unit_tests.TestParseSize

# Run a specific test method
python ai-media.py --unittests ai_media.testing.unit_tests.TestParseSize.test_resolution_presets_standard

# Run a specific test method (Verbose)
python ai-media.py --unittests-verbose ai_media.testing.unit_tests.TestParseSize.test_resolution_presets_standard
```

## Integration Tests

Run integration tests defined in `ai_media/testing/integration-tests.json`.

```bash
# Run all integration tests (Summary)
python ai-media.py --test

# Run all integration tests (Verbose real-time output)
python ai-media.py --test-verbose

# Run specific tests
python ai-media.py --test "Image - SDXL"
```

> **Note**: Test output is buffered to prevent console spam. For long-running tests (like downloading models), use `--test-verbose` to see progress in real-time.

### Single Test Execution

You can run a specific test by providing its name (exact match):

```bash
# Run specific test
python ai-media.py --test "Image - Default (Z-Image Turbo)"

# Run specific test with verbose output
python ai-media.py --test-verbose "Image - Auto Filename"
```

### Multiple Test Execution

You can run a specific subset of tests by passing them as a space-separated list:

```bash
# Run multiple specific tests
python ai-media.py --test "Validation - Image Generation" "Validation - Video Generation"
python ai-media.py --test-verbose "Validation - Image Generation" "Validation - Video Generation" "Validation - Audio Generation"
```

### Glob Pattern Filtering

Use glob patterns to match multiple integration tests by name pattern, avoiding the need to list each test individually:

> **Note:** Pattern matching is **case-insensitive** on all platforms.

#### Supported Patterns

| Pattern | Matches |
|---------|---------|
| `*` | Everything (zero or more characters) |
| `?` | Single character |
| `[seq]` | Any character in seq (e.g., `[ABC]`) |
| `[!seq]` | Any character NOT in seq |

```bash
# Run all Interactive tests (64 tests)
python ai-media.py --test "Interactive*"

# Same as above (case-insensitive on Windows)
python ai-media.py --test "interactive*"

# Run all Image tests
python ai-media.py --test "Image*"

# Run tests containing "SDXL"
python ai-media.py --test "*SDXL*"

# Run Jump 1 through Jump 9 (single digit)
python ai-media.py --test "Interactive - Jump ?"

# Match tests ending in numbers 1, 2, or 3
python ai-media.py --test "Interactive - Jump [123]"

# Match Jump 1 through Jump 9 but NOT Jump 10+
python ai-media.py --test "Interactive - Jump [1-9]"

# Match all Audio models EXCEPT Bark
python ai-media.py --test "Audio*[!k]"  # Excludes names ending in 'k'

# Match SD 1.5 or SD 2.x
python ai-media.py --test "*SD [12]*"
```

#### Interactive Mode

In the Integration Tests menu, select **"🔍 Run Tests by Pattern"** to enter a custom glob pattern interactively. This provides:
- Pattern syntax help
- Example patterns
- Immediate execution of matching tests

### Skipping Tests

You can permanently skip a specific test by adding `"skip": true` to its definition in `ai_media/testing/integration-tests.json`.
The test runner will report these as skipped in the final summary.

```json
{
  "name": "Image - Auto Filename",
  "skip": true,
  "command": "..."
}
```

### Platform Filtering (runOn)

You can restrict tests to specific compute platforms or operating systems using the `runOn` property. If not specified, tests run on all platforms (default: `"all"`).

#### Compute Platform Filters

| Value | Description | Use Case |
|-------|-------------|----------|
| `cuda` | NVIDIA CUDA GPUs only | Tests requiring CUDA-specific libraries (e.g., bitsandbytes 4-bit quantization) |
| `mps` | Apple Silicon MPS only | Tests specifically for Metal Performance Shaders |
| `cpu` | CPU only | Lightweight tests or CPU-specific behavior |
| `gpu` | Any GPU (CUDA or MPS) | Tests that need GPU acceleration but work on either platform |
| `all` | All platforms (default) | Most tests - runs everywhere |

#### Operating System Filters

| Value | Description |
|-------|-------------|
| `mac` / `macos` / `darwin` | macOS only (regardless of MPS/CPU device) |
| `windows` / `win` / `win32` | Windows only |
| `linux` | Linux only |

#### Combinations

Use comma-separated values for multiple platforms:

```json
{
  "name": "Image - FLUX.2 (4-bit Quantized)",
  "runOn": "cuda",
  "command": "-i -p \"test\" --image-model flux2 ...",
  "description": "CUDA-only: bitsandbytes requires NVIDIA GPU"
}
```

```json
{
  "name": "Some Mac/Linux Test",
  "runOn": "mac,linux",
  "command": "..."
}
```

The test runner displays the current platform at suite start and shows skip reasons:

```
   Platform: MPS

⏭️ Skipping test: Image - FLUX.2 (runOn=cuda, current: mps)
```

> [!NOTE]
> The test name must match exactly what is defined in `ai_media/testing/integration-tests.json`. If the name is not found, the script will list all available tests.

## Combined Test Execution

You can run both Unit and Integration tests in a single command, mixing verbosity levels as needed. This is useful for verifying internal logic and pipeline execution at once:

```bash
# Run All Unit Tests (Summary) + All Integration Tests (Summary)
python ai-media.py --unittests --test

# Run All Unit Tests (Verbose) + All Integration Tests (Verbose)
python ai-media.py --unittests-verbose --test-verbose

# Run Specific Unit Test Class + All Integration Tests (Verbose)
python ai-media.py --unittests ai_media.testing.unit_tests.TestParseSize --test-verbose

# Run All Unit Tests (Verbose) + Specific Integration Test
python ai-media.py --unittests-verbose --test "Image - Default (Z-Image Turbo)"

# Run Unit Tests (Summary) + Multiple Specific Integration Tests
python ai-media.py --unittests --test "Image - Auto Filename" "Audio - Bark TTS"

# Run Specific Unit Method (Verbose) + Specific Video Integration Test
python ai-media.py --unittests-verbose ai_media.testing.unit_tests.TestParseDuration.test_colon_format_hms --test "Video - Zeroscope (Default)"
```

## Test Files

| File/Folder | Description |
| :--- | :--- |
| `ai_media/testing/unit_tests.py` | Unit tests for parsing, helpers, and classes |
| `ai_media/testing/integration_tests.py` | Integration test runner and logic |
| `ai_media/testing/integration-tests.json` | Test configurations (commands, expected outputs) |
| `ai_media/testing/data/inputs/` | Sample input files for tests |
| `ai_media/testing/data/outputs/` | Generated outputs (git-ignored) |

> [!WARNING]
> - This may take a **long time** (30+ minutes)
> - Uses significant system resources (CPU, RAM, GPU)
> - Will download **all models** if not already cached (2-30GB each)
> - Press `CTRL+C` at any time to interrupt

> [!NOTE]
> **Output Buffering:** In default quiet mode (`--test`), the runner buffers output and detects "hanging" behavior during long operations (downloads). Using `--test-verbose` streams output in real-time, allowing you to monitor progress immediately.
>
> **Temporary Files:** During test execution, temporary JSON files (e.g., `*-temp-performance.json`) are created to robustly track resource usage for each test. These files are automatically deleted as each test completes. This JSON IPC approach is used because tests run in isolated subprocesses where shared memory/global variables are not accessible by the parent runner.

---

## Codec Analysis Tool

Included in the `ai_media/testing/` directory is a script to verify your system's hardware and software encoding limits.

```bash
python ai_media/testing/codec_limits_tests.py
```

This tool will:
- Detect your acceleration platform (NVIDIA CUDA or MacOS MPS).
- Stress test H.264, HEVC, and AV1 encoders.
- Check resolutions from 4K up to 20K.
- Provide a summary of which resolutions your hardware can handle vs. software fallback.

← [Back to Main README](../README.md)
