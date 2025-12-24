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
python ai-media.py --unittests tests.ai-media_test.TestParseSize

# Run a specific test class (Verbose)
python ai-media.py --unittests-verbose tests.ai-media_test.TestParseSize

# Run a specific test method
python ai-media.py --unittests tests.ai-media_test.TestParseSize.test_resolution_presets_standard

# Run a specific test method (Verbose)
python ai-media.py --unittests-verbose tests.ai-media_test.TestParseSize.test_resolution_presets_standard
```

## Integration Tests

Run integration tests defined in `tests/integration-tests.json`.

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
python ai-media.py --test "Image - SDXL (Default)"

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

### Skipping Tests

You can permanently skip a specific test by adding `"skip": true` to its definition in `tests/integration-tests.json`.
The test runner will report these as skipped in the final summary.

```json
{
  "name": "Image - Auto Filename",
  "skip": true,
  "command": "..."
}
```

> [!NOTE]
> The test name must match exactly what is defined in `tests/integration-tests.json`. If the name is not found, the script will list all available tests.

## Combined Test Execution

You can run both Unit and Integration tests in a single command, mixing verbosity levels as needed. This is useful for verifying internal logic and pipeline execution at once:

```bash
# Run All Unit Tests (Summary) + All Integration Tests (Summary)
python ai-media.py --unittests --test

# Run All Unit Tests (Verbose) + All Integration Tests (Verbose)
python ai-media.py --unittests-verbose --test-verbose

# Run Specific Unit Test Class + All Integration Tests (Verbose)
python ai-media.py --unittests tests.ai-media_test.TestParseSize --test-verbose

# Run All Unit Tests (Verbose) + Specific Integration Test
python ai-media.py --unittests-verbose --test "Image - SDXL (Default)"

# Run Unit Tests (Summary) + Multiple Specific Integration Tests
python ai-media.py --unittests --test "Image - Auto Filename" "Audio - Bark TTS"

# Run Specific Unit Method (Verbose) + Specific Video Integration Test
python ai-media.py --unittests-verbose tests.ai-media_test.TestParseDuration.test_colon_format_hms --test "Video - Zeroscope (Default)"
```

## Test Files

| File/Folder | Description |
| :--- | :--- |
| `tests/ai-media_test.py` | Unit tests for parsing, helpers, and classes |
| `tests/integration-tests.json` | Test configurations (commands, expected outputs) |
| `tests/testData/inputs/` | Sample input files for tests |
| `tests/testData/outputs/` | Generated outputs (git-ignored) |

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

Included in the `tests/` directory is a script to verify your system's hardware and software encoding limits.

```bash
python tests/test_codec_limits.py
```

This tool will:
- Detect your acceleration platform (NVIDIA CUDA or MacOS MPS).
- Stress test H.264, HEVC, and AV1 encoders.
- Check resolutions from 4K up to 20K.
- Provide a summary of which resolutions your hardware can handle vs. software fallback.

← [Back to Main README](../README.md)
