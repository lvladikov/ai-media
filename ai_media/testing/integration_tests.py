"""
Testing module for AI-Media.

Provides test discovery and execution for unit and integration tests.
Full-featured test runner matching OLD ai-media.py behavior:
- Warning prompt before running
- Input file existence checks
- JSON report aggregation
- Resource tracking (RAM/VRAM/CPU/GPU)
- Suite summary with averages
- Windows process group support
- Verbose real-time streaming
- CTRL+C subprocess cleanup
"""

import os
import sys
import subprocess
import signal
import json
import time
import shlex
import tempfile
import fnmatch
from pathlib import Path
from datetime import datetime

# Import emoji helper for safe output on Windows terminals
from ..utils.interaction import emoji


def get_test_dir():
    """Get the testing directory path."""
    return Path(__file__).parent


def find_unit_tests():
    """Return available unit test modules."""
    # Now we have a single monolithic unit test file in testing dir
    return ["unit_tests"]


def find_integration_tests():
    """Load integration test definitions from JSON."""
    test_file = get_test_dir() / "integration-tests.json"
    
    if not test_file.exists():
        return []
    
    try:
        with open(test_file, "r") as f:
            data = json.load(f)
        return data.get("tests", [])
    except (json.JSONDecodeError, Exception):
        return []


def get_current_platform():
    """Detect the current compute platform.
    
    Returns:
        str: 'cuda', 'mps', or 'cpu'
    """
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    return 'cpu'


def get_current_dtype():
    """Detect the current optimal dtype based on platform.
    
    Uses centralized detection from ai_media.utils.system.
    
    Returns:
        str: 'bfloat16', 'float16', or 'float32'
    """
    try:
        from ai_media.utils.system import is_bfloat16_supported
        import torch
        if torch.cuda.is_available():
            # Use centralized detection
            if is_bfloat16_supported():
                return 'bfloat16'
            return 'float16'
        if torch.backends.mps.is_available():
            return 'float32'  # MPS uses float32 for stability
    except (ImportError, AttributeError):
        pass
    return 'float32'


def should_run_test(test_config, current_platform=None):
    """Check if a test should run on the current platform.
    
    Args:
        test_config: Test configuration dict with optional 'runOn' key
        current_platform: Current platform ('cuda', 'mps', 'cpu') or None to auto-detect
        
    Returns:
        tuple: (should_run: bool, reason: str or None)
        
    runOn values (compute):
        - 'all' or not specified: Run on all platforms (default)
        - 'cuda': Only run on NVIDIA CUDA GPUs (Windows/Linux)
        - 'mps': Only run on Apple Silicon MPS
        - 'cpu': Only run on CPU
        - 'gpu': Run on any GPU (cuda or mps)
        
    runOn values (OS):
        - 'mac': Run on macOS (regardless of MPS/CPU)
        - 'windows': Run on Windows
        - 'linux': Run on Linux
        
    Combinations supported:
        - 'cuda,mps': Run on either cuda or mps (comma-separated)
        - 'mac,linux': Run on Mac or Linux
    """
    import sys
    
    if current_platform is None:
        current_platform = get_current_platform()
    
    run_on = test_config.get("runOn", "all").lower().strip()
    
    # Default: run on all platforms
    if run_on == "all" or run_on == "":
        return True, None
    
    # Parse comma-separated list
    allowed_platforms = [p.strip() for p in run_on.split(",")]
    
    # Check for 'gpu' shorthand (cuda or mps)
    if "gpu" in allowed_platforms:
        allowed_platforms.extend(["cuda", "mps"])

    # Check for 'mlx' - allow if on Mac and mlx is importable
    if "mlx" in allowed_platforms:
        try:
            import mlx.core
            allowed_platforms.append(current_platform) # Allow current if valid
            # Or better: check if we are on mac
            if sys.platform != 'darwin':
                 return False, f"runOn={run_on} (MLX requires macOS)"
            # If on mac, we can run it, assuming current_platform is 'mps' or 'cpu'
            # We just need to return True if we are on Mac
            return True, None
        except ImportError:
            return False, f"runOn={run_on} (mlx module not found)"
    
    # Check for OS-specific filters first
    current_os = sys.platform  # 'darwin', 'win32', 'linux'
    os_mapping = {
        'mac': 'darwin',
        'macos': 'darwin',
        'darwin': 'darwin',
        'windows': 'win32',
        'win': 'win32',
        'win32': 'win32',
        'linux': 'linux',
    }
    
    # Check if any OS filter is specified
    os_filters = [p for p in allowed_platforms if p in os_mapping]
    if os_filters:
        allowed_os_values = [os_mapping[p] for p in os_filters]
        if current_os in allowed_os_values:
            return True, None
        # If only OS filters are specified and we don't match, skip
        compute_filters = [p for p in allowed_platforms if p in ['cuda', 'mps', 'cpu', 'gpu']]
        if not compute_filters:
            return False, f"runOn={run_on} (current OS: {current_os})"
    
    # Check compute platform (cuda/mps/cpu)
    if current_platform in allowed_platforms:
        return True, None
    else:
        return False, f"runOn={run_on} (current: {current_platform})"


def run_unit_test(test_name, verbose=True):
    """Run a single Python unittest module.
    
    Args:
        test_name: Name of test module (e.g. 'unit_tests')
        verbose: Show detailed output
        
    Returns:
        bool: True if test passed
    """
    if test_name == "unit_tests" or test_name == "ai_media.unit_tests" or test_name == "ai_media.testing.unit_tests":
        test_path = Path(__file__).parent / "unit_tests.py"
    else:
        # Fallback for legacy
        test_path = get_test_dir() / f"{test_name}.py"
    
    if not test_path.exists():
        print(f"{emoji('❌ ', 'X ')}Test not found: {test_name}")
        return False
    
    print(f"{emoji('🧪 ', '')}Running: {test_name}")
    
    args = [sys.executable, "-m", "unittest", str(test_path)]
    if verbose:
        args.append("-v")
    
    # Run from project root
    result = subprocess.run(args, cwd=str(get_test_dir().parent.parent))
    return result.returncode == 0


def run_all_unit_tests():
    """Run all unit tests."""
    # Just run the main unit_tests module
    if run_unit_test("unit_tests"):
        print(f"\n{emoji('✅ ', '')}All unit tests passed.")
    else:
        print(f"\n{emoji('❌ ', 'X ')}Unit tests failed.")


def format_time(seconds):
    """Convert seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.0f}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m {int(s)}s"


def run_integration_test(test_config, script_path, verbose=False, test_state=None, json_report_path=None):
    """Run a single integration test.
    
    Args:
        test_config: Test configuration dict from JSON
        script_path: Path to main ai-media.py script
        verbose: Stream output in real-time if True
        test_state: Global test state dict for CTRL+C handling
        json_report_path: Path for JSON performance report
        
    Returns:
        dict: {passed: bool, elapsed: float, reason: str, resources: dict}
    """
    name = test_config.get("name", "Unknown")
    command = test_config.get("command", "")
    expected_stdout = test_config.get("expectedStdoutItems", [])
    expected_outputs = test_config.get("expectedOutputItems", [])
    expected_inputs = test_config.get("expectedInputItems", [])
    is_interactive = test_config.get("interactive", False)
    interactive_wait = test_config.get("interactiveWait", 3.0)
    timeout_limit = test_config.get("timeout", 600)
    description = test_config.get("description", "")
    
    script_dir = Path(script_path).parent
    result = {"passed": True, "elapsed": 0, "reason": None, "resources": {}}
    
    # Skip if flagged
    if test_config.get("skip") is True:
        print(f"   ⏭️ Skipped (skip: true)")
        return {"passed": None, "elapsed": 0, "reason": "skipped", "resources": {}}
    
    # 1. Check expected input items exist
    for input_item in expected_inputs:
        input_path = script_dir / input_item
        if not input_path.exists():
            print(f"   ❌ Missing input: {input_item}")
            print(f"   ⏭️ Skipping due to missing inputs")
            return {"passed": False, "elapsed": 0, "reason": f"Missing input: {input_item}", "resources": {}}
    
    # 2. Delete expected outputs before run (clean slate)
    for output_item in expected_outputs:
        output_path = script_dir / output_item
        if output_path.exists():
            output_path.unlink()
            if verbose:
                print(f"   🗑️ Deleted: {output_item}")
    
    # Build full command
    full_command = [sys.executable, "-u", str(script_path)] + shlex.split(command)
    
    # Add JSON report arg if path provided
    if json_report_path:
        full_command.extend(["--report-json", json_report_path])
    
    start_time = time.time()
    stdout_lines = []
    current_process = None
    
    try:
        # Set UTF-8 for subprocess
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        # Windows: create subprocess in new process group for clean CTRL+C
        creation_flags = 0
        if os.name == 'nt':
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
        
        current_process = subprocess.Popen(
            full_command,
            cwd=str(script_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            stdin=subprocess.PIPE if is_interactive else None,  # PIPE allows menu to render before stdin close
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
            bufsize=1,
            universal_newlines=True,
            creationflags=creation_flags if os.name == 'nt' else 0,
        )
        
        # Track subprocess for signal handler cleanup
        if test_state:
            test_state['current_process'] = current_process
        
        if is_interactive:
            # Close stdin immediately so msvcrt.getch() gets EOF rather than hanging
            if current_process.stdin:
                current_process.stdin.close()
            
            # Poll stdout for expected content, terminate early when found
            if verbose:
                print(f"   {emoji('⏳ ', '')}Waiting {interactive_wait}s for interactive output...")
            
            collected_output = []
            start_wait = time.time()
            expected_stdout = test_config.get("expectedStdoutItems", [])
            
            # Use threaded reader for cross-platform compatibility
            import threading
            
            def read_output():
                try:
                    while True:
                        line = current_process.stdout.readline()
                        if not line:
                            break
                        collected_output.append(line)
                except Exception:
                    pass
            
            reader_thread = threading.Thread(target=read_output, daemon=True)
            reader_thread.start()
            
            # Wait and check for expected content periodically
            while (time.time() - start_wait) < interactive_wait:
                time.sleep(0.3)
                # Check if we found all expected items
                combined = ''.join(collected_output)
                if expected_stdout and all(item in combined for item in expected_stdout):
                    if verbose:
                        elapsed = time.time() - start_wait
                        print(f"   {emoji('✓ ', 'OK ')}Found stdout: '{expected_stdout[0]}' (after {elapsed:.1f}s)")
                    break
                # Check if process has exited
                if current_process.poll() is not None:
                    break
            
            # Terminate gracefully
            if os.name == 'nt':
                current_process.terminate()
            else:
                current_process.send_signal(signal.SIGINT)
            
            try:
                remaining_stdout, _ = current_process.communicate(timeout=5)
                if remaining_stdout:
                    collected_output.append(remaining_stdout)
            except subprocess.TimeoutExpired:
                current_process.kill()
                try:
                    current_process.communicate()
                except (OSError, IOError, ValueError):
                    pass  # Mac/Linux: Pipes may already be closed after kill
            except (OSError, IOError, ValueError):
                # Mac/Linux: After SIGINT, pipes may already be closed
                # ValueError can occur when communicate() tries to flush closed stdin
                # This is expected behavior on Unix after termination
                pass
            
            stdout = ''.join(collected_output)
            stdout_lines = [stdout] if stdout else []
        else:
            # Non-interactive: Stream output real-time if verbose
            start_read_time = time.time()
            
            while True:
                # Check for timeout
                if time.time() - start_read_time > timeout_limit:
                    current_process.kill()
                    raise subprocess.TimeoutExpired(full_command, timeout_limit)
                
                line = current_process.stdout.readline()
                if not line and current_process.poll() is not None:
                    break
                
                if line:
                    # Stream to user if verbose mode is on
                    if verbose:
                        print(line, end='', flush=True)
                    stdout_lines.append(line)
            
            current_process.wait()
        
        elapsed = time.time() - start_time
        result["elapsed"] = elapsed
        stdout = "".join(stdout_lines)
        
        # Show verbose end marker
        if verbose and not is_interactive:
            print(f"\n--- END ({elapsed:.1f}s) ---\n")
        
        # Check return code
        if current_process.returncode != 0 and not is_interactive:
            # Check if it was a Ctrl+C interrupt
            is_ctrl_c = current_process.returncode in [130, -2, 3221225786, -1073741510]
            
            if is_ctrl_c:
                print(f"\n\n{emoji('⚠️  ', '')}Interrupted! Cleaning up...")
                raise KeyboardInterrupt()
            else:
                print(f"   {emoji('❌ ', 'X ')}Command failed with exit code {current_process.returncode}")
                result["passed"] = False
                result["reason"] = f"Exit code {current_process.returncode}"
                return result
        
        # Check expected stdout items
        for expected in expected_stdout:
            if expected not in stdout:
                print(f"   {emoji('❌ ', 'X ')}Missing stdout: '{expected}'")
                result["passed"] = False
                result["reason"] = f"Missing stdout: '{expected}'"
                return result
            elif verbose:
                print(f"   {emoji('✓ ', '')}Found stdout: '{expected}'")
        
        # Check expected output files exist
        for output_item in expected_outputs:
            output_path = script_dir / output_item
            if not output_path.exists():
                print(f"   {emoji('❌ ', 'X ')}Missing output file: {output_item}")
                result["passed"] = False
                result["reason"] = f"Missing output: {output_item}"
                return result
            else:
                print(f"   {emoji('✓ ', '')}Output exists: {output_item}")
        
        return result
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        result["elapsed"] = elapsed
        result["passed"] = False
        result["reason"] = f"Timeout ({timeout_limit}s)"
        print(f"   {emoji('⏱️ ', '')}{{result['reason']}}")
        return result
        
    except KeyboardInterrupt:
        # Kill subprocess and re-raise
        if current_process:
            pid = current_process.pid
            print(f"\n   {emoji('⚠️ ', 'Warning ')} Killing subprocess tree (PID {pid})...")
            
            if os.name == 'nt':
                try:
                    subprocess.run(['taskkill', '/T', '/F', '/PID', str(pid)], 
                                   capture_output=True, timeout=5)
                except:
                    try:
                        current_process.kill()
                    except:
                        pass
            else:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except:
                    try:
                        current_process.kill()
                    except:
                        pass
            
            try:
                current_process.wait(timeout=2)
            except:
                pass
        raise
        
    except Exception as e:
        elapsed = time.time() - start_time
        result["elapsed"] = elapsed
        result["passed"] = False
        result["reason"] = str(e)
        print(f"   {emoji('❌ ', 'Error: ')}Error: {e}")
        return result


def run_tests(test_type="all", verbose=True, test_filter=None, exit_on_finish=True):
    """Run tests of specified type.
    
    Args:
        test_type: 'unit', 'integration', or 'all'
        verbose: Show detailed output
        test_filter: List of test names to run (integration only)
        exit_on_finish: Call sys.exit with result code
    """
    # Import test state for CTRL+C handling
    try:
        from ai_media.utils.system import _test_state
    except ImportError:
        _test_state = {'active': False, 'passed': 0, 'failed': 0, 'total': 0, 'current_process': None}
    
    if test_type in ["unit", "all"] and not test_filter:
        print("\n" + "="*50)
        print("UNIT TESTS")
        print("="*50 + "\n")
        run_all_unit_tests()
    
    if test_type in ["integration", "all"]:
        print("\n" + "="*50)
        print("INTEGRATION TESTS")
        print("="*50 + "\n")
        
        tests = find_integration_tests()
        script_path = Path(__file__).parent.parent.parent / "ai-media.py"
        script_dir = script_path.parent
        
        if not tests:
            print("No integration tests found.")
            return
            
        if test_filter:
            original_count = len(tests)
            
            # Filter tests using exact match first, then glob patterns (case-insensitive)
            def matches_filter(test_name, patterns):
                """Check if test name matches any filter pattern (exact or glob, case-insensitive)."""
                test_lower = test_name.lower()
                for pattern in patterns:
                    pattern_lower = pattern.lower()
                    # First try exact match (case-insensitive)
                    if test_lower == pattern_lower:
                        return True
                    # Then try glob pattern match (supports *, ?, [seq], [!seq])
                    if fnmatch.fnmatch(test_lower, pattern_lower):
                        return True
                return False
            
            tests = [t for t in tests if matches_filter(t.get("name", ""), test_filter)]
            print(f" {emoji('🔎 ', '[Filtered] ')}{len(tests)} of {original_count} tests")
            if not tests:
                print(f"{emoji('❌ ', '[X] ')}No tests match filter: {test_filter}")
                print(f"   {emoji('💡 ', 'Tip: ')}Use glob patterns like 'Interactive*' to match multiple tests")
                return
        
        print(f"Found {len(tests)} integration tests.\n")
        
        # Warning prompt
        print(f"\n{'='*60}")
        print(f" {emoji('⚠️ ', '')} WARNING: Test Suite")
        print(f"{'='*60}")
        print(f"   • Integration tests can take a long time")
        print(f"   • Models will be downloaded if not present (2-30GB each)")
        print(f"   • High system resource consumption")
        print(f"   • Press CTRL+C at any time to interrupt")
        print(f"{'='*60}")
        
        if os.environ.get("AI_MEDIA_FORCE") != "1":
            try:
                choice = input(f"\n   Continue? [Y/n]: ").lower().strip()
                if choice in ['n', 'no']:
                    print(f"{emoji('❌ ', 'X ')}Test cancelled.")
                    if exit_on_finish:
                        sys.exit(0)
                    return
            except KeyboardInterrupt:
                print(f"\n{emoji('❌ ', 'X ')}Test cancelled.")
                if exit_on_finish:
                    sys.exit(0)
                return
        else:
            print(f"\n   (Skipping confirmation due to --force)\n")
        
        print(f"\n{'='*60}")
        print(f"{emoji('🧪 ', '')}Running {len(tests)} test(s)")
        print(f"{'='*60}\n")
        
        passed = 0
        failed = 0
        skipped = 0
        results = []
        
        # Resource aggregation
        total_ram = 0.0
        total_vram = 0.0
        total_cpu = 0.0
        total_gpu = 0.0
        resource_count = 0
        
        # Set global test state for CTRL+C handler
        _test_state['active'] = True
        _test_state['total'] = len(tests)
        _test_state['passed'] = 0
        _test_state['failed'] = 0
        
        suite_start_time = time.time()
        start_dt = datetime.fromtimestamp(suite_start_time)
        suite_timestamp = start_dt.strftime("%Y%m%d-%H%M%S-%f")[:-3]
        
        # Detect current platform for runOn filtering
        current_platform = get_current_platform()
        print(f"   Platform: {current_platform.upper()}\n")
        
        for i, test in enumerate(tests):
            test_name = test.get("name", f"Test {i+1}")
            description = test.get("description", "")
            
            # Check for skip flag early
            if test.get("skip") is True:
                print(f"\n{emoji('⏭️ ', '')}Skipping test: {test_name} (skip: true)")
                skipped += 1
                continue
            
            # Check runOn platform filter
            should_run, skip_reason = should_run_test(test, current_platform)
            if not should_run:
                print(f"\n{emoji('⏭️ ', '')}Skipping test: {test_name} ({skip_reason})")
                skipped += 1
                continue
            
            # Test header box
            start_t_str = datetime.now().strftime("%H:%M:%S")
            header = f"{emoji('📋 ', '')}Test {i+1}/{len(tests)}: {test_name}"
            desc = f"   {description}" if description else ""
            time_line = f"   Start at: {start_t_str}"
            
            lines = [header, desc, time_line]
            max_len = max(50, *[len(l) for l in lines if l])
            sep = "-" * max_len
            
            print(f"\n{sep}")
            print(header)
            if description:
                print(desc)
            print("")
            print(time_line)
            print(f"{sep}")
            
            # Display command
            cmd_display = f"python ai-media.py {test.get('command', '')}"
            print(f"{emoji('🚀 ', '')}Running: {cmd_display}")
            
            # Prepare JSON report path
            json_report_path = str(script_dir / f"{suite_timestamp}-{i+1:03d}-temp-performance.json")
            
            try:
                result = run_integration_test(
                    test, 
                    str(script_path), 
                    verbose=verbose, 
                    test_state=_test_state,
                    json_report_path=json_report_path
                )
            except KeyboardInterrupt:
                # Mark test as no longer active before re-raising
                _test_state['active'] = False
                raise
            
            elapsed = result.get("elapsed", 0)
            
            # Check if skipped
            if result.get("passed") is None:
                skipped += 1
                continue
            
            # Read JSON report if available
            if result.get("passed") and os.path.exists(json_report_path):
                try:
                    with open(json_report_path, 'r') as f:
                        stats = json.load(f)
                    
                    r_time = stats.get("time", elapsed)
                    r_ram = stats.get("ram", 0)
                    r_vram = stats.get("vram", 0)
                    r_cpu = stats.get("cpu", 0)
                    r_gpu = stats.get("gpu", 0)
                    
                    total_ram += r_ram
                    total_vram += r_vram
                    total_cpu += r_cpu
                    total_gpu += r_gpu
                    resource_count += 1
                    
                    result["resources"] = {"ram": r_ram, "vram": r_vram, "cpu": r_cpu, "gpu": r_gpu}
                except Exception as e:
                    if verbose:
                        print(f"   {emoji('⚠️ ', 'WARN: ')}Failed to read stats JSON: {e}")
            
            # Cleanup report file
            if os.path.exists(json_report_path):
                try:
                    os.remove(json_report_path)
                except:
                    pass
            
            if result.get("passed"):
                print(f"{emoji('✅ ', '')}PASSED ({elapsed:.1f}s)")
                passed += 1
                _test_state['passed'] = passed
                results.append((test_name, True, f"{elapsed:.1f}s"))
            else:
                print(f"{emoji('❌ ', 'X ')}FAILED ({elapsed:.1f}s)")
                failed += 1
                _test_state['failed'] = failed
                results.append((test_name, False, result.get("reason", "Unknown")))
        
        # Mark test as no longer active
        _test_state['active'] = False
        
        total_duration = time.time() - suite_start_time
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"📊 TEST SUMMARY")
        print(f"{'='*60}")
        print(f"   Total:   {len(tests)}")
        print(f"   Passed:  {passed} {emoji('✅', '')}")
        
        if failed > 0:
            print(f"   Failed:  {failed} {emoji('❌', 'X')}")
        else:
            print(f"   Failed:  {failed}")
        
        if skipped > 0:
            print(f"   Skipped: {skipped} {emoji('⏭️', '')}")
        
        print(f"   Duration: {format_time(total_duration)}")
        
        if resource_count > 0:
            avg_ram = total_ram / resource_count
            avg_vram = total_vram / resource_count
            avg_cpu = total_cpu / resource_count
            avg_gpu = total_gpu / resource_count
            
            if len(tests) >= 2:
                print(f"\n   {emoji('⚖️ ', '')}Averages:\n")
            
            print(f"   RAM:  {avg_ram:.1f} GB")
            print(f"   VRAM: {avg_vram:.1f} GB")
            print(f"   CPU:  {avg_cpu:.1f} %")
            print(f"   GPU:  {avg_gpu:.1f} %")
        
        # Always show platform and dtype (system-level settings)
        current_dtype = get_current_dtype()
        print(f"   Platform: {current_platform.upper()}")
        print(f"   Dtype: {current_dtype}")
        
        print(f"{'='*60}")
        
        if failed > 0:
            print(f"\n{emoji('❌ ', 'X ')}Failed Tests:")
            for name, success, reason in results:
                if not success:
                    print(f"   - {name}: {reason}")
        
        print(f"\n{emoji('✅ ', '')}Test Run Complete")
        
        if exit_on_finish:
            sys.exit(0 if failed == 0 else 1)

# Alias for compatibility with ai-media.py
run_unit_tests = run_unit_test
