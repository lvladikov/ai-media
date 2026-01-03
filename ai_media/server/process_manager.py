"""Process manager for job execution with true cancellation support.

Spawns jobs in separate processes that can be terminated via SIGTERM.
Child processes ignore SIGINT so Ctrl+C only affects the server.
"""

import multiprocessing
import signal
import os
from typing import Dict, Callable, Any, Tuple
from multiprocessing import Queue
import threading
import sys
import io


class StreamLogger(io.StringIO):
    """Redirects stdout/stderr to both original stream and progress queue."""
    def __init__(self, stream, queue, job_id, pipe_name="stdout"):
        super().__init__()
        self.stream = stream
        self.queue = queue
        self.job_id = job_id
        self.pipe_name = pipe_name

    def write(self, message):
        # Write to original stream (terminal)
        self.stream.write(message)
        self.stream.flush()  # Ensure immediate terminal output
        
        # Send to queue if it's a non-empty line
        if message and message.strip():
            try:
                # Send as special "log_line" type
                self.queue.put({
                    "job_id": self.job_id,
                    "log_line": message.rstrip()
                })
            except Exception:
                pass

    def flush(self):
        self.stream.flush()


# Global registry of active job processes
job_processes: Dict[str, multiprocessing.Process] = {}

# Queue for progress updates from child processes
progress_queue: Queue = None


def _init_child():
    """Initialize child process - ignore SIGINT so only parent handles Ctrl+C."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _child_wrapper(target: Callable, args: Tuple, progress_queue: Queue, job_id: str):
    """Wrapper that runs in child process with signal handling."""
    _init_child()
    
    
    # Redirect stdout/stderr to capture logs
    if progress_queue:
        sys.stdout = StreamLogger(sys.stdout, progress_queue, job_id, "stdout")
        sys.stderr = StreamLogger(sys.stderr, progress_queue, job_id, "stderr")

    # Inject progress_queue into args if the target expects it
    # The target functions will check for this queue and use it for updates
    try:
        target(*args, progress_queue=progress_queue)
    except TypeError:
        # Fallback if target doesn't accept progress_queue
        target(*args)


def spawn_job_process(job_id: str, target: Callable, args: Tuple) -> multiprocessing.Process:
    """Spawn a job in a separate process that can be terminated.
    
    Args:
        job_id: Unique job identifier
        target: Function to run in child process
        args: Arguments to pass to target
        
    Returns:
        The spawned Process object
    """
    global progress_queue
    
    if progress_queue is None:
        progress_queue = multiprocessing.Queue()
        # Start background thread to process updates
        _start_progress_listener()
    
    process = multiprocessing.Process(
        target=_child_wrapper,
        args=(target, args, progress_queue, job_id),
        daemon=False  # Not daemon so it can complete if server exits gracefully
    )
    process.start()
    job_processes[job_id] = process
    
    print(f"🚀 Spawned process {process.pid} for job {job_id[:8]}...")
    return process


def terminate_job_process(job_id: str, timeout: float = 3.0) -> bool:
    """Terminate a job's process by PID.
    
    Args:
        job_id: Job to terminate
        timeout: Seconds to wait for graceful termination before SIGKILL
        
    Returns:
        True if process was terminated, False if not found
    """
    if job_id not in job_processes:
        return False
        
    process = job_processes[job_id]
    
    if process.is_alive():
        print(f"🛑 Terminating process {process.pid} for job {job_id[:8]}...")
        process.terminate()  # SIGTERM
        process.join(timeout=timeout)
        
        # Force kill if still alive
        if process.is_alive():
            print(f"⚠️ Force killing process {process.pid}...")
            process.kill()  # SIGKILL
            process.join(timeout=1)
    
    # Clean up
    del job_processes[job_id]
    return True


def terminate_all_processes():
    """Terminate all active job processes. Called on server shutdown."""
    for job_id in list(job_processes.keys()):
        terminate_job_process(job_id, timeout=1.0)
    print("🧹 All job processes terminated")


def _start_progress_listener():
    """Start background thread to listen for progress updates from child processes."""
    def listener():
        from .jobs import update_job
        
        while True:
            try:
                msg = progress_queue.get()
                if msg is None:
                    break  # Shutdown signal
                    
                job_id = msg.get("job_id")
                if job_id:
                    # Remove job_id from msg before passing to update_job
                    update_data = {k: v for k, v in msg.items() if k != "job_id"}
                    update_job(job_id, **update_data)
            except Exception as e:
                print(f"⚠️ Progress listener error: {e}")
    
    thread = threading.Thread(target=listener, daemon=True)
    thread.start()


def send_progress(queue: Queue, job_id: str, **kwargs):
    """Helper for child processes to send progress updates.
    
    Args:
        queue: The progress queue passed to the child
        job_id: Job identifier
        **kwargs: Update fields (status, progress, message, phase, etc.)
    """
    if queue:
        try:
            queue.put({"job_id": job_id, **kwargs})
        except Exception:
            pass  # Queue may be closed
