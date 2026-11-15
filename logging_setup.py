import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

ERROR_LOG_PATH = Path(__file__).resolve().parent / "runtime_errors.log"

def log_exception(context, exc_type, exc_value, exc_tb):
    """Log exceptions safely to console and file, bypassing fragile sys.excepthook output."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"[{timestamp}] {context}\n"
    trace = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    payload = f"{header}{trace}\n"

    try:
        with ERROR_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(payload)
    except Exception:
        pass

    try:
        sys.__stderr__.write(payload)
        sys.__stderr__.flush()
    except Exception:
        try:
            sys.__stderr__.buffer.write(payload.encode("utf-8", "backslashreplace"))
            sys.__stderr__.buffer.flush()
        except Exception:
            pass

def _global_excepthook(exc_type, exc_value, exc_tb):
    log_exception("Unhandled exception (main thread)", exc_type, exc_value, exc_tb)

def _thread_excepthook(args):
    thread_name = args.thread.name if args.thread else "unknown"
    log_exception(
        f"Unhandled exception (thread: {thread_name})",
        args.exc_type,
        args.exc_value,
        args.exc_traceback,
    )
    
def install_exception_hooks():
    sys.excepthook = _global_excepthook
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_excepthook