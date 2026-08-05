import logging
import threading
import contextlib
import torch

logger = logging.getLogger("LEPAUTE.Core") 
_mps_lock = threading.RLock()

@contextlib.contextmanager
def mps_safe(device=None):
    """
    Thread-safe context manager for Apple Silicon MPS operations to ensure synchronized,
    race-condition-free tensor computations with atomic execution boundaries and device synchronization.
    """
    is_mps = False
    if device is not None:
        if isinstance(device, str):
            is_mps = (device == "mps")
        elif isinstance(device, torch.device):
            is_mps = (device.type == "mps")
    else:
        is_mps = torch.backends.mps.is_available()

    if is_mps:
        with _mps_lock:
            yield
            if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
    else:
        yield