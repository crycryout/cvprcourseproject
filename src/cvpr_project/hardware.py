"""CPU-only checks of actual CUDA device identity and frozen timing scope."""


def require_device_scope(actual, reference=None, *, allow_mig=False):
    if "H800" not in actual["name"]:
        raise ValueError(f"This protocol requires an H800, selected {actual['name']}")
    if not allow_mig and "MIG" in actual["name"]:
        raise ValueError(
            f"Full H800 required, selected {actual['name']} with {actual['multiprocessors']} SMs "
            f"and {actual['total_memory_bytes']/1e9:.2f} GB. Numeric CUDA and nvidia-smi indices "
            "are not interchangeable when MIG is enabled. Select a verified full GPU UUID; "
            "do not change MIG mode or other jobs automatically.")
    if reference is not None and actual != reference:
        raise ValueError("Actual CUDA device/software differs from the pilot or frozen reference; "
                         "preserve prior evidence and recalibrate before comparing policies")
