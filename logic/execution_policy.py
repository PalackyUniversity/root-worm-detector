"""Conservative resource bounds; decisions use free memory, not GPU model names."""
import os
import psutil


def available_device():
    import torch
    try:
        if torch.cuda.is_available():
            return 'cuda:0'
    except RuntimeError:
        pass
    try:
        if torch.backends.mps.is_available():
            return 'mps'
    except (AttributeError, RuntimeError):
        pass
    return 'cpu'


def resources(device):
    import torch
    try:
        cores = min(len(psutil.Process().cpu_affinity()), psutil.cpu_count(logical=False) or 1)
    except (AttributeError, psutil.Error):
        cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    free, total = None, 0
    if device.startswith('cuda'):
        try:
            free, total = (value / 2**20 for value in torch.cuda.mem_get_info(device))
        except (RuntimeError, AssertionError):
            pass
    return dict(cores=cores, ram_mib=psutil.virtual_memory().available / 2**20,
                free_gpu_mib=free, total_gpu_mib=total)


def worker_limit(device, cores, ram_mib, free_gpu_mib, total_gpu_mib,
                 peak_mib, active=1, rss_mib=700):
    core_cap = max(1, cores - 1)
    ram_slots = max(0, int((ram_mib - max(512, ram_mib * .15)) / max(512, rss_mib * 1.5)))
    cap = min(core_cap, active + ram_slots)
    if device == 'cpu':
        return max(1, cap)
    if not device.startswith('cuda') or free_gpu_mib is None:
        return 1 # Unknown/unified GPU memory: first establish safe sequential execution.
    reserve = max(512, total_gpu_mib * .12)
    per_worker = max(768, (peak_mib + 512) * 1.35)
    extra = max(0, int((free_gpu_mib - reserve) / per_worker))
    return max(1, min(cap, active + extra))


def recoverable_accelerator_error(error):
    message = str(error).lower()
    if not isinstance(error, (RuntimeError, AssertionError, NotImplementedError)):
        return False
    return any(text in message for text in (
        'cuda out of memory', 'cuda error', 'cuda driver', 'cuda-capable',
        'not compiled with cuda', 'no nvidia driver', 'no kernel image',
        'cudnn_status', 'cublas_status', 'hip out of memory', 'hip error',
        'mps backend out of memory', 'not implemented for the mps',
        'not currently implemented for the mps', 'mps device not found'))


def recoverable_memory_error(error):
    if isinstance(error, MemoryError):
        return True
    return isinstance(error, RuntimeError) and any(marker in str(error).lower() for marker in (
        'defaultcpuallocator', 'std::bad_alloc', 'not enough memory: you tried to allocate'))
