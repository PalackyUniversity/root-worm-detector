"""Bounded, process-isolated prediction with device recovery and reusable models."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import multiprocessing as mp
import threading
import time

from logic.execution_policy import available_device, resources, worker_limit, recoverable_accelerator_error, recoverable_memory_error
from logic.model_registry import model_spec

_PIPELINE = None
_PIPELINE_KEY = None


def predict_job(path, model_id, device, threads, dpi=None):
    """Child-process entry point. No Qt, shared model state or sidecar writes."""
    global _PIPELINE, _PIPELINE_KEY
    import cv2
    import psutil
    import torch
    from threadpoolctl import threadpool_limits
    from logic.inference_pipeline import InferencePipeline
    key = (model_id, device, threads)
    cold = key != _PIPELINE_KEY
    try:
        cv2.setNumThreads(1)
        if device == "cpu":
            try:
                process = psutil.Process()
                process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS") else 5)
            except (psutil.Error, OSError):
                pass
        if cold:
            _PIPELINE = InferencePipeline(device, model_id=model_id, cpu_threads=threads)
            _PIPELINE_KEY = key
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with threadpool_limits(limits=threads):
            result = _PIPELINE.predict(path, dpi=dpi) if dpi is not None else _PIPELINE.predict(path)
        peak = torch.cuda.max_memory_reserved(device)/2**20 if device.startswith('cuda') else 0
        return dict(result=result, seconds=time.perf_counter()-start, cold=cold,
                    peak_mib=peak, rss_mib=psutil.Process().memory_info().rss/2**20)
    except Exception as error:
        return dict(error=f'{type(error).__name__}: {error}',
                    recoverable=(device != 'cpu' and recoverable_accelerator_error(error)) or recoverable_memory_error(error))


def process_executor():
    return ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn'))


class AdaptivePredictor:
    """One coordinator owns this instance; callbacks execute on that thread.

    Bounded waves simplify retries/cancel: every submitted job is drained before
    changing pool configuration, and successful images are never re-enqueued.
    """
    def __init__(self, device=None, executor_factory=None, max_workers=None, cpu_threads=4):
        self.device = device
        self._requested_device = device
        self._factory = executor_factory or process_executor
        self.max_workers = max_workers
        self.cpu_threads = cpu_threads
        self._pools = []
        self._model = None
        self._target = 1
        self._ceiling = None
        self._peak = 0
        self._rss = 700
        self._rates = {}
        self._best_width = 1
        self._best_rate = 0

    def close(self):
        pools, self._pools = self._pools, []
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)

    def _resize(self, width):
        while len(self._pools) > width:
            self._pools.pop().shutdown(wait=True, cancel_futures=True)
        while len(self._pools) < width:
            self._pools.append(self._factory())

    def run(self, paths, model_id, *, cancel=None, on_started=None, on_result=None, on_status=None, dpi_by_path=None):
        model_spec(model_id) # invalid models fail before starting processes
        cancel = cancel or threading.Event()
        on_started = on_started or (lambda path: None)
        on_result = on_result or (lambda path, result: None)
        on_status = on_status or (lambda device, workers, fallback: None)
        desired_device = self._requested_device or available_device()
        if self._model != model_id or self.device != desired_device:
            self.close()
            self._target, self._ceiling = 1, None
            self._rates, self._best_rate, self._best_width = {}, 0, 1
            self._peak, self._rss = 0, 700
            self._model = model_id
            self.device = desired_device
        pending = deque(dict.fromkeys(paths))
        fallback = False
        try:
            while pending and not cancel.is_set():
                snapshot = resources(self.device)
                cap = worker_limit(self.device, **snapshot, peak_mib=self._peak,
                                   active=max(1,len(self._pools)), rss_mib=self._rss)
                if self.device == "cpu":
                    cap = min(cap, max(1, snapshot["cores"]//self.cpu_threads))
                    if not self._pools and self._target == 1 and self._ceiling is None:
                        # CPU model working sets are bounded by the RAM policy;
                        # do not spend a whole slow plate ramping from one core group.
                        self._target = cap
                cap = min(cap, self.max_workers or cap, self._ceiling or cap)
                desired = max(1, min(self._target, cap))
                width = min(desired, len(pending))
                self._resize(min(desired, max(width, len(self._pools))))
                on_status(self.device, width, fallback)
                futures = {}
                submission_errors = []
                for pool in self._pools[:width]:
                    if cancel.is_set():break
                    path = pending.popleft()
                    on_started(path)
                    try:
                        args = (path, model_id, self.device, min(snapshot["cores"], self.cpu_threads) if self.device == "cpu" else 4)
                        if dpi_by_path is not None:
                            args += (dpi_by_path.get(path),)
                        future = pool.submit(predict_job, *args)
                    except BrokenProcessPool as error:
                        submission_errors.append((path, dict(error=str(error), recoverable=True)))
                        break
                    futures[future] = path
                errors, durations, cold = submission_errors, [], False
                for future in as_completed(futures):
                    path = futures[future]
                    try:
                        payload = future.result()
                    except BrokenProcessPool as error:
                        payload = dict(error=str(error), recoverable=True)
                    if 'error' in payload:
                        errors.append((path,payload))
                        continue
                    durations.append(payload['seconds'])
                    cold = cold or payload.get('cold', False)
                    self._peak = max(self._peak, payload['peak_mib'])
                    self._rss = max(self._rss, payload['rss_mib'])
                    on_result(path, payload['result'])
                if errors:
                    allocated_workers = len(self._pools)
                    self.close() # a failed CUDA context must not be reused
                    if cancel.is_set():break
                    fatal = next((value for _,value in errors if not value['recoverable']), None)
                    if fatal:raise RuntimeError(fatal['error'])
                    pending.extendleft(path for path,_ in reversed(errors))
                    self._rates, self._best_rate = {}, 0
                    if allocated_workers > 1:
                        self._target = self._ceiling = max(1, min(width, allocated_workers//2))
                    elif self.device != 'cpu':
                        self.device = 'cpu'
                        self._target, self._ceiling, self._peak = 1, None, 0
                        fallback = True
                    else:
                        raise RuntimeError(errors[0][1]['error'])
                    continue
                # Warm service times exclude process/model startup; require a
                # full warm wave before judging scaling at a concurrency level.
                if durations and len(durations) == width == desired and not cold:
                    rate = width/max(durations)
                    if rate > self._best_rate*1.05:
                        self._best_rate, self._best_width = rate, width
                    elif width > self._best_width:
                        self._target = self._ceiling = self._best_width
                        continue
                    self._rates[width] = rate
                    self._target = min(width+1, cap) if pending else width
                elif width == desired == 1 and durations:
                    # First real image also establishes a memory estimate.
                    self._target = 2 if len(pending) >= 2 else 1
        except Exception:
            self.close()
            raise
