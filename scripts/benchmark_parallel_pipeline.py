"""Compare independent image workers and bounded TIFF prefetch; no production edits."""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from unittest.mock import patch

from benchmark_pipeline import ROOT, compare, differences
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
import logic.inference_pipeline as pipeline_module
from logic import measurement
from logic.image_logic import ImageLogic
from sahi.utils.cv import read_image_as_pil

PIPELINE = None
BARRIER = None


def initialize(barrier):
    global PIPELINE, BARRIER
    PIPELINE = pipeline_module.InferencePipeline('cuda:0')
    BARRIER = barrier


def warm(path):
    PIPELINE.predict(path)
    torch.cuda.synchronize()
    BARRIER.wait(timeout=180)
    return os.getpid()


def chunk(paths, mode, output):
    results = []
    original_sliced = pipeline_module.get_sliced_prediction
    cached = {}
    def sliced(path, model, **kwargs):
        return original_sliced(cached['rgb'], model, **kwargs)
    def read(path):
        return cv2.cvtColor(np.array(cached['rgb']), cv2.COLOR_RGB2BGR)
    torch.cuda.reset_peak_memory_stats()
    with ExitStack() as stack:
        if mode != 'baseline':
            stack.enter_context(patch.object(pipeline_module, 'get_sliced_prediction', sliced))
            stack.enter_context(patch.object(measurement, 'read_image', read))
        loader = stack.enter_context(ThreadPoolExecutor(max_workers=1)) if mode == 'prefetch' else None
        future = loader.submit(read_image_as_pil, str(paths[0])) if loader and paths else None
        for index, path in enumerate(paths):
            begin = time.perf_counter()
            if loader:
                cached['rgb'] = future.result()
                future = loader.submit(read_image_as_pil, str(paths[index+1])) if index+1 < len(paths) else None
            elif mode != 'baseline':
                cached['rgb'] = read_image_as_pil(str(path))
            result = PIPELINE.predict(path)
            ImageLogic.save_image_data(dict(result, path=str(Path(output)/Path(path).name)))
            results.append(dict(file=Path(path).name, result=result, seconds=time.perf_counter()-begin))
    torch.cuda.synchronize()
    return dict(results=results, pid=os.getpid(), peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)


class Monitor:
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
    def sample(self):
        while not self.stop.is_set():
            text = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'], text=True)
            self.samples.append([int(v.strip()) for v in text.splitlines()[0].split(',')])
            self.stop.wait(.5)
    def __enter__(self):
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self
    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/parallel_pipeline_benchmark')
    parser.add_argument('--workers', type=int, nargs='+', default=[1,2,3])
    parser.add_argument('--limit', type=int, default=6)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(ROOT/'train/runs/5wpi_nice_area/plates_5wpi.csv').sort_values('detected_count')
    table = table.iloc[np.linspace(0,len(table)-1,args.limit).astype(int)]
    paths = [str(ROOT/'data-test'/r.folder/r.file) for r in table.itertuples()]
    for path in paths:
        with Image.open(path) as im:
            assert im.getexif().get(274,1) == 1
    context = mp.get_context('spawn')
    references, rows, startup = {}, [], {}
    with tempfile.TemporaryDirectory() as tmp:
        for workers in args.workers:
            begin = time.perf_counter()
            with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                     initializer=initialize, initargs=(context.Barrier(workers),)) as pool:
                futures = [pool.submit(warm, paths[-1]) for _ in range(workers)]
                pids = [f.result() for f in futures]
                startup[str(workers)] = time.perf_counter()-begin
                print(f'{workers} workers ready in {startup[str(workers)]:.2f}s: {pids}', flush=True)
                if not references:
                    # Reference generated before all measured runs; no per-tile timing patches.
                    references = {r['file']:r['result'] for r in pool.submit(chunk, paths, 'baseline', tmp).result()['results']}
                modes = ['baseline','decode_once','prefetch'] if workers == 1 else ['baseline','decode_once']
                for repeat in range(args.repeats):
                    for mode in (modes if repeat%2 == 0 else list(reversed(modes))):
                        with Monitor() as monitor:
                            begin = time.perf_counter()
                            futures = [pool.submit(chunk, paths[i::workers], mode, tmp) for i in range(workers)]
                            completed = [f.result() for f in futures]
                            elapsed = time.perf_counter()-begin
                        objects = [r for worker in completed for r in worker['results']]
                        parity = [dict(file=r['file'], seconds=r['seconds'],
                                       parity=compare(references[r['file']], r['result']),
                                       differences=differences(references[r['file']], r['result'])) for r in objects]
                        row = dict(workers=workers, mode=mode, repeat=repeat, elapsed_seconds=elapsed,
                                   throughput_images_per_second=len(paths)/elapsed,
                                   effective_seconds_per_image=elapsed/len(paths), images=parity,
                                   worker_memory=[{k:v for k,v in worker.items() if k!='results'} for worker in completed],
                                   gpu_samples=monitor.samples)
                        rows.append(row)
                        (args.output/'measurements.json').write_text(json.dumps(dict(startup_seconds=startup, runs=rows), indent=2))
                        final_same = all(all(v for k,v in r['parity'].items() if k!='exact') for r in parity)
                        print(f'{workers} workers {mode} repeat {repeat+1}: {elapsed:.3f}s total, {elapsed/len(paths):.3f}s/image; final outputs equal={final_same}', flush=True)
    summary = []
    for workers,mode in sorted({(r['workers'],r['mode']) for r in rows}):
        group = [r for r in rows if (r['workers'],r['mode']) == (workers,mode)]
        checks = [x for r in group for x in r['images']]
        summary.append(dict(workers=workers, mode=mode,
                            mean_seconds_per_image=float(np.mean([r['effective_seconds_per_image'] for r in group])),
                            final_parity_passes=sum(all(v for k,v in x['parity'].items() if k!='exact') for x in checks),
                            strict_parity_passes=sum(x['parity']['exact'] for x in checks), total_checks=len(checks),
                            max_gpu_memory_mib=max(s[0] for r in group for s in r['gpu_samples']),
                            mean_gpu_utilization=float(np.mean([s[1] for r in group for s in r['gpu_samples']]))))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
