"""Benchmark deployed inference without editing models, source images or sidecars.

Experimental variants are process-local patches, never production settings.
"""
import argparse
from collections import defaultdict
from contextlib import ExitStack
from importlib.metadata import version
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
import logic.inference_pipeline as pipeline_module
from logic import measurement
from logic.image_logic import ImageLogic
from sahi.utils.cv import read_image_as_pil


def canonical(result):
    return dict(scores=result['scores'], contours=[c.tolist() for c in result['contours']],
                measurements=result['measurements'], dpi=result['dpi'])


def compare(reference, result):
    a, b = canonical(reference), canonical(result)
    exact = a == b
    records_a, records_b = a['measurements'], b['measurements']
    same_centres = [(r['x'], r['y']) for r in records_a] == [(r['x'], r['y']) for r in records_b]
    fields = ['nice', 'nice_probability', 'area_mm2', 'mask', 'detector_score']
    return dict(exact=exact, same_count=len(records_a)==len(records_b), same_centres=same_centres,
                **{f'same_{field}': [r.get(field) for r in records_a] == [r.get(field) for r in records_b] for field in fields})



def differences(reference, result):
    from scipy.optimize import linear_sum_assignment
    a, b = reference['measurements'], result['measurements']
    if len(a) != len(b):
        return dict(count_changed=True)
    if not a:
        return {}
    distances = np.linalg.norm(np.array([[r['x'],r['y']] for r in a])[:,None] - np.array([[r['x'],r['y']] for r in b])[None], axis=2)
    ia, ib = linear_sum_assignment(distances)
    output = dict(max_position_error_px=float(distances[ia,ib].max()))
    for field in ['detector_score','area_mm2','nice_probability']:
        pairs = [(a[i].get(field), b[j].get(field)) for i,j in zip(ia,ib)]
        numeric = [(x,y) for x,y in pairs if x is not None and y is not None]
        output['max_delta_'+field] = max((abs(x-y) for x,y in numeric), default=0)
        if field == 'area_mm2':
            output['max_relative_area_change'] = max((abs(x-y)/x for x,y in numeric if x), default=0)
    output['changed_masks'] = sum(a[i].get('mask') != b[j].get('mask') for i,j in zip(ia,ib))
    output['changed_nice_decisions'] = sum(a[i].get('nice') != b[j].get('nice') for i,j in zip(ia,ib))
    return output


def run(pipeline, path, variant, directory):
    timing, calls = defaultdict(float), defaultdict(int)
    def timed(name, fn, sync=False):
        def wrapper(*args, **kwargs):
            if sync: torch.cuda.synchronize()
            begin = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                if sync: torch.cuda.synchronize()
                timing[name] += time.perf_counter()-begin
                calls[name] += 1
        return wrapper
    original_extract = pipeline.detector._extract_predictions
    def extract(results):
        for result in results:
            for name, milliseconds in result.speed.items():
                timing['yolo_'+name] += milliseconds / 1000
        return original_extract(results)
    original_sliced = pipeline_module.get_sliced_prediction
    cached = {}
    def sliced(path, detector, **kwargs):
        if variant != 'baseline':
            begin = time.perf_counter()
            cached['rgb'] = read_image_as_pil(path)
            timing['shared_decode'] += time.perf_counter()-begin
            path = cached['rgb']
        if variant.startswith('batch'):
            kwargs['batch_size'] = int(variant[5:])
        result = original_sliced(path, detector, **kwargs)
        for name, value in result.durations_in_seconds.items():
            timing['sahi_'+name] += value
        return result
    original_read = measurement.read_image
    def read(path):
        if 'rgb' in cached:
            return cv2.cvtColor(np.array(cached['rgb']), cv2.COLOR_RGB2BGR)
        return original_read(path)
    with ExitStack() as stack:
        stack.enter_context(patch.object(pipeline.detector, '_extract_predictions', extract))
        stack.enter_context(patch.object(pipeline_module, 'get_sliced_prediction', timed('detection_total', sliced)))
        stack.enter_context(patch.object(measurement, 'read_image', timed('refiner_image_read', read)))
        for name in ['read_dpi', 'prepare_crop', 'centre_component', 'describe', 'encode_mask']:
            stack.enter_context(patch.object(measurement, name, timed(name, getattr(measurement, name))))
        stack.enter_context(patch.object(pipeline.shape, 'forward', timed('shape_forward', pipeline.shape.forward, True)))
        for name in ['perform_batch_inference', 'perform_inference', 'convert_original_predictions']:
            stack.enter_context(patch.object(pipeline.detector, name, timed('detector_'+name, getattr(pipeline.detector, name))))
        for estimator in pipeline.classifier['models']:
            stack.enter_context(patch.object(estimator, 'predict_proba', timed('classifier', estimator.predict_proba)))
        torch.cuda.synchronize()
        begin = time.perf_counter()
        result = pipeline.predict(path)
        torch.cuda.synchronize()
        timing['predict_total'] = time.perf_counter()-begin
        begin = time.perf_counter()
        ImageLogic.save_image_data(dict(result, path=str(directory / path.name)))
        timing['sidecar_write'] = time.perf_counter()-begin
        timing['total'] = timing['predict_total']+timing['sidecar_write']
    return result, dict(timing), dict(calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/pipeline_benchmark')
    parser.add_argument('--limit', type=int, default=6)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--variants', nargs='+', default=['baseline','decode_once','batch4','batch8','batch16'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plates = pd.read_csv(ROOT/'train/runs/5wpi_nice_area/plates_5wpi.csv').sort_values('detected_count')
    selected = plates.iloc[np.linspace(0, len(plates)-1, args.limit).astype(int)]
    paths = [ROOT/'data-test'/row.folder/row.file for row in selected.itertuples()]
    for path in paths:
        with Image.open(path) as img:
            if img.getexif().get(274, 1) != 1:
                raise ValueError('Shared-decode experiment requires identity EXIF orientation')
    begin = time.perf_counter()
    pipeline = pipeline_module.InferencePipeline('cuda:0')
    torch.cuda.synchronize()
    initialization = time.perf_counter()-begin
    rows, references = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        result, timing, calls = run(pipeline, paths[0], 'baseline', directory)
        cold = timing
        print('Cold first plate:', json.dumps(cold), flush=True)
        # Warm every variant before timing; alternate order to reduce order bias.
        for variant in args.variants:
            run(pipeline, paths[0], variant, directory)
        for repeat in range(args.repeats):
            order = args.variants if repeat % 2 == 0 else list(reversed(args.variants))
            for path in paths:
                if path.name not in references:
                    references[path.name] = run(pipeline, path, 'baseline', directory)[0]
                for variant in order:
                    result, timing, calls = run(pipeline, path, variant, directory)
                    parity = compare(references[path.name], result)
                    rows.append(dict(file=path.name, variant=variant, repeat=repeat,
                                     detected=len(result['scores']), timing=timing, calls=calls, parity=parity, differences=differences(references[path.name], result)))
                    (args.output/'measurements.json').write_text(json.dumps(rows, indent=2))
                    print(f'{repeat+1} {path.name} {variant}: {timing["total"]:.3f}s exact={parity["exact"]}', flush=True)
    summary = dict(initialization_seconds=initialization, first_plate=cold, images=len(paths),
                   repeats=args.repeats, gpu=torch.cuda.get_device_name(),
                   runtime={name:version(name) for name in ['torch','ultralytics','sahi','numpy','opencv-python']}, variants={})
    for variant in args.variants:
        group = [r for r in rows if r['variant']==variant]
        summary['variants'][variant] = dict(mean_seconds={key:float(np.mean([r['timing'].get(key,0) for r in group])) for key in sorted(set().union(*(r['timing'] for r in group)))},
            parity_passes={key:sum(r['parity'][key] for r in group) for key in group[0]['parity']}, runs=len(group))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
