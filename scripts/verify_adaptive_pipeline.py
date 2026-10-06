"""Audit the production adaptive scheduler against the frozen S outputs."""
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from logic.adaptive_prediction import AdaptivePredictor
"""Re-run the GUI inference path against frozen 5WPI outputs without changing them."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.model import Model
from logic.image_logic import ImageLogic
from logic.measurement import FEATURES
from logic.prediction_logic import PredictionLogic


def output_parity(table, objects):
    if not ((table.detected == table.reference_detected).all()
            and (table.classified == table.reference_classified).all()
            and (table.nice == table.reference_nice_at_030).all()
            and len(objects) == int(table.reference_classified.sum())):
        return False
    if objects.empty:
        return True
    return bool((objects.position_error_px <= 1e-12).all()
                and (objects.nice == objects.reference_nice_at_030).all()
                and all((objects['delta_'+field].abs() <= 1e-12).all()
                        for field in ['area_mm2', 'area_pixels_source', 'nice_probability', 'detector_score']))


def audit_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/gui_pipeline_parity")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reference = ROOT / "train/runs/5wpi_nice_area"
    plates = pd.read_csv(reference / "plates_5wpi.csv")
    saved = pd.read_csv(reference / "detections_all_5wpi.csv")
    plate_rows, object_rows, differences = [], [], []
    started = time.monotonic()
    for number, (_, plate) in enumerate(plates.head(args.limit or len(plates)).iterrows(), 1):
        path = ROOT / "data-test" / plate.folder / plate.file
        result = PredictionLogic.predict_image(str(path), device=args.device)
        records = result["measurements"]
        current = pd.DataFrame([dict(record, **record["features"]) for record in records if record["status"] == "classified"])
        expected = saved[saved.file == plate.file].reset_index(drop=True)
        nice_count = sum(record.get("nice") is True for record in records)
        target_nice = int((expected.nice_probability >= Model.NICE_THRESHOLD).sum())
        row = dict(file=plate.file, detected=len(records), reference_detected=int(plate.detected_count),
                   classified=len(current), reference_classified=len(expected), nice=nice_count,
                   reference_nice_at_030=target_nice, original_nice=int(plate.nice_count),
                   area_mm2_sum=sum(record["area_mm2"] for record in records if record.get("nice") is True),
                   reference_area_mm2_sum=float(expected.loc[expected.nice_probability >= Model.NICE_THRESHOLD, "area_mm2"].sum()))
        with tempfile.TemporaryDirectory() as directory:
            cache_path = str(Path(directory) / plate.file)
            ImageLogic.save_image_data(dict(result, path=cache_path))
            payload = json.loads(Path(cache_path+"_contours.json").read_text())
            assert payload["measurements"] == records
            assert payload["scores"] == result["scores"]
        if len(current) == len(expected) and len(current):
            distance = np.linalg.norm(current[["x", "y"]].to_numpy()[:, None]-expected[["x", "y"]].to_numpy()[None], axis=2)
            actual_indices, expected_indices = linear_sum_assignment(distance)
            for actual_index, expected_index in zip(actual_indices, expected_indices):
                actual, old = current.iloc[actual_index], expected.iloc[expected_index]
                entry = dict(file=plate.file, det_index=int(actual.det_index), reference_det_index=int(old.det_index),
                             position_error_px=float(distance[actual_index, expected_index]),
                             nice=bool(actual.nice), reference_nice_at_030=bool(old.nice_probability >= Model.NICE_THRESHOLD))
                for field in ["area_mm2", "area_pixels_source", "nice_probability", "detector_score", *FEATURES]:
                    entry[field] = float(actual[field])
                    entry["reference_"+field] = float(old[field])
                    entry["delta_"+field] = float(actual[field]-old[field])
                object_rows.append(entry)
                if (entry["position_error_px"] > 1e-12
                        or any(abs(entry["delta_"+field]) > 1e-12 for field in ["area_mm2", "area_pixels_source", "nice_probability", "detector_score", *FEATURES])
                        or entry["nice"] != entry["reference_nice_at_030"]):
                    differences.append(entry)
        elif len(current) != len(expected):
            differences.append(dict(file=plate.file, error="Classified count differs; object pairing not attempted"))
        plate_rows.append(row)
        pd.DataFrame(plate_rows).to_csv(args.output / "plates.csv", index=False)
        pd.DataFrame(object_rows).to_csv(args.output / "females.csv", index=False)
        print(f"{number}/{args.limit or len(plates)} {plate.file}: counts {len(records)}/{int(plate.detected_count)}; nice {nice_count}/{target_nice}", flush=True)
    table, objects = pd.DataFrame(plate_rows), pd.DataFrame(object_rows)
    summary = dict(pipeline=Model.PIPELINE_ID, nice_threshold=Model.NICE_THRESHOLD, images=len(table),
                   detections=int(table.detected.sum()), classified=int(table.classified.sum()), nice=int(table.nice.sum()),
                   reference_nice_at_030=int(table.reference_nice_at_030.sum()), original_nice=int(table.original_nice.sum()),
                   exact_detection_count_plates=int((table.detected == table.reference_detected).sum()),
                   exact_classified_count_plates=int((table.classified == table.reference_classified).sum()),
                   exact_nice_count_plates=int((table.nice == table.reference_nice_at_030).sum()),
                   matched_objects=len(objects), differences=len(differences), elapsed_seconds=time.monotonic()-started,
                   source_csv_tolerance=1e-12, sidecars_roundtripped_without_loss=True,
                   runtime={name: version(name) for name in ["torch", "torchvision", "ultralytics", "sahi", "opencv-python", "numpy", "scikit-learn"]})
    if len(objects):
        summary.update(max_position_error_px=float(objects.position_error_px.max()),
                       max_area_error_mm2=float(objects.delta_area_mm2.abs().max()),
                       max_niceness_probability_error=float(objects.delta_nice_probability.abs().max()),
                       exact_niceness_decisions=int((objects.nice == objects.reference_nice_at_030).sum()))
    summary["matches_reference"] = bool(summary["differences"] == 0 and summary["exact_detection_count_plates"] == len(table))
    summary["matches_reference_outputs"] = output_parity(table, objects)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "differences.json").write_text(json.dumps(differences, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["matches_reference"]:
        raise SystemExit(1)




def main():
    output = ROOT / 'outputs/adaptive_inference_benchmark/full_parity'
    output.mkdir(parents=True, exist_ok=True)
    plates = pd.read_csv(ROOT / 'train/runs/5wpi_nice_area/plates_5wpi.csv')
    paths = [str(ROOT / 'data-test' / r.folder / r.file) for r in plates.itertuples()]
    results, statuses = {}, []
    engine = AdaptivePredictor()
    start = time.perf_counter()
    try:
        engine.run(paths, 's', on_result=lambda p, r: results.update({p: r}),
                   on_status=lambda d, w, f: statuses.append(dict(device=d, workers=w, fallback=f)))
    finally:
        engine.close()
    elapsed = time.perf_counter() - start
    (output / 'execution.json').write_text(json.dumps(dict(seconds=elapsed, images=len(results), statuses=statuses), indent=2))
    with patch.object(PredictionLogic, 'predict_image', lambda path, device=None: results.pop(str(path))), patch.object(sys, 'argv', [__file__, '--output', str(output)]):
        audit_main()


if __name__ == '__main__':
    main()
