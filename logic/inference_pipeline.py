"""The GUI's reproducible detector → outline → niceness pipeline."""

import hashlib
from importlib.metadata import version
import pickle

import cv2
import numpy as np
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from threadpoolctl import threadpool_limits

from config.model import Model
from logic import measurement
from logic.shape_model import SeededUNet


class InferencePipeline:
    def __init__(self, device=None):
        self.device = str(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        torch.set_num_threads(4)
        for name, expected in Model.SHA256.items():
            path = Model.MODEL_DIR / name
            if not path.is_file():
                raise FileNotFoundError(f"Required model is missing: {path}. Fetch the model files with Git LFS.")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Model checksum mismatch: {path}")
        self.detector = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=str(Model.DETECTOR_PATH),
            confidence_threshold=Model.MIN_OBJECT_CONFIDENCE, image_size=640, device=self.device)
        self.shape = SeededUNet().to(self.device)
        self.shape.load_state_dict(torch.load(Model.SHAPE_PATH, map_location=self.device, weights_only=True))
        self.shape.eval()
        with Model.CLASSIFIER_PATH.open("rb") as handle:
            self.classifier = pickle.load(handle)
        if self.classifier["features"] != measurement.FEATURES:
            raise ValueError("Classifier feature order does not match the deployment pipeline")

    @torch.no_grad()
    def predict(self, path):
        dpi = measurement.read_dpi(path)
        tile_size = measurement.tile_size_for_dpi(dpi)
        result = get_sliced_prediction(
            str(path), self.detector, slice_height=tile_size, slice_width=tile_size,
            overlap_height_ratio=.2, overlap_width_ratio=.2, postprocess_type="GREEDYNMM",
            postprocess_match_metric="IOS", force_postprocess_type=True, verbose=0)
        image = measurement.read_image(path)
        contours, scores, records, prepared = [], [], [], []
        for index, prediction in enumerate(result.object_prediction_list):
            box = prediction.bbox
            centre = ((box.minx+box.maxx)/2., (box.miny+box.maxy)/2.)
            rings = prediction.mask.segmentation if prediction.mask is not None else []
            polygon = np.asarray(rings[0], dtype=np.float64).reshape(-1, 2) if rings and len(rings[0]) >= 6 else None
            display = polygon if polygon is not None else np.array([[box.minx, box.miny], [box.maxx, box.miny], [box.maxx, box.maxy], [box.minx, box.maxy]])
            contours.append(np.rint(display).astype(np.int32).reshape(-1, 1, 2))
            score = float(prediction.score.value)
            scores.append(score)
            records.append(dict(det_index=index, detector_score=score, x=float(centre[0]), y=float(centre[1]),
                                status="unclassified", nice=None, nice_probability=None, area_mm2=None,
                                area_pixels_source=None, reason="No usable seeded outline"))
            crop = measurement.prepare_crop(image, centre, polygon, dpi)
            if crop is not None and crop["seed"] is not None and crop["seed"].sum() >= 4:
                prepared.append((index, crop))
        if prepared:
            stacked = np.stack([np.concatenate([crop["tile"], (crop["seed"][:, :, None]*255).astype(np.uint8)], axis=2) for _, crop in prepared])
            inputs = torch.from_numpy(stacked).permute(0, 3, 1, 2).float().div_(255.).to(self.device)
            probabilities = torch.sigmoid(self.shape(inputs)).cpu().numpy()[:, 0]
            for (index, crop), probability in zip(prepared, probabilities):
                mask = measurement.centre_component((probability > .5).astype(np.uint8), crop["centre"])
                if mask is None or mask.sum() < 20:
                    continue
                features = measurement.describe(crop["tile"], mask)
                if features is None:
                    continue
                records[index].update(status="classified", reason=None, features=features,
                                      area_mm2=measurement.area_mm2(mask, crop["scale"], dpi),
                                      area_pixels_source=float(mask.sum())/crop["scale"]**2,
                                      mask=measurement.encode_mask(mask), mask_offset=list(crop["offset"]),
                                      mask_scale=crop["scale"])
                outlines, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                outline = max(outlines, key=cv2.contourArea)
                display = outline.reshape(-1, 2)/crop["scale"]+np.asarray(crop["offset"])
                contours[index] = np.rint(display).astype(np.int32).reshape(-1, 1, 2)
        classified = [record for record in records if record["status"] == "classified"]
        if classified:
            median = np.median([record["features"]["value"] for record in classified])
            for record in classified:
                record["features"]["value_vs_plate"] = float(record["features"]["value"]-median)
            inputs = np.array([[record["features"][name] for name in measurement.FEATURES] for record in classified])
            if not np.isfinite(inputs).all():
                raise ValueError("Non-finite niceness features; no prediction was saved")
            with threadpool_limits(limits=4):
                probabilities = np.mean([estimator.predict_proba(inputs)[:, 1] for estimator in self.classifier["models"]], axis=0)
            for record, probability in zip(classified, probabilities):
                record.update(nice_probability=float(probability), nice=bool(probability >= Model.NICE_THRESHOLD))
        return dict(contours=contours, scores=scores, measurements=records, dpi=dpi,
                    pipeline=Model.PIPELINE_ID, predicted=True, nice_threshold=Model.NICE_THRESHOLD,
                    provenance=dict(models=Model.SHA256, confidence=Model.MIN_OBJECT_CONFIDENCE,
                                    nice_threshold=Model.NICE_THRESHOLD, device=self.device,
                                    tile_size=tile_size, overlap=.2, mask_threshold=.5,
                                    runtime={name: version(name) for name in ["torch", "torchvision", "ultralytics", "sahi", "opencv-python", "numpy", "scikit-learn"]}))
