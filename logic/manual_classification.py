"""Classify a user-drawn outline with the deployed niceness ensemble."""
from functools import lru_cache
import hashlib
import pickle

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from config.model import Model
from config.strings import Strings
from logic import measurement


@lru_cache(maxsize=1)
def load_classifier():
    payload = Model.CLASSIFIER_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != Model.SHA256[Model.CLASSIFIER_PATH.name]:
        raise ValueError(f"Model checksum mismatch: {Model.CLASSIFIER_PATH}")
    classifier = pickle.loads(payload)
    if classifier['features'] != measurement.FEATURES:
        raise ValueError('Classifier feature order does not match the deployment pipeline')
    return classifier


def classify_contour(data, contour):
    points = contour.reshape(-1, 2)
    moments = cv2.moments(contour)
    centre = ((moments['m10'] / moments['m00'], moments['m01'] / moments['m00'])
              if moments['m00'] else tuple(points.mean(axis=0)))
    height, width = data['image'].shape[:2]
    if not (0 <= centre[0] < width and 0 <= centre[1] < height):
        raise ValueError(Strings.CONTOUR_CLASSIFICATION_INVALID_SHAPE)
    dpi = data.get('dpi') or 600
    crop = measurement.prepare_crop(data['image'], centre, points, dpi)
    features = measurement.describe(crop['tile'], crop['seed']) if crop is not None else None
    if features is None:
        raise ValueError(Strings.CONTOUR_CLASSIFICATION_INVALID_SHAPE)
    # Use the same plate-relative brightness feature as the detector pipeline.
    values = [record['features']['value'] for record in data.get('measurements', [])
              if record.get('features', {}).get('value') is not None]
    median = np.median(values) if values else features['value']
    features['value_vs_plate'] = float(features['value'] - median)
    inputs = np.array([[features[name] for name in measurement.FEATURES]])
    if not np.isfinite(inputs).all():
        raise ValueError(Strings.CONTOUR_CLASSIFICATION_INVALID_FEATURES)
    classifier = load_classifier()
    with threadpool_limits(limits=4):
        probability = float(np.mean([model.predict_proba(inputs)[0, 1]
                                     for model in classifier['models']]))
    if not np.isfinite(probability):
        raise ValueError(Strings.CONTOUR_CLASSIFICATION_INVALID_PROBABILITY)
    area = float(cv2.contourArea(contour))
    return dict(status='manual', nice=bool(probability >= Model.NICE_THRESHOLD),
                nice_probability=probability, features=features, reason=None,
                x=float(centre[0]), y=float(centre[1]), area_pixels_source=area,
                area_mm2=area * (25.4 / dpi) ** 2, area_method='drawn_contour')
