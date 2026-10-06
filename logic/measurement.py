"""Deployment equivalents of the frozen 5WPI crop and GB-9 feature pipeline."""

import base64

import cv2
import numpy as np
from PIL import Image

FEATURES = ["ellipse_error", "aspect_ratio", "edge_strength", "edge_strength_cv",
            "weak_edge_fraction", "contrast", "value_std", "value_vs_plate", "saturation"]


def read_image(path):
    with Image.open(path) as handle:
        return cv2.cvtColor(np.array(handle.convert("RGB")), cv2.COLOR_RGB2BGR)


def valid_dpi(value):
    try:
        value = float(value)
        return int(round(value)) if np.isfinite(value) and 1 <= value <= 100000 else None
    except (TypeError, ValueError, OverflowError):
        return None


def read_dpi(path):
    with Image.open(path) as handle:
        if hasattr(handle, "tag_v2"):
            value = handle.tag_v2.get(282)
            unit = handle.tag_v2.get(296, 2)
            if unit == 3:
                try:
                    return valid_dpi(float(value) * 2.54)
                except (TypeError, ValueError, OverflowError):
                    return None
            return valid_dpi(value) if unit == 2 else None
        values = handle.info.get("dpi")
        return valid_dpi(values[0]) if isinstance(values, (tuple, list)) and values else None


def tile_size_for_dpi(dpi):
    return max(32, int(round((640 * float(dpi) / 1200) / 32.0)) * 32)


def prepare_crop(image, centre, polygon, dpi):
    diameter = .44 / (25.4 / float(dpi))
    size = max(4, 2 * int(round(4. * diameter / 2)))
    half = size // 2
    centre_x, centre_y = int(round(centre[0])), int(round(centre[1]))
    left, top = centre_x-half, centre_y-half
    right, bottom = left+size, top+size
    height, width = image.shape[:2]
    tile = image[max(0, top):min(height, bottom), max(0, left):min(width, right)]
    if tile.size == 0:
        return None
    padding = (max(0, -top), max(0, bottom-height), max(0, -left), max(0, right-width))
    if any(padding):
        tile = cv2.copyMakeBorder(tile, *padding, cv2.BORDER_REPLICATE)
    scale = 160 / tile.shape[1]
    tile = cv2.resize(tile, (160, 160), interpolation=cv2.INTER_CUBIC)
    seed = None
    if polygon is not None:
        points = (np.asarray(polygon, dtype=np.float64)-np.asarray((left, top)))*scale
        seed = np.zeros((160, 160), np.uint8)
        cv2.fillPoly(seed, [points.round().astype(np.int32)], 1)
    return dict(tile=tile, seed=seed, offset=(left, top), scale=scale,
                centre=((centre[0]-left)*scale, (centre[1]-top)*scale))


def centre_component(binary, centre):
    count, labels = cv2.connectedComponents(binary.astype(np.uint8))
    if count <= 1:
        return None
    centre_x = int(np.clip(centre[0], 0, labels.shape[1]-1))
    centre_y = int(np.clip(centre[1], 0, labels.shape[0]-1))
    label = labels[centre_y, centre_x]
    if label == 0:
        best, best_distance = 0, np.inf
        for candidate in range(1, count):
            positions_y, positions_x = np.nonzero(labels == candidate)
            distance = np.hypot(positions_x.mean()-centre[0], positions_y.mean()-centre[1])
            if distance < best_distance:
                best, best_distance = candidate, distance
        label = best
    return (labels == label).astype(np.uint8) if label else None


def area_mm2(mask, scale, dpi):
    return float(mask.sum()) / (scale ** 2) * ((25.4 / float(dpi)) ** 2)


def encode_mask(mask):
    return dict(shape=list(mask.shape), packed=base64.b64encode(np.packbits(mask).tobytes()).decode("ascii"))


def decode_mask(value):
    shape = tuple(value["shape"])
    packed = np.frombuffer(base64.b64decode(value["packed"]), dtype=np.uint8)
    return np.unpackbits(packed, count=int(np.prod(shape))).reshape(shape)


def describe(bgr, mask):
    if mask.sum() < 20:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5 or cv2.contourArea(contour) <= 0 or cv2.arcLength(contour, True) <= 0:
        return None
    (centre_x, centre_y), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
    major, minor = max(axis_a, axis_b), max(1e-6, min(axis_a, axis_b))
    points = contour.reshape(-1, 2).astype(np.float64)
    theta = np.deg2rad(angle)
    delta_x, delta_y = points[:, 0]-centre_x, points[:, 1]-centre_y
    ellipse_x = (delta_x*np.cos(theta)+delta_y*np.sin(theta))/max(1e-6, axis_a/2)
    ellipse_y = (-delta_x*np.sin(theta)+delta_y*np.cos(theta))/max(1e-6, axis_b/2)
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gradient_x = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    positions_y = np.clip(points[:, 1].astype(int), 0, magnitude.shape[0]-1)
    positions_x = np.clip(points[:, 0].astype(int), 0, magnitude.shape[1]-1)
    along = magnitude[positions_y, positions_x]
    reference = np.percentile(along, 75)
    weak = along < .35*reference if reference > 1e-6 else np.ones(len(along), bool)
    inside = mask.astype(bool)
    ring = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=5).astype(bool) & ~inside
    if ring.sum() < 20:
        ring = ~inside
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    value, saturation = hsv[:, :, 2], hsv[:, :, 1]
    return dict(ellipse_error=float(np.std(np.hypot(ellipse_x, ellipse_y))),
                aspect_ratio=float(major/minor), edge_strength=float(along.mean()),
                edge_strength_cv=float(along.std()/max(1e-6, along.mean())),
                weak_edge_fraction=float(weak.mean()), value=float(value[inside].mean()),
                saturation=float(saturation[inside].mean()), value_std=float(value[inside].std()),
                contrast=float(value[inside].mean()-value[ring].mean()))


def read_prediction_images(path):
    """Share decoded pixels only when SAHI and refiner orientation rules agree."""
    with Image.open(path) as handle:
        if handle.getexif().get(274, 1) != 1:
            return str(path), read_image(path)
        rgb = handle.convert('RGB')
    return rgb, cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
