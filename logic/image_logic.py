from datetime import datetime
from copy import deepcopy

import numpy as np
import json
import cv2
import os
import shutil
import tempfile

from config.model import Model
from config.strings import Strings
from logic import measurement
from logic.model_registry import model_for_pipeline, model_spec, compatible_sidecar


class ImageLogic:
    @staticmethod
    def load_image(file_path, *, load_pixels=True):
        """Read headers and saved results; optionally defer decoding the scan."""
        data = {
            "path": file_path,
            "image": None,
            "contours": [],
            "scores": [],
            "measurements": [],
            "dpi": measurement.read_dpi(file_path),
            "predicted": False,
            "processing": False
        }
        json_path = file_path + "_contours.json"
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as fp:
                    meta = json.load(fp)
                if (int(meta.get("model_version", 0)) == Model.CURRENT_MODEL_VERSION
                        and compatible_sidecar(meta)
                        and len(meta.get("scores", [])) == len(meta.get("contours", []))
                        and len(meta.get("measurements", [])) == len(meta.get("contours", []))):
                    for key in ("scores", "measurements", "dpi", "pipeline", "nice_threshold", "provenance", "predicted", "original_annotations", "model_id"):
                        if key in meta:
                            data[key] = meta[key]
                    data["model_id"] = model_for_pipeline(meta.get("pipeline"))
                    data["contours"] = [np.array(contour, dtype=np.int32) for contour in meta["contours"]]
            except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
        if load_pixels:
            ImageLogic.ensure_pixels(data)
        return data

    @staticmethod
    def ensure_pixels(data):
        """Decode on demand without replacing annotations or manual edits."""
        if data.get("image") is None:
            img = measurement.read_image(data["path"])
            if img is None:
                raise ValueError(Strings.IMAGE_LOAD_ERROR_MESSAGE.format(file_path=data["path"]))
            data["image"] = img
        return data["image"]

    @staticmethod
    def draw_annotations(data, cross_preview_mode, group_selected_indices, effective_scale, show_contours=True):
        img = data["image"].copy()

        # Draw contours using cross preview if enabled, but only if show_contours is True
        if show_contours:
            for i, cnt in enumerate(data["contours"]):
                records = data.get("measurements", [])
                record = records[i] if i < len(records) else {}
                color = (0, 255, 0) if record.get("nice") is True else (0, 0, 255)
                if record.get("nice") is None:
                    color = (170, 170, 170)

                if cross_preview_mode:
                    m = cv2.moments(cnt)
                    if m["m00"] != 0:
                        cx = int(m["m10"] / m["m00"])
                        cy = int(m["m01"] / m["m00"])
                    else:
                        # Fallback: if the contour area is zero, use the first point.
                        pt = cnt[0].ravel()
                        cx, cy = int(pt[0]), int(pt[1])

                    marker_size = max(10, int(30 / effective_scale))  # Marker size scales with zoom
                    cv2.drawMarker(img, (cx, cy), color,
                                   markerType=cv2.MARKER_CROSS, markerSize=marker_size,
                                   thickness=2, line_type=cv2.LINE_AA)

                else:
                    # Draw the full contour outline.
                    cv2.drawContours(img, [cnt], -1, color, 1)

                if i in group_selected_indices:
                    left, top, width, height = cv2.boundingRect(cnt)
                    margin = max(2, int(6 / effective_scale))
                    start, end = (left-margin, top-margin), (left+width+margin, top+height+margin)
                    ImageLogic.draw_dashed_rectangle(img, start, end, (255, 255, 255), effective_scale, desired_thickness=4)
                    ImageLogic.draw_dashed_rectangle(img, start, end, (255, 210, 0), effective_scale, desired_thickness=2)

        return img

    @staticmethod
    def draw_dashed_rectangle(img, pt1, pt2, color, effective_scale, desired_thickness=2, desired_dash=10):
        dash = max(1, int(desired_dash / effective_scale))
        thick = max(1, int(desired_thickness / effective_scale))

        x1, y1 = pt1
        x2, y2 = pt2
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        for x in range(x1, x2, dash * 2):
            cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thick)
            cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thick)

        for y in range(y1, y2, dash * 2):
            cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thick)
            cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thick)

        return img

    @classmethod
    def draw_contour(cls, img, cnt, color, effective_scale):
        thick = max(2, int(2 / effective_scale))

        if len(cnt) > 1:
            smoothed = cls.smooth_contour(np.array(cnt, dtype=np.int32))
            cv2.polylines(img, [smoothed], False, color, thick)

        elif len(cnt) == 1:
            cv2.circle(img, cnt[0], 10, color, thick)  # TODO 10 const, je to jeste na jednom miste, kde se vytvari

        return img

    @staticmethod
    def smooth_contour(points, num_points=100):
        ## TODO
        return points

        # if len(points) < 3:
        #     return np.array(points, dtype=np.int32)
        #
        # points = np.array(points)
        # x = points[:, 0]
        # y = points[:, 1]
        #
        # try:
        #     # s=0 for interpolation through points; k=3 for cubic spline.
        #     tck, _ = splprep([x, y], s=0, k=3)
        #     unew = np.linspace(0, 1, max(num_points, len(points)))
        #     out = splev(unew, tck)
        #     smoothed = np.stack(out, axis=1)
        #     return np.array(smoothed, dtype=np.int32)
        #
        # except Exception:
        #     return np.array(points, dtype=np.int32)

    @staticmethod
    def annotation_snapshot(data):
        return dict(contours=[cnt.tolist() for cnt in data["contours"]],
                    scores=deepcopy(data.get("scores", [])),
                    measurements=deepcopy(data.get("measurements", [])))

    @staticmethod
    def preserve_original_annotations(data):
        if "original_annotations" in data:
            return
        snapshot = ImageLogic.annotation_snapshot(data)
        indices = [i for i, record in enumerate(snapshot["measurements"])
                   if record.get("status") != "manual"]
        snapshot = {key: [values[i] for i in indices] for key, values in snapshot.items()}
        for record in snapshot["measurements"]:
            if "nice_override" in record:
                record["nice"] = record.get("model_nice")
                record.pop("nice_override")
        data["original_annotations"] = snapshot

    @classmethod
    def manual_contour_geometry(cls, cnt):
        """Build the same outline for live drawing and the saved contour."""
        # If only one point is selected, create a circle contour.
        if len(cnt) <= 5:
            center = cnt[0]
            num_points = 20  # Number of points to approximate the circle.
            angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            circle_points = [
                [int(center[0] + 10 * np.cos(a)), int(center[1] + 10 * np.sin(a))]
                for a in angles
            ]
            circle_contour = np.array(circle_points, dtype=np.int32).reshape((-1, 1, 2))
            contour = circle_contour

        # If more than one point is drawn, proceed as before.
        else:
            contour = np.array(cnt, dtype=np.int32)
            contour = ImageLogic.smooth_contour(contour)

        return contour

    @classmethod
    def prepare_manual_contour(cls, data, cnt):
        from logic.manual_classification import classify_contour

        contour = cls.manual_contour_geometry(cnt)
        cls.ensure_pixels(data)
        return contour, classify_contour(data, contour)

    @classmethod
    def add_contour(cls, data, cnt, *, prepared=None):
        contour, record = prepared if prepared is not None else cls.prepare_manual_contour(data, cnt)
        cls.preserve_original_annotations(data)
        data.setdefault("scores", [None]*len(data["contours"]))
        data.setdefault("measurements", [dict(status="manual", nice=None, area_mm2=None)
                                         for _ in data["contours"]])
        data["contours"].append(contour.copy())
        data["scores"].append(None)
        data["measurements"].append(deepcopy(record))

        cls.save_image_data(data)

        return len(data["contours"]) - 1  # Return the index of the newly added contour

    @staticmethod
    def save_image_data(data):
        identity = model_for_pipeline(data.get("pipeline", Model.PIPELINE_ID))
        provenance = data.get("provenance", {})
        if identity and not provenance:
            spec = model_spec(identity)
            provenance = dict(models=spec["hashes"], confidence=spec["confidence"], model_id=identity)
        meta = {
            "prediction_time": datetime.now().isoformat(),
            "model_version": Model.CURRENT_MODEL_VERSION,
            "contours": [cnt.tolist() for cnt in data["contours"]],
            "scores": data.get("scores", []),
            "measurements": data.get("measurements", []),
            "pipeline": data.get("pipeline", Model.PIPELINE_ID),
            "model_id": model_for_pipeline(data.get("pipeline", Model.PIPELINE_ID)),
            "predicted": bool(data.get("predicted", False)),
            "dpi": data.get("dpi", 600),
            "nice_threshold": Model.NICE_THRESHOLD,
            "provenance": provenance,
        }
        if "original_annotations" in data:
            meta["original_annotations"] = data["original_annotations"]
        if len(meta["scores"]) != len(meta["contours"]) or len(meta["measurements"]) != len(meta["contours"]):
            raise ValueError("Contours, scores and measurement records must stay aligned")
        path = data["path"] + "_contours.json"
        if os.path.exists(path) and not os.path.exists(path+".legacy.bak"):
            try:
                with open(path) as handle:
                    previous = json.load(handle)
                if not compatible_sidecar(previous):
                    shutil.copy2(path, path+".legacy.bak")
            except (ValueError, OSError):
                shutil.copy2(path, path+".legacy.bak")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(os.path.abspath(path)), delete=False) as handle:
                temporary = handle.name
                json.dump(meta, handle, allow_nan=False)
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def draw_prediction_scores(img, contours, scores, group_selected_indices, effective_scale):
        for i, (cnt, score) in enumerate(zip(contours, scores)):
            if score is None:
                continue
            m = cv2.moments(cnt)
            if m["m00"] != 0:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
            else:
                pt = cnt[0].ravel()
                cx, cy = int(pt[0]), int(pt[1])

            font_scale = max(0.5, 1.0 * effective_scale)
            thickness = max(1, int(2 * effective_scale))
            text = f"{score:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            # Place text above the contour center
            text_x = int(cx - tw / 2)
            text_y = int(cy - th / 2 - max(10, 10 * effective_scale))
            color = (0, 0, 255) if i in group_selected_indices else (0, 255, 0)
            cv2.putText(
                img,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA
            )
        return img
