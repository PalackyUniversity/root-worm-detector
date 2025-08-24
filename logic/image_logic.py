from datetime import datetime

import numpy as np
import json
import cv2
import os

from config.model import Model
from logic.prediction_logic import PredictionLogic


class ImageLogic:
    @staticmethod
    def load_image(file_path):
        data = {
            "path": file_path,
            "image": cv2.imread(file_path),
            "contours": [],
            "predicted": False,
            "processing": False
        }
        json_path = file_path + "_contours.json"
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as fp:
                    meta = json.load(fp)
                if int(meta.get("model_version", 0)) >= Model.CURRENT_MODEL_VERSION:
                    data["predicted"] = True
                    data["contours"] = [np.array(cnt, dtype=np.int32) for cnt in meta.get("contours", [])]
                    data["scores"] = [float(i) for i in meta.get("scores", [])]
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                pass
        return data

    @staticmethod
    def draw_annotations(data, cross_preview_mode, group_selected_indices, effective_scale):
        img = data["image"].copy()

        # Draw contours using cross preview if enabled.
        for i, cnt in enumerate(data["contours"]):
            color = (0, 0, 255) if i in group_selected_indices else (0, 255, 0)  # TODO color

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
                cv2.drawContours(img, [cnt], -1, color, 2)

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

    @classmethod
    def add_contour(cls, data, cnt):
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
            data["contours"].append(circle_contour)

        # If more than one point is drawn, proceed as before.
        else:
            contour = np.array(cnt, dtype=np.int32)
            data["contours"].append(ImageLogic.smooth_contour(contour))

        cls.save_image_data(data)


    @staticmethod
    def save_image_data(data):
        meta = {
            "prediction_time": datetime.now().isoformat(),
            "model_version": PredictionLogic.CURRENT_MODEL_VERSION,
            "contours": [cnt.tolist() for cnt in data["contours"]]
        }
        with open(data["path"] + "_contours.json", "w") as f:
            json.dump(meta, f)

    @staticmethod
    def draw_prediction_scores(img, contours, scores, group_selected_indices, effective_scale):
        for i, (cnt, score) in enumerate(zip(contours, scores)):
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

