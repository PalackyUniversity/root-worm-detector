from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from config.model import Model
import numpy as np
import datetime
import json
import cv2

class PredictionLogic:
    @staticmethod
    def predict(
            model_path: str,
            source: np.ndarray,
            model_type: str = "ultralytics",
            model_device: str = "cuda:0",
            model_confidence_threshold: float = 0.1,
            slice_height: int = 640,
            slice_width: int = 640,
            overlap_height_ratio: float = 0.2,
            overlap_width_ratio: float = 0.2,
    ):
        """
        Runs sliced prediction on a numpy image and returns OpenCV contours.

        Args:
            model_path: Path to the YOLO/Ultralytics model weights.
            source: Numpy array image (H, W, C) in BGR or RGB.
            model_type: Model type for sahi AutoDetectionModel (default: ultralytics).
            model_device: Torch device string ("cpu" or "cuda:0").
            model_confidence_threshold: Confidence threshold for predictions.
            slice_height: Height of each slice for tiled inference.
            slice_width: Width of each slice for tiled inference.
            overlap_height_ratio: Slice overlap ratio for height.
            overlap_width_ratio: Slice overlap ratio for width.

        Returns:
            contours: List of OpenCV contours (list of numpy arrays).
        """

        # Convert BGR to RGB if needed
        if source.shape[2] == 3:
            img_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = source

        # Initialize detection model
        detection_model = AutoDetectionModel.from_pretrained(
            model_type=model_type,
            model_path=model_path,
            confidence_threshold=model_confidence_threshold,
            device=model_device,
        )

        # Perform sliced prediction
        prediction_result = get_sliced_prediction(
            image=img_rgb,
            detection_model=detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )

        contours = []
        scores = []
        for obj_pred in prediction_result.object_prediction_list:
            if obj_pred.mask:  # segmentation mask available
                mask = obj_pred.mask.bool_mask.astype(np.uint8) * 255
                mask = np.ascontiguousarray(mask)
                obj_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours.extend(obj_contours)
                scores.extend([obj_pred.score.value] * len(obj_contours))

        return contours, scores

    @classmethod
    def predict_contours(cls, image, file_path):
        predicted_contours, predicted_scores = cls.predict(
                model_type="ultralytics",
                model_path="models/v2_best.pt",
                model_confidence_threshold=Model.MIN_OBJECT_CONFIDENCE,
                source=image,
            )

        contours, scores = [], []
        for cnt, score in zip(predicted_contours, predicted_scores):
            x, y, w, h = cv2.boundingRect(cnt)
            if w*h < Model.MAX_OBJECT_SIZE:
                contours.append(cnt)
                scores.append(float(score))

        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        meta = {
            "prediction_time": datetime.datetime.now().isoformat(),
            "model_version": Model.CURRENT_MODEL_VERSION,
            "contours": [cnt.tolist() for cnt in contours],
            "scores": scores
        }
        with open(file_path + "_contours.json", "w") as f:
            json.dump(meta, f)
        return contours, scores