import datetime
import json
import cv2

class PredictionLogic:
    CURRENT_MODEL_VERSION = 1

    @classmethod
    def predict_contours(cls, image, file_path):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        meta = {
            "prediction_time": datetime.datetime.now().isoformat(),
            "model_version": cls.CURRENT_MODEL_VERSION,
            "contours": [cnt.tolist() for cnt in contours]
        }
        with open(file_path + "_contours.json", "w") as f:
            json.dump(meta, f)
        return contours