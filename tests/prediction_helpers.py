"""Deterministic in-process backend for Qt interaction tests (no GPU required)."""
from logic.prediction_logic import PredictionLogic

class InProcessPredictor:
    def run(self, paths, model_id, *, cancel, on_started, on_result, on_status, dpi_by_path=None):
        for path in paths:
            if cancel.is_set():break
            on_started(path)
            result=PredictionLogic.predict_image(path)
            on_result(path,result)
    def close(self):pass
