from PySide6.QtCore import QThread

from logic.image_logic import ImageLogic
from logic.prediction_logic import PredictionLogic


class PredictionWorker(QThread):
    """Process one image without accessing any widgets or shared image records."""

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path
        self.result = None
        self.error = None

    def run(self):
        try:
            result = PredictionLogic.predict_image(self.path)
            ImageLogic.save_image_data(dict(result, path=self.path, predicted=True))
            self.result = result
        except Exception as error:
            self.error = str(error)


class BatchPredictionWorker(QThread):
    """Qt bridge for the process scheduler; only signals cross into widgets."""
    from PySide6.QtCore import Signal
    image_started = Signal(str)
    image_ready = Signal(str, object)
    status_changed = Signal(str, int, bool)

    def __init__(self, paths, model_id, predictor, parent=None, replace=False):
        super().__init__(parent)
        import threading
        self.paths = paths
        self.model_id = model_id
        self.predictor = predictor
        self.replace = replace
        self.cancelled = threading.Event()
        self.error = None

    def cancel(self):
        self.cancelled.set()

    def _save_result(self, path, result):
        if self.replace:
            from pathlib import Path
            import shutil
            import uuid
            sidecar = Path(path + '_contours.json')
            if sidecar.exists():
                shutil.copy2(sidecar, str(sidecar) + '.before-repredict-' + uuid.uuid4().hex + '.bak')
        ImageLogic.save_image_data(dict(result, path=path, predicted=True))
        self.image_ready.emit(path, result)

    def run(self):
        try:
            self.predictor.run(self.paths, self.model_id, cancel=self.cancelled,
                               on_started=self.image_started.emit, on_result=self._save_result,
                               on_status=self.status_changed.emit)
        except Exception as error:
            self.error = str(error)
