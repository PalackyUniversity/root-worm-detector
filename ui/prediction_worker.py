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
