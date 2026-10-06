import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class GuiSmokeTests(unittest.TestCase):
    def test_prediction_replaces_manual_data_without_stale_undo(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import numpy as np
        from PIL import Image
        from PySide6.QtWidgets import QApplication
        from config.model import Model
        from logic.commands import AddContourCommand
        from logic.image_logic import ImageLogic
        from ui.main_window import MainWindow

        application = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plate.tif"
            Image.new("RGB", (40, 40)).save(path, dpi=(600, 600))
            data = ImageLogic.load_image(str(path))
            window = MainWindow()
            window._MainWindow__image_data = [data]
            stack = window._MainWindow__undo_stack
            stack.push(AddContourCommand(data, [(10, 10)]))
            result = dict(contours=[np.array([[[1, 1]], [[5, 1]], [[3, 4]]], np.int32)],
                          scores=[.7], measurements=[dict(status="classified", nice=True, nice_probability=.3, area_mm2=.02)],
                          dpi=600, pipeline=Model.PIPELINE_ID, predicted=True, nice_threshold=.3)
            with patch("ui.main_window.PredictionLogic.predict_image", return_value=result) as predict:
                window.start_prediction()
                deadline = time.monotonic() + 10
                while getattr(window, "_prediction_thread", None) is not None and time.monotonic() < deadline:
                    application.processEvents()
                    time.sleep(.01)
                self.assertIsNone(getattr(window, "_prediction_thread", None))
                deadline = time.monotonic() + 5
                while window._prediction_thread is not None and time.monotonic() < deadline:
                    application.processEvents()
                    time.sleep(.005)
                self.assertIsNone(window._prediction_thread)
            predict.assert_called_once_with(str(path))
            self.assertFalse(stack.canUndo())
            saved = ImageLogic.load_image(str(path))
            self.assertEqual(saved["measurements"], result["measurements"])
            self.assertEqual(saved["nice_threshold"], .3)
            window.close()
            window.deleteLater()
            application.processEvents()

    def test_python_matches_project_version(self):
        expected = tuple(map(int, (ROOT / ".python-version").read_text().strip().split(".")))
        self.assertEqual(sys.version_info[:2], expected)

    def test_window_starts_and_exits(self):
        environment = os.environ.copy()
        environment["QT_QPA_PLATFORM"] = "offscreen"
        script = """
import main
from PySide6.QtCore import QTimer

class TimedApplication(main.QApplication):
    def __init__(self, *args):
        super().__init__(*args)
        QTimer.singleShot(100, self.quit)

main.QApplication = TimedApplication
main.main()
"""
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT, env=environment,
            capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_excel_export(self):
        import pandas as pd
        from logic.export_logic import ExportLogic

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "counts.xlsx"
            ExportLogic.export_data(
                str(output), {"count": True}, [{"path": "empty.tif", "contours": []}],
            )
            result = pd.read_excel(output)
            self.assertEqual(result.loc[0, "Image File"], "empty.tif")
            self.assertEqual(result.loc[0, "Contour Count"], 0)


if __name__ == "__main__":
    unittest.main()
