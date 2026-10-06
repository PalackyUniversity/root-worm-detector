import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from logic.image_logic import ImageLogic
from ui.main_window import MainWindow


class PredictionResponsivenessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_event_loop_runs_during_prediction_and_cancel_skips_next_image(self):
        with tempfile.TemporaryDirectory() as directory:
            records = []
            for name in ('a.png', 'b.png'):
                path = Path(directory) / name
                Image.new('RGB', (40, 40)).save(path)
                records.append(ImageLogic.load_image(str(path)))
            window = MainWindow()
            window._MainWindow__image_data = records
            ticks_during_inference = []
            running = False

            def slow_predict(path):
                nonlocal running
                running = True
                time.sleep(.25)
                running = False
                return dict(contours=[], scores=[], predicted=True)

            def tick():
                if running:
                    ticks_during_inference.append(True)
                    window.cancel_prediction_process()

            timer = QTimer()
            timer.timeout.connect(tick)
            timer.start(10)
            try:
                with patch('ui.main_window.PredictionLogic.predict_image', side_effect=slow_predict):
                    window.start_prediction()
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        self.app.processEvents()
                        if window.button_predict.isVisible() or (records[0]['predicted'] and not records[0]['processing']):
                            if getattr(window, '_prediction_thread', None) is None:
                                break
                        time.sleep(.005)
                self.assertTrue(ticks_during_inference, 'GUI event loop stalled throughout inference')
                self.assertTrue(records[0]['predicted'])
                self.assertFalse(records[1]['predicted'])
                self.assertTrue(Path(records[0]['path'] + '_contours.json').exists())
                self.assertTrue(window.panel_image_list.isEnabled())
            finally:
                timer.stop()
                window.close()
                window.deleteLater()
                self.app.processEvents()

    def test_failure_restores_controls_and_allows_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plate.png'
            Image.new('RGB', (40, 40)).save(path)
            data = ImageLogic.load_image(str(path))
            window = MainWindow()
            window._MainWindow__image_data = [data]
            try:
                with patch('ui.main_window.PredictionLogic.predict_image', side_effect=RuntimeError('inference failed')), patch('ui.main_window.QMessageBox.critical') as dialog:
                    window.start_prediction()
                    self.wait_for_prediction(window)
                    self.assertEqual(dialog.call_args.args[2], 'inference failed')
                self.assertFalse(data['processing'])
                self.assertFalse(data['predicted'])
                self.assertTrue(window.panel_image_list.isEnabled())
                self.assertTrue(window.menu_start_prediction.isEnabled())
                with patch('ui.main_window.PredictionLogic.predict_image', return_value=dict(contours=[], scores=[], predicted=True)):
                    window.start_prediction()
                    self.wait_for_prediction(window)
                self.assertTrue(data['predicted'])
            finally:
                window.close()

    def test_close_during_prediction_defers_shutdown_until_worker_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plate.png'
            Image.new('RGB', (40, 40)).save(path)
            window = MainWindow()
            window._MainWindow__image_data = [ImageLogic.load_image(str(path))]
            window.show()

            def slow_predict(path):
                time.sleep(.15)
                return dict(contours=[], scores=[], predicted=True)

            with patch('ui.main_window.PredictionLogic.predict_image', side_effect=slow_predict):
                window.start_prediction()
                window.close()
                self.assertTrue(window.isVisible())
                self.wait_for_prediction(window)
            self.assertFalse(window.isVisible())

    def wait_for_prediction(self, window):
        deadline = time.monotonic() + 5
        while window._prediction_thread is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.assertIsNone(window._prediction_thread)

    def test_prediction_keeps_zoom_and_pan_controls_and_view_unchanged(self):
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QMouseEvent
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plate.png'
            Image.new('RGB', (1600, 1200)).save(path)
            window = MainWindow()
            window.show()
            window.load_files([str(path)])
            window.zoom(.8)
            h = window.panel_image.horizontalScrollBar()
            v = window.panel_image.verticalScrollBar()
            before = (h.value(), v.value(), window.label_image.scale)
            cursor = window.label_image.cursor().shape()
            def slow_predict(path):
                time.sleep(.2)
                return dict(contours=[], scores=[], predicted=True)
            try:
                with patch('ui.main_window.PredictionLogic.predict_image', side_effect=slow_predict):
                    window.start_prediction()
                    try:
                        self.assertEqual((h.value(), v.value(), window.label_image.scale), before)
                        self.assertEqual(window.label_image.cursor().shape(), cursor)
                        for control in (window.label_image, window.button_pan, window.button_zoom_in,
                                        window.button_zoom_out, window.menu_zoom_in, window.menu_zoom_out):
                            self.assertTrue(control.isEnabled())
                        window.button_zoom_in.click()
                        self.assertGreater(window.label_image.scale, before[2])
                        window.start_group_selection()
                        window.button_pan.click()
                        self.assertTrue(window.button_pan.isChecked())
                        start = window.label_image.widget_point(QPointF(800, 600))
                        old_scroll = (h.value(), v.value())
                        for kind, point, button, buttons in [
                            (QMouseEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton),
                            (QMouseEvent.MouseMove, start + QPointF(60, 40), Qt.NoButton, Qt.LeftButton),
                            (QMouseEvent.MouseButtonRelease, start + QPointF(60, 40), Qt.LeftButton, Qt.NoButton),
                        ]:
                            event = QMouseEvent(kind, point, point, button, buttons, Qt.NoModifier)
                            {QMouseEvent.MouseButtonPress: window.preview_mouse_press,
                             QMouseEvent.MouseMove: window.preview_mouse_move,
                             QMouseEvent.MouseButtonRelease: window.preview_mouse_release}[kind](event)
                        self.assertEqual((h.value(), v.value()), (old_scroll[0] - 60, old_scroll[1] - 40))
                        self.assertFalse(window.button_contour_add.isEnabled())
                    finally:
                        self.wait_for_prediction(window)
            finally:
                window.close()
