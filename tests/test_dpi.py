import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QSettings
from logic import measurement
from logic.image_logic import ImageLogic
from ui.main_window import MainWindow


class DpiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / 'image.png')
        Image.new('RGB', (40, 40)).save(self.path)
        self.settings = QSettings(str(Path(self.tmp.name) / 'settings.ini'), QSettings.IniFormat)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_metadata_is_unknown_and_png_metadata_is_read(self):
        self.assertIsNone(measurement.read_dpi(self.path))
        Image.new('RGB', (40, 40)).save(self.path, dpi=(720, 720))
        self.assertEqual(measurement.read_dpi(self.path), 720)

    def test_override_survives_reload_and_invalidates_prediction(self):
        data = ImageLogic.load_image(self.path)
        data['predicted'] = True
        ImageLogic.set_dpi_override(data, 1200)
        loaded = ImageLogic.load_image(self.path)
        self.assertEqual(loaded['dpi'], 1200)
        self.assertEqual(loaded['dpi_override'], 1200)
        self.assertFalse(loaded['predicted'])
        with self.assertRaises(ValueError):
            ImageLogic.set_dpi_override(data, 0)

    def test_gui_default_and_override_precedence_and_error(self):
        window = MainWindow(settings=self.settings)
        try:
            window.load_files([self.path])
            self.assertFalse(window.button_predict.isEnabled())
            self.assertTrue(window.panel_image_list.item(0).data(Qt.UserRole + 3))
            window.set_default_dpi(600)
            self.assertTrue(window.button_predict.isEnabled())
            window.set_image_dpi(0, 1200)
            window.set_default_dpi(None)
            self.assertTrue(window.button_predict.isEnabled())
            self.assertEqual(window._MainWindow__image_data[0]['dpi'], 1200)
            window.set_image_dpi(0, None)
            self.assertFalse(window.button_predict.isEnabled())
        finally:
            window.close()
            window.deleteLater()
            self.app.processEvents()

    def test_saved_predictions_match_resolved_dpi_on_reload(self):
        data = ImageLogic.load_image(self.path)
        data.update(dpi=600, predicted=True)
        ImageLogic.save_image_data(data)
        self.assertTrue(ImageLogic.load_image(self.path, default_dpi=600)['predicted'])
        self.assertFalse(ImageLogic.load_image(self.path, default_dpi=1200)['predicted'])
        self.assertFalse(ImageLogic.load_image(self.path)['predicted'])

    def test_tiff_resolution_units_and_missing_tags(self):
        from PIL.TiffImagePlugin import ImageFileDirectory_v2
        path = str(Path(self.tmp.name) / 'scan.tif')
        Image.new('RGB', (40, 40)).save(path)
        self.assertIsNone(measurement.read_dpi(path))
        tags = ImageFileDirectory_v2()
        tags[282], tags[283], tags[296] = 300, 300, 3
        Image.new('RGB', (40, 40)).save(path, tiffinfo=tags)
        self.assertEqual(measurement.read_dpi(path), 762)

    def test_centimetre_resolution_is_converted_before_rounding(self):
        from PIL.TiffImagePlugin import ImageFileDirectory_v2
        path = str(Path(self.tmp.name) / 'scan.tif')
        tags = ImageFileDirectory_v2()
        tags[282], tags[283], tags[296] = 720 / 2.54, 720 / 2.54, 3
        Image.new('RGB', (40, 40)).save(path, tiffinfo=tags)
        self.assertEqual(measurement.read_dpi(path), 720)

    def test_batch_skips_missing_dpi_and_passes_effective_dpi(self):
        import time
        from logic.adaptive_prediction import AdaptivePredictor
        from test_adaptive_scheduler import InlineExecutor
        seen = []
        def execute(path, model, device, threads, dpi):
            seen.append((path, dpi))
            return dict(result=dict(contours=[], scores=[], measurements=[], dpi=dpi),
                        seconds=1, peak_mib=0, rss_mib=100)
        known = str(Path(self.tmp.name) / 'known.png')
        Image.new('RGB', (40, 40)).save(known, dpi=(720, 720))
        window = MainWindow(settings=self.settings)
        window._predictor = AdaptivePredictor(device='cpu', executor_factory=lambda: InlineExecutor(execute))
        try:
            window.load_files([self.path, known])
            window.start_prediction()
            deadline = time.monotonic() + 5
            while window._prediction_thread is not None and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(.005)
            self.assertIsNone(window._prediction_thread)
            self.assertEqual(seen, [(known, 720)])
            window.set_image_dpi(0, 1200)
            window.start_prediction()
            while window._prediction_thread is not None and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(.005)
            self.assertEqual(seen, [(known, 720), (self.path, 1200)])
            loaded = ImageLogic.load_image(self.path)
            self.assertEqual(loaded['dpi_override'], 1200)
            self.assertTrue(loaded['predicted'])
        finally:
            window.close()
            window.deleteLater()
            self.app.processEvents()

    def test_default_persists_but_does_not_override_embedded_dpi(self):
        Image.new('RGB', (40, 40)).save(self.path, dpi=(720, 720))
        window = MainWindow(settings=self.settings)
        window.set_default_dpi(1200)
        window.close()
        window.deleteLater()
        window = MainWindow(settings=self.settings)
        try:
            window.load_files([self.path])
            self.assertEqual(window._default_dpi, 1200)
            self.assertEqual(window._MainWindow__image_data[0]['dpi'], 720)
        finally:
            window.close()
            window.deleteLater()
            self.app.processEvents()

    def test_default_menu_presets_none_and_custom_cancel(self):
        from unittest.mock import patch
        from config.strings import Strings
        window = MainWindow(settings=self.settings)
        try:
            window.load_files([self.path])
            menu = window.menu_default_dpi.menu()
            self.assertIsNotNone(menu)
            actions = {a.text(): a for a in menu.actions() if not a.isSeparator()}
            self.assertTrue(actions[Strings.DPI_NONE].isChecked())
            actions['720 DPI'].trigger()
            self.assertEqual(window._MainWindow__image_data[0]['dpi'], 720)
            self.assertTrue(actions['720 DPI'].isChecked())
            self.assertTrue(window.button_predict.isEnabled())
            with patch('ui.main_window.QInputDialog.getInt', return_value=(900, False)):
                actions[Strings.DPI_CUSTOM].trigger()
            self.assertEqual(window._default_dpi, 720)
            self.assertTrue(actions['720 DPI'].isChecked())
            with patch('ui.main_window.QInputDialog.getInt', return_value=(900, True)) as dialog:
                actions[Strings.DPI_CUSTOM].trigger()
                self.assertEqual(dialog.call_args.args[4], 1)
            self.assertEqual(window._MainWindow__image_data[0]['dpi'], 900)
            self.assertTrue(actions[Strings.DPI_CUSTOM].isChecked())
            actions[Strings.DPI_NONE].trigger()
            self.assertIsNone(window._default_dpi)
            self.assertFalse(window.button_predict.isEnabled())
            self.assertTrue(actions[Strings.DPI_NONE].isChecked())
        finally:
            window.close()
            window.deleteLater()
            self.app.processEvents()
