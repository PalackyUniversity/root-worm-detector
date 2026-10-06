"""Image imports must not retain the whole dataset's decoded pixels."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import gc
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication

from logic import measurement
from logic.commands import AddContourCommand
from logic.image_logic import ImageLogic
from ui.main_window import MainWindow


class LazyImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.paths = []
        for i in range(3):
            path = str(Path(self.directory.name) / f'{i}.png')
            Image.new('RGB', (80, 60), (i * 40, 20, 30)).save(path, dpi=(600, 600))
            self.paths.append(path)
        self.window = MainWindow()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.directory.cleanup()

    def test_import_decodes_only_selected_image_and_switch_releases_previous_pixels(self):
        with patch.object(measurement, 'read_image', wraps=measurement.read_image) as decoder:
            self.window.load_files(self.paths)
            self.assertEqual([call.args[0] for call in decoder.call_args_list], self.paths[:1])
            records = self.window._MainWindow__image_data
            previous = weakref.ref(records[0]['image'])
            self.window.panel_image_list.setCurrentRow(2)
            gc.collect()
            self.assertIsNone(previous())
            self.assertEqual(sum(d.get('image') is not None for d in records), 1)
            np.testing.assert_array_equal(records[2]['image'][0, 0], [30, 20, 80])
            self.window.panel_image_list.setCurrentRow(0)
            np.testing.assert_array_equal(records[0]['image'][0, 0], [30, 20, 0])

    def test_status_refresh_preserves_selection_and_prepared_preview(self):
        self.window.load_files(self.paths)
        self.window.panel_image_list.setCurrentRow(2)
        level = self.window.label_image._levels[0]
        with patch.object(measurement, 'read_image', wraps=measurement.read_image) as decoder:
            self.window.update_image_list()
            self.assertEqual(self.window.panel_image_list.currentRow(), 2)
            self.assertIs(self.window.label_image._levels[0], level)
            decoder.assert_not_called()

    def test_undo_keeps_annotations_without_pinning_old_pixels(self):
        self.window.load_files(self.paths)
        record = self.window._MainWindow__image_data[0]
        command = AddContourCommand(record, [(25, 25)])
        command.redo()
        self.window.panel_image_list.setCurrentRow(1)
        self.assertIsNone(record.get('image'))
        command.undo()
        self.assertEqual(len(record['contours']), 0)
        command.redo()
        self.window.panel_image_list.setCurrentRow(0)
        self.assertEqual(len(record['contours']), 1)
        reloaded = ImageLogic.load_image(self.paths[0])
        np.testing.assert_array_equal(reloaded['contours'][0], record['contours'][0])

    def test_unavailable_image_clears_preview_and_can_be_retried(self):
        self.window.load_files(self.paths)
        Path(self.paths[1]).unlink()
        self.window.panel_image_list.setCurrentRow(1)
        self.assertIsNone(self.window.label_image.data)
        self.assertIn(self.paths[1], self.window.label_image.text())
        self.window.update_preview()  # repaint must not retry or show a dialog
        self.window.panel_image_list.setCurrentRow(2)
        self.assertIsNotNone(self.window.label_image.data)
        Image.new('RGB', (80, 60), (40, 20, 30)).save(self.paths[1])
        self.window.panel_image_list.setCurrentRow(1)
        np.testing.assert_array_equal(self.window.label_image.data['image'][0, 0], [30, 20, 40])

    def test_each_image_restores_its_zoom_and_pan_while_new_images_fit(self):
        from PySide6.QtCore import QPointF
        for path, size in zip(self.paths, [(1600, 1200), (900, 1800), (2400, 1000)]):
            Image.new('RGB', size).save(path, dpi=(600, 600))
        self.window.load_files(self.paths)
        view = self.window.panel_image.viewport()
        h = self.window.panel_image.horizontalScrollBar()
        v = self.window.panel_image.verticalScrollBar()

        def center():
            return self.window.label_image.image_point(QPointF(
                h.value() + view.width() / 2, v.value() + view.height() / 2))

        self.window.zoom(1.5)
        h.setValue(h.value() + 123)
        v.setValue(v.value() - 77)
        saved = center()
        self.window.panel_image_list.setCurrentRow(1)
        self.assertAlmostEqual(self.window.label_image.scale,
                               min(view.width() / 900, view.height() / 1800, 1))
        self.assertLess((center() - QPointF(450, 900)).manhattanLength(), 4)
        self.window.zoom(.8)
        h.setValue(h.value() - 85)
        second = center()
        self.window.panel_image_list.setCurrentRow(0)
        self.assertEqual(self.window.label_image.scale, 1.5)
        self.assertLess((center() - saved).manhattanLength(), 2)
        self.window.panel_image_list.setCurrentRow(1)
        self.assertEqual(self.window.label_image.scale, .8)
        self.assertLess((center() - second).manhattanLength(), 2)
        self.window.panel_image_list.setCurrentRow(2)
        self.window.panel_image_list.setCurrentRow(0)
        self.window.resize(1300, 900)
        self.app.processEvents()
        self.window.panel_image_list.setCurrentRow(2)
        self.assertAlmostEqual(self.window.label_image.scale,
                               min(view.width() / 2400, view.height() / 1000, 1))

    def test_panning_without_zoom_is_remembered_after_switch_and_resize(self):
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QMouseEvent
        Image.new('RGB', (1600, 1200)).save(self.paths[0])
        self.window.load_files(self.paths)
        label = self.window.label_image
        start = label.widget_point(QPointF(800, 600))
        scale = label.scale
        for kind, point, button, buttons in [
            (QMouseEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton),
            (QMouseEvent.MouseMove, start + QPointF(70, 40), Qt.NoButton, Qt.LeftButton),
            (QMouseEvent.MouseButtonRelease, start + QPointF(70, 40), Qt.LeftButton, Qt.NoButton),
        ]:
            event = QMouseEvent(kind, point, point, button, buttons, Qt.NoModifier)
            {QMouseEvent.MouseButtonPress: self.window.preview_mouse_press,
             QMouseEvent.MouseMove: self.window.preview_mouse_move,
             QMouseEvent.MouseButtonRelease: self.window.preview_mouse_release}[kind](event)
        view = self.window.panel_image.viewport()
        def center():
            return label.image_point(QPointF(
                self.window.panel_image.horizontalScrollBar().value() + view.width() / 2,
                self.window.panel_image.verticalScrollBar().value() + view.height() / 2))
        saved = center()
        self.window.panel_image_list.setCurrentRow(1)
        self.window.resize(1300, 900)
        self.app.processEvents()
        self.window.panel_image_list.setCurrentRow(0)
        self.assertEqual(label.scale, scale)
        self.assertLess((center() - saved).manhattanLength() * scale, 2)
