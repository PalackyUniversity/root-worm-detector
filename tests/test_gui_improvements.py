"""Regression coverage for the independently committable GUI improvements."""
import gc
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import tempfile
import unittest
import weakref

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.theme import apply_theme


class GuiImprovementsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme(cls.app)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.paths = []
        for i in range(3):
            path = str(Path(self.directory.name) / f'scan-{i}.png')
            cv2.imwrite(path, np.full((600, 800, 3), 40 + i * 40, np.uint8))
            self.paths.append(path)
        self.window = MainWindow()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.directory.cleanup()

    def test_import_keeps_only_selected_pixels_and_releases_on_switch(self):
        self.window.load_files(self.paths)
        records = self.window._MainWindow__image_data
        self.assertEqual(sum(d.get('image') is not None for d in records), 1)
        old = weakref.ref(records[0]['image'])
        self.window.panel_image_list.setCurrentRow(2)
        gc.collect()
        self.assertIsNone(old())
        self.assertEqual(sum(d.get('image') is not None for d in records), 1)
        np.testing.assert_array_equal(records[2]['image'][0, 0], [120, 120, 120])

    def test_status_refresh_preserves_selection_and_decoded_buffer(self):
        self.window.load_files(self.paths)
        self.window.panel_image_list.setCurrentRow(2)
        pixels = self.window._MainWindow__image_data[2]['image']
        self.window.update_image_list()
        self.assertEqual(self.window.panel_image_list.currentRow(), 2)
        self.assertIs(self.window._MainWindow__image_data[2]['image'], pixels)

    def test_empty_list_offers_import_and_long_names_do_not_scroll_horizontally(self):
        panel = self.window.panel_image_list
        self.assertTrue(panel.import_button.isVisible())
        self.assertEqual(len(panel.import_button.menu().actions()), 2)
        self.window.load_files(self.paths)
        self.assertFalse(panel.import_button.isVisible())
        panel.item(0).setText('very-long-image-name-' * 20)
        panel.setFixedWidth(140)
        self.app.processEvents()
        self.assertFalse(panel.horizontalScrollBar().isVisible())
        self.assertLessEqual(panel.visualItemRect(panel.item(0)).width(), panel.viewport().width())

    def test_progress_is_compact_and_remaining_time_clears(self):
        self.window._begin_progress()
        self.window._progress_started -= 30
        self.window._update_progress(1, 4)
        self.assertEqual(self.window.progress_bar.width(), 150)
        self.assertEqual(self.window.progress_bar.value(), 25)
        self.assertIn('remaining', self.window.label_time_remaining.text())
        self.assertLess(self.window.label_time_remaining.x(), self.window.progress_bar.x())
        self.window._end_progress()
        self.assertTrue(self.window.label_time_remaining.isHidden())

    def test_remaining_time_counts_down_between_results_and_stops(self):
        from unittest.mock import patch
        from PySide6.QtTest import QTest
        from config.strings import Strings

        with patch('ui.main_window.monotonic', return_value=100) as clock:
            self.window._begin_progress()
            clock.return_value = 110
            self.window._update_progress(1, 4)
            self.assertEqual(self.window.label_time_remaining.text(),
                             Strings.TIME_REMAINING_SECONDS.format(seconds=30))
            clock.return_value = 111
            QTest.qWait(1150)
            self.assertEqual(self.window.label_time_remaining.text(),
                             Strings.TIME_REMAINING_SECONDS.format(seconds=29))
            self.assertEqual(self.window.progress_bar.value(), 25)
            clock.return_value = 120
            self.window._update_progress(2, 4)
            self.assertEqual(self.window.label_time_remaining.text(),
                             Strings.TIME_REMAINING_SECONDS.format(seconds=20))
            clock.return_value = 150
            QTest.qWait(1150)
            self.assertEqual(self.window.label_time_remaining.text(),
                             Strings.TIME_REMAINING_SECONDS.format(seconds=0))
            self.window._end_progress()
            QTest.qWait(1150)
            self.assertEqual(self.window.label_time_remaining.text(), '')
            self.window._begin_progress()
            QTest.qWait(1150)
            self.assertEqual(self.window.label_time_remaining.text(), Strings.ESTIMATING)
            self.window._end_progress()

    def _select(self, point, offset=QPointF()):
        self.window.start_group_selection()
        label = self.window.label_image
        if hasattr(label, 'widget_point'):
            start = label.widget_point(QPointF(*point))
        else:
            pixmap = label.pixmap()
            scale = self.window._MainWindow__effective_scale
            start = QPointF(point[0] * scale + (label.width() - pixmap.width()) / 2,
                            point[1] * scale + (label.height() - pixmap.height()) / 2)
        for kind, local, button, buttons in [
            (QMouseEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton),
            (QMouseEvent.MouseMove, start + offset, Qt.NoButton, Qt.LeftButton),
            (QMouseEvent.MouseButtonRelease, start + offset, Qt.LeftButton, Qt.NoButton),
        ]:
            event = QMouseEvent(kind, local, local, button, buttons, Qt.NoModifier)
            {QMouseEvent.MouseButtonPress: self.window.preview_mouse_press,
             QMouseEvent.MouseMove: self.window.preview_mouse_move,
             QMouseEvent.MouseButtonRelease: self.window.preview_mouse_release}[kind](event)

    def _contour(self):
        self.window.load_files(self.paths[:1])
        data = self.window._MainWindow__image_data[0]
        data['contours'] = [np.array([[[300, 200]], [[500, 200]], [[500, 400]], [[300, 400]]], np.int32)]
        data['scores'] = [.9]
        self.window.update_preview()

    def test_click_selection_tolerates_small_pointer_movement(self):
        self._contour()
        self._select((320, 220), QPointF(2, 2))
        self.assertEqual(self.window._MainWindow__group_selected_indices, [0])

    def test_near_edge_click_and_rectangle_selection(self):
        self._contour()
        scale = self.window._MainWindow__effective_scale
        self._select((300 - 3 / scale, 250))
        self.assertEqual(self.window._MainWindow__group_selected_indices, [0])
        self._select((300 - 15 / scale, 250))
        self.assertEqual(self.window._MainWindow__group_selected_indices, [])
        self._select((250, 150), QPointF(300 * scale, 300 * scale))
        self.assertEqual(self.window._MainWindow__group_selected_indices, [0])
        self.assertIsNone(self.window._MainWindow__group_selection_rect)

    def test_preview_can_zoom_beyond_previous_limit(self):
        self.window.load_files(self.paths[:1])
        self.window.zoom(8)
        previous = self.window._MainWindow__effective_scale
        self.window.zoom_step_in()
        self.assertGreater(self.window._MainWindow__effective_scale, previous)
