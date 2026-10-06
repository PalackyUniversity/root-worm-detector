import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication, QMenu
from ui.main_window import MainWindow


class ContourContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        path = str(Path(self.directory.name) / 'scan.png')
        cv2.imwrite(path, np.zeros((100, 100, 3), np.uint8))
        self.window = MainWindow()
        self.window.show()
        self.window.load_files([path])
        self.data = self.window._MainWindow__image_data[0]
        self.data.update(contours=[np.array([[[20, 20]], [[40, 20]], [[40, 40]], [[20, 40]]], np.int32)],
                         scores=[.85], measurements=[dict(status='classified', nice=False, nice_probability=.2, detector_score=.85)], predicted=True)
        self.window.update_preview()
        label = self.window.label_image
        if hasattr(label, 'widget_point'):
            self.position = label.widget_point(QPointF(30, 30)).toPoint()
        else:
            pixmap = label.pixmap()
            scale = self.window._MainWindow__effective_scale
            self.position = QPointF(30 * scale + (label.width() - pixmap.width()) / 2,
                                   30 * scale + (label.height() - pixmap.height()) / 2).toPoint()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.directory.cleanup()

    def open_menu(self, action_text=None):
        actions = {}
        class InspectMenu(QMenu):
            def exec(self, *_):
                actions.update({a.text(): a for a in self.actions() if not a.isSeparator()})
                if action_text in actions:
                    actions[action_text].trigger()
        with patch('ui.main_window.QMenu', InspectMenu):
            self.window.show_preview_context_menu(self.position)
        return actions

    def test_context_menu_targets_clicked_contour_and_replaces_group_actions(self):
        actions = self.open_menu()
        self.assertNotIn('Group Select', actions)
        self.assertNotIn('Clear Group Selection', actions)
        self.assertIn('Properties…', actions)
        self.assertTrue(actions['Remove Contour'].isEnabled())
        self.assertEqual(self.window._MainWindow__group_selected_indices, [0])

    def test_marking_from_context_menu_is_undoable_and_keeps_model_probability(self):
        self.open_menu('Mark nice')
        self.assertIs(self.data['measurements'][0]['nice'], True)
        self.assertEqual(self.data['measurements'][0]['nice_probability'], .2)
        from logic.image_logic import ImageLogic
        loaded = ImageLogic.load_image(self.data['path'], load_pixels=False)
        self.assertIs(loaded['measurements'][0]['nice'], True)
        self.assertEqual(loaded['measurements'][0]['nice_probability'], .2)
        self.assertEqual(loaded['scores'], [.85])
        self.window._MainWindow__undo_stack.undo()
        self.assertIs(self.data['measurements'][0]['nice'], False)

    def test_properties_preserve_model_values_after_manual_override_and_handle_missing(self):
        from ui.contour_properties_dialog import ContourPropertiesDialog
        self.data['measurements'][0].update(nice=True, nice_override=True, model_nice=False)
        dialog = ContourPropertiesDialog(self.data, [0], self.window)
        values = [dialog.table.item(row, 1).text() for row in range(dialog.table.rowCount())]
        self.assertIn('85.00%', values)
        self.assertIn('20.00%', values)
        dialog.close()
        self.data['scores'] = [None]
        self.data['measurements'] = [dict(status='manual', nice=None, nice_probability=None)]
        dialog = ContourPropertiesDialog(self.data, [0], self.window)
        values = [dialog.table.item(row, 1).text() for row in range(dialog.table.rowCount())]
        self.assertGreaterEqual(values.count('Not available'), 3)
        dialog.close()

    def test_selected_outline_uses_accent_without_bounding_box(self):
        from PySide6.QtGui import QPalette
        self.window._MainWindow__group_selected_indices = [0]
        self.window._show_confidences = False
        self.window.update_preview()
        self.app.processEvents()
        label = self.window.label_image
        def screen(x, y):
            if hasattr(label, 'widget_point'):
                return label.widget_point(QPointF(x, y)).toPoint()
            pixmap = label.pixmap()
            scale = self.window._MainWindow__effective_scale
            return QPointF(x * scale + (label.width() - pixmap.width()) / 2,
                           y * scale + (label.height() - pixmap.height()) / 2).toPoint()
        shot = label.grab().toImage()
        accent = label.palette().color(QPalette.Highlight)
        edge = shot.pixelColor(screen(20, 30))
        self.assertLess(abs(edge.red() - accent.red()), 5)
        self.assertLess(abs(edge.green() - accent.green()), 5)
        self.assertLess(abs(edge.blue() - accent.blue()), 5)
        outside = shot.pixelColor(screen(14, 30))
        self.assertEqual((outside.red(), outside.green(), outside.blue()), (0, 0, 0))

    def test_context_removal_and_undo_keep_probability_records_aligned(self):
        self.open_menu('Remove Contour')
        self.assertEqual(self.data['contours'], [])
        self.assertEqual(self.data['measurements'], [])
        self.window._MainWindow__undo_stack.undo()
        self.assertEqual(len(self.data['contours']), 1)
        self.assertEqual(self.data['measurements'][0]['nice_probability'], .2)
