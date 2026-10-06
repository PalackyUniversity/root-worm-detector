import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from PySide6.QtCore import QSize
from PySide6.QtGui import QUndoStack
from PySide6.QtWidgets import QApplication

from logic.image_logic import ImageLogic
from logic.export_logic import ExportLogic
from ui.image_preview import ImagePreview
from ui.main_window import MainWindow


class ReclassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        path = Path(self.directory.name) / "image.tif"
        Image.new("RGB", (150, 150), (80, 80, 80)).save(path, dpi=(600, 600))
        self.data = ImageLogic.load_image(str(path))
        contour = np.array([[[50, 50]], [[100, 50]], [[100, 100]], [[50, 100]]], np.int32)
        self.data.update(contours=[contour, contour.copy()], scores=[.9, None], predicted=True,
                         measurements=[dict(status="classified", det_index=0, nice=False, nice_probability=.2, area_mm2=.08),
                                       dict(status="manual", nice=None, nice_probability=None, area_mm2=None)])

    def test_bulk_override_persists_and_undo_restores_all_metadata(self):
        from logic.commands import ReclassifyContoursCommand

        original = deepcopy(self.data['measurements'])
        stack = QUndoStack()
        stack.push(ReclassifyContoursCommand(self.data, [0, 1], True))
        self.assertTrue(all(record['nice'] for record in self.data['measurements']))
        self.assertEqual(self.data['measurements'][0]['nice_probability'], .2)
        self.assertEqual(self.data['measurements'][0]['area_mm2'], .08)
        self.assertIsNone(self.data['measurements'][1]['area_mm2'])
        saved = ImageLogic.load_image(self.data['path'])
        self.assertEqual(saved['measurements'], self.data['measurements'])
        summary = ExportLogic.build_row(saved, {})
        self.assertEqual(summary['Nice Count'], 2)
        self.assertEqual(summary['Nice Area Count'], 1)
        self.assertEqual(summary['Classified Count'], 2)
        self.assertEqual(summary['Model-Classified Count'], 1)
        self.assertTrue(ExportLogic.female_rows(saved)[0]['Manual Nice Override'])
        stack.undo()
        self.assertEqual(self.data['measurements'], original)
        stack.redo()
        self.assertTrue(self.data['measurements'][0]['nice'])
        stack.push(ReclassifyContoursCommand(self.data, [0, 1], None))
        self.assertFalse(self.data['measurements'][0]['nice'])
        self.assertIsNone(self.data['measurements'][1]['nice'])
        stack.undo()
        self.assertTrue(self.data['measurements'][0]['nice'])

    def test_selection_colors_interior_and_deselection_restores_class_color(self):
        preview = ImagePreview("")
        preview.set_data(dict(self.data, contours=self.data["contours"][:1]))
        preview.set_view(1., QSize(150, 150))
        preview.selected = {0}
        preview.show_scores = False
        preview.show()
        self.app.processEvents()
        shot = preview.grab().toImage()
        colour = shot.pixelColor(225, 225)
        from PySide6.QtGui import QPalette
        accent = preview.palette().color(QPalette.Highlight)
        for channel in ('red', 'green', 'blue'):
            expected = round(80 * (1 - 65 / 255) + getattr(accent, channel)() * 65 / 255)
            self.assertLessEqual(abs(getattr(colour, channel)() - expected), 1)
        outline = shot.pixelColor(200, 225)
        self.assertLess(abs(outline.red() - accent.red()), 5)
        self.assertLess(abs(outline.green() - accent.green()), 5)
        self.assertLess(abs(outline.blue() - accent.blue()), 5)
        # The old selection bounding box sat six pixels outside the contour.
        for x, y in [(194, 225), (256, 225), (225, 194), (225, 256)]:
            outside = shot.pixelColor(x, y)
            self.assertEqual((outside.red(), outside.green(), outside.blue()), (80, 80, 80))
        preview.selected = set()
        preview.update()
        self.app.processEvents()
        colour = preview.grab().toImage().pixelColor(225, 225)
        self.assertGreater(colour.red(), colour.green() + 30)
        self.data['measurements'][0]['nice'] = True
        preview.update()
        self.app.processEvents()
        colour = preview.grab().toImage().pixelColor(225, 225)
        self.assertGreater(colour.green(), colour.red() + 30)
        preview.close()

    def test_toolbar_reclassifies_selection_and_restores_model(self):
        window = MainWindow()
        self.addCleanup(window.close)
        window._MainWindow__image_data = [self.data]
        window.on_image_selected(0)
        window._MainWindow__group_selected_indices = [0, 1]
        window.update_controls()
        self.assertFalse(hasattr(window, 'button_mark_nice'))
        self.assertFalse(hasattr(window, 'button_mark_not_nice'))
        self.assertEqual(window.button_restore_classification.text(), '')
        self.assertFalse(window.button_restore_classification.icon().isNull())
        self.assertTrue(window.button_restore_classification.toolTip())
        self.assertTrue(window.button_restore_classification.property('iconButton'))
        toolbar_layout = window.button_restore_classification.parentWidget().layout()
        self.assertIs(toolbar_layout.itemAt(toolbar_layout.count()-1).widget(), window.button_restore_classification)
        self.assertIsNotNone(toolbar_layout.itemAt(toolbar_layout.count()-2).spacerItem())
        icon = window.button_restore_classification.icon().pixmap(24, 24).toImage()
        self.assertTrue(any(icon.pixelColor(horizontal, vertical).alpha()
                            for horizontal in range(24) for vertical in range(24)))
        window.reclassify_selected(True)
        self.assertTrue(all(record['nice'] for record in self.data['measurements']))
        window.reclassify_selected(False)
        self.assertTrue(all(record['nice'] is False for record in self.data['measurements']))
        window.button_restore_classification.click()
        self.assertFalse(self.data['measurements'][0]['nice'])
        self.assertEqual(len(self.data['measurements']), 1)
        window._MainWindow__undo_stack.undo()
        self.assertTrue(all(record['nice'] is False for record in self.data['measurements']))
        self.assertTrue(window.button_restore_classification.isEnabled())
        window._MainWindow__group_selected_indices = []
        window.update_controls()
        self.assertTrue(window.button_restore_classification.isEnabled())


if __name__ == '__main__':
    unittest.main()
