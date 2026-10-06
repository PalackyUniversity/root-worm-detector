import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from PySide6.QtGui import QUndoStack
from PySide6.QtWidgets import QApplication

from config.model import Model
from logic.commands import AddContourCommand
from logic.image_logic import ImageLogic
from logic.export_logic import ExportLogic


class ManualClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_added_contour_classification_survives_undo_redo_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plate.tif'
            Image.new('RGB', (150, 150), (80, 80, 80)).save(path)
            data = ImageLogic.load_image(str(path))
            stack = QUndoStack()
            stack.push(AddContourCommand(data, [(75, 75)]))
            record = deepcopy(data['measurements'][0])
            self.assertIsInstance(record['nice'], bool)
            self.assertEqual(record['nice'], record['nice_probability'] >= Model.NICE_THRESHOLD)
            self.assertEqual(record['status'], 'manual')
            self.assertEqual(ExportLogic.build_row(data, {})['Model-Classified Count'], 1)
            self.assertIsNone(data['scores'][0])
            contour = data['contours'][0].copy()
            stack.undo()
            self.assertEqual(data['measurements'], [])
            stack.redo()
            self.assertEqual(data['measurements'], [record])
            np.testing.assert_array_equal(data['contours'][0], contour)
            self.assertEqual(ImageLogic.load_image(str(path))['measurements'], [record])
            self.assertEqual(data['original_annotations']['contours'], [])

    def test_plate_brightness_context_does_not_reclassify_existing_objects(self):
        from logic.manual_classification import classify_contour
        records = [dict(features=dict(value=60), nice=True),
                   dict(features=dict(value=100), nice=False)]
        before = deepcopy(records)
        data = dict(image=np.full((150, 150, 3), 90, np.uint8), dpi=600,
                    measurements=records)
        contour = np.array([[[65, 65]], [[85, 65]], [[85, 85]], [[65, 85]]], np.int32)
        record = classify_contour(data, contour)
        self.assertAlmostEqual(record['features']['value_vs_plate'], 10)
        self.assertEqual(records, before)
        self.assertAlmostEqual(record['area_mm2'], 400 * (25.4 / 600) ** 2)

    def test_invalid_contour_does_not_change_annotations_or_create_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plate.tif'
            Image.new('RGB', (150, 150)).save(path)
            data = ImageLogic.load_image(str(path))
            before = ImageLogic.annotation_snapshot(data)
            with self.assertRaises(ValueError):
                AddContourCommand(data, [(-100, -100)])
            self.assertEqual(ImageLogic.annotation_snapshot(data), before)
            self.assertFalse(Path(str(path) + '_contours.json').exists())
