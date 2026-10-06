import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
import numpy as np
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication
from logic.image_logic import ImageLogic
from ui.image_preview import ImagePreview


class DrawingPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_draft_matches_finished_unclassified_contour(self):
        strokes = [[(40, 40)], [(40, 40), (45, 45), (50, 45)],
                   [(20, 20), (40, 20), (60, 30), (60, 60), (40, 60), (20, 40)]]
        for stroke in strokes:
            for scale, crosses, visible in [(1., False, True), (.2, False, True),
                                            (1., True, True), (1., False, False)]:
                with self.subTest(stroke=stroke, scale=scale, crosses=crosses, visible=visible):
                    preview = ImagePreview("")
                    data = dict(image=np.full((100, 100, 3), 80, np.uint8),
                                contours=[], measurements=[], scores=[])
                    preview.set_data(data)
                    preview.set_view(scale, QSize(100, 100))
                    preview.crosses = crosses
                    preview.show_contours = visible
                    preview.drawing = stroke
                    draft = preview.grab().toImage()
                    # The same geometry used when the mouse is released.
                    from unittest.mock import patch
                    with patch("logic.manual_classification.classify_contour", return_value=dict(nice=None)):
                        contour, record = ImageLogic.prepare_manual_contour(data, stroke)
                    data.update(contours=[contour], measurements=[record], scores=[None])
                    preview.drawing = []
                    preview.set_data(data)
                    finished = preview.grab().toImage()
                    self.assertEqual(draft, finished)
                    preview.close()
