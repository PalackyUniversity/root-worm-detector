import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow


class PreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.window.resize(1100, 800)
        self.window.show()
        self.app.processEvents()
        self.data = dict(image=np.full((3000, 4000, 3), 80, np.uint8),
                         contours=[np.array([[[1800, 1300]], [[2200, 1300]],
                                             [[2200, 1700]], [[1800, 1700]]], np.int32)],
                         measurements=[dict(nice=True)], predicted=True)
        self.window._MainWindow__image_data = [self.data]
        self.window.on_image_selected(0)
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def image_point(self, viewport_point):
        label = self.window.label_image
        local = label.mapFrom(self.window.panel_image.viewport(), viewport_point)
        event = QMouseEvent(QMouseEvent.MouseMove, QPointF(local), QPointF(label.mapToGlobal(local)),
                            Qt.NoButton, Qt.NoButton, Qt.NoModifier)
        return self.window.get_image_coordinates(event)

    def wheel(self, viewport_point, delta, modifiers=Qt.ControlModifier, pixels=None):
        label = self.window.label_image
        local = label.mapFrom(self.window.panel_image.viewport(), viewport_point)
        event = QWheelEvent(QPointF(local), QPointF(label.mapToGlobal(local)),
                            pixels or QPoint(), QPoint(0, delta), Qt.NoButton,
                            modifiers, Qt.NoScrollPhase, False)
        self.window.preview_wheel_event(event)
        self.app.processEvents()

    def test_ctrl_wheel_preserves_image_point_under_off_center_cursor(self):
        point = QPoint(260, 240)
        before = self.image_point(point)
        for delta in (120, 120, 120, -120):
            self.wheel(point, delta)
            after = self.image_point(point)
            self.assertLessEqual((after - before).manhattanLength(), 10)

    def test_detection_interior_has_translucent_highlight(self):
        view = self.window.panel_image.viewport()
        shot = view.grab().toImage()
        color = shot.pixelColor(shot.width() // 2, shot.height() // 2)
        self.assertGreater(color.green(), color.red() + 20)
        self.assertGreater(color.red(), 0)  # still shows the scan beneath
        np.testing.assert_array_equal(self.data['image'][1500, 2000], [80, 80, 80])

    def test_hidden_overlay_reveals_original_image(self):
        self.window._show_contours = False
        self.window.update_preview()
        self.app.processEvents()
        shot = self.window.panel_image.viewport().grab().toImage()
        color = shot.pixelColor(shot.width() // 2, shot.height() // 2)
        self.assertEqual((color.red(), color.green(), color.blue()), (80, 80, 80))

    def test_drag_can_move_fitted_image_toward_corner(self):
        from PySide6.QtTest import QTest
        view = self.window.panel_image.viewport()
        label = self.window.label_image
        start = view.rect().center()
        before = self.image_point(start)
        local = label.mapFrom(view, start)
        QTest.mousePress(label, Qt.LeftButton, Qt.NoModifier, local)
        QTest.mouseMove(label, local + QPoint(180, 130))
        QTest.mouseRelease(label, Qt.LeftButton, Qt.NoModifier,
                           label.mapFrom(view, start + QPoint(180, 130)))
        self.app.processEvents()
        after = self.image_point(start + QPoint(180, 130))
        self.assertLessEqual((after - before).manhattanLength(), 10)

    def test_zoom_preserves_anchor_below_fit_and_after_panning(self):
        self.window.zoom(0.4)
        self.app.processEvents()
        view = self.window.panel_image.viewport()
        point = view.rect().center() + QPoint(70, -50)
        for delta in [120] * 12 + [-120] * 12:
            before = self.image_point(point)
            self.wheel(point, delta)
            after = self.image_point(point)
            self.assertLessEqual((after - before).manhattanLength()
                                 * self.window._MainWindow__effective_scale, 2)

    def test_small_detection_stays_visible_at_fit(self):
        self.data['contours'] = [np.array([[[1995, 1495]], [[2005, 1495]],
                                          [[2005, 1505]], [[1995, 1505]]], np.int32)]
        self.window.update_preview()
        self.app.processEvents()
        shot = self.window.panel_image.viewport().grab().toImage()
        cx, cy = shot.width() // 2, shot.height() // 2
        visible = sum(shot.pixelColor(x, y).green() > shot.pixelColor(x, y).red() + 40
                      for x in range(cx - 7, cx + 8) for y in range(cy - 7, cy + 8))
        self.assertGreater(visible, 20)

    def test_overlay_tracks_removed_and_restored_contours(self):
        contour = self.data['contours'].pop()
        self.window.update_preview()
        self.app.processEvents()
        shot = self.window.panel_image.viewport().grab().toImage()
        center = QPoint(shot.width() // 2, shot.height() // 2)
        self.assertEqual(shot.pixelColor(center).green(), 80)
        self.data['contours'].append(contour)
        self.window.update_preview()
        self.app.processEvents()
        shot = self.window.panel_image.viewport().grab().toImage()
        self.assertGreater(shot.pixelColor(center).green(), 100)

    def test_resize_preserves_image_point_at_view_center(self):
        self.window.zoom(2)
        self.app.processEvents()
        view = self.window.panel_image.viewport()
        before = self.image_point(view.rect().center())
        self.window.resize(1300, 900)
        self.app.processEvents()
        after = self.image_point(view.rect().center())
        self.assertLessEqual((after - before).manhattanLength(), 10)

    def test_plain_wheel_zooms_at_cursor(self):
        point = QPoint(260, 240)
        before = self.image_point(point)
        scale = self.window._MainWindow__effective_scale
        self.wheel(point, 120, Qt.NoModifier)
        self.assertGreater(self.window._MainWindow__effective_scale, scale)
        self.assertLessEqual((self.image_point(point) - before).manhattanLength(), 10)

    def test_modified_wheel_pans_requested_axis_without_zoom(self):
        point = QPoint(260, 240)
        h = self.window.panel_image.horizontalScrollBar()
        v = self.window.panel_image.verticalScrollBar()
        for modifiers, horizontal in [(Qt.ShiftModifier, False), (Qt.AltModifier, True)]:
            for delta, pixels in [(120, None), (-120, None), (0, QPoint(0, 25))]:
                with self.subTest(modifiers=modifiers, delta=delta, pixels=pixels):
                    before = (h.value(), v.value())
                    scale = self.window._MainWindow__effective_scale
                    self.wheel(point, delta, modifiers, pixels)
                    after = (h.value(), v.value())
                    axis = 0 if horizontal else 1
                    self.assertEqual(after[1-axis], before[1-axis])
                    self.assertLess((after[axis] - before[axis]) * (delta or 25), 0)
                    if pixels:
                        self.assertEqual(after[axis] - before[axis], -25)
                    self.assertEqual(self.window._MainWindow__effective_scale, scale)






    def test_max_zoom_uses_actual_image_scale_at_any_window_size(self):
        from config.general import Config
        for width, height in [(850, 650), (1400, 950)]:
            with self.subTest(size=(width, height)):
                self.window.resize(width, height)
                self.app.processEvents()
                for _ in range(60):
                    self.window.zoom_step_in()
                self.assertAlmostEqual(self.window.label_image.scale, Config.MAX_ZOOM_FACTOR)
                self.wheel(QPoint(260, 240), 120, Qt.NoModifier)
                self.assertAlmostEqual(self.window.label_image.scale, Config.MAX_ZOOM_FACTOR)
                self.wheel(QPoint(260, 240), -120, Qt.NoModifier)
                self.assertAlmostEqual(self.window.label_image.scale, Config.MAX_ZOOM_FACTOR / 1.2)

    def test_resize_keeps_user_chosen_magnification(self):
        self.window.zoom(2)
        self.app.processEvents()
        self.assertAlmostEqual(self.window.label_image.scale, 2)
        self.window.resize(1400, 950)
        self.app.processEvents()
        self.assertAlmostEqual(self.window.label_image.scale, 2)
