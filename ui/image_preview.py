"""Cached scan with viewport-painted annotations; never bake overlays into pixels."""
import math

import cv2
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPalette, QPen, QPixmap, QTransform
from PySide6.QtWidgets import QLabel


class ImagePreview(QLabel):
    def __init__(self, text):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self._source = None
        self._levels = []
        self._contours = ()
        self._geometry = []
        self.scale = 1.0
        self.origin = QPointF()
        self.data = None
        self.selected = set()
        self.crosses = False
        self.show_contours = True
        self.show_scores = True
        self.drawing = []
        self.selection_rect = None
        self._markers = {}

    def set_data(self, data):
        changed = self._source is not data['image']
        self.data = data
        if changed:
            self._source = data['image']  # Keep the NumPy buffer alive for QImage.
            h, w = self._source.shape[:2]
            self._levels = [QImage(self._source.data, w, h, self._source.strides[0], QImage.Format_BGR888)]
            # One bounded pyramid per active image, built only when switching scans.
            # At paint time, sample only the exposed region of a suitable level.
            while min(self._levels[-1].width(), self._levels[-1].height()) > 512:
                last = self._levels[-1]
                self._levels.append(last.scaled(max(1, last.width() // 2), max(1, last.height() // 2),
                                                Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        contours = tuple(data.get('contours', ()))
        if len(contours) != len(self._contours) or any(a is not b for a, b in zip(contours, self._contours)):
            self._contours = contours
            self._geometry = []
            for contour in contours:
                points = contour.reshape(-1, 2)
                path = QPainterPath()
                if len(points):
                    path.moveTo(float(points[0][0]), float(points[0][1]))
                    for x, y in points[1:]:
                        path.lineTo(float(x), float(y))
                    path.closeSubpath()
                    moments = cv2.moments(contour)
                    center = (QPointF(moments['m10'] / moments['m00'], moments['m01'] / moments['m00'])
                              if moments['m00'] else QPointF(float(points[0][0]), float(points[0][1])))
                else:
                    center = QPointF()
                self._geometry.append((path, path.boundingRect(), center))
        return changed

    def clear_image(self, text):
        self.data = self._source = None
        self._levels = []
        self._contours = ()
        self._geometry = []
        self.setText(text)
        self.update()

    def set_view(self, scale, viewport_size):
        self.scale = scale
        # Like Root Tracker, allow a full viewport of travel beyond every edge.
        self.origin = QPointF(viewport_size.width(), viewport_size.height())
        h, w = self._source.shape[:2]
        self.resize(math.ceil(w * scale) + 2 * viewport_size.width(),
                    math.ceil(h * scale) + 2 * viewport_size.height())
        self.update()

    def image_point(self, position):
        return (QPointF(position) - self.origin) / self.scale

    def widget_point(self, position):
        return self.origin + QPointF(position) * self.scale

    def paintEvent(self, event):
        if self.data is None:
            super().paintEvent(event)
            return
        painter = QPainter(self)
        painter.setClipRegion(event.region())
        h, w = self._source.shape[:2]
        image_rect = QRectF(self.origin.x(), self.origin.y(), w * self.scale, h * self.scale)
        exposed = QRectF(event.rect()).intersected(image_rect)
        if not exposed.isEmpty():
            level_index = min(len(self._levels) - 1, max(0, int(math.log2(1 / self.scale))))
            level = self._levels[level_index]
            source_rect = QRectF((exposed.x() - self.origin.x()) / self.scale * level.width() / w,
                                 (exposed.y() - self.origin.y()) / self.scale * level.height() / h,
                                 exposed.width() / self.scale * level.width() / w,
                                 exposed.height() / self.scale * level.height() / h)
            painter.drawImage(exposed, level, source_rect)
        painter.setRenderHint(QPainter.Antialiasing)
        transform = QTransform().translate(self.origin.x(), self.origin.y()).scale(self.scale, self.scale)
        visible = QRectF(event.rect()).adjusted(-60, -40, 60, 40)
        selection_color = self.palette().color(QPalette.Highlight)
        records = self.data.get('measurements', [])
        scores = self.data.get('scores', [])
        font = painter.font()
        font.setPixelSize(12)
        painter.setFont(font)
        for i, (path, bounds, center) in enumerate(self._geometry):
            screen_bounds = transform.mapRect(bounds)
            if not visible.intersects(screen_bounds.adjusted(-5, -5, 5, 5)):
                continue
            nice = records[i].get('nice') if i < len(records) else None
            color = QColor(0, 255, 0) if nice is True else QColor(255, 165, 0)
            if nice is None:
                color = QColor(190, 190, 190)
            if i in self.selected:
                color = QColor(255, 50, 50)
            anchor = transform.map(center)
            small = max(screen_bounds.width(), screen_bounds.height()) < 7
            if self.show_contours:
                if self.crosses or small:
                    # Reuse a tiny raster symbol instead of stroking thousands of
                    # subpixel polygons. Its size stays constant in screen pixels.
                    dpr = self.devicePixelRatioF()
                    key = (color.rgba(), self.crosses, dpr)
                    marker = self._markers.get(key)
                    if marker is None:
                        marker = QPixmap(round(20 * dpr), round(20 * dpr))
                        marker.setDevicePixelRatio(dpr)
                        marker.fill(Qt.transparent)
                        ink = QPainter(marker)
                        ink.setRenderHint(QPainter.Antialiasing)
                        symbol = QPainterPath()
                        if self.crosses:
                            symbol.moveTo(3, 10)
                            symbol.lineTo(17, 10)
                            symbol.moveTo(10, 3)
                            symbol.lineTo(10, 17)
                        else:
                            symbol.addEllipse(QPointF(10, 10), 4, 4)
                            fill = QColor(color)
                            fill.setAlpha(65)
                            ink.fillPath(symbol, fill)
                        ink.setPen(QPen(QColor(0, 0, 0, 180), 4))
                        ink.drawPath(symbol)
                        ink.setPen(QPen(color, 2))
                        ink.drawPath(symbol)
                        ink.end()
                        self._markers[key] = marker
                    painter.drawPixmap(anchor - QPointF(10, 10), marker)
                else:
                    overlay = transform.map(path)
                    fill = QColor(color)
                    fill.setAlpha(65)
                    painter.fillPath(overlay, fill)
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(QPen(QColor(0, 0, 0, 180), 4))
                    painter.drawPath(overlay)
                    painter.setPen(QPen(color, 2))
                    painter.drawPath(overlay)
            # Avoid a carpet of overlapping numbers in the plate overview.
            if (self.show_scores and i < len(scores) and scores[i] is not None
                    and (max(screen_bounds.width(), screen_bounds.height()) >= 14 or i in self.selected)):
                text = f'{scores[i]:.2f}'
                metrics = painter.fontMetrics()
                rect = QRectF(anchor.x() - metrics.horizontalAdvance(text) / 2 - 3,
                              screen_bounds.top() - 21, metrics.horizontalAdvance(text) + 6, 17)
                painter.fillRect(rect, QColor(0, 0, 0, 180))
                painter.setPen(color)
                painter.drawText(rect, Qt.AlignCenter, text)
        if self.drawing:
            path = QPainterPath()
            path.moveTo(transform.map(QPointF(*self.drawing[0])))
            for point in self.drawing[1:]:
                path.lineTo(transform.map(QPointF(*point)))
            if len(self.drawing) == 1:
                path.addEllipse(transform.map(QPointF(*self.drawing[0])), 4, 4)
            painter.setPen(QPen(QColor(0, 100, 255), 2))
            painter.drawPath(path)
        if self.selection_rect is not None:
            rect = transform.mapRect(QRectF(self.selection_rect))
            accent = selection_color
            fill = QColor(accent)
            fill.setAlpha(35)
            painter.setPen(QPen(QColor(255, 255, 255, 180), 3))
            painter.setBrush(fill)
            painter.drawRoundedRect(rect, 2, 2)
            painter.setPen(QPen(accent, 1.5))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect, 2, 2)
        painter.end()
