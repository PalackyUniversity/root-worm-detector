"""Palette-aware outline icons sharing a 24-unit grid and rounded strokes."""

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QIcon, QIconEngine, QPainter, QPainterPath, QPalette, QPen, QPixmap
from PySide6.QtWidgets import QApplication


class _OutlineIconEngine(QIconEngine):
    def __init__(self, symbol):
        super().__init__()
        self.symbol = symbol

    def clone(self):
        return _OutlineIconEngine(self.symbol)

    def pixmap(self, size, mode, state):
        pixmap = QPixmap(size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        self.paint(painter, pixmap.rect(), mode, state)
        painter.end()
        return pixmap

    def paint(self, painter, rect, mode, state):
        palette = QApplication.palette()
        group = QPalette.Disabled if mode == QIcon.Disabled else QPalette.Active
        role = QPalette.HighlightedText if mode == QIcon.Selected else QPalette.ButtonText
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        side = min(rect.width(), rect.height())
        painter.translate(rect.x() + (rect.width() - side) / 2,
                          rect.y() + (rect.height() - side) / 2)
        painter.scale(side / 24, side / 24)
        painter.setPen(QPen(palette.color(group, role), 1.8,
                            Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)

        def line(x1, y1, x2, y2):
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        def path(points, closed=False):
            shape = QPainterPath(QPointF(*points[0]))
            for point in points[1:]:
                shape.lineTo(*point)
            if closed:
                shape.closeSubpath()
            painter.drawPath(shape)

        if self.symbol == 'draw':
            path([(4, 20), (5, 15), (16, 4), (20, 8), (9, 19)], closed=True)
            line(13.5, 6.5, 17.5, 10.5)
        elif self.symbol == 'pan':
            path([(8, 12), (8, 5), (11, 5), (11, 11), (11, 3),
                  (14, 3), (14, 11), (14, 5), (17, 5), (17, 12),
                  (17, 8), (20, 8), (20, 16), (17, 21), (10, 21),
                  (4, 14), (4, 12), (6, 11), (8, 14)])
        elif self.symbol == 'remove':
            line(4, 6, 20, 6)
            path([(9, 6), (9, 3), (15, 3), (15, 6)])
            path([(6, 6), (7, 20), (17, 20), (18, 6)])
            line(10, 10, 10, 16)
            line(14, 10, 14, 16)
        elif self.symbol == 'select':
            for points in ([(4, 8), (4, 4), (8, 4)],
                           [(16, 4), (20, 4), (20, 8)],
                           [(20, 16), (20, 20), (16, 20)],
                           [(8, 20), (4, 20), (4, 16)]):
                path(points)
            line(11, 4, 13, 4)
            line(20, 11, 20, 13)
            line(11, 20, 13, 20)
            line(4, 11, 4, 13)
        elif self.symbol == 'markers':
            for x, y in ((7, 8), (17, 7), (13, 17)):
                line(x - 2.5, y, x + 2.5, y)
                line(x, y - 2.5, x, y + 2.5)
        elif self.symbol in ('zoom-in', 'zoom-out'):
            painter.drawEllipse(QRectF(3, 3, 13, 13))
            line(14.5, 14.5, 21, 21)
            line(6.5, 9.5, 12.5, 9.5)
            if self.symbol == 'zoom-in':
                line(9.5, 6.5, 9.5, 12.5)
        elif self.symbol == 'restore':
            painter.drawArc(QRectF(5, 5, 15, 15), 140 * 16, -290 * 16)
            path([(4, 4), (4, 10), (10, 10)])
        elif self.symbol == 'loading':
            painter.drawArc(QRectF(4, 4, 16, 16), 40 * 16, 280 * 16)
            path([(20, 4), (20, 9), (15, 9)])
        elif self.symbol == 'done':
            path([(5, 12), (10, 17), (19, 7)])
        painter.restore()


class Icons:
    @classmethod
    def create_restore_icon(cls):
        return QIcon(_OutlineIconEngine('restore'))

    @classmethod
    def create_pan_icon(cls, size=24):
        return QIcon(_OutlineIconEngine('pan'))

    @classmethod
    def create_draw_icon(cls, size=24):
        return QIcon(_OutlineIconEngine('draw'))

    @classmethod
    def create_dot_icon(cls, size=24):
        return QIcon(_OutlineIconEngine('markers'))

    @classmethod
    def create_remove_icon(cls, size=24):
        return QIcon(_OutlineIconEngine('remove'))

    @classmethod
    def create_group_select_icon(cls, size=24):
        return QIcon(_OutlineIconEngine('select'))

    @classmethod
    def create_loading_icon(cls):
        return QIcon(_OutlineIconEngine('loading'))

    @classmethod
    def create_done_icon(cls):
        return QIcon(_OutlineIconEngine('done'))

    @classmethod
    def create_zoom_out_icon(cls):
        return QIcon(_OutlineIconEngine('zoom-out'))

    @classmethod
    def create_zoom_in_icon(cls):
        return QIcon(_OutlineIconEngine('zoom-in'))
