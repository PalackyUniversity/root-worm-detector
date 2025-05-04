from PySide6.QtGui import QPixmap, QPainter, QPen, QIcon, QPalette
from PySide6.QtCore import Qt, QPoint, QRect
from PySide6.QtWidgets import QApplication

class Icons:
    @classmethod
    def __is_dark_mode(cls):
        bg = QApplication.instance().palette().color(QPalette.Window)
        return bg.value() < 128

    @classmethod
    def get_foreground_color(cls):
        return Qt.white if cls.__is_dark_mode() else Qt.black
    
    @classmethod
    def __prepare_pixmap(cls, size: int, color=Qt.transparent):
        pixmap = QPixmap(size, size)
        pixmap.fill(color)
        return pixmap

    @classmethod
    def create_dot_icon(cls, size=24):
        pixmap = cls.__prepare_pixmap(size=size)
        painter = QPainter(pixmap)
        painter.setPen(QPen(Icons.get_foreground_color()))
        painter.setBrush(Icons.get_foreground_color())
        radius = size // 4
        center = QPoint(size // 2, size // 2)
        painter.drawEllipse(center, radius, radius)
        painter.end()
        return QIcon(pixmap)

    @classmethod
    def create_plus_icon(cls, size=24):
        pixmap = cls.__prepare_pixmap(size=size)
        painter = QPainter(pixmap)
        painter.setPen(QPen(Icons.get_foreground_color(), 2))
        painter.drawLine(size // 2, 4, size // 2, size - 4)
        painter.drawLine(4, size // 2, size - 4, size // 2)
        painter.end()
        return QIcon(pixmap)

    @classmethod
    def create_remove_icon(cls, size=24):
        pixmap = cls.__prepare_pixmap(size=size)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Icons.get_foreground_color(), 2))
        painter.drawRoundedRect(QRect(4, 3, 16, 4), 2, 2)
        painter.drawArc(QRect(8, 0, 8, 8), 0 * 16, 180 * 16)
        painter.drawRoundedRect(QRect(6, 7, 12, 12), 2, 2)
        painter.end()
        return QIcon(pixmap)

    @classmethod
    def create_group_select_icon(cls, size=24):
        pixmap = cls.__prepare_pixmap(size=size)
        painter = QPainter(pixmap)
        painter.setPen(QPen(Icons.get_foreground_color(), 2, Qt.DashLine))
        painter.drawRect(2, 2, size - 4, size - 4)
        painter.end()
        return QIcon(pixmap)

    @classmethod
    def create_loading_icon(cls):
        return QIcon.fromTheme("emblem-synchronizing")

    @classmethod
    def create_done_icon(cls):
        return QIcon.fromTheme("emblem-default")

    @classmethod
    def create_zoom_out_icon(cls):
        return QIcon.fromTheme("zoom-out")

    @classmethod
    def create_zoom_in_icon(cls):
        return QIcon.fromTheme("zoom-in")
