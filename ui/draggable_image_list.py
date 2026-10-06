import os

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import (QListWidget, QMenu, QPushButton, QStyle,
                               QStyledItemDelegate, QStyleOptionViewItem, QVBoxLayout)

from config.general import Config
from config.strings import Strings


class ImageRowDelegate(QStyledItemDelegate):
    """Keep Root Tracker status marks visible to the right of elided filenames."""

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        # Let the viewport determine row width instead of the longest filename.
        size.setWidth(0)
        return size

    def paint(self, painter, option, index):
        row = QStyleOptionViewItem(option)
        self.initStyleOption(row, index)
        view = self.parent()
        text_rect = view.style().subElementRect(QStyle.SE_ItemViewItemText, row, view)
        completed = bool(index.data(Qt.UserRole))
        processing = bool(index.data(Qt.UserRole + 2))
        error = bool(index.data(Qt.UserRole + 3))
        tick_x = row.rect.right() - 15
        if completed or processing or error:
            # Reserve the status mark's bounds plus a small text gap, accounting
            # for the padding already supplied by the item style.
            text_rect.setRight(min(text_rect.right(), tick_x - 8))
        row.text = row.fontMetrics.elidedText(
            row.text, Qt.ElideRight, max(0, text_rect.width()))
        view.style().drawControl(QStyle.CE_ItemViewItem, row, painter, view)
        if not (completed or processing or error):
            return
        light = row.palette.color(QPalette.Base).lightness() >= 128
        status_color = (QColor('#bb2222' if light else '#ff8888') if error else QColor('#1766a5' if light else '#80bfff') if processing
                        else QColor('#237a45' if light else '#74c69d'))
        color = (row.palette.color(QPalette.HighlightedText)
                 if row.state & QStyle.State_Selected else status_color)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(tick_x, row.rect.center().y())
        if error:
            painter.setPen(QPen(color, 2, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(QPointF(0, -5), QPointF(0, 1))
            painter.drawPoint(QPointF(0, 5))
            painter.restore()
            return
        if processing:
            # Paint the ellipsis to keep the same size across fonts/platforms.
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            for x in (-4, 0, 4):
                painter.drawEllipse(QPointF(x, 0), 1.1, 1.1)
            painter.restore()
            return
        painter.setPen(QPen(color, 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)
        path = QPainterPath()
        path.moveTo(-4, 0)
        path.lineTo(-1, 3)
        path.lineTo(5, -3)
        painter.drawPath(path)
        painter.restore()


class DraggableImageList(QListWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._main_window = main_window  # Store reference to MainWindow
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTextElideMode(Qt.ElideRight)
        self.setWordWrap(False)
        self.setItemDelegate(ImageRowDelegate(self))

        self.import_button = QPushButton(Strings.IMPORT, self.viewport())
        import_menu = QMenu(self.import_button)
        import_menu.addAction(Strings.IMPORT_FILES, main_window.import_files)
        import_menu.addAction(Strings.IMPORT_FOLDER, main_window.import_folder)
        self.import_button.setMenu(import_menu)
        empty_layout = QVBoxLayout(self.viewport())
        empty_layout.addStretch()
        empty_layout.addWidget(self.import_button, 0, Qt.AlignHCenter)
        empty_layout.addStretch()
        self.model().rowsInserted.connect(self._update_empty_state)
        self.model().rowsRemoved.connect(self._update_empty_state)
        self.model().modelReset.connect(self._update_empty_state)

    def _update_empty_state(self, *args):
        self.import_button.setVisible(self.count() == 0)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            # Only accept if at least one file is a supported image
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if os.path.isfile(path) and path.lower().endswith(Config.IMAGE_EXTENSIONS):
                    event.acceptProposedAction()
                    return
                elif os.path.isdir(path):
                    for root, _, fs in os.walk(path):
                        for f in fs:
                            if f.lower().endswith(Config.IMAGE_EXTENSIONS):
                                event.acceptProposedAction()
                                return
            event.ignore()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        # Same logic as dragEnterEvent
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            files = []
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if os.path.isfile(path) and path.lower().endswith(Config.IMAGE_EXTENSIONS):
                    files.append(path)
                elif os.path.isdir(path):
                    for root, _, fs in os.walk(path):
                        for f in fs:
                            if f.lower().endswith(Config.IMAGE_EXTENSIONS):
                                files.append(os.path.join(root, f))
            if files:
                self._main_window.load_files(files)  # Use main_window reference
            event.acceptProposedAction()
        else:
            super().dropEvent(event)
