import os

from PySide6.QtCore import Qt
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
        row.text = row.fontMetrics.elidedText(
            row.text, Qt.ElideRight, max(0, text_rect.width() - 30))
        view.style().drawControl(QStyle.CE_ItemViewItem, row, painter, view)
        if not index.data(Qt.UserRole):
            return
        light = row.palette.color(QPalette.Base).lightness() >= 128
        color = (row.palette.color(QPalette.HighlightedText)
                 if row.state & QStyle.State_Selected
                 else QColor('#237a45' if light else '#74c69d'))
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(row.rect.right() - 15, row.rect.center().y())
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
