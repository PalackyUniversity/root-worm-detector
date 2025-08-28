import os

from PySide6.QtWidgets import QListWidget

from config.general import Config


class DraggableImageList(QListWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._main_window = main_window  # Store reference to MainWindow

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
