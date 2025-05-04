from PySide6.QtCore import QSize


class Config:
    WINDOW_SIZE_MIN = QSize(800, 600)
    WINDOW_SIZE_PREFERRED = QSize(1_000, 600)

    APP_TITLE = "Root Worm Detector"
    APP_VERSION = "1.0"

    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
    IMAGE_EXTENSIONS_FILTER = "Images (*.png *.jpg *.jpeg *.bmp)"