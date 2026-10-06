from PySide6.QtCore import QSize


class Config:
    # Absolute image magnification: 1.0 means 100%, independent of window size.
    MIN_ZOOM_FACTOR = 0.01
    MAX_ZOOM_FACTOR = 16.0
    WINDOW_SIZE_MIN = QSize(800, 600)
    WINDOW_SIZE_PREFERRED = QSize(1_000, 600)

    APP_TITLE = "Root Worm Detector"
    APP_VERSION = "1.0"

    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
    IMAGE_EXTENSIONS_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
