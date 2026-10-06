# from PySide6.QtCore import QTranslator, QLocale
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from ui.main_window import MainWindow
from ui.theme import apply_theme
import sys
import os


def main():
    app = QApplication(sys.argv)
    apply_theme(app)

    # Load translations if available (e.g. i18n/app_<locale>.qm)
    # translator = QTranslator()
    # locale = QLocale.system().name()
    # translator.load(f"i18n/app_{locale}.qm")
    # app.installTranslator(translator)

    # Use logo
    icon_path = os.path.abspath("resources/logo.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    else:
        raise Exception("Missing resources: logo.png")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()