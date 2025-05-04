# from PySide6.QtCore import QTranslator, QLocale
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from ui.main_window import MainWindow
import sys
import os


def main():
    app = QApplication(sys.argv)

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
    main()