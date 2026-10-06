"""Palette-based controls adapted from Root Tracker, with local assets."""

from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QMenu


def blend(base: QColor, accent: QColor, amount: float) -> QColor:
    """Mix two palette colors without losing the active light or dark theme."""
    channels = (
        round(a * (1 - amount) + b * amount)
        for a, b in zip(base.getRgb()[:3], accent.getRgb()[:3])
    )
    return QColor.fromRgb(*channels)


def is_light(palette: QPalette) -> bool:
    return palette.color(QPalette.ColorRole.Window).lightness() >= 128


def menu_border(palette: QPalette) -> str:
    surface = palette.color(QPalette.ColorRole.Button)
    text = palette.color(QPalette.ColorRole.WindowText)
    return blend(surface, text, 0.32 if is_light(palette) else 0.27).name()


def button_stylesheet(palette: QPalette) -> str:
    light = is_light(palette)
    window = palette.color(QPalette.ColorRole.Window)
    surface = palette.color(QPalette.ColorRole.Base if light else QPalette.ColorRole.Button)
    text = palette.color(QPalette.ColorRole.ButtonText)
    accent = palette.color(QPalette.ColorRole.Highlight)
    border = blend(surface, text, 0.22 if light else 0.24).name()
    hover = blend(surface, accent, 0.10 if light else 0.18).name()
    hover_border = blend(surface, accent, 0.40).name()
    pressed = blend(surface, accent, 0.20 if light else 0.26).name()
    disabled_bg = blend(window, surface, 0.30).name()
    disabled_text = blend(window, text, 0.42).name()
    divider = blend(window, text, .14 if light else .22).name()
    check_icon = (Path(__file__).parent / "icons" / "check.svg").as_posix()
    arrow_tone = "dark" if light else "light"
    down_icon = (Path(__file__).parent / "icons" / f"chevron-down-{arrow_tone}.svg").as_posix()
    spin_down_icon = (Path(__file__).parent / "icons" / f"spin-down-{arrow_tone}.svg").as_posix()
    up_icon = (Path(__file__).parent / "icons" / f"chevron-up-{arrow_tone}.svg").as_posix()
    return f"""
        QDialogButtonBox {{ dialogbuttonbox-buttons-have-icons: 0; }}
        QMenuBar {{ border-bottom: 1px solid {divider}; }}
        QStatusBar {{ border-top: 1px solid {divider}; }}
        QStatusBar::item {{ border: none; }}
        QSplitter::handle {{ background-color: {divider}; }}
        QGroupBox {{
            border: 1px solid {border};
            border-radius: 7px;
            margin-top: 10px;
            padding-top: 8px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0px 4px;
        }}
        QPushButton {{
            background-color: {surface.name()};
            color: {text.name()};
            border: 1px solid {border};
            border-radius: 5px;
            padding: 6px 10px;
            min-height: 18px;
        }}
        QPushButton:hover {{
            background-color: {hover};
            border-color: {hover_border};
        }}
        QPushButton:checked {{ background-color: {pressed}; border-color: {accent.name()}; }}
        QPushButton[iconButton="true"] {{ padding: 6px; }}
        QPushButton:pressed {{ background-color: {pressed}; }}
        QPushButton:focus {{ border-color: {accent.name()}; }}
        QPushButton:disabled {{
            background-color: {disabled_bg};
            color: {disabled_text};
            border-color: {border};
        }}
        QToolButton {{
            background: transparent;
            border: 1px solid transparent;
            border-radius: 5px;
        }}
        QToolButton:hover {{
            background-color: {hover};
            border-color: {border};
        }}
        QToolButton:pressed {{ background-color: {pressed}; }}
        QToolButton:focus {{ border-color: {accent.name()}; }}
        QLineEdit {{
            background-color: {surface.name()}; color: {text.name()};
            border: 1px solid {border}; border-radius: 5px;
            padding: 5px 8px 6px 8px; min-height: 18px;
            selection-background-color: {accent.name()};
            selection-color: palette(highlighted-text);
        }}
        QLineEdit:hover {{ border-color: {hover_border}; }}
        QLineEdit:focus {{ border-color: {accent.name()}; }}
        QLineEdit:disabled {{ background-color: {disabled_bg}; color: {disabled_text}; border-color: {border}; }}
        QComboBox {{
            combobox-popup: 0;
            background-color: {surface.name()}; color: {text.name()};
            border: 1px solid {border}; border-radius: 5px;
            padding: 5px 24px 6px 8px; min-height: 18px;
        }}
        QComboBox:hover {{ background-color: {hover}; border-color: {hover_border}; }}
        QComboBox:focus {{ border-color: {accent.name()}; }}
        QComboBox:on {{ background-color: {pressed}; border-color: {accent.name()}; }}
        QComboBox:disabled {{ background-color: {disabled_bg}; color: {disabled_text}; border-color: {border}; }}
        QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: top right; width: 22px; border: none; }}
        QComboBox::down-arrow {{ image: url("{down_icon}"); width: 12px; height: 12px; }}
        QComboBox QAbstractItemView {{
            background-color: {surface.name()}; color: {text.name()};
            border: 1px solid {border}; border-radius: 5px; padding: 3px;
            outline: none; selection-background-color: {accent.name()};
            selection-color: palette(highlighted-text);
        }}
        QComboBox QAbstractItemView::item {{ min-height: 24px; padding: 2px 6px; margin: 1px; border-radius: 4px; }}
        QComboBox QAbstractItemView::item:hover:!selected {{ background-color: {hover}; }}
        QComboBox QAbstractItemView::item:selected {{ background-color: {accent.name()}; color: palette(highlighted-text); }}
        QAbstractSpinBox {{
            background-color: {surface.name()}; color: {text.name()};
            border: 1px solid {border}; border-radius: 5px;
            padding: 3px 24px 3px 8px; min-height: 18px;
            selection-background-color: {accent.name()};
            selection-color: palette(highlighted-text);
        }}
        QAbstractSpinBox:hover {{ border-color: {hover_border}; }}
        QAbstractSpinBox:focus {{ border-color: {accent.name()}; }}
        QAbstractSpinBox:disabled {{ background-color: {disabled_bg}; color: {disabled_text}; }}
        QAbstractSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right; width: 22px; border: none; border-top-right-radius: 5px; margin: 1px; }}
        QAbstractSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right; width: 22px; border: none; border-bottom-right-radius: 5px; margin: 1px; }}
        QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{ background-color: {hover}; }}
        QAbstractSpinBox::up-button:pressed, QAbstractSpinBox::down-button:pressed {{ background-color: {pressed}; }}
        QAbstractSpinBox::up-arrow {{ image: url("{up_icon}"); width: 12px; height: 12px; position: relative; top: 2px; }}
        QAbstractSpinBox::down-arrow {{ image: url("{spin_down_icon}"); width: 12px; height: 12px; position: relative; top: -2px; }}
        QCheckBox {{ spacing: 7px; min-height: 22px; }}
        QCheckBox::indicator {{
            width: 16px; height: 16px; border: 1px solid {border};
            border-radius: 4px; background-color: {surface.name()};
        }}
        QCheckBox::indicator:hover {{ background-color: {hover}; border-color: {hover_border}; }}
        QCheckBox::indicator:checked {{ background-color: {accent.name()}; border-color: {accent.name()}; image: url("{check_icon}"); }}
        QCheckBox::indicator:checked:hover {{ background-color: {accent.lighter(110).name()}; }}
        QCheckBox::indicator:focus {{ border-color: {accent.name()}; }}
        QCheckBox::indicator:disabled {{ background-color: {disabled_bg}; border-color: {border}; }}
        QCheckBox::indicator:checked:disabled {{ background-color: {disabled_text}; }}
    """



def scrollbar_stylesheet(palette: QPalette) -> str:
    """Shared scrollbar geometry for navigation trees and image viewports."""
    base = palette.color(QPalette.ColorRole.Base)
    text = palette.color(QPalette.ColorRole.Text)
    handle = blend(base, text, .30).name()
    handle_hover = blend(base, text, .46).name()
    return f"""
        QScrollBar:vertical {{ background: transparent; width: 12px; margin: 2px; }}
        QScrollBar::handle:vertical {{ background: {handle}; min-height: 28px; border-radius: 4px; }}
        QScrollBar::handle:vertical:hover {{ background: {handle_hover}; }}
        QScrollBar::handle:vertical:pressed {{ background: {handle_hover}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; border: none; }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
        QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 2px; }}
        QScrollBar::handle:horizontal {{ background: {handle}; min-width: 28px; border-radius: 4px; }}
        QScrollBar::handle:horizontal:hover, QScrollBar::handle:horizontal:pressed {{ background: {handle_hover}; }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; border: none; }}
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: transparent; }}
        QAbstractScrollArea::corner {{ background: transparent; border: none; }}
    """



_MENU_STYLE = """
    QMenu {
        background-color: palette(button);
        border: 1px solid MENU_BORDER;
        border-radius: 7px;
        padding: 3px;
    }
    QMenu::item {
        padding: 5px 10px 5px 12px;
        border-radius: 4px;
    }
    QMenu::item:selected {
        background-color: palette(highlight);
        color: palette(highlighted-text);
    }
    QMenu::indicator {
        width: 16px;
        height: 16px;
        subcontrol-position: center left;
        left: 4px;
        border: 1px solid MENU_BORDER;
        border-radius: 4px;
        background-color: palette(base);
    }
    QMenu::indicator:checked {
        background-color: palette(highlight);
        border-color: palette(highlight);
        image: url("CHECK_ICON");
    }
    QMenu::indicator:checked:selected { border-color: palette(highlighted-text); }
    QMenu::separator {
        height: 1px;
        background-color: SEPARATOR_COLOR;
        margin: 6px 10px;
        border: none;
    }
"""


def application_stylesheet(palette: QPalette) -> str:
    check_icon = (Path(__file__).parent / "icons" / "check.svg").as_posix()
    surface = palette.color(QPalette.ColorRole.Button)
    text = palette.color(QPalette.ColorRole.WindowText)
    accent = palette.color(QPalette.ColorRole.Highlight)
    hover = blend(palette.color(QPalette.ColorRole.Base), accent, .12).name()
    menus = (_MENU_STYLE.replace("MENU_BORDER", menu_border(palette))
             .replace("CHECK_ICON", check_icon)
             .replace("SEPARATOR_COLOR", blend(surface, text, .18).name()))
    return button_stylesheet(palette) + scrollbar_stylesheet(palette) + menus + f"""
        QMenuBar::item {{ background: transparent; font-size: 14px; padding: 6px 8px; margin: 4px 0px; }}
        QMenuBar::item:selected {{ background-color: palette(highlight); color: palette(highlighted-text); border-radius: 3px; }}
        QListWidget {{ background-color: palette(base); border: none; outline: none; }}
        QListWidget::item {{ padding: 5px 8px; margin: 1px 0px; border-radius: 4px; }}
        QListWidget::item:hover:!selected {{ background-color: {hover}; }}
        QListWidget::item:selected {{ background-color: palette(highlight); color: palette(highlighted-text); }}
        QScrollArea {{ border: none; }}
    """


class _MenuTransparency(QObject):
    """Give rounded menu backgrounds a transparent native popup surface."""

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Polish and isinstance(watched, QMenu):
            # Configure before the native window is shown, including submenus
            # and menus owned by buttons such as Import.
            watched.setWindowFlag(Qt.FramelessWindowHint, True)
            watched.setAttribute(Qt.WA_TranslucentBackground, True)
        return False


def apply_theme(app):
    """Style every window and popup, and follow system palette changes."""
    if not hasattr(app, "_menu_transparency"):
        app._menu_transparency = _MenuTransparency(app)
        app.installEventFilter(app._menu_transparency)
    app.setStyle("Fusion")
    app.setStyleSheet(application_stylesheet(app.palette()))
    app.paletteChanged.connect(lambda palette: app.setStyleSheet(application_stylesheet(palette)))
