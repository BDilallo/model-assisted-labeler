"""Centralized dark theme: palette, stylesheet, and app icon.

Every color used across the UI is defined once here so the app reads as
one consistent, professional surface instead of per-widget styling.
"""

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import QApplication


class Color:
    WINDOW = "#1c1d21"
    BASE = "#202126"
    SURFACE = "#26282e"
    SURFACE_RAISED = "#2e3038"
    BORDER = "#3a3c44"
    BORDER_STRONG = "#4a4d57"

    TEXT_PRIMARY = "#e9eaed"
    TEXT_SECONDARY = "#9a9da8"
    TEXT_DISABLED = "#5c5e66"

    ACCENT = "#4f8cff"
    ACCENT_HOVER = "#6fa1ff"
    ACCENT_PRESSED = "#3d72d8"
    ACCENT_TEXT = "#ffffff"

    DANGER = "#e5484d"
    DANGER_HOVER = "#ff6167"
    DANGER_SURFACE = "#3a2226"

    WARNING = "#e0a030"


def build_palette() -> QPalette:
    palette = QPalette()

    palette.setColor(QPalette.ColorRole.Window, QColor(Color.WINDOW))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(Color.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Base, QColor(Color.BASE))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(Color.SURFACE))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(Color.SURFACE_RAISED))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(Color.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Text, QColor(Color.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button, QColor(Color.SURFACE))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(Color.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(Color.DANGER))
    palette.setColor(QPalette.ColorRole.Link, QColor(Color.ACCENT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(Color.ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(Color.ACCENT_TEXT))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(Color.TEXT_DISABLED))

    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.Text,
        QColor(Color.TEXT_DISABLED),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.ButtonText,
        QColor(Color.TEXT_DISABLED),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.WindowText,
        QColor(Color.TEXT_DISABLED),
    )

    return palette


def build_stylesheet() -> str:
    return f"""
    QWidget {{
        color: {Color.TEXT_PRIMARY};
        font-family: "Segoe UI", sans-serif;
        font-size: 10pt;
        selection-background-color: {Color.ACCENT};
        selection-color: {Color.ACCENT_TEXT};
    }}

    QToolTip {{
        background-color: {Color.SURFACE_RAISED};
        color: {Color.TEXT_PRIMARY};
        border: 1px solid {Color.BORDER_STRONG};
        padding: 4px 6px;
        border-radius: 3px;
    }}

    QMainWindow, QDialog {{
        background-color: {Color.WINDOW};
    }}

    QLabel {{
        background: transparent;
    }}

    QLabel[muted="true"] {{
        color: {Color.TEXT_SECONDARY};
    }}

    QLabel[heading="true"] {{
        font-size: 13pt;
        font-weight: 600;
        color: {Color.TEXT_PRIMARY};
        padding-bottom: 6px;
        border-bottom: 1px solid {Color.BORDER};
        margin-bottom: 4px;
    }}

    QLabel[warning="true"] {{
        color: {Color.WARNING};
    }}

    QFrame[card="true"] {{
        background-color: {Color.SURFACE};
        border: 1px solid {Color.BORDER};
        border-radius: 6px;
    }}

    QFrame#imageFilterBar {{
        background-color: {Color.SURFACE};
        border: 1px solid {Color.BORDER};
        border-radius: 6px;
    }}

    QFrame#sessionTile {{
        background-color: {Color.SURFACE};
        border: 1px solid {Color.BORDER};
        border-radius: 8px;
    }}

    QFrame#sessionTile:hover {{
        border: 1px solid {Color.BORDER_STRONG};
        background-color: {Color.SURFACE_RAISED};
    }}

    QMenuBar {{
        background-color: {Color.BASE};
        border-bottom: 1px solid {Color.BORDER};
        padding: 2px;
    }}

    QToolBar#topBar {{
        background-color: {Color.BASE};
        border: none;
        border-bottom: 1px solid {Color.BORDER};
        padding: 4px 8px;
        spacing: 6px;
    }}

    QToolBar#topBar QToolButton {{
        border-radius: 4px;
        padding: 5px 12px;
    }}

    QMenuBar::item {{
        background: transparent;
        padding: 4px 10px;
        border-radius: 4px;
    }}

    QMenuBar::item:selected {{
        background-color: {Color.SURFACE_RAISED};
    }}

    QMenu {{
        background-color: {Color.SURFACE_RAISED};
        border: 1px solid {Color.BORDER_STRONG};
        padding: 4px;
    }}

    QMenu::item {{
        padding: 5px 24px 5px 12px;
        border-radius: 4px;
    }}

    QMenu::item:selected {{
        background-color: {Color.ACCENT};
        color: {Color.ACCENT_TEXT};
    }}

    QMenu::separator {{
        height: 1px;
        background: {Color.BORDER};
        margin: 4px 6px;
    }}

    QStatusBar {{
        background-color: {Color.BASE};
        border-top: 1px solid {Color.BORDER};
        color: {Color.TEXT_SECONDARY};
    }}

    QPushButton {{
        background-color: {Color.SURFACE_RAISED};
        border: 1px solid {Color.BORDER_STRONG};
        border-radius: 4px;
        padding: 6px 14px;
        color: {Color.TEXT_PRIMARY};
    }}

    QPushButton:hover {{
        background-color: #363842;
        border-color: {Color.BORDER_STRONG};
    }}

    QPushButton:pressed {{
        background-color: {Color.SURFACE};
    }}

    QPushButton:disabled {{
        color: {Color.TEXT_DISABLED};
        background-color: {Color.SURFACE};
        border-color: {Color.BORDER};
    }}

    QPushButton:checkable:checked {{
        background-color: {Color.ACCENT};
        border-color: {Color.ACCENT};
        color: {Color.ACCENT_TEXT};
    }}

    QPushButton[cta="primary"] {{
        background-color: {Color.ACCENT};
        border: 1px solid {Color.ACCENT};
        color: {Color.ACCENT_TEXT};
        font-weight: 600;
    }}

    QPushButton[cta="primary"]:hover {{
        background-color: {Color.ACCENT_HOVER};
        border-color: {Color.ACCENT_HOVER};
    }}

    QPushButton[cta="primary"]:pressed {{
        background-color: {Color.ACCENT_PRESSED};
        border-color: {Color.ACCENT_PRESSED};
    }}

    QPushButton[cta="primary"]:disabled {{
        background-color: {Color.SURFACE};
        border-color: {Color.BORDER};
        color: {Color.TEXT_DISABLED};
    }}

    QPushButton[cta="destructive"] {{
        background-color: transparent;
        border: 1px solid {Color.DANGER};
        color: {Color.DANGER};
    }}

    QPushButton[cta="destructive"]:hover {{
        background-color: {Color.DANGER_SURFACE};
    }}

    QPushButton[cta="destructive"]:pressed {{
        background-color: {Color.DANGER_SURFACE};
        border-color: {Color.DANGER_HOVER};
    }}

    QToolButton {{
        background-color: {Color.SURFACE_RAISED};
        border: 1px solid {Color.BORDER_STRONG};
        border-radius: 11px;
        color: {Color.TEXT_SECONDARY};
    }}

    QToolButton:hover {{
        color: {Color.TEXT_PRIMARY};
        border-color: {Color.ACCENT};
    }}

    QLineEdit, QSpinBox, QComboBox {{
        background-color: {Color.BASE};
        border: 1px solid {Color.BORDER_STRONG};
        border-radius: 4px;
        padding: 4px 6px;
    }}

    QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{
        border: 1px solid {Color.ACCENT};
    }}

    QLineEdit:read-only {{
        color: {Color.TEXT_SECONDARY};
    }}

    QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled {{
        color: {Color.TEXT_DISABLED};
        background-color: {Color.SURFACE};
    }}

    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}

    QComboBox QAbstractItemView {{
        background-color: {Color.SURFACE_RAISED};
        border: 1px solid {Color.BORDER_STRONG};
        selection-background-color: {Color.ACCENT};
        selection-color: {Color.ACCENT_TEXT};
        outline: none;
    }}

    QSpinBox::up-button, QSpinBox::down-button {{
        background-color: {Color.SURFACE_RAISED};
        border-left: 1px solid {Color.BORDER};
        width: 16px;
    }}

    QListWidget {{
        background-color: {Color.BASE};
        border: 1px solid {Color.BORDER};
        border-radius: 4px;
        outline: none;
        alternate-background-color: {Color.SURFACE};
    }}

    QListWidget::item {{
        padding: 5px 6px;
        border-radius: 3px;
    }}

    QListWidget::item:selected {{
        background-color: {Color.ACCENT};
        color: {Color.ACCENT_TEXT};
    }}

    QListWidget::item:hover:!selected {{
        background-color: {Color.SURFACE_RAISED};
    }}

    QScrollArea {{
        border: none;
    }}

    QSplitter::handle {{
        background-color: {Color.WINDOW};
    }}

    QSplitter::handle:hover {{
        background-color: {Color.ACCENT};
    }}

    QScrollBar:vertical {{
        background: transparent;
        width: 12px;
        margin: 0;
    }}

    QScrollBar::handle:vertical {{
        background: {Color.BORDER_STRONG};
        min-height: 24px;
        border-radius: 5px;
        margin: 2px;
    }}

    QScrollBar::handle:vertical:hover {{
        background: {Color.ACCENT};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}

    QScrollBar:horizontal {{
        background: transparent;
        height: 12px;
        margin: 0;
    }}

    QScrollBar::handle:horizontal {{
        background: {Color.BORDER_STRONG};
        min-width: 24px;
        border-radius: 5px;
        margin: 2px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background: {Color.ACCENT};
    }}

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}

    QProgressDialog {{
        background-color: {Color.WINDOW};
    }}

    QProgressBar {{
        background-color: {Color.BASE};
        border: 1px solid {Color.BORDER};
        border-radius: 4px;
        text-align: center;
        color: {Color.TEXT_PRIMARY};
    }}

    QProgressBar::chunk {{
        background-color: {Color.ACCENT};
        border-radius: 3px;
    }}

    QMessageBox {{
        background-color: {Color.WINDOW};
    }}
    """


def build_app_icon() -> QIcon:
    """Draw a small bounding-box glyph icon so the app doesn't rely on
    an external asset file."""
    size = 64
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(Color.ACCENT))
    painter.drawRoundedRect(2, 2, size - 4, size - 4, 14, 14)

    box_pen = QPen(QColor(Color.ACCENT_TEXT))
    box_pen.setWidthF(4.0)
    painter.setPen(box_pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(16, 16, size - 32, size - 32, 4, 4)

    corner_radius = 4.0
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(Color.ACCENT_TEXT))

    for x, y in ((16, 16), (size - 16, 16), (16, size - 16), (size - 16, size - 16)):
        painter.drawEllipse(QPointF(x, y), corner_radius, corner_radius)

    painter.end()

    return QIcon(pixmap)


def apply_theme(application: QApplication) -> None:
    """Apply the dark palette, stylesheet, and window icon app-wide."""
    application.setPalette(build_palette())
    application.setStyleSheet(build_stylesheet())
    application.setWindowIcon(build_app_icon())
