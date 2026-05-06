"""Styl aplikacji — ciemny motyw."""

DARK_THEME = """
QMainWindow {
    background-color: #1e1e1e;
}

QWidget {
    background-color: #1e1e1e;
    color: #d4d4d4;
    font-family: 'Segoe UI', 'Helvetica', 'Arial', sans-serif;
    font-size: 13px;
}

QMainWindow::separator {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                              stop:0 #3a3a3a, stop:1 #2d2d2d);
    width: 6px;
}

QMenuBar {
    background-color: #2d2d2d;
    color: #d4d4d4;
    border-bottom: 1px solid #3a3a3a;
}

QMenuBar::item {
    padding: 6px 12px;
    background: transparent;
}

QMenuBar::item:selected {
    background-color: #3a3a3a;
}

QMenu {
    background-color: #2d2d2d;
    color: #d4d4d4;
    border: 1px solid #3a3a3a;
}

QMenu::item {
    padding: 6px 30px 6px 15px;
}

QMenu::item:selected {
    background-color: #094771;
}

QLabel {
    color: #d4d4d4;
}

QGroupBox {
    color: #d4d4d4;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 12px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px 0 5px;
}

QPushButton {
    background-color: #0e639c;
    color: #ffffff;
    border: none;
    border-radius: 3px;
    padding: 7px 18px;
    font-weight: bold;
    min-width: 70px;
}

QPushButton:hover {
    background-color: #1177bb;
}

QPushButton:pressed {
    background-color: #0d5a8f;
}

QPushButton:disabled {
    background-color: #3a3a3a;
    color: #6a6a6a;
}

QLineEdit {
    background-color: #3c3c3c;
    border: 1px solid #555555;
    border-radius: 3px;
    padding: 5px 8px;
    color: #d4d4d4;
    selection-background-color: #094771;
}

QLineEdit:focus {
    border: 1px solid #0e639c;
}

QComboBox {
    background-color: #3c3c3c;
    border: 1px solid #555555;
    border-radius: 3px;
    padding: 5px 8px;
    color: #d4d4d4;
    min-width: 120px;
}

QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #2d2d2d;
    color: #d4d4d4;
    selection-background-color: #094771;
    border: 1px solid #3a3a3a;
}

QTableWidget {
    background-color: #252526;
    alternate-background-color: #2d2d2d;
    border: 1px solid #3a3a3a;
    border-radius: 3px;
    gridline-color: #333333;
    selection-background-color: #094771;
    color: #d4d4d4;
}

QTableWidget::item {
    padding: 4px;
}

QTableWidget::item:selected {
    background-color: #094771;
    color: #ffffff;
}

QHeaderView::section {
    background-color: #2d2d2d;
    color: #cccccc;
    padding: 5px;
    border: none;
    border-bottom: 1px solid #3a3a3a;
    border-right: 1px solid #3a3a3a;
    font-weight: bold;
}

QProgressBar {
    border: 1px solid #3a3a3a;
    border-radius: 3px;
    text-align: center;
    background-color: #252526;
}

QProgressBar::chunk {
    background-color: #0e639c;
    border-radius: 2px;
}

QScrollBar:vertical {
    background-color: #1e1e1e;
    width: 10px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background-color: #424242;
    border-radius: 5px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #4f4f4f;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background-color: #1e1e1e;
    height: 10px;
    margin: 0;
}

QScrollBar::handle:horizontal {
    background-color: #424242;
    border-radius: 5px;
    min-width: 20px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #4f4f4f;
}

QStatusBar {
    background-color: #2d2d2d;
    color: #888888;
    border-top: 1px solid #3a3a3a;
}

QTabWidget::pane {
    border: 1px solid #3a3a3a;
    border-radius: 3px;
}

QTabBar::tab {
    background-color: #2d2d2d;
    color: #999999;
    padding: 8px 16px;
    border-top-left-radius: 3px;
    border-top-right-radius: 3px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #1e1e1e;
    color: #d4d4d4;
    border-bottom: 2px solid #0e639c;
}

QTabBar::tab:!selected {
    margin-top: 1px;
}

QTextEdit {
    background-color: #252526;
    border: 1px solid #3a3a3a;
    border-radius: 3px;
    padding: 6px;
    color: #d4d4d4;
    selection-background-color: #094771;
}

QCheckBox {
    color: #d4d4d4;
    spacing: 6px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #555555;
    border-radius: 3px;
    background-color: #3c3c3c;
}

QCheckBox::indicator:checked {
    background-color: #0e639c;
    border: 1px solid #0e639c;
}

QToolTip {
    background-color: #2d2d2d;
    color: #d4d4d4;
    border: 1px solid #0e639c;
    border-radius: 3px;
    padding: 4px;
}
"""
