#!/usr/bin/env python3
"""vLLM / sglang Model Manager — punkt wejścia."""

import sys

from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.styles import DARK_THEME


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("vLLM / sglang Model Manager")
    app.setOrganizationName("vllm-manager")

    # Ciemny motyw
    app.setStyleSheet(DARK_THEME)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
