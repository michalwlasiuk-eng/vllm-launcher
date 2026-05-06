"""Główne okno aplikacji — vLLM / sglang Model Manager."""

import os
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QProcess, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from backend import config, model_scanner
from backend.model_scanner import (
    EngineSupport,
    ModelFormat,
    ModelInfo,
    scan_models_directory,
)


class CustomStatusBar(QStatusBar):
    """Status bar z ikonami i kolorami statusu."""

    def __init__(self):
        super().__init__()
        self._status_indicator = QLabel("●")
        self._status_indicator.setStyleSheet(
            "color: #4caf50; font-size: 16px; font-weight: bold;"
        )
        self.addWidget(self._status_indicator)

    def set_status(self, msg: str, level: str = "info"):
        """Ustaw tekst + kolor wskaźnika.

        level: 'info' (zielony), 'warn' (żółty), 'error' (czerwony)
        """
        colors = {
            "info": "#4caf50",
            "warn": "#ff9800",
            "error": "#f44336",
        }
        self._status_indicator.setStyleSheet(
            f"color: {colors.get(level, '#4caf50')}; "
            f"font-size: 16px; font-weight: bold;"
        )
        self.showMessage(msg)


class ModelDetailWidget(QWidget):
    """Widget do wyświetlania szczegółów modelu."""

    vllm_start_clicked = pyqtSignal(object)
    sglang_start_clicked = pyqtSignal(object)
    llama_start_clicked = pyqtSignal(object)
    vllm_stop_clicked = pyqtSignal()
    sglang_stop_clicked = pyqtSignal()
    llama_stop_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self.detail_text = QPlainTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setPlaceholderText("Wybierz model z tabeli...")
        layout.addWidget(self.detail_text)

        self._current_model: Optional[ModelInfo] = None

        # Przyciski start/stop
        self.button_layout = QHBoxLayout()
        self.vllm_start_btn = QPushButton("🚀 Start vLLM")
        self.vllm_stop_btn = QPushButton("⏹️ Stop vLLM")
        self.sglang_start_btn = QPushButton("🚀 Start sglang")
        self.sglang_stop_btn = QPushButton("⏹️ Stop sglang")
        self.llama_start_btn = QPushButton("🚀 Start llama.cpp")
        self.llama_stop_btn = QPushButton("⏹️ Stop llama")

        self.vllm_start_btn.setEnabled(False)
        self.vllm_stop_btn.setEnabled(False)
        self.sglang_start_btn.setEnabled(False)
        self.sglang_stop_btn.setEnabled(False)
        self.llama_start_btn.setEnabled(False)
        self.llama_stop_btn.setEnabled(False)

        self.vllm_start_btn.clicked.connect(self._on_vllm_start)
        self.vllm_stop_btn.clicked.connect(self._on_vllm_stop)
        self.sglang_start_btn.clicked.connect(self._on_sglang_start)
        self.sglang_stop_btn.clicked.connect(self._on_sglang_stop)
        self.llama_start_btn.clicked.connect(self._on_llama_start)
        self.llama_stop_btn.clicked.connect(self._on_llama_stop)

        self.button_layout.addWidget(self.vllm_start_btn)
        self.button_layout.addWidget(self.vllm_stop_btn)
        self.button_layout.addWidget(self.sglang_start_btn)
        self.button_layout.addWidget(self.sglang_stop_btn)
        self.button_layout.addWidget(self.llama_start_btn)
        self.button_layout.addWidget(self.llama_stop_btn)
        layout.addLayout(self.button_layout)

    def _on_vllm_start(self):
        if self._current_model:
            self.vllm_start_clicked.emit(self._current_model)

    def _on_vllm_stop(self):
        self.vllm_stop_clicked.emit()

    def _on_sglang_start(self):
        if self._current_model:
            self.sglang_start_clicked.emit(self._current_model)

    def _on_sglang_stop(self):
        self.sglang_stop_clicked.emit()

    def _on_llama_start(self):
        if self._current_model:
            self.llama_start_clicked.emit(self._current_model)

    def _on_llama_stop(self):
        self.llama_stop_clicked.emit()

    def set_model_info(self, info: ModelInfo):
        """Wyświetl informacje o modelu."""
        self._current_model = info

        text = f"""
Nazwa: {info.name}
Ścieżka: {info.path}
Rozmiar: {info.size_human}
Format: {info.format.value}
Engine: {info.engine_support.value}
"""
        if info.gguf_file:
            text += f"GGUF file: {info.gguf_file}\n"

        if info.arch:
            text += f"\n--- Architektura ---\n"
            text += f"Architektura: {info.arch}\n"
        if info.hidden_size:
            text += f"Hidden size: {info.hidden_size}\n"
        if info.num_attention_heads:
            text += f"Attention heads: {info.num_attention_heads}\n"
        if info.num_hidden_layers:
            text += f"Layers: {info.num_hidden_layers}\n"
        if info.vocab_size:
            text += f"Vocab size: {info.vocab_size}\n"

        if info.gguf_arch:
            text += f"\n--- GGUF ---\n"
            text += f"GGUF arch: {info.gguf_arch}\n"
            text += f"GGUF dtype: {info.gguf_dtype}\n"

        if info.notes:
            text += f"\n--- Uwagi ---\n"
            text += "\n".join(f"• {n}" for n in info.notes)

        if info.files[:20]:
            text += f"\n\n--- Pliki ({len(info.files)} łącznie) ---\n"
            for f in info.files[:20]:
                text += f"  • {f}\n"
            if len(info.files) > 20:
                text += f"  ... i {len(info.files) - 20} więcej\n"

        self.detail_text.setPlainText(text.strip())

        self.vllm_start_btn.setEnabled(
            info.engine_support in (EngineSupport.VLLM, EngineSupport.BOTH)
        )
        self.sglang_start_btn.setEnabled(
            info.engine_support in (EngineSupport.SGLANG, EngineSupport.BOTH)
        )
        # Llama.cpp obsługuje GGUF zawsze
        self.llama_start_btn.setEnabled(info.format == ModelFormat.GGUF)

    def set_engine_running(self, engine: str, running: bool):
        """Aktualizuj stan przycisków start/stop."""
        if engine == "vllm":
            self.vllm_start_btn.setEnabled(not running)
            self.vllm_stop_btn.setEnabled(running)
        elif engine == "sglang":
            self.sglang_start_btn.setEnabled(not running)
            self.sglang_stop_btn.setEnabled(running)
        elif engine == "llama":
            self.llama_start_btn.setEnabled(not running)
            self.llama_stop_btn.setEnabled(running)


class MainWindow(QMainWindow):
    """Okno główne aplikacji."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("vLLM / sglang / llama.cpp Model Manager")
        self.resize(1200, 750)
        self.setMinimumSize(900, 600)

        self.vllm_process = None
        self.sglang_process = None
        self.llama_process = None
        self.vllm_running = False
        self.sglang_running = False
        self.llama_running = False

        # --- Central widget ---
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Splitter: lewy panel + prawy panel ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # ---- Lewy panel — lista modeli ----
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(6)

        # --- Sekcja konfiguracji ---
        config_group = QGroupBox("Konfiguracja")
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(8)

        # vLLM venv
        self.vllm_path_box = self._make_path_row("vLLM venv:", "vllm_path")
        config_layout.addWidget(self.vllm_path_box)

        # sglang venv
        self.sglang_path_box = self._make_path_row("sglang venv:", "sglang_path")
        config_layout.addWidget(self.sglang_path_box)

        # llama.cpp binary
        self.llama_path_box = self._make_path_row("llama.cpp bin:", "llama_path")
        config_layout.addWidget(self.llama_path_box)

        # Katalog modeli
        self.models_dir_box = self._make_path_row("Katalog modeli:", "models_dir")
        config_layout.addWidget(self.models_dir_box)

        # Przyciski do parametrów uruchomienia
        params_btn_layout = QHBoxLayout()
        params_btn_layout.addStretch()
        self.btn_vllm_params = self._add_btn("⚙️ vLLM")
        self.btn_vllm_params.clicked.connect(
            lambda: self._show_launch_params_dialog("vllm")
        )
        params_btn_layout.addWidget(self.btn_vllm_params)
        self.btn_sglang_params = self._add_btn("⚙️ sglang")
        self.btn_sglang_params.clicked.connect(
            lambda: self._show_launch_params_dialog("sglang")
        )
        params_btn_layout.addWidget(self.btn_sglang_params)
        self.btn_llama_params = self._add_btn("⚙️ llama")
        self.btn_llama_params.clicked.connect(
            lambda: self._show_launch_params_dialog("llama")
        )
        params_btn_layout.addWidget(self.btn_llama_params)
        config_layout.addLayout(params_btn_layout)

        left_layout.addWidget(config_group)

        # --- Przyciski akcji ---
        btn_layout = QHBoxLayout()
        self.btn_scan = self._add_btn("📂 Skanuj")
        btn_layout.addWidget(self.btn_scan)
        self.btn_scan.clicked.connect(self._on_scan_clicked)
        self.btn_refresh = self._add_btn("🔄 Odśwież")
        btn_layout.addWidget(self.btn_refresh)
        self.btn_refresh.clicked.connect(self._on_refresh_clicked)
        self.btn_auto = self._add_btn("🔍 Auto-wykryj")
        btn_layout.addWidget(self.btn_auto)
        self.btn_auto.clicked.connect(self._auto_detect_venvs)
        btn_layout.addStretch()
        left_layout.addLayout(btn_layout)

        # --- Filtry ---
        filter_layout = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(
            ["Wszystkie", "GGUF", "safetensors", "vLLM", "sglang"]
        )
        self.filter_combo.setMinimumWidth(140)
        filter_layout.addWidget(QLabel("Filtr:"))
        filter_layout.addWidget(self.filter_combo)
        filter_layout.addStretch()
        left_layout.addLayout(filter_layout)

        # Tabela modeli
        self.model_table = QTableWidget()
        self.model_table.setColumnCount(6)
        self.model_table.setHorizontalHeaderLabels(
            ["Nazwa", "Ścieżka", "Rozmiar", "Format", "Engine", "Uwagi"]
        )
        self.model_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.model_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.model_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.model_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self.model_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
        )
        self.model_table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Stretch
        )
        self.model_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.model_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.model_table.setAlternatingRowColors(True)
        self.model_table.verticalHeader().setVisible(False)
        self.model_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.model_table.customContextMenuRequested.connect(self._show_context_menu)
        left_layout.addWidget(self.model_table)

        splitter.addWidget(left_widget)

        # ---- Prawy panel — szczegóły + logi ----
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(6)

        # Tabs
        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs)

        # Tab: Szczegóły modelu
        self.detail_widget = ModelDetailWidget()
        self.detail_widget.vllm_start_clicked.connect(self._launch_vllm)
        self.detail_widget.sglang_start_clicked.connect(self._launch_sglang)
        self.detail_widget.llama_start_clicked.connect(self._launch_llama)
        self.detail_widget.vllm_stop_clicked.connect(lambda: self._stop_process("vllm"))
        self.detail_widget.sglang_stop_clicked.connect(
            lambda: self._stop_process("sglang")
        )
        self.detail_widget.llama_stop_clicked.connect(
            lambda: self._stop_process("llama")
        )
        self.tabs.addTab(self.detail_widget, "Szczegóły modelu")

        # Tab: Logi
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Logi z procesu pojawią się tutaj…")
        log_layout = QVBoxLayout()
        log_layout.addWidget(self.log_view)
        btn_clear = self._add_btn("Wyczyść logi")
        btn_clear.clicked.connect(self.log_view.clear)
        log_layout.addWidget(btn_clear)
        log_widget = QWidget()
        log_widget.setLayout(log_layout)
        self.tabs.addTab(log_widget, "Logi")

        splitter.addWidget(right_widget)
        splitter.setSizes([500, 700])

        # --- Status bar ---
        self.setStatusBar(CustomStatusBar())
        self.status("Gotowe")

        # --- Dock: GPU monitor ---
        from gui.gpu_widget import GPUMonitorWidget

        self.gpu_dock = QDockWidget("GPU Monitor", self)
        self.gpu_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.gpu_widget = GPUMonitorWidget()
        self.gpu_dock.setWidget(self.gpu_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.gpu_dock)

        # --- Menu ---
        self._create_menu()

        # Wczytaj konfigurację
        self._load_config_to_ui()

        # Sygnały tabeli
        self.model_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.filter_combo.currentIndexChanged.connect(self._on_filter_changed)

    def _create_menu(self):
        menu = self.menuBar()

        # Plik
        file_menu = menu.addMenu("&Plik")
        act_open = QAction("📂 Otwórz katalog", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._on_scan_clicked)
        file_menu.addAction(act_open)
        file_menu.addSeparator()
        act_exit = QAction("Zakończ", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # Widok
        view_menu = menu.addMenu("&Widok")
        act_gpu = QAction("GPU Monitor", self)
        act_gpu.setCheckable(True)
        act_gpu.setChecked(True)
        act_gpu.toggled.connect(self.gpu_dock.setVisible)
        view_menu.addAction(act_gpu)
        view_menu.addSeparator()
        act_fullscreen = QAction("Pełny ekran", self)
        act_fullscreen.setShortcut("F11")
        act_fullscreen.triggered.connect(
            lambda: (
                self.showMaximized() if not self.isMaximized() else self.showNormal()
            )
        )
        view_menu.addAction(act_fullscreen)

        # Pomoc
        help_menu = menu.addMenu("&Pomoc")
        act_about = QAction("O aplikacji", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _show_about(self):
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.about(
            self,
            "O aplikacji",
            "<h3>vLLM / sglang / llama.cpp Model Manager</h3>"
            "<p>Program do zarządzania modelami LLM.</p>"
            "<p>Obsługuje: vLLM, sglang, llama.cpp</p>",
        )

    def _make_path_row(self, label_text: str, config_key: str) -> QWidget:
        """Pasek: etykieta + QLineEdit + browse button + status."""
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 2)
        layout.setSpacing(6)

        label = QLabel(label_text)
        label.setFixedWidth(120)
        layout.addWidget(label)

        path_edit = QLineEdit()
        path_edit.setPlaceholderText("Wskaż katalog...")
        path_edit.setObjectName(config_key)
        path_edit.textChanged.connect(lambda: self._on_path_changed(config_key))
        setattr(self, f"{config_key}_edit", path_edit)
        layout.addWidget(path_edit, 1)

        browse_btn = self._add_btn("...")
        browse_btn.clicked.connect(lambda _, key=config_key: self._browse_path(key))
        layout.addWidget(browse_btn)

        # Status indicator
        status_label = QLabel("")
        status_label.setObjectName(f"{config_key}_status")
        status_label.setFixedWidth(30)
        status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        setattr(self, f"{config_key}_status", status_label)
        layout.addWidget(status_label)

        return w

    def _browse_path(self, config_key: str) -> None:
        """Otwórz dialog wyboru katalogu."""
        dir_dialog = QFileDialog(self)
        dir_dialog.setFileMode(QFileDialog.FileMode.Directory)
        dir_dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)

        if dir_dialog.exec():
            dirs = dir_dialog.selectedFiles()
            if dirs:
                path = dirs[0]
                edit = getattr(self, f"{config_key}_edit")
                edit.setText(path)

    def _on_path_changed(self, config_key: str) -> None:
        """Wywołany gdy użytkownik zmieni ścieżkę."""
        edit = getattr(self, f"{config_key}_edit")
        path = edit.text().strip()
        status = getattr(self, f"{config_key}_status")

        if not path:
            status.setText("")
            status.setStyleSheet("")
            return

        if config_key == "llama_path":
            # Sprawdź czy plik istnieje
            if os.path.exists(path):
                # Może być plikiem lub katalogiem (dla kompatybilności)
                if os.path.isfile(path):
                    if os.access(path, os.X_OK):
                        status.setText("✅")
                        status.setStyleSheet("color: #4caf50; font-size: 14px;")
                    else:
                        status.setText("⚠️")
                        status.setStyleSheet("color: #ff9800; font-size: 14px;")
                elif os.path.isdir(path):
                    # Szukaj llama-server w katalogu
                    candidates = [
                        os.path.join(path, "llama-server"),
                        os.path.join(path, "bin", "llama-server"),
                        os.path.join(path, "build", "bin", "llama-server"),
                    ]
                    for cand in candidates:
                        if os.path.isfile(cand) and os.access(cand, os.X_OK):
                            status.setText("✅")
                            status.setStyleSheet("color: #4caf50; font-size: 14px;")
                            self.llama_path_edit.setText(cand)  # Update to full path
                            break
                    else:
                        status.setText("⚠️")
                        status.setStyleSheet("color: #ff9800; font-size: 14px;")
                else:
                    status.setText("❌")
                    status.setStyleSheet("color: #f44336; font-size: 14px;")
            else:
                status.setText("❌")
                status.setStyleSheet("color: #f44336; font-size: 14px;")
            return

        if config_key == "models_dir":
            model_count = model_scanner.get_model_count(path)
            if model_count > 0:
                status.setText(f"📦{model_count}")
                status.setStyleSheet("color: #4caf50; font-size: 12px;")
            else:
                status.setText("❌")
                status.setStyleSheet("color: #f44336; font-size: 14px;")
            return

        # Sprawdź czy to venv (ma bin/python)
        has_python, py_path = config.detect_python(path)
        if not has_python:
            status.setText("❌")
            status.setStyleSheet("color: #f44336; font-size: 14px;")
            return

        # Sprawdź odpowiednie binarium
        if config_key == "vllm_path":
            has_vllm, vllm_bin = config.detect_binary(path, "vllm")
            if has_vllm:
                status.setText("✅")
                status.setStyleSheet("color: #4caf50; font-size: 14px;")
            else:
                status.setText("⚠️")
                status.setStyleSheet("color: #ff9800; font-size: 14px;")
        elif config_key == "sglang_path":
            has_sglang, sglang_bin = config.detect_binary(path, "sglang")
            if has_sglang:
                status.setText("✅")
                status.setStyleSheet("color: #4caf50; font-size: 14px;")
            else:
                status.setText("⚠️")
                status.setStyleSheet("color: #ff9800; font-size: 14px;")

        self._save_config_from_ui()

    def _add_btn(self, text: str):
        btn = QPushButton(text)
        btn.setMinimumHeight(28)
        return btn

    def _on_scan_clicked(self) -> None:
        """Otwórz dialog wyboru katalogu i rozpocznij skanowanie."""
        dir_dialog = QFileDialog(self)
        dir_dialog.setFileMode(QFileDialog.FileMode.Directory)
        dir_dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)

        if dir_dialog.exec():
            dirs = dir_dialog.selectedFiles()
            if dirs:
                self._start_scan(dirs[0])

    def _start_scan(self, root_dir: str) -> None:
        """Uruchamia skanowanie w tle."""
        cfg = config.load_config()
        cfg["models_dir"] = root_dir
        config.save_config(cfg)
        self.models_dir_edit.setText(root_dir)

        # Anuluj poprzedni wątek
        if hasattr(self, "scan_thread") and self.scan_thread.isRunning():
            self.scan_thread.terminate()
            self.scan_thread.wait()

        # Utwórz i uruchom wątek
        self.scan_thread = ModelScanThread(root_dir, max_depth=3)
        self.scan_thread.finished.connect(self._on_scan_finished)
        self.scan_thread.error.connect(self._on_scan_error)

        self.status(f"Skanowanie: {root_dir}")
        self.scan_thread.start()

    def _on_scan_finished(self, models: list) -> None:
        """Skanowanie zakończone."""
        self.model_data = models
        self._populate_table(models)
        self.status(f"Znaleziono {len(models)} modeli")

    def _on_scan_error(self, error: str) -> None:
        """Błąd skanowania."""
        self.status(f"Błąd skanowania: {error}", level="error")

    def _on_refresh_clicked(self) -> None:
        """Przycisk: Odśwież — ponowne skanowanie ostatniego katalogu."""
        cfg = config.load_config()
        models_dir = cfg.get("models_dir", "")
        if models_dir:
            self._start_scan(models_dir)
        else:
            self.status("Brak katalogu modeli", duration=3)

    def _auto_detect_venvs(self) -> None:
        """Auto-wykryj venv z vLLM i sglang."""
        self.status("Szukam venv...")
        found = config.scan_common_venv_locations()

        if not found:
            self.status("Nie znaleziono venv z vLLM/sglang", duration=5)
            return

        vllm_found = [f for f in found if f["has_vllm"]]
        sglang_found = [f for f in found if f["has_sglang"]]

        if vllm_found:
            venv = vllm_found[0]
            self.vllm_path_edit.setText(venv["path"])
            self.vllm_status.setText("✅")
            self.vllm_status.setStyleSheet("color: #4caf50; font-size: 14px;")
            self.status(f"vLLM wykryty: {venv['path']}")

        if sglang_found:
            venv = sglang_found[0]
            self.sglang_path_edit.setText(venv["path"])
            self.sglang_status.setText("✅")
            self.sglang_status.setStyleSheet("color: #4caf50; font-size: 14px;")
            self.status(f"sglang wykryty: {venv['path']}")

        self._save_config_from_ui()

    def _populate_table(self, models: list) -> None:
        """Wypełnij tabelę modelami."""
        self.model_table.setRowCount(0)
        self.model_table.setRowCount(len(models))

        for row, info in enumerate(models):
            self._set_model_row(row, info)

    def _set_model_row(self, row: int, info: ModelInfo) -> None:
        """Ustaw wiersz tabeli."""
        items = [
            info.name,
            info.path,
            info.size_human,
            info.format.value,
            info.engine_support.value,
            ", ".join(info.notes) if info.notes else "",
        ]

        for col, text in enumerate(items):
            item = QTableWidgetItem(text)
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
            )
            if col == 3:  # Format
                if info.format == ModelFormat.GGUF:
                    item.setForeground(Qt.GlobalColor.darkBlue)
            self.model_table.setItem(row, col, item)

    def _on_table_selection_changed(self) -> None:
        """Model wybrany z tabeli."""
        selected = self.model_table.selectedItems()
        if not selected:
            return

        row = self.model_table.row(selected[0])
        if not hasattr(self, "model_data") or row >= len(self.model_data):
            return

        info = self.model_data[row]
        self.detail_widget.set_model_info(info)
        self.status(f"Wybrano: {info.name}")

    def _on_filter_changed(self) -> None:
        """Filtruje modele."""
        if not hasattr(self, "model_data"):
            return

        filter_text = self.filter_combo.currentText()
        filtered = []

        for info in self.model_data:
            if filter_text == "Wszystkie":
                filtered.append(info)
            elif filter_text == "GGUF" and info.format == ModelFormat.GGUF:
                filtered.append(info)
            elif (
                filter_text == "safetensors" and info.format == ModelFormat.SAFETENSORS
            ):
                filtered.append(info)
            elif filter_text == "vLLM" and info.engine_support in (
                EngineSupport.VLLM,
                EngineSupport.BOTH,
            ):
                filtered.append(info)
            elif filter_text == "sglang" and info.engine_support in (
                EngineSupport.SGLANG,
                EngineSupport.BOTH,
            ):
                filtered.append(info)

        self.model_table.setRowCount(0)
        for row, info in enumerate(filtered):
            self._set_model_row(row, info)

    def _show_context_menu(self, pos) -> None:
        """Menu kontekstowe dla tabeli."""
        row = self.model_table.rowAt(pos.y())
        if row < 0 or not hasattr(self, "model_data") or row >= len(self.model_data):
            return

        info = self.model_data[row]
        menu = QMenu(self)

        act_launch_vllm = menu.addAction("🚀 Uruchom vLLM")
        act_launch_vllm.triggered.connect(lambda: self._launch_vllm(info))
        menu.addAction(act_launch_vllm)

        act_launch_sglang = menu.addAction("🚀 Uruchom sglang")
        act_launch_sglang.triggered.connect(lambda: self._launch_sglang(info))
        menu.addAction(act_launch_sglang)

        if info.format == ModelFormat.GGUF:
            act_launch_llama = menu.addAction("🚀 Uruchom llama.cpp")
            act_launch_llama.triggered.connect(lambda: self._launch_llama(info))
            menu.addAction(act_launch_llama)

        menu.exec(self.model_table.mapToGlobal(pos))

    def _launch_vllm(self, info: ModelInfo) -> None:
        """Generuje i uruchamia komendę vLLM."""
        cfg = config.load_config()
        venv_path = cfg.get("vllm", {}).get("venv_path", "")
        if not venv_path:
            self.status("Błąd: Nie ustawiono ścieżki do vLLM", duration=5)
            return

        has_bin, vllm_bin = config.detect_binary(venv_path, "vllm")
        if not has_bin:
            self.status("Błąd: Nie znaleziono binarium vllm", duration=5)
            return

        params = cfg.get("vllm", {}).get("launch_params", {})

        cmd_parts = [vllm_bin, "serve", info.model_path]
        cmd_parts += ["--host", str(params.get("host", "127.0.0.1"))]
        cmd_parts += ["--port", str(params.get("port", 8000))]

        tp = params.get("tensor_parallel_size", 1)
        if tp > 1:
            cmd_parts += ["--tensor-parallel-size", str(tp)]

        gpu_mem = params.get("gpu_memory_utilization", 0.9)
        cmd_parts += ["--gpu-memory-utilization", str(gpu_mem)]

        dtype = params.get("dtype", "auto")
        if dtype and dtype != "auto":
            cmd_parts += ["--dtype", dtype]

        max_len = params.get("max_model_len", 0)
        if max_len > 0:
            cmd_parts += ["--max-model-len", str(max_len)]

        if params.get("enforce_eager", False):
            cmd_parts.append("--enforce-eager")

        max_seqs = params.get("max_num_seqs", 256)
        if max_seqs != 256:
            cmd_parts += ["--max-num-seqs", str(max_seqs)]

        swap = params.get("swap_space", 4)
        if swap != 4:
            cmd_parts += ["--swap-space", str(swap)]

        api_key = params.get("api_key", "")
        if api_key:
            cmd_parts += ["--api-key", api_key]

        if params.get("trust_remote_code", True):
            cmd_parts.append("--trust-remote-code")

        ssl_cert = params.get("ssl_certfile", "")
        if ssl_cert:
            cmd_parts += ["--ssl-certfile", ssl_cert]

        tool_parser = params.get("tool_call_parser", "")
        if tool_parser:
            cmd_parts += ["--tool-call-parser", tool_parser]

        cmd = " ".join(cmd_parts)
        self._show_command_and_launch("vllm", cmd, info)

    def _launch_sglang(self, info: ModelInfo) -> None:
        """Generuje i uruchamia komendę sglang."""
        cfg = config.load_config()
        venv_path = cfg.get("sglang", {}).get("venv_path", "")
        if not venv_path:
            self.status("Błąd: Nie ustawiono ścieżki do sglang", duration=5)
            return

        has_bin, sglang_bin = config.detect_binary(venv_path, "sglang")
        if not has_bin:
            self.status("Błąd: Nie znaleziono binarium sglang", duration=5)
            return

        params = cfg.get("sglang", {}).get("launch_params", {})

        cmd_parts = [sglang_bin, "serve", "--model-path", info.model_path]
        cmd_parts += ["--host", str(params.get("host", "127.0.0.1"))]
        cmd_parts += ["--port", str(params.get("port", 8000))]

        tp = params.get("tensor_parallel_size", 1)
        if tp > 1:
            cmd_parts += ["--tp", str(tp)]

        mem_frac = params.get("mem_fraction_static", 0.9)
        cmd_parts += ["--mem-fraction-static", str(mem_frac)]

        dtype = params.get("dtype", "auto")
        if dtype and dtype != "auto":
            cmd_parts += ["--dtype", dtype]

        max_tokens = params.get("max_total_tokens", 0)
        if max_tokens > 0:
            cmd_parts += ["--max-total-tokens", str(max_tokens)]

        load_format = params.get("load_format", "auto")
        if load_format != "auto":
            cmd_parts += ["--load-format", load_format]
        elif info.format == ModelFormat.GGUF:
            cmd_parts += ["--load-format", "gguf"]

        quant = params.get("quantization", "")
        if quant:
            cmd_parts += ["--quantization", quant]

        if params.get("trust_remote_code", True):
            cmd_parts.append("--trust-remote-code")

        cmd = " ".join(cmd_parts)
        self._show_command_and_launch("sglang", cmd, info)

    def _launch_llama(self, info: ModelInfo) -> None:
        """Generuje i uruchamia komendę llama.cpp."""
        cfg = config.load_config()
        llama_path = cfg.get("llama", {}).get("binary_path", "")
        if not llama_path or not Path(llama_path).exists():
            self.status("Błąd: Nie ustawiono ścieżki do llama.cpp", duration=5)
            return

        params = cfg.get("llama", {}).get("launch_params", {})

        # Use model_path for GGUF files
        model_arg = info.model_path

        cmd_parts = [llama_path, "-m", model_arg]
        cmd_parts += ["--host", str(params.get("host", "127.0.0.1"))]
        cmd_parts += ["--port", str(params.get("port", 8080))]

        ctx_size = params.get("ctx_size", 0)
        if ctx_size > 0:
            cmd_parts += ["-c", str(ctx_size)]

        threads = params.get("threads", -1)
        if threads != -1:
            cmd_parts += ["-t", str(threads)]

        threads_batch = params.get("threads_batch", -1)
        if threads_batch != -1:
            cmd_parts += ["-tb", str(threads_batch)]

        batch_size = params.get("batch_size", 2048)
        if batch_size != 2048:
            cmd_parts += ["-b", str(batch_size)]

        ubatch_size = params.get("ubatch_size", 512)
        if ubatch_size != 512:
            cmd_parts += ["-ub", str(ubatch_size)]

        gpu_layers = params.get("gpu_layers", -1)
        if gpu_layers > 0:
            cmd_parts += ["-ngl", str(gpu_layers)]

        n_predict = params.get("n_predict", -1)
        if n_predict != -1:
            cmd_parts += ["-n", str(n_predict)]

        flash_attn = params.get("flash_attn", "auto")
        if flash_attn != "auto":
            cmd_parts += ["-fa", flash_attn]

        cache_type_k = params.get("cache_type_k", "f16")
        if cache_type_k != "f16":
            cmd_parts += ["-ctk", cache_type_k]

        cache_type_v = params.get("cache_type_v", "f16")
        if cache_type_v != "f16":
            cmd_parts += ["-ctv", cache_type_v]

        kv_offload = params.get("kv_offload", True)
        if not kv_offload:
            cmd_parts.append("--no-kv-offload")

        rope_scaling = params.get("rope_scaling", "linear")
        if rope_scaling and rope_scaling not in ("", "none"):
            cmd_parts += ["--rope-scaling", rope_scaling]

        rope_freq_base = params.get("rope_freq_base", 0)
        if rope_freq_base > 0:
            cmd_parts += ["--rope-freq-base", str(rope_freq_base)]

        rope_scale = params.get("rope_scale", 1.0)
        if rope_scale != 1.0:
            cmd_parts += ["--rope-scale", str(rope_scale)]

        main_gpu = params.get("main_gpu", 0)
        if main_gpu != 0:
            cmd_parts += ["-mg", str(main_gpu)]

        tensor_split = params.get("tensor_split", "")
        if tensor_split.strip():
            cmd_parts += ["-ts", tensor_split.strip()]

        split_mode = params.get("split_mode", "layer")
        if split_mode != "layer":
            cmd_parts += ["-sm", split_mode]

        if params.get("mlock", False):
            cmd_parts.append("--mlock")

        if params.get("mmap", True):
            cmd_parts.append("--mmap")
        else:
            cmd_parts.append("--no-mmap")

        if params.get("no_direct_io", False):
            cmd_parts.append("--no-direct-io")

        if params.get("numa", False):
            cmd_parts.append("--numa")

        lora = params.get("lora", "")
        if lora.strip():
            cmd_parts += ["--lora", lora.strip()]

        control_vector = params.get("control_vector", "")
        if control_vector.strip():
            cmd_parts += ["--control-vector", control_vector.strip()]

        keep = params.get("keep", 0)
        if keep != 0:
            cmd_parts += ["--keep", str(keep)]

        if params.get("log_disable", False):
            cmd_parts.append("--log-disable")

        log_file = params.get("log_file", "")
        if log_file.strip():
            cmd_parts += ["--log-file", log_file.strip()]

        if params.get("no_escape", False):
            cmd_parts.append("--no-escape")

        if params.get("verbose", False):
            cmd_parts.append("-v")

        cmd = " ".join(cmd_parts)
        self._show_command_and_launch("llama", cmd, info)

    def _show_launch_params_dialog(self, engine: str) -> None:
        """Pokazuje dialog z parametrami uruchomienia."""
        from PyQt6.QtWidgets import (
            QCheckBox,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QSpinBox,
            QTabWidget,
            QVBoxLayout,
            QWidget,
        )

        cfg = config.load_config()
        params = cfg.get(engine, {}).get("launch_params", {})

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Parametry — {engine}")
        dlg.resize(500, 600)

        main_layout = QVBoxLayout(dlg)

        # Sieć
        net_group = QGroupBox("Sieć")
        net_layout = QFormLayout(net_group)

        host_edit = QLineEdit(params.get("host", "127.0.0.1"))
        net_layout.addRow("Host:", host_edit)

        port_spin = QSpinBox()
        port_spin.setRange(1000, 65535)
        port_spin.setValue(params.get("port", 8080 if engine == "llama" else 8000))
        net_layout.addRow("Port:", port_spin)

        main_layout.addWidget(net_group)

        if engine == "llama":
            tabs = QTabWidget()

            # --- Performance tab ---
            perf_widget = QWidget()
            perf_layout = QFormLayout(perf_widget)

            threads_spin = QSpinBox()
            threads_spin.setRange(-1, 128)
            threads_spin.setValue(params.get("threads", -1))
            perf_layout.addRow("Threads (-t):", threads_spin)

            threads_batch_spin = QSpinBox()
            threads_batch_spin.setRange(-1, 128)
            threads_batch_spin.setValue(params.get("threads_batch", -1))
            perf_layout.addRow("Threads Batch (-tb):", threads_batch_spin)

            batch_spin = QSpinBox()
            batch_spin.setRange(1, 8192)
            batch_spin.setValue(params.get("batch_size", 2048))
            perf_layout.addRow("Batch Size (-b):", batch_spin)

            ubatch_spin = QSpinBox()
            ubatch_spin.setRange(1, 8192)
            ubatch_spin.setValue(params.get("ubatch_size", 512))
            perf_layout.addRow("Ubatch Size (-ub):", ubatch_spin)

            ctx_spin = QSpinBox()
            ctx_spin.setRange(0, 131072)
            ctx_spin.setSingleStep(1024)
            ctx_spin.setValue(params.get("ctx_size", 0))
            perf_layout.addRow("Context (-c):", ctx_spin)

            n_pred_spin = QSpinBox()
            n_pred_spin.setRange(-1, 16384)
            n_pred_spin.setValue(params.get("n_predict", -1))
            perf_layout.addRow("N Predict (-n):", n_pred_spin)

            flash_combo = QComboBox()
            flash_combo.addItems(["auto", "on", "off"])
            flash_combo.setCurrentText(params.get("flash_attn", "auto"))
            perf_layout.addRow("Flash Attn (-fa):", flash_combo)

            cache_k_combo = QComboBox()
            cache_k_combo.addItems(
                [
                    "f16",
                    "f32",
                    "bf16",
                    "q8_0",
                    "q4_0",
                    "q4_1",
                    "iq4_nl",
                    "q5_0",
                    "q5_1",
                    "turbo2",
                    "turbo3",
                    "turbo4",
                ]
            )
            cache_k_combo.setCurrentText(params.get("cache_type_k", "f16"))
            perf_layout.addRow("Cache K (-ctk):", cache_k_combo)

            cache_v_combo = QComboBox()
            cache_v_combo.addItems(
                [
                    "f16",
                    "f32",
                    "bf16",
                    "q8_0",
                    "q4_0",
                    "q4_1",
                    "iq4_nl",
                    "q5_0",
                    "q5_1",
                    "turbo2",
                    "turbo3",
                    "turbo4",
                ]
            )
            cache_v_combo.setCurrentText(params.get("cache_type_v", "f16"))
            perf_layout.addRow("Cache V (-ctv):", cache_v_combo)

            mmap_cb = QCheckBox()
            mmap_cb.setChecked(params.get("mmap", True))
            perf_layout.addRow("mmap (--mmap):", mmap_cb)

            mlock_cb = QCheckBox()
            mlock_cb.setChecked(params.get("mlock", False))
            perf_layout.addRow("mlock (--mlock):", mlock_cb)

            kv_offload_cb = QCheckBox()
            kv_offload_cb.setChecked(params.get("kv_offload", True))
            perf_layout.addRow("KV Offload:", kv_offload_cb)

            no_direct_io_cb = QCheckBox()
            no_direct_io_cb.setChecked(params.get("no_direct_io", False))
            perf_layout.addRow("No Direct IO:", no_direct_io_cb)

            numa_cb = QCheckBox()
            numa_cb.setChecked(params.get("numa", False))
            perf_layout.addRow("NUMA (--numa):", numa_cb)

            tabs.addTab(perf_widget, "Performance")

            # --- GPU tab ---
            gpu_widget = QWidget()
            gpu_layout = QFormLayout(gpu_widget)

            gpu_spin = QSpinBox()
            gpu_spin.setRange(-1, 256)
            gpu_spin.setValue(params.get("gpu_layers", -1))
            gpu_layout.addRow("GPU Layers (-ngl):", gpu_spin)

            main_gpu_spin = QSpinBox()
            main_gpu_spin.setRange(0, 7)
            main_gpu_spin.setValue(params.get("main_gpu", 0))
            gpu_layout.addRow("Main GPU (-mg):", main_gpu_spin)

            tensor_split_edit = QLineEdit(params.get("tensor_split", ""))
            gpu_layout.addRow("Tensor Split (-ts):", tensor_split_edit)

            split_mode_combo = QComboBox()
            split_mode_combo.addItems(["none", "layer", "row", "tensor"])
            split_mode_combo.setCurrentText(params.get("split_mode", "layer"))
            gpu_layout.addRow("Split Mode (-sm):", split_mode_combo)

            tabs.addTab(gpu_widget, "GPU")

            # --- Advanced tab ---
            adv_widget = QWidget()
            adv_layout = QFormLayout(adv_widget)

            rope_scaling_combo = QComboBox()
            rope_scaling_combo.addItems(["", "none", "linear", "yarn"])
            rope_scaling_combo.setCurrentText(params.get("rope_scaling", "linear"))
            adv_layout.addRow("Rope Scaling:", rope_scaling_combo)

            rope_freq_base_spin = QSpinBox()
            rope_freq_base_spin.setRange(0, 10000000)
            rope_freq_base_spin.setValue(params.get("rope_freq_base", 0))
            adv_layout.addRow("Rope Freq Base:", rope_freq_base_spin)

            rope_scale_spin = QDoubleSpinBox()
            rope_scale_spin.setRange(0.1, 100.0)
            rope_scale_spin.setSingleStep(0.1)
            rope_scale_spin.setValue(params.get("rope_scale", 1.0))
            adv_layout.addRow("Rope Scale:", rope_scale_spin)

            keep_spin = QSpinBox()
            keep_spin.setRange(-1, 131072)
            keep_spin.setValue(params.get("keep", 0))
            adv_layout.addRow("Keep (--keep):", keep_spin)

            lora_edit = QLineEdit(params.get("lora", ""))
            adv_layout.addRow("LoRA (--lora):", lora_edit)

            control_vector_edit = QLineEdit(params.get("control_vector", ""))
            adv_layout.addRow("Control Vector:", control_vector_edit)

            log_file_edit = QLineEdit(params.get("log_file", ""))
            adv_layout.addRow("Log File (--log-file):", log_file_edit)

            log_disable_cb = QCheckBox()
            log_disable_cb.setChecked(params.get("log_disable", False))
            adv_layout.addRow("Log Disable:", log_disable_cb)

            verbose_cb = QCheckBox()
            verbose_cb.setChecked(params.get("verbose", False))
            adv_layout.addRow("Verbose (-v):", verbose_cb)

            no_escape_cb = QCheckBox()
            no_escape_cb.setChecked(params.get("no_escape", False))
            adv_layout.addRow("No Escape:", no_escape_cb)

            tabs.addTab(adv_widget, "Advanced")

            main_layout.addWidget(tabs)

        elif engine == "vllm":
            vllm_group = QGroupBox("vLLM")
            vllm_layout = QFormLayout(vllm_group)

            tp_spin = QSpinBox()
            tp_spin.setRange(1, 8)
            tp_spin.setValue(params.get("tensor_parallel_size", 1))
            vllm_layout.addRow("Tensor Parallel:", tp_spin)

            gpu_mem = QDoubleSpinBox()
            gpu_mem.setRange(0.1, 1.0)
            gpu_mem.setSingleStep(0.05)
            gpu_mem.setValue(params.get("gpu_memory_utilization", 0.9))
            vllm_layout.addRow("GPU Memory:", gpu_mem)

            max_len_spin = QSpinBox()
            max_len_spin.setRange(0, 131072)
            max_len_spin.setValue(params.get("max_model_len", 0))
            vllm_layout.addRow("Max Model Len:", max_len_spin)

            trust_cb = QCheckBox("Trust Remote Code")
            trust_cb.setChecked(params.get("trust_remote_code", True))
            vllm_layout.addRow("", trust_cb)

            main_layout.addWidget(vllm_group)

        elif engine == "sglang":
            sglang_group = QGroupBox("sglang")
            sglang_layout = QFormLayout(sglang_group)

            tp_spin = QSpinBox()
            tp_spin.setRange(1, 8)
            tp_spin.setValue(params.get("tensor_parallel_size", 1))
            sglang_layout.addRow("Tensor Parallel:", tp_spin)

            mem_spin = QDoubleSpinBox()
            mem_spin.setRange(0.1, 1.0)
            mem_spin.setSingleStep(0.05)
            mem_spin.setValue(params.get("mem_fraction_static", 0.9))
            sglang_layout.addRow("Mem Fraction:", mem_spin)

            trust_cb = QCheckBox("Trust Remote Code")
            trust_cb.setChecked(params.get("trust_remote_code", True))
            sglang_layout.addRow("", trust_cb)

            main_layout.addWidget(sglang_group)

        # Przyciski
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Zapisz")
        button_box.button(QDialogButtonBox.StandardButton.Cancel).setText("Anuluj")
        main_layout.addWidget(button_box)

        def save_params():
            new_params = {
                "host": host_edit.text().strip(),
                "port": port_spin.value(),
            }
            if engine == "llama":
                new_params["threads"] = threads_spin.value()
                new_params["threads_batch"] = threads_batch_spin.value()
                new_params["batch_size"] = batch_spin.value()
                new_params["ubatch_size"] = ubatch_spin.value()
                new_params["ctx_size"] = ctx_spin.value()
                new_params["gpu_layers"] = gpu_spin.value()
                new_params["main_gpu"] = main_gpu_spin.value()
                new_params["tensor_split"] = tensor_split_edit.text().strip()
                new_params["split_mode"] = split_mode_combo.currentText()
                new_params["n_predict"] = n_pred_spin.value()
                new_params["flash_attn"] = flash_combo.currentText()
                new_params["cache_type_k"] = cache_k_combo.currentText()
                new_params["cache_type_v"] = cache_v_combo.currentText()
                new_params["kv_offload"] = kv_offload_cb.isChecked()
                new_params["mmap"] = mmap_cb.isChecked()
                new_params["mlock"] = mlock_cb.isChecked()
                new_params["no_direct_io"] = no_direct_io_cb.isChecked()
                new_params["numa"] = numa_cb.isChecked()
                new_params["rope_scaling"] = rope_scaling_combo.currentText()
                new_params["rope_freq_base"] = rope_freq_base_spin.value()
                new_params["rope_scale"] = rope_scale_spin.value()
                new_params["keep"] = keep_spin.value()
                new_params["lora"] = lora_edit.text().strip()
                new_params["control_vector"] = control_vector_edit.text().strip()
                new_params["log_file"] = log_file_edit.text().strip()
                new_params["log_disable"] = log_disable_cb.isChecked()
                new_params["no_escape"] = no_escape_cb.isChecked()
                new_params["verbose"] = verbose_cb.isChecked()
            elif engine == "vllm":
                new_params["tensor_parallel_size"] = tp_spin.value()
                new_params["gpu_memory_utilization"] = gpu_mem.value()
                new_params["max_model_len"] = max_len_spin.value()
                new_params["trust_remote_code"] = trust_cb.isChecked()
            elif engine == "sglang":
                new_params["tensor_parallel_size"] = tp_spin.value()
                new_params["mem_fraction_static"] = mem_spin.value()
                new_params["trust_remote_code"] = trust_cb.isChecked()

            cfg[engine]["launch_params"] = new_params
            config.save_config(cfg)
            dlg.accept()

        button_box.accepted.connect(save_params)
        button_box.rejected.connect(dlg.reject)

        dlg.exec()

    def _show_command_and_launch(self, engine: str, cmd: str, info: ModelInfo) -> None:
        """Pokazuje dialog z komendą i opcją edycji."""
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QTextEdit

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Uruchom {engine}")
        dlg.resize(700, 300)

        layout = QVBoxLayout(dlg)
        label = QLabel(f"Komenda (edytuj przed uruchomieniem):")
        layout.addWidget(label)

        cmd_edit = QTextEdit()
        cmd_edit.setPlainText(cmd)
        layout.addWidget(cmd_edit)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Uruchom")
        button_box.button(QDialogButtonBox.StandardButton.Cancel).setText("Anuluj")
        layout.addWidget(button_box)

        button_box.accepted.connect(dlg.accept)
        button_box.rejected.connect(dlg.reject)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            final_cmd = cmd_edit.toPlainText().strip()
            if final_cmd:
                self._start_process(engine, final_cmd, info)

    def _start_process(self, engine: str, cmd: str, info: ModelInfo) -> None:
        """Uruchamia proces."""
        process = QProcess(self)
        process.setProgram(cmd.split()[0])
        process.setArguments(cmd.split()[1:])

        process.started.connect(lambda: self._on_process_started(engine, info))
        process.readyReadStandardOutput.connect(
            lambda: self._on_process_output(engine, process, "stdout")
        )
        process.readyReadStandardError.connect(
            lambda: self._on_process_output(engine, process, "stderr")
        )
        process.finished.connect(
            lambda code, status: self._on_process_finished(engine, code, status, info)
        )

        setattr(self, f"{engine}_process", process)
        setattr(self, f"{engine}_running", True)

        self.append_log(f"[{engine.upper()}] Uruchamianie: {cmd}")
        process.start()
        self.status(f"Uruchamianie {engine}...")

    def _on_process_started(self, engine: str, info: ModelInfo) -> None:
        """Proces uruchomiony."""
        self.status(f"{engine.upper()} uruchomiony dla {info.name}")
        self.append_log(f"[{engine.upper()}] Proces uruchomiony")
        self.detail_widget.set_engine_running(engine, True)

    def _on_process_output(self, engine: str, process: QProcess, stream: str) -> None:
        """Otrzymujemy dane z procesu."""
        if stream == "stdout":
            data = process.readAllStandardOutput().data().decode("utf-8")
        else:
            data = process.readAllStandardError().data().decode("utf-8")

        if data.strip():
            self.append_log(f"[{engine.upper()}] {data.strip()}")

    def _on_process_finished(
        self, engine: str, exit_code: int, exit_status, info: ModelInfo
    ) -> None:
        """Proces zakończony."""
        setattr(self, f"{engine}_running", False)
        process = getattr(self, f"{engine}_process")
        if process:
            process.deleteLater()
            setattr(self, f"{engine}_process", None)

        self.status(f"{engine.upper()} zakończony (exit: {exit_code})")
        self.append_log(f"[{engine.upper()}] Zakończony (exit code: {exit_code})")
        self.detail_widget.set_engine_running(engine, False)

    def _stop_process(self, engine: str) -> None:
        """Zatrzymuje proces."""
        process = getattr(self, f"{engine}_process", None)
        if process and process.state() == QProcess.ProcessState.Running:
            self.append_log(f"[{engine.upper()}] Zatrzymywanie...")
            process.terminate()
            setattr(self, f"{engine}_running", False)
            self.detail_widget.set_engine_running(engine, False)

    def append_log(self, text: str) -> None:
        """Dodaj tekst do logów."""
        current = self.log_view.toPlainText()
        lines = current.split("\n") if current else []
        lines.append(text)
        # Keep last 1000 lines
        if len(lines) > 1000:
            lines = lines[-1000:]
        self.log_view.setPlainText("\n".join(lines))
        # Scroll to bottom
        self.log_view.verticalScrollBar().setValue(
            self.log_view.verticalScrollBar().maximum()
        )

    def status(self, msg: str, duration: int = 0, level: str = "info") -> None:
        """Ustaw status bar."""
        if hasattr(self, "_status_bar_initialized"):
            self.statusBar().set_status(msg, level)
        else:
            self.statusBar().showMessage(msg)

    def _save_config_from_ui(self) -> None:
        """Zapisz obecne wartości z UI do konfiguracji."""
        cfg = config.load_config()
        cfg["vllm"]["venv_path"] = self.vllm_path_edit.text().strip()
        cfg["sglang"]["venv_path"] = self.sglang_path_edit.text().strip()
        cfg["llama"]["binary_path"] = self.llama_path_edit.text().strip()
        cfg["models_dir"] = self.models_dir_edit.text().strip()
        config.save_config(cfg)

    def _load_config_to_ui(self) -> None:
        """Wczytaj konfigurację z YAML do UI."""
        cfg = config.load_config()

        self.vllm_path_edit.setText(cfg.get("vllm", {}).get("venv_path", ""))
        self.sglang_path_edit.setText(cfg.get("sglang", {}).get("venv_path", ""))
        self.llama_path_edit.setText(cfg.get("llama", {}).get("binary_path", ""))
        self.models_dir_edit.setText(cfg.get("models_dir", ""))

        self._update_all_statuses()

    def _update_all_statuses(self) -> None:
        """Aktualizuj wskaźniki statusu dla wszystkich ścieżek."""
        self._on_path_changed("vllm_path")
        self._on_path_changed("sglang_path")
        self._on_path_changed("llama_path")
        self._on_path_changed("models_dir")


class ModelScanThread(QThread):
    """Wątek do skanowania modeli w tle."""

    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, root_dir: str, max_depth: int = 3):
        super().__init__()
        self.root_dir = root_dir
        self.max_depth = max_depth

    def run(self):
        try:
            models = scan_models_directory(
                self.root_dir,
                max_depth=self.max_depth,
                recursive=True,
            )
            self.finished.emit(models)
        except Exception as e:
            self.error.emit(str(e))
