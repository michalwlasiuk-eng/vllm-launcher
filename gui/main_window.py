"""Główne okno aplikacji — vLLM / sglang Model Manager."""

import os
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QProcess, Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QProgressBar,
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
    vllm_stop_clicked = pyqtSignal()
    sglang_stop_clicked = pyqtSignal()

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

        self.vllm_start_btn.setEnabled(False)
        self.vllm_stop_btn.setEnabled(False)
        self.sglang_start_btn.setEnabled(False)
        self.sglang_stop_btn.setEnabled(False)

        self.vllm_start_btn.clicked.connect(self._on_vllm_start)
        self.vllm_stop_btn.clicked.connect(self._on_vllm_stop)
        self.sglang_start_btn.clicked.connect(self._on_sglang_start)
        self.sglang_stop_btn.clicked.connect(self._on_sglang_stop)

        self.button_layout.addWidget(self.vllm_start_btn)
        self.button_layout.addWidget(self.vllm_stop_btn)
        self.button_layout.addWidget(self.sglang_start_btn)
        self.button_layout.addWidget(self.sglang_stop_btn)
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

    def set_model_info(self, info: ModelInfo):
        """Wyświetl informacje o modelu."""
        self._current_model = info

        text = f"""
Nazwa: {info.name}
Ścieżka: {info.path}
Rozmiar: {info.size_human}
Format: {info.format.value}
Engine: {info.engine_support.value}

--- Architektura ---
"""
        if info.arch:
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

    def set_engine_running(self, engine: str, running: bool):
        """Aktualizuj stan przycisków start/stop."""
        if engine == "vllm":
            self.vllm_start_btn.setEnabled(not running)
            self.vllm_stop_btn.setEnabled(running)
        elif engine == "sglang":
            self.sglang_start_btn.setEnabled(not running)
            self.sglang_stop_btn.setEnabled(running)


class MainWindow(QMainWindow):
    """Okno główne aplikacji."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("vLLM / sglang Model Manager")
        self.resize(1200, 750)
        self.setMinimumSize(900, 600)

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

        # Przycisk do parametrów uruchomienia
        params_btn_layout = QHBoxLayout()
        params_btn_layout.addStretch()
        self.btn_vllm_params = self._add_btn("⚙️ Parametry vLLM")
        self.btn_vllm_params.clicked.connect(
            lambda: self._show_launch_params_dialog("vllm")
        )
        params_btn_layout.addWidget(self.btn_vllm_params)
        self.btn_sglang_params = self._add_btn("⚙️ Parametry sglang")
        self.btn_sglang_params.clicked.connect(
            lambda: self._show_launch_params_dialog("sglang")
        )
        params_btn_layout.addWidget(self.btn_sglang_params)
        self.btn_llama_params = self._add_btn("⚙️ Parametry llama")
        self.btn_llama_params.clicked.connect(
            lambda: self._show_launch_params_dialog("llama")
        )
        params_btn_layout.addWidget(self.btn_llama_params)
        config_layout.addLayout(params_btn_layout)

        left_layout.addWidget(config_group)

        # --- Przyciski akcji ---
        btn_layout = QHBoxLayout()
        self.btn_scan = self._add_btn("📂 Skanuj katalog")
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
        left_layout.addWidget(self.model_table)

        splitter.addWidget(left_widget)

        # ---- Prawy panel — zakładki ----
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(6, 6, 6, 6)

        self.tabs = QTabWidget()

        # Tab: Szczegóły / Uruchom
        details_tab = self._create_details_tab()
        self.tabs.addTab(details_tab, "Szczegóły")

        # Tab: Testy
        from gui.model_test_widget import ModelTestWidget

        self.test_widget = ModelTestWidget()
        self.tabs.addTab(self.test_widget, "Testy")

        # Tab: Logi
        logs_tab = self._create_logs_tab()
        self.tabs.addTab(logs_tab, "Logi")

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right_widget)

        # --- Połączenia ---
        self.model_table.cellClicked.connect(self._on_model_selected)
        self.filter_combo.currentTextChanged.connect(self._apply_filter)
        self.model_table.customContextMenuRequested.connect(self._on_context_menu)

        # --- Status bar ---
        self.status_bar = CustomStatusBar()
        self.setStatusBar(self.status_bar)

        # Proporcze splittera: 45% lewo / 55% prawo
        splitter.setSizes([450, 750])

        # --- Wczytaj konfigurację ---
        self._load_config_to_ui()
        self.status_bar.showMessage("Gotowy — wskaż katalog z modelami")

        # Zainicjalizuj zmienne do śledzenia procesów
        self.vllm_process = None
        self.sglang_process = None
        self.vllm_running = False
        self.sglang_running = False

        # --- Progress bar w status barze ---
        self.progress = QProgressBar()
        self.progress.setFixedWidth(160)
        self.progress.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress)

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

    # ================================================================
    #  Skanowanie modeli
    # ================================================================

    def _on_scan_clicked(self) -> None:
        """Przycisk: Skanuj katalog."""
        print("[DEBUG] _on_scan_clicked: otwieranie dialogu")
        dir_dialog = QFileDialog(self)
        dir_dialog.setFileMode(QFileDialog.FileMode.Directory)
        dir_dialog.setDirectory("/media/black/free/train")

        if dir_dialog.exec():
            dirs = dir_dialog.selectedFiles()
            print(f"[DEBUG] Wybrano: {dirs}")
            if dirs:
                self._start_scan(dirs[0])

    def _on_refresh_clicked(self) -> None:
        """Przycisk: Odśwież — ponowne skanowanie ostatniego katalogu."""
        cfg = config.load_config()
        models_dir = cfg.get("models_dir", "")
        if models_dir:
            self._start_scan(models_dir)
        else:
            self.status("Brak katalogu modeli — najpierw wskaż katalog", duration=3)

    def _start_scan(self, root_dir: str) -> None:
        """Uruchamia skanowanie w tle."""
        # Zapisz katalog modeli
        cfg = config.load_config()
        cfg["models_dir"] = root_dir
        config.save_config(cfg)
        self.models_dir_edit.setText(root_dir)

        # Sprawdź czy wątek już działa
        if hasattr(self, "scan_thread") and self.scan_thread.isRunning():
            self.scan_thread.terminate()
            self.scan_thread.wait()

        # Utwórz i uruchom wątek
        self.scan_thread = ModelScanThread(root_dir)
        self.scan_thread.progress.connect(self._on_scan_progress)
        self.scan_thread.finished.connect(self._on_scan_finished)
        self.scan_thread.error.connect(self._on_scan_error)

        self.status(f"Skanowanie: {root_dir}")
        self.set_progress(0)
        self.scan_thread.start()

    def _on_scan_progress(self, current: int, total: int) -> None:
        """Aktualizacja progressbaru."""
        if total > 0:
            self.set_progress(int(current / total * 100))

    def _on_scan_finished(self, models: list) -> None:
        """Skanowanie zakończone."""
        print(f"[DEBUG] _on_scan_finished: {len(models)} modeli")
        self.hide_progress()
        self.model_data = models  # zapisz do filtrowania
        self._populate_table(models)
        self.status(f"Znaleziono {len(models)} modeli")

    def _on_scan_error(self, error_msg: str) -> None:
        """Błąd skanowania."""
        self.hide_progress()
        self.status(f"Błąd skanowania: {error_msg}", duration=5)

    def _populate_table(self, models: list[ModelInfo]) -> None:
        """Wypełnia tabelę modelami."""
        print(f"[DEBUG] _populate_table: {len(models)} modeli")
        self.model_table.setRowCount(0)  # czyścimy najpierw
        self.model_table.setRowCount(len(models))

        for row, info in enumerate(models):
            self._set_model_row(row, info)

        # model_data jest już zapisany w _on_scan_finished
        self.model_data = models

    def _set_model_row(self, row: int, info: ModelInfo) -> None:
        """Ustawia jeden wiersz w tabeli."""
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

            # Kolory według formatu
            if col == 3:  # Format
                if info.format == ModelFormat.GGUF:
                    item.setForeground(Qt.GlobalColor.darkYellow)
                elif info.format == ModelFormat.SAFETENSORS:
                    item.setForeground(Qt.GlobalColor.blue)

            self.model_table.setItem(row, col, item)

        # Debug: verify first row
        if row == 0:
            print(f"[DEBUG] _set_model_row: row={row} items={len(items)}")
            for c in range(len(items)):
                it = self.model_table.item(row, c)
                print(f"  col{c}: {repr(it)}")

    def _apply_filter(self) -> None:
        """Filtruje modele według wybranego kryterium."""
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
        self.status(f"Pokazano {len(filtered)} z {len(self.model_data)} modeli")

    def _on_model_selected(self, row: int, col: int) -> None:
        """Model wybrany z tabeli — pokaż szczegóły."""
        if not hasattr(self, "model_data") or row >= len(self.model_data):
            return

        info = self.model_data[row]

        # Znajdź zakładkę szczegółów
        detail_tab = self.tabs.widget(0)  # pierwsza zakładka

        if isinstance(detail_tab, ModelDetailWidget):
            detail_tab.set_model_info(info)
        else:
            # Jeśli to jeszcze nie jest ModelDetailWidget, zamień
            self._replace_details_tab(info)

        # Przełącz na zakładkę szczegółów
        self.tabs.setCurrentIndex(0)

        self.status(f"Wybrano: {info.name} ({info.size_human}, {info.format.value})")

    def _replace_details_tab(self, info: ModelInfo) -> None:
        """Zamienia zakładkę szczegółów na ModelDetailWidget."""
        detail_widget = ModelDetailWidget()
        detail_widget.vllm_start_clicked.connect(self._launch_vllm)
        detail_widget.sglang_start_clicked.connect(self._launch_sglang)
        detail_widget.vllm_stop_clicked.connect(lambda: self._stop_process("vllm"))
        detail_widget.sglang_stop_clicked.connect(lambda: self._stop_process("sglang"))
        detail_widget.set_model_info(info)

        # Usuń starą zakładkę
        old_widget = self.tabs.widget(0)
        if old_widget:
            self.tabs.removeWidget(old_widget)
            old_widget.deleteLater()

        # Dodaj nową
        self.tabs.insertWidget(0, detail_widget)
        self.tabs.setCurrentWidget(detail_widget)
        self.detail_widget = detail_widget

    def _on_context_menu(self, pos) -> None:
        """Menu kontekstowe na tabeli."""
        row = self.model_table.rowAt(pos)
        if row < 0:
            return

        if not hasattr(self, "model_data") or row >= len(self.model_data):
            return

        info = self.model_data[row]

        menu = QMenu(self)

        act_details = menu.addAction("📋 Szczegóły")
        act_details.triggered.connect(lambda: self._replace_details_tab(info))

        menu.addSeparator()

        act_launch_vllm = menu.addAction("🚀 Uruchom vLLM")
        act_launch_vllm.triggered.connect(lambda: self._launch_vllm(info))

        act_launch_sglang = menu.addAction("🚀 Uruchom sglang")
        act_launch_sglang.triggered.connect(lambda: self._launch_sglang(info))

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
            self.status("Błąd: Nie znaleziono binarium vllm w venv", duration=5)
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

        if params.get("disable_custom_all_reduce", False):
            cmd_parts.append("--disable-custom-all-reduce")

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

        chat_tpl = params.get("chat_template", "")
        if chat_tpl:
            cmd_parts += ["--chat-template", chat_tpl]

        if params.get("disable_log_stats", False):
            cmd_parts.append("--disable-log-stats")

        if params.get("enable_log_requests", False):
            cmd_parts.append("--enable-log-requests")

        ssl_cert = params.get("ssl_certfile", "")
        ssl_key = params.get("ssl_keyfile", "")
        if ssl_cert:
            cmd_parts += ["--ssl-certfile", ssl_cert]
            if ssl_key:
                cmd_parts += ["--ssl-keyfile", ssl_key]

        root_path = params.get("root_path", "")
        if root_path:
            cmd_parts += ["--root-path", root_path]

        hf_token = params.get("huggingface_token", "")
        if hf_token:
            cmd_parts += ["--hf-token", hf_token]

        if params.get("enable_auto_tool_choice", False):
            cmd_parts.append("--enable-auto-tool-choice")

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
            self.status("Błąd: Nie znaleziono binarium sglang w venv", duration=5)
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

        if params.get("disable_cuda_graph", False):
            cmd_parts.append("--disable-cuda-graph")

        if params.get("disable_radix_cache", False):
            cmd_parts.append("--disable-radix-cache")

        max_req = params.get("max_running_requests", 512)
        if max_req != 512:
            cmd_parts += ["--max-running-requests", str(max_req)]

        load_format = params.get("load_format", "auto")
        if load_format != "auto":
            cmd_parts += ["--load-format", load_format]
        elif info.format == ModelFormat.GGUF:
            cmd_parts += ["--load-format", "gguf"]

        quant = params.get("quantization", "")
        if quant:
            cmd_parts += ["--quantization", quant]

        ctx_len = params.get("context_length", 0)
        if ctx_len > 0:
            cmd_parts += ["--context-length", str(ctx_len)]

        max_queued = params.get("max_queued_requests", 256)
        if max_queued != 256:
            cmd_parts += ["--max-queued-requests", str(max_queued)]

        chunked = params.get("chunked_prefill_size", 0)
        if chunked > 0:
            cmd_parts += ["--chunked-prefill-size", str(chunked)]

        if params.get("trust_remote_code", True):
            cmd_parts.append("--trust-remote-code")

        sched_pol = params.get("schedule_policy", "lpm")
        if sched_pol != "lpm":
            cmd_parts += ["--schedule-policy", sched_pol]

        if params.get("enable_priority_scheduling", False):
            cmd_parts.append("--enable-priority-scheduling")

        prefill_max = params.get("prefill_max_requests", 0)
        if prefill_max > 0:
            cmd_parts += ["--prefill-max-requests", str(prefill_max)]

        if params.get("enable_dynamic_chunking", False):
            cmd_parts.append("--enable-dynamic-chunking")

        api_key = params.get("api_key", "")
        if api_key:
            cmd_parts += ["--api-key", api_key]

        ssl_cert = params.get("ssl_certfile", "")
        ssl_key = params.get("ssl_keyfile", "")
        if ssl_cert:
            cmd_parts += ["--ssl-certfile", ssl_cert]
            if ssl_key:
                cmd_parts += ["--ssl-keyfile", ssl_key]

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

        # Use model_path for GGUF files, path for directories
        model_arg = info.model_path if info.format == ModelFormat.GGUF else info.path

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

        gpu_layers = params.get("gpu_layers", -1)
        if gpu_layers != -1:
            cmd_parts += ["-ngl", str(gpu_layers)]

        batch_size = params.get("batch_size", 2048)
        if batch_size != 2048:
            cmd_parts += ["-b", str(batch_size)]

        ubatch_size = params.get("ubatch_size", 512)
        if ubatch_size != 512:
            cmd_parts += ["-ub", str(ubatch_size)]

        n_predict = params.get("n_predict", -1)
        if n_predict != -1:
            cmd_parts += ["-n", str(n_predict)]

        flash_attn = params.get("flash_attn", "auto")
        if flash_attn != "auto":
            cmd_parts += ["-fa", flash_attn]

        if not params.get("kv_offload", True):
            cmd_parts.append("--no-kv-offload")

        cache_type_k = params.get("cache_type_k", "f16")
        if cache_type_k != "f16":
            cmd_parts += ["-ctk", cache_type_k]

        cache_type_v = params.get("cache_type_v", "f16")
        if cache_type_v != "f16":
            cmd_parts += ["-ctv", cache_type_v]

        rope_scaling = params.get("rope_scaling", "linear")
        if rope_scaling != "linear":
            cmd_parts += ["--rope-scaling", rope_scaling]

        rope_scale = params.get("rope_scale", 1.0)
        if rope_scale != 1.0:
            cmd_parts += ["--rope-scale", str(rope_scale)]

        main_gpu = params.get("main_gpu", 0)
        if main_gpu != 0:
            cmd_parts += ["-mg", str(main_gpu)]

        tensor_split = params.get("tensor_split", "")
        if tensor_split:
            cmd_parts += ["-ts", tensor_split]

        split_mode = params.get("split_mode", "layer")
        if split_mode != "layer":
            split_map = {"none": "none", "layer": "layer", "row": "row", "tensor": "tensor"}
            if split_mode in split_map:
                cmd_parts += ["-sm", split_map[split_mode]]

        if params.get("mlock", False):
            cmd_parts.append("--mlock")

        if not params.get("mmap", True):
            cmd_parts.append("--no-mmap")

        if params.get("no_direct_io", True):
            cmd_parts.append("--no-direct-io")

        numa = params.get("numa", "")
        if numa:
            cmd_parts += ["--numa", numa]

        lora = params.get("lora", "")
        if lora:
            cmd_parts += ["--lora", lora]

        control_vector = params.get("control_vector", "")
        if control_vector:
            cmd_parts += ["--control-vector", control_vector]

        keep = params.get("keep", 0)
        if keep != 0:
            cmd_parts += ["--keep", str(keep)]

        if params.get("verbose", False):
            cmd_parts.append("-v")

        if params.get("log_disable", False):
            cmd_parts.append("--log-disable")

        log_file = params.get("log_file", "")
        if log_file:
            cmd_parts += ["--log-file", log_file]

        if not params.get("escape", True):
            cmd_parts.append("--no-escape")

        cmd = " ".join(cmd_parts)

        self._show_command_and_launch("llama", cmd, info)

    def _show_command_and_launch(self, engine: str, cmd: str, info: ModelInfo) -> None:
        """Pokazuje dialog z komendą i opcją edycji przed uruchomieniem."""
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QTextEdit

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Uruchom {engine}")
        dlg.resize(700, 300)

        layout = QVBoxLayout(dlg)

        label = QLabel(f"Komenda do uruchomienia (możesz edytować):")
        layout.addWidget(label)

        cmd_edit = QTextEdit()
        cmd_edit.setPlainText(cmd)
        cmd_edit.setFont(cmd_edit.font())
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
        """Uruchamia proces w tle i śledzi jego wyjście."""
        # Sprawdź czy nie działa już proces
        if hasattr(self, f"{engine}_process") and getattr(self, f"{engine}_process"):
            self.status(f"Błąd: {engine} już działa")
            return

        # Utwórz QProcess
        process = QProcess(self)
        process.setProgram(cmd.split()[0])
        process.setArguments(cmd.split()[1:])
        process.setWorkingDirectory("/")

        # Połączenia
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
        process.errorOccurred.connect(
            lambda error: self._on_process_error(engine, error, process)
        )

        # Zapisz referencję
        setattr(self, f"{engine}_process", process)

        # Uruchom
        self.append_log(f"[{engine.upper()}] Uruchamianie: {cmd}")
        process.start()

        # Ustaw flagę statusu
        setattr(self, f"{engine}_running", True)
        self.status(f"Uruchamianie {engine} dla {info.name}...")

    def _on_process_started(self, engine: str, info: ModelInfo) -> None:
        """Proces się uruchomił."""
        self.status(f"{engine.upper()} uruchomiony dla {info.name}")
        self.append_log(f"[{engine.upper()}] Proces uruchomiony")
        if hasattr(self, "detail_widget"):
            self.detail_widget.set_engine_running(engine, True)

    def _on_process_output(self, engine: str, process: QProcess, stream: str) -> None:
        """Otrzymujemy dane z procesu."""
        if stream == "stdout":
            data = process.readAllStandardOutput().data().decode("utf-8")
        else:
            data = process.readAllStandardError().data().decode("utf-8")

        # Wysyłaj do logów
        if data.strip():
            self.append_log(f"[{engine.upper()}] {data.strip()}")

    def _on_process_finished(
        self,
        engine: str,
        exit_code: int,
        exit_status: QProcess.ExitStatus,
        info: ModelInfo,
    ) -> None:
        """Proces się zakończył."""
        setattr(self, f"{engine}_running", False)
        process = getattr(self, f"{engine}_process", None)
        if process:
            process.deleteLater()
            setattr(self, f"{engine}_process", None)

        self.status(
            f"{engine.upper()} zakończony dla {info.name} (exit code: {exit_code})"
        )
        self.append_log(f"[{engine.upper()}] Zakończony (exit code: {exit_code})")
        if hasattr(self, "detail_widget"):
            self.detail_widget.set_engine_running(engine, False)

    def _stop_process(self, engine: str) -> None:
        """Zatrzymuje proces."""
        process = getattr(self, f"{engine}_process", None)
        if process and process.state() == QProcess.ProcessState.Running:
            self.append_log(f"[{engine.upper()}] Zatrzymywanie...")
            process.terminate()
            setattr(self, f"{engine}_running", False)
            if hasattr(self, "detail_widget"):
                self.detail_widget.set_engine_running(engine, False)

    def _on_process_error(
        self, engine: str, error: QProcess.ProcessError, process: QProcess
    ) -> None:
        """Błąd procesu."""
        setattr(self, f"{engine}_running", False)
        process = getattr(self, f"{engine}_process", None)
        if process:
            process.deleteLater()
            setattr(self, f"{engine}_process", None)

        error_msg = {
            QProcess.ProcessError.FailedToStart: "Nie udało się uruchomić",
            QProcess.ProcessError.Crashed: "Proces się zawiesił",
            QProcess.ProcessError.Timedout: "Przekroczony czas",
            QProcess.ProcessError.WriteError: "Błąd zapisu",
            QProcess.ProcessError.ReadError: "Błąd odczytu",
            QProcess.ProcessError.UnknownError: "Nieznany błąd",
        }.get(error, str(error))

        self.status(f"Błąd {engine.upper()}: {error_msg}")
        self.append_log(f"[{engine.upper()}] Błąd: {error_msg}")
        if hasattr(self, "detail_widget"):
            self.detail_widget.set_engine_running(engine, False)

    # ================================================================
    #  Pomocnicze
    # ================================================================

    def _make_path_row(self, label_text: str, config_key: str) -> QWidget:
        """Pasek: etykieta + QLineEdit + browse button + status."""
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 2)
        layout.setSpacing(6)

        label = QLabel(label_text)
        label.setFixedWidth(140)
        layout.addWidget(label)

        path_edit = QLineEdit()
        path_edit.setPlaceholderText("Wskaż katalog venv...")
        path_edit.setObjectName(config_key)
        setattr(self, f"{config_key}_edit", path_edit)
        layout.addWidget(path_edit, 1)

        browse_btn = self._add_btn("...")
        browse_btn.clicked.connect(lambda _, key=config_key: self._browse_path(key))
        layout.addWidget(browse_btn)

        # Status indicator
        status_label = QLabel("")
        status_label.setObjectName(f"{config_key}_status")
        status_label.setFixedWidth(20)
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
                self._on_path_changed(config_key)

    def _add_btn(self, text: str):
        return QPushButton(text)

    def _on_path_changed(self, config_key: str) -> None:
        """Wywołany gdy użytkownik zmieni ścieżkę."""
        edit = getattr(self, f"{config_key}_edit")
        path = edit.text().strip()
        status = getattr(self, f"{config_key}_status")

        if not path:
            status.setText("")
            status.setStyleSheet("")
            return

        # Sprawdź czy to istnieje
        if not os.path.isdir(path):
            status.setText("❌")
            status.setStyleSheet("color: #f44336; font-size: 14px;")
            return

        # models_dir — osobna logika (to nie jest venv)
        if config_key == "models_dir":
            model_count = model_scanner.get_model_count(path)
            if model_count > 0:
                status.setText(f"📦{model_count}")
                status.setStyleSheet("color: #4caf50; font-size: 12px;")
            else:
                status.setText("❌")
                status.setStyleSheet("color: #f44336; font-size: 14px;")
            self._save_config_from_ui()
            return

        # Sprawdź czy to venv (czy ma bin/python)
        has_python, py_path = config.detect_python(path)
        if not has_python:
            status.setText("❌")
            status.setStyleSheet("color: #f44336; font-size: 14px;")
            return

        # Sprawdź odpowiednie binarium
        if config_key == "vllm_path":
            has_vllm, vllm_bin = config.detect_binary(path, "vllm")
            if has_vllm and has_python:
                status.setText("✅")
                status.setStyleSheet("color: #4caf50; font-size: 14px;")
                self.status(f"vLLM wykryty: {vllm_bin}")
            else:
                status.setText("⚠️")
                status.setStyleSheet("color: #ff9800; font-size: 14px;")
                self.status("vLLM nie wykryte — venv istnieje ale brak vllm")
        elif config_key == "sglang_path":
            has_sglang, sglang_bin = config.detect_binary(path, "sglang")
            if has_sglang and has_python:
                status.setText("✅")
                status.setStyleSheet("color: #4caf50; font-size: 14px;")
                self.status(f"sglang wykryty: {sglang_bin}")
            else:
                status.setText("⚠️")
                status.setStyleSheet("color: #ff9800; font-size: 14px;")
                self.status("sglang nie wykryty — venv istnieje ale brak sglang")

        # Zapisz konfigurację
        self._save_config_from_ui()

    def _auto_detect_venvs(self) -> None:
        """Auto-wykryj venv z vLLM i sglang."""
        self.status("Szukam venv...")
        found = config.scan_common_venv_locations()

        if not found:
            self.status("Nie znaleziono venv z vLLM/sglang", duration=5)
            return

        vllm_found = [f for f in found if f["has_vllm"]]
        sglang_found = [f for f in found if f["has_sglang"]]

        # Ustaw pierwsze znalezione
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
        self.status(
            f"Znaleziono {len(found)} venv (vLLM: {len(vllm_found)}, sglang: {len(sglang_found)})"
        )

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

        # Aktualizuj statusy
        self._update_all_statuses()

    def _update_all_statuses(self) -> None:
        """Aktualizuj wskaźniki statusu dla wszystkich ścieżek."""
        self._on_path_changed("vllm_path")
        self._on_path_changed("sglang_path")
        self._on_path_changed("llama_path")
        self._on_path_changed("models_dir")


    def _create_menu(self):
        menu = self.menuBar()

        # Plik
        file_menu = menu.addMenu("&Plik")

        act_open = QAction("📂 Otwórz katalog", self)
        act_open.setShortcut("Ctrl+O")
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

    def _create_details_tab(self) -> ModelDetailWidget:
        """Zakładka: szczegóły modelu + parametry uruchomienia."""
        self.detail_widget = ModelDetailWidget()
        self.detail_widget.vllm_start_clicked.connect(self._launch_vllm)
        self.detail_widget.sglang_start_clicked.connect(self._launch_sglang)
        self.detail_widget.vllm_stop_clicked.connect(lambda: self._stop_process("vllm"))
        self.detail_widget.sglang_stop_clicked.connect(
            lambda: self._stop_process("sglang")
        )
        return self.detail_widget

    def _create_logs_tab(self) -> QWidget:
        """Zakładka: logi z procesu."""
        w = QWidget()
        layout = QVBoxLayout(w)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Logi z procesu pojawią się tutaj…")
        layout.addWidget(self.log_view)

        btn_clear = self._add_btn("Wyczyść logi")
        btn_clear.clicked.connect(self.log_view.clear)
        layout.addWidget(btn_clear)

        return w

    def _show_launch_params_dialog(self, engine: str) -> None:
        """Pokazuje dialog z parametrami uruchomienia dla engine."""
        from PyQt6.QtWidgets import (
            QCheckBox,
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QFormLayout,
            QGroupBox,
            QSpinBox,
        )

        cfg = config.load_config()
        params = cfg.get(engine, {}).get("launch_params", {})

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Parametry uruchomienia — {engine}")
        dlg.resize(500, 650)

        layout = QVBoxLayout(dlg)

        # Sekcja: Sieć
        net_group = QGroupBox("Sieć")
        net_layout = QFormLayout(net_group)

        host_edit = QLineEdit(params.get("host", "127.0.0.1"))
        net_layout.addRow("Host:", host_edit)

        port_spin = QSpinBox()
        port_spin.setRange(1000, 65535)
        port_spin.setValue(params.get("port", 8000))
        net_layout.addRow("Port:", port_spin)

        api_key_edit = QLineEdit(params.get("api_key", ""))
        api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        net_layout.addRow("API Key:", api_key_edit)

        layout.addWidget(net_group)

        # Sekcja: GPU
        gpu_group = QGroupBox("GPU")
        gpu_layout = QFormLayout(gpu_group)

        tp_spin = QSpinBox()
        tp_spin.setRange(1, 8)
        tp_spin.setValue(params.get("tensor_parallel_size", 1))
        gpu_layout.addRow("Tensor Parallel Size:", tp_spin)

        if engine == "vllm":
            gpu_mem_spin = QDoubleSpinBox()
            gpu_mem_spin.setRange(0.1, 1.0)
            gpu_mem_spin.setSingleStep(0.05)
            gpu_mem_spin.setValue(params.get("gpu_memory_utilization", 0.9))
            gpu_layout.addRow("GPU Memory Utilization:", gpu_mem_spin)

            swap_spin = QSpinBox()
            swap_spin.setRange(0, 64)
            swap_spin.setValue(params.get("swap_space", 4))
            gpu_layout.addRow("Swap Space (GB):", swap_spin)
        else:
            mem_frac_spin = QDoubleSpinBox()
            mem_frac_spin.setRange(0.1, 1.0)
            mem_frac_spin.setSingleStep(0.05)
            mem_frac_spin.setValue(params.get("mem_fraction_static", 0.9))
            gpu_layout.addRow("Mem Fraction Static:", mem_frac_spin)

            max_queued_spin = QSpinBox()
            max_queued_spin.setRange(1, 1024)
            max_queued_spin.setValue(params.get("max_queued_requests", 256))
            gpu_layout.addRow("Max Queued Requests:", max_queued_spin)

        layout.addWidget(gpu_group)

        # Sekcja: Model
        model_group = QGroupBox("Model")
        model_layout = QFormLayout(model_group)

        dtype_combo = QComboBox()
        dtype_combo.addItems(["auto", "float16", "bfloat16", "float32"])
        current_dtype = params.get("dtype", "auto")
        idx = dtype_combo.findText(current_dtype)
        if idx >= 0:
            dtype_combo.setCurrentIndex(idx)
        model_layout.addRow("DType:", dtype_combo)

        trust_remote_cb = QCheckBox("Trust Remote Code")
        trust_remote_cb.setChecked(params.get("trust_remote_code", True))
        model_layout.addRow("", trust_remote_cb)

        if engine == "vllm":
            max_len_spin = QSpinBox()
            max_len_spin.setRange(0, 131072)
            max_len_spin.setSingleStep(1024)
            max_len_spin.setValue(params.get("max_model_len", 0))
            max_len_spin.setSpecialValueText("Auto")
            model_layout.addRow("Max Model Len:", max_len_spin)

            max_seqs_spin = QSpinBox()
            max_seqs_spin.setRange(1, 1024)
            max_seqs_spin.setValue(params.get("max_num_seqs", 256))
            model_layout.addRow("Max Num Seqs:", max_seqs_spin)

            tool_parser_combo = QComboBox()
            tool_parser_combo.addItems(
                [
                    "",
                    "deepseek_v3",
                    "deepseek_v31",
                    "deepseek_v32",
                    "qwen3_coder",
                    "qwen3_xml",
                    "llama3_json",
                    "gemma4",
                ]
            )
            current_tool = params.get("tool_call_parser", "")
            idx = tool_parser_combo.findText(current_tool)
            if idx >= 0:
                tool_parser_combo.setCurrentIndex(idx)
            model_layout.addRow("Tool Parser:", tool_parser_combo)
        else:
            load_format_combo = QComboBox()
            load_format_combo.addItems(
                [
                    "auto",
                    "safetensors",
                    "pt",
                    "gguf",
                    "bitsandbytes",
                    "awq",
                    "gptq",
                    "marlin",
                ]
            )
            current_lf = params.get("load_format", "auto")
            idx = load_format_combo.findText(current_lf)
            if idx >= 0:
                load_format_combo.setCurrentIndex(idx)
            model_layout.addRow("Load Format:", load_format_combo)

            quant_combo = QComboBox()
            quant_combo.addItems(
                [
                    "",
                    "awq",
                    "fp8",
                    "gptq",
                    "marlin",
                    "gguf",
                    "bitsandbytes",
                    "w4a8_int8",
                    "w8a8_fp8",
                ]
            )
            current_q = params.get("quantization", "")
            idx = quant_combo.findText(current_q)
            if idx >= 0:
                quant_combo.setCurrentIndex(idx)
            model_layout.addRow("Quantization:", quant_combo)

            max_tokens_spin = QSpinBox()
            max_tokens_spin.setRange(0, 262144)
            max_tokens_spin.setSingleStep(1024)
            max_tokens_spin.setValue(params.get("max_total_tokens", 0))
            max_tokens_spin.setSpecialValueText("Auto")
            model_layout.addRow("Max Total Tokens:", max_tokens_spin)

            max_req_spin = QSpinBox()
            max_req_spin.setRange(1, 2048)
            max_req_spin.setValue(params.get("max_running_requests", 512))
            model_layout.addRow("Max Running Requests:", max_req_spin)

            ctx_len_spin = QSpinBox()
            ctx_len_spin.setRange(0, 262144)
            ctx_len_spin.setSingleStep(1024)
            ctx_len_spin.setValue(params.get("context_length", 0))
            ctx_len_spin.setSpecialValueText("Auto")
            model_layout.addRow("Context Length:", ctx_len_spin)

            sched_combo = QComboBox()
            sched_combo.addItems(["lpm", "fcfs", "random", "priority", "lof"])
            current_sched = params.get("schedule_policy", "lpm")
            idx = sched_combo.findText(current_sched)
            if idx >= 0:
                sched_combo.setCurrentIndex(idx)
            model_layout.addRow("Schedule Policy:", sched_combo)

        layout.addWidget(model_group)

        # Sekcja: Opcje
        opts_group = QGroupBox("Opcje")
        opts_layout = QVBoxLayout(opts_group)

        if engine == "vllm":
            enforce_eager_cb = QCheckBox("Enforce Eager")
            enforce_eager_cb.setChecked(params.get("enforce_eager", False))
            opts_layout.addWidget(enforce_eager_cb)

            disable_all_reduce_cb = QCheckBox("Disable Custom All Reduce")
            disable_all_reduce_cb.setChecked(
                params.get("disable_custom_all_reduce", False)
            )
            opts_layout.addWidget(disable_all_reduce_cb)

            disable_log_stats_cb = QCheckBox("Disable Log Stats")
            disable_log_stats_cb.setChecked(params.get("disable_log_stats", False))
            opts_layout.addWidget(disable_log_stats_cb)

            enable_log_requests_cb = QCheckBox("Enable Log Requests")
            enable_log_requests_cb.setChecked(params.get("enable_log_requests", False))
            opts_layout.addWidget(enable_log_requests_cb)

            auto_tool_cb = QCheckBox("Enable Auto Tool Choice")
            auto_tool_cb.setChecked(params.get("enable_auto_tool_choice", False))
            opts_layout.addWidget(auto_tool_cb)
        else:
            disable_cuda_graph_cb = QCheckBox("Disable CUDA Graph")
            disable_cuda_graph_cb.setChecked(params.get("disable_cuda_graph", False))
            opts_layout.addWidget(disable_cuda_graph_cb)

            disable_radix_cb = QCheckBox("Disable Radix Cache")
            disable_radix_cb.setChecked(params.get("disable_radix_cache", False))
            opts_layout.addWidget(disable_radix_cb)

            priority_sched_cb = QCheckBox("Enable Priority Scheduling")
            priority_sched_cb.setChecked(
                params.get("enable_priority_scheduling", False)
            )
            opts_layout.addWidget(priority_sched_cb)

            dynamic_chunking_cb = QCheckBox("Enable Dynamic Chunking")
            dynamic_chunking_cb.setChecked(params.get("enable_dynamic_chunking", False))
            opts_layout.addWidget(dynamic_chunking_cb)

        layout.addWidget(opts_group)

        # Przyciski
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults
        )
        button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Zapisz")
        button_box.button(QDialogButtonBox.StandardButton.Cancel).setText("Anuluj")
        button_box.button(QDialogButtonBox.StandardButton.RestoreDefaults).setText(
            "Domyślne"
        )
        layout.addWidget(button_box)

        def save_params():
                        elif engine == "sglang":
                            new_params["schedule_policy"] = sched_combo.currentText()

                        elif engine == "llama":
                            new_params["threads"] = threads_spin.value()
                            new_params["threads_batch"] = threads_batch_spin.value()
                            new_params["ctx_size"] = ctx_spin.value()
                            new_params["gpu_layers"] = gpu_layers_spin.value()
                            new_params["n_predict"] = n_pred_spin.value()
                            new_params["flash_attn"] = flash_attn_combo.currentText()
                            new_params["cache_type_k"] = cache_k_combo.currentText()
                            new_params["cache_type_v"] = cache_v_combo.currentText()
                            new_params["split_mode"] = split_mode_combo.currentText()
                            new_params["main_gpu"] = main_gpu_spin.value()
                            new_params["tensor_split"] = tensor_split_edit.text().strip()
                            new_params["rope_scaling"] = rope_scaling_combo.currentText()
                            new_params["rope_scale"] = rope_scale_spin.value()
                            new_params["keep"] = keep_spin.value()
                            new_params["lora"] = lora_edit.text().strip()
                            new_params["control_vector"] = ctrl_vec_edit.text().strip()
                            new_params["numa"] = numa_combo.currentText()
                            new_params["mlock"] = mlock_cb.isChecked()
                            new_params["mmap"] = mmap_cb.isChecked()
                            new_params["kv_offload"] = kv_offload_cb.isChecked()
                            new_params["no_direct_io"] = no_dio_cb.isChecked()
                            new_params["escape"] = escape_cb.isChecked()
                            new_params["verbose"] = verbose_cb.isChecked()
                            new_params["log_disable"] = log_disable_cb.isChecked()

                        cfg[engine]["launch_params"] = new_params
                        config.save_config(cfg)
                        dlg.accept()

                    def restore_defaults():
                        defaults = config.DEFAULT_CONFIG.get(engine, {}).get("launch_params", {})
                        host_edit.setText(defaults.get("host", "127.0.0.1"))
                        port_spin.setValue(defaults.get("port", 8000))
                        dtype_combo.setCurrentText(defaults.get("dtype", "auto"))
                        api_key_edit.setText("")
                        trust_remote_cb.setChecked(True)

                        if engine == "vllm":
                new_params["gpu_memory_utilization"] = gpu_mem_spin.value()
                new_params["max_model_len"] = max_len_spin.value()
                new_params["max_num_seqs"] = max_seqs_spin.value()
                new_params["swap_space"] = swap_spin.value()
                new_params["enforce_eager"] = enforce_eager_cb.isChecked()
                new_params["disable_custom_all_reduce"] = (
                    disable_all_reduce_cb.isChecked()
                )
                new_params["disable_log_stats"] = disable_log_stats_cb.isChecked()
                new_params["enable_log_requests"] = enable_log_requests_cb.isChecked()
                new_params["enable_auto_tool_choice"] = auto_tool_cb.isChecked()
                new_params["tool_call_parser"] = tool_parser_combo.currentText()
            else:
                new_params["mem_fraction_static"] = mem_frac_spin.value()
                new_params["max_queued_requests"] = max_queued_spin.value()
                new_params["max_total_tokens"] = max_tokens_spin.value()
                new_params["max_running_requests"] = max_req_spin.value()
                new_params["context_length"] = ctx_len_spin.value()
                new_params["schedule_policy"] = sched_combo.currentText()
                new_params["load_format"] = load_format_combo.currentText()
                new_params["quantization"] = quant_combo.currentText()
                new_params["disable_cuda_graph"] = disable_cuda_graph_cb.isChecked()
                new_params["disable_radix_cache"] = disable_radix_cb.isChecked()
                new_params["enable_priority_scheduling"] = priority_sched_cb.isChecked()
                new_params["enable_dynamic_chunking"] = dynamic_chunking_cb.isChecked()

            cfg[engine]["launch_params"] = new_params
            config.save_config(cfg)
            dlg.accept()

        def restore_defaults():
            defaults = config.DEFAULT_CONFIG.get(engine, {}).get("launch_params", {})
            host_edit.setText(defaults.get("host", "127.0.0.1"))
            port_spin.setValue(defaults.get("port", 8000))
            tp_spin.setValue(defaults.get("tensor_parallel_size", 1))
            dtype_combo.setCurrentText(defaults.get("dtype", "auto"))
            api_key_edit.setText("")
            trust_remote_cb.setChecked(True)

            if engine == "vllm":
                gpu_mem_spin.setValue(defaults.get("gpu_memory_utilization", 0.9))
                max_len_spin.setValue(defaults.get("max_model_len", 0))
                max_seqs_spin.setValue(defaults.get("max_num_seqs", 256))
                swap_spin.setValue(defaults.get("swap_space", 4))
                enforce_eager_cb.setChecked(False)
                disable_all_reduce_cb.setChecked(False)
                disable_log_stats_cb.setChecked(False)
                enable_log_requests_cb.setChecked(False)
                auto_tool_cb.setChecked(False)
                tool_parser_combo.setCurrentIndex(0)
            else:
                mem_frac_spin.setValue(defaults.get("mem_fraction_static", 0.9))
                max_queued_spin.setValue(defaults.get("max_queued_requests", 256))
                max_tokens_spin.setValue(defaults.get("max_total_tokens", 0))
                max_req_spin.setValue(defaults.get("max_running_requests", 512))
                ctx_len_spin.setValue(defaults.get("context_length", 0))
                sched_combo.setCurrentText("lpm")
                load_format_combo.setCurrentText("auto")
                quant_combo.setCurrentIndex(0)
                disable_cuda_graph_cb.setChecked(False)
                disable_radix_cb.setChecked(False)
                priority_sched_cb.setChecked(False)
                dynamic_chunking_cb.setChecked(False)

        button_box.accepted.connect(save_params)
        button_box.rejected.connect(dlg.reject)
        button_box.button(
            QDialogButtonBox.StandardButton.RestoreDefaults
        ).clicked.connect(restore_defaults)

        dlg.exec()

    def _show_about(self):
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.about(
            self,
            "O aplikacji",
            "vLLM / sglang Model Manager\n\n"
            "Narzędzie do zarządzania modelami AI z\n"
            "obsługą serwerów vLLM i sglang.\n\n"
            "PyQt6 • Python 3",
        )

    # ================================================================
    #  Public API (następne kroki będą to wykorzystywać)
    # ================================================================

    def set_model_count(self, count: int):
        self.model_table.setRowCount(count)

    def set_model_row(self, row: int, data: dict):
        """data = {'name': ..., 'path': ..., 'size': ..., 'type': ..., 'status': ...}"""
        for col, key in enumerate(["name", "path", "size", "type", "status"]):
            item = QTableWidgetItem(str(data.get(key, "")))
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
            )
            self.model_table.setItem(row, col, item)

    def append_log(self, text: str):
        self.log_view.appendPlainText(text)

    def set_progress(self, value: int):
        self.progress.setValue(value)
        self.progress.setVisible(True)

    def hide_progress(self):
        self.progress.setVisible(False)

    def status(self, msg: str, duration: int = 0):
        """duration=0 = permanentny."""
        if duration:
            self.status_bar.showMessage(msg, duration * 1000)
        else:
            self.status_bar.showMessage(msg)


class ModelScanThread(QThread):
    """Wątek do skanowania modeli w tle."""

    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(list)  # list[ModelInfo]
    error = pyqtSignal(str)

    def __init__(self, root_dir: str, max_depth: int = 3):
        super().__init__()
        self.root_dir = root_dir
        self.max_depth = max_depth
        self._cancelled = False

    def run(self):
        try:
            print(
                f"[DEBUG] ModelScanThread.run: scanning {self.root_dir}, max_depth={self.max_depth}"
            )
            models = scan_models_directory(
                self.root_dir,
                max_depth=self.max_depth,
                recursive=True,
            )
            print(f"[DEBUG] ModelScanThread.run: got {len(models)} models")
            if models:
                print(
                    f"[DEBUG] First model: {models[0].name}, format={models[0].format}"
                )
            self.finished.emit(models)
        except Exception as e:
            print(f"[DEBUG] ModelScanThread.run error: {e}")
            import traceback

            traceback.print_exc()
            self.error.emit(str(e))

    def cancel(self):
        self._cancelled = True
