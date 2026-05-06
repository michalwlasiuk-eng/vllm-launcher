"""Widget do testowania modeli przez API."""

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from backend.model_tester import TestResult, check_server_status, test_model


class TestThread(QThread):
    """Wątek do testowania modelu w tle."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(object)  # TestResult
    error = pyqtSignal(str)

    def __init__(
        self,
        endpoint: str,
        prompt: str,
        model_name: str,
        max_tokens: int,
        temperature: float,
        timeout: int,
    ):
        super().__init__()
        self.endpoint = endpoint
        self.prompt = prompt
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    def run(self):
        try:
            self.progress.emit("Wysyłanie zapytania...")
            result = test_model(
                self.endpoint,
                self.prompt,
                self.model_name,
                self.max_tokens,
                self.temperature,
                self.timeout,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ModelTestWidget(QWidget):
    """Widget do testowania modeli przez API."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.test_thread = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # --- Konfiguracja API ---
        api_group = QGroupBox("Konfiguracja API")
        api_layout = QFormLayout(api_group)

        self.endpoint_edit = QPlainTextEdit()
        self.endpoint_edit.setPlaceholderText("http://127.0.0.1:8000")
        self.endpoint_edit.setMaximumHeight(40)
        self.endpoint_edit.setPlaceholderText("URL serwera (np. http://127.0.0.1:8000)")
        api_layout.addRow("Endpoint:", self.endpoint_edit)

        self.model_name_edit = QPlainTextEdit()
        self.model_name_edit.setPlaceholderText("default")
        self.model_name_edit.setMaximumHeight(40)
        api_layout.addRow("Model:", self.model_name_edit)

        # Przycisk sprawdzenia statusu
        self.check_status_btn = QPushButton("🔍 Sprawdź status")
        self.check_status_btn.clicked.connect(self._check_status)
        api_layout.addRow("", self.check_status_btn)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #ff9800;")
        api_layout.addRow("", self.status_label)

        layout.addWidget(api_group)

        # --- Parametry zapytania ---
        params_group = QGroupBox("Parametry")
        params_layout = QFormLayout(params_group)

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(1, 4096)
        self.max_tokens_spin.setValue(512)
        params_layout.addRow("Max tokeny:", self.max_tokens_spin)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(0.7)
        params_layout.addRow("Temperatura:", self.temp_spin)

        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(5, 300)
        self.timeout_spin.setValue(60)
        params_layout.addRow("Timeout (s):", self.timeout_spin)

        layout.addWidget(params_group)

        # --- Prompt ---
        prompt_group = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout(prompt_group)

        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText(
            "Wpisz prompt do wysłania na serwer...\n\n"
            "Przykład: What is the capital of France?"
        )
        self.prompt_edit.setMinimumHeight(80)
        prompt_layout.addWidget(self.prompt_edit)

        layout.addWidget(prompt_group)

        # --- Przycisk testu ---
        btn_layout = QHBoxLayout()
        self.test_btn = QPushButton("🚀 Wyślij zapytanie")
        self.test_btn.clicked.connect(self._run_test)
        btn_layout.addWidget(self.test_btn)

        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_test)
        btn_layout.addWidget(self.stop_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # --- Wynik ---
        result_group = QGroupBox("Wynik")
        result_layout = QVBoxLayout(result_group)

        self.result_text = QPlainTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText(
            "Wynik pojawi się tutaj...\n\n"
            "Upewnij się, że serwer jest uruchomiony i odpowiada na zapytania."
        )
        result_layout.addWidget(self.result_text)

        layout.addWidget(result_group)

    def _check_status(self):
        """Sprawdź status serwera."""
        endpoint = self.endpoint_edit.toPlainText().strip()
        if not endpoint:
            self.status_label.setText("⚠️ Wpisz endpoint")
            return

        self.status_label.setText("⏳ Sprawdzanie...")
        self.check_status_btn.setEnabled(False)

        if check_server_status(endpoint):
            self.status_label.setText("✅ Serwer online")
            self.status_label.setStyleSheet("color: #4caf50;")
        else:
            self.status_label.setText("❌ Serwer offline")
            self.status_label.setStyleSheet("color: #f44336;")

        self.check_status_btn.setEnabled(True)

    def _run_test(self):
        """Uruchom test modelu."""
        endpoint = self.endpoint_edit.toPlainText().strip()
        model_name = self.model_name_edit.toPlainText().strip()
        prompt = self.prompt_edit.toPlainText().strip()

        if not endpoint:
            self.result_text.setPlainText("⚠️ Wpisz endpoint")
            return

        if not model_name:
            model_name = "default"

        if not prompt:
            self.result_text.setPlainText("⚠️ Wpisz prompt")
            return

        # Zablokuj przyciski
        self.test_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        # Wyczyść wynik
        self.result_text.setPlainText("")

        # Utwórz i uruchom wątek
        self.test_thread = TestThread(
            endpoint=endpoint,
            prompt=prompt,
            model_name=model_name,
            max_tokens=self.max_tokens_spin.value(),
            temperature=self.temp_spin.value(),
            timeout=self.timeout_spin.value(),
        )
        self.test_thread.progress.connect(self._on_test_progress)
        self.test_thread.finished.connect(self._on_test_finished)
        self.test_thread.error.connect(self._on_test_error)
        self.test_thread.start()

    def _stop_test(self):
        """Zatrzymaj test."""
        if self.test_thread and self.test_thread.isRunning():
            self.test_thread.terminate()
            self.test_thread.wait()
            self.result_text.setPlainText("⏹️ Test zatrzymany")

        self.test_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_test_progress(self, msg: str):
        """Aktualizuj postęp."""
        self.result_text.setPlainText(msg)

    def _on_test_finished(self, result: TestResult):
        """Test zakończony."""
        self.test_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        if result.error:
            self.result_text.setPlainText(f"❌ Błąd:\n\n{result.error}")
            self.result_text.setStyleSheet("color: #f44336;")
        else:
            output = f"""
✅ Odpowiedź otrzymana:

{result.response}

---
Czas: {result.time_taken:.2f}s
Tokeny: {result.tokens_used if result.tokens_used else 'N/A'}
Model: {result.model_name}
Endpoint: {result.endpoint}
""".strip()
            self.result_text.setPlainText(output)
            self.result_text.setStyleSheet("color: #d4d4d4;")

    def _on_test_error(self, error: str):
        """Błąd testu."""
        self.test_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.result_text.setPlainText(f"❌ Błąd: {error}")
        self.result_text.setStyleSheet("color: #f44336;")

    def set_endpoint(self, endpoint: str):
        """Ustaw endpoint z zewnątrz."""
        self.endpoint_edit.setPlainText(endpoint)

    def set_model_name(self, model_name: str):
        """Ustaw nazwę modelu z zewnątrz."""
        self.model_name_edit.setPlainText(model_name)
