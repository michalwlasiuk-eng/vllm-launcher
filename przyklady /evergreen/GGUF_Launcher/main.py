#!/usr/bin/env python3
"""
GGUF AI Model Launcher - Main GUI Application
PyQt5-based interface for managing and launching llama.cpp servers
"""

import sys
import os
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QLabel,
    QPushButton,
    QLineEdit,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QCheckBox,
    QFormLayout,
    QSplitter,
    QListWidget,
    QListWidgetItem,
    QStatusBar,
    QProgressBar,
    QFrame,
    QScrollArea,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

from profile_manager import ProfileManager
from gguf_parser import GGUFParser
from server_launcher import ServerLauncher


class ConfigWorker(QThread):
    """Worker thread for parsing GGUF files"""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath

    def run(self):
        try:
            parser = GGUFParser(self.filepath)
            data = parser.parse()
            self.finished.emit(data)
        except Exception as e:
            self.error.emit(str(e))


class MainApp(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GGUF AI Model Launcher v1.0")
        self.setMinimumSize(1200, 800)

        self.profile_manager = ProfileManager()
        self.server_launcher = ServerLauncher()
        self.current_config = {}
        self.current_model_info = None

        self.init_ui()
        self.load_profile_list()

    def init_ui(self):
        """Initialize user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Model and Profile management
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel - Configuration and Server control
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([400, 800])

        main_layout.addWidget(splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def create_left_panel(self) -> QWidget:
        """Create left panel with model selection and profiles"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Model selection section
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()

        self.model_list = QListWidget()
        self.model_list.itemDoubleClicked.connect(self.on_model_selected)
        model_layout.addWidget(self.model_list)

        btn_layout = QHBoxLayout()
        browse_btn = QPushButton("📁 Browse GGUF")
        browse_btn.clicked.connect(self.browse_model)
        btn_layout.addWidget(browse_btn)

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_models)
        btn_layout.addWidget(refresh_btn)

        model_layout.addLayout(btn_layout)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Profile selection
        profile_group = QGroupBox("Configuration Profiles")
        profile_layout = QVBoxLayout()

        self.profile_combo = QComboBox()
        self.profile_combo.currentIndexChanged.connect(self.on_profile_changed)
        profile_layout.addWidget(self.profile_combo)

        profile_btn_layout = QHBoxLayout()
        new_profile_btn = QPushButton("➕ New Profile")
        new_profile_btn.clicked.connect(self.create_new_profile)
        profile_btn_layout.addWidget(new_profile_btn)

        delete_profile_btn = QPushButton("🗑️ Delete")
        delete_profile_btn.clicked.connect(self.delete_profile)
        profile_btn_layout.addWidget(delete_profile_btn)

        profile_layout.addLayout(profile_btn_layout)
        profile_group.setLayout(profile_layout)
        layout.addWidget(profile_group)

        # Model info display
        info_group = QGroupBox("Model Information")
        info_layout = QVBoxLayout()

        self.model_info_text = QTextEdit()
        self.model_info_text.setReadOnly(True)
        self.model_info_text.setMaximumHeight(200)
        info_layout.addWidget(self.model_info_text)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()
        return panel

    def create_right_panel(self) -> QWidget:
        """Create right panel with server configuration"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Server control buttons
        control_group = QGroupBox("Server Control")
        control_layout = QHBoxLayout()

        self.launch_btn = QPushButton("▶️ Launch Server")
        self.launch_btn.clicked.connect(self.launch_server)
        self.launch_btn.setEnabled(False)
        control_layout.addWidget(self.launch_btn)

        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.clicked.connect(self.stop_server)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # Configuration tabs
        config_tabs = QTabWidget()

        # General settings tab
        general_tab = self.create_general_settings_tab()
        config_tabs.addTab(general_tab, "General")

        # GPU settings tab
        gpu_tab = self.create_gpu_settings_tab()
        config_tabs.addTab(gpu_tab, "GPU")

        # Performance tab
        perf_tab = self.create_performance_tab()
        config_tabs.addTab(perf_tab, "Performance")

        # Advanced tab
        advanced_tab = self.create_advanced_tab()
        config_tabs.addTab(advanced_tab, "Advanced")

        layout.addWidget(config_tabs)

        # Server logs
        logs_group = QGroupBox("Server Logs")
        logs_layout = QVBoxLayout()

        self.logs_text = QTextEdit()
        self.logs_text.setReadOnly(True)
        self.logs_text.setMaximumHeight(150)
        logs_layout.addWidget(self.logs_text)

        logs_group.setLayout(logs_layout)
        layout.addWidget(logs_group)

        layout.addStretch()
        return panel

     def create_general_settings_tab(self) -> QWidget:
        """Create general settings tab"""
        widget = QWidget()
        layout = QFormLayout(widget)

        # Model path
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Select GGUF model file")
        layout.addRow("Model File:", self.model_path_input)

        # Server port
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(8080)
        layout.addRow("Port:", self.port_spin)

        # Chat template
        self.chat_template_check = QCheckBox("Use chat template")
        layout.addRow(self.chat_template_check)

        return widget

    def create_gpu_settings_tab(self) -> QWidget:
        """Create GPU settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # GPU layers
        layers_group = QGroupBox("GPU Layers")
        layers_layout = QFormLayout()

        self.gpu_layers_spin = QSpinBox()
        self.gpu_layers_spin.setRange(0, 100)
        self.gpu_layers_spin.setValue(0)
        self.gpu_layers_spin.setSuffix(" / Auto")
        layers_layout.addRow("GPU Layers:", self.gpu_layers_spin)

        layers_group.setLayout(layers_layout)
        layout.addWidget(layers_group)

        # Memory usage
        mem_group = QGroupBox("Memory Usage")
        mem_layout = QFormLayout()

        self.memory_combo = QComboBox()
        self.memory_combo.addItems(["Auto", "Low", "Medium", "High"])
        mem_layout.addRow("Optimization:", self.memory_combo)

        mem_group.setLayout(mem_layout)
        layout.addWidget(mem_group)

        layout.addStretch()
        return widget

    def create_performance_tab(self) -> QWidget:
        """Create performance settings tab"""
        widget = QWidget()
        main_layout = QVBoxLayout(widget)

        # Context size
        ctx_group = QGroupBox("Context")
        ctx_layout = QFormLayout()

        self.context_size_spin = QSpinBox()
        self.context_size_spin.setRange(256, 32768)
        self.context_size_spin.setValue(4096)
        ctx_layout.addRow("Context Size:", self.context_size_spin)

        ctx_group.setLayout(ctx_layout)
        main_layout.addWidget(ctx_group)

        # Threads
        threads_group = QGroupBox("Threads")
        threads_layout = QFormLayout()

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 64)
        self.threads_spin.setValue(4)
        threads_layout.addRow("CPU Threads:", self.threads_spin)

        threads_group.setLayout(threads_layout)
        main_layout.addWidget(threads_group)

        # Batch size
        batch_group = QGroupBox("Batch Size")
        batch_layout = QFormLayout()

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(64, 4096)
        self.batch_size_spin.setValue(512)
        batch_layout.addRow("Batch Size:", self.batch_size_spin)

        batch_group.setLayout(batch_layout)
        main_layout.addWidget(batch_group)

        main_layout.addStretch()
        return widget

    def create_advanced_tab(self) -> QWidget:
        """Create advanced settings tab"""
        widget = QWidget()
        main_layout = QVBoxLayout(widget)

        # Temperature
        temp_group = QGroupBox("Generation")
        temp_layout = QFormLayout()

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.1, 2.0)
        self.temp_spin.setValue(0.8)
        self.temp_spin.setSingleStep(0.1)
        temp_layout.addRow("Temperature:", self.temp_spin)

        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 128)
        self.top_k_spin.setValue(40)
        temp_layout.addRow("Top K:", self.top_k_spin)

        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setValue(0.9)
        self.top_p_spin.setSingleStep(0.01)
        temp_layout.addRow("Top P:", self.top_p_spin)

        self.repeat_penalty_spin = QDoubleSpinBox()
        self.repeat_penalty_spin.setRange(1.0, 2.0)
        self.repeat_penalty_spin.setValue(1.1)
        self.repeat_penalty_spin.setSingleStep(0.1)
        temp_layout.addRow("Repeat Penalty:", self.repeat_penalty_spin)

        temp_group.setLayout(temp_layout)
        main_layout.addWidget(temp_group)

        # Stop sequences
        stop_group = QGroupBox("Stop Sequences")
        stop_layout = QVBoxLayout()

        self.stop_sequences_input = QLineEdit()
        self.stop_sequences_input.setPlaceholderText("comma, separated, values")
        stop_layout.addWidget(self.stop_sequences_input)

        stop_group.setLayout(stop_layout)
        main_layout.addWidget(stop_group)

        main_layout.addStretch()
        return widget

    def browse_model(self):
        """Open file dialog to browse for GGUF model"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Model", "", "GGUF Files (*.gguf)"
        )
        if filepath:
            self.load_model(filepath)

    def load_model(self, filepath: str):
        """Load and display model information"""
        self.status_bar.showMessage(f"Loading model: {os.path.basename(filepath)}")

        self.model_path_input.setText(filepath)
        self.current_config["model_path"] = filepath

        # Start parsing in background thread
        self.config_worker = ConfigWorker(filepath)
        self.config_worker.finished.connect(self.on_model_parsed)
        self.config_worker.error.connect(self.on_parse_error)
        self.config_worker.start()

    def on_model_parsed(self, data: dict):
        """Handle model parsing completion"""
        self.current_model_info = data

        # Display model info
        info_text = f"Name: {data['metadata'].get('general.name', 'Unknown')}\n"
        info_text += (
            f"Description: {data['metadata'].get('general.description', 'N/A')}\n\n"
        )

        if "architecture" in data and data["architecture"]:
            info_text += "Architecture:\n"
            for key, value in data["architecture"].items():
                info_text += f"  {key}: {value}\n"

        self.model_info_text.setPlainText(info_text)

        # Auto-fill config from metadata if available
        if "architecture" in data:
            arch = data["architecture"]
            if "context_length" in arch:
                self.context_size_spin.setValue(min(arch["context_length"], 32768))
            if "embedding_length" in arch:
                self.model_path_input.setToolTip(
                    f"Embedding dim: {arch['embedding_length']}"
                )

        self.status_bar.showMessage("Model loaded successfully")
        self.launch_btn.setEnabled(True)

    def on_parse_error(self, error_msg: str):
        """Handle parsing error"""
        QMessageBox.critical(self, "Error", f"Failed to parse model: {error_msg}")
        self.status_bar.showMessage("Failed to load model")

    def refresh_models(self):
        """Refresh model list"""
        self.model_list.clear()

        # Look for GGUF files in current directory and subdirectories
        base_path = Path(__file__).parent

        for gguf_file in base_path.rglob("*.gguf"):
            self.model_list.addItem(str(gguf_file))

        self.status_bar.showMessage(f"Found {self.model_list.count()} GGUF file(s)")

    def on_model_selected(self, item: QListWidgetItem):
        """Handle model selection from list"""
        filepath = item.text()
        self.load_model(filepath)

    def load_profile_list(self):
        """Load profiles into combo box"""
        profiles = self.profile_manager.get_all_profiles()
        self.profile_combo.clear()
        self.profile_combo.addItems(profiles)

    def on_profile_changed(self, index: int):
        """Handle profile selection change"""
        if index >= 0:
            profile_name = self.profile_combo.currentText()
            config = self.profile_manager.get_profile(profile_name)
            if config:
                self.apply_profile_config(config)

    def apply_profile_config(self, config: dict):
        """Apply configuration from profile"""
        self.current_config.update(config)

        if "model_path" in config:
            self.model_path_input.setText(config["model_path"])

        if "port" in config:
            self.port_spin.setValue(config["port"])

        if "gpu_layers" in config:
            self.gpu_layers_spin.setValue(config["gpu_layers"])

        if "context_size" in config:
            self.context_size_spin.setValue(config["context_size"])

        if "n_threads" in config:
            self.threads_spin.setValue(config["n_threads"])

        if "batch_size" in config:
            self.batch_size_spin.setValue(config["batch_size"])

        if "temp" in config:
            self.temp_spin.setValue(config["temp"])

        if "top_k" in config:
            self.top_k_spin.setValue(config["top_k"])

        if "top_p" in config:
            self.top_p_spin.setValue(config["top_p"])

        if "repeat_penalty" in config:
            self.repeat_penalty_spin.setValue(config["repeat_penalty"])

        if "use_chat_template" in config:
            self.chat_template_check.setChecked(config["use_chat_template"])

        if "memory_usage" in config:
            mem_idx = self.memory_combo.findText(str(config["memory_usage"]))
            if mem_idx >= 0:
                self.memory_combo.setCurrentIndex(mem_idx)

    def create_new_profile(self):
        """Create a new configuration profile"""
        name, ok = QLineEdit.getText(self, "New Profile", "Enter profile name:")
        if ok and name:
            config = self.get_current_config()
            self.profile_manager.add_profile(name, config)
            self.load_profile_list()
            self.profile_combo.setCurrentText(name)
            self.status_bar.showMessage(f"Profile '{name}' created")

    def delete_profile(self):
        """Delete selected profile"""
        name = self.profile_combo.currentText()
        if (
            name
            and QMessageBox.question(
                self,
                "Delete Profile",
                f"Delete profile '{name}'?",
                QMessageBox.Yes | QMessageBox.No,
            )
            == QMessageBox.Yes
        ):
            self.profile_manager.delete_profile(name)
            self.load_profile_list()
            self.status_bar.showMessage(f"Profile '{name}' deleted")

    def get_current_config(self) -> dict:
        """Get current configuration from UI"""
        return {
            "model_path": self.model_path_input.text(),
            "port": self.port_spin.value(),
            "gpu_layers": self.gpu_layers_spin.value(),
            "context_size": self.context_size_spin.value(),
            "n_threads": self.threads_spin.value(),
            "batch_size": self.batch_size_spin.value(),
            "temp": self.temp_spin.value(),
            "top_k": self.top_k_spin.value(),
            "top_p": self.top_p_spin.value(),
            "repeat_penalty": self.repeat_penalty_spin.value(),
            "use_chat_template": self.chat_template_check.isChecked(),
            "memory_usage": self.memory_combo.currentText(),
            "stop_sequences": self.stop_sequences_input.text(),
        }

    def launch_server(self):
        """Launch llama.cpp server"""
        if not self.current_config.get("model_path"):
            QMessageBox.warning(self, "Warning", "Please select a model file first")
            return

        config = self.get_current_config()

        if self.server_launcher.launch_server(config):
            self.status_bar.showMessage("Server launched successfully")
            self.launch_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.update_server_logs()
        else:
            QMessageBox.critical(self, "Error", "Failed to launch server")

    def stop_server(self):
        """Stop the running server"""
        if self.server_launcher.stop_server():
            self.status_bar.showMessage("Server stopped")
            self.launch_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.logs_text.clear()

    def update_server_logs(self):
        """Update server logs display"""
        if self.server_launcher.is_running():
            logs = self.server_launcher.get_server_logs()
            if logs:
                self.logs_text.append("\n".join(logs))

            QTimer.singleShot(2000, self.update_server_logs)

    def closeEvent(self, event):
        """Handle application close"""
        if self.server_launcher.is_running():
            self.server_launcher.stop_server()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("GGUF AI Model Launcher")
    app.setOrganizationName("GGUF_Launcher")

    window = MainApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
