"""Widget monitora GPU/CPU — aktualizuje się co 2 sekundy."""

from typing import Optional

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from backend.gpu_monitor import CPUMetrics, GPUMetrics, get_cpu_metrics, get_gpu_metrics


def make_label(text: str, bold: bool = False) -> QLabel:
    """Utwórz etykietę z opcjonalnie pogrubioną czcionką."""
    label = QLabel(text)
    label.setStyleSheet("color: #d4d4d4;")
    if bold:
        label.setStyleSheet("color: #d4d4d4; font-weight: bold;")
    return label


def make_value_label(value: str, color: str = "#4caf50") -> QLabel:
    """Etykieta z wartością w kolorze."""
    label = QLabel(value)
    label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 14px;")
    return label


def make_progress_bar(
    value: float, color: str = "#0e639c", max_value: float = 100.0
) -> QProgressBar:
    """Pasek postępu z kolorem."""
    bar = QProgressBar()
    bar.setRange(0, int(max_value))
    bar.setValue(int(value))
    bar.setTextVisible(False)
    bar.setStyleSheet(f"""
        QProgressBar::chunk {{
            background-color: {color};
        }}
    """)
    return bar


class GPUInfoCard(QWidget):
    """Karta z informacjami o pojedynczej karcie GPU."""

    def __init__(self, gpu: GPUMetrics, parent: QWidget = None):
        super().__init__(parent)
        self.gpu = gpu
        # Referencje do widgetów do aktualizacji
        self.header_label: Optional[QLabel] = None
        self.temp_label: Optional[QLabel] = None
        self.util_bar: Optional[QProgressBar] = None
        self.util_label: Optional[QLabel] = None
        self.vram_bar: Optional[QProgressBar] = None
        self.vram_label: Optional[QLabel] = None
        self.power_label: Optional[QLabel] = None
        self.fan_bar: Optional[QProgressBar] = None
        self.fan_label: Optional[QLabel] = None
        # Row layouty do show/hide
        self.temp_row: Optional[QWidget] = None
        self.util_row: Optional[QWidget] = None
        self.vram_row: Optional[QWidget] = None
        self.power_row: Optional[QWidget] = None
        self.fan_row: Optional[QWidget] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Header z nazwą GPU
        header = QHBoxLayout()
        self.header_label = make_label(f"GPU {self.gpu.gpu_id}: {self.gpu.name}", bold=True)
        header.addWidget(self.header_label)
        header.addStretch()
        layout.addLayout(header)

        # Temperatura — w osobnym QWidget aby show/hide
        self.temp_row = QWidget()
        temp_layout = QHBoxLayout(self.temp_row)
        temp_layout.setContentsMargins(0, 0, 0, 0)
        temp_layout.addWidget(make_label("Temp:"))
        self.temp_label = None
        if self.gpu.temperature is not None:
            temp_color = (
                "#4caf50" if self.gpu.temperature < 70
                else "#ff9800" if self.gpu.temperature < 85
                else "#f44336"
            )
            self.temp_label = make_value_label(
                f"{self.gpu.temperature}°C", temp_color
            )
            temp_layout.addWidget(self.temp_label)
        temp_layout.addStretch()
        layout.addWidget(self.temp_row)
        self.temp_row.setVisible(self.gpu.temperature is not None)

        # Wykorzystanie GPU
        self.util_row = QWidget()
        util_layout = QHBoxLayout(self.util_row)
        util_layout.setContentsMargins(0, 0, 0, 0)
        util_layout.addWidget(make_label("GPU:"))
        self.util_bar = None
        self.util_label = None
        if self.gpu.utilization is not None:
            self.util_bar = make_progress_bar(self.gpu.utilization)
            self.util_bar.setFixedWidth(150)
            util_layout.addWidget(self.util_bar)
            self.util_label = make_value_label(
                f"{self.gpu.utilization}%", "#0e639c"
            )
            util_layout.addWidget(self.util_label)
        util_layout.addStretch()
        layout.addWidget(self.util_row)
        self.util_row.setVisible(self.gpu.utilization is not None)

        # VRAM
        self.vram_row = QWidget()
        vram_layout = QHBoxLayout(self.vram_row)
        vram_layout.setContentsMargins(0, 0, 0, 0)
        vram_layout.addWidget(make_label("VRAM:"))
        self.vram_bar = None
        self.vram_label = None
        if self.gpu.memory_used is not None and self.gpu.memory_percent is not None:
            self.vram_bar = make_progress_bar(self.gpu.memory_percent)
            self.vram_bar.setFixedWidth(150)
            vram_layout.addWidget(self.vram_bar)
            vram_text = f"{self.gpu.memory_used_human}/{self.gpu.memory_total_human}"
            self.vram_label = make_value_label(vram_text, "#4caf50")
            vram_layout.addWidget(self.vram_label)
        vram_layout.addStretch()
        layout.addWidget(self.vram_row)
        self.vram_row.setVisible(
            self.gpu.memory_used is not None and self.gpu.memory_percent is not None
        )

        # Power
        self.power_row = QWidget()
        power_layout = QHBoxLayout(self.power_row)
        power_layout.setContentsMargins(0, 0, 0, 0)
        power_layout.addWidget(make_label("Power:"))
        self.power_label = None
        if self.gpu.power_draw is not None:
            self.power_label = make_value_label(
                f"{self.gpu.power_draw:.1f}W", "#ff9800"
            )
            power_layout.addWidget(self.power_label)
        power_layout.addStretch()
        layout.addWidget(self.power_row)
        self.power_row.setVisible(self.gpu.power_draw is not None)

        # Wentylator
        self.fan_row = QWidget()
        fan_layout = QHBoxLayout(self.fan_row)
        fan_layout.setContentsMargins(0, 0, 0, 0)
        fan_layout.addWidget(make_label("Fan:"))
        self.fan_bar = None
        self.fan_label = None
        if self.gpu.fan_speed is not None:
            self.fan_bar = make_progress_bar(self.gpu.fan_speed)
            self.fan_bar.setFixedWidth(100)
            fan_layout.addWidget(self.fan_bar)
            self.fan_label = make_value_label(
                f"{self.gpu.fan_speed}%", "#4caf50"
            )
            fan_layout.addWidget(self.fan_label)
        fan_layout.addStretch()
        layout.addWidget(self.fan_row)
        self.fan_row.setVisible(self.gpu.fan_speed is not None)

    def _temp_color(self, temp: float) -> str:
        if temp < 70:
            return "#4caf50"
        if temp < 85:
            return "#ff9800"
        return "#f44336"

    def update_metrics(self, gpu: GPUMetrics):
        """Aktualizuj metryki bez niszczenia layoutu."""
        self.gpu = gpu

        # Header
        if self.header_label:
            self.header_label.setText(f"GPU {gpu.gpu_id}: {gpu.name}")

        # Temperatura
        if gpu.temperature is not None and self.temp_label:
            color = self._temp_color(gpu.temperature)
            self.temp_label.setText(f"{gpu.temperature}°C")
            self.temp_label.setStyleSheet(
                f"color: {color}; font-weight: bold; font-size: 14px;"
            )
        self.temp_row.setVisible(gpu.temperature is not None)

        # Wykorzystanie GPU
        if gpu.utilization is not None:
            if self.util_bar:
                self.util_bar.setValue(int(gpu.utilization))
            if self.util_label:
                self.util_label.setText(f"{gpu.utilization}%")
        self.util_row.setVisible(gpu.utilization is not None)

        # VRAM
        if gpu.memory_used is not None and gpu.memory_percent is not None:
            if self.vram_bar:
                self.vram_bar.setValue(int(gpu.memory_percent))
            if self.vram_label:
                vram_text = f"{gpu.memory_used_human}/{gpu.memory_total_human}"
                self.vram_label.setText(vram_text)
        self.vram_row.setVisible(
            gpu.memory_used is not None and gpu.memory_percent is not None
        )

        # Power
        if gpu.power_draw is not None and self.power_label:
            self.power_label.setText(f"{gpu.power_draw:.1f}W")
        self.power_row.setVisible(gpu.power_draw is not None)

        # Wentylator
        if gpu.fan_speed is not None:
            if self.fan_bar:
                self.fan_bar.setValue(int(gpu.fan_speed))
            if self.fan_label:
                self.fan_label.setText(f"{gpu.fan_speed}%")
        self.fan_row.setVisible(gpu.fan_speed is not None)


class CPUInfoCard(QWidget):
    """Karta z informacjami o CPU i RAM."""

    def __init__(self, cpu: CPUMetrics, parent: QWidget = None):
        super().__init__(parent)
        self.cpu = cpu
        # Referencje do widgetów
        self.cpu_bar: Optional[QProgressBar] = None
        self.cpu_label: Optional[QLabel] = None
        self.ram_bar: Optional[QProgressBar] = None
        self.ram_label: Optional[QLabel] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Header
        header = QHBoxLayout()
        header.addWidget(make_label(f"CPU ({self.cpu.cpu_count} rdzeni)", bold=True))
        header.addStretch()
        layout.addLayout(header)

        # CPU usage
        self.cpu_layout = QHBoxLayout()
        self.cpu_layout.addWidget(make_label("CPU:"))
        self.cpu_bar = make_progress_bar(self.cpu.cpu_percent, "#4caf50")
        self.cpu_bar.setFixedWidth(150)
        self.cpu_layout.addWidget(self.cpu_bar)
        self.cpu_label = make_value_label(f"{self.cpu.cpu_percent:.1f}%", "#4caf50")
        self.cpu_layout.addWidget(self.cpu_label)
        self.cpu_layout.addStretch()
        layout.addLayout(self.cpu_layout)

        # RAM
        self.ram_layout = QHBoxLayout()
        self.ram_layout.addWidget(make_label("RAM:"))
        self.ram_bar = make_progress_bar(self.cpu.memory_percent)
        self.ram_bar.setFixedWidth(150)
        self.ram_layout.addWidget(self.ram_bar)
        ram_text = f"{self.cpu.memory_used_human}/{self.cpu.memory_total_human}"
        self.ram_label = make_value_label(ram_text, "#4caf50")
        self.ram_layout.addWidget(self.ram_label)
        self.ram_layout.addStretch()
        layout.addLayout(self.ram_layout)

    def update_metrics(self, cpu: CPUMetrics):
        """Aktualizuj metryki bez niszczenia layoutu."""
        self.cpu = cpu

        # CPU
        if self.cpu_bar:
            self.cpu_bar.setValue(int(cpu.cpu_percent))
        if self.cpu_label:
            cpu_color = "#4caf50" if cpu.cpu_percent < 60 else "#ff9800" if cpu.cpu_percent < 90 else "#f44336"
            self.cpu_label.setText(f"{cpu.cpu_percent:.1f}%")
            self.cpu_label.setStyleSheet(f"color: {cpu_color}; font-weight: bold; font-size: 14px;")

        # RAM
        if self.ram_bar:
            self.ram_bar.setValue(int(cpu.memory_percent))
        if self.ram_label:
            ram_text = f"{cpu.memory_used_human}/{cpu.memory_total_human}"
            self.ram_label.setText(ram_text)


class GPUMonitorWidget(QWidget):
    """Główny widget monitora GPU/CPU z automatycznym odświeżaniem."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.gpu_cards: list[GPUInfoCard] = []
        self.cpu_card: Optional[CPUInfoCard] = None
        self._setup_ui()

        # Timer do odświeżania
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh)
        self.refresh_timer.start(2000)  # co 2 sekundy

        # Pierwsze odświeżenie
        self._refresh()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area dla wielu GPU
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("border: none; background: transparent;")
        main_layout.addWidget(self.scroll)

        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(8, 8, 8, 8)
        self.scroll_layout.setSpacing(8)
        self.scroll.setWidget(self.scroll_content)

    def _refresh(self):
        """Odśwież metryki GPU i CPU."""
        # CPU
        cpu = get_cpu_metrics()
        if self.cpu_card:
            self.cpu_card.update_metrics(cpu)
        else:
            self.cpu_card = CPUInfoCard(cpu)
            self.scroll_layout.insertWidget(0, self.cpu_card)

        # GPU
        gpus = get_gpu_metrics()

        # Usuń stare karty GPU jeśli zmieniła się liczba
        while len(self.gpu_cards) > len(gpus):
            card = self.gpu_cards.pop()
            card.deleteLater()

        # Zaktualizuj lub utwórz nowe karty GPU
        for i, gpu in enumerate(gpus):
            if i < len(self.gpu_cards):
                self.gpu_cards[i].update_metrics(gpu)
            else:
                card = GPUInfoCard(gpu)
                self.gpu_cards.append(card)
                self.scroll_layout.addWidget(card)

        # Jeśli brak GPU, pokaż komunikat
        if not gpus:
            for card in self.gpu_cards:
                card.deleteLater()
            self.gpu_cards = []

            # Sprawdź czy to brak pynvml czy brak GPU
            if not hasattr(self, '_no_gpu_warning_shown'):
                from backend.gpu_monitor import HAS_PYNVML
                if not HAS_PYNVML:
                    msg = "pynvml nie zainstalowany — GPU monitoring niedostępny"
                else:
                    msg = "Nie wykryto kart NVIDIA — sprawdz NVML"

                warning = QLabel(f"⚠️ {msg}")
                warning.setStyleSheet("color: #ff9800; font-style: italic;")
                self.scroll_layout.addWidget(warning)
                self._no_gpu_warning_shown = True

    def set_refresh_interval(self, ms: int):
        """Ustaw interwał odświeżania w milisekundach."""
        self.refresh_timer.setInterval(ms)

    def stop_refresh(self):
        """Zatrzymaj automatyczne odświeżanie."""
        self.refresh_timer.stop()

    def start_refresh(self):
        """Wznów odświeżanie."""
        self.refresh_timer.start(2000)
