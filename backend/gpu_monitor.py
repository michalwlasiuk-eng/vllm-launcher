"""Monitor GPU i CPU — zbieranie metryk z pynvml i psutil."""

from dataclasses import dataclass
from typing import List, Optional

import psutil

# Próba importu pynvml
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


@dataclass
class GPUMetrics:
    """Metryki pojedynczej karty GPU."""
    gpu_id: int
    name: str
    temperature: Optional[float]  # °C
    utilization: Optional[float]  # %
    memory_used: Optional[int]   # bytes
    memory_total: Optional[int]  # bytes
    power_draw: Optional[float]  # watts
    fan_speed: Optional[float]   # %

    @property
    def memory_used_human(self) -> str:
        """Rozmiar pamięci w GB."""
        if self.memory_used is None:
            return "N/A"
        return f"{self.memory_used / (1024**3):.1f} GB"

    @property
    def memory_total_human(self) -> str:
        """Całkowita pamięć w GB."""
        if self.memory_total is None:
            return "N/A"
        return f"{self.memory_total / (1024**3):.1f} GB"

    @property
    def memory_percent(self) -> Optional[float]:
        """Procent zużycia pamięci."""
        if self.memory_used is None or self.memory_total is None or self.memory_total == 0:
            return None
        return (self.memory_used / self.memory_total) * 100


@dataclass
class CPUMetrics:
    """Metryki procesora."""
    cpu_percent: float
    memory_total: int
    memory_used: int
    memory_available: int
    memory_percent: float
    cpu_count: int

    @property
    def memory_used_human(self) -> str:
        return f"{self.memory_used / (1024**3):.1f} GB"

    @property
    def memory_total_human(self) -> str:
        return f"{self.memory_total / (1024**3):.1f} GB"


def get_cpu_metrics() -> CPUMetrics:
    """Zbierz metryki CPU i RAM."""
    mem = psutil.virtual_memory()
    cpu_pct = psutil.cpu_percent(interval=0.1)
    cpu_count = psutil.cpu_count(logical=True)

    return CPUMetrics(
        cpu_percent=cpu_pct,
        memory_total=mem.total,
        memory_used=mem.used,
        memory_available=mem.available,
        memory_percent=mem.percent,
        cpu_count=cpu_count,
    )


def get_gpu_metrics() -> List[GPUMetrics]:
    """Zbierz metryki wszystkich dostępnych GPU."""
    if not HAS_PYNVML:
        return []

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        metrics = []

        for i in range(device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                try:
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8", errors="replace")
                except Exception:
                    name = f"GPU {i}"

                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    temp = None

                try:
                    utils = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    util = utils.gpu
                except Exception:
                    util = None

                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used = mem_info.used
                    mem_total = mem_info.total
                except Exception:
                    mem_used = None
                    mem_total = None

                try:
                    power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except Exception:
                    power_draw = None

                try:
                    fan = pynvml.nvmlDeviceGetFanSpeed(handle)
                except Exception:
                    fan = None

                metrics.append(GPUMetrics(
                    gpu_id=i,
                    name=name,
                    temperature=temp,
                    utilization=util,
                    memory_used=mem_used,
                    memory_total=mem_total,
                    power_draw=power_draw,
                    fan_speed=fan,
                ))
            except Exception as e:
                print(f"[GPU Monitor] Błąd odczytu GPU {i}: {e}")
                continue

    except Exception as e:
        print(f"[GPU Monitor] Błąd inicjalizacji NVML: {e}")
        return []

    return metrics


def format_gpu_status(gpu: GPUMetrics) -> str:
    """Sformatuj status GPU jako czytelny tekst."""
    lines = [f"GPU {gpu.gpu_id}: {gpu.name}"]

    if gpu.temperature is not None:
        lines.append(f"  Temp: {gpu.temperature}°C")
    if gpu.utilization is not None:
        lines.append(f"  GPU: {gpu.utilization}%")
    if gpu.memory_used is not None:
        lines.append(f"  VRAM: {gpu.memory_used_human}/{gpu.memory_total_human}")
        if gpu.memory_percent is not None:
            lines.append(f"       ({gpu.memory_percent:.1f}%)")
    if gpu.power_draw is not None:
        lines.append(f"  Power: {gpu.power_draw:.1f}W")
    if gpu.fan_speed is not None:
        lines.append(f"  Fan: {gpu.fan_speed}%")

    return "\n".join(lines)
