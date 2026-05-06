"""Menedżer konfiguracji — zapis/odczyt YAML."""

import json
import os
import subprocess
from pathlib import Path

import yaml

# Ścieżka pliku konfiguracyjnego
CONFIG_DIR = Path.home() / ".config" / "vllm-manager"
CONFIG_FILE = CONFIG_DIR / "config.yaml"

DEFAULT_CONFIG = {
    "vllm": {
        "venv_path": "",
        "detected": False,
        "launch_params": {
            "host": "127.0.0.1",
            "port": 8000,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "dtype": "auto",
            "max_model_len": 0,
            "enforce_eager": False,
            "disable_custom_all_reduce": False,
            "max_num_seqs": 256,
            "swap_space": 4,
            "api_key": "",
            "trust_remote_code": True,
            "chat_template": "",
            "disable_log_stats": False,
            "enable_log_requests": False,
            "ssl_certfile": "",
            "ssl_keyfile": "",
            "root_path": "",
            "huggingface_token": "",
            "enable_auto_tool_choice": False,
            "tool_call_parser": "",
        },
    },
    "sglang": {
        "venv_path": "",
        "detected": False,
        "launch_params": {
            "host": "127.0.0.1",
            "port": 8000,
            "tensor_parallel_size": 1,
            "mem_fraction_static": 0.9,
            "dtype": "auto",
            "max_total_tokens": 0,
            "disable_cuda_graph": False,
            "disable_radix_cache": False,
            "max_running_requests": 512,
            "load_format": "auto",
            "quantization": "",
            "context_length": 0,
            "max_queued_requests": 256,
            "chunked_prefill_size": 0,
            "trust_remote_code": True,
            "schedule_policy": "lpm",
            "enable_priority_scheduling": False,
            "prefill_max_requests": 0,
            "enable_dynamic_chunking": False,
            "api_key": "",
            "ssl_certfile": "",
            "ssl_keyfile": "",
        },
    },
    "llama": {
        "binary_path": "",
        "detected": False,
        "launch_params": {
            "model": "",
            "host": "127.0.0.1",
            "port": 8080,
            "ctx_size": 0,
            "threads": -1,
            "threads_batch": -1,
            "gpu_layers": -1,
            "batch_size": 2048,
            "ubatch_size": 512,
            "n_predict": -1,
            "flash_attn": "auto",
            "kv_offload": True,
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "rope_scaling": "linear",
            "rope_scale": 1.0,
            "main_gpu": 0,
            "tensor_split": "",
            "split_mode": "layer",
            "mlock": False,
            "mmap": True,
            "no_direct_io": True,
            "numa": "",
            "lora": "",
            "control_vector": "",
            "verbose": False,
            "log_disable": False,
            "log_file": "",
            "escape": True,
            "keep": 0,
        },
    },
    "models_dir": "",
    "scan_cache": True,
}


def get_config_dir() -> Path:
    """Gwarantuje istnienie katalogu konfiguracyjnego."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


def load_config() -> dict:
    """Wczytaj konfigurację z YAML. Zwraca domyślną jeśli plik nie istnieje."""
    get_config_dir()  # gwarantuje istnienie katalogu

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            # scal z domyślną (na wypadek nowych kluczy)
            merged = {**DEFAULT_CONFIG, **cfg}
            # scal podslowniki
            for key in ("vllm", "sglang", "llama"):
                merged[key] = {**DEFAULT_CONFIG[key], **cfg.get(key, {})}
                # scal launch_params
                if "launch_params" in DEFAULT_CONFIG[key]:
                    merged[key]["launch_params"] = {
                        **DEFAULT_CONFIG[key]["launch_params"],
                        **cfg.get(key, {}).get("launch_params", {}),
                    }
            return merged
        except Exception:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict) -> None:
    """Zapisz konfigurację do YAML."""
    get_config_dir()
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


def detect_binary(venv_path: str, binary_name: str) -> tuple[bool, str]:
    """Sprawdź czy binarium istnieje w podanym venv.

    Zwraca (znaleziono, ścieżka_do_binarium).
    """
    if not venv_path:
        return False, ""

    candidates = [
        Path(venv_path) / "bin" / binary_name,
        Path(venv_path) / "Scripts" / f"{binary_name}.exe",
    ]

    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return True, str(candidate)

    return False, ""


def detect_python(venv_path: str) -> tuple[bool, str]:
    """Sprawdź czy python istnieje w podanym venv."""
    if not venv_path:
        return False, ""

    candidates = [
        Path(venv_path) / "bin" / "python3",
        Path(venv_path) / "bin" / "python",
        Path(venv_path) / "Scripts" / "python.exe",
    ]

    for candidate in candidates:
        if candidate.exists():
            return True, str(candidate)

    return False, ""


def check_module_in_venv(python_path: str, module_name: str) -> bool:
    """Sprawdź czy moduł jest zainstalowany w podanym venv."""
    try:
        result = subprocess.run(
            [python_path, "-c", f"import {module_name}"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def scan_common_venv_locations() -> list[dict]:
    """Przeskanuj popularne lokalizacje venv.

    Zwraca listę znalezionych venv z informacją co zawierają.
    """
    found = []

    # Popularne lokalizacje
    search_paths = [
        Path.home() / "venvs",
        Path.home() / ".venvs",
        Path.home() / "Envs",
        Path("/opt/venvs"),
        Path("/usr/local/venvs"),
        Path("/media/black/free"),
        Path("/home/black/venvs"),
    ]

    for search_path in search_paths:
        if not search_path.exists():
            continue

        for item in search_path.iterdir():
            if not item.is_dir():
                continue

            has_vllm = False
            has_sglang = False

            # Sprawdź pip listę
            py_path = item / "bin" / "python3"
            if not py_path.exists():
                py_path = item / "bin" / "python"
            if not py_path.exists():
                continue

            try:
                result = subprocess.run(
                    [str(py_path), "-m", "pip", "list", "--format=json"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                packages = [p["name"].lower() for p in json.loads(result.stdout)]

                has_vllm = any("vllm" in p for p in packages)
                has_sglang = any("sglang" in p for p in packages)
            except PermissionError:
                # Skip directories we don't have access to
                continue
            except Exception:
                continue

            if has_vllm or has_sglang:
                found.append(
                    {
                        "path": str(item),
                        "name": item.name,
                        "has_vllm": has_vllm,
                        "has_sglang": has_sglang,
                        "python": str(py_path),
                    }
                )

    return found
