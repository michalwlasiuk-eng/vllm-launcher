"""Skaner modeli — wykrywanie formatów (GGUF, HF/safetensors, PyTorch)."""

import json
import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import psutil


class ModelFormat(Enum):
    """Obsługiwane formaty modeli."""

    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    MERGED = "merged"  # HF + GGUF w tym samym folderze


class EngineSupport(Enum):
    """Współpraca z silnikami."""

    VLLM = "vLLM"
    SGLANG = "sglang"
    BOTH = "obaj"
    UNKNOWN = "?"


@dataclass
class ModelInfo:
    """Informacje o jednym modelu."""

    name: str
    path: str
    size_bytes: int
    format: ModelFormat
    engine_support: EngineSupport
    files: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # metadata z config.json
    arch: Optional[str] = None
    hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    vocab_size: Optional[int] = None
    # dla GGUF
    gguf_arch: Optional[str] = None
    gguf_dtype: Optional[str] = None
    gguf_file: Optional[str] = None  # pełna ścieżka do pliku .gguf

    @property
    def model_path(self) -> str:
        """Zwraca ścieżkę do modelu - dla GGUF to plik, dla innych folder."""
        if self.format == ModelFormat.GGUF and self.gguf_file:
            return self.gguf_file
        return self.path

    @property
    def size_human(self) -> str:
        """Rozmiar w formacie czytelnym dla człowieka."""
        return format_size(self.size_bytes)

    @property
    def is_quantized(self) -> bool:
        """Czy model jest kwantyzowany."""
        return any("q" in f.lower() for f in self.files)


def format_size(size_bytes: int) -> str:
    """Formatowanie rozmiaru w MB/GB."""
    if size_bytes == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    unit_idx = 0

    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024
        unit_idx += 1

    if unit_idx == 0:
        return f"{int(size)} B"
    return f"{size:.1f} {units[unit_idx]}"


def get_folder_size(folder_path: str) -> int:
    """Oblicza rozmiar folderu w bajtach."""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
    except PermissionError:
        pass
    return total


def scan_gguf_file(file_path: str) -> dict:
    """Skanuje plik GGUF i wyciąga metadane.

    Używa gguf-info z vLLM lub ręcznego parsowania.
    """
    metadata = {}

    # Spróbuj przez vLLM
    try:
        result = subprocess.run(
            ["vllm", "quantization", "gguf", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        pass

    # Ręczne parsowanie GGUF header (uproszczone)
    try:
        with open(file_path, "rb") as f:
            # GGUF magic number
            magic = f.read(4)
            if magic != b"GGUF":
                return metadata

            # Version
            version = int.from_bytes(f.read(4), "little")

            # Number of tensors (v1, v2, v3)
            if version == 1:
                num_tensors = int.from_bytes(f.read(4), "little")
                num_kv = int.from_bytes(f.read(4), "little")
            else:
                num_tensors = int.from_bytes(f.read(8), "little")
                num_kv = int.from_bytes(f.read(8), "little")

            # Read key-value pairs
            for _ in range(num_tensors + num_kv if version == 1 else num_tensors):
                pass  # złożone do pełnej implementacji

            metadata["gguf_format"] = "v2"
    except Exception:
        pass

    return metadata


def detect_gguf_arch(folder_path: str) -> tuple[Optional[str], Optional[str]]:
    """Wykrywa architekturę modelu GGUF z plików w folderze.

    Zwraca (arch, dtype) lub (None, None).
    """
    # Szukaj pliku .gguf
    gguf_files = list(Path(folder_path).glob("*.gguf"))
    if not gguf_files:
        return None, None

    # Szukaj metadata w pliku GGUF
    arch = None
    dtype = None

    for gguf_file in gguf_files:
        try:
            with open(gguf_file, "rb") as f:
                magic = f.read(4)
                if magic != b"GGUF":
                    continue

                # W GGUF metadata są zakodowane jako string keys
                # Szukamy "general.architecture"
                # To uproszczenie — pełne parsowanie jest złożone
                # Na razie szukamy w nazwach plików
                fname = gguf_file.name.lower()

                if "qwen2" in fname or "qwen_2" in fname:
                    arch = "qwen2"
                elif "llama" in fname:
                    arch = "llama"
                elif "mistral" in fname:
                    arch = "mistral"
                elif "mixtral" in fname:
                    arch = "mixtral"
                elif "gemma" in fname:
                    arch = "gemma"
                elif "qwen" in fname:
                    arch = "qwen"
                elif "deepseek" in fname:
                    arch = "deepseek"

                # Wykryj kwantyzację
                if "q4_0" in fname or "q4_1" in fname:
                    dtype = "Q4_0"
                elif "q5_0" in fname or "q5_1" in fname:
                    dtype = "Q5_0"
                elif "q8_0" in fname:
                    dtype = "Q8_0"
                elif "f16" in fname or "f16" in str(gguf_file):
                    dtype = "F16"
                elif "f32" in fname:
                    dtype = "F32"

                break
        except Exception:
            continue

    return arch, dtype


def scan_hf_config(folder_path: str) -> dict:
    """Skanuje config.json i wyciąga metadane modelu."""
    config_path = Path(folder_path) / "config.json"
    if not config_path.exists():
        return {}

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        return {
            "arch": config.get("architectures", [None])[0],
            "hidden_size": config.get("hidden_size"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "vocab_size": config.get("vocab_size"),
            "max_position_embeddings": config.get("max_position_embeddings"),
            "torch_dtype": str(config.get("torch_dtype", "")),
        }
    except Exception:
        return {}


def check_safetensors(folder_path: str) -> bool:
    """Sprawdza czy folder zawiera safetensors."""
    safetensor_files = list(Path(folder_path).glob("*.safetensors"))
    return len(safetensor_files) > 0


def check_pytorch(folder_path: str) -> bool:
    """Sprawdza czy folder zawiera PyTorch checkpoints."""
    pt_files = list(Path(folder_path).glob("*.bin"))
    pt_files += list(Path(folder_path).glob("pytorch_model.bin"))
    return len(pt_files) > 0


def supports_vllm(model_info: ModelInfo) -> bool:
    """Sprawdza czy vLLM obsługuje ten model."""
    # GGUF — vLLM obsługuje przez convert
    if model_info.format == ModelFormat.GGUF:
        return True

    # Safetensors — standard dla vLLM
    if model_info.format == ModelFormat.SAFETENSORS:
        return True

    # PyTorch — vLLM może wymagać konwersji
    return False


def supports_sglang(model_info: ModelInfo) -> bool:
    """Sprawdza czy sglang obsługuje ten model."""
    # GGUF — sglang obsługuje przez load-format gguf
    if model_info.format == ModelFormat.GGUF:
        return True

    # Safetensors — standard dla sglang
    if model_info.format == ModelFormat.SAFETENSORS:
        return True

    # PyTorch
    return model_info.format == ModelFormat.PYTORCH


def scan_model_folder(folder_path: str) -> Optional[ModelInfo]:
    """Skana pojedynczy folder i zwraca informacje o modelu.

    Zwraca None jeśli to nie jest folder z modelem.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        return None

    # Sprawdź czy to model folder
    has_config = (folder / "config.json").exists()
    has_safetensors = check_safetensors(folder)
    has_gguf = any(folder.glob("*.gguf"))
    has_pytorch = check_pytorch(folder)

    if not (has_config or has_safetensors or has_gguf or has_pytorch):
        return None

    # Określ format
    if has_gguf and has_config:
        fmt = ModelFormat.MERGED
    elif has_gguf:
        fmt = ModelFormat.GGUF
    elif has_safetensors or has_config:
        fmt = ModelFormat.SAFETENSORS
    elif has_pytorch:
        fmt = ModelFormat.PYTORCH
    else:
        return None

    # Oblicz rozmiar
    size = get_folder_size(str(folder))

    # Zbierz pliki
    files = []
    for f in folder.rglob("*"):
        if f.is_file():
            files.append(f.name)
            if len(files) > 50:  # limit
                break

    # GGUF: znajdź plik .gguf
    gguf_file_path = None
    if has_gguf:
        gguf_list = list(folder.glob("*.gguf"))
        if gguf_list:
            gguf_file_path = str(gguf_list[0])

    # Utwórz ModelInfo
    info = ModelInfo(
        name=folder.name,
        path=str(folder),
        size_bytes=size,
        format=fmt,
        engine_support=EngineSupport.UNKNOWN,
        files=files,
    )

    # Przypisz plik GGUF
    info.gguf_file = gguf_file_path

    # Wczytaj config.json
    if has_config:
        cfg = scan_hf_config(str(folder))
        info.arch = cfg.get("arch")
        info.hidden_size = cfg.get("hidden_size")
        info.num_attention_heads = cfg.get("num_attention_heads")
        info.num_key_value_heads = cfg.get("num_key_value_heads")
        info.num_hidden_layers = cfg.get("num_hidden_layers")
        info.vocab_size = cfg.get("vocab_size")

    # Dla GGUF
    if fmt == ModelFormat.GGUF:
        gguf_arch, gguf_dtype = detect_gguf_arch(str(folder))
        info.gguf_arch = gguf_arch
        info.gguf_dtype = gguf_dtype

    # Sprawdź engine support
    vllm_ok = supports_vllm(info)
    sglang_ok = supports_sglang(info)

    if vllm_ok and sglang_ok:
        info.engine_support = EngineSupport.BOTH
    elif vllm_ok:
        info.engine_support = EngineSupport.VLLM
    elif sglang_ok:
        info.engine_support = EngineSupport.SGLANG
    else:
        info.engine_support = EngineSupport.UNKNOWN

    # Uwagi
    if info.is_quantized:
        info.notes.append("Kwantyzowany")
    if size < 2 * 1024 * 1024 * 1024:
        info.notes.append("Mały (<2GB)")
    elif size > 50 * 1024 * 1024 * 1024:
        info.notes.append("Duży (>50GB)")

    return info


def scan_models_directory(
    root_dir: str,
    max_depth: int = 3,
    recursive: bool = True,
) -> list[ModelInfo]:
    """Skana katalog z modelami.

    Args:
        root_dir: Katalog główny do skanowania
        max_depth: Maksymalna głębokość rekurencji
        recursive: Czy skanować rekurencyjnie

    Returns:
        Lista ModelInfo dla wykrytych modeli
    """
    models = []
    root = Path(root_dir)

    if not root.exists():
        return models

    # Zbierz wszystkie potencjalne modele
    folders = []
    for f in root.iterdir():
        try:
            if not f.is_dir():
                continue
            folders.append(f)
            if recursive:
                for sub in f.rglob("*"):
                    if sub.is_dir():
                        depth = len(sub.relative_to(f).parts)
                        if depth < max_depth:
                            folders.append(sub)
        except PermissionError:
            continue

    # Skanuj każdy folder
    for folder in folders:
        try:
            info = scan_model_folder(str(folder))
            if info:
                models.append(info)
        except Exception as e:
            # Kontynuuj pomijając błędne foldery
            continue

    # Sortuj: po nazwie, potem rozmiar
    models.sort(key=lambda m: (m.name.lower(), m.size_bytes))

    return models


def get_model_count(root_dir: str) -> int:
    """Szybki licznik modeli (rekurencyjnie)."""
    root = Path(root_dir)
    if not root.exists():
        return 0

    model_dirs = set()
    for pattern in ["**/config.json", "**/*.safetensors", "**/*.gguf", "**/*.bin"]:
        for item in root.glob(pattern):
            if item.is_file():
                model_dirs.add(item.parent)

    return len(model_dirs)


def format_size(size_bytes: int) -> str:
    """Formatowanie rozmiaru w MB/GB."""
    if size_bytes == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    unit_idx = 0

    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024
        unit_idx += 1

    if unit_idx == 0:
        return f"{int(size)} B"
    return f"{size:.1f} {units[unit_idx]}"


def get_folder_size(folder_path: str) -> int:
    """Oblicza rozmiar folderu w bajtach."""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
    except PermissionError:
        pass
    return total


def scan_gguf_file(file_path: str) -> dict:
    """Skanuje plik GGUF i wyciąga metadane.

    Używa gguf-info z vLLM lub ręcznego parsowania.
    """
    metadata = {}

    # Spróbuj przez vLLM
    try:
        result = subprocess.run(
            ["vllm", "quantization", "gguf", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        pass

    # Ręczne parsowanie GGUF header (uproszczone)
    try:
        with open(file_path, "rb") as f:
            # GGUF magic number
            magic = f.read(4)
            if magic != b"GGUF":
                return metadata

            # Version
            version = int.from_bytes(f.read(4), "little")

            # Number of tensors (v1, v2, v3)
            if version == 1:
                num_tensors = int.from_bytes(f.read(4), "little")
                num_kv = int.from_bytes(f.read(4), "little")
            else:
                num_tensors = int.from_bytes(f.read(8), "little")
                num_kv = int.from_bytes(f.read(8), "little")

            # Read key-value pairs
            for _ in range(num_tensors + num_kv if version == 1 else num_tensors):
                pass  # złożone do pełnej implementacji

            metadata["gguf_format"] = "v2"
    except Exception:
        pass

    return metadata


def detect_gguf_arch(folder_path: str) -> tuple[Optional[str], Optional[str]]:
    """Wykrywa architekturę modelu GGUF z plików w folderze.

    Zwraca (arch, dtype) lub (None, None).
    """
    # Szukaj pliku .gguf
    gguf_files = list(Path(folder_path).glob("*.gguf"))
    if not gguf_files:
        return None, None

    # Szukaj metadata w pliku GGUF
    arch = None
    dtype = None

    for gguf_file in gguf_files:
        try:
            with open(gguf_file, "rb") as f:
                magic = f.read(4)
                if magic != b"GGUF":
                    continue

                # W GGUF metadata są zakodowane jako string keys
                # Szukamy "general.architecture"
                # To uproszczenie — pełne parsowanie jest złożone
                # Na razie szukamy w nazwach plików
                fname = gguf_file.name.lower()

                if "qwen2" in fname or "qwen_2" in fname:
                    arch = "qwen2"
                elif "llama" in fname:
                    arch = "llama"
                elif "mistral" in fname:
                    arch = "mistral"
                elif "mixtral" in fname:
                    arch = "mixtral"
                elif "gemma" in fname:
                    arch = "gemma"
                elif "qwen" in fname:
                    arch = "qwen"
                elif "deepseek" in fname:
                    arch = "deepseek"

                # Wykryj kwantyzację
                if "q4_0" in fname or "q4_1" in fname:
                    dtype = "Q4_0"
                elif "q5_0" in fname or "q5_1" in fname:
                    dtype = "Q5_0"
                elif "q8_0" in fname:
                    dtype = "Q8_0"
                elif "f16" in fname or "f16" in str(gguf_file):
                    dtype = "F16"
                elif "f32" in fname:
                    dtype = "F32"

                break
        except Exception:
            continue

    return arch, dtype


def scan_hf_config(folder_path: str) -> dict:
    """Skanuje config.json i wyciąga metadane modelu."""
    config_path = Path(folder_path) / "config.json"
    if not config_path.exists():
        return {}

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        return {
            "arch": config.get("architectures", [None])[0],
            "hidden_size": config.get("hidden_size"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "vocab_size": config.get("vocab_size"),
            "max_position_embeddings": config.get("max_position_embeddings"),
            "torch_dtype": str(config.get("torch_dtype", "")),
        }
    except Exception:
        return {}


def check_safetensors(folder_path: str) -> bool:
    """Sprawdza czy folder zawiera safetensors."""
    safetensor_files = list(Path(folder_path).glob("*.safetensors"))
    return len(safetensor_files) > 0


def check_pytorch(folder_path: str) -> bool:
    """Sprawdza czy folder zawiera PyTorch checkpoints."""
    pt_files = list(Path(folder_path).glob("*.bin"))
    pt_files += list(Path(folder_path).glob("pytorch_model.bin"))
    return len(pt_files) > 0


def supports_vllm(model_info: ModelInfo) -> bool:
    """Sprawdza czy vLLM obsługuje ten model."""
    # GGUF — vLLM obsługuje przez convert
    if model_info.format == ModelFormat.GGUF:
        return True

    # Safetensors — standard dla vLLM
    if model_info.format == ModelFormat.SAFETENSORS:
        return True

    # PyTorch — vLLM może wymagać konwersji
    return False


def supports_sglang(model_info: ModelInfo) -> bool:
    """Sprawdza czy sglang obsługuje ten model."""
    # GGUF — sglang obsługuje przez load-format gguf
    if model_info.format == ModelFormat.GGUF:
        return True

    # Safetensors — standard dla sglang
    if model_info.format == ModelFormat.SAFETENSORS:
        return True

    # PyTorch
    return model_info.format == ModelFormat.PYTORCH


def scan_model_folder(folder_path: str) -> Optional[ModelInfo]:
    """Skana pojedynczy folder i zwraca informacje o modelu.

    Zwraca None jeśli to nie jest folder z modelem.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        return None

    # Sprawdź czy to model folder
    has_config = (folder / "config.json").exists()
    has_safetensors = check_safetensors(folder)
    has_gguf = any(folder.glob("*.gguf"))
    has_pytorch = check_pytorch(folder)

    if not (has_config or has_safetensors or has_gguf or has_pytorch):
        return None

    # Określ format
    if has_gguf and has_config:
        fmt = ModelFormat.MERGED
    elif has_gguf:
        fmt = ModelFormat.GGUF
    elif has_safetensors or has_config:
        fmt = ModelFormat.SAFETENSORS
    elif has_pytorch:
        fmt = ModelFormat.PYTORCH
    else:
        return None

    # Oblicz rozmiar
    size = get_folder_size(str(folder))

    # Zbierz pliki
    files = []
    for f in folder.rglob("*"):
        if f.is_file():
            files.append(f.name)
            if len(files) > 50:  # limit
                break

    # Utwórz ModelInfo
    info = ModelInfo(
        name=folder.name,
        path=str(folder),
        size_bytes=size,
        format=fmt,
        engine_support=EngineSupport.UNKNOWN,
        files=files,
    )

    # Wczytaj config.json
    if has_config:
        cfg = scan_hf_config(str(folder))
        info.arch = cfg.get("arch")
        info.hidden_size = cfg.get("hidden_size")
        info.num_attention_heads = cfg.get("num_attention_heads")
        info.num_key_value_heads = cfg.get("num_key_value_heads")
        info.num_hidden_layers = cfg.get("num_hidden_layers")
        info.vocab_size = cfg.get("vocab_size")

    # Dla GGUF
    if fmt == ModelFormat.GGUF:
        gguf_arch, gguf_dtype = detect_gguf_arch(str(folder))
        info.gguf_arch = gguf_arch
        info.gguf_dtype = gguf_dtype

    # Sprawdź engine support
    vllm_ok = supports_vllm(info)
    sglang_ok = supports_sglang(info)

    if vllm_ok and sglang_ok:
        info.engine_support = EngineSupport.BOTH
    elif vllm_ok:
        info.engine_support = EngineSupport.VLLM
    elif sglang_ok:
        info.engine_support = EngineSupport.SGLANG
    else:
        info.engine_support = EngineSupport.UNKNOWN

    # Uwagi
    if info.is_quantized:
        info.notes.append("Kwantyzowany")
    if size < 2 * 1024 * 1024 * 1024:
        info.notes.append("Mały (<2GB)")
    elif size > 50 * 1024 * 1024 * 1024:
        info.notes.append("Duży (>50GB)")

    return info


def scan_models_directory(
    root_dir: str,
    max_depth: int = 3,
    recursive: bool = True,
) -> list[ModelInfo]:
    """Skana katalog z modelami.

    Args:
        root_dir: Katalog główny do skanowania
        max_depth: Maksymalna głębokość rekurencji
        recursive: Czy skanować rekurencyjnie

    Returns:
        Lista ModelInfo dla wykrytych modeli
    """
    models = []
    root = Path(root_dir)

    if not root.exists():
        return models

    # Zbierz wszystkie potencjalne modele
    folders = []
    for f in root.iterdir():
        try:
            if not f.is_dir():
                continue
            folders.append(f)
            if recursive:
                for sub in f.rglob("*"):
                    if sub.is_dir():
                        depth = len(sub.relative_to(f).parts)
                        if depth < max_depth:
                            folders.append(sub)
        except PermissionError:
            continue

    # Skanuj każdy folder
    for folder in folders:
        try:
            info = scan_model_folder(str(folder))
            if info:
                models.append(info)
        except Exception as e:
            # Kontynuuj pomijając błędne foldery
            continue

    # Sortuj: po nazwie, potem rozmiar
    models.sort(key=lambda m: (m.name.lower(), m.size_bytes))

    return models


def get_model_count(root_dir: str) -> int:
    """Szybki licznik modeli (rekurencyjnie, bez pełnego skanowania)."""
    root = Path(root_dir)
    if not root.exists():
        return 0

    # Znajdź wszystkie katalogi zawierające pliki modeli
    model_dirs = set()
    for pattern in ["**/config.json", "**/*.safetensors", "**/*.gguf", "**/*.bin"]:
        for item in root.glob(pattern):
            if item.is_file():
                model_dirs.add(item.parent)

    return len(model_dirs)
