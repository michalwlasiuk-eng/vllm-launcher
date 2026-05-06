#!/usr/bin/env python3
"""llama.cpp Server Launcher GUI — configure and launch llama-server with ease."""

import json
import os
import re
import shlex
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    import gguf

    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Flag definitions — each entry drives both the UI and the command builder
# ---------------------------------------------------------------------------


def _find_executable(name: str) -> str | None:
    """Find executable in PATH or common installation directories."""
    # First check PATH
    result = subprocess.run(
        ["which", name],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()

    # Check common llama.cpp locations
    home = Path.home()
    common_paths = [
        "/usr/bin/" + name,
        "/usr/local/bin/" + name,
        "/opt/llama.cpp/" + name,
        "/opt/llm/" + name,
        home / ".local" / "bin" / name,
        home / "llama.cpp" / name,
        home / "llama.cpp" / "build" / "bin" / name,
        home / "llama.cpp" / "build" / "out" / name,
        home / ".cache" / "llama.cpp" / name,
        home / "llama.cpp" / "build" / "bin",
        home / "llama.cpp" / "build" / "out",
    ]

    for path in common_paths:
        path_obj = Path(path)
        if path_obj.is_dir():
            exe = path_obj / name
            if exe.exists() and exe.is_file():
                return str(exe)
        elif path_obj.exists():
            return str(path_obj)

    return None


def _detect_gpus() -> list[dict]:
    """Wykryj karty GPU przez nvidia-smi.
    Zwraca listę: [{"idx": int, "name": str, "total_mib": int, "free_mib": int}]
    """
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode != 0:
            return []
        gpus = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                try:
                    gpus.append(
                        {
                            "idx": int(parts[0]),
                            "name": parts[1],
                            "total_mib": int(parts[2]),
                            "free_mib": int(parts[3]),
                        }
                    )
                except ValueError:
                    pass
        return gpus
    except Exception:
        return []


_CHAT_TEMPLATES = [
    "",
    "bailing", "bailing-think", "chatglm3", "chatglm4", "chatml",
    "command-r", "deepseek", "deepseek2", "deepseek3",
    "exaone3", "exaone4", "falcon3", "gemma", "granite", "granite-4.0",
    "grok-2", "llama2", "llama3", "llama4", "megrez", "minicpm",
    "mistral-v1", "mistral-v3", "mistral-v7", "monarch", "openchat",
    "phi3", "phi4", "rwkv-world", "smolvlm", "vicuna", "zephyr",
]

FLAG_DEFS = [
    # ── Model i Serwer ──────────────────────────────────────────────────────
    {
        "key": "model",
        "label": "Plik modelu",
        "flag": "-m",
        "type": "model_selector",
        "default": "",
        "tip": "Ścieżka do pliku modelu GGUF.",
        "tab": "Model i Serwer",
    },
    {
        "key": "ctx_size",
        "label": "Kontekst",
        "flag": "-c",
        "type": "ctx_selector",
        "default": 0,
        "min": 0,
        "max": 2097152,
        "tip": "Okno kontekstu w tokenach.\n"
               "0 = wczytaj z modelu (zalecane — llama.cpp ustawi max obsługiwany).\n"
               "Większe = więcej RAM/VRAM. Max odczytywany z GGUF po wyborze modelu.",
        "tab": "Model i Serwer",
    },
    {
        "key": "fit",
        "label": "Auto-fit VRAM",
        "flag": "--fit",
        "type": "combobox",
        "default": "on",
        "values": ["on", "off"],
        "tip": "Automatyczne dopasowanie -ngl i -c do dostępnego VRAM.\n"
               "on (domyślnie) = serwer sam zmniejsza warstwy/kontekst jeśli nie mieści się w VRAM.\n"
               "off = uruchom dokładnie z podanymi parametrami (błąd jeśli za mało VRAM).",
        "tab": "Model i Serwer",
    },
    {
        "key": "fit_target",
        "label": "Margines VRAM (MiB)",
        "flag": "--fit-target",
        "type": "entry",
        "default": "",
        "tip": "Margines wolnego VRAM jaki --fit ma zostawić na każdej karcie (w MiB).\n"
               "Domyślnie: 1024 MiB (1 GB) na kartę.\n"
               "Przykład dla 2× RTX 3060: '1024,1024' lub po prostu '1024'.\n"
               "Zmniejsz jeśli chcesz wykorzystać więcej VRAM.",
        "tab": "Model i Serwer",
    },
    {
        "key": "fit_ctx",
        "label": "Min. kontekst (fit)",
        "flag": "--fit-ctx",
        "type": "spinbox",
        "default": 4096,
        "min": 128,
        "max": 131072,
        "tip": "Minimalny rozmiar kontekstu jaki --fit może ustawić.\n"
               "Domyślnie 4096. Serwer nie zmniejszy kontekstu poniżej tej wartości\n"
               "nawet jeśli VRAM jest zbyt mały.",
        "tab": "Model i Serwer",
    },
    {
        "key": "chat_template",
        "label": "Szablon czatu",
        "flag": "--chat-template",
        "type": "combobox",
        "default": "",
        "values": _CHAT_TEMPLATES,
        "tip": "Nadpisz szablon czatu wbudowany w model.\n"
               "Puste = użyj szablonu z pliku GGUF (zalecane).\n"
               "Użyj jeśli model ma błędny/brakujący szablon.\n"
               "Dostępne: llama3, chatml, gemma, phi3, deepseek2, deepseek3, mistral-v7 i inne.",
        "tab": "Model i Serwer",
    },
    {
        "key": "n_predict",
        "label": "Maks. tokenów",
        "flag": "-n",
        "type": "spinbox",
        "default": -1,
        "min": -1,
        "max": 100000,
        "tip": "Maks. tokenów generowanych na jedno zapytanie.\n-1 = bez limitu (zalecane).",
        "tab": "Model i Serwer",
    },
    {
        "key": "seed",
        "label": "Ziarno losowości",
        "flag": "-s",
        "type": "spinbox",
        "default": -1,
        "min": -1,
        "max": 999999999,
        "tip": "Ziarno losowe dla powtarzalnych wyników.\n-1 = losowe przy każdym uruchomieniu.",
        "tab": "Model i Serwer",
    },
    # ── Wydajność ────────────────────────────────────────────────────────────
    {
        "key": "n_gpu_layers",
        "label": "Warstwy GPU (-ngl)",
        "flag": "-ngl",
        "type": "spinbox",
        "default": -1,
        "min": -1,
        "max": 999,
        "tip": "Liczba warstw modelu przeniesionych na GPU.\n"
               "-1 lub 'auto' = serwer sam ustali (domyślne, zalecane z --fit=on).\n"
               "0 = tylko CPU.\n999 = wszystkie warstwy.\n"
               "Przy 2× RTX 3060 (24 GB) większość modeli 30B mieści się w całości.",
        "tab": "Wydajność",
    },
    {
        "key": "cpu_moe",
        "label": "MoE w CPU",
        "flag": "--cpu-moe",
        "type": "checkbox",
        "default": False,
        "tip": "Trzymaj wagi Mixture-of-Experts (MoE) w pamięci CPU zamiast VRAM.\n"
               "WAŻNE dla modeli MoE: DeepSeek, GLM, Qwen-MoE, Mistral-MoE.\n"
               "Zwalnia VRAM na warstwy uwagi — znacznie więcej warstw mieści się na GPU.\n"
               "Wolniejsze od pełnego GPU, ale często jedyna opcja dla dużych MoE.",
        "tab": "Wydajność",
    },
    {
        "key": "n_cpu_moe",
        "label": "Warstwy MoE w CPU",
        "flag": "--n-cpu-moe",
        "type": "spinbox",
        "default": -1,
        "min": -1,
        "max": 999,
        "tip": "Trzymaj wagi MoE pierwszych N warstw w CPU.\n"
               "-1 = wszystkie (jak --cpu-moe).\n"
               "Przydatne gdy chcesz część warstw MoE w GPU, resztę w CPU.",
        "tab": "Wydajność",
    },
    {
        "key": "threads",
        "label": "Wątki CPU",
        "flag": "-t",
        "type": "spinbox",
        "default": -1,
        "min": -1,
        "max": 256,
        "tip": "Wątki CPU do generowania tokenów.\n-1 = automatyczne wykrywanie (zalecane).",
        "tab": "Wydajność",
    },
    {
        "key": "threads_batch",
        "label": "Wątki wsadowe",
        "flag": "-tb",
        "type": "spinbox",
        "default": -1,
        "min": -1,
        "max": 256,
        "tip": "Wątki do przetwarzania promptu (prefill).\n-1 = tyle samo co wątki CPU.",
        "tab": "Wydajność",
    },
    {
        "key": "batch_size",
        "label": "Rozmiar batcha",
        "flag": "-b",
        "type": "spinbox",
        "default": 2048,
        "min": 1,
        "max": 16384,
        "tip": "Logiczny rozmiar wsadu do przetwarzania promptu.\nTypowo: 2048–4096.",
        "tab": "Wydajność",
    },
    {
        "key": "ubatch_size",
        "label": "Micro-batch",
        "flag": "-ub",
        "type": "spinbox",
        "default": 512,
        "min": 1,
        "max": 16384,
        "tip": "Fizyczny rozmiar wsadu dla obliczeń GPU.\nWyższy = krótszy czas do pierwszego tokenu.\nTypowo: 512–2048.",
        "tab": "Wydajność",
    },
    {
        "key": "flash_attn",
        "label": "Flash Attention",
        "flag": "-fa",
        "type": "combobox",
        "default": "auto",
        "values": ["auto", "on", "off"],
        "tip": "Tryb Flash Attention.\nauto = użyj jeśli dostępne (zalecane).\n"
               "Przyspiesza generowanie i zmniejsza zużycie VRAM.",
        "tab": "Wydajność",
    },
    {
        "key": "poll",
        "label": "Polling CPU",
        "flag": "--poll",
        "type": "spinbox",
        "default": 50,
        "min": 0,
        "max": 100,
        "tip": "Poziom aktywnego czekania CPU na pracę (0–100).\n"
               "0 = brak pollingu (mniej CPU w trakcie idle, większe opóźnienie).\n"
               "50 = domyślne.\n100 = maksymalny polling (najmniejsze opóźnienie, więcej CPU).",
        "tab": "Wydajność",
    },
    {
        "key": "cont_batching",
        "label": "Cont. batching",
        "flag": "--cont-batching",
        "type": "checkbox",
        "default": True,
        "tip": "Ciągłe wsadowanie (continuous batching) — domyślnie włączone.\n"
               "Pozwala obsługiwać wiele zapytań jednocześnie bez czekania na ukończenie.\n"
               "Wyłącz tylko przy debugowaniu lub specyficznych problemach.",
        "tab": "Wydajność",
        "flag_off": "--no-cont-batching",
    },
    {
        "key": "mmap",
        "label": "Memory map",
        "flag": "--mmap",
        "type": "checkbox",
        "default": True,
        "tip": "Mapowanie pamięci modelu — szybsze ładowanie (zalecane).",
        "tab": "Wydajność",
        "flag_off": "--no-mmap",
    },
    {
        "key": "mlock",
        "label": "Blokada RAM",
        "flag": "--mlock",
        "type": "checkbox",
        "default": False,
        "tip": "Zablokuj model w RAM — zapobiega stronicowaniu na dysk.\n"
               "Zapewnia stałe opóźnienia. Wymaga wystarczającej wolnej pamięci RAM.",
        "tab": "Wydajność",
    },
    {
        "key": "cache_prompt",
        "label": "Cache promptu",
        "flag": "--cache-prompt",
        "type": "checkbox",
        "default": True,
        "tip": "Ponowne użycie cache KV dla pasujących prefiksów promptu — domyślnie włączony.\n"
               "Znacznie skraca czas do pierwszego tokenu przy powtarzających się promptach.\n"
               "Wyłącz (--no-cache-prompt) tylko przy debugowaniu.",
        "tab": "Wydajność",
        "flag_off": "--no-cache-prompt",
    },
    {
        "key": "prio",
        "label": "Priorytet wątków",
        "flag": "--prio",
        "type": "spinbox",
        "default": 0,
        "min": 0,
        "max": 3,
        "tip": "Priorytet wątków serwera.\n0 = normalny, 1 = średni, 2 = wysoki, 3 = realtime.",
        "tab": "Wydajność",
    },
    # ── GPU i Pamięć ─────────────────────────────────────────────────────────
    {
        "key": "device",
        "label": "Urządzenia GPU",
        "flag": "--device",
        "type": "entry",
        "default": "",
        "tip": "Lista urządzeń do offloadingu (przecinkami).\n"
               "Przykład: 'CUDA0,CUDA1' dla 2× RTX 3060.\n"
               "Puste = serwer wykryje automatycznie.\n"
               "Użyj --list-devices w terminalu aby zobaczyć dostępne urządzenia.",
        "tab": "GPU i Pamięć",
    },
    {
        "key": "tensor_split",
        "label": "Podział VRAM (GPU)",
        "flag": "--tensor-split",
        "type": "tensor_split",
        "default": "",
        "tip": "Proporcje podziału VRAM między kartami GPU.\n"
               "Suwaki ustawiają wagę każdej karty (suma nie musi = 100).\n"
               "llama.cpp normalizuje automatycznie.\n"
               "Np. 50/50 = równy podział, 70/30 = więcej na GPU0.\n"
               "Puste = cały model na jednej karcie.",
        "tab": "GPU i Pamięć",
    },
    {
        "key": "split_mode",
        "label": "Tryb podziału",
        "flag": "--split-mode",
        "type": "combobox",
        "default": "layer",
        "values": ["layer", "row", "tensor", "none"],
        "tip": "Sposób dystrybucji modelu między GPU.\n"
               "layer = kolejne warstwy (domyślny, pipeline).\n"
               "row = podział wierszy tensorów (większa równoległość).\n"
               "tensor = podział wag i KV (eksperymentalny, max równoległość).\n"
               "none = tylko główny GPU.",
        "tab": "GPU i Pamięć",
    },
    {
        "key": "main_gpu",
        "label": "Główny GPU",
        "flag": "--main-gpu",
        "type": "spinbox",
        "default": 0,
        "min": 0,
        "max": 7,
        "tip": "Główny GPU dla tensorów jednokartowych.\nIstotne tylko przy wielu kartach GPU.",
        "tab": "GPU i Pamięć",
    },
    {
        "key": "cache_type_k",
        "label": "Typ cache K",
        "flag": "--cache-type-k",
        "type": "combobox",
        "default": "f16",
        "values": ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
        "tip": "Typ danych dla cache K (klucze uwagi).\n"
               "f16 = pełna precyzja.\nbf16 = podobna jakość do f16, lepsza na nowych GPU.\n"
               "q8_0 = 2× mniejszy VRAM, minimalna strata jakości (dobry wybór).\n"
               "q4_0 = 4× mniejszy VRAM, zauważalna strata.",
        "tab": "GPU i Pamięć",
    },
    {
        "key": "cache_type_v",
        "label": "Typ cache V",
        "flag": "--cache-type-v",
        "type": "combobox",
        "default": "f16",
        "values": ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
        "tip": "Typ danych dla cache V (wartości uwagi).\nTakie same kompromisy jak cache K.",
        "tab": "GPU i Pamięć",
    },
    {
        "key": "kv_offload",
        "label": "KV offload",
        "flag": "--kv-offload",
        "type": "checkbox",
        "default": True,
        "tip": "Offload cache KV na GPU (domyślnie włączony).\n"
               "Wyłącz (--no-kv-offload) aby trzymać cache KV w RAM CPU — oszczędza VRAM\n"
               "kosztem wolniejszego dostępu.",
        "tab": "GPU i Pamięć",
        "flag_off": "--no-kv-offload",
    },
    {
        "key": "override_tensor",
        "label": "Override tensor",
        "flag": "--override-tensor",
        "type": "entry_wide",
        "default": "",
        "tip": "Nadpisz typ bufora dla konkretnych tensorów modelu.\n"
               "Format: 'wzorzec_nazwy=TYP' (przecinkami dla wielu).\n"
               "Przykład: 'blk.*.attn=CUDA0,blk.*.ffn=CPU'\n"
               "Przydatne do manualnego sterowania co trafia na GPU, a co w RAM.",
        "tab": "GPU i Pamięć",
    },
    {
        "key": "swa_full",
        "label": "Pełny cache SWA",
        "flag": "--swa-full",
        "type": "checkbox",
        "default": False,
        "tip": "Użyj cache pełnowymiarowy dla Sliding Window Attention (SWA).\n"
               "Dotyczy modeli z SWA (Gemma3, Mistral v3+).\n"
               "Wyłączone = mniejszy cache (szybciej, mniej VRAM).\n"
               "Włączone = pełny cache (lepsza jakość długich kontekstów).",
        "tab": "GPU i Pamięć",
    },
    {
        "key": "numa",
        "label": "NUMA",
        "flag": "--numa",
        "type": "combobox",
        "default": "",
        "values": ["", "distribute", "isolate", "numactl"],
        "tip": "Optymalizacja NUMA dla procesorów wielogniazdowych.\n"
               "Zostawić puste przy jednym gnieździe CPU.",
        "tab": "GPU i Pamięć",
    },
    {
        "key": "slot_save_path",
        "label": "Ścieżka slotów",
        "flag": "--slot-save-path",
        "type": "entry",
        "default": "",
        "tip": "Katalog do zapisu slotów cache KV na dysk.\nZostawić puste = nie zapisuj.",
        "tab": "GPU i Pamięć",
    },
    # ── Próbkowanie ───────────────────────────────────────────────────────────
    {
        "key": "temperature",
        "label": "Temperatura",
        "flag": "--temp",
        "type": "entry",
        "default": "",
        "tip": "Temperatura próbkowania (domyślnie: 0.80).\n"
               "Niższa (0.1–0.5) = bardziej przewidywalne, deterministyczne odpowiedzi.\n"
               "Wyższa (0.8–1.5) = bardziej kreatywne, różnorodne odpowiedzi.\n"
               "Puste = użyj domyślnej wartości z serwera (0.80).",
        "tab": "Próbkowanie",
    },
    {
        "key": "top_k",
        "label": "Top-K",
        "flag": "--top-k",
        "type": "spinbox",
        "default": 40,
        "min": 0,
        "max": 100000,
        "tip": "Próbkowanie Top-K — ogranicz do K najbardziej prawdopodobnych tokenów.\n"
               "0 = wyłączone. 40 = domyślne.\n"
               "Mniejsze = bardziej skupione odpowiedzi.",
        "tab": "Próbkowanie",
    },
    {
        "key": "top_p",
        "label": "Top-P",
        "flag": "--top-p",
        "type": "entry",
        "default": "",
        "tip": "Próbkowanie Top-P (nucleus) — ogranicz do tokenów których suma prob. = P.\n"
               "1.0 = wyłączone. 0.95 = domyślne.\n"
               "Mniejsze = bardziej skupione, większe = bardziej różnorodne.",
        "tab": "Próbkowanie",
    },
    {
        "key": "min_p",
        "label": "Min-P",
        "flag": "--min-p",
        "type": "entry",
        "default": "",
        "tip": "Próbkowanie Min-P — odrzuć tokeny z prob. < min_p × max_prob.\n"
               "0.0 = wyłączone. 0.05 = domyślne.\n"
               "Dobre uzupełnienie Top-P dla unikania absurdalnych tokenów.",
        "tab": "Próbkowanie",
    },
    {
        "key": "repeat_penalty",
        "label": "Kara za powtórzenia",
        "flag": "--repeat-penalty",
        "type": "entry",
        "default": "",
        "tip": "Kara za powtarzanie tokenów (domyślnie 1.0 = wyłączone).\n"
               "1.1–1.3 = lekkie penalizowanie powtórzeń.\n"
               "Zbyt wysokie wartości mogą pogorszyć jakość.",
        "tab": "Próbkowanie",
    },
    {
        "key": "repeat_last_n",
        "label": "Okno powtórzeń",
        "flag": "--repeat-last-n",
        "type": "spinbox",
        "default": 64,
        "min": -1,
        "max": 4096,
        "tip": "Ile ostatnich tokenów brać pod uwagę przy karze za powtórzenia.\n"
               "0 = wyłączone. -1 = cały kontekst. 64 = domyślne.",
        "tab": "Próbkowanie",
    },
    {
        "key": "dry_multiplier",
        "label": "DRY multiplier",
        "flag": "--dry-multiplier",
        "type": "entry",
        "default": "",
        "tip": "Mnożnik próbkowania DRY (Don't Repeat Yourself).\n"
               "0.0 = wyłączone (domyślne). Wartości > 0 aktywują DRY.\n"
               "DRY silniej karze powtarzanie sekwencji niż klasyczny repeat-penalty.",
        "tab": "Próbkowanie",
    },
    # ── Sieć ────────────────────────────────────────────────────────────────
    {
        "key": "host",
        "label": "Host",
        "flag": "--host",
        "type": "entry",
        "default": "127.0.0.1",
        "tip": "Adres IP do nasłuchiwania.\n"
               "127.0.0.1 = tylko lokalnie.\n0.0.0.0 = wszystkie interfejsy.",
        "tab": "Sieć",
    },
    {
        "key": "port",
        "label": "Port",
        "flag": "--port",
        "type": "spinbox",
        "default": 8080,
        "min": 1,
        "max": 65535,
        "tip": "Port HTTP serwera. opencode domyślnie używa portu 8080.",
        "tab": "Sieć",
    },
    {
        "key": "api_key",
        "label": "Klucz API",
        "flag": "--api-key",
        "type": "entry",
        "default": "",
        "tip": "Wymagaj tego klucza w nagłówku Authorization.\n"
               "Zostawić puste = brak uwierzytelniania.",
        "tab": "Sieć",
    },
    {
        "key": "parallel",
        "label": "Równoległe sloty",
        "flag": "-np",
        "type": "spinbox",
        "default": -1,
        "min": -1,
        "max": 32,
        "tip": "Liczba równoległych slotów zapytań.\n"
               "-1 = auto. 1 = jedno zapytanie na raz (mniej VRAM).\n"
               "Więcej slotów = więcej VRAM na cache KV.",
        "tab": "Sieć",
    },
    {
        "key": "timeout",
        "label": "Timeout (s)",
        "flag": "--timeout",
        "type": "spinbox",
        "default": 600,
        "min": 1,
        "max": 3600,
        "tip": "Limit czasu odczytu/zapisu serwera w sekundach.\n"
               "600 = domyślne (10 minut). Zwiększ dla długich generowań.",
        "tab": "Sieć",
    },
    {
        "key": "threads_http",
        "label": "Wątki HTTP",
        "flag": "--threads-http",
        "type": "spinbox",
        "default": -1,
        "min": -1,
        "max": 64,
        "tip": "Liczba wątków do obsługi żądań HTTP.\n"
               "-1 = automatyczne (zalecane).",
        "tab": "Sieć",
    },
    {
        "key": "context_shift",
        "label": "Przesuwanie kontekstu",
        "flag": "--context-shift",
        "type": "checkbox",
        "default": False,
        "tip": "Włącz automatyczne przesuwanie kontekstu gdy jest pełny.\n"
               "Domyślnie wyłączone — gdy kontekst się zapełni, generowanie się zatrzyma.\n"
               "Włącz dla nieskończonego generowania (kosztem utraty starszego kontekstu).",
        "tab": "Sieć",
    },
    {
        "key": "no_webui",
        "label": "Wyłącz Web UI",
        "flag": "--no-webui",
        "type": "checkbox",
        "default": False,
        "tip": "Wyłącz wbudowany interfejs webowy.\n"
               "Przydatne gdy używasz tylko API (opencode).",
        "tab": "Sieć",
    },
    {
        "key": "metrics",
        "label": "Endpoint metryk",
        "flag": "--metrics",
        "type": "checkbox",
        "default": False,
        "tip": "Włącz endpoint /metrics kompatybilny z Prometheus.",
        "tab": "Sieć",
    },
    # ── Zaawansowane ────────────────────────────────────────────────────────
    {
        "key": "alias",
        "label": "Alias modelu",
        "flag": "--alias",
        "type": "entry",
        "default": "",
        "tip": "Nazwa modelu widoczna przez OpenAI-compatible API.\n"
               "Auto-uzupełniany z metadanych GGUF po wyborze pliku.",
        "tab": "Zaawansowane",
    },
    {
        "key": "jinja",
        "label": "Jinja (wymuś włącz.)",
        "flag": "--jinja",
        "type": "checkbox",
        "default": False,
        "tip": "Wymuś włączenie silnika szablonów Jinja.\n"
               "Jinja jest domyślnie włączone — zaznacz tylko jeśli masz problemy\n"
               "z akceptacją niestandardowych szablonów.",
        "tab": "Zaawansowane",
    },
    {
        "key": "embedding",
        "label": "Osadzenia (embeddings)",
        "flag": "--embedding",
        "type": "checkbox",
        "default": False,
        "tip": "Włącz endpoint /embeddings.\nWymagane dla narzędzi RAG.",
        "tab": "Zaawansowane",
    },
    {
        "key": "reasoning",
        "label": "Tryb rozumowania",
        "flag": "--reasoning",
        "type": "combobox",
        "default": "auto",
        "values": ["auto", "on", "off"],
        "tip": "Włącz/wyłącz tryb rozumowania/myślenia (chain-of-thought).\n"
               "auto = wykryj z szablonu modelu (zalecane).\n"
               "on = wymuś tryb myślenia.\n"
               "off = wyłącz (nie generuj tagu <think>).",
        "tab": "Zaawansowane",
    },
    {
        "key": "reasoning_budget",
        "label": "Budżet rozumowania",
        "flag": "--reasoning-budget",
        "type": "spinbox",
        "default": -1,
        "min": -1,
        "max": 100000,
        "tip": "Maks. tokenów na etap rozumowania.\n"
               "-1 = bez ograniczeń, 0 = wyłącz myślenie.\n"
               "Dla DeepSeek-R1, QwQ, Qwen3 i innych modeli reasoning.",
        "tab": "Zaawansowane",
    },
    {
        "key": "mmproj",
        "label": "Projektor multimodal.",
        "flag": "--mmproj",
        "type": "entry",
        "default": "",
        "tip": "Ścieżka do pliku projektora multimodalnego (vision).\n"
               "Wymagane dla modeli z obsługą obrazów: Gemma3, LLaVA, Qwen-VL itp.\n"
               "Plik zazwyczaj obok pliku modelu, np. mmproj-*.gguf.",
        "tab": "Zaawansowane",
    },
    {
        "key": "model_draft",
        "label": "Model draft (spec.)",
        "flag": "--model-draft",
        "type": "entry",
        "default": "",
        "tip": "Ścieżka do małego modelu draft do spekulatywnego dekodowania.\n"
               "Przyspiesza generowanie o 2–4× bez straty jakości.\n"
               "Model draft musi być kompatybilny (ten sam tokenizer co model główny).",
        "tab": "Zaawansowane",
    },
    {
        "key": "gpu_layers_draft",
        "label": "Warstwy GPU (draft)",
        "flag": "--gpu-layers-draft",
        "type": "spinbox",
        "default": -1,
        "min": -1,
        "max": 999,
        "tip": "Warstwy modelu draft na GPU.\n-1 = auto.",
        "tab": "Zaawansowane",
    },
    {
        "key": "draft_max",
        "label": "Draft tokenów",
        "flag": "--draft",
        "type": "spinbox",
        "default": 16,
        "min": 1,
        "max": 128,
        "tip": "Liczba tokenów do spekulatywnego wygenerowania przez model draft.\n"
               "Wyższe = większy potencjalny zysk, ale też większy koszt przy odrzuceniu.\n"
               "16 = domyślne (dobry kompromis).",
        "tab": "Zaawansowane",
    },
    {
        "key": "lora",
        "label": "Adapter LoRA",
        "flag": "--lora",
        "type": "entry",
        "default": "",
        "tip": "Ścieżka do adaptera LoRA (.gguf).\n"
               "Zostawić puste jeśli nie używasz LoRA.",
        "tab": "Zaawansowane",
    },
    {
        "key": "override_kv",
        "label": "Override KV",
        "flag": "--override-kv",
        "type": "entry_wide",
        "default": "",
        "tip": "Nadpisz metadane modelu kluczem KV.\n"
               "Format: klucz=typ:wartość (przecinkami dla wielu).\n"
               "Przykład: tokenizer.ggml.add_bos_token=bool:false\n"
               "Typy: int, float, bool, str.",
        "tab": "Zaawansowane",
    },
    {
        "key": "rope_scaling",
        "label": "RoPE scaling",
        "flag": "--rope-scaling",
        "type": "combobox",
        "default": "",
        "values": ["", "none", "linear", "yarn"],
        "tip": "Metoda skalowania częstotliwości RoPE.\n"
               "Puste = wartość z modelu.\n"
               "linear = liniowe skalowanie kontekstu.\n"
               "yarn = YaRN (lepsze przy dużych rozszerzeniach kontekstu).",
        "tab": "Zaawansowane",
    },
    {
        "key": "rope_freq_base",
        "label": "RoPE freq base",
        "flag": "--rope-freq-base",
        "type": "entry",
        "default": "",
        "tip": "Podstawa częstotliwości RoPE.\n"
               "Puste = wartość z modelu.\n"
               "Przykład: 1000000 dla Llama-3 przy dużych kontekstach.",
        "tab": "Zaawansowane",
    },
    {
        "key": "log_verbosity",
        "label": "Poziom logowania",
        "flag": "-lv",
        "type": "spinbox",
        "default": 3,
        "min": 0,
        "max": 4,
        "tip": "Szczegółowość logowania:\n0 = ogólne, 1 = błędy, 2 = ostrzeżenia,\n3 = info (domyślne), 4 = debug.",
        "tab": "Zaawansowane",
    },
    {
        "key": "extra_flags",
        "label": "Dodatkowe flagi",
        "flag": "",
        "type": "entry_wide",
        "default": "",
        "tip": "Dodatkowe flagi llama-server nieujęte powyżej, dołączane dosłownie.\n"
               "Przykład: --yarn-orig-ctx 4096 --rope-scale 4",
        "tab": "Zaawansowane",
    },
]

TAB_ORDER = [
    "Model i Serwer",
    "Wydajność",
    "GPU i Pamięć",
    "Próbkowanie",
    "Sieć",
    "Zaawansowane",
    "Metadane GGUF",
]


# Build a reverse lookup: CLI flag string -> FLAG_DEFS key
def _build_flag_map() -> dict[str, str]:
    fm: dict[str, str] = {}
    for fdef in FLAG_DEFS:
        if fdef.get("flag"):
            fm[fdef["flag"]] = fdef["key"]
        if fdef.get("flag_off"):
            fm[fdef["flag_off"]] = fdef["key"]
    return fm


FLAG_MAP = _build_flag_map()

# Set of flags that are boolean (no value argument)
_BOOL_FLAGS: set[str] = set()
for _fd in FLAG_DEFS:
    if _fd["type"] == "checkbox":
        if _fd.get("flag"):
            _BOOL_FLAGS.add(_fd["flag"])
        if _fd.get("flag_off"):
            _BOOL_FLAGS.add(_fd["flag_off"])

DEFAULT_MODEL_DIR = Path.home() / ".local" / "share" / "llama.cpp"
PROFILES_FILE = Path(__file__).with_name("profiles.json")


# ---------------------------------------------------------------------------
# Tooltip helper
# ---------------------------------------------------------------------------
class ToolTip:
    """Hover tooltip for any tkinter widget."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None):
        if self.tip_window:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Segoe UI", 9),
            padx=6,
            pady=4,
        )
        label.pack()

    def _hide(self, _event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


# ---------------------------------------------------------------------------
# Model scanner
# ---------------------------------------------------------------------------
SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def scan_models(directory: Path) -> list[dict]:
    """Return a list of model entries found in *directory*.

    Each entry: {"display": str, "path": str, "parts": int}
    Multi-part shards are grouped; only the first shard is returned with a
    "(N parts)" annotation.
    """
    if not directory.is_dir():
        return []

    gguf_files = sorted(directory.glob("*.gguf"))
    seen_bases: dict[str, dict] = {}
    results: list[dict] = []

    for f in gguf_files:
        m = SHARD_RE.search(f.name)
        if m:
            shard_idx = int(m.group(1))
            total = int(m.group(2))
            base = f.name[: m.start()]
            if base not in seen_bases:
                display = f"{base}  ({total} parts)"
                entry = {"display": display, "path": str(f), "parts": total}
                seen_bases[base] = entry
                results.append(entry)
        else:
            display = f.name
            results.append({"display": display, "path": str(f), "parts": 1})

    return results


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class LlamaLauncher:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("llama.cpp Server Launcher")
        self.root.minsize(700, 600)

        self.process: subprocess.Popen | None = None
        self.output_thread: threading.Thread | None = None
        self.widgets: dict[str, tk.Widget] = {}  # key -> input widget
        self.vars: dict[str, tk.Variable] = {}  # key -> tk variable
        self.model_entries: list[dict] = []

        # GPU info (auto-detected from nvidia-smi at startup)
        self.gpu_info: list[dict] = _detect_gpus()
        # Tensor split per-GPU Scale widgets
        self._ts_scales: list[tk.Scale] = []
        # Token speed tracking
        self.last_token_speed: float = 0.0
        # VRAM monitor timer job id
        self._vram_job = None
        # VRAM bar canvas items per GPU: list of (canvas, bar_id, text_id)
        self._vram_bars: list[tuple] = []

        # Fit-params settings (not in FLAG_DEFS — only used by llama-fit-params)
        self.fit_target_var = tk.IntVar(value=1024)
        self.fit_ctx_var = tk.IntVar(value=4096)

        # Profile state
        self.profiles: dict[str, dict] = {}
        self.active_profile = tk.StringVar(value="Default")
        self._load_profiles()

        self._build_ui()
        self._apply_profile("Default")

    # ── Profiles ────────────────────────────────────────────────────────────

    def _default_values(self) -> dict:
        vals = {"exe_path": "", "fit_target": 1024, "fit_ctx": 4096}
        for f in FLAG_DEFS:
            vals[f["key"]] = f["default"]
        return vals

    def _load_profiles(self):
        if PROFILES_FILE.exists():
            try:
                data = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
                self.profiles = data.get("profiles", {})
                self.active_profile.set(data.get("active", "Default"))
            except (json.JSONDecodeError, KeyError):
                self.profiles = {}
        if "Default" not in self.profiles:
            self.profiles["Default"] = self._default_values()

    def _save_profiles(self):
        data = {"active": self.active_profile.get(), "profiles": self.profiles}
        PROFILES_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _get_current_values(self) -> dict:
        vals: dict = {"exe_path": self.exe_var.get()}
        vals["fit_target"] = self.fit_target_var.get()
        vals["fit_ctx"] = self.fit_ctx_var.get()
        for f in FLAG_DEFS:
            key = f["key"]
            if f["type"] == "checkbox":
                vals[key] = self.vars[key].get()
            elif f["type"] in ("spinbox", "ctx_selector"):
                try:
                    vals[key] = int(self.vars[key].get())
                except ValueError:
                    vals[key] = f["default"]
            else:
                vals[key] = self.vars[key].get()
        return vals

    def _apply_profile(self, name: str):
        vals = self.profiles.get(name, self._default_values())
        self.exe_var.set(vals.get("exe_path", ""))
        self.fit_target_var.set(vals.get("fit_target", 1024))
        self.fit_ctx_var.set(vals.get("fit_ctx", 4096))
        for f in FLAG_DEFS:
            key = f["key"]
            v = vals.get(key, f["default"])
            if f["type"] == "checkbox":
                self.vars[key].set(bool(v))
            elif f["type"] == "model_selector":
                self.vars[key].set(str(v))
                # Update combobox value too
                cb = self.widgets.get(key)
                if cb and isinstance(cb, ttk.Combobox):
                    cb.set(str(v))
            else:
                self.vars[key].set(v)

    def _on_profile_switch(self, _event=None):
        self._apply_profile(self.active_profile.get())

    def _on_save_profile(self):
        name = self.active_profile.get()
        self.profiles[name] = self._get_current_values()
        self._save_profiles()
        self._log(f"Profile '{name}' saved.\n")

    def _on_save_as(self):
        name = tk.simpledialog.askstring(
            "Save Profile As", "Profile name:", parent=self.root
        )
        if not name:
            return
        self.profiles[name] = self._get_current_values()
        self.active_profile.set(name)
        self.profile_combo["values"] = list(self.profiles.keys())
        self.profile_combo.set(name)
        self._save_profiles()
        self._log(f"Profile '{name}' created.\n")

    def _on_delete_profile(self):
        name = self.active_profile.get()
        if name == "Default":
            messagebox.showinfo("Delete Profile", "Cannot delete the Default profile.")
            return
        if not messagebox.askyesno("Delete Profile", f"Delete profile '{name}'?"):
            return
        del self.profiles[name]
        self.active_profile.set("Default")
        self.profile_combo["values"] = list(self.profiles.keys())
        self.profile_combo.set("Default")
        self._apply_profile("Default")
        self._save_profiles()

    # ── UI construction ─────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_profile_bar()
        self._build_exe_bar()
        self._build_tabs()
        self._build_launch_bar()
        self._build_vram_monitor()
        self._build_log_area()

    def _build_profile_bar(self):
        bar = ttk.Frame(self.root, padding=4)
        bar.pack(fill=tk.X)
        ttk.Label(bar, text="Profile:").pack(side=tk.LEFT)
        self.profile_combo = ttk.Combobox(
            bar,
            textvariable=self.active_profile,
            values=list(self.profiles.keys()),
            state="readonly",
            width=24,
        )
        self.profile_combo.pack(side=tk.LEFT, padx=(4, 8))
        self.profile_combo.bind("<<ComboboxSelected>>", self._on_profile_switch)
        ttk.Button(bar, text="Save", command=self._on_save_profile, width=6).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(bar, text="Save As", command=self._on_save_as, width=8).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(bar, text="Delete", command=self._on_delete_profile, width=7).pack(
            side=tk.LEFT, padx=2
        )

    def _build_exe_bar(self):
        bar = ttk.Frame(self.root, padding=4)
        bar.pack(fill=tk.X)
        ttk.Label(bar, text="llama-server:").pack(side=tk.LEFT)
        self.exe_var = tk.StringVar()
        entry = ttk.Entry(bar, textvariable=self.exe_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4))
        ToolTip(entry, "Path to llama-server executable (auto-detected from PATH).")
        ttk.Button(bar, text="Browse...", command=self._browse_exe, width=9).pack(
            side=tk.LEFT
        )

        # Auto-detect on startup
        detected = _find_executable("llama-server")
        if detected:
            self.exe_var.set(detected)

    def _browse_exe(self):
        # Auto-detect llama-server first
        detected = _find_executable("llama-server")
        if detected:
            self.exe_var.set(detected)
            return

        # If not found, let user browse
        path = filedialog.askopenfilename(
            title="Select llama-server executable",
            filetypes=[("Executable", "*"), ("All files", "*.*")],
        )
        if path:
            self.exe_var.set(path)

    def _build_tabs(self):
        """Scrollowalny formularz z sekcjami zamiast zakładek."""
        outer = ttk.Frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._scroll_canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(
            outer, orient=tk.VERTICAL, command=self._scroll_canvas.yview
        )
        self._scroll_canvas.configure(yscrollcommand=vscroll.set)

        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._scroll_inner = ttk.Frame(self._scroll_canvas)
        self._scroll_win = self._scroll_canvas.create_window(
            (0, 0), window=self._scroll_inner, anchor="nw"
        )

        self._scroll_inner.bind(
            "<Configure>",
            lambda e: self._scroll_canvas.configure(
                scrollregion=self._scroll_canvas.bbox("all")
            ),
        )
        self._scroll_canvas.bind(
            "<Configure>",
            lambda e: self._scroll_canvas.itemconfig(
                self._scroll_win, width=e.width
            ),
        )
        # Mouse wheel support
        self._scroll_canvas.bind("<Enter>", self._bind_mousewheel)
        self._scroll_canvas.bind("<Leave>", self._unbind_mousewheel)

        sections: dict[str, ttk.Frame] = {}
        for tab_name in TAB_ORDER:
            lf = ttk.LabelFrame(self._scroll_inner, text=tab_name, padding=(8, 4))
            lf.pack(fill=tk.X, padx=6, pady=(4, 2))
            sections[tab_name] = lf

        for fdef in FLAG_DEFS:
            parent = sections[fdef["tab"]]
            self._build_field(parent, fdef)

        if "Metadane GGUF" in sections:
            self._build_gguf_metadata_tab(sections["Metadane GGUF"])

    def _bind_mousewheel(self, _event=None):
        self._scroll_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self._scroll_canvas.bind_all("<Button-4>", self._on_mousewheel)
        self._scroll_canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event=None):
        self._scroll_canvas.unbind_all("<MouseWheel>")
        self._scroll_canvas.unbind_all("<Button-4>")
        self._scroll_canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        if event.num == 4:
            self._scroll_canvas.yview_scroll(-3, "units")
        elif event.num == 5:
            self._scroll_canvas.yview_scroll(3, "units")
        else:
            self._scroll_canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def _build_gguf_metadata_tab(self, parent: ttk.Frame):
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(
            top_frame,
            text="Wybierz plik GGUF aby zobaczyć metadane i szablon czatu:",
        ).pack(side=tk.LEFT)
        ttk.Button(
            top_frame,
            text="Wybierz plik GGUF…",
            command=self._browse_gguf_file,
        ).pack(side=tk.LEFT, padx=(8, 0))

        if not GGUF_AVAILABLE:
            ttk.Label(
                top_frame,
                text="  Brak biblioteki gguf — zainstaluj: pip install gguf",
                foreground="red",
            ).pack(side=tk.LEFT, padx=(8, 0))

        self.gguf_paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        self.gguf_paned.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(self.gguf_paned)
        right_frame = ttk.Frame(self.gguf_paned)
        self.gguf_paned.add(left_frame, weight=1)
        self.gguf_paned.add(right_frame, weight=1)

        ttk.Label(left_frame, text="Metadane", font=("TkDefaultFont", 9, "bold")).pack(
            anchor=tk.W
        )
        metadata_scroll = ttk.Scrollbar(left_frame)
        metadata_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.metadata_text = tk.Text(
            left_frame, wrap=tk.WORD, height=14, yscrollcommand=metadata_scroll.set
        )
        self.metadata_text.pack(fill=tk.BOTH, expand=True)
        metadata_scroll.config(command=self.metadata_text.yview)

        ttk.Label(
            right_frame, text="Szablon czatu", font=("TkDefaultFont", 9, "bold")
        ).pack(anchor=tk.W)
        template_scroll = ttk.Scrollbar(right_frame)
        template_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.template_text = tk.Text(
            right_frame, wrap=tk.WORD, height=14, yscrollcommand=template_scroll.set
        )
        self.template_text.pack(fill=tk.BOTH, expand=True)
        template_scroll.config(command=self.template_text.yview)

    def _build_field(self, parent: ttk.Frame, fdef: dict):
        key = fdef["key"]
        ftype = fdef["type"]
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)

        if ftype == "model_selector":
            self._build_model_selector(row, fdef)
        elif ftype == "tensor_split":
            self._build_tensor_split_widget(row, fdef)
        elif ftype == "ctx_selector":
            self._build_ctx_field(row, fdef)
        elif ftype == "checkbox":
            var = tk.BooleanVar(value=fdef["default"])
            self.vars[key] = var
            cb = ttk.Checkbutton(row, text=fdef["label"], variable=var)
            cb.pack(side=tk.LEFT)
            self.widgets[key] = cb
            ToolTip(cb, fdef["tip"])
        elif ftype == "spinbox":
            ttk.Label(row, text=fdef["label"] + ":", width=18, anchor=tk.W).pack(
                side=tk.LEFT
            )
            var = tk.StringVar(value=str(fdef["default"]))
            self.vars[key] = var
            sb = ttk.Spinbox(
                row,
                textvariable=var,
                width=10,
                from_=fdef.get("min", 0),
                to=fdef.get("max", 999999),
            )
            sb.pack(side=tk.LEFT, padx=(4, 0))
            self.widgets[key] = sb
            ToolTip(sb, fdef["tip"])
        elif ftype == "combobox":
            ttk.Label(row, text=fdef["label"] + ":", width=18, anchor=tk.W).pack(
                side=tk.LEFT
            )
            var = tk.StringVar(value=fdef["default"])
            self.vars[key] = var
            cb = ttk.Combobox(
                row, textvariable=var, values=fdef["values"], state="readonly", width=14
            )
            cb.pack(side=tk.LEFT, padx=(4, 0))
            self.widgets[key] = cb
            ToolTip(cb, fdef["tip"])
        elif ftype in ("entry", "entry_wide"):
            ttk.Label(row, text=fdef["label"] + ":", width=18, anchor=tk.W).pack(
                side=tk.LEFT
            )
            var = tk.StringVar(value=fdef["default"])
            self.vars[key] = var
            width = 40 if ftype == "entry_wide" else 24
            entry = ttk.Entry(row, textvariable=var, width=width)
            if ftype == "entry_wide":
                entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
            else:
                entry.pack(side=tk.LEFT, padx=(4, 0))
            self.widgets[key] = entry
            ToolTip(entry, fdef["tip"])

    def _build_tensor_split_widget(self, parent: ttk.Frame, fdef: dict):
        """Suwaki podziału VRAM — po jednym na każdy wykryty GPU."""
        key = fdef["key"]
        var = tk.StringVar(value=fdef["default"])
        self.vars[key] = var

        # Nagłówek
        ttk.Label(parent, text=fdef["label"] + ":", width=18, anchor=tk.W).pack(
            side=tk.LEFT, anchor=tk.N, pady=2
        )

        inner = ttk.Frame(parent)
        inner.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._ts_scales = []
        self._ts_scale_vars: list[tk.IntVar] = []

        gpus = self.gpu_info
        if not gpus:
            # Brak NVIDIA — fallback do pola tekstowego
            entry = ttk.Entry(inner, textvariable=var, width=24)
            entry.pack(side=tk.LEFT, padx=(4, 0))
            self.widgets[key] = entry
            ToolTip(entry, fdef["tip"])
            ttk.Label(
                inner,
                text="  (brak NVIDIA GPU — wpisz ręcznie: np. 1,1)",
                foreground="gray",
            ).pack(side=tk.LEFT)
            return

        # Domyślna równa waga dla każdej karty
        default_weight = 100 // len(gpus)

        self._ts_label = ttk.Label(inner, text="", foreground="#888888")

        def _update_ts_label(*_):
            vals = [sv.get() for sv in self._ts_scale_vars]
            total = sum(vals) or 1
            pcts = [f"{v * 100 // total}%" for v in vals]
            var.set(",".join(str(v) for v in vals))
            self._ts_label.configure(
                text="  → --tensor-split " + ",".join(str(v) for v in vals)
                + "   (" + " / ".join(pcts) + ")"
            )

        for gpu in gpus:
            row_g = ttk.Frame(inner)
            row_g.pack(fill=tk.X, pady=1)

            used_mib = gpu["total_mib"] - gpu["free_mib"]
            pct = used_mib * 100 // gpu["total_mib"] if gpu["total_mib"] else 0
            info_txt = (
                f"GPU{gpu['idx']}  {gpu['name']}  "
                f"({gpu['total_mib'] // 1024:.0f} GB)"
            )
            ttk.Label(row_g, text=info_txt, width=36, anchor=tk.W).pack(side=tk.LEFT)

            sv = tk.IntVar(value=default_weight)
            self._ts_scale_vars.append(sv)
            sc = tk.Scale(
                row_g,
                variable=sv,
                from_=0,
                to=100,
                orient=tk.HORIZONTAL,
                length=200,
                command=_update_ts_label,
                showvalue=True,
            )
            sc.pack(side=tk.LEFT, padx=(4, 0))
            self._ts_scales.append(sc)
            ToolTip(sc, fdef["tip"])

        self._ts_label.pack(side=tk.LEFT, anchor=tk.W)
        self.widgets[key] = self._ts_scales[0] if self._ts_scales else None

        # Przycisk odświeżenia GPU info
        def _refresh_gpus():
            self.gpu_info = _detect_gpus()
            # Reload tylko całej sekcji nie jest trivialny — poinformuj użytkownika
            self._log("[VRAM] GPU info odświeżony. Uruchom ponownie aplikację aby zaktualizować suwaki.\n")

        ttk.Button(inner, text="⟳ GPU", command=_refresh_gpus, width=7).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        # Inicjalizacja etykiety
        _update_ts_label()

    def _build_ctx_field(self, parent: ttk.Frame, fdef: dict):
        """Spinbox + przyciski presetów + przycisk Max dla rozmiaru kontekstu."""
        key = fdef["key"]
        self._max_ctx_from_model = 0

        ttk.Label(parent, text=fdef["label"] + ":", width=18, anchor=tk.W).pack(
            side=tk.LEFT
        )
        var = tk.StringVar(value=str(fdef["default"]))
        self.vars[key] = var

        self._ctx_spinbox = ttk.Spinbox(
            parent,
            textvariable=var,
            width=8,
            from_=fdef.get("min", 128),
            to=fdef.get("max", 2097152),
        )
        self._ctx_spinbox.pack(side=tk.LEFT, padx=(4, 6))
        self.widgets[key] = self._ctx_spinbox
        ToolTip(self._ctx_spinbox, fdef["tip"])

        # Preset buttons — typowe rozmiary kontekstu
        CTX_PRESETS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
        for p in CTX_PRESETS:
            label = f"{p // 1024}K"
            btn = ttk.Button(
                parent,
                text=label,
                width=4,
                command=lambda v=p: self.vars["ctx_size"].set(str(v)),
            )
            btn.pack(side=tk.LEFT, padx=1)
            ToolTip(btn, f"Ustaw kontekst na {p:,} tokenów")

        # Przycisk Max — ustawia maksymalny kontekst z metadanych modelu
        self._ctx_max_btn = ttk.Button(
            parent,
            text="Max",
            width=4,
            command=self._set_ctx_max,
        )
        self._ctx_max_btn.pack(side=tk.LEFT, padx=(4, 0))
        ToolTip(
            self._ctx_max_btn,
            "Ustaw maksymalny kontekst obsługiwany przez model\n"
            "(odczytany z metadanych GGUF po wyborze pliku).",
        )

        self._ctx_max_label = ttk.Label(parent, text="", foreground="gray")
        self._ctx_max_label.pack(side=tk.LEFT, padx=(4, 0))

    def _set_ctx_max(self):
        if self._max_ctx_from_model > 0:
            self.vars["ctx_size"].set(str(self._max_ctx_from_model))
        else:
            self._log("Brak danych o maks. kontekście — wybierz plik modelu.\n")

    def _build_model_selector(self, parent: ttk.Frame, fdef: dict):
        key = fdef["key"]
        ttk.Label(parent, text=fdef["label"] + ":", width=18, anchor=tk.W).pack(
            side=tk.LEFT
        )
        var = tk.StringVar(value=fdef["default"])
        self.vars[key] = var

        self.model_entries = scan_models(DEFAULT_MODEL_DIR)
        display_list = [e["display"] for e in self.model_entries]

        combo = ttk.Combobox(parent, textvariable=var, values=display_list, width=50)
        combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4))
        combo.bind("<<ComboboxSelected>>", lambda _e: self._on_model_selected())
        self.widgets[key] = combo
        ToolTip(combo, fdef["tip"])

        ttk.Button(parent, text="Browse...", command=self._browse_model, width=9).pack(
            side=tk.LEFT, padx=(0, 2)
        )
        ttk.Button(parent, text="Refresh", command=self._refresh_models, width=7).pack(
            side=tk.LEFT
        )

    def _on_model_selected(self):
        display = self.vars["model"].get()
        for entry in self.model_entries:
            if entry["display"] == display:
                self.vars["model"].set(entry["path"])
                self._autofill_alias_from_model(entry["path"])
                break

    def _autofill_alias_from_model(self, path: str):
        """Odczytuje metadane GGUF: ustawia alias i maks. kontekst."""
        try:
            metadata, _ = self._parse_gguf_header(path)

            # Auto-alias z general.name
            model_name = metadata.get("general.name", "")
            if model_name and not self.vars.get("alias", tk.StringVar()).get():
                self.vars["alias"].set(model_name)

            # Maks. kontekst z metadanych architektury
            arch = metadata.get("general.architecture", "")
            max_ctx = 0
            for key_suffix in ("context_length",):
                if arch:
                    max_ctx = max_ctx or metadata.get(f"{arch}.{key_suffix}", 0)
                max_ctx = max_ctx or metadata.get(key_suffix, 0)

            if max_ctx and isinstance(max_ctx, (int, float)):
                max_ctx = int(max_ctx)
                self._max_ctx_from_model = max_ctx
                # Aktualizuj górny limit spinboxa
                if hasattr(self, "_ctx_spinbox"):
                    self._ctx_spinbox.configure(to=max_ctx)
                # Pokaż info obok przycisku Max
                if hasattr(self, "_ctx_max_label"):
                    label = (
                        f"max: {max_ctx // 1024}K"
                        if max_ctx >= 1024
                        else f"max: {max_ctx}"
                    )
                    self._ctx_max_label.configure(text=label)
        except Exception:
            pass

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select GGUF model",
            initialdir=str(DEFAULT_MODEL_DIR) if DEFAULT_MODEL_DIR.is_dir() else "",
            filetypes=[("GGUF models", "*.gguf"), ("All files", "*.*")],
        )
        if path:
            self.vars["model"].set(path)

    def _refresh_models(self):
        self.model_entries = scan_models(DEFAULT_MODEL_DIR)
        display_list = [e["display"] for e in self.model_entries]
        combo = self.widgets["model"]
        combo["values"] = display_list
        self._log(f"Found {len(self.model_entries)} model(s) in {DEFAULT_MODEL_DIR}\n")

    def _browse_gguf_file(self):
        path = filedialog.askopenfilename(
            title="Select GGUF file",
            initialdir=str(DEFAULT_MODEL_DIR) if DEFAULT_MODEL_DIR.is_dir() else "",
            filetypes=[("GGUF models", "*.gguf"), ("All files", "*.*")],
        )
        if path:
            self._load_gguf_metadata(path)

    def _parse_gguf_header(self, path: str):
        """Parse GGUF metadata.  Uses gguf.GGUFReader when available (preferred),
        falls back to a pure-Python parser otherwise."""
        if GGUF_AVAILABLE:
            return self._parse_gguf_with_reader(path)
        return self._parse_gguf_custom(path)

    def _parse_gguf_with_reader(self, path: str):
        """Use gguf.GGUFReader — handles all versions and large arrays correctly."""
        import numpy as np
        from gguf import GGUFReader, GGUFValueType

        reader = GGUFReader(path, "r")
        metadata: dict = {}
        chat_template = ""

        for key, field in reader.fields.items():
            if key.startswith("GGUF."):
                continue  # skip internal bookkeeping fields

            types = field.types
            parts = field.parts
            data_indices = field.data
            last_type = types[-1] if types else None

            try:
                if last_type == GGUFValueType.STRING:
                    if types[0] == GGUFValueType.ARRAY:
                        value = [
                            bytes(parts[i]).decode("utf-8", errors="replace")
                            for i in data_indices
                        ]
                    else:
                        value = bytes(parts[data_indices[0]]).decode(
                            "utf-8", errors="replace"
                        )
                else:
                    # Numeric scalar or array
                    all_vals = []
                    for i in data_indices:
                        all_vals.extend(parts[i].tolist())
                    if types[0] == GGUFValueType.ARRAY:
                        value = all_vals
                    else:
                        value = all_vals[0] if all_vals else None
            except Exception:
                continue

            metadata[key] = value
            if "chat_template" in key and isinstance(value, str):
                chat_template = value

        return metadata, chat_template

    def _parse_gguf_custom(self, path: str):
        """Pure-Python fallback parser (no gguf library required)."""
        import struct

        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                raise ValueError("Not a valid GGUF file")
            version = struct.unpack("<I", f.read(4))[0]  # uint32
            _tensor_count = struct.unpack("<Q", f.read(8))[0]
            metadata_kv_count = struct.unpack("<Q", f.read(8))[0]
            metadata_start = f.tell()  # offset 24

        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version: {version}")

        # Read enough data to cover all metadata (up to 32 MB)
        with open(path, "rb") as f:
            f.seek(metadata_start)
            data = f.read(32 * 1024 * 1024)

        metadata: dict = {}
        chat_template = ""
        offset = 0

        for _ in range(metadata_kv_count):
            # Key (uint64 length + bytes) for v3; same for v2
            if offset + 8 > len(data):
                break
            key_len = struct.unpack("<Q", data[offset : offset + 8])[0]
            offset += 8
            if offset + key_len > len(data):
                break
            key = data[offset : offset + key_len].decode("utf-8", errors="replace")
            offset += key_len

            # Value type: uint32 for v3, uint8 for v2
            if version == 3:
                if offset + 4 > len(data):
                    break
                value_type = struct.unpack("<I", data[offset : offset + 4])[0]
                offset += 4
                result = self._read_gguf_value(data, offset, value_type)
            else:
                if offset + 1 > len(data):
                    break
                value_type = data[offset]
                offset += 1
                result = self._read_gguf_value_v2(data, offset, value_type)

            if result is None:
                break
            metadata[key] = result[0]
            if "chat_template" in key and isinstance(result[0], str):
                chat_template = result[0]
            offset += result[1]

        return metadata, chat_template

    def _read_gguf_value(self, data, offset, value_type):
        import struct

        GGUF_VALUE_TYPES = {
            0: "uint8",
            1: "int8",
            2: "uint16",
            3: "int16",
            4: "uint32",
            5: "int32",
            6: "float32",
            7: "bool",
            8: "string",
            9: "array",
            10: "uint64",
            11: "int64",
            12: "float64",
        }

        type_name = GGUF_VALUE_TYPES.get(value_type, "unknown")

        if type_name in ("uint8", "int8"):
            if offset + 1 > len(data):
                return None
            val = struct.unpack(
                "<B" if type_name == "uint8" else "<b", data[offset : offset + 1]
            )[0]
            return (val, 1)

        elif type_name in ("uint16", "int16"):
            if offset + 2 > len(data):
                return None
            val = struct.unpack(
                "<H" if type_name == "uint16" else "<h", data[offset : offset + 2]
            )[0]
            return (val, 2)

        elif type_name in ("uint32", "int32"):
            if offset + 4 > len(data):
                return None
            val = struct.unpack(
                "<I" if type_name == "uint32" else "<i", data[offset : offset + 4]
            )[0]
            return (val, 4)

        elif type_name == "uint64":
            if offset + 8 > len(data):
                return None
            val = struct.unpack("<Q", data[offset : offset + 8])[0]
            return (val, 8)

        elif type_name == "int64":
            if offset + 8 > len(data):
                return None
            val = struct.unpack("<q", data[offset : offset + 8])[0]
            return (val, 8)

        elif type_name == "float32":
            if offset + 4 > len(data):
                return None
            val = struct.unpack("<f", data[offset : offset + 4])[0]
            return (val, 4)

        elif type_name == "float64":
            if offset + 8 > len(data):
                return None
            val = struct.unpack("<d", data[offset : offset + 8])[0]
            return (val, 8)

        elif type_name == "bool":
            if offset + 1 > len(data):
                return None
            val = data[offset] != 0
            return (val, 1)

        elif type_name == "string":
            # GGUF v3 encodes string length as uint64 (8 bytes)
            if offset + 8 > len(data):
                return None
            str_len = struct.unpack("<Q", data[offset : offset + 8])[0]
            offset += 8
            if offset + str_len > len(data):
                return None
            val = data[offset : offset + str_len].decode("utf-8", errors="replace")
            return (val, 8 + str_len)

        elif type_name == "array":
            if offset + 12 > len(data):
                return None
            item_type = struct.unpack("<I", data[offset : offset + 4])[0]
            arr_len = struct.unpack("<Q", data[offset + 4 : offset + 12])[0]
            offset += 12
            total_bytes = 12

            # For fixed-size item types we can calculate the skip size directly
            FIXED_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4,
                           6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
            if item_type in FIXED_SIZES:
                item_sz = FIXED_SIZES[item_type]
                total_item_bytes = arr_len * item_sz
                items = []
                for i in range(min(arr_len, 64)):
                    r = self._read_gguf_value(data, offset + i * item_sz, item_type)
                    if r is None:
                        break
                    items.append(r[0])
                if arr_len > 64:
                    items.append(f"...+{arr_len - 64} more")
                return (items, total_bytes + total_item_bytes)
            else:
                # Variable-size items (string, nested array): must iterate all
                items = []
                for i in range(arr_len):
                    result = self._read_gguf_value(data, offset, item_type)
                    if result is None:
                        break
                    if i < 16:
                        items.append(result[0])
                    elif i == 16:
                        items.append(f"...+{arr_len - 16} more")
                    offset += result[1]
                    total_bytes += result[1]
                return (items, total_bytes)

        return (None, 0)

    def _read_gguf_value_v2(self, data, offset, value_type):
        import struct

        GGUF_V2_TYPES = {
            0: "uint8",
            1: "int8",
            2: "uint16",
            3: "int16",
            4: "uint32",
            5: "int32",
            6: "float32",
            7: "bool",
            8: "string",
            9: "array",
        }

        type_name = GGUF_V2_TYPES.get(value_type, "unknown")

        if type_name in ("uint8", "int8"):
            if offset + 1 > len(data):
                return None
            val = struct.unpack(
                "<B" if type_name == "uint8" else "<b", data[offset : offset + 1]
            )[0]
            return (val, 1)

        elif type_name in ("uint16", "int16"):
            if offset + 2 > len(data):
                return None
            val = struct.unpack(
                "<H" if type_name == "uint16" else "<h", data[offset : offset + 2]
            )[0]
            return (val, 2)

        elif type_name in ("uint32", "int32"):
            if offset + 4 > len(data):
                return None
            val = struct.unpack(
                "<I" if type_name == "uint32" else "<i", data[offset : offset + 4]
            )[0]
            return (val, 4)

        elif type_name == "float32":
            if offset + 4 > len(data):
                return None
            val = struct.unpack("<f", data[offset : offset + 4])[0]
            return (val, 4)

        elif type_name == "bool":
            if offset + 1 > len(data):
                return None
            val = data[offset] != 0
            return (val, 1)

        elif type_name == "string":
            if offset + 4 > len(data):
                return None
            str_len = struct.unpack("<I", data[offset : offset + 4])[0]
            offset += 4
            if offset + str_len > len(data):
                return None
            val = data[offset : offset + str_len].decode("utf-8", errors="replace")
            return (val, 4 + str_len)

        elif type_name == "array":
            if offset + 1 > len(data):
                return None
            item_type = data[offset]
            offset += 1

            if item_type in (4, 5):
                if offset + 4 > len(data):
                    return None
                arr_len = struct.unpack("<I", data[offset : offset + 4])[0]
                offset += 4
            else:
                if offset + 8 > len(data):
                    return None
                arr_len = struct.unpack("<Q", data[offset : offset + 8])[0]
                offset += 8

            items = []
            total_bytes = 1 if item_type in (4, 5) else 9
            for _ in range(arr_len):
                result = self._read_gguf_value_v2(data, offset, item_type)
                if result is None:
                    break
                items.append(result[0])
                offset += result[1]
                total_bytes += result[1]

            return (items, total_bytes)

        return (None, 0)

    def _load_gguf_metadata(self, path: str):
        try:
            metadata, chat_template = self._parse_gguf_header(path)

            # Keys whose values are large bulk arrays — useless to display
            _SKIP_KEYS = {
                "tokenizer.ggml.tokens",
                "tokenizer.ggml.merges",
                "tokenizer.ggml.token_type",
                "tokenizer.ggml.scores",
                "tokenizer.ggml.added_tokens",
            }

            def _skip(key: str, value) -> bool:
                if "chat_template" in key:
                    return True
                if key in _SKIP_KEYS:
                    return True
                # Drop any other large list (>16 items and not strings)
                if isinstance(value, list) and len(value) > 16:
                    return True
                return False

            def format_value(value):
                if isinstance(value, list):
                    if all(isinstance(v, str) for v in value):
                        return ", ".join(value)
                    return ", ".join(str(v) for v in value[:16])
                if isinstance(value, float):
                    return f"{value:.6g}"
                return str(value)

            metadata_text = ""
            for key, value in metadata.items():
                if not _skip(key, value):
                    metadata_text += f"{key}: {format_value(value)}\n"

            if not metadata_text:
                metadata_text = "No metadata found in this file."

            if not chat_template:
                chat_template = "No chat template found in this file."

            self.metadata_text.delete(1.0, tk.END)
            self.metadata_text.insert(tk.END, metadata_text)

            self.template_text.delete(1.0, tk.END)
            self.template_text.insert(tk.END, chat_template)

            # Auto-fill alias from general.name if alias field is empty
            model_name = metadata.get("general.name", "")
            if model_name and not self.vars.get("alias", tk.StringVar()).get():
                self.vars["alias"].set(model_name)

        except Exception as e:
            messagebox.showerror("Error", f"Could not read GGUF file:\n{e}")

    # ── Launch bar ──────────────────────────────────────────────────────────

    def _build_launch_bar(self):
        bar = ttk.Frame(self.root, padding=4)
        bar.pack(fill=tk.X)
        self.launch_btn = ttk.Button(
            bar, text="Launch Server", command=self._launch_server
        )
        self.launch_btn.pack(side=tk.LEFT, padx=(0, 4))
        self.stop_btn = ttk.Button(
            bar, text="Stop Server", command=self._stop_server, state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 12))
        # Separator + Estymator GPU layers
        ttk.Separator(bar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=2
        )
        self.fit_btn = ttk.Button(
            bar, text="Estymuj GPU layers", command=self._on_estimate_gpu_layers
        )
        self.fit_btn.pack(side=tk.LEFT, padx=(0, 8))
        ToolTip(
            self.fit_btn,
            "Automatycznie szacuje liczbę warstw GPU (-ngl) na podstawie:\n"
            "• dostępnego VRAM (nvidia-smi)\n"
            "• rozmiaru pliku modelu i liczby warstw (z metadanych GGUF)\n\n"
            "Wynik to przybliżenie — ustaw nieco niżej jeśli brakuje VRAM.",
        )

        self.status_var = tk.StringVar(value="Stopped")
        self.status_label = ttk.Label(
            bar, textvariable=self.status_var, foreground="gray"
        )
        self.status_label.pack(side=tk.LEFT)

        # Right-side buttons
        copy_api_btn = ttk.Button(
            bar, text="Copy API URL", command=self._copy_api_url, width=14
        )
        copy_api_btn.pack(side=tk.RIGHT, padx=(0, 4))
        ToolTip(
            copy_api_btn,
            "Copy OpenAI-compatible base URL to clipboard.\n"
            "Use this in opencode / other clients:\n"
            "  baseURL: http://host:port/v1",
        )
        ttk.Button(bar, text="Copy Command", command=self._copy_command, width=14).pack(
            side=tk.RIGHT
        )
        self.open_ui_btn = ttk.Button(
            bar,
            text="Open Web UI",
            command=self._open_web_ui,
            width=12,
            state=tk.DISABLED,
        )
        self.open_ui_btn.pack(side=tk.RIGHT, padx=(0, 4))
        ToolTip(self.open_ui_btn, "Open the llama-server web UI in your browser.")

    # ── VRAM Monitor ────────────────────────────────────────────────────────

    def _build_vram_monitor(self):
        """Panel monitorowania VRAM z kolorowymi paskami i prędkością tokenów."""
        gpus = self.gpu_info
        if not gpus:
            return  # brak NVIDIA GPU — nie pokazuj panelu

        frame = ttk.LabelFrame(self.root, text="Monitor VRAM", padding=(6, 4))
        frame.pack(fill=tk.X, padx=4, pady=(0, 2))
        self._vram_frame = frame

        BAR_W = 280
        BAR_H = 16

        self._vram_bars = []
        for gpu in gpus:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=1)

            name_lbl = ttk.Label(
                row,
                text=f"GPU{gpu['idx']} {gpu['name']}",
                width=28,
                anchor=tk.W,
            )
            name_lbl.pack(side=tk.LEFT)

            cv = tk.Canvas(row, width=BAR_W, height=BAR_H, bg="#2a2a2a",
                           highlightthickness=1, highlightbackground="#555555")
            cv.pack(side=tk.LEFT, padx=(4, 6))

            # Tło (całość paska)
            cv.create_rectangle(0, 0, BAR_W, BAR_H, fill="#2a2a2a", outline="")
            # Wypełnienie (dynamiczne)
            bar_id = cv.create_rectangle(0, 0, 0, BAR_H, fill="#4caf50", outline="")
            # Tekst na pasku
            text_id = cv.create_text(
                BAR_W // 2, BAR_H // 2,
                text="", fill="white",
                font=("Consolas", 8, "bold"),
            )
            self._vram_bars.append((cv, bar_id, text_id, BAR_W, gpu["total_mib"]))

        # Linia z prędkością tokenów
        stat_row = ttk.Frame(frame)
        stat_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(stat_row, text="Prędkość generowania:", anchor=tk.W).pack(side=tk.LEFT)
        self._speed_var = tk.StringVar(value="—")
        self._speed_lbl = ttk.Label(
            stat_row,
            textvariable=self._speed_var,
            foreground="#4fc3f7",
            font=("Consolas", 9, "bold"),
        )
        self._speed_lbl.pack(side=tk.LEFT, padx=(6, 20))

        self._vram_status_var = tk.StringVar(value="")
        ttk.Label(
            stat_row,
            textvariable=self._vram_status_var,
            foreground="#888888",
            font=("Consolas", 8),
        ).pack(side=tk.LEFT)

        # Uruchom ciągły monitor (adapter interwału: 2s gdy serwer działa, 8s idle)
        self._schedule_vram_update()

    def _update_vram_bars_once(self):
        """Odśwież paski VRAM jeden raz (wywoływane przez timer i bezpośrednio)."""
        if not self._vram_bars:
            return
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode != 0:
                return
            lines = r.stdout.strip().splitlines()
        except Exception:
            return

        for i, line in enumerate(lines):
            if i >= len(self._vram_bars):
                break
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                used = int(parts[0].strip())
                total = int(parts[1].strip())
            except ValueError:
                continue

            cv, bar_id, text_id, bar_w, _ = self._vram_bars[i]
            pct = used / total if total else 0
            fill_w = int(bar_w * pct)

            if pct < 0.60:
                color = "#4caf50"   # zielony
            elif pct < 0.80:
                color = "#ff9800"   # pomarańczowy/żółty
            else:
                color = "#f44336"   # czerwony

            cv.itemconfigure(bar_id, fill=color)
            cv.coords(bar_id, 0, 0, fill_w, 16)
            cv.itemconfigure(
                text_id,
                text=f"{used} / {total} MiB  ({pct * 100:.0f}%)",
            )

    def _schedule_vram_update(self):
        """Zaplanuj kolejną aktualizację monitora VRAM.
        Gdy serwer działa: co 2 s. Gdy zatrzymany: co 8 s (idle).
        """
        self._update_vram_bars_once()
        interval = 2000 if (self.process and self.process.poll() is None) else 8000
        self._vram_job = self.root.after(interval, self._schedule_vram_update)

    def _stop_vram_monitor(self):
        if self._vram_job is not None:
            self.root.after_cancel(self._vram_job)
            self._vram_job = None

    # ── Log area ────────────────────────────────────────────────────────────

    def _build_log_area(self):
        frame = ttk.LabelFrame(self.root, text="Output Log", padding=4)
        frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        self.log_text = tk.Text(
            frame,
            height=10,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Consolas", 9),
            background="#1e1e1e",
            foreground="#cccccc",
            insertbackground="#cccccc",
        )
        scrollbar = ttk.Scrollbar(
            frame, orient=tk.VERTICAL, command=self.log_text.yview
        )
        self.log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _log(self, text: str):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    # ── Command builder ─────────────────────────────────────────────────────

    def _build_command(self) -> list[str]:
        exe = self.exe_var.get().strip()
        if not exe:
            raise ValueError("llama-server path is not set.")
        cmd = [exe]

        for fdef in FLAG_DEFS:
            key = fdef["key"]
            ftype = fdef["type"]
            flag = fdef["flag"]
            default = fdef["default"]

            if key == "extra_flags":
                continue  # handled separately at the end

            if ftype == "model_selector":
                val = self.vars[key].get().strip()
                if val:
                    cmd += [flag, val]

            elif ftype == "tensor_split":
                val = self.vars[key].get().strip()
                # Emit only if non-empty and not all-zero
                if val and any(v.strip() != "0" for v in val.split(",")):
                    # Normalize: remove leading zeros, keep as-is
                    cmd += [flag, val]

            elif ftype == "checkbox":
                checked = self.vars[key].get()
                if "flag_off" in fdef:
                    # Inverted flag (mmap): checked = default (no flag), unchecked = flag_off
                    if not checked:
                        cmd.append(fdef["flag_off"])
                else:
                    if checked:
                        cmd.append(flag)

            elif ftype in ("spinbox", "ctx_selector"):
                try:
                    val = int(self.vars[key].get())
                except ValueError:
                    continue
                if val != default:
                    cmd += [flag, str(val)]

            elif ftype == "combobox":
                val = self.vars[key].get().strip()
                if val and val != default:
                    cmd += [flag, val]

            elif ftype in ("entry", "entry_wide"):
                val = self.vars[key].get().strip()
                if val and val != default:
                    cmd += [flag, val]

        # Extra flags — split by whitespace and append raw
        extra = self.vars.get("extra_flags")
        if extra:
            extra_text = extra.get().strip()
            if extra_text:
                cmd += extra_text.split()

        return cmd

    def _copy_command(self):
        try:
            cmd = self._build_command()
        except ValueError as e:
            messagebox.showwarning("Command", str(e))
            return
        cmd_str = " ".join(cmd)
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd_str)
        self._log(f"Copied to clipboard:\n{cmd_str}\n\n")

    def _copy_api_url(self):
        host = self.vars["host"].get().strip() or "127.0.0.1"
        if host == "0.0.0.0":
            host = "127.0.0.1"
        try:
            port = int(self.vars["port"].get())
        except ValueError:
            port = 8080
        alias = self.vars.get("alias", tk.StringVar()).get().strip()
        url = f"http://{host}:{port}/v1"
        self.root.clipboard_clear()
        self.root.clipboard_append(url)
        info = f"Copied API URL: {url}"
        if alias:
            info += f"\nModel alias: {alias}"
        self._log(info + "\n")

    def _open_web_ui(self):
        import webbrowser

        host = self.vars["host"].get().strip() or "127.0.0.1"
        if host == "0.0.0.0":
            host = "127.0.0.1"
        try:
            port = int(self.vars["port"].get())
        except ValueError:
            port = 8080
        url = f"http://{host}:{port}"
        webbrowser.open(url)
        self._log(f"Opened {url} in browser\n")

    # ── Process management ──────────────────────────────────────────────────

    def _set_status(self, status: str, color: str = "gray"):
        self.status_var.set(status)
        self.status_label.configure(foreground=color)

    def _set_running_state(self, running: bool):
        if running:
            self.launch_btn.configure(state=tk.DISABLED)
            self.stop_btn.configure(state=tk.NORMAL)
            self.open_ui_btn.configure(state=tk.NORMAL)
        else:
            self.launch_btn.configure(state=tk.NORMAL)
            self.stop_btn.configure(state=tk.DISABLED)
            self.open_ui_btn.configure(state=tk.DISABLED)
            self.last_token_speed = 0.0
            if hasattr(self, "_speed_var"):
                self._speed_var.set("—")
            if hasattr(self, "_speed_lbl"):
                self._speed_lbl.configure(foreground="#4fc3f7")

    def _launch_server(self):
        if self.process and self.process.poll() is None:
            messagebox.showinfo("Launch", "Server is already running.")
            return

        try:
            cmd = self._build_command()
        except ValueError as e:
            messagebox.showwarning("Launch", str(e))
            return

        model_path = self.vars["model"].get().strip()
        if not model_path:
            messagebox.showwarning("Launch", "No model file selected.")
            return

        cmd_str = " ".join(cmd)
        self._log(f"> {cmd_str}\n\n")
        self._set_status("Starting...", "#cc8800")
        self._set_running_state(True)

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW
                if sys.platform == "win32"
                else 0,
                pass_fds=(),
            )
        except FileNotFoundError:
            self._log("ERROR: Could not find the executable. Check the path.\n")
            self._set_status("Error", "red")
            self._set_running_state(False)
            return
        except OSError as e:
            self._log(f"ERROR: {e}\n")
            self._set_status("Error", "red")
            self._set_running_state(False)
            return

        self._set_status("Running", "green")
        self.output_thread = threading.Thread(target=self._read_output, daemon=True)
        self.output_thread.start()

    # Wzorce prędkości tokenów w logach llama-server
    _TOKEN_SPEED_RE = re.compile(
        r"(?:"
        r"n_tokens_second\s*=\s*(\d+\.?\d*)"        # slot 0 : n_tokens_second = 45.23
        r"|(\d+\.?\d+)\s*tok(?:en)?s?/s"            # 45.23 tok/s  lub  tokens/s
        r"|t_token\s*=\s*(\d+\.?\d*)\s*ms"          # t_token = 21.5 ms (→ 1000/ms tok/s)
        r"|predicted_tokens_seconds[\":\s]+(\d+\.?\d*)"  # JSON: predicted_tokens_seconds: 45.23
        r")",
        re.IGNORECASE,
    )

    def _parse_token_speed(self, line: str) -> float | None:
        m = self._TOKEN_SPEED_RE.search(line)
        if not m:
            return None
        if m.group(1):
            return float(m.group(1))
        if m.group(2):
            return float(m.group(2))
        if m.group(3):
            ms = float(m.group(3))
            return 1000.0 / ms if ms > 0 else None
        if m.group(4):
            return float(m.group(4))
        return None

    def _read_output(self):
        proc = self.process
        try:
            for line in proc.stdout:
                self.root.after(0, self._log, line)
                # Parsuj prędkość tokenów
                speed = self._parse_token_speed(line)
                if speed is not None and speed > 0:
                    self.last_token_speed = speed
                    self.root.after(0, self._update_speed_display, speed)
        except Exception:
            pass
        finally:
            ret = proc.wait()
            msg = f"\n[Server exited with code {ret}]\n\n"
            self.root.after(0, self._log, msg)
            if ret == 0:
                self.root.after(0, self._set_status, "Stopped", "gray")
            else:
                self.root.after(0, self._set_status, f"Exited ({ret})", "red")
            self.root.after(0, self._set_running_state, False)

    def _update_speed_display(self, speed: float):
        if not hasattr(self, "_speed_var"):
            return
        self._speed_var.set(f"{speed:.1f} tok/s")
        # Kolor w zależności od prędkości
        if speed >= 15:
            color = "#4caf50"    # zielony — dobra prędkość
        elif speed >= 5:
            color = "#ff9800"    # żółty — umiarkowana
        else:
            color = "#f44336"    # czerwony — wolno
        if hasattr(self, "_speed_lbl"):
            self._speed_lbl.configure(foreground=color)

    def _stop_server(self):
        if self.process and self.process.poll() is None:
            self._log("[Stopping server...]\n")
            self.process.terminate()
            self._set_status("Stopping...", "#cc8800")
        else:
            self._set_status("Stopped", "gray")
            self._set_running_state(False)

    # ── Estymator GPU layers ─────────────────────────────────────────────────

    def _on_estimate_gpu_layers(self):
        """Szacuje -ngl na podstawie VRAM (nvidia-smi) i rozmiaru modelu (GGUF)."""
        model_path = self.vars["model"].get().strip()
        if not model_path or not Path(model_path).exists():
            messagebox.showwarning(
                "Estymuj GPU layers", "Najpierw wybierz plik modelu."
            )
            return

        self.fit_btn.configure(state=tk.DISABLED)
        self._log("--- Estymacja GPU layers ---\n")
        threading.Thread(
            target=self._estimate_gpu_layers_thread,
            args=(model_path,),
            daemon=True,
        ).start()

    def _estimate_gpu_layers_thread(self, model_path: str):
        try:
            # 1. Pobierz dostępny VRAM z nvidia-smi (suma wszystkich kart)
            vram_free_mib = 0
            vram_total_mib = 0
            try:
                r = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.free,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                for line in r.stdout.strip().splitlines():
                    parts = line.split(",")
                    if len(parts) == 2:
                        vram_free_mib += int(parts[0].strip())
                        vram_total_mib += int(parts[1].strip())
            except Exception:
                pass

            # 2. Rozmiar pliku i metadane GGUF
            file_size_mib = Path(model_path).stat().st_size / (1024 * 1024)
            block_count = 0
            try:
                meta, _ = self._parse_gguf_header(model_path)
                arch = meta.get("general.architecture", "")
                for key in (f"{arch}.block_count", "block_count"):
                    v = meta.get(key, 0)
                    if v:
                        block_count = int(v)
                        break
            except Exception:
                pass

            # 3. Oblicz estymację
            lines = []
            lines.append(
                f"Plik modelu: {Path(model_path).name}  ({file_size_mib:.0f} MiB)"
            )

            if block_count > 0:
                # Każda warstwa zajmuje przybliżenie file_size / block_count
                # Dodajemy ~5% na embedding/output/overhead
                mib_per_layer = file_size_mib / block_count
                lines.append(
                    f"Warstwy: {block_count}  (~{mib_per_layer:.0f} MiB/warstwa)"
                )

                if vram_free_mib > 0:
                    lines.append(
                        f"VRAM wolny: {vram_free_mib} MiB  (łącznie: {vram_total_mib} MiB)"
                    )
                    # Zostaw 5% VRAM na overhead + KV cache
                    usable = vram_free_mib * 0.90
                    ngl_est = int(usable / mib_per_layer)
                    ngl_est = max(0, min(ngl_est, block_count))
                    lines.append(
                        f"Szacowane -ngl: {ngl_est}  (przy 90% wolnego VRAM)"
                    )
                    if ngl_est >= block_count:
                        lines.append(
                            "→ Wszystkie warstwy mieszczą się w VRAM — ustaw -ngl = 999"
                        )
                        ngl_est = 999
                    self.root.after(
                        0, self.vars["n_gpu_layers"].set, str(ngl_est)
                    )
                    self.root.after(
                        0,
                        self._log,
                        "\n".join(lines)
                        + f"\n\nUstawiono -ngl = {ngl_est}\n"
                        "Uwaga: to przybliżenie. KV cache też zużywa VRAM.\n"
                        "Obniż o 5-10 warstw jeśli serwer nie startuje.\n\n",
                    )
                else:
                    lines.append(
                        "VRAM: nie wykryto (brak nvidia-smi lub nie-NVIDIA GPU)"
                    )
                    lines.append(
                        f"Plik modelu zajmuje {file_size_mib:.0f} MiB — "
                        f"ustaw -ngl ręcznie."
                    )
                    self.root.after(0, self._log, "\n".join(lines) + "\n\n")
            else:
                lines.append(
                    "Nie można odczytać liczby warstw z metadanych GGUF."
                )
                self.root.after(0, self._log, "\n".join(lines) + "\n\n")

        except Exception as e:
            self.root.after(0, self._log, f"[Błąd estymacji: {e}]\n\n")
        finally:
            self.root.after(
                0, self.fit_btn.configure, {"state": tk.NORMAL}
            )
            self.root.after(
                0, self._log, "--- Estymacja zakończona ---\n\n"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    # simpledialog is needed for Save As — lazy import avoids issues if missing
    import tkinter.simpledialog  # noqa: F401

    root = tk.Tk()
    root.geometry("780x650")
    app = LlamaLauncher(root)

    # Ctrl+Enter to launch
    root.bind("<Control-Return>", lambda _e: app._launch_server())

    root.mainloop()


if __name__ == "__main__":
    main()
