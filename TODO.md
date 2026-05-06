# vLLM Launcher - Lista TODO

## Stan na 2025-05-06

### ✅ ZROBIONE:
1. [x] Naprawiono wykrywanie binariów vLLM/sglang
2. [x] Czyszczenie logów - przycisk działa
3. [x] CPU/GPU monitor - naprawiono aktualizację
4. [x] Nowe parametry vLLM - api_key, trust_remote_code, ssl, itp.
5. [x] Nowe parametry sglang - load_format, quantization, context_length, itp.
6. [x] ModelInfo.model_path - property zwraca proper ścieżkę dla GGUF
7. [x] Zapisywanie gguf_file w scannerze
8. [x] Użycie info.model_path w uruchamianiu - vLLM i sglang
9. [x] Dodano obsługę llama.cpp (wersja podstawowa)
10. [x] Naprawiono _show_launch_params_dialog
11. [x] Wykrywanie llama.cpp (plik lub folder z bin/)
12. [x] Git push do GitHub

---

### 🔄 W TRAKCIE / DO ZROBIENIA:

#### 1. Przekazywanie parametrów llama.cpp ❌ PRIORYTET
- **Problem**: Funkcja `_launch_llama` nie przekazuje wszystkich zapisanych parametrów
- **Do zrobienia**: Rozszerzyć `_launch_llama` o wszystkie parametry z config:
  - `threads_batch`, `batch_size`, `ubatch_size`
  - `kv_offload`, `rope_scaling`, `rope_scale`
  - `main_gpu`, `tensor_split`, `split_mode`
  - `mlock`, `mmap`, `no_direct_io`, `numa`
  - `lora`, `control_vector`, `keep`
  - `verbose`, `log_disable`, `log_file`, `escape`

#### 2. Zakładka z zaawansowanymi ustawieniami llama.cpp
- **Lokalizacja**: Obok "Logi" w prawym panelu
- **Zakładki w ramach**:
  - **Model i Serwer**: model, ctx_size, fit, n_predict, chat_template, seed
  - **Wydajność**: threads, threads_batch, batch_size, ubatch_size, flash_attn, poll, cont_batching, mmap, mlock, cache_prompt, prio, cpu_moe
  - **GPU i Pamięć**: n_gpu_layers, tensor_split, split_mode, main_gpu, device, mlock
  - **Sieć**: host, port, timeout, threads_http, no_webui, metrics
  - **Zaawansowane**: rope_scaling, rope_freq_base, embedding, reasoning, lora, mmproj, override_kv, log_verbosity

---

### 📁 STRUKTURA PLIKÓW

```
vllm/
├── gui/
│   ├── main_window.py      ✅ Działa (vLLM, sglang, llama.cpp)
│   └── main_window_broken.py ❌ Uszkodzony oryginał
├── backend/
│   ├── config.py            ✅ Zawiera sekcję llama
│   └── model_scanner.py     ✅ Skaner z GGUF
└── przyklady/               📁 Przykładowe implementacje
    └── lau.py               💡 Wzorzec z pełnymi parametrami llama.cpp
```

---

### 🚀 URUCHOMIENIE

```bash
python3 main.py
```

### ⚙️ KONFIGURACJA LLAMA.CPP

Ścieżka domyślna: `/media/black/ext4/llama-cpp-turboquant/build/bin/llama-server`

Parametry podstawowe (już w dialogu):
- Host/Port
- Threads (-t)
- Context (-c)
- GPU Layers (-ngl)
- N Predict (-n)
- Flash Attention (-fa)
- Cache K/V (-ctk, -ctv)
- Verbose (-v)

Parametry do dodania:
- fit (--fit)
- fit_target (--fit-target)
- fit_ctx (--fit-ctx)
- chat_template (--chat-template)
- seed (-s)
- threads_batch (-tb)
- batch_size (-b)
- ubatch_size (-ub)
- poll (--poll)
- cont_batching (--cont-batching)
- mmap (--mmap)
- mlock (--mlock)
- cache_prompt (--cache-prompt)
- prio (--prio)
- cpu_moe (--cpu-moe)
- n_cpu_moe (--n-cpu-moe)
- device (--device)
- tensor_split (--tensor-split)
- split_mode (--split-mode)
- main_gpu (-mg)
- timeout (--timeout)
- threads_http (--threads-http)
- context_shift (--context-shift)
- no_webui (--no-webui)
- metrics (--metrics)
- alias (--alias)
- jinja (--jinja)
- embedding (--embedding)
- reasoning (--reasoning)
- reasoning_budget (--reasoning-budget)
- mmproj (--mmproj)
- model_draft (--model-draft)
- gpu_layers_draft (--gpu-layers-draft)
- draft_max (--draft)
- lora (--lora)
- override_kv (--override-kv)
- rope_scaling (--rope-scaling)
- rope_freq_base (--rope-freq-base)
- rope_scale (--rope-scale)
- log_verbosity (-lv)
- log_file (--log-file)
- log_disable (--log-disable)
- extra_flags

---

### 📋 ODNIESIENIE DO PRZYKŁADÓW

Parametry wzorowane na: `przyklady/lau.py` (FLAG_DEFS)
Tablice: Model i Serwer, Wydajność, GPU i Pamięć, Sieć, Zaawansowane