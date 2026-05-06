# vLLM Launcher - TODO List

## Stan na 2025-05-06

### ✅ ZROBIONE:
1. [x] Naprawiono wykrywanie binariów vLLM/sglang
2. [x] Czyszczenie logów - przycisk działa
3. [x] CPU/GPU monitor - naprawiono aktualizację
4. [x] Nowe parametry vLLM - api_key, trust_remote_code, ssl, itp.
5. [x] Nowe parametry sglang - load_format, quantization, context_length, itp.
6. [x] ModelInfo.model_path - property zwraca proper ścieżkę dla GGUF
7. [x] Zapisywanie gguf_file w scannerze - `info.gguf_file` poprawnie ustawiany
8. [x] Użycie info.model_path w uruchamianiu - vLLM i sglang
9. [x] Dodano obsługę llama.cpp (wersja podstawowa)
10. [x] Naprawiono _show_launch_params_dialog
11. [x] Wykrywanie llama.cpp (plik lub folder z bin/)
12. [x] Git push do GitHub (https://github.com/michalwlasiuk-eng/vllm-launcher.git)

---

### ❌ BŁĘDY / DO NAPRAWY:

#### 1. `model_scanner.py` - gguf_file = None ❌ PRIORYTET
**Problem**: Scanner nie przypisuje `gguf_file` do `ModelInfo`
**Lokalizacja**: `backend/model_scanner.py` linia ~325
**Objawy**: 
```
[DEBUG scan_model_folder] folder=/media/black/modele/llama, has_gguf=True, gguf_file_path=None
[DEBUG set_model_info] name=llama, path=/media/black/modele/llama, gguf_file=None
```
**Przyczyna**: Zły wcięcia kodu - logika przypisania `gguf_file_path` jest poza pętlą `for f in folder.rglob("*"):`

**Do naprawy**:
```python
# Właściwe położenie (PO pętli zbierania plików):
    # Zbierz pliki
    files = []
    for f in folder.rglob("*"):
        if f.is_file():
            files.append(f.name)
            if len(files) > 50:  # limit
                break
    
    # GGUF: znajdź plik .gguf - TERAZ POPRAWNIE
    gguf_file_path = None
    if has_gguf:
        gguf_list = list(folder.glob("*.gguf"))
        if gguf_list:
            gguf_file_path = str(gguf_list[0])

    # Utwórz ModelInfo
    info = ModelInfo(...)
    # Przypisz plik GGUF
    info.gguf_file = gguf_file_path  # TO MUSI BYĆ!
```

#### 2. `_launch_llama()` - nie przekazuje parametrów ❌ PRIORYTET
**Problem**: Funkcja buduje komendę ale NIE dodaje większości parametrów z config
**Lokalizacja**: `gui/main_window.py` linia ~882-1003
**Objawy**: 
```
[DEBUG _launch_llama] CMD: /media/black/ext4/buun-llama-cpp/build/bin/llama-server -m /media/black/modele/llama/p-e-w_Llama-3.1-8B-Instruct-heretic-IQ4_XS.gguf --host 127.0.0.1 --port 8080
```
**Brakuje**: `-t`, `-ngl`, `-c`, `-n`, `-fa`, `-ctk`, `-ctv`, `--fit`, `--rope-scaling`, itp.

**Parametry z configa które NIE są przekazywane**:
- `threads` (-t), `threads_batch` (-tb)
- `batch_size` (-b), `ubatch_size` (-ub)
- `gpu_layers` (-ngl), `main_gpu` (-mg)
- `n_predict` (-n), `ctx_size` (-c)
- `flash_attn` (-fa), `cache_type_k` (-ctk), `cache_type_v` (-ctv)
- `rope_scaling`, `rope_freq_base`, `rope_scale`
- `tensor_split` (-ts), `split_mode` (-sm)
- `mlock`, `mmap`, `no_direct_io`, `numa`
- `lora`, `control_vector`, `keep`
- `verbose`, `log_disable`, `log_file`, `no_escape`

#### 3. Brak zakładek z parametrami llama.cpp (⚙️ llama)
**Problem**: Dialog "⚙️ llama" ma tylko podstawowe parametry
**Oczekiwania użytkownika** (według `przyklady/lau.py`):
- **Model i Serwer**: model, ctx_size, fit, n_predict, chat_template, seed
- **Wydajność**: threads, threads_batch, batch_size, ubatch_size, flash_attn, poll, cont_batching, mmap, mlock, cache_prompt, prio, cpu_moe
- **GPU i Pamięć**: n_gpu_layers, tensor_split, split_mode, main_gpu, device, mlock
- **Sieć**: host, port, timeout, threads_http, context_shift, no_webui, metrics
- **Zaawansowane**: rope_scaling, rope_freq_base, embedding, reasoning, lora, mmproj, override_kv, log_verbosity

---

### 📁 STRUKTURA PLIKÓW:

```
vllm-launcher/                    🔗 Repozytorium GitHub
├── gui/
│   ├── main_window.py          ✅ Działa (vLLM, sglang, llama.cpp)
│   ├── main_window_broken.py  ❌ Uszkodzony oryginał (zapas)
│   ├── gpu_widget.py
│   ├── model_test_widget.py
│   └── styles.py
├── backend/
│   ├── config.py                ✅ Zawiera sekcję llama (wszystkie parametry)
│   ├── model_scanner.py         ❌ Błąd: gguf_file=None
│   ├── model_tester.py
│   └── gpu_monitor.py
├── przyklady/                   📁 Przykładowe implementacje
│   ├── lau.py                 💡 Wzorzec z pełnymi parametrami llama.cpp
│   ├── AGENTS.md
│   └── evergreen/GGUF_Launcher/
│       └── server_launcher.py     💡 Wzorzec zakładek
├── config/                       📁 (pusty katalog)
├── assets/
├── main.py
├── config.yaml                   📁 Generowany automatycznie
├── llama_help.txt
├── vllm_help.txt
├── sglang_help.txt
└── TODO.md                      📝 Ten plik
```

---

### 🚀 URUCHAMIENIE:

```bash
cd /media/black/free/vllm
# Z PyQt6 z pipx:
/home/black/.local/share/pipx/venvs/pyqt6/bin/python main.py

# Lub z systemowego python3:
python3 main.py
```

---

### 📋 OBSŁUGIWANE SILNIKI:

1. **vLLM** - dla modeli safetensors i niektórych GGUF (wymaga bitsandbytes)
2. **sglang** - dla modeli safetensors i GGUF
3. **llama.cpp** - dla modeli GGUF (zalecany dla starszych kart, niższe wymagania pamięci)

---

### ⚙️ KONFIGURACJA LLAMA.CPP:

**Ścieżka domyślna**: `/media/black/ext4/llama-cpp-turboquant/build/bin/llama-server`

**Parametry podstawowe (już w dialogu)**:
- Host/Port
- Threads (-t)
- Context (-c)
- GPU Layers (-ngl)
- N Predict (-n)
- Flash Attention (-fa)
- Cache K/V (-ctk, -ctv)
- Verbose (-v)

**Parametry do dodania**:
- fit (--fit), fit_target (--fit-target), fit_ctx (--fit-ctx)
- chat_template (--chat-template), seed (-s)
- threads_batch (-tb), batch_size (-b), ubatch_size (-ub)
- poll (--poll), cont_batching (--cont-batching)
- mmap (--mmap), mlock (--mlock), kv_offload (--no-kv-offload)
- numa (--numa), rope_scaling (--rope-scaling), rope_freq_base (--rope-freq-base)
- rope_scale (--rope-scale), tensor_split (--tensor-split), split_mode (--split-mode)
- main_gpu (-mg), timeout (--timeout), threads_http (--threads-http)
- context_shift (--context-shift), no_webui (--no-webui), metrics (--metrics)
- alias (--alias), jinja (--jinja), embedding (--embedding)
- reasoning (--reasoning), reasoning_budget (--reasoning-budget)
- mmproj (--mmproj), model_draft (--model-draft), gpu_layers_draft (--gpu-layers-draft)
- draft_max (--draft), lora (--lora), override_kv (--override-kv)
- control_vector (--control-vector), keep (--keep)
- log_verbosity (-lv), log_file (--log-file), log_disable (--log-disable)
- extra_flags

---

### 📋 ODNIESIENIE DO PRZYKŁADÓW:

Parametry wzorowane na: `przyklady/lau.py` (FLAG_DEFS)
Tablice: Model i Serwer, Wydajność, GPU i Pamięć, Sieć, Zaawansowane

---

### 🔧 NASTĘPNE KROKI (Priorytet):

#### 1. Naprawić `model_scanner.py` (gguf_file) - 15 min
- [ ] Przenieść logikę `gguf_file_path` POZA pętlę zbierania plików
- [ ] Upewnij się że `info.gguf_file = gguf_file_path` jest wykonywane
- [ ] Test: `python3 -c "from backend import model_scanner; info = model_scanner.scan_model_folder('/media/black/modele/llama'); print(info.gguf_file)"`

#### 2. Naprawić `_launch_llama()` (przekazywanie parametrów) - 30 min
- [ ] Dodaj WSZYSTKIE parametry z config do `cmd_parts` w `_launch_llama()`
- [ ] Sprawdź czy parametry "non-default" są dodawane (np. tylko gdy `threads != -1`)
- [ ] Test: Uruchom model GGUF przez "🚀 Start llama.cpp"

#### 3. Rozbudować dialog parametrów llama.cpp - 45 min
- [ ] Dodaj 5 zakładek według wzorca `lau.py`:
  - Model i Serwer
  - Wydajność  
  - GPU i Pamięć
  - Sieć
  - Zaawansowane
- [ ] Połącz zakładki z `_show_launch_params_dialog()`
- [ ] Test: "⚙️ llama" → sprawdź czy wszystkie parametry są widoczne

#### 4. Testy całości - 15 min
- [ ] Test vLLM ze modelem safetensors
- [ ] Test sglang ze modelem safetensors
- [ ] Test llama.cpp z modelem GGUF (/media/black/modele/llama/)
- [ ] Sprawdź czy parametry są prawidłowo przekazywane (podgląd komendy przed uruchomieniem)

#### 5. Wyczyścić kod - 10 min
- [ ] Usunąć debug logi (`[DEBUG ...]`) z kodu
- [ ] Zaktualizować `README.md` dla nowych użytkowników
- [ ] Commit i push do GitHub

---

### 🏷️ KONTEKST UŻYTKOWNIKA:

- **Sprzęt**: 2× RTX 3060 (12GB każda), i7 4770K (4 cores/8 threads), 32GB RAM
- **System**: Linux Mint
- **Pakiety**: ~560GB modeli AI (Qwen, DeepSeek, Llama, itp.)
- **Preferencje**: 
  - GUI z obsługą myszki
  - Monitorowanie GPU (temperatura, VRAM, obciążenie)
  - Praca głównie z modelami Qwen (30B A3B, Qwen3.6 30B A3B)
  - Jeśli brak bitsandbytes → używać llama.cpp dla GGUF

---

**OSTATNIA AKTUALIZACJA**: 2025-05-06 20:15
**STATUS**: 3 główne błędy do naprawy (gguf_file, params, zakładki)