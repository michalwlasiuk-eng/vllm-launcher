# vLLM Manager - Lista TODO

## Stan na 2025-05-06

### ✅ ZROBIONE:
1. [x] Naprawiono wykrywanie binariów vLLM/sglang
2. [x] Czyszczenie logów - przycisk działa
3. [x] CPU monitor - naprawiono aktualizację
4. [x] Nowe parametry vLLM - api_key, trust_remote_code, ssl, itp.
5. [x] Nowe parametry sglang - load_format, quantization, context_length, itp.
6. [x] ModelInfo.model_path - property zwraca proper ścieżkę dla GGUF
7. [x] Zapisywanie gguf_file w scannerze - `info.gguf_file` poprawnie ustawiany
8. [x] Użycie info.model_path w uruchamianiu - zmieniono na `info.model_path` w vLLM i sglang
9. [x] Dodano obsługę llama.cpp (wersja podstawowa)

---

### 🔄 STATUS PLIKÓW (2025-05-06):

**`gui/main_window.py`** - ✅ DZIAŁAJĄCA WERSJA
- Zawiera vLLM, sglang i llama.cpp
- Stan: Gotowa do użytku

**`gui/main_window_broken.py`** - ❌ USZKODZONA WERSJA
- Oryginalny plik który został uszkodzony podczas edycji
- NIE UŻYWAĆ

**`backend/config.py`** - ✅ Poprawna konfiguracja z sekcją llama

**`backend/model_scanner.py`** - ✅ Poprawny scanner z obsługą GGUF

---

### 🚀 URUCHOMIENIE:
```bash
cd /media/black/free/vllm
python3 main.py
```

### 📋 OBSŁUGIWANE SILNIKI:
1. **vLLM** - dla modeli safetensors i niektórych GGUF
2. **sglang** - dla modeli safetensors i GGUF
3. **llama.cpp** - dla modeli GGUF (zalecany dla starszych kart)

### ⚙️ KONFIGURACJA LLAMA.CPP:
- Ścieżka do binarium: `/media/black/ext4/llama-cpp-turboquant/build/bin/llama-server`
- Parametry: threads, ctx_size, gpu_layers, cache_type_k/v, itp.

---

### 📌 DO ZROBIENIA:
- [ ] Test uruchomienia llama.cpp z modelem GGUF
- [ ] Rozszerzenie parametrów llama.cpp w dialogu
- [ ] Auto-detect dla llama.cpp binary