# vLLM Manager - Lista TODO

## Stan na 2025-05-06

### ✅ ZROBIONE:
1. [x] Naprawiono wykrywanie binariów vLLM/sglang - program poprawnie lokalizuje pliki wykonywalne
2. [x] Czyszczenie logów - przycisk działa  
3. [x] CPU monitor - naprawiono aktualizację bez znikania
4. [x] Nowe parametry vLLM - api_key, trust_remote_code, chat_template, ssl, itp.
5. [x] Nowe parametry sglang - load_format, quantization, context_length, itp.
6. [x] ModelInfo.model_path - property zwraca proper ścieżkę dla GGUF
7. [x] Zapisywanie gguf_file w scannerze - `info.gguf_file` poprawnie ustawiany
8. [x] Użycie info.model_path w uruchamianiu - zmieniono na `info.model_path` w vLLM i sglang

### 🔄 W TRAKCIE:

#### Obsługa llama.cpp (2025-05-06)
- Plik: `backend/config.py` - dodano sekcję "llama" z parametrami uruchomienia
- Plik: `gui/main_window.py` - dodano:
  - Pole "llama.cpp bin:" w konfiguracji
  - Przycisk "⚙️ Parametry llama"
  - Funkcję `_launch_llama()`
  - Zapis/odczyt binary_path do konfiguracji

#### Parametry llama.cpp do zaimplementowania:
- threads, threads_batch
- ctx_size, gpu_layers (ngl)
- batch_size, ubatch_size, n_predict
- flash_attn, kv_offload
- cache_type_k, cache_type_v (turbo2, turbo3, turbo4)
- rope_scaling, rope_scale
- main_gpu, tensor_split, split_mode
- mlock, mmap, direct_io, numa
- lora, control_vector, keep
- verbose, log_disable, log_file, escape

---

## Jak uruchamiać:
```bash
cd /media/black/free/vllm
python3 main.py
```

## Uwagi:
- vLLM/sglang GGUF - nadal problemy, użytkownik doinstaluje bitsandbytes
- llama.cpp - alternatywne rozwiązanie dla modeli GGUF