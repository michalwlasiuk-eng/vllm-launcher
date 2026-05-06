# Project Summary

## Overall Goal
Build a PyQt6 GUI application for managing vLLM and sglang model serving with model scanning, launch controls, GPU monitoring, and testing capabilities.

## Key Knowledge
- User: Michał, works with AI models (Qwen, DeepSeek families), ~560GB models
- Hardware: Dual RTX 3060 12GB, i7 4770K (4c/8t), 32GB RAM, Linux Mint
- GUI: PyQt6 installed system-wide (pip install --break-system-packages), not in venv
- vLLM venv: /media/black/ext4/venv/inerence (binary: vllm)
- sglang venv: /media/black/ext4/sglang-env (binary: sglang)
- Models directory: /media/black/free/train (20 models detected)
- Config stored at: ~/.config/vllm-manager/config.yaml
- Helper files: vllm_help.txt, sglang_help.txt in project root
- Language: Always respond in Polish unless requested otherwise
- Work style: Step-by-step with TODO list, user approves each step before proceeding
- 5 web fetch limit for research, prefer curl fallback

## Recent Actions
1. **Project structure setup** - Created directory structure with gui/, backend/, config/, assets/ directories
2. **PyQt6 dark theme** - Implemented comprehensive stylesheet with 180 lines CSS for PyQt6
3. **Configuration manager** - Created YAML config manager with auto-detection of vLLM/sglang venvs
4. **Model scanner** - Built GGUF/safetensors detection with metadata extraction, QThread background processing
5. **UI enhancements** - Added filter system (GGUF, safetensors, vLLM, sglang, All), context menu with Launch vLLM/sglang options
6. **Bug fixes** - Fixed QHeaderView import, CustomStatusBar, Qt import, ModelInfo engine_support argument, status bar initialization order
7. **Permission handling** - Fixed PermissionError issues in both `get_model_count` and `scan_models_directory`
8. **Table clearing** - Fixed model table not clearing before scanning, added proper filtering that preserves original data
9. **Recursive scanning** - Updated `get_model_count` to recursively scan directories for accurate model counts
10. **Process launching** - Implemented `_launch_vllm` and `_launch_sglang` with QProcess-based execution
11. **GUI controls** - Added start/stop buttons in model details tab for testing launch functionality
12. **GPU monitoring** - Added real-time GPU/CPU metrics with pynvml/psutil, automatic refresh every 2s
13. **Model testing** - Added OpenAI-compatible API testing with configurable endpoint, prompt, max_tokens, temperature

## Current Plan
1. [DONE] Project skeleton: structure, main.py, requirements.txt, dark theme
2. [DONE] Configuration: vLLM/sglang venv paths, auto-detection, YAML config
3. [DONE] Model scanning: GGUF/safetensors detection, filters, QThread background, metadata
4. [DONE] vLLM launch panel: generate command from vllm_help.txt, start/stop, live logs
5. [DONE] sglang launch panel: generate command from sglang_help.txt, start/stop, live logs
6. [DONE] GPU dashboard: VRAM/CPU usage monitoring using pynvml/psutil
7. [DONE] Model testing: prompt via OpenAI-compatible API, display results

---

## Summary Metadata
**Update time**: 2026-05-05T00:00:00.000Z 
