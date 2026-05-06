# AGENTS.md

## Project Type
- llama.cpp Server Launcher project with Python GUI and C# service
- Python GUI: tkinter-based launcher for llama.cpp server
- C# service: GGUF (GPT-Generated Unified Format) model file parser

## Entry Points
- `lau.py` - Main GUI application (run with `python lau.py`)
- `GgufService.cs` - C# class for GGUF parsing (integrated into launcher)

## Key Components
- `lau.py` - 82462-byte Python script providing tkinter GUI for configuring and launching llama-server
- `GgufService.cs` - C# service for parsing GGUF model files to extract metadata, architecture info, tensor data, and chat templates
- `profiles.json` - Configuration profiles storage for saving/loading launcher settings

## Configuration
- GPU management settings in GUI
- Memory settings (GPU memory usage)
- Model loading parameters
- llama.cpp server configuration with various performance parameters
- Chat template selection and model metadata display

## Architecture Notes
- Python GUI controls llama.cpp server launch via subprocess or direct execution
- C# service provides GGUF parsing functionality for model metadata extraction
- Configuration profiles allow saving/loading different server setups
- Project combines Python GUI with C# service components

## Development Notes
- GUI uses tkinter for interface
- Profile system allows persistence of configuration across sessions
- GGUF parsing extracts detailed model information including architecture, tensors, and templates