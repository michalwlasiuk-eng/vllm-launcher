# GGUF AI Model Launcher

A PyQt5-based GUI application for launching and managing llama.cpp servers with GGUF format AI models.

## Features

- **Model Browser**: Browse and select GGUF model files
- **Profile Management**: Save and load server configurations
- **GGUF Parser**: Extract metadata, architecture info, and chat templates from model files
- **Server Control**: Launch and stop llama.cpp server instances
- **GPU Configuration**: Configure GPU layer offloading
- **Performance Tuning**: Adjust context size, threads, batch size, and generation parameters
- **Real-time Logs**: Monitor server output in real-time

## Requirements

- Python 3.8+
- PyQt5
- psutil
- llama.cpp (llama-server binary)

## Installation

```bash
cd GGUF_Launcher
pip install -r requirements.txt
```

Make sure you have the `llama-server` binary available in your PATH or specify the full path in the configuration.

## Usage

```bash
python main.py
```

### Basic Workflow

1. **Browse for Model**: Click "📁 Browse GGUF" to select a GGUF model file
2. **Configure Settings**: Adjust GPU layers, context size, and other parameters
3. **Save Profile**: Click "➕ New Profile" to save your configuration
4. **Launch Server**: Click "▶️ Launch Server" to start the llama.cpp server
5. **Monitor Logs**: View real-time server output in the logs panel

## Configuration Options

### General
- **Model File**: Path to GGUF model file
- **Port**: Server port (default: 8080)
- **Chat Template**: Enable chat template formatting

### GPU
- **GPU Layers**: Number of layers to offload to GPU (0 = auto)
- **Memory Usage**: Optimization level (Auto/Low/Medium/High)

### Performance
- **Context Size**: Maximum context window (default: 4096)
- **CPU Threads**: Number of CPU threads (default: 4)
- **Batch Size**: Processing batch size (default: 512)

### Advanced
- **Temperature**: Generation temperature (default: 0.8)
- **Top K**: Sampling parameter (default: 40)
- **Top P**: Nucleus sampling parameter (default: 0.9)
- **Repeat Penalty**: Penalty for repeated tokens (default: 1.1)
- **Stop Sequences**: Comma-separated stop tokens

## Profile System

Profiles store all configuration settings and allow you to:
- Quickly switch between different model configurations
- Save optimal settings for each model
- Share configurations with others

## Project Structure

```
GGUF_Launcher/
├── main.py              # Main GUI application
├── profile_manager.py   # Profile management system
├── gguf_parser.py       # GGUF file parser
├── server_launcher.py   # Server process management
├── profiles.json        # Profile storage
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## License

MIT License
