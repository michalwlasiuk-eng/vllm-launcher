import subprocess
import psutil
from typing import Optional, List, Dict, Any
import socket


class ServerLauncher:
    """Launch and manage llama.cpp server processes"""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.port: int = 8080

    def is_port_in_use(self, port: int) -> bool:
        """Check if port is already in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    def find_available_port(
        self, start_port: int = 8080, max_attempts: int = 100
    ) -> int:
        """Find an available port"""
        for port in range(start_port, start_port + max_attempts):
            if not self.is_port_in_use(port):
                return port
        raise RuntimeError("No available ports found")

    def kill_existing_server(self, port: int) -> bool:
        """Kill any existing server on the specified port"""
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info["cmdline"]
                if cmdline and any(
                    "llama-server" in str(c) or "llama.cpp" in str(c) for c in cmdline
                ):
                    proc.terminate()
                    proc.wait(timeout=5)
                    return True
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                pass
        return False

    def build_server_command(self, config: Dict[str, Any]) -> List[str]:
        """Build llama-server command from configuration"""
        cmd = []

        # Server executable path
        server_path = config.get("server_path", "bin/llama-server")
        cmd.append(server_path)

        # Model path
        if "model_path" in config:
            cmd.extend(["-m", config["model_path"]])

        # GPU layers
        gpu_layers = config.get("gpu_layers", 0)
        if gpu_layers:
            cmd.extend(["--n-gpu-layers", str(gpu_layers)])

        # Context size
        ctx_size = config.get("context_size", 4096)
        cmd.extend(["-c", str(ctx_size)])

        # Number of threads
        n_threads = config.get("n_threads", 4)
        cmd.extend(["-t", str(n_threads)])

        # Port
        port = config.get("port", 8080)
        cmd.extend(["--port", str(port)])

        # Batch size
        batch_size = config.get("batch_size", 512)
        cmd.extend(["--batch-size", str(batch_size)])

        # Memory usage optimization
        if config.get("memory_usage") == "high":
            cmd.extend(["--mlock"])

        # Chat template
        if config.get("use_chat_template"):
            cmd.extend(["--chat-template", "chatml"])

        # Additional parameters
        if config.get("temp"):
            cmd.extend(["--temp", str(config["temp"])])

        if config.get("repeat_penalty"):
            cmd.extend(["--repeat-penalty", str(config["repeat_penalty"])])

        if config.get("top_k"):
            cmd.extend(["--top-k", str(config["top_k"])])

        if config.get("top_p"):
            cmd.extend(["--top-p", str(config["top_p"])])

        if config.get("stop_sequences"):
            for stop in config["stop_sequences"].split(","):
                cmd.extend(["--stop", stop.strip()])

        return cmd

    def launch_server(self, config: Dict[str, Any]) -> bool:
        """Launch llama.cpp server with given configuration"""
        try:
            # Find available port if not specified
            if "port" not in config:
                config["port"] = self.find_available_port()

            self.port = config["port"]

            # Kill existing server if running on this port
            self.kill_existing_server(self.port)

            cmd = self.build_server_command(config)

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            return True

        except Exception as e:
            print(f"Failed to launch server: {e}")
            return False

    def stop_server(self) -> bool:
        """Stop the running server"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
                self.process = None
                return True
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process = None
                return True
        return False

    def is_running(self) -> bool:
        """Check if server is currently running"""
        if self.process:
            return self.process.poll() is None
        return False

    def get_server_logs(self, lines: int = 50) -> List[str]:
        """Get last n lines of server output"""
        if self.process and self.process.stdout:
            self.process.stdout.seek(0, 2)
            output = []
            for line in self.process.stdout:
                output.append(line.strip())
            return output[-lines:] if lines else output
        return []
