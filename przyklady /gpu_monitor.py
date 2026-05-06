import tkinter as tk
from tkinter import ttk
import subprocess
import re
import threading
import time
from collections import deque


class GPUMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("NVIDIA GPU Monitor")
        self.root.geometry("600x350")

        self.gpu_data = {}
        self.max_history = 60
        self.history = {}

        self.create_widgets()
        self.start_monitoring()

    def create_widgets(self):
        style = ttk.Style()
        style.configure(
            "Green.Horizontal.TProgressbar", foreground="green", background="green"
        )
        style.configure(
            "Yellow.Horizontal.TProgressbar", foreground="yellow", background="yellow"
        )
        style.configure(
            "Red.Horizontal.TProgressbar", foreground="red", background="red"
        )

        self.tree = ttk.Treeview(
            self.root, columns=("gpu_name", "vram", "temp", "fan"), show="headings"
        )
        self.tree.heading("gpu_name", text="GPU")
        self.tree.heading("vram", text="VRAM Usage")
        self.tree.heading("temp", text="Temperature")
        self.tree.heading("fan", text="Fan Speed")
        self.tree.pack(fill=tk.BOTH, expand=True)

    def get_gpu_info(self):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.used,memory.total,temperature.gpu,fan.speed",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            lines = result.stdout.strip().split("\n")
            gpu_info = []
            for line in lines:
                parts = line.split(",")
                gpu_info.append(
                    {
                        "index": int(parts[0].strip()),
                        "name": parts[1].strip(),
                        "vram_used": int(parts[2].strip()),
                        "vram_total": int(parts[3].strip()),
                        "temp": int(parts[4].strip()),
                        "fan": int(parts[5].strip()),
                    }
                )
            return gpu_info
        except Exception as e:
            print(f"Error getting GPU info: {e}")
            return []

    def get_color_for_vram(self, used, total):
        percentage = (used / total) * 100 if total > 0 else 0
        if percentage < 50:
            return "green"
        elif percentage < 80:
            return "yellow"
        else:
            return "red"

    def update_display(self):
        gpu_info = self.get_gpu_info()
        self.tree.delete(*self.tree.get_children())

        for gpu in gpu_info:
            vram_percent = (
                (gpu["vram_used"] / gpu["vram_total"]) * 100
                if gpu["vram_total"] > 0
                else 0
            )
            color = self.get_color_for_vram(gpu["vram_used"], gpu["vram_total"])

            self.tree.insert(
                "",
                "end",
                values=(
                    f"GPU {gpu['index']}: {gpu['name']}",
                    f"{vram_percent:.1f}% ({gpu['vram_used']}MB / {gpu['vram_total']}MB)",
                    f"{gpu['temp']}°C",
                    f"{gpu['fan']}%",
                ),
            )

    def start_monitoring(self):
        self.update_display()
        self.root.after(1000, self.start_monitoring)


if __name__ == "__main__":
    root = tk.Tk()
    app = GPUMonitor(root)
    root.mainloop()
