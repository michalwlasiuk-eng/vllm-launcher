import struct
import os
from typing import Dict, List, Any, Optional, BinaryIO
from io import BytesIO


class GGUFParser:
    """Parser for GGUF (GPT-Generated Unified Format) model files"""

    GGUF_MAGIC = 0x46554747  # "GGUF"

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.metadata: Dict[str, Any] = {}
        self.architecture: Dict[str, Any] = {}
        self.tensor_names: List[str] = []
        self.tensor_shapes: List[tuple] = []
        self.chat_template: Optional[str] = None

    def parse(self) -> Dict[str, Any]:
        """Parse GGUF file and extract metadata"""
        with open(self.filepath, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]

            if magic != self.GGUF_MAGIC:
                raise ValueError(f"Not a valid GGUF file. Magic: {hex(magic)}")

            version = struct.unpack("<I", f.read(4))[0]
            self._parse_header(f, version)

        return self._get_parsed_data()

    def _parse_header(self, f: BinaryIO, version: int) -> None:
        """Parse GGUF header section"""
        kv_count = struct.unpack("<Q", f.read(8))[0]

        for _ in range(kv_count):
            key_len = struct.unpack("<Q", f.read(8))[0]
            key = f.read(key_len).decode("utf-8")

            (kind,) = struct.unpack("<I", f.read(4))

            if kind == 0:  # uint8
                value = struct.unpack("<B", f.read(1))[0]
            elif kind == 1:  # int8
                value = struct.unpack("<b", f.read(1))[0]
            elif kind == 2:  # uint16
                value = struct.unpack("<H", f.read(2))[0]
            elif kind == 3:  # int16
                value = struct.unpack("<h", f.read(2))[0]
            elif kind == 4:  # uint32
                value = struct.unpack("<I", f.read(4))[0]
            elif kind == 5:  # int32
                value = struct.unpack("<i", f.read(4))[0]
            elif kind == 6:  # float32
                value = struct.unpack("<f", f.read(4))[0]
            elif kind == 7:  # bool
                value = struct.unpack("<?", f.read(1))[0]
            elif kind == 8:  # string
                str_len = struct.unpack("<Q", f.read(8))[0]
                value = f.read(str_len).decode("utf-8")
            elif kind == 9:  # array
                arr_type, arr_len = struct.unpack("<II", f.read(8))
                value = []
                for _ in range(arr_len):
                    if arr_type == 0:
                        value.append(struct.unpack("<B", f.read(1))[0])
                    elif arr_type == 1:
                        value.append(struct.unpack("<b", f.read(1))[0])
                    elif arr_type == 2:
                        value.append(struct.unpack("<H", f.read(2))[0])
                    elif arr_type == 3:
                        value.append(struct.unpack("<h", f.read(2))[0])
                    elif arr_type == 4:
                        value.append(struct.unpack("<I", f.read(4))[0])
                    elif arr_type == 5:
                        value.append(struct.unpack("<i", f.read(4))[0])
                    elif arr_type == 6:
                        value.append(struct.unpack("<f", f.read(4))[0])
                    elif arr_type == 7:
                        value.append(struct.unpack("<?", f.read(1))[0])
                    elif arr_type == 8:
                        str_len = struct.unpack("<Q", f.read(8))[0]
                        value.append(f.read(str_len).decode("utf-8"))
            elif kind == 10:  # binary
                bin_len = struct.unpack("<Q", f.read(8))[0]
                value = f.read(bin_len)
            else:
                continue

            self._process_key_value(key, value)

    def _process_key_value(self, key: str, value: Any) -> None:
        """Process key-value pair from GGUF metadata"""
        if key.startswith("general."):
            self.metadata[key.split(".", 1)[1]] = value
        elif key.startswith("tokenizer."):
            if key == "tokenizer.chat_template":
                self.chat_template = value
            else:
                self.metadata[key] = value
        elif key.startswith("llama."):
            self.architecture[key.split(".", 1)[1]] = value
        else:
            self.metadata[key] = value

    def _get_parsed_data(self) -> Dict[str, Any]:
        """Return all parsed data"""
        return {
            "metadata": self.metadata,
            "architecture": self.architecture,
            "chat_template": self.chat_template,
            "filepath": self.filepath,
            "filename": os.path.basename(self.filepath),
        }

    def get_model_name(self) -> str:
        """Get model name from metadata"""
        return self.metadata.get("general.name", "Unknown Model")

    def get_model_description(self) -> str:
        """Get model description from metadata"""
        return self.metadata.get("general.description", "")

    def get_architecture_info(self) -> Dict[str, Any]:
        """Get architecture-specific information"""
        return self.architecture

    def get_chat_template(self) -> Optional[str]:
        """Get chat template if available"""
        return self.chat_template
