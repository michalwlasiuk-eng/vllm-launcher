import json
import os
from typing import List, Dict, Any, Optional


class ProfileManager:
    """Manages configuration profiles for GGUF Launcher"""

    def __init__(self, profiles_file: str = "profiles.json"):
        self.profiles_file = profiles_file
        self.profiles = self._load_profiles()

    def _load_profiles(self) -> List[Dict[str, Any]]:
        """Load profiles from JSON file"""
        if os.path.exists(self.profiles_file):
            try:
                with open(self.profiles_file, "r") as f:
                    data = json.load(f)
                    return data.get("profiles", [])
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save_profiles(self) -> None:
        """Save profiles to JSON file"""
        data = {"version": "1.0.0", "profiles": self.profiles}
        with open(self.profiles_file, "w") as f:
            json.dump(data, f, indent=4)

    def add_profile(self, name: str, config: Dict[str, Any]) -> None:
        """Add a new configuration profile"""
        profile = {
            "name": name,
            "config": config,
            "model_path": config.get("model_path", ""),
            "gpu_layers": config.get("gpu_layers", 0),
            "memory_usage": config.get("memory_usage", "auto"),
            "port": config.get("port", 8080),
        }
        self.profiles.append(profile)
        self._save_profiles()

    def get_profile(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a profile by name"""
        for profile in self.profiles:
            if profile["name"] == name:
                return profile["config"]
        return None

    def get_all_profiles(self) -> List[str]:
        """Get all profile names"""
        return [profile["name"] for profile in self.profiles]

    def delete_profile(self, name: str) -> bool:
        """Delete a profile by name"""
        for i, profile in enumerate(self.profiles):
            if profile["name"] == name:
                self.profiles.pop(i)
                self._save_profiles()
                return True
        return False

    def update_profile(self, name: str, config: Dict[str, Any]) -> bool:
        """Update an existing profile"""
        for profile in self.profiles:
            if profile["name"] == name:
                profile["config"] = config
                profile["model_path"] = config.get("model_path", "")
                profile["gpu_layers"] = config.get("gpu_layers", 0)
                profile["memory_usage"] = config.get("memory_usage", "auto")
                profile["port"] = config.get("port", 8080)
                self._save_profiles()
                return True
        return False
