"""Server configuration."""

import json
import os
from pathlib import Path
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    """Load configuration from config.json."""
    config_path = Path(__file__).parent.parent.parent / "config.json"
    default_config = {
        "paths": {
            "hf_home": None,
            "python_venv": "./venv",
            "ai_media": str(Path(__file__).parent.parent.parent),
            "ffmpeg": "ffmpeg",
            "media_output": "output",
        },
        "server": {
            "host": "127.0.0.1",
            "port": 8000,
        },
        "client": {
            "host": "127.0.0.1",
            "port": 5173,
        },
        "preferences": {
            "theme": "dark",
        }
    }
    
    if config_path.exists():
        try:
            with open(config_path) as f:
                user_config = json.load(f)
                # Deep merge
                for key in default_config:
                    if key in user_config:
                        if isinstance(default_config[key], dict):
                            default_config[key].update(user_config[key])
                        else:
                            default_config[key] = user_config[key]
        except Exception as e:
            print(f"⚠️ Error loading config.json: {e}, using defaults")
    
    return default_config


# Load config on import
CONFIG = load_config()

# Set HF Home if configured
if CONFIG["paths"]["hf_home"]:
    os.environ["HF_HOME"] = CONFIG["paths"]["hf_home"]
