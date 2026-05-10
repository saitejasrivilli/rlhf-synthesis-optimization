import json
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

def get_config():
    config_path = Path(__file__).parent.parent / "config" / "config.json"
    with open(config_path) as f:
        config_dict = json.load(f)
    return Config(config_dict)
