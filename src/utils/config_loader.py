import yaml
from typing import Dict, Any
from pathlib import Path

class ConfigLoader:
    """Loads and manage configuration files"""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, Any] = {}

    def load_all(self) -> Dict[str, Any]:
        config_files = self.config_dir.glob("*.yaml")

        for config_file in config_files:
            config_name = config_file.stem.replace("_config", "")
            self.configs[config_name] = self.load_yaml(config_file)

        return self.configs
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self.configs

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else: 
                return default
            
        return value

    def load_yaml(self, file_path: Path) -> Dict[str, Any]:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)