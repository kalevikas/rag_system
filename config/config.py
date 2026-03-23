"""
Unified Configuration Manager
Loads settings from config.yaml and provides a clean interface
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Configuration management class"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config from YAML file
        
        Args:
            config_path: Path to config.yaml (defaults to config/config.yaml)
        """
        if config_path is None:
            # Find config.yaml relative to this file
            root_dir = Path(__file__).parent.parent
            config_path = root_dir / "config" / "config.yaml"
        
        self.config_path = str(config_path)
        self.root_dir = str(Path(self.config_path).parent.parent)
        
        # Load YAML
        with open(self.config_path, 'r') as f:
            self.data = yaml.safe_load(f) or {}
        
        # Set attributes from config
        self._load_attributes()
    
    def _load_attributes(self):
        """Convert config dict to object attributes"""
        for key, value in self.data.items():
            if isinstance(value, dict):
                setattr(self, key, DotDict(value))
            else:
                setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default fallback"""
        keys = key.split('.')
        value = self.data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def get_pdf_dir(self) -> str:
        """Get PDF directory path (absolute)"""
        pdf_dir = self.ingest.pdf_dir
        if os.path.isabs(pdf_dir):
            return pdf_dir
        return os.path.join(self.root_dir, pdf_dir)
    
    def get_data_dir(self) -> str:
        """Get data directory path (absolute)"""
        return os.path.join(self.root_dir, "data")
    
    def get_logs_dir(self) -> str:
        """Get logs directory path (absolute)"""
        return os.path.join(self.root_dir, "logs")


class DotDict(dict):
    """Dictionary that supports dot notation access"""
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"No attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"No attribute '{key}'")


# Global config instance
_config_instance = None

def get_config() -> Config:
    """Get or create global config instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
