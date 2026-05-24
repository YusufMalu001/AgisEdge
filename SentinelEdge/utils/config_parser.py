import os
import yaml
from utils.logger import logger

class Config:
    def __init__(self, config_dict=None):
        self._config = config_dict or {}

    def get(self, path, default=None):
        keys = path.split('.')
        val = self._config
        for key in keys:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                return default
        return val

    def set(self, path, value):
        keys = path.split('.')
        val = self._config
        for key in keys[:-1]:
            val = val.setdefault(key, {})
        val[keys[-1]] = value

    @property
    def raw(self):
        return self._config

def load_config(config_path="configs/default.yaml"):
    """
    Loads and parses the YAML configuration file.
    """
    if not os.path.exists(config_path):
        # Check relative to SentinelEdge root
        fallback_path = os.path.join(os.path.dirname(__file__), "..", config_path)
        if os.path.exists(fallback_path):
            config_path = fallback_path
        else:
            logger.warning(f"Configuration file not found at {config_path}. Using empty config.")
            return Config()

    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        logger.info(f"Configuration successfully loaded from {config_path}")
        return Config(config_dict)
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise e
