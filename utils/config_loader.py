"""
Configuration Loader
Loads and manages YAML configuration files
"""

import yaml
from pathlib import Path
from typing import Any, Optional
import logging
import os

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Centralized configuration loader for YAML files
    
    Supports:
    - Loading multiple config files
    - Environment variable substitution
    - Config validation
    - Hot-reloading (optional)
    """
    
    _instance:  Optional['ConfigLoader'] = None
    _configs: dict[str, dict] = {}
    
    def __init__(self, config_dir: Optional[Path] = None):
        if config_dir is None:
            # Default to config/ directory relative to project root
            config_dir = Path(__file__).parent.parent / "config"
        
        self. config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise ValueError(f"Config directory not found: {self.config_dir}")
        
        logger.info(f"ConfigLoader initialized with directory: {self.config_dir}")
    
    @classmethod
    def get_instance(cls, config_dir: Optional[Path] = None) -> 'ConfigLoader':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls(config_dir)
        return cls._instance
    
    def load(self, config_name: str, reload: bool = False) -> dict[str, Any]:
        """
        Load a configuration file
        
        Args: 
            config_name: Name of config file (without . yaml extension)
            reload: Force reload from disk
        
        Returns:
            Configuration dictionary
        """
        if config_name in self._configs and not reload:
            return self._configs[config_name]
        
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Substitute environment variables
            config = self._substitute_env_vars(config)
            
            self._configs[config_name] = config
            logger.info(f"Loaded configuration: {config_name}")
            
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {config_path}: {e}")
            raise ValueError(f"Invalid YAML in {config_name}. yaml: {e}")
    
    def load_all(self) -> dict[str, dict]:
        """Load all YAML files in config directory"""
        configs = {}
        for config_file in self.config_dir.glob("*.yaml"):
            config_name = config_file.stem
            try:
                configs[config_name] = self. load(config_name)
            except Exception as e:
                logger.warning(f"Failed to load {config_name}: {e}")
        
        return configs
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """
        Recursively substitute environment variables in config
        
        Supports ${VAR_NAME} and ${VAR_NAME: default_value} syntax
        """
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Check for ${VAR_NAME} or ${VAR_NAME:default}
            import re
            pattern = r'\$\{([^}: ]+)(?::([^}]*))?\}'
            
            def replace_var(match):
                var_name = match. group(1)
                default = match.group(2)
                return os.getenv(var_name, default if default is not None else match.group(0))
            
            return re. sub(pattern, replace_var, obj)
        else:
            return obj
    
    def get(self, config_name: str, key_path: str, default: Any = None) -> Any:
        """
        Get a specific value from config using dot notation
        
        Example:  get('prompts', 'action_prompts. summarize. template')
        """
        config = self.load(config_name)
        
        keys = key_path.split('.')
        value = config
        
        try:
            for key in keys: 
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def reload_all(self):
        """Reload all configurations from disk"""
        logger.info("Reloading all configurations")
        config_names = list(self._configs.keys())
        for config_name in config_names: 
            self. load(config_name, reload=True)
