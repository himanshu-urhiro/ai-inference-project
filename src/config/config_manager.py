"""
Configuration Manager Module for PyTorch Video Inference

This module handles loading, validation, and access to configuration
settings for the video inference system on edge devices.
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages configuration for the video inference system.
    
    Handles loading, validation, and access to configuration settings.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = {}
        self.load_config()
        
    def load_config(self) -> bool:
        """
        Load configuration from file.
        
        Returns:
            bool: True if configuration loaded successfully, False otherwise
        """
        if not os.path.exists(self.config_path):
            logger.error(f"Configuration file not found: {self.config_path}")
            return False
            
        try:
            file_ext = os.path.splitext(self.config_path)[1].lower()
            
            if file_ext == '.json':
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            elif file_ext in ['.yaml', '.yml']:
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                logger.error(f"Unsupported configuration file format: {file_ext}")
                return False
                
            # Validate configuration
            if not self._validate_config():
                logger.error("Configuration validation failed")
                return False
                
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            return False
    
    def _validate_config(self) -> bool:
        """
        Validate the loaded configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        # Check for required sections
        required_sections = ['models', 'video', 'inference', 'api']
        for section in required_sections:
            if section not in self.config:
                logger.warning(f"Missing required configuration section: {section}")
                # Initialize with empty dict to prevent errors
                self.config[section] = {}
        
        # Set default values for missing configurations
        if 'models' in self.config:
            self.config['models'].setdefault('model_dir', './models')
            
        if 'video' in self.config:
            self.config['video'].setdefault('frame_skip', 1)
            self.config['video'].setdefault('queue_size', 30)
            self.config['video'].setdefault('target_size', (224, 224))
            
        if 'inference' in self.config:
            self.config['inference'].setdefault('batch_size', 1)
            
        if 'api' in self.config:
            self.config['api'].setdefault('host', '0.0.0.0')
            self.config['api'].setdefault('port', 8000)
            
        return True
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the complete configuration.
        
        Returns:
            Dict[str, Any]: Complete configuration dictionary
        """
        return self.config
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a specific configuration section.
        
        Args:
            section: Name of the configuration section
            
        Returns:
            Dict[str, Any]: Configuration section or empty dict if not found
        """
        return self.config.get(section, {})
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value.
        
        Args:
            key: Configuration key (can use dot notation for nested keys)
            default: Default value if key not found
            
        Returns:
            Any: Configuration value or default if not found
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def save_config(self, config_path: Optional[str] = None) -> bool:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration (uses current path if None)
            
        Returns:
            bool: True if configuration saved successfully, False otherwise
        """
        if config_path is None:
            config_path = self.config_path
            
        try:
            file_ext = os.path.splitext(config_path)[1].lower()
            
            if file_ext == '.json':
                with open(config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            elif file_ext in ['.yaml', '.yml']:
                with open(config_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            else:
                logger.error(f"Unsupported configuration file format: {file_ext}")
                return False
                
            logger.info(f"Configuration saved successfully to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            return False
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            
        Returns:
            bool: True if configuration updated successfully, False otherwise
        """
        try:
            # Recursively update configuration
            self._update_dict(self.config, updates)
            logger.info("Configuration updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update configuration: {str(e)}")
            return False
    
    def _update_dict(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively update a dictionary.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with updates
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._update_dict(target[key], value)
            else:
                target[key] = value
