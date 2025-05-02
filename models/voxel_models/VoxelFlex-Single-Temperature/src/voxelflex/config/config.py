"""
Configuration module for Voxelflex.

This module handles loading and validating YAML configuration files.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    # Expand user path if necessary
    config_path = os.path.expanduser(config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load configuration from YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate configuration
    validate_config(config)
    
    # Expand paths in configuration
    config = expand_paths(config)
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check for required sections
    required_sections = ['input', 'output', 'model', 'training']
    missing_sections = [section for section in required_sections if section not in config]
    
    if missing_sections:
        raise ValueError(f"Missing required configuration sections: {missing_sections}")
    
    # Validate input section
    if 'voxel_file' not in config['input']:
        raise ValueError("Missing required input parameter: voxel_file")
    
    if 'rmsf_dir' not in config['input']:
        raise ValueError("Missing required input parameter: rmsf_dir")
    
    # Validate output section
    if 'base_dir' not in config['output']:
        raise ValueError("Missing required output parameter: base_dir")
    
    # Validate model section
    if 'architecture' not in config['model']:
        raise ValueError("Missing required model parameter: architecture")
    
    valid_architectures = ['voxelflex_cnn', 'dilated_resnet3d', 'multipath_rmsf_net']
    if config['model']['architecture'] not in valid_architectures:
        raise ValueError(f"Invalid model architecture: {config['model']['architecture']}. "
                         f"Must be one of: {valid_architectures}")
    
    #Validate system memory and utilization
    if 'system_utilization' in config:
        if 'memory_ceiling_percent' in config['system_utilization']:
            memory_ceiling = config['system_utilization']['memory_ceiling_percent']
            if not isinstance(memory_ceiling, (int, float)) or memory_ceiling <= 0 or memory_ceiling > 100:
                raise ValueError(f"Invalid memory_ceiling_percent: {memory_ceiling}. Must be between 0 and 100.")
            
    # Validate training section
    if 'batch_size' not in config['training']:
        raise ValueError("Missing required training parameter: batch_size")
    
    if 'num_epochs' not in config['training']:
        raise ValueError("Missing required training parameter: num_epochs")


def expand_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expand relative paths in configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with expanded paths
    """
    # Expand paths in input section
    if 'data_dir' in config['input']:
        config['input']['data_dir'] = os.path.expanduser(config['input']['data_dir'])
    
    config['input']['voxel_file'] = os.path.expanduser(config['input']['voxel_file'])
    config['input']['rmsf_dir'] = os.path.expanduser(config['input']['rmsf_dir'])
    
    # Expand output base directory
    config['output']['base_dir'] = os.path.expanduser(config['output']['base_dir'])
    
    return config


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.
    
    Returns:
        Default configuration dictionary
    """
    # Get the path to the default configuration file
    default_config_path = os.path.join(
        os.path.dirname(__file__), 
        'default_config.yaml'
    )
    
    # Load default configuration
    with open(default_config_path, 'r') as f:
        default_config = yaml.safe_load(f)
    
    return default_config