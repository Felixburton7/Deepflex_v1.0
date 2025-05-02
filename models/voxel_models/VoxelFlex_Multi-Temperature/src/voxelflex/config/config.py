"""
Configuration module for VoxelFlex (Temperature-Aware).

Handles loading, validation, merging with defaults, and path expansion.
"""
import os
import logging
import time
import socket
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

logger = logging.getLogger("voxelflex.config")
from voxelflex.utils.file_utils import resolve_path, ensure_dir

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file and merge with defaults.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Complete configuration dictionary
        
    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config YAML is invalid
        RuntimeError: If default config could not be loaded
    """
    # Resolve config path
    config_path_resolved = resolve_path(config_path)
    logger.info(f"Loading user config: {config_path_resolved}")
    
    if not os.path.exists(config_path_resolved):
        raise FileNotFoundError(f"Config file not found: {config_path_resolved}")
        
    # Load user config
    try:
        with open(config_path_resolved, 'r') as f:
            user_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {config_path_resolved}") from e
        
    # Handle empty config
    if user_config is None:
        logger.warning(f"User config {config_path_resolved} empty.")
        user_config = {}
        
    # Load default config
    default_config = get_default_config()
    if not default_config:
        raise RuntimeError("Failed to load default config.")
        
    # Merge configs
    config = merge_configs(default_config, user_config)
    
    # Validate merged config
    validate_config(config)
    
    # Expand paths
    config = expand_paths(config)
    
    # Add timestamp to run name if needed
    if "run_name" not in config["output"] or "{timestamp}" in config["output"].get("run_name", ""):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name_template = config["output"].get("run_name", "voxelflex_run_{timestamp}")
        config["output"]["run_name"] = run_name_template.format(timestamp=timestamp)
        logger.info(f"Run name: {config['output']['run_name']}")
    
    # Create output directory structure
    base_output_dir = config["output"]["base_dir"]
    run_output_dir = os.path.join(base_output_dir, config["output"]["run_name"])
    
    # Add output directories to config
    config["output"]["run_dir"] = run_output_dir
    config["output"]["log_dir"] = os.path.join(run_output_dir, "logs")
    config["output"]["models_dir"] = os.path.join(run_output_dir, "models")
    config["output"]["metrics_dir"] = os.path.join(run_output_dir, "metrics")
    config["output"]["visualizations_dir"] = os.path.join(run_output_dir, "visualizations")
    
    # Create directories
    ensure_dir(config["output"]["log_dir"])
    ensure_dir(config["output"]["models_dir"])
    ensure_dir(config["output"]["metrics_dir"])
    ensure_dir(config["output"]["visualizations_dir"])
    
    # Setup temperature scaling file path
    scaling_file_name = os.path.basename(config["data"]["temp_scaling_params_file"])
    if not scaling_file_name:
        scaling_file_name = "temp_scaling_params.json"
    config["data"]["temp_scaling_params_file"] = os.path.join(config["output"]["models_dir"], scaling_file_name)
    logger.debug(f"Temp scaling file path: {config['data']['temp_scaling_params_file']}")
    
    # Set master samples path
    processed_base = config["data"]["processed_dir"]
    config["data"]["master_samples_path"] = os.path.join(processed_base, config["data"]["master_samples_file"])
    logger.debug(f"Master samples file path: {config['data']['master_samples_path']}")
    
    # Remove old processed paths if they exist from previous versions
    config["data"].pop("processed_train_dir", None)
    config["data"].pop("processed_val_dir", None)
    config["data"].pop("processed_test_dir", None)
    config["data"].pop("processed_train_meta", None)
    config["data"].pop("processed_val_meta", None)
    config["data"].pop("processed_test_meta", None)
    
    # Add runtime information
    config["runtime"] = {
        "timestamp": timestamp,
        "hostname": socket.gethostname(),
        "python_version": sys.version
    }
    
    logger.debug("Configuration loaded successfully.")
    return config

def merge_configs(default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge user config with default config.
    User values override defaults.
    
    Args:
        default: Default configuration
        user: User configuration
        
    Returns:
        Merged configuration
    """
    merged = default.copy()
    
    for key, value in user.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override default with user value
            merged[key] = value
            
    return merged

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure and essential parameters.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: For invalid configurations
    """
    logger.debug("Validating configuration structure...")
    
    # Check required sections
    required_sections = [
        'input', 'output', 'model', 'training', 'data', 
        'logging', 'evaluation', 'visualization', 'system_utilization'
    ]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing config section: '{section}'")
        if not isinstance(config[section], dict):
            raise ValueError(f"Section '{section}' must be a dictionary.")
    
    # Check input configuration
    input_cfg = config['input']
    req_input = ['voxel_file', 'aggregated_rmsf_file', 'train_split_file', 'val_split_file']
    
    for key in req_input:
        if key not in input_cfg or not input_cfg[key]:
            raise ValueError(f"Missing input param: 'input.{key}'")
            
    if 'test_split_file' not in input_cfg or not input_cfg['test_split_file']:
        logger.warning(f"Optional input 'input.test_split_file' missing.")
    
    # Check data configuration
    data_cfg = config['data']
    req_data = ['processed_dir', 'master_samples_file', 'temp_scaling_params_file']
    
    for key in req_data:
        if key not in data_cfg or not data_cfg[key]:
            raise ValueError(f"Missing data param: 'data.{key}'")
    
    # Check output configuration
    if 'base_dir' not in config['output'] or not config['output']['base_dir']:
        raise ValueError("Missing output param: 'output.base_dir'")
    
    # Check model configuration
    model_cfg = config.get('model', {})
    if 'architecture' not in model_cfg:
        raise ValueError("Missing 'model.architecture'")
        
    valid_arch = ['densenet3d_regression', 'dilated_resnet3d', 'multipath_rmsf_net']
    if model_cfg['architecture'] not in valid_arch:
        raise ValueError(f"Invalid 'model.architecture'. Use: {valid_arch}")
    
    # Check model dimensions
    req_model_dims = ['input_channels', 'voxel_depth', 'voxel_height', 'voxel_width']
    for key in req_model_dims:
        if key not in model_cfg or not isinstance(model_cfg[key], int) or model_cfg[key] <= 0:
            raise ValueError(f"Missing/invalid positive integer for 'model.{key}' (needed by VoxelDataset)")
    
    # Check specific model architecture parameters
    if model_cfg['architecture'] == 'densenet3d_regression':
        if 'densenet' not in model_cfg or not isinstance(model_cfg['densenet'], dict):
            raise ValueError("Missing 'model.densenet' section.")
            
        req_densenet = ['growth_rate', 'block_config', 'num_init_features', 'bn_size']
        for key in req_densenet:
            if key not in model_cfg['densenet']:
                raise ValueError(f"Missing 'model.densenet.{key}'")
                
        if not isinstance(model_cfg['densenet']['block_config'], list):
            raise ValueError("'model.densenet.block_config' must be a list.")
    
    # Check training configuration
    train_cfg = config.get('training', {})
    req_train = ['batch_size', 'num_epochs', 'learning_rate', 'weight_decay', 'seed']
    
    for key in req_train:
        if key not in train_cfg:
            raise ValueError(f"Missing 'training.{key}'")
    
    # Check training parameters
    if not isinstance(train_cfg.get('batch_size'), int) or train_cfg.get('batch_size', 0) <= 0:
        raise ValueError("'training.batch_size' must be positive.")
        
    if not isinstance(train_cfg.get('num_epochs'), int) or train_cfg.get('num_epochs', 0) <= 0:
        raise ValueError("'training.num_epochs' must be positive.")
        
    if not isinstance(train_cfg.get('num_workers', 0), int) or train_cfg.get('num_workers', 0) < 0:
        raise ValueError("'training.num_workers' must be non-negative.")
    
    # Check metrics
    valid_metrics = ['val_loss', 'val_pearson']
    
    def validate_monitor_metric(cfg_section: dict, section_key: str, section_name: str):
        metric = cfg_section.get(section_key, {}).get('monitor_metric')
        if metric and metric not in valid_metrics:
            raise ValueError(f"'training.{section_name}.monitor_metric' ('{metric}') must be one of {valid_metrics}")
    
    if 'save_best_metric' in train_cfg and train_cfg['save_best_metric'] not in valid_metrics:
        raise ValueError(f"'training.save_best_metric' must be one of {valid_metrics}")
        
    validate_monitor_metric(train_cfg, 'scheduler', 'scheduler')
    validate_monitor_metric(train_cfg, 'early_stopping', 'early_stopping')
    
    # Check scheduler
    sched_cfg = train_cfg.get('scheduler', {})
    if 'type' in sched_cfg and sched_cfg['type'] not in ['reduce_on_plateau', 'cosine_annealing', 'step']:
        raise ValueError(f"Invalid scheduler type: {sched_cfg['type']}")
    
    # Check system configuration
    sys_cfg = config.get('system_utilization', {})
    if not isinstance(sys_cfg.get('detect_cores'), bool):
        raise ValueError("'system_utilization.detect_cores' must be boolean.")
        
    if not isinstance(sys_cfg.get('adjust_for_gpu'), bool):
        raise ValueError("'system_utilization.adjust_for_gpu' must be boolean.")
    
    logger.debug("Configuration validation passed.")

def expand_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expand relative paths in configuration to absolute paths.
    
    Args:
        config: Configuration with paths to expand
        
    Returns:
        Configuration with expanded paths
    """
    logger.debug("Expanding paths in configuration...")
    
    # List of (section, key) tuples for paths to expand
    paths_to_expand = {
        ('input', 'voxel_file'), 
        ('input', 'aggregated_rmsf_file'), 
        ('input', 'train_split_file'), 
        ('input', 'val_split_file'), 
        ('input', 'test_split_file'), 
        ('output', 'base_dir'), 
        ('data', 'processed_dir'), 
        ('training', 'resume_checkpoint'),
    }
    
    for section, key in paths_to_expand:
        if config.get(section) is not None and isinstance(config[section], dict) and config[section].get(key):
            original_path = config[section][key]
            if isinstance(original_path, str) and original_path:
                config[section][key] = resolve_path(original_path)
            elif not original_path:
                config[section][key] = None
                
    return config

def get_default_config() -> Dict[str, Any]:
    """
    Load default configuration from default_config.yaml.
    
    Returns:
        Default configuration or empty dict if file not found/valid
    """
    default_config_path = os.path.join(os.path.dirname(__file__), 'default_config.yaml')
    local_logger = logging.getLogger("voxelflex.config.default")
    local_logger.debug(f"Loading default config: {default_config_path}")
    
    if not os.path.exists(default_config_path):
        local_logger.error(f"Default config NOT FOUND: {default_config_path}")
        return {}
        
    try:
        with open(default_config_path, 'r') as f:
            default_config = yaml.safe_load(f)
            
        if default_config is None:
            local_logger.error("Default config empty!")
            return {}
            
        local_logger.debug("Default config loaded.")
        return default_config
    except Exception as e:
        local_logger.error(f"Failed load default config: {e}")
        return {}