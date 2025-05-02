"""
Configuration loading and validation for the pipeline.
"""
import yaml
import os
import logging
from typing import Dict, Any, Optional

# Define potential default config locations
DEFAULT_CONFIG_NAME = 'default_config.yaml'
# Assume config dir is sibling to src dir if run from root, or relative to this file
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, DEFAULT_CONFIG_NAME)


class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass


def _find_config_file(config_path: Optional[str] = None) -> Optional[str]:
    """Finds the configuration file path."""
    if config_path:
        if os.path.exists(config_path):
            return config_path
        else:
            raise ConfigurationError(f"Specified config file not found: {config_path}")
    elif os.path.exists(DEFAULT_CONFIG_PATH):
         return DEFAULT_CONFIG_PATH
    elif os.path.exists(os.path.join('config', DEFAULT_CONFIG_NAME)): # Check relative to current dir
         return os.path.join('config', DEFAULT_CONFIG_NAME)
    else:
         return None # Let caller handle missing default


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads the YAML configuration file.

    Args:
        config_path (Optional[str]): Path to the config file. If None, tries default locations.

    Returns:
        Dict[str, Any]: The loaded configuration dictionary.

    Raises:
        ConfigurationError: If the config file cannot be found or parsed, or fails validation.
    """
    actual_config_path = _find_config_file(config_path)
    if not actual_config_path:
        raise ConfigurationError("Configuration file not found. Provide path via --config or place default_config.yaml in ./config/")

    logging.info(f"Loading configuration from: {actual_config_path}")
    try:
        with open(actual_config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        if not isinstance(config_data, dict):
            raise ConfigurationError("Configuration file is not a valid YAML dictionary.")

        _validate_config(config_data, actual_config_path) # Basic validation
        return config_data
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error parsing YAML file {actual_config_path}: {e}") from e
    except Exception as e:
         # Catch other potential errors like permissions
         raise ConfigurationError(f"Error reading configuration file {actual_config_path}: {e}") from e

def _validate_config(config: Dict[str, Any], path: str):
    """Performs basic validation checks on the loaded config."""
    logging.debug("Validating configuration...")
    # 1. Check required top-level keys
    required_keys = ['input', 'output', 'processing', 'performance', 'logging', 'visualization']
    for key in required_keys:
        if key not in config:
            raise ConfigurationError(f"Missing required top-level key '{key}' in config file: {path}")

    # 2. Check 'input.mdcath_folder' existence (if not using explicit domain list)
    input_cfg = config['input']
    if not input_cfg.get('domain_ids'): # If domain_ids is null or empty list
        folder = input_cfg.get('mdcath_folder')
        if not folder or not os.path.isdir(folder):
             raise ConfigurationError(f"'input.mdcath_folder' ({folder}) is not specified or not a valid directory in config: {path}, "
                                     "and 'input.domain_ids' is not provided.")

    # 3. Check 'output.base_dir'
    output_dir = config['output'].get('base_dir')
    if not output_dir:
         raise ConfigurationError(f"Missing required key 'output.base_dir' in config file: {path}")
    # Check if parent dir is writable? Might be too strict. Let makedirs handle it later.


    # Add more specific checks as needed (e.g., valid enum values for methods)
    proc_cfg = config['processing']
    frame_method = proc_cfg.get('frame_selection', {}).get('method')
    valid_frame_methods = ['regular', 'rmsd', 'gyration', 'random', 'last']
    if frame_method not in valid_frame_methods:
         logging.warning(f"Invalid 'processing.frame_selection.method': {frame_method}. Should be one of {valid_frame_methods}. Defaulting behavior might occur.")

    logging.debug("Configuration validation passed (basic checks).")


# Modify PipelineExecutor to accept config dict directly
# Add this method or modify __init__ in pipeline/executor.py
def __init__(self, config_dict: Dict[str, Any]):
        """
        Initializes the executor with a configuration dictionary.

        Args:
            config_dict (Dict[str, Any]): The loaded configuration dictionary.
        """
        try:
            self.config = config_dict # Store the loaded config
            self.output_dir = self.config.get('output', {}).get('base_dir', './outputs')
            # ... rest of initialization as before ...

        except Exception as e:
            logging.exception(f"Failed to initialize PipelineExecutor with config dict: {e}")
            raise PipelineExecutionError(f"Initialization failed: {e}") from e

# Make sure PipelineExecutor.__init__ in executor.py reflects this change
# Update the call in cli.py: executor = PipelineExecutor(config_dict=config)

