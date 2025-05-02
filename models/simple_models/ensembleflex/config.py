# /home/s_felix/ensembleflex/ensembleflex/config.py

"""
Configuration handling for the ensembleflex ML pipeline.

Handles loading, validation, and management of configuration settings
for the unified temperature-aware model.
"""

import os
import logging
# from pathlib import Path # Not strictly needed anymore
from typing import Dict, Any, Optional, List, Union
import yaml
# import re # No longer needed for templating

logger = logging.getLogger(__name__)

def deep_merge(base_dict: Dict, overlay_dict: Dict) -> Dict:
    """
    Recursively merge two dictionaries, with values from overlay_dict taking precedence.

    Args:
        base_dict: Base dictionary to merge into
        overlay_dict: Dictionary with values that should override base_dict

    Returns:
        Dict containing merged configuration
    """
    result = base_dict.copy()

    for key, value in overlay_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result

def get_env_var_config() -> Dict[str, Any]:
    """
    Get configuration from environment variables.
    Environment variables should be prefixed with ensembleflex_ and use
    underscore separators for nested keys.

    Examples:
        ensembleflex_PATHS_DATA_DIR=/path/to/data
        ensembleflex_MODELS_RANDOM_FOREST_N_ESTIMATORS=200

    Returns:
        Dict containing configuration from environment variables
    """
    config = {}
    prefix = "ensembleflex_" # Changed prefix

    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue

        # Remove prefix and convert to lowercase
        key = key[len(prefix):].lower()

        # Split into parts and create nested dict
        parts = key.split("_")
        current = config

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set value, converting to appropriate type
        value_part = parts[-1]

        # Try to convert to appropriate type
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.lower() == "null" or value.lower() == "none":
            value = None
        else:
            try:
                # Try int first, then float
                value = int(value)
            except ValueError:
                try:
                     value = float(value)
                except ValueError:
                     # Keep as string if not convertible
                     pass

        current[value_part] = value

    return config

def parse_param_overrides(params: List[str]) -> Dict[str, Any]:
    """
    Parse parameter overrides from CLI arguments.

    Args:
        params: List of parameter overrides in format "key=value"

    Returns:
        Dict containing parameter overrides
    """
    if not params:
        return {}

    override_dict = {}

    for param in params:
        if "=" not in param:
            logger.warning(f"Ignoring invalid parameter override: {param}")
            continue

        key, value_str = param.split("=", 1)
        value: Any = value_str # Keep type flexible initially

        # Convert value to appropriate type
        if value_str.lower() == "true":
            value = True
        elif value_str.lower() == "false":
            value = False
        elif value_str.lower() == "null" or value_str.lower() == "none":
            value = None
        else:
            try:
                # Try int first, then float
                value = int(value_str)
            except ValueError:
                try:
                    value = float(value_str)
                except ValueError:
                    # Keep as string if not convertible
                    value = value_str

        # Split key into parts and create nested dict
        parts = key.split(".")
        current = override_dict

        for part in parts[:-1]:
            # Ensure intermediate levels are dictionaries
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
            # Check if trying to assign value to a non-leaf node
            if not isinstance(current, dict):
                 logger.warning(f"Trying to override intermediate key '{part}' in '{key}' which is not a dictionary. Override might fail.")
                 break # Stop processing this override

        # Only assign if we reached the intended leaf node
        if isinstance(current, dict):
            current[parts[-1]] = value
        else:
            logger.warning(f"Could not apply override '{key}={value_str}' due to path conflict.")


    return override_dict

# REMOVED: template_config_for_temperature function

def load_config(
    config_path: Optional[str] = None,
    param_overrides: Optional[List[str]] = None,
    use_env_vars: bool = True,
    # temperature: Optional[Union[int, str]] = None # REMOVED temperature argument
) -> Dict[str, Any]:
    """
    Load configuration from default and user-provided sources.
    Handles merging of defaults, user file, environment variables,
    and command-line parameters.

    Args:
        config_path: Optional path to user config file.
        param_overrides: Optional list of parameter overrides from CLI.
        use_env_vars: Whether to use environment variables (ensembleflex_*).

    Returns:
        Dict containing the final merged configuration.

    Raises:
        FileNotFoundError: If default or specified user config file doesn't exist.
        ValueError: If configuration is invalid after merging.
    """
    # Determine default config path relative to this file's location
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project root
    default_path = os.path.join(base_dir, "default_config.yaml")

    # Load default config
    if not os.path.exists(default_path):
        raise FileNotFoundError(f"Default config not found at expected path: {default_path}")

    logger.debug(f"Loading default configuration from: {default_path}")
    with open(default_path, 'r') as f:
        config = yaml.safe_load(f)
    if config is None:
         config = {}
         logger.warning(f"Default config file '{default_path}' seems empty.")


    # Overlay user config if provided
    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"User config file not found at: {config_path}")
        logger.info(f"Loading user configuration from: {config_path}")
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            if user_config:
                config = deep_merge(config, user_config)
            else:
                 logger.warning(f"User config file '{config_path}' is empty.")


    # Apply environment variable overrides
    if use_env_vars:
        env_config = get_env_var_config()
        if env_config:
            logger.debug(f"Applying {len(env_config)} environment variable overrides.")
            config = deep_merge(config, env_config)
        else:
             logger.debug("No environment variable overrides found.")


    # Apply CLI parameter overrides
    if param_overrides:
        override_config = parse_param_overrides(param_overrides)
        if override_config:
            logger.debug(f"Applying {len(override_config)} CLI parameter overrides.")
            config = deep_merge(config, override_config)
        else:
            logger.debug("No valid CLI parameter overrides provided.")


    # REMOVED: Temperature override and templating logic

    # Handle OmniFlex mode settings (Ensure features are correctly toggled based on mode)
    # This needs to happen *after* all overrides are applied.
    try:
        if config.get("mode", {}).get("active", "standard").lower() == "omniflex":
            logger.info("OmniFlex mode active. Ensuring relevant features are enabled.")
            omniflex_cfg = config.get("mode", {}).get("omniflex", {})
            use_features = config.setdefault("dataset", {}).setdefault("features", {}).setdefault("use_features", {})

            if omniflex_cfg.get("use_esm", False):
                if 'esm_rmsf' not in use_features: logger.warning("Enabling 'esm_rmsf' feature for OmniFlex mode.")
                use_features["esm_rmsf"] = True
            else:
                 if use_features.get("esm_rmsf", False): logger.info("Disabling 'esm_rmsf' feature as OmniFlex use_esm is false.")
                 use_features["esm_rmsf"] = False # Explicitly disable if OmniFlex doesn't want it


            if omniflex_cfg.get("use_voxel", False):
                 if 'voxel_rmsf' not in use_features: logger.warning("Enabling 'voxel_rmsf' feature for OmniFlex mode.")
                 use_features["voxel_rmsf"] = True
            else:
                 if use_features.get("voxel_rmsf", False): logger.info("Disabling 'voxel_rmsf' feature as OmniFlex use_voxel is false.")
                 use_features["voxel_rmsf"] = False # Explicitly disable

        elif config.get("mode", {}).get("active", "standard").lower() == "standard":
             logger.info("Standard mode active. Ensuring OmniFlex features are disabled.")
             use_features = config.setdefault("dataset", {}).setdefault("features", {}).setdefault("use_features", {})
             if use_features.get("esm_rmsf", False): logger.info("Disabling 'esm_rmsf' feature for Standard mode.")
             use_features["esm_rmsf"] = False
             if use_features.get("voxel_rmsf", False): logger.info("Disabling 'voxel_rmsf' feature for Standard mode.")
             use_features["voxel_rmsf"] = False

    except Exception as e:
         logger.error(f"Error applying OmniFlex mode settings: {e}", exc_info=True)


    # Validate final config
    validate_config(config)

    # Set system-wide logging level
    try:
        log_level_str = config.get("system", {}).get("log_level", "INFO").upper()
        numeric_level = getattr(logging, log_level_str, None)
        if isinstance(numeric_level, int):
            # Set level for the root logger and potentially specific package loggers
            logging.getLogger().setLevel(numeric_level)
            logging.getLogger("ensembleflex").setLevel(numeric_level) # Set for our package too
            logger.info(f"Logging level set to {log_level_str}")
        else:
             logger.warning(f"Invalid log level '{log_level_str}' in config. Using default.")

    except Exception as e:
         logger.warning(f"Could not set logging level from config: {e}")


    return config

def validate_config(config: Dict[str, Any]) -> None:
    """
    Perform basic validation of the merged ensembleflex configuration.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    logger.debug("Validating final configuration...")
    # Check required top-level sections
    required_sections = ["paths", "dataset", "models", "evaluation", "system", "temperature", "mode"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")

    # Validate dataset section
    dataset_cfg = config.get("dataset", {})
    if "target" not in dataset_cfg:
        raise ValueError("Missing required 'dataset.target' configuration")
    if dataset_cfg["target"] != "rmsf":
        logger.warning(f"Expected 'dataset.target' to be 'rmsf', but found '{dataset_cfg['target']}'.")
    if "file_pattern" not in dataset_cfg:
        raise ValueError("Missing required 'dataset.file_pattern' configuration")
    if "{temperature}" in dataset_cfg["file_pattern"]:
        logger.warning(f"Found '{{temperature}}' in 'dataset.file_pattern'. This is likely incorrect for ensembleflex aggregated data.")

    # Validate features section
    features_cfg = dataset_cfg.get("features", {})
    if "required" not in features_cfg:
        raise ValueError("Missing required 'dataset.features.required' configuration")
    required_list = features_cfg["required"]
    if "rmsf" not in required_list:
        logger.warning("Target 'rmsf' not listed in 'dataset.features.required'.")
    if "temperature" not in required_list:
        logger.warning("Input feature 'temperature' not listed in 'dataset.features.required'.")

    if "use_features" not in features_cfg:
        raise ValueError("Missing required 'dataset.features.use_features' configuration")
    if "temperature" not in features_cfg["use_features"]:
        logger.warning("Feature toggle 'dataset.features.use_features.temperature' is missing. It might not be included as input.")

    # REMOVED: Temperature section validation for current/available consistency

    # Check paths
    paths_cfg = config.get("paths", {})
    if "output_dir" not in paths_cfg or "{temperature}" in paths_cfg["output_dir"]:
         logger.warning(f"Path 'paths.output_dir' might be misconfigured: {paths_cfg.get('output_dir')}")
    if "models_dir" not in paths_cfg or "{temperature}" in paths_cfg["models_dir"]:
         logger.warning(f"Path 'paths.models_dir' might be misconfigured: {paths_cfg.get('models_dir')}")


    # Check that at least one model is enabled
    any_model_enabled = False
    for model_name, model_config in config.get("models", {}).items():
        if model_name != "common" and isinstance(model_config, dict) and model_config.get("enabled", False):
            any_model_enabled = True
            break

    if not any_model_enabled:
        logger.warning("No models are enabled in configuration ('models.*.enabled: true'). No training will occur.")

    # Add more specific checks as needed (e.g., data types, valid choices)
    logger.debug("Configuration validation passed.")


def get_enabled_models(config: Dict[str, Any]) -> List[str]:
    """
    Get list of enabled model names from config.

    Args:
        config: Configuration dictionary

    Returns:
        List of enabled model names
    """
    enabled_models = []

    for model_name, model_config in config.get("models", {}).items():
        # Ensure model_config is a dictionary and has 'enabled' key
        if model_name != "common" and isinstance(model_config, dict) and model_config.get("enabled", False):
            enabled_models.append(model_name)

    return enabled_models

def get_model_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model, with common settings applied.

    Args:
        config: Full configuration dictionary
        model_name: Name of the model

    Returns:
        Model-specific configuration with common settings merged in

    Raises:
        ValueError: If model_name is not found in configuration
    """
    models_config = config.get("models", {})

    if model_name not in models_config:
        raise ValueError(f"Model '{model_name}' not found in configuration")

    model_config = models_config[model_name]
    if not isinstance(model_config, dict):
         logger.warning(f"Configuration for model '{model_name}' is not a dictionary. Using empty config.")
         model_config = {}

    common_config = models_config.get("common", {})
    if not isinstance(common_config, dict):
         logger.warning("Section 'models.common' is not a dictionary. Using empty common config.")
         common_config = {}


    # Merge common config with model-specific config
    merged_config = deep_merge(common_config, model_config)

    return merged_config

def get_available_temperatures(config: Dict[str, Any]) -> List[Union[int, str]]:
    """
    Get list of available temperatures from config (informational only).

    Args:
        config: Configuration dictionary

    Returns:
        List of available temperatures defined in the config
    """
    return config.get("temperature", {}).get("available", [])

# --- ADAPTED Path Functions ---
# These now ignore any temperature argument and return the static paths

def get_output_dir(config: Dict[str, Any]) -> str:
    """
    Get the unified output directory path.

    Args:
        config: Configuration dictionary

    Returns:
        Path to the output directory.
    """
    return config.get("paths", {}).get("output_dir", "./output/ensembleflex/") # Provide default

def get_models_dir(config: Dict[str, Any]) -> str:
    """
    Get the unified models directory path.

    Args:
        config: Configuration dictionary

    Returns:
        Path to the models directory.
    """
    return config.get("paths", {}).get("models_dir", "./models/ensembleflex/") # Provide default

