"""
Logging configuration for the mdCATH processor.
"""
import logging
import os
import sys
from typing import Optional, Dict, Any

def setup_logging(log_level_console: str = 'INFO',
                  log_level_file: str = 'DEBUG',
                  log_file: Optional[str] = None,
                  config: Optional[Dict[str, Any]] = None):
    """
    Sets up logging for the application.

    Args:
        log_level_console (str): Logging level for console output (e.g., 'INFO', 'DEBUG').
        log_level_file (str): Logging level for file output.
        log_file (Optional[str]): Path to the log file. If None, file logging is disabled.
        config (Optional[Dict[str, Any]]): If provided, overrides default levels and file path
                                            using keys from the 'logging' section.
    """
    # Get settings from config if provided
    log_filename_from_config = 'pipeline.log' # Default filename
    if config and 'logging' in config:
        log_cfg = config['logging']
        log_level_console = log_cfg.get('log_level_console', log_level_console).upper()
        log_level_file = log_cfg.get('log_level_file', log_level_file).upper()
        log_filename_from_config = log_cfg.get('log_filename', log_filename_from_config) # Use configured name
        # Construct log file path relative to output base directory
        output_base_dir = config.get('output', {}).get('base_dir', '.')
        log_dir = os.path.join(output_base_dir, 'logs')
        log_file = os.path.join(log_dir, log_filename_from_config)
    else:
        # Convert default levels to uppercase
        log_level_console = log_level_console.upper()
        log_level_file = log_level_file.upper()
        # If no config and no log_file path provided, disable file logging
        if log_file:
             # Assume log_file path is absolute or relative to current dir if no config
             log_dir = os.path.dirname(log_file)
        else:
             log_dir = None # Disable file logging

    # --- Configure Root Logger ---
    # Get the root logger
    root_logger = logging.getLogger()

    # *** Forcefully clear existing handlers to prevent conflicts/duplicates ***
    if root_logger.hasHandlers():
        # print("DEBUG: Clearing existing logging handlers.") # Optional: for debugging setup
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close() # Close handler to release file locks etc.

    # Set root logger level - set to lowest level desired across all handlers
    min_level_val = min(getattr(logging, log_level_console, logging.INFO),
                        getattr(logging, log_level_file, logging.DEBUG))
    root_logger.setLevel(min_level_val) # Capture everything up to the lowest level needed

    # Basic formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    date_formatter = logging.Formatter('%(levelname)s: %(message)s') # Even simpler format for console

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    try:
        console_level_val = getattr(logging, log_level_console, logging.INFO)
    except AttributeError:
        print(f"Warning: Invalid console log level '{log_level_console}'. Defaulting to INFO.")
        console_level_val = logging.INFO
    console_handler.setLevel(console_level_val)
    console_handler.setFormatter(date_formatter) # Use simpler format for console
    root_logger.addHandler(console_handler)

    # --- File Handler ---
    if log_file and log_dir: # Ensure both log_file and log_dir are set
        try:
            os.makedirs(log_dir, exist_ok=True)
            # Use 'a' for append or 'w' for overwrite
            file_handler = logging.FileHandler(log_file, mode='w') # Overwrite log each run
            try:
                file_level_val = getattr(logging, log_level_file, logging.DEBUG)
            except AttributeError:
                 print(f"Warning: Invalid file log level '{log_level_file}'. Defaulting to DEBUG.")
                 file_level_val = logging.DEBUG
            file_handler.setLevel(file_level_val)
            file_handler.setFormatter(formatter) # Use more detailed format for file
            root_logger.addHandler(file_handler)
            # Use print for this initial message as logging might not be fully ready
            print(f"Logging to file: {log_file}")
        except Exception as e:
            # Use print here as logging might be failing
            print(f"ERROR: Failed to set up file handler for {log_file}: {e}")
            # Optionally, print traceback if needed for debugging setup itself
            # import traceback
            # print(traceback.format_exc())
            # Continue with console logging only
    else:
        print("File logging disabled (no log file path configured).")

    # print("DEBUG: Logging setup complete.") # Optional: for debugging setup