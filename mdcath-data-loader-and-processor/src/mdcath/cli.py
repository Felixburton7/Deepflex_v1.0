"""
Command-Line Interface (CLI) for the mdCATH processor pipeline.
Uses argparse to handle command-line arguments.
"""

import argparse
import logging
import os
import sys
import traceback # Import traceback
from .config import load_config, DEFAULT_CONFIG_PATH, ConfigurationError
from .pipeline.executor import PipelineExecutor, PipelineExecutionError
from .utils.logging_config import setup_logging

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="mdCATH Dataset Processor: Extracts features and generates outputs for ML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to the pipeline configuration YAML file."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Path to the base output directory (overrides config)."
    )
    parser.add_argument(
        "-d", "--domains",
        nargs='*',
        default=None,
        help="List of specific domain IDs to process (overrides config)."
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=None,
        help="Number of CPU cores for parallel processing (0=auto, 1=sequential, overrides config)."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase console logging verbosity (-v for INFO, -vv for DEBUG)."
    )

    return parser.parse_args()

def main():
    """Main CLI entry point."""
    args = parse_arguments()

    config = None # Initialize config to None
    try:
        # Load configuration first
        config = load_config(args.config)

        # --- Setup Logging based on args and config ---
        log_cfg = config.setdefault('logging', {})
        if args.verbose == 1:
            log_cfg['log_level_console'] = 'INFO'
        elif args.verbose >= 2:
            log_cfg['log_level_console'] = 'DEBUG'
        # setup_logging will clear existing handlers
        setup_logging(config=config)

        # --- Override config with CLI arguments ---
        if args.output_dir:
            config['output']['base_dir'] = args.output_dir
            logging.info(f"Overriding output directory with CLI argument: {args.output_dir}")
        # Check if args.domains is explicitly provided (not None)
        if args.domains is not None:
             # Handle case where no domains are given on CLI (empty list []) vs not using the arg (None)
             config['input']['domain_ids'] = args.domains if args.domains else [] # Use empty list if args.domains is []
             logging.info(f"Overriding domain list with CLI arguments: {config['input']['domain_ids'] or 'All Found'}")
        if args.num_cores is not None:
            config['performance']['num_cores'] = args.num_cores
            logging.info(f"Overriding number of cores with CLI argument: {args.num_cores}")

        # --- Execute Pipeline ---
        executor = PipelineExecutor(config_dict=config)
        executor.run()

    except PipelineExecutionError as e:
        logging.error(f"Pipeline execution failed: {e}")
        # WORKAROUND for logging TypeError
        tb_str = traceback.format_exc()
        logging.error(f"Traceback for PipelineExecutionError:\n{tb_str}")
        sys.exit(1)
    except ConfigurationError as e: # Catch config errors specifically
        # Config error might happen before logging is fully set up
        print(f"ERROR: Configuration Error - {e}", file=sys.stderr)
        # Log traceback if logging *might* be available and level is DEBUG
        if logging.getLogger().isEnabledFor(logging.DEBUG):
             tb_str = traceback.format_exc()
             logging.debug(f"Traceback for ConfigurationError:\n{tb_str}")
        sys.exit(1)
    except Exception as e:
        # WORKAROUND for logging TypeError
        logging.error(f"An unexpected error occurred: {e}")
        tb_str = traceback.format_exc()
        logging.error(f"Traceback for unexpected error:\n{tb_str}")
        sys.exit(1)