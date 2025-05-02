#!/usr/bin/env python3
"""
Main entry point for mdCATH dataset processing pipeline.
"""
import sys
import logging
import mdcath.cli
# from mdcath.utils.logging_config import setup_logging # Removed initial setup call

# Basic configuration just to catch early errors before full setup
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

def run_pipeline():
    """Parses arguments and runs the main pipeline executor."""
    try:
        # cli.main() handles argument parsing, config loading, FULL logging setup, and executor run
        mdcath.cli.main()
        logging.info("mdCATH processing pipeline completed successfully.") # This should use the configured logger
    except SystemExit:
         # Allow SystemExit (e.g., from argparse --help) to pass through without logging failure
         pass
    except Exception as e:
        # Catch broad exceptions here as a last resort
        # Use the root logger, which *should* be configured by cli.main if it got that far
        # If cli.main failed very early, basicConfig might be used.
        logging.critical(f"Pipeline execution failed at the top level: {e}", exc_info=True)
        sys.exit(1) # Exit with a non-zero code to indicate failure

if __name__ == "__main__":
    run_pipeline()