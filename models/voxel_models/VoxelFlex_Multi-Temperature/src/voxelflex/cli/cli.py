"""
Command Line Interface for VoxelFlex (Temperature-Aware).

Main entry point for voxelflex commands: preprocess, train, predict, evaluate, visualize.
"""

import argparse
import logging
import os
import sys
import time
from typing import List, Optional

from voxelflex.utils.logging_utils import setup_logging, get_logger, log_section_header, log_final_memory_state
logger = get_logger("cli")

# Define master samples filename constant
MASTER_SAMPLES_FILENAME = "master_samples.parquet"

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Args:
        args: Optional list of command line arguments (default: sys.argv[1:])
        
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        prog="voxelflex", 
        description="VoxelFlex: Preprocess metadata, train, predict, evaluate, and visualize temperature-aware protein flexibility predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Define common arguments for all commands
    common_parser_args = argparse.ArgumentParser(add_help=False)
    common_parser_args.add_argument(
        '-v', '--verbose', 
        action='count', 
        default=0, 
        help="Increase verbosity (-v INFO, -vv DEBUG)."
    )
    common_parser_args.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to YAML config file."
    )
    
    # Define subcommands
    subparsers = parser.add_subparsers(
        dest="command", 
        help="Sub-command to run", 
        required=True
    )

    # --- Preprocess command ---
    preprocess_parser = subparsers.add_parser(
        "preprocess", 
        help="Preprocess metadata (generate sample list).", 
        parents=[common_parser_args], 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Train command ---
    train_parser = subparsers.add_parser(
        "train", 
        help="Train model (loads HDF5 on demand).", 
        parents=[common_parser_args], 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    train_parser.add_argument(
        "--force_preprocess", 
        action="store_true", 
        help="Run metadata preprocessing first."
    )
    train_parser.add_argument(
        "--subset", 
        type=float, 
        default=1.0,
        help="Use a subset of training data (0.0-1.0)"
    )
    
    # --- Predict command ---
    predict_parser = subparsers.add_parser(
        "predict", 
        help="Predict RMSF at a target temperature.", 
        parents=[common_parser_args], 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    predict_parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Path to trained model checkpoint (.pt)."
    )
    predict_parser.add_argument(
        "--temperature", 
        type=float, 
        required=True, 
        help="Target prediction temperature (K)."
    )
    predict_parser.add_argument(
        "--domains", 
        type=str, 
        nargs='*', 
        default=None, 
        help="Optional: List HDF5 domain keys. Uses test split if omitted."
    )
    predict_parser.add_argument(
        "--output_csv", 
        type=str, 
        default=None, 
        help="Optional: Specify output CSV filename."
    )
    
    # --- Evaluate command ---
    evaluate_parser = subparsers.add_parser(
        "evaluate", 
        help="Evaluate model predictions.", 
        parents=[common_parser_args], 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    evaluate_parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Path to trained model checkpoint (.pt)."
    )
    evaluate_parser.add_argument(
        "--predictions", 
        type=str, 
        required=True, 
        help="Path to predictions CSV file."
    )
    
    # --- Visualize command ---
    visualize_parser = subparsers.add_parser(
        "visualize", 
        help="Create performance visualizations.", 
        parents=[common_parser_args], 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    visualize_parser.add_argument(
        "--predictions", 
        type=str, 
        required=True, 
        help="Path to predictions CSV file."
    )
    visualize_parser.add_argument(
        "--history", 
        type=str, 
        default=None, 
        help="Optional: Path to training history JSON."
    )

    return parser.parse_args(args)

def main(cli_args: Optional[List[str]] = None) -> int:
    """
    Main CLI entry point for voxelflex.
    
    Args:
        cli_args: Optional list of command line arguments (default: sys.argv[1:])
        
    Returns:
        Exit code (0 = success, >0 = error)
    """
    start_time = time.time()
    args = parse_args(cli_args)
    
    # Configure initial logging based on verbosity
    console_log_level = "DEBUG" if args.verbose >= 2 else "INFO" if args.verbose == 1 else "WARNING"
    setup_logging(console_level=console_log_level, file_level="DEBUG", log_file=None)
    
    config = None
    log_file_path = None

    try:
        if not hasattr(args, 'config') or not args.config:
            raise ValueError("--config required.")
            
        # Load configuration
        from voxelflex.config.config import load_config
        config = load_config(args.config)
        
        # Set up logging to file
        log_file_path = os.path.join(config["output"]["log_dir"], config["output"]["log_file"])
        os.makedirs(config["output"]["log_dir"], exist_ok=True)
        setup_logging(log_file=log_file_path, console_level=console_log_level, file_level=config["logging"].get("file_level", "DEBUG"))
        logger.info(f"Logging re-initialized. Log file: {log_file_path}")
        logger.info(f"Run output directory: {config['output']['run_dir']}")
        
        # Log GPU information
        from voxelflex.utils.system_utils import log_gpu_details
        log_gpu_details(logger)

        log_section_header(logger, f"EXECUTING COMMAND: {args.command}")

        # Get the master sample filename from config
        master_samples_filename_from_config = config["data"].get("master_samples_file", MASTER_SAMPLES_FILENAME)

        # --- Command Execution ---
        if args.command == "preprocess":
            from voxelflex.cli.commands.preprocess import run_preprocessing
            run_preprocessing(config)
            
        elif args.command == "train":
            from voxelflex.cli.commands.train import train_model
            
            # Check for master sample file
            master_samples_path = os.path.join(config["data"]["processed_dir"], master_samples_filename_from_config)
            preprocessed_exists = os.path.exists(master_samples_path)

            if not preprocessed_exists or args.force_preprocess:
                if args.force_preprocess:
                    logger.info("Metadata preprocessing forced.")
                else:
                    logger.warning("Master sample file not found. Running preprocessing first...")
                    
                from voxelflex.cli.commands.preprocess import run_preprocessing
                run_preprocessing(config)
                
                if not os.path.exists(master_samples_path):
                    raise RuntimeError(f"Preprocessing ran but master sample file missing: {master_samples_path}")
                    
                logger.info("Preprocessing finished. Proceeding with training.")
            else:
                logger.info("Master sample file found. Skipping preprocessing.")
                
            # Apply subset parameter if provided
            if hasattr(args, 'subset') and args.subset < 1.0:
                if args.subset <= 0.0 or args.subset > 1.0:
                    logger.warning(f"Invalid subset value {args.subset}. Using full dataset.")
                else:
                    logger.info(f"Using {args.subset:.1%} of training data as requested")
                    config['training']['subset'] = args.subset
                
            train_model(config)
            
        elif args.command == "predict":
            from voxelflex.cli.commands.predict import predict_rmsf
            predict_rmsf(
                config=config, 
                model_path=args.model, 
                target_temperature=args.temperature, 
                domain_ids_to_predict=args.domains, 
                output_csv_filename=args.output_csv
            )
            
        elif args.command == "evaluate":
            from voxelflex.cli.commands.evaluate import evaluate_model
            evaluate_model(
                config=config, 
                model_path=args.model, 
                predictions_path=args.predictions
            )
            
        elif args.command == "visualize":
            from voxelflex.cli.commands.visualize import create_visualizations
            
            history_file_resolved = None
            if args.history:
                from voxelflex.utils.file_utils import resolve_path
                history_file_resolved = resolve_path(args.history)
                
            if history_file_resolved and not os.path.exists(history_file_resolved):
                logger.warning(f"History file not found: {history_file_resolved}")
                history_file_resolved = None
                
            create_visualizations(
                config=config, 
                predictions_path=args.predictions, 
                history_path=history_file_resolved
            )
            
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1

        log_section_header(logger, f"COMMAND '{args.command}' COMPLETED")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File Not Found Error: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Value Error: {e}")
        return 1
    except RuntimeError as e:
        logger.error(f"Runtime Error: {e}")
        return 1
    except ImportError as e:
        logger.error(f"Import Error: {e}. Ensure required libraries (e.g., pyarrow) are installed.")
        return 1
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        return 1
    finally:
        end_time = time.time()
        total_duration = end_time - start_time
        logger.info(f"Total execution time: {total_duration:.2f} seconds.")
        log_final_memory_state(logger)
        logging.shutdown()

if __name__ == "__main__":
    sys.exit(main())