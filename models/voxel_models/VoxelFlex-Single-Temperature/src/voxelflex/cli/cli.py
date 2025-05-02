"""
Command Line Interface for Voxelflex.

This module provides the main CLI functionality for the Voxelflex package,
including argument parsing and command dispatching.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from voxelflex.config.config import load_config
from voxelflex.utils.logging_utils import pipeline_tracker, ProgressBar, get_logger, setup_logging, EnhancedProgressBar, log_memory_usage, log_operation_result, log_section_header, log_stage, log_step
from voxelflex.utils.system_utils import check_system_resources
from voxelflex.cli.commands.train import train_model
from voxelflex.cli.commands.predict import predict_rmsf
from voxelflex.cli.commands.evaluate import evaluate_model
from voxelflex.cli.commands.visualize import create_visualizations

logger = get_logger(__name__)

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Args:
        args: Command line arguments (if None, sys.argv[1:] is used)
        
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        prog="voxelflex",
        description="Voxelflex: A package for predicting per-residue RMSF values from voxelized protein data",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run command (runs the entire pipeline)
    run_parser = subparsers.add_parser("run", help="Run the entire pipeline")
    run_parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration file"
    )
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration file with options for domain processing and memory management"
    )
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions with a trained model")
    predict_parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration file with options for domain processing and memory management"
    )
    predict_parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Path to trained model file"
    )
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate model performance")
    evaluate_parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration file with options for domain processing and memory management"
    )
    evaluate_parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Path to trained model file"
    )
    
    # Visualize command
    visualize_parser = subparsers.add_parser("visualize", help="Create visualizations")
    visualize_parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration file"
    )
    visualize_parser.add_argument(
        "--predictions", 
        type=str, 
        required=True,
        help="Path to predictions file"
    )
    
    parsed_args = parser.parse_args(args)
    
    if parsed_args.command is None:
        parser.print_help()
        sys.exit(1)
    
    return parsed_args

def run_pipeline(config_path: str) -> None:
    """
    Run the entire pipeline: train, predict, evaluate, and visualize.
    
    Args:
        config_path: Path to configuration file
    """
    # Start pipeline tracking
    pipeline_tracker.start_stage("INITIALIZATION", "Loading configuration and setting up environment")
    
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    log_file = os.path.join(
        config["output"]["base_dir"], 
        "logs", 
        config["output"]["log_file"]
    )
    setup_logging(
        log_file=log_file, 
        console_level=config["logging"]["console_level"],
        file_level=config["logging"]["file_level"]
    )
    
    logger = get_logger(__name__)
    
    # Log section header
    log_section_header(logger, "VOXELFLEX PIPELINE EXECUTION")
    
    # Check system resources
    system_info = check_system_resources(
        detect_cores=config["system_utilization"]["detect_cores"],
        adjust_for_gpu=config["system_utilization"]["adjust_for_gpu"]
    )
    logger.info(f"System resources: {system_info}")
    
    # Create output directories
    logger.info("Creating output directories")
    os.makedirs(os.path.join(config["output"]["base_dir"], "logs"), exist_ok=True)
    os.makedirs(os.path.join(config["output"]["base_dir"], "models"), exist_ok=True)
    os.makedirs(os.path.join(config["output"]["base_dir"], "metrics"), exist_ok=True)
    os.makedirs(os.path.join(config["output"]["base_dir"], "visualizations"), exist_ok=True)
    
    # End initialization stage
    pipeline_tracker.end_stage("INITIALIZATION")
    
    # Log initial memory usage
    log_memory_usage(logger)
    
    try:
        # Train the model
        logger.info("Starting model training phase")
        model_path, train_history = train_model(config)
        
        # Make predictions
        logger.info("Starting prediction phase")
        pipeline_tracker.start_stage("PREDICTION", "Making predictions with trained model")
        predictions_path = predict_rmsf(config, model_path)
        pipeline_tracker.end_stage("PREDICTION")
        
        # Evaluate the model
        logger.info("Starting evaluation phase")
        pipeline_tracker.start_stage("EVALUATION", "Evaluating model performance")
        metrics_path = evaluate_model(config, model_path, predictions_path)
        pipeline_tracker.end_stage("EVALUATION")
        
        # Create visualizations
        logger.info("Starting visualization phase")
        pipeline_tracker.start_stage("VISUALIZATION", "Creating performance visualizations")
        visualization_paths = create_visualizations(config, train_history, predictions_path)
        pipeline_tracker.end_stage("VISUALIZATION")
        
        # Cleanup and log summary
        pipeline_tracker.start_stage("CLEANUP", "Finalizing pipeline and logging results")
        
        # Log final memory usage
        log_memory_usage(logger)
        
        # Log summary of results
        logger.info("Pipeline completed successfully")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Predictions saved to: {predictions_path}")
        logger.info(f"Metrics saved to: {metrics_path}")
        logger.info(f"Visualizations saved to: {os.path.join(config['output']['base_dir'], 'visualizations')}")
        
        pipeline_tracker.end_stage("CLEANUP")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        if pipeline_tracker.current_stage:
            pipeline_tracker.end_stage()
        raise

# def run_pipeline(config_path: str) -> None:
#     """
#     Run the entire pipeline: train, predict, evaluate, and visualize.
    
#     Args:
#         config_path: Path to configuration file
#     """
#     # Load configuration
#     config = load_config(config_path)
    
#     # Set up logging
#     log_file = os.path.join(
#         config["output"]["base_dir"], 
#         "logs", 
#         config["output"]["log_file"]
#     )
#     setup_logging(
#         log_file=log_file, 
#         console_level=config["logging"]["console_level"],
#         file_level=config["logging"]["file_level"]
#     )
    
#     # Check system resources
#     system_info = check_system_resources(
#         detect_cores=config["system_utilization"]["detect_cores"],
#         adjust_for_gpu=config["system_utilization"]["adjust_for_gpu"]
#     )
#     logger.info(f"System resources: {system_info}")
    
#     # Create output directories
#     os.makedirs(os.path.join(config["output"]["base_dir"], "logs"), exist_ok=True)
#     os.makedirs(os.path.join(config["output"]["base_dir"], "models"), exist_ok=True)
#     os.makedirs(os.path.join(config["output"]["base_dir"], "metrics"), exist_ok=True)
#     os.makedirs(os.path.join(config["output"]["base_dir"], "visualizations"), exist_ok=True)
    
#     # Train the model
#     logger.info("Starting model training")
#     model_path, train_history = train_model(config)
    
#     # Make predictions
#     logger.info("Making predictions")
#     predictions_path = predict_rmsf(config, model_path)
    
#     # Evaluate the model
#     logger.info("Evaluating model performance")
#     metrics_path = evaluate_model(config, model_path, predictions_path)
    
#     # Create visualizations
#     logger.info("Creating visualizations")
#     create_visualizations(config, train_history, predictions_path)
    
#     logger.info("Pipeline completed successfully")
#     logger.info(f"Model saved to: {model_path}")
#     logger.info(f"Predictions saved to: {predictions_path}")
#     logger.info(f"Metrics saved to: {metrics_path}")
#     logger.info(f"Visualizations saved to: {os.path.join(config['output']['base_dir'], 'visualizations')}")


def main(args: Optional[List[str]] = None) -> None:
    """
    Main entry point for the CLI.
    
    Args:
        args: Command line arguments (if None, sys.argv[1:] is used)
    """
    parsed_args = parse_args(args)
    
    if parsed_args.command == "run":
        run_pipeline(parsed_args.config)
    elif parsed_args.command == "train":
        config = load_config(parsed_args.config)
        setup_logging(
            log_file=os.path.join(config["output"]["base_dir"], "logs", config["output"]["log_file"]),
            console_level=config["logging"]["console_level"],
            file_level=config["logging"]["file_level"]
        )
        train_model(config)
    elif parsed_args.command == "predict":
        config = load_config(parsed_args.config)
        setup_logging(
            log_file=os.path.join(config["output"]["base_dir"], "logs", config["output"]["log_file"]),
            console_level=config["logging"]["console_level"],
            file_level=config["logging"]["file_level"]
        )
        predict_rmsf(config, parsed_args.model)
    elif parsed_args.command == "evaluate":
        config = load_config(parsed_args.config)
        setup_logging(
            log_file=os.path.join(config["output"]["base_dir"], "logs", config["output"]["log_file"]),
            console_level=config["logging"]["console_level"],
            file_level=config["logging"]["file_level"]
        )
        evaluate_model(config, parsed_args.model)
    elif parsed_args.command == "visualize":
        config = load_config(parsed_args.config)
        setup_logging(
            log_file=os.path.join(config["output"]["base_dir"], "logs", config["output"]["log_file"]),
            console_level=config["logging"]["console_level"],
            file_level=config["logging"]["file_level"]
        )
        create_visualizations(config, None, parsed_args.predictions)


if __name__ == "__main__":
    main()