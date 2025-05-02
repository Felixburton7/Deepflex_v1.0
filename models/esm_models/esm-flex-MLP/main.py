#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
import logging # Add logging import

# Set project root directory relative to this script file
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Add project root to Python path to allow importing modules
sys.path.insert(0, PROJECT_ROOT)

# Now import project modules
try:
    from data_processor import process_data
    from train import train
    from predict import predict
except ImportError as e:
     print(f"Error: Failed to import project modules. Is the script run from the project root? Error: {e}")
     sys.exit(1)


# Setup basic logging for the main script orchestrator
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(module)s] - %(message)s')
logger = logging.getLogger(__name__) # Get logger for this module


def main():
    parser = argparse.ArgumentParser(
        description='ESM-Flex (ESM-3): Protein Flexibility (RMSF) Prediction Pipeline.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
    )
    subparsers = parser.add_subparsers(
        dest='command',
        help='Select the command to run: process, train, or predict.',
        required=True # Make selecting a command mandatory
    )

    # --- Process data command ---
    process_parser = subparsers.add_parser(
        'process',
        help='Process raw RMSF data from CSV into standardized training/validation/test splits.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    process_parser.add_argument('--csv', type=str, required=True, help='Path to the input raw RMSF CSV file (e.g., data/raw/rmsf_..._fixed.csv).')
    process_parser.add_argument('--output', type=str, default='data/processed', help='Output directory for processed data splits (train/val/test files).')
    process_parser.add_argument('--train_ratio', type=float, default=0.7, help='Fraction of protein topologies allocated to the training set.')
    process_parser.add_argument('--val_ratio', type=float, default=0.15, help='Fraction of protein topologies allocated to the validation set.')
    process_parser.add_argument('--seed', type=int, default=42, help='Random seed for topology shuffling before splitting.')

    # --- Train command ---
    train_parser = subparsers.add_parser(
        'train',
        help='Train the ESM-3 Regression model using processed data and a config file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    train_parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file defining training parameters and model settings.')

    # --- Predict command ---
    predict_parser = subparsers.add_parser(
        'predict',
        help='Predict RMSF for new sequences using a trained model checkpoint.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    predict_parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to the trained model checkpoint (.pt file) to use for prediction.')
    predict_parser.add_argument('--fasta_path', type=str, required=True, help='Path to the input FASTA file containing protein sequences for which to predict RMSF.')
    predict_parser.add_argument('--output_dir', type=str, default='predictions', help='Directory where prediction results (CSV, plots, log file) will be saved.')
    predict_parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing sequences during prediction (adjust based on GPU memory).')
    predict_parser.add_argument('--max_length', type=int, default=None, help='Optional: Filter out sequences longer than this length before prediction.')
    predict_parser.add_argument('--plot_predictions', action=argparse.BooleanOptionalAction, default=True, help='Generate individual RMSF plots for each predicted sequence.')
    predict_parser.add_argument('--smoothing_window', type=int, default=1, help='Window size for moving average smoothing on the generated plots (1 means no smoothing).')


    # Parse arguments
    args = parser.parse_args()

    # --- Execute Command ---
    if args.command == 'process':
        logger.info(f"Initiating data processing command...")
        process_data(
            csv_path=args.csv,
            output_dir=args.output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed
        )
        logger.info("Data processing finished.")

    elif args.command == 'train':
        logger.info(f"Initiating model training command...")
        config_path = args.config
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded training configuration from {config_path}")
            # Call the train function from train.py, passing the loaded config dictionary
            train(config)
            logger.info("Training finished.")
        except FileNotFoundError:
            logger.error(f"Training configuration file not found: {config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing training configuration file {config_path}: {e}")
            sys.exit(1)
        except Exception as e:
             logger.error(f"An unexpected error occurred during training setup or execution: {e}", exc_info=True) # Log traceback
             sys.exit(1)


    elif args.command == 'predict':
        logger.info(f"Initiating prediction command...")
        # Prepare the configuration dictionary for the predict function
        predict_config = {
            'model_checkpoint': args.model_checkpoint,
            'fasta_path': args.fasta_path,
            'output_dir': args.output_dir,
            'batch_size': args.batch_size,
            'max_length': args.max_length,
            'plot_predictions': args.plot_predictions,
            'smoothing_window': args.smoothing_window
            # Add other prediction-specific args here if needed
        }
        try:
            # Call the predict function from predict.py
            predict(predict_config)
            logger.info("Prediction finished.")
        except Exception as e:
             logger.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
             sys.exit(1)

    else:
        # This should not be reachable if subparsers `required=True`
        logger.error(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
