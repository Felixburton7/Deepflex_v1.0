#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
import logging
import torch # Import torch early to check availability

# Set project root directory relative to this script file
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Add project root to Python path to allow importing modules
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Setup logging (configure root logger)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(name)s] - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__) # Get logger for this script

# Now import project modules with error handling
try:
    from data_processor import process_data
    from train import train
    from predict import predict
except ImportError as e:
     logger.critical(f"Error: Failed to import project modules. Is the script run from the project root?", exc_info=True)
     sys.exit(1)
except Exception as e:
     logger.critical(f"An unexpected error occurred during initial imports: {e}", exc_info=True)
     sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='ESM-Flex (ESM-C + LoRA): Protein Flexibility (RMSF) Prediction Pipeline.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run', required=True)

    # Process data command
    process_parser = subparsers.add_parser('process', help='Process raw RMSF CSV data.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    process_parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML config file.')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the ESM-C LoRA model.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train_parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML config file.')
    # Note: Use 'accelerate launch main.py train ...' for multi-GPU

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict RMSF using a trained model.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    predict_parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to trained model *directory*.')
    predict_parser.add_argument('--fasta_path', type=str, required=True, help='Input FASTA file.')
    predict_parser.add_argument('--output_dir', type=str, default='predictions', help='Output directory for predictions.')
    predict_parser.add_argument('--batch_size', type=int, default=8, help='Prediction batch size.')
    predict_parser.add_argument('--max_length', type=int, default=None, help='Max sequence length for prediction.')
    predict_parser.add_argument('--plot_predictions', action=argparse.BooleanOptionalAction, default=True, help='Generate RMSF plots.')
    predict_parser.add_argument('--smoothing_window', type=int, default=1, help='Smoothing window for plots.')

    args = parser.parse_args()

    # --- Load Config for Process/Train ---
    config = None
    if args.command in ['process', 'train']:
        config_path = args.config
        try:
            with open(config_path, 'r') as f: config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e: logger.critical(f"Error loading config '{config_path}': {e}", exc_info=True); sys.exit(1)
        if not config: logger.critical("Config file is empty or invalid."); sys.exit(1)

    # --- Execute Command ---
    try:
        if args.command == 'process':
            logger.info("Initiating data processing...")
            if not process_data(config): sys.exit(1) # process_data returns bool
            logger.info("Data processing finished.")
        elif args.command == 'train':
            logger.info("Initiating model training...")
            if 'ACCELERATE_PROCESS_ID' not in os.environ and torch.cuda.is_available() and torch.cuda.device_count() > 1:
                 logger.warning("Multiple GPUs detected but not using 'accelerate launch'. Training may be suboptimal.")
            train(config) # Train function handles detailed logging and potential errors
            logger.info("Training script finished.")
        elif args.command == 'predict':
            logger.info("Initiating prediction...")
            predict_config = vars(args) # Use predict's args directly
            predict(predict_config)
            logger.info("Prediction script finished.")
        else: # Should be unreachable
            logger.error(f"Unknown command: {args.command}"); parser.print_help(); sys.exit(1)
    except Exception as e:
         # Catch errors raised from the called functions
         logger.critical(f"A critical error occurred during command execution: {e}", exc_info=True)
         sys.exit(1)

if __name__ == "__main__":
    main()
