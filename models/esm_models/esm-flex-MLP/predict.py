import os
import torch
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import time

# Import the correct model
from model import ESMRegressionModel
# Import helper from dataset
from dataset import load_sequences_from_fasta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_gpu_memory(detail=False):
    """Log GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"GPU Memory: Allocated={allocated:.2f} MB, Reserved={reserved:.2f} MB")
        if detail:
            logger.info(torch.cuda.memory_summary())


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[Optional[ESMRegressionModel], Optional[Dict]]:
    """
    Load a trained ESMRegressionModel from a checkpoint file.

    Args:
        checkpoint_path: Path to the model checkpoint (.pt file).
        device: The device ('cuda' or 'cpu') to load the model onto.

    Returns:
        A tuple containing (loaded_model, config_from_checkpoint).
        Returns (None, None) if loading fails.
    """
    logger.info(f"Loading model checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return None, None

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        logger.info(f"Checkpoint loaded successfully.")
    except Exception as e:
         logger.error(f"Failed to load checkpoint file {checkpoint_path}: {e}", exc_info=True)
         return None, None

    # --- Verify checkpoint structure and extract config ---
    required_keys = ['config', 'model_state_dict', 'epoch']
    if not all(key in checkpoint for key in required_keys):
         logger.error(f"Checkpoint file {checkpoint_path} is missing required keys ({required_keys}). Found keys: {list(checkpoint.keys())}")
         return None, None

    config_from_ckpt = checkpoint['config']
    logger.info("Config loaded from checkpoint.")

    # --- Recreate model instance based on config ---
    try:
        logger.info(f"Recreating model architecture based on checkpoint config:")
        logger.info(f"  ESM Version: {config_from_ckpt['model']['esm_version']}")
        logger.info(f"  Regression Hidden Dim: {config_from_ckpt['model']['regression']['hidden_dim']}")
        logger.info(f"  Regression Dropout: {config_from_ckpt['model']['regression']['dropout']}")

        model = ESMRegressionModel(
            esm_model_name=config_from_ckpt['model']['esm_version'],
            regression_hidden_dim=config_from_ckpt['model']['regression']['hidden_dim'],
            regression_dropout=config_from_ckpt['model']['regression']['dropout']
        )
        logger.info("Model instance created.")
    except KeyError as e:
         logger.error(f"Missing expected key in checkpoint config: {e}")
         return None, None
    except Exception as e:
         logger.error(f"Error creating model instance from config: {e}", exc_info=True)
         return None, None

    # --- Load model state dictionary ---
    try:
        # Load weights, be flexible with strict=False first
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing_keys:
             logger.warning(f"State dict missing keys: {missing_keys}")
        if unexpected_keys:
             logger.warning(f"State dict has unexpected keys: {unexpected_keys}")
        # Optionally, try strict=True if needed, but strict=False is often better for compatibility
        # model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        logger.info(f"Model weights loaded into recreated structure.")

    except Exception as e:
         logger.error(f"Error loading state_dict into model: {e}", exc_info=True)
         return None, None

    # --- Final setup ---
    model = model.to(device)
    model.eval() # Set to evaluation mode

    logger.info(f"Model successfully loaded and transferred to {device}.")
    logger.info(f"  Model trained for {checkpoint['epoch']+1} epochs.")
    if 'val_loss' in checkpoint and 'val_corr' in checkpoint:
         logger.info(f"  Validation metrics at save: Loss={checkpoint['val_loss']:.6f}, Corr={checkpoint['val_corr']:.6f}")

    return model, config_from_ckpt


def group_sequences_by_length(sequences: Dict[str, str], batch_size: int, bucket_size: int = 50) -> List[List[Tuple[str, str]]]:
    """
    Groups sequences by length into buckets and then creates batches.
    Helps improve padding efficiency for transformer models during inference.

    Args:
        sequences: Dictionary mapping sequence IDs to amino acid sequences.
        batch_size: Target number of sequences per batch.
        bucket_size: Size of length ranges for grouping.

    Returns:
        List of batches, where each batch is a list of (sequence_id, sequence) tuples.
        Batches are sorted approximately by length (shortest first).
    """
    if not sequences:
        return []

    # Group by length bucket index
    length_buckets = defaultdict(list)
    for seq_id, seq in sequences.items():
        bucket_idx = len(seq) // bucket_size
        length_buckets[bucket_idx].append((seq_id, seq))

    # Create batches within each bucket, keeping buckets sorted by length
    all_batches = []
    for bucket_idx in sorted(length_buckets.keys()):
        bucket_items = length_buckets[bucket_idx]
        # Optional: Sort items within bucket by exact length (minor effect usually)
        # bucket_items.sort(key=lambda x: len(x[1]))
        for i in range(0, len(bucket_items), batch_size):
            batch = bucket_items[i : i + batch_size]
            all_batches.append(batch)

    logger.info(f"Grouped {len(sequences)} sequences into {len(all_batches)} batches using length bucketing.")
    return all_batches


def predict_rmsf(
    model: ESMRegressionModel,
    sequences: Dict[str, str],
    batch_size: int,
    device: torch.device,
    use_amp: bool = True # Enable/disable AMP for prediction
) -> Dict[str, np.ndarray]:
    """
    Predict RMSF values for a dictionary of sequences using the trained model.

    Args:
        model: The trained ESMRegressionModel instance.
        sequences: Dictionary mapping sequence IDs (str) to sequences (str).
        batch_size: Batch size for inference.
        device: Device ('cuda' or 'cpu') to run inference on.
        use_amp: Whether to use Automatic Mixed Precision for inference (GPU only).

    Returns:
        Dictionary mapping sequence IDs (str) to predicted RMSF values (NumPy array).
    """
    model.eval() # Ensure model is in evaluation mode

    if not sequences:
         logger.warning("No sequences provided for prediction.")
         return {}

    # Group sequences for efficient batching
    # Using a default bucket size, adjust if needed
    batches = group_sequences_by_length(sequences, batch_size, bucket_size=50)
    results = {}

    logger.info(f"Starting RMSF prediction for {len(sequences)} sequences...")
    prediction_start_time = time.time()

    autocast_device_type = device.type
    amp_enabled = (device.type == 'cuda' and use_amp)

    with torch.no_grad(): # Disable gradient calculations
        for batch_data in tqdm(batches, desc="Predicting", leave=False):
            batch_ids = [item[0] for item in batch_data]
            batch_seqs = [item[1] for item in batch_data] # List of sequence strings

            try:
                # Forward pass with optional AMP
                with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                    # Use the model's predict method which returns numpy arrays
                    batch_predictions_np = model.predict(batch_seqs)

                # Store results (already numpy arrays)
                if len(batch_predictions_np) == len(batch_ids):
                    for seq_id, pred_np in zip(batch_ids, batch_predictions_np):
                        results[seq_id] = pred_np
                else:
                     logger.error(f"Mismatch between batch IDs ({len(batch_ids)}) and predictions ({len(batch_predictions_np)}) count.")
                     # Attempt partial assignment if possible
                     for i, seq_id in enumerate(batch_ids):
                          if i < len(batch_predictions_np):
                              results[seq_id] = batch_predictions_np[i]
                          else:
                              logger.warning(f"No prediction found for sequence ID: {seq_id}")


            except Exception as e:
                 logger.error(f"Error during prediction for batch starting with ID {batch_ids[0]}: {e}", exc_info=True)
                 logger.warning("Skipping batch due to error.")
                 # Add placeholder or skip IDs in this batch based on requirements
                 # for seq_id in batch_ids: results[seq_id] = np.array([]) # Example: empty array on error
                 continue


            # Optional: Clear CUDA cache periodically for very large models/sequences
            if device.type == 'cuda':
                 if len(results) % (5 * batch_size) == 0: # Every 5 batches approx
                      torch.cuda.empty_cache()

    prediction_duration = time.time() - prediction_start_time
    num_predicted = len(results)
    logger.info(f"Prediction completed for {num_predicted} sequences in {prediction_duration:.2f}s.")
    if num_predicted > 0:
         logger.info(f"Average time per sequence: {prediction_duration / num_predicted:.4f}s")

    return results


def plot_rmsf(
    sequence: str,
    predictions: np.ndarray,
    title: str,
    output_path: str,
    window_size: int = 1,
    figsize: Tuple[int, int] = (15, 6) # Wider plot
):
    """
    Plot predicted RMSF values against residue position for a single protein.

    Args:
        sequence: The amino acid sequence string (used for length and labels).
        predictions: NumPy array of predicted RMSF values.
        title: Title for the plot (usually the sequence ID).
        output_path: Full path to save the plot image file (e.g., 'plots/protein_id.png').
        window_size: Window size for optional moving average smoothing (1 = no smoothing).
        figsize: Dimensions of the plot figure.
    """
    if predictions is None or len(predictions) == 0:
        logger.warning(f"No prediction data to plot for {title}. Skipping plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=figsize)

    # Determine residue indices based on prediction length
    # Prediction length might differ from original sequence length due to tokenization/truncation
    pred_len = len(predictions)
    residue_indices = np.arange(1, pred_len + 1)

    # Apply smoothing if requested
    if window_size > 1:
        # Use pandas rolling average for robustness (handles edges better)
        s = pd.Series(predictions)
        smoothed_predictions = s.rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()
        plot_data = smoothed_predictions
        plot_label = f'RMSF Prediction (Smoothed, window={window_size})'
    else:
        plot_data = predictions
        plot_label = 'RMSF Prediction'

    # Plot RMSF values
    plt.plot(residue_indices, plot_data, '-', color='dodgerblue', linewidth=2, label=plot_label)

    # Add annotations and labels
    plt.xlabel('Residue Position')
    plt.ylabel('Predicted RMSF')
    plt.title(f'Predicted RMSF for {title} (Length: {pred_len})')
    plt.xlim(0, pred_len + 1) # Set x-axis limits
    plt.grid(True, linestyle=':', alpha=0.7)

    # Add basic statistics to the plot
    mean_rmsf = np.mean(predictions) # Use original predictions for stats
    max_rmsf = np.max(predictions)
    min_rmsf = np.min(predictions)
    median_rmsf = np.median(predictions)
    stats_text = (f'Mean: {mean_rmsf:.3f}\n'
                  f'Median: {median_rmsf:.3f}\n'
                  f'Min: {min_rmsf:.3f}\n'
                  f'Max: {max_rmsf:.3f}')
    # Place stats box using axes coordinates (0=left/bottom, 1=right/top)
    plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.4))


    # Optional: Highlight highly flexible regions (e.g., > 90th percentile)
    try:
         threshold = np.percentile(predictions, 90)
         plt.axhline(y=threshold, color='red', linestyle='--', linewidth=1, alpha=0.6, label=f'90th Percentile ({threshold:.3f})')
         # Optionally fill above threshold
         # plt.fill_between(residue_indices, plot_data, threshold, where=plot_data >= threshold,
         #                  color='red', alpha=0.1, interpolate=True)
    except IndexError: # Handles empty predictions case if not caught earlier
         pass

    plt.legend(loc='upper right')
    plt.tight_layout()

    # Ensure output directory exists and save plot
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight') # Use moderate DPI for file size
    except Exception as e:
        logger.error(f"Failed to save plot to {output_path}: {e}")
    finally:
        plt.close() # Close the figure to release memory


def save_predictions(predictions: Dict[str, np.ndarray], output_path: str):
    """
    Save predictions to a CSV file.

    Args:
        predictions: Dictionary mapping sequence IDs to predicted RMSF (NumPy arrays).
        output_path: Path to the output CSV file.
    """
    if not predictions:
        logger.warning("No predictions to save.")
        return

    data_to_save = []
    for domain_id, rmsf_values in predictions.items():
        if rmsf_values is None or len(rmsf_values) == 0:
            logger.warning(f"Skipping empty prediction for {domain_id} in CSV output.")
            continue
        # Generate 1-based residue indices corresponding to the predictions
        for i, rmsf in enumerate(rmsf_values):
            data_to_save.append({
                'domain_id': domain_id,
                'resid': i + 1,  # 1-based residue index for the prediction
                'rmsf_pred': rmsf
            })

    if not data_to_save:
        logger.warning("No valid prediction data points found to save in CSV.")
        return

    # Convert to DataFrame and save
    try:
        df = pd.DataFrame(data_to_save)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, float_format='%.6f') # Format float precision
        logger.info(f"Predictions saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save predictions DataFrame to {output_path}: {e}")


def predict(config: Dict[str, Any]):
    """
    Main prediction function orchestrating loading, prediction, saving, and plotting.

    Args:
        config: Dictionary containing prediction settings, typically derived from
                command-line arguments or a config file section. Expected keys:
                'model_checkpoint', 'fasta_path', 'output_dir', 'batch_size',
                'max_length' (Optional), 'plot_predictions' (bool), 'smoothing_window'.
    """
    predict_start_time = time.time()

    # --- Setup Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log_gpu_memory()


    # --- Output Directory & Logging ---
    output_dir = config.get('output_dir', 'predictions')
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, 'prediction.log')
    # Remove existing handlers for this file to avoid duplicates
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_path:
            logger.removeHandler(handler)
            handler.close()
    file_handler = logging.FileHandler(log_path, mode='w') # Overwrite log for new prediction run
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info("--- Starting Prediction Run ---")
    logger.info(f"Prediction config: {config}")


    # --- Load Model ---
    model_checkpoint = config.get('model_checkpoint')
    if not model_checkpoint:
        logger.critical("Model checkpoint path ('model_checkpoint') not provided in config.")
        return

    model, model_config_from_ckpt = load_model(model_checkpoint, device)
    if model is None:
        logger.critical(f"Failed to load model from {model_checkpoint}. Aborting prediction.")
        return
    # Log memory after loading model
    if device.type == 'cuda': log_gpu_memory()


    # --- Load Sequences ---
    fasta_path = config.get('fasta_path')
    if not fasta_path:
        logger.critical("Input FASTA file path ('fasta_path') not provided in config.")
        return

    try:
        sequences = load_sequences_from_fasta(fasta_path)
        if not sequences:
             logger.critical(f"No sequences found in FASTA file: {fasta_path}. Aborting.")
             return
        logger.info(f"Loaded {len(sequences)} sequences from {fasta_path}")
        # Log sequence length stats
        seq_lengths = [len(s) for s in sequences.values()]
        logger.info(f"  Sequence length stats: Min={np.min(seq_lengths)}, Max={np.max(seq_lengths)}, "
                    f"Mean={np.mean(seq_lengths):.1f}, Median={np.median(seq_lengths):.1f}")

    except FileNotFoundError:
         logger.critical(f"FASTA file not found: {fasta_path}")
         return
    except Exception as e:
         logger.critical(f"Error loading sequences from {fasta_path}: {e}", exc_info=True)
         return


    # --- Filter Sequences by Max Length (Optional) ---
    max_length = config.get('max_length')
    if max_length is not None and isinstance(max_length, int) and max_length > 0:
        original_count = len(sequences)
        sequences = {seq_id: seq for seq_id, seq in sequences.items() if len(seq) <= max_length}
        filtered_count = len(sequences)
        if filtered_count < original_count:
            logger.info(f"Filtered out {original_count - filtered_count} sequences longer than max_length ({max_length}).")
        if not sequences:
            logger.critical(f"No sequences remaining after filtering by max_length={max_length}. Aborting.")
            return


    # --- Predict RMSF ---
    prediction_batch_size = config.get('batch_size', 8) # Use prediction-specific batch size
    use_amp_predict = config.get('use_amp', True) and (device.type == 'cuda') # AMP only on CUDA

    predictions = predict_rmsf(
        model,
        sequences,
        prediction_batch_size,
        device,
        use_amp=use_amp_predict
    )

    if not predictions:
         logger.warning("Prediction step resulted in an empty dictionary. No results to save or plot.")
         # Continue to log completion time, but skip saving/plotting
    else:
         # --- Save Predictions ---
         output_csv_path = os.path.join(output_dir, 'predictions.csv')
         save_predictions(predictions, output_csv_path)

         # --- Plot Predictions (Optional) ---
         if config.get('plot_predictions', True):
             plots_dir = os.path.join(output_dir, 'plots')
             os.makedirs(plots_dir, exist_ok=True)
             smoothing_window = config.get('smoothing_window', 1)

             logger.info(f"Generating plots for {len(predictions)} predicted proteins in {plots_dir}...")
             plot_count = 0
             # Iterate through the predictions dict to only plot what was predicted
             for domain_id, pred_array in tqdm(predictions.items(), desc="Plotting", leave=False):
                  if domain_id in sequences: # Check if original sequence exists for plotting context
                      try:
                          plot_rmsf(
                              sequence=sequences[domain_id], # Original sequence
                              predictions=pred_array,       # Predicted RMSF array
                              title=domain_id,
                              output_path=os.path.join(plots_dir, f'{domain_id}.png'),
                              window_size=smoothing_window
                          )
                          plot_count += 1
                      except Exception as e:
                          logger.error(f"Failed to generate plot for {domain_id}: {e}", exc_info=True)
                  else:
                      logger.warning(f"Cannot plot for {domain_id}: Original sequence not found (might have been filtered).")
             logger.info(f"Generated {plot_count} plots.")


    # --- Finalize ---
    predict_end_time = time.time()
    logger.info(f"--- Prediction Run Finished ---")
    logger.info(f"Total prediction time: {predict_end_time - predict_start_time:.2f} seconds.")
    logger.info(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict RMSF using a trained ESM-3 Regression model')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to the trained model checkpoint (.pt file)')
    parser.add_argument('--fasta_path', type=str, required=True, help='Path to the input FASTA file containing sequences to predict')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Directory to save prediction results (CSV, plots, log)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for prediction (adjust based on GPU memory)')
    parser.add_argument('--max_length', type=int, default=None, help='Optional: Filter out sequences longer than this *before* prediction.')
    parser.add_argument('--plot_predictions', action=argparse.BooleanOptionalAction, default=True, help='Generate plots for each prediction (default: True, use --no-plot-predictions to disable)')
    parser.add_argument('--smoothing_window', type=int, default=1, help='Window size for moving average smoothing on plots (1 = no smoothing)')
    # Add AMP toggle if needed
    # parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=True, help='Use Automatic Mixed Precision (AMP) for prediction (GPU only)')

    args = parser.parse_args()

    # Convert args Namespace to dictionary for the predict function
    config_dict = vars(args)

    # Run the prediction process
    predict(config_dict)
