# === FILE: predict.py ===
import os
import torch
import torch.nn as nn # <--- ADDED THIS IMPORT
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
from collections import defaultdict

# Import the correct model definition
from model import ESMRegressionModelWithLoRA
# Import helpers from dataset
from dataset import load_sequences_from_fasta

# PEFT for loading LoRA adapters
try:
    from peft import PeftModel, PeftConfig
    peft_available = True
except ImportError:
    logging.error("PEFT library not found. Cannot load LoRA models.")
    peft_available = False
    # Prediction might still work if loading a non-PEFT model, but unlikely for this project
    class PeftModel: pass # Dummy
    class PeftConfig: pass # Dummy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(name)s] - %(message)s', force=True)
logger = logging.getLogger(__name__)

def log_gpu_memory(detail=False):
    """Log GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2; reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"GPU Memory: Allocated={allocated:.1f}MB, Reserved={reserved:.1f}MB")
        if detail: logger.info(torch.cuda.memory_summary())

def load_model_for_prediction(checkpoint_dir: str, device: torch.device) -> Optional[ESMRegressionModelWithLoRA]:
    """Load a trained model for prediction, handling PEFT checkpoints."""
    logger.info(f"Loading model for prediction from directory: {checkpoint_dir}")
    if not os.path.isdir(checkpoint_dir): logger.error(f"Checkpoint dir not found: {checkpoint_dir}"); return None

    # --- Load Configuration ---
    config_from_ckpt = None
    training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if os.path.exists(training_state_path):
        try:
            # Load state dict on CPU first to avoid GPU memory issues if model is large
            training_state = torch.load(training_state_path, map_location='cpu')
            config_from_ckpt = training_state.get('config')
            if config_from_ckpt: logger.info("Loaded training config from training_state.pt")
            else: logger.warning("Found training_state.pt but 'config' key is missing.")
        except Exception as e: logger.warning(f"Could not load training_state.pt: {e}. Trying PEFT config.")

    adapter_config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    is_peft_model = os.path.exists(adapter_config_path)

    if not config_from_ckpt and is_peft_model:
        try:
            peft_config = PeftConfig.from_pretrained(checkpoint_dir)
            config_from_ckpt = {'model': {'esm_version': peft_config.base_model_name_or_path,
                                          'lora': {'enabled': True},
                                          'regression': {}}} # Minimal config
            logger.info(f"Loaded base model name '{config_from_ckpt['model']['esm_version']}' from adapter_config.json")
        except Exception as e: logger.error(f"Failed to load PeftConfig: {e}"); return None

    if not config_from_ckpt: logger.error("Cannot determine model config from checkpoint files."); return None
    if 'model' not in config_from_ckpt or 'esm_version' not in config_from_ckpt['model']:
         logger.error("Loaded config is incomplete (missing model/esm_version)."); return None

    # --- Instantiate Base Model Structure ---
    try:
        temp_config = config_from_ckpt.copy()
        temp_config['model']['lora'] = temp_config['model'].get('lora', {})
        temp_config['model']['lora']['enabled'] = False # Create base + head first
        model = ESMRegressionModelWithLoRA(temp_config)
        logger.info("Base model and head structure created.")
    except Exception as e: logger.error(f"Error creating model structure: {e}", exc_info=True); return None

    # --- Load Weights (PEFT or Full) ---
    if is_peft_model:
        if not peft_available: logger.error("PEFT needed to load adapters but not installed."); return None
        logger.info("Loading PEFT LoRA adapters and head weights...")
        try:
            # Load onto the created structure (model). device_map can help with large models.
            model = PeftModel.from_pretrained(model, checkpoint_dir) #, device_map="auto")
            logger.info(f"PEFT adapters/head loaded successfully from {checkpoint_dir}")
        except Exception as e: logger.error(f"Error loading PEFT model: {e}", exc_info=True); return None
    else: # Handle non-PEFT loading if necessary (less likely path for this project)
        logger.warning("Attempting to load as a non-PEFT model.")
        full_ckpt_path = os.path.join(checkpoint_dir, "full_model_checkpoint.pt") # Adjust name if needed
        if os.path.exists(full_ckpt_path):
             try:
                  state_dict = torch.load(full_ckpt_path, map_location='cpu')['model_state_dict']
                  model.load_state_dict(state_dict)
                  logger.info("Loaded full model state dict.")
             except Exception as e: logger.error(f"Error loading full state dict: {e}"); return None
        else: logger.error("Cannot load non-PEFT model: Checkpoint file not found."); return None

    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded to {device} and set to eval mode.")
    return model

def group_sequences_by_length(sequences: Dict[str, str], batch_size: int, bucket_size: int = 50) -> List[List[Tuple[str, str]]]:
    """Groups sequences by length into buckets and then creates batches."""
    if not sequences: return []
    length_buckets = defaultdict(list); seq_items = list(sequences.items())
    seq_items.sort(key=lambda item: len(item[1])) # Sort by length
    for seq_id, seq in seq_items:
        bucket_idx = len(seq) // bucket_size
        length_buckets[bucket_idx].append((seq_id, seq))
    all_batches = []
    for bucket_idx in sorted(length_buckets.keys()): # Process buckets by length
        bucket_items = length_buckets[bucket_idx]
        for i in range(0, len(bucket_items), batch_size):
            all_batches.append(bucket_items[i : i + batch_size])
    logger.info(f"Grouped {len(sequences)} sequences into {len(all_batches)} batches.")
    return all_batches

@torch.no_grad()
def predict_rmsf(model: nn.Module, sequences: Dict[str, str], batch_size: int, device: torch.device) -> Dict[str, np.ndarray]:
    """Predict RMSF values for sequences."""
    model.eval(); results = {}; 
    if not sequences: return {}
    batches = group_sequences_by_length(sequences, batch_size)
    logger.info(f"Starting RMSF prediction for {len(sequences)} sequences...")
    start_time = time.time()

    for batch_data in tqdm(batches, desc="Predicting", leave=False):
        batch_ids = [item[0] for item in batch_data]
        batch_seqs = [item[1] for item in batch_data]
        try:
            batch_predictions_np = model.predict(batch_seqs) # Model's predict method
            if len(batch_predictions_np) == len(batch_ids):
                for seq_id, pred_np in zip(batch_ids, batch_predictions_np): results[seq_id] = pred_np
            else: logger.error(f"Prediction mismatch: {len(batch_predictions_np)} vs {len(batch_ids)} IDs.")
        except Exception as e: logger.error(f"Batch prediction error: {e}", exc_info=True); continue

    duration = time.time() - start_time
    logger.info(f"Prediction completed for {len(results)} sequences in {duration:.2f}s.")
    if results: logger.info(f"Avg time/seq: {duration / len(results):.4f}s")
    return results

def plot_rmsf(predictions: np.ndarray, title: str, output_path: str, window_size: int = 1, figsize: Tuple[int, int] = (15, 6)):
    """Plot predicted RMSF values."""
    if predictions is None or predictions.size == 0: logger.warning(f"No plot data for {title}"); return
    plt.style.use('seaborn-v0_8-whitegrid'); fig, ax = plt.subplots(figsize=figsize)
    pred_len = len(predictions); indices = np.arange(1, pred_len + 1)
    data = predictions; label = 'RMSF Prediction'
    if window_size > 1:
        data = pd.Series(predictions).rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()
        label = f'RMSF Prediction (Smoothed, w={window_size})'
    ax.plot(indices, data, '-', color='dodgerblue', linewidth=1.5, label=label)
    ax.set_xlabel('Residue Position'); ax.set_ylabel('Predicted RMSF'); ax.set_title(f'Predicted RMSF for {title} (Length: {pred_len})')
    ax.set_xlim(0, pred_len + 1); ax.grid(True, linestyle=':', alpha=0.7)
    try: # Add stats
        stats = {'Mean': np.nanmean(predictions), 'Median': np.nanmedian(predictions), 'Min': np.nanmin(predictions), 'Max': np.nanmax(predictions)}
        stats_text = "\n".join([f"{k}: {v:.3f}" for k,v in stats.items()])
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=8, va='top', ha='right', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.4))
        p90 = np.nanpercentile(predictions, 90)
        ax.axhline(y=p90, color='red', linestyle='--', lw=1, alpha=0.6, label=f'90th Perc. ({p90:.3f})')
    except Exception: pass
    ax.legend(loc='upper left'); plt.tight_layout()
    try: os.makedirs(os.path.dirname(output_path), exist_ok=True); plt.savefig(output_path, dpi=120, bbox_inches='tight')
    except Exception as e: logger.error(f"Failed save plot {output_path}: {e}")
    plt.close(fig)

def save_predictions(predictions: Dict[str, np.ndarray], output_path: str):
    """Save predictions to CSV."""
    if not predictions: logger.warning("No predictions to save."); return
    data = [{'domain_id': did, 'resid': i + 1, 'rmsf_pred': rmsf}
            for did, rmsf_arr in predictions.items() if rmsf_arr is not None and rmsf_arr.size > 0
            for i, rmsf in enumerate(rmsf_arr)]
    if not data: logger.warning("No valid prediction data points to save."); return
    try:
        df = pd.DataFrame(data); os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, float_format='%.6f')
        logger.info(f"Predictions saved to {output_path}")
    except Exception as e: logger.error(f"Failed save predictions CSV {output_path}: {e}")

def predict(config: Dict[str, Any]):
    """Main prediction function."""
    start_time = time.time(); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = config.get('output_dir', 'predictions'); os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'prediction.log')
    # Ensure exclusive file handler for this run
    root_logger = logging.getLogger()
    # Remove previous handlers associated with this file path
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_path:
             handler.close()
             root_logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_path, mode='w'); file_handler.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s-%(message)s'))
    root_logger.addHandler(file_handler) # Add handler to root logger

    logger.info("--- Starting Prediction Run ---"); logger.info(f"Using device: {device}")
    logger.info(f"Prediction config: {config}")
    if device.type == 'cuda': log_gpu_memory()

    model_checkpoint_dir = config.get('model_checkpoint'); fasta_path = config.get('fasta_path')
    if not model_checkpoint_dir or not fasta_path: logger.critical("Missing model checkpoint dir or fasta path."); return

    model = load_model_for_prediction(model_checkpoint_dir, device)
    if model is None: logger.critical("Failed load model."); return
    if device.type == 'cuda': log_gpu_memory()

    try: sequences = load_sequences_from_fasta(fasta_path); assert sequences
    except Exception as e: logger.critical(f"Failed load FASTA: {e}"); return

    max_length = config.get('max_length')
    if max_length:
        count_before = len(sequences)
        sequences = {k: v for k, v in sequences.items() if len(v) <= max_length}
        logger.info(f"Filtered {count_before - len(sequences)} sequences > {max_length} residues.")
        if not sequences: logger.critical("No sequences left after filter."); return

    predictions = predict_rmsf(model, sequences, config.get('batch_size', 8), device)

    if predictions:
        output_csv = os.path.join(output_dir, 'predictions.csv')
        save_predictions(predictions, output_csv)
        if config.get('plot_predictions', True):
            plots_dir = os.path.join(output_dir, 'plots')
            window = config.get('smoothing_window', 1)
            logger.info(f"Generating plots in {plots_dir} (Smooth={window})...")
            for did, pred_arr in tqdm(predictions.items(), desc="Plotting", leave=False):
                if did in sequences: plot_rmsf(pred_arr, did, os.path.join(plots_dir, f'{did}.png'), window)
            logger.info("Plotting complete.")

    end_time = time.time()
    logger.info(f"--- Prediction Run Finished ({end_time - start_time:.2f}s) ---")
    logger.info(f"Results saved in: {output_dir}")
    # Remove handler at the end to avoid interference if script is run multiple times in one session
    root_logger.removeHandler(file_handler)
    file_handler.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict RMSF using trained ESM-C+LoRA model.')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to *directory* containing saved model/adapter files.')
    parser.add_argument('--fasta_path', type=str, required=True, help='Input FASTA file.')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Directory for results.')
    parser.add_argument('--batch_size', type=int, default=8, help='Prediction batch size.')
    parser.add_argument('--max_length', type=int, default=None, help='Optional: Max sequence length filter.')
    parser.add_argument('--plot_predictions', action=argparse.BooleanOptionalAction, default=True, help='Generate plots.')
    parser.add_argument('--smoothing_window', type=int, default=1, help='Smoothing window for plots (1=none).')
    args = parser.parse_args(); predict(vars(args))