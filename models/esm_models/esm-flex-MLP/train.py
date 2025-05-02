import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
import logging
import time
from pathlib import Path

from dataset import create_length_batched_dataloader
from model import ESMRegressionModel 

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set seed for reproducibility across libraries."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Potentially make things slower, but more reproducible
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")

def log_gpu_memory(detail=False):
    """Log GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"GPU Memory: Allocated={allocated:.2f} MB, Reserved={reserved:.2f} MB")
        if detail:
            logger.info(torch.cuda.memory_summary())


def train_epoch(model, dataloader, optimizer, device, accumulation_steps=1, max_gradient_norm=1.0):
    """
    Train the model for one epoch.

    Args:
        model: The ESMRegressionModel instance.
        dataloader: DataLoader providing training batches.
        optimizer: The optimizer instance.
        device: The device to train on ('cuda' or 'cpu').
        accumulation_steps: Number of steps to accumulate gradients over.
        max_gradient_norm: Maximum norm for gradient clipping (0 to disable).

    Returns:
        Tuple of (average epoch loss, average epoch correlation).
    """
    model.train() # Set model to training mode (enables dropout in head)
    total_loss = 0.0
    total_corr = 0.0
    num_samples_processed = 0 # Track number of sequences processed

    # Use torch.cuda.amp for mixed precision if on GPU
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    autocast_device_type = device.type # 'cuda' or 'cpu'

    # Reset gradients at the beginning of the epoch
    optimizer.zero_grad(set_to_none=True)

    epoch_start_time = time.time()
    batch_iterator = tqdm(dataloader, desc="Training", leave=False)
    for i, batch in enumerate(batch_iterator):
        sequences = batch['sequences'] # List of sequence strings
        # Targets are already tensors from collate_fn, ensure correct type if needed
        rmsf_values = [t.to(torch.float32) for t in batch['rmsf_values']]

        current_batch_size = len(sequences)
        if current_batch_size == 0: continue # Skip empty batches if they somehow occur

        try:
            # Forward pass with Automatic Mixed Precision (AMP)
            with torch.amp.autocast(device_type=autocast_device_type, enabled=(scaler is not None)):
                outputs = model(sequences=sequences, rmsf_values=rmsf_values)
                loss = outputs['loss']

                # Check for invalid loss
                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Batch {i}: Invalid loss detected ({loss}). Skipping batch.")
                    # Crucially, ensure gradients are cleared if skipping optimizer step later
                    if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                        optimizer.zero_grad(set_to_none=True)
                    continue # Move to next batch

                # Normalize loss for gradient accumulation
                loss = loss / accumulation_steps

            # Backward pass (gradient computation)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # --- Gradient Accumulation & Optimizer Step ---
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                if scaler is not None:
                    # Unscale gradients before clipping
                    if max_gradient_norm > 0:
                        scaler.unscale_(optimizer) # Unscales the gradients of optimizer's assigned params
                        torch.nn.utils.clip_grad_norm_(
                            (p for p in model.parameters() if p.requires_grad), # Clip only trainable params
                            max_gradient_norm
                        )
                    # Optimizer step - scaler implicitly checks for inf/NaN gradients
                    scaler.step(optimizer)
                    # Update the scaler for next iteration
                    scaler.update()
                else:
                    # Clip gradients if not using scaler
                    if max_gradient_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            (p for p in model.parameters() if p.requires_grad),
                            max_gradient_norm
                        )
                    optimizer.step()

                # Reset gradients for the next accumulation cycle or batch
                optimizer.zero_grad(set_to_none=True)

            # Update cumulative metrics
            # Use loss.item() * accumulation_steps to get the non-normalized loss for this step
            total_loss += loss.item() * accumulation_steps * current_batch_size
            correlation = outputs['metrics'].get('pearson_correlation', 0.0)
            if not np.isnan(correlation): # Avoid adding NaN correlations
                total_corr += correlation * current_batch_size
            num_samples_processed += current_batch_size

            # Update progress bar
            avg_loss = total_loss / num_samples_processed if num_samples_processed > 0 else 0.0
            avg_corr = total_corr / num_samples_processed if num_samples_processed > 0 else 0.0
            batch_iterator.set_postfix(
                loss=f"{avg_loss:.4f}",
                corr=f"{avg_corr:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}" # Show learning rate
            )

        except Exception as e:
             logger.error(f"Error during training batch {i}: {e}", exc_info=True)
             logger.warning("Skipping batch due to error.")
             # Attempt to clear gradients if error occurred mid-accumulation
             optimizer.zero_grad(set_to_none=True)
             if device.type == 'cuda': torch.cuda.empty_cache() # Try to free memory
             continue # Skip to the next batch

        # Optional: Periodic memory logging
        # if i % 50 == 0 and device.type == 'cuda': log_gpu_memory()


    epoch_duration = time.time() - epoch_start_time
    logger.info(f"Training epoch duration: {epoch_duration:.2f}s")

    # Calculate final epoch averages
    final_avg_loss = total_loss / num_samples_processed if num_samples_processed > 0 else 0.0
    final_avg_corr = total_corr / num_samples_processed if num_samples_processed > 0 else 0.0

    return final_avg_loss, final_avg_corr


def validate(model, dataloader, device):
    """
    Validate the model on the validation set.

    Args:
        model: The ESMRegressionModel instance.
        dataloader: DataLoader providing validation batches.
        device: The device to run validation on.

    Returns:
        Tuple of (average validation loss, average validation correlation).
    """
    model.eval() # Set model to evaluation mode (disables dropout, etc.)
    total_loss = 0.0
    total_corr = 0.0
    num_samples_processed = 0
    domain_correlations = {} # Store per-domain correlation

    # Use torch.cuda.amp for mixed precision inference as well
    autocast_device_type = device.type

    epoch_start_time = time.time()
    batch_iterator = tqdm(dataloader, desc="Validation", leave=False)
    with torch.no_grad(): # Disable gradient calculations for validation
        for batch in batch_iterator:
            sequences = batch['sequences']
            domain_ids = batch['domain_ids']
            rmsf_values = [t.to(torch.float32) for t in batch['rmsf_values']]

            current_batch_size = len(sequences)
            if current_batch_size == 0: continue

            try:
                # Forward pass with AMP
                with torch.amp.autocast(device_type=autocast_device_type, enabled=(device.type == 'cuda')):
                    outputs = model(sequences=sequences, rmsf_values=rmsf_values)
                    loss = outputs['loss']

                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Validation: Invalid loss detected ({loss}). Skipping batch.")
                    continue

                # Update cumulative metrics
                total_loss += loss.item() * current_batch_size
                correlation = outputs['metrics'].get('pearson_correlation', 0.0)
                if not np.isnan(correlation):
                    total_corr += correlation * current_batch_size
                num_samples_processed += current_batch_size

                # Calculate and store per-domain correlations (on CPU for numpy)
                predictions_list = outputs['predictions'] # List of tensors
                for i, domain_id in enumerate(domain_ids):
                    if i < len(predictions_list): # Ensure prediction exists
                        pred_tensor = predictions_list[i].cpu() # Move to CPU
                        true_tensor = rmsf_values[i].cpu()      # Move target to CPU

                        min_len = min(len(pred_tensor), len(true_tensor))
                        if min_len > 1:
                            pred_np = pred_tensor[:min_len].numpy()
                            true_np = true_tensor[:min_len].numpy()

                            # Use model's static safe correlation method
                            corr_val = ESMRegressionModel.safe_pearson_correlation(
                                torch.from_numpy(pred_np), torch.from_numpy(true_np)
                            ).item()
                            domain_correlations[domain_id] = corr_val
                        else:
                            domain_correlations[domain_id] = 0.0 # Undefined for < 2 points

                # Update progress bar
                avg_loss = total_loss / num_samples_processed if num_samples_processed > 0 else 0.0
                avg_corr = total_corr / num_samples_processed if num_samples_processed > 0 else 0.0
                batch_iterator.set_postfix(loss=f"{avg_loss:.4f}", corr=f"{avg_corr:.4f}")

            except Exception as e:
                logger.error(f"Error during validation batch: {e}", exc_info=True)
                logger.warning("Skipping validation batch due to error.")
                continue # Skip to next batch


    epoch_duration = time.time() - epoch_start_time
    logger.info(f"Validation duration: {epoch_duration:.2f}s")

    # Log detailed correlation statistics from this epoch
    if domain_correlations:
         correlations = np.array(list(domain_correlations.values())) # Convert to numpy array
         correlations = correlations[~np.isnan(correlations)] # Remove NaNs if any slip through
         if len(correlations) > 0:
              logger.info(f"Per-Domain Validation Correlation stats (n={len(correlations)}):")
              logger.info(f"  Min: {np.min(correlations):.4f}, Max: {np.max(correlations):.4f}")
              logger.info(f"  Mean: {np.mean(correlations):.4f}, Median: {np.median(correlations):.4f}")
              logger.info(f"  Std Dev: {np.std(correlations):.4f}")
         else:
              logger.warning("No valid per-domain correlations calculated during validation.")
    else:
         logger.warning("No per-domain correlations were calculated during validation.")


    # Calculate final epoch averages
    final_avg_loss = total_loss / num_samples_processed if num_samples_processed > 0 else 0.0
    final_avg_corr = total_corr / num_samples_processed if num_samples_processed > 0 else 0.0

    return final_avg_loss, final_avg_corr


def save_model(model, optimizer, epoch, val_loss, val_corr, config, save_path):
    """
    Save model checkpoint, including state dict, optimizer state, epoch, metrics, and config.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Prepare the state to save
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_corr': val_corr,
        'config': config # Save the config used for this training run
    }

    # Save the checkpoint
    try:
        torch.save(checkpoint, save_path)
        logger.info(f"Model checkpoint saved to {save_path}")
    except Exception as e:
        logger.error(f"Error saving checkpoint to {save_path}: {e}")

def plot_metrics(train_losses, val_losses, train_corrs, val_corrs, save_dir, lr_values=None):
    """
    Plot training and validation loss and correlation over epochs.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- Loss Plot ---
    ax1.plot(epochs, train_losses, 'o-', color='royalblue', label='Train Loss', markersize=4)
    ax1.plot(epochs, val_losses, 's-', color='orangered', label='Validation Loss', markersize=4)
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    # Add text for best validation loss
    if val_losses:
        best_val_loss_epoch = np.argmin(val_losses)
        best_val_loss = val_losses[best_val_loss_epoch]
        ax1.annotate(f'Best Val Loss: {best_val_loss:.4f}\n(Epoch {best_val_loss_epoch+1})',
                     xy=(best_val_loss_epoch + 1, best_val_loss),
                     xytext=(10, 10), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                     fontsize=9, ha='left')


    # --- Correlation Plot ---
    ax2.plot(epochs, train_corrs, 'o-', color='royalblue', label='Train Correlation', markersize=4)
    ax2.plot(epochs, val_corrs, 's-', color='orangered', label='Validation Correlation', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Pearson Correlation')
    ax2.set_title('Training and Validation Correlation')
    ax2.legend(loc='lower left')
    ax2.grid(True, linestyle='--', alpha=0.6)
    # Add text for best validation correlation
    if val_corrs:
        best_val_corr_epoch = np.argmax(val_corrs)
        best_val_corr = val_corrs[best_val_corr_epoch]
        ax2.annotate(f'Best Val Corr: {best_val_corr:.4f}\n(Epoch {best_val_corr_epoch+1})',
                     xy=(best_val_corr_epoch + 1, best_val_corr),
                     xytext=(10, -20), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2"),
                     fontsize=9, ha='left')


    # --- Learning Rate Plot (Optional) ---
    if lr_values:
        ax3 = ax2.twinx() # Share x-axis with correlation plot
        ax3.plot(epochs, lr_values, 'd--', color='green', label='Learning Rate', markersize=3, alpha=0.7)
        ax3.set_ylabel('Learning Rate', color='green')
        ax3.tick_params(axis='y', labelcolor='green')
        # Set y-axis to logarithmic if LR changes significantly
        if len(set(lr_values)) > 2: # More than just initial and one drop
             ax3.set_yscale('log')
        ax3.legend(loc='lower right')


    plt.tight_layout(pad=1.5) # Add padding between subplots

    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, 'training_metrics.png')
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics plot saved to {plot_path}")
    except Exception as e:
        logger.error(f"Error saving metrics plot to {plot_path}: {e}")
    plt.close(fig) # Close the figure to free memory


def train(config):
    """
    Main training function, orchestrates the entire training process.
    """
    start_time_train = time.time()

    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log_gpu_memory() # Log initial memory

    set_seed(config['training']['seed'])

    model_save_dir = config['output']['model_dir']
    log_dir = os.path.join(model_save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    # Configure file logging handler
    log_path = os.path.join(log_dir, 'training.log')
    # Remove existing handlers for this file to avoid duplicates if re-running
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_path:
            logger.removeHandler(handler)
            handler.close()
    file_handler = logging.FileHandler(log_path, mode='a') # Append mode
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info("--- Starting New Training Run ---")
    logger.info(f"Configuration loaded: {config}")


    # --- Data Loaders ---
    logger.info("Creating data loaders...")
    try:
        train_dataloader = create_length_batched_dataloader(
            config['data']['data_dir'], 'train', config['training']['batch_size'],
            shuffle=True, max_length=config['training'].get('max_length'),
            length_bucket_size=config['training'].get('length_bucket_size', 50),
            num_workers=0 # Often 0 is fine unless I/O is bottleneck
        )
        val_dataloader = create_length_batched_dataloader(
            config['data']['data_dir'], 'val', config['training']['batch_size'],
            shuffle=False, max_length=config['training'].get('max_length'),
            length_bucket_size=config['training'].get('length_bucket_size', 50),
             num_workers=0
        )
        if not train_dataloader or not val_dataloader:
            logger.error("Failed to create one or both dataloaders. Aborting training.")
            return
    except Exception as e:
        logger.error(f"Error creating dataloaders: {e}", exc_info=True)
        return

    # --- Model ---
    logger.info("Creating model...")
    try:
        model = ESMRegressionModel(
            esm_model_name=config['model']['esm_version'],
            regression_hidden_dim=config['model']['regression']['hidden_dim'],
            regression_dropout=config['model']['regression']['dropout']
        )
        model = model.to(device)
        if device.type == 'cuda': log_gpu_memory() # Log memory after model load
    except Exception as e:
        logger.error(f"Error creating model: {e}", exc_info=True)
        return

    # --- Optimizer ---
    # Only optimize the regression head parameters (already frozen in model init)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
         logger.error("Model has no trainable parameters! Check model initialization.")
         return

    logger.info(f"Number of parameter tensors to optimize: {len(trainable_params)}")
    learning_rate = float(config['training']['learning_rate'])
    adam_epsilon = float(config['training'].get('adam_epsilon', 1e-8))
    weight_decay = float(config['training']['weight_decay'])

    optimizer = optim.AdamW(trainable_params, lr=learning_rate, eps=adam_epsilon, weight_decay=weight_decay)
    logger.info(f"Optimizer: AdamW (LR={learning_rate}, WeightDecay={weight_decay}, Epsilon={adam_epsilon})")


    # --- Scheduler ---
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5,
        patience=config['training']['scheduler_patience'],
        verbose=True, threshold=0.001 # Monitor validation correlation
    )
    logger.info(f"Scheduler: ReduceLROnPlateau (Patience={config['training']['scheduler_patience']}, Factor=0.5)")


    # --- Training Loop ---
    logger.info("--- Starting Training Loop ---")
    best_val_corr = -float('inf')
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses, train_corrs, val_corrs, lr_values = [], [], [], [], []

    num_epochs = config['training']['num_epochs']
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        logger.info(f"--- Epoch {epoch+1}/{num_epochs} ---")

        # Train one epoch
        train_loss, train_corr = train_epoch(
            model, train_dataloader, optimizer, device,
            config['training']['accumulation_steps'],
            config['training'].get('max_gradient_norm', 1.0)
        )
        train_losses.append(train_loss)
        train_corrs.append(train_corr)

        # Validate
        val_loss, val_corr = validate(model, val_dataloader, device)
        val_losses.append(val_loss)
        val_corrs.append(val_corr)

        current_lr = optimizer.param_groups[0]['lr']
        lr_values.append(current_lr)

        epoch_duration = time.time() - epoch_start_time

        # Log epoch summary
        logger.info(f"Epoch {epoch+1} Summary (Duration: {epoch_duration:.2f}s):")
        logger.info(f"  Train Loss: {train_loss:.6f}, Train Corr: {train_corr:.6f}")
        logger.info(f"  Val Loss:   {val_loss:.6f}, Val Corr:   {val_corr:.6f}")
        logger.info(f"  Learning Rate: {current_lr:.8f}")

        # --- Checkpointing & Early Stopping ---
        scheduler.step(val_corr) # Update LR scheduler based on validation correlation

        is_best = val_corr > best_val_corr
        if is_best:
            improvement = val_corr - best_val_corr
            logger.info(f"  Validation correlation improved! ({best_val_corr:.6f} -> {val_corr:.6f}, +{improvement:.6f})")
            best_val_corr = val_corr
            best_val_loss = val_loss
            patience_counter = 0
            save_model(model, optimizer, epoch, val_loss, val_corr, config, os.path.join(model_save_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            logger.info(f"  Validation correlation did not improve. Patience: {patience_counter}/{config['training']['early_stopping_patience']}")

        # Save latest model regardless
        save_model(model, optimizer, epoch, val_loss, val_corr, config, os.path.join(model_save_dir, 'latest_model.pt'))

        # Save periodic checkpoint
        if (epoch + 1) % config['training'].get('checkpoint_interval', 5) == 0:
            save_model(model, optimizer, epoch, val_loss, val_corr, config, os.path.join(model_save_dir, f'checkpoint_epoch_{epoch+1}.pt'))

        # Plot metrics after each epoch
        plot_metrics(train_losses, val_losses, train_corrs, val_corrs, model_save_dir, lr_values)

        # Check for early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            logger.info(f"Early stopping triggered after {epoch+1} epochs.")
            break

        # Small cleanup
        if device.type == 'cuda': torch.cuda.empty_cache()

    # --- End of Training ---
    total_training_time = time.time() - start_time_train
    logger.info("--- Training Finished ---")
    logger.info(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")
    logger.info(f"Best validation correlation achieved: {best_val_corr:.6f}")
    logger.info(f"Best validation loss achieved: {best_val_loss:.6f}")
    logger.info(f"Final training metrics plot saved in {model_save_dir}")
    logger.info("--- Training Run Ended ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ESM-3 Regression model for RMSF prediction')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {args.config}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {args.config}")
        exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {args.config}: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred loading config: {e}")
        exit(1)


    # Run training process
    try:
        train(config)
    except Exception as e:
         logger.critical(f"A critical error occurred during the training process: {e}", exc_info=True)
         exit(1)
