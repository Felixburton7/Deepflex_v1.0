# === FILE: train.py ===
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
from typing import Dict, List, Tuple, Optional, Any

# Use Accelerate for device placement and potentially distributed training later
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed as accelerate_set_seed

from dataset import create_dataloader
from model import ESMRegressionModelWithLoRA # Import the LoRA-enabled model

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(name)s] - %(message)s', force=True)
logger = logging.getLogger(__name__)


def log_gpu_memory(accelerator, prefix=""):
    """Log GPU memory usage on the current accelerator device."""
    if accelerator.device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(accelerator.device) / 1024**2
        reserved = torch.cuda.memory_reserved(accelerator.device) / 1024**2
        logger.info(f"{prefix}GPU Memory (Device {accelerator.local_process_index}): Allocated={allocated:.1f}MB, Reserved={reserved:.1f}MB")

def save_model_checkpoint(accelerator: Accelerator, model_to_save, optimizer, epoch, val_loss, val_corr, config, save_dir, is_peft=False):
    """Save model checkpoint, handling PEFT models and distributed training correctly."""
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model_to_save) # Get the underlying ESMRegressionModelWithLoRA

        if is_peft:
            try:
                # --- FIXED: Call save_pretrained on the PEFT model attribute ---
                # The PeftModel instance is stored within our custom model class, typically as '.model'
                peft_model_instance = unwrapped_model.model # Access the PeftModel wrapper
                peft_model_instance.save_pretrained(save_dir)
                # Note: This PEFT method should save adapters AND the state_dict of
                # non-PEFT modules like our regression_head if defined correctly.
                logger.info(f"PEFT model components (adapters/head) saved to directory: {save_dir}")

                # Save optimizer and other training state separately
                optimizer_path = os.path.join(save_dir, "optimizer.pt")
                training_state_path = os.path.join(save_dir, "training_state.pt")

                accelerator.save(optimizer.state_dict(), optimizer_path) # Use accelerator.save

                training_state = {
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_corr': val_corr,
                    'config': config # Save the config used for this training run
                }
                accelerator.save(training_state, training_state_path) # Use accelerator.save
                logger.info(f"Optimizer and training state saved in {save_dir}")

            except AttributeError as ae:
                 logger.error(f"AttributeError during PEFT save. Does 'unwrapped_model' have a '.model' attribute holding the PeftModel? Error: {ae}", exc_info=True)
            except Exception as e:
                logger.error(f"Error saving PEFT model components to {save_dir}: {e}", exc_info=True)
        else: # Standard saving for non-PEFT models
             logger.warning("Attempting to save non-PEFT model.")
             save_path = os.path.join(save_dir, "full_model_checkpoint.pt")
             checkpoint = {
                 'epoch': epoch,
                 'model_state_dict': unwrapped_model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'val_loss': val_loss, 'val_corr': val_corr, 'config': config
             }
             try: accelerator.save(checkpoint, save_path)
             except Exception as e: logger.error(f"Error saving non-PEFT checkpoint: {e}")


def train_epoch(model, dataloader, optimizer, accelerator: Accelerator, max_gradient_norm: float):
    """Train the model for one epoch using Accelerate."""
    model.train()
    total_loss_accum = 0.0
    total_corr_accum = 0.0
    samples_processed_accum = 0
    start_time = time.time()

    optimizer.zero_grad()

    batch_iterator = tqdm(dataloader, desc="Training Epoch", disable=not accelerator.is_local_main_process, leave=False, ncols=100)
    for step, batch in enumerate(batch_iterator):
        sequences = batch['sequences']
        rmsf_values = batch['rmsf_values']
        # Estimate global batch size for logging avg loss/corr (adjust if using uneven batches)
        current_global_batch_size = len(sequences) * accelerator.num_processes

        with accelerator.accumulate(model): # Handles sync and grad accumulation context
            outputs = model(sequences=sequences, rmsf_values=rmsf_values)
            loss = outputs['loss']
            correlation = outputs['metrics'].get('pearson_correlation', 0.0) # Metric from this rank

            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"[Rank {accelerator.process_index}] Step {step}: Invalid loss ({loss}). Skipping backward.")
                continue # Skip gradient calc and step

            # Ensure loss is valid before backward
            if not torch.isnan(loss) and not torch.isinf(loss):
                 # Gather loss across GPUs *before* backward for accurate logging of step loss
                 avg_loss_step = accelerator.gather(loss.detach()).mean()
                 accelerator.backward(loss) # Scales loss and calls backward()

                 if accelerator.sync_gradients: # Only clip/step when grads are synced
                     if max_gradient_norm > 0:
                          accelerator.clip_grad_norm_(model.parameters(), max_gradient_norm)
                     optimizer.step()
                     optimizer.zero_grad()

                 # --- Accumulate Metrics for Logging (on main process) ---
                 if accelerator.is_main_process:
                      # Accumulate using the gathered average loss for the step
                      total_loss_accum += avg_loss_step.item() * current_global_batch_size
                      # Gather correlation - average across devices for a better estimate
                      gathered_corr_tensor = accelerator.gather(torch.tensor(correlation, device=accelerator.device))
                      gathered_corr_tensor_valid = gathered_corr_tensor[~torch.isnan(gathered_corr_tensor)]
                      avg_corr_step = gathered_corr_tensor_valid.mean().item() if gathered_corr_tensor_valid.numel() > 0 else 0.0
                      total_corr_accum += avg_corr_step * current_global_batch_size
                      samples_processed_accum += current_global_batch_size

                      # Update progress bar
                      avg_loss_epoch = total_loss_accum / samples_processed_accum if samples_processed_accum > 0 else 0.0
                      avg_corr_epoch = total_corr_accum / samples_processed_accum if samples_processed_accum > 0 else 0.0
                      batch_iterator.set_postfix(
                          loss=f"{avg_loss_step.item():.4f}", # Step avg loss
                          corr=f"{avg_corr_step:.4f}", # Step avg corr
                          ep_loss=f"{avg_loss_epoch:.4f}", # Running epoch avg loss
                          ep_corr=f"{avg_corr_epoch:.4f}" # Running epoch avg corr
                      )
            else:
                logger.warning(f"[Rank {accelerator.process_index}] Step {step}: NaN or Inf loss before backward. Gradients not computed.")

    # --- Epoch End Calculation ---
    accelerator.wait_for_everyone()
    final_avg_loss, final_avg_corr = None, None
    if accelerator.is_main_process:
        final_avg_loss = total_loss_accum / samples_processed_accum if samples_processed_accum > 0 else 0.0
        final_avg_corr = total_corr_accum / samples_processed_accum if samples_processed_accum > 0 else 0.0
        epoch_duration = time.time() - start_time
        logger.info(f"Train Epoch completed in {epoch_duration:.2f}s. Avg Loss: {final_avg_loss:.6f}, Avg Corr: {final_avg_corr:.6f}")

    return final_avg_loss, final_avg_corr


@torch.no_grad()
def validate(model, dataloader, accelerator: Accelerator):
    """Validate the model using Accelerate."""
    model.eval()
    total_loss_accum = 0.0
    total_corr_accum = 0.0
    samples_processed_accum = 0
    start_time = time.time()

    batch_iterator = tqdm(dataloader, desc="Validation", disable=not accelerator.is_local_main_process, leave=False, ncols=100)
    for step, batch in enumerate(batch_iterator):
        sequences = batch['sequences']
        rmsf_values = batch['rmsf_values']
        current_global_batch_size = len(sequences) * accelerator.num_processes

        outputs = model(sequences=sequences, rmsf_values=rmsf_values)
        loss = outputs['loss']
        correlation = outputs['metrics'].get('pearson_correlation', 0.0)

        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"[Rank {accelerator.process_index}] Validation Step {step}: Invalid loss ({loss}). Skipping.")
            continue

        # Gather results across all processes
        gathered_loss = accelerator.gather(loss.detach()).mean()
        gathered_corr_tensor = accelerator.gather(torch.tensor(correlation, device=accelerator.device))
        gathered_corr_tensor_valid = gathered_corr_tensor[~torch.isnan(gathered_corr_tensor)]
        avg_corr_batch = gathered_corr_tensor_valid.mean().item() if gathered_corr_tensor_valid.numel() > 0 else 0.0

        # Accumulate on main process
        if accelerator.is_main_process:
            total_loss_accum += gathered_loss.item() * current_global_batch_size
            total_corr_accum += avg_corr_batch * current_global_batch_size
            samples_processed_accum += current_global_batch_size

            # Update progress bar
            avg_loss_epoch = total_loss_accum / samples_processed_accum if samples_processed_accum > 0 else 0.0
            avg_corr_epoch = total_corr_accum / samples_processed_accum if samples_processed_accum > 0 else 0.0
            batch_iterator.set_postfix(loss=f"{gathered_loss.item():.4f}", corr=f"{avg_corr_batch:.4f}")

    # --- Epoch End Calculation ---
    accelerator.wait_for_everyone()
    final_avg_loss, final_avg_corr = None, None
    if accelerator.is_main_process:
        final_avg_loss = total_loss_accum / samples_processed_accum if samples_processed_accum > 0 else 0.0
        final_avg_corr = total_corr_accum / samples_processed_accum if samples_processed_accum > 0 else 0.0
        epoch_duration = time.time() - start_time
        logger.info(f"Validation completed in {epoch_duration:.2f}s. Avg Loss: {final_avg_loss:.6f}, Avg Corr: {final_avg_corr:.6f}")

    return final_avg_loss, final_avg_corr


def plot_metrics(train_losses, val_losses, train_corrs, val_corrs, save_dir, lr_values=None):
    """Plot training and validation metrics."""
    if not train_losses or not val_losses: return # Cannot plot if lists are empty
    epochs = range(1, len(train_losses) + 1)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Loss Plot
    ax1.plot(epochs, train_losses, 'o-', color='royalblue', label='Train Loss', markersize=4)
    ax1.plot(epochs, val_losses, 's-', color='orangered', label='Validation Loss', markersize=4)
    ax1.set_ylabel('Loss (MSE)'); ax1.set_title('Training and Validation Loss'); ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.6)
    if val_losses: # Check if list is not empty
        best_val_loss_epoch = np.argmin(val_losses)
        ax1.scatter(best_val_loss_epoch + 1, val_losses[best_val_loss_epoch], marker='*', color='red', s=100, zorder=5, label=f'Best Val Loss ({val_losses[best_val_loss_epoch]:.4f})')
        ax1.legend()

    # Correlation Plot
    ax2.plot(epochs, train_corrs, 'o-', color='royalblue', label='Train Correlation', markersize=4)
    ax2.plot(epochs, val_corrs, 's-', color='orangered', label='Validation Correlation', markersize=4)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Pearson Correlation'); ax2.set_title('Training and Validation Correlation'); ax2.grid(True, linestyle='--', alpha=0.6)
    if val_corrs: # Check if list is not empty
        best_val_corr_epoch = np.argmax(val_corrs)
        ax2.scatter(best_val_corr_epoch + 1, val_corrs[best_val_corr_epoch], marker='*', color='red', s=100, zorder=5, label=f'Best Val Corr ({val_corrs[best_val_corr_epoch]:.4f})')
    ax2.legend(loc='lower right')

    # Learning Rate Plot (Optional)
    if lr_values:
        ax3 = ax2.twinx()
        ax3.plot(epochs, lr_values, 'd--', color='green', label='Learning Rate', markersize=3, alpha=0.7)
        ax3.set_ylabel('Learning Rate', color='green'); ax3.tick_params(axis='y', labelcolor='green')
        # Use log scale only if LR actually changes significantly
        if len(lr_values) > 1 and min(lr_values) < max(lr_values) * 0.9:
             ax3.set_yscale('log')
        ax3.legend(loc='center right')

    plt.tight_layout(pad=1.5)
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, 'training_metrics.png')
    try: plt.savefig(plot_path, dpi=150, bbox_inches='tight'); logger.info(f"Metrics plot saved to {plot_path}")
    except Exception as e: logger.error(f"Error saving metrics plot: {e}")
    plt.close(fig)


def train(config: Dict):
    """Main training function using Accelerate."""
    start_time_train = time.time()
    train_config = config['training']

    # --- Initialize Accelerator ---
    # Handle find_unused_parameters based on config or default
    find_unused = train_config.get('ddp_find_unused_parameters', False) # Default False
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused)
    # Check for AMP setting in config
    mixed_precision = "fp16" if train_config.get('use_amp', False) else "no"

    accelerator = Accelerator(
        gradient_accumulation_steps=train_config['accumulation_steps'],
        log_with="tensorboard", # Or "wandb" etc.
        project_dir=config['output']['model_dir'],
        kwargs_handlers=[ddp_kwargs],
        mixed_precision=mixed_precision # Set mixed precision type
    )

    # --- Setup Logging ---
    log_dir = os.path.join(config['output']['model_dir'], 'logs')
    if accelerator.is_main_process:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'training.log')
        # Ensure root logger has file handler only on main process
        root_logger = logging.getLogger()
        # Remove previous file handlers for this path if any
        for handler in root_logger.handlers[:]:
             if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_path:
                  handler.close()
                  root_logger.removeHandler(handler)
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s [%(name)s] - %(message)s'))
        root_logger.addHandler(file_handler)
        logger.info("--- Starting New Training Run ---")
        logger.info(f"Accelerate state: {accelerator.state}")
        logger.info(f"Using device: {accelerator.device}")
        logger.info(f"Mixed precision: {accelerator.mixed_precision}")
        logger.info(f"Configuration loaded.")
    else: # Reduce logging on non-main processes
         logging.getLogger().setLevel(logging.WARNING)

    accelerate_set_seed(train_config['seed'])
    if accelerator.is_main_process: log_gpu_memory(accelerator, prefix="Initial ")

    # --- Data Loaders ---
    if accelerator.is_main_process: logger.info("Creating data loaders...")
    try:
        num_workers = train_config.get('num_workers', 0) # Default to 0, increase if IO allows
        pin_memory = train_config.get('pin_memory', True)
        train_dataloader = create_dataloader(
            config['data']['data_dir'], 'train', train_config['batch_size'], shuffle=True,
            max_length=train_config.get('max_length'),
            length_bucket_size=train_config.get('length_bucket_size', 50),
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True
        )
        val_dataloader = create_dataloader(
            config['data']['data_dir'], 'val', train_config['batch_size'], shuffle=False,
            max_length=train_config.get('max_length'),
            length_bucket_size=train_config.get('length_bucket_size', 50),
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False
        )
        if not train_dataloader or not val_dataloader: logger.error("Failed to create dataloaders."); return
    except Exception as e: logger.error(f"Error creating dataloaders: {e}", exc_info=True); return
    if accelerator.is_main_process: logger.info("Dataloaders created.")

    # --- Model ---
    if accelerator.is_main_process: logger.info("Creating model...")
    try:
        model = ESMRegressionModelWithLoRA(config)
        is_peft_model = config['model'].get('lora', {}).get('enabled', False)
    except Exception as e: logger.error(f"Error creating model: {e}", exc_info=True); return
    if accelerator.is_main_process: logger.info("Model created.")

    # --- Optimizer ---
    try:
        params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        if not params_to_optimize: logger.error("Model has no trainable parameters!"); return
        optimizer = optim.AdamW(
            params_to_optimize, lr=float(train_config['learning_rate']),
            eps=float(train_config.get('adam_epsilon', 1e-8)),
            weight_decay=float(train_config['weight_decay'])
        )
    except Exception as e: logger.error(f"Error creating optimizer: {e}", exc_info=True); return
    if accelerator.is_main_process: logger.info("Optimizer created.")

    # --- Scheduler ---
    scheduler = ReduceLROnPlateau(
        optimizer, mode=train_config.get('scheduler_mode', 'max'),
        factor=train_config.get('scheduler_factor', 0.5),
        patience=train_config['scheduler_patience'], verbose=False, # Log manually
        threshold=train_config.get('scheduler_threshold', 0.001)
    )
    if accelerator.is_main_process: logger.info("Scheduler created.")

    # --- Prepare with Accelerate ---
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )
    if accelerator.is_main_process:
        logger.info("Components prepared with Accelerate.")
        log_gpu_memory(accelerator, prefix="After Accelerate Prepare ")

    # --- Training Loop ---
    if accelerator.is_main_process: logger.info("--- Starting Training Loop ---")
    best_val_corr = -float('inf')
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses, train_corrs, val_corrs, lr_values = [], [], [], [], []
    num_epochs = train_config['num_epochs']
    early_stopping_patience = train_config['early_stopping_patience']
    early_stopping_threshold = train_config['early_stopping_threshold']
    model_save_dir = config['output']['model_dir']

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        if accelerator.is_main_process: logger.info(f"--- Epoch {epoch+1}/{num_epochs} ---")

        train_loss_epoch, train_corr_epoch = train_epoch(
            model, train_dataloader, optimizer, accelerator, train_config.get('max_gradient_norm', 1.0)
        )
        val_loss_epoch, val_corr_epoch = validate(model, val_dataloader, accelerator)

        # --- Aggregation, Logging, Checkpointing, Early Stopping (Main Process Only) ---
        # Use accelerator.gather to collect results if needed for more precise averaging
        # For now, rely on main process results as returned by train/validate functions
        if accelerator.is_main_process:
             # Ensure we have valid numbers before proceeding
             if train_loss_epoch is None or val_loss_epoch is None:
                  logger.error(f"Epoch {epoch+1}: Training or Validation function did not return results on main process. Cannot proceed.")
                  break # Stop training if main process didn't get results

             train_losses.append(train_loss_epoch); train_corrs.append(train_corr_epoch)
             val_losses.append(val_loss_epoch); val_corrs.append(val_corr_epoch)
             current_lr = optimizer.param_groups[0]['lr']; lr_values.append(current_lr)
             epoch_duration = time.time() - epoch_start_time

             logger.info(f"Epoch {epoch+1} Summary (Duration: {epoch_duration:.2f}s):")
             logger.info(f"  Train Loss: {train_loss_epoch:.6f}, Train Corr: {train_corr_epoch:.6f}")
             logger.info(f"  Val Loss:   {val_loss_epoch:.6f}, Val Corr:   {val_corr_epoch:.6f}")
             logger.info(f"  Learning Rate: {current_lr:.8f}")

             # Scheduler Step
             old_lr = current_lr
             scheduler.step(val_corr_epoch)
             new_lr = optimizer.param_groups[0]['lr']
             if new_lr < old_lr: logger.info(f"Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")

             # Checkpointing & Early Stopping
             is_best = val_corr_epoch > best_val_corr + early_stopping_threshold
             if is_best:
                 improvement = val_corr_epoch - best_val_corr
                 logger.info(f"  Val Corr improved! ({best_val_corr:.6f} -> {val_corr_epoch:.6f}, +{improvement:.6f}) Saving best model...")
                 best_val_corr = val_corr_epoch; best_val_loss = val_loss_epoch
                 patience_counter = 0
                 best_save_dir = os.path.join(model_save_dir, 'best_model')
                 # accelerator.wait_for_everyone() # Ensure all processes are ready before saving
                 save_model_checkpoint(accelerator, model, optimizer, epoch, val_loss_epoch, val_corr_epoch, config, best_save_dir, is_peft=is_peft_model)
             else:
                 patience_counter += 1
                 logger.info(f"  Val Corr did not improve sufficiently. Patience: {patience_counter}/{early_stopping_patience}")

             # Save latest model checkpoint periodically
             if (epoch + 1) % train_config.get('checkpoint_interval', 1) == 0:
                 latest_save_dir = os.path.join(model_save_dir, 'latest_model')
                 logger.info(f"Saving latest model checkpoint (Epoch {epoch+1})...")
                 # accelerator.wait_for_everyone()
                 save_model_checkpoint(accelerator, model, optimizer, epoch, val_loss_epoch, val_corr_epoch, config, latest_save_dir, is_peft=is_peft_model)

             # Plot metrics
             plot_metrics(train_losses, val_losses, train_corrs, val_corrs, model_save_dir, lr_values)

             # Check for early stopping
             if patience_counter >= early_stopping_patience:
                 logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                 break # Exit the loop

        # Sync all processes before starting next epoch
        accelerator.wait_for_everyone()
        if accelerator.device.type == 'cuda': torch.cuda.empty_cache()


    # --- End of Training ---
    if accelerator.is_main_process:
        total_training_time = time.time() - start_time_train
        logger.info("--- Training Finished ---")
        logger.info(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")
        logger.info(f"Best validation correlation: {best_val_corr:.6f} (Loss: {best_val_loss:.6f})")
        logger.info(f"Checkpoints/logs saved in {model_save_dir}")
        logger.info("--- Training Run Ended ---")
        # Ensure logs are flushed before exiting
        for handler in logging.getLogger().handlers:
             handler.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ESM-C + LoRA model for RMSF prediction using Accelerate')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    args = parser.parse_args()
    try:
        with open(args.config, 'r') as f: config = yaml.safe_load(f)
    except Exception as e: logger.error(f"Error loading config {args.config}: {e}"); exit(1)
    try: train(config)
    except Exception as e: logger.critical(f"Critical error during training: {e}", exc_info=True); exit(1)