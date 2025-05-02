"""
Training command for VoxelFlex (Temperature-Aware).

Implements training using the chunked data loading approach for
memory-efficient processing of large HDF5 files.
"""

import os
import time
import json
import logging
import math
from typing import Dict, Any, Tuple, List, Optional, Callable, Union
import shutil
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from scipy.stats import pearsonr

logger = logging.getLogger("voxelflex.cli.train")

# Project imports
from voxelflex.data.data_loader import (
    ChunkedVoxelDataset,  # Use this for train/val/test
    worker_init_fn,       # Simplified version is fine
    simple_collate_fn,    # Keep using this
    load_list_from_file   # Utility to load domain lists
)
from voxelflex.models.cnn_models import get_model
from voxelflex.utils.logging_utils import (
    log_stage, EnhancedProgressBar, log_memory_usage, log_section_header, get_logger, Timing
)
from voxelflex.utils.file_utils import ensure_dir, save_json, load_json, resolve_path, convert_to_serializable
from voxelflex.utils.system_utils import (
    get_device, clear_memory, check_memory_usage, set_num_threads
)
from voxelflex.utils.temp_scaling import get_temperature_scaler

# --- Train/Validate Epoch Functions ---
def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    timing: Optional[Timing] = None
) -> Tuple[float, float]:
    """
    Performs one training epoch with performance tracking.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        config: Configuration dictionary
        scaler: Optional GradScaler for mixed precision training
        timing: Optional timing tracker
        
    Returns:
        Tuple of (average loss, pearson correlation)
    """
    model.train()
    running_loss_sum = 0.0
    batch_preds_all = []
    batch_targets_all = []
    num_samples_processed = 0
    
    train_cfg = config['training']
    grad_clip_cfg = train_cfg.get('gradient_clipping', {})
    grad_clip_norm = grad_clip_cfg.get('max_norm') if grad_clip_cfg.get('enabled') else None
    show_progress = config['logging'].get('show_progress_bars', True)
    autocast_enabled = scaler is not None and scaler.is_enabled()
    grad_accum_steps = train_cfg.get('gradient_accumulation_steps', 1)

    # Estimate total batches for progress bar
    total_batches_estimate = None
    est_samples = config.get('runtime', {}).get('estimated_samples_per_epoch')
    batch_size = train_cfg.get('batch_size')
    if est_samples and batch_size:
        total_batches_estimate = math.ceil(est_samples / batch_size)
        logger.debug(f"Using estimated train batches for progress: {total_batches_estimate}")

    progress = EnhancedProgressBar(total=total_batches_estimate, desc=f"Epoch {epoch+1} Train") if show_progress else None

    if timing: timing.start('train_epoch_total')
    optimizer.zero_grad(set_to_none=True)
    running_grad_norm = 0.0

    # --- Batch Loop ---
    for i, batch_dict in enumerate(train_loader):
        if timing: timing.start('batch_process')
        if batch_dict is None:
            logger.debug(f"Skipping None batch {i}")
            if progress: progress.update(1)
            if timing: timing.end('batch_process')
            continue
            
        try: # Data Transfer
            if timing: timing.start('data_transfer')
            voxel_inputs = batch_dict['voxels'].to(device, non_blocking=True)
            scaled_temps = batch_dict['scaled_temps'].to(device, non_blocking=True)
            targets = batch_dict['targets'].to(device, non_blocking=True)
            if timing: timing.end('data_transfer')
            current_batch_size = voxel_inputs.size(0)
        except Exception as load_e:
            logger.debug(f"Batch prep error {i}: {load_e}")
            if progress: progress.update(1)
            if timing: timing.end('batch_process')
            continue
            
        try: # Forward Pass
            if timing: timing.start('forward')
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                outputs = model(voxel_input=voxel_inputs, scaled_temp=scaled_temps)
                loss = criterion(outputs, targets) / grad_accum_steps
            if timing: timing.end('forward')
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss batch {i}. Skip.")
                if progress: progress.update(1)
                if timing: timing.end('batch_process')
                continue
        except Exception as forward_e:
            logger.debug(f"Forward error batch {i}: {forward_e}")
            if progress: progress.update(1)
            if timing: timing.end('batch_process')
            continue
            
        try: # Backward Pass
            if timing: timing.start('backward')
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if timing: timing.end('backward')
            
            if grad_clip_norm: # Grad norm calculation
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            param_norm = param.grad.detach().data.norm(2)
                            running_grad_norm += param_norm.item() ** 2
        except Exception as backward_e:
            logger.debug(f"Backward error batch {i}: {backward_e}")
            optimizer.zero_grad(set_to_none=True)
            if progress: progress.update(1)
            if timing: timing.end('batch_process')
            continue

        # --- Step Optimizer ---
        is_last_batch = False # Hard to determine precisely with IterableDataset
        if progress and total_batches_estimate and progress.n >= total_batches_estimate - 1:
            is_last_batch = True # Approximation
            
        if (i + 1) % grad_accum_steps == 0 or is_last_batch:
            if timing: timing.start('optimizer_step')
            total_norm = 0.0
            if grad_clip_norm:
                total_norm = running_grad_norm**0.5
                running_grad_norm = 0.0
                
            try:
                if scaler is not None:
                    if grad_clip_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if grad_clip_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()
                    
                optimizer.zero_grad(set_to_none=True)
            except Exception as optim_e:
                logger.debug(f"Optim step error {i}: {optim_e}")
                optimizer.zero_grad(set_to_none=True)
                
            if timing: timing.end('optimizer_step')

        # --- Accumulate & Cleanup ---
        running_loss_sum += (loss.item() * grad_accum_steps) * current_batch_size
        batch_preds_all.append(outputs.detach().cpu().numpy())
        batch_targets_all.append(targets.detach().cpu().numpy())
        num_samples_processed += current_batch_size
        
        if progress: progress.update(1)
        del voxel_inputs, scaled_temps, targets, outputs, loss
        if i % 50 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()
            
        if timing: timing.end('batch_process')
    # --- End Batch Loop ---

    if progress: progress.close()

    # --- Calculate Metrics ---
    if timing: timing.start('metrics_calculation')
    
    avg_loss = running_loss_sum / num_samples_processed if num_samples_processed > 0 else 0.0
    epoch_pearson = 0.0
    
    if num_samples_processed > 1 and batch_preds_all and batch_targets_all:
        try:
            all_preds_flat = np.concatenate([b.flatten() for b in batch_preds_all])
            all_targets_flat = np.concatenate([b.flatten() for b in batch_targets_all])
            
            valid_mask = np.isfinite(all_preds_flat) & np.isfinite(all_targets_flat)
            preds_valid = all_preds_flat[valid_mask]
            targets_valid = all_targets_flat[valid_mask]
            
            if len(preds_valid) > 1 and np.std(preds_valid) > 1e-6 and np.std(targets_valid) > 1e-6:
                corr, _ = pearsonr(preds_valid, targets_valid)
                epoch_pearson = corr if not np.isnan(corr) else 0.0
        except Exception as e:
            logger.debug(f"Train Corr calc error: {e}")
            
    if timing: timing.end('metrics_calculation')
    if timing: timing.end('train_epoch_total')
    
    logger.debug(f"Train epoch timing: {timing.get_stats().get('train_epoch_total', {}).get('total', 0):.2f}s")
    return avg_loss, epoch_pearson


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Dict[str, Any],
    timing: Optional[Timing] = None
) -> Tuple[float, float]:
    """
    Performs validation with performance tracking.
    
    Args:
        model: Model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        config: Configuration dictionary
        timing: Optional timing tracker
        
    Returns:
        Tuple of (average loss, pearson correlation)
    """
    model.eval()
    running_loss_sum = 0.0
    batch_preds_all = []
    batch_targets_all = []
    num_samples_processed = 0
    
    autocast_enabled = config['training'].get('mixed_precision', {}).get('enabled', False) and device.type == 'cuda'
    show_progress = config['logging'].get('show_progress_bars', True)

    # Estimate total batches for progress bar
    val_batches_estimate = None
    est_val_samples = config.get('runtime', {}).get('estimated_val_samples')
    batch_size = config['training'].get('batch_size')  # Assuming same base batch size logic
    if est_val_samples and batch_size:
        val_batch_size = batch_size * 2  # Match loader config
        val_batches_estimate = math.ceil(est_val_samples / val_batch_size)
        logger.debug(f"Using estimated val batches for progress: {val_batches_estimate}")

    progress = EnhancedProgressBar(total=val_batches_estimate, desc="Validation") if show_progress else None

    if timing: timing.start('val_epoch_total')
    
    with torch.no_grad():
        for i, batch_dict in enumerate(val_loader):
            if timing: timing.start('val_batch_process')
            
            if batch_dict is None:
                if progress: progress.update(1)
                if timing: timing.end('val_batch_process')
                continue
                
            try: # Data Transfer
                if timing: timing.start('val_data_transfer')
                voxel_inputs = batch_dict['voxels'].to(device, non_blocking=True)
                scaled_temps = batch_dict['scaled_temps'].to(device, non_blocking=True)
                targets = batch_dict['targets'].to(device, non_blocking=True)
                if timing: timing.end('val_data_transfer')
                current_batch_size = voxel_inputs.size(0)
            except Exception as e:
                logger.debug(f"Val batch prep {i}: {e}")
                if progress: progress.update(1)
                if timing: timing.end('val_batch_process')
                continue
                
            try: # Forward
                if timing: timing.start('val_forward')
                with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                    outputs = model(voxel_input=voxel_inputs, scaled_temp=scaled_temps)
                    loss = criterion(outputs, targets)
                if timing: timing.end('val_forward')
                
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.debug(f"Val NaN/Inf loss {i}. Skip.")
                    if progress: progress.update(1)
                    if timing: timing.end('val_batch_process')
                    continue
                    
                running_loss_sum += loss.item() * current_batch_size
                batch_preds_all.append(outputs.cpu().numpy())
                batch_targets_all.append(targets.cpu().numpy())
                num_samples_processed += current_batch_size
            except Exception as e:
                logger.debug(f"Val forward error {i}: {e}")
                if progress: progress.update(1)
                if timing: timing.end('val_batch_process')
                continue
                
            # Cleanup & Progress
            if progress: progress.update(1)
            del voxel_inputs, scaled_temps, targets, outputs, loss
            if i % 50 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
                
            if timing: timing.end('val_batch_process')
    # End Batch Loop
    
    if progress: progress.close()

    # --- Calculate Metrics ---
    if timing: timing.start('val_metrics_calculation')
    
    avg_loss = running_loss_sum / num_samples_processed if num_samples_processed > 0 else float('inf')
    epoch_pearson = 0.0
    
    if num_samples_processed > 1 and batch_preds_all and batch_targets_all:
        try:
            all_preds_flat = np.concatenate([b.flatten() for b in batch_preds_all])
            all_targets_flat = np.concatenate([b.flatten() for b in batch_targets_all])
            
            valid_mask = np.isfinite(all_preds_flat) & np.isfinite(all_targets_flat)
            preds_valid = all_preds_flat[valid_mask]
            targets_valid = all_targets_flat[valid_mask]
            
            if len(preds_valid) > 1 and np.std(preds_valid) > 1e-6 and np.std(targets_valid) > 1e-6:
                corr, _ = pearsonr(preds_valid, targets_valid)
                epoch_pearson = corr if not np.isnan(corr) else 0.0
        except Exception as e:
            logger.debug(f"Val Corr calc error: {e}")
            
    if timing: timing.end('val_metrics_calculation')
    if timing: timing.end('val_epoch_total')
    
    logger.debug(f"Val epoch timing: {timing.get_stats().get('val_epoch_total', {}).get('total', 0):.2f}s")
    return avg_loss, epoch_pearson


# --- Helper Functions: Optimizer and Scheduler ---
def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """
    Get optimizer based on config.
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
        
    Returns:
        PyTorch optimizer
    """
    lr = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    logger.info(f"Optimizer: AdamW (LR={lr:.2e}, WD={weight_decay:.2e})")
    return optimizer

def get_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any], num_epochs: int) -> Tuple[Optional[Any], str, str]:
    """
    Get learning rate scheduler based on config.
    
    Args:
        optimizer: Optimizer to schedule
        config: Configuration dictionary
        num_epochs: Total number of epochs
        
    Returns:
        Tuple of (scheduler, metric_to_monitor, mode)
    """
    s_cfg = config['training'].get('scheduler', {})
    s_type = s_cfg.get('type', 'reduce_on_plateau').lower()
    metric = s_cfg.get('monitor_metric', 'val_pearson')
    
    if metric not in ['val_loss', 'val_pearson']:
        logger.warning(f"Invalid scheduler metric '{metric}'. Default='val_pearson'.")
        metric = 'val_pearson'
        
    mode = s_cfg.get('mode')
    if mode not in ['min', 'max']:
        mode = 'max' if 'pearson' in metric else 'min'
        logger.debug(f"Scheduler mode auto={mode}.")
        
    scheduler = None
    
    if s_type == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=float(s_cfg.get('factor', 0.5)),
            patience=int(s_cfg.get('patience', 5)),
            threshold=float(s_cfg.get('threshold', 0.001)),
            min_lr=float(s_cfg.get('min_lr', 1e-7)),
            verbose=True
        )
        logger.info(f"Scheduler: ReduceLROnPlateau (Metric={metric}, Mode={mode})")
        
    elif s_type == 'cosine_annealing':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(s_cfg.get('T_max', num_epochs)),
            eta_min=float(s_cfg.get('eta_min', 1e-7))
        )
        logger.info(f"Scheduler: CosineAnnealingLR")
        
    elif s_type == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=int(s_cfg.get('step_size', 10)),
            gamma=float(s_cfg.get('gamma', 0.1))
        )
        logger.info(f"Scheduler: StepLR")
        
    else:
        logger.warning(f"Unknown scheduler type: {s_type}. None used.")
        
    return scheduler, metric, mode


# def train_model(config: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, List[float]]]]:
#     """
#     Main function to train the model using chunked data loading.
    
#     Args:
#         config: Configuration dictionary
        
#     Returns:
#         Tuple of (path to best model, training history)
#     """
#     run_output_dir = config["output"]["run_dir"]
#     models_dir = config["output"]["models_dir"]
#     ensure_dir(models_dir)
    
#     log_section_header(logger, f"STARTING TRAINING RUN: {config['output']['run_name']} (Chunked Loading)")
#     log_memory_usage(logger)
    
#     # Set up environment
#     device = get_device(config["system_utilization"]["adjust_for_gpu"])
#     seed = config['training']['seed']
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if device.type == 'cuda':
#         torch.cuda.manual_seed_all(seed)
        
#     log_timing = config.get('logging', {}).get('log_timing', False)
#     timing = Timing() if log_timing else None

#     # --- Load Datasets & DataLoaders ---
#     with log_stage("DATA_PREPARATION", "Creating Datasets & DataLoaders (Chunked)"):
#         try:
#             master_samples_path = os.path.join(config["data"]["processed_dir"], config["data"]["master_samples_file"])
#             if not os.path.exists(master_samples_path):
#                 raise FileNotFoundError(f"Master sample file missing: {master_samples_path}")
                
#             voxel_hdf5_path = config['input']['voxel_file']
#             if not os.path.exists(voxel_hdf5_path):
#                 raise FileNotFoundError(f"HDF5 file missing: {voxel_hdf5_path}")

#             # Set up voxel shape information
#             voxel_c = config['model']['input_channels']
#             voxel_d = config['model'].get('voxel_depth', 21)
#             voxel_h = config['model'].get('voxel_height', 21)
#             voxel_w = config['model'].get('voxel_width', 21)
#             voxel_shape = (voxel_c, voxel_d, voxel_h, voxel_w)
#             expected_channels = voxel_c

#             # Load temp scaling params with better error handling
#             temp_scaling_params = None
#             scaler_path = config.get('data', {}).get('temp_scaling_params_file')
#             if scaler_path and os.path.exists(scaler_path):
#                 try:
#                     temp_scaling_params = load_json(scaler_path)
#                     logger.info(f"Loaded temp scaling: {temp_scaling_params}")
#                 except Exception as e:
#                     logger.error(f"Failed to load scaler file '{scaler_path}': {e}")
                    
#             if not temp_scaling_params:
#                 temp_scaling_params = {'temp_min': 280.0, 'temp_max': 360.0}
#                 logger.warning(f"Using default temp scaling: {temp_scaling_params}")
                
#             config.setdefault('runtime', {})['temp_scaling_params'] = temp_scaling_params

#             # Load domain lists
#             train_split_file = config['input'].get('train_split_file')
#             val_split_file = config['input'].get('val_split_file')
            
#             if not train_split_file or not os.path.exists(train_split_file):
#                 raise FileNotFoundError(f"Train split file missing: {train_split_file}")
                
#             if not val_split_file or not os.path.exists(val_split_file):
#                 raise FileNotFoundError(f"Validation split file missing: {val_split_file}")
                
#             train_domain_list = load_list_from_file(train_split_file)
#             val_domain_list = load_list_from_file(val_split_file)
            
#             if not train_domain_list:
#                 raise ValueError("Training domain list is empty.")
                
#             if not val_domain_list:
#                 raise ValueError("Validation domain list is empty.")

#             # Estimate samples for progress bars (handle PyArrow availability correctly)
#             try:
#                 import pyarrow.parquet as pq
#                 PYARROW_AVAILABLE = True
#             except ImportError:
#                 PYARROW_AVAILABLE = False
                
#             try:
#                 logger.debug("Estimating samples per epoch for progress bars...")
#                 n_train, n_val = 0, 0
                
#                 if PYARROW_AVAILABLE:
#                     train_filter = [('split', '==', 'train')]
#                     val_filter = [('split', '==', 'val')]
#                     train_meta = pq.read_table(master_samples_path, columns=['split'], filters=train_filter)
#                     val_meta = pq.read_table(master_samples_path, columns=['split'], filters=val_filter)
#                     n_train = len(train_meta)
#                     n_val = len(val_meta)
#                     del train_meta, val_meta  # Free memory
#                 else:
#                     # Fallback: load full and filter (slower, more memory)
#                     df_full = pd.read_csv(master_samples_path, usecols=['split'])
#                     n_train = len(df_full[df_full['split'] == 'train'])
#                     n_val = len(df_full[df_full['split'] == 'val'])
#                     del df_full
                    
#                 config['runtime']['estimated_samples_per_epoch'] = n_train
#                 config['runtime']['estimated_val_samples'] = n_val
#                 logger.info(f"Estimated samples: Train={n_train}, Val={n_val}")
#                 gc.collect()
#             except Exception as e:
#                 logger.warning(f"Could not estimate sample counts from {master_samples_path}: {e}. "
#                               f"Progress bars may be indefinite.")

#             # Instantiate Datasets using ChunkedVoxelDataset
#             train_chunk_size = config['training'].get('chunk_size', 100)
#             logger.info(f"Creating training dataset (ChunkedIterable, chunk_size={train_chunk_size})")
#             train_dataset = ChunkedVoxelDataset(
#                 master_samples_path=master_samples_path, 
#                 split='train', 
#                 domain_list=train_domain_list,
#                 voxel_hdf5_path=voxel_hdf5_path, 
#                 temp_scaling_params=temp_scaling_params,
#                 chunk_size=train_chunk_size, 
#                 shuffle_domain_list=True  # Shuffle for training
#             )
            
#             # Added metadata info logging for better debugging
#             logger.info(f"Train dataset created with {len(train_domain_list)} domains and {len(train_dataset.metadata_lookup)} metadata entries")
            
#             logger.info(f"Creating validation dataset (ChunkedIterable, chunk_size={train_chunk_size})")
#             val_dataset = ChunkedVoxelDataset(
#                 master_samples_path=master_samples_path, 
#                 split='val', 
#                 domain_list=val_domain_list,
#                 voxel_hdf5_path=voxel_hdf5_path, 
#                 temp_scaling_params=temp_scaling_params,
#                 chunk_size=train_chunk_size, 
#                 shuffle_domain_list=False  # No shuffle for validation
#             )
            
#             logger.info(f"Val dataset created with {len(val_domain_list)} domains and {len(val_dataset.metadata_lookup)} metadata entries")

#             # Instantiate DataLoaders
#             train_batch_size = config['training']['batch_size']
#             val_batch_size = train_batch_size * 2  # Larger batches for validation
#             num_workers = config['training'].get('num_workers', 4)
#             pin_memory = config['training'].get('pin_memory', True) and (device.type == 'cuda')

#             logger.info(f"Train DataLoader: Chunked, BatchSize={train_batch_size}, Workers={num_workers}")
#             train_loader = DataLoader(
#                 train_dataset, 
#                 batch_size=train_batch_size, 
#                 num_workers=num_workers,
#                 pin_memory=pin_memory, 
#                 worker_init_fn=worker_init_fn, 
#                 collate_fn=simple_collate_fn,
#                 prefetch_factor=config['training'].get('prefetch_factor', 2) if num_workers > 0 else None
#             )
            
#             logger.info(f"Val DataLoader: Chunked, BatchSize={val_batch_size}, Workers={num_workers}")
#             val_loader = DataLoader(
#                 val_dataset, 
#                 batch_size=val_batch_size, 
#                 num_workers=num_workers,
#                 pin_memory=pin_memory, 
#                 worker_init_fn=worker_init_fn, 
#                 collate_fn=simple_collate_fn,
#                 prefetch_factor=config['training'].get('prefetch_factor', 2) if num_workers > 0 else None
#             )

#         except Exception as e:
#             logger.exception(f"Fatal error creating Datasets/DataLoaders: {e}")
#             return None, None

#     # --- Model & Optimizer ---
#     with log_stage("MODEL_CREATION", "Creating model and optimizer"):
#         try:
#             input_shape = voxel_shape
#             logger.info(f"Using model input shape: {input_shape}")
#             model = get_model(config['model'], input_shape=input_shape)
#             model.to(device)
            
#             # Log model size
#             total_params = sum(p.numel() for p in model.parameters())
#             trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#             logger.info(f"Model '{config['model']['architecture']}' created. "
#                        f"Params: Total={total_params:,}, Trainable={trainable_params:,}")
                       
#             # Create optimizer and scheduler
#             optimizer = get_optimizer(model, config)
#             num_epochs = config['training']['num_epochs']
#             scheduler, monitor_metric, scheduler_mode = get_scheduler(optimizer, config, num_epochs)
#         except Exception as e:
#             logger.exception(f"Fatal error creating model/optimizer: {e}")
#             return None, None

#     # --- Resume / Training Prep ---
#     start_epoch = 0
#     history = {
#         'train_loss': [], 
#         'val_loss': [], 
#         'train_pearson': [], 
#         'val_pearson': [], 
#         'lr': [], 
#         'epoch_time': []
#     }
#     best_metric_value = -float('inf') if scheduler_mode == 'max' else float('inf')
#     best_epoch = -1
    
#     resume_path = config['training'].get('resume_checkpoint')
#     if resume_path and os.path.exists(resume_path):
#         logger.info(f"Attempting resume from: {resume_path}")
#         try:
#             checkpoint = torch.load(resume_path, map_location=device)
            
#             # Load model state dict first
#             model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # Allow non-strict for flexibility
#             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#             start_epoch = checkpoint.get('epoch', -1) + 1
#             history = checkpoint.get('history', history)
            
#             # Load saved best metric value from checkpoint
#             best_metric_value_saved = checkpoint.get(f'best_{monitor_metric}', best_metric_value)
#             # Ensure compatibility if metric name changed
#             if isinstance(best_metric_value_saved, (float, int)):
#                 best_metric_value = best_metric_value_saved
                
#             best_epoch = checkpoint.get('best_epoch', -1)
            
#             # Load scheduler state if compatible
#             if scheduler and 'scheduler_state_dict' in checkpoint and type(scheduler).__name__ == checkpoint.get('scheduler_type'):
#                 try:
#                     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#                     logger.info("Resuming scheduler state.")
#                 except Exception as sch_e:
#                     logger.warning(f"Could not load scheduler state: {sch_e}")
                    
#             # Update config with potentially loaded temp params from checkpoint for consistency
#             if 'config' in checkpoint and 'runtime' in checkpoint['config'] and 'temp_scaling_params' in checkpoint['config']['runtime']:
#                 config['runtime']['temp_scaling_params'] = checkpoint['config']['runtime']['temp_scaling_params']
#                 logger.info("Loaded temp scaling params from checkpoint into runtime config.")

#             logger.info(f"Resumed from epoch {start_epoch}. Best '{monitor_metric}': {best_metric_value:.6f} @ epoch {best_epoch}")
#             del checkpoint
#             gc.collect()
#         except Exception as e:
#             logger.exception(f"Failed resume: {e}. Starting fresh.")
#             start_epoch = 0
#             history = {
#                 'train_loss': [], 
#                 'val_loss': [], 
#                 'train_pearson': [], 
#                 'val_pearson': [], 
#                 'lr': [], 
#                 'epoch_time': []
#             }
#             best_metric_value = -float('inf') if scheduler_mode == 'max' else float('inf')
#             best_epoch = -1

#     # --- Training Loop Setup ---
#     criterion = nn.MSELoss()
#     mixed_precision_enabled = config['training']['mixed_precision'].get('enabled', False)
#     use_amp = mixed_precision_enabled and device.type == 'cuda'
    
#     if use_amp:
#         logger.info("Using AMP")
#         scaler = torch.cuda.amp.GradScaler(enabled=True)
#     else:
#         logger.info("AMP disabled")
#         scaler = torch.cuda.amp.GradScaler(enabled=False)
        
#     # Early stopping configuration
#     early_stopping_cfg = config['training'].get('early_stopping', {})
#     early_stopping_enabled = early_stopping_cfg.get('enabled', True)
#     early_stopping_patience = early_stopping_cfg.get('patience', 10)
#     early_stopping_delta = early_stopping_cfg.get('min_delta', 0.001)
#     early_stopping_counter = 0
#     early_stopping_metric = early_stopping_cfg.get('monitor_metric', monitor_metric)
#     early_stopping_mode = early_stopping_cfg.get('mode', scheduler_mode)
    
#     if early_stopping_metric != monitor_metric or early_stopping_mode != scheduler_mode:
#         logger.warning(f"Early stopping metric/mode differs. Using scheduler's.")
#         early_stopping_metric = monitor_metric
#         early_stopping_mode = scheduler_mode
        
#     # Best model saving configuration
#     save_best_metric = config['training'].get('save_best_metric', monitor_metric)
#     save_best_mode = config['training'].get('save_best_mode', scheduler_mode)
    
#     # Gradient accumulation for large batches
#     gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
#     if gradient_accumulation_steps > 1:
#         logger.info(f"Using gradient accumulation: {gradient_accumulation_steps} steps")
#         logger.info(f"Effective batch size: {config['training']['batch_size'] * gradient_accumulation_steps}")

#     # --- TRAINING LOOP ---
#     start_train_loop = time.time()
#     log_section_header(logger, f"STARTING TRAINING LOOP (Epochs {start_epoch+1} to {num_epochs})")
#     for epoch in range(start_epoch, num_epochs):
#         epoch_start_time = time.time()
        
#         # Train epoch
#         avg_epoch_train_loss, avg_epoch_train_corr = train_epoch(
#             model, train_loader, criterion, optimizer, device, epoch, config, scaler, timing
#         )
        
#         # Validation epoch
#         avg_epoch_val_loss, avg_epoch_val_corr = validate(
#             model, val_loader, criterion, device, config, timing
#         )
        
#         # --- Log, History, Scheduler ---
#         history['train_loss'].append(avg_epoch_train_loss)
#         history['train_pearson'].append(avg_epoch_train_corr)
#         history['val_loss'].append(avg_epoch_val_loss)
#         history['val_pearson'].append(avg_epoch_val_corr)
#         history['lr'].append(optimizer.param_groups[0]['lr'])
        
#         epoch_duration = time.time() - epoch_start_time
#         history['epoch_time'].append(epoch_duration)
        
#         logger.info(f"--- Epoch {epoch+1}/{num_epochs} --- | Time: {epoch_duration:.1f}s | "
#                   f"LR: {history['lr'][-1]:.2e}")
#         logger.info(f"  Train -> Loss: {avg_epoch_train_loss:.6f} | Pearson: {avg_epoch_train_corr:.6f}")
#         logger.info(f"  Valid -> Loss: {avg_epoch_val_loss:.6f} | Pearson: {avg_epoch_val_corr:.6f}")
        
#         # Update scheduler
#         current_metric_val = avg_epoch_val_corr if monitor_metric == 'val_pearson' else avg_epoch_val_loss
#         if scheduler:
#             if isinstance(scheduler, ReduceLROnPlateau):
#                 scheduler.step(current_metric_val)
#             else:
#                 scheduler.step()
                
#         # --- Checkpointing & Early Stopping ---
#         is_best = False
#         if not np.isnan(current_metric_val):
#             if save_best_mode == 'max' and current_metric_val > best_metric_value + early_stopping_delta:
#                 is_best = True
#             elif save_best_mode == 'min' and current_metric_val < best_metric_value - early_stopping_delta:
#                 is_best = True
#         else:
#             logger.warning(f"Ep {epoch+1}: Monitored metric '{save_best_metric}' NaN.")
            
#         # Create checkpoint data
#         checkpoint_data = {
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'config': config,  # Save potentially updated config
#             'input_shape': input_shape,
#             'temp_scaling_params': config['runtime']['temp_scaling_params'],  # Explicitly save params used
#             'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
#             'scheduler_type': type(scheduler).__name__ if scheduler else None,
#             f'best_{save_best_metric}': best_metric_value,
#             'best_epoch': best_epoch  # Save best info in all checkpoints
#         }

#         if is_best:
#             logger.info(f"  >>> New best {save_best_metric}: {current_metric_val:.6f}. Saving best model...")
#             best_metric_value = current_metric_val
#             best_epoch = epoch + 1
#             early_stopping_counter = 0
            
#             best_model_path = os.path.join(models_dir, "best_model.pt")
#             best_checkpoint_data = checkpoint_data.copy()  # Already includes best info
#             try:
#                 torch.save(best_checkpoint_data, best_model_path)
#             except Exception as e:
#                 logger.error(f"Failed save best model: {e}")
#         else:
#             early_stopping_counter += 1
#             logger.info(f"  {save_best_metric} did not improve. Best: {best_metric_value:.6f} @ Ep {best_epoch}. "
#                       f"EarlyStop: {early_stopping_counter}/{early_stopping_patience}")

#         # Save periodic checkpoints if configured
#         chkpt_interval = config['training'].get('checkpoint_interval', 0)
#         if chkpt_interval > 0 and (epoch + 1) % chkpt_interval == 0:
#             chkpt_path = os.path.join(models_dir, f"checkpoint_epoch_{epoch+1}.pt")
#             logger.info(f"Saving periodic checkpoint: {chkpt_path}")
#             periodic_checkpoint_data = checkpoint_data.copy()
#             periodic_checkpoint_data['history'] = history
#             try:
#                 torch.save(periodic_checkpoint_data, chkpt_path)
#             except Exception as e:
#                 logger.error(f"Failed save periodic ckpt: {e}")
                
#         # Always save latest model
#         latest_model_path = os.path.join(models_dir, "latest_model.pt")
#         latest_checkpoint_data = checkpoint_data.copy()
#         latest_checkpoint_data['history'] = history
#         try:
#             torch.save(latest_checkpoint_data, latest_model_path)
#         except Exception as e:
#             logger.error(f"Failed save latest model: {e}")
            
#         # Check for early stopping
#         if early_stopping_enabled and early_stopping_counter >= early_stopping_patience:
#             logger.info(f"Early stopping triggered at epoch {epoch+1}.")
#             break
            
#         # Clean up memory
#         clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))
#         if config.get('logging', {}).get('log_memory_usage', True):
#             log_memory_usage(logger)
            
#         # Log timing stats if enabled
#         if timing:
#             stats = timing.get_stats()
#             logger.debug("--- Epoch Timing Stats ---")
#             for section, s_data in stats.items():
#                 logger.debug(f"  {section:<20}: avg={s_data['avg']:.4f}s, total={s_data['total']:.2f}s, count={s_data['count']}")
#             logger.debug("--------------------------")

#     # --- Finalization ---
#     training_duration = time.time() - start_train_loop
#     log_section_header(logger, "TRAINING FINISHED")
#     logger.info(f"Total Training Time: {training_duration:.2f}s ({training_duration/60:.1f} mins)")
    
#     final_model_path = None
#     if best_epoch != -1:
#         logger.info(f"Best {save_best_metric}: {best_metric_value:.6f} @ epoch {best_epoch}")
#         final_model_path = os.path.join(models_dir, "best_model.pt")
#     else:
#         logger.warning("No improvement detected during training.")
#         final_model_path = os.path.join(models_dir, "latest_model.pt")
#         logger.info(f"Using latest model: {final_model_path}")
        
#     if not final_model_path or not os.path.exists(final_model_path):
#         logger.error("Could not find final model.")
#         final_model_path = None
        
#     # Save training history
#     history_path = os.path.join(run_output_dir, "training_history.json")
#     try:
#         save_json(history, history_path)
#         logger.info(f"History saved: {history_path}")
#     except Exception as e:
#         logger.error(f"Failed save history: {e}")
        
#     # Generate basic plots if visualization module is available
#     try:
#         from voxelflex.cli.commands.visualize import create_loss_curve, create_correlation_curve
#         viz_dir = config["output"]["visualizations_dir"]
#         logger.info("Generating final plots...")
        
#         if viz_dir:
#             if history.get('train_loss') and history.get('val_loss'):
#                 create_loss_curve(
#                     history, viz_dir, 
#                     save_format=config['visualization']['save_format'], 
#                     dpi=config['visualization']['dpi']
#                 )
                
#             if history.get('train_pearson') and history.get('val_pearson'):
#                 create_correlation_curve(
#                     history, viz_dir, 
#                     save_format=config['visualization']['save_format'], 
#                     dpi=config['visualization']['dpi']
#                 )
#     except ImportError:
#         logger.warning("Could not import visualize functions.")
        
#     # Final cleanup
#     clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))
#     return final_model_path, history




def train_model(config: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, List[float]]]]:
    """
    Main function to train the model using chunked data loading.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (path to best model, training history)
    """
    run_output_dir = config["output"]["run_dir"]
    models_dir = config["output"]["models_dir"]
    ensure_dir(models_dir)

    log_section_header(logger, f"STARTING TRAINING RUN: {config['output']['run_name']} (Chunked Loading)")
    log_memory_usage(logger)

    # Set up environment
    device = get_device(config["system_utilization"].get("adjust_for_gpu", True))
    seed = config['training']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        # Consider adding for reproducibility, but might impact performance slightly
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    log_timing = config.get('logging', {}).get('log_timing', False)
    timing = Timing() if log_timing else None

    # --- Load Datasets & DataLoaders ---
    with log_stage("DATA_PREPARATION", "Creating Datasets & DataLoaders (Chunked)"):
        try:
            # *** NOTE: Resolving paths here using the function ***
            master_samples_path = resolve_path(config["data"]["master_samples_path"])
            if not os.path.exists(master_samples_path):
                raise FileNotFoundError(f"Master sample file missing: {master_samples_path}")

            voxel_hdf5_path = resolve_path(config['input']['voxel_file'])
            if not os.path.exists(voxel_hdf5_path):
                raise FileNotFoundError(f"HDF5 file missing: {voxel_hdf5_path}")

            # Set up voxel shape information
            voxel_c = config['model']['input_channels']
            voxel_d = config['model'].get('voxel_depth', 21)
            voxel_h = config['model'].get('voxel_height', 21)
            voxel_w = config['model'].get('voxel_width', 21)
            voxel_shape = (voxel_c, voxel_d, voxel_h, voxel_w)

            # Load temp scaling params with better error handling
            temp_scaling_params = None
            scaler_path = resolve_path(config.get('data', {}).get('temp_scaling_params_file'))
            if scaler_path and os.path.exists(scaler_path):
                try:
                    temp_scaling_params = load_json(scaler_path)
                    logger.info(f"Loaded temp scaling: {temp_scaling_params}")
                except Exception as e:
                    logger.error(f"Failed to load scaler file '{scaler_path}': {e}")
                    temp_scaling_params = None # Fallback to default

            if not temp_scaling_params:
                temp_scaling_params = {'temp_min': 320.0, 'temp_max': 450.0} # Define default scaling
                logger.warning(f"Using default temp scaling: {temp_scaling_params}")

            config.setdefault('runtime', {})['temp_scaling_params'] = temp_scaling_params

            # Load domain lists
            train_split_file = resolve_path(config['input'].get('train_split_file'))
            val_split_file = resolve_path(config['input'].get('val_split_file'))

            if not train_split_file or not os.path.exists(train_split_file):
                raise FileNotFoundError(f"Train split file missing: {train_split_file}")

            if not val_split_file or not os.path.exists(val_split_file):
                raise FileNotFoundError(f"Validation split file missing: {val_split_file}")

            train_domain_list = load_list_from_file(train_split_file)
            val_domain_list = load_list_from_file(val_split_file)

            if not train_domain_list:
                raise ValueError("Training domain list is empty.")

            if not val_domain_list:
                raise ValueError("Validation domain list is empty.")

            # Apply subset percentage if specified
            train_subset_fraction = config['training'].get('subset', 1.0)
            if 0.0 < train_subset_fraction < 1.0:
                 subset_size = int(len(train_domain_list) * train_subset_fraction)
                 if subset_size < 1: subset_size = 1 # Ensure at least one domain
                 # Shuffle before taking subset for randomness
                 np.random.shuffle(train_domain_list)
                 train_domain_list = train_domain_list[:subset_size]
                 logger.info(f"Using a subset of {len(train_domain_list)} training domains ({train_subset_fraction:.1%})")


            # Estimate samples for progress bars (handle PyArrow availability correctly)
            try:
                # Check PyArrow availability
                try:
                    import pyarrow.parquet as pq
                    PYARROW_AVAILABLE = True
                except ImportError:
                    PYARROW_AVAILABLE = False

                logger.debug("Estimating samples per epoch for progress bars...")
                n_train, n_val = 0, 0

                # Create sets of domains actually used
                train_domain_set = set(train_domain_list)
                val_domain_set = set(val_domain_list)

                if PYARROW_AVAILABLE and master_samples_path.endswith(".parquet"):
                    # Efficiently read only necessary columns and filter
                    filters = [('split', 'in', ['train', 'val'])]
                    # Read relevant splits and domain IDs
                    meta = pq.read_table(master_samples_path, columns=['split', 'hdf5_domain_id'], filters=filters).to_pandas()
                    n_train = len(meta[(meta['split'] == 'train') & (meta['hdf5_domain_id'].isin(train_domain_set))])
                    n_val = len(meta[(meta['split'] == 'val') & (meta['hdf5_domain_id'].isin(val_domain_set))])
                    del meta
                elif master_samples_path.endswith(".csv"):
                     try:
                        df_full = pd.read_csv(master_samples_path, usecols=['split', 'hdf5_domain_id'], dtype={'split': 'category', 'hdf5_domain_id': str})
                        n_train = len(df_full[(df_full['split'] == 'train') & (df_full['hdf5_domain_id'].isin(train_domain_set))])
                        n_val = len(df_full[(df_full['split'] == 'val') & (df_full['hdf5_domain_id'].isin(val_domain_set))])
                        del df_full
                     except MemoryError:
                        logger.warning("MemoryError reading full CSV for sample estimation. Progress bars may be inaccurate.")
                        n_train, n_val = None, None # Indicate estimation failed
                else:
                     logger.warning(f"Unsupported master samples format for estimation: {master_samples_path}. Progress bars may be inaccurate.")
                     n_train, n_val = None, None

                # Only set if estimation was successful
                if n_train is not None: config['runtime']['estimated_samples_per_epoch'] = n_train
                if n_val is not None: config['runtime']['estimated_val_samples'] = n_val
                logger.info(f"Estimated samples: Train={n_train if n_train is not None else 'N/A'}, Val={n_val if n_val is not None else 'N/A'}")

                gc.collect()
            except Exception as e:
                logger.warning(f"Could not estimate sample counts from {master_samples_path}: {e}. "
                              f"Progress bars may be inaccurate or indefinite.")


            # Instantiate Datasets using ChunkedVoxelDataset
            train_chunk_size = config['training'].get('chunk_size', 100)
            logger.info(f"Creating training dataset (ChunkedIterable, chunk_size={train_chunk_size})")
            train_dataset = ChunkedVoxelDataset(
                master_samples_path=master_samples_path,
                split='train',
                domain_list=train_domain_list, # Use potentially subsetted list
                voxel_hdf5_path=voxel_hdf5_path,
                temp_scaling_params=temp_scaling_params,
                chunk_size=train_chunk_size,
                shuffle_domain_list=True  # Shuffle for training
            )

            # Check if metadata lookup is populated
            if not train_dataset.metadata_lookup:
                 raise ValueError("Training metadata lookup is empty after dataset initialization. Check preprocessing and split files.")
            logger.info(f"Train dataset created with {len(train_domain_list)} domains and {len(train_dataset.metadata_lookup)} metadata entries")


            logger.info(f"Creating validation dataset (ChunkedIterable, chunk_size={train_chunk_size})")
            # Use same chunk size for validation loading? Or maybe larger if RAM allows? Configurable?
            val_chunk_size = config['training'].get('val_chunk_size', train_chunk_size)
            val_dataset = ChunkedVoxelDataset(
                master_samples_path=master_samples_path,
                split='val',
                domain_list=val_domain_list,
                voxel_hdf5_path=voxel_hdf5_path,
                temp_scaling_params=temp_scaling_params,
                chunk_size=val_chunk_size,
                shuffle_domain_list=False  # No shuffle for validation
            )
            if not val_dataset.metadata_lookup:
                 logger.warning("Validation metadata lookup is empty after dataset initialization.")
            logger.info(f"Val dataset created with {len(val_domain_list)} domains and {len(val_dataset.metadata_lookup)} metadata entries")


            # Instantiate DataLoaders
            train_batch_size = config['training']['batch_size']
            val_batch_size = config.get('prediction', {}).get('batch_size', train_batch_size * 2)
            num_workers = config['training'].get('num_workers', 4)
            pin_memory = config['training'].get('pin_memory', True) and (device.type == 'cuda')
            prefetch_factor = config['training'].get('prefetch_factor', 2) if num_workers > 0 else None

            logger.info(f"Train DataLoader: Chunked, BatchSize={train_batch_size}, Workers={num_workers}, Prefetch={prefetch_factor}")
            train_loader = DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                worker_init_fn=worker_init_fn,
                collate_fn=simple_collate_fn,
                prefetch_factor=prefetch_factor,
                # persistent_workers=(num_workers > 0)
            )

            val_num_workers = config['training'].get('val_num_workers', num_workers)
            val_prefetch_factor = config['training'].get('val_prefetch_factor', prefetch_factor)
            logger.info(f"Val DataLoader: Chunked, BatchSize={val_batch_size}, Workers={val_num_workers}, Prefetch={val_prefetch_factor}")
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_batch_size,
                num_workers=val_num_workers,
                pin_memory=pin_memory,
                worker_init_fn=worker_init_fn,
                collate_fn=simple_collate_fn,
                prefetch_factor=val_prefetch_factor,
                # persistent_workers=(val_num_workers > 0)
            )

        except Exception as e:
            logger.exception(f"Fatal error creating Datasets/DataLoaders: {e}")
            return None, None

    # --- Model & Optimizer ---
    with log_stage("MODEL_CREATION", "Creating model and optimizer"):
        try:
            input_shape = voxel_shape
            logger.info(f"Using model input shape: {input_shape}")
            model = get_model(config['model'], input_shape=input_shape)
            model.to(device)

            # Log model size
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model '{config['model']['architecture']}' created. "
                       f"Params: Total={total_params:,}, Trainable={trainable_params:,}")

            # Create optimizer and scheduler
            optimizer = get_optimizer(model, config)
            num_epochs = config['training']['num_epochs']
            scheduler, monitor_metric, scheduler_mode = get_scheduler(optimizer, config, num_epochs)
        except Exception as e:
            logger.exception(f"Fatal error creating model/optimizer: {e}")
            return None, None

    # --- Resume / Training Prep ---
    start_epoch = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_pearson': [],
        'val_pearson': [],
        'lr': [],
        'epoch_time': []
    }
    best_metric_value = -float('inf') if scheduler_mode == 'max' else float('inf')
    best_epoch = -1

    resume_path = config['training'].get('resume_checkpoint')
    if resume_path:
         resume_path_resolved = resolve_path(resume_path) # Resolve path
         if os.path.exists(resume_path_resolved):
            logger.info(f"Attempting resume from: {resume_path_resolved}")
            try:
                checkpoint = torch.load(resume_path_resolved, map_location=device)

                # Load model state dict first (allow non-strict for flexibility)
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                if missing_keys: logger.warning(f"Resume Warning: Missing keys in model state_dict: {missing_keys}")
                if unexpected_keys: logger.warning(f"Resume Warning: Unexpected keys in model state_dict: {unexpected_keys}")

                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint.get('epoch', -1) + 1 # Start next epoch
                loaded_history = checkpoint.get('history', history) # Get history from checkpoint

                # Ensure history lists are appended correctly, not overwritten
                # And handle potential type issues from older checkpoints
                for key in history.keys():
                     if key in loaded_history and isinstance(loaded_history[key], list):
                          # Take only up to the saved epoch index
                          history[key] = loaded_history[key][:start_epoch]
                     else:
                          history[key] = [] # Reset if format mismatch

                # Load saved best metric value from checkpoint, ensure compatibility
                best_metric_value_saved = checkpoint.get(f'best_{monitor_metric}', best_metric_value)
                if isinstance(best_metric_value_saved, (float, int)) and np.isfinite(best_metric_value_saved):
                    best_metric_value = best_metric_value_saved
                else: # Reset if invalid type or NaN/inf
                     best_metric_value = -float('inf') if scheduler_mode == 'max' else float('inf')


                best_epoch = checkpoint.get('best_epoch', -1)

                # Load scheduler state if compatible
                if scheduler and 'scheduler_state_dict' in checkpoint and type(scheduler).__name__ == checkpoint.get('scheduler_type'):
                    try:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        logger.info("Resuming scheduler state.")
                    except Exception as sch_e:
                        logger.warning(f"Could not load scheduler state (may reset LR): {sch_e}")

                # Load GradScaler state if present and using AMP
                # Check for scaler state before initializing the main scaler instance
                scaler_state_dict = checkpoint.get('scaler_state_dict')
                # if use_amp and 'scaler_state_dict' in checkpoint: # This check happens before scaler init
                #      try:
                #           scaler = torch.cuda.amp.GradScaler() # Need to instantiate first
                #           scaler.load_state_dict(checkpoint['scaler_state_dict'])
                #           logger.info("Resuming GradScaler state.")
                #      except Exception as scale_e:
                #           logger.warning(f"Could not load GradScaler state: {scale_e}")
                #           scaler = torch.cuda.amp.GradScaler(enabled=use_amp) # Re-init if loading failed
                # else:
                #      scaler = torch.cuda.amp.GradScaler(enabled=use_amp) # Standard init


                # Update config with potentially loaded temp params from checkpoint for consistency
                if 'config' in checkpoint and 'runtime' in checkpoint['config'] and 'temp_scaling_params' in checkpoint['config']['runtime']:
                    loaded_temp_params = checkpoint['config']['runtime']['temp_scaling_params']
                    logger.info(f"Loaded temp scaling params from checkpoint ({loaded_temp_params}). Using these.")
                    # config['runtime']['temp_scaling_params'] = loaded_temp_params # Update runtime config if needed

                logger.info(f"Resumed from epoch {start_epoch}. Best '{monitor_metric}': {best_metric_value:.6f} @ epoch {best_epoch}")
                del checkpoint
                gc.collect()
            except Exception as e:
                logger.exception(f"Failed to load checkpoint from {resume_path_resolved}: {e}. Starting fresh.")
                start_epoch = 0
                history = {k: [] for k in history} # Reset history
                best_metric_value = -float('inf') if scheduler_mode == 'max' else float('inf')
                best_epoch = -1
                scaler_state_dict = None # Ensure scaler state is not loaded
         else:
              logger.warning(f"Resume checkpoint specified but not found: {resume_path_resolved}. Starting fresh.")
              scaler_state_dict = None


    # --- Training Loop Setup ---
    criterion = nn.MSELoss()
    mixed_precision_enabled = config['training'].get('mixed_precision', {}).get('enabled', True)
    use_amp = mixed_precision_enabled and device.type == 'cuda'

    # Initialize GradScaler here, after potential resume attempt
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        logger.info("Using Automatic Mixed Precision (AMP).")
        # Load state if it was successfully retrieved from checkpoint
        if 'scaler_state_dict' in locals() and scaler_state_dict is not None:
            try:
                scaler.load_state_dict(scaler_state_dict)
                logger.info("Loaded GradScaler state from checkpoint.")
            except Exception as e:
                 logger.warning(f"Could not load scaler state dict during setup: {e}")
    else:
        logger.info("AMP disabled.")


    # Early stopping configuration
    early_stopping_cfg = config['training'].get('early_stopping', {})
    early_stopping_enabled = early_stopping_cfg.get('enabled', True)
    early_stopping_patience = early_stopping_cfg.get('patience', 10)
    early_stopping_delta = early_stopping_cfg.get('min_delta', 0.001)
    early_stopping_counter = 0 # Initialize counter
    # Determine metric and mode for early stopping, ensure consistency with scheduler
    early_stopping_metric = early_stopping_cfg.get('monitor_metric', monitor_metric)
    early_stopping_mode = early_stopping_cfg.get('mode', scheduler_mode)
    if early_stopping_metric != monitor_metric or early_stopping_mode != scheduler_mode:
        logger.warning(f"Early stopping metric/mode differs from scheduler. Using scheduler's: {monitor_metric}/{scheduler_mode}")
        early_stopping_metric = monitor_metric
        early_stopping_mode = scheduler_mode
    logger.info(f"Early Stopping: {'Enabled' if early_stopping_enabled else 'Disabled'}. Metric: {early_stopping_metric}, Mode: {early_stopping_mode}, Patience: {early_stopping_patience}, Delta: {early_stopping_delta}")


    # Best model saving configuration
    save_best_metric = config['training'].get('save_best_metric', monitor_metric)
    save_best_mode = config['training'].get('save_best_mode', scheduler_mode)
    # Use consistent metric/mode for comparison logic
    _comparison_metric = early_stopping_metric # Use the (potentially corrected) early stopping metric
    _comparison_mode = early_stopping_mode   # Use the (potentially corrected) early stopping mode


    # Gradient accumulation for large batches
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    if gradient_accumulation_steps > 1:
        logger.info(f"Using gradient accumulation: {gradient_accumulation_steps} steps")
        effective_batch_size = config['training']['batch_size'] * gradient_accumulation_steps
        logger.info(f"Effective batch size: {effective_batch_size}")


    # --- TRAINING LOOP ---
    start_train_loop = time.time()
    log_section_header(logger, f"STARTING TRAINING LOOP (Epochs {start_epoch+1} to {num_epochs})")
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        # Train epoch
        avg_epoch_train_loss, avg_epoch_train_corr = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config, scaler, timing
        )

        # Validation epoch
        if len(val_loader) > 0 and len(val_dataset.metadata_lookup) > 0:
             avg_epoch_val_loss, avg_epoch_val_corr = validate(
                 model, val_loader, criterion, device, config, timing
             )
        else:
             logger.warning(f"Epoch {epoch+1}: Skipping validation due to empty validation dataset/loader.")
             avg_epoch_val_loss, avg_epoch_val_corr = float('inf'), 0.0 # Assign default poor values

        # --- Log, History, Scheduler ---
        history['train_loss'].append(avg_epoch_train_loss)
        history['train_pearson'].append(avg_epoch_train_corr)
        history['val_loss'].append(avg_epoch_val_loss)
        history['val_pearson'].append(avg_epoch_val_corr)
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        epoch_duration = time.time() - epoch_start_time
        history['epoch_time'].append(epoch_duration)

        logger.info(f"--- Epoch {epoch+1}/{num_epochs} --- | Time: {epoch_duration:.1f}s | "
                  f"LR: {current_lr:.2e}")
        logger.info(f"  Train -> Loss: {avg_epoch_train_loss:.6f} | Pearson: {avg_epoch_train_corr:.6f}")
        logger.info(f"  Valid -> Loss: {avg_epoch_val_loss:.6f} | Pearson: {avg_epoch_val_corr:.6f}")

        # Update scheduler
        # Use the metric designated for scheduler monitoring
        scheduler_metric_val = avg_epoch_val_corr if monitor_metric == 'val_pearson' else avg_epoch_val_loss
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                 # Ensure scheduler step gets a valid number
                 if np.isfinite(scheduler_metric_val):
                      scheduler.step(scheduler_metric_val)
                 else:
                      logger.warning(f"Scheduler {monitor_metric} is NaN/inf. Skipping scheduler step.")
            else: # For other schedulers like CosineAnnealing, StepLR
                scheduler.step()

        # --- Checkpointing & Early Stopping ---
        is_best = False
        # Use the metric designated for early stopping / best model saving comparison
        current_comparison_metric_val = avg_epoch_val_corr if _comparison_metric == 'val_pearson' else avg_epoch_val_loss

        if np.isfinite(current_comparison_metric_val): # Check if the metric is valid
            # Apply delta for comparison
            improvement_threshold = early_stopping_delta
            if _comparison_mode == 'max':
                if current_comparison_metric_val > best_metric_value + improvement_threshold:
                    is_best = True
            elif _comparison_mode == 'min':
                if current_comparison_metric_val < best_metric_value - improvement_threshold:
                    is_best = True
        else:
            logger.warning(f"Epoch {epoch+1}: Monitored metric '{_comparison_metric}' for best model/early stopping is NaN/inf.")


        # --- Checkpointing ---
        # Prepare checkpoint data dictionary
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,  # Save config used for this run
            'input_shape': input_shape,
            'temp_scaling_params': config['runtime']['temp_scaling_params'],
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scheduler_type': type(scheduler).__name__ if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if use_amp else None, # Save scaler state if using AMP
            f'best_{_comparison_metric}': best_metric_value, # Save the best value *so far*
            'best_epoch': best_epoch # Save the epoch number where best value was achieved
            # Note: History is NOT saved in individual model checkpoints, only in latest/final history file
        }


        if is_best:
            logger.info(f"  >>> New best {_comparison_metric}: {current_comparison_metric_val:.6f} (vs {best_metric_value:.6f}). Saving best model...")
            best_metric_value = current_comparison_metric_val # Update best value
            best_epoch = epoch + 1 # Update best epoch (use 1-based index for logging)
            early_stopping_counter = 0 # Reset early stopping counter

            best_model_path = os.path.join(models_dir, "best_model.pt")
            try:
                # Save checkpoint specifically for the best model
                # Update the best metric value in the checkpoint being saved
                checkpoint_data[f'best_{_comparison_metric}'] = best_metric_value
                checkpoint_data['best_epoch'] = best_epoch
                torch.save(checkpoint_data, best_model_path)
                logger.info(f"Best model checkpoint saved to {best_model_path}")
            except Exception as e:
                logger.error(f"Failed save best model checkpoint: {e}")
        else:
            # Only increment counter if early stopping is enabled
            if early_stopping_enabled:
                 early_stopping_counter += 1
                 logger.info(f"  {_comparison_metric} did not improve by delta {early_stopping_delta}. Best: {best_metric_value:.6f} @ Ep {best_epoch}. "
                             f"EarlyStop Counter: {early_stopping_counter}/{early_stopping_patience}")


        # Save periodic checkpoints if configured
        chkpt_interval = config['training'].get('checkpoint_interval', 0) # Default 0 means disabled
        if chkpt_interval > 0 and (epoch + 1) % chkpt_interval == 0:
            chkpt_path = os.path.join(models_dir, f"checkpoint_epoch_{epoch+1}.pt")
            logger.info(f"Saving periodic checkpoint: {chkpt_path}")
            try:
                # Save same checkpoint data, history is saved separately at the end
                torch.save(checkpoint_data, chkpt_path)
            except Exception as e:
                logger.error(f"Failed save periodic checkpoint: {e}")

        # Always save latest model state (can be used for resume)
        latest_model_path = os.path.join(models_dir, "latest_model.pt")
        try:
            # Include history in the 'latest' checkpoint for easier resume if needed?
            # Let's keep history separate for now to avoid large latest checkpoint.
            torch.save(checkpoint_data, latest_model_path)
        except Exception as e:
            logger.error(f"Failed save latest model checkpoint: {e}")

        # Check for early stopping condition
        if early_stopping_enabled and early_stopping_counter >= early_stopping_patience:
            logger.info(f"EARLY STOPPING triggered at epoch {epoch+1} after {early_stopping_patience} epochs without improvement.")
            break # Exit the training loop

        # Clean up memory at end of epoch
        clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))
        if config.get('logging', {}).get('log_memory_usage', True):
            log_memory_usage(logger, level=logging.DEBUG) # Log memory usage (DEBUG level perhaps)

        # Log timing stats if enabled
        if timing:
            # Maybe log only every few epochs or if DEBUG level is high?
            if (epoch + 1) % 5 == 0 or config.get('logging', {}).get('level') == 'DEBUG':
                 stats = timing.get_stats()
                 logger.debug("--- Epoch Timing Stats ---")
                 for section, s_data in stats.items():
                     logger.debug(f"  {section:<20}: avg={s_data['avg']:.4f}s, total={s_data['total']:.2f}s, count={s_data['count']}")
                 logger.debug("--------------------------")
                 timing.reset() # Reset timing for next interval if logging periodically


    # --- Finalization ---
    training_duration = time.time() - start_train_loop
    log_section_header(logger, "TRAINING FINISHED")
    logger.info(f"Total Training Time: {training_duration:.2f}s ({training_duration/60:.1f} mins)")

    final_model_path = None
    if best_epoch != -1:
        logger.info(f"Best {_comparison_metric} achieved: {best_metric_value:.6f} at epoch {best_epoch}")
        # Verify best model file exists before declaring it final
        _best_path_check = os.path.join(models_dir, "best_model.pt")
        if os.path.exists(_best_path_check):
             final_model_path = _best_path_check
             logger.info(f"Using best model: {final_model_path}")
        else:
             logger.warning("Best metric was recorded, but best_model.pt not found. Falling back to latest.")
             final_model_path = os.path.join(models_dir, "latest_model.pt")
             if not os.path.exists(final_model_path): final_model_path = None # Check if latest exists

    else: # No improvement or validation failed consistently
        logger.warning("No improvement detected during training or validation failed.")
        final_model_path = os.path.join(models_dir, "latest_model.pt")
        if os.path.exists(final_model_path):
             logger.info(f"Using latest model: {final_model_path}")
        else:
             logger.error("No models saved (latest_model.pt not found).")
             final_model_path = None


    # Save training history
    history_path = os.path.join(run_output_dir, "training_history.json")
    try:
        # *** FIX: Convert history contents to JSON-serializable types before saving ***
        serializable_history = convert_to_serializable(history)
        save_json(serializable_history, history_path)
        logger.info(f"Training history saved to {history_path}")
    except Exception as e:
        logger.error(f"Failed to save training history JSON: {e}")
        # Don't return history if saving failed? Or return original history?
        history = None # Indicate history saving failed

    # Generate basic plots if visualization module is available
    # Check if history is valid before plotting
    if history is not None:
         try:
             from voxelflex.cli.commands.visualize import create_loss_curve, create_correlation_curve
             viz_dir = config["output"]["visualizations_dir"]
             # Ensure visualization dir exists (should be created by config.py)
             ensure_dir(viz_dir)
             logger.info("Generating final training plots...")

             plot_config = config.get("visualization", {})
             save_format = plot_config.get("save_format", "png")
             dpi = plot_config.get("dpi", 150)
             save_plot_data = plot_config.get("save_plot_data", True) # Pass save_data flag

             if viz_dir:
                 # Check specific history keys exist and are not empty before plotting
                 if plot_config.get("plot_loss", True) and history.get('train_loss') and history.get('val_loss'):
                     create_loss_curve(
                         history, viz_dir,
                         save_format=save_format,
                         dpi=dpi,
                         save_data=save_plot_data # Pass flag
                     )

                 if plot_config.get("plot_correlation", True) and history.get('train_pearson') and history.get('val_pearson'):
                     create_correlation_curve(
                         history, viz_dir,
                         save_format=save_format,
                         dpi=dpi,
                         save_data=save_plot_data # Pass flag
                     )
         except ImportError:
             logger.warning("Visualization module not found or failed to import. Skipping final plots.")
         except Exception as plot_e:
              logger.error(f"Error during final plot generation: {plot_e}")


    # Final cleanup
    clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))
    return final_model_path, history # Return potentially None history if save failed