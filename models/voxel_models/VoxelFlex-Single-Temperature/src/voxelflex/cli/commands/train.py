"""
Training command for Voxelflex.

This module handles the training of RMSF prediction models.
"""

import os
import time
import json
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
import gc
import torch.nn as nn
import h5py
import torch.optim as optim
from torch.utils.data import DataLoader

from voxelflex.data.data_loader import load_voxel_data, load_domain_batch, load_rmsf_data, prepare_dataloaders, create_domain_mapping, check_memory_usage, clear_memory, RMSFDataset, validate_rmsf_data
from voxelflex.models.cnn_models import get_model
from voxelflex.utils.logging_utils import pipeline_tracker, ProgressBar, get_logger, setup_logging, EnhancedProgressBar, log_memory_usage, log_operation_result, log_section_header, log_stage, log_step
from voxelflex.utils.file_utils import ensure_dir, save_json, load_json, save_domain_registry, load_domain_registry
from voxelflex.utils.system_utils import get_device, check_system_resources, clear_memory, check_memory_usage, emergency_memory_reduction, set_num_threads, is_memory_critical, estimate_batch_size, adjust_workers_for_memory, MEMORY_WARNING_THRESHOLD, MEMORY_CRITICAL_THRESHOLD, MEMORY_EMERGENCY_THRESHOLD

logger = get_logger(__name__)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    show_progress: bool = True,
    memory_efficient: bool = True,  # Enable memory efficiency by default
    scaler = None,  # Add scaler for mixed precision training
    gradient_clip_norm: Optional[float] = None  # For gradient clipping
) -> float:
    """
    Train model for one epoch with enhanced GPU utilization.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (CPU or GPU)
        epoch: Current epoch number
        show_progress: Whether to show progress bar
        memory_efficient: Whether to use memory-efficient mode
        scaler: GradScaler for mixed precision training
        gradient_clip_norm: Maximum norm for gradient clipping
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader)
    logger = get_logger(__name__)
    
    if show_progress:
        # Enhanced progress bar with memory tracking
        progress = EnhancedProgressBar(
            total=total_batches, 
            prefix=f"Epoch {epoch+1} Training", 
            suffix="Complete",
            stage_info="Training"
        )
    
    # Log at beginning of epoch
    log_memory_usage(logger)
    
    # Track time for detailed performance monitoring
    batch_times = []
    forward_times = []
    backward_times = []
    
    # MEMORY OPTIMIZATION: Process in larger chunks (increased from 50)
    processed_batches = 0
    chunk_size = 200 if memory_efficient else total_batches  # Process 200 batches at a time
    
    while processed_batches < total_batches:
        end_batch = min(processed_batches + chunk_size, total_batches)
        
        # Process a chunk of batches
        for i, (inputs, targets) in enumerate(train_loader):
            if i < processed_batches:
                continue
            if i >= end_batch:
                break
            
            batch_start = time.time()
            
            # Move data to device
            inputs = inputs.to(device, non_blocking=True)  # Use non_blocking for better CPU-GPU overlap
            targets = targets.to(device, non_blocking=True)
            
            # Zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)  # set_to_none=True is more memory efficient
            
            # Forward pass with mixed precision if available
            forward_start = time.time()
            
            # Use mixed precision training if scaler is provided
            if scaler is not None and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # Backward pass with scaler and gradient clipping
                backward_start = time.time()
                scaler.scale(loss).backward()
                
                # Apply gradient clipping if enabled
                if gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision training
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass
                backward_start = time.time()
                loss.backward()
                
                # Apply gradient clipping if enabled
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                
                optimizer.step()
            
            forward_time = backward_start - forward_start
            backward_time = time.time() - backward_start
            
            forward_times.append(forward_time)
            backward_times.append(backward_time)
            
            # Statistics
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            running_loss += loss.item()
            
            if show_progress:
                progress.update(i + 1)
            
            # Periodically log detailed performance metrics (reduced frequency)
            if (i + 1) % 100 == 0 or i + 1 == total_batches:  # Increased from 50
                logger.debug(
                    f"Batch {i+1}/{total_batches} - Loss: {loss.item():.6f}, "
                    f"Time: {batch_time:.3f}s (F: {forward_time:.3f}s, B: {backward_time:.3f}s)"
                )
            
            # Monitor for training divergence
            if loss.item() > 10.0:  # Set a reasonable threshold for divergence
                logger.warning(f"Potential training divergence detected. Batch loss: {loss.item():.6f}")
                
                # Additional debugging for divergence
                if gradient_clip_norm is None:
                    # Log gradient norms to check for explosion
                    grad_norm = 0.0
                    for param in model.parameters():
                        if param.grad is not None:
                            grad_norm += param.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5
                    logger.warning(f"Gradient norm: {grad_norm:.6f} - Consider enabling gradient clipping")
            
            # MEMORY OPTIMIZATION: Clear memory for inputs and outputs
            del inputs, outputs, loss
            
            # MEMORY OPTIMIZATION: Clear CUDA cache less frequently
            if memory_efficient and device.type == 'cuda' and (i + 1) % 100 == 0:  # Increased from 20
                torch.cuda.empty_cache()
        
        # Update processed batches count
        processed_batches = end_batch
        
        # MEMORY OPTIMIZATION: Check memory less frequently
        if memory_efficient and processed_batches < total_batches:
            memory_stats = check_memory_usage()
            if memory_stats['system_percent'] > 95:  # Increased from 80
                logger.warning(f"Very high memory usage after processing {processed_batches}/{total_batches} batches "
                              f"({memory_stats['system_percent']}%). Clearing memory.")
                clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))
    
    if show_progress:
        progress.finish()
    
    # Calculate and log performance statistics
    avg_batch_time = sum(batch_times) / len(batch_times)
    avg_forward_time = sum(forward_times) / len(forward_times)
    avg_backward_time = sum(backward_times) / len(backward_times)
    
    logger.debug(
        f"Epoch {epoch+1} performance - "
        f"Avg batch: {avg_batch_time:.3f}s, "
        f"Avg forward: {avg_forward_time:.3f}s, "
        f"Avg backward: {avg_backward_time:.3f}s"
    )
    
    # MEMORY OPTIMIZATION: Final memory cleanup at end of epoch
    if memory_efficient:
        clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))
    
    return running_loss / len(train_loader)

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = True,
    memory_efficient: bool = True  # Enable memory efficiency by default
) -> float:
    """
    Validate model on validation set with improved memory management.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on (CPU or GPU)
        show_progress: Whether to show progress bar
        memory_efficient: Whether to use memory-efficient mode
        
    Returns:
        Average validation loss
    """
    model.eval()
    running_loss = 0.0
    total_batches = len(val_loader)
    
    if show_progress:
        progress = EnhancedProgressBar(
            total=total_batches, 
            prefix="Validation", 
            suffix="Complete",
            stage_info="Validation"
        )
    
    # MEMORY OPTIMIZATION: Process in chunks if dataset is large
    processed_batches = 0
    chunk_size = 50 if memory_efficient else total_batches  # Process 50 batches at a time in memory-efficient mode
    
    with torch.no_grad():
        while processed_batches < total_batches:
            end_batch = min(processed_batches + chunk_size, total_batches)
            
            # Process a chunk of batches
            for i, (inputs, targets) in enumerate(val_loader):
                if i < processed_batches:
                    continue
                if i >= end_batch:
                    break
                
                # Move data to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item()
                
                if show_progress:
                    progress.update(i + 1)
                
                # MEMORY OPTIMIZATION: Clear memory for inputs and outputs
                del inputs, outputs, loss
                
                # MEMORY OPTIMIZATION: Clear CUDA cache periodically
                if memory_efficient and device.type == 'cuda' and (i + 1) % 20 == 0:
                    torch.cuda.empty_cache()
            
            # Update processed batches count
            processed_batches = end_batch
            
            # MEMORY OPTIMIZATION: Check memory and clear if needed between chunks
            if memory_efficient and processed_batches < total_batches:
                memory_stats = check_memory_usage()
                if memory_stats['system_percent'] > 80:  # If system memory usage > 80%
                    logger.warning(f"High memory usage during validation "
                                  f"({memory_stats['system_percent']}%). Clearing memory.")
                    clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))
    
    if show_progress:
        progress.finish()
    
    # MEMORY OPTIMIZATION: Final memory cleanup at end of validation
    if memory_efficient:
        clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))
    
    return running_loss / len(val_loader)

def create_domain_batches(domain_indices: List[int], domains_per_batch: int) -> List[List[int]]:
    """
    Create batches of domain indices with a specified number of domains per batch.
    
    Args:
        domain_indices: List of domain indices to split into batches
        domains_per_batch: Number of domains per batch
        
    Returns:
        List of domain index batches
    """
    # Create batches of domain indices
    domain_batches = [
        domain_indices[i:i+domains_per_batch] 
        for i in range(0, len(domain_indices), domains_per_batch)
    ]
    
    return domain_batches



def train_model(config: Dict[str, Any]) -> Tuple[str, Dict[str, List[float]]]:
    """
    Train an RMSF prediction model with domain streaming and improved memory management.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (model_path, training_history)
    """
    # Start the overall training stage
    with log_stage("TRAINING", f"Training {config['model']['architecture']} model"):
        # Log section header
        logger = get_logger(__name__)
        log_section_header(logger, "MODEL TRAINING WITH DOMAIN STREAMING")
        
        # Check if we're resuming from a checkpoint
        resume_checkpoint = config.get("training", {}).get("resume_checkpoint", None)
        start_epoch = 0
        processed_domains_history = set()
        
        # MEMORY OPTIMIZATION: Check and log initial memory usage
        memory_stats = check_memory_usage()
        logger.info(f"Initial memory: System: {memory_stats['system_percent']}% used, "
                   f"Process: {memory_stats['process_rss_gb']:.2f} GB")
        
        # MEMORY OPTIMIZATION: Emergency check - if already at critical levels, try to recover
        if memory_stats['system_percent'] > MEMORY_CRITICAL_THRESHOLD * 100:
            logger.warning(f"Starting with critically high memory usage: {memory_stats['system_percent']}%")
            logger.warning("Attempting emergency memory reduction before training")
            emergency_memory_reduction()
            memory_stats = check_memory_usage()
            if memory_stats['system_percent'] > MEMORY_EMERGENCY_THRESHOLD * 100:
                logger.error(f"Cannot start training with memory usage at {memory_stats['system_percent']}%")
                logger.error("Please free up system memory before starting training")
                raise MemoryError("Insufficient memory to begin training")
        
        # Get device (CPU or GPU)
        device = get_device(config["system_utilization"]["adjust_for_gpu"])
        logger.info(f"Using device: {device}")
        
        # Setup mixed precision training if on GPU
        scaler = None
        if device.type == 'cuda' and config["training"].get("mixed_precision", {}).get("enabled", False):
            mixed_precision_dtype = config["training"]["mixed_precision"].get("dtype", "float16")
            if mixed_precision_dtype == "bfloat16" and torch.cuda.is_available() and hasattr(torch, 'bfloat16'):
                amp_dtype = torch.bfloat16
                logger.info("Using mixed precision training with bfloat16")
            else:
                amp_dtype = torch.float16
                logger.info("Using mixed precision training with float16")
            
            scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled with GradScaler")
        
        # Check for domain registry if it exists
        domain_registry_path = os.path.join(config["output"]["base_dir"], "domain_registry.json")
        
        # Load or create domain registry for tracking processing status
        if os.path.exists(domain_registry_path):
            try:
                domain_registry = load_json(domain_registry_path)
                logger.info(f"Loaded domain registry with {len(domain_registry)} entries")
            except Exception as e:
                logger.warning(f"Error loading domain registry: {str(e)}")
                domain_registry = {}
        else:
            domain_registry = {}
        
        # ==== DOMAIN DISCOVERY PHASE ====
        logger.info("Starting domain discovery phase")
        
        all_domains = []
        
        try:
            # Open the HDF5 file to get all available domains
            with h5py.File(config["input"]["voxel_file"], 'r') as f:
                all_available_domains = list(f.keys())
                logger.info(f"Found {len(all_available_domains)} total domains in HDF5 file")
                
                # Check if we have a list of specific domains to use
                if config["input"]["domain_ids"] and len(config["input"]["domain_ids"]) > 0:
                    # Filter domains based on specified IDs
                    filtered_domains = []
                    for domain in all_available_domains:
                        base_domain = domain.split('_')[0]
                        if base_domain in config["input"]["domain_ids"]:
                            filtered_domains.append(domain)
                    
                    if not filtered_domains:
                        logger.warning("None of the specified domain_ids were found in the voxel file")
                        logger.warning("Using all available domains instead")
                        all_domains = all_available_domains
                    else:
                        all_domains = filtered_domains
                        logger.info(f"Filtered to {len(all_domains)} domains based on specified domain_ids")
                else:
                    # Use all domains
                    all_domains = all_available_domains
                
                # Apply max_domains limit if specified in config
                if config["input"].get("max_domains") is not None and len(all_domains) > config["input"]["max_domains"]:
                    logger.info(f"Limiting to {config['input']['max_domains']} domains (out of {len(all_domains)} available)")
                    all_domains = all_domains[:config["input"]["max_domains"]]
                    
        except Exception as e:
            logger.error(f"Error during domain discovery: {str(e)}")
            raise
        
        # Update domain registry with all discovered domains
        for domain_id in all_domains:
            if domain_id not in domain_registry:
                domain_registry[domain_id] = {
                    "processed": False,
                    "processing_attempts": 0,
                    "last_processed": None,
                    "error_count": 0
                }
        
        # Prioritize unprocessed domains
        unprocessed_domains = [d for d in all_domains if not domain_registry[d].get("processed", False)]
        processed_domains = [d for d in all_domains if domain_registry[d].get("processed", False)]
        
        if unprocessed_domains:
            logger.info(f"Prioritizing {len(unprocessed_domains)} unprocessed domains")
            
            # Move unprocessed domains to the front but keep some processed ones for continuity
            # Include some processed domains to maintain training continuity
            num_processed_to_include = min(len(processed_domains), len(unprocessed_domains) // 4)
            if num_processed_to_include > 0:
                # Include some processed domains that performed well previously
                processed_domains_sample = processed_domains[:num_processed_to_include]
                all_domains = unprocessed_domains + processed_domains_sample + processed_domains[num_processed_to_include:]
                logger.info(f"Including {num_processed_to_include} previously processed domains for training continuity")
            else:
                all_domains = unprocessed_domains + processed_domains
        
        # Save the full domain list for reference
        full_domain_registry_path = os.path.join(config["output"]["base_dir"], "full_domain_registry.txt")
        save_domain_registry(all_domains, full_domain_registry_path)
        logger.info(f"Saved full domain registry with {len(all_domains)} domains")
        
        # ==== LOAD RMSF DATA ====
        logger.info("Loading RMSF data")
        rmsf_data = load_rmsf_data(
            config["input"]["rmsf_dir"],
            config["input"].get("replica", "replica_average"),
            config["input"]["temperature"]
        )
        
        # MEMORY OPTIMIZATION: Check memory after loading RMSF data
        memory_stats = check_memory_usage()
        logger.info(f"Memory after loading RMSF data: {memory_stats['system_percent']}% used")
        
        # ==== CREATE MODEL ====
        with log_stage("MODEL_CREATION", f"Creating {config['model']['architecture']} model"):
            # Determine input shape by loading a single domain temporarily
            logger.info("Sampling a domain to determine input shape")
            input_shape = None
            
            # Try multiple domains in case some are invalid
            for test_domain in all_domains[:min(10, len(all_domains))]:
                try:
                    with h5py.File(config["input"]["voxel_file"], 'r') as f:
                        domain_group = f[test_domain]
                        first_child_key = list(domain_group.keys())[0]
                        residue_group = domain_group[first_child_key]
                        residue_keys = [k for k in residue_group.keys() if isinstance(k, str) and k.isdigit()]
                        
                        if not residue_keys:
                            logger.warning(f"No residues found in sample domain {test_domain}")
                            continue
                        
                        first_residue = residue_keys[0]
                        residue_data = residue_group[first_residue]
                        voxel_data_raw = residue_data[:]
                        
                        # Transpose if necessary
                        if len(residue_data.shape) == 4 and residue_data.shape[3] in [4, 5]:
                            voxel = np.transpose(voxel_data_raw, (3, 0, 1, 2))
                        else:
                            voxel = voxel_data_raw
                        
                        input_shape = voxel.shape
                        logger.info(f"Determined input shape: {input_shape}")
                        break
                except Exception as e:
                    logger.warning(f"Error sampling domain {test_domain}: {str(e)}")
            
            if input_shape is None:
                raise ValueError("Could not determine input shape from any sampled domain")
            
            # MEMORY OPTIMIZATION: Clear memory after sampling
            clear_memory(force_gc=True, clear_cuda=True)
            
            # Check if resuming from checkpoint
            if resume_checkpoint and os.path.exists(resume_checkpoint):
                logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
                checkpoint = torch.load(resume_checkpoint, map_location=device)
                
                # Extract model architecture and config
                model_config = checkpoint.get('config', {}).get('model', config['model'])
                
                # Create model with same architecture
                model = get_model(
                    architecture=model_config['architecture'],
                    input_channels=model_config['input_channels'],
                    channel_growth_rate=model_config['channel_growth_rate'],
                    num_residual_blocks=model_config['num_residual_blocks'],
                    dropout_rate=model_config['dropout_rate'],
                    base_filters=model_config['base_filters']
                )
                
                # Load model weights
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Move model to device
                model = model.to(device)
                
                # Get training history and processed domains
                train_losses = checkpoint.get('train_losses', [])
                val_losses = checkpoint.get('val_losses', [])
                processed_domains_history = set(checkpoint.get('processed_domains', []))
                start_epoch = checkpoint.get('epoch', 0)
                
                logger.info(f"Resumed from epoch {start_epoch} with {len(processed_domains_history)} domains processed")
                
                # Clean up checkpoint to free memory
                del checkpoint
                clear_memory(force_gc=True, clear_cuda=True)
            else:
                # Create new model
                logger.info("Building model architecture")
                model = get_model(
                    architecture=config["model"]["architecture"],
                    input_channels=input_shape[0],
                    channel_growth_rate=config["model"]["channel_growth_rate"],
                    num_residual_blocks=config["model"]["num_residual_blocks"],
                    dropout_rate=config["model"]["dropout_rate"],
                    base_filters=config["model"]["base_filters"]
                )
                
                # Move model to device
                model = model.to(device)
                
                # Initialize training history
                train_losses = []
                val_losses = []
            
            # Log model summary (params count)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
            
            # Define loss function and optimizer
            criterion = nn.MSELoss()
            
            # Reduced learning rate to prevent divergence
            initial_lr = float(config["training"]["learning_rate"])
            
            # Check for warmup settings
            use_warmup = config["training"].get("warmup", {}).get("enabled", False)
            if use_warmup and start_epoch < config["training"]["warmup"].get("epochs", 1):
                warmup_epochs = config["training"]["warmup"].get("epochs", 1)
                # Start with a lower learning rate if using warmup
                effective_lr = initial_lr * 0.1
                logger.info(f"Using learning rate warmup over {warmup_epochs} epochs: starting at {effective_lr:.6f}")
            else:
                effective_lr = initial_lr
            
            # Create optimizer (or load from checkpoint)
            if resume_checkpoint and 'optimizer_state_dict' in locals().get('checkpoint', {}):
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=effective_lr,
                    weight_decay=float(config["training"]["weight_decay"])
                )
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Loaded optimizer state from checkpoint")
            else:
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=effective_lr,
                    weight_decay=float(config["training"]["weight_decay"])
                )
            
            # Create scheduler if specified
            scheduler = None
            if "scheduler" in config.get("training", {}):
                scheduler_config = config["training"]["scheduler"]
                if scheduler_config["type"] == "reduce_on_plateau":
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode=scheduler_config.get("mode", "min"),
                        factor=float(scheduler_config.get("factor", 0.5)),  # Changed from 0.1 to 0.5
                        patience=int(scheduler_config.get("patience", 10)),
                        verbose=True
                    )
                    logger.info(f"Using ReduceLROnPlateau scheduler with patience {scheduler_config.get('patience', 10)}")
                elif scheduler_config["type"] == "step":
                    scheduler = optim.lr_scheduler.StepLR(
                        optimizer,
                        step_size=int(scheduler_config.get("step_size", 30)),
                        gamma=float(scheduler_config.get("gamma", 0.5))  # Changed from 0.1 to 0.5
                    )
                    logger.info(f"Using StepLR scheduler with step size {scheduler_config.get('step_size', 30)}")
                elif scheduler_config["type"] == "cosine_annealing":
                    T_max = int(scheduler_config.get("T_max", config["training"]["num_epochs"]))
                    # Explicitly convert eta_min to float to avoid type errors
                    eta_min = float(scheduler_config.get("eta_min", 1e-6))
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=T_max,
                        eta_min=eta_min
                    )
                    logger.info(f"Using CosineAnnealingLR scheduler with T_max={T_max}, eta_min={eta_min}")
                    
            # Gradient clipping configuration
            gradient_clip_norm = None
            if config["training"].get("gradient_clipping", {}).get("enabled", False):
                gradient_clip_norm = config["training"]["gradient_clipping"].get("max_norm", 1.0)
                logger.info(f"Gradient clipping enabled with max_norm={gradient_clip_norm}")
                
            # MEMORY OPTIMIZATION: Check memory after model creation
            memory_stats = check_memory_usage()
            logger.info(f"Memory after model creation: {memory_stats['system_percent']}% used")
            
            # Clear memory before starting training
            clear_memory(force_gc=True, clear_cuda=True)
        
        # ==== DOMAIN STREAMING TRAINING ====
        with log_stage("TRAINING", f"Training for {config['training']['num_epochs']} epochs with domain streaming"):
            num_epochs = int(config["training"]["num_epochs"])
            show_progress = config["logging"]["show_progress_bars"]
            
            # Record losses
            epoch_domain_counts = []
            
            # Track best model
            if resume_checkpoint and 'best_val_loss' in locals().get('checkpoint', {}):
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                best_epoch = checkpoint.get('best_epoch', 0)
                logger.info(f"Resuming with best validation loss: {best_val_loss:.6f} from epoch {best_epoch}")
            else:
                best_val_loss = float('inf')
                best_epoch = 0
            
            # MEMORY OPTIMIZATION: Calculate domain batch size more conservatively
            memory_stats = check_memory_usage()
            available_memory_gb = memory_stats["system_available_gb"]
            system_memory_total = memory_stats["system_total_gb"]
            system_memory_percent = memory_stats["system_percent"]

            # MAJOR CHANGE: Much more conservative domain batch sizing based on memory pressure
            if system_memory_percent > 60:  # Higher memory pressure
                domains_per_batch = max(10, config["training"]["domain_streaming"].get("initial_domains_per_batch", 100) // 4)
                logger.warning(f"High initial memory usage ({system_memory_percent:.1f}%), using reduced batch size of {domains_per_batch} domains")
            elif available_memory_gb > 30:  # Plenty of memory
                domains_per_batch = max(50, config["training"]["domain_streaming"].get("initial_domains_per_batch", 100) // 2)
                logger.info(f"Good memory availability ({available_memory_gb:.1f} GB free), using {domains_per_batch} domains per batch")
            else:  # Limited memory
                domains_per_batch = max(20, config["training"]["domain_streaming"].get("initial_domains_per_batch", 100) // 3)
                logger.info(f"Limited memory availability ({available_memory_gb:.1f} GB free), using reduced batch size of {domains_per_batch} domains")

            logger.info(f"Using streaming approach with {domains_per_batch} domains per batch")
            logger.info(f"Total domains to process: {len(all_domains)}")

            # Prepare domain batches
            np.random.seed(config.get("training", {}).get("seed", 42))
            domain_indices = np.arange(len(all_domains))
            np.random.shuffle(domain_indices)

            # Split into training, validation and test sets by domains
            train_split = config["training"]["train_split"]
            val_split = config["training"]["val_split"]

            train_idx = int(len(domain_indices) * train_split)
            val_idx = train_idx + int(len(domain_indices) * val_split)

            train_domain_indices = domain_indices[:train_idx]
            val_domain_indices = domain_indices[train_idx:val_idx]
            test_domain_indices = domain_indices[val_idx:]

            logger.info(f"Split domains: {len(train_domain_indices)} training, "
                    f"{len(val_domain_indices)} validation, {len(test_domain_indices)} test")

            # Create domain batches with the updated size
            train_domain_batches = create_domain_batches(train_domain_indices, domains_per_batch)
            val_domain_batches = create_domain_batches(val_domain_indices, domains_per_batch)

            logger.info(f"Created {len(train_domain_batches)} training batches, {len(val_domain_batches)} validation batches")

            # Create global RMSF lookup to reuse across all batches
            from voxelflex.data.data_loader import create_optimized_rmsf_lookup
            global_rmsf_lookup = create_optimized_rmsf_lookup(rmsf_data)
            
            # Create checkpoint directory
            checkpoint_dir = os.path.join(config["output"]["base_dir"], "checkpoints")
            ensure_dir(checkpoint_dir)
            
            # Train the model across epochs
            start_time = time.time()
            total_domains_processed = len(processed_domains_history)
            
            # Track domains processed in this run
            domains_processed_this_run = set(processed_domains_history)
            
            # Create model save directory
            model_dir = os.path.join(config["output"]["base_dir"], "models")
            ensure_dir(model_dir)
            
            # Main epoch loop
            for epoch in range(start_epoch, num_epochs):
                epoch_start_time = time.time()
                
                # Apply learning rate warmup if configured
                if use_warmup and epoch < warmup_epochs:
                    warmup_factor = (epoch + 1) / warmup_epochs
                    adjusted_lr = initial_lr * warmup_factor
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = adjusted_lr
                    logger.info(f"Warmup epoch {epoch+1}/{warmup_epochs}: LR = {adjusted_lr:.6f}")
                
                # Shuffle domain batches at the start of each epoch
                np.random.shuffle(train_domain_batches)
                
                # Track epoch statistics
                epoch_train_loss = 0.0
                epoch_domains_processed = 0
                epoch_batches_processed = 0
                
                # Training phase
                model.train()
                
                # Process each domain batch
                batch_progress = EnhancedProgressBar(
                    total=len(train_domain_batches),
                    prefix=f"Epoch {epoch+1}/{num_epochs} Batches",
                    suffix="Complete",
                    stage_info="DOMAIN_BATCH"
                )
                
                # Keep track of domains processed in this epoch
                domains_processed_this_epoch = set()
                
                for batch_idx, domain_batch_indices in enumerate(train_domain_batches):
                    batch_start_time = time.time()
                    
                    # MEMORY OPTIMIZATION: Check memory before loading new batch
                    memory_stats = check_memory_usage()
                    if memory_stats['system_percent'] > MEMORY_CRITICAL_THRESHOLD * 100:
                        logger.warning(f"Critical memory usage ({memory_stats['system_percent']:.1f}%) before batch {batch_idx+1}")
                        logger.warning("Performing emergency memory reduction")
                        freed_memory = emergency_memory_reduction()
                        
                        # If not enough memory was freed, skip this batch
                        memory_stats = check_memory_usage()
                        if memory_stats['system_percent'] > MEMORY_EMERGENCY_THRESHOLD * 100:
                            logger.error(f"Memory usage still critical ({memory_stats['system_percent']:.1f}%) after emergency reduction")
                            logger.error(f"Skipping batch {batch_idx+1} to prevent system crash")
                            batch_progress.update(batch_idx + 1)
                            continue
                    
                    # Load domain batch
                    batch_voxel_data = load_domain_batch(domain_batch_indices, all_domains, config)
                    
                    # Count domains successfully loaded
                    domains_in_batch = len(batch_voxel_data)
                    
                    # Update domain processing tracking
                    for domain_id in batch_voxel_data.keys():
                        domains_processed_this_epoch.add(domain_id)
                        domains_processed_this_run.add(domain_id)
                    
                    if domains_in_batch == 0:
                        logger.warning(f"No valid domains loaded in batch {batch_idx+1}. Skipping.")
                        batch_progress.update(batch_idx + 1)
                        continue
                    
                    logger.info(f"Processing {domains_in_batch} domains in batch {batch_idx+1}/{len(train_domain_batches)}")
                    
                    # MEMORY OPTIMIZATION: Check memory after loading batch
                    memory_stats = check_memory_usage()
                    if memory_stats['system_percent'] > MEMORY_CRITICAL_THRESHOLD * 100:
                        logger.warning(f"Critical memory usage ({memory_stats['system_percent']:.1f}%) after loading batch {batch_idx+1}")
                        logger.warning("Performing emergency memory reduction")
                        freed_memory = emergency_memory_reduction()
                        
                        # If still critical after reduction, we need to skip this batch
                        memory_stats = check_memory_usage()
                        if memory_stats['system_percent'] > MEMORY_EMERGENCY_THRESHOLD * 100:
                            logger.error(f"Memory usage still critical ({memory_stats['system_percent']:.1f}%) after emergency reduction")
                            logger.error(f"Skipping batch {batch_idx+1} to prevent system crash")
                            
                            # Free the batch data
                            del batch_voxel_data
                            clear_memory(force_gc=True, clear_cuda=True, aggressive=True)
                            batch_progress.update(batch_idx + 1)
                            continue
                    
                    # Create dataloaders for this batch
                    try:
                        # Create domain mapping for this batch
                        voxel_domains = list(batch_voxel_data.keys())
                        rmsf_domains = rmsf_data['domain_id'].unique().tolist()
                        domain_mapping = create_domain_mapping(voxel_domains, rmsf_domains)

                        # MEMORY OPTIMIZATION: Calculate optimal batch size for GPU more conservatively
                        data_batch_size = config["training"]["batch_size"]  # e.g., 512 from config
                        if device.type == 'cuda':
                            # Calculate dynamically based on available GPU memory
                            gpu_id = 0
                            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
                            allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                            free_memory = total_memory - allocated

                            # MAJOR CHANGE: More conservative memory allocation
                            element_size = 4  # float32 = 4 bytes
                            input_bytes = np.prod(input_shape) * element_size
                            # Increased overhead factor from 2.0 to 3.0 for more safety
                            max_elements = (free_memory * 0.7 * (1024**3)) / (input_bytes * 3.0)
                            # Reduce max batch size for safety
                            max_batch = min(512, int(max_elements))
                            
                            # Use the smaller of configured batch size or calculated max
                            data_batch_size = min(config["training"]["batch_size"], max_batch)
                            logger.info(f"Using GPU-optimized batch size: {data_batch_size}")
                            
                        # Create dataset for this batch
                        batch_dataset = RMSFDataset(
                            batch_voxel_data,
                            rmsf_data,
                            domain_mapping,
                            memory_efficient=config["training"].get("memory_efficient", True),  # Use from config
                            global_rmsf_lookup=global_rmsf_lookup  # Use the pre-computed lookup
                        )

                        # MEMORY OPTIMIZATION: Determine optimal worker count more conservatively
                        memory_stats = check_memory_usage()
                        cpu_count = os.cpu_count() or 8
                        
                        if config["training"].get("safe_mode", False):
                            # Safe mode forces single-threaded operation
                            num_workers = 0
                            logger.info("Safe mode enabled. Using 0 workers (single-threaded mode).")
                        elif memory_stats['system_percent'] > 80:
                            # Very high memory pressure - use minimal workers
                            num_workers = 0
                            logger.warning(f"High memory pressure ({memory_stats['system_percent']:.1f}%). Using 0 workers.")
                        elif memory_stats['system_percent'] > 70:
                            # High memory pressure - use minimal workers
                            num_workers = 1
                            logger.warning(f"High memory pressure ({memory_stats['system_percent']:.1f}%). Using 1 worker.")
                        elif memory_stats['system_percent'] > 60:
                            # Moderate memory pressure
                            num_workers = min(2, cpu_count // 4)
                            logger.info(f"Moderate memory pressure ({memory_stats['system_percent']:.1f}%). Using {num_workers} workers.")
                        else:
                            # Normal memory pressure
                            num_workers = min(4, cpu_count // 2)
                            logger.info(f"Normal memory usage ({memory_stats['system_percent']:.1f}%). Using {num_workers} workers.")

                        # Create data loader
                        train_loader = DataLoader(
                            batch_dataset,
                            batch_size=data_batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=(device.type == 'cuda'),
                            persistent_workers=(num_workers > 0),
                            prefetch_factor=2 if num_workers > 0 else None  # CHANGE: Reduced from 4 to 2
                        )

                        logger.info(f"Created dataloader with batch size {data_batch_size}, {num_workers} workers")
                        logger.info(f"Dataloader contains {len(train_loader)} batches from {len(batch_dataset)} samples")

                        # Train on this domain batch
                        batch_train_loss = train_epoch(
                            model=model,
                            train_loader=train_loader,
                            criterion=criterion,
                            optimizer=optimizer,
                            device=device,
                            epoch=epoch,
                            show_progress=show_progress and len(train_loader) > 10,
                            memory_efficient=config["training"].get("memory_efficient", True),  # Use from config
                            scaler=scaler,
                            gradient_clip_norm=gradient_clip_norm
                        )

                        # Add to epoch statistics
                        epoch_train_loss += batch_train_loss * len(train_loader)
                        epoch_batches_processed += len(train_loader)
                        epoch_domains_processed += domains_in_batch

                        logger.info(f"Batch {batch_idx+1} completed: {domains_in_batch} domains, loss: {batch_train_loss:.6f}")

                    except Exception as e:
                        logger.error(f"Error processing domain batch {batch_idx+1}: {str(e)}", exc_info=True)

                        # Update domain registry with error information
                        for domain_id in [all_domains[i] for i in domain_batch_indices]:
                            if domain_id in domain_registry:
                                domain_registry[domain_id]["error_count"] = domain_registry[domain_id].get("error_count", 0) + 1
                                domain_registry[domain_id]["last_error"] = str(e)
                        
                        # MEMORY OPTIMIZATION: In case of error, perform aggressive cleanup
                        clear_memory(force_gc=True, clear_cuda=True, aggressive=True)
                    
                    # Clean up batch data to free memory
                    del batch_voxel_data
                    if 'train_loader' in locals():
                        del train_loader
                    if 'batch_dataset' in locals():
                        del batch_dataset
                    if 'domain_mapping' in locals():
                        del domain_mapping
                        
                    clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))
                    
                    # Update batch progress
                    batch_duration = time.time() - batch_start_time
                    logger.info(f"Batch {batch_idx+1} processing time: {batch_duration:.2f}s")
                    batch_progress.update(batch_idx + 1)
                    
                    # MEMORY OPTIMIZATION: Check memory periodically during training
                    if (batch_idx + 1) % 5 == 0 or batch_idx == len(train_domain_batches) - 1:
                        memory_stats = check_memory_usage()
                        logger.info(f"Memory after batch {batch_idx+1}: {memory_stats['system_percent']:.1f}% used")
                        
                        # If memory is getting high, perform cleanup
                        if memory_stats['system_percent'] > MEMORY_CRITICAL_THRESHOLD * 100:
                            logger.warning(f"Critical memory usage ({memory_stats['system_percent']:.1f}%) after batch {batch_idx+1}")
                            logger.warning("Performing emergency memory reduction")
                            emergency_memory_reduction()
                
                # End batch progress
                batch_progress.finish()
                
                # Calculate average training loss for the epoch
                avg_train_loss = epoch_train_loss / max(1, epoch_batches_processed)
                train_losses.append(avg_train_loss)
                
                # Validation phase
                model.eval()
                epoch_val_loss = 0.0
                val_steps = 0
                
                # Process validation domain batches
                val_progress = EnhancedProgressBar(
                    total=len(val_domain_batches),
                    prefix="Validation Batches",
                    suffix="Complete",
                    stage_info="VALIDATION"
                )
                
                # MEMORY OPTIMIZATION: Clear memory before validation
                clear_memory(force_gc=True, clear_cuda=True)
                
                for val_batch_idx, val_domain_batch_indices in enumerate(val_domain_batches):
                    # MEMORY OPTIMIZATION: Check memory before validation batch
                    memory_stats = check_memory_usage()
                    if memory_stats['system_percent'] > MEMORY_CRITICAL_THRESHOLD * 100:
                        logger.warning(f"Critical memory usage ({memory_stats['system_percent']:.1f}%) before validation batch {val_batch_idx+1}")
                        logger.warning("Performing emergency memory reduction")
                        emergency_memory_reduction()
                        
                        # If still critical, skip this validation batch
                        memory_stats = check_memory_usage()
                        if memory_stats['system_percent'] > MEMORY_EMERGENCY_THRESHOLD * 100:
                            logger.error(f"Memory usage still critical ({memory_stats['system_percent']:.1f}%) after emergency reduction")
                            logger.error(f"Skipping validation batch {val_batch_idx+1} to prevent system crash")
                            val_progress.update(val_batch_idx + 1)
                            continue
                    
                    # Load validation domain batch
                    val_voxel_data = load_domain_batch(val_domain_batch_indices, all_domains, config)
                    
                    # Count domains successfully loaded
                    val_domains_in_batch = len(val_voxel_data)
                    
                    if val_domains_in_batch == 0:
                        logger.warning(f"No valid domains loaded in validation batch {val_batch_idx+1}. Skipping.")
                        val_progress.update(val_batch_idx + 1)
                        continue
                    
                    # Create dataloaders for validation
                    try:
                        # Create domain mapping for this validation batch
                        val_voxel_domains = list(val_voxel_data.keys())
                        val_domain_mapping = create_domain_mapping(val_voxel_domains, rmsf_domains)
                        
                        # MEMORY OPTIMIZATION: Use smaller batch size for validation
                        val_batch_size = min(128, config["training"]["batch_size"])
                        
                        # Create dataset and loader with memory-efficient mode
                        val_dataset = RMSFDataset(
                            val_voxel_data, 
                            rmsf_data, 
                            val_domain_mapping, 
                            memory_efficient=config["training"].get("memory_efficient", True),  # Use from config
                            global_rmsf_lookup=global_rmsf_lookup
                        )
                        
                        val_loader = DataLoader(
                            val_dataset,
                            batch_size=val_batch_size,
                            shuffle=False,
                            num_workers=0 if config["training"].get("safe_mode", False) else 0,  # Use 0 workers for validation to reduce memory pressure
                            pin_memory=(device.type == 'cuda')
                        )
                        
                        # Validate
                        with torch.no_grad():
                            batch_val_steps = 0
                            batch_val_loss = 0.0
                            
                            for val_inputs, val_targets in val_loader:
                                val_inputs = val_inputs.to(device, non_blocking=True)
                                val_targets = val_targets.to(device, non_blocking=True)
                                
                                # Use mixed precision for validation if available
                                if device.type == 'cuda' and scaler is not None:
                                    with torch.cuda.amp.autocast():
                                        val_outputs = model(val_inputs)
                                        val_loss = criterion(val_outputs, val_targets)
                                else:
                                    val_outputs = model(val_inputs)
                                    val_loss = criterion(val_outputs, val_targets)
                                
                                batch_val_loss += val_loss.item()
                                batch_val_steps += 1
                                
                                # Clean up
                                del val_inputs, val_outputs, val_targets
                                
                                # MEMORY OPTIMIZATION: Clear CUDA cache periodically during validation
                                if device.type == 'cuda' and batch_val_steps % 10 == 0:
                                    torch.cuda.empty_cache()
                        
                        # Add to epoch validation loss
                        if batch_val_steps > 0:
                            epoch_val_loss += batch_val_loss
                            val_steps += batch_val_steps
                        
                    except Exception as e:
                        logger.error(f"Error in validation batch {val_batch_idx+1}: {str(e)}")
                        
                        # MEMORY OPTIMIZATION: In case of error, perform aggressive cleanup
                        clear_memory(force_gc=True, clear_cuda=True, aggressive=True)
                    
                    # Clean up
                    del val_voxel_data
                    if 'val_loader' in locals():
                        del val_loader
                    if 'val_dataset' in locals():
                        del val_dataset
                    if 'val_domain_mapping' in locals():
                        del val_domain_mapping
                        
                    clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))
                    
                    # Update progress
                    val_progress.update(val_batch_idx + 1)
                
                # End validation progress
                val_progress.finish()
                
                # Calculate average validation loss
                avg_val_loss = epoch_val_loss / max(1, val_steps)
                val_losses.append(avg_val_loss)
                
                # Update scheduler if used
                if scheduler is not None:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(avg_val_loss)
                    else:
                        scheduler.step()
                
                # Track best model
                is_best = avg_val_loss < best_val_loss
                if is_best:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1
                    
                    # Save best model checkpoint
                    best_model_path = os.path.join(model_dir, f"best_model_streaming.pt")
                    
                    # MEMORY OPTIMIZATION: Clear memory before saving model
                    clear_memory(force_gc=True, clear_cuda=True)
                    
                    # Save best model with minimal state to reduce file size
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': epoch + 1,
                        'val_loss': best_val_loss,
                        'config': config,
                        'input_shape': input_shape,
                        'processed_domains': list(domains_processed_this_run)  # Track processed domains
                    }, best_model_path)
                    
                    logger.info(f"Saved best model checkpoint (val_loss: {best_val_loss:.6f})")
                
                # MAJOR NEW FEATURE: Save regular checkpoints every 3 epochs
                if (epoch + 1) % 3 == 0 or epoch == num_epochs - 1:
                    # Create checkpoint filename
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
                    
                    # MEMORY OPTIMIZATION: Clear memory before saving checkpoint
                    clear_memory(force_gc=True, clear_cuda=True)
                    
                    logger.info(f"Saving checkpoint for epoch {epoch+1}")
                    
                    # Save checkpoint with all necessary information
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler': scaler.state_dict() if scaler else None,
                        'epoch': epoch + 1,
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'best_val_loss': best_val_loss,
                        'best_epoch': best_epoch,
                        'config': config,
                        'input_shape': input_shape,
                        'processed_domains': list(domains_processed_this_run)
                    }, checkpoint_path)
                    
                    logger.info(f"Checkpoint saved to {checkpoint_path}")
                    
                    # MEMORY OPTIMIZATION: Clean up old checkpoints to save disk space
                    # Keep only the 3 most recent checkpoints
                    try:
                        checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                                          if f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
                        checkpoint_files.sort(key=lambda x: int(x.split("_")[2].split(".")[0]), reverse=True)
                        
                        # Keep the 3 most recent checkpoints
                        for old_checkpoint in checkpoint_files[3:]:
                            old_path = os.path.join(checkpoint_dir, old_checkpoint)
                            os.remove(old_path)
                            logger.info(f"Removed old checkpoint: {old_path}")
                    except Exception as e:
                        logger.warning(f"Error cleaning up old checkpoints: {str(e)}")
                
                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time
                
                # Log progress with detailed metrics
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s - "
                    f"Domains: {epoch_domains_processed} - "
                    f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
                    f"{' (Best)' if is_best else ''}"
                )
                
                # Check for training divergence
                if avg_train_loss > avg_val_loss * 5:  # Training loss much higher than validation
                    logger.warning(f"Potential training divergence detected: train_loss={avg_train_loss:.6f}, "
                                  f"val_loss={avg_val_loss:.6f}")
                    if gradient_clip_norm is None:
                        logger.warning("Consider enabling gradient clipping to address divergence")
                
                # Track domains processed in this epoch
                epoch_domain_counts.append(epoch_domains_processed)
                total_domains_processed += epoch_domains_processed
                
                # Update domain registry with successfully processed domains
                for domain_id in domains_processed_this_epoch:
                    if domain_id in domain_registry:
                        domain_registry[domain_id]["processed"] = True
                        domain_registry[domain_id]["processing_attempts"] = domain_registry[domain_id].get("processing_attempts", 0) + 1
                        domain_registry[domain_id]["last_processed"] = time.strftime("%Y%m%d_%H%M%S")
                
                # Save updated domain registry
                save_json(domain_registry, domain_registry_path)
                logger.info(f"Updated domain registry with {len(domains_processed_this_epoch)} newly processed domains")
                
                # MEMORY OPTIMIZATION: Check memory at end of epoch
                memory_stats = check_memory_usage()
                logger.info(f"Memory at end of epoch {epoch+1}: {memory_stats['system_percent']:.1f}% used")
                
                # MEMORY OPTIMIZATION: Clear memory at end of epoch
                clear_memory(force_gc=True, clear_cuda=True)
                
                # MEMORY OPTIMIZATION: Dynamically adjust domains_per_batch based on memory usage
                memory_stats = check_memory_usage()
                
                # If memory usage is too high, reduce batch size
                if memory_stats['system_percent'] > 75:
                    # Reduce domains per batch by 30%
                    old_domains_per_batch = domains_per_batch
                    domains_per_batch = max(10, int(domains_per_batch * 0.7))
                    
                    # Recalculate batches
                    train_domain_batches = create_domain_batches(train_domain_indices, domains_per_batch)
                    val_domain_batches = create_domain_batches(val_domain_indices, domains_per_batch)
                    
                    logger.warning(f"Reduced domains per batch from {old_domains_per_batch} to {domains_per_batch} due to high memory usage")
                    logger.info(f"New batch counts: {len(train_domain_batches)} training, {len(val_domain_batches)} validation")
                # If memory usage is very low and we've processed all domains, try to increase batch size slightly
                elif memory_stats['system_percent'] < 50 and epoch_domains_processed >= len(train_domain_indices):
                    # Increase domains per batch by 10% (more conservative than before)
                    old_domains_per_batch = domains_per_batch
                    domains_per_batch = min(len(train_domain_indices), int(domains_per_batch * 1.1))
                    
                    # Only adjust if there's a meaningful change
                    if domains_per_batch > old_domains_per_batch:
                        # Recalculate batches with new size
                        train_domain_batches = create_domain_batches(train_domain_indices, domains_per_batch)
                        val_domain_batches = create_domain_batches(val_domain_indices, domains_per_batch)
                        
                        logger.info(f"Increased domains per batch from {old_domains_per_batch} to {domains_per_batch} due to low memory usage")
                        logger.info(f"New batch counts: {len(train_domain_batches)} training, {len(val_domain_batches)} validation")
            
            # Training summary
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
            logger.info(f"Total domains processed across all epochs: {total_domains_processed}")
            logger.info(f"Unique domains processed in this run: {len(domains_processed_this_run)}")
            
            # Calculate and report domain coverage
            total_domains = len(all_domains)
            coverage = (len(domains_processed_this_run) / total_domains) * 100
            logger.info(f"Domain coverage: {coverage:.1f}% ({len(domains_processed_this_run)}/{total_domains})")
            
            # Calculate and report total processed domains
            total_processed = len([d for d in domain_registry if domain_registry[d].get("processed", False)])
            total_coverage = (total_processed / total_domains) * 100
            logger.info(f"Total domain coverage (all runs): {total_coverage:.1f}% ({total_processed}/{total_domains})")
            
            # Final memory usage
            log_memory_usage(logger)
        
        # Save the final model
        with log_stage("MODEL_SAVING", "Saving trained model"):
            # MEMORY OPTIMIZATION: Clear memory before final save
            clear_memory(force_gc=True, clear_cuda=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(model_dir, f"{config['model']['architecture']}_streaming_{timestamp}.pt")
            
            logger.info(f"Saving final model to {model_path}")
            
            # Save model with all relevant information
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'input_shape': input_shape,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epoch_domain_counts': epoch_domain_counts,
                'epoch': num_epochs,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'processed_domains': list(domains_processed_this_run)  # Save domains processed in this run
            }, model_path)
            
            logger.info(f"Final model saved to {model_path}")
            
            # Save training history
            history = {
                'train_loss': train_losses,
                'val_loss': val_losses,
                'best_epoch': best_epoch,
                'best_val_loss': best_val_loss,
                'epoch_domain_counts': epoch_domain_counts,
                'domains_processed': list(domains_processed_this_run),
                'total_domains': len(all_domains),
                'domain_coverage_percent': coverage
            }
            
            history_path = os.path.join(model_dir, f"training_history_streaming_{timestamp}.json")
            save_json(history, history_path)
            
            logger.info(f"Training history saved to {history_path}")
        
        return model_path, history