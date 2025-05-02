"""
Evaluation command for VoxelFlex (Temperature-Aware).

Calculates performance metrics by comparing predictions against ground truth
from the aggregated RMSF data. Includes stratification and permutation importance.
"""

import os
import time
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable, DefaultDict
from collections import defaultdict
import gc 

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler

# Use centralized logger
logger = logging.getLogger("voxelflex.cli.evaluate")

# Project imports
from voxelflex.utils.logging_utils import (
    get_logger, log_stage, log_memory_usage, log_section_header, EnhancedProgressBar
)
from voxelflex.utils.file_utils import ensure_dir, save_json, load_json, resolve_path
from voxelflex.utils.system_utils import clear_memory, check_memory_usage, get_device
from voxelflex.models.cnn_models import get_model
from voxelflex.utils.temp_scaling import get_temperature_scaler
from voxelflex.data.data_loader import (
    PredictionDataset,
    load_process_domain_from_handle,  # Use robust HDF5 loader for permutation importance
    load_aggregated_rmsf_data,
    create_domain_mapping  # Needed for resname lookup
)

# --- Metric Calculation Helpers ---

def safe_metric(metric_func: Callable, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """
    Safely compute metric, handling NaNs, Infs, and insufficient data.
    
    Args:
        metric_func: Metric function to call (e.g., pearsonr, r2_score)
        y_true: Ground truth values
        y_pred: Predicted values
        **kwargs: Additional arguments for metric function
        
    Returns:
        Computed metric value or NaN if calculation fails
    """
    try:
        # Ensure inputs are numpy arrays and float64 for precision
        x_np = np.asarray(y_true).astype(np.float64)
        y_np = np.asarray(y_pred).astype(np.float64)

        # Filter out non-finite values
        valid_mask = np.isfinite(x_np) & np.isfinite(y_np)
        x_clean = x_np[valid_mask]
        y_clean = y_np[valid_mask]

        # Check for sufficient data points
        if len(x_clean) < 2:
            return np.nan

        # Check for zero variance in correlation metrics
        is_correlation = 'pearsonr' in metric_func.__name__ or 'spearmanr' in metric_func.__name__
        if is_correlation:
            if np.std(x_clean) < 1e-8 or np.std(y_clean) < 1e-8:
                return np.nan

        # Calculate metric based on function type
        if 'pearsonr' in metric_func.__name__:
            result, _ = metric_func(x_clean, y_clean)
        elif 'spearmanr' in metric_func.__name__:
            result, _ = metric_func(x_clean, y_clean)
        else:
            result = metric_func(x_clean, y_clean, **kwargs)

        # Return result only if finite
        return float(result) if np.isfinite(result) else np.nan

    except Exception as e:
        logger.debug(f"Error calculating metric {metric_func.__name__}: {e}")
        return np.nan  # Return NaN on any exception


def calculate_metrics(df: pd.DataFrame, label: str = "Overall") -> Dict[str, float]:
    """
    Calculate standard regression metrics from a DataFrame.
    
    Args:
        df: DataFrame with 'target_rmsf' and 'predicted_rmsf' columns
        label: Label for logging purposes
        
    Returns:
        Dictionary with calculated metrics
    """
    metrics = {}
    count = len(df)
    metrics['count'] = int(count)

    if count < 2:
        logger.warning(f"Metrics ({label}): Insufficient samples ({count}) for calculation.")
        # Return dict with count and NaNs for other metrics
        nan_metrics = [
            'pearson', 'spearman', 'r2', 'mse', 'rmse', 'mae',
            'mean_absolute_relative_error', 'median_absolute_relative_error',
            'cv_rmse_percent'
        ]
        for key in nan_metrics:
            metrics[key] = np.nan
        return metrics

    if 'target_rmsf' not in df or 'predicted_rmsf' not in df:
        logger.warning(f"Metrics ({label}): Missing target_rmsf or predicted_rmsf columns.")
        return metrics  # Should not happen if merge worked, but safety check

    y_true = df['target_rmsf'].values
    y_pred = df['predicted_rmsf'].values

    # Calculate standard metrics safely
    metrics['pearson'] = safe_metric(pearsonr, y_true, y_pred)
    metrics['spearman'] = safe_metric(spearmanr, y_true, y_pred)
    metrics['r2'] = safe_metric(r2_score, y_true, y_pred)
    metrics['mse'] = safe_metric(mean_squared_error, y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse']) if pd.notna(metrics['mse']) else np.nan
    metrics['mae'] = safe_metric(mean_absolute_error, y_true, y_pred)

    # Calculate relative errors safely
    try:
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        
        if len(y_true_clean) > 0:
            abs_error = np.abs(y_true_clean - y_pred_clean)
            # Avoid division by zero or very small numbers for relative error
            safe_y_true = np.maximum(np.abs(y_true_clean), 1e-6)
            relative_error = abs_error / safe_y_true
            
            # Filter NaNs/Infs that might arise from division
            finite_rel_err = relative_error[np.isfinite(relative_error)]
            
            if len(finite_rel_err) > 0:
                metrics['mean_absolute_relative_error'] = float(np.mean(finite_rel_err)) * 100
                metrics['median_absolute_relative_error'] = float(np.median(finite_rel_err)) * 100
            else:
                metrics['mean_absolute_relative_error'] = np.nan
                metrics['median_absolute_relative_error'] = np.nan
                
            # Calculate CV(RMSE)
            mean_true = np.mean(y_true_clean)
            if pd.notna(metrics['rmse']) and abs(mean_true) > 1e-6:
                metrics['cv_rmse_percent'] = (metrics['rmse'] / mean_true) * 100
            else:
                metrics['cv_rmse_percent'] = np.nan
        else:
            metrics['mean_absolute_relative_error'] = np.nan
            metrics['median_absolute_relative_error'] = np.nan
            metrics['cv_rmse_percent'] = np.nan

    except Exception as e:
        logger.warning(f"Error calculating relative/CV metrics ({label}): {e}")
        metrics['mean_absolute_relative_error'] = np.nan
        metrics['median_absolute_relative_error'] = np.nan
        metrics['cv_rmse_percent'] = np.nan

    return metrics

# --- Permutation Importance ---
def perform_permutation_importance(
    model: nn.Module,
    perm_dataset: PredictionDataset,  # Dataset containing samples for permutation
    temp_scaler_func: Callable[[float], float],
    target_temperature: float,  # The original temperature (used to find baseline performance)
    baseline_metrics: Dict[str, float],  # Baseline performance on the same dataset
    device: torch.device,
    config: Dict[str, Any],
    n_repeats: int = 5
) -> Dict[str, float]:
    """
    Performs permutation importance analysis for the temperature feature.
    
    Args:
        model: The trained model
        perm_dataset: PredictionDataset containing the samples to evaluate on
        temp_scaler_func: Function to scale raw temperatures
        target_temperature: The original temperature the baseline metrics correspond to
        baseline_metrics: Dictionary of baseline metrics (e.g., pearson, rmse)
        device: The device to run inference on
        config: Configuration dictionary
        n_repeats: Number of times to repeat the permutation
        
    Returns:
        Dictionary containing permutation importance scores (e.g., 'pearson_drop')
    """
    logger.info(f"Performing Permutation Importance for Temperature (n_repeats={n_repeats})...")
    model.eval()  # Ensure model is in eval mode
    batch_size = config.get('prediction', {}).get('batch_size', 128)
    rng = np.random.default_rng(config['training']['seed'])

    # Create a shuffling pool of scaled temperatures from the training data distribution
    # This avoids biasing the permutation test with only the target temperature
    scaled_shuffling_pool = []
    try:
        # Try loading the full RMSF file to get all unique temperatures
        rmsf_df_full = load_aggregated_rmsf_data(config['input']['aggregated_rmsf_file'])
        all_temps_unique = rmsf_df_full['temperature_feature'].dropna().unique()
        scaled_shuffling_pool = [temp_scaler_func(t) for t in all_temps_unique]
        logger.info(f"Created temperature shuffling pool with {len(scaled_shuffling_pool)} unique scaled values.")
        del rmsf_df_full
        gc.collect()
    except Exception as e:
        logger.error(f"Could not load temps for shuffling pool from {config['input']['aggregated_rmsf_file']}: {e}. "
                    f"Using default [0, 1] range.")
        # Fallback to a simple range if file loading fails
        scaled_shuffling_pool = np.linspace(0.0, 1.0, 20).tolist()  # Use 20 points between 0 and 1

    if not scaled_shuffling_pool:
        logger.warning("Temp shuffling pool is empty. Using default [0, 1] range.")
        scaled_shuffling_pool = np.linspace(0.0, 1.0, 20).tolist()

    # Get baseline metrics from parameter
    baseline_pearson = baseline_metrics.get('pearson', np.nan)
    baseline_rmse = baseline_metrics.get('rmse', np.nan)

    if pd.isna(baseline_pearson) or pd.isna(baseline_rmse):
        logger.warning("Baseline metrics (Pearson, RMSE) are missing or NaN. "
                      "Cannot calculate importance drop/increase.")

    # Store metrics from each permutation repeat
    permuted_metrics_repeats: Dict[str, List[float]] = {'pearson': [], 'rmse': []}

    # Get the ground truth targets aligned with the dataset order
    aligned_targets_np = np.full(len(perm_dataset), np.nan, dtype=np.float64)  # Initialize with NaNs
    try:
        gt_df = load_aggregated_rmsf_data(config['input']['aggregated_rmsf_file'])
        gt_df['resid'] = pd.to_numeric(gt_df['resid'], errors='coerce').astype('Int64')
        gt_df['temperature_feature'] = pd.to_numeric(gt_df['temperature_feature'], errors='coerce')
        gt_df_filtered = gt_df[np.isclose(gt_df['temperature_feature'], target_temperature)].copy()
        gt_lookup = gt_df_filtered.set_index(['domain_id', 'resid'])['target_rmsf']
        
        # Create domain mapping for this lookup
        hdf5_keys_in_dataset = list(set(s[0] for s in perm_dataset.samples))
        domain_mapping_perm = create_domain_mapping(
            hdf5_keys_in_dataset, 
            gt_df_filtered['domain_id'].unique().tolist()
        )

        for i, (domain_id, resid_str) in enumerate(perm_dataset.samples):
            try:
                resid_int = int(resid_str)
                mapped_id = domain_mapping_perm.get(domain_id, domain_id)
                target = gt_lookup.get((mapped_id, resid_int))
                
                if target is None:  # Try base name
                    base_id = mapped_id.split('_')[0]
                    if base_id != mapped_id:
                        target = gt_lookup.get((base_id, resid_int))
                        
                if target is not None and not pd.isna(target):
                    aligned_targets_np[i] = float(target)
            except (ValueError, TypeError, KeyError):
                continue  # Ignore errors for specific samples
                
        del gt_df, gt_df_filtered, gt_lookup, domain_mapping_perm
        gc.collect()
        
        if np.isnan(aligned_targets_np).all():
            raise ValueError("Could not align any ground truth targets for permutation importance samples.")
    except Exception as e:
        logger.error(f"Failed to get aligned ground truth targets for permutation importance: {e}")
        return {"error": "Failed to get ground truth targets"}

    # Use mixed precision if enabled during training (get from config)
    autocast_enabled = (config.get('training', {})
                       .get('mixed_precision', {})
                       .get('enabled', False) and device.type == 'cuda')

    # --- Permutation Loop ---
    for n in range(n_repeats):
        logger.info(f"  Permutation importance repeat {n+1}/{n_repeats}...")
        permuted_preds_batches: List[np.ndarray] = []  # Store predictions for this repeat
        
        # Create a data loader for sequential processing
        permuted_loader = DataLoader(
            perm_dataset, 
            batch_size=batch_size, 
            sampler=SequentialSampler(perm_dataset), 
            num_workers=0
        )
        progress = EnhancedProgressBar(len(permuted_loader), desc=f"  Permutation {n+1}")

        with torch.no_grad():
            for i, batch_data in enumerate(permuted_loader):
                if not batch_data or len(batch_data) != 3:
                    continue
                    
                _, _, batch_voxels = batch_data

                if batch_voxels.numel() == 0:
                    continue
                    
                batch_voxels = batch_voxels.to(device, non_blocking=True)
                current_batch_size = batch_voxels.size(0)

                # --- Permute Temperature Feature ---
                # Create permuted temperature tensor for this batch
                shuffled_scaled_values = rng.choice(scaled_shuffling_pool, size=current_batch_size, replace=True)
                batch_shuffled_temps = torch.tensor(
                    shuffled_scaled_values, device=device, dtype=torch.float32
                ).unsqueeze(1)

                try:
                    # Run inference with permuted temperature
                    with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                        batch_outputs = model(voxel_input=batch_voxels, scaled_temp=batch_shuffled_temps)
                    permuted_preds_batches.append(batch_outputs.detach().cpu().numpy().flatten())
                except Exception as e:
                    logger.warning(f"Error predicting with permuted temps (batch {i}): {e}")
                    # Append NaNs if prediction fails for a batch
                    permuted_preds_batches.append(np.full(current_batch_size, np.nan))

                progress.update(1)
                del batch_voxels, batch_shuffled_temps, batch_outputs
                if i % 100 == 0:
                    clear_memory(force_gc=False, clear_cuda=True)
        progress.finish()

        # Concatenate predictions for the current repeat
        if permuted_preds_batches:
            perm_preds_np = np.concatenate(permuted_preds_batches)
            if len(perm_preds_np) != len(aligned_targets_np):
                logger.error(f"Permutation repeat {n+1}: Prediction length mismatch ({len(perm_preds_np)}) "
                            f"vs target length ({len(aligned_targets_np)}). Skipping metrics for this repeat.")
                continue

            # Calculate metrics for this repeat using the permuted predictions
            perm_pearson = safe_metric(pearsonr, aligned_targets_np, perm_preds_np)
            perm_mse = safe_metric(mean_squared_error, aligned_targets_np, perm_preds_np)
            perm_rmse = np.sqrt(perm_mse) if pd.notna(perm_mse) else np.nan

            if pd.notna(perm_pearson):
                permuted_metrics_repeats['pearson'].append(perm_pearson)
            if pd.notna(perm_rmse):
                permuted_metrics_repeats['rmse'].append(perm_rmse)
        else:
            logger.warning(f"No predictions generated for permutation repeat {n+1}.")

    # --- Calculate Final Importance Scores ---
    importance_scores: Dict[str, float] = {}
    if permuted_metrics_repeats['pearson'] and pd.notna(baseline_pearson):
        mean_perm_pearson = float(np.mean(permuted_metrics_repeats['pearson']))
        std_perm_pearson = float(np.std(permuted_metrics_repeats['pearson']))
        importance_scores['pearson_permuted_mean'] = mean_perm_pearson
        importance_scores['pearson_permuted_std'] = std_perm_pearson
        importance_scores['pearson_drop'] = baseline_pearson - mean_perm_pearson
    else:
        logger.warning("Could not calculate Pearson drop for permutation importance "
                      "(missing baseline or permuted values).")

    if permuted_metrics_repeats['rmse'] and pd.notna(baseline_rmse):
        mean_perm_rmse = float(np.mean(permuted_metrics_repeats['rmse']))
        std_perm_rmse = float(np.std(permuted_metrics_repeats['rmse']))
        importance_scores['rmse_permuted_mean'] = mean_perm_rmse
        importance_scores['rmse_permuted_std'] = std_perm_rmse
        importance_scores['rmse_increase'] = mean_perm_rmse - baseline_rmse
    else:
        logger.warning("Could not calculate RMSE increase for permutation importance "
                      "(missing baseline or permuted values).")

    logger.info("--- Permutation Importance Results ---")
    if importance_scores:
        for key, val in importance_scores.items():
            logger.info(f"  {key}: {val:.4f}")
    else:
        logger.info("  No valid permutation importance scores calculated.")

    return importance_scores


# --- Main Evaluation Function ---
def evaluate_model(
    config: Dict[str, Any],
    model_path: str,
    predictions_path: str,
) -> Optional[str]:
    """
    Evaluates model performance using a predictions file and ground truth data.
    
    Args:
        config: Configuration dictionary
        model_path: Path to the trained model checkpoint (.pt)
        predictions_path: Path to the predictions CSV file generated by 'predict'
        
    Returns:
        Path to the saved evaluation metrics JSON file, or None on failure
    """
    run_output_dir = config["output"]["run_dir"]
    metrics_dir = config["output"]["metrics_dir"]
    ensure_dir(metrics_dir)

    # Construct metrics filename based on predictions filename
    pred_basename = os.path.splitext(os.path.basename(predictions_path))[0]
    metrics_filename = f"evaluation_metrics_{pred_basename}.json"
    metrics_path = os.path.join(metrics_dir, metrics_filename)

    log_section_header(logger, "MODEL EVALUATION")
    logger.info(f"Predictions file: {predictions_path}")
    logger.info(f"Model file: {model_path}")
    logger.info(f"Output metrics JSON: {metrics_path}")
    log_memory_usage(logger)

    all_metrics: Dict[str, Any] = {  # Initialize results dict
        'overall': {},
        'stratified': {},
        'permutation_importance': {},
        'evaluation_temperature': None,
        'model_path': model_path,
        'predictions_path': predictions_path,
        'input_data': {
            'voxel_file': config['input'].get('voxel_file'),
            'aggregated_rmsf_file': config['input'].get('aggregated_rmsf_file'),
            'test_split_file': config['input'].get('test_split_file', 'N/A')
        }
    }

    # Use try-finally to ensure cleanup actions happen
    model = None  # Define outside try block for finally
    perm_dataset = None
    try:
        # --- Load Predictions ---
        with log_stage("EVAL_SETUP", "Loading predictions"):
            try:
                preds_df = pd.read_csv(predictions_path, dtype={'domain_id': str, 'resname': str})
                
                # Validate required columns and types
                preds_df['resid'] = pd.to_numeric(preds_df['resid'], errors='coerce').astype('Int64')
                preds_df['predicted_rmsf'] = pd.to_numeric(preds_df['predicted_rmsf'], errors='coerce')
                preds_df['prediction_temperature'] = pd.to_numeric(preds_df['prediction_temperature'], errors='coerce')
                preds_df.dropna(subset=['domain_id', 'resid', 'predicted_rmsf', 'prediction_temperature'], inplace=True)

                if preds_df.empty:
                    raise ValueError("Predictions file contains no valid rows after validation.")

                # Determine evaluation temperature from predictions file
                unique_temps = preds_df['prediction_temperature'].unique()
                if len(unique_temps) > 1:
                    logger.warning(f"Predictions file contains multiple temperatures ({unique_temps}). "
                                  f"Evaluation will compare against ground truth at the first temperature found: "
                                  f"{unique_temps[0]:.1f}K")
                elif len(unique_temps) == 0:
                    raise ValueError("Cannot determine evaluation temperature from predictions file.")
                    
                eval_temp = unique_temps[0]
                all_metrics['evaluation_temperature'] = float(eval_temp)  # Store eval temp
                logger.info(f"Evaluating predictions made at temperature: {eval_temp:.1f}K")

            except Exception as e:
                logger.exception(f"Failed to load or validate predictions CSV '{predictions_path}': {e}")
                return None  # Exit if predictions fail to load

        # --- Load Ground Truth ---
        with log_stage("EVAL_SETUP", "Loading and filtering ground truth data"):
            try:
                gt_df_raw = load_aggregated_rmsf_data(config['input']['aggregated_rmsf_file'])
                
                # Select necessary columns, including optional ones for stratification
                cols_to_keep = ['domain_id', 'resid', 'resname', 'target_rmsf', 'temperature_feature']
                optional_cols = ['relative_accessibility', 'dssp', 'secondary_structure_encoded']
                available_optional = [col for col in optional_cols if col in gt_df_raw.columns]
                cols_to_keep.extend(available_optional)

                gt_df = gt_df_raw[cols_to_keep].copy()
                gt_df['resid'] = pd.to_numeric(gt_df['resid'], errors='coerce').astype('Int64')
                gt_df['temperature_feature'] = pd.to_numeric(gt_df['temperature_feature'], errors='coerce')
                
                # Drop rows where essential GT info is missing BEFORE filtering by temp
                gt_df.dropna(subset=['domain_id', 'resid', 'target_rmsf', 'temperature_feature'], inplace=True)

                # Filter ground truth to match the evaluation temperature
                gt_df_filtered = gt_df[np.isclose(gt_df['temperature_feature'], eval_temp)].copy()
                logger.info(f"Loaded and filtered ground truth to {len(gt_df_filtered)} rows matching "
                           f"temp ~{eval_temp:.1f}K.")

                if gt_df_filtered.empty:
                    raise ValueError(f"No ground truth data matches evaluation temperature {eval_temp:.1f}K.")
                    
                del gt_df_raw, gt_df
                gc.collect()  # Free memory
            except Exception as e:
                logger.exception(f"Failed to load or process ground truth data: {e}")
                return None  # Exit if GT fails

        # --- Merge Predictions and Ground Truth ---
        with log_stage("EVAL_SETUP", "Merging predictions and ground truth"):
            try:
                logger.info("Merging predictions with filtered ground truth data...")
                
                # Ensure 'resid' is compatible type for merging (int after dropna)
                preds_df['resid'] = preds_df['resid'].astype(int)
                gt_df_filtered['resid'] = gt_df_filtered['resid'].astype(int)

                eval_df = pd.merge(
                    preds_df,
                    gt_df_filtered.drop(columns=['temperature_feature']),  # Don't need temp column from GT anymore
                    on=['domain_id', 'resid'],  # Merge keys
                    how='inner',  # Only keep overlapping entries
                    suffixes=('_pred', '_gt')  # Suffixes for potentially overlapping columns like resname
                )

                # Handle potential duplicate resname columns after merge
                if 'resname_pred' in eval_df.columns and 'resname_gt' in eval_df.columns:
                    eval_df['resname'] = eval_df['resname_gt']  # Prioritize ground truth resname
                    eval_df.drop(columns=['resname_pred', 'resname_gt'], inplace=True)
                elif 'resname_gt' in eval_df.columns:  # Rename if only GT suffix exists
                    eval_df.rename(columns={'resname_gt': 'resname'}, inplace=True)
                elif 'resname_pred' in eval_df.columns:  # Rename if only Pred suffix exists
                    eval_df.rename(columns={'resname_pred': 'resname'}, inplace=True)
                elif 'resname' not in eval_df.columns:  # If neither exists
                    eval_df['resname'] = 'UNK'  # Add placeholder

                logger.info(f"Merged data contains {len(eval_df)} overlapping entries.")
                if eval_df.empty:
                    logger.error("Merge resulted in empty DataFrame. Check domain/resid matching "
                                "between predictions and ground truth.")
                    return None

                # Final check for NaNs in prediction/target columns
                eval_df.dropna(subset=['predicted_rmsf', 'target_rmsf'], inplace=True)
                logger.info(f"{len(eval_df)} valid entries remaining after RMSF NaN check.")
                if eval_df.empty:
                    logger.error("No valid overlapping entries after RMSF NaN check.")
                    return None
            except Exception as e:
                logger.exception(f"Error merging prediction and ground truth data: {e}")
                return None  # Exit on merge error

        # --- Calculate Overall Metrics ---
        with log_stage("EVALUATION", "Calculating overall metrics"):
            overall_metrics = calculate_metrics(eval_df, label="Overall")
            all_metrics['overall'] = overall_metrics
            logger.info("--- Overall Performance ---")
            if overall_metrics:
                for key, val in overall_metrics.items():
                    logger.info(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
            else:
                logger.warning("Overall metrics could not be calculated.")

        # --- Stratified Metrics ---
        if config['evaluation'].get('calculate_stratified_metrics', True):
            with log_stage("EVALUATION", "Calculating stratified metrics"):
                all_metrics['stratified'] = {}  # Ensure key exists

                # By Secondary Structure
                ss_col = next((c for c in ['dssp', 'secondary_structure_encoded'] if c in eval_df.columns), None)
                if ss_col:
                    logger.info(f"Stratifying metrics by Secondary Structure ('{ss_col}')...")
                    strat_ss_metrics = {}
                    for ss_type, group in eval_df.groupby(ss_col):
                        if isinstance(group, pd.DataFrame):  # Ensure group is DataFrame
                            metrics = calculate_metrics(group, label=f"SS={ss_type}")
                            if metrics.get('count', 0) > 10:  # Only report if enough samples
                                strat_ss_metrics[str(ss_type)] = metrics
                            else:
                                logger.debug(f"Skipping SS type '{ss_type}' due to insufficient samples "
                                            f"({metrics.get('count', 0)}).")
                    all_metrics['stratified']['secondary_structure'] = strat_ss_metrics
                else:
                    logger.warning("Secondary structure column ('dssp' or 'secondary_structure_encoded') "
                                  "not found for stratification.")

                # By Solvent Accessibility
                sasa_col = 'relative_accessibility'
                if sasa_col in eval_df.columns:
                    logger.info(f"Stratifying metrics by Relative Accessibility ('{sasa_col}')...")
                    sasa_bins = config['evaluation'].get('sasa_bins', [0.0, 0.1, 0.4, 1.01])
                    bin_labels = [f"{sasa_bins[i]:.1f}-{sasa_bins[i+1]:.1f}" for i in range(len(sasa_bins)-1)]
                    strat_sasa_metrics = {}
                    try:
                        # Ensure column is numeric and handle potential errors during binning
                        eval_df[sasa_col] = pd.to_numeric(eval_df[sasa_col], errors='coerce')
                        eval_df['sasa_bin'] = pd.cut(
                            eval_df[sasa_col], 
                            bins=sasa_bins, 
                            labels=bin_labels, 
                            right=False, 
                            include_lowest=True
                        )

                        for bin_label, group in eval_df.groupby('sasa_bin', observed=False):  # Use observed=False for categorical
                            if isinstance(group, pd.DataFrame):
                                metrics = calculate_metrics(group, label=f"SASA={bin_label}")
                                if metrics.get('count', 0) > 10:
                                    strat_sasa_metrics[str(bin_label)] = metrics
                                else:
                                    logger.debug(f"Skipping SASA bin '{bin_label}' due to insufficient samples "
                                                f"({metrics.get('count', 0)}).")
                        all_metrics['stratified']['sasa'] = strat_sasa_metrics
                    except Exception as sasa_e:
                        logger.error(f"Error during SASA binning/stratification: {sasa_e}")
                else:
                    logger.warning("SASA column ('relative_accessibility') not found for stratification.")

        # --- Permutation Importance Calculation ---
        if config['evaluation'].get('calculate_permutation_importance', True):
            with log_stage("EVALUATION", "Calculating permutation importance"):
                perm_importance_scores = {}  # Initialize scores dict
                try:
                    logger.info("Loading resources for permutation importance...")
                    device = get_device(config["system_utilization"]["adjust_for_gpu"])
                    checkpoint = torch.load(model_path, map_location='cpu')
                    model_config = checkpoint.get('config', {}).get('model', {})
                    input_shape = checkpoint.get('input_shape')
                    scaling_params = checkpoint.get('temp_scaling_params')
                    
                    if not model_config or not input_shape:
                        raise ValueError("Model config or shape missing from checkpoint.")
                    if not scaling_params:
                        raise ValueError("Temperature scaling parameters missing from checkpoint.")

                    model = get_model(model_config, input_shape=input_shape)
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    model.to(device)
                    model.eval()
                    del checkpoint
                    gc.collect()

                    temp_scaler_func = get_temperature_scaler(params=scaling_params)

                    # Get samples present in the final evaluation dataframe
                    perm_samples_to_load = list(eval_df[['domain_id', 'resid']].apply(
                        lambda x: (x['domain_id'], str(x['resid'])), axis=1
                    ).unique())
                    
                    if not perm_samples_to_load:
                        raise ValueError("No samples available in eval_df for permutation importance.")

                    logger.info(f"Creating permutation dataset for {len(perm_samples_to_load)} unique residues...")
                    perm_dataset = PredictionDataset(
                        samples_to_load=perm_samples_to_load,
                        voxel_hdf5_path=config['input']['voxel_file'],
                        expected_channels=model_config['input_channels'],
                        target_shape_chw=input_shape
                    )

                    if len(perm_dataset) == 0:
                        raise ValueError("Permutation dataset is empty after loading voxels.")
                        
                    if len(perm_dataset) != len(eval_df):
                        logger.warning(f"Permutation dataset size ({len(perm_dataset)}) differs from "
                                      f"eval_df size ({len(eval_df)}). Ensure samples match.")
                        # This might happen if some voxels failed loading for eval_df samples
                        # We need to use the intersection for fair comparison
                        # Filter eval_df to only include samples in perm_dataset
                        perm_dataset_keys = set(perm_dataset.samples)
                        eval_df_keys = set(eval_df[['domain_id', 'resid']].apply(
                            lambda x: (x['domain_id'], str(x['resid'])), axis=1
                        ))
                        common_keys = perm_dataset_keys.intersection(eval_df_keys)
                        eval_df_filtered_for_perm = eval_df[eval_df[['domain_id', 'resid']].apply(
                            lambda x: (x['domain_id'], str(x['resid'])), axis=1
                        ).isin(common_keys)].copy()
                        
                        logger.info(f"Filtered eval_df to {len(eval_df_filtered_for_perm)} samples "
                                   f"matching permutation dataset.")
                        if len(eval_df_filtered_for_perm) != len(perm_dataset):
                            # This should not happen if logic is correct
                            raise RuntimeError("Mismatch between filtered eval_df and permutation dataset sizes.")
                            
                        # Recalculate baseline on the filtered set for fair comparison
                        filtered_baseline_metrics = calculate_metrics(
                            eval_df_filtered_for_perm, label="Permutation Baseline"
                        )
                    else:
                        # Sizes match, use original overall metrics
                        filtered_baseline_metrics = overall_metrics

                    n_repeats = config['evaluation'].get('permutation_n_repeats', 5)

                    # Perform permutation importance
                    perm_importance_scores = perform_permutation_importance(
                        model, perm_dataset, temp_scaler_func, eval_temp, 
                        filtered_baseline_metrics, device, config, n_repeats
                    )

                except Exception as perm_e:
                    logger.exception(f"Failed to perform permutation importance: {perm_e}")
                    perm_importance_scores = {"error": str(perm_e)}
                finally:
                    # Clean up permutation specific resources
                    del model  # Ensure model loaded for perm is deleted
                    del perm_dataset
                    clear_memory(force_gc=True, clear_cuda=True)

                all_metrics['permutation_importance'] = perm_importance_scores

        # --- Save All Metrics ---
        with log_stage("OUTPUT", "Saving evaluation metrics"):
            try:
                # Helper function to convert numpy types for JSON serialization
                def convert_numpy_types(obj):
                    if isinstance(obj, (np.int_, np.integer)):
                        return int(obj)
                    elif isinstance(obj, (np.float_, np.floating)):
                        return float(obj) if np.isfinite(obj) else None  # Convert non-finite floats to None
                    elif isinstance(obj, (np.bool_)):
                        return bool(obj)
                    elif isinstance(obj, (np.void)):
                        return None
                    elif isinstance(obj, np.ndarray):
                        return [convert_numpy_types(item) for item in obj]  # Recursively convert array elements
                    elif isinstance(obj, dict):
                        return {k: convert_numpy_types(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [convert_numpy_types(i) for i in obj]
                    elif pd.isna(obj):
                        return None  # Handle pandas NaNs
                    else:
                        return obj  # Keep other types as is

                metrics_serializable = convert_numpy_types(all_metrics)
                save_json(metrics_serializable, metrics_path)
                logger.info(f"Evaluation metrics saved successfully to {metrics_path}")
            except Exception as e:
                logger.exception(f"Failed to save metrics JSON to {metrics_path}: {e}")
                return None  # Indicate failure to save

        log_memory_usage(logger)
        logger.info("Evaluation finished successfully.")
        return metrics_path  # Return path on success

    finally:
        # Final cleanup
        clear_memory(force_gc=True)