"""
Evaluation command for Voxelflex.

This module handles evaluating the performance of trained RMSF models.
"""


import os
import time
import json
import gc
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch  # Add this import
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from voxelflex.utils.logging_utils import get_logger
from voxelflex.utils.file_utils import ensure_dir, save_json
from voxelflex.utils.system_utils import clear_memory, check_memory_usage  # Add these imports

logger = get_logger(__name__)

logger = get_logger(__name__)




def evaluate_model(
    config: Dict[str, Any],
    model_path: str,
    predictions_path: Optional[str] = None
) -> str:
    """
    Evaluate model performance using various metrics with enhanced memory management.
    
    Args:
        config: Configuration dictionary
        model_path: Path to the trained model file
        predictions_path: Path to predictions file. If None, predictions will be made.
        
    Returns:
        Path to the metrics file
    """
    logger.info("Evaluating model performance with optimized memory handling")
    
    # MEMORY OPTIMIZATION: Check initial memory usage
    memory_stats = check_memory_usage()
    logger.info(f"Initial memory: System: {memory_stats['system_percent']}% used, "
               f"Process: {memory_stats['process_rss_gb']:.2f} GB")
    
    # Extract domain information from model if available
    try:
        # Load checkpoint but be careful with memory
        device = torch.device('cpu')  # Use CPU for checkpoint loading to save GPU memory
        checkpoint = torch.load(model_path, map_location=device)
        processed_domains = checkpoint.get('processed_domains', None)
        
        if processed_domains:
            logger.info(f"Model was trained on {len(processed_domains)} domains")
        
        # Extract only what we need and clear the rest
        config_from_checkpoint = checkpoint.get('config', {})
        # We don't need the full checkpoint anymore
        del checkpoint
        clear_memory(force_gc=True)
        
    except Exception as e:
        logger.warning(f"Could not extract domain information from model: {str(e)}")
    
    # If predictions_path is not provided, generate predictions
    if predictions_path is None:
        from voxelflex.cli.commands.predict import predict_rmsf
        # Force memory clearing before prediction
        clear_memory(force_gc=True, clear_cuda=True)
        predictions_path = predict_rmsf(config, model_path)
    
    # MEMORY OPTIMIZATION: Load predictions in chunks for large files
    logger.info(f"Loading predictions from {predictions_path}")
    
    try:
        # Check file size to determine if chunking is needed
        file_size_mb = os.path.getsize(predictions_path) / (1024 * 1024)
        
        if file_size_mb > 500:  # For files larger than 500MB
            logger.info(f"Large predictions file detected ({file_size_mb:.1f} MB). Loading in chunks.")
            
            # Read in chunks with specified dtype to reduce memory usage
            predictions_df = pd.read_csv(
                predictions_path,
                dtype={
                    'domain_id': 'category',  # Use category dtype for domain_id to save memory
                    'resid': 'int32',         # Use int32 instead of int64
                    'resname': 'category',    # Use category dtype for resname to save memory
                    'predicted_rmsf': 'float32',  # Use float32 instead of float64
                    'actual_rmsf': 'float32'      # Use float32 instead of float64
                },
                chunksize=100000  # Load 100k rows at a time
            )
            
            # Process chunks to calculate basic statistics first
            chunk_list = []
            total_rows = 0
            y_true_sum = 0
            y_pred_sum = 0
            y_true_min = float('inf')
            y_true_max = float('-inf')
            
            for i, chunk in enumerate(predictions_df):
                logger.info(f"Processing chunk {i+1} for basic statistics")
                total_rows += len(chunk)
                
                # Calculate running statistics
                y_true_sum += chunk['actual_rmsf'].sum()
                y_pred_sum += chunk['predicted_rmsf'].sum()
                y_true_min = min(y_true_min, chunk['actual_rmsf'].min())
                y_true_max = max(y_true_max, chunk['actual_rmsf'].max())
                
                # Add to list but monitor memory
                chunk_list.append(chunk)
                
                # Check memory pressure and clear if needed
                if i % 5 == 0 and i > 0:
                    memory_stats = check_memory_usage()
                    if memory_stats['system_percent'] > 80:
                        logger.warning(f"High memory pressure ({memory_stats['system_percent']}%). "
                                      f"Clearing memory after chunk {i+1}")
                        clear_memory(force_gc=True)
            
            # Now combine chunks but be careful with memory
            try:
                predictions_df = pd.concat(chunk_list, ignore_index=True)
                del chunk_list
                clear_memory(force_gc=True)
            except Exception as e:
                logger.error(f"Error combining chunks: {str(e)}")
                # Fall back to processing each chunk separately
                logger.warning("Falling back to processing chunks separately")
                predictions_df = pd.read_csv(
                    predictions_path,
                    dtype={
                        'domain_id': 'category',
                        'resid': 'int32',
                        'resname': 'category',
                        'predicted_rmsf': 'float32',
                        'actual_rmsf': 'float32'
                    },
                    chunksize=100000
                )
        else:
            # Standard loading for smaller files with optimized dtypes
            predictions_df = pd.read_csv(
                predictions_path,
                dtype={
                    'domain_id': 'category',
                    'resid': 'int32',
                    'resname': 'category',
                    'predicted_rmsf': 'float32',
                    'actual_rmsf': 'float32'
                }
            )
    except Exception as e:
        logger.error(f"Error loading predictions: {str(e)}")
        raise
    
    # MEMORY OPTIMIZATION: Check memory after loading predictions
    memory_stats = check_memory_usage()
    logger.info(f"Memory after loading predictions: {memory_stats['system_percent']}%, "
               f"{memory_stats['process_rss_gb']:.2f} GB")
    
    # Log some information about the domains in the predictions
    if isinstance(predictions_df, pd.DataFrame) and 'domain_id' in predictions_df.columns:
        domains_in_predictions = predictions_df['domain_id'].nunique()
        total_residues = len(predictions_df)
        logger.info(f"Evaluating predictions for {domains_in_predictions} domains with {total_residues} total residues")
    
    # Calculate overall metrics with memory-efficient approach
    if isinstance(predictions_df, pd.DataFrame):
        # For DataFrame case (smaller files)
        y_true = predictions_df['actual_rmsf'].values
        y_pred = predictions_df['predicted_rmsf'].values
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate coefficient of variation of RMSD
        cv_rmsd = rmse / np.mean(y_true) * 100
        
        # Calculate relative error metrics
        mean_relative_error = np.mean(np.abs(y_pred - y_true) / np.maximum(0.01, y_true)) * 100
        median_relative_error = np.median(np.abs(y_pred - y_true) / np.maximum(0.01, y_true)) * 100
        
    else:
        # For chunked DataFrame case (larger files)
        # Initialize metrics
        sum_squared_error = 0
        sum_absolute_error = 0
        sum_true = 0
        sum_pred = 0
        sum_true_squared = 0
        sum_pred_squared = 0
        sum_product = 0
        count = 0
        sum_relative_error = 0
        relative_errors = []
        
        # Process each chunk
        for i, chunk in enumerate(predictions_df):
            logger.info(f"Processing metrics for chunk {i+1}")
            
            chunk_y_true = chunk['actual_rmsf'].values
            chunk_y_pred = chunk['predicted_rmsf'].values
            chunk_count = len(chunk_y_true)
            
            # Update running sums for metrics
            sum_squared_error += np.sum((chunk_y_pred - chunk_y_true) ** 2)
            sum_absolute_error += np.sum(np.abs(chunk_y_pred - chunk_y_true))
            sum_true += np.sum(chunk_y_true)
            sum_pred += np.sum(chunk_y_pred)
            sum_true_squared += np.sum(chunk_y_true ** 2)
            sum_pred_squared += np.sum(chunk_y_pred ** 2)
            sum_product += np.sum(chunk_y_true * chunk_y_pred)
            
            # Collect relative errors for median calculation
            chunk_relative_errors = np.abs(chunk_y_pred - chunk_y_true) / np.maximum(0.01, chunk_y_true) * 100
            sum_relative_error += np.sum(chunk_relative_errors)
            relative_errors.extend(chunk_relative_errors.tolist())
            
            count += chunk_count
            
            # Clear memory between chunks
            if i % 3 == 0:
                clear_memory(force_gc=True)
        
        # Calculate final metrics
        mse = sum_squared_error / count
        rmse = np.sqrt(mse)
        mae = sum_absolute_error / count
        
        # Calculate R² manually
        mean_true = sum_true / count
        mean_pred = sum_pred / count
        numerator = sum_product - count * mean_true * mean_pred
        denominator_true = sum_true_squared - count * mean_true ** 2
        denominator_pred = sum_pred_squared - count * mean_pred ** 2
        r2 = (numerator ** 2) / (denominator_true * denominator_pred)
        
        # Calculate coefficient of variation of RMSD
        cv_rmsd = rmse / (sum_true / count) * 100
        
        # Calculate relative error metrics
        mean_relative_error = sum_relative_error / count
        median_relative_error = np.median(np.array(relative_errors))
        
        # Clear relative errors array to save memory
        del relative_errors
        clear_memory(force_gc=True)
    
    # Calculate domain-level metrics if domain info is available - with memory optimization
    domain_metrics = {}
    if isinstance(predictions_df, pd.DataFrame) and 'domain_id' in predictions_df.columns:
        # Process domains in batches to manage memory
        unique_domains = predictions_df['domain_id'].unique()
        domain_batch_size = min(50, len(unique_domains))  # Process 50 domains at a time or fewer
        
        for batch_start in range(0, len(unique_domains), domain_batch_size):
            batch_end = min(batch_start + domain_batch_size, len(unique_domains))
            batch_domains = unique_domains[batch_start:batch_end]
            
            logger.info(f"Processing domain metrics batch {batch_start//domain_batch_size + 1} "
                       f"(domains {batch_start} to {batch_end-1})")
            
            for domain in batch_domains:
                domain_df = predictions_df[predictions_df['domain_id'] == domain]
                domain_y_true = domain_df['actual_rmsf'].values
                domain_y_pred = domain_df['predicted_rmsf'].values
                
                # Skip if we don't have enough data
                if len(domain_y_true) < 2:
                    continue
                
                try:
                    domain_mse = mean_squared_error(domain_y_true, domain_y_pred)
                    domain_rmse = np.sqrt(domain_mse)
                    domain_mae = mean_absolute_error(domain_y_true, domain_y_pred)
                    domain_r2 = r2_score(domain_y_true, domain_y_pred)
                    
                    # Additional metrics for better evaluation
                    domain_mean_true = np.mean(domain_y_true)
                    domain_mean_pred = np.mean(domain_y_pred)
                    domain_median_true = np.median(domain_y_true)
                    domain_median_pred = np.median(domain_y_pred)
                    domain_cv_rmsd = domain_rmse / domain_mean_true * 100 if domain_mean_true > 0 else float('inf')
                    
                    # Calculate correlation coefficient
                    domain_corr = np.corrcoef(domain_y_true, domain_y_pred)[0, 1]
                    
                    domain_metrics[domain] = {
                        'mse': float(domain_mse),
                        'rmse': float(domain_rmse),
                        'mae': float(domain_mae),
                        'r2': float(domain_r2),
                        'corr': float(domain_corr),
                        'cv_rmsd': float(domain_cv_rmsd),
                        'mean_true': float(domain_mean_true),
                        'mean_pred': float(domain_mean_pred),
                        'median_true': float(domain_median_true),
                        'median_pred': float(domain_median_pred),
                        'num_residues': int(len(domain_df))
                    }
                except Exception as e:
                    logger.warning(f"Error calculating metrics for domain {domain}: {str(e)}")
            
            # Clear memory after each batch
            clear_memory(force_gc=True)
            
            # Check memory pressure and take emergency measures if needed
            memory_stats = check_memory_usage()
            if memory_stats['system_percent'] > 85:
                logger.warning(f"Critical memory pressure during domain metrics: {memory_stats['system_percent']}%")
                logger.warning(f"Stopping domain metrics calculation to preserve memory integrity")
                break
    elif not isinstance(predictions_df, pd.DataFrame):
        # For chunked processing (large files)
        logger.info("Processing domain metrics for large chunked files")
        
        # Get unique domains from all chunks
        unique_domains = set()
        domain_residue_counts = {}
        
        # First pass to identify unique domains and count residues
        for i, chunk in enumerate(predictions_df):
            for domain in chunk['domain_id'].unique():
                if domain not in unique_domains:
                    unique_domains.add(domain)
                    domain_residue_counts[domain] = 0
                
                domain_residue_counts[domain] += len(chunk[chunk['domain_id'] == domain])
            
            # Reset iterator for next pass
            if i == 0:
                predictions_df = pd.read_csv(
                    predictions_path,
                    dtype={
                        'domain_id': 'category',
                        'resid': 'int32',
                        'resname': 'category',
                        'predicted_rmsf': 'float32',
                        'actual_rmsf': 'float32'
                    },
                    chunksize=100000
                )
        
        # Second pass for metrics calculation
        unique_domains = list(unique_domains)
        domain_batch_size = min(20, len(unique_domains))  # Process fewer domains at a time for chunked files
        
        for batch_start in range(0, len(unique_domains), domain_batch_size):
            batch_end = min(batch_start + domain_batch_size, len(unique_domains))
            batch_domains = unique_domains[batch_start:batch_end]
            
            logger.info(f"Processing domain metrics batch {batch_start//domain_batch_size + 1} "
                       f"(domains {batch_start} to {batch_end-1})")
            
            # Initialize domain data collectors
            domain_data_collectors = {domain: {'true': [], 'pred': []} for domain in batch_domains}
            
            # Reset iterator for processing
            predictions_df = pd.read_csv(
                predictions_path,
                dtype={
                    'domain_id': 'category',
                    'resid': 'int32',
                    'resname': 'category',
                    'predicted_rmsf': 'float32',
                    'actual_rmsf': 'float32'
                },
                chunksize=100000
            )
            
            # Collect data for each domain
            for chunk in predictions_df:
                for domain in batch_domains:
                    domain_rows = chunk[chunk['domain_id'] == domain]
                    if len(domain_rows) > 0:
                        domain_data_collectors[domain]['true'].extend(domain_rows['actual_rmsf'].values)
                        domain_data_collectors[domain]['pred'].extend(domain_rows['predicted_rmsf'].values)
                
                # Clear chunk from memory
                del chunk
                clear_memory(force_gc=True)
            
            # Calculate metrics for collected domain data
            for domain in batch_domains:
                domain_y_true = np.array(domain_data_collectors[domain]['true'])
                domain_y_pred = np.array(domain_data_collectors[domain]['pred'])
                
                # Skip if we don't have enough data
                if len(domain_y_true) < 2:
                    continue
                
                try:
                    domain_mse = mean_squared_error(domain_y_true, domain_y_pred)
                    domain_rmse = np.sqrt(domain_mse)
                    domain_mae = mean_absolute_error(domain_y_true, domain_y_pred)
                    domain_r2 = r2_score(domain_y_true, domain_y_pred)
                    
                    # Additional metrics for better evaluation
                    domain_mean_true = np.mean(domain_y_true)
                    domain_mean_pred = np.mean(domain_y_pred)
                    domain_median_true = np.median(domain_y_true)
                    domain_median_pred = np.median(domain_y_pred)
                    domain_cv_rmsd = domain_rmse / domain_mean_true * 100 if domain_mean_true > 0 else float('inf')
                    
                    # Calculate correlation coefficient
                    domain_corr = np.corrcoef(domain_y_true, domain_y_pred)[0, 1]
                    
                    domain_metrics[domain] = {
                        'mse': float(domain_mse),
                        'rmse': float(domain_rmse),
                        'mae': float(domain_mae),
                        'r2': float(domain_r2),
                        'corr': float(domain_corr),
                        'cv_rmsd': float(domain_cv_rmsd),
                        'mean_true': float(domain_mean_true),
                        'mean_pred': float(domain_mean_pred),
                        'median_true': float(domain_median_true),
                        'median_pred': float(domain_median_pred),
                        'num_residues': int(domain_residue_counts[domain])
                    }
                except Exception as e:
                    logger.warning(f"Error calculating metrics for domain {domain}: {str(e)}")
            
            # Clear collectors to free memory
            del domain_data_collectors
            clear_memory(force_gc=True)
            
            # Check memory pressure and take emergency measures if needed
            memory_stats = check_memory_usage()
            if memory_stats['system_percent'] > 85:
                logger.warning(f"Critical memory pressure during domain metrics: {memory_stats['system_percent']}%")
                logger.warning(f"Stopping domain metrics calculation to preserve memory integrity")
                break
    
    # MEMORY OPTIMIZATION: Check memory after domain-level metrics
    memory_stats = check_memory_usage()
    logger.info(f"Memory after domain metrics: {memory_stats['system_percent']}%, "
              f"{memory_stats['process_rss_gb']:.2f} GB")
    
    # Calculate residue type metrics if residue type info is available - with memory optimization
    residue_type_metrics = {}
    
    # Define a function to calculate metrics for residue types
    def calculate_residue_type_metrics(data_source):
        nonlocal residue_type_metrics
        
        if isinstance(data_source, pd.DataFrame):
            # For DataFrame case
            if 'resname' in data_source.columns:
                # Get unique residue types with at least 10 data points
                valid_restypes = []
                restype_counts = {}
                
                for resname in data_source['resname'].dropna().unique():
                    resname_df = data_source[data_source['resname'] == resname]
                    restype_counts[resname] = len(resname_df)
                    
                    if len(resname_df) >= 10:  # Need at least 10 points for meaningful statistics
                        valid_restypes.append(resname)
                
                # Calculate metrics for each valid residue type
                for resname in valid_restypes:
                    resname_df = data_source[data_source['resname'] == resname]
                    resname_y_true = resname_df['actual_rmsf'].values
                    resname_y_pred = resname_df['predicted_rmsf'].values
                    
                    try:
                        restype_mse = mean_squared_error(resname_y_true, resname_y_pred)
                        restype_rmse = np.sqrt(restype_mse)
                        restype_mae = mean_absolute_error(resname_y_true, resname_y_pred)
                        restype_r2 = r2_score(resname_y_true, resname_y_pred)
                        
                        # Additional statistics for better analysis
                        mean_true = np.mean(resname_y_true)
                        mean_pred = np.mean(resname_y_pred)
                        median_true = np.median(resname_y_true)
                        median_pred = np.median(resname_y_pred)
                        
                        # Calculate normalized RMSE (NRMSE)
                        range_true = np.max(resname_y_true) - np.min(resname_y_true)
                        nrmse = restype_rmse / range_true if range_true > 0 else float('inf')
                        
                        # Calculate correlation coefficient
                        restype_corr = np.corrcoef(resname_y_true, resname_y_pred)[0, 1]
                        
                        residue_type_metrics[resname] = {
                            'mse': float(restype_mse),
                            'rmse': float(restype_rmse),
                            'mae': float(restype_mae),
                            'r2': float(restype_r2),
                            'nrmse': float(nrmse),
                            'corr': float(restype_corr),
                            'mean_true': float(mean_true),
                            'mean_pred': float(mean_pred),
                            'median_true': float(median_true),
                            'median_pred': float(median_pred),
                            'num_residues': int(len(resname_df))
                        }
                    except Exception as e:
                        logger.warning(f"Error calculating metrics for residue type {resname}: {str(e)}")
        else:
            # For chunked DataFrame case
            # First pass to identify unique residue types and count
            residue_type_counts = {}
            
            for chunk in data_source:
                if 'resname' in chunk.columns:
                    for resname in chunk['resname'].dropna().unique():
                        if resname not in residue_type_counts:
                            residue_type_counts[resname] = 0
                        
                        residue_type_counts[resname] += len(chunk[chunk['resname'] == resname])
            
            # Identify valid residue types (at least 10 data points)
            valid_restypes = [resname for resname, count in residue_type_counts.items() if count >= 10]
            
            # Reset iterator for next pass
            nonlocal predictions_path
            predictions_df = pd.read_csv(
                predictions_path,
                dtype={
                    'domain_id': 'category',
                    'resid': 'int32',
                    'resname': 'category',
                    'predicted_rmsf': 'float32',
                    'actual_rmsf': 'float32'
                },
                chunksize=100000
            )
            
            # Second pass for metrics calculation
            # Process residue types in batches to manage memory
            restype_batch_size = min(10, len(valid_restypes))
            
            for batch_start in range(0, len(valid_restypes), restype_batch_size):
                batch_end = min(batch_start + restype_batch_size, len(valid_restypes))
                batch_restypes = valid_restypes[batch_start:batch_end]
                
                logger.info(f"Processing residue type metrics batch {batch_start//restype_batch_size + 1} "
                           f"(types {batch_start} to {batch_end-1})")
                
                # Initialize residue type data collectors
                restype_data_collectors = {resname: {'true': [], 'pred': []} for resname in batch_restypes}
                
                # Collect data for each residue type
                for chunk in predictions_df:
                    for resname in batch_restypes:
                        resname_rows = chunk[chunk['resname'] == resname]
                        if len(resname_rows) > 0:
                            restype_data_collectors[resname]['true'].extend(resname_rows['actual_rmsf'].values)
                            restype_data_collectors[resname]['pred'].extend(resname_rows['predicted_rmsf'].values)
                    
                    # Clear chunk from memory
                    del chunk
                    clear_memory(force_gc=True)
                
                # Reset iterator for next batch
                predictions_df = pd.read_csv(
                    predictions_path,
                    dtype={
                        'domain_id': 'category',
                        'resid': 'int32',
                        'resname': 'category',
                        'predicted_rmsf': 'float32',
                        'actual_rmsf': 'float32'
                    },
                    chunksize=100000
                )
                
                # Calculate metrics for collected residue type data
                for resname in batch_restypes:
                    resname_y_true = np.array(restype_data_collectors[resname]['true'])
                    resname_y_pred = np.array(restype_data_collectors[resname]['pred'])
                    
                    # Skip if we don't have enough data
                    if len(resname_y_true) < 10:
                        continue
                    
                    try:
                        restype_mse = mean_squared_error(resname_y_true, resname_y_pred)
                        restype_rmse = np.sqrt(restype_mse)
                        restype_mae = mean_absolute_error(resname_y_true, resname_y_pred)
                        restype_r2 = r2_score(resname_y_true, resname_y_pred)
                        
                        # Additional statistics for better analysis
                        mean_true = np.mean(resname_y_true)
                        mean_pred = np.mean(resname_y_pred)
                        median_true = np.median(resname_y_true)
                        median_pred = np.median(resname_y_pred)
                        
                        # Calculate normalized RMSE (NRMSE)
                        range_true = np.max(resname_y_true) - np.min(resname_y_true)
                        nrmse = restype_rmse / range_true if range_true > 0 else float('inf')
                        
                        # Calculate correlation coefficient
                        restype_corr = np.corrcoef(resname_y_true, resname_y_pred)[0, 1]
                        
                        residue_type_metrics[resname] = {
                            'mse': float(restype_mse),
                            'rmse': float(restype_rmse),
                            'mae': float(restype_mae),
                            'r2': float(restype_r2),
                            'nrmse': float(nrmse),
                            'corr': float(restype_corr),
                            'mean_true': float(mean_true),
                            'mean_pred': float(mean_pred),
                            'median_true': float(median_true),
                            'median_pred': float(median_pred),
                            'num_residues': int(residue_type_counts[resname])
                        }
                    except Exception as e:
                        logger.warning(f"Error calculating metrics for residue type {resname}: {str(e)}")
                
                # Clear collectors to free memory
                del restype_data_collectors
                clear_memory(force_gc=True)
                
                # Check memory pressure and take emergency measures if needed
                memory_stats = check_memory_usage()
                if memory_stats['system_percent'] > 85:
                    logger.warning(f"Critical memory pressure during residue type metrics: {memory_stats['system_percent']}%")
                    logger.warning(f"Stopping residue type metrics calculation to preserve memory integrity")
                    break
    
    # Calculate residue type metrics
    if isinstance(predictions_df, pd.DataFrame) or not isinstance(predictions_df, pd.DataFrame):
        try:
            calculate_residue_type_metrics(predictions_df)
        except Exception as e:
            logger.error(f"Error calculating residue type metrics: {str(e)}")
    
    # MEMORY OPTIMIZATION: Check memory after residue type metrics
    memory_stats = check_memory_usage()
    logger.info(f"Memory after residue type metrics: {memory_stats['system_percent']}%, "
               f"{memory_stats['process_rss_gb']:.2f} GB")
    
    # Calculate metrics for different RMSF ranges with memory optimization
    flexibility_metrics = {}
    
    # Define a function to calculate flexibility metrics
    def calculate_flexibility_metrics(data_source):
        nonlocal flexibility_metrics
        
        if isinstance(data_source, pd.DataFrame):
            # For DataFrame case
            y_true = data_source['actual_rmsf'].values
            y_pred = data_source['predicted_rmsf'].values
            
            # Define RMSF flexibility ranges
            low_flex_threshold = np.percentile(y_true, 33.3)
            high_flex_threshold = np.percentile(y_true, 66.7)
            
            # Low flexibility regions
            low_flex_mask = y_true <= low_flex_threshold
            if np.sum(low_flex_mask) >= 10:
                low_flex_true = y_true[low_flex_mask]
                low_flex_pred = y_pred[low_flex_mask]
                
                low_flex_metrics = {
                    'mse': float(mean_squared_error(low_flex_true, low_flex_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(low_flex_true, low_flex_pred))),
                    'mae': float(mean_absolute_error(low_flex_true, low_flex_pred)),
                    'r2': float(r2_score(low_flex_true, low_flex_pred)),
                    'count': int(np.sum(low_flex_mask)),
                    'threshold': float(low_flex_threshold),
                    'mean_true': float(np.mean(low_flex_true)),
                    'mean_pred': float(np.mean(low_flex_pred))
                }
                flexibility_metrics['low_flexibility'] = low_flex_metrics
            
            # Medium flexibility regions
            med_flex_mask = (y_true > low_flex_threshold) & (y_true <= high_flex_threshold)
            if np.sum(med_flex_mask) >= 10:
                med_flex_true = y_true[med_flex_mask]
                med_flex_pred = y_pred[med_flex_mask]
                
                med_flex_metrics = {
                    'mse': float(mean_squared_error(med_flex_true, med_flex_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(med_flex_true, med_flex_pred))),
                    'mae': float(mean_absolute_error(med_flex_true, med_flex_pred)),
                    'r2': float(r2_score(med_flex_true, med_flex_pred)),
                    'count': int(np.sum(med_flex_mask)),
                    'lower_threshold': float(low_flex_threshold),
                    'upper_threshold': float(high_flex_threshold),
                    'mean_true': float(np.mean(med_flex_true)),
                    'mean_pred': float(np.mean(med_flex_pred))
                }
                flexibility_metrics['medium_flexibility'] = med_flex_metrics
            
            # High flexibility regions
            high_flex_mask = y_true > high_flex_threshold
            if np.sum(high_flex_mask) >= 10:
                high_flex_true = y_true[high_flex_mask]
                high_flex_pred = y_pred[high_flex_mask]
                
                high_flex_metrics = {
                    'mse': float(mean_squared_error(high_flex_true, high_flex_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(high_flex_true, high_flex_pred))),
                    'mae': float(mean_absolute_error(high_flex_true, high_flex_pred)),
                    'r2': float(r2_score(high_flex_true, high_flex_pred)),
                    'count': int(np.sum(high_flex_mask)),
                    'threshold': float(high_flex_threshold),
                    'mean_true': float(np.mean(high_flex_true)),
                    'mean_pred': float(np.mean(high_flex_pred))
                }
                flexibility_metrics['high_flexibility'] = high_flex_metrics
        else:
            # For chunked DataFrame case
            # First pass to calculate percentiles
            y_true_values = []
            
            # Sample up to 100,000 values for percentile calculation
            total_sampled = 0
            for chunk in data_source:
                if total_sampled < 100000:
                    sample_size = min(len(chunk), 100000 - total_sampled)
                    if sample_size > 0:
                        sampled_indices = np.random.choice(len(chunk), sample_size, replace=False)
                        y_true_values.extend(chunk['actual_rmsf'].values[sampled_indices])
                        total_sampled += sample_size
            
            # Calculate percentiles
            y_true_array = np.array(y_true_values)
            low_flex_threshold = np.percentile(y_true_array, 33.3)
            high_flex_threshold = np.percentile(y_true_array, 66.7)
            
            # Clear sampled values to save memory
            del y_true_values, y_true_array
            clear_memory(force_gc=True)
            
            # Reset iterator for next pass
            nonlocal predictions_path
            predictions_df = pd.read_csv(
                predictions_path,
                dtype={
                    'domain_id': 'category',
                    'resid': 'int32',
                    'resname': 'category',
                    'predicted_rmsf': 'float32',
                    'actual_rmsf': 'float32'
                },
                chunksize=100000
            )
            
            # Initialize collectors for each flexibility range
            low_flex_collector = {'true': [], 'pred': [], 'count': 0}
            med_flex_collector = {'true': [], 'pred': [], 'count': 0}
            high_flex_collector = {'true': [], 'pred': [], 'count': 0}
            
            # Second pass to collect data for each flexibility range
            for chunk in predictions_df:
                chunk_y_true = chunk['actual_rmsf'].values
                chunk_y_pred = chunk['predicted_rmsf'].values
                
                # Low flexibility
                low_flex_mask = chunk_y_true <= low_flex_threshold
                if np.sum(low_flex_mask) > 0:
                    low_flex_collector['true'].extend(chunk_y_true[low_flex_mask])
                    low_flex_collector['pred'].extend(chunk_y_pred[low_flex_mask])
                    low_flex_collector['count'] += np.sum(low_flex_mask)
                
                # Medium flexibility
                med_flex_mask = (chunk_y_true > low_flex_threshold) & (chunk_y_true <= high_flex_threshold)
                if np.sum(med_flex_mask) > 0:
                    med_flex_collector['true'].extend(chunk_y_true[med_flex_mask])
                    med_flex_collector['pred'].extend(chunk_y_pred[med_flex_mask])
                    med_flex_collector['count'] += np.sum(med_flex_mask)
                
                # High flexibility
                high_flex_mask = chunk_y_true > high_flex_threshold
                if np.sum(high_flex_mask) > 0:
                    high_flex_collector['true'].extend(chunk_y_true[high_flex_mask])
                    high_flex_collector['pred'].extend(chunk_y_pred[high_flex_mask])
                    high_flex_collector['count'] += np.sum(high_flex_mask)
                
                # Clear chunk from memory
                del chunk
                clear_memory(force_gc=True)
                
                # Sample data if collectors get too large to avoid memory issues
                for collector in [low_flex_collector, med_flex_collector, high_flex_collector]:
                    if len(collector['true']) > 100000:
                        sample_indices = np.random.choice(len(collector['true']), 50000, replace=False)
                        collector['true'] = [collector['true'][i] for i in sample_indices]
                        collector['pred'] = [collector['pred'][i] for i in sample_indices]
                
                # Check memory pressure and take emergency measures if needed
                memory_stats = check_memory_usage()
                if memory_stats['system_percent'] > 85:
                    logger.warning(f"Critical memory pressure during flexibility metrics: {memory_stats['system_percent']}%")
                    logger.warning(f"Stopping flexibility metrics calculation to preserve memory integrity")
                    break
            
            # Calculate metrics for each flexibility range
            # Low flexibility
            if low_flex_collector['count'] >= 10:
                low_flex_true = np.array(low_flex_collector['true'])
                low_flex_pred = np.array(low_flex_collector['pred'])
                
                low_flex_metrics = {
                    'mse': float(mean_squared_error(low_flex_true, low_flex_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(low_flex_true, low_flex_pred))),
                    'mae': float(mean_absolute_error(low_flex_true, low_flex_pred)),
                    'r2': float(r2_score(low_flex_true, low_flex_pred)),
                    'count': int(low_flex_collector['count']),
                    'threshold': float(low_flex_threshold),
                    'mean_true': float(np.mean(low_flex_true)),
                    'mean_pred': float(np.mean(low_flex_pred))
                }
                flexibility_metrics['low_flexibility'] = low_flex_metrics
                
                # Clear collector to save memory
                del low_flex_true, low_flex_pred
                clear_memory(force_gc=True)
            
            # Medium flexibility
            if med_flex_collector['count'] >= 10:
                med_flex_true = np.array(med_flex_collector['true'])
                med_flex_pred = np.array(med_flex_collector['pred'])
                
                med_flex_metrics = {
                    'mse': float(mean_squared_error(med_flex_true, med_flex_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(med_flex_true, med_flex_pred))),
                    'mae': float(mean_absolute_error(med_flex_true, med_flex_pred)),
                    'r2': float(r2_score(med_flex_true, med_flex_pred)),
                    'count': int(med_flex_collector['count']),
                    'lower_threshold': float(low_flex_threshold),
                    'upper_threshold': float(high_flex_threshold),
                    'mean_true': float(np.mean(med_flex_true)),
                    'mean_pred': float(np.mean(med_flex_pred))
                }
                flexibility_metrics['medium_flexibility'] = med_flex_metrics
                
                # Clear collector to save memory
                del med_flex_true, med_flex_pred
                clear_memory(force_gc=True)
            
            # High flexibility
            if high_flex_collector['count'] >= 10:
                high_flex_true = np.array(high_flex_collector['true'])
                high_flex_pred = np.array(high_flex_collector['pred'])
                
                high_flex_metrics = {
                    'mse': float(mean_squared_error(high_flex_true, high_flex_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(high_flex_true, high_flex_pred))),
                    'mae': float(mean_absolute_error(high_flex_true, high_flex_pred)),
                    'r2': float(r2_score(high_flex_true, high_flex_pred)),
                    'count': int(high_flex_collector['count']),
                    'threshold': float(high_flex_threshold),
                    'mean_true': float(np.mean(high_flex_true)),
                    'mean_pred': float(np.mean(high_flex_pred))
                }
                flexibility_metrics['high_flexibility'] = high_flex_metrics
                
                # Clear collector to save memory
                del high_flex_true, high_flex_pred
                clear_memory(force_gc=True)
            
            # Clear all collectors to save memory
            del low_flex_collector, med_flex_collector, high_flex_collector
            clear_memory(force_gc=True)
    
    # Calculate flexibility metrics
    if isinstance(predictions_df, pd.DataFrame) or not isinstance(predictions_df, pd.DataFrame):
        try:
            calculate_flexibility_metrics(predictions_df)
        except Exception as e:
            logger.error(f"Error calculating flexibility metrics: {str(e)}")
    
    # MEMORY OPTIMIZATION: Check memory after flexibility metrics
    memory_stats = check_memory_usage()
    logger.info(f"Memory after flexibility metrics: {memory_stats['system_percent']}%, "
               f"{memory_stats['process_rss_gb']:.2f} GB")
    
    # Log overall metrics
    logger.info(f"Overall Mean Squared Error (MSE): {mse:.6f}")
    logger.info(f"Overall Root Mean Squared Error (RMSE): {rmse:.6f}")
    logger.info(f"Overall Mean Absolute Error (MAE): {mae:.6f}")
    logger.info(f"Overall R²: {r2:.6f}")
    logger.info(f"Overall CV-RMSD: {cv_rmsd:.2f}%")
    logger.info(f"Mean Relative Error: {mean_relative_error:.2f}%")
    
    # Save metrics
    metrics = {
        'overall': {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'cv_rmsd': float(cv_rmsd),
            'mean_relative_error': float(mean_relative_error),
            'median_relative_error': float(median_relative_error),
            'num_samples': int(count) if not isinstance(predictions_df, pd.DataFrame) else int(len(predictions_df)),
            'num_domains': int(len(domain_metrics)) if domain_metrics else None,
        }
    }
    
    # Add mean and median info to overall metrics if possible
    if isinstance(predictions_df, pd.DataFrame):
        metrics['overall'].update({
            'mean_true': float(np.mean(predictions_df['actual_rmsf'])),
            'mean_pred': float(np.mean(predictions_df['predicted_rmsf'])),
            'median_true': float(np.median(predictions_df['actual_rmsf'])),
            'median_pred': float(np.median(predictions_df['predicted_rmsf'])),
            'std_true': float(np.std(predictions_df['actual_rmsf'])),
            'std_pred': float(np.std(predictions_df['predicted_rmsf']))
        })
    
    # Add other metrics
    metrics['domains'] = domain_metrics
    metrics['residue_types'] = residue_type_metrics
    metrics['flexibility_regions'] = flexibility_metrics
    
    # Add model information if available
    try:
        device = torch.device('cpu')  # Use CPU for checkpoint loading to save GPU memory
        checkpoint = torch.load(model_path, map_location=device)
        
        model_info = {
            'architecture': checkpoint.get('config', {}).get('model', {}).get('architecture', 'unknown'),
            'total_domains_used': len(checkpoint.get('processed_domains', [])),
            'learning_rate': checkpoint.get('config', {}).get('training', {}).get('learning_rate', 'unknown'),
            'epochs': checkpoint.get('epoch', 'unknown'),
            'best_val_loss': checkpoint.get('best_val_loss', 'unknown')
        }
        metrics['model_info'] = model_info
        
        del checkpoint
        clear_memory(force_gc=True)
    except Exception as e:
        logger.warning(f"Could not extract model information: {str(e)}")
    
    # Save domain performance rankings
    if domain_metrics:
        try:
            # Rank domains by R² performance
            domain_performances = [(domain, metrics['r2']) for domain, metrics in domain_metrics.items()]
            domain_performances.sort(key=lambda x: x[1], reverse=True)
            
            # Only take top/bottom 10 for memory efficiency
            best_domains = domain_performances[:10]
            worst_domains = domain_performances[-10:]
            
            domain_rankings = {
                'best_domains': {domain: domain_metrics[domain] for domain, _ in best_domains},
                'worst_domains': {domain: domain_metrics[domain] for domain, _ in worst_domains}
            }
            
            metrics['domain_rankings'] = domain_rankings
        except Exception as e:
            logger.warning(f"Error creating domain rankings: {str(e)}")
    
    # Save residue type rankings
    if residue_type_metrics:
        try:
            # Rank residue types by R² performance
            restype_performances = [(restype, metrics['r2']) for restype, metrics in residue_type_metrics.items()]
            restype_performances.sort(key=lambda x: x[1], reverse=True)
            
            # Only take top/bottom 5 for memory efficiency
            best_restypes = restype_performances[:5]
            worst_restypes = restype_performances[-5:]
            
            restype_rankings = {
                'best_residue_types': {restype: residue_type_metrics[restype] for restype, _ in best_restypes},
                'worst_residue_types': {restype: residue_type_metrics[restype] for restype, _ in worst_restypes}
            }
            
            metrics['residue_type_rankings'] = restype_rankings
        except Exception as e:
            logger.warning(f"Error creating residue type rankings: {str(e)}")
    
    # Clear memory before saving
    clear_memory(force_gc=True)
    
    # Save metrics
    metrics_dir = os.path.join(config["output"]["base_dir"], "metrics")
    ensure_dir(metrics_dir)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    metrics_path = os.path.join(metrics_dir, f"metrics_{timestamp}.json")
    
    save_json(metrics, metrics_path)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Generate summary for console output
    logger.info("=" * 40)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 40)
    logger.info(f"Overall R²: {r2:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"CV-RMSD: {cv_rmsd:.2f}%")
    
    if 'domains' in metrics:
        r2_values = [m['r2'] for m in metrics['domains'].values()]
        logger.info(f"Domain R² - Mean: {np.mean(r2_values):.4f}, Median: {np.median(r2_values):.4f}")
        logger.info(f"Domains with R² > 0.5: {sum(1 for r in r2_values if r > 0.5)} out of {len(r2_values)}")
    
    logger.info("=" * 40)
    
    # Final memory check
    memory_stats = check_memory_usage()
    logger.info(f"Final memory: {memory_stats['system_percent']}%, "
               f"{memory_stats['process_rss_gb']:.2f} GB")
    
    return metrics_path