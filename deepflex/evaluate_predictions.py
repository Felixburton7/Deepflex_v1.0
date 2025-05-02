# evaluate_predictions.py

import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import logging
import argparse
from collections import defaultdict
import time
from typing import Optional
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def safe_pearsonr(x, y):
    """Calculates Pearson correlation safely, returning NaN on error or insufficient data."""
    try:
        # Ensure numpy arrays and handle potential all-NaN slices after filtering
        x_np = np.asarray(x).astype(np.float64)
        y_np = np.asarray(y).astype(np.float64)
        valid_mask = ~np.isnan(x_np) & ~np.isnan(y_np)
        x_clean = x_np[valid_mask]
        y_clean = y_np[valid_mask]

        if len(x_clean) < 2:
            return np.nan
        # Check for near-zero variance AFTER cleaning NaNs
        if np.std(x_clean) < 1e-8 or np.std(y_clean) < 1e-8:
            return np.nan

        corr, _ = pearsonr(x_clean, y_clean)
        return corr if not np.isnan(corr) else np.nan
    except (ValueError, FloatingPointError):
        return np.nan
    except Exception as e:
        # Log unexpected errors during correlation calculation
        logger.error(f"Unexpected error during pearsonr calculation: {e}", exc_info=True)
        return np.nan

def evaluate(predictions_csv_path: str, ground_truth_rmsf_npy: str, target_temp: float, output_json_path: Optional[str] = None):
    """
    Evaluates predictions against ground truth for a specific temperature.

    Args:
        predictions_csv_path: Path to the prediction CSV file.
        ground_truth_rmsf_npy: Path to the test_rmsf.npy file.
        target_temp: The specific temperature (float) to evaluate.
        output_json_path: Optional path to save evaluation metrics as JSON.
    """
    logger.info(f"--- Starting Evaluation for Temperature: {target_temp:.1f}K ---")
    logger.info(f"Prediction file: {predictions_csv_path}")
    logger.info(f"Ground truth file: {ground_truth_rmsf_npy}")

    # 1. Load Predictions
    try:
        pred_df = pd.read_csv(predictions_csv_path)
        logger.info(f"Loaded {len(pred_df):,} prediction rows.")
        # The prediction script saves the instance key in the 'domain_id' column
        if 'domain_id' not in pred_df.columns or 'resid' not in pred_df.columns or 'rmsf_pred' not in pred_df.columns:
             raise ValueError("Prediction CSV missing required columns ('domain_id', 'resid', 'rmsf_pred')")
        # Rename for clarity and consistency
        pred_df.rename(columns={'domain_id': 'instance_key'}, inplace=True)
        # Ensure correct types
        pred_df['resid'] = pd.to_numeric(pred_df['resid'], errors='coerce').astype('Int64') # Use Int64 to handle potential NaNs during coerce
        pred_df['rmsf_pred'] = pd.to_numeric(pred_df['rmsf_pred'], errors='coerce')
        pred_df.dropna(subset=['instance_key', 'resid', 'rmsf_pred'], inplace=True)
        if pred_df.empty:
             logger.error("No valid prediction rows after loading and cleaning.")
             return
    except FileNotFoundError:
        logger.error(f"Prediction file not found: {predictions_csv_path}")
        return
    except Exception as e:
        logger.error(f"Error loading prediction file: {e}", exc_info=True)
        return

    # 2. Load Ground Truth
    try:
        gt_rmsf_dict = np.load(ground_truth_rmsf_npy, allow_pickle=True).item()
        logger.info(f"Loaded ground truth RMSF for {len(gt_rmsf_dict):,} instances.")
    except FileNotFoundError:
        logger.error(f"Ground truth RMSF file not found: {ground_truth_rmsf_npy}")
        return
    except Exception as e:
        logger.error(f"Error loading ground truth RMSF file: {e}", exc_info=True)
        return

    # 3. Prepare Ground Truth DataFrame for the Target Temperature
    gt_data = []
    skipped_keys = 0
    processed_keys = 0
    logger.info(f"Filtering ground truth for target temperature {target_temp:.1f}K...")
    for instance_key, rmsf_array in gt_rmsf_dict.items():
        try:
            # Expect key like 'domain_id@temp_float'
            domain_part, temp_part = instance_key.rsplit('@', 1)
            temp_val = float(temp_part)
        except (ValueError, AttributeError):
            logger.warning(f"Skipping malformed instance key in ground truth: {instance_key}")
            skipped_keys += 1
            continue

        # Check if the temperature matches the target
        if abs(temp_val - target_temp) < 1e-6: # Robust float comparison
            processed_keys += 1
            if isinstance(rmsf_array, np.ndarray):
                for i, rmsf_val in enumerate(rmsf_array):
                    if not np.isnan(rmsf_val): # Check for NaN in ground truth RMSF
                        gt_data.append({
                            'instance_key': instance_key,
                            'resid': i + 1, # Generate 1-based residue index
                            'target_rmsf': float(rmsf_val) # Ensure float
                        })
            # else: logger.warning(f"RMSF data for {instance_key} is not a numpy array.")

    if not gt_data:
        logger.error(f"No ground truth data found for target temperature {target_temp:.1f}K. Cannot evaluate.")
        if skipped_keys > 0: logger.warning(f"Skipped {skipped_keys} potentially malformed keys during filtering.")
        logger.info(f"Processed {processed_keys} keys matching temperature {target_temp:.1f}K.")

        return

    gt_df = pd.DataFrame(gt_data)
    # Ensure correct types
    gt_df['resid'] = gt_df['resid'].astype('Int64')
    gt_df['target_rmsf'] = pd.to_numeric(gt_df['target_rmsf'], errors='coerce')
    gt_df.dropna(inplace=True)
    logger.info(f"Created ground truth DataFrame with {len(gt_df):,} rows for temperature {target_temp:.1f}K.")

    # 4. Merge Predictions and Ground Truth
    logger.info("Merging predictions and ground truth...")
    try:
        # Ensure resid types match before merge (already done above)
        eval_df = pd.merge(pred_df[['instance_key', 'resid', 'rmsf_pred']],
                           gt_df[['instance_key', 'resid', 'target_rmsf']],
                           on=['instance_key', 'resid'],
                           how='inner') # Inner join ensures we only evaluate matching residues
        logger.info(f"Merged DataFrame contains {len(eval_df):,} aligned residue predictions.")

        if eval_df.empty:
             logger.error("Merging predictions and ground truth resulted in an empty DataFrame. Check instance keys and residue IDs.")
             return

    except Exception as e:
         logger.error(f"Error merging DataFrames: {e}", exc_info=True)
         return

    # 5. Calculate Overall Metrics
    logger.info("Calculating overall metrics...")
    results = {'evaluation_temperature': target_temp, 'overall': {}, 'per_instance_summary': {}, 'per_instance_detail': {}}
    try:
        if len(eval_df) < 2:
            logger.warning("Fewer than 2 aligned data points. Cannot calculate overall correlation.")
            results['overall']['pearson_correlation'] = None
        else:
            overall_corr = safe_pearsonr(eval_df['target_rmsf'], eval_df['rmsf_pred'])
            results['overall']['pearson_correlation'] = float(overall_corr) if not np.isnan(overall_corr) else None

        overall_mse = mean_squared_error(eval_df['target_rmsf'], eval_df['rmsf_pred'])
        overall_rmse = np.sqrt(overall_mse)
        overall_mae = mean_absolute_error(eval_df['target_rmsf'], eval_df['rmsf_pred'])

        results['overall']['rmse'] = float(overall_rmse)
        results['overall']['mae'] = float(overall_mae)
        results['overall']['mse'] = float(overall_mse)
        results['overall']['num_residues'] = len(eval_df)

        logger.info(f"--- Overall Metrics (Temp: {target_temp:.1f}K) ---")
        logger.info(f"  Pearson Correlation: {results['overall']['pearson_correlation']:.4f}" if results['overall']['pearson_correlation'] is not None else "  Pearson Correlation: N/A (<2 points)")
        logger.info(f"  RMSE: {overall_rmse:.4f}")
        logger.info(f"  MAE: {overall_mae:.4f}")
        logger.info(f"  MSE: {overall_mse:.4f}")
        logger.info(f"  Evaluated on: {len(eval_df):,} residues")

    except Exception as e:
        logger.error(f"Error calculating overall metrics: {e}", exc_info=True)

    # 6. Calculate Per-Instance Metrics
    logger.info("Calculating per-instance metrics...")
    per_instance_metrics_list = []
    instance_groups = eval_df.groupby('instance_key')
    num_instances_eval = 0

    for instance_key, group in tqdm(instance_groups, desc="Per-Instance Metrics", total=len(instance_groups)):
        num_residues = len(group)
        if num_residues < 2: # Need at least 2 points for correlation
            continue

        num_instances_eval += 1
        try:
            corr = safe_pearsonr(group['target_rmsf'], group['rmsf_pred'])
            rmse = np.sqrt(mean_squared_error(group['target_rmsf'], group['rmsf_pred']))
            mae = mean_absolute_error(group['target_rmsf'], group['rmsf_pred'])

            instance_metrics = {
                'correlation': float(corr) if not np.isnan(corr) else None,
                'rmse': float(rmse),
                'mae': float(mae),
                'num_residues': num_residues
            }
            per_instance_metrics_list.append(instance_metrics)
            results['per_instance_detail'][instance_key] = instance_metrics

        except Exception as e:
            logger.warning(f"Error calculating metrics for instance {instance_key}: {e}")

    # Summarize Per-Instance Metrics
    if per_instance_metrics_list:
        per_instance_df = pd.DataFrame(per_instance_metrics_list)
        # Calculate summary stats, ignoring NaNs for correlation
        summary = {
            'mean_correlation': float(per_instance_df['correlation'].mean(skipna=True)),
            'median_correlation': float(per_instance_df['correlation'].median(skipna=True)),
            'std_dev_correlation': float(per_instance_df['correlation'].std(skipna=True)),
            'mean_rmse': float(per_instance_df['rmse'].mean()),
            'median_rmse': float(per_instance_df['rmse'].median()),
            'mean_mae': float(per_instance_df['mae'].mean()),
            'median_mae': float(per_instance_df['mae'].median()),
            'num_instances': num_instances_eval
        }
        results['per_instance_summary'] = summary

        logger.info("\n--- Per-Protein Instance Metrics Summary ---")
        logger.info(f"  Instances Evaluated: {num_instances_eval}")
        logger.info(f"  Mean Correlation: {summary['mean_correlation']:.4f} +/- {summary['std_dev_correlation']:.4f}")
        logger.info(f"  Median Correlation: {summary['median_correlation']:.4f}")
        logger.info(f"  Mean RMSE: {summary['mean_rmse']:.4f}")
        logger.info(f"  Median RMSE: {summary['median_rmse']:.4f}")
        logger.info(f"  Mean MAE: {summary['mean_mae']:.4f}")
        logger.info(f"  Median MAE: {summary['median_mae']:.4f}")
    else:
        logger.warning("No instances had enough data points (>=2) for per-instance metric calculation.")
        results['per_instance_summary'] = {'num_instances': 0}


    # 7. Save Results to JSON if path provided
    if output_json_path:
        logger.info(f"Saving evaluation results to: {output_json_path}")
        try:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            # Use a custom encoder for potential numpy types if any sneak through
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer): return int(obj)
                    if isinstance(obj, np.floating): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    return super(NpEncoder, self).default(obj)
            with open(output_json_path, 'w') as f:
                json.dump(results, f, indent=4, cls=NpEncoder)
            logger.info("Results saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save results JSON: {e}", exc_info=True)

    logger.info(f"--- Evaluation Complete for Temperature: {target_temp:.1f}K ---")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate RMSF predictions against ground truth.')
    parser.add_argument('--predictions_csv', type=str, required=True,
                        help='Path to the prediction CSV file generated by predict.py (e.g., predictions/test_set_results/320K/predictions_320K.csv)')
    parser.add_argument('--ground_truth_npy', type=str, required=True,
                        help='Path to the ground truth test set RMSF .npy file (e.g., data/processed/test_rmsf.npy)')
    parser.add_argument('--temperature', type=float, required=True,
                        help='The specific temperature (in Kelvin) for which the predictions were made and should be evaluated.')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Optional: Path to save the evaluation metrics as a JSON file.')

    args = parser.parse_args()

    # Basic validation
    if not os.path.exists(args.predictions_csv):
        logger.error(f"Prediction file not found: {args.predictions_csv}")
        sys.exit(1)
    if not os.path.exists(args.ground_truth_npy):
        logger.error(f"Ground truth file not found: {args.ground_truth_npy}")
        sys.exit(1)

    evaluate(
        predictions_csv_path=args.predictions_csv,
        ground_truth_rmsf_npy=args.ground_truth_npy,
        target_temp=args.temperature,
        output_json_path=args.output_json
    )