# src/voxelflex/cli/commands/visualize.py
"""
Visualization command for VoxelFlex (Temperature-Aware).

Generates plots for model performance analysis, optionally saving plot data.
"""

import os
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union # Add Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
# Ensure 'viridis' is imported correctly if used directly
from matplotlib.cm import ScalarMappable, viridis #, viridis_r if needed
from scipy.stats import gaussian_kde, pearsonr

# Use centralized logger
logger = logging.getLogger("voxelflex.cli.visualize")

# Project imports
from voxelflex.utils.logging_utils import get_logger, log_stage, log_section_header # Add log_section_header
from voxelflex.utils.file_utils import ensure_dir, load_json, save_json, resolve_path
from voxelflex.data.data_loader import load_aggregated_rmsf_data # Needed for merging GT
# Import the metric calculation from evaluate for consistency
from voxelflex.cli.commands.evaluate import calculate_metrics

# --- Plotting Functions ---

def _save_plot_and_data(
    fig: Figure,
    plot_df: Optional[pd.DataFrame],
    base_filename: str,
    output_dir: str,
    save_format: str,
    dpi: int,
    save_data: bool
) -> Optional[str]:
    """Helper function to save plot image and optionally its data."""
    ensure_dir(output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"{base_filename}_{timestamp}.{save_format}")
    csv_path = os.path.join(output_dir, f"{base_filename}_{timestamp}_data.csv")

    plot_saved = False
    try:
        fig.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved plot: {plot_path}")
        plot_saved = True
    except Exception as e:
        logger.error(f"Failed to save plot {plot_path}: {e}")
    finally:
        plt.close(fig) # Close figure to free memory, regardless of save success

    if save_data and plot_df is not None and not plot_df.empty:
        try:
            plot_df.to_csv(csv_path, index=False, float_format='%.6f')
            logger.info(f"Saved plot data: {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save plot data {csv_path}: {e}")

    return plot_path if plot_saved else None

def create_metric_curve(
    history: Dict[str, List[float]],
    metric_key: str, # e.g., 'loss' or 'pearson'
    output_dir: str,
    save_format: str = 'png',
    dpi: int = 150,
    save_data: bool = True
) -> Optional[str]:
    """Creates a plot of training and validation metrics over epochs."""
    train_key = f'train_{metric_key}'
    val_key = f'val_{metric_key}'
    lr_key = 'lr'

    if train_key not in history or val_key not in history or not history[train_key] or not history[val_key]:
        logger.warning(f"History data missing for '{metric_key}'. Skipping {metric_key} curve plot.")
        return None

    logger.info(f"Creating {metric_key} curve plot...")
    train_values = history[train_key]
    val_values = history[val_key]
    epochs = range(1, len(train_values) + 1)

    plot_df_dict: Dict[str, List[Union[int, float]]] = { # Use Union for types
        'epoch': list(epochs),
        train_key: train_values,
        val_key: val_values
    }
    # Handle potential length mismatch for LR if resuming incomplete epoch
    if lr_key in history and len(history[lr_key]) >= len(epochs):
        plot_df_dict[lr_key] = history[lr_key][:len(epochs)]
    elif lr_key in history:
         logger.warning(f"Length mismatch for '{lr_key}' in history. Cannot plot LR.")

    plot_df = pd.DataFrame(plot_df_dict)


    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_values, 'o-', color='royalblue', label=f'Training {metric_key.capitalize()}', markersize=4, alpha=0.8)
    ax.plot(epochs, val_values, 's-', color='orangered', label=f'Validation {metric_key.capitalize()}', markersize=4, alpha=0.8)

    # Determine best epoch based on metric (loss=min, corr=max)
    try:
        is_loss = 'loss' in metric_key.lower()
        valid_val_values = [v for v in val_values if pd.notna(v)] # Filter NaNs for argmin/argmax
        if not valid_val_values: raise ValueError("No valid validation values found.")

        if is_loss:
            best_val_epoch_idx = np.nanargmin(val_values) # Use nanargmin
        else:
            best_val_epoch_idx = np.nanargmax(val_values) # Use nanargmax

        best_val_epoch = best_val_epoch_idx + 1
        best_val_value = val_values[best_val_epoch_idx]
        ax.axvline(best_val_epoch, linestyle='--', color='gray', alpha=0.7, label=f'Best Val @ Ep {best_val_epoch} ({best_val_value:.4f})')
    except (ValueError, IndexError) as e:
         logger.warning(f"Could not determine best epoch for {metric_key} curve: {e}")


    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric_key.capitalize(), fontsize=12)
    ax.set_title(f'Training and Validation {metric_key.capitalize()}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Optional: Add Learning Rate on secondary axis if plotting correlation
    if not is_loss and lr_key in plot_df.columns:
        ax2 = ax.twinx()
        ax2.plot(epochs, plot_df[lr_key], 'd--', color='green', label='Learning Rate', markersize=3, alpha=0.5)
        ax2.set_ylabel('Learning Rate', color='green', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='green', labelsize=9)
        # Use log scale if LR varies significantly and has no zeros/negatives
        lr_vals = plot_df[lr_key].dropna()
        if len(lr_vals.unique()) > 2 and (lr_vals > 0).all():
            try:
                 ax2.set_yscale('log')
            except ValueError as e: # Catch potential issues with non-positive values if check failed
                 logger.warning(f"Could not set log scale for LR axis: {e}")
        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='best', fontsize=9)


    fig.tight_layout()
    base_filename = f"{metric_key}_curve"
    # Pass the DataFrame used for plotting
    return _save_plot_and_data(fig, plot_df, base_filename, output_dir, save_format, dpi, save_data)

# Alias specific curve functions
def create_loss_curve(history, output_dir, save_format='png', dpi=150, save_data=True):
    return create_metric_curve(history, 'loss', output_dir, save_format, dpi, save_data)

def create_correlation_curve(history, output_dir, save_format='png', dpi=150, save_data=True):
    return create_metric_curve(history, 'pearson', output_dir, save_format, dpi, save_data)

def create_prediction_scatter(
    eval_df: pd.DataFrame,
    output_dir: str,
    save_format: str = 'png',
    dpi: int = 150,
    max_points: int = 1000,
    save_data: bool = True
) -> Optional[str]:
    """Creates Predicted vs. Actual RMSF scatter plot."""
    if eval_df.empty or 'target_rmsf' not in eval_df or 'predicted_rmsf' not in eval_df:
        logger.warning("Cannot create prediction scatter: DataFrame empty or missing columns.")
        return None

    logger.info("Creating prediction scatter plot...")
    # Use calculate_metrics for consistency
    metrics = calculate_metrics(eval_df, label="Scatter")

    y_true = eval_df['target_rmsf'].values
    y_pred = eval_df['predicted_rmsf'].values

    # Filter only finite values for plotting and metrics text
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_plot = y_true[valid_mask]
    y_pred_plot = y_pred[valid_mask]
    plot_df_full = eval_df[valid_mask].copy() # Dataframe with only valid points

    if len(y_true_plot) == 0:
        logger.warning("No finite data points for prediction scatter plot.")
        return None

    # Sample points *after* filtering NaNs if needed
    if len(y_true_plot) > max_points:
        logger.debug(f"Sampling {max_points} points for scatter plot from {len(y_true_plot)} valid points.")
        indices = np.random.choice(len(y_true_plot), max_points, replace=False)
        y_true_sampled, y_pred_sampled = y_true_plot[indices], y_pred_plot[indices]
        # Data to save should be the sampled points if sampling occurred
        plot_df_to_save = plot_df_full.iloc[indices][['target_rmsf', 'predicted_rmsf']].copy()
    else:
        y_true_sampled, y_pred_sampled = y_true_plot, y_pred_plot
        plot_df_to_save = plot_df_full[['target_rmsf', 'predicted_rmsf']].copy() # Save all valid points

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true_sampled, y_pred_sampled, c='steelblue', s=20, alpha=0.5, edgecolors='none', label='Predictions')

    # Add identity line
    # Calculate limits based on the *sampled* data for the plot view
    min_val = min(y_true_sampled.min(), y_pred_sampled.min()) - 0.1 * abs(min(y_true_sampled.min(), y_pred_sampled.min())) # Add padding
    max_val = max(y_true_sampled.max(), y_pred_sampled.max()) + 0.1 * abs(max(y_true_sampled.max(), y_pred_sampled.max()))
    lims = [min_val, max_val]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label="y=x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')

    # Use metrics calculated on the *full* valid dataset for annotation
    metrics_text = (
        f"Pearson: {metrics.get('pearson', np.nan):.3f}\n"
        f"R²: {metrics.get('r2', np.nan):.3f}\n"
        f"RMSE: {metrics.get('rmse', np.nan):.3f}\n"
        f"MAE: {metrics.get('mae', np.nan):.3f}\n"
        f"N: {metrics.get('count', 0):,}" # Count is from full valid dataset
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    ax.text(0.03, 0.97, metrics_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    ax.set_title('Predicted vs. Actual RMSF', fontsize=14, fontweight='bold')
    ax.set_xlabel('Actual RMSF', fontsize=12)
    ax.set_ylabel('Predicted RMSF', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)

    fig.tight_layout()
    base_filename = "prediction_scatter"
    # Save the potentially sampled data
    return _save_plot_and_data(fig, plot_df_to_save, base_filename, output_dir, save_format, dpi, save_data)


def create_predicted_vs_validation_scatter_density(
    eval_df: pd.DataFrame,
    output_dir: str,
    save_format: str = 'png',
    dpi: int = 150,
    save_data: bool = True # Save the underlying x, y data
) -> Optional[str]:
    """Creates Predicted vs. Actual RMSF scatter plot with density coloring."""
    if eval_df.empty or 'target_rmsf' not in eval_df or 'predicted_rmsf' not in eval_df:
        logger.warning("Cannot create density scatter: DataFrame empty or missing columns.")
        return None

    logger.info("Creating prediction density scatter plot...")
    y_true = eval_df['target_rmsf'].values
    y_pred = eval_df['predicted_rmsf'].values

    # Filter out NaNs/Infs before KDE
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    plot_df = eval_df[valid_mask][['target_rmsf', 'predicted_rmsf']].copy() # DataFrame for saving

    if len(y_true_clean) < 5: # Need points for KDE
        logger.warning("Too few valid points (<5) for density estimation. Skipping density scatter.")
        return None

    # Calculate metrics on valid points only
    metrics = calculate_metrics(plot_df, label="Density Scatter")

    fig, ax = plt.subplots(figsize=(8, 8))

    # Calculate the point density
    try:
        xy = np.vstack([y_true_clean, y_pred_clean])
        # Handle potential singular matrix in KDE
        try:
             z = gaussian_kde(xy)(xy)
        except np.linalg.LinAlgError:
             logger.warning("Singular matrix in KDE calculation. Adding small jitter.")
             jitter_scale = 1e-6 * (np.max(xy, axis=1) - np.min(xy, axis=1))
             jitter = jitter_scale[:, np.newaxis] * np.random.randn(*xy.shape)
             z = gaussian_kde(xy + jitter)(xy + jitter)

        # Sort points by density, so dense points are plotted last
        idx = z.argsort()
        x_plot, y_plot, z_plot = y_true_clean[idx], y_pred_clean[idx], z[idx]

        scatter = ax.scatter(x_plot, y_plot, c=z_plot, s=10, cmap=viridis, alpha=0.7, edgecolors='none')
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
        cbar.set_label('Point Density', fontsize=10)
        cbar.ax.tick_params(labelsize=8)
    except Exception as kde_e:
         logger.warning(f"KDE calculation failed ({kde_e}). Falling back to simple scatter.")
         ax.scatter(y_true_clean, y_pred_clean, c='steelblue', s=10, alpha=0.5, edgecolors='none')


    # Add identity line
    min_val = min(y_true_clean.min(), y_pred_clean.min()) - 0.1 * abs(min(y_true_clean.min(), y_pred_clean.min()))
    max_val = max(y_true_clean.max(), y_pred_clean.max()) + 0.1 * abs(max(y_true_clean.max(), y_pred_clean.max()))
    lims = [min_val, max_val]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=1, label="y=x") # Ensure line is visible
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')

    metrics_text = (
        f"Pearson: {metrics.get('pearson', np.nan):.3f}\n"
        f"R²: {metrics.get('r2', np.nan):.3f}\n"
        f"RMSE: {metrics.get('rmse', np.nan):.3f}\n"
        f"MAE: {metrics.get('mae', np.nan):.3f}\n"
        f"N: {metrics.get('count', 0):,}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    ax.text(0.03, 0.97, metrics_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    ax.set_title('Predicted vs. Actual RMSF (Density Scatter)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Actual RMSF', fontsize=12)
    ax.set_ylabel('Predicted RMSF', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)

    fig.tight_layout()
    base_filename = "prediction_density_scatter"
    return _save_plot_and_data(fig, plot_df, base_filename, output_dir, save_format, dpi, save_data)


def create_error_distribution(
    eval_df: pd.DataFrame,
    output_dir: str,
    save_format: str = 'png',
    dpi: int = 150,
    save_data: bool = True
) -> Optional[str]:
    """Creates histogram of prediction errors (Predicted - Actual)."""
    if eval_df.empty or 'target_rmsf' not in eval_df or 'predicted_rmsf' not in eval_df:
        logger.warning("Cannot create error distribution: DataFrame empty or missing columns.")
        return None

    logger.info("Creating error distribution plot...")
    eval_df_copy = eval_df.copy() # Work on a copy
    eval_df_copy['error'] = eval_df_copy['predicted_rmsf'] - eval_df_copy['target_rmsf']
    errors = eval_df_copy['error'].dropna()
    if errors.empty:
        logger.warning("No valid error values found for distribution plot.")
        return None

    plot_df = pd.DataFrame({'error': errors}) # Data for saving

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(errors, kde=True, bins=50, ax=ax, color='coral', edgecolor='black', alpha=0.7, stat="density")

    mean_error = errors.mean()
    median_error = errors.median()
    std_error = errors.std()

    ax.axvline(mean_error, color='k', linestyle='--', linewidth=1.5, label=f'Mean: {mean_error:.3f}')
    ax.axvline(median_error, color='k', linestyle=':', linewidth=1.5, label=f'Median: {median_error:.3f}')
    # ax.axvspan(mean_error - std_error, mean_error + std_error, alpha=0.15, color='gray', label=f'StdDev: {std_error:.3f}')

    ax.set_title('Distribution of Prediction Errors (Predicted - Actual)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Prediction Error', fontsize=12)
    ax.set_ylabel('Density', fontsize=12) # Use Density since kde=True
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=10)

    fig.tight_layout()
    base_filename = "error_distribution"
    return _save_plot_and_data(fig, plot_df, base_filename, output_dir, save_format, dpi, save_data)


def create_stratified_error_boxplot(
    eval_df: pd.DataFrame,
    stratify_by_col: str,
    plot_title: str,
    x_label: str,
    output_dir: str,
    base_filename: str,
    save_format: str = 'png',
    dpi: int = 150,
    save_data: bool = True,
    min_samples_per_group: int = 10
) -> Optional[str]:
    """Creates a boxplot of absolute errors stratified by a given column."""
    if eval_df.empty or stratify_by_col not in eval_df.columns:
        logger.warning(f"Cannot create stratified boxplot: DataFrame empty or column '{stratify_by_col}' missing.")
        return None

    logger.info(f"Creating stratified error boxplot by '{stratify_by_col}'...")
    eval_df_copy = eval_df.copy()
    if 'error' not in eval_df_copy.columns:
         eval_df_copy['error'] = eval_df_copy['predicted_rmsf'] - eval_df_copy['target_rmsf']
    eval_df_copy['abs_error'] = np.abs(eval_df_copy['error'])

    # Ensure stratification column is suitable type (e.g., string or category) and handle NaNs
    eval_df_copy = eval_df_copy.dropna(subset=[stratify_by_col, 'abs_error'])
    eval_df_copy[stratify_by_col] = eval_df_copy[stratify_by_col].astype(str) # Convert to string for consistent grouping

    # Filter groups with enough samples
    group_counts = eval_df_copy[stratify_by_col].value_counts()
    valid_groups = group_counts[group_counts >= min_samples_per_group].index.tolist()
    plot_df = eval_df_copy[eval_df_copy[stratify_by_col].isin(valid_groups)].copy()

    if plot_df.empty:
        logger.warning(f"No groups in '{stratify_by_col}' met the minimum sample requirement ({min_samples_per_group}). Skipping boxplot.")
        return None

    # Sort categories for consistent plotting
    # Try numeric sort first if they look like numbers/bins, else alphabetical
    try:
        # Attempt to extract leading number for sorting (e.g., for SASA bins)
        categories_sorted = sorted(valid_groups, key=lambda x: float(x.split('-')[0]))
    except (ValueError, IndexError):
        # Fallback to alphabetical sort if numeric extraction fails
        categories_sorted = sorted(valid_groups)


    fig, ax = plt.subplots(figsize=(max(8, len(categories_sorted)*0.5), 6)) # Adjust width based on # categories
    sns.boxplot(x=stratify_by_col, y='abs_error', data=plot_df, ax=ax, order=categories_sorted,
                palette='viridis', fliersize=2, linewidth=1.0, showfliers=False) # Hide outliers for clarity?

    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Absolute Prediction Error', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=10, ha='right')
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, linestyle=':', alpha=0.6, axis='y')

    # Add counts below boxes
    y_min, y_max = ax.get_ylim() # Get current y-limits AFTER plotting boxes
    # y_range = y_max - y_min if y_max > y_min else 1.0 # Avoid zero range
    # Place text slightly below the minimum y-axis value shown
    text_y_pos = y_min - (y_max - y_min) * 0.05 # Position below plot area

    for i, cat in enumerate(categories_sorted):
        count = group_counts[cat]
        ax.text(i, text_y_pos, f"n={count}", ha='center', va='top', fontsize=8, color='gray')

    # Adjust y-limit slightly to make space for text
    ax.set_ylim(bottom=text_y_pos - (y_max - y_min) * 0.02)

    fig.tight_layout()
    # Data to save includes the stratification column and the absolute error
    plot_data_to_save = plot_df[[stratify_by_col, 'abs_error']].copy()
    return _save_plot_and_data(fig, plot_data_to_save, base_filename, output_dir, save_format, dpi, save_data)


# Specific wrappers for stratified plots
def create_residue_type_analysis(eval_df, output_dir, save_format='png', dpi=150, save_data=True):
    return create_stratified_error_boxplot(
        eval_df, 'resname', 'Absolute Error by Residue Type', 'Residue Type',
        output_dir, 'residue_type_error_boxplot', save_format, dpi, save_data
    )

def create_sasa_error_analysis(eval_df, output_dir, sasa_bins, save_format='png', dpi=150, save_data=True):
    sasa_col = 'relative_accessibility'
    if sasa_col not in eval_df.columns:
        logger.warning(f"SASA column '{sasa_col}' not found. Skipping SASA error analysis.")
        return None
    eval_df_copy = eval_df.copy()
    # Ensure bin column exists
    if 'sasa_bin' not in eval_df_copy.columns:
        bin_labels = [f"{sasa_bins[i]:.1f}-{sasa_bins[i+1]:.1f}" for i in range(len(sasa_bins)-1)]
        try:
            # Ensure numeric conversion before cutting
            eval_df_copy[sasa_col] = pd.to_numeric(eval_df_copy[sasa_col], errors='coerce')
            eval_df_copy['sasa_bin'] = pd.cut(eval_df_copy[sasa_col], bins=sasa_bins, labels=bin_labels, right=False, include_lowest=True)
        except Exception as e:
             logger.error(f"Failed to create SASA bins for plotting: {e}")
             return None
    # Pass the dataframe with the 'sasa_bin' column
    return create_stratified_error_boxplot(
        eval_df_copy, 'sasa_bin', 'Absolute Error by Relative Solvent Accessibility', 'SASA Bin',
        output_dir, 'sasa_error_boxplot', save_format, dpi, save_data
    )

def create_ss_error_analysis(eval_df, output_dir, save_format='png', dpi=150, save_data=True):
    ss_col = next((c for c in ['dssp', 'secondary_structure_encoded'] if c in eval_df.columns), None)
    if ss_col is None:
        logger.warning("No secondary structure column found. Skipping SS error analysis.")
        return None
    return create_stratified_error_boxplot(
        eval_df, ss_col, 'Absolute Error by Secondary Structure', f'SS Type ({ss_col})',
        output_dir, 'ss_error_boxplot', save_format, dpi, save_data
    )


def create_amino_acid_performance(
    eval_df: pd.DataFrame,
    output_dir: str,
    save_format: str = 'png',
    dpi: int = 150,
    save_data: bool = True
) -> Optional[str]:
    """Creates bar plots of various metrics grouped by amino acid type."""
    if eval_df.empty or 'resname' not in eval_df.columns:
        logger.warning("Cannot create AA performance plot: DataFrame empty or 'resname' missing.")
        return None

    logger.info("Creating amino acid performance plots...")
    aa_metrics_list = []
    # Ensure resname is string and handle potential NaNs dropped by earlier steps
    eval_df_copy = eval_df.dropna(subset=['resname']).copy()
    eval_df_copy['resname'] = eval_df_copy['resname'].astype(str)

    grouped = eval_df_copy.groupby('resname')
    # Define standard AA order if possible, otherwise sort alphabetically
    aa_order = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    resnames_present = eval_df_copy['resname'].unique()
    resnames_sorted = [aa for aa in aa_order if aa in resnames_present] + sorted([aa for aa in resnames_present if aa not in aa_order])

    for resname in resnames_sorted:
        group = grouped.get_group(resname)
        if len(group) >= 2: # Need min 2 points for most metrics
             metrics = calculate_metrics(group, label=f"AA={resname}")
             # Only add if key metrics were calculable
             if pd.notna(metrics.get('pearson')) or pd.notna(metrics.get('rmse')):
                 metrics['resname'] = resname
                 aa_metrics_list.append(metrics)
        # else: logger.debug(f"Skipping AA '{resname}' due to insufficient samples ({len(group)})")

    if not aa_metrics_list:
         logger.warning("No amino acid groups had sufficient valid samples for performance plotting.")
         return None

    metrics_df = pd.DataFrame(aa_metrics_list)
    plot_df_to_save = metrics_df.copy() # Data to save

    # Plotting setup
    metrics_to_plot = ['pearson', 'rmse', 'mae', 'count']
    titles = ['Pearson Correlation', 'RMSE', 'MAE', 'Sample Count']
    palettes = ['coolwarm', 'viridis_r', 'magma_r', 'crest']
    num_metrics = len(metrics_to_plot)
    ncols = 2
    nrows = (num_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), sharex=True) # Share x-axis
    axes = axes.flatten()

    for i, metric in enumerate(metrics_to_plot):
        if metric not in metrics_df.columns:
            logger.warning(f"Metric '{metric}' not found in calculated AA metrics. Skipping its plot.")
            # Disable the unused subplot
            if i < len(axes): axes[i].set_visible(False)
            continue

        # Filter out NaN values for the specific metric before plotting
        plot_data = metrics_df.dropna(subset=[metric])
        if plot_data.empty:
            logger.warning(f"No valid data for metric '{metric}' after dropping NaNs. Skipping plot.")
            if i < len(axes): axes[i].set_visible(False)
            continue

        sns.barplot(x='resname', y=metric, data=plot_data, ax=axes[i], palette=palettes[i % len(palettes)], order=resnames_sorted)
        axes[i].set_title(titles[i], fontsize=12, fontweight='bold')
        axes[i].set_xlabel(None) # Remove redundant x-label from upper plots if sharing x
        axes[i].set_ylabel(metric.upper() if metric != 'count' else 'Count', fontsize=10)
        axes[i].tick_params(axis='x', rotation=45, labelsize=9, ha='right')
        axes[i].tick_params(axis='y', labelsize=9)
        axes[i].grid(True, linestyle=':', alpha=0.6, axis='y')

    # Set x-label only on the bottom plots
    for i in range(ncols * (nrows - 1), len(axes)):
        if axes[i].get_visible(): # Only if plot was actually drawn
             axes[i].set_xlabel("Residue Type", fontsize=11)

    # Remove any unused subplots if num_metrics < nrows*ncols
    for i in range(num_metrics, len(axes)):
        axes[i].set_visible(False)


    fig.suptitle("Performance Metrics by Amino Acid Type", fontsize=16, fontweight='bold')#, y=1.02)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap

    base_filename = "amino_acid_performance"
    return _save_plot_and_data(fig, plot_df_to_save, base_filename, output_dir, save_format, dpi, save_data)


# --- Main Visualization Function ---

def create_visualizations(
    config: Dict[str, Any],
    predictions_path: str,
    history_path: Optional[str] = None # Allow passing history file path
) -> List[str]:
    """
    Generate and save performance visualizations based on config settings.

    Args:
        config: Configuration dictionary.
        predictions_path: Path to the prediction CSV file.
        history_path: Optional path to the training history JSON file.

    Returns:
        List of paths to the generated plot files.
    """
    log_section_header(logger, "GENERATING VISUALIZATIONS")
    predictions_path = resolve_path(predictions_path)
    if not os.path.exists(predictions_path):
        logger.error(f"Predictions file not found: {predictions_path}")
        return []
    history_path = resolve_path(history_path) if history_path else None
    viz_output_dir = config["output"]["visualizations_dir"]
    ensure_dir(viz_output_dir) # Ensure output directory exists

    viz_config = config.get("visualization", {})
    save_format = viz_config.get("save_format", "png")
    dpi = viz_config.get("dpi", 150)
    save_plot_data = viz_config.get("save_plot_data", True)
    max_scatter = viz_config.get("max_scatter_points", 1000)
    sasa_bins = config.get('evaluation', {}).get('sasa_bins', [0.0, 0.1, 0.4, 1.01]) # Get bins from eval config

    generated_plots: List[str] = []
    eval_df: Optional[pd.DataFrame] = None # Initialize

    # --- Load Data ---
    try:
        with log_stage("VIS_SETUP", "Loading prediction and ground truth data"):
            logger.info(f"Loading predictions from: {predictions_path}")
            preds_df = pd.read_csv(predictions_path, dtype={'domain_id': str, 'resname': str})
            preds_df['resid'] = pd.to_numeric(preds_df['resid'], errors='coerce').astype('Int64')
            preds_df['predicted_rmsf'] = pd.to_numeric(preds_df['predicted_rmsf'], errors='coerce')
            preds_df['prediction_temperature'] = pd.to_numeric(preds_df['prediction_temperature'], errors='coerce')
            preds_df.dropna(subset=['domain_id', 'resid', 'predicted_rmsf', 'prediction_temperature'], inplace=True)

            if preds_df.empty:
                 logger.error("Predictions file is empty or contains no valid rows. Cannot generate comparison visualizations.")
                 return [] # Stop if no predictions

            # Load ground truth for comparison plots
            gt_file_path = config['input'].get('aggregated_rmsf_file')
            if not gt_file_path:
                logger.warning("Aggregated RMSF file path not specified in config. Cannot generate comparison plots.")
                eval_df = preds_df.copy() # Can only plot prediction distribution
            else:
                logger.info(f"Loading ground truth from: {gt_file_path}")
                gt_df_raw = load_aggregated_rmsf_data(gt_file_path)
                # Select necessary columns, including optional ones
                cols_to_keep = ['domain_id', 'resid', 'resname', 'target_rmsf', 'temperature_feature']
                optional_cols = ['relative_accessibility', 'dssp', 'secondary_structure_encoded']
                available_optional = [col for col in optional_cols if col in gt_df_raw.columns]
                cols_to_keep.extend(available_optional)
                gt_df = gt_df_raw[cols_to_keep].copy()
                gt_df['resid'] = pd.to_numeric(gt_df['resid'], errors='coerce').astype('Int64')
                gt_df['temperature_feature'] = pd.to_numeric(gt_df['temperature_feature'], errors='coerce')
                gt_df.dropna(subset=['domain_id', 'resid', 'target_rmsf', 'temperature_feature'], inplace=True)

                # Merge predictions and ground truth based on prediction temperature
                pred_temp = preds_df['prediction_temperature'].iloc[0]
                gt_df_filtered = gt_df[np.isclose(gt_df['temperature_feature'], pred_temp)].copy()
                logger.info(f"Merging predictions with {len(gt_df_filtered)} ground truth entries for temp ~{pred_temp:.1f}K...")

                preds_df['resid'] = preds_df['resid'].astype(int) # Ensure int for merge
                gt_df_filtered['resid'] = gt_df_filtered['resid'].astype(int)
                eval_df = pd.merge(preds_df, gt_df_filtered.drop(columns=['temperature_feature']), on=['domain_id', 'resid'], how='inner', suffixes=('_pred', '_gt'))

                # Handle resname column merge issues
                if 'resname_pred' in eval_df.columns and 'resname_gt' in eval_df.columns:
                    eval_df['resname'] = eval_df['resname_gt']
                    eval_df.drop(columns=['resname_pred', 'resname_gt'], inplace=True)
                elif 'resname_gt' in eval_df.columns: eval_df.rename(columns={'resname_gt': 'resname'}, inplace=True)
                elif 'resname_pred' in eval_df.columns: eval_df.rename(columns={'resname_pred': 'resname'}, inplace=True)
                elif 'resname' not in eval_df.columns: eval_df['resname'] = 'UNK'

                eval_df.dropna(subset=['predicted_rmsf', 'target_rmsf'], inplace=True)
                logger.info(f"Merged data for plots contains {len(eval_df)} entries.")

                if eval_df.empty:
                    logger.error("No overlapping data found between predictions and ground truth. Cannot generate comparison plots.")
                    eval_df = None # Reset eval_df if merge failed

    except Exception as e:
        logger.exception(f"Error loading data for visualization: {e}")
        return [] # Cannot proceed without data

    # --- Load History ---
    train_history = None
    if history_path and os.path.exists(history_path):
        logger.info(f"Loading training history from: {history_path}")
        try:
             train_history = load_json(history_path)
        except Exception as e:
             logger.error(f"Failed to load history file {history_path}: {e}")
    elif viz_config.get("plot_loss") or viz_config.get("plot_correlation"):
        logger.warning("Training history file not found or specified. Skipping metric curve plots.")


    # --- Generate Plots ---
    with log_stage("VISUALIZATION", "Creating plots"):
        # Training Curves (require history)
        if train_history:
            if viz_config.get("plot_loss", False):
                 path = create_loss_curve(train_history, viz_output_dir, save_format, dpi, save_plot_data)
                 if path: generated_plots.append(path)
            if viz_config.get("plot_correlation", False):
                 if 'train_pearson' in train_history and 'val_pearson' in train_history:
                      path = create_correlation_curve(train_history, viz_output_dir, save_format, dpi, save_plot_data)
                      if path: generated_plots.append(path)
                 else: logger.warning("Pearson correlation data not found in history. Skipping correlation curve.")
        else:
             if viz_config.get("plot_loss") or viz_config.get("plot_correlation"):
                 logger.info("Skipping loss/correlation curves as history file was not provided or loaded.")

        # Comparison Plots (require merged eval_df)
        if eval_df is not None and not eval_df.empty:
            if viz_config.get("plot_predictions", False):
                path = create_prediction_scatter(eval_df, viz_output_dir, save_format, dpi, max_scatter, save_plot_data)
                if path: generated_plots.append(path)
            if viz_config.get("plot_density_scatter", False):
                 path = create_predicted_vs_validation_scatter_density(eval_df, viz_output_dir, save_format, dpi, save_plot_data)
                 if path: generated_plots.append(path)
            if viz_config.get("plot_error_distribution", False):
                path = create_error_distribution(eval_df, viz_output_dir, save_format, dpi, save_plot_data)
                if path: generated_plots.append(path)
            if viz_config.get("plot_residue_type_analysis", False):
                path = create_residue_type_analysis(eval_df, viz_output_dir, save_format, dpi, save_plot_data)
                if path: generated_plots.append(path)
            if viz_config.get("plot_sasa_error_analysis", False):
                 path = create_sasa_error_analysis(eval_df, viz_output_dir, sasa_bins, save_format, dpi, save_plot_data)
                 if path: generated_plots.append(path)
            if viz_config.get("plot_ss_error_analysis", False):
                 path = create_ss_error_analysis(eval_df, viz_output_dir, save_format, dpi, save_plot_data)
                 if path: generated_plots.append(path)
            if viz_config.get("plot_amino_acid_performance", False):
                 path = create_amino_acid_performance(eval_df, viz_output_dir, save_format, dpi, save_plot_data)
                 if path: generated_plots.append(path)
        elif eval_df is None:
             logger.warning("Skipping comparison plots as merged evaluation data could not be created.")


    logger.info(f"Finished generating {len(generated_plots)} plots.")
    return generated_plots
