# /home/s_felix/ensembleflex/ensembleflex/utils/visualization.py

"""
Visualization functions for the ensembleflex ML pipeline.

Generates plots and saves visualization data CSVs based on the single model's
results and the aggregated dataset structure. Focuses on analyzing performance
relative to features like temperature.
"""

import os
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
# from scipy import stats # Only needed if doing KDE manually, seaborn handles it

# Updated imports
from ensembleflex.utils.helpers import (
    # get_temperature_color, # Less relevant
    make_model_color_map, # Still useful for comparing models in specific plots if needed
    ensure_dir
)
# Import metrics calculation if needed within plotting (e.g., for binned analysis)
from ensembleflex.utils.metrics import evaluate_predictions

logger = logging.getLogger(__name__)

def save_plot(plt_or_fig, output_path: str, dpi: int = 300) -> None:
    """
    Save a matplotlib plot or figure to disk, ensuring directory exists and closing plot.

    Args:
        plt_or_fig: Matplotlib pyplot instance or Figure object.
        output_path: Path to save the plot.
        dpi: Resolution in dots per inch.
    """
    if output_path is None:
        logger.warning("No output path provided for saving plot. Skipping save.")
        plt.close('all')
        return

    ensure_dir(os.path.dirname(output_path))
    try:
        fig = None
        # Check if it's a Figure object or pyplot module
        if isinstance(plt_or_fig, plt.Figure):
            fig = plt_or_fig
        elif hasattr(plt_or_fig, 'gcf'): # Check if it's likely the pyplot module
             fig = plt_or_fig.gcf()

        if fig is not None and fig.get_axes(): # Check if figure has content
            # Try tight_layout, but handle potential errors gracefully
            try:
                fig.tight_layout()
            except ValueError as tle:
                logger.warning(f"tight_layout failed for {os.path.basename(output_path)}: {tle}. Proceeding without it.")
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Plot saved successfully to {output_path}")
        elif fig is not None:
             logger.warning(f"Attempted to save an empty figure to {output_path}. Skipping save.")
        else:
             logger.error("Could not get figure object to save plot.")

    except Exception as e:
        logger.error(f"Failed to save plot to {output_path}: {e}", exc_info=True)
    finally:
        # Always close all figures to prevent memory leaks
        plt.close('all')


# --- Removed Obsolete Functions ---
# plot_temperature_comparison (Metrics vs Temp handled by plot_error_vs_temperature)
# plot_model_metrics_table (Metrics saved directly in evaluate step)
# plot_r2_comparison (Metrics saved directly in evaluate step)
# plot_r2_comparison_scatter (Could be adapted, but less central)

# --- Adapted Visualization Functions ---

# Place this corrected function in ensembleflex/utils/visualization.py

def plot_feature_importance(
    importance_dict: Dict[str, float], # Expects a dictionary
    plot_path: str,
    csv_path: Optional[str] = None,
    top_n: int = 20
) -> None:
    """
    Generate feature importance bar chart and save data from a dictionary.
    Highlights the 'temperature' feature if present.

    Args:
        importance_dict: Dictionary mapping feature names (str) to importance values (float).
        plot_path: Path to save the bar chart plot (.png).
        csv_path: Optional path to save the importance data (.csv).
        top_n: Number of top features to display in the plot.
    """
    if not importance_dict or not isinstance(importance_dict, dict):
        logger.warning("Invalid or empty importance_dict provided. Skipping plot/save.")
        return

    try:
        # Create DataFrame directly from the dictionary
        importance_df = pd.DataFrame(importance_dict.items(), columns=['feature', 'importance'])
        # Handle potential NaN/Inf values before sorting
        importance_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        importance_df.dropna(subset=['importance'], inplace=True)
        if importance_df.empty:
             logger.warning("No valid importance values found after cleaning.")
             return
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

        # Save to CSV if path provided
        if csv_path:
            ensure_dir(os.path.dirname(csv_path))
            try:
                importance_df.to_csv(csv_path, index=False)
                logger.info(f"Feature importance data saved to {csv_path}")
            except Exception as e:
                logger.error(f"Failed to save feature importance CSV: {e}")

        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, max(6, min(top_n, len(importance_df)) * 0.4)))
        top_features = importance_df.head(top_n)

        # Highlight temperature feature using the correct feature names
        colors = ['#d62728' if feat == 'temperature' else '#1f77b4' for feat in top_features['feature']]

        sns.barplot(x='importance', y='feature', data=top_features, palette=colors, ax=ax)
        ax.set_title(f'Top {min(top_n, len(top_features))} Feature Importances') # Adjust title if fewer than top_n
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Feature Name') # Y-axis now uses the actual names

        # Add annotation for temperature rank
        if 'temperature' in importance_df['feature'].values:
             try:
                 temp_rank = importance_df[importance_df['feature'] == 'temperature'].index[0] + 1
                 # Add annotation only if not already displayed in top_n
                 if temp_rank > top_n:
                      ax.text(0.05, 0.01, f"'temperature' rank: {temp_rank}", transform=ax.transAxes,
                              fontsize=9, color='red', ha='left', va='bottom')
             except (IndexError, TypeError):
                 logger.warning("Could not determine rank for 'temperature' feature.")


        save_plot(fig, plot_path) # Pass the figure object

    except Exception as e:
        logger.error(f"Failed to create feature importance plot: {e}", exc_info=True)
        plt.close('all')


def plot_scatter_with_density_contours(
    results_df: pd.DataFrame,
    model_name: str,
    target_col: str,
    plot_path: str,
    csv_path: Optional[str] = None,
    sample_size: int = 5000
) -> None:
    """
    Generate scatter plot of Actual vs. Predicted with density contours for the single model.

    Args:
        results_df: DataFrame containing 'all_results.csv' data.
        model_name: Name of the model (used to find prediction column).
        target_col: Name of the actual target column (e.g., 'rmsf').
        plot_path: Path to save the plot (.png).
        csv_path: Optional path to save the sampled data (.csv).
        sample_size: Number of points to sample for plotting.
    """
    pred_col = f"{model_name}_predicted"
    if target_col not in results_df.columns or pred_col not in results_df.columns:
        logger.error(f"Missing required columns ('{target_col}', '{pred_col}') for scatter plot. Skipping.")
        return

    # Select data and drop NaNs in target or prediction
    plot_data = results_df[[target_col, pred_col]].dropna().copy()
    plot_data.rename(columns={target_col: 'actual', pred_col: 'predicted'}, inplace=True)

    if plot_data.empty:
         logger.warning("No valid data points found for scatter plot after dropping NaNs.")
         return

    # Sample data for plotting
    n_samples = min(sample_size, len(plot_data))
    sampled_df = plot_data.sample(n_samples, random_state=42) # Use fixed seed

    # Save sampled data if path provided
    if csv_path:
        ensure_dir(os.path.dirname(csv_path))
        try:
            sampled_df.to_csv(csv_path, index=False)
            logger.info(f"Sampled scatter plot data saved to {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save scatter plot data CSV: {e}")

    # Create the plot
    try:
        fig, ax = plt.subplots(figsize=(8, 8)) # Get figure and axes objects
        # Use seaborn for easier density plotting
        sns.kdeplot(
            x='actual',
            y='predicted',
            data=sampled_df,
            fill=True,
            cmap='Blues',
            alpha=0.6,
            levels=8, # Adjust number of contours
            ax=ax
        )
        # Overlay scatter points
        ax.scatter(
            sampled_df['actual'],
            sampled_df['predicted'],
            alpha=0.15, # Make points less prominent
            s=15,
            color='darkblue',
            edgecolors='none' # Remove edges for dense plots
        )

        # Add diagonal line
        # Calculate limits based on the data range
        min_val = min(sampled_df['actual'].min(), sampled_df['predicted'].min())
        max_val = max(sampled_df['actual'].max(), sampled_df['predicted'].max())
        lim_min = min_val - 0.1 * (max_val - min_val) # Add some buffer
        lim_max = max_val + 0.1 * (max_val - min_val)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1.5, label="Ideal (y=x)")

        ax.set_xlabel(f'Actual {target_col.upper()}')
        ax.set_ylabel(f'{model_name} Predicted {target_col.upper()}')
        ax.set_title(f'Actual vs. Predicted {target_col.upper()} ({model_name})')
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.5)

        save_plot(fig, plot_path) # Pass the figure object

    except Exception as e:
        logger.error(f"Failed to create density scatter plot: {e}", exc_info=True)
        plt.close('all')


def plot_amino_acid_error_analysis(
    results_df: pd.DataFrame,
    model_name: str,
    target_col: str, # Keep for context, though error should be precalculated
    csv_path: str,
    plot_path: str
) -> None:
    """
    Generate amino acid error analysis data and plot for the single model.

    Args:
        results_df: DataFrame containing 'all_results.csv' data.
        model_name: Name of the model.
        target_col: Name of the actual target column (e.g., 'rmsf').
        csv_path: Path to save the error summary CSV.
        plot_path: Path to save the bar plot (.png).
    """
    error_col = f"{model_name}_abs_error"
    if 'resname' not in results_df.columns:
        logger.warning("Missing 'resname' column for amino acid error analysis. Skipping.")
        return
    if error_col not in results_df.columns:
         # Try to calculate it if prediction and target exist
         pred_col = f"{model_name}_predicted"
         if target_col in results_df.columns and pred_col in results_df.columns:
             logger.info(f"Calculating '{error_col}' on the fly for amino acid analysis.")
             results_df[error_col] = (results_df[pred_col] - results_df[target_col]).abs()
         else:
              logger.error(f"Cannot perform amino acid analysis: Missing '{error_col}' or source columns ('{pred_col}', '{target_col}').")
              return

    # Group and aggregate, handling potential NaNs in error column
    aa_errors = results_df.groupby('resname')[error_col].agg(['mean', 'median', 'std', 'count']).reset_index()
    aa_errors.rename(columns={'mean': 'mean_abs_error', 'median': 'median_abs_error', 'std': 'std_abs_error'}, inplace=True)

    if aa_errors.empty:
         logger.warning("No data found after grouping by resname for error analysis.")
         return

    # Save to CSV
    ensure_dir(os.path.dirname(csv_path))
    try:
        aa_errors.to_csv(csv_path, index=False)
        logger.info(f"Amino acid error summary saved to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save amino acid error CSV: {e}")

    # Create bar plot
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        sorted_df = aa_errors.sort_values('mean_abs_error')
        sns.barplot(x='resname', y='mean_abs_error', data=sorted_df, ax=ax)
        ax.set_title(f'Mean Absolute Error by Amino Acid ({model_name})')
        ax.set_xlabel('Amino Acid')
        ax.set_ylabel('Mean Absolute Error')
        ax.tick_params(axis='x', rotation=45)

        save_plot(fig, plot_path)

    except Exception as e:
        logger.error(f"Failed to create amino acid error plot: {e}", exc_info=True)
        plt.close('all')


def plot_error_analysis_by_property(
    results_df: pd.DataFrame,
    model_name: str,
    target_col: str, # Keep for context
    output_base_path: str # e.g., ../output/ensembleflex/residue_analysis/model_error_by
) -> None:
    """
    Generate error analysis data and plots grouped by various properties for the single model.

    Args:
        results_df: DataFrame containing 'all_results.csv' data.
        model_name: Name of the model.
        target_col: Name of the actual target column (e.g., 'rmsf').
        output_base_path: Base path for saving files (property name and extension added).
    """
    error_col = f"{model_name}_abs_error"
    if error_col not in results_df.columns:
         pred_col = f"{model_name}_predicted"
         if target_col in results_df.columns and pred_col in results_df.columns:
             logger.info(f"Calculating '{error_col}' on the fly for property analysis.")
             # Ensure calculation happens on a temporary copy if needed downstream
             # Use .loc to potentially avoid SettingWithCopyWarning if results_df is a view
             df_copy_for_calc = results_df.copy()
             df_copy_for_calc[error_col] = (df_copy_for_calc[pred_col] - df_copy_for_calc[target_col]).abs()
             # We'll use this df_copy_for_calc below if error had to be calculated
         else:
              logger.error(f"Cannot perform property analysis: Missing '{error_col}' or source columns ('{pred_col}', '{target_col}').")
              return
    else:
        # If error column already exists, still use a copy for modifications like binning
        df_copy_for_calc = results_df.copy()


    properties_to_analyze = {}
    # Define properties based on columns available in the df_copy_for_calc
    if 'secondary_structure_encoded' in df_copy_for_calc.columns:
        properties_to_analyze['secondary_structure'] = {
            'column': 'secondary_structure_encoded',
            'labels': {0: 'Helix', 1: 'Sheet', 2: 'Loop/Other'}
        }
    if 'core_exterior_encoded' in df_copy_for_calc.columns:
        properties_to_analyze['surface_exposure'] = {
            'column': 'core_exterior_encoded',
            'labels': {0: 'Core', 1: 'Surface'}
        }
    if 'normalized_resid' in df_copy_for_calc.columns:
        properties_to_analyze['sequence_position'] = {
            'column': 'normalized_resid',
            'labels': None, # Will be binned
            'bins': 5,
            'bin_labels': ['N-term', 'N-quarter', 'Middle', 'C-quarter', 'C-term']
        }
    if 'relative_accessibility' in df_copy_for_calc.columns:
         properties_to_analyze['accessibility_bin'] = {
             'column': 'relative_accessibility',
             'labels': None,
             'bins': 5
             # Example: Use default numeric labels for bins
         }

    if not properties_to_analyze:
         logger.warning("No suitable columns found for property error analysis.")
         return

    for prop_name, prop_config in properties_to_analyze.items():
        col = prop_config['column']
        if col not in df_copy_for_calc.columns:
            logger.warning(f"Column '{col}' for property '{prop_name}' not found. Skipping.")
            continue

        group_col = col
        df_group = df_copy_for_calc # Start with the dataframe possibly containing calculated error

        # Handle binning for continuous properties
        if 'bins' in prop_config:
            bin_col_name = f"{prop_name}_bin"
            group_col = bin_col_name
            bin_labels = prop_config.get('bin_labels', False)
            try:
                 # Ensure labels match number of bins if provided
                 if isinstance(bin_labels, list) and len(bin_labels) != prop_config['bins']:
                      logger.warning(f"Num labels ({len(bin_labels)}) != num bins ({prop_config['bins']}) for {prop_name}. Using default.")
                      bin_labels = False

                 # Important: Use pd.cut on the DataFrame copy
                 df_group[bin_col_name] = pd.cut(
                     df_group[col],
                     bins=prop_config['bins'],
                     labels=bin_labels,
                     include_lowest=True,
                     duplicates='drop'
                 )
                 # Remove rows where binning failed (resulted in NaN bin)
                 initial_rows = len(df_group)
                 df_group = df_group.dropna(subset=[bin_col_name])
                 if len(df_group) < initial_rows:
                      logger.debug(f"Dropped {initial_rows - len(df_group)} rows with NaN bins for '{col}'.")

                 logger.debug(f"Binned '{col}' into '{bin_col_name}'.")
            except Exception as e:
                 logger.error(f"Failed to bin column '{col}': {e}. Skipping analysis for {prop_name}.")
                 continue

        if df_group.empty:
             logger.warning(f"DataFrame empty after potential binning for property {prop_name}. Skipping.")
             continue

        # Group and aggregate
        try:
            # Use observed=False if using categorical bins to include empty bins
            observed_flag = False if pd.api.types.is_categorical_dtype(df_group.get(group_col)) else True
            # Group by the potentially new bin column
            prop_errors = df_group.groupby(group_col, observed=observed_flag)[error_col].agg(
                ['mean', 'median', 'std', 'count']
            ).reset_index()
            prop_errors.rename(columns={
                group_col: 'value', 'mean': 'mean_abs_error',
                'median': 'median_abs_error', 'std': 'std_abs_error'
                }, inplace=True)
            prop_errors.insert(0, 'property', prop_name)

            # Apply labels if available
            if prop_config.get('labels'):
                # Convert 'value' column to the type of the keys in labels dict before mapping
                label_keys = list(prop_config['labels'].keys())
                if label_keys:
                    key_type = type(label_keys[0])
                    try:
                         prop_errors['value'] = prop_errors['value'].astype(key_type)
                    except (ValueError, TypeError):
                         logger.warning(f"Could not convert 'value' column to type {key_type} for mapping property {prop_name}.")
                prop_errors['value'] = prop_errors['value'].map(prop_config['labels']).fillna(prop_errors['value'].astype(str))


            if prop_errors.empty:
                 logger.warning(f"No results after grouping for property {prop_name}.")
                 continue

            # Save CSV
            csv_path = f"{output_base_path}_{prop_name}.csv"
            ensure_dir(os.path.dirname(csv_path))
            prop_errors.to_csv(csv_path, index=False)
            logger.info(f"Error analysis by {prop_name} saved to {csv_path}")

            # Create Plot
            plot_path = f"{output_base_path}_{prop_name}.png"
            fig, ax = plt.subplots(figsize=(max(8, len(prop_errors)*0.8), 5))
            plot_order = None
            if prop_name == 'sequence_position': plot_order = prop_config.get('bin_labels')
            elif prop_name == 'surface_exposure': plot_order = ['Core', 'Surface']
            elif prop_name == 'secondary_structure': plot_order = ['Helix', 'Sheet', 'Loop/Other']

            sns.barplot(x='value', y='mean_abs_error', data=prop_errors, order=plot_order, ax=ax, palette="viridis")
            ax.set_title(f'Mean Absolute Error by {prop_name.replace("_", " ").title()} ({model_name})')
            ax.set_xlabel(prop_name.replace('_', ' ').title())
            ax.set_ylabel('Mean Absolute Error')

            # --- CORRECTED LINE ---
            ax.tick_params(axis='x', rotation=45 if prop_name not in ['sequence_position', 'surface_exposure'] else 0)
            # --- END CORRECTED LINE ---

            # Optionally improve label readability further if rotated
            if prop_name not in ['sequence_position', 'surface_exposure']:
                 fig.autofmt_xdate(rotation=45) # Alternative way to handle rotated labels

            save_plot(fig, plot_path)

        except Exception as e:
            logger.error(f"Failed processing or plotting for property {prop_name}: {e}", exc_info=True)
            plt.close('all')


# --- New Visualization Functions ---

def plot_prediction_vs_temperature(
    results_df: pd.DataFrame,
    model_name: str,
    plot_path: str,
    sample_size: int = 5000,
    color_by: Optional[str] = 'actual' # 'actual', 'error', 'resname', None
) -> None:
    """
    Generate scatter plot of Predicted RMSF vs. Input Temperature feature.

    Args:
        results_df: DataFrame containing 'all_results.csv' data.
        model_name: Name of the model.
        plot_path: Path to save the plot (.png).
        sample_size: Number of points to sample for plotting.
        color_by: Column to use for coloring points ('actual', 'error', 'resname', or None).
    """
    pred_col = f"{model_name}_predicted"
    temp_col = 'temperature'
    target_col = 'rmsf' # Assumes target is 'rmsf'
    error_col = f"{model_name}_abs_error"

    required_cols = [pred_col, temp_col]
    color_col = None
    if color_by == 'actual': color_col = target_col
    elif color_by == 'error': color_col = error_col
    elif color_by == 'resname': color_col = 'resname'

    if color_col: required_cols.append(color_col)

    if not all(col in results_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in results_df.columns]
        logger.error(f"Missing required columns ({missing}) for prediction vs temperature plot. Skipping.")
        return

    plot_data = results_df[required_cols].dropna().copy()

    if plot_data.empty:
        logger.warning("No valid data points found for prediction vs temperature plot.")
        return

    # Sample data
    n_samples = min(sample_size, len(plot_data))
    sampled_df = plot_data.sample(n_samples, random_state=42)

    # Create plot
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        hue_column_data = None
        palette = None
        cbar_label = None
        legend_title = None
        is_categorical_hue = False

        if color_col and color_col != 'resname': # Numeric coloring
            hue_column_data = sampled_df[color_col]
            palette = 'viridis' if color_col == target_col else 'coolwarm'
            cbar_label = f'Actual {target_col.upper()}' if color_col == target_col else 'Absolute Error'
        elif color_col == 'resname': # Categorical coloring
             hue_column_data = sampled_df['resname']
             is_categorical_hue = True
             unique_res = sampled_df['resname'].unique()
             n_colors = len(unique_res)
             if n_colors <= 20:
                  palette = sns.color_palette("tab20", n_colors=n_colors)
                  legend_title = "Residue"
             else: # Too many categories, disable coloring
                  logger.warning("Too many residue types for distinct coloring. Disabling coloring.")
                  hue_column_data = None
                  is_categorical_hue = False

        # Use seaborn for easier hue mapping
        sns.scatterplot(
            x=temp_col,
            y=pred_col,
            data=sampled_df,
            hue=hue_column_data,
            palette=palette,
            s=25,
            alpha=0.7,
            edgecolor='none',
            legend='brief' if is_categorical_hue and n_colors <= 20 else False, # Show legend only for categorical & manageable N
            ax=ax
        )

        # Add colorbar for numeric hue
        # Get the collection used by scatterplot to create a colorbar
        if not is_categorical_hue and hue_column_data is not None:
            norm = plt.Normalize(hue_column_data.min(), hue_column_data.max())
            sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label=cbar_label)


        if is_categorical_hue and legend_title and ax.get_legend() is not None:
             ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')
        elif ax.get_legend() is not None:
             ax.get_legend().remove() # Remove default legend if not needed

        ax.set_xlabel("Input Temperature Feature (K)")
        ax.set_ylabel(f"{model_name} Predicted {target_col.upper()}")
        ax.set_title(f"Predicted {target_col.upper()} vs. Input Temperature ({model_name})")
        ax.grid(True, linestyle=':', alpha=0.5)

        save_plot(fig, plot_path)

    except Exception as e:
        logger.error(f"Failed to create prediction vs temperature plot: {e}", exc_info=True)
        plt.close('all')


def plot_error_vs_temperature(
    results_df: pd.DataFrame,
    model_name: str,
    config: Dict[str, Any], # Needed for metrics config
    plot_path: str,
    csv_path: Optional[str] = None,
    n_bins: int = 10
) -> None:
    """
    Generate plots of evaluation metrics (calculated in bins) vs. Input Temperature.

    Args:
        results_df: DataFrame containing 'all_results.csv' data.
        model_name: Name of the model.
        config: Global configuration dictionary (for metric definitions).
        plot_path: Path to save the plot (.png).
        csv_path: Optional path to save the binned metric data (.csv).
        n_bins: Number of temperature bins to create.
    """
    temp_col = 'temperature'
    target_col = config['dataset']['target'] # 'rmsf'
    pred_col = f"{model_name}_predicted"

    if not all(c in results_df.columns for c in [temp_col, target_col, pred_col]):
        missing = [c for c in [temp_col, target_col, pred_col] if c not in results_df.columns]
        logger.error(f"Missing required columns ({missing}) for error vs temperature analysis. Skipping.")
        return

    df_valid = results_df[[temp_col, target_col, pred_col]].dropna().copy()
    if df_valid.empty:
        logger.warning("No valid data points found for error vs temperature analysis.")
        return

    # Create temperature bins
    min_temp, max_temp = df_valid[temp_col].min(), df_valid[temp_col].max()
    if max_temp <= min_temp: # Handle single temperature value or invalid range
         logger.warning(f"Cannot create bins for temperature range: min={min_temp}, max={max_temp}. Skipping plot.")
         return

    try:
        # Create bins with slightly extended range to include endpoints
        bins = np.linspace(min_temp - 0.01 * abs(min_temp) if min_temp != 0 else -0.01,
                           max_temp + 0.01 * abs(max_temp) if max_temp != 0 else 0.01,
                           n_bins + 1)
        # Use bin midpoints as labels for plotting later
        bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(n_bins)]
        # Use pd.cut without labels first to get bin indices, then map to centers
        df_valid['temp_bin_idx'] = pd.cut(df_valid[temp_col], bins=bins, labels=False, right=True, include_lowest=True)

    except Exception as e:
        logger.error(f"Failed to create temperature bins: {e}. Skipping plot.")
        return


    # Calculate metrics per bin
    binned_metrics = []
    metrics_to_calc_config = config['evaluation']['metrics'] # Get metrics config
    metrics_to_calculate = [m for m, enabled in metrics_to_calc_config.items() if enabled]

    for bin_idx, group in df_valid.groupby('temp_bin_idx', observed=False):
        if group.empty: continue
        if pd.isna(bin_idx): continue # Skip rows that didn't fall into a bin

        y_true = group[target_col].values
        y_pred = group[pred_col].values

        # Calculate metrics using the evaluate_predictions function
        # Need to create a temporary config for evaluate_predictions with only the required metrics
        temp_eval_config = {'evaluation': {'metrics': {m: True for m in metrics_to_calculate}}}
        bin_result = evaluate_predictions(y_true, y_pred, temp_eval_config) # Use temp config

        bin_center = bin_centers[int(bin_idx)]
        bin_result['temp_center'] = bin_center
        bin_result['count'] = len(group)

        binned_metrics.append(bin_result)

    if not binned_metrics:
        logger.warning("No metrics calculated across temperature bins.")
        return

    metrics_df = pd.DataFrame(binned_metrics).sort_values('temp_center')

     # Save binned metrics data if path provided
    if csv_path:
        ensure_dir(os.path.dirname(csv_path))
        try:
            metrics_df.to_csv(csv_path, index=False)
            logger.info(f"Binned metrics vs temperature data saved to {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save binned metrics CSV: {e}")

    # Create plots (e.g., RMSE and R2 vs Temp)
    try:
        # Identify which metrics were successfully calculated
        plot_cols = [m for m in metrics_to_calculate if m in metrics_df.columns]

        if not plot_cols:
             logger.warning("No suitable metrics available for plotting vs temperature.")
             return

        n_plots = len(plot_cols)
        fig, axes = plt.subplots(n_plots, 1, figsize=(8, n_plots * 3.5), sharex=True, squeeze=False) # Ensure axes is 2D array
        axes = axes.flatten() # Flatten for easy iteration

        for i, metric in enumerate(plot_cols):
            ax = axes[i]
            ax.plot(metrics_df['temp_center'], metrics_df[metric], marker='o', linestyle='-')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} vs. Input Temperature ({model_name})')
            ax.grid(True, linestyle=':', alpha=0.5)
            # Add count as secondary y-axis? Or text labels?
            # ax2 = ax.twinx()
            # ax2.bar(metrics_df['temp_center'], metrics_df['count'], alpha=0.2, width=(bins[1]-bins[0])*0.8, color='grey')
            # ax2.set_ylabel('Count', color='grey')

        axes[-1].set_xlabel("Temperature Bin Center (K)")
        fig.suptitle(f"Performance Metrics vs. Temperature ({model_name})", y=1.02) # Add overall title
        fig.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap

        save_plot(fig, plot_path) # Pass the figure object

    except Exception as e:
        logger.error(f"Failed to create error vs temperature plot: {e}", exc_info=True)
        plt.close('all')

def plot_training_validation_curves(
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    model_name: str,
    plot_path: str # Changed from config to direct path
    # config: Dict[str, Any] # Config might not be needed if path is direct
) -> None:
    """
    Generate training and validation curves plot. CSV saving handled by Pipeline.

    Args:
        train_metrics: Dictionary of training metrics by epoch (e.g., {'train_loss': [], 'train_r2': []}).
        val_metrics: Dictionary of validation metrics by epoch (e.g., {'val_loss': [], 'val_r2': []}).
        model_name: Name of the model.
        plot_path: Full path to save the plot (.png).
    """
    if not train_metrics and not val_metrics:
        logger.warning(f"No training or validation metrics provided for {model_name}. Skipping curve plot.")
        return

    # Determine available epochs and metrics
    epochs = 0
    if train_metrics: epochs = max(epochs, len(next(iter(train_metrics.values()), [])))
    if val_metrics: epochs = max(epochs, len(next(iter(val_metrics.values()), [])))

    if epochs == 0:
        logger.warning(f"No epochs found in metrics data for {model_name}. Skipping curve plot.")
        return

    # Identify metrics present in both train and val (or just one)
    possible_metrics = set(k.replace('train_', '').replace('val_', '') for k in train_metrics) | \
                       set(k.replace('train_', '').replace('val_', '') for k in val_metrics)
    metrics_to_plot = []
    if 'loss' in possible_metrics: metrics_to_plot.append(('loss', 'Loss'))
    if 'r2' in possible_metrics: metrics_to_plot.append(('r2', 'R²'))
    if 'mae' in possible_metrics: metrics_to_plot.append(('mae', 'MAE'))
    # Add others if needed

    if not metrics_to_plot:
         logger.warning(f"No plottable metrics (loss, r2, mae) found for {model_name}.")
         return

    n_plots = len(metrics_to_plot)
    try:
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, n_plots * 4), sharex=True, squeeze=False)
        axes = axes.flatten()

        x_values = range(epochs)

        for i, (metric_key, metric_title) in enumerate(metrics_to_plot):
            ax = axes[i]
            train_key = f"train_{metric_key}"
            val_key = f"val_{metric_key}"

            if train_key in train_metrics and len(train_metrics[train_key]) == epochs:
                ax.plot(x_values, train_metrics[train_key], marker='.', linestyle='-', label=f'Training {metric_title}')
            if val_key in val_metrics and len(val_metrics[val_key]) == epochs:
                ax.plot(x_values, val_metrics[val_key], marker='.', linestyle='-', label=f'Validation {metric_title}')

            ax.set_ylabel(metric_title)
            ax.set_title(f'{metric_title} Curve')
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.5)

        axes[-1].set_xlabel('Epoch')
        fig.suptitle(f'Training & Validation Curves for {model_name}', y=1.02)
        fig.tight_layout(rect=[0, 0.03, 1, 0.98])

        save_plot(fig, plot_path)

    except Exception as e:
         logger.error(f"Failed to generate training curves plot: {e}", exc_info=True)
         plt.close('all')
         