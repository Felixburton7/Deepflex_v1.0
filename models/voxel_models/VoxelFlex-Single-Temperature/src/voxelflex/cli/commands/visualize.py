"""
Revised visualization command for Voxelflex with improved visualizations.

This module handles creating visualizations for model performance and analysis.
"""

import os
import time
import json
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import r2_score
from scipy.stats import gaussian_kde

from voxelflex.utils.logging_utils import get_logger
from voxelflex.utils.file_utils import ensure_dir, load_json

logger = get_logger(__name__)

def create_loss_curve(
    train_history: Dict[str, List[float]],
    output_dir: str,
    save_format: str = 'png',
    dpi: int = 300
) -> str:
    """
    Create a plot of training and validation loss curves.
    
    Args:
        train_history: Dictionary containing training and validation losses
        output_dir: Directory to save the plot
        save_format: Format to save the plot (png, jpg, etc.)
        dpi: DPI for the saved plot
        
    Returns:
        Path to the saved plot
    """
    logger.info("Creating loss curve plot")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_history['train_loss']) + 1)
    ax.plot(epochs, train_history['train_loss'], 'b-', label='Training Loss', linewidth=2.5)
    ax.plot(epochs, train_history['val_loss'], 'r-', label='Validation Loss', linewidth=2.5)
    
    ax.set_title('Training and Validation Loss', fontsize=16, fontweight='bold')
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add min/max annotations
    min_val_loss = min(train_history['val_loss'])
    min_val_epoch = train_history['val_loss'].index(min_val_loss) + 1
    
    ax.annotate(f'Min val loss: {min_val_loss:.4f}',
                xy=(min_val_epoch, min_val_loss),
                xytext=(min_val_epoch, min_val_loss * 1.2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
    
    # Save plot
    ensure_dir(output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"loss_curve_{timestamp}.{save_format}")
    
    fig.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Loss curve plot saved to {plot_path}")
    return plot_path


def create_prediction_scatter(
    predictions_df: pd.DataFrame,
    output_dir: str,
    save_format: str = 'png',
    dpi: int = 300,
    max_points: int = 1000
) -> str:
    """
    Create a simplified scatter plot of predicted vs. validation RMSF values with R².
    
    Args:
        predictions_df: DataFrame containing predictions and actual values
        output_dir: Directory to save the plot
        save_format: Format to save the plot (png, jpg, etc.)
        dpi: DPI for the saved plot
        max_points: Maximum number of points to plot (to avoid overcrowding)
        
    Returns:
        Path to the saved plot
    """
    logger.info("Creating prediction scatter plot with R²")
    
    # Extract predictions and actual values
    y_true = predictions_df['actual_rmsf'].values
    y_pred = predictions_df['predicted_rmsf'].values
    
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    mse = np.mean((y_pred - y_true)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true))
    
    # Sample points if there are too many
    if len(y_true) > max_points:
        logger.info(f"Sampling {max_points} points for scatter plot (out of {len(y_true)} total)")
        indices = np.random.choice(len(y_true), max_points, replace=False)
        y_true_plot = y_true[indices]
        y_pred_plot = y_pred[indices]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a simple scatter plot with alpha for overlap visibility
    scatter = ax.scatter(y_true_plot, y_pred_plot, c='steelblue', s=30, alpha=0.6, edgecolor='none')
    
    # Add perfect prediction line
    max_val = max(np.max(y_true), np.max(y_pred))
    min_val = min(np.min(y_true), np.min(y_pred))
    margin = (max_val - min_val) * 0.1
    ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 'r--', linewidth=2.0, label='Perfect Prediction')
    
    # Add metrics as text box
    metrics_text = (
        f'R²: {r2:.4f}\n'
        f'RMSE: {rmse:.4f}\n'
        f'MAE: {mae:.4f}\n'
        f'Samples: {len(y_true)}'
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    ax.set_title('Predicted vs. Validation RMSF Values', fontsize=16, fontweight='bold')
    ax.set_xlabel('Validation RMSF', fontsize=14)
    ax.set_ylabel('Predicted RMSF', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right', fontsize=12)
    
    # Make plot square
    ax.set_aspect('equal')
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    
    # Save plot
    ensure_dir(output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"prediction_scatter_{timestamp}.{save_format}")
    
    fig.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Prediction scatter plot saved to {plot_path}")
    return plot_path


def create_error_distribution(
    predictions_df: pd.DataFrame,
    output_dir: str,
    save_format: str = 'png',
    dpi: int = 300
) -> str:
    """
    Create a histogram of prediction errors.
    
    Args:
        predictions_df: DataFrame containing predictions and actual values
        output_dir: Directory to save the plot
        save_format: Format to save the plot (png, jpg, etc.)
        dpi: DPI for the saved plot
        
    Returns:
        Path to the saved plot
    """
    logger.info("Creating error distribution plot")
    
    # Calculate errors
    predictions_df['error'] = predictions_df['predicted_rmsf'] - predictions_df['actual_rmsf']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot with more bins and a nicer color
    sns.histplot(predictions_df['error'], kde=True, bins=30, ax=ax, color='steelblue', 
                 edgecolor='black', alpha=0.7, line_kws={'linewidth': 2})
    
    ax.set_title('Distribution of Prediction Errors', fontsize=16, fontweight='bold')
    ax.set_xlabel('Error (Predicted - Validation)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add error statistics
    mean_error = predictions_df['error'].mean()
    std_error = predictions_df['error'].std()
    median_error = predictions_df['error'].median()
    
    ax.axvline(mean_error, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.4f}')
    ax.axvline(median_error, color='g', linestyle='--', linewidth=2, label=f'Median: {median_error:.4f}')
    ax.axvline(mean_error + std_error, color='purple', linestyle=':', linewidth=1.5, label=f'Std: {std_error:.4f}')
    ax.axvline(mean_error - std_error, color='purple', linestyle=':', linewidth=1.5)
    
    ax.legend(fontsize=12)
    
    # Save plot
    ensure_dir(output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"error_distribution_{timestamp}.{save_format}")
    
    fig.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Error distribution plot saved to {plot_path}")
    return plot_path


def create_residue_type_analysis(
    predictions_df: pd.DataFrame,
    output_dir: str,
    save_format: str = 'png',
    dpi: int = 300
) -> Optional[str]:
    """
    Create a box plot of prediction errors grouped by residue type.
    
    Args:
        predictions_df: DataFrame containing predictions and actual values
        output_dir: Directory to save the plot
        save_format: Format to save the plot (png, jpg, etc.)
        dpi: DPI for the saved plot
        
    Returns:
        Path to the saved plot, or None if residue type information is not available
    """
    if 'resname' not in predictions_df.columns:
        logger.warning("Residue type information not available for residue type analysis")
        return None
    
    logger.info("Creating residue type analysis plot")
    
    # Calculate errors if not already done
    if 'error' not in predictions_df.columns:
        predictions_df['error'] = predictions_df['predicted_rmsf'] - predictions_df['actual_rmsf']
    
    # Calculate absolute errors
    predictions_df['abs_error'] = np.abs(predictions_df['error'])
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Create box plot with improved styling
    ax = sns.boxplot(x='resname', y='abs_error', hue='resname', data=predictions_df, 
                    palette='viridis', width=0.6, linewidth=1.5, legend=False,
                    fliersize=5, flierprops=dict(marker='o', markerfacecolor='red'))
    
    ax.set_title('Prediction Error by Residue Type', fontsize=16, fontweight='bold')
    ax.set_xlabel('Residue Type', fontsize=14)
    ax.set_ylabel('Absolute Error', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels if there are many residue types
    if predictions_df['resname'].nunique() > 10:
        plt.xticks(rotation=45, ha='right', fontsize=12)
    
    # Add number of samples for each residue type
    restype_counts = predictions_df['resname'].value_counts()
    for i, restype in enumerate(ax.get_xticklabels()):
        restype_text = restype.get_text()
        if restype_text in restype_counts:
            count = restype_counts[restype_text]
            ax.text(i, -0.1, f'n={count}', ha='center', va='top', rotation=45,
                   transform=ax.get_xaxis_transform(), fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    ensure_dir(output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"residue_type_analysis_{timestamp}.{save_format}")
    
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Residue type analysis plot saved to {plot_path}")
    return plot_path


def create_amino_acid_performance(
    predictions_df: pd.DataFrame,
    output_dir: str,
    save_format: str = 'png',
    dpi: int = 300
) -> Optional[str]:
    """
    Create a histogram of amino acid performance metrics.
    
    Args:
        predictions_df: DataFrame containing predictions and actual values
        output_dir: Directory to save the plot
        save_format: Format to save the plot (png, jpg, etc.)
        dpi: DPI for the saved plot
        
    Returns:
        Path to the saved plot, or None if residue type information is not available
    """
    if 'resname' not in predictions_df.columns:
        logger.warning("Residue type information not available for amino acid performance analysis")
        return None
    
    logger.info("Creating amino acid performance plot")
    
    # Calculate errors if not already done
    if 'error' not in predictions_df.columns:
        predictions_df['error'] = predictions_df['predicted_rmsf'] - predictions_df['actual_rmsf']
    
    # Calculate metrics for each amino acid
    aa_metrics = []
    
    for resname in sorted(predictions_df['resname'].unique()):
        resname_df = predictions_df[predictions_df['resname'] == resname]
        y_true = resname_df['actual_rmsf'].values
        y_pred = resname_df['predicted_rmsf'].values
        
        mse = ((y_pred - y_true) ** 2).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(y_pred - y_true).mean()
        r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
        
        aa_metrics.append({
            'resname': resname,
            'count': len(resname_df),
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })
    
    aa_metrics_df = pd.DataFrame(aa_metrics)
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Plot RMSE
    sns.barplot(x='resname', y='rmse', hue='resname', data=aa_metrics_df, ax=axes[0], palette='viridis', legend=False)
    axes[0].set_title('RMSE by Amino Acid', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Amino Acid', fontsize=12)
    axes[0].set_ylabel('RMSE', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot MAE
    sns.barplot(x='resname', y='mae', hue='resname', data=aa_metrics_df, ax=axes[1], palette='magma', legend=False)
    axes[1].set_title('MAE by Amino Acid', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Amino Acid', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot R²
    sns.barplot(x='resname', y='r2', hue='resname', data=aa_metrics_df, ax=axes[2], palette='plasma')
    axes[2].set_title('R² by Amino Acid', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Amino Acid', fontsize=12)
    axes[2].set_ylabel('R²', fontsize=12)
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    # Plot sample count
    sns.barplot(x='resname', y='count',hue='resname', data=aa_metrics_df, ax=axes[3], palette='crest')
    axes[3].set_title('Sample Count by Amino Acid', fontsize=14, fontweight='bold')
    axes[3].set_xlabel('Amino Acid', fontsize=12)
    axes[3].set_ylabel('Count', fontsize=12)
    axes[3].grid(True, linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels
    for ax in axes:
    # First get the current tick positions
        ticks = ax.get_xticks()
        # Then set the labels with rotation
        ax.set_xticks(ticks)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    ensure_dir(output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"amino_acid_performance_{timestamp}.{save_format}")
    
    fig.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Amino acid performance plot saved to {plot_path}")
    return plot_path


def create_residue_error_histogram(
    predictions_df: pd.DataFrame,
    output_dir: str,
    save_format: str = 'png',
    dpi: int = 300
) -> str:
    """
    Create a histogram showing error distribution for each residue.
    
    Args:
        predictions_df: DataFrame containing predictions and actual values
        output_dir: Directory to save the plot
        save_format: Format to save the plot (png, jpg, etc.)
        dpi: DPI for the saved plot
        
    Returns:
        Path to the saved plot
    """
    logger.info("Creating residue error histogram")
    
    # Ensure we have error column
    if 'error' not in predictions_df.columns:
        predictions_df['error'] = predictions_df['predicted_rmsf'] - predictions_df['actual_rmsf']
    
    # Get absolute error
    predictions_df['abs_error'] = np.abs(predictions_df['error'])
    
    # Group by residue ID if available and compute mean error
    if 'resid' in predictions_df.columns:
        residue_errors = predictions_df.groupby('resid')['abs_error'].mean().reset_index()
        residue_errors = residue_errors.sort_values('resid')
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create bar plot with color gradient based on error magnitude
        cmap = plt.cm.get_cmap('viridis_r')
        norm = plt.Normalize(residue_errors['abs_error'].min(), residue_errors['abs_error'].max())
        colors = [cmap(norm(value)) for value in residue_errors['abs_error']]
        
        bars = ax.bar(residue_errors['resid'], residue_errors['abs_error'], color=colors, width=0.8)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Absolute Error Magnitude', fontsize=12)
        
        ax.set_title('Mean Absolute Error by Residue Position', fontsize=16, fontweight='bold')
        ax.set_xlabel('Residue ID', fontsize=14)
        ax.set_ylabel('Mean Absolute Error', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add statistics
        highest_error_resid = residue_errors.loc[residue_errors['abs_error'].idxmax()]
        lowest_error_resid = residue_errors.loc[residue_errors['abs_error'].idxmin()]
        
        # Annotate highest and lowest error residues
        ax.annotate(f"Highest error: {highest_error_resid['abs_error']:.4f}",
                    xy=(highest_error_resid['resid'], highest_error_resid['abs_error']),
                    xytext=(0, 20), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                    fontsize=12, ha='center')
        
        ax.annotate(f"Lowest error: {lowest_error_resid['abs_error']:.4f}",
                    xy=(lowest_error_resid['resid'], lowest_error_resid['abs_error']),
                    xytext=(0, -20), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                    fontsize=12, ha='center')
        
        # Add a horizontal line for mean error
        mean_error = residue_errors['abs_error'].mean()
        ax.axhline(mean_error, color='red', linestyle='--', linewidth=1.5, 
                   label=f'Mean Error: {mean_error:.4f}')
        ax.legend(fontsize=12)
        
        # Remove every second tick if there are too many residues
        if len(residue_errors) > 30:
            for idx, label in enumerate(ax.xaxis.get_ticklabels()):
                if idx % 2 != 0:
                    label.set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        ensure_dir(output_dir)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(output_dir, f"residue_error_histogram_{timestamp}.{save_format}")
        
        fig.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Residue error histogram saved to {plot_path}")
        return plot_path
    else:
        logger.warning("Residue ID information not available for residue error histogram")
        return None


def create_residue_type_error_histogram(
    predictions_df: pd.DataFrame,
    output_dir: str,
    save_format: str = 'png',
    dpi: int = 300
) -> str:
    """
    Create a histogram showing error distribution by residue type (amino acid).
    
    Args:
        predictions_df: DataFrame containing predictions and actual values
        output_dir: Directory to save the plot
        save_format: Format to save the plot (png, jpg, etc.)
        dpi: DPI for the saved plot
        
    Returns:
        Path to the saved plot
    """
    logger.info("Creating residue type error histogram")
    
    # Check if we have residue type information
    if 'resname' not in predictions_df.columns:
        logger.warning("Residue type information not available for residue type error histogram")
        return None
    
    # Ensure we have error column
    if 'error' not in predictions_df.columns:
        predictions_df['error'] = predictions_df['predicted_rmsf'] - predictions_df['actual_rmsf']
    
    # Get absolute error
    predictions_df['abs_error'] = np.abs(predictions_df['error'])
    
    # Group by residue type and compute mean absolute error
    residue_type_errors = predictions_df.groupby('resname')['abs_error'].mean().reset_index()
    
    # Sort by error for better visualization
    residue_type_errors = residue_type_errors.sort_values('abs_error', ascending=False)
    
    # Get sample counts for each residue type
    residue_counts = predictions_df['resname'].value_counts().reset_index()
    residue_counts.columns = ['resname', 'count']
    
    # Merge with errors
    residue_type_errors = pd.merge(residue_type_errors, residue_counts, on='resname', how='left')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create bar plot with color gradient based on error magnitude
    cmap = plt.cm.get_cmap('viridis_r')
    norm = plt.Normalize(residue_type_errors['abs_error'].min(), residue_type_errors['abs_error'].max())
    colors = [cmap(norm(value)) for value in residue_type_errors['abs_error']]
    
    # Create bars
    bars = ax.bar(residue_type_errors['resname'], residue_type_errors['abs_error'], color=colors, width=0.7)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Absolute Error Magnitude', fontsize=12)
    
    ax.set_title('Mean Absolute Error by Amino Acid Type', fontsize=16, fontweight='bold')
    ax.set_xlabel('Amino Acid', fontsize=14)
    ax.set_ylabel('Mean Absolute Error', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add sample count above each bar
    for i, (_, row) in enumerate(residue_type_errors.iterrows()):
        ax.text(i, row['abs_error'] + 0.01, f"n={row['count']}", 
                ha='center', va='bottom', fontsize=10, rotation=0)
    
    # Add statistics
    highest_error_aa = residue_type_errors.iloc[0]
    lowest_error_aa = residue_type_errors.iloc[-1]
    
    # Annotate highest and lowest error amino acids
    ax.annotate(f"{highest_error_aa['resname']}: {highest_error_aa['abs_error']:.4f}",
                xy=(0, highest_error_aa['abs_error']),
                xytext=(1, highest_error_aa['abs_error'] * 1.1), 
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                fontsize=12)
    
    ax.annotate(f"{lowest_error_aa['resname']}: {lowest_error_aa['abs_error']:.4f}",
                xy=(len(residue_type_errors)-1, lowest_error_aa['abs_error']),
                xytext=(len(residue_type_errors)-2, lowest_error_aa['abs_error'] * 1.5), 
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                fontsize=12)
    
    # Add a horizontal line for mean error
    mean_error = residue_type_errors['abs_error'].mean()
    ax.axhline(mean_error, color='red', linestyle='--', linewidth=1.5, 
               label=f'Mean Error: {mean_error:.4f}')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    ensure_dir(output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"residue_type_error_histogram_{timestamp}.{save_format}")
    
    fig.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Residue type error histogram saved to {plot_path}")
    return plot_path


def create_predicted_vs_validation_scatter_density(
    predictions_df: pd.DataFrame,
    output_dir: str,
    save_format: str = 'png',
    dpi: int = 300
) -> str:
    """
    Create a scatter plot with density contours for predicted vs. validation RMSF values.
    
    Args:
        predictions_df: DataFrame containing predictions and actual values
        output_dir: Directory to save the plot
        save_format: Format to save the plot (png, jpg, etc.)
        dpi: DPI for the saved plot
        
    Returns:
        Path to the saved plot
    """
    logger.info("Creating predicted vs. validation scatter density plot")
    
    # Extract predictions and actual values
    y_true = predictions_df['actual_rmsf'].values
    y_pred = predictions_df['predicted_rmsf'].values
    
    # Calculate R² and other metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use kernel density estimate for the joint distribution
    xy = np.vstack([y_true, y_pred])
    kernel = gaussian_kde(xy)
    
    # Get the range for creating the density map
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    margin = (max_val - min_val) * 0.1
    x_range = np.linspace(min_val - margin, max_val + margin, 100)
    y_range = np.linspace(min_val - margin, max_val + margin, 100)
    X, Y = np.meshgrid(x_range, y_range)
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    # Evaluate the KDE at the grid positions
    Z = kernel(positions)
    Z = Z.reshape(X.shape)
    
    # Create the scatter plot of actual points
    scatter = ax.scatter(y_true, y_pred, c='steelblue', s=20, alpha=0.3, edgecolor='none')
    
    # Add contour lines for the density
    contour = ax.contour(X, Y, Z, cmap='viridis', levels=7, alpha=0.8)
    
    # Add perfect prediction line
    ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 'r--', linewidth=2.0, label='Perfect Prediction')
    
    # Add metrics as text box
    metrics_text = (
        f'R²: {r2:.4f}\n'
        f'RMSE: {rmse:.4f}\n'
        f'Samples: {len(y_true)}'
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    ax.set_title('Predicted vs. Validation RMSF with Density Contours', fontsize=16, fontweight='bold')
    ax.set_xlabel('Validation RMSF', fontsize=14)
    ax.set_ylabel('Predicted RMSF', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right', fontsize=12)
    
    # Make plot square
    ax.set_aspect('equal')
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    
    # Save plot
    ensure_dir(output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"predicted_vs_validation_density_{timestamp}.{save_format}")
    
    fig.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Predicted vs. validation scatter density plot saved to {plot_path}")
    return plot_path


def create_visualizations(
    config: Dict[str, Any],
    train_history: Optional[Dict[str, List[float]]],
    predictions_path: str
) -> List[str]:
    """
    Create visualizations for model performance and analysis.
    
    Args:
        config: Configuration dictionary
        train_history: Training history dictionary (optional)
        predictions_path: Path to predictions file
        
    Returns:
        List of paths to created visualizations
    """
    logger.info("Creating visualizations")
    
    # Load predictions
    predictions_df = pd.read_csv(predictions_path)
    
    # Create output directory
    output_dir = os.path.join(config["output"]["base_dir"], "visualizations")
    ensure_dir(output_dir)
    
    # Get visualization settings
    viz_config = config.get("visualization", {})
    save_format = viz_config.get("save_format", "png")
    dpi = viz_config.get("dpi", 300)
    max_scatter_points = viz_config.get("max_scatter_points", 1000)
    
    # Create visualizations
    visualization_paths = []
    
    if train_history is None and viz_config.get("plot_loss", True):
        logger.warning("Training history not provided, skipping loss curve plot")

    # Loss curve (if training history is available)
    if train_history is not None and viz_config.get("plot_loss", True):
        loss_curve_path = create_loss_curve(
            train_history, output_dir, save_format, dpi
        )
        visualization_paths.append(loss_curve_path)
    
    # Prediction scatter plot (improved with R²)
    if viz_config.get("plot_predictions", True):
        scatter_path = create_prediction_scatter(
            predictions_df, output_dir, save_format, dpi, max_scatter_points
        )
        visualization_paths.append(scatter_path)
    
    # Predicted vs validation with density (new and improved)
    density_scatter_path = create_predicted_vs_validation_scatter_density(
        predictions_df, output_dir, save_format, dpi
    )
    visualization_paths.append(density_scatter_path)
    
    # Error distribution
    if viz_config.get("plot_error_distribution", True):
        error_dist_path = create_error_distribution(
            predictions_df, output_dir, save_format, dpi
        )
        visualization_paths.append(error_dist_path)
    
    # Residue error histogram
    residue_error_path = create_residue_error_histogram(
        predictions_df, output_dir, save_format, dpi
    )
    if residue_error_path:
        visualization_paths.append(residue_error_path)
    
    # Residue type error histogram (new)
    residue_type_error_path = create_residue_type_error_histogram(
        predictions_df, output_dir, save_format, dpi
    )
    if residue_type_error_path:
        visualization_paths.append(residue_type_error_path)
    
    # Residue type analysis
    if viz_config.get("plot_residue_type_analysis", True):
        residue_type_path = create_residue_type_analysis(
            predictions_df, output_dir, save_format, dpi
        )
        if residue_type_path:
            visualization_paths.append(residue_type_path)
    
    # Amino acid performance
    if viz_config.get("plot_amino_acid_performance", True):
        aa_performance_path = create_amino_acid_performance(
            predictions_df, output_dir, save_format, dpi
        )
        if aa_performance_path:
            visualization_paths.append(aa_performance_path)
    
    logger.info(f"Created {len(visualization_paths)} visualizations")
    return visualization_paths