#!/usr/bin/env python3
"""
Temperature Comparison Visualizer for FlexSeq Results

This script generates publication-quality visualizations from
temperature comparison data produced by the FlexSeq pipeline.

Usage:
    python temperature_comparison_visualizer.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR] [--focus_temp FOCUS_TEMP]

Arguments:
    --input_dir INPUT_DIR     Directory containing comparison data files (default: /home/s_felix/flexseq/output/outputs_comparison)
    --output_dir OUTPUT_DIR   Directory to save visualization files (default: same as input_dir)
    --focus_temp FOCUS_TEMP   Focus temperature for single-temperature visualizations (default: 320)
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
# sns.set_style("whitegrid")
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.titlesize'] = 16
mpl.rcParams['figure.dpi'] = 300

# Define a colorblind-friendly palette
# Avoiding red-green combinations for better accessibility
# Mainly using blues, purples, and grays
CB_PALETTE = ['#4477AA', '#66CCEE', '#228833', '#CCBB44', '#EE6677', '#AA3377', '#BBBBBB']

# Mapping between temperature and a blue-to-purple color scale
# (using blues and purples instead of pure red for better accessibility)
TEMP_COLORS = {
    "320": "#083D77",  # Dark blue
    "348": "#5F84A2",  # Medium blue
    "379": "#9EADC8",  # Light blue/periwinkle
    "413": "#8B5FBF",  # Purple
    "450": "#6A0572",  # Dark purple
    "average": "#404040"  # Dark gray
}

# Define category folders
CATEGORIES = {
    "model_performance": "Model Performance Metrics",
    "correlation_analysis": "Temperature Correlation Analysis",
    "error_analysis": "Error Analysis",
    "structural_analysis": "Structural Analysis",
    "distribution": "Distribution Analysis",
    "amino_acid_analysis": "Amino Acid Analysis"
}

def load_data(input_dir):
    """
    Load all CSV files from the input directory.
    
    Args:
        input_dir: Directory containing CSV files
        
    Returns:
        Dictionary mapping filenames to pandas DataFrames
    """
    data = {}
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_dir, filename)
            try:
                # For correlation files, explicitly set the first column as index
                if filename in ['actual_correlations.csv', 'predicted_correlations.csv']:
                    df = pd.read_csv(file_path, index_col=0)
                else:
                    df = pd.read_csv(file_path)
                data[filename] = df
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return data

def ensure_category_dirs(output_dir):
    """
    Ensure category directories exist.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Dictionary mapping category keys to full paths
    """
    category_dirs = {}
    for key, name in CATEGORIES.items():
        category_path = os.path.join(output_dir, key)
        os.makedirs(category_path, exist_ok=True)
        category_dirs[key] = category_path
    return category_dirs

def save_figure(fig, filename, category, output_dirs, dpi=300):
    """
    Save figure to appropriate category directory.
    
    Args:
        fig: Matplotlib figure object
        filename: Output filename
        category: Category key for organization
        output_dirs: Dictionary mapping category keys to paths
        dpi: Resolution in dots per inch
    """
    output_path = os.path.join(output_dirs[category], filename)
    fig.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f"Saved {output_path}")
    

def plot_temperature_correlation_heatmaps(data, output_dirs, focus_temp=None):
    """
    Create heatmaps comparing RMSF correlations between temperatures.
    
    Args:
        data: Dictionary of loaded data
        output_dirs: Dictionary mapping category keys to paths
        focus_temp: Optional focus temperature for specific visualization
    """
    if 'actual_correlations.csv' in data and 'predicted_correlations.csv' in data:
        # Load correlation data
        actual_corr = data['actual_correlations.csv']
        predicted_corr = data['predicted_correlations.csv']
        
        # Create a single figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot actual correlations
        sns.heatmap(actual_corr, annot=True, cmap='Blues', vmin=0, vmax=1, 
                    ax=ax1, fmt='.2f', linewidths=.5)
        ax1.set_title('Actual RMSF Correlations Between Temperatures')
        
        # Plot predicted correlations
        sns.heatmap(predicted_corr, annot=True, cmap='Blues', vmin=0, vmax=1, 
                    ax=ax2, fmt='.2f', linewidths=.5)
        ax2.set_title('Predicted RMSF Correlations Between Temperatures')
        
        plt.tight_layout()
        save_figure(fig, 'temperature_correlations.png', 'correlation_analysis', output_dirs)
        
        # If focus temperature is provided, create a focused visualization
        if focus_temp is not None:
            focus_temp_str = str(focus_temp)
            if focus_temp_str in actual_corr.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Extract correlations for focus temperature
                focus_corr = actual_corr[focus_temp_str].copy()
                if 'average' in focus_corr.index:
                    # Handle 'average' specially if needed
                    pass
                
                # Sort values
                focus_corr = focus_corr.sort_values()
                
                # Create bar plot
                bars = ax.bar(focus_corr.index, focus_corr.values, 
                             color=[TEMP_COLORS.get(t, '#AAAAAA') for t in focus_corr.index])
                
                ax.set_title(f'Correlation with {focus_temp}K (Actual RMSF)')
                ax.set_xlabel('Temperature (K)')
                ax.set_ylabel('Correlation Coefficient')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                save_figure(fig, f'temperature_correlation_{focus_temp}K.png', 'correlation_analysis', output_dirs)
                
        # Create special plots showing correlation with average
        if 'average' in actual_corr.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Extract correlations with average
            avg_corr = actual_corr['average'].copy()
            avg_corr = avg_corr.sort_values()
            
            # Create bar plot
            bars = ax.bar(avg_corr.index, avg_corr.values,
                         color=[TEMP_COLORS.get(t, '#AAAAAA') for t in avg_corr.index])
            
            ax.set_title('Correlation with Average RMSF')
            ax.set_xlabel('Temperature (K)')
            ax.set_ylabel('Correlation Coefficient')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            save_figure(fig, 'temperature_correlation_with_average.png', 'correlation_analysis', output_dirs)
            
def plot_model_performance(data, output_dirs, focus_temp=None):
    """
    Plot model performance metrics across temperatures.
    
    Args:
        data: Dictionary of loaded data
        output_dirs: Dictionary mapping category keys to paths
        focus_temp: Optional focus temperature for specific visualization
    """
    if 'temperature_metrics.csv' in data:
        metrics_df = data['temperature_metrics.csv']
        
        # Convert temperature to string to ensure proper ordering
        metrics_df['temperature'] = metrics_df['temperature'].astype(str)
        
        # Create a figure with multiple subplots for different metrics
        metrics_to_plot = ['rmse', 'r2', 'pearson_correlation']
        fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 12), sharex=True)
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            # Create bar plot
            sns.barplot(x='temperature', y=metric, data=metrics_df, 
                         palette=[TEMP_COLORS.get(t, '#AAAAAA') for t in metrics_df['temperature']], 
                         ax=ax)
            
            # Set titles and labels
            metric_titles = {
                'rmse': 'Root Mean Squared Error',
                'r2': 'R² Score',
                'pearson_correlation': 'Pearson Correlation'
            }
            ax.set_title(f"{metric_titles.get(metric, metric)}")
            ax.set_ylabel('Score')
            
            if i == len(metrics_to_plot) - 1:
                ax.set_xlabel('Temperature (K)')
            else:
                ax.set_xlabel('')
        
        plt.tight_layout()
        save_figure(fig, 'model_performance_by_temperature.png', 'model_performance', output_dirs)
        
        # If focus temperature is provided, create a focused visualization
        if focus_temp is not None:
            # Create focused performance plot
            focus_metrics = metrics_df[metrics_df['temperature'] == str(focus_temp)]
            
            if not focus_metrics.empty:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Reorder metrics for better visualization
                focus_data = []
                for metric in metrics_to_plot:
                    metric_title = metric_titles.get(metric, metric)
                    focus_data.append({
                        'Metric': metric_title,
                        'Value': focus_metrics[metric].values[0]
                    })
                
                focus_df = pd.DataFrame(focus_data)
                
                # Create horizontal bar plot
                bars = ax.barh(focus_df['Metric'], focus_df['Value'], color=TEMP_COLORS.get(str(focus_temp), '#AAAAAA'))
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                            ha='left', va='center')
                
                ax.set_title(f'Model Performance at {focus_temp}K')
                ax.set_xlabel('Score')
                
                plt.tight_layout()
                save_figure(fig, f'model_performance_{focus_temp}K.png', 'model_performance', output_dirs)

def plot_amino_acid_temperature_response(data, output_dirs, focus_temp=None):
    """
    Plot amino acid response to temperature.
    
    Args:
        data: Dictionary of loaded data
        output_dirs: Dictionary mapping category keys to paths
        focus_temp: Optional focus temperature for specific visualization
    """
    if 'aa_responses.csv' in data:
        aa_df = data['aa_responses.csv']
        
        # Sort by average slope
        aa_df = aa_df.sort_values('avg_slope', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create barplot with error bars
        bars = ax.bar(aa_df['resname'], aa_df['avg_slope'], 
                yerr=aa_df['std_slope'], 
                color=CB_PALETTE[0], 
                alpha=0.7)
        
        # Add data points for R² values (secondary y-axis)
        ax2 = ax.twinx()
        ax2.scatter(range(len(aa_df)), aa_df['avg_r_squared'], 
                   color=CB_PALETTE[1], 
                   alpha=0.7, 
                   s=50, 
                   label='Avg R²')
        
        # Set labels and titles
        ax.set_xlabel('Amino Acid')
        ax.set_ylabel('Average Slope (RMSF/K)')
        ax2.set_ylabel('Average R²')
        ax.set_title('Amino Acid Response to Temperature')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add legend for the secondary axis
        ax2.legend(loc='upper right')
        
        # Set limits for secondary axis (R² typically between 0 and 1)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        save_figure(fig, 'amino_acid_temperature_response.png', 'amino_acid_analysis', output_dirs)

def plot_secondary_structure_errors(data, output_dirs, focus_temp=None):
    """
    Plot error distribution by secondary structure.
    
    Args:
        data: Dictionary of loaded data
        output_dirs: Dictionary mapping category keys to paths
        focus_temp: Optional focus temperature for specific visualization
    """
    if 'ss_error_summary_random_forest.csv' in data:
        ss_df = data['ss_error_summary_random_forest.csv']
        
        # Create a long-form dataframe for easier plotting
        error_cols = [col for col in ss_df.columns if 'mean_abs_error' in col]
        
        # Extract temperature values from column names
        temps = [col.split('_')[-1] for col in error_cols]
        
        # Prepare data for plotting
        plot_data = []
        for _, row in ss_df.iterrows():
            ss = row['secondary_structure_encoded']
            if ss == 0:
                ss_name = 'Helix'
            elif ss == 1:
                ss_name = 'Sheet'
            elif ss == 2:
                ss_name = 'Loop/Other'
            else:
                ss_name = f'Unknown ({ss})'
                
            for temp, col in zip(temps, error_cols):
                plot_data.append({
                    'Secondary Structure': ss_name,
                    'Temperature': temp,
                    'Mean Absolute Error': row[col]
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use a grouped bar plot
        sns.barplot(x='Secondary Structure', y='Mean Absolute Error', 
                   hue='Temperature', data=plot_df, 
                   palette=[TEMP_COLORS.get(t, '#AAAAAA') for t in temps],
                   ax=ax)
        
        ax.set_title('Error Distribution by Secondary Structure')
        ax.set_xlabel('Secondary Structure')
        ax.set_ylabel('Mean Absolute Error')
        
        plt.legend(title='Temperature (K)')
        plt.tight_layout()
        
        save_figure(fig, 'secondary_structure_errors.png', 'structural_analysis', output_dirs)
        
        # If focus temperature is provided, create a focused visualization
        if focus_temp is not None:
            # Filter data for focus temperature
            focus_df = plot_df[plot_df['Temperature'] == str(focus_temp)]
            
            if not focus_df.empty:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Create bar plot for focus temperature
                sns.barplot(x='Secondary Structure', y='Mean Absolute Error', 
                           data=focus_df, 
                           color=TEMP_COLORS.get(str(focus_temp), '#AAAAAA'),
                           ax=ax)
                
                ax.set_title(f'Error Distribution by Secondary Structure at {focus_temp}K')
                ax.set_xlabel('Secondary Structure')
                ax.set_ylabel('Mean Absolute Error')
                
                plt.tight_layout()
                save_figure(fig, f'secondary_structure_errors_{focus_temp}K.png', 'structural_analysis', output_dirs)

def plot_error_vs_position(data, output_dirs, focus_temp=None):
    """
    Plot error vs normalized residue position.
    
    Args:
        data: Dictionary of loaded data
        output_dirs: Dictionary mapping category keys to paths
        focus_temp: Optional focus temperature for specific visualization
    """
    if 'error_by_normalized_resid_bins_random_forest.csv' in data:
        pos_df = data['error_by_normalized_resid_bins_random_forest.csv']
        
        # Extract position bin centers
        pos_df['bin_center'] = (pos_df['bin_start'] + pos_df['bin_end']) / 2
        
        # Extract error columns and temperatures
        error_cols = [col for col in pos_df.columns if 'mean_abs_error' in col]
        temps = [col.split('_')[-1] for col in error_cols]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot a line for each temperature
        for temp, col in zip(temps, error_cols):
            ax.plot(pos_df['bin_center'], pos_df[col], 
                   marker='o', 
                   label=f'{temp}K', 
                   color=TEMP_COLORS.get(temp, '#AAAAAA'))
        
        ax.set_title('Error vs Normalized Residue Position')
        ax.set_xlabel('Normalized Residue Position')
        ax.set_ylabel('Mean Absolute Error')
        
        ax.legend(title='Temperature')
        plt.tight_layout()
        
        save_figure(fig, 'error_vs_position.png', 'error_analysis', output_dirs)
        
        # If focus temperature is provided, create a focused visualization
        if focus_temp is not None:
            # Find column for focus temperature
            focus_col = f'mean_abs_error_{focus_temp}'
            
            if focus_col in pos_df.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Create line plot for focus temperature
                ax.plot(pos_df['bin_center'], pos_df[focus_col], 
                       marker='o', 
                       color=TEMP_COLORS.get(str(focus_temp), '#AAAAAA'),
                       linewidth=2)
                
                ax.set_title(f'Error vs Normalized Residue Position at {focus_temp}K')
                ax.set_xlabel('Normalized Residue Position')
                ax.set_ylabel('Mean Absolute Error')
                
                plt.tight_layout()
                save_figure(fig, f'error_vs_position_{focus_temp}K.png', 'error_analysis', output_dirs)

def plot_error_vs_accessibility(data, output_dirs, focus_temp=None):
    """
    Plot error vs relative accessibility.
    
    Args:
        data: Dictionary of loaded data
        output_dirs: Dictionary mapping category keys to paths
        focus_temp: Optional focus temperature for specific visualization
    """
    if 'error_by_relative_accessibility_bins_random_forest.csv' in data:
        acc_df = data['error_by_relative_accessibility_bins_random_forest.csv']
        
        # Extract position bin centers
        acc_df['bin_center'] = (acc_df['bin_start'] + acc_df['bin_end']) / 2
        
        # Extract error columns and temperatures
        error_cols = [col for col in acc_df.columns if 'mean_abs_error' in col]
        temps = [col.split('_')[-1] for col in error_cols]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot a line for each temperature
        for temp, col in zip(temps, error_cols):
            ax.plot(acc_df['bin_center'], acc_df[col], 
                   marker='o', 
                   label=f'{temp}K', 
                   color=TEMP_COLORS.get(temp, '#AAAAAA'))
        
        ax.set_title('Error vs Relative Accessibility')
        ax.set_xlabel('Relative Accessibility')
        ax.set_ylabel('Mean Absolute Error')
        
        ax.legend(title='Temperature')
        plt.tight_layout()
        
        save_figure(fig, 'error_vs_accessibility.png', 'error_analysis', output_dirs)
        
        # If focus temperature is provided, create a focused visualization
        if focus_temp is not None:
            # Find column for focus temperature
            focus_col = f'mean_abs_error_{focus_temp}'
            
            if focus_col in acc_df.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Create line plot for focus temperature
                ax.plot(acc_df['bin_center'], acc_df[focus_col], 
                       marker='o', 
                       color=TEMP_COLORS.get(str(focus_temp), '#AAAAAA'),
                       linewidth=2)
                
                ax.set_title(f'Error vs Relative Accessibility at {focus_temp}K')
                ax.set_xlabel('Relative Accessibility')
                ax.set_ylabel('Mean Absolute Error')
                
                plt.tight_layout()
                save_figure(fig, f'error_vs_accessibility_{focus_temp}K.png', 'error_analysis', output_dirs)

def plot_domain_trends(data, output_dirs):
    """
    Plot domain statistics analysis.
    
    Args:
        data: Dictionary of loaded data
        output_dirs: Dictionary mapping category keys to paths
    """
    if 'domain_stats.csv' in data:
        domain_df = data['domain_stats.csv']
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot average slope vs R^2
        scatter = axes[0].scatter(domain_df['avg_slope'], domain_df['avg_r_squared'], 
                                 c=domain_df['outliers_count'], 
                                 cmap='viridis', 
                                 alpha=0.7,
                                 s=50)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[0])
        cbar.set_label('Outlier Count')
        
        axes[0].set_xlabel('Average Slope (RMSF/K)')
        axes[0].set_ylabel('Average R²')
        axes[0].set_title('Domain Temperature Response')
        
        # Create a histogram of slopes
        axes[1].hist(domain_df['avg_slope'], bins=20, 
                    alpha=0.7, 
                    color=CB_PALETTE[0])
        
        axes[1].set_xlabel('Average Slope (RMSF/K)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Distribution of Domain Temperature Response')
        
        plt.tight_layout()
        save_figure(fig, 'domain_trends_analysis.png', 'structural_analysis', output_dirs)

def plot_rmsf_histograms(data, output_dirs, focus_temp=None):
    """
    Plot RMSF distribution histograms.
    
    Args:
        data: Dictionary of loaded data
        output_dirs: Dictionary mapping category keys to paths
        focus_temp: Optional focus temperature for specific visualization
    """
    if 'histogram_data.csv' in data:
        hist_df = data['histogram_data.csv']
        
        # Get unique temperatures
        temps = hist_df['temperature'].unique()
        
        # Create a figure with subplots for each temperature
        n_temps = len(temps)
        fig, axes = plt.subplots(n_temps, 1, figsize=(10, 4*n_temps), sharex=True)
        
        # If only one temperature, axes will not be an array
        if n_temps == 1:
            axes = [axes]
        
        for i, temp in enumerate(temps):
            temp_data = hist_df[hist_df['temperature'] == temp]
            
            # Get actual and predicted data
            actual_data = temp_data[temp_data['type'] == 'actual']
            predicted_data = temp_data[temp_data['type'] == 'predicted_random_forest']
            
            # Plot bin centers
            bin_centers_actual = (actual_data['bin_start'] + actual_data['bin_end']) / 2
            bin_centers_pred = (predicted_data['bin_start'] + predicted_data['bin_end']) / 2
            
            # Plot histograms as step plots
            axes[i].step(bin_centers_actual, actual_data['count'], 
                        where='mid', 
                        label='Actual',
                        color=CB_PALETTE[0],
                        linewidth=2)
            
            axes[i].step(bin_centers_pred, predicted_data['count'], 
                        where='mid', 
                        label='Predicted',
                        color=CB_PALETTE[1],
                        linewidth=2)
            
            axes[i].set_title(f'RMSF Distribution at {temp}K')
            axes[i].set_ylabel('Count')
            axes[i].legend()
            
            if i == n_temps - 1:
                axes[i].set_xlabel('RMSF Value')
        
        plt.tight_layout()
        save_figure(fig, 'rmsf_histograms.png', 'distribution', output_dirs)
        
        # If focus temperature is provided, create a focused visualization
        if focus_temp is not None:
            # Filter data for focus temperature
            focus_data = hist_df[hist_df['temperature'] == str(focus_temp)]
            
            if not focus_data.empty:
                # Get actual and predicted data
                actual_data = focus_data[focus_data['type'] == 'actual']
                predicted_data = focus_data[focus_data['type'] == 'predicted_random_forest']
                
                # Plot bin centers
                bin_centers_actual = (actual_data['bin_start'] + actual_data['bin_end']) / 2
                bin_centers_pred = (predicted_data['bin_start'] + predicted_data['bin_end']) / 2
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot histograms as step plots
                ax.step(bin_centers_actual, actual_data['count'], 
                      where='mid', 
                      label='Actual',
                      color=CB_PALETTE[0],
                      linewidth=2)
                
                ax.step(bin_centers_pred, predicted_data['count'], 
                      where='mid', 
                      label='Predicted',
                      color=CB_PALETTE[1],
                      linewidth=2)
                
                ax.set_title(f'RMSF Distribution at {focus_temp}K')
                ax.set_xlabel('RMSF Value')
                ax.set_ylabel('Count')
                ax.legend()
                
                plt.tight_layout()
                save_figure(fig, f'rmsf_histogram_{focus_temp}K.png', 'distribution', output_dirs)

def plot_amino_acid_errors(data, output_dirs, focus_temp=None):
    """
    Plot error distribution by amino acid.
    
    Args:
        data: Dictionary of loaded data
        output_dirs: Dictionary mapping category keys to paths
        focus_temp: Optional focus temperature for specific visualization
    """
    if 'aa_error_summary_random_forest.csv' in data:
        aa_df = data['aa_error_summary_random_forest.csv']
        
        # Extract error columns and temperatures
        error_cols = [col for col in aa_df.columns if 'mean_abs_error' in col]
        temps = [col.split('_')[-1] for col in error_cols]
        
        # Sort amino acids by average error across temperatures
        aa_df['avg_error'] = aa_df[error_cols].mean(axis=1)
        aa_df = aa_df.sort_values('avg_error', ascending=False)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot grouped bars for each amino acid
        bar_width = 0.15
        x = np.arange(len(aa_df))
        
        for i, (temp, col) in enumerate(zip(temps, error_cols)):
            offset = bar_width * (i - len(temps)/2 + 0.5)
            ax.bar(x + offset, aa_df[col], 
                  width=bar_width, 
                  label=f'{temp}K',
                  color=TEMP_COLORS.get(temp, '#AAAAAA'))
        
        ax.set_title('Error Distribution by Amino Acid')
        ax.set_xlabel('Amino Acid')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_xticks(x)
        ax.set_xticklabels(aa_df['resname'], rotation=45)
        
        ax.legend(title='Temperature')
        plt.tight_layout()
        
        save_figure(fig, 'amino_acid_errors.png', 'amino_acid_analysis', output_dirs)
        
        # If focus temperature is provided, create a focused visualization
        if focus_temp is not None:
            # Find column for focus temperature
            focus_col = f'mean_abs_error_{focus_temp}'
            
            if focus_col in aa_df.columns:
                # Sort amino acids by error at focus temperature
                focus_aa_df = aa_df.sort_values(focus_col, ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create bar plot for focus temperature
                bars = ax.bar(focus_aa_df['resname'], focus_aa_df[focus_col], 
                             color=TEMP_COLORS.get(str(focus_temp), '#AAAAAA'))
                
                ax.set_title(f'Error Distribution by Amino Acid at {focus_temp}K')
                ax.set_xlabel('Amino Acid')
                ax.set_ylabel('Mean Absolute Error')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                save_figure(fig, f'amino_acid_errors_{focus_temp}K.png', 'amino_acid_analysis', output_dirs)
                
                
def plot_model_comparison_curves(data, output_dirs):
    """
    Plot model performance comparison across temperatures.
    
    Args:
        data: Dictionary of loaded data
        output_dirs: Dictionary mapping category keys to paths
    """
    if 'temperature_metrics.csv' in data:
        metrics_df = data['temperature_metrics.csv']
        
        # Convert temperature to string for proper sorting
        metrics_df['temperature'] = metrics_df['temperature'].astype(str)
        
        # Check if we have multiple models
        models = metrics_df['model'].unique()
        
        if len(models) > 1:
            # Plot metrics by temperature for each model
            metrics_to_plot = ['rmse', 'r2', 'pearson_correlation']
            
            for metric in metrics_to_plot:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot each model as a line
                for model in models:
                    model_data = metrics_df[metrics_df['model'] == model]
                    ax.plot(model_data['temperature'], model_data[metric], 
                           marker='o', 
                           label=model,
                           linewidth=2)
                
                metric_titles = {
                    'rmse': 'Root Mean Squared Error',
                    'r2': 'R² Score',
                    'pearson_correlation': 'Pearson Correlation'
                }
                
                ax.set_title(f"{metric_titles.get(metric, metric)} by Model and Temperature")
                ax.set_xlabel('Temperature (K)')
                ax.set_ylabel('Score')
                ax.legend(title='Model')
                
                plt.tight_layout()
                save_figure(fig, f'model_comparison_{metric}.png', 'model_performance', output_dirs)
        else:
            print("Only one model found in data, skipping model comparison")

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Generate visualizations from FlexSeq temperature comparison data')
    
    parser.add_argument('--input_dir', 
                        default='/home/s_felix/flexseq/output/outputs_comparison',
                        help='Directory containing comparison data files')
    
    parser.add_argument('--output_dir', 
                        default=None,
                        help='Directory to save visualization files (default: same as input_dir)')
    
    parser.add_argument('--focus_temp', 
                        default='320',
                        help='Focus temperature for specific visualizations (default: 320)')
    
    return parser.parse_args()

def main():
    """
    Main function to generate all visualizations.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Set output directory to input directory if not specified
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create category directories
    output_dirs = ensure_category_dirs(args.output_dir)
    
    # Load data from input directory
    print(f"Loading data from {args.input_dir}")
    data = load_data(args.input_dir)
    
    # Parse focus temperature
    focus_temp = args.focus_temp
    print(f"Focus temperature: {focus_temp}K")
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Temperature correlation heatmaps
    plot_temperature_correlation_heatmaps(data, output_dirs)
    
    # Model performance across temperatures
    plot_model_performance(data, output_dirs, focus_temp)
    
    # Amino acid response to temperature
    plot_amino_acid_temperature_response(data, output_dirs, focus_temp)
    
    # Error distribution by secondary structure
    plot_secondary_structure_errors(data, output_dirs, focus_temp)
    
    # Error vs normalized residue position
    plot_error_vs_position(data, output_dirs, focus_temp)
    
    # Error vs relative accessibility
    plot_error_vs_accessibility(data, output_dirs, focus_temp)
    
    # Domain trends analysis
    plot_domain_trends(data, output_dirs)
    
    # RMSF distribution histograms
    plot_rmsf_histograms(data, output_dirs, focus_temp)
    
    # Amino acid error distribution
    plot_amino_acid_errors(data, output_dirs, focus_temp)
    
    
    
    plot_model_comparison_curves(data, output_dirs)
    
    print(f"All visualizations saved to {args.output_dir} in organized folders")

if __name__ == "__main__":
    main()