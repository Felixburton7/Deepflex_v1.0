# -*- coding: utf-8 -*-
"""
Enhanced module for generating visualizations of processed mdCATH data.
Includes fixes, stylistic improvements, and additional plots.
"""
import glob
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, Optional, Any, List, Tuple

# Specific imports used in functions
from matplotlib.colors import LinearSegmentedColormap, Normalize
# from scipy.stats import kde # Removed as kdeplot is used directly
from scipy.ndimage import gaussian_filter1d
import h5py # Used in voxel info plot
# Import statsmodels now that it's installed (for lowess)
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.info("statsmodels not found. Plots using LOWESS smoothing will be skipped.")


# Import the constant from the structure properties module
try:
    from ..structure.properties import PDB_MAX_ASA
except ImportError:
    # Define fallback if run standalone or structure changes
    logging.warning("Could not import PDB_MAX_ASA, using fallback.")
    PDB_MAX_ASA = {
        'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0, 'CYS': 167.0,
        'GLN': 225.0, 'GLU': 223.0, 'GLY': 104.0, 'HIS': 224.0, 'ILE': 197.0,
        'LEU': 201.0, 'LYS': 236.0, 'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0,
        'SER': 155.0, 'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0,
        'UNK': 197.0
    }


# --- Plotting Configuration ---
DEFAULT_PALETTE = "colorblind"
DEFAULT_DPI = 300
plt.style.use('seaborn-v0_8-whitegrid') # Use a seaborn style globally

# --- Helper Functions ---
def _setup_plot_style(palette_name: Optional[str] = None):
    """Sets consistent Seaborn style and color palette."""
    # Style is set globally above, just handle palette here
    palette = palette_name or DEFAULT_PALETTE
    try:
        current_palette = sns.color_palette(palette)
        # sns.set_palette(current_palette) # Avoid resetting palette globally within function
        logging.debug(f"Using seaborn palette: {palette}")
        return current_palette
    except ValueError:
        logging.warning(f"Invalid palette '{palette}' specified. Using default '{DEFAULT_PALETTE}'.")
        current_palette = sns.color_palette(DEFAULT_PALETTE)
        # sns.set_palette(current_palette)
        return current_palette

def _save_plot(fig, output_dir: str, filename: str, dpi: int):
    """Handles saving the plot with directory creation."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        logging.info(f"Saved visualization: {path}")
        plt.close(fig)
    except Exception as e:
        logging.error(f"Failed to save plot {filename}: {e}", exc_info=logging.getLogger().isEnabledFor(logging.DEBUG))
        if 'fig' in locals() and plt.fignum_exists(fig.number):
             plt.close(fig)

# --- Plotting Functions ---

def create_temperature_summary_heatmap(rmsf_data: Dict[str, pd.DataFrame],
                                     output_dir: str,
                                     viz_config: Dict[str, Any]) -> Optional[str]:
    vis_dir = os.path.join(output_dir, "visualizations")
    dpi = viz_config.get('dpi', DEFAULT_DPI)
    palette = _setup_plot_style(viz_config.get('palette')) # Get palette but might not be used directly

    temps = [temp for temp in rmsf_data.keys() if temp.isdigit()]
    if not temps:
        logging.warning("No numeric temperature data available for heatmap")
        return None

    domain_ids = set()
    for temp in temps:
        if temp in rmsf_data and rmsf_data[temp] is not None and not rmsf_data[temp].empty:
             if 'domain_id' in rmsf_data[temp].columns:
                 domain_ids.update(rmsf_data[temp]["domain_id"].unique())
             else:
                 logging.warning(f"Missing 'domain_id' column in RMSF data for temp {temp}")

    if not domain_ids:
        logging.warning("No domain IDs found across temperature data for heatmap.")
        return None

    domain_ids = sorted(list(domain_ids))
    heatmap_data = []
    expected_rmsf_col_pattern = "rmsf_"

    for domain_id in domain_ids:
        domain_data = {"domain_id": domain_id}
        for temp in temps:
            if temp in rmsf_data and rmsf_data[temp] is not None:
                rmsf_col = f"rmsf_{temp}"
                domain_temp_data = rmsf_data[temp][rmsf_data[temp]["domain_id"] == domain_id]

                if not domain_temp_data.empty and rmsf_col in domain_temp_data.columns:
                    domain_data[temp] = domain_temp_data[rmsf_col].mean()
                else:
                    fallback_cols = [c for c in domain_temp_data.columns if c.startswith(expected_rmsf_col_pattern)]
                    if not domain_temp_data.empty and fallback_cols:
                        fallback_col = fallback_cols[0]
                        logging.warning(f"Column '{rmsf_col}' not found for T={temp}, Domain={domain_id}. Using fallback '{fallback_col}'.")
                        domain_data[temp] = domain_temp_data[fallback_col].mean()
                    else:
                        domain_data[temp] = np.nan

        heatmap_data.append(domain_data)

    if not heatmap_data:
        logging.warning("No data aggregated for heatmap")
        return None

    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df.columns = [int(c) if isinstance(c, str) and c.isdigit() else c for c in heatmap_df.columns]
    temp_cols = sorted([c for c in heatmap_df.columns if isinstance(c, int)])
    if not temp_cols:
        logging.warning("No numeric temperature columns found for heatmap pivot.")
        return None
    heatmap_pivot = heatmap_df.set_index("domain_id")[temp_cols]

    if heatmap_pivot.empty or heatmap_pivot.isnull().all().all():
         logging.warning("Heatmap pivot table is empty or all NaN.")
         return None

    fig_height = max(8, len(domain_ids) * 0.1)
    fig_width = max(10, len(temp_cols) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(heatmap_pivot, annot=False, cmap="viridis", ax=ax, cbar_kws={'label': 'Mean RMSF (nm)'})
    ax.set_title("Average RMSF by Domain and Temperature")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(f"Domains (n={len(domain_ids)})")

    if len(domain_ids) > 50:
        ax.set_yticks([])
    else:
        ax.tick_params(axis='y', labelsize=min(10, 400 / len(domain_ids)))

    ax.text(0.01, 0.01, f"Domains: {len(domain_ids)}\nTemps: {', '.join(map(str, temp_cols))}",
            transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    output_path = os.path.join(vis_dir, "temperature_summary_heatmap.png")
    _save_plot(fig, vis_dir, "temperature_summary_heatmap.png", dpi)
    return output_path


def create_temperature_average_summary(feature_df_average: pd.DataFrame,
                                     output_dir: str,
                                     viz_config: Dict[str, Any]) -> Optional[str]:
    vis_dir = os.path.join(output_dir, "visualizations")
    dpi = viz_config.get('dpi', DEFAULT_DPI)
    palette = _setup_plot_style(viz_config.get('palette'))

    if feature_df_average is None or feature_df_average.empty or 'rmsf_average' not in feature_df_average.columns:
        logging.warning("No temperature average data available for summary plot (expected in feature_df_average)")
        return None

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[2, 1])

    ax_density = plt.subplot(gs[0, 0])
    domain_stats = feature_df_average.groupby("domain_id")["rmsf_average"].mean().reset_index()
    domain_stats = domain_stats.rename(columns={'rmsf_average': 'mean_domain_rmsf'})

    if domain_stats.empty:
        logging.warning("No domain stats calculated for density plot.")
        plt.close(fig)
        return None

    rmsf_mean_q1, rmsf_mean_q3 = domain_stats["mean_domain_rmsf"].quantile([0.01, 0.99])
    density_plot_data = domain_stats[
        (domain_stats["mean_domain_rmsf"] >= rmsf_mean_q1) & (domain_stats["mean_domain_rmsf"] <= rmsf_mean_q3)
    ]["mean_domain_rmsf"]

    sns.kdeplot(density_plot_data, fill=True, color=palette[0], ax=ax_density, bw_adjust=0.5) # Use palette color
    sns.rugplot(density_plot_data, ax=ax_density, height=0.05, color=palette[1], alpha=0.5) # Use palette color

    quartiles = np.percentile(domain_stats["mean_domain_rmsf"], [25, 50, 75])
    labels = ["Q1", "Median", "Q3"]
    q_colors = ["green", "red", "green"]
    for q, label, color in zip(quartiles, labels, q_colors):
        ax_density.axvline(q, color=color, linestyle="--", alpha=0.7, linewidth=1.5)
        ax_density.text(q, ax_density.get_ylim()[1] * 0.9, f"{label}: {q:.4f}",
                     ha="center", va="top", fontsize=9,
                     bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"))

    stats_text = (
        f"Mean Domain Avg RMSF: {domain_stats['mean_domain_rmsf'].mean():.4f} nm\n"
        f"Std Dev: {domain_stats['mean_domain_rmsf'].std():.4f} nm\n"
        f"Range: {domain_stats['mean_domain_rmsf'].min():.4f} - {domain_stats['mean_domain_rmsf'].max():.4f} nm\n"
        f"Number of domains: {len(domain_stats)}"
    )
    ax_density.text(0.02, 0.98, stats_text, transform=ax_density.transAxes,
                 fontsize=10, va="top", ha="left",
                 bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"))
    ax_density.set_title("Distribution of Mean RMSF Across Domains (Avg Temp)", fontsize=14)
    ax_density.set_xlabel("Mean RMSF per Domain (nm)", fontsize=12)
    ax_density.set_ylabel("Density", fontsize=12)
    ax_density.set_xlim(left=max(0, rmsf_mean_q1 * 0.9), right=rmsf_mean_q3 * 1.1)

    ax_box = plt.subplot(gs[0, 1])
    rmsf_all_q1, rmsf_all_q3 = feature_df_average["rmsf_average"].quantile([0.01, 0.99])
    boxplot_data = feature_df_average[
        (feature_df_average["rmsf_average"] >= rmsf_all_q1) & (feature_df_average["rmsf_average"] <= rmsf_all_q3)
    ]
    sns.boxplot(y="rmsf_average", data=boxplot_data, ax=ax_box, color=palette[0], showfliers=False)
    stripplot_data = feature_df_average.sample(min(1000, len(feature_df_average)))
    sns.stripplot(y="rmsf_average", data=stripplot_data,
                  ax=ax_box, alpha=0.2, size=2, color=palette[1])
    ax_box.set_title("Overall Residue RMSF Distribution", fontsize=14)
    ax_box.set_ylabel("RMSF (nm)", fontsize=12)
    ax_box.set_xlabel("")
    ax_box.set_ylim(bottom=max(0, rmsf_all_q1 * 0.9), top=rmsf_all_q3 * 1.1)

    ax_size = plt.subplot(gs[1, 0])
    if "protein_size" in feature_df_average.columns:
        domain_size_info = feature_df_average[['domain_id', 'protein_size']].drop_duplicates()
        domain_size_rmsf = pd.merge(domain_stats, domain_size_info, on="domain_id", how="left")

        if not domain_size_rmsf.empty and 'protein_size' in domain_size_rmsf.columns and domain_size_rmsf['protein_size'].notna().any():
            scatter = ax_size.scatter(domain_size_rmsf["protein_size"],
                                   domain_size_rmsf["mean_domain_rmsf"],
                                   alpha=0.6, s=30, c=domain_size_rmsf["mean_domain_rmsf"],
                                   cmap="viridis")
            cbar = plt.colorbar(scatter, ax=ax_size)
            cbar.set_label("Mean Domain RMSF (nm)")

            try:
                valid_fit_data = domain_size_rmsf[['protein_size', 'mean_domain_rmsf']].dropna()
                x = valid_fit_data["protein_size"].astype(float)
                y = valid_fit_data["mean_domain_rmsf"].astype(float)
                if len(x) > 1:
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(x.min(), x.max(), 100)
                    ax_size.plot(x_range, p(x_range), "r--", linewidth=1.5, alpha=0.7)
                    corr = np.corrcoef(x, y)[0, 1]
                    ax_size.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax_size.transAxes,
                              fontsize=10, va="top", ha="left",
                              bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"))
            except Exception as e:
                logging.warning(f"Failed to create trend line for size vs RMSF: {e}")

            ax_size.set_title("Mean Domain RMSF vs Protein Size", fontsize=14)
            ax_size.set_xlabel("Number of Residues", fontsize=12)
            ax_size.set_ylabel("Mean Domain RMSF (nm)", fontsize=12)
        else:
             ax_size.text(0.5, 0.5, "Protein size data processed incorrectly or missing", ha="center", va="center", fontsize=12)
             ax_size.axis("off")
    else:
        ax_size.text(0.5, 0.5, "Protein size data not available in feature_df_average", ha="center", va="center", fontsize=12)
        ax_size.axis("off")

    ax_var = plt.subplot(gs[1, 1])
    domain_variability = feature_df_average.groupby("domain_id")["rmsf_average"].std().reset_index()
    domain_variability = domain_variability.rename(columns={'rmsf_average': 'rmsf_std'})

    if not domain_variability.empty and 'rmsf_std' in domain_variability.columns and domain_variability['rmsf_std'].notna().any():
        valid_std_data = domain_variability["rmsf_std"].dropna()
        std_q1, std_q3 = valid_std_data.quantile([0.01, 0.99])
        hist_std_data = valid_std_data[(valid_std_data >= std_q1) & (valid_std_data <= std_q3)]

        sns.histplot(hist_std_data, kde=True, ax=ax_var, color=palette[2], bins=30) # Use palette color
        median_std = valid_std_data.median()
        ax_var.axvline(median_std, color="red", linestyle="--", linewidth=1.5)
        ax_var.text(median_std, ax_var.get_ylim()[1] * 0.9, f"Median Std: {median_std:.4f}",
                  ha="center", va="top", fontsize=9,
                  bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"))
        ax_var.set_title("RMSF Variability Within Domains", fontsize=14)
        ax_var.set_xlabel("Std Dev of RMSF per Domain (nm)", fontsize=12)
        ax_var.set_ylabel("Count", fontsize=12)
        ax_var.set_xlim(left=max(0, std_q1*0.9), right=std_q3*1.1)
    else:
         ax_var.text(0.5, 0.5, "RMSF variability data not available", ha="center", va='center', fontsize=12)
         ax_var.axis("off")

    plt.tight_layout()
    output_path = os.path.join(vis_dir, "temperature_average_summary.png")
    _save_plot(fig, vis_dir, "temperature_average_summary.png", dpi)
    return output_path


def create_rmsf_distribution_plots(replica_avg_data: Dict[str, pd.DataFrame],
                                  overall_avg_data: Optional[pd.DataFrame],
                                  output_dir: str,
                                  viz_config: Dict[str, Any]) -> List[Optional[str]]:
    """
    Create distribution plots (violin, separate histograms) for RMSF by temperature.

    Args:
        replica_avg_data: Dict {temp: avg_replica_df}.
        overall_avg_data: DataFrame with overall average RMSF.
        output_dir: Base output directory.
        viz_config: Visualization config section.

    Returns:
        List[Optional[str]]: Paths to the saved figures (violin, separated histograms).
    """
    vis_dir = os.path.join(output_dir, "visualizations")
    dpi = viz_config.get('dpi', DEFAULT_DPI)
    palette_name = viz_config.get('palette', DEFAULT_PALETTE)
    palette = _setup_plot_style(palette_name)
    histogram_bins = viz_config.get('histogram_bins', 50) # Use configured bins

    saved_paths = []

    temps = sorted([temp for temp in replica_avg_data.keys() if temp.isdigit()], key=int)
    if not temps:
        logging.warning("No numeric temperature data available for RMSF distribution plots")
        return [None, None]

    # Prepare data for plotting
    dist_data = []
    for temp in temps:
        if temp in replica_avg_data and replica_avg_data[temp] is not None:
            temp_df = replica_avg_data[temp]
            rmsf_col = f"rmsf_{temp}"
            if rmsf_col in temp_df.columns:
                # Append data with temperature label
                temp_subset = temp_df[[rmsf_col]].copy()
                temp_subset['Temperature'] = temp # Keep as int for sorting
                temp_subset.rename(columns={rmsf_col: "RMSF"}, inplace=True)
                dist_data.append(temp_subset)

    if not dist_data:
        logging.warning("No valid RMSF data collected for distribution plots")
        return [None, None]

    dist_df = pd.concat(dist_data, ignore_index=True)
    # Ensure Temperature is treated categorically for plotting if needed
    dist_df['Temperature_Str'] = dist_df['Temperature'].astype(str) + "K"
    temp_order_str = [str(t)+"K" for t in temps]

    # 1. Create violin plot
    try:
        fig_violin, ax_violin = plt.subplots(figsize=(max(8, len(temps)*1.2), 6))
        violin_palette = sns.color_palette(palette_name, n_colors=len(temps))
        sns.violinplot(x="Temperature_Str", y="RMSF", data=dist_df, order=temp_order_str,
                       palette=violin_palette, ax=ax_violin, hue="Temperature_Str", legend=False)
        ax_violin.set_title("RMSF Distribution by Temperature")
        ax_violin.set_xlabel("Temperature (K)")
        ax_violin.set_ylabel("RMSF (nm)")
        ax_violin.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        violin_path = os.path.join(vis_dir, "rmsf_violin_plot.png")
        _save_plot(fig_violin, vis_dir, "rmsf_violin_plot.png", dpi)
        saved_paths.append(violin_path)
    except Exception as e:
        logging.error(f"Failed to create RMSF violin plot: {e}", exc_info=True)
        saved_paths.append(None)

    # 2. Create separate histograms
    try:
        # Determine grid size
        num_plots = len(temps) + (1 if overall_avg_data is not None and not overall_avg_data.empty else 0)
        ncols = 3
        nrows = (num_plots + ncols - 1) // ncols
        fig_hist, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), sharex=True, sharey=True)
        axs = axs.flatten()

        hist_palette = sns.color_palette(palette_name, n_colors=len(temps))

        for i, temp in enumerate(temps):
            ax = axs[i]
            temp_data = dist_df[dist_df["Temperature"] == temp]["RMSF"].dropna()
            if not temp_data.empty:
                sns.histplot(temp_data, kde=True, bins=histogram_bins, ax=ax, color=hist_palette[i], stat="density")
                ax.set_title(f"Temperature {temp}K")
                ax.set_xlabel("RMSF (nm)")
                ax.set_ylabel("Density" if i % ncols == 0 else "") # Label only first column y-axis

                mean_val = temp_data.mean()
                std_val = temp_data.std()
                ax.text(0.95, 0.95, f"Mean: {mean_val:.4f}\nStd: {std_val:.4f}",
                        transform=ax.transAxes, fontsize=9, # Smaller font
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                ax.axvline(mean_val, color='k', linestyle='--', linewidth=1) # Use black for mean line
            else:
                 ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                 ax.axis('off')


        # Add overall average histogram if available
        if overall_avg_data is not None and not overall_avg_data.empty and 'rmsf_average' in overall_avg_data.columns:
            ax = axs[len(temps)]
            avg_data = overall_avg_data["rmsf_average"].dropna()
            if not avg_data.empty:
                sns.histplot(avg_data, kde=True, bins=histogram_bins, ax=ax, color="grey", stat="density") # Neutral color
                ax.set_title("Average RMSF (All Temps)")
                ax.set_xlabel("RMSF (nm)")
                ax.set_ylabel("Density" if len(temps) % ncols == 0 else "") # Label y-axis if it starts a new row

                mean_val = avg_data.mean()
                std_val = avg_data.std()
                ax.text(0.95, 0.95, f"Mean: {mean_val:.4f}\nStd: {std_val:.4f}",
                        transform=ax.transAxes, fontsize=9,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                ax.axvline(mean_val, color='k', linestyle='--', linewidth=1)
            else:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                ax.axis('off')


        # Hide unused axes
        for j in range(num_plots, nrows * ncols):
            axs[j].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout
        fig_hist.suptitle("RMSF Distributions by Temperature", fontsize=16, y=0.99)
        hist_path = os.path.join(vis_dir, "rmsf_histogram_separated.png") # Changed filename
        _save_plot(fig_hist, vis_dir, "rmsf_histogram_separated.png", dpi)
        saved_paths.append(hist_path)

    except Exception as e:
        logging.error(f"Failed to create RMSF separated histograms: {e}", exc_info=True)
        saved_paths.append(None)

    return saved_paths


def create_amino_acid_rmsf_plot(feature_df_average: pd.DataFrame,
                              output_dir: str,
                              viz_config: Dict[str, Any]) -> Optional[str]:
    vis_dir = os.path.join(output_dir, "visualizations")
    dpi = viz_config.get('dpi', DEFAULT_DPI)
    palette_name = viz_config.get('palette', DEFAULT_PALETTE)
    _setup_plot_style(palette_name)

    if feature_df_average is None or feature_df_average.empty or 'rmsf_average' not in feature_df_average.columns or 'resname' not in feature_df_average.columns:
        logging.warning("No average feature data available for amino acid plot")
        return None

    plot_data = feature_df_average[['resname', 'rmsf_average']].copy()
    plot_data['resname'] = plot_data['resname'].replace({'HSD': 'HIS', 'HSE': 'HIS', 'HSP': 'HIS'})
    standard_aa = sorted(list(PDB_MAX_ASA.keys()))
    if 'UNK' in standard_aa: standard_aa.remove('UNK')
    plot_data = plot_data[plot_data['resname'].isin(standard_aa)]

    if plot_data.empty:
         logging.error("No standard amino acid data left for RMSF plot.")
         return None

    order = sorted(plot_data['resname'].unique())

    fig, ax = plt.subplots(figsize=(16, 7))
    num_colors_needed = len(order)
    plot_palette = sns.color_palette(palette_name, n_colors=num_colors_needed)

    sns.violinplot(data=plot_data, x='resname', y='rmsf_average', ax=ax,
                   hue='resname',
                   palette=plot_palette, order=order,
                   density_norm='width', cut=0, legend=False)

    sns.pointplot(data=plot_data, x='resname', y='rmsf_average', order=order, ax=ax,
                  color='black', markers='.', linestyle='none',
                  errorbar=None)

    ax.set_title('Average RMSF Distribution by Amino Acid Type')
    ax.set_xlabel('Amino Acid')
    ax.set_ylabel('Average RMSF (nm)')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    plt.tight_layout()

    output_path = os.path.join(vis_dir, "amino_acid_rmsf_violin_plot.png")
    _save_plot(fig, vis_dir, "amino_acid_rmsf_violin_plot.png", dpi)
    return output_path


def create_amino_acid_rmsf_plot_colored(feature_df_average: pd.DataFrame,
                                         output_dir: str,
                                         viz_config: Dict[str, Any]) -> Optional[str]:
    vis_dir = os.path.join(output_dir, "visualizations")
    dpi = viz_config.get('dpi', DEFAULT_DPI)
    _setup_plot_style(viz_config.get('palette'))

    if feature_df_average is None or feature_df_average.empty or 'rmsf_average' not in feature_df_average.columns or 'resname' not in feature_df_average.columns:
        logging.warning("No average feature data available for colored amino acid plot")
        return None

    avg_df = feature_df_average.copy()
    avg_df["resname"] = avg_df["resname"].apply(lambda x: "HIS" if x in ["HSE", "HSP", "HSD"] else x)

    aa_properties = {
        "ALA": {"color": "salmon", "type": "hydrophobic"}, "ARG": {"color": "royalblue", "type": "basic"},
        "ASN": {"color": "mediumseagreen", "type": "polar"}, "ASP": {"color": "crimson", "type": "acidic"},
        "CYS": {"color": "gold", "type": "special"}, "GLN": {"color": "mediumseagreen", "type": "polar"},
        "GLU": {"color": "crimson", "type": "acidic"}, "GLY": {"color": "lightgray", "type": "special"},
        "HIS": {"color": "cornflowerblue", "type": "basic"}, "ILE": {"color": "darksalmon", "type": "hydrophobic"},
        "LEU": {"color": "darksalmon", "type": "hydrophobic"}, "LYS": {"color": "royalblue", "type": "basic"},
        "MET": {"color": "orange", "type": "hydrophobic"}, "PHE": {"color": "chocolate", "type": "aromatic"},
        "PRO": {"color": "greenyellow", "type": "special"}, "SER": {"color": "mediumseagreen", "type": "polar"},
        "THR": {"color": "mediumseagreen", "type": "polar"}, "TRP": {"color": "chocolate", "type": "aromatic"},
        "TYR": {"color": "chocolate", "type": "aromatic"}, "VAL": {"color": "darksalmon", "type": "hydrophobic"}
    }
    standard_aa_props = list(aa_properties.keys())
    aa_df = avg_df[avg_df['resname'].isin(standard_aa_props)][['resname', 'rmsf_average']].copy()

    if aa_df.empty:
        logging.warning("No standard AA data remaining for colored plot.")
        return None

    all_residues_present = sorted(aa_df["resname"].unique())
    aa_stats = aa_df.groupby("resname")["rmsf_average"].agg(['mean', 'std', 'count']).reset_index()
    colors_dict = {aa: aa_properties.get(aa, {"color": "gray"})["color"] for aa in all_residues_present}

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.violinplot(x="resname", y="rmsf_average", data=aa_df, order=all_residues_present,
                   hue="resname", palette=colors_dict, inner="box", ax=ax, legend=False)

    for i, aa in enumerate(all_residues_present):
        stats = aa_stats[aa_stats["resname"] == aa]
        if not stats.empty:
            mean_val = stats.iloc[0]['mean']
            ax.scatter(i, mean_val, color='black', s=30, zorder=10)
            count = stats.iloc[0]['count']
            ax.annotate(f"n={int(count):,}", xy=(i, -0.05), xycoords=('data', 'axes fraction'),
                         ha='center', va='top', fontsize=8, rotation=90)


    type_colors = {
        "hydrophobic": "darksalmon", "polar": "mediumseagreen", "acidic": "crimson",
        "basic": "royalblue", "aromatic": "chocolate", "special": "gold"
    }
    legend_elements = [plt.Line2D([0], [0], color=color, marker='o', linestyle='', markersize=10, label=type_name)
                     for type_name, color in type_colors.items()]
    ax.legend(handles=legend_elements, title="Amino Acid Types", loc='upper right')

    plt.title("RMSF Distribution by Amino Acid Type (Colored by Property)", fontsize=14)
    plt.xlabel("Amino Acid")
    plt.ylabel("RMSF (nm)")
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    colored_output_path = os.path.join(vis_dir, "amino_acid_rmsf_colored.png")
    _save_plot(fig, vis_dir, "amino_acid_rmsf_colored.png", dpi)
    return colored_output_path


def create_replica_variance_plot(combined_rmsf_data: Dict[str, Dict[str, pd.DataFrame]],
                               output_dir: str,
                               viz_config: Dict[str, Any]) -> Optional[str]:
    vis_dir = os.path.join(output_dir, "visualizations")
    dpi = viz_config.get('dpi', DEFAULT_DPI)
    palette_name = viz_config.get('palette', DEFAULT_PALETTE)
    palette = _setup_plot_style(palette_name)

    variance_data = []
    temps = list(combined_rmsf_data.keys())
    if not temps:
        logging.warning("No combined RMSF data available for replica variance plot")
        return None

    for temp_str in temps:
        replica_dict = combined_rmsf_data.get(temp_str, {})
        if not replica_dict or len(replica_dict) < 2:
            logging.debug(f"Skipping T={temp_str} for variance plot: Not enough replicas ({len(replica_dict)}).")
            continue

        all_reps_df_list = []
        rmsf_col = f"rmsf_{temp_str}"
        for rep_str, df in replica_dict.items():
            if df is not None and not df.empty:
                 if rmsf_col in df.columns:
                      rep_df = df[['domain_id', 'resid', rmsf_col]].copy()
                      rep_df['replica'] = rep_str
                      all_reps_df_list.append(rep_df)
                 else:
                      logging.warning(f"RMSF column '{rmsf_col}' not found in replica data for {temp_str}, {rep_str}")

        if not all_reps_df_list: continue
        temp_combined_df = pd.concat(all_reps_df_list, ignore_index=True)

        grouped = temp_combined_df.groupby(['domain_id', 'resid'])
        stats = grouped[rmsf_col].agg(['mean', 'var', 'std', 'count'])
        stats = stats[stats['count'] > 1].reset_index()

        if stats.empty: continue

        stats['Temperature'] = temp_str
        stats['CV'] = (stats['std'] / stats['mean'].replace(0, np.nan)).fillna(0) * 100
        variance_data.append(stats)


    if not variance_data:
        logging.warning("No variance data calculated across replicas.")
        return None

    variance_df = pd.concat(variance_data, ignore_index=True)
    variance_df = variance_df.rename(columns={'mean': 'Mean_RMSF', 'var': 'Variance'})

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])
    ax_main = plt.subplot(gs[0, 0])
    colors = ["blue", "green", "yellow", "red"]
    cmap_name = "cv_colormap"
    cm = LinearSegmentedColormap.from_list(cmap_name, colors)

    plot_x = variance_df["Mean_RMSF"].replace([np.inf, -np.inf], np.nan).dropna()
    plot_y = variance_df["Variance"].replace([np.inf, -np.inf], np.nan).fillna(0).dropna()
    if plot_x.empty or plot_y.empty or len(plot_x) != len(plot_y):
         logging.warning("Insufficient valid data for hist2d in replica variance plot.")
         plt.close(fig)
         return None

    h = ax_main.hist2d(plot_x, plot_y, bins=50, cmap="Blues", alpha=0.8, cmin=1)
    plt.colorbar(h[3], ax=ax_main, label="Number of residues")

    high_cv_threshold = np.percentile(variance_df['CV'].dropna(), 90)
    high_cv = variance_df[(variance_df["CV"] > high_cv_threshold) & variance_df["CV"].notna()]
    if not high_cv.empty:
        scatter = ax_main.scatter(high_cv["Mean_RMSF"], high_cv["Variance"],
                                c=high_cv["CV"], cmap=cm, alpha=0.7, s=20, edgecolor='k', vmin=0, vmax=max(100, high_cv["CV"].max()))
        cb = plt.colorbar(scatter, ax=ax_main, label="CV (%) [Top 10%]")
    else:
         logging.info("No high CV outliers found for scatter overlay.")

    ax_main.set_title("RMSF Variance vs Mean RMSF (with density)")
    ax_main.set_xlabel("Mean RMSF (nm)")
    ax_main.set_ylabel("Variance of RMSF (nm²)")
    xlim_right = np.percentile(plot_x, 99.5) if not plot_x.empty else 1
    ylim_top = np.percentile(plot_y[plot_y > 0], 99.5) if any(plot_y > 0) else 0.1
    ax_main.set_xlim(left=0, right=xlim_right)
    ax_main.set_ylim(bottom=0, top=ylim_top)

    ax_right = plt.subplot(gs[0, 1], sharey=ax_main)
    ax_right.hist(plot_y, bins=50, orientation='horizontal', color=palette[0], alpha=0.7, range=(0, ax_main.get_ylim()[1])) # Use palette
    ax_right.set_xlabel("Count")
    ax_right.set_title("Variance Dist.")
    plt.setp(ax_right.get_yticklabels(), visible=False)

    ax_bottom = plt.subplot(gs[1, 0], sharex=ax_main)
    ax_bottom.hist(plot_x, bins=50, color=palette[0], alpha=0.7, range=ax_main.get_xlim()) # Use palette
    ax_bottom.set_ylabel("Count")
    ax_bottom.set_title("Mean RMSF Dist.")
    plt.setp(ax_bottom.get_xticklabels(), visible=False)

    ax_temp = plt.subplot(gs[1, 1])
    try:
        variance_df['Temperature_Num'] = pd.to_numeric(variance_df['Temperature'])
        temp_order = sorted(variance_df['Temperature'].unique(), key=lambda x: int(x))
    except ValueError:
        temp_order = sorted(variance_df['Temperature'].unique())

    n_temps = len(temp_order)
    temp_palette = sns.color_palette(palette_name, n_colors=n_temps)

    sns.boxplot(x="Temperature", y="Variance", data=variance_df, ax=ax_temp, order=temp_order,
                hue="Temperature", palette=temp_palette,
                showfliers=False, legend=False)

    ax_temp.set_title("Variance by Temperature")
    ax_temp.set_xlabel("Temperature (K)")
    ax_temp.set_ylabel("Variance (nm²)")
    ax_temp.tick_params(axis='x', rotation=45)
    ax_temp.set_ylim(bottom=0, top=ylim_top)

    total_residues = len(variance_df)
    high_cv_pct = len(high_cv) / total_residues * 100 if total_residues > 0 and not high_cv.empty else 0
    mean_cv_val = variance_df['CV'].mean()
    ax_main.text(0.01, 0.99,
               f"Total residues (var calculated): {total_residues:,}\n"
               f"High CV outliers (>{high_cv_threshold:.1f}%): {len(high_cv):,} ({high_cv_pct:.1f}%)\n"
               f"Mean CV: {mean_cv_val:.2f}%" if pd.notna(mean_cv_val) else "Mean CV: N/A",
               transform=ax_main.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    output_path = os.path.join(vis_dir, "replica_variance_plot.png")
    _save_plot(fig, vis_dir, "replica_variance_plot.png", dpi)
    return output_path


def create_dssp_rmsf_correlation_plot(feature_df_average: pd.DataFrame,
                                    output_dir: str,
                                    viz_config: Dict[str, Any]) -> Optional[str]:
    vis_dir = os.path.join(output_dir, "visualizations")
    dpi = viz_config.get('dpi', DEFAULT_DPI)
    palette_name = viz_config.get('palette', DEFAULT_PALETTE)
    palette = _setup_plot_style(palette_name)

    if feature_df_average is None or feature_df_average.empty:
        logging.warning("No average feature data available for DSSP correlation plot")
        return None
    required_cols = ["dssp", "rmsf_average"]
    if not all(col in feature_df_average.columns for col in required_cols):
         logging.warning(f"DSSP correlation plot requires columns: {required_cols}. Found: {feature_df_average.columns.tolist()}")
         return None

    avg_df = feature_df_average.copy()

    dssp_category_map = {
        'H': 'Alpha Helix', 'G': 'Alpha Helix', 'I': 'Alpha Helix',
        'E': 'Beta Sheet', 'B': 'Beta Sheet',
        'T': 'Turn/Bend', 'S': 'Turn/Bend',
        'C': 'Coil/Loop', ' ': 'Coil/Loop', '-': 'Coil/Loop'
    }
    avg_df['ss_category'] = avg_df['dssp'].apply(
        lambda x: dssp_category_map.get(str(x).upper(), 'Coil/Loop')
    ).astype('category')

    category_order = ['Alpha Helix', 'Beta Sheet', 'Turn/Bend', 'Coil/Loop']
    present_categories = [cat for cat in category_order if cat in avg_df['ss_category'].cat.categories]
    if not present_categories:
         logging.warning("No recognized secondary structure categories found for DSSP plot.")
         return None
    avg_df['ss_category'] = avg_df['ss_category'].cat.set_categories(present_categories)


    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.5])

    ax1 = plt.subplot(gs[0, 0:2])
    category_palette = {'Alpha Helix': 'crimson', 'Beta Sheet': 'royalblue', 'Turn/Bend': 'gold', 'Coil/Loop': 'mediumseagreen'}
    plot_palette = {k: v for k, v in category_palette.items() if k in present_categories}

    sns.violinplot(x='ss_category', y='rmsf_average', data=avg_df,
                 order=present_categories, ax=ax1, inner='quartile',
                 hue='ss_category', palette=plot_palette, legend=False)

    stats = avg_df.groupby('ss_category', observed=False)['rmsf_average'].agg(['mean', 'std', 'count'])
    stats = stats.reindex(present_categories)

    y_min, y_max = ax1.get_ylim()
    text_y_base = y_max * 0.95

    for i, cat in enumerate(present_categories):
        if cat in stats.index and pd.notna(stats.loc[cat, 'count']):
            row = stats.loc[cat]
            text_content = f"n={int(row['count']):,}\nMean={row['mean']:.4f}\nStd={row['std']:.4f}"
            ax1.text(i, text_y_base , text_content,
                   ha='center', va='top', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'), fontsize=8)
        else:
            ax1.text(i, text_y_base, "No Data", ha='center', va='top', fontsize=8)


    ax1.set_title("RMSF Distribution by Secondary Structure Category", fontsize=14)
    ax1.set_xlabel("Secondary Structure Type", fontsize=12)
    ax1.set_ylabel("RMSF (nm)", fontsize=12)

    ax2 = plt.subplot(gs[0, 2])
    category_counts = avg_df['ss_category'].value_counts().reindex(present_categories).fillna(0)
    bar_colors = [plot_palette.get(cat, 'gray') for cat in category_counts.index]
    bars = ax2.bar(range(len(category_counts)), category_counts, color=bar_colors)
    ax2.set_xticks(range(len(category_counts)))
    ax2.set_xticklabels(category_counts.index, rotation=45, ha='right')
    ax2.set_title("Residue Distribution by Structure Type", fontsize=14)
    ax2.set_ylabel("Number of Residues", fontsize=12)

    total = category_counts.sum()
    if total > 0:
        ax2.bar_label(bars, fmt=lambda x: f'{x/total:.1%}' if total > 0 else '0%', label_type='edge', fontsize=8, padding=3)

    ax3 = plt.subplot(gs[1, 0])
    valid_dssp_codes = avg_df['dssp'].replace([' ', '-'], np.nan).dropna()
    top_dssp = []
    dssp_subset = pd.DataFrame()
    if not valid_dssp_codes.empty:
        top_dssp = valid_dssp_codes.value_counts().head(10).index.tolist()
        dssp_subset = avg_df[avg_df['dssp'].isin(top_dssp)].copy()

    if not dssp_subset.empty:
        medians = dssp_subset.groupby('dssp')['rmsf_average'].median().sort_values(ascending=False)
        sorted_dssp = medians.index.tolist()

        dssp_colors = {
            'H': '#FF0000', 'G': '#FFA500', 'I': '#FFC0CB', 'E': '#0000FF', 'B': '#ADD8E6',
            'T': '#008000', 'S': '#FFFF00', 'C': '#808080', ' ': '#808080', '-': '#808080'
        }
        color_map_dict = {code: dssp_colors.get(code, 'gray') for code in sorted_dssp}

        sns.boxplot(x='dssp', y='rmsf_average', data=dssp_subset,
                    order=sorted_dssp, hue='dssp', palette=color_map_dict,
                    ax=ax3, showfliers=False, legend=False)

        ax3.set_title("RMSF by Specific DSSP Code (Top 10)", fontsize=14)
        ax3.set_xlabel("DSSP Code", fontsize=12)
        ax3.set_ylabel("RMSF (nm)", fontsize=12)
    else:
        ax3.text(0.5, 0.5, "No valid top DSSP codes found.", ha='center', va='center', fontsize=12)
        ax3.axis('off')

    ax4 = plt.subplot(gs[1, 1:])
    if "normalized_resid" in avg_df.columns:
        plot_data_heatmap = avg_df[['normalized_resid', 'ss_category', 'rmsf_average']].copy()
        try:
             plot_data_heatmap['position_bin'] = pd.qcut(plot_data_heatmap['normalized_resid'], 20,
                                                          labels=False, duplicates='drop')
             pivot_data = plot_data_heatmap.groupby(['position_bin', 'ss_category'], observed=False)['rmsf_average'].mean().reset_index()
             if not pivot_data.empty:
                 pivot_table = pivot_data.pivot(index='position_bin', columns='ss_category', values='rmsf_average')
                 pivot_table = pivot_table.reindex(columns=present_categories)

                 sns.heatmap(pivot_table, cmap="YlOrRd", annot=True, fmt=".3f", linewidths=.5,
                           cbar_kws={'label': 'Mean RMSF (nm)'}, ax=ax4, annot_kws={"size": 7})
                 ax4.set_title("RMSF by Structure Type and Relative Position", fontsize=14)
                 ax4.set_xlabel("Secondary Structure Type", fontsize=12)
                 ax4.set_ylabel("Normalized Residue Position (bins)", fontsize=12)
                 ax4.tick_params(axis='y', labelsize=8)
             else:
                  logging.warning("Pivot table for position vs structure heatmap is empty.")
                  ax4.text(0.5, 0.5, "Pivot table empty", ha='center', va='center', fontsize=12)
                  ax4.axis('off')

        except Exception as e:
             logging.warning(f"Could not generate position vs structure heatmap: {e}")
             ax4.text(0.5, 0.5, "Could not generate heatmap", ha='center', va='center', fontsize=12)
             ax4.axis('off')
    else:
        ax4.text(0.5, 0.5, "Normalized position data missing", ha='center', va='center', fontsize=12)
        ax4.axis('off')


    dssp_descriptions = {
        'H': 'α-helix', 'G': '3₁₀-helix', 'I': 'π-helix', 'E': 'β-strand', 'B': 'β-bridge',
        'T': 'Turn', 'S': 'Bend', 'C': 'Coil', ' ': 'Undefined', '-': 'Undefined'
    }
    codes_in_plot = set(top_dssp)
    description = "DSSP Codes:\n" + "\n".join([f"• {k}: {v}" for k, v in dssp_descriptions.items() if k in codes_in_plot or k in [' ', '-','C']])
    fig.text(0.5, 0.01, description, ha='center', va='bottom', fontsize=8, wrap=True,
               bbox=dict(facecolor='white', alpha=0.9, boxstyle='round'))

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    output_path = os.path.join(vis_dir, "dssp_rmsf_correlation_plot.png")
    _save_plot(fig, vis_dir, "dssp_rmsf_correlation_plot.png", dpi)
    return output_path


def create_feature_correlation_plot(feature_df_average: pd.DataFrame,
                                  output_dir: str,
                                  viz_config: Dict[str, Any]) -> Optional[str]:
    vis_dir = os.path.join(output_dir, "visualizations")
    dpi = viz_config.get('dpi', DEFAULT_DPI)
    _setup_plot_style(viz_config.get('palette'))

    if feature_df_average is None or feature_df_average.empty:
        logging.warning("No average feature data available for feature correlation plot")
        return None

    numerical_cols = []
    potential_cols = ['rmsf_average', 'rmsf_log', 'protein_size', 'normalized_resid',
                      'relative_accessibility', 'phi_norm', 'psi_norm',
                      'core_exterior_encoded', 'secondary_structure_encoded', 'resname_encoded']
    for col in potential_cols:
        if col in feature_df_average.columns and pd.api.types.is_numeric_dtype(feature_df_average[col]):
             numerical_cols.append(col)

    if len(numerical_cols) < 2:
        logging.warning(f"Not enough numerical feature columns ({len(numerical_cols)}) found for correlation plot")
        return None

    plot_data = feature_df_average[numerical_cols].copy()
    plot_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    plot_data.dropna(inplace=True)

    if len(plot_data) < 2:
        logging.warning("Not enough valid data points for correlation plot after handling NaNs.")
        return None

    corr_df = plot_data.corr(method='spearman')

    fig, ax = plt.subplots(figsize=(max(10, len(numerical_cols)*0.8), max(8, len(numerical_cols)*0.7)))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f",
               vmin=-1, vmax=1, center=0, ax=ax, linewidths=.5, annot_kws={"size": 8})
    ax.set_title("Spearman Correlation Between Features (Average Dataset)")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path = os.path.join(vis_dir, "feature_correlation_plot.png")
    _save_plot(fig, vis_dir, "feature_correlation_plot.png", dpi)
    return output_path


def create_frames_visualization(pdb_results: Dict[str, Any], config: Dict[str, Any],
                              # domain_results: Dict[str, Dict[str, Any]], # Not used here
                              output_dir: str,
                              viz_config: Dict[str, Any]) -> Optional[str]:
    logging.warning("Frame visualization currently uses SIMULATED/PLACEHOLDER data. Needs adaptation for real data.")
    vis_dir = os.path.join(output_dir, "visualizations")
    dpi = viz_config.get('dpi', DEFAULT_DPI)
    palette = _setup_plot_style(viz_config.get('palette'))

    frame_selection = config.get("processing", {}).get("frame_selection", {})
    method = frame_selection.get("method", "rmsd")
    num_frames = frame_selection.get("num_frames", 1)
    cluster_method = frame_selection.get("cluster_method", "kmeans")

    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 2, 2])

    ax_meta = plt.subplot(gs[0, :])
    selection_info = (
        f"Frame Selection Configuration:\n"
        f"• Method: {method}\n"
        f"• Frames per domain/temp/rep: {num_frames}\n"
        f"• Clustering (RMSD): {cluster_method}\n"
    )
    frame_base_dir = os.path.join(output_dir, "frames")
    total_domains = len(pdb_results) if pdb_results else 0 # Placeholder count
    saved_frame_files = glob.glob(os.path.join(frame_base_dir, "*", "*", "*.pdb"), recursive=True)
    domains_with_frames = set()
    if saved_frame_files:
        for f_path in saved_frame_files:
            fname = os.path.basename(f_path)
            parts = fname.split('_')
            if len(parts) >= 3 and parts[-1].endswith('.pdb'):
                domains_with_frames.add(parts[0])

    num_domains_with_frames = len(domains_with_frames)
    frame_percentage = (num_domains_with_frames / total_domains * 100) if total_domains > 0 else 0
    selection_info += f"• Domains processed (Est.): {total_domains}\n" # Mark as estimate
    selection_info += f"• Domains with saved frames: {num_domains_with_frames} ({frame_percentage:.1f}%)\n"
    selection_info += f"• Total frames saved: {len(saved_frame_files):,}"

    ax_meta.text(0.5, 0.5, selection_info, ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.2))
    ax_meta.set_title("Frame Selection Metadata", fontsize=14)
    ax_meta.axis('off')

    # Placeholder Panels
    for i in range(4):
        row, col = divmod(i, 2)
        ax = plt.subplot(gs[row + 1, col])
        ax.text(0.5, 0.5, f"Placeholder Panel {i+2}\n(Requires Real Data Analysis)",
                ha='center', va='center', fontsize=10, color='gray')
        ax.set_title(f"Analysis Panel {i+2}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle("Frame Analysis Summary (Placeholder Data)", fontsize=16, y=0.99)

    output_path = os.path.join(vis_dir, "frames_analysis.png")
    _save_plot(fig, vis_dir, "frames_analysis.png", dpi)
    return output_path


def create_ml_features_plot(feature_df_average: pd.DataFrame,
                          output_dir: str,
                          viz_config: Dict[str, Any]) -> Optional[str]:
    vis_dir = os.path.join(output_dir, "visualizations")
    dpi = viz_config.get('dpi', DEFAULT_DPI)
    palette_name = viz_config.get('palette', DEFAULT_PALETTE)
    palette = _setup_plot_style(palette_name)

    if feature_df_average is None or feature_df_average.empty:
        logging.warning("No average feature data available for ML features plot")
        return None

    avg_df = feature_df_average.copy()

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1.5, 1.5])

    ax_summary = plt.subplot(gs[0, :])
    feature_info = [
        f"ML Feature Dataset Overview (Avg Temp):",
        f"• Total residues: {len(avg_df):,}",
        f"• Unique domains: {avg_df['domain_id'].nunique()}",
        f"• Features: {', '.join([col for col in avg_df.columns if col not in ['domain_id', 'resid']])}"
    ]
    if "secondary_structure_encoded" in avg_df.columns:
        ss_dist = avg_df["secondary_structure_encoded"].value_counts(normalize=True)
        feature_info.append(f"• SS Dist: Helix {ss_dist.get(0, 0):.1%}, Sheet {ss_dist.get(1, 0):.1%}, Coil {ss_dist.get(2, 0):.1%}")
    if "core_exterior_encoded" in avg_df.columns:
        ce_dist = avg_df["core_exterior_encoded"].value_counts(normalize=True)
        feature_info.append(f"• Core/Ext Dist: Core {ce_dist.get(0, 0):.1%}, Exterior {ce_dist.get(1, 0):.1%}")
    ax_summary.text(0.5, 0.5, "\n".join(feature_info), ha='center', va='center', fontsize=10,
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))
    ax_summary.set_title("ML Features Overview (Average)", fontsize=14)
    ax_summary.axis('off')

    ax_corr = plt.subplot(gs[1, 0:2])
    corr_features = []
    potential_corr_cols = ["rmsf_average", "relative_accessibility", "normalized_resid",
                   "secondary_structure_encoded", "core_exterior_encoded", "protein_size",
                   "rmsf_log", "phi_norm", "psi_norm", "resname_encoded"]
    for col in potential_corr_cols:
        if col in avg_df.columns and pd.api.types.is_numeric_dtype(avg_df[col]):
             corr_features.append(col)

    if len(corr_features) > 1:
        plot_data_corr = avg_df[corr_features].copy()
        plot_data_corr.replace([np.inf, -np.inf], np.nan, inplace=True)
        plot_data_corr.dropna(inplace=True)
        if len(plot_data_corr) > 1:
             corr_matrix = plot_data_corr.corr(method='spearman')
             sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr, annot_kws={"size": 7})
             ax_corr.set_title("Feature Correlation (Spearman)")
             ax_corr.tick_params(axis='x', rotation=45, labelsize=8)
             ax_corr.tick_params(axis='y', labelsize=8)
        else:
             ax_corr.text(0.5, 0.5, "Not enough valid data for correlation", ha='center', va='center')
             ax_corr.axis('off')
    else:
        ax_corr.text(0.5, 0.5, "Not enough numeric features for correlation", ha='center', va='center')
        ax_corr.axis('off')

    ax_access = plt.subplot(gs[1, 2])
    if "relative_accessibility" in avg_df.columns and "rmsf_average" in avg_df.columns:
        sample_size = min(5000, len(avg_df))
        sample_df = avg_df.sample(sample_size, random_state=42)
        if "core_exterior" in sample_df.columns:
            hue_order = sorted([h for h in sample_df['core_exterior'].unique() if pd.notna(h)])
            plot_palette_dict = {"core": palette[0], "exterior": palette[1]}

            if all(item in plot_palette_dict for item in hue_order):
                sns.scatterplot(data=sample_df, x="relative_accessibility", y="rmsf_average",
                              hue="core_exterior", palette=plot_palette_dict, hue_order=hue_order,
                              alpha=0.4, s=10, ax=ax_access)
                handles, labels = ax_access.get_legend_handles_labels()
                if handles: ax_access.legend(handles=handles, labels=labels, title="Location", fontsize=8)
            else:
                 logging.warning("Core/Exterior values mismatch palette keys. Plotting without hue.")
                 sns.scatterplot(data=sample_df, x="relative_accessibility", y="rmsf_average",
                               alpha=0.4, s=10, ax=ax_access, color=palette[0])
        else:
             sns.scatterplot(data=sample_df, x="relative_accessibility", y="rmsf_average",
                           alpha=0.4, s=10, ax=ax_access, color=palette[0])

        ax_access.set_title(f"RMSF vs Rel. Accessibility")
        ax_access.set_xlabel("Relative Accessibility")
        ax_access.set_ylabel("RMSF (nm)")
        ax_access.set_xlim(left=0, right=1.0)
    else:
        ax_access.text(0.5, 0.5, "RMSF/Access. data missing", ha='center', va='center')
        ax_access.axis('off')

    ax_pos = plt.subplot(gs[2, 0])
    if "normalized_resid" in avg_df.columns and "rmsf_average" in avg_df.columns:
        sample_size = min(5000, len(avg_df))
        sample_df = avg_df.sample(sample_size, random_state=42)
        sns.scatterplot(data=sample_df, x="normalized_resid", y="rmsf_average",
                      alpha=0.4, s=10, ax=ax_pos, color=palette[2])
        if STATSMODELS_AVAILABLE:
            try:
                 sns.regplot(data=sample_df, x="normalized_resid", y="rmsf_average",
                             scatter=False, lowess=True, ax=ax_pos, line_kws={'color': 'red', 'lw': 1.5})
            except Exception as e:
                 logging.warning(f"Could not add LOWESS smoothed line to pos vs RMSF: {e}")
        else:
             logging.debug("statsmodels not found, skipping LOWESS smoothing.")
        ax_pos.set_title(f"RMSF vs Norm. Position")
        ax_pos.set_xlabel("Normalized Residue Position")
        ax_pos.set_ylabel("RMSF (nm)")
        ax_pos.set_xlim(left=-0.05, right=1.05)
    else:
        ax_pos.text(0.5, 0.5, "RMSF/Pos. data missing", ha='center', va='center')
        ax_pos.axis('off')

    ax_ss = plt.subplot(gs[2, 1])
    if "secondary_structure_encoded" in avg_df.columns and "rmsf_average" in avg_df.columns:
        ss_map = {0: "Helix", 1: "Sheet", 2: "Coil"}
        plot_data_ss = avg_df[['secondary_structure_encoded', 'rmsf_average']].copy()
        plot_data_ss["SS_Type"] = plot_data_ss["secondary_structure_encoded"].map(ss_map)
        ss_order = [ss_map[i] for i in sorted(ss_map.keys()) if ss_map[i] in plot_data_ss['SS_Type'].unique()]
        ss_palette_dict = {"Helix": palette[3], "Sheet": palette[4], "Coil": palette[5]}
        plot_ss_palette = {k: v for k, v in ss_palette_dict.items() if k in ss_order}

        if ss_order:
            sns.boxplot(data=plot_data_ss, x="SS_Type", y="rmsf_average", ax=ax_ss, order=ss_order,
                        hue="SS_Type", palette=plot_ss_palette,
                        showfliers=False, legend=False)
            ax_ss.set_title("RMSF Dist by SS Type")
            ax_ss.set_xlabel("Secondary Structure Type")
            ax_ss.set_ylabel("RMSF (nm)")
        else:
             ax_ss.text(0.5, 0.5, "No SS data to plot", ha='center', va='center')
             ax_ss.axis('off')
    else:
        ax_ss.text(0.5, 0.5, "RMSF/SS data missing", ha='center', va='center')
        ax_ss.axis('off')

    ax_ce = plt.subplot(gs[2, 2])
    if "core_exterior_encoded" in avg_df.columns and "rmsf_average" in avg_df.columns:
        ce_map = {0: "Core", 1: "Exterior"}
        ce_plot_map = {"Core": "core", "Exterior": "exterior"}
        plot_data_ce = avg_df[['core_exterior_encoded', 'rmsf_average']].copy()
        plot_data_ce["Location_Full"] = plot_data_ce["core_exterior_encoded"].map(ce_map)
        plot_data_ce["Location"] = plot_data_ce["Location_Full"].map(ce_plot_map)
        loc_order = [loc for loc in ["core", "exterior"] if loc in plot_data_ce['Location'].unique()]
        ce_palette_dict = {"core": palette[0], "exterior": palette[1]}
        plot_ce_palette = {k: v for k, v in ce_palette_dict.items() if k in loc_order}

        if loc_order:
            sns.boxplot(data=plot_data_ce, x="Location", y="rmsf_average", ax=ax_ce, order=loc_order,
                        hue="Location", palette=plot_ce_palette,
                        showfliers=False, legend=False)
            ax_ce.set_title("RMSF Dist by Location")
            ax_ce.set_xlabel("Residue Location")
            ax_ce.set_ylabel("RMSF (nm)")
        else:
            ax_ce.text(0.5, 0.5, "No Core/Ext data to plot", ha='center', va='center')
            ax_ce.axis('off')
    else:
        ax_ce.text(0.5, 0.5, "RMSF/CoreExt data missing", ha='center', va='center')
        ax_ce.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle("ML Features Analysis (Average Dataset)", fontsize=16, y=0.99)

    output_path = os.path.join(vis_dir, "ml_features_analysis.png")
    _save_plot(fig, vis_dir, "ml_features_analysis.png", dpi)
    return output_path


def create_summary_plot(replica_avg_data: Dict[str, pd.DataFrame],
                      feature_df_average: pd.DataFrame,
                      domain_status: Dict[str, str],
                      output_dir: str,
                      viz_config: Dict[str, Any]) -> Optional[str]:
    vis_dir = os.path.join(output_dir, "visualizations")
    dpi = viz_config.get('dpi', DEFAULT_DPI)
    palette_name = viz_config.get('palette', DEFAULT_PALETTE)
    palette = _setup_plot_style(palette_name)

    has_avg_feature_data = feature_df_average is not None and not feature_df_average.empty
    has_replica_avg_data = replica_avg_data is not None and bool(replica_avg_data)
    has_status_data = domain_status is not None and bool(domain_status)

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
    fig.suptitle("Pipeline Summary Report", fontsize=16, y=0.99)

    # Panel 1: Processing Status
    ax_status = plt.subplot(gs[0, 0])
    ax_status.set_title("Processing Status", fontsize=12)
    if has_status_data:
        status_series = pd.Series(domain_status)
        status_counts = status_series.value_counts()
        defined_order = ['Success', 'Failed HDF5 Read/Access', 'Failed PDB Read', 'Failed PDB Clean',
                         'Failed Properties Calc', 'Failed RMSF Proc', 'Failed Unexpected', 'Failed Component Init']
        present_statuses = status_counts.index.tolist()
        status_order = [s for s in defined_order if s in present_statuses] + \
                       [s for s in present_statuses if s not in defined_order]
        color_map = {
            'Success': 'mediumseagreen', 'Failed HDF5 Read/Access': 'tomato',
            'Failed PDB Read': 'tomato', 'Failed PDB Clean': 'tomato',
            'Failed Properties Calc': 'tomato', 'Failed RMSF Proc': 'tomato',
            'Failed Unexpected': 'darkred', 'Failed Component Init': 'salmon'
        }
        status_colors = [color_map.get(s, 'lightgray') for s in status_order]

        ax_status.pie(status_counts[status_order], labels=status_order, autopct='%1.1f%%',
                      startangle=90, colors=status_colors, textprops={'fontsize': 8}, pctdistance=0.85)
        ax_status.axis('equal')
    else:
        ax_status.text(0.5, 0.5, "No Status Data", ha='center', va='center')
        ax_status.axis('off')

    # Panel 2: RMSF Distribution (Overall Average)
    ax_rmsf_hist = plt.subplot(gs[0, 1])
    ax_rmsf_hist.set_title("Avg RMSF Distribution", fontsize=12)
    if has_avg_feature_data and 'rmsf_average' in feature_df_average.columns:
        avg_rmsf = feature_df_average["rmsf_average"].dropna()
        if not avg_rmsf.empty:
            sns.histplot(avg_rmsf, bins=30, kde=True, ax=ax_rmsf_hist, color=palette[0])
            mean_rmsf = avg_rmsf.mean()
            ax_rmsf_hist.axvline(mean_rmsf, color='r', linestyle='--', lw=1, label=f'Mean: {mean_rmsf:.3f}')
            ax_rmsf_hist.legend(fontsize=8)
            ax_rmsf_hist.set_xlabel("RMSF (nm)", fontsize=9)
            ax_rmsf_hist.set_ylabel("Count", fontsize=9)
        else:
             ax_rmsf_hist.text(0.5, 0.5, "No RMSF Values", ha='center', va='center')
             ax_rmsf_hist.axis('off')
    else:
        ax_rmsf_hist.text(0.5, 0.5, "No Avg RMSF Data", ha='center', va='center')
        ax_rmsf_hist.axis('off')

    # Panel 3: RMSF Temperature Trend
    ax_temp_trend = plt.subplot(gs[0, 2])
    ax_temp_trend.set_title("RMSF vs Temperature", fontsize=12)
    if has_replica_avg_data:
        temps = sorted([t for t in replica_avg_data.keys() if t.isdigit()], key=int)
        temp_rmsf_means = []
        temp_rmsf_stds = []
        valid_temps_for_plot = []
        for temp in temps:
            rmsf_col = f"rmsf_{temp}"
            if temp in replica_avg_data and replica_avg_data[temp] is not None and rmsf_col in replica_avg_data[temp].columns:
                 rmsf_vals = replica_avg_data[temp][rmsf_col].dropna()
                 if not rmsf_vals.empty:
                      temp_rmsf_means.append(rmsf_vals.mean())
                      temp_rmsf_stds.append(rmsf_vals.std())
                      valid_temps_for_plot.append(str(temp)) # Keep as string

        if temp_rmsf_means:
            ax_temp_trend.errorbar(valid_temps_for_plot, temp_rmsf_means, yerr=temp_rmsf_stds,
                                fmt='o-', capsize=3, color=palette[1], markersize=4, elinewidth=1, alpha=0.8)
            ax_temp_trend.set_xlabel("Temperature (K)", fontsize=9)
            ax_temp_trend.set_ylabel("Mean RMSF (nm)", fontsize=9)
            ax_temp_trend.tick_params(axis='x', rotation=45)
        else:
             ax_temp_trend.text(0.5, 0.5, "No Temp RMSF Data Calculated", ha='center', va='center')
             ax_temp_trend.axis('off')
    else:
        ax_temp_trend.text(0.5, 0.5, "No Temp RMSF Data Found", ha='center', va='center')
        ax_temp_trend.axis('off')

    # Panel 4: Secondary Structure Distribution
    ax_ss_pie = plt.subplot(gs[1, 0])
    ax_ss_pie.set_title("Secondary Structure", fontsize=12)
    if has_avg_feature_data and "secondary_structure_encoded" in feature_df_average.columns:
        ss_map = {0: "Helix", 1: "Sheet", 2: "Coil"}
        ss_counts = feature_df_average["secondary_structure_encoded"].map(ss_map).value_counts()
        ss_colors = {'Helix': palette[2], 'Sheet': palette[3], 'Coil': palette[4]}
        pie_colors = [ss_colors.get(lbl, 'grey') for lbl in ss_counts.index]
        ax_ss_pie.pie(ss_counts, labels=ss_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=pie_colors,
                   textprops={'fontsize': 8})
        ax_ss_pie.axis('equal')
    else:
        ax_ss_pie.text(0.5, 0.5, "No SS Data", ha='center', va='center')
        ax_ss_pie.axis('off')

    # Panel 5: Core/Exterior Distribution
    ax_ce_pie = plt.subplot(gs[1, 1])
    ax_ce_pie.set_title("Core vs Exterior", fontsize=12)
    if has_avg_feature_data and "core_exterior_encoded" in feature_df_average.columns:
        ce_map = {0: "Core", 1: "Exterior"}
        plot_data_ce_pie = feature_df_average["core_exterior_encoded"].map(ce_map)
        ce_counts = plot_data_ce_pie.value_counts()
        ce_colors = {'Core': palette[5], 'Exterior': palette[6]}
        pie_colors_ce = [ce_colors.get(lbl, 'grey') for lbl in ce_counts.index]
        ax_ce_pie.pie(ce_counts, labels=ce_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=pie_colors_ce,
                   textprops={'fontsize': 8})
        ax_ce_pie.axis('equal')
    else:
        ax_ce_pie.text(0.5, 0.5, "No Core/Ext Data", ha='center', va='center')
        ax_ce_pie.axis('off')

    # Panel 6: RMSF vs Accessibility Scatter
    ax_rmsf_access = plt.subplot(gs[1, 2])
    ax_rmsf_access.set_title("RMSF vs Access.", fontsize=12)
    if has_avg_feature_data and "rmsf_average" in feature_df_average.columns and "relative_accessibility" in feature_df_average.columns:
        sample_size = min(2000, len(feature_df_average))
        sample_df = feature_df_average.sample(sample_size, random_state=42)
        sns.scatterplot(data=sample_df, x="relative_accessibility", y="rmsf_average",
                      alpha=0.3, s=5, ax=ax_rmsf_access, color=palette[7])
        ax_rmsf_access.set_xlabel("Rel. Accessibility", fontsize=9)
        ax_rmsf_access.set_ylabel("RMSF (nm)", fontsize=9)
        ax_rmsf_access.set_xlim(0, 1)
    else:
        ax_rmsf_access.text(0.5, 0.5, "No RMSF/Access. Data", ha='center', va='center')
        ax_rmsf_access.axis('off')

    # Panel 7: Top Amino Acids Bar Chart
    ax_aa_bar = plt.subplot(gs[2, 0])
    ax_aa_bar.set_title("Top 5 Amino Acids", fontsize=12)
    if has_avg_feature_data and "resname" in feature_df_average.columns:
        aa_counts = feature_df_average["resname"].value_counts().head(5)
        bar_colors_aa = palette[:len(aa_counts)]
        # *** Corrected bar plot and label call ***
        bars_aa = ax_aa_bar.bar(aa_counts.index, aa_counts.values, color=bar_colors_aa)
        ax_aa_bar.bar_label(bars_aa, fmt='{:,.0f}') # Use the container returned by ax.bar
        ax_aa_bar.set_xlabel("Amino Acid", fontsize=9)
        ax_aa_bar.set_ylabel("Count", fontsize=9)
        ax_aa_bar.tick_params(axis='x', labelsize=8)
    else:
        ax_aa_bar.text(0.5, 0.5, "No Resname Data", ha='center', va='center')
        ax_aa_bar.axis('off')

    # Panel 8: RMSF by SS Type Bar Chart
    ax_ss_rmsf = plt.subplot(gs[2, 1])
    ax_ss_rmsf.set_title("RMSF by SS Type", fontsize=12)
    if has_avg_feature_data and "secondary_structure_encoded" in feature_df_average.columns and "rmsf_average" in feature_df_average.columns:
        ss_map = {0: "Helix", 1: "Sheet", 2: "Coil"}
        plot_data_ss = feature_df_average[['secondary_structure_encoded', 'rmsf_average']].copy()
        plot_data_ss["SS_Type"] = plot_data_ss["secondary_structure_encoded"].map(ss_map)
        ss_rmsf_means = plot_data_ss.groupby("SS_Type", observed=False)['rmsf_average'].mean().reindex(["Helix", "Sheet", "Coil"]).fillna(0)
        ss_colors = {'Helix': palette[2], 'Sheet': palette[3], 'Coil': palette[4]}
        bar_colors = [ss_colors.get(idx, 'grey') for idx in ss_rmsf_means.index]
        # *** Corrected bar plot and label call ***
        bars_ss = ax_ss_rmsf.bar(ss_rmsf_means.index, ss_rmsf_means.values, color=bar_colors)
        ax_ss_rmsf.bar_label(bars_ss, fmt='%.4f', fontsize=8)
        ax_ss_rmsf.set_xlabel("Secondary Structure", fontsize=9)
        ax_ss_rmsf.set_ylabel("Mean RMSF (nm)", fontsize=9)
        ax_ss_rmsf.tick_params(axis='x', rotation=0, labelsize=8)
    else:
        ax_ss_rmsf.text(0.5, 0.5, "No RMSF/SS Data", ha='center', va='center')
        ax_ss_rmsf.axis('off')

    # Panel 9: RMSF by Core/Exterior Bar Chart
    ax_ce_rmsf = plt.subplot(gs[2, 2])
    ax_ce_rmsf.set_title("RMSF by Location", fontsize=12)
    if has_avg_feature_data and "core_exterior" in feature_df_average.columns and "rmsf_average" in feature_df_average.columns:
        plot_data_ce = feature_df_average[['core_exterior', 'rmsf_average']].copy()
        ce_rmsf_means = plot_data_ce.groupby("core_exterior", observed=False)['rmsf_average'].mean().reindex(["core", "exterior"]).fillna(0)
        ce_colors = {'core': palette[5], 'exterior': palette[6]}
        bar_colors_ce = [ce_colors.get(idx, 'grey') for idx in ce_rmsf_means.index]
        # *** Corrected bar plot and label call ***
        bars_ce = ax_ce_rmsf.bar(ce_rmsf_means.index, ce_rmsf_means.values, color=bar_colors_ce)
        ax_ce_rmsf.bar_label(bars_ce, fmt='%.4f', fontsize=8)
        ax_ce_rmsf.set_xlabel("Location", fontsize=9)
        ax_ce_rmsf.set_ylabel("Mean RMSF (nm)", fontsize=9)
        ax_ce_rmsf.tick_params(axis='x', rotation=0, labelsize=8)
    else:
        ax_ce_rmsf.text(0.5, 0.5, "No RMSF/CoreExt Data", ha='center', va='center')
        ax_ce_rmsf.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = os.path.join(vis_dir, "pipeline_summary_report.png")
    _save_plot(fig, vis_dir, "pipeline_summary_report.png", dpi)
    return output_path


def create_voxel_info_plot(config: Dict[str, Any],
                           voxel_output_file: Optional[str],
                           output_dir: str,
                           viz_config: Dict[str, Any]) -> Optional[str]:
    vis_dir = os.path.join(output_dir, "visualizations")
    dpi = viz_config.get('dpi', DEFAULT_DPI)
    _setup_plot_style(viz_config.get('palette'))

    voxel_config = config.get("processing", {}).get("voxelization", {})
    if not voxel_config.get("enabled", True):
        logging.info("Voxelization was disabled, skipping parameter summary plot.")
        return None

    fig = plt.figure(figsize=(10, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 1])

    ax_config = plt.subplot(gs[0])
    summary_lines = ["Voxelization Parameters (Aposteriori):"]
    try:
         resolution = f"{voxel_config.get('frame_edge_length', 0) / voxel_config.get('voxels_per_side', 1):.2f} Å/voxel"
    except (TypeError, ValueError, ZeroDivisionError):
         resolution = "N/A"

    params = [
        ('Enabled', voxel_config.get("enabled", False)),
        ('Executable Used', voxel_config.get("aposteriori_executable") or "make-frame-dataset (PATH)"),
        ('Output File', os.path.basename(voxel_output_file) if voxel_output_file and os.path.exists(voxel_output_file) else "Not Generated/Found"),
        ('Frame Edge Length', f"{voxel_config.get('frame_edge_length', 'N/A')} Å"),
        ('Voxels Per Side', voxel_config.get('voxels_per_side', 'N/A')),
        ('Resolution', resolution),
        ('Atom Encoder', voxel_config.get('atom_encoder', 'N/A')),
        ('Encode CB', voxel_config.get('encode_cb', 'N/A')),
        ('Compression (Gzip)', voxel_config.get('compression_gzip', 'N/A')),
        ('Voxelise All NMR States', voxel_config.get('voxelise_all_states', 'N/A')),
    ]
    for name, value in params:
        summary_lines.append(f"• {name}: {value}")

    if voxel_output_file and os.path.exists(voxel_output_file):
         try:
             size_mb = os.path.getsize(voxel_output_file) / (1024 * 1024)
             summary_lines.append(f"\n• Output File Size: {size_mb:.2f} MB")
             try:
                 with h5py.File(voxel_output_file, 'r') as f:
                     num_domains = len(list(f.keys()))
                     num_residues = 0
                     for domain_key in f:
                         if isinstance(f[domain_key], h5py.Group):
                             for chain_key in f[domain_key]:
                                  if isinstance(f[f"{domain_key}/{chain_key}"], h5py.Group):
                                       if 'residue_idx' in f[f"{domain_key}/{chain_key}"]:
                                            num_residues += len(f[f"{domain_key}/{chain_key}/residue_idx"][:])
                                       else:
                                             num_residues += len([k for k in f[f"{domain_key}/{chain_key}"] if k.isdigit()])
                     summary_lines.append(f"• Domains in Output: {num_domains}")
                     summary_lines.append(f"• Residues in Output: {num_residues:,}")
             except ImportError:
                 summary_lines.append("• (h5py not installed - cannot read content stats)")
             except Exception as h5_err:
                 summary_lines.append(f"• (Error reading HDF5 content: {h5_err})")
         except Exception as e:
              summary_lines.append(f"• (Could not get file stats: {e})")

    summary_text = "\n".join(summary_lines)
    ax_config.text(0.05, 0.95, summary_text, transform=ax_config.transAxes, fontsize=9,
                   verticalalignment='top', family='monospace')
    ax_config.axis('off')
    ax_config.set_title('Voxelization Parameters & Output', fontsize=12)

    ax_slice = plt.subplot(gs[1])
    ax_slice.set_title("Illustrative Voxel Slice (Example)", fontsize=12)
    try:
        grid_size = int(voxel_config.get('voxels_per_side', 21))
        fake_slice = np.zeros((grid_size, grid_size))
        center = grid_size // 2
        if center > 0:
            fake_slice[center, center] = 1
            if center+1 < grid_size and center-1 >= 0: fake_slice[center+1, center-1] = 0.8
            if center-1 >= 0: fake_slice[center-1, center-1] = 0.7
            if center+2 < grid_size: fake_slice[center, center+2] = 0.6
        if grid_size > 5:
            fake_slice = gaussian_filter1d(fake_slice, sigma=1.0)

        im = ax_slice.imshow(fake_slice, cmap='viridis', origin='lower')
        plt.colorbar(im, ax=ax_slice, label='Atom Density (Illustrative)')
        step = max(1, grid_size // 5)
        ax_slice.set_xticks(np.arange(-.5, grid_size-.5, step))
        ax_slice.set_yticks(np.arange(-.5, grid_size-.5, step))
        ax_slice.set_xticklabels(np.arange(0, grid_size, step))
        ax_slice.set_yticklabels(np.arange(0, grid_size, step))
        ax_slice.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.3)
        ax_slice.set_xlabel("Voxel Index (X)")
        ax_slice.set_ylabel("Voxel Index (Y)")
    except Exception as plot_err:
         logging.warning(f"Could not generate illustrative voxel slice: {plot_err}")
         ax_slice.text(0.5, 0.5, "Could not generate slice", ha='center', va='center')
         ax_slice.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_path = os.path.join(vis_dir, "voxelization_info.png")
    _save_plot(fig, vis_dir, "voxelization_info.png", dpi)
    return output_path


def create_additional_ml_features_plot(feature_df_average: pd.DataFrame,
                                     output_dir: str,
                                     viz_config: Dict[str, Any]) -> Optional[str]:
    vis_dir = os.path.join(output_dir, "visualizations")
    dpi = viz_config.get('dpi', DEFAULT_DPI)
    palette_name = viz_config.get('palette', DEFAULT_PALETTE)
    palette = _setup_plot_style(palette_name)

    if feature_df_average is None or feature_df_average.empty:
        logging.warning("No average feature data available for additional ML features plot")
        return None

    avg_df = feature_df_average.copy()

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1.5, 1.5])
    fig.suptitle("Additional Features Analysis (Average Dataset)", fontsize=16, y=0.99)

    ax_importance = plt.subplot(gs[0, :])
    ax_importance.set_title("Feature Correlation with RMSF (Spearman)", fontsize=12)
    features_to_correlate = []
    potential_features = [col for col in avg_df.columns
                          if col not in ['domain_id', 'resid', 'resname', 'chain', 'dssp', 'core_exterior',
                                         'rmsf_average', 'rmsf_log']]
    for col in potential_features:
         if col in avg_df.columns and pd.api.types.is_numeric_dtype(avg_df[col]):
             features_to_correlate.append(col)

    if "rmsf_average" in avg_df.columns and features_to_correlate:
        corrs = []
        temp_df_corr = avg_df[['rmsf_average'] + features_to_correlate].copy()
        temp_df_corr.replace([np.inf, -np.inf], np.nan, inplace=True)
        temp_df_corr.dropna(inplace=True)

        if len(temp_df_corr) > 1:
            for feat in features_to_correlate:
                try:
                    corr = temp_df_corr[feat].corr(temp_df_corr["rmsf_average"], method='spearman')
                    corrs.append((feat, corr))
                except Exception as corr_err:
                     logging.debug(f"Could not calculate Spearman correlation for {feat}: {corr_err}")
                     corrs.append((feat, np.nan))

            corrs = sorted([c for c in corrs if pd.notna(c[1])], key=lambda x: abs(x[1]), reverse=True)

            if corrs:
                feat_names = [name.replace("_encoded", "").replace("_norm", "").replace("_", " ").title() for name, _ in corrs]
                corr_values = [corr for _, corr in corrs]
                colors = [palette[1] if c < 0 else palette[0] for c in corr_values]
                y_pos = range(len(feat_names))
                ax_importance.barh(y_pos, corr_values, color=colors, align='center')
                ax_importance.set_yticks(y_pos)
                ax_importance.set_yticklabels(feat_names, fontsize=8)
                ax_importance.axvline(0, color='black', linestyle='-', lw=0.5, alpha=0.5)
                ax_importance.set_xlabel("Spearman Correlation Coefficient", fontsize=9)
                ax_importance.invert_yaxis()
            else:
                ax_importance.text(0.5, 0.5, "No valid correlations calculated.", ha='center', va='center')
                ax_importance.axis('off')
        else:
             ax_importance.text(0.5, 0.5, "Not enough valid data for correlations.", ha='center', va='center')
             ax_importance.axis('off')
    else:
        ax_importance.text(0.5, 0.5, "RMSF or numeric features missing.", ha='center', va='center')
        ax_importance.axis('off')

    ax_ss_ce = plt.subplot(gs[1, 0:2])
    ax_ss_ce.set_title("Mean RMSF by SS Type & Location", fontsize=12)
    if ("secondary_structure_encoded" in avg_df.columns and
        "core_exterior" in avg_df.columns and
        "rmsf_average" in avg_df.columns):

        ss_map = {0: "Helix", 1: "Sheet", 2: "Coil"}
        plot_data_ssce = avg_df.copy()
        plot_data_ssce["SS_Type"] = plot_data_ssce["secondary_structure_encoded"].map(ss_map)
        plot_data_ssce["Location"] = plot_data_ssce["core_exterior"]

        plot_data_ssce["SS_Type"] = pd.Categorical(plot_data_ssce["SS_Type"], categories=["Helix", "Sheet", "Coil"], ordered=True)
        plot_data_ssce["Location"] = pd.Categorical(plot_data_ssce["Location"], categories=["core", "exterior"], ordered=True)
        plot_data_ssce.dropna(subset=["SS_Type", "Location", "rmsf_average"], inplace=True)

        if not plot_data_ssce.empty:
             pivot_data = plot_data_ssce.groupby(["SS_Type", "Location"], observed=False)["rmsf_average"].agg(["mean", "count"])
             pivot_mean = pivot_data['mean'].unstack(level='Location').reindex(["Helix", "Sheet", "Coil"]).reindex(columns=["core", "exterior"])
             pivot_count = pivot_data['count'].unstack(level='Location').reindex(["Helix", "Sheet", "Coil"]).reindex(columns=["core", "exterior"])

             sns.heatmap(pivot_mean, annot=True, fmt=".4f", cmap="viridis", ax=ax_ss_ce,
                       linewidths=.5, cbar_kws={'label': 'Mean RMSF (nm)'}, annot_kws={"size": 8})
             for i, ss_type in enumerate(pivot_mean.index):
                 for j, location in enumerate(pivot_mean.columns):
                     if ss_type in pivot_count.index and location in pivot_count.columns:
                          count = pivot_count.loc[ss_type, location]
                          mean_val = pivot_mean.loc[ss_type, location]
                          if pd.notna(count) and pd.notna(mean_val):
                               text_color = 'white' if mean_val > pivot_mean.values[~np.isnan(pivot_mean.values)].mean() else 'black'
                               ax_ss_ce.text(j + 0.5, i + 0.7, f"n={int(count):,}",
                                            ha='center', va='center', fontsize=7, color=text_color)

             ax_ss_ce.set_xlabel("Location", fontsize=9)
             ax_ss_ce.set_ylabel("Secondary Structure", fontsize=9)
        else:
             ax_ss_ce.text(0.5, 0.5, "No valid SS/Location data.", ha='center', va='center')
             ax_ss_ce.axis('off')
    else:
        ax_ss_ce.text(0.5, 0.5, "Required columns missing (SS_enc, core_ext, RMSF).", ha='center', va='center')
        ax_ss_ce.axis('off')

    ax_pos_access = plt.subplot(gs[1, 2])
    ax_pos_access.set_title("Accessibility vs Norm. Position", fontsize=12)
    if ("normalized_resid" in avg_df.columns and "relative_accessibility" in avg_df.columns):
        try:
             sample_df = avg_df.sample(min(5000, len(avg_df)), random_state=42)
             sns.scatterplot(data=sample_df, x="normalized_resid", y="relative_accessibility",
                             alpha=0.4, s=10, ax=ax_pos_access, color=palette[3])
             if STATSMODELS_AVAILABLE:
                 sns.regplot(data=sample_df, x="normalized_resid", y="relative_accessibility",
                             scatter=False, lowess=True,
                             ax=ax_pos_access, line_kws={'color': 'red', 'lw': 1})
             ax_pos_access.set_xlabel("Norm. Residue Position", fontsize=9)
             ax_pos_access.set_ylabel("Rel. Accessibility", fontsize=9)
             ax_pos_access.set_xlim(-0.05, 1.05)
             ax_pos_access.set_ylim(-0.05, 1.05)

        except Exception as e:
             logging.warning(f"Scatter/Regplot failed for Pos vs Access: {e}")
             ax_pos_access.text(0.5, 0.5, "Plotting failed.", ha='center', va='center')
             ax_pos_access.axis('off')
    else:
        ax_pos_access.text(0.5, 0.5, "Required columns missing (norm_resid, rel_access).", ha='center', va='center')
        ax_pos_access.axis('off')

    ax_phi_psi = plt.subplot(gs[2, 0:2])
    ax_phi_psi.set_title("RMSF vs Torsion Angles (Density)", fontsize=12)
    if ("phi_norm" in avg_df.columns and "psi_norm" in avg_df.columns and "rmsf_average" in avg_df.columns):
        plot_data_torsion = avg_df[['phi_norm', 'psi_norm', 'rmsf_average']].copy()
        plot_data_torsion.dropna(inplace=True)
        if len(plot_data_torsion) > 100:
            try:
                sns.kdeplot(data=plot_data_torsion, x="phi_norm", y="psi_norm",
                          fill=True, thresh=0.05, levels=5, cmap="viridis",
                          cbar=True, cbar_kws={'label': 'Density'}, ax=ax_phi_psi)
                ax_phi_psi.set_xlabel("Normalized Phi Angle (-1 to 1)", fontsize=9)
                ax_phi_psi.set_ylabel("Normalized Psi Angle (-1 to 1)", fontsize=9)
                ax_phi_psi.set_xlim(-1, 1)
                ax_phi_psi.set_ylim(-1, 1)
                ax_phi_psi.axhline(0, color='k', lw=0.5, ls=':')
                ax_phi_psi.axvline(0, color='k', lw=0.5, ls=':')
            except Exception as e:
                 logging.warning(f"KDE plot failed for Phi/Psi: {e}")
                 ax_phi_psi.text(0.5, 0.5, "KDE Plot Failed", ha='center', va='center')
                 ax_phi_psi.axis('off')
        else:
             ax_phi_psi.text(0.5, 0.5, "Not enough data points", ha='center', va='center')
             ax_phi_psi.axis('off')
    else:
        ax_phi_psi.text(0.5, 0.5, "Phi/Psi/RMSF data missing.", ha='center', va='center')
        ax_phi_psi.axis('off')

    ax_size_hist = plt.subplot(gs[2, 2])
    ax_size_hist.set_title("Protein Size Distribution", fontsize=12)
    if "protein_size" in avg_df.columns:
        protein_sizes = avg_df[['domain_id', 'protein_size']].drop_duplicates()['protein_size'].dropna()
        if not protein_sizes.empty:
            sns.histplot(protein_sizes, bins=30, kde=True, ax=ax_size_hist, color=palette[5])
            ax_size_hist.set_xlabel("Number of Residues", fontsize=9)
            ax_size_hist.set_ylabel("Number of Domains", fontsize=9)
        else:
             ax_size_hist.text(0.5, 0.5, "No Size Data", ha='center', va='center')
             ax_size_hist.axis('off')
    else:
        ax_size_hist.text(0.5, 0.5, "Protein size column missing.", ha='center', va='center')
        ax_size_hist.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = os.path.join(vis_dir, "ml_features_additional_analysis.png")
    _save_plot(fig, vis_dir, "ml_features_additional_analysis.png", dpi)
    return output_path

# --- NEW PLOTS ---

def create_rmsf_density_plots(feature_df_average: pd.DataFrame,
                              output_dir: str,
                              viz_config: Dict[str, Any]) -> Optional[str]:
    """
    Create 2D density plots showing RMSF distribution against other features.

    Args:
        feature_df_average: DataFrame containing average features.
        output_dir: Base output directory.
        viz_config: Visualization config section.

    Returns:
        Path to the saved figure or None if creation fails.
    """
    vis_dir = os.path.join(output_dir, "visualizations")
    dpi = viz_config.get('dpi', DEFAULT_DPI)
    palette_name = viz_config.get('palette', DEFAULT_PALETTE) # Use configured palette for cmap if desired
    _setup_plot_style(palette_name) # Set base style

    if feature_df_average is None or feature_df_average.empty:
        logging.warning("No average feature data available for RMSF density plots")
        return None

    required_cols = ["rmsf_average", "relative_accessibility", "normalized_resid"]
    if not all(col in feature_df_average.columns for col in required_cols):
        logging.warning(f"RMSF density plot requires columns: {required_cols}. Skipping.")
        return None

    avg_df = feature_df_average.copy()
    avg_df.dropna(subset=required_cols, inplace=True)

    if len(avg_df) < 50: # Need sufficient points for density estimation
        logging.warning("Not enough data points for RMSF density plots after dropping NaNs.")
        return None

    # Sample data for performance if dataframe is very large
    sample_df = avg_df.sample(min(10000, len(avg_df)), random_state=42)

    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("RMSF Density Analysis", fontsize=16, y=1.02)

    # Define a sequential colormap (e.g., viridis, plasma, magma)
    cmap = plt.get_cmap("viridis")

    # Panel 1: RMSF vs Relative Accessibility (Density colored by RMSF)
    try:
        # Use hexbin for density visualization colored by RMSF mean
        hb1 = axs[0].hexbin(sample_df["relative_accessibility"], sample_df["rmsf_average"],
                           C=sample_df["rmsf_average"], reduce_C_function=np.mean,
                           gridsize=40, cmap=cmap, mincnt=1) # mincnt=1 ensures bins with one point are shown
        cb1 = plt.colorbar(hb1, ax=axs[0])
        cb1.set_label('Mean RMSF (nm) in Bin')
        axs[0].set_title("RMSF vs Relative Accessibility (Density)")
        axs[0].set_xlabel("Relative Accessibility")
        axs[0].set_ylabel("RMSF (nm)")
        axs[0].set_xlim(0, 1)
        # Set ylim based on RMSF quantiles to avoid extreme outliers
        rmsf_q01, rmsf_q99 = sample_df["rmsf_average"].quantile([0.01, 0.99])
        axs[0].set_ylim(max(0, rmsf_q01 * 0.9), rmsf_q99 * 1.1)
    except Exception as e:
        logging.error(f"Failed to create RMSF vs Accessibility density plot: {e}")
        axs[0].text(0.5, 0.5, "Plot Failed", ha='center', va='center')
        axs[0].axis('off')

    # Panel 2: RMSF vs Normalized Position (Density colored by RMSF)
    try:
        hb2 = axs[1].hexbin(sample_df["normalized_resid"], sample_df["rmsf_average"],
                           C=sample_df["rmsf_average"], reduce_C_function=np.mean,
                           gridsize=40, cmap=cmap, mincnt=1)
        cb2 = plt.colorbar(hb2, ax=axs[1])
        cb2.set_label('Mean RMSF (nm) in Bin')
        axs[1].set_title("RMSF vs Normalized Position (Density)")
        axs[1].set_xlabel("Normalized Residue Position (N->C)")
        axs[1].set_ylabel("RMSF (nm)")
        axs[1].set_xlim(-0.05, 1.05)
        axs[1].set_ylim(max(0, rmsf_q01 * 0.9), rmsf_q99 * 1.1) # Use same y-lim as panel 1
    except Exception as e:
        logging.error(f"Failed to create RMSF vs Position density plot: {e}")
        axs[1].text(0.5, 0.5, "Plot Failed", ha='center', va='center')
        axs[1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_path = os.path.join(vis_dir, "rmsf_density_plots.png")
    _save_plot(fig, vis_dir, "rmsf_density_plots.png", dpi)
    return output_path


def create_rmsf_by_aa_ss_density(feature_df_average: pd.DataFrame,
                                 output_dir: str,
                                 viz_config: Dict[str, Any]) -> Optional[str]:
    """
    Create a ridgeline plot showing RMSF density distribution for major amino acids,
    separated by secondary structure type.

    Args:
        feature_df_average: DataFrame containing average features.
        output_dir: Base output directory.
        viz_config: Visualization config section.

    Returns:
        Path to the saved figure or None if creation fails.
    """
    vis_dir = os.path.join(output_dir, "visualizations")
    dpi = viz_config.get('dpi', DEFAULT_DPI)
    palette_name = viz_config.get('palette', DEFAULT_PALETTE)
    palette = _setup_plot_style(palette_name)

    if feature_df_average is None or feature_df_average.empty:
        logging.warning("No average feature data available for RMSF AA/SS density plot")
        return None

    required_cols = ["rmsf_average", "resname", "secondary_structure_encoded"]
    if not all(col in feature_df_average.columns for col in required_cols):
        logging.warning(f"RMSF AA/SS density plot requires columns: {required_cols}. Skipping.")
        return None

    avg_df = feature_df_average.copy()

    # Prepare data: Standardize HIS, map SS encoding, filter top AAs
    avg_df['resname'] = avg_df['resname'].replace({'HSD': 'HIS', 'HSE': 'HIS', 'HSP': 'HIS'})
    ss_map = {0: "Helix", 1: "Sheet", 2: "Coil"}
    avg_df['SS_Type'] = avg_df["secondary_structure_encoded"].map(ss_map).astype('category')

    # Get Top N amino acids (e.g., top 12)
    top_n_aa = 12
    top_aa = avg_df["resname"].value_counts().nlargest(top_n_aa).index.tolist()
    plot_df = avg_df[avg_df['resname'].isin(top_aa)].copy()

    # Set order for SS types and AAs (by overall mean RMSF)
    ss_order = ["Helix", "Sheet", "Coil"]
    plot_df['SS_Type'] = pd.Categorical(plot_df['SS_Type'], categories=ss_order, ordered=True)
    aa_order = plot_df.groupby('resname')['rmsf_average'].mean().sort_values().index.tolist()
    plot_df['resname'] = pd.Categorical(plot_df['resname'], categories=aa_order, ordered=True)

    plot_df.dropna(subset=['RMSF', 'resname', 'SS_Type'], inplace=True)

    if plot_df.empty:
        logging.warning("No data remaining for RMSF AA/SS density plot after filtering.")
        return None

    # Use FacetGrid for Ridgeline plot (requires seaborn >= 0.11 for ridgeplot)
    try:
        # Define a color palette for SS types
        ss_colors = {"Helix": palette[0], "Sheet": palette[1], "Coil": palette[2]}

        g = sns.FacetGrid(plot_df, row="resname", hue="SS_Type", aspect=5, height=1.2,
                          row_order=aa_order, hue_order=ss_order, palette=ss_colors)

        # Draw the densities in a few steps
        g.map(sns.kdeplot, "rmsf_average", bw_adjust=.5, clip_on=False, fill=True, alpha=0.7, linewidth=1.5)
        # Draw reference lines for each facet
        g.map(plt.axhline, y=0, linewidth=1, linestyle="-", color=None, clip_on=False)

        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0.02, .2, label, fontweight="bold", color=color,
                    ha="left", va="center", transform=ax.transAxes)

        g.map(label, "rmsf_average")

        # Set the subplots to overlap
        g.figure.subplots_adjust(hspace=-.5)

        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)

        g.fig.suptitle(f'RMSF Density Distribution by Secondary Structure (Top {top_n_aa} Amino Acids)', y=1.02)
        g.set_xlabels("RMSF (nm)")
        g.add_legend(title="SS Type")

        output_path = os.path.join(vis_dir, "rmsf_density_by_aa_ss.png")
        # Save using the FacetGrid figure
        g.figure.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(g.figure)
        logging.info(f"Saved RMSF density by AA/SS plot: {output_path}")
        return output_path

    except Exception as e:
        logging.error(f"Failed to create RMSF AA/SS density plot: {e}", exc_info=True)
        # Clean up potentially open figure
        if 'g' in locals() and hasattr(g, 'figure'):
            plt.close(g.figure)
        return None

# Ensure all functions intended to be called by the executor exist:
# create_temperature_summary_heatmap
# create_temperature_average_summary
# create_rmsf_distribution_plots (violin + separated histograms)
# create_amino_acid_rmsf_plot
# create_amino_acid_rmsf_plot_colored
# create_replica_variance_plot
# create_dssp_rmsf_correlation_plot
# create_feature_correlation_plot
# create_frames_visualization (Placeholder)
# create_ml_features_plot
# create_summary_plot
# create_voxel_info_plot
# create_additional_ml_features_plot
# create_rmsf_density_plots (New)
# create_rmsf_by_aa_ss_density (New)