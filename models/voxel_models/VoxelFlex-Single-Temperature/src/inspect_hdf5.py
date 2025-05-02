#!/usr/bin/env python3
"""
inspect_hdf5.py

A utility script to thoroughly inspect an HDF5 file and report its structure,
including top-level groups (domains), child groups/datasets, dataset shapes, 
and detailed information about residues, atom types, and actual data values.

Usage:
    python inspect_hdf5.py /path/to/your/file.hdf5 --domain-sample 3 --print-residues --show-data
"""

import os
import argparse
import h5py
import numpy as np
import pandas as pd
from collections import Counter
from tabulate import tabulate

def inspect_hdf5_structure(file_path, domain_sample=2, print_residues=False, show_data=False):
    """
    Thoroughly inspect the structure of an HDF5 file with detailed data samples.

    Args:
        file_path (str): Path to the HDF5 file.
        domain_sample (int): Number of top-level groups (domains) to sample for detailed inspection.
        print_residues (bool): If True, prints details of numeric (likely residue) groups.
        show_data (bool): If True, shows sample data values from datasets.
    """
    file_path = os.path.expanduser(file_path)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"\nInspecting HDF5 file: {file_path}")
    print("=" * 80)
    
    try:
        with h5py.File(file_path, 'r') as f:
            # List top-level keys (domains)
            domains = list(f.keys())
            print(f"File contains {len(domains)} top-level groups (domains)")
            print(f"First 10 domains: {domains[:min(10, len(domains))]}")
            
            # Count patterns in domain names
            pattern_counts = {}
            for domain in domains:
                # Extract pattern (e.g., _pdb_clean_fixed)
                parts = domain.split('_')
                if len(parts) > 1:
                    pattern = '_'.join(parts[1:])
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            print("\nDomain naming patterns:")
            for pattern, count in pattern_counts.items():
                print(f"  {pattern}: {count} domains ({count/len(domains)*100:.1f}%)")
            
            # Sample domains for detailed inspection - mixture of first few and some random ones
            print(f"\nSampling {domain_sample} domains for detailed inspection:")
            
            # Choose some domains to sample - mix of first ones and random selections
            sample_domains = []
            # Add first few domains
            first_domains = min(domain_sample // 2, len(domains))
            sample_domains.extend(domains[:first_domains])
            
            # Add some random domains if needed
            if domain_sample > first_domains:
                import random
                random.seed(42)  # For reproducibility
                remaining = domain_sample - first_domains
                random_domains = random.sample(domains[first_domains:], min(remaining, len(domains) - first_domains))
                sample_domains.extend(random_domains)
                
            print(f"Selected domains: {sample_domains}")
            
            for domain_idx, domain in enumerate(sample_domains):
                print(f"\n{'='*20} DOMAIN: {domain} {'='*20}")
                
                domain_group = f[domain]
                domain_keys = list(domain_group.keys())
                print(f"Domain has {len(domain_keys)} direct children")
                print(f"Children keys: {domain_keys}")
                
                # Domain statistics summary
                total_residues = 0
                total_datasets = 0
                data_shapes = set()
                data_types = set()
                residue_types_found = set()
                voxel_dimensions = set()
                
                # Inspect each child (typically chains)
                for chain_key in domain_keys:
                    chain = domain_group[chain_key]
                    if isinstance(chain, h5py.Group):
                        print(f"\n  CHAIN: {chain_key}")
                        residue_keys = list(chain.keys())
                        print(f"  Contains {len(residue_keys)} items")
                        
                        # Look for residue keys (numeric keys)
                        numeric_keys = [k for k in residue_keys if k.isdigit()]
                        if numeric_keys:
                            total_residues += len(numeric_keys)
                            print(f"  Contains {len(numeric_keys)} numeric keys (residues)")
                            print(f"  Residue ID range: {min(numeric_keys)} to {max(numeric_keys)}")
                            
                            # Group residues by tens for a histogram-like view
                            residue_counts = {}
                            for resid in numeric_keys:
                                bin_key = int(resid) // 10 * 10
                                residue_counts[bin_key] = residue_counts.get(bin_key, 0) + 1
                            
                            print("  Residue distribution:")
                            for bin_start, count in sorted(residue_counts.items()):
                                print(f"    {bin_start}-{bin_start+9}: {count} residues")
                            
                            if print_residues:
                                # Sample evenly distributed residues for a representative view
                                if len(numeric_keys) <= 5:
                                    sampled_residues = numeric_keys
                                else:
                                    indices = np.linspace(0, len(numeric_keys)-1, 5, dtype=int)
                                    sampled_residues = [numeric_keys[i] for i in indices]
                                
                                print(f"\n  Sampling {len(sampled_residues)} residues:")
                                for res_idx, res in enumerate(sampled_residues):
                                    print(f"\n    RESIDUE: {res}")
                                    residue = chain[res]
                                    
                                    if isinstance(residue, h5py.Dataset):
                                        shape = residue.shape
                                        dtype = residue.dtype
                                        data_shapes.add(str(shape))
                                        data_types.add(str(dtype))
                                        total_datasets += 1
                                        
                                        print(f"    Direct dataset: shape={shape}, dtype={dtype}")
                                        
                                        # Try to determine residue type if available in attributes or metadata
                                        res_type = None
                                        try:
                                            if hasattr(residue, 'attrs') and 'resname' in residue.attrs:
                                                res_type = residue.attrs['resname']
                                                print(f"    Residue type (from attributes): {res_type}")
                                        except Exception:
                                            pass
                                            
                                        if show_data:
                                            # Print shape details
                                            print(f"    Dimensions: {len(shape)}-D tensor")
                                            for dim_idx, dim_size in enumerate(shape):
                                                print(f"      Dimension {dim_idx}: {dim_size}")
                                            
                                            # Sample and print data
                                            try:
                                                data = residue[:]
                                                
                                                # Basic statistics
                                                if dtype in [np.float32, np.float64]:
                                                    print(f"    Data statistics:")
                                                    print(f"      Min: {data.min():.4f}")
                                                    print(f"      Max: {data.max():.4f}")
                                                    print(f"      Mean: {data.mean():.4f}")
                                                    print(f"      Median: {np.median(data):.4f}")
                                                    print(f"      Standard deviation: {np.std(data):.4f}")
                                                    print(f"      Non-zero elements: {np.count_nonzero(data)}/{data.size} ({np.count_nonzero(data)/data.size*100:.1f}%)")
                                                    
                                                    # Show actual values
                                                    flat_data = data.flatten()
                                                    unique_values = np.unique(flat_data)
                                                    if len(unique_values) <= 20:
                                                        print(f"      Unique values: {unique_values}")
                                                    else:
                                                        print(f"      Sample values: {flat_data[:10]} ...")
                                                
                                                # Identify channels if available
                                                if len(shape) == 4 and shape[3] in [4, 5]:
                                                    print(f"    Likely voxel data with {shape[3]} channels")
                                                    channels = ["C", "N", "O", "CA"]
                                                    if shape[3] == 5:
                                                        channels.append("CB")
                                                    
                                                    # Transpose for accessing channels
                                                    transposed = np.transpose(data, (3, 0, 1, 2))
                                                    
                                                    print(f"    Channel information:")
                                                    for ch_idx, ch_name in enumerate(channels):
                                                        channel_data = transposed[ch_idx]
                                                        print(f"      Channel {ch_idx} ({ch_name}):")
                                                        print(f"        Non-zero voxels: {np.count_nonzero(channel_data)}/{channel_data.size} ({np.count_nonzero(channel_data)/channel_data.size*100:.2f}%)")
                                                        print(f"        Value range: {channel_data.min():.4f} to {channel_data.max():.4f}")
                                                        
                                                        # Distribution of values
                                                        if show_data:
                                                            # Get distribution of non-zero values
                                                            non_zero_values = channel_data[channel_data > 0]
                                                            if len(non_zero_values) > 0:
                                                                percentiles = [0, 25, 50, 75, 100]
                                                                percentile_values = np.percentile(non_zero_values, percentiles)
                                                                print(f"        Value distribution (percentiles):")
                                                                for p, v in zip(percentiles, percentile_values):
                                                                    print(f"          {p}%: {v:.4f}")
                                                                    
                                                                # Identify voxel density pattern
                                                                density_pattern = "Uniform"
                                                                variance = np.var(non_zero_values)
                                                                if variance > 0.1:
                                                                    density_pattern = "Highly varied"
                                                                elif variance > 0.01:
                                                                    density_pattern = "Moderately varied"
                                                                
                                                                print(f"        Density pattern: {density_pattern} (variance: {variance:.4f})")
                                                
                                                    # Further analyze the data - no plotting
                                                        # Check for patterns in voxel distribution
                                                        for ch_idx, ch_name in enumerate(channels):
                                                            channel_data = transposed[ch_idx]
                                                            
                                                            # Add voxel dimensions to domain stats
                                                    voxel_dimensions.add(f"{data.shape[0]}x{data.shape[1]}x{data.shape[2]}")
                                                    
                                                    # Check for central tendency
                                                    if np.count_nonzero(channel_data) > 0:
                                                                
                                                        # Find center of mass (weighted average of coordinates)
                                                        positions = np.where(channel_data > 0)
                                                        if len(positions[0]) > 0:
                                                            center_x = np.mean(positions[0])
                                                            center_y = np.mean(positions[1])
                                                            center_z = np.mean(positions[2])
                                                            print(f"        Center of mass for {ch_name}: "
                                                                  f"({center_x:.1f}, {center_y:.1f}, {center_z:.1f})")
                                                            
                                                            # Check for atoms near the center vs. periphery
                                                            # Calculate distances from center of voxel grid
                                                            grid_center_x = data.shape[0] / 2
                                                            grid_center_y = data.shape[1] / 2
                                                            grid_center_z = data.shape[2] / 2
                                                            
                                                            center_offset = np.sqrt(
                                                                (center_x - grid_center_x)**2 + 
                                                                (center_y - grid_center_y)**2 + 
                                                                (center_z - grid_center_z)**2
                                                            )
                                                            
                                                            max_distance = min(data.shape[0], data.shape[1], data.shape[2]) / 2
                                                            relative_offset = center_offset / max_distance
                                                            
                                                            if relative_offset < 0.2:
                                                                position = "centered in voxel grid"
                                                            elif relative_offset < 0.5:
                                                                position = "moderately offset from center"
                                                            else:
                                                                position = "significantly offset from center"
                                                                
                                                            print(f"        {ch_name} atom is {position} "
                                                                  f"(offset: {relative_offset:.2f})")
                                            except Exception as e:
                                                print(f"    Error sampling data: {e}")
                                    
                                    elif isinstance(residue, h5py.Group):
                                        print(f"    Group with {len(residue.keys())} items")
                                        print(f"    Keys: {list(residue.keys())}")
                                        
                                        # Explore one level deeper if there are datasets
                                        for sub_key in residue.keys():
                                            sub_item = residue[sub_key]
                                            if isinstance(sub_item, h5py.Dataset):
                                                shape = sub_item.shape
                                                dtype = sub_item.dtype
                                                data_shapes.add(str(shape))
                                                data_types.add(str(dtype))
                                                total_datasets += 1
                                                
                                                print(f"      Dataset '{sub_key}': shape={shape}, dtype={dtype}")
                                                
                                                if show_data:
                                                    try:
                                                        data = sub_item[:]
                                                        print(f"      Data sample: {data.flatten()[:5]} ...")
                                                    except Exception as e:
                                                        print(f"      Error sampling data: {e}")
                    
                    elif isinstance(chain, h5py.Dataset):
                        shape = chain.shape
                        dtype = chain.dtype
                        data_shapes.add(str(shape))
                        data_types.add(str(dtype))
                        total_datasets += 1
                        
                        print(f"\n  Direct dataset '{chain_key}': shape={shape}, dtype={dtype}")
                        
                        if show_data:
                            try:
                                data = chain[:]
                                print(f"  Data sample: {data.flatten()[:5]} ...")
                            except Exception as e:
                                print(f"  Error sampling data: {e}")
                
                # Summarize domain statistics
                print(f"\n  DOMAIN SUMMARY:")
                print(f"  Total residues: {total_residues}")
                print(f"  Total datasets: {total_datasets}")
                print(f"  Unique data shapes: {data_shapes}")
                print(f"  Data types: {data_types}")
                
                # Add residue types if found
                if residue_types_found:
                    print(f"  Residue types found: {sorted(residue_types_found)}")
                
                # Add voxel dimensions if found
                if voxel_dimensions:
                    print(f"  Voxel grid dimensions: {sorted(voxel_dimensions)}")
            
            # Memory usage information
            memory_info = {
                "File size (MB)": os.path.getsize(file_path) / (1024 * 1024),
                "Total domains": len(domains),
                "Sample domains": len(sample_domains)
            }
            print("\nFile information:")
            for key, value in memory_info.items():
                print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"Error opening or processing HDF5 file: {e}")
    
    print("\nFile inspection complete.")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Thoroughly inspect HDF5 file structure and content")
    parser.add_argument("file", help="Path to the HDF5 file")
    parser.add_argument("--domain-sample", type=int, default=2, help="Number of top-level groups (domains) to sample")
    parser.add_argument("--print-residues", action="store_true", help="Print detailed residue (numeric key) information")
    parser.add_argument("--show-data", action="store_true", help="Show sample data values from datasets")
    
    args = parser.parse_args()
    inspect_hdf5_structure(
        args.file, 
        domain_sample=args.domain_sample, 
        print_residues=args.print_residues,
        show_data=args.show_data
    )