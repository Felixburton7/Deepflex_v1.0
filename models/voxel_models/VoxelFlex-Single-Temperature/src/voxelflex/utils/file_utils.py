"""
File utility functions for Voxelflex.

This module provides utility functions for file and directory operations.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional, List

def ensure_dir(directory: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
    """
    os.makedirs(directory, exist_ok=True)


def resolve_path(path: str) -> str:
    """
    Resolve a file path, expanding user directory and making it absolute.
    
    Args:
        path: File path
        
    Returns:
        Resolved path
    """
    expanded_path = os.path.expanduser(path)
    return os.path.abspath(expanded_path)


def save_json(data: Dict[str, Any], file_path: str, indent: int = 4) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the file
        indent: JSON indentation level
    """
    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    ensure_dir(directory)
    
    # Save the JSON file
    with open(file_path, 'w') as f:
        # Handle numpy arrays and other non-serializable objects
        json.dump(data, f, indent=indent, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def get_file_extension(file_path: str) -> str:
    """
    Get the file extension from a file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension (lowercase, without the dot)
    """
    return os.path.splitext(file_path)[1].lower()[1:]


def inspect_hdf5_structure(file_path: str, domain_sample: int = 2, print_residues: bool = False):
    """
    Utility function to inspect and print the structure of an HDF5 file.
    
    This is useful for debugging data loading issues.
    
    Args:
        file_path: Path to the HDF5 file
        domain_sample: Number of domains to sample and print (default: 2)
        print_residues: Whether to print individual residue details (default: False)
    """
    import h5py
    import os
    
    file_path = os.path.expanduser(file_path)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    print(f"\nInspecting HDF5 file: {file_path}")
    print("=" * 80)
    
    with h5py.File(file_path, 'r') as f:
        # Print top-level keys (domains)
        domains = list(f.keys())
        print(f"File contains {len(domains)} top-level groups (domains)")
        print(f"First few domains: {domains[:min(5, len(domains))]}")
        
        # Sample a few domains to inspect more deeply
        sample_domains = domains[:min(domain_sample, len(domains))]
        print(f"\nSampling {len(sample_domains)} domains for detailed inspection:")
        
        for domain in sample_domains:
            print(f"\nDomain: {domain}")
            print("-" * 40)
            
            domain_group = f[domain]
            domain_keys = list(domain_group.keys())
            print(f"Domain has {len(domain_keys)} direct children")
            print(f"Children keys: {domain_keys}")
            
            # Find residue groups
            for key in domain_keys:
                item = domain_group[key]
                if isinstance(item, h5py.Group):
                    print(f"\n  Group: {key}")
                    sub_keys = list(item.keys())
                    print(f"  Contains {len(sub_keys)} items")
                    
                    # Check if these are likely residues (numeric keys)
                    numeric_keys = [k for k in sub_keys if isinstance(k, str) and k.isdigit()]
                    if numeric_keys:
                        print(f"  Contains {len(numeric_keys)} numeric keys (likely residues)")
                        
                        if print_residues:
                            # Sample a few residues
                            sample_residues = numeric_keys[:min(3, len(numeric_keys))]
                            for res_id in sample_residues:
                                print(f"\n    Residue: {res_id}")
                                residue = item[res_id]
                                
                                if isinstance(residue, h5py.Group):
                                    res_keys = list(residue.keys())
                                    print(f"    Contains keys: {res_keys}")
                                    
                                    # Check for voxel data
                                    for res_key in res_keys:
                                        res_item = residue[res_key]
                                        if isinstance(res_item, h5py.Dataset):
                                            print(f"      Dataset '{res_key}': shape={res_item.shape}, dtype={res_item.dtype}")
                                
                                elif isinstance(residue, h5py.Dataset):
                                    print(f"    Direct dataset: shape={residue.shape}, dtype={residue.dtype}")
                elif isinstance(item, h5py.Dataset):
                    print(f"\n  Dataset: {key}, shape={item.shape}, dtype={item.dtype}")
    
    print("\nFile inspection complete.")
    print("=" * 80)



def save_domain_registry(domains: List[str], file_path: str) -> None:
    """
    Save a list of processed domain IDs to a file.
    
    Args:
        domains: List of domain IDs that were processed
        file_path: Path to save the registry file
    """
    with open(file_path, 'w') as f:
        for domain in domains:
            f.write(f"{domain}\n")

def load_domain_registry(file_path: str) -> List[str]:
    """
    Load a list of processed domain IDs from a file.
    
    Args:
        file_path: Path to the registry file
        
    Returns:
        List of domain IDs
    """
    if not os.path.exists(file_path):
        return []
        
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]