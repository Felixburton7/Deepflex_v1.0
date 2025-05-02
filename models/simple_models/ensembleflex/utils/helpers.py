"""
Helper functions for the FlexSeq ML pipeline.

This module provides utility functions used throughout the pipeline.
"""

import os
import logging
import re
from typing import Dict, List, Any, Tuple, Optional, Union, Callable, Iterable, TypeVar
from functools import wraps
from time import time

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

T = TypeVar('T')

def timer(func):
    """
    Decorator for measuring function execution time.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        logger.info(f"Function '{func.__name__}' executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def ensure_dir(directory: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Directory path
    """
    os.makedirs(directory, exist_ok=True)
    return directory

def estimate_memory_usage(df: pd.DataFrame) -> Tuple[float, str]:
    """
    Estimate memory usage of a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (size, unit) where size is a number and unit is a string
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    
    if memory_bytes < 1024:
        return memory_bytes, "bytes"
    elif memory_bytes < 1024**2:
        return memory_bytes / 1024, "KB"
    elif memory_bytes < 1024**3:
        return memory_bytes / (1024**2), "MB"
    else:
        return memory_bytes / (1024**3), "GB"

def chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
    """
    Split a DataFrame into chunks of specified size.
    
    Args:
        df: Input DataFrame
        chunk_size: Number of rows per chunk
        
    Returns:
        List of DataFrame chunks
    """
    return [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

def safe_open(file_path: str, mode: str = 'r'):
    """
    Safely open a file with proper directory creation.
    
    Args:
        file_path: Path to file
        mode: File opening mode
        
    Returns:
        File object
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return open(file_path, mode)

def truncate_filename(filename: str, max_length: int = 255) -> str:
    """
    Truncate a filename to ensure it doesn't exceed maximum path length.
    
    Args:
        filename: Original filename
        max_length: Maximum allowed length
        
    Returns:
        Truncated filename
    """
    if len(filename) <= max_length:
        return filename
    
    name, ext = os.path.splitext(filename)
    return name[:max_length - len(ext)] + ext

def safe_parse_float(value: Any) -> Optional[float]:
    """
    Safely parse a value to float, returning None if not possible.
    
    Args:
        value: Value to convert
        
    Returns:
        Float value or None if conversion fails
    """
    if pd.isna(value):
        return None
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def get_amino_acid_properties() -> Dict[str, Dict[str, Any]]:
    """
    Get dictionary of amino acid properties.
    
    Returns:
        Dictionary mapping amino acid codes to property dictionaries
    """
    properties = {
        # Hydrophobic residues
        'ALA': {'hydropathy': 1.8, 'volume': 88.6, 'charge': 0, 'group': 'hydrophobic'},
        'VAL': {'hydropathy': 4.2, 'volume': 140.0, 'charge': 0, 'group': 'hydrophobic'},
        'LEU': {'hydropathy': 3.8, 'volume': 166.7, 'charge': 0, 'group': 'hydrophobic'},
        'ILE': {'hydropathy': 4.5, 'volume': 166.7, 'charge': 0, 'group': 'hydrophobic'},
        'MET': {'hydropathy': 1.9, 'volume': 162.9, 'charge': 0, 'group': 'hydrophobic'},
        'PHE': {'hydropathy': 2.8, 'volume': 189.9, 'charge': 0, 'group': 'hydrophobic'},
        'TRP': {'hydropathy': -0.9, 'volume': 227.8, 'charge': 0, 'group': 'hydrophobic'},
        'PRO': {'hydropathy': -1.6, 'volume': 112.7, 'charge': 0, 'group': 'special'},
        'GLY': {'hydropathy': -0.4, 'volume': 60.1, 'charge': 0, 'group': 'special'},
        
        # Polar residues
        'SER': {'hydropathy': -0.8, 'volume': 89.0, 'charge': 0, 'group': 'polar'},
        'THR': {'hydropathy': -0.7, 'volume': 116.1, 'charge': 0, 'group': 'polar'},
        'CYS': {'hydropathy': 2.5, 'volume': 108.5, 'charge': 0, 'group': 'polar'},
        'TYR': {'hydropathy': -1.3, 'volume': 193.6, 'charge': 0, 'group': 'polar'},
        'ASN': {'hydropathy': -3.5, 'volume': 111.1, 'charge': 0, 'group': 'polar'},
        'GLN': {'hydropathy': -3.5, 'volume': 143.8, 'charge': 0, 'group': 'polar'},
        
        # Charged residues
        'LYS': {'hydropathy': -3.9, 'volume': 168.6, 'charge': 1, 'group': 'positive'},
        'ARG': {'hydropathy': -4.5, 'volume': 173.4, 'charge': 1, 'group': 'positive'},
        'HIS': {'hydropathy': -3.2, 'volume': 153.2, 'charge': 0.5, 'group': 'positive'},
        'ASP': {'hydropathy': -3.5, 'volume': 111.1, 'charge': -1, 'group': 'negative'},
        'GLU': {'hydropathy': -3.5, 'volume': 138.4, 'charge': -1, 'group': 'negative'},
        
        # Non-standard
        'HSE': {'hydropathy': -3.2, 'volume': 153.2, 'charge': 0.5, 'group': 'positive'},
        'HSD': {'hydropathy': -3.2, 'volume': 153.2, 'charge': 0.5, 'group': 'positive'},
        'HSP': {'hydropathy': -3.2, 'volume': 153.2, 'charge': 1, 'group': 'positive'},
        'UNK': {'hydropathy': 0.0, 'volume': 0.0, 'charge': 0, 'group': 'unknown'}
    }
    
    return properties

def is_glycine_or_proline(residue: str) -> bool:
    """
    Check if residue is Glycine or Proline, which significantly affect flexibility.
    
    Args:
        residue: Residue name
        
    Returns:
        True if residue is GLY or PRO
    """
    return residue in ["GLY", "PRO"]

def analyze_hydrogen_bonds(
    secondary_structure: str, 
    residue_name: str
) -> float:
    """
    Analyze potential hydrogen bonding based on secondary structure and residue type.
    Returns a relative measure of hydrogen bond stabilization (higher means more stable).
    
    Args:
        secondary_structure: DSSP code
        residue_name: Residue name
        
    Returns:
        Relative stability score (0-1)
    """
    # Secondary structure types have different H-bond patterns
    ss_stability = {
        'H': 1.0,  # Alpha helix - most stable H-bond pattern
        'G': 0.8,  # 3-10 helix
        'I': 0.9,  # Pi helix
        'E': 0.9,  # Beta sheet - very stable
        'B': 0.7,  # Beta bridge
        'T': 0.4,  # Turn
        'S': 0.3,  # Bend
        'C': 0.1   # Coil - least stable
    }
    
    # Some residues have different H-bond propensities
    residue_factors = {
        'SER': 1.2,  # Can form side-chain H-bonds
        'THR': 1.2,  # Can form side-chain H-bonds
        'ASN': 1.2,  # Can form side-chain H-bonds
        'GLN': 1.2,  # Can form side-chain H-bonds
        'TYR': 1.1,  # Can form side-chain H-bonds
        'PRO': 0.7,  # Disrupts H-bond patterns
        'GLY': 0.9   # More flexible backbone
    }
    
    # Get base stability from secondary structure
    stability = ss_stability.get(secondary_structure, 0.1)
    
    # Apply residue-specific factor
    factor = residue_factors.get(residue_name, 1.0)
    
    return min(1.0, stability * factor)

def calculate_sequence_complexity(sequence: List[str], window_size: int = 5) -> List[float]:
    """
    Calculate sequence complexity using Shannon entropy in a sliding window.
    Higher values indicate more diverse/complex sequence regions.
    
    Args:
        sequence: List of amino acid types
        window_size: Size of sliding window
        
    Returns:
        List of complexity values for each position
    """
    # Convert to single-letter amino acid codes if needed
    if all(len(aa) == 3 for aa in sequence if aa):
        three_to_one = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
            'HSD': 'H', 'HSE': 'H', 'HSP': 'H', 'UNK': 'X'
        }
        sequence = [three_to_one.get(aa, 'X') for aa in sequence]
    
    # Compute complexity for each position
    complexity = []
    half_window = window_size // 2
    padded_seq = ['X'] * half_window + list(sequence) + ['X'] * half_window
    
    for i in range(half_window, len(padded_seq) - half_window):
        window = padded_seq[i - half_window:i + half_window + 1]
        
        # Calculate Shannon entropy
        aa_counts = {}
        for aa in window:
            if aa in aa_counts:
                aa_counts[aa] += 1
            else:
                aa_counts[aa] = 1
        
        entropy = 0
        for count in aa_counts.values():
            p = count / window_size
            entropy -= p * np.log2(p)
        
        # Normalize by maximum possible entropy (all different amino acids)
        max_entropy = np.log2(min(20, window_size))
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 0
        
        complexity.append(normalized_entropy)
    
    return complexity

def progress_bar(
    iterable: Iterable[T],
    desc: str = None,
    total: Optional[int] = None,
    disable: bool = False,
    leave: bool = True,
    **kwargs
) -> Iterable[T]:
    """
    Create a progress bar for an iterable.
    
    Args:
        iterable: Iterable to track progress of
        desc: Description for the progress bar
        total: Total number of items (inferred if not provided)
        disable: Whether to disable the progress bar
        leave: Whether to leave the progress bar after completion
        **kwargs: Additional arguments to pass to tqdm
        
    Returns:
        Wrapped iterable with progress tracking
    """
    # Disable progress bar if verbose logging is not enabled
    log_level = logging.getLogger().getEffectiveLevel()
    if log_level > logging.INFO:
        disable = True
        
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        disable=disable,
        leave=leave,
        ncols=100,  # Fixed width
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
        **kwargs
    )

class ProgressCallback:
    """
    Callback class to track progress of operations that don't use iterables.
    """
    
    def __init__(
        self, 
        total: int, 
        desc: str = None,
        disable: bool = False,
        leave: bool = True,
        **kwargs
    ):
        """
        Initialize progress callback.
        
        Args:
            total: Total number of steps
            desc: Description for the progress bar
            disable: Whether to disable the progress bar
            leave: Whether to leave the progress bar after completion
            **kwargs: Additional arguments to pass to tqdm
        """
        # Disable progress bar if verbose logging is not enabled
        log_level = logging.getLogger().getEffectiveLevel()
        if log_level > logging.INFO:
            disable = True
            
        self.pbar = tqdm(
            total=total,
            desc=desc,
            disable=disable,
            leave=leave,
            ncols=100,  # Fixed width
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            **kwargs
        )
    
    def update(self, n: int = 1):
        """Update the progress bar by n steps."""
        self.pbar.update(n)
    
    def set_description(self, desc: str):
        """Set the description of the progress bar."""
        self.pbar.set_description(desc)
    
    def set_postfix(self, **kwargs):
        """Set the postfix of the progress bar."""
        self.pbar.set_postfix(**kwargs)
    
    def close(self):
        """Close the progress bar."""
        self.pbar.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

def get_temperature_color(temperature: Union[int, str]) -> str:
    """
    Get a color code for a temperature value.
    Colors range from blue (cold) to red (hot).
    
    Args:
        temperature: Temperature value
        
    Returns:
        Hex color code
    """
    # Handle special case for "average"
    if temperature == "average" or not isinstance(temperature, (int, float)):
        return "#7F7F7F"  # Gray
    
    # Temperature ranges for FlexSeq
    min_temp = 320
    max_temp = 450
    
    # Clamp temperature to range
    temp = max(min_temp, min(temperature, max_temp))
    
    # Normalize to 0-1 range
    normalized = (temp - min_temp) / (max_temp - min_temp)
    
    # Generate color (blue to red)
    r = int(255 * normalized)
    b = int(255 * (1 - normalized))
    g = int(100 * (1 - abs(2 * normalized - 1)))
    
    return f"#{r:02x}{g:02x}{b:02x}"

def make_model_color_map(model_names: List[str]) -> Dict[str, str]:
    """
    Create a consistent color mapping for models.
    
    Args:
        model_names: List of model names
        
    Returns:
        Dictionary mapping model names to color codes
    """
    # Standard colors for models
    standard_colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf"   # Cyan
    ]
    
    # Create mapping
    color_map = {}
    for i, model in enumerate(model_names):
        color_map[model] = standard_colors[i % len(standard_colors)]
    
    return color_map