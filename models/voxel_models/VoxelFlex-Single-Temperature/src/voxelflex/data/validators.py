
"""
Data validation module for Voxelflex.

This module provides functions to validate voxel and RMSF data
to ensure consistency and proper format before processing.
"""

import logging
from typing import Dict, List, Set, Any

import numpy as np
import pandas as pd

from voxelflex.utils.logging_utils import get_logger

logger = get_logger(__name__)

def validate_voxel_data(voxel_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Validate voxel data to ensure consistent shapes and formats.
    
    Args:
        voxel_data: Dictionary mapping domain IDs to dictionaries of residue voxel data
        
    Returns:
        Validated voxel data
    """
    logger.info("Validating voxel data")
    
    if not voxel_data:
        logger.error("Empty voxel data. Check your voxel file structure and domain_ids configuration.")
        logger.error("If using specific domain_ids, ensure they match the base names in the voxel file.")
        raise ValueError("Empty voxel data. Check logs for troubleshooting details.")
    
    valid_data = {}
    expected_ndim = None
    expected_channels = None
    
    # First pass: determine expected dimensions
    for domain_id, domain_data in voxel_data.items():
        if not domain_data:
            logger.warning(f"Empty domain data for {domain_id}")
            continue
            
        # Get the first residue's voxel data to determine expected shape
        first_resid = next(iter(domain_data))
        first_voxel = domain_data[first_resid]
        
        if expected_ndim is None:
            expected_ndim = first_voxel.ndim
            if expected_ndim != 4:  # [channels, x, y, z]
                logger.warning(f"Unexpected voxel dimensions: {expected_ndim}, expected 4")
            
        if expected_channels is None and expected_ndim >= 1:
            expected_channels = first_voxel.shape[0]
            logger.info(f"Detected {expected_channels} channels in voxel data")
            
            # Check if we have the right number of channels
            if expected_channels not in [4, 5]:
                logger.warning(f"Unexpected number of channels: {expected_channels}, expected 4 or 5")
    
    # Second pass: validate and filter data
    total_residues = 0
    valid_residues = 0
    
    for domain_id, domain_data in voxel_data.items():
        valid_domain_data = {}
        
        for resid, voxel in domain_data.items():
            total_residues += 1
            
            # Check dimensions
            if voxel.ndim != expected_ndim:
                logger.debug(f"Skipping {domain_id}:{resid} - Inconsistent dimensions: {voxel.ndim} vs {expected_ndim}")
                continue
                
            # Check number of channels
            if voxel.shape[0] != expected_channels:
                logger.debug(f"Skipping {domain_id}:{resid} - Inconsistent channels: {voxel.shape[0]} vs {expected_channels}")
                continue
            
            # Check for NaN or Inf values
            if np.isnan(voxel).any() or np.isinf(voxel).any():
                logger.debug(f"Skipping {domain_id}:{resid} - Contains NaN or Inf values")
                continue
            
            valid_domain_data[resid] = voxel
            valid_residues += 1
        
        if valid_domain_data:
            valid_data[domain_id] = valid_domain_data
    
    logger.info(f"Validated {valid_residues}/{total_residues} residues "
                f"across {len(valid_data)}/{len(voxel_data)} domains")
    
    if not valid_data:
        # Provide detailed troubleshooting information
        logger.error("No valid voxel data after validation")
        logger.error("Potential issues:")
        logger.error("1. The structure of your voxel file may not match what the code expects")
        logger.error("2. The voxel data may have inconsistent shapes or contain NaN/Inf values")
        logger.error("3. You may need to adjust the validation criteria in the code")
        logger.error("Try running with verbose logging and check the file structure directly")
        raise ValueError("No valid voxel data after validation. Check logs for details.")
    
    return valid_data

def validate_rmsf_data(rmsf_data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate RMSF data to ensure it has the required columns and format.
    
    Args:
        rmsf_data: DataFrame containing RMSF data
        
    Returns:
        Validated RMSF data
    """
    logger.info("Validating RMSF data")
    
    # Check if the dataframe is empty
    if rmsf_data.empty:
        raise ValueError("Empty RMSF data")
    
    # Check for required columns
    required_columns = ['domain_id', 'resid', 'resname', 'average_rmsf']
    missing_columns = [col for col in required_columns if col not in rmsf_data.columns]
    
    if missing_columns:
        raise ValueError(f"RMSF data missing required columns: {missing_columns}")
    
    # Check for NaN values
    nan_counts = rmsf_data[required_columns].isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"RMSF data contains NaN values: {nan_counts}")
        
        # Drop rows with NaN values in required columns
        rmsf_data = rmsf_data.dropna(subset=required_columns)
        logger.info(f"Dropped rows with NaN values, {len(rmsf_data)} rows remain")
    
    # Check for negative RMSF values
    negative_rmsf = (rmsf_data['average_rmsf'] < 0).sum()
    if negative_rmsf > 0:
        logger.warning(f"RMSF data contains {negative_rmsf} negative values")
        
        # Filter out negative values
        rmsf_data = rmsf_data[rmsf_data['average_rmsf'] >= 0]
        logger.info(f"Filtered out negative RMSF values, {len(rmsf_data)} rows remain")
    
    # Check for duplicate (domain_id, resid) pairs
    duplicates = rmsf_data.duplicated(subset=['domain_id', 'resid']).sum()
    if duplicates > 0:
        logger.warning(f"RMSF data contains {duplicates} duplicate (domain_id, resid) pairs")
        
        # Keep only the first occurrence of each pair
        rmsf_data = rmsf_data.drop_duplicates(subset=['domain_id', 'resid'])
        logger.info(f"Removed duplicates, {len(rmsf_data)} rows remain")
    
    if rmsf_data.empty:
        raise ValueError("No valid RMSF data after validation")
    
    logger.info(f"RMSF data validation complete: {len(rmsf_data)} valid entries")
    
    # Print summary statistics
    logger.info(f"RMSF value range: [{rmsf_data['average_rmsf'].min():.4f}, {rmsf_data['average_rmsf'].max():.4f}]")
    logger.info(f"RMSF value mean: {rmsf_data['average_rmsf'].mean():.4f}")
    logger.info(f"Number of unique domains: {rmsf_data['domain_id'].nunique()}")
    logger.info(f"Number of unique residue types: {rmsf_data['resname'].nunique()}")
    
    return rmsf_data


def validate_domain_residue_mapping(
    voxel_data: Dict[str, Dict[str, np.ndarray]],
    rmsf_data: pd.DataFrame,
    domain_mapping: Dict[str, str]
) -> None:
    """
    Validate mapping between voxel and RMSF data at domain and residue level.
    
    Args:
        voxel_data: Dictionary mapping domain IDs to voxel data
        rmsf_data: DataFrame containing RMSF data
        domain_mapping: Mapping from voxel domain IDs to RMSF domain IDs
    """
    logger.info("Validating domain and residue mapping")
    
    # Count the number of domains in each dataset
    num_voxel_domains = len(voxel_data)
    num_rmsf_domains = rmsf_data['domain_id'].nunique()
    num_mapped_domains = len(domain_mapping)
    
    logger.info(f"Voxel domains: {num_voxel_domains}, RMSF domains: {num_rmsf_domains}, "
                f"Mapped domains: {num_mapped_domains}")
    
    # Analyze domain patterns
    voxel_domain_patterns = set()
    for domain in list(voxel_data.keys())[:min(100, len(voxel_data))]:
        if '_' in domain:
            pattern = domain.split('_', 1)[1]  # Everything after first '_'
            voxel_domain_patterns.add(pattern)
    
    rmsf_domain_patterns = set()
    for domain in rmsf_data['domain_id'].unique()[:min(100, len(rmsf_data['domain_id'].unique()))]:
        if '_' in domain:
            pattern = domain.split('_', 1)[1]  # Everything after first '_'
            rmsf_domain_patterns.add(pattern)
    
    logger.info(f"Voxel domain patterns: {voxel_domain_patterns}")
    logger.info(f"RMSF domain patterns: {rmsf_domain_patterns}")
    
    # Check mapping coverage
    unmapped_domains = [d for d in voxel_data.keys() if d not in domain_mapping]
    if unmapped_domains:
        logger.warning(f"{len(unmapped_domains)}/{num_voxel_domains} voxel domains could not be mapped to RMSF domains")
        logger.debug(f"Unmapped domains: {unmapped_domains[:5]}{'...' if len(unmapped_domains) > 5 else ''}")
        
        # Check if unmapped domains could be due to suffix issues
        suffix_issues = 0
        for domain in unmapped_domains[:min(50, len(unmapped_domains))]:
            base_name = domain.split('_')[0]
            for rmsf_domain in rmsf_data['domain_id'].unique():
                if rmsf_domain == base_name or rmsf_domain.startswith(base_name):
                    suffix_issues += 1
                    break
        
        if suffix_issues > 0:
            logger.warning(f"At least {suffix_issues} unmapped domains may be due to suffix differences between datasets")
    
    # Check residue mapping
    total_voxel_residues = sum(len(domain_data) for domain_data in voxel_data.values())
    mapped_residues = 0
    
    # Create a set of (domain_id, resid) pairs from RMSF data for faster lookup
    rmsf_domain_resid_set = set(
        (domain, resid) for domain, resid in 
        zip(rmsf_data['domain_id'], rmsf_data['resid'])
    )
    
    # Also create a set with base domain names for more flexible mapping
    rmsf_base_domain_resid_set = set()
    for domain, resid in zip(rmsf_data['domain_id'], rmsf_data['resid']):
        base_domain = domain.split('_')[0] if '_' in domain else domain
        rmsf_base_domain_resid_set.add((base_domain, resid))
    
    # Check each voxel residue
    residue_mapping_issues = 0
    sample_issues = []
    
    for voxel_domain, domain_data in voxel_data.items():
        if voxel_domain not in domain_mapping:
            continue
            
        rmsf_domain = domain_mapping[voxel_domain]
        
        for resid in domain_data:
            try:
                resid_int = int(resid)
                if (rmsf_domain, resid_int) in rmsf_domain_resid_set:
                    mapped_residues += 1
                else:
                    # Try base domain name as fallback
                    base_domain = voxel_domain.split('_')[0] if '_' in voxel_domain else voxel_domain
                    if (base_domain, resid_int) in rmsf_base_domain_resid_set:
                        mapped_residues += 1
                    else:
                        residue_mapping_issues += 1
                        if len(sample_issues) < 5:
                            sample_issues.append((voxel_domain, resid_int, rmsf_domain))
            except ValueError:
                logger.debug(f"Could not convert residue ID to integer: {resid}")
    
    logger.info(f"Successfully mapped {mapped_residues}/{total_voxel_residues} voxel residues to RMSF data "
                f"({mapped_residues/total_voxel_residues*100:.1f}%)")
    
    if residue_mapping_issues > 0:
        logger.warning(f"Found {residue_mapping_issues} residues that couldn't be mapped from voxel to RMSF data")
        if sample_issues:
            logger.debug(f"Sample mapping issues: {sample_issues}")
    
    if mapped_residues == 0:
        raise ValueError("No residues could be mapped between voxel and RMSF data")