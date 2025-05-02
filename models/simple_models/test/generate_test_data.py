# test/generate_test_data.py
"""
Generate synthetic protein data for testing the ensembleflex pipeline setup.

This script creates test data files at different temperatures which can then
be aggregated for ensembleflex testing.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# --- Configuration ---
TEST_DATA_DIR = Path('test/test_data')
N_DOMAINS = 3  # Reduced for faster testing
RESIDUES_PER_DOMAIN_AVG = 30 # Reduced for faster testing
TEMPERATURES = [320, 450] # Reduced, only need numeric for aggregation test
# Exclude "average" for default testing as aggregate script excludes it

def generate_data():
    """Generates the test data files."""
    # Create test_data directory if it doesn't exist
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ensured test data directory exists: {TEST_DATA_DIR.resolve()}")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Define domains
    domains = [f"test{i}A00" for i in range(1, N_DOMAINS + 1)]

    # Amino acids and their properties
    amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                   'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

    # Secondary structure types
    ss_types = ['H', 'E', 'C', 'T']  # Simplified

    # Surface exposure
    locations = ['core', 'surface'] # Changed from 'exterior' to match implementation

    # Temperature scaling factors (relative to 320K) - Simplified
    temp_scaling = {
        320: 1.0,
        450: 1.8,
    }

    # Generate data
    all_temp_data = {temp: [] for temp in TEMPERATURES}

    print("Generating synthetic data points...")
    for domain in domains:
        # Domain size
        protein_size = RESIDUES_PER_DOMAIN_AVG + np.random.randint(-3, 4)
        protein_size = max(10, protein_size) # Ensure minimum size

        # Generate secondary structure segments realistically
        ss_segments = []
        pos = 0
        while pos < protein_size:
            p = np.random.rand()
            if p < 0.4: ss_type, length = 'H', np.random.randint(5, 12)
            elif p < 0.7: ss_type, length = 'E', np.random.randint(3, 7)
            else: ss_type, length = 'C', np.random.randint(2, 6) # Coil/Turn
            ss_segments.append((ss_type, min(length, protein_size - pos)))
            pos += length
        secondary_structure = [ss for ss_type, length in ss_segments for ss in [ss_type] * length][:protein_size]

        # Generate data for each residue
        for resid in range(protein_size):
            normalized_resid = resid / max(1, protein_size - 1) # Avoid div by zero if size is 1
            dssp = secondary_structure[resid]

            # Assign other features randomly but plausibly
            resname = np.random.choice(amino_acids)
            core_exterior = np.random.choice(locations, p=[0.4, 0.6]) # Slightly more surface
            relative_accessibility = np.random.uniform(0, 0.3) if core_exterior == 'core' else np.random.uniform(0.3, 1.0)
            phi = np.random.uniform(-180, 180)
            psi = np.random.uniform(-180, 180)
            resname_encoded = amino_acids.index(resname) + 1 if resname in amino_acids else 0
            core_exterior_encoded = 0 if core_exterior == 'core' else 1
            ss_map_simple = {'H': 0, 'E': 1, 'C': 2, 'T': 2}
            secondary_structure_encoded = ss_map_simple.get(dssp, 2)
            phi_norm = np.sin(np.radians(phi))
            psi_norm = np.sin(np.radians(psi))

            # Base RMSF influenced by SS and position
            base_rmsf = 0.3 + 0.5 * (dssp == 'C') + 0.8 * (normalized_resid * (1 - normalized_resid))
            base_rmsf *= np.random.normal(1.0, 0.1) # Add noise

            # Generate plausible ESM/Voxel RMSF based on base RMSF
            esm_rmsf = base_rmsf * np.random.normal(1.0, 0.25)
            voxel_rmsf = base_rmsf * np.random.normal(1.0, 0.25)


            # Base data common to all temps for this residue
            base_row = {
                'domain_id': domain, 'resid': resid, 'resname': resname,
                'protein_size': protein_size, 'normalized_resid': normalized_resid,
                'core_exterior': core_exterior, 'relative_accessibility': relative_accessibility,
                'dssp': dssp, 'phi': phi, 'psi': psi,
                'resname_encoded': resname_encoded, 'core_exterior_encoded': core_exterior_encoded,
                'secondary_structure_encoded': secondary_structure_encoded,
                'phi_norm': phi_norm, 'psi_norm': psi_norm,
                'esm_rmsf': max(0, esm_rmsf), # Ensure non-negative
                'voxel_rmsf': max(0, voxel_rmsf) # Ensure non-negative
            }

            # Create row for each temperature
            for temp in TEMPERATURES:
                temp_row = base_row.copy()
                scaling = temp_scaling[temp]
                rmsf = base_rmsf * scaling * np.random.normal(1.0, 0.15) # Temp scaling + noise
                temp_row[f"rmsf_{temp}"] = max(0, rmsf) # Ensure non-negative
                all_temp_data[temp].append(temp_row)

    print("Synthetic data generation complete. Saving files...")
    # Save temperature-specific files
    for temp in TEMPERATURES:
        temp_df = pd.DataFrame(all_temp_data[temp])
        output_path = TEST_DATA_DIR / f'temperature_{temp}_train.csv'
        temp_df.to_csv(output_path, index=False)
        print(f"Created test data file: {output_path} with {len(temp_df)} rows")

if __name__ == "__main__":
    print("Running test data generation script...")
    generate_data()
    print("Test data generation finished!")