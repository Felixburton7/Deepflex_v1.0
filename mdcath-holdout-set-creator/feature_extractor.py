import pandas as pd
import numpy as np
from collections import defaultdict
from utils import parallel_map  # Import our multiprocessing helper
from tqdm import tqdm  # Progress bar library

def _load_csv(file_path):
    """Helper function to load a CSV file and extract its temperature key."""
    temp = file_path.split('_')[-1].split('.')[0]
    if temp == 'average':
        temp = 'avg'
    df = pd.read_csv(file_path)
    return temp, df

def load_temperature_data(file_paths):
    """Load data from all temperature CSV files using multiprocessing."""
    temperature_data = {}
    results = parallel_map(_load_csv, file_paths)
    for temp, df in results:
        temperature_data[temp] = df
    return temperature_data

def aggregate_domain_features(temperature_data):
    """Compute domain-level features from residue-level data."""
    domain_features = {}

    # Get unique domain IDs from the first temperature file
    first_temp = list(temperature_data.keys())[0]
    domains = temperature_data[first_temp]['domain_id'].unique()

    # Use tqdm to wrap the domain processing loop
    for domain in tqdm(domains, desc="Computing domain-level features"):
        # Initialize feature dictionary for this domain
        domain_features[domain] = {}

        # Process each temperature dataset
        for temp, data in temperature_data.items():
            domain_data = data[data['domain_id'] == domain]

            if len(domain_data) == 0:
                continue

            # Basic features
            domain_features[domain]['size'] = domain_data['protein_size'].iloc[0]

            # Secondary structure composition
            dssp_counts = domain_data['dssp'].value_counts()
            total_residues = len(domain_data)

            # Calculate % of helix (H,G,I), sheet (E,B), and coil/turn (others)
            helix_pct = dssp_counts.get('H', 0) + dssp_counts.get('G', 0) + dssp_counts.get('I', 0)
            sheet_pct = dssp_counts.get('E', 0) + dssp_counts.get('B', 0)

            domain_features[domain]['helix_pct'] = helix_pct / total_residues if total_residues > 0 else 0
            domain_features[domain]['sheet_pct'] = sheet_pct / total_residues if total_residues > 0 else 0
            domain_features[domain]['coil_pct'] = 1 - (domain_features[domain]['helix_pct'] + domain_features[domain]['sheet_pct'])

            # Core/exterior ratio
            core_count = (domain_data['core_exterior'] == 'core').sum()
            domain_features[domain]['core_ratio'] = core_count / total_residues if total_residues > 0 else 0

            # Average accessibility
            domain_features[domain]['avg_accessibility'] = domain_data['relative_accessibility'].mean()

            # RMSF statistics for this temperature
            rmsf_col = f'rmsf_{temp}' if temp != 'avg' else 'rmsf_average'
            if rmsf_col in domain_data.columns:
                domain_features[domain][f'avg_rmsf_{temp}'] = domain_data[rmsf_col].mean()
                domain_features[domain][f'std_rmsf_{temp}'] = domain_data[rmsf_col].std()

    # Calculate temperature stability profiles
    for domain in domain_features:
        if '320' in temperature_data and '450' in temperature_data:
            if 'avg_rmsf_320' in domain_features[domain] and 'avg_rmsf_450' in domain_features[domain]:
                rmsf_ratio = domain_features[domain]['avg_rmsf_450'] / domain_features[domain]['avg_rmsf_320']

                # Assign stability class based on rmsf ratio
                if rmsf_ratio < 1.0:
                    domain_features[domain]['stability_class'] = 'stable'
                elif rmsf_ratio < 2.0:
                    domain_features[domain]['stability_class'] = 'moderate'
                else:
                    domain_features[domain]['stability_class'] = 'unstable'

    return domain_features
