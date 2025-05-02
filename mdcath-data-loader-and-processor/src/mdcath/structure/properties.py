"""
Module for calculating structural properties like DSSP, SASA, Core/Exterior.
"""
import os
import logging
import subprocess
import tempfile
import shutil
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from functools import lru_cache

# Attempt to import Biopython
try:
    from Bio.PDB import PDBParser, DSSP, HSExposureCB, NeighborSearch, ResidueDepth
    from Bio.PDB.SASA import ShrakeRupley
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    logging.error("BioPython library not found. Structural property calculation cannot proceed.")
    # Define dummy classes/functions if needed for program structure but operations will fail
    class PDBParser: pass
    class DSSP: pass
    class ShrakeRupley: pass

# Max size for LRU cache
CACHE_MAX_SIZE = 256

# Approximate maximum ASA values for normalization (from Tien et al. 2013 PLoS ONE, or similar source)
# Use 3-letter codes. Added default for unknown residues.
PDB_MAX_ASA = {
    'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0, 'CYS': 167.0,
    'GLN': 225.0, 'GLU': 223.0, 'GLY': 104.0, 'HIS': 224.0, 'ILE': 197.0,
    'LEU': 201.0, 'LYS': 236.0, 'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0,
    'SER': 155.0, 'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0,
    'UNK': 197.0 # Default fallback (e.g., average of ILE/LEU)
}


class PropertiesCalculatorError(Exception):
    """Custom exception for PropertiesCalculator errors."""
    pass

class PropertiesCalculator:
    """
    Calculates structural properties from cleaned PDB files.
    Uses DSSP executable and BioPython. Caches results.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the calculator with configuration.

        Args:
            config (Dict[str, Any]): The 'properties_calculation' section of the main config.

        Raises:
            ImportError: If BioPython is not installed.
        """
        if not BIOPYTHON_AVAILABLE:
            raise ImportError("BioPython is required for PropertiesCalculator but not found.")

        self.config = config
        self.dssp_executable = self._find_dssp()
        self.fallback_method = self.config.get('fallback_method', 'biopython') # Currently only 'biopython' fallback supported
        # Use relative ASA threshold by default
        self.relative_asa_core_threshold = self.config.get('relative_asa_core_threshold', 0.20)


        # Initialize BioPython parser and SASA calculator once
        self.parser = PDBParser(QUIET=True)
        self.sr = ShrakeRupley()


    def _find_dssp(self) -> Optional[str]:
        """Finds the DSSP executable path (dssp or mkdssp)."""
        configured_exec = self.config.get('dssp_executable', 'dssp')
        execs_to_try = [configured_exec]
        if configured_exec == 'dssp':
            execs_to_try.append('mkdssp')
        elif configured_exec == 'mkdssp':
             execs_to_try.append('dssp') # Try the other one if primary fails

        for exe in execs_to_try:
            path = shutil.which(exe)
            if path:
                logging.info(f"Found DSSP executable '{exe}' at: {path}")
                return path

        logging.warning(f"DSSP executable ('{configured_exec}' or its alternative) not found in PATH. "
                       f"DSSP calculation will use fallback methods if possible.")
        return None

        # Use file path as cache key. Ensure path is absolute/canonical if needed.
    @lru_cache(maxsize=CACHE_MAX_SIZE)
    def _run_dssp(self, cleaned_pdb_path: str) -> Optional[Dict[Tuple, Tuple]]:
        """Runs DSSP executable via BioPython wrapper and caches results."""
        if not self.dssp_executable:
            logging.debug(f"Skipping DSSP run: No DSSP executable found.")
            return None

        logging.debug(f"Running DSSP on {os.path.basename(cleaned_pdb_path)}...")
        try:
            structure = self.parser.get_structure(os.path.basename(cleaned_pdb_path), cleaned_pdb_path)
            model = structure[0]

            # DSSP execution
            dssp_results = DSSP(model, cleaned_pdb_path, dssp=self.dssp_executable)

            # DSSP object itself acts like a dictionary
            if not dssp_results or len(dssp_results) == 0:
                 logging.warning(f"DSSP returned empty results for {cleaned_pdb_path}.")
                 return None

            # Convert to a standard dict to ensure consistency before returning/caching
            # If iteration below fails, the issue might be here or how DSSP object behaves
            standardized_results = {key: dssp_results[key] for key in dssp_results.keys()}

            logging.debug(f"Successfully ran DSSP on {cleaned_pdb_path}. Found {len(standardized_results)} residues.")
            return standardized_results

        except FileNotFoundError:
             logging.error(f"DSSP executable '{self.dssp_executable}' not found during execution attempt.", exc_info=True)
             self.dssp_executable = None
             return None
        except Exception as e:
            # *** CORRECTED LOGGING CALL ***
            logging.log(logging.WARNING, f"DSSP execution or parsing failed for {cleaned_pdb_path}: {e}", exc_info=True)
            return None

    # Use file path as cache key. Consider if config changes should invalidate cache.
        # Use file path as cache key. Consider if config changes should invalidate cache.
    @lru_cache(maxsize=CACHE_MAX_SIZE)
    def calculate_properties(self, cleaned_pdb_path: str) -> Optional[pd.DataFrame]:
        """
        Calculates DSSP, SASA, Core/Exterior for a cleaned PDB file.
        Results are cached based on the pdb path.

        Args:
            cleaned_pdb_path (str): Path to the cleaned PDB file.

        Returns:
            Optional[pd.DataFrame]: DataFrame with columns
                ['resid', 'chain', 'dssp', 'relative_accessibility', 'core_exterior', 'phi', 'psi'],
                or None if calculations fail completely.

        Raises:
            PropertiesCalculatorError: If file not found or critical calculation fails.
        """
        if not os.path.exists(cleaned_pdb_path):
            raise PropertiesCalculatorError(f"PDB file not found for property calculation: {cleaned_pdb_path}")

        domain_id = os.path.splitext(os.path.basename(cleaned_pdb_path))[0]
        logging.debug(f"Calculating properties for {domain_id} from {cleaned_pdb_path}")

        # Attempt to get DSSP results (from cache or by running)
        dssp_results_dict = self._run_dssp(cleaned_pdb_path) # Returns a dict or None

        # --- Prepare structure for SASA and fallback ---
        try:
            structure = self.parser.get_structure(domain_id, cleaned_pdb_path)
            model = structure[0]
        except Exception as e:
             logging.error(f"Failed to parse PDB {cleaned_pdb_path} with BioPython: {e}", exc_info=True)
             raise PropertiesCalculatorError(f"BioPython parsing failed for {domain_id}") from e

        # --- Process residues ---
        property_list = []
        processed_keys = set() # Track DSSP keys to add fallback info later

        # 1. Process residues found by DSSP
        if dssp_results_dict:
            # *** CORRECTED ITERATION ***
            for key in dssp_results_dict.keys(): # Iterate over keys
                dssp_data = dssp_results_dict[key] # Get data for the key
                try:
                    chain_id = key[0]
                    res_id_tuple = key[1]
                    # Skip HETATM/water records potentially included by DSSP key format
                    if res_id_tuple[0] != ' ':
                         logging.debug(f"Skipping non-standard residue key from DSSP: {key}")
                         continue
                    res_num = res_id_tuple[1]

                    ss_code = dssp_data[2] if len(dssp_data) > 2 and dssp_data[2] not in [' ', '-'] else 'C'
                    phi = dssp_data[4] if len(dssp_data) > 4 and isinstance(dssp_data[4], (int, float)) else 0.0
                    psi = dssp_data[5] if len(dssp_data) > 5 and isinstance(dssp_data[5], (int, float)) else 0.0

                    property_list.append({
                        "key": key, "resid": res_num, "chain": chain_id,
                        "dssp": ss_code, "phi": phi, "psi": psi,
                        "relative_accessibility": np.nan, # Placeholder
                        "core_exterior": "unknown"      # Placeholder
                    })
                    processed_keys.add(key)
                except (IndexError, TypeError, KeyError) as e:
                     logging.warning(f"Could not parse DSSP data for key {key} in {domain_id}: {e}")

        # 2. Calculate SASA using ShrakeRupley for all standard residues in the parsed structure
        sasa_data = {} # Store {key: relative_asa}
        try:
            self.sr.compute(model, level="R") # Compute SASA at residue level
            for chain in model:
                for residue in chain:
                    key = (chain.id, residue.get_id())
                    if key[1][0] == ' ': # Standard residue (' ' flag)
                        resname = residue.get_resname()
                        # Handle non-standard AAs, ensure 3 letters before MAX_ASA lookup
                        if len(resname) != 3:
                            logging.debug(f"Skipping SASA for non-3-letter residue {resname} in {domain_id} {key}")
                            continue
                        if resname not in PDB_MAX_ASA:
                             logging.debug(f"Residue {resname} in {domain_id} {key} not in PDB_MAX_ASA map, using UNK default.")
                             resname_lookup = 'UNK'
                        else:
                             resname_lookup = resname

                        sasa_abs = residue.sasa if hasattr(residue, 'sasa') else 0.0
                        max_asa = PDB_MAX_ASA.get(resname_lookup, PDB_MAX_ASA['UNK'])
                        relative_accessibility = min(1.0, sasa_abs / max_asa) if max_asa > 0 else 0.0
                        sasa_data[key] = relative_accessibility

                        # Add defaults if residue was missed by DSSP run (e.g., if DSSP failed entirely)
                        if key not in processed_keys:
                             logging.debug(f"Adding fallback property data for residue {key} missed by DSSP in {domain_id}")
                             property_list.append({
                                 "key": key, "resid": key[1][1], "chain": key[0],
                                 "dssp": 'C', "phi": 0.0, "psi": 0.0,
                                 "relative_accessibility": np.nan, # Placeholder
                                 "core_exterior": "unknown"      # Placeholder
                             })
                             processed_keys.add(key) # Mark as processed

        except Exception as e:
             logging.error(f"SASA calculation failed for {domain_id}: {e}", exc_info=True)
             # Continue, relative_accessibility will be filled with default later

        # 3. Combine data and finalize
        if not property_list:
            logging.error(f"No property data could be generated for {domain_id}.")
            return None

        final_df = pd.DataFrame(property_list)

        # Fill in SASA and Core/Exterior using the calculated sasa_data map
        final_df['relative_accessibility'] = final_df['key'].map(sasa_data)
        final_df['relative_accessibility'].fillna(0.5, inplace=True) # Use default 0.5 if SASA mapping failed

        # Calculate core_exterior based on the relative accessibility
        final_df['core_exterior'] = np.where(
             final_df['relative_accessibility'] > self.relative_asa_core_threshold,
             'exterior',
             'core'
        )

        # Clean up and type conversion
        final_df = final_df.drop(columns=['key'])
        try:
            # Ensure correct types before returning
            final_df['resid'] = pd.to_numeric(final_df['resid'])
            final_df['relative_accessibility'] = pd.to_numeric(final_df['relative_accessibility'])
            final_df['phi'] = pd.to_numeric(final_df['phi'])
            final_df['psi'] = pd.to_numeric(final_df['psi'])
            final_df['chain'] = final_df['chain'].astype(str)
            final_df['dssp'] = final_df['dssp'].astype(str)
            final_df['core_exterior'] = final_df['core_exterior'].astype(str)

        except Exception as e:
            logging.error(f"Type conversion failed for property DataFrame of {domain_id}: {e}", exc_info=True)
            return None

        final_df = final_df.sort_values(by=['chain', 'resid']).reset_index(drop=True)
        logging.debug(f"Generated properties table with {len(final_df)} residues for {domain_id}")

        # Final check for NaNs in critical columns
        if final_df[['resid', 'chain', 'dssp', 'core_exterior']].isnull().any().any():
             logging.warning(f"Found null values in critical property columns for {domain_id}. Check processing.")

        return final_df

