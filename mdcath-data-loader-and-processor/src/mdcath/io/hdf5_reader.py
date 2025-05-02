"""
Module for reading data selectively from mdCATH HDF5 files.
"""
import h5py
import logging
import os
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Any, Dict

class HDF5ReaderError(Exception):
    """Custom exception for HDF5Reader errors."""
    pass

class HDF5Reader:
    """
    Reads data from a single mdCATH HDF5 file.

    Args:
        h5_path (str): Path to the HDF5 file.

    Raises:
        FileNotFoundError: If the HDF5 file does not exist.
        HDF5ReaderError: If the file is invalid or key datasets are missing.
    """
    def __init__(self, h5_path: str):
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        self.h5_path = h5_path
        self.domain_id = self._extract_domain_id()
        self._validate_file() # Basic validation on init

    def _extract_domain_id(self) -> str:
        """Extracts domain ID from the filename."""
        basename = os.path.basename(self.h5_path)
        if basename.startswith("mdcath_dataset_") and basename.endswith(".h5"):
            return basename[len("mdcath_dataset_"):-len(".h5")]
        else:
            logging.warning(f"Could not parse standard domain ID from filename: {basename}. Using filename root.")
            return os.path.splitext(basename)[0]

    def _validate_file(self):
        """Performs basic validation of the HDF5 file structure."""
        try:
            with h5py.File(self.h5_path, 'r') as f:
                if self.domain_id not in f:
                    raise HDF5ReaderError(f"Domain ID group '{self.domain_id}' not found in {self.h5_path}")
                # Check for essential base datasets
                required_base = ['resid', 'resname', 'pdb']
                for req in required_base:
                    if req not in f[self.domain_id]:
                         raise HDF5ReaderError(f"Required base dataset '{req}' not found for domain {self.domain_id}")

                # Check for at least one temperature group (basic check)
                temp_groups = [key for key in f[self.domain_id].keys() if key.isdigit()]
                if not temp_groups:
                     logging.warning(f"No temperature groups found for domain {self.domain_id} in {self.h5_path}")
                else:
                     # Check structure within first temp/replica found
                     first_temp = temp_groups[0]
                     rep_groups = [key for key in f[f"{self.domain_id}/{first_temp}"].keys() if key.isdigit()]
                     if not rep_groups:
                          logging.warning(f"No replica groups found under temp {first_temp} for domain {self.domain_id}")
                     else:
                          first_rep = rep_groups[0]
                          rep_path = f"{self.domain_id}/{first_temp}/{first_rep}"
                          required_rep_data = ['coords', 'rmsf'] # Check essential dynamics data
                          for req_data in required_rep_data:
                              if req_data not in f[rep_path]:
                                   logging.warning(f"Required dataset '{req_data}' not found under {rep_path}. Processing may fail.")


        except HDF5ReaderError:
             raise # Propagate validation errors
        except Exception as e:
            logging.error(f"Failed to open or validate HDF5 file {self.h5_path}: {e}")
            raise HDF5ReaderError(f"HDF5 file access error: {e}") from e


    def get_pdb_string(self) -> Optional[str]:
        """Reads the PDB string from the HDF5 file."""
        path = f"{self.domain_id}/pdb"
        try:
            with h5py.File(self.h5_path, 'r') as f:
                if path not in f:
                     logging.error(f"'pdb' dataset not found for domain {self.domain_id} in {self.h5_path}")
                     return None
                pdb_dataset = f[path]
                pdb_data = pdb_dataset[()] # Read the scalar dataset
                if isinstance(pdb_data, bytes):
                    return pdb_data.decode('utf-8')
                elif isinstance(pdb_data, str):
                    return pdb_data
                else:
                     logging.warning(f"Unexpected data type for PDB string in {self.domain_id}: {type(pdb_data)}")
                     return str(pdb_data) # Attempt conversion
        except Exception as e:
            logging.error(f"Error reading PDB string for {self.domain_id}: {e}")
            return None

    def get_residue_info(self) -> Optional[pd.DataFrame]:
        """Reads unique residue number and name info."""
        try:
            with h5py.File(self.h5_path, 'r') as f:
                domain_group = f[self.domain_id]
                if 'resid' not in domain_group or 'resname' not in domain_group:
                     logging.error(f"Required residue datasets ('resid', 'resname') not found for {self.domain_id}")
                     return None

                resids = domain_group['resid'][:]
                resnames_raw = domain_group['resname'][:]

                # Create a minimal per-atom DataFrame to find unique residues
                # Assumption: first occurrence of a resid corresponds to its resname for the whole residue
                temp_df = pd.DataFrame({'resid': resids, 'resname_raw': resnames_raw})
                unique_residues_df = temp_df.drop_duplicates(subset='resid', keep='first').reset_index(drop=True)

                # Decode bytes if necessary
                unique_residues_df['resname'] = unique_residues_df['resname_raw'].apply(
                    lambda rn: rn.decode('utf-8') if isinstance(rn, bytes) else str(rn)
                )
                unique_residues_df = unique_residues_df[['resid', 'resname']] # Keep only needed columns

                # Get numResidues attribute if available for cross-check
                num_residues_attr = domain_group.attrs.get('numResidues')
                if num_residues_attr is not None and len(unique_residues_df) != num_residues_attr:
                     logging.warning(f"Number of unique residues found ({len(unique_residues_df)}) "
                                     f"does not match '.numResidues' attribute ({num_residues_attr}) for {self.domain_id}.")

                if unique_residues_df.empty:
                     logging.error(f"No unique residues found for domain {self.domain_id}.")
                     return None

                # Ensure resid is numeric
                unique_residues_df['resid'] = pd.to_numeric(unique_residues_df['resid'])

                return unique_residues_df

        except Exception as e:
            logging.error(f"Error reading residue info for {self.domain_id}: {e}", exc_info=True)
            return None

    def get_rmsf(self, temperature: int, replica: int) -> Optional[np.ndarray]:
        """Reads RMSF data for a specific temperature and replica."""
        path = f"{self.domain_id}/{temperature}/{replica}/rmsf"
        try:
            with h5py.File(self.h5_path, 'r') as f:
                if path not in f:
                     logging.warning(f"RMSF data not found at path: {path} in {self.h5_path}")
                     return None
                rmsf_data = f[path][:]
                # Basic shape check: should be 1D
                if rmsf_data.ndim != 1:
                     logging.warning(f"RMSF data at {path} is not 1D (shape: {rmsf_data.shape}). Attempting to flatten.")
                     rmsf_data = rmsf_data.flatten()
                return rmsf_data
        except Exception as e:
            logging.error(f"Error reading RMSF from {path} in {self.h5_path}: {e}")
            return None

    def get_coordinates(self, temperature: int, replica: int, frame_index: int = -1) -> Optional[np.ndarray]:
        """
        Reads coordinate data for a specific temperature, replica, and frame.

        Args:
            temperature (int): Simulation temperature.
            replica (int): Replica index.
            frame_index (int): Frame index to read (-1 for the last frame).
                                Use -999 to read ALL frames.

        Returns:
            Optional[np.ndarray]: Coordinates array (n_atoms, 3) for a single frame,
                                  or (n_frames, n_atoms, 3) if frame_index is -999,
                                  or None if reading fails.
        """
        path = f"{self.domain_id}/{temperature}/{replica}/coords"
        try:
            with h5py.File(self.h5_path, 'r') as f:
                if path not in f:
                    logging.warning(f"Coordinates data not found at path: {path} in {self.h5_path}")
                    return None

                coords_dataset = f[path]
                if coords_dataset.ndim != 3 or coords_dataset.shape[1] == 0 or coords_dataset.shape[2] != 3:
                     logging.error(f"Invalid coordinates shape {coords_dataset.shape} at {path}")
                     return None

                num_frames = coords_dataset.shape[0]
                if num_frames == 0:
                    logging.warning(f"Empty coordinates dataset at path: {path}")
                    return None

                if frame_index == -999: # Read all frames
                    coords_data = coords_dataset[:]
                else:
                    actual_index = frame_index if frame_index >= 0 else num_frames + frame_index
                    if 0 <= actual_index < num_frames:
                        coords_data = coords_dataset[actual_index]
                    else:
                        logging.error(f"Frame index {frame_index} (resolved to {actual_index}) out of bounds "
                                      f"(0-{num_frames-1}) for path {path}")
                        return None
                return coords_data
        except Exception as e:
            logging.error(f"Error reading coordinates from {path} in {self.h5_path}: {e}")
            return None

    def get_scalar_traj(self, temperature: int, replica: int, dataset_name: str) -> Optional[np.ndarray]:
        """Reads scalar trajectory data like 'rmsd' or 'gyrationRadius'."""
        path = f"{self.domain_id}/{temperature}/{replica}/{dataset_name}"
        try:
            with h5py.File(self.h5_path, 'r') as f:
                if path not in f:
                     logging.warning(f"Dataset '{dataset_name}' not found at path: {path} in {self.h5_path}")
                     return None
                data = f[path][:]
                # Ensure it's reasonably 1D
                if data.ndim > 1 and not (data.ndim == 2 and data.shape[1] == 1):
                     logging.warning(f"Expected scalar trajectory for '{dataset_name}' but got shape {data.shape}. Flattening.")
                     data = data.flatten()
                elif data.ndim == 0: # Handle scalar case if possible
                     data = np.array([data])

                # Basic check for expected length based on coords if possible
                # num_frames = self.get_num_frames(temperature, replica)
                # if num_frames is not None and len(data) != num_frames:
                #      logging.warning(f"Length mismatch for '{dataset_name}' ({len(data)}) vs num_frames ({num_frames}) at {path}")

                return data
        except Exception as e:
            logging.error(f"Error reading dataset {dataset_name} from {path} in {self.h5_path}: {e}")
            return None

    def get_num_frames(self, temperature: int, replica: int) -> Optional[int]:
         """Gets the number of frames attribute for a specific replica."""
         path = f"{self.domain_id}/{temperature}/{replica}"
         try:
             with h5py.File(self.h5_path, 'r') as f:
                 if path not in f:
                     logging.warning(f"Replica group not found at path: {path} in {self.h5_path}")
                     return None
                 # Try reading attribute '.numFrames'
                 num_frames_attr = f[path].attrs.get('.numFrames')
                 if num_frames_attr is not None:
                      return int(num_frames_attr)
                 else:
                      # Fallback: read shape of coords dataset if attribute missing
                      coords_path = f"{path}/coords"
                      if coords_path in f and f[coords_path].ndim == 3:
                           return f[coords_path].shape[0]
                      else:
                           logging.warning(f"Could not determine number of frames for {path}")
                           return None
         except Exception as e:
            logging.error(f"Error getting numFrames from {path} in {self.h5_path}: {e}")
            return None


    def get_available_temperatures(self) -> List[int]:
        """Returns a list of available temperatures (as integers) found in the file."""
        temps = []
        try:
            with h5py.File(self.h5_path, 'r') as f:
                if self.domain_id in f:
                    for key in f[self.domain_id].keys():
                        if key.isdigit():
                            temps.append(int(key))
            return sorted(temps)
        except Exception as e:
            logging.error(f"Error listing temperatures for {self.domain_id}: {e}")
            return []

    def get_available_replicas(self, temperature: int) -> List[int]:
         """Returns a list of available replicas (as integers) for a given temperature."""
         reps = []
         temp_path = f"{self.domain_id}/{temperature}"
         try:
             with h5py.File(self.h5_path, 'r') as f:
                 if temp_path in f:
                      for key in f[temp_path].keys():
                           if key.isdigit():
                                reps.append(int(key))
             return sorted(reps)
         except Exception as e:
            logging.error(f"Error listing replicas for {self.domain_id}, temp {temperature}: {e}")
            return []

