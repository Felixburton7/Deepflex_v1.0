"""
Module for extracting specific simulation frames as PDB files.
"""
import os
import logging
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from mdcath.io.writers import save_string
try:
    from sklearn.cluster import KMeans # Import here if using K-Means method
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.info("scikit-learn not found. KMeans frame selection method will not be available.")


# Helper function to format PDB ATOM/HETATM line (ensure consistency)
def format_pdb_line(record_type, atom_num, atom_name, alt_loc, res_name, chain_id, res_num, icode, x, y, z, occupancy, temp_factor, element):
    """Formats a PDB ATOM/HETATM line string with proper spacing."""
    atom_name_padded = f"{atom_name:<4}"
    if len(atom_name_padded) > 4 : atom_name_padded = atom_name_padded[:4] # Ensure max 4 chars

    # Ensure res_name is string and pad
    res_name_str = str(res_name if res_name is not None else "UNK")
    res_name_padded = f"{res_name_str:>3}" # Right-justify resname

    # Ensure element symbol is string and formatted
    element_str = str(element if element is not None else "")
    element_padded = f"{element_str:>2}" # Right align 2 chars

    # Ensure numeric fields are correctly formatted
    occupancy_val = float(occupancy) if occupancy else 0.0
    temp_factor_val = float(temp_factor) if temp_factor else 0.0


    line = (
        f"{record_type:<6}"              # ATOM or HETATM
        f"{int(atom_num):>5} "           # Atom serial number
        f"{atom_name_padded:4}"          # Atom name
        f"{str(alt_loc):1}"              # Alt loc indicator
        f"{res_name_padded:3} "          # Residue name
        f"{str(chain_id):1}"             # Chain identifier
        f"{int(res_num):>4}"            # Residue sequence number
        f"{str(icode):1}"                # Code for insertion of residues
        f"   "                          # 3 spaces separator
        f"{x:8.3f}"                      # X coordinate
        f"{y:8.3f}"                      # Y coordinate
        f"{z:8.3f}"                      # Z coordinate
        f"{occupancy_val:6.2f}"          # Occupancy
        f"{temp_factor_val:6.2f}      "   # Temperature factor + spaces
        f"{element_padded:>2}  "         # Element symbol + charge placeholder space
        "\n"
    )
    return line

def select_frame_indices(num_available_frames: int,
                         num_frames_to_select: int,
                         method: str,
                         cluster_method: str = 'kmeans',
                         rmsd_data: Optional[np.ndarray] = None,
                         gyration_data: Optional[np.ndarray] = None) -> List[int]:
    """
    Selects frame indices based on the specified method.

    Args:
        num_available_frames (int): Total frames available in the trajectory.
        num_frames_to_select (int): Number of frames desired.
        method (str): Selection method ('regular', 'last', 'random', 'rmsd', 'gyration').
        cluster_method (str): Clustering method if method='rmsd'.
        rmsd_data (Optional[np.ndarray]): RMSD values for each frame (required for 'rmsd').
        gyration_data (Optional[np.ndarray]): Gyration radius values (required for 'gyration').

    Returns:
        List[int]: A list of selected frame indices.
    """
    if num_available_frames <= 0: return []
    if num_frames_to_select <= 0: return []

    # Ensure num_frames_to_select is not greater than available
    num_frames_to_select = min(num_frames_to_select, num_available_frames)

    indices = []
    if method == 'last':
        start_index = max(0, num_available_frames - num_frames_to_select)
        indices = list(range(start_index, num_available_frames))
        if num_frames_to_select > 1:
             logging.debug(f"Method 'last' selected. Selecting last {len(indices)} frames.")

    elif method == 'regular':
        # Generate evenly spaced indices including start and end points if possible
        indices = np.linspace(0, num_available_frames - 1, num_frames_to_select, dtype=int).tolist()

    elif method == 'random':
         indices = random.sample(range(num_available_frames), num_frames_to_select)

    elif method == 'gyration':
        if gyration_data is None or len(gyration_data) != num_available_frames:
            logging.warning("Gyration data missing or invalid for 'gyration' selection. Falling back to 'regular'.")
            return select_frame_indices(num_available_frames, num_frames_to_select, 'regular')
        # Select frames representing min, max, and intermediate gyration values
        sorted_indices = np.argsort(gyration_data)
        indices_of_indices = np.linspace(0, num_available_frames - 1, num_frames_to_select, dtype=int)
        indices = [sorted_indices[i] for i in indices_of_indices]

    elif method == 'rmsd':
        if rmsd_data is None or len(rmsd_data) != num_available_frames:
            logging.warning("RMSD data missing or invalid for 'rmsd' selection. Falling back to 'regular'.")
            return select_frame_indices(num_available_frames, num_frames_to_select, 'regular')

        if cluster_method == 'kmeans' and SKLEARN_AVAILABLE and num_available_frames >= num_frames_to_select > 0:
            try:
                rmsd_reshaped = rmsd_data.reshape(-1, 1)
                n_clusters = num_frames_to_select

                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                kmeans.fit(rmsd_reshaped)

                indices = []
                # Find frame closest to each centroid
                for i in range(n_clusters):
                     center = kmeans.cluster_centers_[i]
                     cluster_member_indices = np.where(kmeans.labels_ == i)[0]
                     if len(cluster_member_indices) > 0:
                         distances = np.abs(rmsd_reshaped[cluster_member_indices] - center)
                         closest_point_in_cluster_idx = np.argmin(distances)
                         original_frame_index = cluster_member_indices[closest_point_in_cluster_idx]
                         indices.append(original_frame_index)
                     else:
                         logging.warning(f"KMeans cluster {i} has no points for RMSD selection.")
                # If fewer frames selected than requested (e.g. due to identical RMSD values),
                # potentially top up with random/regular samples? For now, return what's found.

            except Exception as e:
                logging.error(f"KMeans clustering for RMSD selection failed: {e}. Falling back to 'regular'.", exc_info=True)
                return select_frame_indices(num_available_frames, num_frames_to_select, 'regular')
        else:
            if not SKLEARN_AVAILABLE:
                logging.warning("scikit-learn not installed. Cannot use 'kmeans' for RMSD selection.")
            else:
                logging.warning(f"Insufficient data or unsupported cluster method '{cluster_method}' for RMSD selection.")
            logging.info("Falling back to 'regular' frame selection.")
            return select_frame_indices(num_available_frames, num_frames_to_select, 'regular')
    else:
        logging.warning(f"Unknown frame selection method '{method}'. Falling back to 'regular'.")
        return select_frame_indices(num_available_frames, num_frames_to_select, 'regular')

    # Return unique, sorted indices
    return sorted(list(set(indices)))


def extract_and_save_frames(domain_id: str,
                            coords_all_frames: np.ndarray, # Shape (F, N, 3)
                            cleaned_pdb_template_path: str,
                            output_dir: str, # Base output dir (e.g., ./outputs)
                            config: Dict[str, Any],
                            rmsd_data: Optional[np.ndarray] = None,
                            gyration_data: Optional[np.ndarray] = None,
                            temperature: Optional[str] = None, # For naming output folder
                            replica: Optional[str] = None      # For naming output folder
                            ) -> bool:
    """
    Extracts specified frames from coordinate data and saves them as PDB files.

    Args:
        domain_id (str): Domain identifier.
        coords_all_frames (np.ndarray): Coordinate data for all frames (F, N, 3).
        cleaned_pdb_template_path (str): Path to the cleaned static PDB file to use as template.
        output_dir (str): Base output directory (e.g., './outputs').
        config (Dict[str, Any]): Pipeline configuration dictionary.
        rmsd_data (Optional[np.ndarray]): RMSD data for selection methods.
        gyration_data (Optional[np.ndarray]): Gyration data for selection methods.
        temperature (Optional[str]): Temperature identifier for output path.
        replica (Optional[str]): Replica identifier for output path.


    Returns:
        bool: True if at least one frame was successfully extracted and saved, False otherwise.
    """
    frame_cfg = config.get('processing', {}).get('frame_selection', {})
    num_frames_to_select = frame_cfg.get('num_frames', 1)
    method = frame_cfg.get('method', 'rmsd')
    cluster_method = frame_cfg.get('cluster_method', 'kmeans')

    if coords_all_frames is None or coords_all_frames.ndim != 3 or coords_all_frames.shape[0] == 0:
        logging.warning(f"No valid multi-frame coordinate data provided for {domain_id}. Skipping frame extraction.")
        return False

    num_available_frames = coords_all_frames.shape[0]
    num_atoms = coords_all_frames.shape[1]

    if not os.path.exists(cleaned_pdb_template_path):
        logging.error(f"Cleaned PDB template not found: {cleaned_pdb_template_path}. Cannot extract frames for {domain_id}.")
        return False

    # Read the template PDB
    try:
        with open(cleaned_pdb_template_path, 'r') as f:
            template_lines = f.readlines()
    except Exception as e:
        logging.error(f"Failed to read PDB template {cleaned_pdb_template_path}: {e}", exc_info=True)
        return False

    # Select frame indices
    selected_indices = select_frame_indices(num_available_frames, num_frames_to_select, method,
                                            cluster_method, rmsd_data, gyration_data)

    if not selected_indices:
        logging.warning(f"No frames selected for extraction for domain {domain_id}.")
        return False

    logging.info(f"Attempting to extract frames {selected_indices} for {domain_id} (Temp: {temperature}, Rep: {replica})")

    # --- Prepare mapping from template atom index to coordinate index ---
    # Assumption: Atom order in cleaned static PDB matches order in HDF5 coords.
    template_atom_lines_info = []
    for line in template_lines:
         if line.startswith(("ATOM", "HETATM")):
              # Store minimal info needed for line regeneration
              try:
                    info = {
                        "record_type": line[0:6].strip(),
                        "atom_num": int(line[6:11].strip()),
                        "atom_name": line[12:16].strip(),
                        "alt_loc": line[16:17].strip(),
                        "res_name": line[17:20].strip(),
                        "chain_id": line[21:22].strip(),
                        "res_num": int(line[22:26].strip()),
                        "icode": line[26:27].strip(),
                        "occupancy": line[54:60].strip(),
                        "temp_factor": line[60:66].strip(),
                        "element": line[76:78].strip() if len(line) >= 78 else line[12:14].strip()[0:1].upper(), # Guess element
                        "original_line": line # Keep original for non-atom lines
                    }
                    template_atom_lines_info.append(info)
              except ValueError as e:
                   logging.warning(f"Could not parse atom line in template {cleaned_pdb_template_path}: {line.strip()} - {e}")
                   template_atom_lines_info.append({"original_line": line}) # Keep line as is


    num_template_atoms = len([info for info in template_atom_lines_info if "record_type" in info])

    if num_template_atoms != num_atoms:
        logging.warning(f"Atom count mismatch for {domain_id}: Template PDB ({num_template_atoms}) vs Coordinates ({num_atoms}). "
                        f"Using minimum count ({min(num_template_atoms, num_atoms)}) for frame extraction.")
        num_atoms_to_process = min(num_template_atoms, num_atoms)
    else:
        num_atoms_to_process = num_atoms

    # --- Extract and save each selected frame ---
    saved_count = 0
    non_atom_header = [info["original_line"] for info in template_atom_lines_info if "record_type" not in info and not info["original_line"].startswith("END")]
    non_atom_footer = [info["original_line"] for info in template_atom_lines_info if "record_type" not in info and info["original_line"].startswith("END")]
    if not non_atom_footer: non_atom_footer = ["END\n"] # Ensure END record

    atom_template_info_to_use = [info for info in template_atom_lines_info if "record_type" in info][:num_atoms_to_process]

    for frame_idx in selected_indices:
        try:
            frame_coords = coords_all_frames[frame_idx, :num_atoms_to_process, :] # Use potentially truncated coords
            new_pdb_lines = list(non_atom_header) # Start with header lines

            for i, atom_info in enumerate(atom_template_info_to_use):
                x, y, z = frame_coords[i]
                new_line = format_pdb_line(
                    atom_info["record_type"], atom_info["atom_num"], atom_info["atom_name"],
                    atom_info["alt_loc"], atom_info["res_name"], atom_info["chain_id"],
                    atom_info["res_num"], atom_info["icode"], x, y, z,
                    atom_info["occupancy"], atom_info["temp_factor"], atom_info["element"]
                )
                new_pdb_lines.append(new_line)

            new_pdb_lines.extend(non_atom_footer) # Add footer lines

            # Define output path
            frame_output_dir = os.path.join(output_dir, "frames")
            if temperature is not None and replica is not None:
                 frame_output_dir = os.path.join(frame_output_dir, str(temperature), f"replica_{replica}")
            elif temperature is not None:
                 frame_output_dir = os.path.join(frame_output_dir, str(temperature))
            # Handle 'average' path if necessary

            output_filename = f"{domain_id}_frame_{frame_idx}.pdb"
            output_path = os.path.join(frame_output_dir, output_filename)

            # Save the new PDB file
            save_string("".join(new_pdb_lines), output_path)
            saved_count += 1
        except Exception as e:
            logging.error(f"Failed to process or save frame {frame_idx} for {domain_id}: {e}", exc_info=True)

    if saved_count > 0:
         logging.info(f"Successfully saved {saved_count} frames for {domain_id} (Temp: {temperature}, Rep: {replica})")
    else:
         logging.warning(f"Failed to save any frames for {domain_id} (Temp: {temperature}, Rep: {replica})")

    return saved_count > 0

