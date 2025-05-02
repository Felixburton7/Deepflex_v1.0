# src/mdcath/voxel/aposteriori_wrapper.py
"""
Module to wrap and execute the external 'aposteriori' (make-frame-dataset) tool.
"""
import os
import logging
import subprocess
import sys
import shutil
import glob # Import glob
from typing import Dict, Any, List, Optional

class VoxelizationError(Exception):
    """Custom exception for Voxelization errors."""
    pass

def run_aposteriori(config: Dict[str, Any], output_base_dir: str, pdb_input_dir: str) -> bool:
    """
    Runs the 'make-frame-dataset' command from aposteriori, targeting static cleaned PDBs.

    Args:
        config (Dict[str, Any]): The 'voxelization' section of the main config.
        output_base_dir (str): The base output directory (e.g., './outputs').
        pdb_input_dir (str): Path to the directory containing the cleaned static PDB files
                             (e.g., './outputs/pdbs').

    Returns:
        bool: True if execution was successful (or skipped due to no input), False on failure.

    Raises:
        VoxelizationError: If configuration is invalid or execution fails critically.
    """
    if not config.get("enabled", True):
        logging.info("Voxelization step is disabled in the configuration. Skipping.")
        return True

    # --- Check Input Directory ---
    if not os.path.isdir(pdb_input_dir):
         logging.error(f"Input PDB directory for voxelization '{pdb_input_dir}' not found.")
         return False

    # Check if any .pdb files exist within the directory (non-recursively, as specified)
    try:
        # Use glob to find .pdb files directly in the specified directory
        pdb_files = glob.glob(os.path.join(pdb_input_dir, '*.pdb'))
        if not pdb_files:
            pdb_files_found = False
        else:
            pdb_files_found = True
            logging.info(f"Found {len(pdb_files)} PDB files in {pdb_input_dir}.")
    except Exception as glob_err:
        logging.error(f"Error checking for PDB files in {pdb_input_dir}: {glob_err}")
        return False

    if not pdb_files_found:
        logging.warning(f"Input directory '{pdb_input_dir}' contains no '.pdb' files. "
                        "Aposteriori will not be called, and no voxelized output will be generated.")
        return True # Return True as pipeline step didn't fail, just no input

    # --- Find Executable ---
    aposteriori_cmd = config.get("aposteriori_executable") or "make-frame-dataset"
    aposteriori_path = shutil.which(aposteriori_cmd)
    if not aposteriori_path:
         logging.error(f"Aposteriori command '{aposteriori_cmd}' not found in PATH. "
                       f"Install aposteriori ('pip install aposteriori') or provide full path in config.")
         return False
    logging.info(f"Using aposteriori executable: {aposteriori_path}")

    # --- Prepare Arguments ---
    voxel_output_dir = os.path.join(output_base_dir, "voxelized")
    os.makedirs(voxel_output_dir, exist_ok=True)

    output_name = config.get("output_name", "mdcath_voxelized")
    output_path_base = os.path.join(voxel_output_dir, output_name)
    final_output_hdf5 = output_path_base + ".hdf5"

    cmd_args = [
        aposteriori_path,
        "-o", voxel_output_dir,
        "-n", output_name,
        "-e", ".pdb", # Target PDB files
        "--frame-edge-length", str(config.get("frame_edge_length", 12.0)),
        "--voxels-per-side", str(config.get("voxels_per_side", 21)),
        "-ae", config.get("atom_encoder", "CNOCBCA"),
    ]

    cmd_args.extend(["-cb", "true" if config.get("encode_cb", True) else "false"])
    cmd_args.extend(["-comp", "true" if config.get("compression_gzip", True) else "false"])
    cmd_args.extend(["-vas", "true" if config.get("voxelise_all_states", False) else "false"])

    # *** USE INCREASED VERBOSITY (-vv) ***
    cmd_args.append("-vv")

    # *** NO RECURSIVE FLAG (-r) needed if input is ./outputs/pdbs ***

    # Add the input directory LAST (should be ./outputs/pdbs based on user)
    cmd_args.append(pdb_input_dir)

    # --- Execute Command ---
    logging.info(f"Running Aposteriori: {' '.join(cmd_args)}")
    try:
        process = subprocess.Popen(
            cmd_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )

        timeout_seconds = config.get("aposteriori_timeout", 3600)
        stdout, stderr = process.communicate(input='y\n', timeout=timeout_seconds)

        # *** Log stdout and stderr at INFO level for better visibility ***
        if stdout:
            logging.info(f"--- Aposteriori STDOUT ---\n{stdout}\n--- End Aposteriori STDOUT ---")
        else:
            logging.info("--- Aposteriori STDOUT: (empty) ---")

        if stderr:
             # Still log stderr appropriately based on exit code
             stderr_level = logging.ERROR if process.returncode != 0 else logging.WARNING
             logging.log(stderr_level, f"--- Aposteriori STDERR ---\n{stderr}\n--- End Aposteriori STDERR ---")
        else:
            # Log even if stderr is empty if the exit code is non-zero
            if process.returncode != 0:
                 logging.warning("--- Aposteriori STDERR: (empty) ---")


        if process.returncode != 0:
            raise VoxelizationError(f"Aposteriori command failed with exit code {process.returncode}. Check logs (stdout/stderr above) for details.")

        # --- Verify Output ---
        if not os.path.exists(final_output_hdf5):
            logging.error(f"Aposteriori finished (exit code 0) but expected output file '{final_output_hdf5}' was not found.")
            # Additional check: Did it create a file with a slightly different name?
            actual_files = glob.glob(os.path.join(voxel_output_dir, f"{output_name}*.hdf5"))
            if actual_files:
                logging.warning(f"Found potential output files, but not exact match: {actual_files}")
            return False
        elif os.path.getsize(final_output_hdf5) == 0:
             logging.warning(f"Aposteriori finished successfully, but the output file '{final_output_hdf5}' is empty. This indicates no residues/frames were successfully voxelized from the input PDBs in '{pdb_input_dir}'. Check Aposteriori stdout/stderr above for potential reasons (e.g., parsing issues, filtering).")
             # Return True because the command ran without error, even if output is empty.
             # Caller might need to handle the empty file case.
        else:
            logging.info(f"Aposteriori voxelization completed. Output: {final_output_hdf5} (Size: {os.path.getsize(final_output_hdf5)} bytes)")

        return True # Command executed, output file exists (even if empty)

    except subprocess.TimeoutExpired:
         logging.error(f"Aposteriori command timed out after {timeout_seconds} seconds.")
         if process: process.kill()
         try:
             # Attempt to grab any remaining output
             stdout, stderr = process.communicate()
             logging.error(f"Aposteriori stdout (on timeout):\n{stdout}")
             logging.error(f"Aposteriori stderr (on timeout):\n{stderr}")
         except Exception:
             logging.error("Failed to get further output after timeout.")
         return False
    except VoxelizationError as e: # Re-raise specific errors
         logging.error(f"Voxelization failed: {e}")
         return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during aposteriori execution: {e}", exc_info=True)
        if 'process' in locals() and hasattr(process, 'poll') and process.poll() is None:
             try:
                 process.kill()
                 process.communicate()
             except Exception: pass
        return False