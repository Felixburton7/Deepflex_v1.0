#!/usr/bin/env python3

import h5py
import sys
import os
import numpy as np # Import numpy for calculating averages

# --- Configuration ---
# <<< SET THE PATH TO YOUR HDF5 FILE HERE >>>
HDF5_FILE_PATH = "/home/s_felix/mdcath-processor/outputs/voxelized/mdcath_voxelized.hdf5"  # Replace with the actual path to your .h5 file

# Show progress message every N PDB entries processed
PROGRESS_INTERVAL = 100 
# --- End Configuration ---

def summarize_h5_structure(filepath, progress_interval=100):
    """
    Summarizes the structure of an HDF5 file assumed to be created by aposteriori,
    including a progress indicator.

    Args:
        filepath (str): The path to the HDF5 file.
        progress_interval (int): How often (in PDB entries) to print progress.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        sys.exit(1)

    print(f"--- Summarizing HDF5 File: {filepath} ---")

    try:
        with h5py.File(filepath, 'r') as f:

            # --- 1. Root Level Attributes (Metadata) ---
            print("\n=== Dataset Metadata (Root Attributes) ===")
            if f.attrs:
                for key, value in f.attrs.items():
                    # Decode byte strings if necessary for printing
                    if isinstance(value, bytes):
                         try:
                             value_str = value.decode('utf-8')
                         except UnicodeDecodeError:
                             value_str = str(value) # Keep as bytes representation
                    elif isinstance(value, np.ndarray):
                         # Format numpy arrays for cleaner printing
                         value_str = np.array2string(value, precision=4, separator=', ', suppress_small=True)
                         if len(value_str) > 100: # Truncate long arrays
                             value_str = value_str[:100] + "...]" 
                    elif isinstance(value, (list, tuple)):
                         value_str = str(value)
                         if len(value_str) > 100: # Truncate long lists/tuples
                            value_str = value_str[:100] + "...]"
                    else:
                         value_str = str(value)
                    print(f"  - {key}: {value_str}")
            else:
                print("  No attributes found at the root level.")

            # --- 2. Structure Calculation (Internal) ---
            print(f"\n=== Calculating Structure Counts (this may take time, progress every {progress_interval} PDBs) ===") # Added message
            pdb_codes = []
            chains_per_pdb = []
            residues_per_pdb = []
            residues_per_chain = []
            total_chains = 0
            total_residues = 0
            example_residue_path = None
            example_residue_data = None
            pdb_counter = 0 # Initialize counter

            top_level_keys = list(f.keys()) # Get keys once to avoid issues if file changes during iteration (unlikely)

            # Iterate through PDB codes (top-level groups)
            for pdb_code in top_level_keys:
                # Ensure it's a group before proceeding
                if isinstance(f[pdb_code], h5py.Group):
                    pdb_codes.append(pdb_code)
                    pdb_group = f[pdb_code]
                    pdb_counter += 1 # Increment counter

                    # --- Optional Progress Print ---
                    if progress_interval > 0 and pdb_counter % progress_interval == 0:
                        print(f"  Processed {pdb_counter} PDB entries...")
                    # --- End Progress Print ---

                    num_chains_in_pdb = 0
                    num_residues_in_pdb = 0
                    
                    try:
                        chain_keys = list(pdb_group.keys())
                    except Exception as e:
                        print(f"  [Warning] Could not list keys for PDB group '{pdb_code}': {e}")
                        continue # Skip this PDB group if keys can't be read

                    # Iterate through Chain IDs within the PDB group
                    for chain_id in chain_keys:
                        if isinstance(pdb_group[chain_id], h5py.Group):
                            num_chains_in_pdb += 1
                            total_chains += 1
                            chain_group = pdb_group[chain_id]

                            num_residues_in_chain = 0
                            try:
                                residue_keys = list(chain_group.keys())
                            except Exception as e:
                                print(f"  [Warning] Could not list keys for chain '{chain_id}' in PDB '{pdb_code}': {e}")
                                continue # Skip this chain if keys can't be read

                            # Iterate through Residue IDs within the Chain group
                            for residue_id in residue_keys:
                                try:
                                    dataset = chain_group[residue_id]
                                    if isinstance(dataset, h5py.Dataset):
                                        num_residues_in_chain += 1
                                        total_residues += 1

                                        # Capture the first residue dataset as an example
                                        if example_residue_data is None:
                                            example_residue_path = f"/{pdb_code}/{chain_id}/{residue_id}"
                                            # Avoid loading full data here, just get metadata
                                            example_residue_data = dataset 
                                    # else: # Optional: uncomment to warn about unexpected items
                                    #      print(f"    [Internal Warning] Expected Dataset at /{pdb_code}/{chain_id}/{residue_id}, found {type(dataset)}")
                                except Exception as e:
                                     print(f"  [Warning] Could not access residue dataset '{residue_id}' in chain '{chain_id}', PDB '{pdb_code}': {e}")


                            if num_residues_in_chain > 0:
                                residues_per_chain.append(num_residues_in_chain)
                            num_residues_in_pdb += num_residues_in_chain

                        # else: # Optional: uncomment to warn about unexpected items
                        #     print(f"  [Internal Warning] Expected Group at /{pdb_code}/{chain_id}, found {type(pdb_group[chain_id])}")
                    
                    if num_chains_in_pdb > 0:
                        chains_per_pdb.append(num_chains_in_pdb)
                    if num_residues_in_pdb > 0:
                        residues_per_pdb.append(num_residues_in_pdb)

                # else: # Optional: uncomment to warn about unexpected top-level items
                #      print(f"[Internal Warning] Expected Group at top level '/{pdb_code}', found {type(f[pdb_code])}")

            # --- Add a final message after the loop ---
            print(f"  Finished counting across {pdb_counter} PDB entries.")
            # --- End final message ---

            # --- 3. Summary Statistics ---
            print("\n=== Summary Statistics ===")
            num_pdbs = len(pdb_codes) # This is the count of top-level PDB groups
            print(f"- Total PDB Codes (Domains): {num_pdbs}")
            print(f"- Total Chains across all PDBs: {total_chains}")
            print(f"- Total Residues (Frames): {total_residues}")

            if num_pdbs > 0 and chains_per_pdb:
                 avg_chains = np.mean(chains_per_pdb)
                 print(f"- Average Chains per PDB: {avg_chains:.2f}")
            else:
                 print("- Average Chains per PDB: N/A (No PDBs or chains found)")

            if num_pdbs > 0 and residues_per_pdb:
                 avg_residues_pdb = np.mean(residues_per_pdb)
                 print(f"- Average Residues (Frames) per PDB: {avg_residues_pdb:.2f}")
            else:
                 print("- Average Residues (Frames) per PDB: N/A (No PDBs or residues found)")


            if total_chains > 0 and residues_per_chain:
                 avg_residues_chain = np.mean(residues_per_chain)
                 print(f"- Average Residues (Frames) per Chain: {avg_residues_chain:.2f}")
            else:
                 print("- Average Residues (Frames) per Chain: N/A (No chains or residues found)")
            
            # Optional: Add Min/Max/Median if desired and data exists
            if chains_per_pdb:
                print(f"- Min/Median/Max Chains per PDB: {np.min(chains_per_pdb)} / {np.median(chains_per_pdb):.0f} / {np.max(chains_per_pdb)}")
            if residues_per_pdb:
                 print(f"- Min/Median/Max Residues per PDB: {np.min(residues_per_pdb)} / {np.median(residues_per_pdb):.0f} / {np.max(residues_per_pdb)}")
            if residues_per_chain:
                 print(f"- Min/Median/Max Residues per Chain: {np.min(residues_per_chain)} / {np.median(residues_per_chain):.0f} / {np.max(residues_per_chain)}")


            # --- 4. Example Residue Frame Details ---
            print("\n=== Example Residue Frame ===")
            if example_residue_data is not None:
                print(f"- Path: {example_residue_path}")
                print(f"- Shape: {example_residue_data.shape}")
                print(f"- Data Type: {example_residue_data.dtype}")
                print("- Attributes:")
                # Check attributes on the example dataset
                example_attrs = dict(example_residue_data.attrs)
                if example_attrs:
                    for key, value in example_attrs.items():
                        # Decode byte strings if necessary for printing
                        if isinstance(value, bytes):
                            try:
                                value_str = value.decode('utf-8')
                            except UnicodeDecodeError:
                                value_str = str(value) # Keep as bytes representation
                        elif isinstance(value, np.ndarray):
                             value_str = np.array2string(value, precision=4, separator=', ', suppress_small=True)
                             if len(value_str) > 100: value_str = value_str[:100] + "...]" 
                        elif isinstance(value, (list, tuple)):
                            value_str = str(value)
                            if len(value_str) > 100: value_str = value_str[:100] + "...]"
                        else:
                             value_str = str(value)
                        print(f"    - {key}: {value_str}")
                else:
                    print("    No attributes found for this residue dataset.")
            else:
                print("  No residue datasets found in the file to show an example.")


    except OSError as e:
        print(f"\nError opening or reading HDF5 file: {e}")
        print("Please ensure the file exists, is a valid HDF5 file, and you have read permissions.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred during processing: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        sys.exit(1)

    print(f"\n--- Summary Complete: {filepath} ---")

if __name__ == "__main__":
    # Make sure to install numpy and h5py: pip install numpy h5py
    summarize_h5_structure(HDF5_FILE_PATH, progress_interval=PROGRESS_INTERVAL)