"""
Module for cleaning and processing PDB files.
Uses pdbUtils if available, otherwise provides a fallback.
"""
import os
import logging
import subprocess
import tempfile
from typing import Dict, Any

# Attempt to import pdbUtils
try:
    from pdbUtils import pdbUtils
    PDBUTILS_AVAILABLE = True
    logging.debug("pdbUtils library found. Will be used for PDB processing if configured.")
except ImportError:
    PDBUTILS_AVAILABLE = False
    logging.debug("pdbUtils library not found. Fallback method will be used if pdbUtils is configured.")

class PDBProcessorError(Exception):
    """Custom exception for PDBProcessor errors."""
    pass

class PDBProcessor:
    """
    Cleans and processes PDB files according to configuration.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the PDBProcessor with cleaning configuration.

        Args:
            config (Dict[str, Any]): The 'pdb_cleaning' section of the main config.
        """
        self.config = config
        self._setup_tool()

    def _setup_tool(self):
        """Determines which cleaning tool to use based on config and availability."""
        configured_tool = self.config.get("tool", "pdbutils")
        if configured_tool == "pdbutils":
            if PDBUTILS_AVAILABLE:
                self.use_pdbutils = True
                logging.info("Using pdbUtils for PDB cleaning.")
            else:
                self.use_pdbutils = False
                logging.warning("pdbUtils configured but not found. Switching to fallback PDB cleaning.")
        elif configured_tool == "fallback":
            self.use_pdbutils = False
            logging.info("Using fallback method for PDB cleaning.")
        else:
            logging.warning(f"Unknown PDB cleaning tool '{configured_tool}' specified in config. Using fallback.")
            self.use_pdbutils = False

    def clean_pdb_string(self, pdb_string: str, output_pdb_path: str) -> bool:
        """
        Cleans a PDB string and saves the result to a file.

        Args:
            pdb_string (str): The PDB content as a single string.
            output_pdb_path (str): Path where the cleaned PDB file will be saved.

        Returns:
            bool: True if cleaning was successful, False otherwise.
        """
        # Write the input string to a temporary file for processing
        # Use NamedTemporaryFile for automatic cleanup
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".pdb", encoding='utf-8') as temp_in:
                temp_in_path = temp_in.name
                temp_in.write(pdb_string)
            # Now process the temporary file
            success = self.clean_pdb_file(temp_in_path, output_pdb_path)
        except Exception as e:
             logging.error(f"Error creating or writing temporary input PDB: {e}", exc_info=True)
             success = False
        finally:
            # Ensure temporary input file is removed
            if 'temp_in_path' in locals() and os.path.exists(temp_in_path):
                try:
                    os.remove(temp_in_path)
                except OSError as e:
                    logging.warning(f"Could not remove temporary input file {temp_in_path}: {e}")
        return success


    def clean_pdb_file(self, input_pdb_path: str, output_pdb_path: str) -> bool:
        """
        Cleans a PDB file using the configured method.

        Args:
            input_pdb_path (str): Path to the input PDB file.
            output_pdb_path (str): Path where the cleaned PDB file will be saved.

        Returns:
            bool: True if cleaning was successful, False otherwise.
        """
        if not os.path.exists(input_pdb_path):
            logging.error(f"Input PDB file not found: {input_pdb_path}")
            return False

        logging.debug(f"Cleaning PDB file: {os.path.basename(input_pdb_path)} -> {os.path.basename(output_pdb_path)}")
        try:
            os.makedirs(os.path.dirname(output_pdb_path), exist_ok=True)
            if self.use_pdbutils:
                return self._clean_with_pdbutils(input_pdb_path, output_pdb_path)
            else:
                return self._clean_with_fallback(input_pdb_path, output_pdb_path)
        except Exception as e:
            logging.error(f"Failed to clean PDB {input_pdb_path}: {e}", exc_info=True)
            return False

    def _clean_with_pdbutils(self, input_path: str, output_path: str) -> bool:
        """Cleans PDB using pdbUtils library."""
        try:
            pdb_df = pdbUtils.pdb2df(input_path)
            initial_atoms = len(pdb_df)

            # --- Apply Filters First ---
            # Remove solvent/ions first to potentially simplify TER handling
            if self.config.get("remove_solvent_ions", True):
                 skip_resnames = {"TIP", "HOH", "WAT", "SOD", "CLA", "K", "MG", "ZN", "CA", "CL-", "NA+", "K+"} # More variants
                 if "RES_NAME" in pdb_df.columns:
                      pdb_df = pdb_df[~pdb_df["RES_NAME"].isin(skip_resnames)]
                 # Consider chain removal if needed (e.g., Chain 'W')
                 # if "CHAIN_ID" in pdb_df.columns:
                 #     pdb_df = pdb_df[pdb_df["CHAIN_ID"] != "W"]

            # Remove hydrogens
            if self.config.get("remove_hydrogens", False):
                 # More robust hydrogen check: element or atom name start
                 if "ELEMENT" in pdb_df.columns:
                      pdb_df = pdb_df[pdb_df["ELEMENT"] != "H"]
                 elif "ATOM_NAME" in pdb_df.columns: # Fallback if element missing
                      # Handle names like '1H', '2H', 'HD1', etc.
                      pdb_df = pdb_df[~pdb_df["ATOM_NAME"].str.strip().str.match(r"^[1-9]?[HD]")]
                 else:
                      logging.warning("Cannot remove hydrogens: Neither ELEMENT nor ATOM_NAME column found.")


            # Stop after TER: Apply filter after other removals
            if self.config.get("stop_after_ter", True):
                 # Find the index of the first non-ATOM/HETATM record after initial filtering
                 # This is complex with DataFrames. Simpler approach: Re-parse after saving temporarily?
                 # Let's assume for now that pdbUtils' df2pdb writes a sensible structure
                 # and subsequent reads won't include much beyond the main chain if solvent etc. removed.
                 # A truly strict TER stop might need the fallback line-by-line approach.
                 logging.debug("pdbUtils TER handling: Assuming removal of solvent/ions suffices or df2pdb handles it.")


            # --- Apply Corrections ---
            # Chain ID '0' -> 'A'
            if self.config.get("replace_chain_0_with_A", True) and "CHAIN_ID" in pdb_df.columns:
                pdb_df["CHAIN_ID"] = pdb_df["CHAIN_ID"].replace("0", "A")

            # Correct residue names (HIS variants)
            if self.config.get("correct_unusual_residue_names", True) and "RES_NAME" in pdb_df.columns:
                his_map = {"HSD": "HIS", "HSE": "HIS", "HSP": "HIS"}
                pdb_df["RES_NAME"] = pdb_df["RES_NAME"].replace(his_map)

            # Fix atom numbering (sequential) AFTER all filtering
            if self.config.get("fix_atom_numbering", True) and "ATOM_NUM" in pdb_df.columns:
                 pdb_df["ATOM_NUM"] = range(1, len(pdb_df) + 1)

            # Ensure essential columns exist for df2pdb, add placeholders if missing?
            # Depends on pdbUtils requirements.

            pdbUtils.df2pdb(pdb_df, output_path)
            final_atoms = len(pdb_df)
            logging.info(f"Cleaned PDB (pdbUtils): {os.path.basename(input_path)} ({initial_atoms} atoms -> {final_atoms} atoms)")
            # Post-process: Ensure CRYST1 if needed (e.g., for DSSP)
            self._ensure_cryst1(output_path)
            return True
        except Exception as e:
            logging.error(f"pdbUtils cleaning failed for {input_path}: {e}", exc_info=True)
            return False

    def _clean_with_fallback(self, input_path: str, output_path: str) -> bool:
        """Cleans PDB using basic line-by-line processing."""
        cleaned_lines = []
        atom_counter = 1
        ter_encountered = False
        last_res_info = "" # To format TER card correctly

        # Read config settings
        stop_after_ter = self.config.get("stop_after_ter", True)
        skip_resnames = {"TIP", "HOH", "WAT", "SOD", "CLA", "K", "MG", "ZN", "CA", "CL-", "NA+", "K+"} if self.config.get("remove_solvent_ions", True) else set()
        remove_h = self.config.get("remove_hydrogens", False)
        correct_res = self.config.get("correct_unusual_residue_names", True)
        fix_chain0 = self.config.get("replace_chain_0_with_A", True)
        fix_numbering = self.config.get("fix_atom_numbering", True)
        his_map = {"HSD": "HIS", "HSE": "HIS", "HSP": "HIS"}

        try:
            with open(input_path, 'r') as infile:
                # Check for CRYST1 early, add if missing and needed
                # Note: This simplistic check won't fix a malformed CRYST1
                infile_content = infile.readlines() # Read all lines first for CRYST1 check
                has_cryst1 = any(line.startswith("CRYST1") for line in infile_content)
                if not has_cryst1:
                    cleaned_lines.append("CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n")
                    logging.debug("Added default CRYST1 record.")

                infile.seek(0) # Rewind not needed as we iterate over infile_content

                for line_num, line in enumerate(infile_content):
                    line = line.rstrip() # Remove trailing whitespace
                    if not line: continue # Skip empty lines

                    # Check TER condition first
                    if stop_after_ter and ter_encountered:
                        if line.startswith("END"):
                            cleaned_lines.append(line + "\n")
                        continue

                    if line.startswith("TER"):
                        ter_atom_idx = atom_counter # Use the *next* atom number
                        # Try to format TER using last residue info
                        # TER card format: TER serialNumber resName chainID resSeq iCode
                        ter_line = f"TER   {ter_atom_idx:>5}      {last_res_info}\n" # last_res_info is "resName chainID resSeq"
                        cleaned_lines.append(ter_line)
                        ter_encountered = True
                        last_res_info = "" # Reset for next chain if any
                        continue

                    if line.startswith("END"):
                         cleaned_lines.append(line + "\n")
                         break

                    if line.startswith(("ATOM", "HETATM")):
                        # --- Parse line carefully ---
                        try:
                            record_type = line[0:6]
                            atom_name = line[12:16] # Keep spacing
                            alt_loc = line[16:17]
                            res_name = line[17:20].strip()
                            chain_id = line[21:22].strip()
                            res_num_str = line[22:26].strip()
                            icode = line[26:27]
                            x_coord_str = line[30:38].strip()
                            y_coord_str = line[38:46].strip()
                            z_coord_str = line[46:54].strip()
                            occupancy_str = line[54:60].strip()
                            temp_factor_str = line[60:66].strip()
                            element = line[76:78].strip() if len(line) >= 78 else ''
                            # charge = line[78:80].strip() if len(line) >= 80 else ''

                            # Basic validation of parsed numbers
                            res_num = int(res_num_str)
                            x_coord = float(x_coord_str)
                            y_coord = float(y_coord_str)
                            z_coord = float(z_coord_str)
                            occupancy = float(occupancy_str) if occupancy_str else 0.0
                            temp_factor = float(temp_factor_str) if temp_factor_str else 0.0

                        except (ValueError, IndexError) as parse_error:
                            logging.warning(f"Skipping malformed ATOM/HETATM line {line_num+1} in {os.path.basename(input_path)}: {parse_error} -> '{line}'")
                            continue

                        # --- Apply Filters/Modifications ---
                        # Element check for Hydrogen removal
                        if not element: element = atom_name.strip()[0:1].upper() # Simple guess if missing
                        if remove_h and (element == 'H' or atom_name.strip().startswith(('H','D','1','2','3'))): # More robust H check
                            continue

                        # Residue name filter (solvent/ions)
                        if res_name in skip_resnames:
                            continue

                        # --- Apply Corrections ---
                        if correct_res and res_name in his_map: res_name = his_map[res_name]
                        if fix_chain0 and chain_id == '0': chain_id = 'A'

                        # --- Format Output ---
                        current_atom_num = atom_counter if fix_numbering else int(line[6:11].strip())
                        formatted_line = format_pdb_line(
                             record_type, current_atom_num, atom_name.strip(), alt_loc.strip(), res_name,
                             chain_id, res_num, icode.strip(), x_coord, y_coord, z_coord,
                             occupancy, temp_factor, element
                        )
                        cleaned_lines.append(formatted_line)
                        last_res_info = f"{res_name:>3} {chain_id:1}{res_num:>4}{icode:1}".strip() # Store for TER card
                        atom_counter += 1
                    elif not ter_encountered: # Keep other lines only before first TER if stopping
                        cleaned_lines.append(line + "\n") # Append original line (e.g., REMARK, CRYST1)

            # Add END if not present and not stopped early by TER break
            if not cleaned_lines or not cleaned_lines[-1].startswith("END"):
                 cleaned_lines.append("END\n")

            # Write cleaned file
            with open(output_path, 'w') as outfile:
                outfile.writelines(cleaned_lines)

            logging.info(f"Cleaned PDB (Fallback): {os.path.basename(input_path)} ({atom_counter-1} atoms written)")
            # Ensure CRYST1 after writing (in case original was missing)
            self._ensure_cryst1(output_path)
            return True

        except Exception as e:
            logging.error(f"Fallback cleaning failed for {input_path}: {e}", exc_info=True)
            return False

    def _ensure_cryst1(self, pdb_path: str):
        """Checks if a PDB file has a CRYST1 record, adds a default if missing."""
        try:
            with open(pdb_path, 'r') as f:
                lines = f.readlines()
            has_cryst1 = any(line.startswith("CRYST1") for line in lines)
            if not has_cryst1:
                logging.debug(f"Adding default CRYST1 record to {pdb_path}")
                default_cryst1 = "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n"
                with open(pdb_path, 'w') as f:
                    f.write(default_cryst1)
                    f.writelines(lines)
        except Exception as e:
            logging.warning(f"Could not ensure CRYST1 record for {pdb_path}: {e}")


