# -*- coding: utf-8 -*-
"""
Orchestrates the mdCATH processing pipeline steps.
Includes parallel processing worker function to avoid pickling errors
and uses intermediate disk storage to manage memory.
"""
import os
import glob
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from tqdm import tqdm
import importlib
import inspect
import traceback # Import traceback module
import joblib # Import joblib for intermediate data saving/loading
import shutil # For cleaning up intermediate directory

# Import package components
from ..config import load_config
from ..io.hdf5_reader import HDF5Reader, HDF5ReaderError
from ..structure.pdb_processor import PDBProcessor
from ..structure.properties import PropertiesCalculator
from ..structure.frame_extractor import extract_and_save_frames
from ..metrics.rmsf import process_domain_rmsf, aggregate_and_average_rmsf, save_rmsf_results
from ..features.builder import FeatureBuilder, FeatureBuilderError
from ..voxel.aposteriori_wrapper import run_aposteriori, VoxelizationError
from ..visualize import plots as viz_plots
from ..utils.parallel import parallel_map
from ..io.writers import save_dataframe_csv

# Import custom exceptions
from ..exceptions import (
    RmsfProcessingError,
    PDBProcessorError,
    PropertiesCalculatorError,
    HDF5ReaderError,
    PipelineExecutionError,
    FeatureBuilderError,
    VoxelizationError,
    ConfigurationError
)

# Define PipelineExecutionError if not defined in exceptions.py
if 'PipelineExecutionError' not in locals():
    class PipelineExecutionError(Exception):
        """Custom exception for pipeline execution errors."""
        pass

# Define type hint for RMSF result path
RmsfResultPath = Optional[str]
# Define type hint for Properties result path
PropertiesResultPath = Optional[str]

# --- Top-level Worker Function for Parallel Processing (Modified) ---
def _parallel_domain_worker(args_tuple: Tuple[str, Dict, str]) -> Tuple[str, str, RmsfResultPath, PropertiesResultPath, Optional[str]]:
    """
    Worker function for parallel domain processing. Initializes components internally,
    saves results to disk, and returns paths.

    Args:
        args_tuple (Tuple): Contains (domain_id, config_dict, output_dir)

    Returns:
        Tuple: (domain_id, status, rmsf_result_path, properties_result_path, cleaned_pdb_path)
               Paths are None if the corresponding step failed or saving failed.
    """
    domain_id, config_dict, output_dir = args_tuple
    status = f"Starting (worker) {domain_id}"
    rmsf_result = None
    properties_result = None
    cleaned_pdb_path = None
    rmsf_result_path: RmsfResultPath = None
    properties_result_path: PropertiesResultPath = None
    intermediate_dir = os.path.join(output_dir, "intermediate", domain_id)

    try:
        # Create intermediate directory for this domain
        os.makedirs(intermediate_dir, exist_ok=True)

        # Initialize components INSIDE the worker to avoid pickling issues
        pdb_processor = PDBProcessor(config_dict.get('processing', {}).get('pdb_cleaning', {}))
        properties_calculator = PropertiesCalculator(config_dict.get('processing', {}).get('properties_calculation', {}))

        input_cfg = config_dict.get('input', {})
        mdcath_folder = input_cfg.get('mdcath_folder')
        h5_path = os.path.join(mdcath_folder, f"mdcath_dataset_{domain_id}.h5")
        reader = HDF5Reader(h5_path)
        status = f"Read HDF5 OK"

        pdb_string = reader.get_pdb_string()
        if not pdb_string: status = "Failed PDB Read"; raise ValueError("Could not read PDB string")

        pdb_out_dir = os.path.join(output_dir, "pdbs")
        os.makedirs(pdb_out_dir, exist_ok=True)
        cleaned_pdb_path = os.path.join(pdb_out_dir, f"{domain_id}.pdb")

        if not pdb_processor.clean_pdb_string(pdb_string, cleaned_pdb_path):
             status = "Failed PDB Clean"; raise PDBProcessorError("PDB cleaning failed")
        status = "PDB Clean OK"

        properties_result = properties_calculator.calculate_properties(cleaned_pdb_path)
        if properties_result is None or properties_result.empty:
             status = "Failed Properties Calc"; raise PropertiesCalculatorError("Property calculation failed")
        # Save Properties Result
        try:
            properties_result_path_temp = os.path.join(intermediate_dir, "properties.parquet")
            properties_result.to_parquet(properties_result_path_temp, index=False)
            properties_result_path = properties_result_path_temp # Assign path only on success
            status = "Properties Calc OK"
        except Exception as e_save:
             logging.error(f"Worker error {domain_id}: Failed to save properties: {e_save}")
             status = "Failed Properties Save" # Update status
             properties_result_path = None # Ensure path is None on failure

        # Proceed only if properties were calculated and saved (or saving failed but calc OK)
        if status not in ["Properties Calc OK", "Failed Properties Save"]:
             raise PropertiesCalculatorError("Stopping worker due to properties calc/save failure")


        rmsf_result_tuple = process_domain_rmsf(domain_id, reader, config_dict)
        if rmsf_result_tuple is None:
            status = "Failed RMSF Proc"; raise RmsfProcessingError("RMSF processing failed")
        rmsf_result = (rmsf_result_tuple[0].copy(), rmsf_result_tuple[1])

        # Save RMSF Result
        try:
            rmsf_result_path_temp = os.path.join(intermediate_dir, "rmsf_data.joblib")
            joblib.dump(rmsf_result, rmsf_result_path_temp, compress=3) # Use compression
            rmsf_result_path = rmsf_result_path_temp # Assign path only on success
            status = "RMSF Proc OK"
        except Exception as e_save:
             logging.error(f"Worker error {domain_id}: Failed to save RMSF data: {e_save}")
             status = "Failed RMSF Save" # Update status
             rmsf_result_path = None # Ensure path is None on failure

        # Final status depends on successful saving of both
        if properties_result_path and rmsf_result_path:
            status = "Success"
        else:
            # Keep the more specific failure status if saving failed
            if status not in ["Failed Properties Save", "Failed RMSF Save"]:
                 status = "Failed Intermediate Save"


    # --- Exception Handling ---
    # Prioritize specific failure statuses set during saving/processing
    except (FileNotFoundError, HDF5ReaderError, ValueError) as e:
         if not status.startswith("Failed"): status = "Failed HDF5 Read/Access"
         logging.error(f"Worker error {domain_id} (Read): {e}")
    except PDBProcessorError as e:
         if not status.startswith("Failed"): status = "Failed PDB Clean"
         logging.error(f"Worker error {domain_id} (Clean): {e}")
    except PropertiesCalculatorError as e:
         if not status.startswith("Failed"): status = "Failed Properties Calc"
         logging.error(f"Worker error {domain_id} (Properties): {e}")
    except RmsfProcessingError as e:
         if not status.startswith("Failed"): status = "Failed RMSF Proc"
         logging.error(f"Worker error {domain_id} (RMSF): {e}")
    except Exception as e:
        # Catch unexpected errors, keep specific save errors if they occurred
        if not status.startswith("Failed"):
             status = f"Failed Unexpected (Worker): {type(e).__name__}"
        logging.error(f"Unexpected worker error {domain_id}: {e}", exc_info=True)


    # Return paths (which might be None if saving failed)
    return domain_id, status, rmsf_result_path, properties_result_path, cleaned_pdb_path


# --- PipelineExecutor Class (Modified) ---
class PipelineExecutor:
    """ Manages and executes the mdCATH processing pipeline using intermediate disk storage. """
    def __init__(self, config_dict: Dict[str, Any]):
        """ Initializes the executor with a configuration dictionary. """
        try:
            self.config = config_dict
            self.output_dir = self.config.get('output', {}).get('base_dir', './outputs')
            self.intermediate_base_dir = os.path.join(self.output_dir, "intermediate") # Define base intermediate dir
            self.num_cores = self.config.get('performance', {}).get('num_cores', 0)
            cpu_count = os.cpu_count()
            if self.num_cores <= 0:
                 self.num_cores = max(1, cpu_count - 2 if cpu_count is not None and cpu_count > 2 else 1)
            else:
                 self.num_cores = self.num_cores
            logging.info(f"Determined number of cores to use: {self.num_cores}")

            # Pipeline state
            self.domain_list: List[str] = []
            self.domain_status: Dict[str, str] = {}
            # Store paths to intermediate files, not the data itself
            self.rmsf_file_paths: Dict[str, RmsfResultPath] = {}
            self.properties_file_paths: Dict[str, PropertiesResultPath] = {}
            self.cleaned_pdb_paths: Dict[str, Optional[str]] = {}
            # Aggregated data (will be built by reading intermediate files)
            self.voxel_output_file: Optional[str] = None
            self.agg_replica_data: Optional[Dict] = None
            self.agg_replica_avg_data: Optional[Dict] = None
            self.agg_overall_avg_data: Optional[pd.DataFrame] = None
            # Final feature dataframes (built by reading intermediate files)
            self.all_feature_dfs: Dict[str, pd.DataFrame] = {}

        except Exception as e:
            logging.exception(f"Failed during PipelineExecutor initialization: {e}")
            raise PipelineExecutionError(f"Initialization failed: {e}") from e

    def _get_domain_list(self) -> List[str]:
        """Determines the list of domain IDs to process."""
        # (Keep this method as is - No changes needed here)
        input_cfg = self.config.get('input', {})
        configured_domains = input_cfg.get('domain_ids')
        mdcath_folder = input_cfg.get('mdcath_folder')

        if isinstance(configured_domains, list) and configured_domains:
            logging.info(f"Using specified list of {len(configured_domains)} domains.")
            return configured_domains
        elif mdcath_folder and os.path.isdir(mdcath_folder):
            pattern = os.path.join(mdcath_folder, "mdcath_dataset_*.h5")
            logging.debug(f"Searching for HDF5 files with pattern: {pattern}")
            h5_files = glob.glob(pattern)
            if not h5_files:
                 raise PipelineExecutionError(f"No 'mdcath_dataset_*.h5' files found in specified directory: {mdcath_folder}")

            domains = []
            for h5_file in h5_files:
                basename = os.path.basename(h5_file)
                if basename.startswith("mdcath_dataset_") and basename.endswith(".h5"):
                    domain_id = basename[len("mdcath_dataset_"):-len(".h5")]
                    if domain_id: domains.append(domain_id)
                    else: logging.warning(f"Skipping file with potentially empty domain ID: {h5_file}")
                else: logging.warning(f"Skipping file not matching expected pattern: {h5_file}")

            if not domains: raise PipelineExecutionError(f"No valid domains extracted from filenames in: {mdcath_folder}")

            logging.info(f"Found {len(domains)} domains in {mdcath_folder}.")
            return sorted(domains)
        else:
            raise PipelineExecutionError("Config must provide 'input.domain_ids' list or valid 'input.mdcath_folder'.")


    def _process_single_domain(self, domain_id: str) -> Tuple[str, str, RmsfResultPath, PropertiesResultPath, Optional[str]]:
         """ Sequential processing for a single domain (used as fallback). """
         # Call the modified worker function which now saves to disk
         return _parallel_domain_worker((domain_id, self.config, self.output_dir))

    def _extract_all_frames(self):
        """Extracts frames sequentially after initial processing."""
        # (Keep this method as is - No changes needed here as it reads from HDF5)
        frame_cfg = self.config.get('processing', {}).get('frame_selection', {})
        if frame_cfg.get('num_frames', 0) <= 0:
            logging.info("Frame extraction skipped (num_frames <= 0).")
            return

        logging.info("Starting frame extraction...")
        # Use cleaned_pdb_paths which is populated correctly even with intermediate storage
        domains_to_process = [did for did, status in self.domain_status.items() if status == "Success" and self.cleaned_pdb_paths.get(did)]
        if not domains_to_process:
             logging.warning("No successfully processed domains with valid cleaned PDB paths available for frame extraction.")
             return

        input_cfg = self.config.get('input', {})
        mdcath_folder = input_cfg.get('mdcath_folder')
        num_success = 0
        num_failed = 0

        for domain_id in tqdm(domains_to_process, desc="Extracting Frames", disable=not self.config.get('logging',{}).get('show_progress_bars', True)):
            domain_frames_saved_flag = False
            try:
                 h5_path = os.path.join(mdcath_folder, f"mdcath_dataset_{domain_id}.h5")
                 if not os.path.exists(h5_path):
                      logging.error(f"HDF5 file not found for frame extraction: {h5_path}")
                      num_failed += 1; continue

                 reader = HDF5Reader(h5_path)
                 cleaned_pdb_path = self.cleaned_pdb_paths.get(domain_id) # Already checked this exists

                 temps = reader.get_available_temperatures()
                 for temp in temps:
                     replicas = reader.get_available_replicas(temp)
                     for rep in replicas:
                          coords_all = reader.get_coordinates(temp, rep, frame_index=-999)
                          if coords_all is None:
                               logging.warning(f"No coordinates found for {domain_id}, T={temp}, R={rep}. Skipping."); continue

                          rmsd_data = reader.get_scalar_traj(temp, rep, 'rmsd')
                          gyration_data = reader.get_scalar_traj(temp, rep, 'gyrationRadius')

                          success = extract_and_save_frames(
                               domain_id, coords_all, cleaned_pdb_path, self.output_dir,
                               self.config, rmsd_data, gyration_data, str(temp), str(rep)
                          )
                          if success: domain_frames_saved_flag = True

                 if domain_frames_saved_flag: num_success += 1
                 else: num_failed += 1 # Failed if no frames saved for the domain

            except Exception as e:
                 logging.error(f"Error during frame extraction loop for {domain_id}: {e}", exc_info=True)
                 num_failed += 1

        logging.info(f"Frame extraction complete. Domains with frames saved: {num_success}, Domains failed: {num_failed}")

    def _cleanup_intermediate_files(self):
        """Removes the intermediate data directory."""
        if os.path.isdir(self.intermediate_base_dir):
            try:
                logging.info(f"Cleaning up intermediate directory: {self.intermediate_base_dir}")
                shutil.rmtree(self.intermediate_base_dir)
            except Exception as e:
                logging.warning(f"Failed to remove intermediate directory {self.intermediate_base_dir}: {e}")
        else:
            logging.debug("No intermediate directory found to clean up.")


    def run(self):
        """Executes the full processing pipeline."""
        try:
            logging.info("Starting mdCATH processing pipeline...")
            logging.info(f"Using output directory: {self.output_dir}")
            os.makedirs(self.output_dir, exist_ok=True)
            # Clean intermediate directory from previous runs if it exists
            self._cleanup_intermediate_files()
            os.makedirs(self.intermediate_base_dir, exist_ok=True)


            # --- Step 1: Get Domain List ---
            self.domain_list = self._get_domain_list()
            if not self.domain_list: logging.warning("No domains found/specified."); return

            # --- Step 2: Initial Domain Processing (Workers save intermediates) ---
            logging.info(f"Processing {len(self.domain_list)} domains using up to {self.num_cores} cores (saving intermediates)...")
            results = []
            show_progress = self.config.get('logging',{}).get('show_progress_bars', True)
            worker_args = [(domain_id, self.config, self.output_dir) for domain_id in self.domain_list]

            if self.num_cores > 1 and len(self.domain_list) > 1:
                logging.info(f"Using parallel processing with {self.num_cores} cores.")
                try:
                     results = parallel_map(_parallel_domain_worker, worker_args,
                                            num_cores=self.num_cores, use_progress_bar=show_progress,
                                            desc="Processing Domains")
                except Exception as parallel_e:
                     logging.error(f"Parallel processing failed: {parallel_e}. Falling back to sequential.")
                     if logging.getLogger().isEnabledFor(logging.DEBUG):
                         try: logging.debug("Traceback for parallel processing failure:", exc_info=True)
                         except TypeError: logging.debug(f"Traceback:\n{traceback.format_exc()}")
                     results = [self._process_single_domain(did) for did in tqdm(self.domain_list, desc="Processing (Sequential Fallback)", disable=not show_progress)]
            else:
                logging.info("Using sequential processing.")
                results = [self._process_single_domain(did) for did in tqdm(self.domain_list, desc="Processing Domains", disable=not show_progress)]

            # Collect result *paths* and status
            self.rmsf_file_paths = {}
            self.properties_file_paths = {}
            self.cleaned_pdb_paths = {}
            self.domain_status = {}
            for result in results:
                 # Ensure result format matches the worker's return signature
                 if result and isinstance(result, tuple) and len(result) == 5:
                    domain_id, status, rmsf_path, prop_path, pdb_path = result
                    self.domain_status[domain_id] = status
                    # Store paths ONLY if the overall status is Success
                    # (Worker ensures paths are None if saving failed)
                    if status == "Success":
                        self.rmsf_file_paths[domain_id] = rmsf_path
                        self.properties_file_paths[domain_id] = prop_path
                        self.cleaned_pdb_paths[domain_id] = pdb_path
                    elif pdb_path: # Still store PDB path if cleaning was ok but subsequent steps failed
                         self.cleaned_pdb_paths[domain_id] = pdb_path

                 else: logging.error(f"Invalid result format received from worker: {result}")

            successful_domains = [d for d, s in self.domain_status.items() if s == "Success"]
            logging.info(f"Initial processing complete. Successful domains: {len(successful_domains)}/{len(self.domain_list)}")
            if not successful_domains:
                 logging.error("No domains processed successfully. Cannot proceed further.")
                 self._generate_status_visualization()
                 self._cleanup_intermediate_files() # Clean up even on failure
                 return

            # --- Step 3: Aggregate RMSF (Reads intermediate files) ---
            logging.info("Aggregating RMSF data from intermediate files...")
            try:
                # Pass the dictionary of RMSF file paths
                self.agg_replica_data, self.agg_replica_avg_data, self.agg_overall_avg_data = aggregate_and_average_rmsf(
                    self.rmsf_file_paths, # Pass the paths
                    self.config
                )
            except Exception as e:
                logging.exception(f"Error during RMSF aggregation from intermediate files: {e}")
                self.agg_replica_data, self.agg_replica_avg_data, self.agg_overall_avg_data = None, None, None

            # --- Step 4: Save RMSF Results ---
            # (No changes needed here, uses aggregated data)
            try:
                save_rmsf_results(self.output_dir, self.config, self.agg_replica_data, self.agg_replica_avg_data, self.agg_overall_avg_data)
            except Exception as e:
                logging.exception(f"Error saving RMSF results: {e}")


            # --- Step 5: Build ML Features (Reads intermediate files) ---
            logging.info("Building ML feature sets from intermediate files...")
            try:
                 # Prepare valid aggregated RMSF data (still held in memory after aggregation)
                 valid_replica_avg = {k: v for k, v in (self.agg_replica_avg_data or {}).items() if isinstance(v, pd.DataFrame) and not v.empty}
                 valid_overall_avg = self.agg_overall_avg_data if isinstance(self.agg_overall_avg_data, pd.DataFrame) and not self.agg_overall_avg_data.empty else None

                 # Get paths for properties of SUCCESSFUL domains only
                 valid_properties_paths = {d: p for d, p in self.properties_file_paths.items() if p and self.domain_status.get(d) == "Success"}

                 if (not valid_replica_avg and valid_overall_avg is None):
                     logging.warning("Skipping feature building: No valid RMSF average data was aggregated.")
                     self.all_feature_dfs = {}
                 elif not valid_properties_paths:
                      logging.warning("Skipping feature building: No valid intermediate properties files found for successful domains.")
                      self.all_feature_dfs = {}
                 else:
                     feature_builder = FeatureBuilder(
                         config=self.config.get('processing', {}).get('feature_building', {}),
                         # Pass aggregated RMSF data (should fit in memory)
                         replica_avg_rmsf=valid_replica_avg,
                         overall_avg_rmsf=valid_overall_avg,
                         # Pass paths to properties files
                         properties_file_paths=valid_properties_paths
                     )
                     self.all_feature_dfs = {}
                     ml_features_dir = os.path.join(self.output_dir, "ML_features")
                     os.makedirs(ml_features_dir, exist_ok=True)

                     temps_to_build = list(valid_replica_avg.keys())
                     # Build features for each temperature
                     for temp_str in tqdm(temps_to_build, desc="Building Temp Features", disable=not show_progress):
                          features_df = feature_builder.build_features(temperature=temp_str)
                          if features_df is not None:
                               self.all_feature_dfs[temp_str] = features_df
                               save_dataframe_csv(features_df, os.path.join(ml_features_dir, f"final_dataset_temperature_{temp_str}.csv"))
                          else:
                               logging.warning(f"Feature building returned None for temperature {temp_str}")

                     # Build features for the overall average
                     if valid_overall_avg is not None:
                         overall_features_df = feature_builder.build_features(temperature=None)
                         if overall_features_df is not None:
                              self.all_feature_dfs["average"] = overall_features_df
                              save_dataframe_csv(overall_features_df, os.path.join(ml_features_dir, "final_dataset_temperature_average.csv"))
                         else:
                              logging.warning("Feature building returned None for overall average")

            except Exception as e:
                 logging.error(f"Error during feature building from intermediate files: {e}")
                 if logging.getLogger().isEnabledFor(logging.DEBUG):
                     try: logging.debug("Traceback:", exc_info=True)
                     except TypeError: logging.debug(f"Traceback:\n{traceback.format_exc()}")
                 self.all_feature_dfs = {}


            # --- Step 6: Extract Frames ---
            # (No changes needed here)
            try:
                self._extract_all_frames()
            except Exception as e:
                logging.exception(f"Error during frame extraction: {e}")


            # --- Step 7: Voxelization ---
            # (No changes needed here, uses cleaned PDBs)
            logging.info("Running voxelization...")
            try:
                 cleaned_pdb_dir = os.path.join(self.output_dir, "pdbs")
                 voxel_config = self.config.get('processing', {}).get('voxelization', {})
                 # Check if the directory exists and contains *any* PDB files
                 pdb_files_exist = glob.glob(os.path.join(cleaned_pdb_dir, '*.pdb'))

                 if not os.path.isdir(cleaned_pdb_dir) or not pdb_files_exist:
                     logging.warning("Skipping Voxelization: Cleaned PDB dir empty/missing or contains no .pdb files.")
                 elif not voxel_config.get("enabled", True):
                     logging.info("Voxelization skipped by config.")
                 else:
                     voxel_success = run_aposteriori(voxel_config, self.output_dir, cleaned_pdb_dir)
                     if voxel_success:
                         voxel_output_name = voxel_config.get("output_name", "mdcath_voxelized")
                         potential_path = os.path.join(self.output_dir, "voxelized", f"{voxel_output_name}.hdf5")
                         if os.path.exists(potential_path): self.voxel_output_file = potential_path
                         else: logging.warning(f"Voxelization command finished, but output file not found: {potential_path}")
            except Exception as e:
                 logging.error(f"Unexpected error during voxelization: {e}")
                 if logging.getLogger().isEnabledFor(logging.DEBUG):
                     try: logging.debug("Traceback:", exc_info=True)
                     except TypeError: logging.debug(f"Traceback:\n{traceback.format_exc()}")


            # --- Step 8: Generate Visualizations ---
            # (No changes needed here, uses aggregated/final feature data)
            if self.config.get('visualization', {}).get('enabled', True):
                 try:
                     self._generate_visualizations() # Call refactored method
                 except Exception as e:
                     logging.error(f"Error during visualization generation: {e}")
                     if logging.getLogger().isEnabledFor(logging.DEBUG):
                         try: logging.debug("Traceback:", exc_info=True)
                         except TypeError: logging.debug(f"Traceback:\n{traceback.format_exc()}")
            else:
                 logging.info("Visualization generation skipped by configuration.")

            logging.info("Pipeline finished.")

        except Exception as e:
             logging.error(f"An unexpected error occurred during pipeline execution: {e}")
             if logging.getLogger().isEnabledFor(logging.DEBUG):
                 try: logging.debug("Traceback for pipeline execution failure:", exc_info=True)
                 except TypeError: logging.debug(f"Traceback:\n{traceback.format_exc()}")
             raise PipelineExecutionError(f"Pipeline failed unexpectedly: {e}") from e
        finally:
             # --- Final Step: Clean up intermediate files ---
             self._cleanup_intermediate_files()


    # --- run_plot Helper ---
    # (Keep this method as is - No changes needed)
    def run_plot(self, func, *args, **kwargs):
        """ Helper function to run a plotting function with error handling and data validation. """
        arg_name = func.__name__
        logging.debug(f"Attempting to generate plot: {arg_name}")

        # Basic data validation (Check args before kwargs setup)
        for i, data_arg in enumerate(args):
             arg_desc = f"positional argument {i+1}"
             if data_arg is None:
                  logging.warning(f"Skipping plot {arg_name}: Required {arg_desc} is None.")
                  return
             if isinstance(data_arg, pd.DataFrame) and data_arg.empty:
                  logging.warning(f"Skipping plot {arg_name}: Required DataFrame {arg_desc} is empty.")
                  return
             if isinstance(data_arg, (dict, list, tuple)) and not data_arg:
                  logging.warning(f"Skipping plot {arg_name}: Required {type(data_arg).__name__} {arg_desc} is empty.")
                  return
             # Add specific checks if needed for numpy arrays etc.

        try:
            plot_kwargs = kwargs.copy()
            func_sig = inspect.signature(func)
            func_params = func_sig.parameters

            # Inject standard args if function accepts them
            if 'output_dir' in func_params: plot_kwargs.setdefault('output_dir', self.output_dir)
            if 'viz_config' in func_params: plot_kwargs.setdefault('viz_config', self.config.get('visualization', {}))

            # Filter kwargs to only those accepted by the function, unless it accepts **kwargs
            if any(p.kind == p.VAR_KEYWORD for p in func_params.values()):
                final_kwargs = plot_kwargs # Pass all if **kwargs is present
            else:
                final_kwargs = {k: v for k, v in plot_kwargs.items() if k in func_params}

            # Additional validation for kwargs based on signature could be added here if needed

            func(*args, **final_kwargs) # Call the plot function

        except Exception as e:
            error_message = f"Failed to generate plot {func.__name__}: {e}"
            logging.error(error_message)
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                try: logging.debug(f"Traceback for plot failure ({func.__name__}):", exc_info=True)
                except TypeError: logging.debug(f"Traceback for plot failure ({func.__name__}):\n{traceback.format_exc()}")

    # --- _generate_status_visualization Helper ---
    # (Keep this method as is - No changes needed)
    def _generate_status_visualization(self):
         """Helper to generate only the status plot, e.g., on early exit."""
         if self.config.get('visualization', {}).get('enabled', True) and self.domain_status:
             logging.info("Generating processing status visualization...")
             try:
                  # Need average feature df and replica avg data for the summary plot
                  # Use empty structures if they weren't generated
                  avg_feature_df_for_status = self.all_feature_dfs.get("average", pd.DataFrame())
                  replica_avg_data_for_status = self.agg_replica_avg_data if self.agg_replica_avg_data is not None else {}

                  # Call the summary plot function, which includes the status pie chart
                  self.run_plot(viz_plots.create_summary_plot,
                                replica_avg_data_for_status,
                                avg_feature_df_for_status,
                                self.domain_status) # Pass the status dictionary
             except Exception as e:
                  logging.error(f"Failed to generate status plot (via summary): {e}")
                  if logging.getLogger().isEnabledFor(logging.DEBUG):
                      try: logging.debug("Traceback:", exc_info=True)
                      except TypeError: logging.debug(f"Traceback:\n{traceback.format_exc()}")

    # --- _generate_visualizations Method ---
    # (Keep this method as is - No changes needed)
    def _generate_visualizations(self):
          """Generates all configured plots by calling functions from viz_plots."""
          logging.info("Generating visualizations...")

          # --- Prepare Data References (use data aggregated/built in memory) ---
          replica_avg_data = self.agg_replica_avg_data if self.agg_replica_avg_data is not None else {}
          avg_feature_df = self.all_feature_dfs.get("average", pd.DataFrame()) # Use final feature DF
          domain_status_data = self.domain_status if self.domain_status is not None else {}
          voxel_output = self.voxel_output_file
          # combined_replica_data should be the dict of DFs, not paths
          combined_replica_data = self.agg_replica_data if self.agg_replica_data else {}
          overall_avg_data = self.agg_overall_avg_data if self.agg_overall_avg_data is not None else pd.DataFrame()


          # --- Define Plotting Tasks ---
          plotting_tasks = [
              (viz_plots.create_temperature_summary_heatmap, [replica_avg_data], {}),
              (viz_plots.create_temperature_average_summary, [avg_feature_df], {}),
              (viz_plots.create_rmsf_distribution_plots, [replica_avg_data, overall_avg_data], {}),
              (viz_plots.create_amino_acid_rmsf_plot, [avg_feature_df], {}),
              (viz_plots.create_amino_acid_rmsf_plot_colored, [avg_feature_df], {}),
              (viz_plots.create_replica_variance_plot, [combined_replica_data], {}),
              (viz_plots.create_dssp_rmsf_correlation_plot, [avg_feature_df], {}),
              (viz_plots.create_feature_correlation_plot, [avg_feature_df], {}),
              # (viz_plots.create_frames_visualization, ... ), # Still needs data collection if implemented
              (viz_plots.create_ml_features_plot, [avg_feature_df], {}),
              (viz_plots.create_summary_plot, [replica_avg_data, avg_feature_df, domain_status_data], {}),
              (viz_plots.create_additional_ml_features_plot, [avg_feature_df], {}),
              (getattr(viz_plots, 'create_rmsf_density_plots', None), [avg_feature_df], {}),
              (getattr(viz_plots, 'create_rmsf_by_aa_ss_density', None), [avg_feature_df], {}),
              (viz_plots.create_voxel_info_plot, [], {'config': self.config, 'voxel_output_file': voxel_output}),
          ]
          plotting_tasks = [task for task in plotting_tasks if task[0] is not None]


          # --- Execute Plotting Tasks ---
          logging.info(f"Executing {len(plotting_tasks)} visualization tasks...")
          for plot_func, data_args, plot_kwargs in tqdm(plotting_tasks, desc="Generating Visualizations", disable=not self.config.get('logging',{}).get('show_progress_bars', True)):
                self.run_plot(plot_func, *data_args, **plot_kwargs)

          logging.info("Visualization generation attempt complete.")