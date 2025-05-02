#!/usr/bin/env python
import os
import sys
import subprocess
import time
import glob
import logging
import argparse
import psutil # Requires: pip install psutil
import pandas as pd
from pathlib import Path
import shlex # For safe command splitting if needed

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run VoxelFlex predict sequentially on split domain files with RAM monitoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the VoxelFlex configuration YAML file."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the trained VoxelFlex model checkpoint (.pt)."
    )
    parser.add_argument(
        "--split-dir", type=str, required=True,
        help="Directory containing the split domain files (e.g., 'input_data/train_splits_parts')."
    )
    parser.add_argument(
        "--split-prefix", type=str, default="train_domains_part_",
        help="Prefix for the split domain files."
    )
    parser.add_argument(
        "--temperature", type=float, required=True,
        help="Target prediction temperature (K)."
    )
    parser.add_argument(
        "--output-base", type=str, required=True,
        help="Base name for output CSV files (part number and .csv will be added)."
    )
    parser.add_argument(
        "--ram-threshold", type=float, default=70.0,
        help="System RAM usage percentage threshold to trigger cutoff (e.g., 70 for 70%%)."
    )
    parser.add_argument(
        "--check-interval", type=float, default=5.0,
        help="Interval (seconds) for checking RAM usage."
    )
    parser.add_argument(
        "--voxelflex-cmd", type=str, default="voxelflex",
        help="Command to invoke the VoxelFlex CLI (use full path if not in PATH)."
    )
    parser.add_argument(
        "--verbosity", type=str, default="-v", choices=["", "-v", "-vv"],
        help="Verbosity level for the voxelflex command."
    )
    parser.add_argument(
        "--combine-output", type=str, default="combined_predictions.csv",
        help="Filename for the combined output CSV (set to '' or None to disable combining)."
    )
    parser.add_argument(
        "--stop-on-error", action="store_true",
        help="Stop the entire script if any prediction part fails (excluding RAM cutoff)."
    )

    return parser.parse_args()

# --- Main Execution ---
def main():
    args = parse_arguments()

    # --- Validate Paths ---
    config_path = Path(args.config).resolve()
    model_path = Path(args.model).resolve()
    split_dir_path = Path(args.split_dir).resolve()
    voxelflex_cmd = args.voxelflex_cmd

    if not config_path.is_file():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    if not model_path.is_file():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)
    if not split_dir_path.is_dir():
        logger.error(f"Split directory not found: {split_dir_path}")
        sys.exit(1)

    # --- Find Split Files ---
    split_pattern = f"{args.split_prefix}*.txt"
    # Use glob from pathlib for cleaner path handling
    split_files = sorted(list(split_dir_path.glob(split_pattern)))

    if not split_files:
        logger.error(f"No split files found in {split_dir_path} matching pattern '{split_pattern}'")
        sys.exit(1)

    logger.info(f"Found {len(split_files)} split files to process.")
    logger.info(f"RAM Usage Threshold: {args.ram_threshold}%")
    logger.info(f"Check Interval: {args.check_interval}s")

    generated_files = []
    interrupted_by_ram = False
    errors_occurred = False

    # --- Loop Through Parts ---
    for i, part_file in enumerate(split_files):
        part_num_str = part_file.stem.replace(args.split_prefix, '')
        logger.info(f"\n--- Processing Part {part_num_str}/{len(split_files)}: {part_file.name} ---")

        output_csv_name = f"{args.output_base}{part_num_str}.csv"
        # Note: predict.py puts this in outputs/<run_name>/metrics/. This name is just for voxelflex.
        # We will find the actual generated file later for combining.

        # --- Read Domains for this part ---
        try:
            with open(part_file, 'r') as f:
                # Read lines, strip whitespace, filter empty lines
                domains_for_part = [line.strip() for line in f if line.strip()]
            if not domains_for_part:
                 logger.warning(f"Split file {part_file.name} is empty. Skipping.")
                 continue
            logger.info(f"Read {len(domains_for_part)} domains from {part_file.name}")
        except Exception as e:
            logger.error(f"Failed to read domains from {part_file.name}: {e}")
            errors_occurred = True
            if args.stop_on_error:
                logger.error("Exiting due to file read error (--stop-on-error enabled).")
                sys.exit(1)
            else:
                logger.warning("Skipping this part due to file read error.")
                continue

        # --- Construct Command List ---
        command = [
            voxelflex_cmd,
            "predict",
            "--config", str(config_path),
            "--model", str(model_path),
            "--temperature", str(args.temperature),
            "--output_csv", output_csv_name,
        ]
        if args.verbosity:
            command.append(args.verbosity)
        # Add domains explicitly
        command.append("--domains")
        command.extend(domains_for_part)

        logger.info(f"Executing command for part {part_num_str}...")
        # logger.debug(f"Command: {' '.join(shlex.quote(str(c)) for c in command)}") # Debug: print quoted command

        process = None
        try:
            # Start the subprocess
            # Capture output to check for errors later if needed (optional)
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            pid = process.pid
            logger.info(f"Started subprocess for part {part_num_str} with PID: {pid}")

            # --- Monitoring Loop ---
            while process.poll() is None: # While the process is running
                try:
                    current_ram_percent = psutil.virtual_memory().percent
                    logger.debug(f"Part {part_num_str} running (PID: {pid}). System RAM: {current_ram_percent:.1f}%")

                    if current_ram_percent > args.ram_threshold:
                        logger.warning(f"RAM usage ({current_ram_percent:.1f}%) exceeded threshold ({args.ram_threshold}%). Terminating process PID {pid}...")
                        process.terminate() # Send SIGTERM first
                        try:
                             # Wait a moment for graceful termination
                            process.wait(timeout=5)
                            logger.warning(f"Process PID {pid} terminated.")
                        except subprocess.TimeoutExpired:
                            logger.warning(f"Process PID {pid} did not terminate gracefully. Sending SIGKILL.")
                            process.kill() # Force kill
                            process.wait() # Wait for kill confirmation
                            logger.warning(f"Process PID {pid} killed.")
                        interrupted_by_ram = True
                        break # Exit monitoring loop

                except psutil.NoSuchProcess:
                    logger.warning(f"Process PID {pid} disappeared unexpectedly.")
                    break # Exit monitoring loop if process is gone
                except Exception as mon_e:
                    logger.error(f"Error during RAM monitoring: {mon_e}")
                    # Continue monitoring if possible, or decide to terminate

                time.sleep(args.check_interval) # Wait before next check

            # --- Check result after loop ---
            if interrupted_by_ram:
                logger.error(f"Prediction for part {part_num_str} terminated due to high RAM usage.")
                errors_occurred = True
                # Exit the main loop over parts if RAM cutoff happened
                break

            # If process finished normally
            return_code = process.wait() # Get final return code
            if return_code == 0:
                logger.info(f"Prediction for part {part_num_str} completed successfully.")
                # *** Find the actual output file path ***
                # Since voxelflex creates a new run dir, we need to find it.
                # This is tricky. A simpler way is just to track the base name.
                generated_files.append(output_csv_name) # Store the *intended* base name
            else:
                logger.error(f"Prediction for part {part_num_str} failed with exit code {return_code}.")
                # Optionally print stderr
                stderr_output = process.stderr.read()
                if stderr_output:
                     logger.error(f"Stderr:\n{stderr_output}")
                errors_occurred = True
                if args.stop_on_error:
                    logger.error("Exiting due to prediction error (--stop-on-error enabled).")
                    sys.exit(1)

        except FileNotFoundError:
            logger.error(f"Command '{voxelflex_cmd}' not found. Is VoxelFlex installed and in PATH?")
            sys.exit(1)
        except Exception as e:
            logger.error(f"An error occurred launching or monitoring subprocess for part {part_num_str}: {e}")
            if process and process.poll() is None:
                logger.info(f"Attempting to kill lingering process {process.pid}")
                process.kill()
            errors_occurred = True
            if args.stop_on_error:
                 sys.exit(1)

    # --- End Loop Over Parts ---

    if interrupted_by_ram:
        logger.error("Script terminated due to excessive RAM usage during one of the parts.")
        sys.exit(1)
    if errors_occurred and not args.stop_on_error:
         logger.warning("Some prediction parts failed. Combining results might be incomplete.")


    # --- Combine Results ---
    if args.combine_output and generated_files:
        logger.info("\n--- Combining Prediction Results ---")
        all_dfs = []
        output_base_dir = Path(args.config).parent.parent.parent / "outputs" # Assume outputs is peer to src

        # Search for the actual generated files - This is complex due to timestamps in run_name
        # We'll search recent run directories for files matching the pattern.
        logger.info(f"Searching for generated CSVs with base names like '{args.output_base}*.csv'...")

        # Find run directories created roughly after the script started (heuristic)
        # This is not perfectly reliable if runs overlap heavily
        script_start_time_sec = time.time() - (len(split_files) * 10 * 60) # Estimate start window
        possible_run_dirs = sorted([d for d in output_base_dir.glob('voxelflex_run_*') if d.is_dir() and d.stat().st_mtime > script_start_time_sec], reverse=True)

        found_files_to_combine = []
        processed_base_names = set()

        for base_name in generated_files:
            if base_name in processed_base_names: continue # Avoid duplicates if script re-run

            found_in_run = False
            for run_dir in possible_run_dirs:
                metrics_dir = run_dir / "metrics"
                potential_file = metrics_dir / base_name
                if potential_file.is_file():
                    logger.info(f"Found: {potential_file}")
                    found_files_to_combine.append(potential_file)
                    processed_base_names.add(base_name)
                    found_in_run = True
                    break # Found the file for this base name

            if not found_in_run:
                 logger.warning(f"Could not find output file corresponding to base name: {base_name}")


        if found_files_to_combine:
            logger.info(f"Attempting to combine {len(found_files_to_combine)} CSV files...")
            try:
                for f_path in found_files_to_combine:
                    try:
                        all_dfs.append(pd.read_csv(f_path))
                    except Exception as read_e:
                        logger.error(f"Failed to read {f_path}: {read_e}")

                if all_dfs:
                    combined_df = pd.concat(all_dfs, ignore_index=True)
                    combined_output_path = output_base_dir / args.combine_output # Save in outputs dir
                    combined_df.to_csv(combined_output_path, index=False, float_format='%.6f')
                    logger.info(f"Combined results saved to: {combined_output_path} ({len(combined_df)} rows)")
                else:
                     logger.error("No valid CSV dataframes could be read for combining.")

            except Exception as comb_e:
                logger.error(f"Failed to combine prediction CSVs: {comb_e}")
        else:
            logger.warning("No prediction files found to combine.")

    elif args.combine_output:
         logger.info("Skipping result combination as no files were generated or --combine-output is empty.")


    logger.info("\n--- Batch Prediction Script Finished ---")
    if errors_occurred:
        logger.warning("Completed with errors during prediction for some parts.")
        sys.exit(1) # Exit with error code if any part failed

# --- Script Entry Point ---
if __name__ == "__main__":
    main()