# /home/s_felix/ensembleflex/scripts/aggregate_data.py

import os
import glob
import re
import argparse
import logging
import pandas as pd

# --- Configuration ---
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Regex to extract temperature (numeric or 'average') from filename
# Example: temperature_320_train.csv -> 320
# Example: temperature_average_train.csv -> average
FILENAME_PATTERN = re.compile(r"temperature_(\d+|average)_.*\.csv", re.IGNORECASE)

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate temperature-specific RMSF data files into a single CSV."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing the temperature-specific CSV files (e.g., ../data/)."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save the aggregated CSV file (e.g., ../data/aggregated_rmsf_data.csv)."
    )
    parser.add_argument(
        "--file-glob",
        type=str,
        default="temperature_*.csv",
        help="Glob pattern to find input temperature files (default: 'temperature_*.csv')."
    )
    parser.add_argument(
        "--exclude-average",
        action='store_true', # Default is False (include average if found)
        help="Exclude the 'temperature_average_train.csv' file from aggregation."
    )
    parser.add_argument(
        "--target-column-prefix",
        type=str,
        default="rmsf_",
        help="Prefix for the temperature-specific target columns (e.g., 'rmsf_')."
    )
    parser.add_argument(
        "--new-target-column",
        type=str,
        default="rmsf",
        help="Name for the unified target column in the output file (default: 'rmsf')."
    )
    parser.add_argument(
        "--new-temp-column",
        type=str,
        default="temperature",
        help="Name for the new temperature feature column (default: 'temperature')."
    )
    return parser.parse_args()

def process_file(file_path: str, args: argparse.Namespace) -> pd.DataFrame | None:
    """Loads a single temperature file, processes it, and returns a DataFrame."""
    filename = os.path.basename(file_path)
    match = FILENAME_PATTERN.match(filename)

    if not match:
        logger.warning(f"Skipping file (does not match pattern): {filename}")
        return None

    temp_str = match.group(1)
    logger.info(f"Processing file: {filename} for temperature '{temp_str}'")

    if temp_str.lower() == "average":
        if args.exclude_average:
            logger.info(f"Excluding 'average' temperature file as requested: {filename}")
            return None
        # For the ensembleflex model expecting numeric temperature, 'average' doesn't fit well.
        # We will assign NaN. The data processing step later might need to handle/drop these.
        temperature_value = float('nan')
        logger.warning(f"Assigning NaN to temperature column for 'average' file: {filename}. "
                       f"Ensure downstream processing handles this.")
    else:
        try:
            temperature_value = float(temp_str)
        except ValueError:
            logger.error(f"Could not convert temperature '{temp_str}' to float in file: {filename}. Skipping.")
            return None

    # Construct the expected original target column name
    original_target_col = f"{args.target_column_prefix}{temp_str}"

    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Loaded {filename} with shape {df.shape}")

        if original_target_col not in df.columns:
            logger.error(f"Expected target column '{original_target_col}' not found in {filename}. Skipping.")
            return None

        # Add the new temperature column
        df = df.assign(**{args.new_temp_column: temperature_value})

        # Rename the target column
        df = df.rename(columns={original_target_col: args.new_target_column})

        # Drop other potential rmsf_ columns to avoid confusion? Optional.
        # cols_to_drop = [col for col in df.columns if col.startswith(args.target_column_prefix) and col != args.new_target_column]
        # if cols_to_drop:
        #     df = df.drop(columns=cols_to_drop)
        #     logger.debug(f"Dropped other RMSF columns: {cols_to_drop}")

        logger.info(f"Successfully processed: {filename}")
        return df

    except FileNotFoundError:
        logger.error(f"File not found during processing: {file_path}. Skipping.")
        return None
    except pd.errors.EmptyDataError:
        logger.warning(f"Skipping empty file: {filename}")
        return None
    except Exception as e:
        logger.error(f"Error processing file {filename}: {e}", exc_info=True)
        return None


def main():
    """Main execution function."""
    args = parse_arguments()
    logger.info("Starting data aggregation process...")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output file: {args.output_file}")
    logger.info(f"File glob pattern: {args.file_glob}")
    logger.info(f"Exclude 'average' file: {args.exclude_average}")

    input_path_pattern = os.path.join(args.input_dir, args.file_glob)
    source_files = glob.glob(input_path_pattern)

    if not source_files:
        logger.error(f"No files found matching pattern '{input_path_pattern}'. Exiting.")
        return

    logger.info(f"Found {len(source_files)} potential source files.")

    all_dfs = []
    processed_count = 0
    for file_path in sorted(source_files): # Sort for consistent order
        processed_df = process_file(file_path, args)
        if processed_df is not None:
            all_dfs.append(processed_df)
            processed_count += 1

    if not all_dfs:
        logger.error("No dataframes were successfully processed. Aggregated file will not be created.")
        return

    logger.info(f"Successfully processed {processed_count} files. Concatenating dataframes...")

    try:
        aggregated_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Aggregation complete. Final dataframe shape: {aggregated_df.shape}")

        # Basic validation of the output
        if args.new_temp_column not in aggregated_df.columns:
             logger.error(f"Critical Error: The new temperature column '{args.new_temp_column}' is missing in the final dataframe!")
             return
        if args.new_target_column not in aggregated_df.columns:
             logger.error(f"Critical Error: The new target column '{args.new_target_column}' is missing in the final dataframe!")
             return
        if not args.exclude_average and aggregated_df[args.new_temp_column].isnull().any():
             logger.warning(f"NaN values found in the '{args.new_temp_column}' column, likely from 'average' files.")

        # Ensure output directory exists
        output_dir = os.path.dirname(args.output_file)
        if output_dir: # Handle case where output file is in the current directory
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Ensured output directory exists: {output_dir}")


        # Save the aggregated dataframe
        aggregated_df.to_csv(args.output_file, index=False)
        logger.info(f"Aggregated data successfully saved to: {args.output_file}")

    except Exception as e:
        logger.error(f"Error during dataframe concatenation or saving: {e}", exc_info=True)

if __name__ == "__main__":
    main()