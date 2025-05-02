#!/usr/bin/env python3
import os
import sys
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Adjust this path if your processed data is elsewhere
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'processed')

INPUT_FASTA_FILES = [
    "train_sequences.fasta",
    "val_sequences.fasta",
    "test_sequences.fasta",
]

OUTPUT_FASTA_FILE = "predict_sequences.fasta"
# --- End Configuration ---

def concatenate_files():
    """Concatenates specified input FASTA files into one output FASTA file."""

    output_path = os.path.join(PROCESSED_DATA_DIR, OUTPUT_FASTA_FILE)
    input_paths = [os.path.join(PROCESSED_DATA_DIR, fname) for fname in INPUT_FASTA_FILES]

    logger.info(f"Attempting to concatenate FASTA files into: {output_path}")
    total_sequences = 0
    missing_files = []

    # Check if all input files exist first
    for path in input_paths:
        if not os.path.exists(path):
            logger.error(f"Input FASTA file not found: {path}")
            missing_files.append(path)

    if missing_files:
        logger.error("Aborting concatenation due to missing input files.")
        return False

    try:
        with open(output_path, 'w') as outfile:
            for input_path in input_paths:
                logger.info(f"Processing: {os.path.basename(input_path)}")
                sequence_count_in_file = 0
                last_line_was_newline = True # Assume start of file is like after a newline
                with open(input_path, 'r') as infile:
                    for line in infile:
                        # Count sequences for logging
                        if line.startswith('>'):
                            sequence_count_in_file += 1
                            total_sequences += 1
                        # Write line to output
                        outfile.write(line)
                        # Track if last line ended with newline (important for FASTA format)
                        last_line_was_newline = line.endswith('\n')

                logger.info(f" -> Appended {sequence_count_in_file} sequences from {os.path.basename(input_path)}.")

                # Ensure there's a newline between concatenated files if the previous didn't end with one
                if not last_line_was_newline:
                    logger.warning(f"File {os.path.basename(input_path)} did not end with a newline. Adding one.")
                    outfile.write('\n')


        logger.info("-" * 30)
        logger.info(f"Successfully concatenated {len(input_paths)} files into {output_path}")
        logger.info(f"Total sequences written: {total_sequences}")
        logger.info("-" * 30)
        return True

    except IOError as e:
        logger.error(f"An error occurred during file operations: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    if concatenate_files():
        sys.exit(0) # Exit with success code
    else:
        sys.exit(1) # Exit with error code