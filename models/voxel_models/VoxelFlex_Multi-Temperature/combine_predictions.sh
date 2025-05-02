#!/bin/bash

# ============================================================
# Combine VoxelFlex Prediction Parts Script (Updated)
# ============================================================
# Description:
#   Finds prediction CSV files matching a specific pattern
#   (e.g., predictions_train-set_320K_partXX.csv) recursively
#   within the 'outputs' directory and concatenates them
#   into a single output CSV file, preserving the header
#   only from the first file found.
#
# Instructions:
#   1. Adjust the variables in the Configuration section below.
#   2. Run this script from the root directory of the
#      VoxelFlex project (e.g., ~/VoxelFlex_T).
#      Example: ./combine_predictions.sh
# ============================================================

# --- Configuration ---
SEARCH_DIR="outputs" # Directory to search within (relative to script location)
# Adjust temperature (320K) and base name if needed. This is the base filename
# *without* the full path prefix from the outputs directory.
FILE_PATTERN="predictions_train-set_320K_part[0-9][0-9].csv"
# Desired name for the final combined file (will be placed in SEARCH_DIR)
OUTPUT_FILE="outputs/combined_train_predictions_320K.csv"
# --------------------

echo "--- Starting Prediction Combination ---"
echo "Searching in: $SEARCH_DIR"
echo "Filename pattern: $FILE_PATTERN"
echo "Output file: $OUTPUT_FILE"

# --- Check Search Directory ---
if [ ! -d "$SEARCH_DIR" ]; then
  echo "ERROR: Search directory '$SEARCH_DIR' not found." >&2
  exit 1
fi

# --- Find and Sort Files ---
# Use find to locate files recursively based on the name pattern
# -type f ensures we only find files
# Print null-separated for safety with spaces/special chars
# Sort -zV sorts numerically (version sort) handling leading zeros
# mapfile reads null-separated input into the array
declare -a found_files
mapfile -d $'\0' found_files < <(find "$SEARCH_DIR" -name "$FILE_PATTERN" -type f -print0 | sort -zV)

# Check if any files were found
if [ ${#found_files[@]} -eq 0 ]; then
  echo "ERROR: No prediction files found matching pattern '$FILE_PATTERN' in '$SEARCH_DIR'." >&2
  exit 1
fi

echo "Found ${#found_files[@]} prediction file parts to combine:"
# Use printf for safe printing of paths, even with spaces
printf "  %s\n" "${found_files[@]}"

# --- Combine using awk (handles header automatically) ---
echo "Combining files into $OUTPUT_FILE ..."

# Ensure output directory exists (though SEARCH_DIR check helps)
mkdir -p "$(dirname "$OUTPUT_FILE")"

# awk 'condition' file1 file2 ... > output
# Condition: FNR > 1 || NR == 1
#   Prints header only from the first file (NR==1)
#   Prints data lines (skipping header) from subsequent files (FNR > 1)
awk 'FNR > 1 || NR == 1' "${found_files[@]}" > "$OUTPUT_FILE"

# Check if awk command succeeded and output file was created
if [ $? -eq 0 ] && [ -f "$OUTPUT_FILE" ]; then
  echo "Combination successful."
  echo "Combined file saved to: $OUTPUT_FILE"
  # Optional: Show line count
  line_count=$(wc -l < "$OUTPUT_FILE")
  echo "Total lines in combined file (including header): $line_count"
else
  echo "ERROR: File combination failed using awk." >&2
  # Clean up potentially partially created file
  rm -f "$OUTPUT_FILE"
  exit 1
fi

echo "--- Prediction Combination Finished ---"
exit 0