#!/bin/bash

# --- Configuration ---
INPUT_FILE="/home/s_felix/VoxelFlex_T/input_data/train_domains.txt"
NUM_FILES=10
OUTPUT_DIR="/home/s_felix/VoxelFlex_T/input_data/train_splits_parts" # Directory to store the split files
PREFIX="train_domains_part_" # Prefix for the output files
SUFFIX=".txt" # Suffix for the output files

# --- Checks ---
if [ ! -f "$INPUT_FILE" ]; then
  echo "ERROR: Input file not found: $INPUT_FILE" >&2
  exit 1
fi

# --- Calculate lines per file ---
total_lines=$(wc -l < "$INPUT_FILE")
if [ "$total_lines" -lt "$NUM_FILES" ]; then
  echo "WARNING: Fewer lines ($total_lines) than requested files ($NUM_FILES). Creating fewer files." >&2
  NUM_FILES=$total_lines # Adjust number of files if fewer lines exist
fi

# Calculate lines per file, rounding up to ensure all lines are included
lines_per_file=$(( (total_lines + NUM_FILES - 1) / NUM_FILES ))

echo "Input file: $INPUT_FILE ($total_lines lines)"
echo "Splitting into $NUM_FILES files (approx $lines_per_file lines each)."
echo "Output directory: $OUTPUT_DIR"

# --- Create output directory ---
mkdir -p "$OUTPUT_DIR"
if [ $? -ne 0 ]; then
  echo "ERROR: Could not create output directory: $OUTPUT_DIR" >&2
  exit 1
fi

# --- Perform the split ---
# The 'split' command is ideal for this
# -l specifies lines per file
# -a 2 specifies using a 2-digit suffix (00, 01, ..., 09)
# --numeric-suffixes=1 starts numbering from 1 instead of 0
# The final argument is the prefix for the output files
split -l "$lines_per_file" -a 2 --numeric-suffixes=1 "$INPUT_FILE" "$OUTPUT_DIR/$PREFIX"

# --- Rename files to add suffix ---
echo "Renaming split files..."
cd "$OUTPUT_DIR" || exit 1 # Change into the output directory
for f in ${PREFIX}*; do
  # Check if the file doesn't already have the suffix
  if [[ "$f" != *"$SUFFIX" ]]; then
    mv -- "$f" "${f}${SUFFIX}"
  fi
done
cd - > /dev/null # Go back to the previous directory quietly

echo "Splitting complete. Files are in $OUTPUT_DIR"
exit 0