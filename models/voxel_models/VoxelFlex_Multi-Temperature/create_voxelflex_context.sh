#!/bin/bash

# =========================================================================
# Create VoxelFlex Project Context File Script
# =========================================================================
# Description:
#   Analyzes the VoxelFlex project directory to generate a context file.
#   This file details the project's goals, data handling strategy
#   (Metadata Preprocessing / On-Demand HDF5), dataset scale, compute
#   environment, folder structure, and key source code files. It helps
#   provide a snapshot of the project for analysis or sharing.
#   Includes snippets (head -n 10) for domain split list files.
#
# Instructions:
#   Run this script from the root directory of the VoxelFlex project.
#   Example: ./create_voxelflex_context.sh
#
# Output:
#   Generates/overwrites the file "voxelflex_context.txt" in the current directory.
# =========================================================================

# --- Configuration ---
OUTPUT_FILE="voxelflex_context.txt"
CONTEXT_DATE=$(date '+%Y-%m-%d %H:%M:%S %Z')

# --- Basic Checks ---
if [ ! -d "src/voxelflex" ]; then
  echo "ERROR: 'src/voxelflex' directory not found." >&2
  echo "Please run this script from the root directory of the VoxelFlex project." >&2
  exit 1
fi
if [ ! -f "pyproject.toml" ]; then
  echo "ERROR: 'pyproject.toml' not found." >&2
  echo "Please run this script from the root directory of the VoxelFlex project." >&2
  exit 1
fi

echo "Generating project context file: $OUTPUT_FILE ..."

# Clear existing file
rm -f "$OUTPUT_FILE"

# --- Header ---
echo "==========================================================" >> "$OUTPUT_FILE"
echo "        VoxelFlex (Temperature-Aware) Project Context     " >> "$OUTPUT_FILE"
echo "     (Workflow: Metadata Preprocessing / On-Demand HDF5)    " >> "$OUTPUT_FILE"
echo "          Generated on: $CONTEXT_DATE                    " >> "$OUTPUT_FILE"
echo "==========================================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# --- Project Goal & Workflow Overview ---
# Note: Using quoted 'EOF' prevents shell variable expansion inside the block.
echo "Project Goal & Workflow Overview:" >> "$OUTPUT_FILE"
echo "---------------------------------" >> "$OUTPUT_FILE"
cat << 'EOF' >> "$OUTPUT_FILE"
**Scientific Goal**: Develop a robust and efficient Python package to predict per-residue protein flexibility (Root Mean Square Fluctuation, RMSF) from 3D structural voxel representations while incorporating temperature awareness. The model should be capable of predicting RMSF values at arbitrary input temperatures, enabling better understanding of protein dynamics across thermal conditions.

**Biological Significance**: Protein flexibility is critical for understanding function, binding interactions, and thermal stability. Temperature-aware prediction enables insights into how proteins behave across varying thermal conditions, which is crucial for biotechnology applications and understanding temperature adaptation in organisms.

**Core Workflow Strategy (Metadata Preprocessing / On-Demand HDF5):**
To handle potentially very large voxel datasets efficiently without massive intermediate storage, this project uses a two-stage approach:
1.  **Preprocessing (`preprocess` command):** Reads raw RMSF data (CSV) and domain splits (TXT). Maps this information to the structure within the primary Voxel Data HDF5 file. Generates a *metadata-only* manifest file (`master_samples.parquet`) containing pointers (HDF5 domain/residue IDs), target RMSF values, raw temperatures, and split assignments for every valid residue@temperature sample. Also calculates and saves temperature scaling parameters (`temp_scaling_params.json`) based on the training data. **No voxel data is processed or saved at this stage.**
2.  **Training (`train` command):** Uses a custom PyTorch `IterableDataset` (`ChunkedVoxelDataset`). DataLoader workers read the metadata manifest (`master_samples.parquet`). They process assigned domains in chunks (`chunk_size`), reading raw voxel data **directly from the HDF5 file on-demand** into worker RAM. Voxels are processed (bool->float32, transpose), combined with scaled temperature and target RMSF (from metadata), and yielded as tensors. Raw voxel data for completed chunks is discarded by the worker to manage RAM. This avoids storing terabytes of preprocessed voxel tensors.
3.  **Inference (`predict`, `evaluate`):** Also uses on-demand loading from HDF5 for the required residues.

**Current Status:** The pipeline (preprocessing, on-demand data loading, training loop, validation) has been shown to function correctly, successfully training on the full dataset despite some underlying data inconsistencies (missing domains in HDF5, minor HDF5 corruption) which are handled gracefully by skipping affected data points. Performance tuning (e.g., `num_workers`) has identified I/O contention as a key bottleneck, mitigated by reducing concurrent workers.
EOF
echo "" >> "$OUTPUT_FILE"


# --- Dataset Scale ---
echo "Dataset Scale (Approximate):" >> "$OUTPUT_FILE"
echo "----------------------------" >> "$OUTPUT_FILE"
cat << 'EOF' >> "$OUTPUT_FILE"
*   **Domains:** ~5,400 unique domain IDs in the source HDF5 file (~5,378 found relevant across provided splits in testing).
*   **RMSF Data:** ~3.7 million rows in the aggregated CSV file (representing multiple temperatures per residue).
*   **Samples Generated:** The preprocessing step generates ~3.7 million individual samples (residue@temperature points) stored in the `master_samples.parquet` file.
*   **Voxel Data Size:** Individual processed voxel arrays (float32, 5x21x21x21) are ~180KB each. The raw HDF5 file (`mdcath_voxelized.hdf5`) is likely hundreds of GBs.
*   **Intermediate Data:** The `master_samples.parquet` file (metadata only) is relatively small (MBs to low GBs). **This workflow successfully avoids large intermediate voxel storage.**
EOF
echo "" >> "$OUTPUT_FILE"


# --- Compute Environment Specifications ---
echo "Compute Environment Specifications:" >> "$OUTPUT_FILE"
echo "---------------------------------" >> "$OUTPUT_FILE"
# Capturing details as provided by user previously
cat << 'EOF' >> "$OUTPUT_FILE"
*   **CPU:** 36 Cores (Based on user info and `htop` output)
*   **RAM:** ~62.6 GB Total System Memory
*   **GPU:** 1x NVIDIA Quadro RTX 8000 (49152 MiB / ~47.5 GB VRAM)
*   **CUDA Version:** 12.2 (from `nvidia-smi`)
*   **NVIDIA Driver Version:** 535.183.01 (from `nvidia-smi`)
*   **Operating System:** Linux (Likely Ubuntu or similar)
*   **Filesystem Mount:** `/home/s_felix` located on `/dev/sdc2` (Previously experienced read-only remount issues under heavy I/O load, suggesting potential storage sensitivity or underlying issues). Performance improved by reducing concurrent DataLoader workers.
EOF
echo "" >> "$OUTPUT_FILE"


# --- Input Data Formats (Expected) ---
echo "Input Data Formats (Expected):" >> "$OUTPUT_FILE"
echo "------------------------------" >> "$OUTPUT_FILE"
cat << 'EOF' >> "$OUTPUT_FILE"
1.  **Voxel Data (HDF5):**
    *   Path: `input.voxel_file` in config.
    *   Format: HDF5 (`.hdf5`).
    *   Structure: `DomainID` (e.g., '1abcA00') -> `ChainID` (e.g., 'A') -> `ResidueID` (string digit, e.g., '123') -> HDF5 Dataset.
    *   Voxel Dataset: Expected `bool`, shape `(21, 21, 21, 5)`. Processed on-demand to `float32`, shape `(5, 21, 21, 21)`.

2.  **Aggregated RMSF Data (CSV):**
    *   Path: `input.aggregated_rmsf_file` in config.
    *   Format: CSV (`.csv`).
    *   Required Columns: `domain_id`, `resid` (int), `resname` (str), `temperature_feature` (float, Kelvin), `target_rmsf` (float).
    *   Optional Columns: `relative_accessibility`, `dssp`, `secondary_structure_encoded` (used for evaluation).

3.  **Domain Split Files (.txt):**
    *   Paths: `input.train_split_file`, `input.val_split_file`, `input.test_split_file` in config.
    *   Format: Plain text (`.txt`), one HDF5 `DomainID` per line.
EOF
echo "" >> "$OUTPUT_FILE"


# --- Output Data Formats (Expected) ---
echo "Output Data Formats (Expected):" >> "$OUTPUT_FILE"
echo "-------------------------------" >> "$OUTPUT_FILE"
cat << 'EOF' >> "$OUTPUT_FILE"
Outputs saved within `outputs/<run_name>/`.

1.  **From `preprocess`:**
    *   `input_data/processed/master_samples.parquet`: Single file with sample metadata (domain_id, resid_str, resid_int, raw_temp, target_rmsf, split). **NO VOXELS.**
    *   `outputs/<run_name>/models/temp_scaling_params.json`: Min/Max scaler values derived from training temperatures.
    *   `outputs/<run_name>/failed_preprocess_domains.txt`: Domains skipped during preprocessing (mapping/HDF5 issues).

2.  **From `train`:**
    *   `outputs/<run_name>/models/best_model.pt`: Checkpoint of the model with the best validation metric.
    *   `outputs/<run_name>/models/latest_model.pt`: Checkpoint of the model from the latest completed epoch.
    *   `outputs/<run_name>/models/checkpoint_epoch_*.pt` (Optional): Periodic checkpoints if `checkpoint_interval` is set.
    *   `outputs/<run_name>/training_history.json`: JSON file with epoch-wise metrics (loss, pearson, LR, time).
    *   `outputs/<run_name>/logs/voxelflex.log`: Detailed log file for the run.

3.  **From `predict`:**
    *   `outputs/<run_name>/metrics/predictions_*.csv`: CSV file with predicted RMSF values for specified domains at a target temperature.

4.  **From `evaluate`:**
    *   `outputs/<run_name>/metrics/evaluation_metrics_*.json`: JSON file with performance metrics (overall, stratified, permutation importance).

5.  **From `visualize`:**
    *   `outputs/<run_name>/visualizations/*.png` (or other format): Generated performance plots.
    *   `outputs/<run_name>/visualizations/*_data.csv` (Optional): Data used for generating plots.
EOF
echo "" >> "$OUTPUT_FILE"


# --- Folder Structure ---
echo "Project Folder Structure:" >> "$OUTPUT_FILE"
echo "-------------------------" >> "$OUTPUT_FILE"
echo "(Showing relative paths from project root, excluding outputs/build files)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
if command -v tree &> /dev/null; then
  # Exclude common transient/output/env directories and the script's output file
  tree -L 3 -I '.git*|__pycache__|*.egg-info|.venv|env|venv|outputs|snap|dist|build|'$OUTPUT_FILE >> "$OUTPUT_FILE"
else
  echo "(tree command not found, using find...)" >> "$OUTPUT_FILE"
  # Use find with more exclusions
  find . -not \( \
      -path './.git*' -o \
      -path '*/__pycache__*' -o \
      -path './*.egg-info*' -o \
      -path './.venv*' -o \
      -path './venv*' -o \
      -path './env*' -o \
      -path './outputs*' -o \
      -path './snap' -o \
      -path './dist' -o \
      -path './build' -o \
      -path "./$OUTPUT_FILE" \
      \) -maxdepth 4 -print | sed -e 's;[^/]*/;|____;g;s;____|; |;s;[^/]*$;-- &;' >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"


# --- File Contents ---
echo "File Contents:" >> "$OUTPUT_FILE"
echo "----------------" >> "$OUTPUT_FILE"
echo "(Core source files and snippets for input data files like *.txt)" >> "$OUTPUT_FILE"

# List core files - adjust if structure changes significantly
FILES_TO_CAT=(
  "pyproject.toml"
  "README.md"
  "requirements.txt"
  "src/voxelflex/config/default_config.yaml"
  "src/voxelflex/config/config.py"
  "src/voxelflex/data/validators.py"
  "src/voxelflex/data/data_loader.py"
  "src/voxelflex/models/cnn_models.py"
  "src/voxelflex/utils/file_utils.py"
  "src/voxelflex/utils/logging_utils.py"
  "src/voxelflex/utils/system_utils.py"
  "src/voxelflex/utils/temp_scaling.py"
  "src/voxelflex/cli/cli.py"
  "src/voxelflex/cli/commands/preprocess.py"
  "src/voxelflex/cli/commands/train.py"
  "src/voxelflex/cli/commands/predict.py"
  "src/voxelflex/cli/commands/evaluate.py"
  "src/voxelflex/cli/commands/visualize.py"
  "input_data/train_domains.txt"
  "input_data/val_domains.txt"
  "input_data/test_domains.txt"
)

for file in "${FILES_TO_CAT[@]}"; do
  echo "" >> "$OUTPUT_FILE"
  echo "==========================================================" >> "$OUTPUT_FILE"
  if [ -f "$file" ]; then
      # Show only head for potentially large input lists
      if [[ "$file" == input_data/*.txt ]]; then
          echo "===== FILE: $file (Top 10 Lines) =====" >> "$OUTPUT_FILE"
          echo "==========================================================" >> "$OUTPUT_FILE"
          echo "" >> "$OUTPUT_FILE"
          # Use head safely, suppressing error if file is empty/short
          head -n 10 "$file" 2>/dev/null >> "$OUTPUT_FILE"
          echo "" >> "$OUTPUT_FILE"
          echo "===== (End Snippet of $file) =====" >> "$OUTPUT_FILE"
      else
          echo "===== FILE: $file =====" >> "$OUTPUT_FILE"
          echo "==========================================================" >> "$OUTPUT_FILE"
          echo "" >> "$OUTPUT_FILE"
          # Use cat safely, suppressing error if file is unreadable
          cat "$file" 2>/dev/null >> "$OUTPUT_FILE"
      fi
  else
    # Clearly mark if a listed file is not found
    echo "===== FILE: $file (Not Found) =====" >> "$OUTPUT_FILE"
    echo "==========================================================" >> "$OUTPUT_FILE"
  fi
done

echo "" >> "$OUTPUT_FILE"
echo "==========================================================" >> "$OUTPUT_FILE"
echo "              End of VoxelFlex Project Context            " >> "$OUTPUT_FILE"
echo "==========================================================" >> "$OUTPUT_FILE"

echo "Finished generating $OUTPUT_FILE"
exit 0