#!/bin/bash

# Define the output file
OUTPUT_FILE="$PWD/EnsembleFlex_context.txt" # Renamed output file

# Ensure script runs from the project root directory
if [ ! -f "pyproject.toml" ] || [ ! -d "ensembleflex" ]; then
    echo "Error: This script must be run from the ensembleflex project root directory."
    exit 1
fi

{
    # Header
    echo "=========================================================="
    echo "          EnsembleFlex: Temperature-Aware Protein Flexibility ML Pipeline" # Renamed
    echo "=========================================================="
    echo "Context generated on: $(date)"
    echo ""

    # Input Data Format Section (UPDATED FOR AGGREGATED DATA)
    echo "=========================================================="
    echo "Input Data Format (Aggregated)"
    echo "=========================================================="
    echo "EnsembleFlex expects a SINGLE aggregated CSV file (e.g., 'aggregated_rmsf_data.csv') specified in the config."
    echo "This file contains data from multiple temperatures, with temperature as an input feature."
    echo ""
    echo "Expected Columns in the Aggregated CSV:"
    echo "----------------------------------------------------------"
    echo "| Column Name                 | Type    | Description                                                  |"
    echo "|-----------------------------|---------|--------------------------------------------------------------|"
    echo "| domain_id                   | string  | Protein domain identifier (e.g., '1a0aA00')                  |"
    echo "| resid                       | int     | Residue ID (position in the original chain)                  |"
    echo "| resname                     | string  | 3-letter amino acid code (e.g., 'ALA', 'LYS')                |"
    echo "| protein_size                | int     | Total number of residues in the protein/domain               |"
    echo "| normalized_resid            | float   | Residue position normalized to [0, 1] range                  |"
    echo "| core_exterior               | string  | Original classification ('core' or 'surface')                |"
    echo "| relative_accessibility    | float   | Solvent Accessible Surface Area (SASA), typically [0, 1]     |"
    echo "| dssp                        | string  | Secondary structure char (DSSP: H, E, C, T, G, etc.)         |"
    echo "| phi                         | float   | Backbone dihedral angle phi (degrees)                        |"
    echo "| psi                         | float   | Backbone dihedral angle psi (degrees)                        |"
    echo "| resname_encoded             | int     | Numerical encoding of 'resname'                              |"
    echo "| core_exterior_encoded     | int     | Numerical encoding of 'core_exterior' (e.g., 0=core, 1=surf) |"
    echo "| secondary_structure_encoded | int     | Numerical encoding of 'dssp' (e.g., 0=Helix, 1=Sheet, 2=Coil)|"
    echo "| phi_norm                    | float   | Normalized phi angle (e.g., sin(rad(phi)), [-1, 1])          |"
    echo "| psi_norm                    | float   | Normalized psi angle (e.g., sin(rad(psi)), [-1, 1])          |"
    echo "| temperature                 | float   | **INPUT FEATURE:** Temperature (K) for this data point       |" # NEW
    echo "| rmsf                        | float   | **TARGET:** RMSF value for this residue at 'temperature'     |" # RENAMED/UNIFIED
    echo "| esm_rmsf (OmniFlex only)    | float   | External prediction from ESM embeddings                      |" # Optional
    echo "| voxel_rmsf (OmniFlex only)  | float   | External prediction from 3D voxel representation             |" # Optional
    # Add any other generated window features like 'resname_encoded_offset_1', etc. if needed
    echo "| ... (*_offset_*)           | float/int | Optional window features based on neighboring residues     |"
    echo "----------------------------------------------------------"
    echo ""

    # Usage Examples Section (UPDATED)
    echo "=========================================================="
    echo "Usage Examples (EnsembleFlex)"
    echo "=========================================================="
    echo "# Train the unified model (uses aggregated data specified in config)"
    echo "ensembleflex train"
    echo ""
    echo "# Train a specific model type (e.g., random_forest)"
    echo "ensembleflex train --model random_forest"
    echo ""
    echo "# Train a specific model type (e.g., random_forest)"
    echo "ensembleflex run --model random_forest"
    echo ""
    echo "# Evaluate the trained model (uses test split from aggregated data)"
    echo "ensembleflex evaluate --model random_forest"
    echo ""
    echo "# Generate predictions using the trained model REQUIRES temperature"
    echo "ensembleflex predict --input new_proteins_features.csv --temperature 310.5"
    echo ""
    echo ""
    echo "# Use OmniFlex mode (if model was trained with it)"
    echo "ensembleflex train --mode omniflex"
    echo ""
    echo "# Override a config parameter"
    echo "ensembleflex train --param models.neural_network.training.epochs=50"
    echo ""

    # Project Info
    echo "=========================================================="
    echo "Project Information"
    echo "=========================================================="
    echo "Project Root Directory: $(pwd)"
    echo ""

    # Project Structure (Refined Exclusions)
    echo "Project Tree Structure:"
    echo "---------------------------------------------------------"
    # Exclude common cache/build/env/git/data/output/model dirs more reliably
    find . -maxdepth 3 -type d \( \
        -name "__pycache__" -o \
        -name ".git" -o \
        -name ".vscode" -o \
        -name "*.egg-info" -o \
        -name "build" -o \
        -name "dist" -o \
        -name "env_*" -o \
        -name "venv" -o \
        -name "env" -o \
        -path "./data" -o \
        -path "./output" -o \
        -path "./models" -o \
        -path "./test/test_data" -o \
        -path "./test/__pycache__" \
    \) -prune -o -type d -print | sort
    echo ""

    # File Listing (Refined Exclusions & Inclusions)
    echo "File Listing (Key Project Files):"
    echo "---------------------------------------------------------"
    find . -type d \( \
        -name "__pycache__" -o \
        -name ".git" -o \
        -name ".vscode" -o \
        -name "*.egg-info" -o \
        -name "build" -o \
        -name "dist" -o \
        -name "env_*" -o \
        -name "venv" -o \
        -name "env" -o \
        -path "./data" -o \
        -path "./output" -o \
        -path "./models" -o \
        -path "./test/test_data" \
    \) -prune -o -type f \( \
        -name "*.py" -o \
        -name "*.yaml" -o \
        -name "*.toml" -o \
        -name "README.md" -o \
        -name "*.sh" \
    \) -print | grep -v '/__pycache__/' | sort

    echo ""

    # Print default configuration
    echo "=========================================================="
    echo "Default Configuration (default_config.yaml)"
    echo "=========================================================="
    if [ -f "default_config.yaml" ]; then
        cat default_config.yaml
    else
        echo "default_config.yaml not found."
    fi
    echo ""

    # Print package structure and files (UPDATED PATHS)
    echo "=========================================================="
    echo "EnsembleFlex Package Files"
    echo "=========================================================="

    # Main package files
    echo "### Main Package Files ###"
    echo "---------------------------------------------------------"
    for file in pyproject.toml setup.py README.md; do
        if [ -f "$file" ]; then
            echo "===== FILE: $file ====="
            cat "$file"
            echo ""
        fi
    done

    # Core module files
    echo "### Core Module Files ###"
    echo "---------------------------------------------------------"
    for file in ensemble/__init__.py ensembleflex/config.py ensembleflex/pipeline.py ensembleflex/cli.py; do # Updated path
        if [ -f "$file" ]; then
            echo "===== FILE: $file ====="
            cat "$file"
            echo ""
        fi
    done

    # Model files
    echo "### Model Files ###"
    echo "---------------------------------------------------------"
    for file in ensemble/models/__init__.py ensembleflex/models/base.py ensembleflex/models/random_forest.py ensembleflex/models/neural_network.py; do # Updated path
        if [ -f "$file" ]; then
            echo "===== FILE: $file ====="
            cat "$file"
            echo ""
        fi
    done

    # Data handling files
    echo "### Data Handling Files ###"
    echo "---------------------------------------------------------"
    for file in ensemble/data/__init__.py ensembleflex/data/loader.py ensembleflex/data/processor.py; do # Updated path
        if [ -f "$file" ]; then
            echo "===== FILE: $file ====="
            cat "$file"
            echo ""
        fi
    done

    # Temperature handling files (Simplified - might remove later if comparison.py is empty)
    echo "### Temperature Analysis Files (Simplified) ###"
    echo "---------------------------------------------------------"
    for file in ensemble/temperature/__init__.py ensembleflex/temperature/comparison.py; do # Updated path
        if [ -f "$file" ]; then
            echo "===== FILE: $file ====="
            cat "$file"
            echo ""
        fi
    done

    # Utility files
    echo "### Utility Files ###"
    echo "---------------------------------------------------------"
    for file in ensemble/utils/__init__.py ensembleflex/utils/helpers.py ensembleflex/utils/metrics.py ensembleflex/utils/visualization.py; do # Updated path
        if [ -f "$file" ]; then
            echo "===== FILE: $file ====="
            cat "$file"
            echo ""
        fi
    done

    # Script files
    echo "### Script Files ###"
    echo "---------------------------------------------------------"
    for file in scripts/aggregate_data.py; do # Add other scripts if any
        if [ -f "$file" ]; then
            echo "===== FILE: $file ====="
            cat "$file"
            echo ""
        fi
    done


    # Output result files section (UPDATED PATH)
    echo "=========================================================="
    echo "Output Result Files (First 15 lines of each file)"
    echo "=========================================================="
    echo ""

    OUTPUT_DATA_DIR="./output/ensembleflex" # Updated path
    echo "Searching for output files in: $OUTPUT_DATA_DIR"
    if [ ! -d "$OUTPUT_DATA_DIR" ]; then
        echo "Output directory '$OUTPUT_DATA_DIR' not found."
    else
        # Find relevant files (CSV, potentially others), exclude plots/subdirs like training_performance
        output_files=$(find "$OUTPUT_DATA_DIR" -maxdepth 1 -type f \( -name "*.csv" -o -name "*.log" \) 2>/dev/null | sort)

        if [ -z "$output_files" ]; then
            echo "No primary output result files (.csv, .log) found directly in $OUTPUT_DATA_DIR."
            # Optionally search subdirs too, but keep it concise for context
            # echo "Searching subdirectories..."
            # find "$OUTPUT_DATA_DIR" -mindepth 2 -type f -not -name "*.png" ...
        else
            for file in $output_files; do
                echo "===== FILE: $file ====="
                echo "First 15 lines:"
                head -n 15 "$file"
                echo ""
                echo "---------------------------------------------------------"
            done
        fi
    fi

    # Footer
    echo "=========================================================="
    echo "End of EnsembleFlex Context Document" # Renamed
    echo "=========================================================="

} > "$OUTPUT_FILE"

echo "ensembleflex context file created at: $OUTPUT_FILE"