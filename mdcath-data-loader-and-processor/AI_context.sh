#!/bin/bash

# Define output file in the current directory (project root)
OUTPUT_FILE="./AI_context.txt"
# Define the directory where output/result files are expected
OUTPUT_DIR="./outputs"

# --- Script Start ---
echo "Generating project context for AI..."

{
    echo "======================================="
    echo "Project Root Directory Information"
    echo "======================================="
    echo "Working Directory: $(pwd)"
    echo ""
    echo "Project File Structure:"
    echo "(Excluding: __pycache__, *.egg-info, .git, .vscode, ${OUTPUT_DIR})"
    echo "---------------------------------------------------------"
    # Use tree, ignoring specified patterns. Add more patterns if needed, separated by |
    tree -I '__pycache__|*.egg-info|.git|.vscode|outputs'
    echo ""
    echo ""

    echo "======================================="
    echo "Contents of Code, Config, and Metadata Files"
    echo "(Ignoring binary files and excluded directories)"
    echo "======================================="
    echo ""

    # Find relevant files, excluding specified directories and binary files
    # Excluded paths: anything inside __pycache__, *.egg-info, .git, .vscode, or the OUTPUT_DIR
    find . \
        -path "./__pycache__" -prune -o \
        -path "./*.egg-info" -prune -o \
        -path "./.git" -prune -o \
        -path "./.vscode" -prune -o \
        -path "$OUTPUT_DIR" -prune -o \
        -type f \( -name "*.py" -o -name "*.yaml" -o -name "*.md" -o -name "LICENSE" -o -name "requirements.txt" -o -name "pyproject.toml" \) -print0 | while IFS= read -r -d '' file; do
        
        # Check if the file is text-based before attempting to cat
        if file "$file" | grep -qE 'ASCII text|UTF-8 Unicode text|text'; then
            echo "===== FILE: $file ====="
            cat "$file"
            echo ""
            echo "--- End of File: $file ---"
            echo ""
        else
            echo "===== FILE: $file (Skipped - Detected as non-text) ====="
            echo ""
        fi
    done

    echo ""
    echo "======================================="
    echo "Preview of Output Files (First 10 Lines)"
    echo "(Looking in '${OUTPUT_DIR}' directory, ignoring binary files)"
    echo "======================================="
    echo ""

    # Check if the output directory exists
    if [ -d "$OUTPUT_DIR" ]; then
        echo "Found output directory: $OUTPUT_DIR"
        echo "---------------------------------------------------------"
        
        # Find all files within the OUTPUT_DIR
        find "$OUTPUT_DIR" -type f -print0 | while IFS= read -r -d '' file; do
            # Check if the file is text-based
            if file "$file" | grep -qE 'ASCII text|UTF-8 Unicode text|text'; then
                echo "===== FILE: $file (First 10 lines) ====="
                head -n 10 "$file"
                echo ""
                echo "--- End of Preview: $file ---"
                echo ""
            else
                echo "===== FILE: $file (Skipped - Detected as non-text) ====="
                echo ""
            fi
        done
    else
        echo "Output directory '${OUTPUT_DIR}' not found. Skipping output file preview."
    fi

    echo ""
    echo "======================================="
    echo "Context Generation Complete."
    echo "======================================="

} > "$OUTPUT_FILE"

echo "AI context saved to $OUTPUT_FILE"
# --- Script End ---