#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Define output file (in the current directory)
OUTPUT_FILE="$(pwd)/AI_context.txt"

{
    echo "Working Directory: $(pwd)"
    echo ""
    echo "Full File Structure of Project:"
    tree .
    echo ""
    echo "---------------------------------------------------------"
    echo "Contents of Relevant Files (Ignoring Binary Files):"
    echo "---------------------------------------------------------"
    
    # Inspect files in key source directories: config, cli, data, models, visualization, utils
    find src/voxelflex/config \
         src/voxelflex/cli \
         src/voxelflex/data \
         src/voxelflex/models \
         src/voxelflex/visualization \
         src/voxelflex/utils \
         -type f ! -name "*.png" -print0 | while IFS= read -r -d '' file; do
        if file "$file" | grep -qE "text|ASCII|UTF-8"; then
            echo "===== FILE: $file ====="
            cat "$file"
            echo ""
        fi
    done

    echo ""
    echo "======================================="
    echo "Extracting First 10 Lines from Files in outputs/logs"
    echo "======================================="
    echo ""

    # Change directory to outputs/logs
    cd outputs/logs || exit 1
    echo "Current Logs Directory: $(pwd)"
    echo ""
    echo "Folder Structure in Logs:"
    tree .
    echo ""
    echo "Extracting First 10 Lines from Each Text File in Logs (if any):"
    echo "-------------------------------------------------------------------------------------"
    
    find . -type f ! -name "*.png" -print0 | while IFS= read -r -d '' file; do
        if file "$file" | grep -qE "text|ASCII|UTF-8"; then
            echo "===== FILE: $file ====="
            head -n 10 "$file"
            echo ""
        fi
    done

} > "$OUTPUT_FILE"

echo "Inspection complete. Output written to: $OUTPUT_FILE"
