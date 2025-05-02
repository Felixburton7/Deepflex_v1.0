#!/bin/bash

# Define the output file path
OUTPUT_FILE="$PWD/DeepFlex_context.txt"

{
    echo "=========================================================="
    echo "                Deep-flex Context Document"
    echo "=========================================================="
    echo ""
    
    echo "Project Working Directory: $(pwd)"
    echo ""
    
    echo "---------------------------------------------------------"
    echo "List of .yaml and .py files:"
    echo "---------------------------------------------------------"
    # Find and list the matching files
    find . -type f \( -name "*.yaml" -o -name "*.py" \) | sort
    echo ""
    
    echo "=========================================================="
    echo "File Contents:"
    echo "=========================================================="
    
    # Loop through each matching file and print its contents
    find . -type f \( -name "*.yaml" -o -name "*.py" \) | sort | while read -r file; do
        echo "===== FILE: $file ====="
        cat "$file"
        echo ""
        echo "---------------------------------------------------------"
    done
    
    echo "=========================================================="
    echo "End of Deep-flex Context Document"
    echo "=========================================================="
    
} > "$OUTPUT_FILE"

echo "Context file created at: $OUTPUT_FILE"