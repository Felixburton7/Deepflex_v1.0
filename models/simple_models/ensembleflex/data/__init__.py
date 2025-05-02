"""
Data handling modules for the FlexSeq ML pipeline.

This package contains functions for loading, processing, and
manipulating protein data across multiple temperatures.
"""

# Import key functions for easier access
from ensembleflex.data.loader import load_file
from ensembleflex.data.processor import (
    load_and_process_data, 
    clean_data, 
    prepare_data_for_model
)