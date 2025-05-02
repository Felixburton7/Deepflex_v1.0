# scripts/get_rf_importance.py

import joblib
import argparse
import os
import sys
import logging
import pandas as pd # For saving CSV

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_rf_feature_importance(pkl_path, top_n=None, csv_output=None):
    """
    Loads a saved RandomForestModel state from a .pkl file and prints/saves
    feature importances.

    Args:
        pkl_path (str): Path to the .pkl file saved by RandomForestModel.save().
        top_n (int, optional): If specified, prints only the top N features. Defaults to None (print all).
        csv_output (str, optional): If specified, saves the full importance list to this CSV file. Defaults to None.
    """
    if not os.path.exists(pkl_path):
        logger.error(f"Error: PKL file not found at '{pkl_path}'")
        sys.exit(1)

    try:
        logger.info(f"Loading model state from: {pkl_path}")
        # Load the dictionary saved by the model's save method
        loaded_state = joblib.load(pkl_path)
        logger.info("Model state loaded successfully.")

        # --- Validation Checks ---
        if not isinstance(loaded_state, dict):
            logger.error("Error: Loaded object is not a dictionary. Expected the state dictionary saved by the model.")
            sys.exit(1)

        if 'model' not in loaded_state or 'feature_names' not in loaded_state:
            logger.error("Error: Loaded dictionary is missing 'model' or 'feature_names' key.")
            logger.error(f"Available keys: {list(loaded_state.keys())}")
            sys.exit(1)

        rf_model = loaded_state['model']
        feature_names = loaded_state['feature_names']

        # Check if the loaded model has the importance attribute
        if not hasattr(rf_model, 'feature_importances_'):
            logger.error(f"Error: Loaded model object (type: {type(rf_model)}) does not have 'feature_importances_' attribute.")
            sys.exit(1)

        importances = rf_model.feature_importances_

        if feature_names is None:
             logger.warning("Warning: Feature names were not found in the saved state. Showing importance by index.")
             feature_names = [f"feature_{i}" for i in range(len(importances))]

        if len(importances) != len(feature_names):
            logger.error(f"Error: Mismatch between number of importances ({len(importances)}) and feature names ({len(feature_names)}).")
            sys.exit(1)

        # --- Process and Display Importances ---
        importance_list = sorted(zip(importances, feature_names), reverse=True)

        logger.info("\n--- Feature Importances ---")
        features_to_print = importance_list[:top_n] if top_n is not None else importance_list

        for importance, name in features_to_print:
            print(f"{name}: {importance:.6f}")

        if top_n is not None and len(importance_list) > top_n:
             print(f"\n(Showing top {top_n} of {len(importance_list)} features)")

        # --- Save to CSV if requested ---
        if csv_output:
            try:
                df = pd.DataFrame(importance_list, columns=['Importance', 'Feature'])
                # Ensure directory exists
                output_dir = os.path.dirname(csv_output)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                df.to_csv(csv_output, index=False)
                logger.info(f"Full feature importance list saved to: {csv_output}")
            except Exception as e:
                 logger.error(f"Failed to save importance data to CSV '{csv_output}': {e}")


    except FileNotFoundError: # Should be caught above, but for safety
        logger.error(f"Error: File not found at '{pkl_path}'")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and display feature importance from a saved ensembleflex RandomForestModel PKL file.")
    parser.add_argument("pkl_file", help="Path to the RandomForestModel .pkl file.")
    parser.add_argument("-n", "--top-n", type=int, default=None, help="Display only the top N features.")
    parser.add_argument("-o", "--csv", type=str, default=None, help="Optional path to save the full importance list as a CSV file.")

    args = parser.parse_args()

    get_rf_feature_importance(args.pkl_file, args.top_n, args.csv)