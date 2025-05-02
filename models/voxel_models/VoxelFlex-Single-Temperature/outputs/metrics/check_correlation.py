import os 
import sys 
import pandas as pd 
import numpy as np 
from typing import Dict, List, Set, Optional 
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
import logging

INPUT_PATH = "/home/s_felix/FINAL_PROJECT/packages/voxelflex/VoxelFlex/outputs/metrics/predictions_20250413_140319.csv"

root_logger = logging.getLogger()
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

def predict_correlation(INPUT_PATH: str) -> None: 
    if os.path.exists(INPUT_PATH): 
        try: 
            logger.info("Loading file now")
            df = pd.read_csv(INPUT_PATH)
        except Exception as e: 
            logger.warning(f"Failed to load file: {e}")
            return
    else:
        logger.warning(f"No file found at {INPUT_PATH}")
        return
    
    predicted_col = None
    actual_col = None

    for col in df.columns: 
        if "predicted_rmsf" in col:
            predicted_col = col
        elif "actual_rmsf" in col:
            actual_col = col

    if predicted_col and actual_col:
        predicted = df[predicted_col]
        actual = df[actual_col]

        pearson_corr, _ = pearsonr(predicted, actual)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)

        logger.info(f"Pearson correlation (r): {pearson_corr:.4f}")
        logger.info(f"Mean Absolute Error (MAE): {mae:.4f}")
        logger.info(f"Coefficient of Determination (R²): {r2:.4f}")
    else:
        logger.warning("Required columns not found in the DataFrame.")

def main(): 
    predict_correlation(INPUT_PATH)

if __name__ == '__main__': 
    main()