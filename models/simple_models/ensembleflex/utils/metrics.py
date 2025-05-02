# ensembleflex/utils/metrics.py

"""
Evaluation metrics for the ensembleflex ML pipeline.

This module provides functions for evaluating model performance on aggregated data
and potentially performing cross-validation. Temperature-specific logic related
to comparing separate runs has been removed.
"""

import logging
from typing import Dict, List, Any, Union, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    max_error,
    median_absolute_error
)
from sklearn.model_selection import KFold, cross_val_score # Keep cross_val_score for Q2 placeholder
from scipy.stats import pearsonr, spearmanr

# Assuming these are available from elsewhere in the project if needed by CV
# from ensembleflex.data.processor import prepare_data_for_model
# from ensembleflex.utils.helpers import progress_bar

logger = logging.getLogger(__name__)

# Default set of metrics if not specified in config
DEFAULT_METRICS = {
    "rmse": True,
    "mae": True,
    "r2": True,
    "pearson_correlation": True,
    "spearman_correlation": False,
    "explained_variance": False,
    "max_error": False,
    "median_absolute_error": False,
    "adjusted_r2": False,
    "root_mean_square_absolute_error": False, # Note: Interpreted as RMSE
    "q2": False # Placeholder for cross-validated R2
}

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: Dict[str, Any],
    X: Optional[np.ndarray] = None,
    n_features: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate model predictions using multiple metrics specified in the configuration.

    Handles potential calculation errors gracefully by logging warnings and
    returning NaN for failed metrics.

    Args:
        y_true: Array of true target values. Must be 1D.
        y_pred: Array of predicted values. Must be 1D and same length as y_true.
        config: Configuration dictionary, expected to have config['evaluation']['metrics'].
        X: Optional feature matrix (NumPy array) used for metrics like Q2 (currently placeholder).
        n_features: Optional number of features used by the model, required for adjusted R2.

    Returns:
        Dictionary mapping metric names (str) to their calculated values (float).
        Metrics that fail calculation will have a value of np.nan.
    """
    results = {}
    # Safely get metrics config, use defaults if missing
    metrics_config = config.get("evaluation", {}).get("metrics", {})
    if not metrics_config:
        logger.warning("No metrics specified in config['evaluation']['metrics']. Using defaults.")
        metrics_config = DEFAULT_METRICS

    # --- Input Validation ---
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        logger.error("Inputs y_true and y_pred must be NumPy arrays.")
        return {metric: np.nan for metric, enabled in metrics_config.items() if enabled}
    if y_true.ndim != 1 or y_pred.ndim != 1:
        logger.error(f"Inputs y_true (shape {y_true.shape}) and y_pred (shape {y_pred.shape}) must be 1D arrays.")
        return {metric: np.nan for metric, enabled in metrics_config.items() if enabled}
    if len(y_true) != len(y_pred):
        logger.error(f"Inputs y_true ({len(y_true)}) and y_pred ({len(y_pred)}) must have the same length.")
        return {metric: np.nan for metric, enabled in metrics_config.items() if enabled}
    if len(y_true) == 0:
        logger.warning("Input arrays are empty. Returning NaN for all metrics.")
        return {metric: np.nan for metric, enabled in metrics_config.items() if enabled}

    # --- Calculate Metrics ---

    # Helper function to calculate metric safely
    def _calculate_metric(metric_name: str, func, *args, **kwargs):
        if metrics_config.get(metric_name, False):
            try:
                value = func(*args, **kwargs)
                # Check for tuple return (like correlations)
                if isinstance(value, tuple):
                    results[metric_name] = value[0] if not np.isnan(value[0]) else 0.0 # Store corr, handle NaN
                    # Optionally store p-value if needed: results[f"{metric_name}_p_value"] = value[1]
                else:
                    results[metric_name] = float(value) if not np.isnan(value) else np.nan
            except ValueError as ve: # Catches issues like constant input for correlation/r2
                 logger.warning(f"{metric_name.upper()} calculation failed: {ve}. Assigning NaN.")
                 results[metric_name] = np.nan
            except Exception as e:
                 logger.warning(f"{metric_name.upper()} calculation failed unexpectedly: {e}. Assigning NaN.")
                 results[metric_name] = np.nan

    _calculate_metric("rmse", lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), y_true, y_pred)
    _calculate_metric("mae", mean_absolute_error, y_true, y_pred)
    _calculate_metric("r2", r2_score, y_true, y_pred)
    _calculate_metric("pearson_correlation", pearsonr, y_true, y_pred)
    _calculate_metric("spearman_correlation", spearmanr, y_true, y_pred)
    _calculate_metric("explained_variance", explained_variance_score, y_true, y_pred)
    _calculate_metric("max_error", max_error, y_true, y_pred)
    _calculate_metric("median_absolute_error", median_absolute_error, y_true, y_pred)

    # Adjusted R2 (requires n_features)
    if metrics_config.get("adjusted_r2", False):
        if n_features is not None and n_features > 0:
            n = len(y_true)
            r2 = results.get("r2") # Use already calculated R2 if available
            if r2 is None: # Calculate if needed
                 try: r2 = r2_score(y_true, y_pred); results["r2"] = r2 # Store it too
                 except ValueError: r2 = np.nan

            if not np.isnan(r2) and (n - n_features - 1) > 0: # Check denominator
                adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1)
                results["adjusted_r2"] = adj_r2
            else:
                results["adjusted_r2"] = np.nan
                if (n - n_features - 1) <= 0: logger.warning("Adjusted R2 calc failed: n_samples <= n_features + 1.")
                elif np.isnan(r2): logger.warning("Adjusted R2 calc failed: R2 was NaN.")
        else:
            logger.warning("Adjusted R2 requested but n_features not provided or invalid.")
            results["adjusted_r2"] = np.nan

    # Root Mean Square Absolute Error (Interpreted as RMSE)
    if metrics_config.get("root_mean_square_absolute_error", False):
        # This is mathematically equivalent to RMSE
        results["root_mean_square_absolute_error"] = results.get("rmse", np.nan)
        if np.isnan(results["root_mean_square_absolute_error"]): # Recalculate if RMSE failed/disabled
             _calculate_metric("root_mean_square_absolute_error", lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), y_true, y_pred)


    # Q2 Placeholder
    if metrics_config.get("q2", False):
        logger.warning("Q2 metric calculation is not implemented (requires CV setup). Assigning NaN.")
        results["q2"] = np.nan
        # if X is not None:
        #     # Placeholder for actual Q2 calculation using cross-validation
        #     # cv_r2 = cross_val_score(model_placeholder, X, y_true, cv=5, scoring='r2').mean()
        #     # results["q2"] = cv_r2
        #     pass
        # else:
        #     results["q2"] = np.nan


    # Ensure all requested metrics exist in the results dictionary
    for metric, enabled in metrics_config.items():
         if enabled and metric not in results:
              logger.debug(f"Metric '{metric}' was enabled but not calculated (or failed). Setting to NaN.")
              results[metric] = np.nan

    return results


# Note: cross_validate_model needs careful review if used with the new pipeline structure.
# It might be simpler to perform CV within the pipeline's train method if needed.
# The version below is adapted but might require further testing in the ensembleflex context.
def cross_validate_model(
    model_class: Any,
    model_params: Dict[str, Any],
    data: pd.DataFrame, # Expects aggregated data now
    config: Dict[str, Any],
    n_folds: int = 5,
    return_predictions: bool = False
) -> Union[Dict[str, float], Tuple[Dict[str, float], pd.DataFrame]]:
    """
    Perform cross-validation for a model using aggregated data.

    Handles stratified splitting by domain if configured and possible.

    Args:
        model_class: Model class to instantiate (should inherit from BaseModel).
        model_params: Parameters for model initialization (specific to the model).
        data: DataFrame with aggregated features and target.
        config: Configuration dictionary.
        n_folds: Number of cross-validation folds.
        return_predictions: Whether to return Out-Of-Fold (OOF) predictions.

    Returns:
        - Dictionary with cross-validation results (mean/std of metrics).
        - If return_predictions is True, also returns a DataFrame containing
          OOF predictions aligned with the original data index.
    """
    # Import here to avoid circular dependency if metrics is imported by processor
    from ensembleflex.data.processor import prepare_data_for_model
    from ensembleflex.utils.helpers import progress_bar

    logger.info(f"Starting {n_folds}-fold cross-validation...")

    # Use a copy to avoid modifying original data
    cv_data = data.copy()

    # Get metrics to calculate from config
    metrics_config = config.get("evaluation", {}).get("metrics", {})
    metrics_to_calculate = {m: [] for m, enabled in metrics_config.items() if enabled}
    if not metrics_to_calculate: # Add defaults if none specified
         metrics_to_calculate = {'rmse': [], 'mae': [], 'r2': [], 'pearson_correlation': []}
         logger.warning(f"No metrics enabled in config for CV, using defaults: {list(metrics_to_calculate.keys())}")


    # Add storage for OOF predictions if requested
    oof_preds_list = []
    oof_indices_list = []

    # Create cross-validation folds
    stratify_by_domain = config["dataset"]["split"].get("stratify_by_domain", True)
    random_state = config["system"]["random_state"]
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_iterator_desc = f"CV ({n_folds} folds)"
    fold_indices_or_domains: Union[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[List[str], List[str]]]] = []

    # Prepare splits (either domain IDs or row indices)
    if stratify_by_domain and 'domain_id' in cv_data.columns:
        unique_domains = cv_data["domain_id"].unique()
        if len(unique_domains) < n_folds:
            logger.warning(f"Number of unique domains ({len(unique_domains)}) < n_folds ({n_folds}). Stratification might be imperfect.")
        fold_indices_or_domains = [(unique_domains[train_idx], unique_domains[test_idx])
                                   for train_idx, test_idx in folds.split(unique_domains)]
        fold_iterator_desc += " stratified by domain"
    else:
        if stratify_by_domain: # Log if stratification was requested but not possible
            logger.warning("Stratify by domain requested but 'domain_id' column missing. Using regular KFold on rows.")
        fold_indices_or_domains = list(folds.split(cv_data)) # List of (train_indices, test_indices)

    # Perform CV
    fold_iterator = progress_bar(range(n_folds), desc=fold_iterator_desc)
    for i in fold_iterator:
        train_idx_or_domains, test_idx_or_domains = fold_indices_or_domains[i]

        # Select train/test data based on split type
        if stratify_by_domain and 'domain_id' in cv_data.columns:
            train_data = cv_data[cv_data["domain_id"].isin(train_idx_or_domains)]
            test_data = cv_data[cv_data["domain_id"].isin(test_idx_or_domains)]
        else: # Index-based split
            train_data = cv_data.iloc[train_idx_or_domains]
            test_data = cv_data.iloc[test_idx_or_domains]

        fold_test_indices = test_data.index # Store original indices for OOF

        if train_data.empty or test_data.empty:
             logger.warning(f"Skipping fold {i+1} due to empty train or test split.")
             continue

        try:
            # Prepare data for model (important to use fold-specific data)
            X_train, y_train, feature_names = prepare_data_for_model(train_data, config)
            X_test, y_test, _ = prepare_data_for_model(test_data, config)

            if X_train.size == 0 or X_test.size == 0:
                 logger.warning(f"Skipping fold {i+1} due to empty feature matrix after preparation.")
                 continue

            # Create and train model instance for this fold
            # Pass only model-specific params, not the whole config section
            fold_model_params = get_model_config(config, model_class.__name__.lower()) # Assumes class name matches config key
            model_instance = model_class(**fold_model_params)

            # Fit model
            if 'feature_names' in inspect.signature(model_instance.fit).parameters:
                 model_instance.fit(X_train, y_train, feature_names=feature_names)
            else:
                 model_instance.fit(X_train, y_train)

            # Generate predictions for the test fold
            y_pred = model_instance.predict(X_test)

            # Evaluate predictions for this fold
            n_features_fold = X_train.shape[1]
            # Pass the main config for evaluation metric settings
            fold_metrics = evaluate_predictions(y_test, y_pred, config, X=X_test, n_features=n_features_fold)

            # Store results
            for metric, value in fold_metrics.items():
                if metric in metrics_to_calculate:
                     metrics_to_calculate[metric].append(value)

            # Store OOF predictions if requested
            if return_predictions:
                oof_preds_list.append(pd.Series(y_pred, index=fold_test_indices))
                # oof_true_values.append(pd.Series(y_test, index=fold_test_indices)) # Less common to return true

        except Exception as e:
             logger.error(f"Error occurred in CV fold {i+1}: {e}", exc_info=True)
             # Add NaN for this fold's metrics to maintain alignment?
             for metric in metrics_to_calculate:
                  metrics_to_calculate[metric].append(np.nan)


    # Calculate final statistics (mean and stddev over folds)
    results_summary = {}
    for metric, values in metrics_to_calculate.items():
        valid_values = [v for v in values if pd.notna(v)] # Filter out NaNs
        if valid_values:
            results_summary[f"mean_{metric}"] = np.mean(valid_values)
            results_summary[f"std_{metric}"] = np.std(valid_values)
        else: # Handle case where all folds failed for a metric
             results_summary[f"mean_{metric}"] = np.nan
             results_summary[f"std_{metric}"] = np.nan
        # Add raw fold values if desired
        # results_summary[f"{metric}_folds"] = values


    if return_predictions:
        if oof_preds_list:
            # Concatenate predictions, aligning by original index
            oof_predictions_df = pd.concat(oof_preds_list).sort_index()
            # Add original target values for comparison if needed
            oof_predictions_df = pd.DataFrame({'oof_prediction': oof_predictions_df})
            oof_predictions_df = oof_predictions_df.join(data[[config['dataset']['target']]]) # Join original target
        else:
             oof_predictions_df = pd.DataFrame() # Return empty df if no preds generated
        return results_summary, oof_predictions_df
    else:
        return results_summary


# --- Other Utility Functions (Review for ensembleflex context) ---

def calculate_residue_metrics(
    df: pd.DataFrame,
    target_col: str,
    prediction_cols: List[str],
    include_uncertainty: bool = False,
    uncertainty_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate residue-level metrics from a DataFrame containing predictions.

    Assumes input df has 'domain_id', 'resid', 'resname', target_col, and prediction_cols.

    Args:
        df: DataFrame with true values and predictions.
        target_col: Column with true target values (e.g., 'rmsf').
        prediction_cols: List of column names with model predictions (e.g., ['nn_predicted']).
        include_uncertainty: Whether to include uncertainty metrics.
        uncertainty_cols: List of column names with prediction uncertainties (e.g., ['nn_uncertainty']).

    Returns:
        DataFrame with residue-level metrics (one row per residue).
    """
    if not all(c in df.columns for c in ['domain_id', 'resid', target_col]) or not prediction_cols:
        logger.error("Input DataFrame missing essential columns for calculate_residue_metrics.")
        return pd.DataFrame()

    results = []
    required_cols = ['domain_id', 'resid', target_col] + prediction_cols
    if 'resname' in df.columns: required_cols.append('resname')
    if include_uncertainty and uncertainty_cols: required_cols.extend(uncertainty_cols)
    # Add other structural features if needed and available
    for feature in ["secondary_structure_encoded", "core_exterior_encoded", "temperature"]:
        if feature in df.columns: required_cols.append(feature)

    # Group by unique residue identifier
    for (domain_id, resid), residue_group in df.groupby(["domain_id", "resid"], observed=False):
        if residue_group.empty: continue
        # Take the first row's data for residue-specific info (assuming it's constant)
        residue_data = residue_group.iloc[0]
        residue_metrics = {
            "domain_id": domain_id,
            "resid": resid,
        }
        # Copy static features
        for col in required_cols:
             if col not in [target_col] + prediction_cols + (uncertainty_cols or []):
                  residue_metrics[col] = residue_data.get(col, np.nan)

        # Use the mean of target/prediction/uncertainty if multiple temps exist per residue
        true_value = residue_group[target_col].mean() # Average actual value across temps
        residue_metrics["actual_mean"] = true_value

        for pred_col in prediction_cols:
            if pred_col not in residue_group.columns: continue
            pred_value = residue_group[pred_col].mean() # Average predicted value
            residue_metrics[pred_col+"_mean"] = pred_value

            # Calculate error based on average values
            error = pred_value - true_value
            abs_error = abs(error)
            model_name = pred_col.split("_predicted")[0]
            residue_metrics[f"{model_name}_error_mean"] = error
            residue_metrics[f"{model_name}_abs_error_mean"] = abs_error

            # Handle uncertainty
            if include_uncertainty and uncertainty_cols:
                unc_col = next((col for col in uncertainty_cols if model_name in col), None)
                if unc_col and unc_col in residue_group.columns:
                    unc_value = residue_group[unc_col].mean() # Average uncertainty
                    residue_metrics[f"{model_name}_uncertainty_mean"] = unc_value
                    # Calculate normalized error (using average error and average uncertainty)
                    if unc_value > 1e-9: # Avoid division by zero/small numbers
                        normalized_error = abs_error / unc_value
                        residue_metrics[f"{model_name}_normalized_error_mean"] = normalized_error
                    else:
                         residue_metrics[f"{model_name}_normalized_error_mean"] = np.nan

        results.append(residue_metrics)

    return pd.DataFrame(results)

# Note: This function is likely obsolete for ensembleflex as it assumes rmsf_{temp} columns
def calculate_temperature_scaling_factors(
    df: pd.DataFrame,
    temperatures: List[Union[int, float]] # Should be numeric temps
) -> Dict[str, float]:
    """
    [OBSOLETE - Requires Refactor] Calculate scaling factors between RMSF values at different temperatures.
    This requires columns named like 'rmsf_320', 'rmsf_450' which are not present
    in the standard ensembleflex aggregated data. Could be adapted to work on the
    output of the `compare-temperatures` CLI command's CSV.

    Args:
        df: DataFrame with RMSF values (e.g., 'rmsf_320', 'rmsf_450').
        temperatures: List of numeric temperatures corresponding to columns.

    Returns:
        Dictionary mapping temperature pairs (str) to scaling factors (float).
    """
    logger.warning("calculate_temperature_scaling_factors is likely obsolete or needs refactoring for ensembleflex.")
    temp_list = sorted([t for t in temperatures if isinstance(t, (int, float))])
    if len(temp_list) < 2: return {}

    scaling_factors = {}
    for i, temp1 in enumerate(temp_list):
        col1 = f"rmsf_{temp1}" # Assumes old column naming convention
        if col1 not in df.columns: continue
        for temp2 in temp_list[i+1:]:
            col2 = f"rmsf_{temp2}"
            if col2 not in df.columns: continue
            try:
                # Calculate average ratio, avoiding division by zero/NaNs
                valid_mask = (df[col1] > 1e-6) & df[col2].notna()
                if valid_mask.sum() > 0:
                    ratios = df.loc[valid_mask, col2] / df.loc[valid_mask, col1]
                    scaling_factors[f"{temp1}_to_{temp2}"] = ratios.mean()
                else:
                     scaling_factors[f"{temp1}_to_{temp2}"] = np.nan
            except Exception as e:
                 logger.error(f"Error calculating scaling factor {temp1}->{temp2}: {e}")
                 scaling_factors[f"{temp1}_to_{temp2}"] = np.nan

    return scaling_factors


def calculate_uncertainty_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray
) -> Dict[str, float]:
    """
    Calculate metrics related to uncertainty quantification.

    Requires true values, mean predictions, and predicted standard deviations.

    Args:
        y_true: Array of true target values.
        y_pred: Array of mean predicted values.
        y_std: Array of predicted standard deviations (uncertainty).

    Returns:
        Dictionary of uncertainty metrics.
    """
    if not all(isinstance(arr, np.ndarray) and arr.ndim == 1 and len(arr) == len(y_true) for arr in [y_pred, y_std]):
         logger.error("Invalid input shapes or types for calculate_uncertainty_metrics.")
         return {}
    if len(y_true) == 0: return {}

    results = {}
    errors = np.abs(y_true - y_pred)
    # Ensure std dev is not zero or negative for calculations
    y_std_safe = np.maximum(y_std, 1e-9) # Add small epsilon to avoid division by zero

    # Calibration metrics: Percentage within N std devs
    results["within_1std"] = np.mean(errors <= y_std_safe)
    results["within_2std"] = np.mean(errors <= 2 * y_std_safe)
    results["within_3std"] = np.mean(errors <= 3 * y_std_safe)

    # Calibration Error (deviation from ideal Gaussian coverage)
    results["calibration_error_1std"] = np.abs(results["within_1std"] - 0.6827)
    results["calibration_error_2std"] = np.abs(results["within_2std"] - 0.9545)
    results["calibration_error_3std"] = np.abs(results["within_3std"] - 0.9973)
    results["avg_calibration_error"] = np.mean([
        results["calibration_error_1std"],
        results["calibration_error_2std"],
        results["calibration_error_3std"]
    ])

    # Negative Log Predictive Density (NLPD) - Lower is better
    # Assumes Gaussian predictive distribution N(y_pred, y_std^2)
    try:
        variance = y_std_safe**2
        nlpd = 0.5 * np.mean(np.log(2 * np.pi * variance) + ((y_true - y_pred)**2) / variance)
        results["nlpd"] = nlpd
    except Exception as e:
        logger.warning(f"NLPD calculation failed: {e}")
        results["nlpd"] = np.nan


    # Uncertainty-Error Correlation (Pearson) - Higher positive correlation is generally better
    try:
        # Need variation in both std and error
        if np.var(y_std_safe) > 1e-9 and np.var(errors) > 1e-9:
            unc_err_corr, _ = pearsonr(y_std_safe, errors)
            results["uncertainty_error_correlation"] = unc_err_corr if not np.isnan(unc_err_corr) else 0.0
        else:
             results["uncertainty_error_correlation"] = np.nan
    except Exception as e:
        logger.warning(f"Uncertainty-error correlation failed: {e}")
        results["uncertainty_error_correlation"] = np.nan

    return results

def calculate_domain_performance(
    df: pd.DataFrame,
    target_col: str,
    prediction_cols: List[str]
) -> pd.DataFrame:
    """
    Calculate performance metrics grouped by domain ID.

    Assumes input df contains 'domain_id', target_col, prediction_cols,
    and potentially other static features like 'resname', 'secondary_structure_encoded'.

    Args:
        df: DataFrame with predictions and actual values (e.g., from all_results.csv).
        target_col: Target column name ('rmsf').
        prediction_cols: List of columns with model predictions (e.g., ['nn_predicted']).

    Returns:
        DataFrame with domain performance metrics.
    """
    if 'domain_id' not in df.columns or target_col not in df.columns or not prediction_cols:
         logger.error("Missing required columns ('domain_id', target, predictions) for calculate_domain_performance.")
         return pd.DataFrame()

    results = []
    logger.info("Calculating domain performance metrics...")

    for domain_id, domain_df in progress_bar(df.groupby("domain_id", observed=False), desc="Domain Perf."):
        if domain_df.empty: continue
        row = {"domain_id": domain_id}

        # Aggregate static features (take first value)
        static_cols = ['resname', 'secondary_structure_encoded', 'core_exterior_encoded', 'temperature'] # Add temp?
        for col in static_cols:
            if col in domain_df.columns:
                 # If feature is constant per domain (like resname, SS at a residue), take first.
                 # If feature varies (like temperature), calculate mean/median.
                 if domain_df[col].nunique() == 1:
                      row[f"{col}_unique"] = domain_df[col].iloc[0]
                 elif pd.api.types.is_numeric_dtype(domain_df[col].dtype):
                      row[f"{col}_mean"] = domain_df[col].mean()
                 else: # Categorical but not unique
                      row[f"{col}_mode"] = domain_df[col].mode()[0] if not domain_df[col].mode().empty else None


        # Add residue/row counts
        row["num_unique_residues"] = domain_df['resid'].nunique() if 'resid' in domain_df.columns else np.nan
        row["num_rows"] = len(domain_df)

        # Calculate performance metrics for each model
        for pred_col in prediction_cols:
            model_name = pred_col.split("_predicted")[0]
            if pred_col not in domain_df.columns: continue

            # Drop NaNs for metric calculation for this model/domain
            valid_rows = domain_df[[target_col, pred_col]].dropna()
            if valid_rows.empty or len(valid_rows) < 2: # Need at least 2 points for R2/Corr
                 row[f"{model_name}_rmse"] = np.nan
                 row[f"{model_name}_mae"] = np.nan
                 row[f"{model_name}_r2"] = np.nan
                 row[f"{model_name}_pearson"] = np.nan
                 continue

            y_true = valid_rows[target_col].values
            y_pred = valid_rows[pred_col].values

            try: row[f"{model_name}_rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
            except: row[f"{model_name}_rmse"] = np.nan

            try: row[f"{model_name}_mae"] = mean_absolute_error(y_true, y_pred)
            except: row[f"{model_name}_mae"] = np.nan

            try: row[f"{model_name}_r2"] = r2_score(y_true, y_pred) if np.var(y_true) > 1e-9 else np.nan
            except: row[f"{model_name}_r2"] = np.nan

            try:
                 pearson_corr, _ = pearsonr(y_true, y_pred) if np.var(y_true) > 1e-9 and np.var(y_pred) > 1e-9 else (np.nan, np.nan)
                 row[f"{model_name}_pearson"] = pearson_corr if not np.isnan(pearson_corr) else 0.0
            except: row[f"{model_name}_pearson"] = np.nan

        results.append(row)

    logger.info("Finished calculating domain performance metrics.")
    return pd.DataFrame(results)