# /home/s_felix/ensembleflex/ensembleflex/pipeline.py

"""
Main pipeline orchestration for the ensembleflex ML workflow.

Handles training, evaluation, prediction, and analysis for the
single, unified, temperature-aware model using aggregated data.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import time
import inspect
# import joblib # Keep if models use joblib
from ensembleflex.models.neural_network import NeuralNetworkModel
from ensembleflex.models.random_forest import RandomForestModel
from ensembleflex.models.lightgbm import LightGBMModel
from ensembleflex.models.lightgbm_classifier import LightGBMClassifier



# Updated imports for ensembleflex structure
from ensembleflex.config import (
    load_config,
    get_enabled_models,
    get_model_config,
    get_output_dir,  # UPDATED
    get_models_dir   # UPDATED
)
from ensembleflex.utils.helpers import progress_bar, ProgressCallback, ensure_dir
from ensembleflex.models import get_model_class, BaseModel # Import BaseModel for type hint
from ensembleflex.data.processor import (
    load_file,
    load_and_process_data,
    split_data,
    prepare_data_for_model,
    process_features
)

from ensembleflex.utils.metrics import evaluate_predictions
# Import visualization functions that will be added/adapted
from ensembleflex.utils.visualization import (
    plot_feature_importance,
    plot_scatter_with_density_contours,
    # plot_residue_level_rmsf, # Keep if useful for single domain viz
    plot_amino_acid_error_analysis,
    plot_error_analysis_by_property,
    plot_prediction_vs_temperature, # NEW
    plot_error_vs_temperature,      # NEW
    plot_training_validation_curves # Keep for NN
)

logger = logging.getLogger(__name__)

class Pipeline:
    """
    Main pipeline orchestration for ensembleflex.
    Handles the full ML workflow for the unified temperature-aware model.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline with configuration.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.models: Dict[str, BaseModel] = {} # Stores the single trained model instance {model_name: model_object}
        self.output_dir = get_output_dir(config)
        self.models_dir = get_models_dir(config)
        self.prepare_directories() # Use updated paths

        mode = config.get("mode", {}).get("active", "standard")
        logger.info(f"Pipeline initialized in {mode.upper()} mode.")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Models directory: {self.models_dir}")



    def prepare_directories(self) -> None:
        """Create necessary output directories using unified paths."""
        paths = self.config["paths"]
        data_dir = paths.get("data_dir", "./data") # Keep for data loading consistency
        ensure_dir(data_dir)
        ensure_dir(self.output_dir)
        ensure_dir(self.models_dir)

        # Create subdirectories within the unified output directory
        analysis_subdirs = [
            "feature_importance",
            "residue_analysis",
            "domain_analysis", # Keep if domain-level metrics are still generated
            "comparisons", # Keep for plots like scatter comparisons
            "training_performance", # For NN history
            "temperature_analysis" # NEW subdir for temp-specific plots
        ]
        for subdir in analysis_subdirs:
            ensure_dir(os.path.join(self.output_dir, subdir))


    # def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
    #     """
    #     Loads and processes the single aggregated input data file.

    #     Args:
    #         data_path: Optional explicit path to override config's file_pattern.

    #     Returns:
    #         Processed DataFrame.
    #     """
    #     # Determine input data path (CLI override or config default)
    #     effective_data_path = data_path
    #     if not effective_data_path:
    #         data_dir = self.config["paths"]["data_dir"]
    #         file_pattern = self.config["dataset"]["file_pattern"] # Should be the aggregated file name
    #         effective_data_path = os.path.join(data_dir, file_pattern)
    #         logger.info(f"Loading data from config path: {effective_data_path}")
    #     else:
    #          logger.info(f"Loading data from provided path: {effective_data_path}")

    #     if not os.path.exists(effective_data_path):
    #          raise FileNotFoundError(f"Data file not found: {effective_data_path}")

    #     # Call processor's main loading function
    #     return load_and_process_data(data_path=effective_data_path, config=self.config)
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
            """
            Loads and processes the single aggregated input data file.

            Args:
                data_path: Optional explicit path to override config's file_pattern.

            Returns:
                Processed DataFrame.
            """
            # Determine input data path (CLI override or config default)
            effective_data_input: Union[str, None] = data_path # Start with optional path override
            if not effective_data_input:
                # If no path override, construct path from config
                data_dir = self.config["paths"]["data_dir"]
                file_pattern = self.config["dataset"]["file_pattern"] # Should be the aggregated file name
                effective_data_input = os.path.join(data_dir, file_pattern)
                logger.info(f"Loading data from config path: {effective_data_input}")
            else:
                logger.info(f"Loading data from provided path: {effective_data_input}")

            return load_and_process_data(effective_data_input, config=self.config)


    def train(
        self,
        model_names: Optional[List[str]] = None,
        data_path: Optional[str] = None
    ) -> Dict[str, BaseModel]:
        """
        Train the single specified model on the aggregated data.

        Handles passing validation data for early stopping and feature names
        to the model's fit method if supported.

        Args:
            model_names: List containing the single model name to train.
                         If None, uses the first enabled model from config.
            data_path: Optional explicit path to the aggregated data file.

        Returns:
            Dictionary containing the trained model instance: {model_name: model_object},
            or an empty dictionary if training fails.
        """
        # 1. Determine the single model to train
        if not model_names:
            model_names = get_enabled_models(self.config)
            if not model_names:
                logger.error("No models enabled in configuration. Cannot train.")
                return {}
            model_name_to_train = model_names[0]
            logger.info(f"No model specified, training first enabled model: {model_name_to_train}")
        elif len(model_names) > 1:
             model_name_to_train = model_names[0]
             logger.warning(f"Multiple models specified ({model_names}). Training only the first: {model_name_to_train}")
        else:
             model_name_to_train = model_names[0]

        # 2. Load and preprocess the full aggregated data
        logger.info("Loading and processing aggregated data...")
        try:
            with ProgressCallback(total=1, desc="Loading data") as pbar:
                df = self.load_data(data_path=data_path)
                pbar.update()
            if df.empty:
                logger.error("Loaded data is empty. Cannot train.")
                return {}
        except FileNotFoundError as e:
             logger.error(f"Data loading failed: {e}")
             return {}
        except Exception as e:
             logger.error(f"Data loading/processing failed: {e}", exc_info=True)
             return {}


        # 3. Split data into train/validation/test sets once
        logger.info("Splitting data into train/validation/test sets...")
        try:
            with ProgressCallback(total=1, desc="Splitting data") as pbar:
                train_df, val_df, test_df = split_data(df, self.config)
                pbar.update()
            if train_df.empty:
                 logger.error("Training split is empty after splitting data. Cannot train.")
                 return {}
            # val_df can be empty if validation_size is 0 or very small
            if val_df.empty:
                 logger.warning("Validation split is empty. Early stopping based on validation data will be disabled.")

        except Exception as e:
             logger.error(f"Data splitting failed: {e}", exc_info=True)
             return {}


        # 4. Prepare training and validation data (features X and target y)
        logger.info("Preparing features for training and validation...")
        try:
            X_train, y_train, feature_names = prepare_data_for_model(train_df, self.config, include_target=True)
            X_val, y_val = None, None
            if not val_df.empty:
                X_val, y_val, _ = prepare_data_for_model(val_df, self.config, include_target=True)
            else:
                 logger.info("Skipping validation data preparation as validation set is empty.")


            # Sanity check features
            if X_train.size == 0 or y_train.size == 0:
                 logger.error("Training data (X_train or y_train) is empty after preparation. Cannot train.")
                 return {}
            if self.config['dataset']['features']['use_features'].get('temperature', False):
                 if 'temperature' not in feature_names:
                      logger.error("Config enables temperature feature, but it's missing from prepared training data!")
                      # Optionally raise error: raise ValueError("Temp feature missing")
                 else: logger.info(f"Training with {len(feature_names)} features, including 'temperature'.")
            else: logger.info(f"Training with {len(feature_names)} features (temp excluded).")

        except Exception as e:
             logger.error(f"Feature preparation failed: {e}", exc_info=True)
             return {}


        # 5. --- Train the Single Model ---
        trained_models: Dict[str, BaseModel] = {}
        logger.info(f"--- Training Model: {model_name_to_train} ---")

        try:
            model_class = get_model_class(model_name_to_train)
            model_config = get_model_config(self.config, model_name_to_train) # Merges common and specific
            if not model_config.get("enabled", False):
                logger.error(f"Model {model_name_to_train} is not enabled in the final config. Cannot train."); return {}

            # Instantiate the model
            model = model_class(**model_config)

            # --- HPO Check (Informational) ---
            if (model_name_to_train == "neural_network" and model_config.get("hyperparameter_optimization", {}).get("enabled", False)) or \
               (model_name_to_train == "random_forest" and model_config.get("randomized_search", {}).get("enabled", False)):
                logger.warning("HPO enabled. Ensure it's handled within model's .fit() or done prior.")

            # --- Fit the model ---
            start_time = time.time()
            logger.info("Fitting model...")

            # Prepare keyword arguments for model.fit dynamically
            fit_kwargs = {}
            fit_signature = inspect.signature(model.fit)

            # Pass validation data if fit accepts it and data exists
            if 'X_val' in fit_signature.parameters and 'y_val' in fit_signature.parameters:
                 if X_val is not None and y_val is not None:
                      fit_kwargs['X_val'] = X_val
                      fit_kwargs['y_val'] = y_val
                      logger.debug(f"Passing validation data to {model_name_to_train}.fit.")
                 else:
                      logger.debug(f"{model_name_to_train}.fit accepts validation data, but validation set was empty.")

            # Pass feature names if fit accepts it
            if 'feature_names' in fit_signature.parameters:
                 fit_kwargs['feature_names'] = feature_names
                 logger.debug(f"Passing feature_names to {model_name_to_train}.fit.")

            # Call fit with appropriate arguments
            model.fit(X_train, y_train, **fit_kwargs)

            train_time = time.time() - start_time
            logger.info(f"Finished training {model_name_to_train} in {train_time:.2f} seconds")

            # Store the successfully trained model
            trained_models[model_name_to_train] = model
            self.models[model_name_to_train] = model # Cache in pipeline instance

            # --- Post-Training Steps ---

            # Save Training History (NN specific usually)
            if hasattr(model, 'get_training_history') and callable(model.get_training_history):
                history = model.get_training_history()
                if history:
                    try:
                        history_df = pd.DataFrame(history)
                        perf_dir = os.path.join(self.output_dir, "training_performance")
                        ensure_dir(perf_dir)
                        history_path = os.path.join(perf_dir, f"{model_name_to_train}_training_history.csv")
                        history_df.to_csv(history_path, index=False)
                        logger.info(f"Saved training history to {history_path}")
                        # Plot curves
                        curve_plot_path = os.path.join(perf_dir, f"{model_name_to_train}_training_curves.png")
                        plot_training_validation_curves(history, history, model_name_to_train, curve_plot_path) # Pass history dict directly
                    except Exception as hist_e:
                        logger.warning(f"Could not save/plot training history: {hist_e}", exc_info=True)

            # Save the trained model object
            if model_config.get("save_best", True):
                with ProgressCallback(total=1, desc=f"Saving {model_name_to_train}") as pbar:
                    self.save_model(model, model_name_to_train)
                    pbar.update()

            # Evaluate on Validation Set (if validation set exists)
            if X_val is not None and y_val is not None:
                logger.info("Evaluating model performance on validation set...")
                with ProgressCallback(total=1, desc=f"Validating {model_name_to_train}") as pbar:
                    val_predictions = model.predict(X_val)
                    n_features_train = X_train.shape[1]
                    val_metrics = evaluate_predictions(y_val, val_predictions, self.config, X=X_val, n_features=n_features_train)
                    logger.info(f"Validation metrics for {model_name_to_train}: {val_metrics}")
                    # Save validation metrics
                    val_metrics_df = pd.DataFrame([val_metrics], index=[model_name_to_train])
                    val_metrics_df.index.name = "model"
                    val_metrics_path = os.path.join(self.output_dir, f"{model_name_to_train}_validation_metrics.csv")
                    try:
                         val_metrics_df.to_csv(val_metrics_path)
                         logger.info(f"Saved validation metrics to {val_metrics_path}")
                    except Exception as e:
                         logger.error(f"Failed to save validation metrics: {e}")
                    pbar.update()
            else:
                 logger.info("Skipping evaluation on validation set as it was empty.")


        except Exception as e:
            logger.error(f"Error during training process for {model_name_to_train}: {e}", exc_info=True)
            # Return empty dict if training failed catastrophically
            return {}

        return trained_models


    def save_model(self, model: BaseModel, model_name: str) -> None:
            """
            Save the single trained model to the unified models directory.

            Args:
                model: Trained model instance.
                model_name: Name of the model.
            """
            # Use the unified models directory stored in self.models_dir
            # Determine file extension - default to .pkl, use .pt if it's a torch Module
            if isinstance(model.model, torch.nn.Module): # Check the actual model object inside the wrapper
                model_filename = f"{model_name}.pt"
            else:
                model_filename = f"{model_name}.pkl"

            model_path = os.path.join(self.models_dir, model_filename)

            try:
                # The model's save method should handle the actual saving logic
                model.save(model_path) # The model's save method receives the full path including extension
                logger.info(f"Saved model '{model_name}' to: {model_path}")
            except AttributeError:
                logger.error(f"Model object for '{model_name}' does not have a 'save' method.")
            except Exception as e:
                logger.error(f"Error saving model '{model_name}' to {model_path}: {e}", exc_info=True)


    def load_model(self, model_name: str) -> BaseModel:
        """
        Load the single trained model from the unified models directory.

        Args:
            model_name: Name of the model to load.

        Returns:
            Loaded model instance.
        """
        # Use the unified models directory stored in self.models_dir
        # Try common extensions
        possible_extensions = ['.pkl', '.pt'] # Add more if needed
        model_path = None
        for ext in possible_extensions:
             path_try = os.path.join(self.models_dir, f"{model_name}{ext}")
             if os.path.exists(path_try):
                  model_path = path_try
                  break

        if model_path is None:
            raise FileNotFoundError(f"Model file for '{model_name}' not found in {self.models_dir} with extensions {possible_extensions}")

        logger.info(f"Loading model '{model_name}' from: {model_path}")
        try:
            model_class = get_model_class(model_name)
            # The model's load method should handle instantiation and state loading
            model = model_class.load(model_path)
            logger.info(f"Successfully loaded model '{model_name}'.")
            return model
        except AttributeError:
             logger.error(f"Model class for '{model_name}' does not have a 'load' classmethod.")
             raise
        except Exception as e:
            logger.error(f"Error loading model '{model_name}' from {model_path}: {e}", exc_info=True)
            raise


    def evaluate(
        self,
        model_names: Optional[List[str]] = None,
        data_path: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the single trained model on the test split of the aggregated data.
        Includes enhanced logging for feature mismatch debugging.

        Args:
            model_names: List containing the single model name to evaluate.
                         If None, uses the first enabled model from config.
            data_path: Optional explicit path to the aggregated data file.

        Returns:
            Dictionary containing evaluation metrics for the model: {model_name: metrics_dict}.
        """
        # 1. Determine Model to Evaluate
        if not model_names:
            model_names = get_enabled_models(self.config)
            if not model_names:
                logger.error("No models enabled in configuration. Cannot evaluate.")
                return {}
            model_name_to_eval = model_names[0]
            logger.info(f"No model specified, evaluating first enabled model: {model_name_to_eval}")
        elif len(model_names) > 1:
             model_name_to_eval = model_names[0]
             logger.warning(f"Multiple models specified ({model_names}). Evaluating only: '{model_name_to_eval}'")
        else:
             model_name_to_eval = model_names[0]

        # 2. Load Data and Get Evaluation Split
        logger.info("Loading aggregated data for evaluation split...")
        try:
            with ProgressCallback(total=1, desc="Loading data") as pbar:
                df = self.load_data(data_path=data_path)
                pbar.update()
            if df.empty: raise ValueError("Loaded data is empty.")

            with ProgressCallback(total=1, desc="Splitting data") as pbar:
                _, val_df, test_df = split_data(df, self.config) # Get both val and test splits
                pbar.update()

            comparison_set_name = self.config["evaluation"].get("comparison_set", "test").lower()
            if comparison_set_name == "validation":
                eval_df = val_df
                logger.info("Using validation set for evaluation.")
            else:
                if comparison_set_name != "test": logger.warning(f"Unknown comparison_set, using test.")
                eval_df = test_df
                logger.info("Using test set for evaluation.")

            if eval_df.empty: raise ValueError(f"{comparison_set_name} set is empty.")

        except (FileNotFoundError, ValueError, Exception) as e:
             logger.error(f"Failed to load or split data for evaluation: {e}", exc_info=True)
             return {}

        # 3. Prepare Evaluation Data (X, y, feature_names)
        logger.info(f"Preparing {comparison_set_name} features for evaluation...")
        try:
            with ProgressCallback(total=1, desc="Preparing features") as pbar:
                # This is where feature names for evaluation are generated
                X_eval, y_eval, feature_names_eval = prepare_data_for_model(eval_df, self.config, include_target=True)
                pbar.update()
            if X_eval.size == 0 or y_eval.size == 0:
                raise ValueError("Evaluation data (X or y) is empty after preparation.")
            logger.debug(f"Features prepared for EVALUATION ({len(feature_names_eval)}): {feature_names_eval}") # Log prepared features

        except Exception as e:
             logger.error(f"Feature preparation failed for evaluation: {e}", exc_info=True)
             return {}

        # 4. Evaluate the Single Model
        results = {}
        predictions = {}
        uncertainties = {}
        logger.info(f"--- Evaluating Model: {model_name_to_eval} ---")

        try:
            # Load model
            with ProgressCallback(total=1, desc=f"Loading {model_name_to_eval}", leave=False) as pbar:
                model = self.models.get(model_name_to_eval) or self.load_model(model_name_to_eval)
                self.models[model_name_to_eval] = model # Cache loaded model
                pbar.update()

            # --- START: Feature Consistency Check ---
            if hasattr(model, 'feature_names_') and model.feature_names_:
                expected_features = model.feature_names_
                logger.debug(f"Model '{model_name_to_eval}' expects features ({len(expected_features)}): {expected_features}")

                if feature_names_eval != expected_features:
                    logger.error(f"FATAL: Feature mismatch detected for model '{model_name_to_eval}'!")
                    logger.error(f"  Data has {len(feature_names_eval)} features: {feature_names_eval}")
                    logger.error(f"  Model expects {len(expected_features)} features: {expected_features}")

                    # Detailed comparison (optional, can be verbose)
                    set_eval = set(feature_names_eval)
                    set_expected = set(expected_features)
                    missing_in_eval = set_expected - set_eval
                    extra_in_eval = set_eval - set_expected
                    if missing_in_eval:
                        logger.error(f"  Features MISSING in evaluation data: {sorted(list(missing_in_eval))}")
                    if extra_in_eval:
                        logger.error(f"  Features EXTRA in evaluation data: {sorted(list(extra_in_eval))}")
                    if len(feature_names_eval) == len(expected_features) and feature_names_eval != expected_features:
                         logger.error("  Feature ORDER differs.")

                    # Option 1: Raise an error to stop execution
                    raise ValueError(f"Feature mismatch for model '{model_name_to_eval}'. Cannot proceed.")
                    # Option 2: Return empty results (current behavior if error is caught below)
                    # return {}
                else:
                     logger.info(f"Feature consistency check passed for model '{model_name_to_eval}'.")

            else:
                 logger.warning(f"Loaded model '{model_name_to_eval}' does not have 'feature_names_' attribute. Cannot perform consistency check.")
            # --- END: Feature Consistency Check ---


            # Generate predictions
            logger.info("Generating predictions on evaluation set...")
            with ProgressCallback(total=1, desc=f"Predicting", leave=False) as pbar:
                if hasattr(model, 'predict_with_std') and callable(model.predict_with_std):
                    preds, stds = model.predict_with_std(X_eval)
                    uncertainties[model_name_to_eval] = stds
                else:
                    preds = model.predict(X_eval)
                predictions[model_name_to_eval] = preds
                pbar.update()

            # Calculate metrics
            logger.info("Calculating evaluation metrics...")
            with ProgressCallback(total=1, desc="Computing metrics", leave=False) as pbar:
                n_features = X_eval.shape[1] # Use actual shape of X_eval
                metrics = evaluate_predictions(y_eval, preds, self.config, X=X_eval, n_features=n_features)
                pbar.update()

            results[model_name_to_eval] = metrics
            logger.info(f"Evaluation metrics for {model_name_to_eval}: {metrics}")

        except FileNotFoundError as e:
             logger.error(f"Could not load model '{model_name_to_eval}' for evaluation: {e}")
             return {}
        except ValueError as ve: # Catch the explicit ValueError from the feature check
             logger.error(f"Evaluation aborted due to error: {ve}")
             return {}
        except Exception as e:
            logger.error(f"Error during evaluation process for {model_name_to_eval}: {e}", exc_info=True)
            return {} # Return empty if evaluation fails

        # 5. Save evaluation results
        logger.info("Saving evaluation results...")
        with ProgressCallback(total=1, desc="Saving results") as pbar:
            # Pass original eval_df (contains feature names if derived from pandas)
            # Pass predictions dict and uncertainties dict
            self.save_evaluation_results(results, eval_df, predictions, uncertainties)
            pbar.update()

        return results


    def save_evaluation_results(
        self,
        results: Dict[str, Dict[str, float]], # {model_name: metrics_dict}
        eval_df: pd.DataFrame, # The original eval split df (test or val)
        predictions: Dict[str, np.ndarray], # {model_name: predictions_array}
        uncertainties: Dict[str, np.ndarray] # {model_name: uncertainty_array}
    ) -> None:
        """
        Save evaluation results (metrics and detailed predictions) to the unified output directory.

        Args:
            results: Dictionary of evaluation metrics for the evaluated model.
            eval_df: DataFrame with the evaluation data (features and actual target).
            predictions: Dictionary of predictions by model name.
            uncertainties: Dictionary of prediction uncertainties by model name.
        """
        # Save metrics summary to CSV
        results_path = os.path.join(self.output_dir, "evaluation_results.csv")
        try:
            results_summary_df = pd.DataFrame(results).T # Transpose to have models as rows
            results_summary_df.index.name = "model"
            results_summary_df.to_csv(results_path)
            logger.info(f"Saved evaluation metrics summary to: {results_path}")
        except Exception as e:
             logger.error(f"Failed to save evaluation metrics summary: {e}")


        # Save detailed results (original eval data + predictions + errors + uncertainty)
        if predictions:
            # Make a copy to avoid modifying the original eval_df slice
            # Ensure index alignment if eval_df was sliced/diced earlier
            all_results_df = eval_df.copy()
            target_col = self.config["dataset"]["target"] # Should be 'rmsf'

            # Add predictions and calculate errors for each model evaluated (usually just one)
            for model_name, preds in predictions.items():
                pred_col_name = f"{model_name}_predicted"
                error_col_name = f"{model_name}_error"
                abs_error_col_name = f"{model_name}_abs_error"

                # Assign predictions based on index
                all_results_df[pred_col_name] = pd.Series(preds, index=all_results_df.index)

                # Calculate errors (ensure target column exists)
                if target_col in all_results_df.columns:
                    all_results_df[error_col_name] = all_results_df[pred_col_name] - all_results_df[target_col]
                    all_results_df[abs_error_col_name] = np.abs(all_results_df[error_col_name])
                else:
                     logger.warning(f"Target column '{target_col}' not found in eval_df. Cannot calculate errors.")
                     all_results_df[error_col_name] = np.nan
                     all_results_df[abs_error_col_name] = np.nan


                # Add uncertainties if available
                if uncertainties and model_name in uncertainties:
                    uncertainty_col_name = f"{model_name}_uncertainty"
                    all_results_df[uncertainty_col_name] = pd.Series(uncertainties[model_name], index=all_results_df.index)

            # Save the combined DataFrame
            all_results_path = os.path.join(self.output_dir, "all_results.csv")
            try:
                all_results_df.to_csv(all_results_path, index=False)
                logger.info(f"Saved detailed evaluation results to: {all_results_path}")

                # Optionally save domain-level metrics based on this detailed file
                domain_metrics_path = os.path.join(self.output_dir, "domain_analysis", "domain_metrics.csv")
                self.save_domain_metrics(all_results_df, target_col, list(predictions.keys()), domain_metrics_path)

            except Exception as e:
                 logger.error(f"Failed to save detailed evaluation results: {e}")

        else:
             logger.warning("No predictions provided to save_evaluation_results. Skipping detailed results saving.")


    def save_domain_metrics(
        self,
        results_df: pd.DataFrame,
        target_col: str,
        model_names: List[str],
        output_path: str
    ) -> None:
        """
        Calculate and save domain-level metrics from detailed results.

        Args:
            results_df: DataFrame with all results (including predictions/errors).
            target_col: Target column name ('rmsf').
            model_names: List of model names present in results_df.
            output_path: Full path to save the domain metrics CSV.
        """
        if 'domain_id' not in results_df.columns:
             logger.warning("Cannot calculate domain metrics: 'domain_id' column missing.")
             return

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        domain_metrics = []
        logger.info("Calculating domain-level metrics...")

        for domain_id, domain_df in progress_bar(results_df.groupby("domain_id", observed=False), desc="Domain Metrics"):
            if domain_df.empty: continue
            domain_result = {"domain_id": domain_id}

            # Get unique residue count for this domain
            num_unique_residues = domain_df['resid'].nunique()
            domain_result["num_unique_residues"] = num_unique_residues
            domain_result["num_rows"] = len(domain_df) # Rows = residues * temps

            # Add avg temp if temp column exists
            if 'temperature' in domain_df.columns:
                 domain_result["avg_temperature"] = domain_df['temperature'].mean()

            # Calculate metrics for each model
            for model_name in model_names:
                pred_col = f"{model_name}_predicted"
                error_col = f"{model_name}_error"
                abs_error_col = f"{model_name}_abs_error"

                if pred_col not in domain_df.columns or target_col not in domain_df.columns:
                    logger.debug(f"Skipping metrics for model {model_name} in domain {domain_id} (missing columns).")
                    # Add NaN placeholders for consistent columns
                    domain_result[f"{model_name}_rmse"] = np.nan
                    domain_result[f"{model_name}_mae"] = np.nan
                    domain_result[f"{model_name}_r2"] = np.nan
                    if error_col in domain_df.columns:
                        domain_result[f"{model_name}_mean_error"] = np.nan
                        domain_result[f"{model_name}_std_error"] = np.nan
                    continue

                # Drop rows where either actual or prediction is NaN for metric calculation
                valid_rows = domain_df[[target_col, pred_col]].dropna()
                if valid_rows.empty:
                     logger.debug(f"Skipping metrics for model {model_name} in domain {domain_id} (no valid rows).")
                     rmse, mae, r2 = np.nan, np.nan, np.nan
                else:
                    actual = valid_rows[target_col].values
                    predicted = valid_rows[pred_col].values
                    try:
                         rmse = np.sqrt(mean_squared_error(actual, predicted))
                         mae = mean_absolute_error(actual, predicted)
                         # R2 requires variance in actual values
                         r2 = r2_score(actual, predicted) if np.var(actual) > 1e-9 else np.nan
                    except ValueError:
                         rmse, mae, r2 = np.nan, np.nan, np.nan


                # Store metrics
                domain_result[f"{model_name}_rmse"] = rmse
                domain_result[f"{model_name}_mae"] = mae
                domain_result[f"{model_name}_r2"] = r2

                # Basic error stats if error columns exist
                if error_col in domain_df.columns:
                     # Calculate mean/std only on non-NaN errors
                     valid_errors = domain_df[error_col].dropna()
                     domain_result[f"{model_name}_mean_error"] = valid_errors.mean() if not valid_errors.empty else np.nan
                     domain_result[f"{model_name}_std_error"] = valid_errors.std() if len(valid_errors) > 1 else 0.0


            domain_metrics.append(domain_result)

        # Save domain metrics to CSV
        if domain_metrics:
            domain_metrics_df = pd.DataFrame(domain_metrics)
            try:
                ensure_dir(os.path.dirname(output_path))
                domain_metrics_df.to_csv(output_path, index=False)
                logger.info(f"Saved domain-level metrics to: {output_path}")
            except Exception as e:
                 logger.error(f"Failed to save domain metrics: {e}")
        else:
             logger.warning("No domain metrics were calculated.")

    def predict(
        self,
        data: Union[str, pd.DataFrame],
        temperature: float,  # REQUIRED prediction temperature
        model_name: Optional[str] = None,
        with_uncertainty: bool = False,
        batch_size: int = 10000,  # Added batch processing
        check_temp_range: bool = True  # Added temperature validation
    ) -> Tuple[pd.DataFrame, Optional[Dict[str, float]]]:
        """
        Enhanced version: Generate predictions for new data at a specific prediction temperature.
        
        Improvements:
        - Batch processing for large datasets
        - Temperature range validation
        - Better error handling
        - Progress tracking for large datasets

        Args:
            data: DataFrame or path to CSV file with protein data features.
                May optionally contain 'rmsf' and 'temperature' columns.
            temperature: The target temperature (K) for which to predict.
            model_name: Specific model to use (if None, finds best/first enabled).
            with_uncertainty: Whether to include uncertainty estimates.
            batch_size: Number of rows to process at once for large datasets.
            check_temp_range: Whether to validate if temperature is in training range.

        Returns:
            Tuple containing:
            - DataFrame with identifiers, prediction temp, prediction, original temp (if present),
            true target (if present), errors (if present), uncertainty (if requested).
            - Dictionary of evaluation metrics calculated ONLY on the subset where
            original input temperature == prediction temperature (if possible), else None.
        """
        logger.info(f"Starting prediction for prediction_temperature: {temperature} K")
        target_col_name = self.config["dataset"]["target"]  # 'rmsf'
        original_temp_col_name = 'temperature'  # Standard name for original temp feature
        pred_col_name = f"{target_col_name}_predicted"
        unc_col_name = f"{target_col_name}_uncertainty"
        error_col_name = f"{target_col_name}_error"
        abs_error_col_name = f"{target_col_name}_abs_error"

        # --- 1. Load and Process Input Data ---
        input_df_contains_target = False
        input_df_contains_orig_temp = False
        input_df_raw: Optional[pd.DataFrame] = None

        if isinstance(data, str):
            logger.debug(f"Loading prediction input data from path: {data}")
            try:
                input_df_raw = load_file(data)  # Load raw first
                if target_col_name in input_df_raw.columns: input_df_contains_target = True
                if original_temp_col_name in input_df_raw.columns: input_df_contains_orig_temp = True
                logger.info(f"Input file contains target: {input_df_contains_target}, original temperature: {input_df_contains_orig_temp}.")
                
                # Check dataset size for potential batching
                large_dataset = len(input_df_raw) > batch_size
                if large_dataset:
                    logger.info(f"Large dataset detected: {len(input_df_raw)} rows. Will process in batches of {batch_size}.")
                
                df_processed = load_and_process_data(input_df_raw, self.config)
            except Exception as e:
                logger.error(f"Error loading/processing input file {data}: {e}", exc_info=True)
                raise ValueError(f"Failed to load/process input file {data}") from e
        elif isinstance(data, pd.DataFrame):
            logger.debug("Processing provided DataFrame for prediction...")
            if target_col_name in data.columns: input_df_contains_target = True
            if original_temp_col_name in data.columns: input_df_contains_orig_temp = True
            logger.info(f"Input DataFrame contains target: {input_df_contains_target}, original temperature: {input_df_contains_orig_temp}.")
            input_df_raw = data.copy()
            df_processed = process_features(input_df_raw.copy(), self.config)
        else:
            raise TypeError("Input 'data' must be a file path (str) or a pandas DataFrame.")

        if not isinstance(input_df_raw, pd.DataFrame):
            logger.error("Failed to obtain a valid DataFrame from input.")
            raise ValueError("Could not process input data into a DataFrame.")

        # --- 2. Determine and Load Model ---
        if model_name is None:
            try:  # Find best model logic...
                results_path = os.path.join(self.output_dir, "evaluation_results.csv")
                if os.path.exists(results_path):
                    results_df = pd.read_csv(results_path, index_col="model")
                    if not results_df.empty:
                        if "r2" in results_df.columns and results_df["r2"].notna().any(): model_name = results_df["r2"].idxmax()
                        elif "rmse" in results_df.columns and results_df["rmse"].notna().any(): model_name = results_df["rmse"].idxmin()
                        else: model_name = results_df.index[0]
                        logger.info(f"Using best model: '{model_name}'")
                    else: raise ValueError("Eval file empty.")
                else: raise FileNotFoundError
            except (FileNotFoundError, IndexError, ValueError, Exception) as e:
                logger.warning(f"Could not determine best model ({e}). Falling back.")
                enabled_models = get_enabled_models(self.config)
                if not enabled_models: raise RuntimeError("No model specified/enabled.")
                model_name = enabled_models[0]
                logger.info(f"Using first enabled: '{model_name}'")

        logger.info(f"Loading model '{model_name}' for prediction.")
        model = self.models.get(model_name) or self.load_model(model_name)
        self.models[model_name] = model

        # --- 3. Temperature Range Validation (New) ---
        if check_temp_range and input_df_contains_orig_temp:
            # Get temperature range from training data
            available_temps = input_df_raw[original_temp_col_name].dropna().unique()
            if len(available_temps) > 0:
                min_temp, max_temp = np.min(available_temps), np.max(available_temps)
                # Add a small buffer (5% of range)
                temp_range = max_temp - min_temp
                buffer = temp_range * 0.05 if temp_range > 0 else 10.0
                
                if temperature < min_temp - buffer or temperature > max_temp + buffer:
                    logger.warning(f"Requested temperature {temperature}K is outside training range "
                                f"({min_temp:.1f}K - {max_temp:.1f}K ± {buffer:.1f}K buffer). "
                                f"Predictions may be less reliable.")

        # --- 4. Prepare Feature Matrix & Augment Temperature ---
        logger.debug("Preparing feature matrix for prediction...")
        X_input, _, feature_names = prepare_data_for_model(
            df_processed, self.config, include_target=False
        )
        temp_feature_enabled = self.config['dataset']['features']['use_features'].get('temperature', False)
        
        if temp_feature_enabled:
            try:
                temp_feature_index = feature_names.index(original_temp_col_name)  # Use standard temp col name
                logger.debug(f"Found temperature feature at index {temp_feature_index}")
            except ValueError:
                logger.error(f"'{original_temp_col_name}' required but missing from prepared features!")
                raise ValueError(f"Feature mismatch: '{original_temp_col_name}' missing.")
        else: 
            logger.warning("Predicting, but model not trained with temperature feature.")
            temp_feature_index = None
            
        # --- 5. Generate Predictions (with batching) ---
        logger.info(f"Generating predictions with model '{model_name}'...")
        
        # Determine if batching is needed
        large_dataset = len(X_input) > batch_size
        
        predictions_list = []
        uncertainties_list = []
        
        if large_dataset:
            # Process in batches
            num_batches = int(np.ceil(len(X_input) / batch_size))
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_input))
                logger.info(f"Processing batch {i+1}/{num_batches} (rows {start_idx}-{end_idx})")
                
                # Get batch and set temperature
                X_batch = X_input[start_idx:end_idx].copy()
                
                # Set temperature for this batch
                if temp_feature_enabled and temp_feature_index is not None:
                    X_batch[:, temp_feature_index] = temperature
                    
                # Generate predictions for batch
                try:
                    if with_uncertainty and hasattr(model, 'predict_with_std') and callable(model.predict_with_std):
                        batch_preds, batch_uncs = model.predict_with_std(X_batch)
                        predictions_list.append(batch_preds)
                        uncertainties_list.append(batch_uncs)
                    else:
                        if with_uncertainty: 
                            logger.warning(f"Uncertainty requested, but not supported.")
                        batch_preds = model.predict(X_batch)
                        predictions_list.append(batch_preds)
                except Exception as e:
                    logger.error(f"Batch prediction failed: {e}", exc_info=True)
                    # Continue with next batch instead of failing completely
                    predictions_list.append(np.full(end_idx - start_idx, np.nan))
                    if with_uncertainty:
                        uncertainties_list.append(np.full(end_idx - start_idx, np.nan))
        else:
            # Process entire dataset at once
            X_augmented = X_input.copy()
            if temp_feature_enabled and temp_feature_index is not None:
                X_augmented[:, temp_feature_index] = temperature
                
            try:
                if with_uncertainty and hasattr(model, 'predict_with_std') and callable(model.predict_with_std):
                    full_preds, full_uncs = model.predict_with_std(X_augmented)
                    predictions_list.append(full_preds)
                    uncertainties_list.append(full_uncs)
                else:
                    if with_uncertainty: 
                        logger.warning(f"Uncertainty requested, but not supported.")
                    full_preds = model.predict(X_augmented)
                    predictions_list.append(full_preds)
            except Exception as e:
                logger.error(f"Prediction generation failed: {e}", exc_info=True)
                raise

        # Combine predictions from batches
        if predictions_list:
            predictions_array = np.concatenate(predictions_list) if len(predictions_list) > 1 else predictions_list[0]
            uncertainties_array = None
            if uncertainties_list:
                uncertainties_array = np.concatenate(uncertainties_list) if len(uncertainties_list) > 1 else uncertainties_list[0]
        else:
            raise RuntimeError("No predictions were generated")

        # --- 6. Construct Result DataFrame ---
        logger.debug("Constructing result DataFrame...")
        id_cols = ['domain_id', 'resid', 'resname']
        cols_to_copy = id_cols[:]
        if input_df_contains_orig_temp: cols_to_copy.append(original_temp_col_name)
        if input_df_contains_target: cols_to_copy.append(target_col_name)

        result_df_base = pd.DataFrame(index=range(len(predictions_array)))
        if all(c in input_df_raw.columns for c in id_cols):
            temp_ids_df = input_df_raw[id_cols].copy().reset_index(drop=True)
            if len(temp_ids_df) == len(predictions_array):
                result_df_base = temp_ids_df
            else: logger.warning("ID length mismatch.")
        else: logger.warning("Input missing IDs.")

        # Add optional columns if they exist in raw input and length matches
        for col in cols_to_copy:
            if col not in id_cols and col in input_df_raw.columns and len(input_df_raw) == len(predictions_array):
                result_df_base[col] = input_df_raw[col].values
            elif col not in id_cols and col not in result_df_base.columns: # Add as NaN if missing entirely
                result_df_base[col] = np.nan

        # Add prediction temperature and predictions
        result_df_base['prediction_temperature'] = temperature
        result_df_base[pred_col_name] = predictions_array
        if uncertainties_array is not None: result_df_base[unc_col_name] = uncertainties_array

        # Calculate errors *if* target was present
        if input_df_contains_target:
            result_df_base[error_col_name] = result_df_base[pred_col_name] - result_df_base[target_col_name]
            result_df_base[abs_error_col_name] = np.abs(result_df_base[error_col_name])

        # --- Calculate Metrics ONLY on Matching Temperatures ---
        metrics = None
        if input_df_contains_target and input_df_contains_orig_temp:
            logger.info(f"Calculating metrics comparing predictions @{temperature}K vs true values @{temperature}K...")
            # Filter the results to rows where original temp == prediction temp
            # Use np.isclose for robust float comparison
            matching_temp_df = result_df_base[np.isclose(result_df_base[original_temp_col_name], temperature)]

            if not matching_temp_df.empty:
                # Drop NaNs from target and prediction within this subset
                eval_subset = matching_temp_df[[target_col_name, pred_col_name]].dropna()
                if not eval_subset.empty and len(eval_subset) > 1:
                    y_true_eval = eval_subset[target_col_name].values
                    y_pred_eval = eval_subset[pred_col_name].values
                    # Need to prepare X subset for metrics requiring features (e.g., Adj R2)
                    # This requires re-preparing features for the matching_temp_df indices
                    # For simplicity now, we won't pass X to evaluate_predictions here
                    # If Adj R2 is critical, this needs refinement
                    n_feat_for_metrics = len(feature_names) # Number of features model was trained on
                    metrics = evaluate_predictions(y_true_eval, y_pred_eval, self.config, n_features=n_feat_for_metrics)
                    logger.info(f"Prediction Metrics (for T={temperature}K only): {metrics}")
                else:
                    logger.warning(f"Not enough valid data points found where original temperature == {temperature}K for metric calculation.")
            else:
                logger.warning(f"No rows found in input where original temperature == {temperature}K. Cannot calculate specific metrics.")
        # --- End Metrics Calculation ---

        # Reorder columns for clarity
        final_cols_order = ['domain_id', 'resid', 'resname']
        if original_temp_col_name in result_df_base.columns: final_cols_order.append(original_temp_col_name)
        if target_col_name in result_df_base.columns: final_cols_order.append(target_col_name)
        final_cols_order.append('prediction_temperature')
        final_cols_order.append(pred_col_name)
        if unc_col_name in result_df_base.columns: final_cols_order.append(unc_col_name)
        if error_col_name in result_df_base.columns: final_cols_order.append(error_col_name)
        if abs_error_col_name in result_df_base.columns: final_cols_order.append(abs_error_col_name)
        final_cols_order = [col for col in final_cols_order if col in result_df_base.columns]
        result_df_final = result_df_base[final_cols_order]

        logger.info(f"Prediction completed successfully. Generated {len(result_df_final)} predictions.")
        return result_df_final, metrics

    def analyze(
        self,
        model_name: Optional[str] = None,
        results_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Perform analysis of the single trained model's results.
        Focuses on feature importance (including temperature) and error analysis
        as a function of the input temperature feature.

        Args:
            model_name: Name of the model to analyze (if None, finds first enabled).
            results_df: Optional DataFrame containing detailed evaluation results
                        (e.g., from `all_results.csv`). If None, it will be loaded.
        """
        logger.info("--- Starting Model Analysis ---")

        # 1. Determine Model Name & Load Model/Results
        # (This part remains the same as the previous corrected version)
        model: Optional[BaseModel] = None
        try:
            if not model_name:
                enabled_models = get_enabled_models(self.config)
                if not enabled_models: raise ValueError("No models enabled.")
                model_name = enabled_models[0]; logger.info(f"Analyzing first enabled: {model_name}")
            else: logger.info(f"Analyzing model: {model_name}")

            model = self.models.get(model_name) or self.load_model(model_name)
            self.models[model_name] = model # Cache

            if results_df is None:
                results_path = os.path.join(self.output_dir, "all_results.csv")
                if not os.path.exists(results_path): raise FileNotFoundError(f"Results missing: {results_path}")
                logger.info(f"Loading results: {results_path}")
                results_df = pd.read_csv(results_path)

            target_col=self.config['dataset']['target']; pred_col=f"{model_name}_predicted"
            if not all(c in results_df.columns for c in ['temperature',target_col,pred_col]):
                raise ValueError(f"Results missing essential columns.")
            abs_error_col = f"{model_name}_abs_error"
            if abs_error_col not in results_df.columns:
                results_df.loc[:, abs_error_col] = (results_df[pred_col] - results_df[target_col]).abs()
        except (FileNotFoundError, ValueError, Exception) as e:
            logger.error(f"Analysis setup failed: {e}", exc_info=True); return

        # --- 2. Feature Importance Analysis ---
        analysis_cfg = self.config.get("analysis", {})
        importance_cfg = analysis_cfg.get("feature_importance", {})
        if importance_cfg.get("enabled", False):
            logger.info("Calculating feature importance...")
            X_eval, y_eval, feature_names = None, None, None
            importance_dict = None # Initialize importance_dict
            try:
                # Prepare data
                use_val_data = importance_cfg.get("use_validation_data", True)
                eval_set_name = "validation" if use_val_data else "test"
                split_set_index = 1 if use_val_data else 2
                logger.debug(f"Preparing {eval_set_name} data for importance...")
                full_df = self.load_data(); splits = split_data(full_df, self.config)
                eval_df_for_importance = splits[split_set_index]

                if not eval_df_for_importance.empty:
                    # *** Get feature names during data preparation ***
                    X_eval, y_eval, feature_names = prepare_data_for_model(eval_df_for_importance, self.config)
                else: logger.warning(f"{eval_set_name} set empty.")

                if X_eval is not None and y_eval is not None and feature_names:
                    importance_method_cfg = importance_cfg.get("method", "permutation")
                    n_repeats_cfg = importance_cfg.get("n_repeats", 10)
                    importance_values = None

                    # Call get_feature_importance correctly
                    logger.debug(f"Calling importance (method='{importance_method_cfg}') for {type(model).__name__}")
                    if not hasattr(model, 'get_feature_importance'):
                         logger.warning(f"Model missing get_feature_importance.")
                    else:
                         sig = inspect.signature(model.get_feature_importance)
                         params = sig.parameters; call_kwargs = {}
                         # Pass arguments accepted by the specific model's method
                         if 'X_val' in params: call_kwargs['X_val'] = X_eval
                         if 'y_val' in params: call_kwargs['y_val'] = y_eval
                         if 'method' in params: call_kwargs['method'] = importance_method_cfg
                         if 'n_repeats' in params: call_kwargs['n_repeats'] = n_repeats_cfg
                         try: importance_values = model.get_feature_importance(**call_kwargs)
                         except Exception as fe_e: logger.error(f"Call failed: {fe_e}", exc_info=True)

                    # *** Process results AND map to feature names ***
                    if importance_values is not None:
                        # If it's an array, map it using the feature_names list
                        if isinstance(importance_values, np.ndarray):
                             if len(importance_values) == len(feature_names):
                                 importance_dict = dict(zip(feature_names, importance_values.astype(float)))
                                 logger.debug("Mapped importance array to feature names.")
                             else:
                                 logger.warning(f"Importance array length ({len(importance_values)}) != feature names length ({len(feature_names)}). Cannot map names.")
                        elif isinstance(importance_values, dict):
                             importance_dict = importance_values # Assume already mapped
                             logger.debug("Importance already returned as dictionary.")
                        else: logger.warning("Importance format unexpected.")

                        # Plotting/Saving uses the dictionary with correct names
                        if importance_dict:
                            importance_dir = os.path.join(self.output_dir, "feature_importance"); ensure_dir(importance_dir)
                            plot_path = os.path.join(importance_dir, f"{model_name}_feature_importance.png")
                            csv_path = os.path.join(importance_dir, f"{model_name}_feature_importance.csv")
                            # plot_feature_importance now receives the dict with correct names
                            plot_feature_importance(importance_dict, plot_path, csv_path)
                            if 'temperature' in importance_dict: logger.info(f"Importance of 'temperature': {importance_dict['temperature']:.4f}")
                        else: logger.warning(f"Failed to create importance dictionary for {model_name}.")
                    else: logger.warning(f"Could not calculate feature importance for {model_name}.")
                else: logger.warning("Data preparation failed for importance.")
            except Exception as e: logger.error(f"Error during feature importance step: {e}", exc_info=True)
        else: logger.info("Feature importance analysis skipped (config).")

        # --- 3. Error Analysis vs. Temperature ---
        # (This part remains the same as previous correction)
        logger.info("Analyzing performance vs. input temperature feature...")
        temp_analysis_dir = os.path.join(self.output_dir, "temperature_analysis"); ensure_dir(temp_analysis_dir)
        pred_vs_temp_path = os.path.join(temp_analysis_dir, f"{model_name}_prediction_vs_temp.png")
        error_vs_temp_plot_path = os.path.join(temp_analysis_dir, f"{model_name}_error_vs_temp.png")
        error_vs_temp_csv_path = os.path.join(temp_analysis_dir, f"{model_name}_error_vs_temp_binned.csv")
        try: plot_prediction_vs_temperature(results_df, model_name, pred_vs_temp_path)
        except Exception as e: logger.error(f"Pred vs Temp plot failed: {e}", exc_info=True)
        try: plot_error_vs_temperature(results_df, model_name, self.config, error_vs_temp_plot_path, error_vs_temp_csv_path)
        except Exception as e: logger.error(f"Error vs Temp plot failed: {e}", exc_info=True)

        # --- 4. Standard Residue/Property Analysis ---
        # (This part remains the same)
        logger.info("Performing standard residue/property error analysis...")
        residue_analysis_dir = os.path.join(self.output_dir, "residue_analysis"); ensure_dir(residue_analysis_dir)
        aa_error_csv_path = os.path.join(residue_analysis_dir, f"{model_name}_amino_acid_errors.csv")
        aa_error_plot_path = os.path.join(residue_analysis_dir, f"{model_name}_amino_acid_errors.png")
        prop_analysis_base_path = os.path.join(residue_analysis_dir, f"{model_name}_error_by")
        try: plot_amino_acid_error_analysis(results_df, model_name, target_col, aa_error_csv_path, aa_error_plot_path)
        except Exception as e: logger.error(f"AA error analysis failed: {e}", exc_info=True)
        try: plot_error_analysis_by_property(results_df, model_name, target_col, prop_analysis_base_path)
        except Exception as e: logger.error(f"Prop error analysis failed: {e}", exc_info=True)

        # --- 5. Scatter Plot ---
        # (This part remains the same)
        scatter_dir = os.path.join(self.output_dir, "comparisons"); ensure_dir(scatter_dir)
        scatter_plot_path = os.path.join(scatter_dir, f"{model_name}_actual_vs_predicted_scatter.png")
        scatter_csv_path = os.path.join(scatter_dir, f"{model_name}_actual_vs_predicted_data.csv")
        try: plot_scatter_with_density_contours(results_df, model_name, target_col, scatter_plot_path, scatter_csv_path)
        except Exception as e: logger.error(f"Scatter plot failed: {e}", exc_info=True)

        logger.info("--- Model Analysis Complete ---")


    def run_pipeline(
        self,
        model_names: Optional[List[str]] = None,
        data_path: Optional[str] = None,
        skip_analysis: bool = False # Flag name matches CLI
    ) -> Dict[str, Dict[str, float]]:
        """
        Run the complete pipeline: train, evaluate, and analyze the single model.

        Args:
            model_names: List containing the single model name to use.
            data_path: Optional explicit path to the aggregated data file.
            skip_analysis: Whether to skip the analysis step.

        Returns:
            Dictionary of evaluation metrics for the trained model, or empty if failed.
        """
        final_results = {}
        model_to_run = None
        train_successful = False
        eval_successful = False

        try:
            # --- Train ---
            logger.info("--- Starting Training Phase ---")
            trained_models_dict = self.train(model_names, data_path)
            if trained_models_dict:
                model_to_run = list(trained_models_dict.keys())[0]
                train_successful = True
                logger.info(f"--- Training Phase Complete for {model_to_run} ---")
            else:
                logger.error("Training phase failed or returned no models. Aborting pipeline.")
                return {}

            # --- Evaluate ---
            logger.info("--- Starting Evaluation Phase ---")
            eval_results = self.evaluate(model_names=[model_to_run], data_path=data_path)
            if eval_results and model_to_run in eval_results:
                final_results = eval_results
                eval_successful = True
                logger.info(f"--- Evaluation Phase Complete for {model_to_run} ---")
            else:
                logger.error(f"Evaluation phase failed to produce results for model {model_to_run}.")
                # Decide if pipeline should continue to analysis despite failed eval
                # For now, we will stop analysis if eval fails

            # --- Analyze ---
            if not skip_analysis and eval_successful:
                logger.info("--- Starting Analysis Phase ---")
                # Load the detailed results saved by evaluate step
                results_df_path = os.path.join(self.output_dir, "all_results.csv")
                results_df_for_analysis = None
                if os.path.exists(results_df_path):
                    try:
                        results_df_for_analysis = pd.read_csv(results_df_path)
                        logger.debug("Loaded detailed results file for analysis.")
                    except Exception as e:
                        logger.error(f"Failed to load detailed results file '{results_df_path}' for analysis: {e}")

                if results_df_for_analysis is not None:
                    self.analyze(model_name=model_to_run, results_df=results_df_for_analysis)
                    logger.info("--- Analysis Phase Complete ---")
                else:
                    logger.error("Skipping analysis because detailed results file could not be loaded.")
            elif skip_analysis:
                logger.info("--- Analysis Phase Skipped ---")
            else: # Analysis not skipped, but eval failed
                 logger.warning("--- Analysis Phase Skipped due to evaluation failure ---")


        except Exception as pipeline_error:
             logger.error(f"Pipeline execution failed: {pipeline_error}", exc_info=True)
             return final_results # Return whatever eval results were obtained before the crash

        logger.info("--- Pipeline Run Finished ---")
        return final_results