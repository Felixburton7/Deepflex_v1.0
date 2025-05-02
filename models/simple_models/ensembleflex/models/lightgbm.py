# /home/s_felix/ensembleflex/ensembleflex/models/lightgbm_model.py

"""
LightGBM model implementation for the EnsembleFlex ML pipeline.
"""

import os
import logging
import time
import inspect # To check function signatures
from typing import Dict, Any, Optional, Union, List, Tuple

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb # Import LightGBM

from ensembleflex.models import register_model
from ensembleflex.models.base import BaseModel
from ensembleflex.utils.helpers import ensure_dir, ProgressCallback # Keep ProgressCallback for timing

logger = logging.getLogger(__name__)

@register_model("lightgbm") # Register model with the name 'lightgbm'
class LightGBMModel(BaseModel):
    """
    LightGBM Gradient Boosting Regressor model for protein flexibility prediction.

    Utilizes the LightGBM library for efficient gradient boosting. Supports
    early stopping during training if validation data is provided.
    """

    def __init__(
        self,
        objective: str = 'regression_l1', # MAE is often robust for RMSF
        metric: str = 'mae', # Evaluate using MAE
        n_estimators: int = 1000, # Can be large if using early stopping
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        reg_alpha: float = 0.1, # L1 regularization
        reg_lambda: float = 0.1, # L2 regularization
        colsample_bytree: float = 0.8,
        subsample: float = 0.8,
        n_jobs: int = -1, # Use all available cores
        random_state: int = 42,
        early_stopping: Optional[Dict[str, Any]] = None, # Config for early stopping
        **kwargs # Capture any other valid LGBMRegressor parameters from config
    ):
        """
        Initialize the LightGBM model.

        Args:
            objective: Learning task objective (e.g., 'regression', 'regression_l1').
            metric: Metric(s) to evaluate during training (e.g., 'rmse', 'mae').
            n_estimators: Maximum number of boosting rounds.
            learning_rate: Boosting learning rate.
            num_leaves: Maximum tree leaves for base learners.
            max_depth: Maximum tree depth for base learners, -1 means no limit.
            reg_alpha: L1 regularization term on weights.
            reg_lambda: L2 regularization term on weights.
            colsample_bytree: Subsample ratio of columns when constructing each tree.
            subsample: Subsample ratio of the training instance.
            n_jobs: Number of parallel threads.
            random_state: Random number seed.
            early_stopping: Dictionary with early stopping config (e.g., {'enabled': True, 'stopping_rounds': 50, 'verbose': False}).
            **kwargs: Additional parameters passed directly to lgb.LGBMRegressor.
        """
        super().__init__() # Initialize base class if needed
        self.objective = objective
        self.metric = metric
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.early_stopping_config = early_stopping if early_stopping else {}

        # Filter kwargs to pass only valid ones to LGBMRegressor
        lgbm_params = set(inspect.signature(lgb.LGBMRegressor).parameters.keys())
        self.model_params = {k: v for k, v in kwargs.items() if k in lgbm_params}

        self.model: Optional[lgb.LGBMRegressor] = None
        self.feature_names_: Optional[List[str]] = None
        self.best_iteration_: Optional[int] = None

    def fit(self,
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series],
            X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None, # For early stopping
            y_val: Optional[Union[np.ndarray, pd.Series]] = None, # For early stopping
            feature_names: Optional[List[str]] = None) -> 'LightGBMModel':
        """
        Train the LightGBM model.

        Args:
            X: Feature matrix for training.
            y: Target values for training.
            X_val: Optional feature matrix for validation (early stopping).
            y_val: Optional target values for validation (early stopping).
            feature_names: Optional list of feature names.

        Returns:
            Self (the trained model instance).
        """
        start_time = time.time()
        logger.info(f"Starting LightGBM training with {self.n_estimators} max estimators...")

        # Store feature names if provided
        if feature_names is not None:
            self.feature_names_ = feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        else:
            # Create generic names if X is numpy and names not provided
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]

        # Ensure y is a 1D numpy array
        y_train = np.ravel(y.values if isinstance(y, pd.Series) else y)
        if X_val is not None and y_val is not None:
            y_val_train = np.ravel(y_val.values if isinstance(y_val, pd.Series) else y_val)

        # Prepare arguments for LightGBM fit method
        fit_params = {}
        callbacks = []

        # Configure Early Stopping
        use_early_stopping = self.early_stopping_config.get('enabled', False) and \
                             X_val is not None and y_val is not None
        if use_early_stopping:
            stopping_rounds = self.early_stopping_config.get('stopping_rounds', 50)
            verbose_early_stopping = self.early_stopping_config.get('verbose', False) # LGBM's verbose logging for rounds
            fit_params['eval_set'] = [(X_val, y_val_train)]
            fit_params['eval_metric'] = self.metric # Use the configured metric for stopping
            # Use lgb.early_stopping callback
            callbacks.append(lgb.early_stopping(stopping_rounds=stopping_rounds,
                                                verbose=verbose_early_stopping))
            logger.info(f"Early stopping enabled with stopping_rounds={stopping_rounds} using '{self.metric}' metric.")
        else:
            logger.info("Early stopping not enabled or validation set not provided.")

        # Add LightGBM's own progress callback if desired (can be noisy)
        # callbacks.append(lgb.log_evaluation(period=100)) # Log eval results every 100 rounds


        # Instantiate the model
        self.model = lgb.LGBMRegressor(
            objective=self.objective,
            metric=self.metric,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            colsample_bytree=self.colsample_bytree,
            subsample=self.subsample,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            **self.model_params # Pass other valid params from config
        )

        # Use ProgressCallback for overall timing info
        with ProgressCallback(total=1, desc="Training LightGBM") as pbar:
            try:
                 self.model.fit(X, y_train,
                                feature_name=self.feature_names_ if self.feature_names_ else 'auto', # Pass feature names if available
                                **fit_params, # Includes eval_set, eval_metric if early stopping enabled
                                callbacks=callbacks if callbacks else None)
                 pbar.update()
                 if use_early_stopping and self.model.best_iteration_:
                      self.best_iteration_ = self.model.best_iteration_
                      logger.info(f"Early stopping utilized. Best iteration: {self.best_iteration_}")

            except Exception as e:
                logger.error(f"LightGBM training failed: {e}", exc_info=True)
                raise # Re-raise the exception

        end_time = time.time()
        logger.info(f"LightGBM training finished in {end_time - start_time:.2f} seconds.")
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Generate RMSF predictions.

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted RMSF values.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction.")
        # Pass best_iteration if early stopping was used during fit
        num_iteration = self.best_iteration_ if self.best_iteration_ else 0 # 0 means use all trees
        return self.model.predict(X, num_iteration=num_iteration)

    def predict_with_std(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty estimates.
        NOTE: Standard LightGBM doesn't provide simple uncertainty like RF/MC Dropout.
              This method returns the mean prediction and zero standard deviation.

        Args:
            X: Feature matrix.

        Returns:
            Tuple of (predictions, std_deviation=0).
        """
        logger.warning("Standard uncertainty estimation (std dev) not directly available for LightGBM. Returning zero uncertainty.")
        predictions = self.predict(X)
        std_deviation = np.zeros_like(predictions)
        return predictions, std_deviation

    def save(self, path: str) -> None:
        """
        Save the trained LightGBM model state using joblib.

        Args:
            path: Path to save the model file (e.g., model.pkl).
        """
        if self.model is None:
            raise RuntimeError("Cannot save untrained model.")

        ensure_dir(os.path.dirname(path))
        logger.info(f"Saving LightGBM model state to: {path}")

        # Save the relevant state for reloading
        state = {
            'model': self.model, # Save the trained LGBMRegressor object
            'feature_names': self.feature_names_,
            'best_iteration': self.best_iteration_,
            # Optionally save init parameters if needed for re-instantiation flexibility
            'init_params': {
                'objective': self.objective, 'metric': self.metric, 'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate, 'num_leaves': self.num_leaves, 'max_depth': self.max_depth,
                'reg_alpha': self.reg_alpha, 'reg_lambda': self.reg_lambda, 'colsample_bytree': self.colsample_bytree,
                'subsample': self.subsample, 'n_jobs': self.n_jobs, 'random_state': self.random_state,
                'early_stopping': self.early_stopping_config, **self.model_params
            }
        }
        try:
            joblib.dump(state, path)
            logger.info("Model saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save model state with joblib: {e}", exc_info=True)
            # Fallback: Try saving only the booster
            booster_path = os.path.splitext(path)[0] + ".lgb"
            try:
                self.model.booster_.save_model(booster_path)
                logger.warning(f"Saved only the LightGBM booster to {booster_path} as joblib failed.")
            except Exception as booster_e:
                logger.error(f"Failed to save booster as well: {booster_e}")

    @classmethod
    def load(cls, path: str) -> 'LightGBMModel':
        """
        Load a trained LightGBM model state from disk using joblib.

        Args:
            path: Path to the saved model file (.pkl).

        Returns:
            Loaded LightGBMModel instance.
        """
        if not os.path.exists(path):
            # Check for fallback booster file if joblib file missing
            booster_path = os.path.splitext(path)[0] + ".lgb"
            if os.path.exists(booster_path):
                logger.warning(f"Joblib file {path} not found, attempting to load booster from {booster_path}.")
                try:
                     booster = lgb.Booster(model_file=booster_path)
                     # Need to re-instantiate LGBMRegressor and set booster_
                     # This requires knowing the original parameters. Cannot fully restore class state this way.
                     # Return a basic model object for prediction only?
                     logger.error("Loading from booster file only is not fully supported for restoring class state. Cannot proceed.")
                     raise FileNotFoundError(f"Cannot fully load model state, only booster found at {booster_path}")
                except Exception as e:
                     logger.error(f"Failed to load booster file {booster_path}: {e}")
                     raise FileNotFoundError(f"Model file not found at {path} or {booster_path}")
            else:
                 raise FileNotFoundError(f"Model file not found: {path}")

        logger.info(f"Loading LightGBM model state from: {path}")
        try:
            state = joblib.load(path)
            if not isinstance(state, dict) or 'model' not in state:
                 raise ValueError("Loaded state is not a valid dictionary or missing 'model' key.")

            # Re-create instance and restore state
            # Option 1: Re-instantiate with saved params (more robust if class changes)
            init_params = state.get('init_params', {}) # Get saved init params
            instance = cls(**init_params) # Re-create with original settings

            # Option 2: Just restore attributes to a blank instance (simpler if class hasn't changed much)
            # instance = cls()

            instance.model = state['model']
            instance.feature_names_ = state.get('feature_names')
            instance.best_iteration_ = state.get('best_iteration')
            # Restore other necessary attributes if using Option 2

            logger.info("LightGBM model loaded successfully.")
            return instance
        except Exception as e:
            logger.error(f"Error loading LightGBM model state from {path}: {e}", exc_info=True)
            raise ValueError(f"Failed to load model state from {path}.") from e

        # Place this corrected method inside the LightGBMModel class
    def get_feature_importance(self, X_val=None, y_val=None, method="built-in", n_repeats=None) -> Optional[Dict[str, float]]:
        """
        Get feature importance using LightGBM's native feature importance only.
        
        Args:
            X_val: Not used for built-in method
            y_val: Not used for built-in method
            method: Ignored - always uses built-in
            n_repeats: Not used for built-in method
            
        Returns:
            Dictionary mapping feature names to importance values
        """
        if self.model is None:
            return None
        
        # Only use LightGBM's native feature importance (fast)
        try:
            logger.info("Using LightGBM native feature importance (fast method)")
            importances = self.model.feature_importances_
            
            # Return dict with feature names if available
            if hasattr(self, 'feature_names_') and len(self.feature_names_) == len(importances):
                return dict(zip(self.feature_names_, importances))
            else:
                # Create generic feature names if needed
                generic_names = [f"feature_{i}" for i in range(len(importances))]
                return dict(zip(generic_names, importances))
                
        except Exception as e:
            logger.warning(f"Could not get LightGBM feature importance: {e}")
            return None