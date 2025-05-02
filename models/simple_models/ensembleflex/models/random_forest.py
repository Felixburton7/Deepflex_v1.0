


"""
Optimized Random Forest model for the EnsembleFlex ML pipeline.

This module provides a high-performance RandomForestModel for protein flexibility prediction,
focused on simplicity and performance with large datasets.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
import time
import gc

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import inspect

from ensembleflex.models import register_model
from ensembleflex.models.base import BaseModel
from ensembleflex.utils.helpers import ProgressCallback, ensure_dir

logger = logging.getLogger(__name__)

@register_model("random_forest")
class RandomForestModel(BaseModel):
    """
    Efficient Random Forest implementation for protein flexibility prediction.
    
    Optimized for large datasets with many features (like window-based encoding)
    while maintaining simplicity and compatibility with the ensembleflex pipeline.
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 50,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, float, int] = "sqrt",
        bootstrap: bool = True,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize the Random Forest model with configured parameters.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        # Handle n_jobs parameter (from kwargs or system settings)
        n_jobs = kwargs.pop('n_jobs', None)
        if n_jobs is None or n_jobs == -16:
            # Use 90% of cores for large datasets
            import multiprocessing
            total_cores = multiprocessing.cpu_count()
            self.n_jobs = max(1, int(total_cores * 0.3))
            logger.info(f"Using {self.n_jobs} cores for training (system has {total_cores})")
        else:
            self.n_jobs = n_jobs
        
        # Filter kwargs for valid RandomForestRegressor parameters
        rf_params = set(inspect.signature(RandomForestRegressor).parameters.keys())
        self.model_params = {k: v for k, v in kwargs.items() if k in rf_params}
        
        # Store hyperparameter search config separately
        self.randomized_search_config = kwargs.get('randomized_search', {})
        
        # Initialize model state
        self.model = None
        self.feature_names_ = None
        self.best_params_ = None
        self._feature_importances_cache = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], 
            feature_names: Optional[List[str]] = None,
            X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
            y_val: Optional[Union[np.ndarray, pd.Series]] = None) -> 'RandomForestModel':
        """
        Train the Random Forest model with optimized settings.
        
        Args:
            X: Feature matrix
            y: Target RMSF values
            feature_names: Optional list of feature names
            X_val: Optional validation features
            y_val: Optional validation targets
        """
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        elif feature_names is not None:
            self.feature_names_ = feature_names
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]

        # Convert to numpy arrays for better performance
        if isinstance(X, pd.DataFrame): 
            X = X.values
        if isinstance(y, pd.Series): 
            y = y.values
        
        # Ensure y is 1D
        y = np.ravel(y)
        
        # Log dataset properties
        logger.info(f"Training RandomForest on dataset with {X.shape[0]} samples and {X.shape[1]} features")
        
        # Check if using hyperparameter optimization
        use_randomized_search = self.randomized_search_config.get('enabled', False)
        
        # Measure total training time
        fit_start_time = time.time()
        
        try:
            if use_randomized_search:
                self._fit_with_hpo(X, y)
            else:
                self._fit_standard(X, y)
            
            # Log training time
            total_fit_time = time.time() - fit_start_time
            logger.info(f"RandomForest training completed in {total_fit_time:.2f} seconds")
            
            # Force garbage collection to free memory
            gc.collect()
            
            return self
        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            raise
            
    def _fit_standard(self, X, y):
        """Standard efficient training without hyperparameter optimization."""
        logger.info(f"Training Random Forest with {self.n_estimators} trees and {self.n_jobs} parallel jobs")
        
        # Create parameter dictionary for RandomForestRegressor
        rf_params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbose': 1,  # Show progress
            **self.model_params
        }
        
        # Create and train the model
        with ProgressCallback(total=1, desc="Training Random Forest") as pbar:
            self.model = RandomForestRegressor(**rf_params)
            self.model.fit(X, y)
            pbar.update()
        
        logger.info(f"Random Forest training complete with {self.n_estimators} trees")
    
    def _fit_with_hpo(self, X, y):
        """Train with hyperparameter optimization using RandomizedSearchCV."""
        logger.info("Running hyperparameter optimization with RandomizedSearchCV")
        
        # Get HPO settings from config
        search_params = self.randomized_search_config
        n_iter = search_params.get('n_iter', 10)
        cv = search_params.get('cv', 3)
        verbose = search_params.get('verbose', 1)
        param_distributions = search_params.get('param_distributions', {})
        
        if not isinstance(param_distributions, dict) or not param_distributions:
            logger.error("HPO enabled, but 'param_distributions' is missing or invalid")
            raise ValueError("Invalid 'param_distributions' for RandomizedSearchCV")
        
        # Configure the base estimator with minimum parallelization
        # since parallelization happens at the CV level
        base_rf = RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=1,  # Single job here, parallelism is at CV level
            **{k: v for k, v in self.model_params.items() 
               if k not in param_distributions}
        )
        
        logger.info(f"Setting up RandomizedSearchCV with n_iter={n_iter}, cv={cv}, n_jobs={self.n_jobs}")
        
        # Set up search with efficient parameters
        search = RandomizedSearchCV(
            estimator=base_rf,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=verbose,
            # pre_dispatch='2*n_jobs',  # Control worker dispatching
            return_train_score=False,
            error_score='raise'
        )
        
        try:
            # Fit the search
            logger.info("Starting RandomizedSearchCV fitting...")
            search.fit(X, y)
            
            # Store best model and parameters
            self.model = search.best_estimator_
            self.best_params_ = search.best_params_
            
            # Log results
            logger.info(f"Best score: {search.best_score_:.4f}")
            logger.info(f"Best parameters: {self.best_params_}")
            
            # Set optimal n_jobs for the final model
            if hasattr(self.model, 'n_jobs') and self.model.n_jobs == 1:
                self.model.n_jobs = self.n_jobs
            
        except Exception as e:
            logger.error(f"Error during HPO: {e}", exc_info=True)
            
            # Fallback to standard training
            logger.warning("Falling back to standard training after HPO failure")
            self._fit_standard(X, y)
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Generate predictions using the trained model."""
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
    
    def predict_with_std(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions with uncertainty estimates using tree variance."""
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        try:
            # Verify model has estimators
            if not hasattr(self.model, 'estimators_') or not self.model.estimators_:
                logger.warning("Model has no estimators. Cannot calculate uncertainty.")
                mean_prediction = self.model.predict(X)
                return mean_prediction, np.zeros_like(mean_prediction)
            
            # Get predictions from each tree
            all_preds = np.array([tree.predict(X) for tree in self.model.estimators_])
            
            # Calculate mean and standard deviation across trees
            mean_prediction = np.mean(all_preds, axis=0)
            std_prediction = np.std(all_preds, axis=0)
            
            return mean_prediction, std_prediction
        except Exception as e:
            logger.error(f"Error during uncertainty prediction: {e}. Falling back to zeros.")
            mean_prediction = self.model.predict(X)
            return mean_prediction, np.zeros_like(mean_prediction)
    
    def get_feature_importance(self, 
                              X_val=None, 
                              y_val=None, 
                              method="permutation", 
                              n_repeats=10) -> Optional[Union[Dict[str, float], np.ndarray]]:
        """Calculate feature importance using permutation or impurity methods."""
        if self.model is None:
            return None
        
        # Use cached result if available (and no new validation data provided)
        if self._feature_importances_cache is not None and X_val is None:
            return self._feature_importances_cache
        
        # Impurity-based importance (fast, no validation data needed)
        if method == "impurity" or X_val is None or y_val is None:
            try:
                logger.debug("Using impurity-based feature importance")
                importances = self.model.feature_importances_
                
                # Map to feature names if available
                if self.feature_names_ and len(self.feature_names_) == len(importances):
                    result = dict(zip(self.feature_names_, importances))
                    self._feature_importances_cache = result
                    return result
                else:
                    return importances
                    
            except Exception as e:
                logger.warning(f"Could not get impurity-based importance: {e}")
                return None
        
        # Permutation importance with validation data
        elif method == "permutation" and X_val is not None and y_val is not None:
            try:
                from sklearn.inspection import permutation_importance
                
                logger.debug(f"Calculating permutation importance with {n_repeats} repeats")
                
                # Convert validation data to numpy if needed
                if isinstance(X_val, pd.DataFrame):
                    X_val = X_val.values
                if isinstance(y_val, pd.Series):
                    y_val = y_val.values
                
                # Sample data if it's large to avoid memory issues
                if len(X_val) > 10000:
                    indices = np.random.choice(len(X_val), size=10000, replace=False)
                    X_val_sample = X_val[indices]
                    y_val_sample = y_val[indices]
                else:
                    X_val_sample = X_val
                    y_val_sample = y_val
                
                # Calculate permutation importance
                r = permutation_importance(
                    self.model, X_val_sample, y_val_sample,
                    n_repeats=n_repeats,
                    random_state=self.random_state,
                    n_jobs=min(4, self.n_jobs)  # Limit jobs to prevent memory issues
                )
                
                importances = r.importances_mean
                
                # Map to feature names if available
                if self.feature_names_ and len(self.feature_names_) == len(importances):
                    result = dict(zip(self.feature_names_, importances))
                    self._feature_importances_cache = result
                    return result
                else:
                    return importances
                    
            except Exception as e:
                logger.warning(f"Permutation importance failed: {e}. Falling back to impurity.")
                return self.get_feature_importance(method="impurity")
        
        logger.warning(f"Unsupported feature importance method: {method}")
        return None
    
    def hyperparameter_optimize(self, 
                               X, 
                               y, 
                               param_grid, 
                               method="random", 
                               n_trials=20, 
                               cv=3):
        """Perform hyperparameter optimization by calling fit with RandomizedSearchCV."""
        logger.info(f"Setting up hyperparameter optimization with {n_trials} trials")
        
        # Only RandomizedSearchCV is supported
        if method != "random":
            logger.warning(f"Method '{method}' not supported, using 'random'")
        
        # Configure randomized search
        self.randomized_search_config = {
            'enabled': True,
            'n_iter': n_trials,
            'cv': cv,
            'param_distributions': param_grid,
            'verbose': 1
        }
        
        # Train with HPO
        self.fit(X, y)
        
        # Return best parameters
        return self.best_params_ if self.best_params_ else {}
    
    def save(self, path: str) -> None:
        """Save model to disk with all necessary metadata."""
        if self.model is None:
            raise RuntimeError("Cannot save untrained model")
        
        # Ensure directory exists
        ensure_dir(os.path.dirname(path))
        
        # Prepare state dictionary
        state = {
            'model': self.model,
            'feature_names': self.feature_names_,
            'best_params': self.best_params_,
            # Store parameters for reconstruction
            'init_params': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'max_features': self.max_features,
                'bootstrap': self.bootstrap,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                **self.model_params
            },
            'randomized_search_config': self.randomized_search_config
        }
        
        try:
            # Save with compression
            joblib.dump(state, path, compress=3)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model to {path}: {e}", exc_info=True)
            raise
    
    @classmethod
    def load(cls, path: str) -> 'RandomForestModel':
        """Load model from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        try:
            # Load state dictionary
            state = joblib.load(path)
            
            # Create new instance
            instance = cls()
            
            # Restore model and metadata
            instance.model = state['model']
            instance.feature_names_ = state.get('feature_names')
            instance.best_params_ = state.get('best_params')
            
            # Restore parameters from saved state
            init_params = state.get('init_params', {})
            for key, value in init_params.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            
            # Restore model_params dictionary with non-primary parameters
            primary_params = {
                'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
                'max_features', 'bootstrap', 'random_state', 'n_jobs'
            }
            instance.model_params = {k: v for k, v in init_params.items() if k not in primary_params}
            
            # Restore randomized search config
            instance.randomized_search_config = state.get('randomized_search_config', {})
            
            logger.info(f"Model loaded from {path}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}", exc_info=True)
            raise