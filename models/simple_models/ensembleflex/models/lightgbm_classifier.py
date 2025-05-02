"""
LightGBM classifier model implementation for the EnsembleFlex ML pipeline.

This module provides a LightGBM-based classifier for protein flexibility classification
with support for K-means binning, class weighting, and confusion matrix evaluation.
"""


'''
This in default_config.yaml

    lightgbm_classifier:
      enabled: true
      objective: 'multiclass'
      metric: 'multi_logloss'
      n_estimators: 1000
      learning_rate: 0.05
      num_leaves: 31
      max_depth: -1
      reg_alpha: 0.05
      reg_lambda: 0.05
      colsample_bytree: 0.7
      subsample: 0.7
      class_weight: 'balanced'
      n_jobs: -1
      random_state: 42
      num_class: 5
      early_stopping:
      enabled: true
      stopping_rounds: 50
      
      '''

import os
import logging
import time
import inspect
import pickle
from typing import Dict, Any, Optional, Union, List, Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.cluster import KMeans
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from ensembleflex.models import register_model
from ensembleflex.models.base import BaseModel
from ensembleflex.utils.helpers import ProgressCallback, ensure_dir

logger = logging.getLogger(__name__)

# Define flexibility class labels for interpretability
FLEXIBILITY_LABELS = [
    "Very Rigid", 
    "Rigid", 
    "Moderate", 
    "Flexible", 
    "Very Flexible"
]

@register_model("lightgbm_classifier")
class LightGBMClassifier(BaseModel):
    """
    LightGBM classifier for protein flexibility classification.
    
    Classifies protein residues into 5 flexibility categories from 
    very rigid (0) to very flexible (4) using RMSF values and other features.
    """
    
    def __init__(
        self,
        objective: str = 'multiclass',
        metric: str = 'multi_logloss',
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        reg_alpha: float = 0.05,
        reg_lambda: float = 0.05,
        colsample_bytree: float = 0.7,
        subsample: float = 0.7,
        class_weight: Union[str, Dict, None] = 'balanced',
        n_jobs: int = -1,
        random_state: int = 42,
        num_class: int = 5,  # Specific to classification - default 5 flexibility classes
        **kwargs
    ):
        """
        Initialize the LightGBM classifier model with parameters.
        
        Args:
            objective: LightGBM objective function (default: 'multiclass')
            metric: Metric for evaluation during training (default: 'multi_logloss')
            n_estimators: Number of boosting iterations (trees)
            learning_rate: Boosting learning rate
            num_leaves: Maximum number of leaves in one tree
            max_depth: Maximum tree depth, -1 means no limit
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
            colsample_bytree: Fraction of features to use per tree
            subsample: Fraction of samples to use per tree
            class_weight: Class weights for handling imbalanced data
            n_jobs: Number of parallel threads (-1 means all)
            random_state: Random seed for reproducibility
            num_class: Number of classes for classification (default: 5)
            **kwargs: Additional parameters passed to LightGBM
        """
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
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.num_class = num_class

        # Store additional parameters
        self.model_params = kwargs
        self.early_stopping_config = kwargs.get('early_stopping', {})
        
        # Initialize model-related attributes
        self.model = None         # LightGBM model
        self.kmeans = None        # K-means model for binning
        self.feature_names_ = None  # Feature names
        self.best_params_ = None  # Best params from hyperparameter optimization
        self.class_names_ = FLEXIBILITY_LABELS[:num_class]  # Class labels
        self.training_history = None  # Training metrics

    def fit(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series], 
        feature_names: Optional[List[str]] = None,
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> 'LightGBMClassifier':
        """
        Train the LightGBM classifier with K-means binning.
        
        Args:
            X: Feature matrix
            y: Target RMSF values (continuous)
            feature_names: Optional list of feature names
            X_val: Optional validation feature matrix
            y_val: Optional validation target values
            
        Returns:
            Self, for method chaining
        """
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        elif feature_names is not None:
            self.feature_names_ = feature_names
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]

        # Ensure X is a numpy array
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Ensure y is 1D numpy array
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.ravel(y)
        
        # Process validation data if provided
        X_val_array = None
        y_val_array = None
        if X_val is not None and y_val is not None:
            X_val_array = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            y_val_array = y_val.values if isinstance(y_val, pd.Series) else y_val

        # Step 1: Bin the continuous RMSF values using K-means
        logger.info(f"Fitting K-means with {self.num_class} clusters for RMSF binning")
        try:
            # Reshape for K-means
            y_reshaped = y_array.reshape(-1, 1)
            
            # Fit K-means
            self.kmeans = KMeans(
                n_clusters=self.num_class,
                random_state=self.random_state,
                n_init=10
            )
            clustered = self.kmeans.fit_predict(y_reshaped)
            
            # Get cluster centers and sort them to maintain consistent ordering
            # Lower RMSF (more rigid) -> Lower class number
            centers = self.kmeans.cluster_centers_.flatten()
            sorted_indices = np.argsort(centers)
            center_to_class = {i: sorted_indices[i] for i in range(self.num_class)}
            
            # Map clusters to sorted classes
            y_binned = np.array([center_to_class[c] for c in clustered])
            
            # Log K-means results
            centers_sorted = centers[sorted_indices]
            logger.info(f"K-means cluster centers (sorted by RMSF): {centers_sorted}")
            
            # Also bin validation data if provided
            y_val_binned = None
            if y_val_array is not None:
                y_val_reshaped = y_val_array.reshape(-1, 1)
                val_clustered = self.kmeans.predict(y_val_reshaped)
                y_val_binned = np.array([center_to_class[c] for c in val_clustered])
        
        except Exception as e:
            logger.error(f"K-means binning failed: {e}", exc_info=True)
            raise

        # Step 2: Configure and train LightGBM
        fit_start_time = time.time()
        
        try:
            # Prepare LightGBM parameters
            params = {
                'objective': self.objective,
                'metric': self.metric,
                'learning_rate': self.learning_rate,
                'num_leaves': self.num_leaves,
                'max_depth': self.max_depth,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,
                'colsample_bytree': self.colsample_bytree,
                'subsample': self.subsample,
                'n_jobs': self.n_jobs,
                'random_state': self.random_state,
                'verbose': -1,  # Suppress LightGBM output
                'num_class': self.num_class,  # Required for multiclass
            }
            
            # Add any additional parameters
            for key, value in self.model_params.items():
                if key not in ['early_stopping']:
                    params[key] = value

            # Process class weights if needed
            if self.class_weight == 'balanced':
                # Calculate balanced weights
                class_counts = np.bincount(y_binned, minlength=self.num_class)
                total_samples = len(y_binned)
                # Weight = total_samples / (num_classes * class_count)
                weights = {i: total_samples / (self.num_class * count) if count > 0 else 1.0 
                           for i, count in enumerate(class_counts)}
                logger.info(f"Using balanced class weights: {weights}")
                
                # Set class weights in LightGBM params
                for i, weight in weights.items():
                    params[f'class_weight_{i}'] = weight
            elif isinstance(self.class_weight, dict):
                # User provided weights
                for i, weight in self.class_weight.items():
                    params[f'class_weight_{i}'] = weight
                logger.info(f"Using provided class weights: {self.class_weight}")

            # Prepare datasets
            train_data = lgb.Dataset(
                X_array, 
                label=y_binned,
                feature_name=self.feature_names_
            )
            
            valid_data = None
            if X_val_array is not None and y_val_binned is not None:
                valid_data = lgb.Dataset(
                    X_val_array,
                    label=y_val_binned,
                    reference=train_data
                )

            # Early stopping configuration
            callbacks = []
            if self.early_stopping_config.get('enabled', False) and valid_data is not None:
                stopping_rounds = self.early_stopping_config.get('stopping_rounds', 50)
                logger.info(f"Enabling early stopping with patience={stopping_rounds}")
                callbacks.append(lgb.early_stopping(stopping_rounds=stopping_rounds))
                callbacks.append(lgb.log_evaluation(period=100))  # Log every 100 iterations

            # Train the model
            with ProgressCallback(total=1, desc="Training LightGBM Classifier") as pbar:
                self.model = lgb.train(
                    params=params,
                    train_set=train_data,
                    num_boost_round=self.n_estimators,
                    valid_sets=[valid_data] if valid_data is not None else None,
                    callbacks=callbacks
                )
                pbar.update()

            # Store the number of iterations actually used
            if hasattr(self.model, 'best_iteration'):
                logger.info(f"Best iteration: {self.model.best_iteration}")
            
            # Record training time
            train_time = time.time() - fit_start_time
            logger.info(f"LightGBM classifier trained in {train_time:.2f} seconds")
            
            # Generate and log class distribution
            class_counts = np.bincount(y_binned, minlength=self.num_class)
            class_distribution = {FLEXIBILITY_LABELS[i]: int(count) 
                               for i, count in enumerate(class_counts) if i < len(FLEXIBILITY_LABELS)}
            logger.info(f"Class distribution after binning: {class_distribution}")
            
            return self
            
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}", exc_info=True)
            raise

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Generate class predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted class labels (0-4)
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        
        # Ensure X is a numpy array
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Get raw probabilities
        probas = self.predict_proba(X_array)
        
        # Get class with highest probability
        return np.argmax(probas, axis=1)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Generate class probability predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of class probabilities (shape: [n_samples, n_classes])
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        
        # Ensure X is a numpy array
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Use LightGBM's predict method with raw score flag
        return self.model.predict(X_array)

    def predict_with_std(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty estimates.
        Uncertainty is represented by the entropy of the class probability distribution.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predicted_classes, uncertainty)
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        
        # Get class probabilities
        probas = self.predict_proba(X)
        
        # Calculate entropy as uncertainty measure
        # Add a small epsilon to avoid log(0)
        epsilon = 1e-10
        entropy = -np.sum(probas * np.log2(probas + epsilon), axis=1)
        
        # Normalize entropy to 0-1 range
        max_entropy = -np.log2(1.0 / self.num_class)  # Theoretical max entropy
        normalized_entropy = entropy / max_entropy
        
        # Get predicted classes
        predicted_classes = np.argmax(probas, axis=1)
        
        return predicted_classes, normalized_entropy

    def get_feature_importance(
        self, 
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        method: str = "gain"
    ) -> Dict[str, float]:
        """
        Get feature importance values.
        
        Args:
            X_val: Optional validation features for permutation importance
            y_val: Optional validation targets for permutation importance
            method: Importance type ('gain', 'split', or 'permutation')
            
        Returns:
            Dictionary mapping feature names to importance values
        """
        if self.model is None:
            return {}
        
        # Get feature names
        feature_names = self.feature_names_ or [f"feature_{i}" for i in range(self.model.num_feature())]
        
        # Method depends on what was requested
        if method == "permutation" and X_val is not None and y_val is not None:
            try:
                from sklearn.inspection import permutation_importance
                
                # Bin the continuous validation targets if needed
                if len(y_val.shape) == 1 and self.kmeans is not None:
                    y_val_reshaped = y_val.reshape(-1, 1)
                    y_val_binned = self.kmeans.predict(y_val_reshaped)
                else:
                    y_val_binned = y_val
                
                # Calculate permutation importance
                r = permutation_importance(
                    self,  # This class implements scikit-learn's predict
                    X_val,
                    y_val_binned,
                    n_repeats=10,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
                
                # Map importance values to feature names
                return {feature_names[i]: imp for i, imp in enumerate(r.importances_mean)}
            
            except Exception as e:
                logger.warning(f"Permutation importance calculation failed: {e}")
                # Fall back to built-in importance
        
        # Use LightGBM's built-in feature importance
        try:
            # Get importance values
            importance_type = "gain" if method not in ["split", "gain"] else method
            importance_values = self.model.feature_importance(importance_type=importance_type)
            
            # Map to feature names
            return {feature_names[i]: imp for i, imp in enumerate(importance_values)}
        
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")
            return {}

    def save(self, path: str) -> None:
        """
        Save model to disk with all components (LightGBM model and K-means).
        
        Args:
            path: Path to save location
        """
        if self.model is None:
            raise RuntimeError("Cannot save untrained model")
        
        # Ensure directory exists
        ensure_dir(os.path.dirname(path))
        
        # Save model state dictionary
        state = {
            'lightgbm_model': self.model,
            'kmeans_model': self.kmeans,
            'feature_names': self.feature_names_,
            'class_names': self.class_names_,
            'params': {
                'objective': self.objective,
                'metric': self.metric,
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'num_leaves': self.num_leaves,
                'max_depth': self.max_depth,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,
                'colsample_bytree': self.colsample_bytree,
                'subsample': self.subsample,
                'class_weight': self.class_weight,
                'n_jobs': self.n_jobs,
                'random_state': self.random_state,
                'num_class': self.num_class,
                'model_params': self.model_params
            },
            'best_params': self.best_params_
        }
        
        try:
            # Save using pickle
            with open(path, 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model to {path}: {e}", exc_info=True)
            raise

    @classmethod
    def load(cls, path: str) -> 'LightGBMClassifier':
        """
        Load model and all components from disk.
        
        Args:
            path: Path to saved model
            
        Returns:
            Loaded LightGBMClassifier instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        try:
            # Load model state
            with open(path, 'rb') as f:
                state = pickle.load(f)
            
            # Create new instance with saved parameters
            params = state.get('params', {})
            model_params = params.pop('model_params', {})
            instance = cls(**params, **model_params)
            
            # Restore model components
            instance.model = state.get('lightgbm_model')
            instance.kmeans = state.get('kmeans_model')
            instance.feature_names_ = state.get('feature_names')
            instance.class_names_ = state.get('class_names', FLEXIBILITY_LABELS)
            instance.best_params_ = state.get('best_params')
            
            logger.info(f"Model loaded from {path}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}", exc_info=True)
            raise

    def plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: Optional[np.ndarray] = None, 
        X: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Generate and optionally save a confusion matrix visualization.
        
        Args:
            y_true: True class labels or continuous RMSF values
            y_pred: Predicted class labels (optional if X is provided)
            X: Feature matrix to generate predictions (used if y_pred is None)
            save_path: Optional path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        # Process true labels - bin if continuous
        if len(y_true.shape) == 1 and self.kmeans is not None:
            try:
                # Check if values seem continuous (RMSF values)
                if np.unique(y_true).shape[0] > self.num_class:
                    logger.info("Binning continuous true values for confusion matrix")
                    y_true_reshaped = y_true.reshape(-1, 1)
                    y_true_binned = self.kmeans.predict(y_true_reshaped)
                    
                    # Get cluster centers and sort them
                    centers = self.kmeans.cluster_centers_.flatten()
                    sorted_indices = np.argsort(centers)
                    center_to_class = {i: sorted_indices[i] for i in range(self.num_class)}
                    
                    # Map clusters to sorted classes
                    y_true_binned = np.array([center_to_class[c] for c in y_true_binned])
                else:
                    # Already discrete classes
                    y_true_binned = y_true
            except Exception as e:
                logger.warning(f"Error binning true values: {e}")
                y_true_binned = y_true
        else:
            y_true_binned = y_true
        
        # Get predictions if not provided
        if y_pred is None and X is not None:
            y_pred = self.predict(X)
        elif y_pred is None:
            raise ValueError("Either y_pred or X must be provided")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true_binned, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot with seaborn
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            xticklabels=self.class_names_,
            yticklabels=self.class_names_,
            ax=ax
        )
        
        # Add labels
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")
        
        # Add overall accuracy
        accuracy = accuracy_score(y_true_binned, y_pred)
        plt.text(
            0.5, 1.05, 
            f"Accuracy: {accuracy:.3f}",
            horizontalalignment='center',
            fontsize=12,
            transform=ax.transAxes
        )
        
        # Save if path provided
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig

    def plot_feature_importance(
        self, 
        importance_dict: Optional[Dict[str, float]] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        method: str = "gain",
        save_path: Optional[str] = None,
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot feature importance and optionally save the figure.
        
        Args:
            importance_dict: Pre-computed importance dictionary (optional)
            X_val: Validation features (used if importance_dict not provided)
            y_val: Validation targets (used if importance_dict not provided)
            method: Importance type ('gain', 'split', or 'permutation')
            save_path: Optional path to save the plot
            top_n: Number of top features to show
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        # Get importance dict if not provided
        if importance_dict is None:
            importance_dict = self.get_feature_importance(X_val, y_val, method)
        
        if not importance_dict:
            logger.warning("No feature importance values available")
            return None
        
        # Create DataFrame and sort
        imp_df = pd.DataFrame(
            list(importance_dict.items()),
            columns=['feature', 'importance']
        ).sort_values('importance', ascending=False)
        
        # Limit to top_n features
        if len(imp_df) > top_n:
            imp_df = imp_df.head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot horizontal bar chart
        sns.barplot(x='importance', y='feature', data=imp_df, ax=ax)
        
        # Add labels
        ax.set_title(f"Top {len(imp_df)} Feature Importances ({method})")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        
        # Save if path provided
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to {save_path}")
        
        return fig

    def generate_classification_report(
        self, 
        y_true: np.ndarray, 
        y_pred: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        output_dict: bool = False
    ) -> Union[str, Dict]:
        """
        Generate a classification report with precision, recall, f1-score.
        
        Args:
            y_true: True class labels or continuous RMSF values
            y_pred: Predicted class labels (optional if X is provided)
            X: Feature matrix to generate predictions (used if y_pred is None)
            output_dict: Whether to return a dict instead of string
            
        Returns:
            Classification report as string or dictionary
        """
        # Process true labels - bin if continuous
        if len(y_true.shape) == 1 and self.kmeans is not None:
            try:
                # Check if values seem continuous (RMSF values)
                if np.unique(y_true).shape[0] > self.num_class:
                    logger.info("Binning continuous true values for classification report")
                    y_true_reshaped = y_true.reshape(-1, 1)
                    y_true_binned = self.kmeans.predict(y_true_reshaped)
                    
                    # Get cluster centers and sort them
                    centers = self.kmeans.cluster_centers_.flatten()
                    sorted_indices = np.argsort(centers)
                    center_to_class = {i: sorted_indices[i] for i in range(self.num_class)}
                    
                    # Map clusters to sorted classes
                    y_true_binned = np.array([center_to_class[c] for c in y_true_binned])
                else:
                    # Already discrete classes
                    y_true_binned = y_true
            except Exception as e:
                logger.warning(f"Error binning true values: {e}")
                y_true_binned = y_true
        else:
            y_true_binned = y_true
        
        # Get predictions if not provided
        if y_pred is None and X is not None:
            y_pred = self.predict(X)
        elif y_pred is None:
            raise ValueError("Either y_pred or X must be provided")
        
        # Generate classification report
        return classification_report(
            y_true_binned, 
            y_pred, 
            labels=range(self.num_class),
            target_names=self.class_names_,
            output_dict=output_dict
        )