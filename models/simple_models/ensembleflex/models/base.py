"""
Base model class for ensembleflex ML pipeline.

This module defines the BaseModel abstract class that all
protein flexibility prediction models must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple

import numpy as np
import pandas as pd

class BaseModel(ABC):
    """
    Base class for all ensembleflex ML models.
    All models must implement these methods.
    """
    
    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'BaseModel':
        """
        Train the model on input data.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Self, for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Generate predictions for input data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save location
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BaseModel':
        """
        Load model from disk.
        
        Args:
            path: Path to saved model
            
        Returns:
            Loaded model instance
        """
        pass
    
    def predict_with_std(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty estimates.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, std_deviation)
        """
        # Default implementation returns predictions with zeros for std dev
        # Models that support uncertainty should override this
        predictions = self.predict(X)
        std_deviation = np.zeros_like(predictions)
        return predictions, std_deviation
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance values if available.
        
        Returns:
            Dictionary mapping feature names to importance values,
            or None if feature importance is not available
        """
        return None
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_') and key != 'model'
        }
    
    def get_model_name(self) -> str:
        """
        Get model name.
        
        Returns:
            String representing model name
        """
        return self.__class__.__name__
    
    def hyperparameter_optimize(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series],
        param_grid: Dict[str, Any],
        method: str = "bayesian",
        n_trials: int = 20,
        cv: int = 3
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization.
        
        Args:
            X: Feature matrix
            y: Target values
            param_grid: Parameter grid or distributions
            method: Optimization method ("grid", "random", or "bayesian")
            n_trials: Number of trials for random or bayesian methods
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with best parameters
            
        Raises:
            NotImplementedError: If the model doesn't support hyperparameter optimization
        """
        raise NotImplementedError("This model doesn't support hyperparameter optimization")
    
    def get_training_history(self) -> Optional[Dict[str, List[float]]]:
        """
        Get training history if available.
        
        Returns:
            Dictionary with training metrics by epoch, or None if not available
        """
        return None