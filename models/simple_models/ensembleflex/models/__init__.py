"""
Model registry for the ensembleflex ML pipeline.

This module provides a registry for model classes and utility functions
for model discovery and management.
"""

from importlib import import_module
from pathlib import Path
from typing import Dict, Type, List, Optional

from .base import BaseModel

# Global model registry
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}

def register_model(name: str):
    """
    Decorator to register a model class in the global registry.
    
    Args:
        name: Name to register the model under
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model_class(model_name: str) -> Type[BaseModel]:
    """
    Get a model class by name.
    
    Args:
        model_name: Name of the model to get
        
    Returns:
        Model class
        
    Raises:
        ValueError: If model_name is not found in registry
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry. " 
                         f"Available models: {', '.join(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_name]

def get_available_models() -> List[str]:
    """
    Get list of available model names.
    
    Returns:
        List of registered model names
    """
    return list(MODEL_REGISTRY.keys())

# Auto-discover models using importlib
models_dir = Path(__file__).parent
for model_file in models_dir.glob('*.py'):
    if model_file.stem not in ('__init__', 'base'):
        try:
            import_module(f'ensembleflex.models.{model_file.stem}')
        except ImportError as e:
            import logging
            logging.getLogger(__name__).warning(f"Error importing model module {model_file.stem}: {e}")