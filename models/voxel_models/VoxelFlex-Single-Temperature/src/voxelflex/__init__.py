"""
Voxelflex: A package for predicting per-residue RMSF values from voxelized protein data.
"""

__version__ = "0.1.0"
__author__ = "Felix"
__email__ = "s_felix@domain.com"

from voxelflex.cli.cli import main

# Import main modules to make them available at the package level
from voxelflex.data import data_loader, validators
from voxelflex.models import cnn_models
from voxelflex.config import config