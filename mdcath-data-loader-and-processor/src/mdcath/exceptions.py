"""
Custom exception classes for the mdCATH processor package.
Allows for more specific error handling.
"""

class MdCathProcessorError(Exception):
    """Base exception for errors in the mdcath-processor package."""
    pass

class ConfigurationError(MdCathProcessorError):
    """Exception raised for errors in configuration loading or validation."""
    pass

class HDF5ReaderError(MdCathProcessorError):
    """Custom exception for HDF5Reader errors."""
    pass

class PDBProcessorError(MdCathProcessorError):
    """Custom exception for PDBProcessor errors."""
    pass

class PropertiesCalculatorError(MdCathProcessorError):
    """Custom exception for PropertiesCalculator errors."""
    pass

class RmsfProcessingError(MdCathProcessorError):
    """Custom exception for RMSF processing errors."""
    pass

class FeatureBuilderError(MdCathProcessorError):
    """Custom exception for FeatureBuilder errors."""
    pass

class VoxelizationError(MdCathProcessorError):
    """Custom exception for Voxelization errors."""
    pass

class VisualizationError(MdCathProcessorError):
    """Custom exception for Visualization errors."""
    pass

class PipelineExecutionError(MdCathProcessorError):
    """Exception raised for errors during the main pipeline execution flow."""
    pass
