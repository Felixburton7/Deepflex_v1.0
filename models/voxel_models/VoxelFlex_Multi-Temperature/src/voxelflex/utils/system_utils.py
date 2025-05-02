"""
System related utilities (CPU, GPU, Memory).

Provides functions for system resource management, monitoring, and optimization.
"""

import os
import gc
import logging
import psutil
import torch
from typing import Optional, Dict, Any

logger = logging.getLogger("voxelflex.utils.system")

def get_cpu_cores() -> int:
    """
    Get the number of logical CPU cores.
    
    Returns:
        Number of logical CPU cores (1 if detection fails)
    """
    try:
        cores = os.cpu_count()
        logger.debug(f"Detected {cores} logical CPU cores.")
        return cores
    except NotImplementedError:
        logger.warning("Could not detect number of CPU cores. Defaulting to 1.")
        return 1

def get_gpu_details() -> Dict[str, Any]:
    """
    Get details about available NVIDIA GPUs.
    
    Returns:
        Dictionary with GPU details including count, names, memory
    """
    details = {
        "count": 0, 
        "names": [], 
        "memory_gb": [], 
        "cuda_available": False
    }
    
    if torch.cuda.is_available():
        details["cuda_available"] = True
        details["count"] = torch.cuda.device_count()
        
        for i in range(details["count"]):
            try:
                props = torch.cuda.get_device_properties(i)
                details["names"].append(props.name)
                details["memory_gb"].append(props.total_memory / (1024**3))
            except Exception as e:
                logger.error(f"Could not get properties for GPU {i}: {e}")
                details["names"].append("Error")
                details["memory_gb"].append(0)

    return details

def log_gpu_details(logger_instance: logging.Logger = logger):
    """
    Logs GPU details.
    
    Args:
        logger_instance: Logger to use
    """
    gpu_info = get_gpu_details()
    if gpu_info["cuda_available"]:
        logger_instance.info(f"CUDA Available: Yes. Found {gpu_info['count']} GPU(s).")
        for i in range(gpu_info['count']):
            logger_instance.info(f"  GPU {i}: {gpu_info['names'][i]} - Memory: {gpu_info['memory_gb'][i]:.2f} GB")
    else:
        logger_instance.info("CUDA Available: No. Running on CPU.")

def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Gets the recommended device (GPU if available and preferred, else CPU).
    
    Args:
        prefer_gpu: Whether to prefer GPU if available
        
    Returns:
        PyTorch device to use
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        log_gpu_details(logger)  # Log details when selecting GPU
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device.")
        
    logger.info(f"Selected device: {device}")
    return device

def check_memory_usage() -> Dict[str, float]:
    """
    Returns system memory usage statistics.
    
    Returns:
        Dictionary with memory usage information in GB and percentages
    """
    mem_info = psutil.virtual_memory()
    return {
        "total_gb": mem_info.total / (1024**3),
        "available_gb": mem_info.available / (1024**3),
        "percent_used": mem_info.percent,
        "used_gb": mem_info.used / (1024**3)
    }

def clear_memory(force_gc: bool = True, clear_cuda: bool = True):
    """
    Attempts to clear memory by running GC and emptying CUDA cache.
    
    Args:
        force_gc: Whether to run garbage collection
        clear_cuda: Whether to clear CUDA cache if available
    """
    logger.debug("Attempting to clear memory...")
    if force_gc:
        gc.collect()
        logger.debug("Ran garbage collector.")
        
    if clear_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("Emptied CUDA cache.")

def set_num_threads(num_threads: Optional[int] = None):
    """
    Sets the number of threads used by PyTorch.
    
    Args:
        num_threads: Number of threads to use (None for default)
    """
    if num_threads is not None and num_threads > 0:
        logger.info(f"Setting PyTorch CPU threads to: {num_threads}")
        torch.set_num_threads(num_threads)
    else:
        # Use default setting (often uses all cores, which is usually fine)
        cores = get_cpu_cores()
        logger.debug(f"Using default PyTorch CPU thread settings (likely based on {cores} cores).")

def adjust_workers_for_memory(requested_workers: int, memory_threshold: float = 85.0) -> int:
    """
    Adjusts the number of DataLoader workers based on available memory.
    Simple heuristic: reduces workers if memory usage is high.
    
    Args:
        requested_workers: Initially requested number of workers
        memory_threshold: Memory usage percentage threshold for reduction
        
    Returns:
        Adjusted number of workers
    """
    if requested_workers <= 0:
        return 0

    try:
        mem_percent = check_memory_usage().get("percent_used", 0)
        if mem_percent > memory_threshold:
            reduced_workers = max(1, requested_workers // 2)  # Halve workers, minimum 1
            logger.warning(f"High memory usage ({mem_percent:.1f}%) detected. "
                           f"Reducing DataLoader workers from {requested_workers} to {reduced_workers}.")
            return reduced_workers
        else:
            return requested_workers
    except Exception as e:
        logger.warning(f"Could not check memory to adjust workers: {e}. Using requested value: {requested_workers}")
        return requested_workers

class AdaptiveChunkSizer:
    """
    Dynamically adjust chunk size based on memory pressure.
    
    Useful for iterative data loading to prevent out-of-memory errors.
    """
    def __init__(self, 
                initial_chunk_size: int = 100,
                min_chunk_size: int = 10,
                memory_headroom: float = 0.25,  # Aim to keep 25% memory free
                adjustment_factor: float = 0.8):  # Reduce by 20% when needed
        """
        Initialize adaptive chunk sizer.
        
        Args:
            initial_chunk_size: Starting chunk size
            min_chunk_size: Minimum allowed chunk size
            memory_headroom: Fraction of memory to keep free
            adjustment_factor: Factor to multiply chunk size by when adjusting down
        """
        self.current_chunk_size = initial_chunk_size
        self.min_chunk_size = min_chunk_size
        self.memory_headroom = memory_headroom
        self.adjustment_factor = adjustment_factor
        self.reduction_count = 0
        
    def get_chunk_size(self) -> int:
        """
        Get current recommended chunk size.
        
        Returns:
            Current chunk size
        """
        return self.current_chunk_size
        
    def check_and_adjust(self) -> int:
        """
        Check memory conditions and adjust chunk size if needed.
        
        Returns:
            New recommended chunk size
        """
        # Check system RAM
        ram_usage = psutil.virtual_memory().percent / 100.0
        free_ram_fraction = 1.0 - ram_usage
        
        # Check GPU if available
        gpu_free_fraction = 1.0
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i)
                gpu_usage = allocated / props.total_memory
                gpu_free_fraction = min(gpu_free_fraction, 1.0 - gpu_usage)
        
        # Determine if adjustment needed
        headroom_ok = (free_ram_fraction >= self.memory_headroom and 
                       gpu_free_fraction >= self.memory_headroom)
                       
        if not headroom_ok and self.current_chunk_size > self.min_chunk_size:
            # Reduce chunk size to maintain headroom
            old_size = self.current_chunk_size
            self.current_chunk_size = max(
                self.min_chunk_size,
                int(self.current_chunk_size * self.adjustment_factor)
            )
            self.reduction_count += 1
            logger.warning(
                f"Memory pressure detected (RAM free: {free_ram_fraction:.2%}, "
                f"GPU free: {gpu_free_fraction:.2%}). "
                f"Reducing chunk size: {old_size} → {self.current_chunk_size}"
            )
        
        return self.current_chunk_size

def configure_h5py_for_domain_reading(chunk_size_mb: int = 64):
    """
    Configure h5py settings for optimal domain-level reading.
    
    Args:
        chunk_size_mb: Cache size in MB
    """
    try:
        import h5py
        
        # Set chunk cache size
        propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
        settings = list(propfaid.get_cache())
        
        # Adjust cache for our access pattern
        # nslots: number of chunk slots (more means fewer collisions)
        # nbytes: size of cache in bytes
        # w0: chunk preemption policy (0-1, higher keeps chunks in cache longer)
        cache_size_bytes = chunk_size_mb * 1024 * 1024
        settings[1] = cache_size_bytes  # Cache size in bytes
        settings[2] = 1.0  # Keep chunks in cache as long as possible
        propfaid.set_cache(*settings)
        
        # Register the property list with h5py
        h5py.get_config().default_file_access_proplist = propfaid
        
        logger.info(f"Configured h5py for domain reading with {chunk_size_mb}MB cache.")
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not configure h5py cache: {e}")