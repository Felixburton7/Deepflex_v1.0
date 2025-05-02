"""
Enhanced system utility functions with memory management for Voxelflex.

This module provides improved memory monitoring and resource allocation.
"""

import os
import platform
import multiprocessing
import gc
import psutil
from typing import Dict, Any, Tuple, Optional

import torch
import numpy as np

from voxelflex.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Memory thresholds (percentage of system memory)
MEMORY_WARNING_THRESHOLD = 0.70  
MEMORY_CRITICAL_THRESHOLD = 0.80 
MEMORY_EMERGENCY_THRESHOLD = 0.90  


def get_device(adjust_for_gpu: bool = True) -> torch.device:
    if adjust_for_gpu and torch.cuda.is_available():
        try:
            # Explicitly set device
            torch.cuda.set_device(0)
            
            # Verify GPU setup
            test_tensor = torch.tensor([1.0], device='cuda')
            
            # Detailed logging
            logger.info(f"GPU Selected: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
            
            return torch.device('cuda')
        except Exception as e:
            logger.error(f"GPU Setup Failed: {e}")
    
    return torch.device('cpu')

def check_system_resources(detect_cores: bool = True, 
                           adjust_for_gpu: bool = True) -> Dict[str, Any]:
    """
    Check available system resources with enhanced memory reporting.
    
    Args:
        detect_cores: Whether to detect available CPU cores
        adjust_for_gpu: Whether to check for GPU availability
        
    Returns:
        Dictionary containing system resource information
    """
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    
    # Check memory resources - ADD THIS SECTION
    memory = psutil.virtual_memory()
    system_info["memory_total_gb"] = memory.total / (1024**3)
    system_info["memory_available_gb"] = memory.available / (1024**3)
    system_info["memory_percent_used"] = memory.percent
    
    # Add memory thresholds
    system_info["memory_warning_threshold"] = MEMORY_WARNING_THRESHOLD * 100
    system_info["memory_critical_threshold"] = MEMORY_CRITICAL_THRESHOLD * 100
    system_info["memory_emergency_threshold"] = MEMORY_EMERGENCY_THRESHOLD * 100
    
    # Check CPU resources
    if detect_cores:
        cpu_count = multiprocessing.cpu_count()
        system_info["cpu_count"] = cpu_count
        
        # # Reduce worker recommendation to be more conservative
        # system_info["recommended_workers"] = max(1, min(4, cpu_count // 2))
        # More aggressive worker allocation
        system_info["recommended_workers"] = min(32, cpu_count - 4)  # Leave 2 cores free, max 32
    
    # Check GPU resources
    if adjust_for_gpu:
        system_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            system_info["cuda_device_count"] = torch.cuda.device_count()
            system_info["cuda_device_name"] = torch.cuda.get_device_name(0)
            system_info["cuda_version"] = torch.version.cuda
            
            # Add GPU memory info
            try:
                gpu_id = 0
                gpu_properties = torch.cuda.get_device_properties(gpu_id)
                total_memory_gb = gpu_properties.total_memory / (1024**3)  # Convert to GB
                allocated_memory_gb = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                reserved_memory_gb = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                
                system_info["gpu_total_memory_gb"] = total_memory_gb
                system_info["gpu_allocated_memory_gb"] = allocated_memory_gb
                system_info["gpu_reserved_memory_gb"] = reserved_memory_gb
                system_info["gpu_available_memory_gb"] = total_memory_gb - allocated_memory_gb
            except Exception as e:
                logger.warning(f"Error getting GPU memory info: {str(e)}")
    
    # Log enhanced system information
    logger.info(f"Platform: {system_info['platform']}")
    logger.info(f"Python version: {system_info['python_version']}")
    logger.info(f"PyTorch version: {system_info['torch_version']}")
    
    # Log memory information
    logger.info(f"System memory: {system_info['memory_total_gb']:.2f} GB total, "
                f"{system_info['memory_available_gb']:.2f} GB available "
                f"({system_info['memory_percent_used']}% used)")
    
    if detect_cores:
        logger.info(f"CPU cores: {system_info['cpu_count']}")
        logger.info(f"Recommended worker processes: {system_info['recommended_workers']}")
    
    if adjust_for_gpu and system_info.get("cuda_available", False):
        logger.info(f"CUDA available: {system_info['cuda_available']}")
        logger.info(f"CUDA device count: {system_info['cuda_device_count']}")
        logger.info(f"CUDA device name: {system_info['cuda_device_name']}")
        logger.info(f"CUDA version: {system_info['cuda_version']}")
        
        if "gpu_total_memory_gb" in system_info:
            logger.info(f"GPU memory: {system_info['gpu_total_memory_gb']:.2f} GB total, "
                        f"{system_info['gpu_available_memory_gb']:.2f} GB available")
    
    return system_info


def set_num_threads(num_threads: int = None) -> int:
    """
    Set number of threads for PyTorch.
    
    Args:
        num_threads: Number of threads to use (if None, use half of available cores)
        
    Returns:
        Number of threads set
    """
    if num_threads is None:
        cpu_count = multiprocessing.cpu_count()
        # Use half of available cores for better resource sharing
        num_threads = max(1, cpu_count // 2)
    
    torch.set_num_threads(num_threads)
    logger.info(f"Set PyTorch number of threads to {num_threads}")
    
    return num_threads


def check_memory_usage() -> Dict[str, float]:
    """
    Check current memory usage and return detailed statistics.
    
    Returns:
        Dictionary with memory usage statistics
    """
    # Get process memory info
    process = psutil.Process(os.getpid())
    process_memory = process.memory_info()
    
    # Get system memory info
    system_memory = psutil.virtual_memory()
    
    # Calculate memory statistics
    process_rss_gb = process_memory.rss / (1024**3)
    process_vms_gb = process_memory.vms / (1024**3)
    system_total_gb = system_memory.total / (1024**3)
    system_available_gb = system_memory.available / (1024**3)
    system_used_gb = system_total_gb - system_available_gb
    system_percent = system_memory.percent
    
    # Check GPU memory if available
    gpu_stats = {}
    if torch.cuda.is_available():
        try:
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                gpu_stats[f"gpu{i}_allocated_gb"] = allocated
                gpu_stats[f"gpu{i}_reserved_gb"] = reserved
        except Exception as e:
            logger.debug(f"Error getting GPU memory stats: {str(e)}")
    
    # Combine all stats
    memory_stats = {
        "process_rss_gb": process_rss_gb,
        "process_vms_gb": process_vms_gb,
        "system_total_gb": system_total_gb,
        "system_available_gb": system_available_gb,
        "system_used_gb": system_used_gb,
        "system_percent": system_percent,
        **gpu_stats
    }
    
    return memory_stats


def clear_memory(force_gc: bool = True, clear_cuda: bool = True, aggressive: bool = False) -> Dict[str, float]:
    """
    Attempt to clear memory by forcing aggressive garbage collection and clearing CUDA cache.
    
    Args:
        force_gc: Whether to force garbage collection
        clear_cuda: Whether to clear CUDA cache
        aggressive: Whether to use aggressive memory clearing (for emergency situations)
        
    Returns:
        Dictionary with memory usage statistics after clearing
    """
    # Get memory usage before clearing
    before_stats = check_memory_usage()
    
    # Force garbage collection if requested - run multiple times for better collection
    if force_gc:
        # Run multiple collections to better handle reference cycles and fragmentation
        for _ in range(5 if aggressive else 3):  # More cycles in aggressive mode
            gc.collect()
        
        # Attempt to manually clean reference cycles between objects
        gc.collect(generation=2)  # Focus on oldest generation
    
    # Clear CUDA cache if available and requested
    if clear_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Force synchronization to ensure CUDA operations complete
        if torch.cuda.is_initialized():
            try:
                torch.cuda.synchronize()
                
                # Reset the CUDA device on critical memory or in aggressive mode
                if is_memory_critical() or aggressive:
                    current_device = torch.cuda.current_device()
                    torch.cuda.empty_cache()
                    # More aggressive clearing for critical situations
                    if hasattr(torch.cuda, 'memory_stats'):
                        torch.cuda.memory_stats(current_device)  # Force memory stats refresh
                    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                        torch.cuda.reset_peak_memory_stats(current_device)
                    if hasattr(torch.cuda.memory, '_dump_snapshot'):
                        torch.cuda.memory._dump_snapshot()
            except Exception as e:
                logger.warning(f"Error during CUDA synchronization: {str(e)}")
    
    # In aggressive mode, attempt to release memory back to the OS
    if aggressive:
        try:
            import ctypes
            try:
                # Try Linux version first
                libc = ctypes.CDLL('libc.so.6')
                if hasattr(libc, 'malloc_trim'):
                    libc.malloc_trim(0)
            except:
                # Try Windows version
                try:
                    kernel32 = ctypes.windll.kernel32
                    if hasattr(kernel32, 'SetProcessWorkingSetSize'):
                        kernel32.SetProcessWorkingSetSize(
                            ctypes.windll.kernel32.GetCurrentProcess(),
                            ctypes.c_size_t(-1),
                            ctypes.c_size_t(-1)
                        )
                except:
                    pass
        except Exception as e:
            logger.debug(f"Could not call OS memory release: {str(e)}")
    
    # Get memory usage after clearing
    after_stats = check_memory_usage()
    
    # Calculate memory freed
    process_rss_freed = before_stats["process_rss_gb"] - after_stats["process_rss_gb"]
    system_freed = before_stats["system_used_gb"] - after_stats["system_used_gb"]
    
    # Log more detailed information about memory clearing
    logger.info(f"Memory cleared: Process RSS: {process_rss_freed:.2f} GB, System: {system_freed:.2f} GB")
    
    return after_stats

def is_memory_critical() -> bool:
    """
    Check if system memory usage is at a critical level.
    
    Returns:
        True if memory usage is critical, False otherwise
    """
    system_memory = psutil.virtual_memory()
    memory_percent = system_memory.percent / 100.0
    
    return memory_percent >= MEMORY_CRITICAL_THRESHOLD

def estimate_batch_size(
    input_shape: Tuple[int, ...], 
    target_memory_gb: float = None,
    min_batch_size: int = 32, 
    max_batch_size: int = 1024
) -> int:
    """
    Estimate a safe batch size based on input shape and available memory.
    
    Args:
        input_shape: Shape of a single input tensor
        target_memory_gb: Target memory usage in GB (if None, will be estimated)
        min_batch_size: Minimum allowable batch size
        max_batch_size: Maximum allowable batch size
        
    Returns:
        Estimated safe batch size
    """
    # Calculate size of a single input in bytes
    element_size = 4  # float32 = 4 bytes
    single_input_size = np.prod(input_shape) * element_size
    
    # Improved memory estimation for PyTorch overhead
    # PyTorch typically needs extra memory for gradients, optimizer states, etc.
    memory_overhead_factor = 3.0  # More aggressive - allows larger batch sizes
    
    # Use GPU memory if available for batch size estimation
    if torch.cuda.is_available():
        try:
            gpu_id = 0
            gpu_properties = torch.cuda.get_device_properties(gpu_id)
            total_memory_gb = gpu_properties.total_memory / (1024**3)
            allocated_memory_gb = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            reserved_memory_gb = torch.cuda.memory_reserved(gpu_id) / (1024**3)
            
            # Calculate available memory more accurately
            effective_available_gb = total_memory_gb - max(allocated_memory_gb, reserved_memory_gb)
            
            # Use a higher percentage of available GPU memory
            target_gpu_memory_gb = effective_available_gb * 0.85  # Use up to 85% of available memory
            
            # Calculate maximum batch size based on GPU memory
            max_elements = (target_gpu_memory_gb * (1024**3)) / (single_input_size * memory_overhead_factor)
            gpu_batch_size = max(min_batch_size, min(int(max_elements), max_batch_size))
            
            logger.info(f"GPU memory: {total_memory_gb:.2f} GB total, {effective_available_gb:.2f} GB available")
            logger.info(f"Estimated GPU-based batch size: {gpu_batch_size}")
            
            return gpu_batch_size
        
        except Exception as e:
            logger.warning(f"GPU memory estimation failed: {str(e)}")
    
    # Fallback to system memory estimation
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    # Use more aggressive memory estimation for CPU
    system_memory_factor = 0.7  # Use up to 70% of available system memory
    target_system_memory_gb = available_gb * system_memory_factor
    
    # Calculate maximum batch size based on system memory
    max_elements = (target_system_memory_gb * (1024**3)) / (single_input_size * memory_overhead_factor)
    system_batch_size = max(min_batch_size, min(int(max_elements), max_batch_size))
    
    logger.info(f"System memory: {memory.total/(1024**3):.2f} GB total, {available_gb:.2f} GB available")
    logger.info(f"Estimated system memory-based batch size: {system_batch_size}")
    
    return system_batch_size

def adjust_workers_for_memory(default_workers: int = 4) -> int:
    """
    Adjust the number of worker processes based on available system memory.
    
    Args:
        default_workers: Default number of workers
        
    Returns:
        Adjusted number of workers
    """
    memory = psutil.virtual_memory()
    memory_percent = memory.percent / 100.0
    
    if memory_percent >= MEMORY_EMERGENCY_THRESHOLD:
        # Critical memory situation, use single-threaded operation
        logger.warning(f"Emergency memory situation ({memory_percent*100:.1f}%), using 0 workers")
        return 0
    elif memory_percent >= MEMORY_CRITICAL_THRESHOLD:
        # Critical memory situation, use single-threaded operation
        logger.warning(f"Critical memory situation ({memory_percent*100:.1f}%), using 1 worker")
        return 1
    elif memory_percent >= MEMORY_WARNING_THRESHOLD:
        # Warning memory situation, reduce workers
        adjusted_workers = max(1, default_workers // 2)
        logger.warning(f"Memory usage high ({memory_percent*100:.1f}%), reducing workers to {adjusted_workers}")
        return adjusted_workers
    else:
        # Normal memory situation, use default workers
        return default_workers
    
    

def log_gpu_details():
    """
    Comprehensive GPU logging and diagnostics.
    """
    if torch.cuda.is_available():
        gpu_id = 0
        device = torch.device('cuda', gpu_id)
        
        # Device Properties
        props = torch.cuda.get_device_properties(gpu_id)
        logger.info(f"GPU Details:")
        logger.info(f"  Name: {props.name}")
        logger.info(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
        logger.info(f"  CUDA Capability: {props.major}.{props.minor}")
        logger.info(f"  Multi-Processor Count: {props.multi_processor_count}")
        
        # Current Memory Stats
        allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
        reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated(gpu_id) / (1024**3)
        
        logger.info(f"Memory Stats:")
        logger.info(f"  Currently Allocated: {allocated:.2f} GB")
        logger.info(f"  Reserved Memory: {reserved:.2f} GB")
        logger.info(f"  Peak Allocated: {max_allocated:.2f} GB")
        
        # Utilization and Performance
        logger.info(f"  Current GPU Utilization: {torch.cuda.utilization(gpu_id)}%")
        
        
        

def emergency_memory_reduction():
    """
    Emergency procedure to reduce memory usage when critical thresholds are reached.
    This is a last resort before potential system crash.
    
    Returns:
        Amount of memory freed in GB
    """
    logger.warning("EMERGENCY MEMORY REDUCTION INITIATED")
    
    before_stats = check_memory_usage()
    
    # 1. Aggressive garbage collection
    clear_memory(force_gc=True, clear_cuda=True, aggressive=True)
    
    # 2. Trigger Python's internal memory compaction
    import gc
    gc.collect(generation=2)
    
    # 3. Attempt to release memory back to the OS
    try:
        import ctypes
        try:
            # Try Linux version first
            libc = ctypes.CDLL('libc.so.6')
            if hasattr(libc, 'malloc_trim'):
                libc.malloc_trim(0)
        except:
            # Try Windows version
            try:
                kernel32 = ctypes.windll.kernel32
                kernel32.SetProcessWorkingSetSize(
                    ctypes.windll.kernel32.GetCurrentProcess(),
                    ctypes.c_size_t(-1),
                    ctypes.c_size_t(-1)
                )
            except:
                pass
    except Exception as e:
        logger.debug(f"Could not call OS memory release: {str(e)}")
    
    # 4. Force CUDA memory cleanup if available
    if torch.cuda.is_available():
        try:
            for i in range(torch.cuda.device_count()):
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'memory_stats'):
                    torch.cuda.memory_stats(i)
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats(i)
        except Exception as e:
            logger.warning(f"Error clearing CUDA memory: {str(e)}")
    
    # 5. Get memory usage after emergency procedures
    after_stats = check_memory_usage()
    
    # Calculate memory freed
    process_rss_freed = before_stats["process_rss_gb"] - after_stats["process_rss_gb"]
    system_freed = before_stats["system_used_gb"] - after_stats["system_used_gb"]
    
    logger.warning(f"Emergency memory reduction freed: Process RSS: {process_rss_freed:.2f} GB, System: {system_freed:.2f} GB")
    logger.warning(f"Current memory usage: {after_stats['system_percent']:.1f}% ({after_stats['process_rss_gb']:.2f} GB)")
    
    return system_freed