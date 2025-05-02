"""
Logging utilities for VoxelFlex.

Provides enhanced logging functionality, progress bars, and performance monitoring.
"""

import logging
import sys
import time
import psutil
import gc
from contextlib import contextmanager
from typing import Optional, Dict, Any, Union, Callable
import os
from datetime import datetime

import torch
from tqdm import tqdm

# --- Global Logger Setup ---

def setup_logging(
    log_file: Optional[str] = None,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    name: str = "voxelflex"  # Root logger name
) -> logging.Logger:
    """
    Configures logging for the entire application.
    
    Args:
        log_file: Path to the log file. If None, only console logging is enabled.
        console_level: Logging level for console output (e.g., "DEBUG", "INFO", "WARNING").
        file_level: Logging level for file output (e.g., "DEBUG", "INFO").
        name: Name of the root logger for the application.
        
    Returns:
        The configured root logger instance.
    """
    logger = logging.getLogger(name)
    # Prevent adding multiple handlers if called again
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)  # Set root logger level to lowest (DEBUG)

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    try:
        # Convert string level names to logging constants
        console_log_level_int = getattr(logging, console_level.upper(), logging.INFO)
    except AttributeError:
        print(f"Warning: Invalid console log level '{console_level}'. Defaulting to INFO.")
        console_log_level_int = logging.INFO
        
    console_handler.setLevel(console_log_level_int)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # --- File Handler ---
    if log_file:
        try:
            # Ensure log directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            # Convert string level names to logging constants
            file_log_level_int = getattr(logging, file_level.upper(), logging.DEBUG)
        except AttributeError:
            print(f"Warning: Invalid file log level '{file_level}'. Defaulting to DEBUG.")
            file_log_level_int = logging.DEBUG

        file_handler = logging.FileHandler(log_file, mode='a')  # Append mode
        file_handler.setLevel(file_log_level_int)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file} (Level: {file_level.upper()})")

    logger.info(f"Console logging level set to: {console_level.upper()}")

    # --- Handle external libraries ---
    # Reduce verbosity of common libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance, inheriting configuration from the root.
    
    Args:
        name: Logger name suffix (will be prefixed with "voxelflex.")
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"voxelflex.{name}")  # Use hierarchical naming


# --- Progress Bar ---
class EnhancedProgressBar(tqdm):
    """
    Custom tqdm progress bar with additional features.
    
    Features:
    - Customizable formatting
    - Optional timing and ETA
    - Memory usage display
    """
    def __init__(self, *args, stage_info: Optional[str] = None, **kwargs):
        """
        Initialize enhanced progress bar.
        
        Args:
            *args: Arguments to pass to tqdm
            stage_info: Optional stage information to display
            **kwargs: Keyword arguments to pass to tqdm
        """
        # Set some sensible defaults if not provided by the caller
        kwargs.setdefault('ncols', 100)  # Wider default
        kwargs.setdefault('bar_format', '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        
        # Store stage info for potential use
        self.stage_info = stage_info
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Initialize memory tracking
        self.last_mem_check = 0
        self.mem_check_interval = 5  # seconds

    def update(self, n=1):
        """
        Update the progress bar.
        
        Args:
            n: Amount to increment the progress bar
        """
        super().update(n)
        
        # Optionally add memory usage in postfix
        current_time = time.time()
        if current_time - self.last_mem_check > self.mem_check_interval:
            try:
                mem_info = psutil.virtual_memory()
                self.set_postfix_str(f"Mem: {mem_info.percent:.1f}%", refresh=False)
                self.last_mem_check = current_time
            except:
                pass  # Ignore memory info errors

    def finish(self):
        """Close the progress bar."""
        # Ensure the bar is refreshed before closing to show final state
        self.refresh()
        self.close()


# --- Context Managers and Helpers ---

@contextmanager
def log_stage(stage_name: str, description: Optional[str] = None):
    """
    Logs the start and end of a processing stage, with timing.
    
    Args:
        stage_name: Name of the stage
        description: Optional description of the stage
        
    Yields:
        None
    """
    logger = get_logger("pipeline")  # Use a dedicated pipeline logger
    logger.info(f"--- Starting Stage: {stage_name} ---")
    if description:
        logger.info(f"  {description}")
    start_time = time.time()
    log_memory_usage(logger, level=logging.DEBUG)  # Log memory at start
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.info(f"--- Finished Stage: {stage_name} (Duration: {duration:.2f}s) ---")
        log_memory_usage(logger, level=logging.DEBUG)  # Log memory at end
        gc.collect()  # Force GC after stage


def log_section_header(logger_instance: logging.Logger, title: str):
    """
    Logs a formatted section header.
    
    Args:
        logger_instance: Logger to use
        title: Title of the section
    """
    logger_instance.info("")
    logger_instance.info("=" * 80)
    logger_instance.info(f"===== {title.upper()} =====")
    logger_instance.info("=" * 80)


def log_memory_usage(logger_instance: logging.Logger, level: int = logging.INFO):
    """
    Logs current system and GPU memory usage.
    
    Args:
        logger_instance: Logger to use
        level: Logging level for the memory information
    """
    try:
        # System RAM
        mem_info = psutil.virtual_memory()
        total_gb = mem_info.total / (1024**3)
        available_gb = mem_info.available / (1024**3)
        used_gb = mem_info.used / (1024**3)
        percent_used = mem_info.percent
        logger_instance.log(level, f"Sys Memory: {used_gb:.2f}/{total_gb:.2f} GB Used ({percent_used:.1f}%) | Available: {available_gb:.2f} GB")

        # Process Memory (Current Python process)
        process = psutil.Process(os.getpid())
        process_mem_mb = process.memory_info().rss / (1024**2)  # Resident Set Size
        logger_instance.log(level, f"Process Memory: {process_mem_mb:.2f} MB")

        # GPU Memory (if CUDA available)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                reserved_mem = torch.cuda.memory_reserved(i) / (1024**3)
                allocated_mem = torch.cuda.memory_allocated(i) / (1024**3)
                free_mem = total_mem - reserved_mem  # More accurate available estimate
                logger_instance.log(level, f"  GPU {i} Mem: Allocated={allocated_mem:.2f} GB | Reserved={reserved_mem:.2f} GB | Total={total_mem:.2f} GB")
    except Exception as e:
        logger_instance.warning(f"Could not retrieve memory usage details: {e}", exc_info=False)


def log_final_memory_state(logger_instance: logging.Logger):
    """
    Log detailed final memory state for diagnostics.
    
    Args:
        logger_instance: Logger to use
    """
    logger_instance.info("--- Final Memory State ---")
    log_memory_usage(logger_instance, level=logging.INFO)


# --- Performance Timing ---
class Timing:
    """
    Simple class to track execution time of code sections.
    
    Features:
    - Multiple section tracking
    - Statistics calculation
    - Total, average, min, max timing
    """
    def __init__(self):
        """Initialize timing tracker."""
        self.timings = {}
        self.starts = {}
        
    def start(self, section: str):
        """
        Start timing a section.
        
        Args:
            section: Name of the section to time
        """
        self.starts[section] = time.time()
        
    def end(self, section: str) -> float:
        """
        End timing a section.
        
        Args:
            section: Name of the section to end
            
        Returns:
            Elapsed time in seconds
        """
        if section in self.starts:
            elapsed = time.time() - self.starts.pop(section)
            self.timings.setdefault(section, []).append(elapsed)
            return elapsed
        return 0
        
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get timing statistics.
        
        Returns:
            Dictionary with stats for each section
        """
        stats = {}
        for section, times in self.timings.items():
            stats[section] = {
                'avg': sum(times) / len(times) if times else 0,
                'min': min(times) if times else 0,
                'max': max(times) if times else 0,
                'total': sum(times),
                'count': len(times)
            }
        return stats
        
    def reset(self):
        """Reset all timings."""
        self.timings = {}
        self.starts = {}


# --- Resource Monitor ---
class ResourceMonitor:
    """
    Monitor system resources and log or take action if thresholds are exceeded.
    
    Features:
    - Memory and GPU monitoring
    - Configurable thresholds
    - Callback support for handling threshold violations
    """
    def __init__(self, 
                ram_threshold: float = 0.90,  # 90% of system RAM
                gpu_threshold: float = 0.95,  # 95% of GPU memory
                check_interval: float = 5.0,  # Check every 5 seconds
                logger: Optional[logging.Logger] = None):
        """
        Initialize resource monitor.
        
        Args:
            ram_threshold: System RAM threshold (0.0-1.0)
            gpu_threshold: GPU memory threshold (0.0-1.0)
            check_interval: Check interval in seconds
            logger: Logger to use (if None, create a new one)
        """
        self.ram_threshold = ram_threshold
        self.gpu_threshold = gpu_threshold
        self.check_interval = check_interval
        self.logger = logger or get_logger("resources")
        
        self.running = False
        self.on_threshold_exceeded = None  # Callback function
        
        self.logger.info(f"ResourceMonitor initialized with RAM threshold: {ram_threshold:.1%}, GPU threshold: {gpu_threshold:.1%}")
    
    def check(self) -> Dict[str, Any]:
        """
        Check current resource usage.
        
        Returns:
            Dictionary with resource usage information
        """
        import gc
        
        # Collect memory stats
        result = {}
        
        # System RAM
        mem_info = psutil.virtual_memory()
        ram_usage = mem_info.percent / 100.0
        result['ram_usage'] = ram_usage
        result['ram_exceeded'] = ram_usage > self.ram_threshold
        
        # GPU if available
        result['gpu_usage'] = []
        result['gpu_exceeded'] = False
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                total = props.total_memory
                
                # Calculate usage based on allocated memory
                gpu_usage = allocated / total
                result['gpu_usage'].append(gpu_usage)
                
                if gpu_usage > self.gpu_threshold:
                    result['gpu_exceeded'] = True
                    result['gpu_id'] = i
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return result
    
    def log_status(self, level: int = logging.INFO):
        """
        Log current resource status.
        
        Args:
            level: Logging level to use
        """
        status = self.check()
        
        # Log RAM
        ram_usage = status['ram_usage']
        self.logger.log(level, f"System RAM: {ram_usage:.1%} used, threshold: {self.ram_threshold:.1%}")
        
        # Log GPU if available
        if torch.cuda.is_available():
            for i, gpu_usage in enumerate(status['gpu_usage']):
                self.logger.log(level, f"GPU {i}: {gpu_usage:.1%} used, threshold: {self.gpu_threshold:.1%}")
    
    def handle_threshold_violations(self) -> bool:
        """
        Check for threshold violations and handle them.
        
        Returns:
            True if any threshold was exceeded, False otherwise
        """
        status = self.check()
        violations = []
        
        # Check RAM
        if status['ram_exceeded']:
            violations.append(f"System RAM usage ({status['ram_usage']:.1%}) exceeded threshold ({self.ram_threshold:.1%})")
            
        # Check GPU
        if status['gpu_exceeded']:
            gpu_id = status.get('gpu_id', 0)
            gpu_usage = status['gpu_usage'][gpu_id]
            violations.append(f"GPU {gpu_id} memory usage ({gpu_usage:.1%}) exceeded threshold ({self.gpu_threshold:.1%})")
        
        # Handle violations
        if violations:
            message = "RESOURCE THRESHOLD EXCEEDED: " + ", ".join(violations)
            self.logger.warning(message)
            
            # Call callback if registered
            if callable(self.on_threshold_exceeded):
                try:
                    self.on_threshold_exceeded(violations)
                except Exception as e:
                    self.logger.error(f"Error in threshold callback: {e}")
                    
            return True
            
        return False