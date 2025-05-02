
"""
Enhanced logging utilities for Voxelflex.

This module provides advanced logging functionality including:
- Structured pipeline progress tracking
- Memory usage monitoring
- Stage-based progress reporting
- Enhanced progress bar with ETA, memory stats, and stage info
"""

import os
import sys
import time
import logging
import psutil
import shutil
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

# ANSI escape codes for colors
COLOR_RESET = "\033[0m"
COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_MAGENTA = "\033[35m"
COLOR_CYAN = "\033[36m"

# ASCII art for logger
ASCII_HEADER = r"""
 __      __                  _  __ _           
 \ \    / /                 | |/ _| |          
  \ \  / /__  __  _____  ___| | |_| | _____  __
   \ \/ / _ \\ \/ / _ \/ __| |  _| |/ _ \ \/ /
    \  / (_) |>  <  __/ (__| | | | |  __/>  < 
     \/ \___//_/\_\___|\___|_|_| |_|\___/_/\_\                                     
"""

# Pipeline stages for tracking progress
# In src/voxelflex/utils/logging_utils.py

# Update the PIPELINE_STAGES list to include MODEL_SAVING
PIPELINE_STAGES = [
    "INITIALIZATION",
    "DATA_LOADING",
    "DATA_VALIDATION",
    "DATASET_CREATION",
    "MODEL_CREATION",
    "TRAINING",
    "MODEL_SAVING",  # Add this missing stage
    "PREDICTION",
    "EVALUATION",
    "VISUALIZATION",
    "CLEANUP"
]

class PipelineTracker:
    """Tracks the progress of the Voxelflex pipeline through its stages."""
    
    def __init__(self):
        """Initialize the pipeline tracker."""
        self.current_stage = None
        self.stage_start_time = None
        self.stages_completed = []
        self.stage_durations = {}
        self.logger = logging.getLogger("voxelflex.pipeline")
    
    def start_stage(self, stage: str, details: str = "") -> None:
        """
        Mark the beginning of a pipeline stage.
        
        Args:
            stage: Name of the stage
            details: Additional details about the stage
        """
        if stage not in PIPELINE_STAGES:
            raise ValueError(f"Unknown pipeline stage: {stage}")
        
        self.current_stage = stage
        self.stage_start_time = time.time()
        
        # Log the stage start
        detail_text = f" - {details}" if details else ""
        self.logger.info(f"{COLOR_CYAN}Starting pipeline stage: {stage}{COLOR_RESET}{detail_text}")
        
        # Log memory usage
        self._log_memory_usage()
    
    def end_stage(self, stage: str = None) -> None:
        """
        Mark the completion of a pipeline stage.
        
        Args:
            stage: Name of the stage (if None, use current_stage)
        """
        if stage is None:
            stage = self.current_stage
        
        if stage != self.current_stage:
            self.logger.warning(f"Ending stage {stage}, but current stage is {self.current_stage}")
        
        if self.stage_start_time is None:
            self.logger.warning(f"Ending stage {stage}, but no start time was recorded")
            duration = 0
        else:
            duration = time.time() - self.stage_start_time
        
        self.stages_completed.append(stage)
        self.stage_durations[stage] = duration
        
        # Log the stage completion and duration
        self.logger.info(f"{COLOR_GREEN}Completed pipeline stage: {stage} in {self._format_time(duration)}{COLOR_RESET}")
        
        # Log memory usage
        self._log_memory_usage()
        
        self.current_stage = None
        self.stage_start_time = None
    
    def get_current_stage(self) -> str:
        """
        Get the current pipeline stage.
        
        Returns:
            Current stage name or "UNKNOWN" if no stage is active
        """
        return self.current_stage or "UNKNOWN"
    
    def get_pipeline_progress(self) -> float:
        """
        Get the overall pipeline progress as a percentage.
        
        Returns:
            Progress percentage (0-100)
        """
        if not self.stages_completed:
            return 0.0
        
        total_stages = len(PIPELINE_STAGES)
        completed_stages = len(self.stages_completed)
        
        # If a stage is in progress, add partial credit
        if self.current_stage:
            current_stage_idx = PIPELINE_STAGES.index(self.current_stage)
            prev_stages = current_stage_idx
            
            # Calculate progress within the current stage (assuming linear progress)
            # This is a rough approximation and could be improved with more data
            if self.stage_start_time:
                # Average time for completed stages (if any)
                if self.stage_durations:
                    avg_duration = sum(self.stage_durations.values()) / len(self.stage_durations)
                    # Estimate current stage progress (cap at 0.95 to avoid appearing complete)
                    elapsed = time.time() - self.stage_start_time
                    stage_progress = min(0.95, elapsed / (avg_duration * 1.5))  # Apply a 1.5x factor for safety
                else:
                    # No historical data, assume 50% complete
                    stage_progress = 0.5
            else:
                stage_progress = 0
            
            progress = (prev_stages + stage_progress) / total_stages
        else:
            progress = completed_stages / total_stages
        
        return progress * 100
    
    def _log_memory_usage(self) -> None:
        """Log current memory usage."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        self.logger.info(f"Memory usage: {memory_info.rss / (1024 * 1024):.1f} MB")
    
    def _format_time(self, seconds: float) -> str:
        """
        Format time in seconds to a readable string.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            seconds = seconds % 60
            return f"{minutes} minutes {seconds:.1f} seconds"
        else:
            hours = int(seconds / 3600)
            seconds = seconds % 3600
            minutes = int(seconds / 60)
            seconds = seconds % 60
            return f"{hours} hours {minutes} minutes {seconds:.1f} seconds"

# Global pipeline tracker instance
pipeline_tracker = PipelineTracker()

def setup_logging(log_file: Optional[str] = None, console_level: str = "INFO", 
                  file_level: str = "DEBUG") -> None:
    """
    Set up enhanced logging for Voxelflex.
    
    Args:
        log_file: Path to log file (if None, logging to file is disabled)
        console_level: Logging level for console output
        file_level: Logging level for file output
    """
    # Convert string levels to logging levels
    console_level = getattr(logging, console_level.upper())
    file_level = getattr(logging, file_level.upper())
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to lowest level to capture everything
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter with timestamp
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Log ASCII art header
    logger.info(f"\n{ASCII_HEADER}")
    logger.info(f"{COLOR_GREEN}Voxelflex logger initialized{COLOR_RESET}")
    
    # Log system memory information
    vm = psutil.virtual_memory()
    logger.info(f"System memory: {vm.total / (1024**3):.1f} GB total, {vm.available / (1024**3):.1f} GB available")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


@contextmanager
def log_stage(stage: str, details: str = ""):
    """
    Context manager for tracking a pipeline stage.
    
    Args:
        stage: Stage name
        details: Additional details about the stage
    """
    pipeline_tracker.start_stage(stage, details)
    try:
        yield
    finally:
        pipeline_tracker.end_stage(stage)


class EnhancedProgressBar:
    """
    Enhanced progress bar with memory usage tracking and time statistics.
    """
    
    def __init__(self, total: int, prefix: str = "", suffix: str = "", 
                 bar_length: int = 30, stage_info: str = None):
        """
        Initialize progress bar.
        
        Args:
            total: Total number of items
            prefix: Prefix string
            suffix: Suffix string
            bar_length: Length of the progress bar
            stage_info: Current pipeline stage information
        """
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.bar_length = bar_length
        self.current = 0
        self.start_time = time.time()
        self.last_printed_length = 0
        self.stage_info = stage_info or pipeline_tracker.get_current_stage()
        
        # Get terminal width
        self.terminal_width = shutil.get_terminal_size().columns
        
        # Initial memory snapshot
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.process.memory_info().rss
        
        # Print initial progress
        self._print_progress()
    
    def update(self, current: int) -> None:
        """
        Update progress bar.
        
        Args:
            current: Current progress value
        """
        self.current = current
        self._print_progress()
    
    def _print_progress(self) -> None:
        """Print the progress bar with enhanced information."""
        # Calculate progress
        if self.total <= 0:  # Check if total is zero or negative
            percent = 100.0  # Assume 100% complete if total is 0
            filled_length = self.bar_length  # Fill the entire bar
        else:
            percent = float(self.current) / float(self.total) * 100
            filled_length = int(round(self.bar_length * self.current / float(self.total)))
        
        bar = "█" * filled_length + "-" * (self.bar_length - filled_length)
        
        # Calculate elapsed time and ETA
        elapsed_time = time.time() - self.start_time
        
        # Handle items_per_second calculation when total is 0
        if self.total <= 0 or elapsed_time <= 0:
            items_per_second = 0
            eta = 0
        else:
            items_per_second = self.current / elapsed_time if elapsed_time > 0 else 0
            remaining_items = self.total - self.current
            eta = remaining_items / items_per_second if items_per_second > 0 else 0
        
        # Format time strings
        elapsed_str = self._format_time(elapsed_time)
        eta_str = self._format_time(eta)
        
        # Get memory usage
        current_memory = self.process.memory_info().rss
        memory_diff = current_memory - self.initial_memory
        memory_str = f"{current_memory / (1024 * 1024):.1f} MB"
        memory_diff_str = f"{memory_diff / (1024 * 1024):+.1f} MB" if memory_diff != 0 else ""
        
        # Build progress string
        progress_str = f"\r{self.prefix} [{bar}] {percent:.1f}% {self.current}/{self.total} "
        progress_str += f"[{elapsed_str}<{eta_str}, {items_per_second:.1f} it/s]"
        
        # Add memory usage if significant change
        if abs(memory_diff) > 1024 * 1024:  # Only show if change > 1MB
            progress_str += f" | Mem: {memory_str} ({memory_diff_str})"
        
        # Add stage info
        if self.stage_info:
            progress_str += f" | Stage: {self.stage_info}"
        
        progress_str += f" {self.suffix}"
        
        # Ensure the string fits the terminal width
        if len(progress_str) > self.terminal_width:
            progress_str = progress_str[:self.terminal_width - 3] + "..."
        
        # Clear previous output by printing spaces
        clear_str = " " * self.last_printed_length
        sys.stdout.write("\r" + clear_str)
        
        # Print progress
        sys.stdout.write(progress_str)
        sys.stdout.flush()
        
        self.last_printed_length = len(progress_str)
    
    def _format_time(self, seconds: float) -> str:
        """
        Format time in seconds to a readable string.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            seconds = seconds % 60
            return f"{minutes}m {seconds:.1f}s"
        else:
            hours = int(seconds / 3600)
            seconds = seconds % 3600
            minutes = int(seconds / 60)
            seconds = seconds % 60
            return f"{hours}h {minutes}m {seconds:.1f}s"
    
    def finish(self) -> None:
        """Complete the progress bar and move to the next line."""
        self.update(self.total)
        sys.stdout.write("\n")
        sys.stdout.flush()


def log_memory_usage(logger=None) -> Dict[str, float]:
    """
    Log current memory usage statistics.
    
    Args:
        logger: Logger to use (if None, create a new one)
        
    Returns:
        Dictionary with memory usage statistics in MB
    """
    if logger is None:
        logger = get_logger("voxelflex.memory")
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Get system memory info
    system_memory = psutil.virtual_memory()
    
    # Calculate memory statistics
    process_rss_mb = memory_info.rss / (1024 * 1024)
    process_vms_mb = memory_info.vms / (1024 * 1024)
    system_used_mb = (system_memory.total - system_memory.available) / (1024 * 1024)
    system_total_mb = system_memory.total / (1024 * 1024)
    system_percent = system_memory.percent
    
    memory_stats = {
        "process_rss_mb": process_rss_mb,
        "process_vms_mb": process_vms_mb,
        "system_used_mb": system_used_mb,
        "system_total_mb": system_total_mb,
        "system_percent": system_percent
    }
    
    logger.info(
        f"Memory usage: Process RSS: {process_rss_mb:.1f} MB, "
        f"System: {system_used_mb:.1f}/{system_total_mb:.1f} MB ({system_percent:.1f}%)"
    )
    
    return memory_stats


def log_step(logger, step_name: str, step_index: int = None, total_steps: int = None) -> None:
    """
    Log a step in a process with optional progress information.
    
    Args:
        logger: Logger to use
        step_name: Name of the step
        step_index: Current step index (optional)
        total_steps: Total number of steps (optional)
    """
    progress_info = ""
    if step_index is not None and total_steps is not None:
        percent = (step_index / total_steps) * 100
        progress_info = f" ({step_index}/{total_steps}, {percent:.1f}%)"
    
    logger.info(f"Step: {step_name}{progress_info}")


def log_section_header(logger, section_name: str) -> None:
    """
    Log a section header to clearly delineate different parts of the process.
    
    Args:
        logger: Logger to use
        section_name: Name of the section
    """
    separator = "=" * min(70, len(section_name) + 10)
    logger.info(f"\n{separator}")
    logger.info(f"{COLOR_CYAN}{section_name}{COLOR_RESET}")
    logger.info(f"{separator}")


def log_operation_result(logger, operation_name: str, result_summary: str, 
                        success: bool = True, extras: Dict[str, Any] = None) -> None:
    """
    Log the result of an operation with additional details.
    
    Args:
        logger: Logger to use
        operation_name: Name of the operation
        result_summary: Summary of the result
        success: Whether the operation was successful
        extras: Additional information to log
    """
    color = COLOR_GREEN if success else COLOR_RED
    status = "SUCCESS" if success else "FAILURE"
    
    logger.info(f"{color}[{status}]{COLOR_RESET} {operation_name}: {result_summary}")
    
    if extras:
        for key, value in extras.items():
            logger.debug(f"  {key}: {value}")






class ProgressBar:
    """
    Fixed-bottom progress bar that can be updated from anywhere in the code.
    """
    
    def __init__(self, total: int, prefix: str = "", suffix: str = "", 
                 bar_length: int = 30):
        """
        Initialize progress bar.
        
        Args:
            total: Total number of items
            prefix: Prefix string
            suffix: Suffix string
            bar_length: Length of the progress bar
        """
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.bar_length = bar_length
        self.current = 0
        self.start_time = time.time()
        self.last_printed_length = 0
        
        # Get terminal width
        self.terminal_width = shutil.get_terminal_size().columns
    
    def update(self, current: int) -> None:
        """
        Update progress bar.
        
        Args:
            current: Current progress value
        """
        self.current = current
        self._print_progress()
    
    def _print_progress(self) -> None:
        """Print the progress bar."""
        # Calculate progress
        percent = float(self.current) / float(self.total) * 100
        filled_length = int(round(self.bar_length * self.current / float(self.total)))
        bar = "█" * filled_length + "-" * (self.bar_length - filled_length)
        
        # Calculate elapsed time and ETA
        elapsed_time = time.time() - self.start_time
        items_per_second = self.current / elapsed_time if elapsed_time > 0 else 0
        eta = (self.total - self.current) / items_per_second if items_per_second > 0 else 0
        
        # Format time strings
        elapsed_str = self._format_time(elapsed_time)
        eta_str = self._format_time(eta)
        
        # Build progress string
        progress_str = f"\r{self.prefix} [{bar}] {percent:.1f}% {self.current}/{self.total} "
        progress_str += f"[{elapsed_str}<{eta_str}, {items_per_second:.1f} it/s] {self.suffix}"
        
        # Ensure the string fits the terminal width
        if len(progress_str) > self.terminal_width:
            progress_str = progress_str[:self.terminal_width - 3] + "..."
        
        # Clear previous output by printing spaces
        clear_str = " " * self.last_printed_length
        sys.stdout.write("\r" + clear_str)
        
        # Print progress
        sys.stdout.write(progress_str)
        sys.stdout.flush()
        
        self.last_printed_length = len(progress_str)
    
    def _format_time(self, seconds: float) -> str:
        """
        Format time in seconds to a readable string.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            seconds = seconds % 60
            return f"{minutes}m {seconds:.1f}s"
        else:
            hours = int(seconds / 3600)
            seconds = seconds % 3600
            minutes = int(seconds / 60)
            seconds = seconds % 60
            return f"{hours}h {minutes}m {seconds:.1f}s"
    
    def finish(self) -> None:
        """Complete the progress bar and move to the next line."""
        self.update(self.total)
        sys.stdout.write("\n")
        sys.stdout.flush()
