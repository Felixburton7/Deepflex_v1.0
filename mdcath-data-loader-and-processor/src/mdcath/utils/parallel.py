"""
Parallel processing helper functions.
Includes robust error logging for pickling issues.
"""
import logging
import concurrent.futures
from tqdm import tqdm
from typing import Callable, Iterable, List, Any, Optional
import traceback # Import traceback module

def parallel_map(func: Callable,
                 items: Iterable,
                 num_cores: int,
                 use_progress_bar: bool = True,
                 desc: Optional[str] = None,
                 chunksize: int = 1) -> List[Any]:
    """
    Applies a function to items in parallel using ProcessPoolExecutor.

    Args:
        func (Callable): The function to apply (must be pickleable).
        items (Iterable): The iterable of items to process.
        num_cores (int): The number of worker processes. If 1, runs sequentially.
        use_progress_bar (bool): Display tqdm progress bar.
        desc (Optional[str]): Description for progress bar.
        chunksize (int): Chunksize for ProcessPoolExecutor.map.

    Returns:
        List[Any]: List of results or raises RuntimeError on failure.
    """
    if num_cores < 1:
        logging.warning("parallel_map called with num_cores < 1. Running sequentially.")
        num_cores = 1

    results = []
    item_list = []
    try:
         item_list = list(items)
         total_items = len(item_list)
    except Exception as e:
         logging.error(f"Input 'items' must be a finite iterable (like a list): {e}")
         raise TypeError("Input 'items' must be a finite iterable") from e

    if total_items == 0: return [] # Nothing to process

    effective_cores = min(num_cores, total_items)

    if effective_cores == 1:
        logging.info("Running sequentially (num_cores=1 or only 1 item).")
        progress_iterator = tqdm(item_list, desc=desc or "Processing", disable=not use_progress_bar, total=total_items)
        try:
            for item in progress_iterator:
                results.append(func(item)) # Call the function directly
        except Exception as e:
             # Apply robust logging fix for sequential mode error
             logging.error(f"Error during sequential processing in parallel_map helper: {e}")
             if logging.getLogger().isEnabledFor(logging.DEBUG):
                  logging.debug(f"Traceback:\n{traceback.format_exc()}")
             raise RuntimeError("Sequential processing failed") from e # Re-raise
    else:
        logging.info(f"Running in parallel with {effective_cores} cores (chunksize={chunksize}).")
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=effective_cores) as executor:
                future_iterator = executor.map(func, item_list, chunksize=chunksize)
                progress_iterator = tqdm(future_iterator, desc=desc or "Processing", total=total_items, disable=not use_progress_bar)
                results = list(progress_iterator) # Collect results

        except Exception as e:
            # Apply robust logging fix for parallel mode error (like PicklingError)
            logging.error(f"Error during parallel processing in parallel_map helper: {e}")
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                 # Log traceback explicitly using traceback module
                 logging.debug(f"Traceback:\n{traceback.format_exc()}")
            # Re-raise a more informative error
            raise RuntimeError(f"Parallel processing failed: {e}") from e

    return results