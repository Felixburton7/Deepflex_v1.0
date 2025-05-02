import multiprocessing
from tqdm import tqdm  # Ensure tqdm is installed (pip install tqdm)

def parallel_map(func, iterable, processes=None):
    """Apply func to each item in iterable using multiprocessing with a progress bar."""
    if processes is None:
        processes = multiprocessing.cpu_count()
    results = []
    with multiprocessing.Pool(processes=processes) as pool:
        # Use imap for a lazy iterator and wrap it with tqdm for progress tracking
        for result in tqdm(pool.imap(func, iterable), total=len(iterable), desc="Processing CSV files"):
            results.append(result)
    return results
